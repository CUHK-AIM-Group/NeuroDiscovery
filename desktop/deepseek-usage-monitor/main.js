// DeepSeek 用量监控客户端 —— 主进程
//
// 安全设计：
//   - contextIsolation + sandbox，渲染进程无 Node 权限，一切网络请求只在本进程发起；
//   - 只访问两个硬编码官方域名：api.deepseek.com / platform.deepseek.com；
//   - 凭证保存在 userData/config.json，Windows 下用 icacls 锁定为仅当前用户；
//   - 凭证不回传渲染进程（渲染层只知道"是否已配置"），错误信息自动打码；
//   - 不加载任何远程内容，禁止页面导航，外链只允许官方域名并交给系统浏览器。
'use strict';

const { app, BrowserWindow, ipcMain, Notification, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const fsp = fs.promises;
const { execFile } = require('child_process');
const { randomUUID } = require('crypto');
const history = require('./history');

const API_BASE = 'https://api.deepseek.com';
const PLATFORM = 'https://platform.deepseek.com';
const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36';
const FETCH_TIMEOUT_MS = 20000;

const SMOKE = process.argv.includes('--smoke');
const MOCK = process.argv.includes('--mock');
let mainWindow = null;
let lowBalanceNotified = false;

// 固定 userData 路径：打包后（productName 变化）仍沿用开发版路径，
// 已保存的凭证与本地历史记录不丢失。DSUM_USERDATA 仅用于自动化测试隔离。
app.setPath('userData',
  process.env.DSUM_USERDATA
    ? path.resolve(process.env.DSUM_USERDATA)
    : path.join(app.getPath('appData'), 'deepseek-usage-monitor'));

// ── Mock 数据（--mock 模式，用于无凭证的渲染自检）────────
function mockBalance() {
  return {
    total_balance: '88.88',
    granted_balance: '8.88',
    topped_up_balance: '80.00',
    currency: 'CNY',
    is_available: true,
  };
}
function mockUsage(month, year) {
  const days = [];
  for (let d = 1; d <= 5; d++) {
    days.push({
      date: `${year}-${String(month).padStart(2, '0')}-${String(d).padStart(2, '0')}`,
      input: 120000 + d * 8000,
      cache_hit: 80000 + d * 5000,
      cache_miss: 40000 + d * 3000,
      output: 30000 + d * 2000,
      cost: Math.round((0.42 + d * 0.07) * 10000) / 10000,
    });
  }
  return {
    year, month,
    month_cost: 2.52,
    today: { cost: 0.63, input: 160000, output: 40000, cache_hit: 105000, cache_miss: 55000 },
    days,
    models: [
      { model: 'deepseek-chat', input: 520000, output: 140000, cost: 1.68, percent: 66.7 },
      { model: 'deepseek-reasoner', input: 120000, output: 60000, cost: 0.84, percent: 33.3 },
    ],
  };
}

// ── 凭证打码 ──────────────────────────────────────────────
let secrets = [];
function trackSecrets(cfg) {
  secrets = [];
  if (!cfg || !Array.isArray(cfg.accounts)) return;
  for (const a of cfg.accounts) {
    if (a.apiKey) secrets.push(String(a.apiKey));
    if (a.sessionToken) secrets.push(String(a.sessionToken));
  }
}
function redact(text) {
  let t = String(text ?? '');
  for (const s of secrets) if (s) t = t.split(s).join('***');
  return t;
}

// ── 配置存储（ACL 保护，多账号结构）───────────────────────
// config.json 结构：
//   {
//     "accounts": [ { "id": "<uuid>", "name": "账号名", "apiKey": "sk-…", "sessionToken": "…" } ],
//     "activeAccountId": "<uuid>",
//     "refreshInterval": 60, "alertEnabled": false, "alertThreshold": 5, "theme": "dark"
//   }
function configPath() {
  return path.join(app.getPath('userData'), 'config.json');
}
function historyFilePath(accountId) {
  return path.join(app.getPath('userData'), `history-${accountId}.jsonl`);
}
function normalizeConfig(raw) {
  const cfg = raw && typeof raw === 'object' ? raw : {};
  let migrated = false;
  // 旧版单账号结构 → 迁移为账号列表
  if (cfg.apiKey && !Array.isArray(cfg.accounts)) {
    const id = randomUUID();
    cfg.accounts = [{
      id,
      name: '默认账号',
      apiKey: String(cfg.apiKey),
      sessionToken: cfg.sessionToken ? String(cfg.sessionToken) : '',
    }];
    cfg.activeAccountId = id;
    delete cfg.apiKey;
    delete cfg.sessionToken;
    migrated = true;
    // 旧版历史文件 history.jsonl → history-<账号id>.jsonl
    try {
      const legacy = path.join(app.getPath('userData'), 'history.jsonl');
      const target = historyFilePath(id);
      if (fs.existsSync(legacy) && !fs.existsSync(target)) fs.renameSync(legacy, target);
    } catch (e) {
      console.error(`legacy history migration failed: ${e.message}`);
    }
  }
  if (!Array.isArray(cfg.accounts)) cfg.accounts = [];
  cfg.accounts = cfg.accounts.map((a) => ({
    id: a.id || randomUUID(),
    name: String(a.name || '账号'),
    apiKey: a.apiKey ? String(a.apiKey) : '',
    sessionToken: a.sessionToken ? String(a.sessionToken) : '',
  }));
  if (!cfg.activeAccountId || !cfg.accounts.some((a) => a.id === cfg.activeAccountId)) {
    cfg.activeAccountId = cfg.accounts.length ? cfg.accounts[0].id : null;
  }
  return { cfg, migrated };
}
async function writeConfigRaw(cfg) {
  await fsp.mkdir(path.dirname(configPath()), { recursive: true });
  await fsp.writeFile(configPath(), JSON.stringify(cfg, null, 2), 'utf8');
  if (process.platform === 'win32') {
    // 移除继承权限，仅保留当前用户
    await new Promise((resolve) => {
      execFile('icacls', [configPath(), '/inheritance:r', '/grant:r', `${process.env.USERNAME}:F`], () => resolve());
    });
  }
}
function syncRuntime(cfg) {
  // 同步历史存储注册与打码清单（按账号隔离）
  const secretList = [];
  for (const a of cfg.accounts) {
    history.register(a.id, historyFilePath(a.id));
    if (a.apiKey) secretList.push(a.apiKey);
    if (a.sessionToken) secretList.push(a.sessionToken);
  }
  secrets = secretList;
}
async function loadConfig() {
  let raw = {};
  try {
    raw = JSON.parse(await fsp.readFile(configPath(), 'utf8'));
  } catch { /* 无配置 */ }
  const { cfg, migrated } = normalizeConfig(raw);
  if (migrated) await writeConfigRaw(cfg);
  syncRuntime(cfg);
  return cfg;
}
function loadConfigSync() {
  // 同步读取（仅用于窗口创建时的背景色等一次性决策）。
  // 注意：只读原始字段，绝不执行迁移副作用（迁移统一在 loadConfig 中完成一次）。
  try {
    const cfg = JSON.parse(fs.readFileSync(configPath(), 'utf8'));
    return { theme: cfg.theme, accounts: cfg.accounts || [], activeAccountId: cfg.activeAccountId };
  } catch {
    return {};
  }
}
async function saveConfig(patch) {
  const cfg = await loadConfig();
  Object.assign(cfg, patch);
  await writeConfigRaw(cfg);
  syncRuntime(cfg);
  return cfg;
}
function activeAccount(cfg) {
  return cfg.accounts.find((a) => a.id === cfg.activeAccountId) || cfg.accounts[0] || null;
}
function sanitizeAccount(a) {
  return { id: a.id, name: a.name, hasApiKey: !!a.apiKey, hasSessionToken: !!a.sessionToken };
}
function publicSettings(cfg) {
  return {
    refreshInterval: cfg && cfg.refreshInterval ? Number(cfg.refreshInterval) : 60,
    alertEnabled: !!(cfg && cfg.alertEnabled),
    alertThreshold: cfg && cfg.alertThreshold != null ? Number(cfg.alertThreshold) : 5,
    theme: cfg && cfg.theme === 'light' ? 'light' : 'dark',
    accounts: (cfg.accounts || []).map(sanitizeAccount),
    activeAccountId: cfg.activeAccountId,
  };
}

// ── 网络（带超时，仅官方域名）────────────────────────────
async function fetchWithTimeout(url, options = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    return await fetch(url, { ...options, signal: ctrl.signal });
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('请求超时（20 秒），请检查网络');
    throw new Error(`网络请求失败：${redact(e.message)}`);
  } finally {
    clearTimeout(timer);
  }
}

async function fetchBalance(apiKey) {
  const res = await fetchWithTimeout(`${API_BASE}/user/balance`, {
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  if (res.status === 401) throw new Error('API Key 无效或已过期（HTTP 401）');
  if (res.status === 429) throw new Error('请求过于频繁（HTTP 429），请稍后再试');
  if (!res.ok) throw new Error(`余额接口返回 HTTP ${res.status}`);
  const j = await res.json();
  // 账户可能有多个币种条目（如 CNY + USD），且数组顺序不稳定——
  // 不能取 [0]，按余额降序取最高的作为主账户。
  const entries = (Array.isArray(j.balance_infos) ? j.balance_infos : [])
    .map((info) => ({
      total_balance: String(info.total_balance ?? '0'),
      granted_balance: String(info.granted_balance ?? '0'),
      topped_up_balance: String(info.topped_up_balance ?? '0'),
      currency: info.currency || 'CNY',
    }))
    .sort((a, b) => num(b.total_balance) - num(a.total_balance));
  if (!entries.length) throw new Error('余额接口返回格式异常');
  const primary = entries[0];
  return {
    total_balance: primary.total_balance,
    granted_balance: primary.granted_balance,
    topped_up_balance: primary.topped_up_balance,
    currency: primary.currency,
    is_available: j.is_available !== false,
    entries,
  };
}

function platformHeaders(sessionToken) {
  return {
    Authorization: `Bearer ${sessionToken}`,
    'x-app-version': '1.0.0',
    Accept: '*/*',
    Referer: `${PLATFORM}/usage`,
    Origin: PLATFORM,
    'User-Agent': UA,
  };
}

function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

async function fetchUsage(sessionToken, month, year) {
  const h = platformHeaders(sessionToken);
  const url = (kind) => `${PLATFORM}/api/v0/usage/${kind}?month=${month}&year=${year}`;
  const fetchJson = async (kind) => {
    const res = await fetchWithTimeout(url(kind), { headers: h });
    if (res.status === 401) throw new Error('Session Token 无效或已过期（HTTP 401），请重新获取');
    if (res.status === 429) throw new Error('请求过于频繁（HTTP 429），请稍后再试');
    if (!res.ok) throw new Error(`用量接口 ${kind} 返回 HTTP ${res.status}`);
    return res.json();
  };
  const [amountResp, costResp] = await Promise.all([fetchJson('amount'), fetchJson('cost')]);
  const biz = (r) => {
    const bd = r && r.data && r.data.biz_data;
    return Array.isArray(bd) ? bd[0] : bd || null;
  };
  const amountBiz = biz(amountResp);
  const costBiz = biz(costResp);
  if (!amountBiz || !Array.isArray(amountBiz.days)) {
    throw new Error('接口返回格式异常：未找到 days 数据（可能本月暂无用量）');
  }

  // 费用索引：按日期 / 按 日期+模型
  const costByDate = {};
  const costByModelByDate = {};
  let monthCost = 0;
  if (costBiz) {
    for (const day of costBiz.days || []) {
      const dateKey = String(day.date);
      let dayCost = 0;
      for (const mu of day.data || []) {
        let modelCost = 0;
        for (const entry of mu.usage || []) modelCost += num(entry.amount);
        dayCost += modelCost;
        const model = String(mu.model || 'unknown');
        costByModelByDate[dateKey] = costByModelByDate[dateKey] || {};
        costByModelByDate[dateKey][model] = (costByModelByDate[dateKey][model] || 0) + modelCost;
      }
      costByDate[dateKey] = dayCost;
    }
    for (const m of costBiz.total || []) {
      for (const entry of m.usage || []) monthCost += num(entry.amount);
    }
  }

  // 按天汇总 Token
  const days = [];
  for (const day of amountBiz.days) {
    const dateKey = String(day.date);
    let hit = 0, miss = 0, respTok = 0, promptTok = 0;
    for (const mu of day.data || []) {
      for (const entry of mu.usage || []) {
        const type = String(entry.type || '');
        const amt = num(entry.amount);
        if (type.includes('CACHE_HIT')) hit += amt;
        else if (type.includes('CACHE_MISS')) miss += amt;
        else if (type.includes('RESPONSE')) respTok += amt;
        else if (type.includes('PROMPT')) promptTok += amt;
      }
    }
    days.push({
      date: dateKey,
      input: Math.round(promptTok + hit + miss),
      cache_hit: Math.round(hit),
      cache_miss: Math.round(miss),
      output: Math.round(respTok),
      cost: Math.round((costByDate[dateKey] || 0) * 10000) / 10000,
    });
  }

  // 按模型汇总
  const modelTotals = {};
  for (const day of amountBiz.days) {
    const dateKey = String(day.date);
    for (const mu of day.data || []) {
      const model = String(mu.model || 'unknown');
      modelTotals[model] = modelTotals[model] || { input: 0, output: 0, cost: 0 };
      for (const entry of mu.usage || []) {
        const type = String(entry.type || '');
        const amt = num(entry.amount);
        if (type.includes('RESPONSE')) modelTotals[model].output += amt;
        else modelTotals[model].input += amt;
      }
      if (costByModelByDate[dateKey] && costByModelByDate[dateKey][model]) {
        modelTotals[model].cost += costByModelByDate[dateKey][model];
      }
    }
  }
  const totalCost = Object.values(modelTotals).reduce((s, m) => s + m.cost, 0);
  const models = Object.keys(modelTotals)
    .sort()
    .map((model) => {
      const cost = Math.round(modelTotals[model].cost * 10000) / 10000;
      return {
        model,
        input: Math.round(modelTotals[model].input),
        output: Math.round(modelTotals[model].output),
        cost,
        percent: totalCost > 0 ? Math.round((cost / totalCost) * 1000) / 10 : 0,
      };
    });

  // 今日
  const todayStr = new Date().toISOString().slice(0, 10);
  const todayDay = days.find((d) => d.date === todayStr);
  const today = todayDay
    ? { cost: todayDay.cost, input: todayDay.input, output: todayDay.output, cache_hit: todayDay.cache_hit, cache_miss: todayDay.cache_miss }
    : { cost: 0, input: 0, output: 0, cache_hit: 0, cache_miss: 0 };

  return { year, month, month_cost: Math.round(monthCost * 10000) / 10000, today, days, models };
}

// ── 低余额通知 ───────────────────────────────────────────
function maybeNotifyLowBalance(cfg, balance) {
  const threshold = num(cfg.alertThreshold);
  const total = num(balance.total_balance);
  const below = cfg.alertEnabled && threshold > 0 && total < threshold;
  if (below && !lowBalanceNotified && Notification.isSupported()) {
    new Notification({
      title: 'DeepSeek 余额不足',
      body: `当前余额 ${balance.total_balance} ${balance.currency}，已低于阈值 ${threshold} 元`,
    }).show();
    lowBalanceNotified = true;
  }
  if (!below) lowBalanceNotified = false;
}

// ── IPC ──────────────────────────────────────────────────
function registerIpc() {
  ipcMain.handle('settings:get', async () => {
    const cfg = await loadConfig();
    const pub = publicSettings(cfg);
    // mock 模式：无账号时注入一个演示账号，便于渲染自检
    if (MOCK && !pub.accounts.length) {
      return {
        ...pub,
        accounts: [{ id: 'mock', name: 'Mock 账号', hasApiKey: true, hasSessionToken: true }],
        activeAccountId: 'mock',
      };
    }
    return pub;
  });

  ipcMain.handle('settings:save', async (_e, patch = {}) => {
    const clean = {};
    if (patch.refreshInterval != null) clean.refreshInterval = Math.min(3600, Math.max(10, Number(patch.refreshInterval) || 60));
    if (typeof patch.alertEnabled === 'boolean') clean.alertEnabled = patch.alertEnabled;
    if (patch.alertThreshold != null) clean.alertThreshold = Math.max(0, Number(patch.alertThreshold) || 0);
    if (patch.theme === 'light' || patch.theme === 'dark') clean.theme = patch.theme;
    const cfg = await saveConfig(clean);
    return publicSettings(cfg);
  });

  // 清除当前账号的凭证
  ipcMain.handle('settings:clear', async () => {
    const cfg = await loadConfig();
    const acc = activeAccount(cfg);
    if (acc) {
      acc.apiKey = '';
      acc.sessionToken = '';
      await writeConfigRaw(cfg);
      syncRuntime(cfg);
    }
    return publicSettings(cfg);
  });

  // ── 账号管理 ──
  ipcMain.handle('account:add', async (_e, name, apiKey, sessionToken) => {
    const key = apiKey ? String(apiKey).trim() : '';
    if (!key) throw new Error('API Key 不能为空');
    const cfg = await loadConfig();
    const acc = {
      id: randomUUID(),
      name: String(name || '').trim() || `账号 ${cfg.accounts.length + 1}`,
      apiKey: key,
      sessionToken: sessionToken ? String(sessionToken).trim() : '',
    };
    cfg.accounts.push(acc);
    if (!cfg.activeAccountId) cfg.activeAccountId = acc.id;
    await writeConfigRaw(cfg);
    syncRuntime(cfg);
    return publicSettings(cfg);
  });

  ipcMain.handle('account:update', async (_e, id, patch = {}) => {
    const cfg = await loadConfig();
    const acc = cfg.accounts.find((a) => a.id === id);
    if (!acc) throw new Error('账号不存在');
    if (patch.name != null && String(patch.name).trim()) acc.name = String(patch.name).trim();
    // 非空字符串 → 更新；空字符串 → 清除；undefined → 保持不变
    if (patch.apiKey != null) acc.apiKey = patch.apiKey === '' ? '' : String(patch.apiKey).trim();
    if (patch.sessionToken != null) acc.sessionToken = patch.sessionToken === '' ? '' : String(patch.sessionToken).trim();
    await writeConfigRaw(cfg);
    syncRuntime(cfg);
    return publicSettings(cfg);
  });

  ipcMain.handle('account:delete', async (_e, id) => {
    const cfg = await loadConfig();
    cfg.accounts = cfg.accounts.filter((a) => a.id !== id);
    if (cfg.activeAccountId === id) {
      cfg.activeAccountId = cfg.accounts.length ? cfg.accounts[0].id : null;
    }
    await writeConfigRaw(cfg);
    syncRuntime(cfg);
    return publicSettings(cfg);
  });

  ipcMain.handle('account:activate', async (_e, id) => {
    const cfg = await loadConfig();
    if (!cfg.accounts.some((a) => a.id === id)) throw new Error('账号不存在');
    cfg.activeAccountId = id;
    await writeConfigRaw(cfg);
    syncRuntime(cfg);
    return publicSettings(cfg);
  });

  ipcMain.handle('balance:get', async () => {
    if (MOCK) {
      const m = mockBalance();
      history.appendBalance('mock', m).catch(() => {});
      return m;
    }
    const cfg = await loadConfig();
    const acc = activeAccount(cfg);
    if (!acc || !acc.apiKey) throw new Error('未配置 API Key：请先在设置页添加账号或切换账号');
    const balance = await fetchBalance(acc.apiKey);
    history.appendBalance(acc.id, balance).catch(() => {});
    maybeNotifyLowBalance(cfg, balance);
    return balance;
  });

  ipcMain.handle('usage:get', async (_e, month, year) => {
    if (MOCK) {
      const m = mockUsage(Number(month), Number(year));
      history.appendUsage('mock', m).catch(() => {});
      return m;
    }
    const cfg = await loadConfig();
    const acc = activeAccount(cfg);
    if (!acc || !acc.sessionToken) throw new Error('未配置 Session Token：请先在设置页为当前账号填写（获取方法见 README）');
    const usage = await fetchUsage(acc.sessionToken, Number(month), Number(year));
    history.appendUsage(acc.id, usage).catch(() => {});
    return usage;
  });

  ipcMain.handle('history:get', async (_e, opts) => {
    const cfg = await loadConfig();
    const acc = activeAccount(cfg);
    const accountId = MOCK ? 'mock' : (acc ? acc.id : null);
    return accountId ? history.getHistory(accountId, opts || {}) : { balance: [], usage: [], file: null };
  });
  ipcMain.handle('history:clear', async () => {
    const cfg = await loadConfig();
    const acc = activeAccount(cfg);
    const accountId = MOCK ? 'mock' : (acc ? acc.id : null);
    if (accountId) await history.clearHistory(accountId);
  });

  ipcMain.handle('connection:test', async (_e, apiKey, sessionToken) => {
    const cfg = await loadConfig();
    const acc = activeAccount(cfg);
    const key = apiKey || (acc && acc.apiKey);
    const tok = sessionToken || (acc && acc.sessionToken);
    if (!key) throw new Error('请先填写 API Key');
    secrets = secrets.concat(key, tok).filter(Boolean);
    const balance = await fetchBalance(key);
    let usageOk = true;
    let usageMsg = '';
    if (tok) {
      try {
        const now = new Date();
        await fetchUsage(tok, now.getMonth() + 1, now.getFullYear());
      } catch (e) {
        usageOk = false;
        usageMsg = `；用量接口：${redact(e.message)}`;
      }
    }
    return { balanceOk: true, usageOk, message: `连接成功：余额 ${balance.total_balance} ${balance.currency}${usageMsg}` };
  });

  ipcMain.handle('open-external', async (_e, url) => {
    const ok = ['https://platform.deepseek.com', 'https://api-docs.deepseek.com'].some((p) => String(url).startsWith(p));
    if (ok) shell.openExternal(String(url));
  });
}

// ── 窗口 ─────────────────────────────────────────────────
function createWindow() {
  const cfg0 = loadConfigSync();
  mainWindow = new BrowserWindow({
    width: 1180,
    height: 780,
    minWidth: 940,
    minHeight: 620,
    backgroundColor: cfg0.theme === 'light' ? '#f3f4f6' : '#0f0f0a',
    icon: path.join(__dirname, 'build', 'icon.png'),
    autoHideMenuBar: true,
    title: 'DeepSeek 用量监控',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  // 禁止导航与弹窗；官方域名外链交给系统浏览器
  mainWindow.webContents.on('will-navigate', (e) => e.preventDefault());
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https://platform.deepseek.com') || url.startsWith('https://api-docs.deepseek.com')) {
      shell.openExternal(url);
    }
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => { mainWindow = null; });
}

// ── 启动 ─────────────────────────────────────────────────
app.setAppUserModelId('com.neuroclaw.deepseek-usage-monitor');
app.whenReady().then(() => {
  // mock 模式使用独立历史文件，避免污染真实数据
  if (MOCK) {
    history.register('mock', path.join(app.getPath('userData'), 'history-mock.jsonl'));
  }
  registerIpc();
  createWindow();

  if (SMOKE) {
    mainWindow.webContents.once('did-finish-load', async () => {
      try {
        const title = await mainWindow.webContents.executeJavaScript('document.title');
        const hasChart = await mainWindow.webContents.executeJavaScript('typeof window.Chart !== "undefined"');
        const hasBridge = await mainWindow.webContents.executeJavaScript('typeof window.dsApi !== "undefined"');
        let dom = '';
        let historyLines = 0;
        if (MOCK) {
          // 等待异步 refresh() 渲染完成
          await new Promise((r) => setTimeout(r, 1800));
          dom = await mainWindow.webContents.executeJavaScript(
            'JSON.stringify({ cards: document.querySelectorAll("#cards .card").length, models: document.querySelectorAll("#model-table tbody tr").length, days: document.querySelectorAll("#days-table tbody tr").length, balanceChart: Chart.getChart("chart-balance") ? Chart.getChart("chart-balance").data.datasets[0].data.length : -1, monthChart: Chart.getChart("chart-monthcost") ? Chart.getChart("chart-monthcost").data.datasets[0].data.length : -1, accounts: document.querySelectorAll("#account-select option").length, theme: document.documentElement.dataset.theme, status: document.getElementById("status-bar").textContent })');
          const hf = path.join(app.getPath('userData'), 'history-mock.jsonl');
          historyLines = fs.existsSync(hf) ? fs.readFileSync(hf, 'utf8').split('\n').filter(Boolean).length : 0;
        }
        const smokeDir = path.join(app.getPath('userData'), 'smoke');
        await fsp.mkdir(smokeDir, { recursive: true });
        console.log(`SMOKE_OK title=${JSON.stringify(title)} chart=${hasChart} bridge=${hasBridge}${dom ? ' dom=' + dom : ''}${historyLines ? ' historyLines=' + historyLines : ''}`);
        app.exit(0);
      } catch (e) {
        console.error(`SMOKE_FAIL ${e.message}`);
        app.exit(1);
      }
    });
    setTimeout(() => { console.error('SMOKE_TIMEOUT'); app.exit(2); }, 30000);
  }
});

app.on('window-all-closed', () => {
  app.quit();
});
