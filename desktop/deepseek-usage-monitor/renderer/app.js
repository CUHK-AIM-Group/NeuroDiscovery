// DeepSeek 用量监控 —— 渲染进程逻辑
// 通过 window.dsApi（preload 暴露）与主进程通信；本进程无网络权限。
'use strict';

const $ = (id) => document.getElementById(id);
const state = {
  settings: null,
  balance: null,
  usage: null,
  history: null,
  accountForm: null, // null | 'add' | { id }
  month: new Date().getMonth() + 1,
  year: new Date().getFullYear(),
  timer: null,
  busy: false,
};

const PALETTE = ['#22d3ee', '#a855f7', '#f97316', '#22c55e', '#eab308', '#3b82f6', '#ef4444', '#14b8a6'];

// ── 主题 ──
const THEME_COLORS = {
  dark: { text: '#9ca3af', grid: '#242418', legend: '#d1d5db' },
  light: { text: '#6b7280', grid: '#e5e7eb', legend: '#374151' },
};

function applyTheme(theme) {
  const t = theme === 'light' ? 'light' : 'dark';
  document.documentElement.dataset.theme = t;
  const c = THEME_COLORS[t];
  Chart.defaults.color = c.text;
  for (const ch of [dailyChart, costChart, modelChart, balanceChart, monthCostChart]) {
    const s = ch.options.scales;
    if (s) {
      for (const key of Object.keys(s)) {
        if (s[key].ticks) s[key].ticks.color = c.text;
        if (s[key].grid) s[key].grid.color = c.grid;
      }
    }
    const legend = ch.options.plugins && ch.options.plugins.legend;
    if (legend && legend.labels) legend.labels.color = c.legend;
    if (ch === modelChart && ch.data.datasets[0]) {
      ch.data.datasets[0].borderColor =
        getComputedStyle(document.documentElement).getPropertyValue('--panel').trim() || '#ffffff';
    }
    ch.update();
  }
  const btn = $('btn-theme');
  if (btn) btn.textContent = t === 'dark' ? '☀️ 日间' : '🌙 夜间';
}

async function onToggleTheme() {
  const next = state.settings && state.settings.theme === 'light' ? 'dark' : 'light';
  try {
    state.settings = await window.dsApi.saveSettings({ theme: next });
    applyTheme(next);
    setStatus(`已切换到${next === 'light' ? '日间' : '夜间'}模式`, 'ok');
  } catch (e) {
    setStatus(e.message || String(e), 'err');
  }
}

// ── 格式化 ──
const fmtN = (n) => Number(n || 0).toLocaleString('zh-CN');
const fmtTokens = (n) => fmtN(Math.round(Number(n || 0)));
const fmtMoney = (n) => '¥' + Number(n || 0).toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 4 });
const fmtPct = (n) => Number(n || 0).toFixed(1) + '%';

// ── 状态栏 ──
function setStatus(msg, kind = 'info') {
  const bar = $('status-bar');
  bar.textContent = msg || '';
  bar.className = kind;
}
function setLastUpdated() {
  $('last-updated').textContent = '更新于 ' + new Date().toLocaleTimeString('zh-CN');
}

// ── 账号 ──
function accounts() {
  return (state.settings && state.settings.accounts) || [];
}
function activeAccount() {
  const s = state.settings;
  if (!s) return null;
  return s.accounts.find((a) => a.id === s.activeAccountId) || s.accounts[0] || null;
}

function renderAccountSelect() {
  const sel = $('account-select');
  const accs = accounts();
  if (!accs.length) {
    sel.innerHTML = '<option value="">（无账号，请到设置页添加）</option>';
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  const active = activeAccount();
  sel.innerHTML = accs.map((a) =>
    `<option value="${a.id}" ${a.id === active.id ? 'selected' : ''}>${escapeHtml(a.name)}${a.id === active.id ? ' ★' : ''}</option>`).join('');
}

function renderAccountList() {
  const list = $('account-list');
  const accs = accounts();
  const active = activeAccount();
  if (!accs.length) {
    list.innerHTML = '<p class="muted small">还没有账号。点击「＋ 添加账号」开始。</p>';
    return;
  }
  list.innerHTML = accs.map((a) => {
    const isActive = a.id === active.id;
    return `<div class="acc-row ${isActive ? 'active' : ''}">
      <span class="acc-name">${escapeHtml(a.name)}</span>
      <span class="acc-badges">
        <span class="badge ${a.hasApiKey ? 'ok' : 'no'}">API Key</span>
        <span class="badge ${a.hasSessionToken ? 'ok' : 'no'}">Token</span>
        ${isActive ? '<span class="badge active">当前</span>' : ''}
      </span>
      <span class="acc-actions">
        ${isActive ? '' : `<button class="btn" data-act="${a.id}">设为当前</button>`}
        <button class="btn" data-edit="${a.id}">编辑</button>
        <button class="btn danger" data-del="${a.id}">删除</button>
      </span>
    </div>`;
  }).join('');

  list.querySelectorAll('[data-act]').forEach((b) =>
    b.addEventListener('click', () => onActivate(b.dataset.act)));
  list.querySelectorAll('[data-edit]').forEach((b) =>
    b.addEventListener('click', () => openAccountForm(a => a.id === b.dataset.edit)));
  list.querySelectorAll('[data-del]').forEach((b) =>
    b.addEventListener('click', () => onDeleteAccount(b.dataset.del)));
}

function openAccountForm(mode) {
  state.accountForm = mode;
  const editing = mode !== 'add' ? accounts().find((a) => a.id === mode.id) : null;
  $('account-form').hidden = false;
  $('account-form-title').textContent = editing ? `编辑账号：${editing.name}` : '添加账号';
  $('inp-acc-name').value = editing ? editing.name : '';
  $('inp-acc-key').value = '';
  $('inp-acc-token').value = '';
  $('inp-acc-key').placeholder = editing && editing.hasApiKey ? '已保存（留空保持不变，输入新值覆盖）' : 'sk-…';
  $('inp-acc-token').placeholder = editing && editing.hasSessionToken ? '已保存（留空保持不变，输入新值覆盖）' : '登录态 Bearer Token';
  $('account-form-status').textContent = '';
  $('inp-acc-name').focus();
}

function closeAccountForm() {
  state.accountForm = null;
  $('account-form').hidden = true;
}

async function onSaveAccount() {
  const name = $('inp-acc-name').value.trim();
  const key = $('inp-acc-key').value.trim();
  const token = $('inp-acc-token').value.trim();
  try {
    if (state.accountForm === 'add') {
      if (!key) { setStatus('API Key 不能为空', 'err'); return; }
      state.settings = await window.dsApi.addAccount(name, key, token);
      setStatus('账号已添加', 'ok');
    } else if (state.accountForm) {
      const patch = {};
      if (name) patch.name = name;
      if (key) patch.apiKey = key;
      if (token) patch.sessionToken = token;
      if (!Object.keys(patch).length) {
        setStatus('没有需要更新的内容（未填写任何字段）', 'info');
        return;
      }
      state.settings = await window.dsApi.updateAccount(state.accountForm.id, patch);
      setStatus('账号已更新', 'ok');
    }
    closeAccountForm();
    renderAccountSelect();
    renderAccountList();
    scheduleRefresh();
    await refresh();
  } catch (e) {
    setStatus(e.message || String(e), 'err');
  }
}

async function onTestAccount() {
  const key = $('inp-acc-key').value.trim();
  const token = $('inp-acc-token').value.trim();
  const acc = activeAccount();
  if (!key && !(acc && acc.hasApiKey)) {
    setStatus('请先填写 API Key', 'err');
    return;
  }
  setStatus('测试连接中…', 'info');
  try {
    const r = await window.dsApi.testConnection(key, token);
    setStatus(r.message, 'ok');
    $('account-form-status').textContent = r.message;
  } catch (e) {
    setStatus(e.message || String(e), 'err');
    $('account-form-status').textContent = e.message || String(e);
  }
}

async function onActivate(id) {
  try {
    state.settings = await window.dsApi.activateAccount(id);
    renderAccountSelect();
    renderAccountList();
    setStatus(`已切换到账号「${activeAccount().name}」`, 'ok');
    await refresh();
  } catch (e) {
    setStatus(e.message || String(e), 'err');
  }
}

async function onDeleteAccount(id) {
  const acc = accounts().find((a) => a.id === id);
  if (!acc) return;
  if (!confirm(`确定删除账号「${acc.name}」？其凭证将一并删除（历史数据文件保留）。`)) return;
  try {
    state.settings = await window.dsApi.deleteAccount(id);
    closeAccountForm();
    renderAccountSelect();
    renderAccountList();
    state.balance = null;
    state.usage = null;
    state.history = null;
    renderCards();
    renderCharts();
    renderTables();
    renderHistory();
    setStatus('账号已删除', 'ok');
    await refresh();
  } catch (e) {
    setStatus(e.message || String(e), 'err');
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ── 卡片 ──
function renderCards() {
  const cards = $('cards');
  const b = state.balance;
  const u = state.usage;
  cards.innerHTML = '';

  const card = (label, value, cls, sub) =>
    `<div class="card"><div class="label">${label}</div><div class="value ${cls}">${value}</div><div class="sub">${sub}</div></div>`;

  if (b) {
    const multi = b.entries && b.entries.length > 1 ? ` · 另有 ${b.entries.length - 1} 个币种账户` : '';
    cards.insertAdjacentHTML('beforeend', card('总余额（' + b.currency + '）', b.total_balance, 'blue',
      `充值 ${b.topped_up_balance} · 赠送 ${b.granted_balance}${multi}`));
    cards.insertAdjacentHTML('beforeend', card('账户状态', b.is_available ? '可用' : '余额不足', b.is_available ? 'green' : 'red',
      b.is_available ? '可继续调用 API' : '请及时充值'));
  } else {
    cards.insertAdjacentHTML('beforeend', card('总余额', '—', 'muted', accounts().length ? '当前账号未配置 API Key' : '未添加账号'));
    cards.insertAdjacentHTML('beforeend', card('账户状态', '—', 'muted', ''));
  }

  if (u) {
    cards.insertAdjacentHTML('beforeend', card('今日消耗', fmtMoney(u.today.cost), 'orange',
      `输入 ${fmtTokens(u.today.input)} · 输出 ${fmtTokens(u.today.output)} tokens`));
    cards.insertAdjacentHTML('beforeend', card(`${u.year}-${u.month} 本月消耗`, fmtMoney(u.month_cost), 'purple',
      `输入 ${fmtTokens(u.days.reduce((s, d) => s + d.input, 0))} · 输出 ${fmtTokens(u.days.reduce((s, d) => s + d.output, 0))} tokens`));
  } else {
    cards.insertAdjacentHTML('beforeend', card('今日消耗', '—', 'muted', '当前账号未配置 Session Token'));
    cards.insertAdjacentHTML('beforeend', card('本月消耗', '—', 'muted', ''));
  }
}

// ── 图表 ──
function chartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: { ticks: { color: '#9ca3af', maxRotation: 0, autoSkip: true, maxTicksLimit: 15 }, grid: { color: '#242418' } },
      y: { ticks: { color: '#9ca3af' }, grid: { color: '#242418' } },
    },
    plugins: { legend: { labels: { color: '#d1d5db', boxWidth: 10 } } },
  };
}

const dailyChart = new Chart($('chart-daily'), {
  type: 'bar',
  data: {
    labels: [],
    datasets: [
      { label: '缓存命中', data: [], backgroundColor: '#22d3ee', stack: 't' },
      { label: '缓存未命中', data: [], backgroundColor: '#22c55e', stack: 't' },
      { label: '输出', data: [], backgroundColor: '#eab308', stack: 't' },
    ],
  },
  options: { ...chartOptions(), scales: { x: { stacked: true, ticks: { color: '#9ca3af' }, grid: { color: '#242418' } }, y: { stacked: true, ticks: { color: '#9ca3af' }, grid: { color: '#242418' } } } },
});

const costChart = new Chart($('chart-cost'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [{ label: '每日费用（元）', data: [], borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,0.15)', fill: true, tension: 0.3, pointRadius: 2 }],
  },
  options: chartOptions(),
});

const modelChart = new Chart($('chart-model'), {
  type: 'doughnut',
  data: { labels: [], datasets: [{ data: [], backgroundColor: PALETTE, borderColor: '#1a1a0e', borderWidth: 2 }] },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    cutout: '55%',
    plugins: {
      legend: { position: 'right', labels: { color: '#d1d5db', boxWidth: 10 } },
      tooltip: {
        callbacks: {
          label: (ctx) => ` ${ctx.label}: ${fmtMoney(ctx.parsed)}（${fmtPct(state.usage ? state.usage.models[ctx.dataIndex].percent : 0)}）`,
        },
      },
    },
  },
});

const fmtTime = (t) => {
  const d = new Date(t);
  const p = (n) => String(n).padStart(2, '0');
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
};

const balanceChart = new Chart($('chart-balance'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: '余额（元）',
      data: [],
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.15)',
      fill: true,
      tension: 0.3,
      pointRadius: 1.5,
    }],
  },
  options: chartOptions(),
});

const monthCostChart = new Chart($('chart-monthcost'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: '本月累计费用（元）',
      data: [],
      borderColor: '#a855f7',
      backgroundColor: 'rgba(168,85,247,0.15)',
      fill: true,
      tension: 0.3,
      pointRadius: 1.5,
    }],
  },
  options: chartOptions(),
});

function renderCharts() {
  const u = state.usage;
  if (!u) return;
  const days = u.days;
  dailyChart.data.labels = days.map((d) => d.date.slice(5));
  dailyChart.data.datasets[0].data = days.map((d) => d.cache_hit);
  dailyChart.data.datasets[1].data = days.map((d) => d.cache_miss);
  dailyChart.data.datasets[2].data = days.map((d) => d.output);
  dailyChart.update();

  costChart.data.labels = days.map((d) => d.date.slice(5));
  costChart.data.datasets[0].data = days.map((d) => d.cost);
  costChart.update();

  modelChart.data.labels = u.models.map((m) => m.model);
  modelChart.data.datasets[0].data = u.models.map((m) => m.cost);
  modelChart.update();
}

// ── 表格 ──
function renderTables() {
  const u = state.usage;
  const mt = $('model-table').querySelector('tbody');
  const dt = $('days-table').querySelector('tbody');
  if (!u) { mt.innerHTML = ''; dt.innerHTML = ''; return; }

  mt.innerHTML = u.models.map((m) =>
    `<tr><td>${m.model}</td><td class="num">${fmtTokens(m.input)}</td><td class="num">${fmtTokens(m.output)}</td>` +
    `<td class="num">${fmtMoney(m.cost)}</td><td class="num">${fmtPct(m.percent)}</td></tr>`).join('');

  dt.innerHTML = u.days.map((d) =>
    `<tr><td>${d.date}</td><td class="num">${fmtTokens(d.input)}</td><td class="num">${fmtTokens(d.cache_hit)}</td>` +
    `<td class="num">${fmtTokens(d.cache_miss)}</td><td class="num">${fmtTokens(d.output)}</td>` +
    `<td class="num">${fmtMoney(d.cost)}</td></tr>`).join('');
}

function renderHistory() {
  const h = state.history;
  if (!h) return;

  // 余额趋势（近 30 天）
  const bal = h.balance || [];
  balanceChart.data.labels = bal.map((r) => fmtTime(r.t));
  balanceChart.data.datasets[0].data = bal.map((r) => Number(r.total));
  balanceChart.update();

  // 本月消耗累计：只展示当前所选月份的快照
  const use = (h.usage || []).filter((r) => r.year === state.year && r.month === state.month);
  monthCostChart.data.labels = use.map((r) => fmtTime(r.t));
  monthCostChart.data.datasets[0].data = use.map((r) => r.month_cost);
  monthCostChart.update();
}

// ── 数据刷新 ──
async function refresh() {
  if (state.busy) return;
  state.busy = true;
  setStatus('刷新中…', 'info');
  try {
    const [bal, use, hist] = await Promise.allSettled([
      window.dsApi.getBalance(),
      window.dsApi.getUsage(state.month, state.year),
      window.dsApi.getHistory({ days: 30 }),
    ]);
    if (bal.status === 'fulfilled') {
      state.balance = bal.value;
    } else if (!state.balance) {
      setStatus(bal.reason.message, 'err');
    }
    if (use.status === 'fulfilled') {
      state.usage = use.value;
    } else if (use.reason && String(use.reason.message).includes('Session Token')) {
      state.usage = null;
      if (!activeAccount() || !activeAccount().hasSessionToken) {
        setStatus('提示：当前账号未配置 Session Token，消耗明细不可用（设置页可填写）', 'info');
      } else {
        setStatus(use.reason.message, 'err');
      }
    } else if (use.reason) {
      state.usage = null;
      setStatus(use.reason.message, 'err');
    }
    if (hist.status === 'fulfilled') state.history = hist.value;
    renderCards();
    renderCharts();
    renderTables();
    renderHistory();
    setLastUpdated();
    if (bal.status === 'fulfilled' || use.status === 'fulfilled') {
      setStatus('刷新完成', 'ok');
    }
  } catch (e) {
    setStatus(e.message || String(e), 'err');
  } finally {
    state.busy = false;
  }
}

function scheduleRefresh() {
  if (state.timer) clearInterval(state.timer);
  const sec = (state.settings && state.settings.refreshInterval) || 60;
  state.timer = setInterval(() => refresh(), sec * 1000);
}

// ── 全局设置（刷新/告警）──
function fillSettingsForm() {
  const s = state.settings || {};
  if (!$('inp-interval').value) $('inp-interval').value = s.refreshInterval || 60;
  $('chk-alert').checked = !!s.alertEnabled;
  $('inp-threshold').value = s.alertThreshold != null ? s.alertThreshold : 5;
}

async function onSave() {
  try {
    const patch = {};
    patch.refreshInterval = Number($('inp-interval').value) || 60;
    patch.alertEnabled = $('chk-alert').checked;
    patch.alertThreshold = Number($('inp-threshold').value) || 0;
    state.settings = await window.dsApi.saveSettings(patch);
    fillSettingsForm();
    scheduleRefresh();
    setStatus('设置已保存', 'ok');
    $('settings-status').textContent = '已更新刷新与告警设置。';
    await refresh();
  } catch (e) {
    setStatus(e.message || String(e), 'err');
  }
}

async function onClear() {
  if (!confirm('确定清除当前账号保存的 API Key / Session Token？')) return;
  try {
    state.settings = await window.dsApi.clearSettings();
    renderAccountSelect();
    renderAccountList();
    state.balance = null;
    state.usage = null;
    renderCards();
    renderCharts();
    renderTables();
    setStatus('已清除当前账号的凭证', 'ok');
    $('settings-status').textContent = '';
  } catch (e) {
    setStatus(e.message || String(e), 'err');
  }
}

async function onClearHistory() {
  if (!confirm('确定清空当前账号的历史数据（余额/消耗快照）？此操作不可恢复。')) return;
  try {
    await window.dsApi.clearHistory();
    state.history = null;
    renderHistory();
    setStatus('当前账号历史数据已清空', 'ok');
    $('settings-status').textContent = '历史数据已清空，曲线将从下次刷新开始重新积累。';
  } catch (e) {
    setStatus(e.message || String(e), 'err');
  }
}

// ── 视图切换 / 月份切换 ──
function switchView(view) {
  $('view-dashboard').hidden = view !== 'dashboard';
  $('view-settings').hidden = view !== 'settings';
  document.querySelectorAll('.nav-btn').forEach((b) => {
    b.classList.toggle('active', b.dataset.view === view);
  });
}

function updateMonthLabel() {
  $('month-label').textContent = `${state.year}-${String(state.month).padStart(2, '0')}`;
}

function shiftMonth(delta) {
  const d = new Date(state.year, state.month - 1 + delta, 1);
  state.year = d.getFullYear();
  state.month = d.getMonth() + 1;
  updateMonthLabel();
  refresh();
}

// ── 事件绑定与启动 ──
function bindEvents() {
  $('btn-refresh').addEventListener('click', () => refresh());
  $('btn-theme').addEventListener('click', onToggleTheme);
  $('btn-prev').addEventListener('click', () => shiftMonth(-1));
  $('btn-next').addEventListener('click', () => shiftMonth(1));
  $('btn-save').addEventListener('click', onSave);
  $('btn-clear').addEventListener('click', onClear);
  $('btn-clear-history').addEventListener('click', onClearHistory);
  $('btn-add-account').addEventListener('click', () => openAccountForm('add'));
  $('btn-acc-save').addEventListener('click', onSaveAccount);
  $('btn-acc-test').addEventListener('click', onTestAccount);
  $('btn-acc-cancel').addEventListener('click', closeAccountForm);
  $('account-select').addEventListener('change', (e) => {
    if (e.target.value) onActivate(e.target.value);
  });
  document.querySelectorAll('.nav-btn').forEach((b) =>
    b.addEventListener('click', () => switchView(b.dataset.view)));
  document.querySelectorAll('a[data-link]').forEach((a) =>
    a.addEventListener('click', (e) => { e.preventDefault(); window.dsApi.openExternal(a.dataset.link); }));
}

async function init() {
  bindEvents();
  updateMonthLabel();
  state.settings = await window.dsApi.getSettings();
  applyTheme(state.settings.theme);
  fillSettingsForm();
  renderAccountSelect();
  renderAccountList();
  scheduleRefresh();
  await refresh();
}

init();
