// 本地历史记录存储：按账号隔离的追加式 JSONL 文件（userData/history-<账号id>.jsonl）
//
// 设计说明：
//   - 零依赖：不使用 SQLite（原生模块在 Electron 中需重编译，引入构建与供应链风险）；
//   - 按账号隔离：每个账号一个文件，余额/消耗曲线互不混淆；
//   - 追加写入：串行写队列（writeChain）保证并发安全，不会覆盖已写数据；
//   - 惰性加载：文件在账号首次读写时才载入内存缓存，启动开销小；
//   - 降采样：图表查询时均匀抽稀到最多 MAX_POINTS 个点，避免长周期曲线过密；
//   - ACL 保护：Windows 下仅当前用户可读写；
//   - 频率控制：余额每 5 分钟、消耗每 10 分钟至少记一条，避免高频噪音。
'use strict';

const fs = require('fs');
const fsp = fs.promises;
const { execFile } = require('child_process');

const MIN_BALANCE_MS = 5 * 60 * 1000;   // 余额快照最小间隔 5 分钟
const MIN_USAGE_MS = 10 * 60 * 1000;    // 消耗快照最小间隔 10 分钟
const CACHE_CAP = 20000;                // 内存缓存上限（条/账号）
const MAX_POINTS = 250;                 // 图表最大点数（降采样目标）

const stores = new Map();               // accountId -> store
let writeChain = Promise.resolve();

function register(accountId, filePath) {
  if (!accountId || stores.has(accountId)) return;
  stores.set(accountId, {
    file: filePath,
    seeded: false,
    lastAppend: { balance: 0, usage: 0 },
    cache: { balance: [], usage: [] },
  });
}

function store(accountId) {
  return stores.get(accountId) || null;
}

function ensureSeeded(s) {
  if (s.seeded) return;
  s.seeded = true;
  try {
    const raw = fs.readFileSync(s.file, 'utf8');
    for (const line of raw.split('\n')) {
      if (!line.trim()) continue;
      try {
        const rec = JSON.parse(line);
        if (rec.kind === 'balance' && s.cache.balance.length < CACHE_CAP) s.cache.balance.push(rec);
        else if (rec.kind === 'usage' && s.cache.usage.length < CACHE_CAP) s.cache.usage.push(rec);
      } catch { /* 跳过损坏行 */ }
    }
  } catch { /* 文件尚不存在 */ }
}

async function applyAcl(file) {
  if (process.platform !== 'win32') return;
  await new Promise((resolve) => {
    execFile('icacls', [file, '/inheritance:r', '/grant:r', `${process.env.USERNAME}:F`], () => resolve());
  });
}

function append(accountId, kind, rec) {
  const s = store(accountId);
  if (!s) return Promise.resolve(false);
  ensureSeeded(s);
  rec.kind = kind;
  rec.t = Date.now();
  s.cache[kind].push(rec);
  if (s.cache[kind].length > CACHE_CAP) s.cache[kind].splice(0, s.cache[kind].length - CACHE_CAP);
  writeChain = writeChain
    .then(() => applyAcl(s.file))
    .then(() => fsp.appendFile(s.file, JSON.stringify(rec) + '\n', 'utf8'))
    .catch((e) => { console.error(`history append failed: ${e.message}`); });
  return writeChain;
}

function maybeAppend(accountId, kind, rec, minMs) {
  const s = store(accountId);
  if (!s) return Promise.resolve(false);
  const now = Date.now();
  if (now - s.lastAppend[kind] < minMs) return Promise.resolve(false);
  s.lastAppend[kind] = now;
  return append(accountId, kind, rec).then(() => true);
}

function appendBalance(accountId, balance) {
  return maybeAppend(accountId, 'balance', {
    total: String(balance.total_balance ?? '0'),
    granted: String(balance.granted_balance ?? '0'),
    topped_up: String(balance.topped_up_balance ?? '0'),
    currency: balance.currency || 'CNY',
    available: balance.is_available !== false,
  }, MIN_BALANCE_MS);
}

function appendUsage(accountId, usage) {
  return maybeAppend(accountId, 'usage', {
    year: usage.year,
    month: usage.month,
    month_cost: usage.month_cost,
    today_cost: usage.today.cost,
    today_input: usage.today.input,
    today_output: usage.today.output,
  }, MIN_USAGE_MS);
}

function downsample(arr, max) {
  if (arr.length <= max) return arr;
  const step = arr.length / max;
  const out = [];
  for (let i = 0; i < max; i++) out.push(arr[Math.floor(i * step)]);
  return out;
}

function getHistory(accountId, opts = {}) {
  const s = store(accountId);
  if (!s) return { balance: [], usage: [], file: null };
  ensureSeeded(s);
  const days = opts.days || 30;
  const cutoff = Date.now() - days * 86400000;
  return {
    balance: downsample(s.cache.balance.filter((r) => r.t >= cutoff), MAX_POINTS),
    usage: downsample(s.cache.usage.filter((r) => r.t >= cutoff), MAX_POINTS),
    file: s.file,
  };
}

async function clearHistory(accountId) {
  const s = store(accountId);
  if (!s) return Promise.resolve();
  s.cache = { balance: [], usage: [] };
  s.seeded = true;
  writeChain = writeChain
    .then(() => fsp.writeFile(s.file, '', 'utf8'))
    .catch((e) => { console.error(`history clear failed: ${e.message}`); });
  return writeChain;
}

module.exports = { register, appendBalance, appendUsage, getHistory, clearHistory };
