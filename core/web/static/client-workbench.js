/* Native workbench primitives. No provider keys, automatic retries of chat, or scientific policy. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.ClientWorkbench = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const copy = value => JSON.parse(JSON.stringify(value));
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
  const terminal = status => !['running', 'stopping'].includes(status);

  function removeWorkspace(state, projectId, busy) {
    const project = state.projects.find(p => p.id === projectId);
    if (!project) return;
    const sessions = state.sessions.filter(s => s.projectId === projectId);
    if (sessions.some(s => busy(s.id))) throw new Error('Stop or finish active conversations before removing this workspace.');
    sessions.forEach(s => { s.workspacePath = project.workspacePath || s.workspacePath || ''; s.projectId = null; });
    state.projects = state.projects.filter(p => p.id !== projectId);
    if (state.activeProjectId === projectId) state.activeProjectId = null;
  }

  function moveSession(state, sessionId, projectId, busy) {
    const session = state.sessions.find(s => s.id === sessionId);
    if (!session) return;
    if (busy(sessionId)) throw new Error('Stop or finish this conversation before moving it.');
    const previous = state.projects.find(p => p.id === session.projectId);
    const next = state.projects.find(p => p.id === projectId);
    if (projectId && !next) throw new Error('Workspace no longer exists.');
    session.workspacePath = next?.workspacePath || previous?.workspacePath || session.workspacePath || '';
    session.projectId = next?.id || null;
    if (state.activeSessionId === sessionId) state.activeProjectId = session.projectId;
  }

  function createHistory({snapshot, restore, onError, fetcher = (...args) => fetch(...args)}) {
    let revision = 0, ready = false, dirty = false, conflict = false, timer, pending = null;
    async function flush() {
      clearTimeout(timer);
      if (pending) { await pending; if (dirty && !conflict) return flush(); return; }
      if (!ready || !dirty || conflict) return;
      dirty = false;
      const body = {revision, state: copy(snapshot())};
      pending = (async () => {
        try {
          const response = await fetcher('/api/workbench/state', {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
          const data = await response.json();
          if (!response.ok) { conflict = response.status === 409; const error = new Error(data.message || 'History was not saved.'); error.recoveryId=data.recovery_id; throw error; }
          revision = data.revision;
        } catch (error) { dirty = true; onError(error); }
      })();
      await pending;
      pending = null;
      // A failed write waits for the next user action/online event, not a tight retry loop.
    }
    return {
      async load() {
        try {
          const response = await fetcher('/api/workbench/state');
          if (!response.ok) throw new Error('Unable to load local history. Browser backup remains available.');
          const data = await response.json();
          revision = data.revision;
          if (data.state) restore(data.state);
          ready = true;
          if (!data.state) { dirty = true; await flush(); }
        } catch (error) { onError(error); }
      },
      schedule() { dirty = true; clearTimeout(timer); timer = setTimeout(flush, 250); },
      flush,
      get conflict() { return conflict; },
    };
  }

  function dialog({title, message = '', fields = [], confirm = 'Save', cancel = 'Cancel', danger = false, submit, returnFocus}) {
    return new Promise(resolve => {
      const previous = document.activeElement;
      const el = document.createElement('dialog');
      el.className = 'wb-dialog';
      const id = 'wb-dialog-' + Math.random().toString(36).slice(2);
      el.setAttribute('aria-labelledby', id);
      el.innerHTML = `<form><h2 id="${id}">${esc(title)}</h2><p class="wb-dialog-description">${esc(message)}</p><div class="wb-dialog-fields"></div><p class="wb-inline-error" role="alert" hidden></p><footer><button type="button" data-cancel>${esc(cancel)}</button><button type="submit" class="${danger ? 'wb-danger' : 'wb-primary'}">${esc(confirm)}</button></footer></form>`;
      const controls = new Map();
      fields.forEach(field => {
        const label = document.createElement('label');
        label.textContent = field.label;
        const input = document.createElement(field.options ? 'select' : 'input');
        input.name = field.name; input.required = field.required !== false;
        if (field.maxLength) input.maxLength = field.maxLength;
        if (field.placeholder) input.placeholder = field.placeholder;
        (field.options || []).forEach(option => { const item = document.createElement('option'); item.value = option.value; item.textContent = option.label; input.append(item); });
        input.value = field.value || '';
        label.append(input);
        el.querySelector('.wb-dialog-fields').append(label);
        controls.set(field.name, input);
      });
      let saving = false, answer = null;
      const error = el.querySelector('[role="alert"]');
      el.querySelector('[data-cancel]').onclick = () => { if (!saving) el.close(); };
      el.addEventListener('cancel', event => { if (saving) event.preventDefault(); });
      el.addEventListener('click', event => { if (event.target === el && !saving) { const r = el.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) el.close(); } });
      el.querySelector('form').onsubmit = async event => {
        event.preventDefault();
        if (saving) return;
        const values = Object.fromEntries([...controls].map(([name, control]) => [name, control.value.trim()]));
        if (fields.some(field => field.required !== false && !values[field.name])) { controls.get(fields.find(field => !values[field.name]).name)?.focus(); return; }
        saving = true; error.hidden = true;
        el.querySelectorAll('button, input, select').forEach(control => control.disabled = true);
        try { answer = submit ? await submit(values) : values; if (answer === undefined) answer = values; el.close(); }
        catch (err) { error.textContent = err.message || String(err); error.hidden = false; }
        finally { saving = false; el.querySelectorAll('button, input, select').forEach(control => control.disabled = false); }
      };
      el.addEventListener('close', () => { el.remove(); const focus = previous?.isConnected && previous !== document.body ? previous : returnFocus?.(); focus?.focus({preventScroll:true}); resolve(answer); }, {once:true});
      document.body.append(el); el.showModal();
      (danger ? el.querySelector('[data-cancel]') : controls.values().next().value || el.querySelector('[type="submit"]')).focus();
    });
  }

  function reduceEvent(state, event) {
    if (event.request_id !== state.request_id || event.chat_id !== state.chat_id || event.seq <= state.seq) return state;
    state.seq = event.seq;
    if (['model_start','text','reasoning','model_end','usage'].includes(event.type)) {
      let block = state.blocks.find(item => item.call_id === event.call_id);
      if (!block) { block = {call_id:event.call_id, text:'', reasoning:'', status:'running'}; state.blocks.push(block); }
      if (event.type === 'text' || event.type === 'reasoning') {
        const key = event.type === 'text' ? 'text' : 'reasoning';
        block[key] = (block[key] + (event.text || '')).slice(0, 1000000);
      } else ['model','source','status'].forEach(key => { if (event[key] !== undefined) block[key] = event[key]; });
      if (event.type === 'usage') block.usage = event.usage;
    }
    if (['tool_start','tool_end'].includes(event.type)) {
      let tool = state.tools.find(item => item.tool_id === event.tool_id);
      if (!tool) { tool = {}; state.tools.push(tool); }
      Object.assign(tool, event);
    }
    if (event.type === 'finish' || event.type === 'status') state.status = event.status;
    return state;
  }

  async function followRun(requestId, chatId, {onUpdate, onConnection, fetcher = (...args) => fetch(...args)} = {}) {
    let execution = {request_id:requestId, chat_id:chatId, seq:0, status:'running', blocks:[], tools:[]};
    let failures = 0;
    for (;;) {
      let response, data;
      try {
        response = await fetcher(`/api/chat/runs/${encodeURIComponent(requestId)}?after=${execution.seq}`);
        data = await response.json();
        if (!response.ok) {
          if ([400,404,410].includes(response.status)) return {type:'error', message:data.message, execution};
          throw new Error(data.message || 'Reconnecting');
        }
        failures = 0; onConnection?.(true);
      } catch (_error) { failures += 1; onConnection?.(false); await wait(Math.min(5000, 500 * failures)); continue; }
      if (data.snapshot && data.snapshot.request_id === requestId && data.snapshot.chat_id === chatId) execution = data.snapshot;
      (data.events || []).forEach(event => reduceEvent(execution, event));
      execution.status = data.status;
      onUpdate?.(execution);
      if (terminal(data.status)) {
        const {result: _savedResult, ...view} = execution;
        return {...(data.result || {type:'error', message:'Request ended without a result.'}), execution:view};
      }
      await wait(350);
    }
  }

  function statusLabel(status, zh) {
    const labels = {running:['Running','运行中'], stopping:['Stopping…','停止中…'], completed:['Completed','已完成'], failed:['Failed','失败'], cancelled:['Cancelled','已取消'], interrupted:['Interrupted','中断'], blocked:['Blocked','阻塞'], stalled:['Needs attention','需要处理'], budget_exhausted:['Budget reached','达到预算']};
    return labels[status]?.[zh ? 1 : 0] || status;
  }

  function executionView(host, execution, {zh = false, live = false} = {}) {
    if (!execution) return;
    const nearBottom = host.parentElement ? host.parentElement.scrollHeight - host.parentElement.scrollTop - host.parentElement.clientHeight < 110 : false;
    if (!host.querySelector('.wb-execution-status')) host.innerHTML = '<div class="wb-execution-status" role="status"></div><div class="wb-execution-blocks"></div>';
    host.classList.add('wb-execution');
    host.querySelector('.wb-execution-status').textContent = statusLabel(execution.status, zh);
    const parent = host.querySelector('.wb-execution-blocks');
    function details(key, label, text, running, defaultOpen) {
      let node = [...parent.children].find(child => child.dataset.key === key);
      if (!node) {
        node = document.createElement('details'); node.dataset.key = key;
        node.innerHTML = '<summary></summary><pre></pre>';
        node.open = Boolean(defaultOpen);
        node.querySelector('summary').addEventListener('click', () => node.dataset.touched = '1');
        parent.append(node);
      }
      node.classList.toggle('wb-running', running);
      node.querySelector('summary').textContent = label;
      node.querySelector('pre').textContent = text;
      if (!running && node.dataset.touched !== '1') node.open = false;
    }
    (execution.blocks || []).forEach(block => {
      const running = live && block.status === 'running' && !terminal(execution.status);
      const origin = block.source && block.source !== 'main' ? ` · ${block.source}` : '';
      const name = `${block.model || (zh ? '模型' : 'Model')}${origin}`;
      if (block.reasoning) details(`reasoning:${block.call_id}`, `${zh ? '模型提供的思考摘要' : 'Provider reasoning'} · ${name}`, block.reasoning, running, running);
      if (block.text) details(`text:${block.call_id}`, `${zh ? '生成内容' : 'Output'} · ${name}`, block.text, running, running);
    });
    (execution.tools || []).forEach(tool => {
      const running = live && tool.status === 'running' && !terminal(execution.status);
      details(`tool:${tool.tool_id}`, `${tool.tool || 'Tool'} · ${statusLabel(running ? 'running' : tool.status === 'running' ? execution.status : tool.status, zh)}`,
        [tool.command, tool.output, tool.error].filter(Boolean).join('\n\n'), running, running);
    });
    if (live && nearBottom) host.parentElement.scrollTop = host.parentElement.scrollHeight;
  }

  function countLabel(data, key, zh = false) {
    const totals = data?.totals;
    if (!totals || (totals.requests && !totals.known_fields?.[key])) return zh ? '未上报' : 'Not reported';
    const partial = totals.known_fields?.[key] < totals.requests;
    return Number(totals[key] || 0).toLocaleString() + (partial ? (zh ? '（部分）' : ' (partial)') : '');
  }

  function executionUsage(execution) {
    const blocks=execution?.blocks || [];
    const keys=['input_tokens','output_tokens','cache_read_tokens','cache_write_tokens','reasoning_tokens'];
    const totals={requests:blocks.length,known_fields:{}};
    keys.forEach(key=>{
      const values=blocks.map(block=>block.usage?.[key]).filter(value=>Number.isInteger(value) && value>=0);
      totals[key]=values.reduce((sum,value)=>sum+value,0);totals.known_fields[key]=values.length;
    });
    return {totals};
  }

  function usageDetails(data, zh = false) {
    const labels = {input_tokens:zh?'输入（含缓存）':'Input (including cache)', output_tokens:zh?'输出':'Output', cache_read_tokens:zh?'缓存读取':'Cache read', cache_write_tokens:zh?'缓存写入':'Cache write', reasoning_tokens:zh?'推理 token（输出子集）':'Reasoning tokens (output subset)'};
    return `<div class="wb-usage-grid">${Object.entries(labels).map(([key,label]) => `<div><span>${esc(label)}</span><strong>${esc(countLabel(data,key,zh))}</strong></div>`).join('')}</div><p class="wb-note">${zh?'累计用量不是上下文占用。缓存详情不重复相加；未上报不视为零。':'Cumulative usage is not context occupancy. Cache details are not added twice; missing usage is not zero.'}</p>`;
  }

  function csv(rows) {
    const keys = ['call_id','request_id','chat_id','project_id','provider','model','source','started','status','input_tokens','output_tokens','cache_read_tokens','cache_write_tokens','reasoning_tokens','reported'];
    const cell = value => '"' + String(value ?? '').replace(/^[=+@\-\t\r]/, match => "'" + match).replace(/"/g,'""') + '"';
    return [keys, ...rows.map(row => keys.map(key => row[key]))].map(row => row.map(cell).join(',')).join('\r\n');
  }

  async function usagePanel(host, {zh = false, projects = [], fetcher = (...args) => fetch(...args)} = {}) {
    host.innerHTML = `<h2>${zh?'使用量':'Usage'}</h2><p class="wb-note">${zh?'本客户端真实 API 调用记录，包含重试、标题和子任务。不同于供应商账户总用量。':'Actual API calls from this client, including retries, titles and subagents. Separate from provider account totals.'}</p><div class="wb-usage-filters"></div><div class="wb-usage-results" role="status"></div>`;
    const filters = host.querySelector('.wb-usage-filters'), results = host.querySelector('.wb-usage-results');
    const selects = {};
    const addSelect = (key, label, options) => {
      const field = document.createElement('label'); field.textContent = label;
      const select = document.createElement('select'); field.append(select); filters.append(field); selects[key] = select;
      options.forEach(([value,text]) => { const opt = document.createElement('option'); opt.value=value; opt.textContent=text; select.append(opt); });
      select.onchange = refresh;
    };
    addSelect('days', zh?'时间':'Period', [['0',zh?'全部':'All time'],['1',zh?'24 小时':'24 hours'],['7',zh?'7 天':'7 days'],['30',zh?'30 天':'30 days']]);
    addSelect('provider', zh?'提供方':'Provider', [['',zh?'全部':'All']]);
    addSelect('model', zh?'模型':'Model', [['',zh?'全部':'All']]);
    addSelect('project_id', zh?'工作区':'Workspace', [['',zh?'全部':'All'], ...projects.map(p => [p.id,p.name])]);
    const refreshButton = document.createElement('button'); refreshButton.textContent = zh?'刷新':'Refresh'; refreshButton.onclick = refresh; filters.append(refreshButton);
    const exportButton = document.createElement('button'); exportButton.textContent = zh?'导出 CSV':'Export CSV'; filters.append(exportButton);
    let serial = 0, currentRows = [];
    exportButton.onclick = () => { const url = URL.createObjectURL(new Blob(['\ufeff' + csv(currentRows)], {type:'text/csv;charset=utf-8'})); const a=document.createElement('a'); a.href=url; a.download='neurodiscovery-usage.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(url),1000); };
    async function refresh() {
      const request = ++serial; refreshButton.disabled = true; exportButton.disabled = true;
      try {
        const query = new URLSearchParams(Object.entries(selects).map(([key,select]) => [key,select.value]));
        const response = await fetcher('/api/workbench/usage?' + query); const data = await response.json();
        if (!response.ok) throw new Error(data.message || 'Unable to read usage');
        if (request !== serial || !host.isConnected) return;
        currentRows = data.calls || [];
        ['provider','model'].forEach(key => [...new Set(currentRows.map(row => row[key]))].sort().forEach(value => {
          if (![...selects[key].options].some(opt => opt.value === value)) { const option=document.createElement('option'); option.value=value; option.textContent=value; selects[key].append(option); }
        }));
        const t = data.totals;
        results.innerHTML = usageDetails(data,zh) + `<p class="wb-note">${zh?'请求':'Requests'} ${t.requests} · ${zh?'完成':'Completed'} ${t.completed} · ${zh?'失败':'Failed'} ${t.failed} · ${zh?'取消':'Cancelled'} ${t.cancelled} · ${zh?'未完整上报':'Unreported'} ${t.unreported}</p><p class="wb-note">${zh?'费用 / 账户余额：未提供，不作估算。':'Cost / account balance: not provided; no estimate.'} · ${esc(new Date(data.updated_at*1000).toLocaleString())}</p><div class="wb-table-scroll"><table><thead><tr>${(zh?['时间','模型 / 来源','状态','输入','输出']:['Time','Model / source','Status','Input','Output']).map(label=>`<th>${label}</th>`).join('')}</tr></thead><tbody>${currentRows.slice(0,200).map(row=>`<tr><td>${esc(new Date(row.started*1000).toLocaleString())}</td><td>${esc(row.provider)} / ${esc(row.model)}<small>${esc(row.source)}</small></td><td>${esc(statusLabel(row.status,zh))}</td><td>${esc(row.input_tokens ?? (zh?'未上报':'Unknown'))}</td><td>${esc(row.output_tokens ?? (zh?'未上报':'Unknown'))}</td></tr>`).join('')}</tbody></table></div><p class="wb-note">${zh?'表格最多显示最近 200 条；导出包含全部筛选记录。':'Table shows the latest 200 rows; export includes all filtered records.'}</p>`;
      } catch(error) { if (request === serial) results.textContent = error.message; }
      finally { if (request === serial) { refreshButton.disabled = false; exportButton.disabled = false; } }
    }
    await refresh();
  }

  return {removeWorkspace,moveSession,createHistory,dialog,reduceEvent,followRun,executionView,executionUsage,countLabel,usageDetails,usagePanel,csv,esc,terminal};
});
