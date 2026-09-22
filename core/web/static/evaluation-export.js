/* Shared participant handoff. Only session references stay in browser storage. */
(function (root) {
  'use strict';
  const KEY = 'neurodiscovery.human-evaluations.references.v1';
  function read() {
    const value = JSON.parse(root.localStorage.getItem(KEY) || '[]');
    if (!Array.isArray(value)) throw new Error('Invalid saved evaluation index.');
    return value;
  }
  function remember(kind, code, ref) {
    const rows = read().filter(row => !(row.kind === kind && row.id === ref.id));
    rows.push({kind, code: String(code).trim(), ...ref});
    root.localStorage.setItem(KEY, JSON.stringify(rows));
  }
  function forget(kind, id) {
    root.localStorage.setItem(KEY, JSON.stringify(read().filter(row => !(row.kind === kind && row.id === id))));
  }
  function references(code) {
    const own = read().filter(row => row.code === String(code).trim());
    return {participant_code: String(code).trim(),
      discovery_sessions: own.filter(row => row.kind === 'discovery').map(({id, token}) => ({id, token})),
      ranking_session_ids: own.filter(row => row.kind === 'ranking').map(row => row.id)};
  }
  function bridge() {
    try { return root.neuroclawDesktop || root.parent?.neuroclawDesktop; }
    catch (_) { return root.neuroclawDesktop; }
  }
  async function exportResults({code, token, language = 'zh', rankingStudyId}) {
    // Import the previous client's single resumable HE1 reference when present.
    const previous = JSON.parse(root.localStorage.getItem('discoveryExpertPilotResumeV1') || 'null');
    if (previous?.id && previous?.token && previous.display_code?.trim() === String(code).trim()) {
      remember('discovery', code, {id: previous.id, token: previous.token});
    }
    const response = await root.fetch('/api/studies/evaluations/export', {
      method: 'POST', cache: 'no-store', headers: {'Content-Type': 'application/json', 'X-NeuroOracle-Study-Token': token || ''},
      body: JSON.stringify({...references(code), ...(rankingStudyId ? {ranking_study_ids: [rankingStudyId]} : {})}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    const en = language === 'en', one = payload.human_evaluation_1, two = payload.human_evaluation_2;
    const status = value => ({complete: en ? 'Completed' : '已完成', in_progress: en ? 'In progress' : '未完成', not_included: en ? 'No saved record linked' : '尚无关联的保存记录'}[value] || value);
    // A lightweight review step makes partial exports explicit before file creation.
    const modal = root.document.createElement('dialog');
    modal.className = 'evaluation-export-dialog';
    const title = root.document.createElement('h2'); title.textContent = en ? 'Export and send to the researcher' : '导出并交给研究者';
    const detail = root.document.createElement('p'); detail.className = 'evaluation-export-summary';
    detail.textContent = `${en ? 'Name / code' : '名字/代号'}: ${payload.participant_code}\nHuman Evaluation 1: ${status(one.status)}\nHuman Evaluation 2: ${status(two.status)}${two.required_sessions ? ` (${two.completed_sessions}/${two.required_sessions})` : ''}`;
    const note = root.document.createElement('p');
    note.textContent = payload.completion_status === 'complete'
      ? (en ? 'Both evaluations are saved. Export one JSON file and send it to the researcher.' : '两项评估均已保存。导出一个 JSON 文件，将该文件发给研究者即可。')
      : (en ? 'This is a partial export. You can export a new file after finishing the remaining evaluations. Use the same name/code in both evaluations.' : '当前将导出部分结果。其余评估完成后可重新导出一个完整文件。两项评估请使用相同的名字/代号。');
    const actions = root.document.createElement('div'); actions.className = 'evaluation-actions';
    const cancel = root.document.createElement('button'); cancel.className = 'secondary'; cancel.textContent = en ? 'Cancel' : '取消';
    const download = root.document.createElement('button'); download.className = 'primary'; download.textContent = en ? 'Export JSON file' : '导出 JSON 文件';
    actions.append(cancel, download); modal.append(title, detail, note, actions); root.document.body.append(modal);
    const accepted = await new Promise(resolve => {
      cancel.onclick = () => { modal.close(); resolve(false); };
      download.onclick = () => { modal.close(); resolve(true); };
      modal.oncancel = () => resolve(false);
      modal.showModal();
    });
    modal.remove();
    if (!accepted) return {canceled: true};
    const safeCode = String(code).replace(/[^\p{L}\p{N}._-]+/gu, '-').slice(0, 60) || 'participant';
    const defaultFileName = `NeuroDiscovery-Human-Evaluations-${safeCode}-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    const desktop = bridge();
    if (desktop?.exportUserStudyResults) {
      const result = await desktop.exportUserStudyResults({defaultFileName, payload});
      if (result?.error) throw new Error(result.error);
      return result;
    }
    const url = root.URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2) + '\n'], {type: 'application/json;charset=utf-8'}));
    const link = root.document.createElement('a'); link.href = url; link.download = defaultFileName;
    root.document.body.append(link); link.click(); link.remove(); root.setTimeout(() => root.URL.revokeObjectURL(url), 2000);
    return {downloadRequested: true, filename: defaultFileName};
  }
  root.EvaluationExport = {remember, forget, references, exportResults};
  if (typeof module !== 'undefined') module.exports = {remember, forget, references};
})(typeof window === 'undefined' ? globalThis : window);
