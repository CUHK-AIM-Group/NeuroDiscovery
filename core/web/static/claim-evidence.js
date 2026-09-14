/* Accepted shared claims, with complete paper-level observations. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const s = { active: false, offset: 0, limit: 30, total: 0, revision: null, selected: null, data: null, request: 0, detailRequest: 0, timer: null, hooks: null };
  const tr = (en, zh) => s.hooks?.language() === "zh" ? zh : en;
  const badge = (text, kind = "") => `<span class="ce-badge ${kind}">${esc(text)}</span>`;
  const roles = {
    own_result: ["Own study result", "本研究结果"], review_synthesis: ["Review / synthesis", "综述／综合证据"],
    background_assertion: ["Background assertion", "背景陈述"], hypothesis_proposal: ["Hypothesis", "假说"],
    methods_statement: ["Methods statement", "方法陈述"], not_reviewed: ["Not reviewed", "尚未审核"]
  };
  const supports = {
    supports: ["Supports this claim", "支持该主张"], does_not_support: ["Does not support this claim", "不支持该主张"],
    hypothesis_only: ["Hypothesis only", "仅为假说"], not_reviewed: ["Support not reviewed", "支持内容尚未审核"],
    context_only: ["Context only", "仅提供背景信息"]
  };
  const label = (map, value) => map[value] ? tr(...map[value]) : String(value || tr("Unknown", "未知"));
  const formatted = value => typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
  async function api(path, params = {}) {
    const url = new URL(path, location.origin);
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
    const response = await fetch(url, { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) { const error = new Error(data.error || response.statusText); error.status = response.status; error.data = data; throw error; }
    return data;
  }
  function mountView() {
    const view = document.createElement("section");
    view.id = "claimEvidenceView"; view.className = "claim-evidence-view"; view.hidden = true;
    view.innerHTML = `<aside class="ce-sidebar"><div class="ce-search"><h2 id="ceHeading"></h2><p class="ce-muted" id="ceHint"></p>
      <input id="ceSearch" type="search" autocomplete="off"><label><span id="ceFilterLabel"></span><select id="ceMinimum"><option value="2">2+</option><option value="3">3+</option><option value="5">5+</option><option value="0"></option></select><button class="ce-action" id="ceRefresh" type="button"></button></label></div>
      <div class="ce-results" id="ceResults" aria-live="polite"></div><div class="ce-pagination"><button id="cePrevious" type="button"></button><span id="cePage"></span><button id="ceNext" type="button"></button></div></aside>
      <main class="ce-detail" id="ceDetail" tabindex="-1" aria-live="polite"></main>`;
    document.body.append(view);
    const button = document.createElement("button"); button.id = "claimEvidenceToggle"; button.type = "button"; button.className = "icon-btn primary";
    document.querySelector(".header-actions").prepend(button);
    button.addEventListener("click", () => setActive(!s.active));
    $("ceSearch").addEventListener("input", () => { clearTimeout(s.timer); s.offset = 0; s.timer = setTimeout(search, 250); });
    $("ceSearch").addEventListener("keydown", e => { if (e.key === "Enter") { clearTimeout(s.timer); search(); } });
    $("ceMinimum").addEventListener("change", () => { s.offset = 0; search(); });
    $("ceRefresh").addEventListener("click", () => { search(); if (s.selected) openClaim(s.selected); });
    $("cePrevious").addEventListener("click", () => { s.offset = Math.max(0, s.offset - s.limit); search(); });
    $("ceNext").addEventListener("click", () => { s.offset += s.limit; search(); });
    document.addEventListener("click", e => {
      const open = e.target.closest("[data-open-claim]");
      if (open) { e.preventDefault(); setActive(true, false); openClaim(open.dataset.openClaim); }
      const result = e.target.closest("[data-shared-claim]");
      if (result) openClaim(result.dataset.sharedClaim);
      if (e.target.closest("[data-ce-retry]")) { search(); if (s.selected) openClaim(s.selected); }
      if (e.target.closest("[data-ce-export]") && s.data) {
        const blob = new Blob([JSON.stringify(s.data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob), a = document.createElement("a"); a.href = url; a.download = `${s.data.shared_claim_id.replace(/[^\w-]/g, "_")}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    });
    new MutationObserver(() => { translate(); if (s.active) { search(); if (s.data) renderDetail(s.data); } }).observe(document.documentElement, { attributes: true, attributeFilter: ["lang"] });
    translate();
  }
  function translate() {
    $("ceHeading").textContent = tr("Shared claims", "共同主张");
    $("ceHint").textContent = tr("Search a scientific claim, paper, or original claim ID.", "搜索科学主张、论文或原 claim ID。");
    $("ceSearch").placeholder = tr("Concept, PMID, CLM:… or REL:…", "概念、PMID、CLM:… 或 REL:…");
    $("ceSearch").setAttribute("aria-label", tr("Search current claim evidence", "搜索当前主张证据"));
    $("ceFilterLabel").textContent = tr("Reviewed supporting papers", "获审支持论文");
    $("ceMinimum").setAttribute("aria-label", $("ceFilterLabel").textContent);
    $("ceMinimum").lastElementChild.textContent = tr("All shared claims", "全部共同主张");
    $("cePrevious").textContent = tr("Previous", "上一页"); $("ceNext").textContent = tr("Next", "下一页");
    $("ceRefresh").textContent = tr("Refresh", "刷新");
    $("claimEvidenceToggle").textContent = s.active ? tr("Concept graph", "概念关系图") : tr("Shared claims & papers", "共同主张与论文");
  }
  function setActive(active, refresh = true) {
    s.active = active; document.body.classList.toggle("claim-evidence-active", active); $("claimEvidenceView").hidden = !active; translate();
    const url = new URL(location.href); url.searchParams.set("view", active ? "claims" : "concepts"); history.replaceState(null, "", url);
    if (active) { if (refresh) search(); if (!s.data) showEmpty(); }
    else s.hooks.legacyStart();
  }
  function showEmpty() {
    $("ceDetail").innerHTML = `<div class="ce-empty"><h2>${tr("One claim, all its papers", "一个主张，查看全部论文")}</h2><p class="ce-muted">${tr("Select a claim to compare its original observations, populations, statistics and evidence roles. Papers are deduplicated; their independence is not assumed.", "选择左侧主张，逐篇查看原句、人群、统计和证据角色。论文已按确认的版本去重，篇数不代表独立研究数。")}</p></div>`;
  }
  function showError(error, target = "ceDetail") {
    const missing = error.status === 404;
    if (target === "ceResults") {
      $("statsLine").textContent = tr("Evidence temporarily unavailable", "证据暂不可用");
      $("cePage").textContent = "—";
      $("cePrevious").disabled = true;
      $("ceNext").disabled = true;
    }
    $(target).innerHTML = `<div class="ce-empty"><h2>${missing ? tr("Claim not found", "未找到该 claim") : tr("Evidence temporarily unavailable", "证据暂不可用")}</h2><p class="ce-muted">${missing ? tr("This ID is not present in the accepted current graph.", "当前已验收图中没有这个 ID。") : tr("The graph may be updating or its validation may no longer match. Retry to read the accepted revision.", "图谱可能正在更新，或当前文件与验收记录不一致。请重试读取已验收版本。")}</p><button class="ce-action" data-ce-retry type="button">${tr("Retry", "重试")}</button></div>`;
    if (target === "ceDetail") s.data = null;
  }
  async function search() {
    if (!s.active) return;
    const request = ++s.request, q = $("ceSearch").value.trim();
    if (/^(CLM|REL):/.test(q)) { openClaim(q); return; }
    $("ceResults").innerHTML = `<p class="ce-muted" style="padding:15px">${tr("Reading current evidence…", "正在读取当前证据…")}</p>`;
    try {
      const data = await api("/api/kg/shared-claims", { q, minimum_papers: $("ceMinimum").value, offset: s.offset, limit: s.limit });
      if (request !== s.request || !s.active) return;
      s.total = data.total; s.revision = data.graph_revision;
      $("statsLine").textContent = tr(`${data.total} matching shared claims`, `${data.total} 个匹配的共同主张`);
      $("ceResults").innerHTML = data.claims.map(c => `<button class="ce-result" type="button" data-shared-claim="${esc(c.shared_claim_id)}" aria-current="${s.data?.shared_claim_id === c.shared_claim_id}">
        <strong>${esc(c.claim.subject_name)}</strong><span class="ce-predicate">${esc(c.claim.predicate)} →</span><strong>${esc(c.claim.object_name)}</strong><div class="ce-badges">${badge(tr(`${c.reviewed_supporting_article_count} supporting papers`, `${c.reviewed_supporting_article_count} 篇获审支持论文`), "ce-support")}${badge(tr(`${c.article_count} source works total`, `共 ${c.article_count} 项来源`))}</div></button>`).join("") || `<p class="ce-muted" style="padding:15px">${tr("No claims match this search and paper threshold.", "没有符合搜索词和论文数量条件的主张。")}</p>`;
      $("cePage").textContent = data.total ? `${s.offset + 1}–${Math.min(s.offset + s.limit, data.total)} / ${data.total}` : "0 / 0";
      $("cePrevious").disabled = s.offset === 0; $("ceNext").disabled = s.offset + s.limit >= data.total;
      if (s.data && (s.data.graph_revision !== data.graph_revision || s.data.claim_layer_revision !== data.claim_layer_revision)) openClaim(s.selected);
    } catch (error) { if (request === s.request && s.active) { showError(error, "ceResults"); showError(error); } }
  }
  async function openClaim(id) {
    setActive(true, false); const request = ++s.detailRequest; s.selected = id; s.data = null;
    $("ceDetail").innerHTML = `<div class="ce-empty">${tr("Reading all papers and original observations…", "正在读取全部论文和原始观察…")}</div>`;
    try {
      const data = await api("/api/kg/claim-evidence", id.startsWith("CLM:") ? { claim_id: id } : { relation_id: id });
      if (request !== s.detailRequest || !s.active) return;
      s.data = data; renderDetail(data);
      document.querySelectorAll("[data-shared-claim]").forEach(el => el.setAttribute("aria-current", String(el.dataset.sharedClaim === data.shared_claim_id)));
      const url = new URL(location.href); url.searchParams.delete("claim"); url.searchParams.delete("relation"); url.searchParams.set(id.startsWith("CLM:") ? "claim" : "relation", id); history.replaceState(null, "", url);
      $("ceDetail").scrollTop = 0;
    } catch (error) { if (request === s.detailRequest && s.active) showError(error); }
  }
  function bibliographyLinks(b) {
    const links = [];
    if (/^\d+$/.test(String(b.pmid || ""))) links.push(`<a class="ce-badge" href="https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(b.pmid)}/" target="_blank" rel="noopener noreferrer">PMID ${esc(b.pmid)} ↗</a>`);
    if (/^10\.\d{4,9}\//.test(String(b.doi || ""))) links.push(`<a class="ce-badge" href="https://doi.org/${encodeURIComponent(b.doi)}" target="_blank" rel="noopener noreferrer">DOI ↗</a>`);
    return links.join("");
  }
  function observation(o) {
    const evidence = o.evidence || {}, original = o.original_claim || o.source_derived_claim || {};
    const supplemental = o.observation_origin === "supplemental_own_abstract";
    const review = o.source_review || {};
    const scopeNotes = [...new Set([review.source_specific_scope_note, review.scope_note].filter(v => typeof v === "string" && v.trim()))];
    const rows = [[tr("Study type (original record)", "研究类型（原记录）"), evidence.study_type], [tr("Method", "方法"), evidence.methodology], ["n", evidence.sample_size], ["p", evidence.p_value], [evidence.effect_metric || tr("Effect", "效应"), evidence.effect_size], [tr("Direction", "方向"), evidence.direction], [tr("Population", "人群"), o.population], [tr("Conditions", "条件"), o.conditions]];
    const kept = rows.filter(([,v]) => v !== null && v !== undefined && v !== "" && (!Array.isArray(v) || v.length));
    return `<section class="ce-observation"><div class="ce-badges">${supplemental ? badge(tr("Supplemental source evidence", "补充来源证据")) : ""}${badge(label(roles, o.observation_role))}${badge(label(supports, o.proposition_support), o.proposition_support === "supports" && !o.negated ? "ce-support" : "ce-nonsupport")}${o.negated ? badge(tr("Negated observation", "否定观察"), "ce-nonsupport") : ""}</div>
      ${o.raw_text ? `<blockquote>${esc(o.raw_text)}</blockquote>` : `<p class="ce-muted">${tr("No original sentence is stored for this observation.", "这条观察没有保存原句。")}</p>`}
      ${kept.length ? `<dl>${kept.map(([k,v]) => `<dt>${esc(k)}</dt><dd>${esc(formatted(v))}</dd>`).join("")}</dl>` : ""}
      ${review.source_anchor || scopeNotes.length ? `<details><summary>${tr("Reviewed source passage and scope", "已核对的来源原句与支持范围")}</summary>${review.source_anchor ? `<blockquote>${esc(review.source_anchor)}</blockquote>` : ""}${scopeNotes.map(note => `<p class="ce-muted">${esc(note)}</p>`).join("")}</details>` : ""}
      <details><summary>${supplemental ? tr("Supplemental source record", "补充来源记录") : tr("Original claim and study fields", "原 claim 与研究字段")}</summary><code>${esc(o.claim_id)}</code><pre>${esc(JSON.stringify({ subject: original.subject_name, predicate: original.predicate, object: original.object_name, evidence, population: o.population, conditions: o.conditions }, null, 2))}</pre></details></section>`;
  }
  function renderDetail(data) {
    const c = data.claim;
    $("ceDetail").innerHTML = `<p class="ce-muted">${tr("Current shared scientific claim", "当前共同科学主张")}</p>
      <div class="ce-triple"><div class="ce-endpoint">${esc(c.subject_name)}</div><div class="ce-relation">${esc(c.predicate)}<span>→</span></div><div class="ce-endpoint">${esc(c.object_name)}</div></div>
      <div class="ce-summary">${badge(tr(`${data.reviewed_supporting_article_count} reviewed supporting papers`, `${data.reviewed_supporting_article_count} 篇获审支持论文`), "ce-support")}${badge(tr(`${data.article_count} source works total`, `共 ${data.article_count} 项来源`))}${badge(tr(`${data.observation_count} original observations`, `${data.observation_count} 条原始观察`))}</div>
      <p class="ce-muted">${tr("Counts combine reviewed support from results, reviews and background assertions. Paper count does not establish independent replication or consensus.", "支持篇数包含经审核的研究结果、综述和背景陈述。论文篇数不代表独立重复验证或科学共识。")}</p>
      ${data.canonical_scope_note ? `<details><summary>${tr("Scope of this shared claim", "这一共同主张的支持范围")}</summary><p class="ce-muted">${esc(data.canonical_scope_note)}</p></details>` : ""}
      <details class="ce-ids"><summary>${tr("Shared ID and original claim IDs", "共同 ID 与原 claim ID")}</summary><code>${esc(data.shared_claim_id)}</code>${data.original_claim_ids.map(id => `<code>${esc(id)}</code>`).join("")}</details>
      <div class="ce-toolbar"><h2>${tr("Paper-level evidence", "逐篇论文证据")}</h2><button type="button" class="ce-action" data-ce-export>${tr("Export evidence", "导出证据")}</button></div>
      ${data.papers.map((p,i) => { const b = p.bibliography, status = p.publication_review?.status; return `<article class="ce-paper" data-paper-key="${esc(p.work_key)}"><h3>${i + 1}. ${esc(b.title || tr("Untitled source", "无题名来源"))}</h3><p class="ce-muted">${esc([b.authors, b.journal, b.year].filter(v => v !== undefined && v !== null && v !== "").join(" · "))}</p><div class="ce-badges">${bibliographyLinks(b)}${badge(p.supports_reviewed_proposition ? tr("Reviewed support", "支持内容已审核") : tr("Not counted as reviewed support", "未计入获审支持"), p.supports_reviewed_proposition ? "ce-support" : "ce-nonsupport")}${!p.verified ? badge(tr("Publication identity unverified", "论文身份未核验"), "ce-nonsupport") : ""}${status === "retracted" ? badge(tr("Retracted", "已撤稿"), "ce-nonsupport") : ""}</div>
      ${p.publication_versions.length > 1 ? `<details class="ce-ids"><summary>${tr(`${p.publication_versions.length} publication versions · counted once`, `${p.publication_versions.length} 个发表版本 · 合计一篇`)}</summary>${p.publication_versions.map(v => `<div>${esc(v.bibliography.title)} ${bibliographyLinks(v.bibliography)}</div>`).join("")}</details>` : ""}${p.observations.map(observation).join("")}</article>`; }).join("")}`;
  }
  async function mount(hooks) {
    s.hooks = hooks; mountView(); const params = new URLSearchParams(location.search);
    const direct = params.get("claim") || params.get("relation");
    try {
      const status = await api("/api/kg/evidence-status");
      if (!status.configured) { $("claimEvidenceToggle").hidden = true; hooks.legacyStart(); return; }
      s.revision = status.graph_revision;
      if (params.get("view") === "concepts" && !direct) { setActive(false); return; }
      setActive(true); if (direct) openClaim(direct);
    } catch (error) { setActive(true, false); showError(error); showError(error, "ceResults"); }
  }
  window.ClaimEvidenceExplorer = {
    get active() { return s.active; }, mount, openClaim,
    claimButton: id => typeof id === "string" && id.startsWith("CLM:") ? `<button type="button" class="ce-action ce-full-evidence" data-open-claim="${esc(id)}">${tr("All papers & evidence", "全部论文与证据")} →</button>` : ""
  };
})();
