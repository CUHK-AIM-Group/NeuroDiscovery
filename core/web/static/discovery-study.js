"use strict";
const $ = (s) => document.querySelector(s);
const escapeHTML = (x) => String(x ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const h = escapeHTML;
const API = "/api/studies/discovery";
const STORAGE = "discoveryExpertPilotResumeV1";
let config, data, credentials, current = 0, drafts = {}, dirty = new Set(), saving = null, pending = null;
let saveTimer, toastTimer, activeSeconds = 0, lastActivity = Date.now(), authToken = "", hostVisible = true;
let referenceNotes = {};
let definitionPlain = {};
let significance = {};
const PARTICIPANT_CODE_KEY = "neurodiscovery.expert-study.participant-code.v1";
const domainName = (x) => ({ADHD:"ADHD",bipolar:"双相",psychosis_SZ_SZA:"SZ/SZA",psychosis:"精神病",SCHZ:"SCHZ",SZ:"SZ",nonaffective_psychosis:"非情感性精神病",affective_psychosis:"情感性精神病"}[x] || x);
function comparisonName(r) {
  if(r.cohort==="UCLA" && r.domain==="psychosis_SZ_SZA") return "SCHZ";
  if(r.variant==="sensitivity_strict_SZ") return "SZ（不含 SZA）";
  if(r.domain==="non_affective_psychosis") return "非情感性精神病";
  return domainName(r.domain);
}
const variantName=x=>({primary_FD:"主要模型 · 调整 FD",sensitivity_no_FD:"敏感性 · 不调整 FD",primary_SZ_SZA:"主要模型 · SZ/SZA",sensitivity_strict_SZ:"敏感性 · 仅 SZ",primary_non_affective:"主要 · 非情感性早期精神病",secondary_affective_context:"背景 · 情感性早期精神病（非双相验证）"}[x]||x);
const f = (x, digits=3) => typeof x === "number" ? x.toFixed(digits) : "—";
const pc = (x) => typeof x === "number" ? `${(100*x).toFixed(2)}%` : "—";
const singleRound = (meta=data.meta) => meta.review_flow === "single_round";
const multiTopic = (meta=data?.meta) => meta?.material_layout === "multitopic";
const reviewPhase = () => data.session.stage === "complete" ? (singleRound() ? "review" : "B") : data.session.stage;

function message(text) { clearTimeout(toastTimer); $("#message").textContent=text; $("#message").hidden=false; toastTimer=setTimeout(()=>$("#message").hidden=true,8500); }
function saveStatus(text, error=false) { $("#save-status").textContent=text; $("#save-status").classList.toggle("error",error); }
function studyStorage() { try { return window.parent.sessionStorage; } catch { return sessionStorage; } }
try { authToken = studyStorage().getItem("neurodiscoveryStudyToken") || ""; } catch { /* stand-alone mode */ }

async function request(path, options={}) {
  const headers = {"Content-Type":"application/json", "X-NeuroOracle-Study-Token":authToken, ...(credentials ? {"X-Discovery-Session-Token":credentials.token} : {})};
  const response = await fetch(path,{...options,headers,cache:"no-store"});
  const body = await response.json();
  if (!response.ok) {
    const error = new Error(body.error || body.detail || `请求失败 (${response.status})`);
    error.status=response.status;
    if(response.status===401) { $("#auth-form").hidden=false; if(data)message("研究入口需要重新解锁。请保存继续码后回到入口，原答案仍保留。"); }
    throw error;
  }
  return body;
}
const uid=()=>crypto.randomUUID();
const sessionPath=(suffix="")=>`${API}/sessions/${encodeURIComponent(credentials.id)}${suffix}`;
function storeCredentials() { localStorage.setItem(STORAGE,JSON.stringify(credentials)); }
function readCredentials() { try { return JSON.parse(localStorage.getItem(STORAGE)); } catch { return null; } }
function dialog(title,text,label="确认") {
  return new Promise(resolve=>{
    $("#dialog-title").textContent=title; $("#dialog-text").textContent=text; $("#dialog-ok").textContent=label;
    const modal=$("#confirm-dialog");
    const finish=value=>{modal.close(); resolve(value);};
    $("#dialog-ok").onclick=()=>finish(true); $("#dialog-cancel").onclick=()=>finish(false);
    modal.oncancel=event=>{event.preventDefault();finish(false);}; modal.showModal();
  });
}

async function loadConfig() {
  try {
    config=await request(`${API}/config`);
    display.addCatalog(config.presentation);
    referenceNotes=config.reference_notes||{};
    definitionPlain=config.definition_plain||{};
    significance=config.significance||{};
    const codeInput=$('input[name=code]');
    if(codeInput && !codeInput.value.trim()) {
      let shared=''; try{shared=localStorage.getItem(PARTICIPANT_CODE_KEY)||'';}catch(_e){}
      const fallback=String(config.default_code||'').trim();
      if(shared || fallback) codeInput.value=shared || fallback;
    }
    $("#auth-form").hidden=true; $("#setup-form").hidden=false;
    $("#start").disabled=false;
    const mixedPanel=String(config.meta.version_label||"").startsWith("v7");
    const introLead=count=>multiTopic(config.meta)
      ? `审阅跨诊断脑连接、影像遗传学和预后研究材料。本次分配 ${count} 份，每份约 5 分钟，完成 5–6 项能力评分。`
      : mixedPanel
      ? `本面板包含 30 份实验一致的发现材料和 4 份实验不一致的真实校准材料；本次分配给你 ${count} 份，每份 ${config.questions.length} 题，共 ${count*config.questions.length} 题。所有材料使用同一套“越高越好”的能力评分，结果不一致本身不是低分理由。`
      : `我们从系统自动提出并完成实验验证的成果中，选取表现最好的 ${config.meta.card_count} 个完整案例；本次分配给你 ${count} 份，每份 ${config.questions.length} 题，共 ${count*config.questions.length} 题，无需撰写长篇评审。`;
    $("#intro-lead").textContent=introLead(config.meta.card_count);
    // The organizer preview (ALL) is intentionally not offered to participants.
    const assignmentRow=$("#assignment-row"),assignmentSelect=$("#assignment");
    const assignmentOptions=(config.assignments?.options||[]).filter(option=>option.id!=="ALL");
    if(assignmentOptions.length) {
      assignmentSelect.innerHTML=`<option value="">请选择</option>`+assignmentOptions.map(option=>`<option value="${h(option.id)}">${h(`${option.id} · ${option.card_count} 份`)}</option>`).join("");
      assignmentSelect.required=true;
      assignmentRow.hidden=false;
      assignmentSelect.onchange=()=>{
        const option=assignmentOptions.find(item=>item.id===assignmentSelect.value);
        $("#intro-lead").textContent=introLead(option?option.card_count:config.meta.card_count);
      };
    } else {assignmentSelect.required=false;assignmentRow.hidden=true;}
    $("#start").textContent=singleRound(config.meta)?"开始评审 →":"进入阶段 A →";
    $("#intro-flow").innerHTML=singleRound(config.meta)?'<div><span>01</span><strong>阅读一个完整案例</strong><p>假设、依据、方法、内外部结果和反馈同页展示。</p></div><div><span>02</span><strong>完成本例六项评分</strong><p>评分区保留当前假设，可随时对照材料。</p></div><div><span>03</span><strong>继续下一个案例</strong><p>一轮完成分配的案例，不再分两轮往返。</p></div>':'<div><span>01</span><strong>先看假设与方法</strong><p>完成全部材料的初评，不显示候选实验结果。</p></div><div><span>02</span><strong>再看完整结果</strong><p>锁定初评后，查看内外部分析及真实反馈链。</p></div><div><span>03</span><strong>独立作出判断</strong><p>选择题与 1–5 级评分，允许无法判断。</p></div>';
    if(multiTopic(config.meta)) $("#intro-flow").innerHTML=$("#intro-flow").innerHTML.replace("完成本例六项评分","完成本例能力评分");
    $("#pack-note").textContent=`固定材料：${config.meta.version_label} · ${config.meta.card_count} 份材料 · 自动保存`;
    const saved=readCredentials();
    if(saved?.id && saved?.token) {
      $("#resume-banner").hidden=false; $("#setup-form").hidden=true;
      $("#resume-description").textContent=(saved.display_code?`${saved.display_code}。`:"")+"答案与材料版本已绑定。继续原会话不会改用更新的材料。";
    }
  } catch(error) {
    if(error.status===401) {$("#setup-form").hidden=true; $("#pack-note").textContent="请先解锁研究入口。";}
    else {$("#pack-note").textContent="连接失败："+error.message; $("#start").disabled=true;}
  }
}

function initialize(nextData) {
  data=nextData; current=0; drafts={}; dirty.clear(); pending=null; activeSeconds=0;
  display.addCatalog(data.presentation);
  credentials.display_code=data.session.profile.code; delete credentials.mode; storeCredentials();
  rememberEvaluation();
  for(const card of data.cards) drafts[card.id]={answers:{...data.session.answers[card.id]},notes:{...data.session.notes[card.id]},issues:[...data.session.issues[card.id]]};
  if(singleRound() && data.session.stage!=="complete") current=Math.max(0,data.cards.findIndex(c=>completeCount(c.id)<stageQuestions(data.session.stage,c.id).length));
  $("#welcome").hidden=true; $("#workspace").hidden=false;
  render(); StudyWorkspace.resetReadingPosition(); window.scrollTo(0,0);
}
function stageQuestions(stage=data.session.stage,cardId=data.cards[current]?.id) {
  const applicable=data.scoring?.applicable_items_by_card?.[cardId];
  return data.questions.filter(q=>q.stage===stage && (!applicable || applicable.includes(q.id)));
}
function completeCount(cardId,stage=data.session.stage) {return stageQuestions(stage,cardId).filter(q=>drafts[cardId].answers[q.id]!==undefined).length;}
function refreshProgress() {
  const stage=data.session.stage, phase=reviewPhase();
  const count=data.cards.reduce((sum,c)=>sum+completeCount(c.id,phase),0),total=data.cards.reduce((sum,c)=>sum+stageQuestions(phase,c.id).length,0);
  $("#progress-text").textContent=`${singleRound()?"单轮评审":`阶段 ${phase}`} · ${count} / ${total} 项已回答`;
  $("#progress-bar").value=100*count/total;
  $("#cards-nav").innerHTML=data.cards.map((card,i)=>`<button class="nav-card ${i===current?"active":""}" data-index="${i}" ${i===current?'aria-current="true"':""}>材料 ${String(i+1).padStart(2,"0")}<small>${completeCount(card.id,phase)} / ${stageQuestions(phase,card.id).length} 项${completeCount(card.id,phase)===stageQuestions(phase,card.id).length?" · 已回答":""}</small></button>`).join("");
  $("#cards-nav").querySelectorAll("button").forEach(button=>button.addEventListener("click",()=>navigate(Number(button.dataset.index))));
  $("#previous").disabled=current===0; $("#next").disabled=current===data.cards.length-1;
  $("#next").textContent=singleRound()?"保存并评下一份":"下一份";
  $("#advance").textContent=stage==="complete"?"已完成 · 答案锁定":stage==="A"?"锁定初评，解锁结果":"完成评审";
  $("#advance").hidden=singleRound() && stage!=="complete" && current!==data.cards.length-1;
  $("#advance").disabled=stage==="complete";
  $("#export-complete").hidden=stage!=="complete";
}
function commonHTML(items) {return items.map(item=>`<div class="common-item"><strong>${h(item.title)}</strong><p>${h(item.text)}</p></div>`).join("");}
function safeUrl(x) {try {const u=new URL(x); return ["https:","http:"].includes(u.protocol)?u.href:"";} catch{return "";}}
function rationaleHTML(pre, cardId) {
  const isEn = display.language === "en";
  const notes = referenceNotes?.[cardId] || {};
  const items = (pre.references || []).map(ref => {
    const note = notes[ref.id];
    const did = note ? (isEn ? note.did_en : note.did_zh) : "";
    return did ? `<li><strong>${h(ref.id)}</strong>：${h(did)}</li>` : "";
  }).filter(Boolean).join("");
  const lead = isEn
    ? "The system formed this idea from these published studies recorded in the knowledge graph:"
    : "系统形成这个猜想，是受了这些已发表研究（知识图谱记录）的启发：";
  const originalLabel = pre.rationale_is_summary ? (isEn ? "Research rationale" : "研究依据") : (isEn ? "The model's reasoning (verbatim)" : "模型的推理（原文）");
  const list = items ? `<p class="ref-lead" data-user-content>${h(lead)}</p><ul class="rationale-refs" data-user-content>${items}</ul>` : "";
  return `${list}<p class="rationale-original"><strong>${h(originalLabel)}：</strong>${h(pre.rationale)}</p><p class="tiny">${h(pre.source_note)}</p>`;
}
function definitionPlainHTML(cardId) {
  const gloss = definitionPlain?.[cardId];
  if (!gloss) return "";
  const isEn = display.language === "en";
  return `<p class="plain-gloss" data-user-content><strong>${h(isEn ? "In plain terms" : "通俗理解")}：</strong>${h(isEn ? gloss.en : gloss.zh)}</p>`;
}
function significanceHTML(cardId) {
  // New prose is frozen with the case in the session's own material snapshot.
  // Legacy sidecars may only be used with the pack they actually describe.
  const embedded = data?.cards?.find(card => card.id === cardId)?.pre?.significance;
  const samePack = typeof config === "undefined" || !config?.pack_id || !data?.session?.pack_id || config.pack_id === data.session.pack_id;
  const entry = embedded || (samePack ? significance?.[cardId] : null);
  if (!entry || (!embedded && !/^v[6789]/.test(String(data?.meta?.version_label || "")))) return "";
  const isEn = display.language === "en";
  return `<p class="sig" data-user-content>${h(isEn ? entry.en : entry.zh)}</p>`;
}
function feedbackGlossHTML(post) {
  const fb = post.feedback, isEn = display.language === "en";
  if (fb.no_parent) {
    return h(isEn
      ? "In plain terms: this card comes from the first-round proposals — there is no earlier parent hypothesis; it is the start of this automated iteration chain."
      : "通俗理解：这张卡来自第一轮提议，没有更早的父假设——它是这条自动迭代链的起点。");
  }
  const named = (fb.parent_named || []).map(domainName).join("、");
  return h(isEn
    ? `In plain terms: this idea did not come from nowhere — the system had already tested a related hypothesis (${fb.parent_endpoint}${named ? `, ${named}` : ""}), which came out highly consistent (${pc(fb.parent_joint)} of replicates in the same direction); this hypothesis pushes that lead one step further.`
    : `通俗理解：这个猜想不是凭空来的——系统上一轮已经实际测过一个相近的猜想（${fb.parent_endpoint}${named ? `，${named}` : ""}），结果方向相当稳定（${pc(fb.parent_joint)} 的重复中方向一致）；本猜想把那条线索继续向前推了一步。`);
}
const NETWORK_PLAIN = {
  Vis: ["视觉网络", "visual network"],
  SomMot: ["躯体运动网络", "somatomotor network"],
  DorsAttn: ["背侧注意网络", "dorsal attention network"],
  SalVentAttn: ["显著性/腹侧注意网络", "salience/ventral attention network"],
  Default: ["默认网络", "default mode network"],
};
const DOMAIN_PLAIN_EN = {ADHD:"ADHD",bipolar:"bipolar disorder",psychosis_SZ_SZA:"schizophrenia/schizoaffective (SZ/SZA)",psychosis:"psychosis",SCHZ:"schizophrenia",SZ:"SZ",nonaffective_psychosis:"non-affective psychosis",affective_psychosis:"affective psychosis"};
const networkPlain = (key, isEn) => (NETWORK_PLAIN[key] ? NETWORK_PLAIN[key][isEn ? 1 : 0] + (isEn ? ` (${key})` : `（${key}）`) : key);
function parentEndpointPlain(endpoint, isEn) {
  const roi = /^ROI_to_network:(\d+):(\w+)$/.exec(endpoint || "");
  if (roi) return isEn
    ? `the average connectivity between ROI ${roi[1]} and all parcels of the ${networkPlain(roi[2], true)}`
    : `${roi[1]} 号感兴趣区（ROI）与${networkPlain(roi[2], false)}中全部分区之间连接的平均强度`;
  const pair = /^network_pair:(\w+):(\w+)$/.exec(endpoint || "");
  if (pair) return isEn
    ? `the average connectivity between all parcels of the ${networkPlain(pair[1], true)} and the ${networkPlain(pair[2], true)}`
    : `${networkPlain(pair[1], false)}与${networkPlain(pair[2], false)}之间全部分区连接的平均强度`;
  return endpoint || "";
}
function parentPlainHTML(fb) {
  const isEn = display.language === "en";
  const groups = fb.parent_named || [];
  const names = isEn ? groups.map(x => DOMAIN_PLAIN_EN[x] || x).join(", ") : groups.map(domainName).join("、");
  const betas = Object.values(fb.parent_components || {}).map(v => v && v.standardized_beta).filter(x => typeof x === "number");
  const sign = !betas.length ? 0 : betas.every(x => x < 0) ? -1 : betas.every(x => x > 0) ? 1 : 2;
  const endpoint = parentEndpointPlain(fb.parent_endpoint, isEn);
  const label = isEn ? "What the parent hypothesis actually is" : "父假设具体是什么";
  const zhDirection = sign === -1 ? "低于健康对照" : sign === 1 ? "高于健康对照" : sign === 2 ? "与健康对照相比方向不一" : "与健康对照不同";
  const zh = `在${names || "相关诊断组"}中，病例组的${endpoint}平均而言${zhDirection}；该父假设在系统内部重复实验中方向一致的比例为 ${pc(fb.parent_joint)}。`;
  const enClause = sign === 2
    ? `patients showed mixed directions versus healthy controls in ${endpoint}`
    : sign === 0
      ? `patients differed from healthy controls in ${endpoint}`
      : `patients showed ${sign === -1 ? "lower" : "higher"} values than healthy controls for ${endpoint}`;
  const en = `In ${names || "the related diagnostic groups"}, ${enClause}; ${pc(fb.parent_joint)} of the system's internal replicates agreed in direction.`;
  return `<p class="plain-gloss" data-user-content><strong>${h(label)}：</strong>${h(isEn ? en : zh)}</p>`;
}
function referencesHTML(references, cardId) {
  const isEn = display.language === "en";
  const didLabel = isEn ? "What this study did" : "该文献做了什么";
  const relationLabel = isEn ? "Relation to this hypothesis" : "与本假设的关系";
  return references.map(ref=>{
    const note = referenceNotes?.[cardId]?.[ref.id];
    const did = note ? (isEn ? note.did_en : note.did_zh) : "";
    const relation = note ? (isEn ? note.relation_en : note.relation_zh) : "";
    return `<div class="reference"><strong>${h(ref.id)} · ${h(ref.title)}</strong><p class="tiny">${h(ref.journal)} · ${h(ref.year)} · PMID ${h(ref.pmid)} ${safeUrl(ref.url)?`<a href="${h(safeUrl(ref.url))}" target="_blank" rel="noopener noreferrer">查原文 ↗</a>`:""}</p>${did?`<p class="ref-note" data-user-content><strong>${h(didLabel)}：</strong>${h(did)}</p>`:""}<blockquote>${h(ref.recorded_sentence)}</blockquote>${relation?`<p class="ref-note" data-user-content><strong>${h(relationLabel)}：</strong>${h(relation)}</p>`:""}</div>`;
  }).join("");
}
function internalHTML(rows) {return `<div class="results-scroll"><table><caption class="tiny">R1 · TCP 固定内部测量（β / 残差 RMS）</caption><thead><tr><th>比较</th><th>病例 / 对照</th><th>β</th><th>标准化效应</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${h(domainName(r.domain))}</td><td>${h(r.cases)} / ${h(r.controls)}</td><td>${f(r.beta)}</td><td>${f(r.standardized_beta)}</td></tr>`).join("")}</tbody></table></div>${connectivityAnalysisHTML(rows,"internal")}`;}
function externalHTML(rows, compact=false) {
  return `<div class="results-scroll"><table><caption class="tiny">R2 · 全部外部组成及敏感性结果（横向滚动）</caption><thead><tr><th>队列 / 诊断</th><th>n 病例/对照</th><th>标准化效应</th><th>描述性百分位范围</th><th>同向频率</th></tr></thead><tbody>${rows.map(r=>{
    const boot=r.bootstrap_standardized_beta,range=boot.bootstrap_percentiles_2_5_97_5_descriptive;
    return `<tr><td>${h(r.cohort)} · ${h(comparisonName(r))}<span class="sub">${h(r.row_id)} · ${h(variantName(r.variant))}</span></td><td>${h(r.cases)} / ${h(r.controls)}</td><td>${f(r.standardized_beta)}</td><td>[${f(range?.[0])}, ${f(range?.[1])}]</td><td>${pc(boot.bootstrap_direction_fraction)}<span class="sub">有效重复 ${h(boot.bootstrap_valid)} / ${h(boot.bootstrap_total)}</span></td></tr>`;
  }).join("")}</tbody></table></div>${connectivityAnalysisHTML(rows,"external")}${compact?"":'<p class="metric-note">百分位范围是描述性 bootstrap 结果，不是校准置信区间；没有 P/q 值。请勿按是否跨零作显著性判断。</p>'}`;
}
function feedbackHTML(post) {
  const fb=post.feedback;
  const gloss=`<p class="plain-gloss" data-user-content>${feedbackGlossHTML(post)}</p>`;
  if(fb.no_parent||!fb.parent_endpoint) return gloss+`<p class="tiny">R3 · 首轮提议：生成该假设时同 seed 尚无已完成的 TCP 反馈，无父假设绑定。</p><p><strong>首轮提议说明：</strong><br>${h(fb.action_text)}</p><p><strong>原始讨论后的决策：</strong>${h(post.discussion.action)}<br>${h(post.discussion.text)}</p><p class="tiny">文中的“三方”是三个独立的大语言模型评审角色（生物统计、方法学、临床神经科学视角）：先各自独立评审，再看到另外两份意见后复核，两轮共六份评议。评议来自模型角色，不是人类专家意见；其赞同不增加独立实验支持。</p>`;
  return gloss+`<p class="tiny">R3 · 同 seed 的历史反馈在当前提议前已可见，绑定核对：${fb.binding_verified?"通过":"未确认"}。</p>${parentPlainHTML(fb)}<p><strong>父假设测量：</strong>${h(fb.parent_endpoint)}<br><strong>具名组成：</strong>${h(fb.parent_named.join(" / "))}<br><strong>父反馈标准化效应：</strong>${Object.entries(fb.parent_components).map(([key,value])=>`${h(domainName(key))} ${f(value.standardized_beta)}`).join("；")}<br><strong>联合方向频率：</strong>${pc(fb.parent_joint)}</p><p><strong>真实后轮利用说明：</strong><br>${h(fb.action_text)}</p><p><strong>原始讨论后的决策：</strong>${h(post.discussion.action)}<br>${h(post.discussion.text)}</p><p class="tiny">文中的“三方”是三个独立的大语言模型评审角色（生物统计、方法学、临床神经科学视角）：先各自独立评审，再看到另外两份意见后复核，两轮共六份评议。评议来自模型角色，不是人类专家意见；其赞同不增加独立实验支持。</p>`;
}

function partHTML(num,title,desc) {
  return `<h3 class="part-title"><span class="part-num">${num}</span>${h(title)}</h3>${desc?`<p class="part-desc">${h(desc)}</p>`:""}`;
}
function hypothesisHTML(pre,eyebrow,cardId) {
  // The plain-language statement is the primary display; the original
  // program-rendered string stays available for provenance. A plain "what it
  // would mean" sentence closes the block when the bound sidecar has one.
  const plain=pre.hypothesis_plain;
  return `<div class="eyebrow">${h(eyebrow)}</div><p>${h(plain||pre.hypothesis)}</p>${significanceHTML(cardId)}${plain?`<details class="hypothesis-raw"><summary>程序原始表述</summary><p class="tiny">${h(pre.hypothesis)}</p></details>`:""}`;
}
function studyContextHTML(pre) {
  const context=pre.study_context;
  if(!context) return "";
  const isEn=display.language==="en";
  const origin=isEn?context.origin_en:context.origin_zh;
  const layer=isEn?context.layer_en:context.layer_zh;
  const alignment=isEn?context.alignment_en:context.alignment_zh;
  const detail=isEn?context.detail_en:context.detail_zh;
  const tone=context.alignment==="consistent"?"consistent":"not-consistent";
  return `<div class="study-context ${tone}" data-user-content><div class="study-context-top"><span>${h(origin)}</span><strong>${h(alignment)}</strong><span class="study-context-layer">${h(layer)}</span></div><p>${h(detail)}</p></div>`;
}
function probabilityText(value) {
  if(typeof value!=="number" || !Number.isFinite(value)) return "—";
  return value>0 && value<0.0001 ? value.toExponential(2) : f(value,4);
}
function resultNumber(value) {
  return typeof value==="number" && Number.isFinite(value);
}
function resultAnalysisHTML(text) {
  return `<p class="result-analysis" data-user-content><strong>${display.language==="en"?"Result analysis: ":"结果分析："}</strong>${h(text)}</p>`;
}
function effectDirectionText(values,negative,positive) {
  const en=display.language==="en";
  if(!values.length) return en?"No valid effect estimates are available to assess direction.":"缺少有效效应估计，无法判断方向。";
  if(values.every(value=>value<0)) return negative;
  if(values.every(value=>value>0)) return positive;
  return en?"The estimates do not share a strictly positive or negative direction.":"各项估计并非一致为正或一致为负。";
}
function connectivityAnalysisHTML(rows,phase,joint,comparison) {
  const en=display.language==="en",valid=rows.filter(row=>resultNumber(row.standardized_beta));
  const effects=valid.map(row=>row.standardized_beta);
  const scope=phase==="internal"?"TCP":phase==="additional"?(en?"Additional comparisons":"补充比较"):(en?"External comparisons":"外部比较");
  const pieces=[effectDirectionText(effects,
    en?`${scope}: all ${valid.length} available effects are negative, indicating lower connectivity in patients.`:`${scope}的 ${valid.length} 项有效效应均为负，患者组连接较低。`,
    en?`${scope}: all ${valid.length} available effects are positive, indicating higher connectivity in patients.`:`${scope}的 ${valid.length} 项有效效应均为正，患者组连接较高。`)];
  if(valid.length) {
    const labels=valid.map(row=>`${row.cohort||"TCP"} · ${comparison?comparison(row):row.domain}: ${f(row.standardized_beta)}`);
    pieces.push(en?`Standardized effects are ${labels.join("; ")}.`:`各比较的标准化效应为 ${labels.join("；")}。`);
  }
  if(valid.length!==rows.length) pieces.push(en?`${rows.length-valid.length} comparisons lack valid effect estimates and are not included in this direction summary.`:`另有 ${rows.length-valid.length} 项比较缺少有效效应估计，未纳入上述方向总结。`);
  if(phase==="internal") {
    if(resultNumber(joint) && joint>=0 && joint<=1) pieces.push(en?`Joint direction agreement across internal resamples is ${pc(joint)}.`:`内部重复抽样的联合方向频率为 ${pc(joint)}。`);
    pieces.push(en?"These within-cohort diagnostic comparisons assess consistency across groups; external reproducibility is assessed in the following table.":"这些同一队列内的诊断比较反映组间方向的一致程度；跨队列可重复性需结合下表判断。");
  } else {
    const stability=rows.map(row=>row.bootstrap_standardized_beta?.bootstrap_direction_fraction).filter(value=>resultNumber(value)&&value>=0&&value<=1);
    if(stability.length) pieces.push(en?`Available bootstrap direction agreement ranges from ${pc(Math.min(...stability))} to ${pc(Math.max(...stability))}; this measures resampling consistency, not statistical significance.`:`有效 bootstrap 同向频率为 ${pc(Math.min(...stability))}–${pc(Math.max(...stability))}，反映重复抽样的方向一致性，而非统计显著性。`);
    if(phase==="additional") pieces.push(en?"The head-motion and diagnosis restrictions probe sensitivity to analysis choices; the affective-psychosis comparison examines a different diagnostic group, rather than another independent replication of the main comparison.":"头动调整与诊断限制检验分析设定的敏感性；情感性精神病比较涉及不同诊断人群，不能将这些补充行当作主比较的新增独立重复验证。");
  }
  return resultAnalysisHTML(pieces.join(" "));
}
function experimentAnalysisHTML(post,phase,names) {
  const en=display.language==="en",ig=post.result_kind==="imaging_genetics";
  const usable=result=>result && !result.failure && (!result.status || result.status==="complete");
  const configs=post.experimental_results;
  const available=configs.filter(cfg=>usable(cfg[phase]) && resultNumber(cfg[phase].primary?.[ig?"effect":"hazard_ratio"]) && (ig || cfg[phase].primary.hazard_ratio>0));
  const pieces=[];
  if(ig) {
    pieces.push(effectDirectionText(available.map(cfg=>cfg[phase].primary.effect),
      en?"All available models estimate a negative association.":"本表有效模型均估计为负向关联。",
      en?"All available models estimate a positive association.":"本表有效模型均估计为正向关联。"));
    const models=available.map(cfg=>`${names[cfg.model]||cfg.model} = ${f(cfg[phase].primary.effect)} (q=${probabilityText(cfg[phase].q_value)})`);
    if(models.length) pieces.push(models.join(en?"; ":"；")+(en?".":"。"));
    const tested=available.filter(cfg=>resultNumber(cfg[phase].q_value) && cfg[phase].q_value>=0 && cfg[phase].q_value<=1);
    const passed=tested.filter(cfg=>cfg[phase].q_value<0.05).length;
    pieces.push(en?`${passed} of ${tested.length} models with valid q values meet q < 0.05 after multiple-testing correction.`:`有有效 q 值的 ${tested.length} 个模型中，${passed} 个达到多重检验校正后的 q < 0.05。`);
    if(tested.length && !passed) pieces.push(en?"This table does not establish an association after correction; this is not evidence that the association is exactly zero.":"本表尚未提供校正后关联证据，但不能据此认定关联为零。");
    if(phase==="external") {
      const paired=configs.filter(cfg=>usable(cfg.internal)&&usable(cfg.external)&&resultNumber(cfg.internal.primary?.effect)&&resultNumber(cfg.external.primary?.effect));
      const agreeing=paired.filter(cfg=>cfg.internal.primary.effect*cfg.external.primary.effect>0).length;
      const internalTests=configs.filter(cfg=>usable(cfg.internal)&&resultNumber(cfg.internal.primary?.effect)&&resultNumber(cfg.internal.q_value)&&cfg.internal.q_value>=0&&cfg.internal.q_value<=1);
      const internalPassed=internalTests.filter(cfg=>cfg.internal.q_value<0.05).length;
      pieces.push(en?`${agreeing} of ${paired.length} paired models have the same nonzero direction internally and externally; the internal table has ${internalPassed}/${internalTests.length} models meeting q < 0.05.`:`可配对的 ${paired.length} 个模型中，${agreeing} 个内外部方向相同且非零；内部表达到 q < 0.05 的模型为 ${internalPassed}/${internalTests.length}。`);
      if(internalTests.length && !internalPassed && passed) pieces.push(en?"The corrected evidence is concentrated in the external cohort, not consistently established in both cohorts.":"校正后的关联证据集中在外部队列，尚非两个队列均得到一致统计验证。");
    }
    pieces.push(en?"Agreement across models describes robustness on the same participants; r and β are different measures and their magnitudes are not directly interchangeable.":"模型间一致性反映同一批参与者上的分析稳健性；r 与 β 是不同指标，不能直接按数值大小互相比较。");
  } else {
    pieces.push(effectDirectionText(available.map(cfg=>cfg[phase].primary.hazard_ratio-1),
      en?"All available HR estimates are below 1, associating a one-SD increase in the predictor with lower progression hazard.":"本表有效 HR 点估计均小于 1，预测变量每增加一个标准差均与较低的进展瞬时风险相关。",
      en?"All available HR estimates are above 1, associating a one-SD increase in the predictor with higher progression hazard.":"本表有效 HR 点估计均大于 1，预测变量每增加一个标准差均与较高的进展瞬时风险相关。"));
    for(const cfg of available) {
      const result=cfg[phase],primary=result.primary;
      const validCI=resultNumber(primary.hazard_ratio_ci_low)&&resultNumber(primary.hazard_ratio_ci_high)&&primary.hazard_ratio_ci_low>0&&primary.hazard_ratio_ci_high>=primary.hazard_ratio_ci_low;
      const excludes=validCI&&(primary.hazard_ratio_ci_high<1||primary.hazard_ratio_ci_low>1);
      const validQ=resultNumber(result.q_value)&&result.q_value>=0&&result.q_value<=1;
      let assessment=!validCI?(en?"the interval is unavailable":"区间不可用"):excludes?(en?"the interval excludes 1":"区间不含 1"):(en?"the interval includes 1":"区间包含 1");
      assessment+=!validQ?(en?", q is unavailable": "，q 值不可用"):result.q_value<0.05?(en?", q < 0.05 after correction":"，校正后 q < 0.05"):(en?", q does not meet 0.05 after correction":"，校正后 q 未达到 0.05 阈值");
      const interval=validCI?`[${f(primary.hazard_ratio_ci_low)}, ${f(primary.hazard_ratio_ci_high)}]`:"—";
      pieces.push(en?`${cfg.horizon_years}-year HR=${f(primary.hazard_ratio)}, 95% CI ${interval}, q=${probabilityText(result.q_value)}: ${assessment}.`:`${cfg.horizon_years} 年 HR=${f(primary.hazard_ratio)}，95% CI ${interval}，q=${probabilityText(result.q_value)}：${assessment}。`);
      if(resultNumber(primary.events)&&primary.events<10) pieces.push(en?`Only ${primary.events} events inform the ${cfg.horizon_years}-year estimate, so even a small q value does not by itself establish reliable validation.`:`${cfg.horizon_years} 年窗口仅有 ${primary.events} 个事件，即使 q 值较小，也不能单凭这一点认定已获得可靠验证。`);
    }
    pieces.push(en?"The follow-up windows reuse participants and are not independent replications; the HRs describe associations, not causal effects.":"各随访窗口重复使用参与者，并非相互独立的重复验证；HR 描述关联，而非因果效应。");
  }
  if(available.length!==configs.length) pieces.push(en?`${configs.length-available.length} rows have missing or unsuccessful estimates and cannot be interpreted as null results.`:`另有 ${configs.length-available.length} 行估计缺失或未成功，不能解读为无效应。`);
  return resultAnalysisHTML(pieces.join(" "));
}
function experimentTablesHTML(post) {
  const en=display.language==="en", ig=post.result_kind==="imaging_genetics";
  const names={association_glm:en?"Linear association · r":"线性关联 · r",rank_association:en?"Rank association · r":"秩关联 · r",robust_huber:en?"Robust Huber · β":"Huber 稳健回归 · β"};
  return ["internal","external"].map(phase=>{
    const title=phase==="internal"?(en?"Internal · ADNI1/GO/2":"内部 · ADNI1/GO/2"):(en?"External · ADNI3":"外部 · ADNI3");
    const headers=ig?[en?"Model / measure":"模型 / 指标","n",en?"Effect":"效应","P","q"]:[en?"Follow-up":"随访窗口",en?"n / events":"n / 事件数","HR",en?"95% CI":"95% 置信区间","P","q"];
    const rows=post.experimental_results.map(cfg=>{
      const result=cfg[phase]||{},p=result.primary||{};
      const cells=ig?[names[cfg.model]||cfg.model,p.n,f(p.effect),probabilityText(p.p_value),probabilityText(result.q_value)]
        :[`${cfg.horizon_years} ${en?"years":"年"}`,`${p.n} / ${p.events}`,f(p.hazard_ratio),`[${f(p.hazard_ratio_ci_low)}, ${f(p.hazard_ratio_ci_high)}]`,probabilityText(p.p_value),probabilityText(result.q_value)];
      return `<tr>${cells.map(cell=>`<td>${h(cell)}</td>`).join("")}</tr>`;
    }).join("");
    return `<div class="results-scroll" data-user-content><table><caption>${h(title)}</caption><thead><tr>${headers.map(label=>`<th>${h(label)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>${experimentAnalysisHTML(post,phase,names)}`;
  }).join("");
}
function multiTopicMaterialHTML(card) {
  if(card.pre.reading) return readableMaterialHTML(card);
  const pre=card.pre,post=card.post,en=display.language==="en";
  const feedback=post?.feedback;
  let html=`<header class="material-heading"><span class="tag">材料 ${String(current+1).padStart(2,"0")} / ${String(data.cards.length).padStart(2,"0")}</span><h2>${h(pre.title)}</h2><p><span>${h(pre.topic)}</span> · NeuroDiscovery</p></header>`;
  html+=`<section class="material-part">${partHTML("01","这个发现是什么","")}<div class="hypothesis" id="current-hypothesis">${hypothesisHTML(pre,"系统提出的假设",card.id)}</div><section class="block"><h4>精确定义</h4><p>${h(pre.definition)}</p>${definitionPlainHTML(card.id)}</section></section>`;
  html+=`<section class="material-part">${partHTML("02",en?"Where it came from":"它从何而来","")}<section class="block"><h4>为何提出这项假设</h4>${rationaleHTML(pre,card.id)}</section><details open><summary>${en?"Related studies":"相关研究"} (${pre.references.length})</summary>${referencesHTML(pre.references,card.id)}</details>`;
  if(feedback?.parent_endpoint && !feedback.no_parent) {
    html+=`<section class="block"><h4>${en?"Learning from earlier experiments":"前序实验与反馈"}</h4>${parentPlainHTML(feedback)}<p>${h(feedback.action_text)}</p><details><summary>${en?"Decision after model-agent discussion":"模型角色讨论后的决策"}</summary><p>${h(post.discussion?.text)}</p></details></section>`;
  }
  html+="</section>";
  const methods=pre.methods||[
    {title:en?"Datasets":"数据集",text:en?"Internal: TCP. External: UCLA, COBRE and HCP-EP. Sample sizes are shown for each comparison.":"内部使用 TCP，外部使用 UCLA、COBRE 和 HCP-EP；各比较的样本量见结果表。"},
    {title:en?"Measures and statistical models":"测量与统计模型",text:en?"Resting-state fMRI connectivity is measured as the mean Pearson correlation across the predefined connections. OLS models compare cases and controls with adjustment for age, sex and site; external models also evaluate head motion. Stability is assessed with 512 internal and 2,000 external resamples.":"以预先指定连接的平均 Pearson 相关系数衡量静息态功能连接。OLS 模型比较病例与对照，调整年龄、性别和站点；外部模型同时考察头动。内部使用 512 次、外部使用 2,000 次重复抽样评估稳定性。"}
  ];
  html+=`<section class="material-part">${partHTML("03","实验设计 · 模型、数据与指标","")}${commonHTML(methods)}</section>`;
  if(post) {
    const results=post.experimental_results?experimentTablesHTML(post):internalHTML(post.internal_rows)+`<div class="result-figures"><div><span>TCP · 联合方向频率</span><b>${pc(post.tcp_joint)}</b></div><div><span>UCLA · 联合方向频率</span><b>${pc(post.ucla_joint)}</b></div></div>`+externalHTML(post.external_rows,true);
    html+=`<section class="material-part">${partHTML("04","实验结果 · 内部与外部验证","")}${results}</section>`;
  }
  return html+'<a href="#ratings-form">前往本份评审题 ↓</a>';
}
function readableResultTablesHTML(card) {
  const post=card.post,en=display.language==="en";
  if(!post) return "";
  if(post.experimental_results) {
    const note=post.result_kind==="imaging_genetics"
      ? (en?"r is the adjusted correlation; β is the standardized regression coefficient. q adjusts for multiple testing.":"r 为调整后的相关系数，β 为标准化回归系数；q 为多重检验校正值。")
      : (en?"HR is the hazard ratio per standard-deviation increase in the predictor; values below 1 indicate lower progression risk. q adjusts for multiple testing.":"HR 为预测变量每增加一个标准差对应的风险比；小于 1 表示进展风险较低。q 为多重检验校正值。");
    return `<div class="experimental-results" data-results-kind="${h(post.result_kind)}">${experimentTablesHTML(post)}<p class="tiny" data-user-content>${h(note)}</p></div>`;
  }
  const diagnoses={
    ADHD:["ADHD","ADHD"],bipolar:["双相障碍","Bipolar disorder"],
    psychosis_SZ_SZA:["精神分裂症 / 分裂情感性障碍","Schizophrenia / schizoaffective disorder"],
    non_affective_psychosis:["非情感性精神病","Non-affective psychosis"],
    affective_psychosis:["情感性精神病","Affective psychosis"]
  };
  const comparison=row=>row.variant==="sensitivity_strict_SZ"?(en?"Schizophrenia only":"仅精神分裂症")
    :row.cohort==="UCLA" && row.domain==="psychosis_SZ_SZA"?(en?"Schizophrenia":"精神分裂症")
    :(diagnoses[row.domain]?.[en?1:0]||row.domain);
  const table=(title,headers,rows)=>`<div class="results-scroll" data-user-content><table><caption>${h(title)}</caption><thead><tr>${headers.map(label=>`<th>${h(label)}</th>`).join("")}</tr></thead><tbody>${rows.map(cells=>`<tr>${cells.map(cell=>`<td>${h(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  const sampleLabel=en?"n patients / controls":"n 患者 / 对照";
  const internal=table(en?"Internal · TCP":"内部 · TCP",[en?"Comparison":"比较",sampleLabel,en?"Standardized effect":"标准化效应"],post.internal_rows.map(row=>[comparison(row),`${row.cases} / ${row.controls}`,f(row.standardized_beta)]))+connectivityAnalysisHTML(post.internal_rows,"internal",post.tcp_joint,comparison);
  const externalRows=rows=>rows.map(row=>[`${row.cohort} · ${comparison(row)}`,`${row.cases} / ${row.controls}`,f(row.standardized_beta),pc(row.bootstrap_standardized_beta?.bootstrap_direction_fraction)]);
  const headers=[en?"Cohort / comparison":"队列 / 比较",sampleLabel,en?"Standardized effect":"标准化效应",en?"Direction stability":"方向稳定性"];
  const primary=post.external_rows.filter(row=>row.variant.startsWith("primary_"));
  const additional=post.external_rows.filter(row=>!row.variant.startsWith("primary_"));
  const extraLabels={sensitivity_no_FD:en?"Without head-motion adjustment":"不调整头动",sensitivity_strict_SZ:en?"Restricted diagnosis":"限制诊断范围",secondary_affective_context:en?"Additional affective-psychosis comparison":"补充情感性精神病比较"};
  const extra=additional.length?`<details><summary>${en?"Additional checks":"补充检验"}</summary>${table(en?"Additional external comparisons":"外部补充比较",[en?"Analysis / comparison":"分析 / 比较",...headers.slice(1)],additional.map(row=>[`${row.cohort} · ${comparison(row)} · ${extraLabels[row.variant]||row.variant}`,`${row.cases} / ${row.controls}`,f(row.standardized_beta),pc(row.bootstrap_standardized_beta?.bootstrap_direction_fraction)]))}${connectivityAnalysisHTML(additional,"additional",null,row=>`${comparison(row)} · ${extraLabels[row.variant]||row.variant}`)}</details>`:"";
  const note=en?"Negative effects indicate lower connectivity in patients. Direction stability is the percentage of resamples agreeing with the hypothesis direction, not a P value.":"负效应表示患者组连接较低。方向稳定性是重复抽样中与假设同向的比例，不是 P 值。";
  return `<div class="experimental-results" data-results-kind="connectivity">${internal}${table(en?"External validation":"外部验证",headers,externalRows(primary))}${connectivityAnalysisHTML(primary,"external",null,comparison)}${extra}<p class="tiny" data-user-content>${h(note)}</p></div>`;
}
function researchChainHTML(chain,forward=false) {
  const en=display.language==="en",lang=en?"en":"zh",explicit=chain.mode==="explicit_parent";
  const labels={
    source:explicit?(en?(forward?"Parent hypothesis · this case":"Parent hypothesis"):(forward?"父假设 · 本例":"父假设")):(en?"Previous-round hypothesis · this case":"前轮假设 · 本例"),
    feedback:en?"Experimental feedback":"实验反馈",
    next:explicit?(en?(forward?"Child hypothesis":"Child hypothesis · this case"):(forward?"子假设":"子假设 · 本例")):(en?"Specific later-round proposals":"后轮具体假设"),
    change:en?"What changed":"具体变化",outcome:en?"Follow-up outcome":"后续进展"
  };
  const row=(key,content)=>`<div><dt>${labels[key]}</dt><dd>${content}</dd></div>`;
  const text=note=>h(note[lang]);
  return `<dl class="research-chain" data-research-chain="${explicit?"parent-child":"round-followup"}">`
    +row("source",text(chain.source))+row("feedback",text(chain.feedback))
    +row("next",chain.next.length===1?text(chain.next[0]):`<ul>${chain.next.map(note=>`<li>${text(note)}</li>`).join("")}</ul>`)
    +row("change",text(chain.change))+(chain.outcome?row("outcome",text(chain.outcome)):"")+`</dl>`;
}
const TERM_GLOSSARY = {
  roi:{zh:["ROI（感兴趣区）","把大脑按图谱划分后选定的一个小区域；这里的编号只代表本研究图谱中的分区，不等同于整个解剖脑区。"],en:["ROI (region of interest)","A small area selected after the brain is divided by an atlas. Its number identifies a parcel in this study's atlas, not an entire anatomical region."]},
  salventattn:{zh:["显著性/腹侧注意网络（SalVentAttn）","一组参与发现重要刺激、切换注意和协调其他脑网络的区域。这里测量的是它与指定分区之间的连接。"],en:["Salience/ventral attention network (SalVentAttn)","A set of regions involved in detecting important events, redirecting attention and coordinating other brain networks. This study measures its connectivity with a specified parcel."]},
  sommot:{zh:["感觉运动网络（SomMot）","主要参与身体感觉和运动控制的一组脑区。网络内连接指这些分区彼此活动同步的程度。"],en:["Somatomotor network (SomMot)","A set of brain regions mainly involved in bodily sensation and movement. Within-network connectivity describes how synchronously its parcels fluctuate."]},
  rsfmri:{zh:["静息态 fMRI","参与者不执行特定任务时记录脑血氧信号的影像方法，用于观察不同脑区的自发活动是否同步。"],en:["Resting-state fMRI","An imaging method that records blood-oxygen signals while a participant performs no specific task, allowing researchers to examine whether spontaneous activity in different regions fluctuates together."]},
  connectivity:{zh:["功能连接","两个脑区信号随时间同步变化的统计关联；连接更高或更低不等于存在直接的神经纤维连接，也不直接证明因果关系。"],en:["Functional connectivity","A statistical association between how two brain signals change over time. Higher or lower connectivity does not necessarily imply a direct anatomical connection or causation."]},
  pearson:{zh:["Pearson r","衡量两个信号同步变化程度的相关系数，通常在 −1 到 1 之间；正负表示变化方向，不代表因果。"],en:["Pearson r","A correlation coefficient, usually ranging from −1 to 1, that summarizes how two signals vary together. Its sign indicates direction, not causation."]},
  standardized_effect:{zh:["标准化效应","把组间差异换算到统一尺度后的数值，便于不同队列比较；负值在这些案例中表示患者组的连接低于对照组。"],en:["Standardized effect","A group difference converted to a common scale so cohorts can be compared. In these cases, a negative value means lower connectivity in the patient group than in controls."]},
  resampling:{zh:["方向稳定性（重复抽样）","多次从现有样本重新抽取数据并重复分析，观察效应方向是否经常一致。结果表和正文中的“方向稳定性”“联合方向频率”都是这一比例：接近 100% 表示方向稳定。它反映重复分析的一致性，不是 P 值，也不能替代独立队列验证。"],en:["Directional stability (resampling)","The analysis is repeated on many resamples of the observed data to see how often the effect points in the same direction. The tables and text labelled directional stability or joint direction frequency report this proportion: close to 100% means a stable direction. It reflects consistency across repeated analyses; it is not a P value and does not replace validation in an independent cohort."]},
  spectrum:{zh:["精神分裂症谱系","包含精神分裂症及分裂情感性障碍等相关诊断的一组疾病；具体纳入范围以各队列说明为准。"],en:["Schizophrenia spectrum","A group of related diagnoses including schizophrenia and schizoaffective disorder. The exact included diagnoses depend on each cohort."]},
  head_motion:{zh:["头动","扫描时头部移动会改变 fMRI 信号并造成假性连接差异，因此分析会进行调整或敏感性检查。"],en:["Head motion","Movement during scanning can alter fMRI signals and create spurious connectivity differences, so analyses adjust for it or test sensitivity to it."]},
  apoe:{zh:["APOE ε4","APOE 基因的一种常见变体，与阿尔茨海默病风险升高有关；携带它不代表一定会患病。"],en:["APOE ε4","A common variant of the APOE gene associated with increased Alzheimer's disease risk. Carrying it does not mean a person will necessarily develop the disease."]},
  allele_dosage:{zh:["等位基因剂量","一个人携带某种基因变体的拷贝数；这里指 APOE ε4 为 0、1 或 2 份。"],en:["Allele dosage","The number of copies of a genetic variant a person carries; here, 0, 1 or 2 copies of APOE ε4."]},
  baseline:{zh:["基线","随访开始时、后续结局发生前的首次测量，用作后续比较或预测的起点。"],en:["Baseline","The initial measurement taken at the start of follow-up, before later outcomes occur, and used as the reference point for comparison or prediction."]},
  entorhinal:{zh:["内嗅皮层","位于内侧颞叶、与记忆形成密切相关的脑区，也是阿尔茨海默病早期常受影响的区域之一。"],en:["Entorhinal cortex","A medial temporal-lobe region closely involved in memory formation and among the areas often affected early in Alzheimer's disease."]},
  hippocampus:{zh:["海马","内侧颞叶中参与学习和记忆的结构；体积减小可见于多种情况，并非某一种疾病所独有。"],en:["Hippocampus","A medial temporal-lobe structure involved in learning and memory. Reduced volume can occur in several conditions and is not unique to one disease."]},
  middle_temporal:{zh:["颞中回","位于颞叶外侧、参与语言和语义加工等功能的脑区；这里研究的是 MRI 测得的区域体积。"],en:["Middle temporal gyrus","A lateral temporal-lobe region involved in functions including language and semantic processing. Here the measure is its MRI-derived volume."]},
  whole_brain:{zh:["全脑体积","MRI 估算的整体脑组织体积，是较概括的结构指标，不指向某个特定脑区。"],en:["Whole-brain volume","An MRI estimate of total brain-tissue volume. It is a broad structural measure rather than a marker of one specific region."]},
  icv:{zh:["颅内容积归一化（ICV）","用颅腔总体积校正脑区体积，以减少个体头部大小差异带来的影响。"],en:["Intracranial-volume normalization (ICV)","Brain-region volume is adjusted for total cranial capacity to reduce differences caused by overall head size."]},
  adni:{zh:["ADNI","阿尔茨海默病神经影像计划，一个包含影像、认知、遗传和随访资料的多中心研究。ADNI1/GO/2 与 ADNI3 是不同阶段的数据批次。"],en:["ADNI","The Alzheimer's Disease Neuroimaging Initiative, a multicenter study with imaging, cognitive, genetic and follow-up data. ADNI1/GO/2 and ADNI3 are data from different phases."]},
  association_models:{zh:["线性、秩关联与稳健回归","三种检验变量关系的方法：线性模型看平均直线关系，秩关联更关注排序，稳健回归降低极端值的影响。多种方法方向一致时，结果通常更稳健。"],en:["Linear, rank and robust analyses","Three ways to test an association: linear analysis estimates an average straight-line relationship, rank analysis focuses on ordering, and robust regression reduces the influence of extreme values. Agreement across methods generally suggests greater robustness."]},
  fdr:{zh:["FDR 与 q 值","同时检验多个结果时控制假阳性的方法。q 值是校正后的统计量；越小表示多重比较后证据越强，但不表示效应一定很大或具有临床价值。"],en:["FDR and q value","A method for controlling false positives when many results are tested. The q value is the adjusted statistic; smaller values indicate stronger evidence after multiple testing, not necessarily a large or clinically important effect."]},
  mci:{zh:["MCI（轻度认知障碍）","认知能力较预期下降、但日常独立生活通常仍基本保留的状态；部分人会进展为痴呆，也有人长期稳定或改善。"],en:["MCI (mild cognitive impairment)","A condition involving greater-than-expected cognitive decline while everyday independence is generally preserved. Some people progress to dementia, while others remain stable or improve."]},
  cox:{zh:["Cox 生存分析","分析某事件是否发生以及何时发生的方法；这里用于研究基线脑体积与随后进展为痴呆的时间关系。"],en:["Cox survival analysis","A method that analyzes whether and when an event occurs. Here it relates baseline brain volume to the subsequent time to dementia progression."]},
  hazard_ratio:{zh:["风险比（HR）","比较事件瞬时发生风险的相对数值。这里 HR 小于 1 表示脑体积较大与较低进展风险相关；它是关联，不直接证明保护作用。"],en:["Hazard ratio (HR)","A relative comparison of the instantaneous event rate. Here, an HR below 1 means larger brain volume is associated with lower progression risk; it does not by itself prove a protective effect."]},
  confidence_interval:{zh:["95% 置信区间（CI）","表示估计值不确定性的范围；区间越宽，估计越不精确。是否跨 1 是解读风险比时的一个统计线索，但仍需结合 q 值、样本量和重复验证。"],en:["95% confidence interval (CI)","A range expressing uncertainty around an estimate; wider intervals indicate less precision. Whether it crosses 1 is one statistical clue for a hazard ratio, but q values, sample size and replication also matter."]},
  cohorts:{zh:["四个队列（TCP / UCLA / COBRE / HCP-EP）","本材料的结果来自四个互相独立的数据集。TCP（Transdiagnostic Connectome Project，跨诊断连接组计划）提供内部实验，纳入双相障碍、精神分裂症／分裂情感性障碍患者和对照。UCLA、COBRE 和 HCP-EP 用于外部验证：UCLA 同时纳入双相和精神分裂症患者，COBRE 以精神分裂症谱系为主，HCP-EP 为早期精神病。外部验证看同一方向能否在其他样本中重复出现。"],en:["Four cohorts (TCP / UCLA / COBRE / HCP-EP)","Results here come from four independent datasets. TCP (the Transdiagnostic Connectome Project) supplies the internal experiment, with bipolar disorder, schizophrenia/schizoaffective disorder and control participants. UCLA, COBRE and HCP-EP provide external validation: UCLA includes both bipolar and schizophrenia participants, COBRE mainly covers the schizophrenia spectrum, and HCP-EP covers early psychosis. External validation asks whether the same direction reappears in other samples."]},
  covariate_adjustment:{zh:["校正（协变量）","统计分析时把已知会影响结果的变量一起放进模型，使比较尽量不被这些因素干扰。本材料用到的协变量包括年龄、性别、扫描站点（参与者是在哪台扫描仪上完成检查）和头动；结构影像部分另含教育程度与遗传祖源。纳入哪些协变量以各案例的“实验怎么做”为准。"],en:["Adjustment (covariates)","Known influences are included in the model so the comparison is not driven by them. Covariates used here include age, sex, scanning site (which scanner measured the participant) and head motion; the structural studies also include education and genetic ancestry. Which covariates apply is stated in each case's methods."]},
  other_networks:{zh:["其他脑网络","视觉网络：处理视觉信息；默认模式网络：安静休息、不专注外部任务时较活跃，与内省和记忆提取有关；背侧注意网络：自上而下地分配注意；额顶网络：参与工作记忆和认知控制；控制网络：统筹其他网络、支持灵活切换任务。"],en:["Other brain networks","Visual network: processes visual information. Default mode network: more active during rest and inward thought, and linked to introspection and memory retrieval. Dorsal attention network: allocates attention in a top-down way. Frontoparietal network: supports working memory and cognitive control. Control network: coordinates other networks and flexible task switching."]},
  bipolar:{zh:["双相障碍","以躁狂或轻躁狂发作与抑郁发作交替出现为特征的精神疾病。"],en:["Bipolar disorder","A psychiatric disorder characterised by episodes of mania or hypomania alternating with depression."]},
  psychosis:{zh:["精神病（精神病性障碍）","出现幻觉、妄想或思维紊乱等精神病性症状的一类障碍；精神分裂症谱系与早期精神病都属于这一范围。"],en:["Psychosis","A group of disorders involving psychotic symptoms such as hallucinations, delusions or disorganised thinking; the schizophrenia spectrum and early psychosis fall in this range."]},
  brain_volume:{zh:["脑区体积（结构性 MRI 指标）","由结构性 MRI 测得的某个脑区或整体脑组织的体积大小。体积差异是描述性测量结果，本身不说明功能变化或病因。"],en:["Regional brain volume (structural MRI)","The volume of a brain region, or of overall brain tissue, measured from structural MRI. A volume difference is a descriptive measurement; it does not by itself establish function or cause."]},
  dementia:{zh:["痴呆（进展结局）","这里的结局是“从轻度认知障碍进展为痴呆”这一事件以及发生时间。痴呆是多种病因共有的临床综合征，不等于某一种特定疾病。"],en:["Dementia (progression outcome)","Here the outcome is progression from mild cognitive impairment to dementia, and when it occurs. Dementia is a clinical syndrome with several causes rather than one specific disease."]},
  beta:{zh:["β（标准化回归系数）","回归模型中表示自变量每变化一个单位、结果变量随之变化多少的系数；标准化后可与其它指标比较。正负只表示关联方向，不等于因果。"],en:["β (standardized regression coefficient)","A regression coefficient describing how much the outcome changes per unit of a predictor; standardised values can be compared across measures. Its sign gives the direction of association, not causation."]},
  p_value:{zh:["P 值","在“两组没有差异（或没有关联）”这一前提下，出现当前或更极端结果的概率；P 越小，越难用偶然解释。同时检验多项指标时应参考校正后的 q 值，单看 P 值容易把偶然当作发现。"],en:["P value","The probability of seeing the current or a more extreme result if there were truly no difference or association. Smaller P values are harder to explain by chance. When many measures are tested, read the adjusted q value as well; a single P value alone can turn chance into an apparent finding."]},
  standard_deviation:{zh:["标准差（SD）","衡量数值在个体之间分散程度的单位。“每增加一个标准差”表示从平均水平向上移动一个分散单位，便于在不同量纲的指标之间比较效应大小。"],en:["Standard deviation (SD)","A measure of how widely values spread between individuals. Per standard deviation means moving one spread unit above the average, which allows effect sizes to be compared across measures with different units."]},
  event_count:{zh:["n / 事件数","n 是该项分析纳入的人数；事件数是随访窗口内真正发生目标结局（如进展为痴呆）的人数。事件数很少时，风险比的估计会很不稳定。"],en:["n / events","n is the number of participants in that analysis; events is how many actually reached the outcome (for example progression to dementia) within the follow-up window. With few events the hazard-ratio estimate is unstable."]},
};
function glossaryKeys(card) {
  // Every card lists the explanations for the nouns that actually appear in its own
  // rendered material, so a reader from another field meets no unexplained jargon.
  const connectivity=["roi","salventattn","sommot","rsfmri","connectivity","pearson","standardized_effect","resampling","spectrum","head_motion","cohorts","covariate_adjustment","other_networks","bipolar","psychosis","p_value"];
  if(["packet-01","packet-02","packet-03","packet-05"].includes(card.id)) return connectivity;
  const genetics=(region,extra=[])=>["apoe","allele_dosage","brain_volume",region,...extra,"baseline","icv","adni","association_models","fdr","beta","covariate_adjustment"];
  if(["packet-36","packet-37","packet-38"].includes(card.id))
    return genetics({"packet-36":"entorhinal","packet-37":"hippocampus","packet-38":"middle_temporal"}[card.id],
                    card.id==="packet-37"?[]:["whole_brain","mci","dementia"]);
  if(["packet-42","packet-44","packet-46"].includes(card.id))
    return ["mci","dementia","brain_volume",{"packet-42":"entorhinal","packet-44":"middle_temporal","packet-46":"whole_brain"}[card.id],
            "baseline","icv","adni","cox","hazard_ratio","confidence_interval","fdr","standard_deviation","event_count","covariate_adjustment","apoe"];
  return [];
}
function terminologyHTML(card) {
  const lang=display.language==="en"?"en":"zh",keys=glossaryKeys(card);
  if(!keys.length)return "";
  const entries=keys.map(key=>TERM_GLOSSARY[key]).filter(Boolean).map(entry=>`<div class="term-entry"><dt>${h(entry[lang][0])}</dt><dd>${h(entry[lang][1])}</dd></div>`).join("");
  const title=lang==="en"?"Quick terminology guide":"术语速查";
  const hint=lang==="en"?"Plain-language explanations for terms used in this case; they do not change the study's original definitions or results.":"本例所用名词的通俗解释；不改变研究中的原始定义和实验结果。";
  return `<details class="terminology" data-user-content><summary>${h(title)} <span>${keys.length}</span></summary><p class="term-hint">${h(hint)}</p><dl>${entries}</dl></details>`;
}
function readableMaterialHTML(card) {
  // The narrative is stored with each session; historical cards keep their original presentation.
  const pre=card.pre, reading=pre.reading, en=display.language==="en", lang=en?"en":"zh";
  const paragraph=note=>`<p data-user-content>${h(note[lang])}</p>`;
  const refs=pre.references.map(ref=>{
    const url=safeUrl(ref.url),label=h(ref.id);
    const detail=reading.reference_details?.[ref.id];
    if(detail) {
      const title=`${label} · ${h(detail.title)}`;
      return `<li class="related-study"><div class="related-study-title">${url?`<a href="${h(url)}" target="_blank" rel="noopener noreferrer">${title} ↗</a>`:title}</div><p class="tiny">${h(detail.journal)} · ${h(detail.year)} · PMID ${h(ref.pmid)}</p><p><strong>${en?"What the study did":"这项研究做了什么"}：</strong>${h(detail.did[lang])}</p><p><strong>${en?"How it relates to this hypothesis":"与当前假设的关系"}：</strong>${h(detail.relation[lang])}</p></li>`;
    }
    return `<li>${url?`<a href="${h(url)}" target="_blank" rel="noopener noreferrer">${label} ↗</a>`:label} · ${h(reading.references[ref.id][lang])}</li>`;
  }).join("");
  let html=`<header class="material-heading"><span class="tag">${en?"Case":"材料"} ${String(current+1).padStart(2,"0")} / ${String(data.cards.length).padStart(2,"0")}</span><h2>${h(pre.title)}</h2><p><span>${h(pre.topic)}</span> · NeuroDiscovery</p></header>${terminologyHTML(card)}`;
  html+=`<section class="material-part">${partHTML("01",en?"What this finding is":"这个发现是什么","")}<div class="hypothesis" id="current-hypothesis"><div class="eyebrow">${en?"The system's hypothesis":"系统提出的假设"}</div><p>${h(pre.hypothesis_plain||pre.hypothesis)}</p>${significanceHTML(card.id)}</div></section>`;
  html+=`<section class="material-part">${partHTML("02",en?"Where it came from":"它从何而来","")}${paragraph(reading.rationale)}<details open><summary>${en?"Related studies":"相关研究"} (${pre.references.length})</summary><ul class="rationale-refs" data-user-content>${refs}</ul></details>`;
  if(reading.feedback) html+=`<section class="block"><h4>${en?"Learning from earlier experiments":"前序实验的启发"}</h4>${reading.feedback_chain?researchChainHTML(reading.feedback_chain):paragraph(reading.feedback)}</section>`;
  html+='</section>';
  html+=`<section class="material-part">${partHTML("03",en?"How it was tested":"实验怎么做","")}${reading.scale?`<div class="study-scale" data-user-content><strong>${en?"Study scale":"实验规模"}</strong>${paragraph(reading.scale)}</div>`:""}${paragraph(reading.methods)}</section>`;
  if(card.post) {
    const process=reading.runtime_process?.length?`<section class="runtime-process" data-runtime-process><h4>${en?"NeuroRuntime experimental process":"NeuroRuntime 实验执行过程"}</h4>${reading.runtime_process.map(step=>`<div class="block"><h5>${h(step.title[lang])}</h5>${paragraph(step.text)}</div>`).join("")}</section>`:"";
    html+=`<section class="material-part">${partHTML("04",en?"What the experiments found":"实验结果","")}${process}${paragraph(reading.results)}${readableResultTablesHTML(card)}</section>`;
  }
  if(card.post && reading.next_research?.zh?.trim() && reading.next_research?.en?.trim()) html+=`<section class="material-part" data-next-research>${partHTML("05",en?"How this case informed the next research round":"本例结果如何进入下一轮研究","")}${reading.next_research_chain?researchChainHTML(reading.next_research_chain,true):paragraph(reading.next_research)}</section>`;
  return html+`<a href="#ratings-form">${en?"Go to the assessment ↓":"前往本份评审题 ↓"}</a>`;
}
function renderMaterial(card) {
  if(multiTopic()) {$("#material").innerHTML=multiTopicMaterialHTML(card);return;}
  const pre=card.pre;
  const heading=`<header class="material-heading"><span class="tag">材料 ${String(current+1).padStart(2,"0")} / ${String(data.cards.length).padStart(2,"0")}</span><h2>${h(pre.title)}</h2>${studyContextHTML(pre)}<p>请分别评价系统能力，并保留测量、诊断与队列的边界。</p></header>`;
  if(!singleRound()) {
    let html=heading+`<div class="hypothesis" id="current-hypothesis">${hypothesisHTML(pre,"待检验的假设")}</div><section class="block"><h3>精确定义</h3><p>${h(pre.definition)}</p></section><section class="block"><h3>为何提出这项假设</h3><p>${h(pre.rationale)}</p><p class="tiny">${h(pre.source_note)}</p></section><section class="block"><h3>证据迁移的限制</h3><p>${h(pre.transfer_limit)}</p></section><details ${data.session.stage==="A"?"open":""}><summary>共同方法、数据接触与统计边界 · 必读</summary>${commonHTML(data.common_pre)}</details><details ${data.session.stage==="A"?"open":""}><summary>既有文献与核查缺口（${pre.references.length} 份）</summary>${referencesHTML(pre.references, card.id)}</details>`;
    if(card.post) {
      const post=card.post;
      html+=`<section class="block"><h3>内部与外部结果</h3>${internalHTML(post.internal_rows)}<div class="result-figures"><div><span>TCP · 联合方向频率</span><b>${pc(post.tcp_joint)}</b></div><div><span>UCLA · 联合方向频率</span><b>${pc(post.ucla_joint)}</b></div></div>${externalHTML(post.external_rows)}<details><summary>逐行模型与原尺度参数</summary>${post.external_rows.map(r=>`<p class="tiny"><strong>${h(r.row_id)} · ${h(r.cohort)} / ${h(r.variant)}</strong><br>模型列：${h(r.columns.join(", "))}<br>β ${f(r.beta,6)}；残差 RMS ${f(r.residual_RMS,6)}</p>`).join("")}</details></section><details open><summary>完整解释边界 · 必读</summary>${commonHTML(data.common_post)}</details><details open><summary>从真实反馈到后轮假设</summary>${feedbackHTML(post)}</details>`;
      if(data.meta.show_curator_summary!==false && post.curator_summary) html+=`<section class="block"><h3>整理者摘要 <span class="tag">非模型总结原文</span></h3><p>${h(post.curator_summary)}</p></section>`;
      html+=`<p class="tiny">${h(post.audit_note)}</p>`;
    } else html+=`<div class="notice"><strong>实验结果尚未解锁</strong><p>完成全部 ${data.cards.length} 份材料的阶段 A 后统一解锁，避免先看到相关材料的结果。无需猜测实验是否成功。</p></div>`;
    $("#material").innerHTML=html+'<a href="#ratings-form">前往本份评审题 ↓</a>';
    return;
  }
  // Single-round reading order: what the finding is, where it came from
  // (prior studies and earlier experiment feedback), the design, then results.
  const post=card.post;
  let html=heading;
  html+=`<section class="material-part">${partHTML("01","这个发现是什么","系统提出的假设与它的精确定义。")}<div class="hypothesis" id="current-hypothesis">${hypothesisHTML(pre,"系统提出的假设",card.id)}</div><section class="block"><h4>精确定义</h4><p>${h(pre.definition)}</p>${definitionPlainHTML(card.id)}</section></section>`;
  html+=`<section class="material-part">${partHTML("02","它从何而来 · 相关研究与前序实验的启发","相关研究做了什么；系统前序假设实验留下了什么反馈。")}<section class="block"><h4>为何提出这项假设</h4>${rationaleHTML(pre, card.id)}</section><details open><summary>既有文献与核查缺口（${pre.references.length} 份）</summary>${referencesHTML(pre.references, card.id)}</details>`;
  if(post) html+=`<section class="block"><h4>前序假设实验与真实反馈 · 自动迭代</h4>${feedbackHTML(post)}</section>`;
  html+=`</section>`;
  // Keep only what experts need: the same datasets, measure and model apply
  // to every card. Row-level parameter detail and boundary notes stay in the
  // exported record, not in this panel.
  html+=`<section class="material-part">${partHTML("03","实验设计 · 模型、数据与指标","所有案例共用同一套数据、测量与模型。")}`
    +`<section class="block"><h4>数据集</h4><p>内部实验用 TCP 队列 234 人（ADHD 13 人、双相 26 人、SZ/SZA 16 人，各比较的对照组 91 人）；外部验证用三个互不相干的独立队列：UCLA 257 人、COBRE 158 人、HCP-EP 145 人。</p></section>`
    +`<section class="block"><h4>测量指标</h4><p>静息态功能连接（fMRI）。每个假设预先锁定一条连接：某个脑分区到某个网络、或两个网络之间连接强度的平均值。看两个读数：病例组与对照的差异大小（标准化效应），以及差异方向在重复抽样中的一致程度（联合方向频率）。</p></section>`
    +`<section class="block"><h4>统计模型</h4><p>普通最小二乘（OLS）回归，比较病例组与对照，调整年龄、性别和扫描站点；外部分析另调整头动。结果的稳定性用重复抽样检验：内部 512 次、外部 2,000 次。候选假设由系统在神经影像知识图谱上提出、经三个大语言模型评审角色讨论；验证刻意使用可解释的线性模型，没有使用深度学习。</p></section>`
    +`</section>`;
  if(post) {
    html+=`<section class="material-part">${partHTML("04","实验结果 · 内部与外部验证","内部固定测量与外部组成、敏感性结果。")}${internalHTML(post.internal_rows)}<div class="result-figures"><div><span>TCP · 联合方向频率</span><b>${pc(post.tcp_joint)}</b></div><div><span>UCLA · 联合方向频率</span><b>${pc(post.ucla_joint)}</b></div></div>${externalHTML(post.external_rows)}`;
    if(data.meta.show_curator_summary!==false && post.curator_summary) html+=`<section class="block"><h4>整理者摘要 <span class="tag">非模型总结原文</span></h4><p>${h(post.curator_summary)}</p></section>`;
    html+=`<p class="tiny">${h(post.audit_note)}</p></section>`;
  } else html+=`<div class="notice"><strong>实验结果尚未解锁</strong><p>完成全部 ${data.cards.length} 份材料的阶段 A 后统一解锁，避免先看到相关材料的结果。无需猜测实验是否成功。</p></div>`;
  $("#material").innerHTML=html+'<a href="#ratings-form">前往本份评审题 ↓</a>';
}
const QUESTION_DISPLAY_ORDER=["novelty","grounding","feedback","design","validation","value"];
function displayQuestions(list) {
  // Single-round question order follows the material: the finding itself,
  // its origins and feedback-driven iteration, the design, then results.
  // Answers stay keyed by question id; only presentation order changes.
  return [...list].sort((a,b)=>{
    const ia=QUESTION_DISPLAY_ORDER.indexOf(a.id), ib=QUESTION_DISPLAY_ORDER.indexOf(b.id);
    return (ia<0?QUESTION_DISPLAY_ORDER.length:ia)-(ib<0?QUESTION_DISPLAY_ORDER.length:ib);
  });
}
function questionHTML(q,i,card,disabled=false) {
  const value=drafts[card.id].answers[q.id];
  const number=singleRound()?i+1:(data.scoring?data.questions.findIndex(x=>x.id===q.id)+1:i+1);
  return `<fieldset><legend><span class="question-number">${String(number).padStart(2,"0")}</span>${h(q.title)}</legend>${q.capability?`<span class="tag">评测能力 · ${h(q.capability)}</span>`:""}${q.help?`<p class="q-help">${h(q.help)}</p>`:""}${q.options.map(option=>`<label class="choice ${["unable","insufficient","outside","not_assessable"].includes(option.value)?"unable":""}"><input type="radio" name="${h(q.id)}" value="${h(option.value)}" ${value===option.value?"checked":""} ${disabled?"disabled":""}><span>${h(option.label)}</span></label>`).join("")}</fieldset>`;
}
function scoreHTML() {
  const summary=data.score_summary;
  if(!summary) return "";
  return `<details class="score-summary"><summary>查看本次能力评分汇总</summary><p>综合均分：<strong>${f(summary.composite_mean,2)} / 5</strong> · ${multiTopic()?"适用项均有数值的材料":"六项均有数值的材料"} ${summary.complete_case_count} / ${summary.case_count} 份。</p><table><thead><tr><th>能力</th><th>均分 / 5</th><th>有效材料数</th></tr></thead><tbody>${Object.values(summary.dimensions).map(d=>`<tr><td>${h(d.label)}</td><td>${f(d.mean,2)}</td><td>${d.n} / ${d.applicable_count??summary.case_count}</td></tr>`).join("")}</tbody></table><p class="tiny">仅汇总本次个人评价；不同能力可有不同有效数。未评分项及原因单独保留。全部统计使用导出中的未舍入数值。</p></details>`;
}
function renderQuestions(card) {
  const stage=data.session.stage,complete=stage==="complete",phase=reviewPhase();
  $("#question-head").innerHTML=`<h2>${complete?"已完成的独立判断":phase==="A"?"假设与设计初评":"完整成果评审"}</h2><p class="instruction">${complete?"答案已锁定，可导出完整记录。":"每题选择一项，不预选答案。对不同维度分别判断，避免用单一印象代替评价。"} <a href="#material">回看材料 ↑</a></p>${complete?scoreHTML():""}`;
  if(singleRound()) $("#question-head").innerHTML+=`<div class="hypothesis hypothesis-reminder"><div class="eyebrow">当前案例 · 假设对照</div><p>${h(card.pre.hypothesis_plain||card.pre.hypothesis)}</p><a href="#current-hypothesis">回看假设、方法与结果 ↑</a></div><p class="q-help">请阅读本例完整材料后作答。</p>`;
  let formHTML="";
  if(phase==="B") {
    formHTML=`<details class="locked-answers"><summary>查看阶段 A 已锁定判断</summary><dl class="breakdown">${stageQuestions("A").map(q=>`<dt>${h(q.title)}</dt><dd>${h(q.options.find(o=>o.value===drafts[card.id].answers[q.id])?.label||"未回答")}</dd>`).join("")}</dl><p>${h(drafts[card.id].notes.A)}</p></details>`;
  }
  formHTML+=(singleRound()?displayQuestions(stageQuestions(phase)):stageQuestions(phase)).map((q,i)=>questionHTML(q,i,card,complete)).join("");
  if(phase==="B" && data.issue_options.length) formHTML+=`<fieldset><legend>可选 · 主要问题标签</legend><div class="issue-grid">${data.issue_options.map(issue=>`<label class="choice"><input type="checkbox" name="issue" value="${h(issue)}" ${drafts[card.id].issues.includes(issue)?"checked":""} ${complete?"disabled":""}>${h(issue)}</label>`).join("")}</div></fieldset>`;
  formHTML+=`<label class="notes-label">可选 · 关键理由、缺失证据或原文核查线索<textarea name="note" maxlength="4000" placeholder="例如：P2 不能直接支持本次测量；还需双相独立样本…" ${complete?"disabled":""}>${h(drafts[card.id].notes[phase])}</textarea></label>`;
  $("#ratings-form").innerHTML=formHTML;
  $("#question-bottom").innerHTML=data.scoring?`<p class="tiny">每项能力按所选描述记为 1–5 分；六项等权。无法判断的回答单独记录，不纳入数值均分。</p>`:`<p class="tiny">此会话保留旧版问卷与计分规则，不自动转换为新版能力分数。</p>`;
  if(multiTopic()) $("#question-bottom").innerHTML='<p class="tiny">每项 1–5 分，适用项等权。无法判断不计分。</p>';
  if(!complete) {
    $("#ratings-form").onchange=captureForm;
    $("textarea[name=note]").addEventListener("input",captureForm);
  }
}
function captureForm() {
  const card=data.cards[current],phase=data.session.stage;
  if(phase==="complete") return;
  for(const q of stageQuestions()) {
    const selected=$(`#ratings-form input[name="${q.id}"]:checked`);
    if(selected) drafts[card.id].answers[q.id]=selected.value;
  }
  drafts[card.id].notes[phase]=$("textarea[name=note]").value;
  if(phase==="B") drafts[card.id].issues=Array.from(document.querySelectorAll('#ratings-form input[name="issue"]:checked')).map(el=>el.value);
  dirty.add(card.id); saveStatus("有更改，正在自动保存…"); refreshProgress();
  clearTimeout(saveTimer); saveTimer=setTimeout(()=>saveAll().catch(()=>{}),650);
}
async function saveAll() {
  clearTimeout(saveTimer);
  if(saving) return saving;
  saving=(async()=>{
    while(dirty.size || pending) {
      if(!pending) {
        const cardId=dirty.values().next().value; dirty.delete(cardId);
        const draft=drafts[cardId],stage=data.session.stage;
        const answers=Object.fromEntries(stageQuestions(stage,cardId).filter(q=>draft.answers[q.id]!==undefined).map(q=>[q.id,draft.answers[q.id]]));
        const seconds=Math.min(activeSeconds,28800); activeSeconds=0;
        pending={revision:data.session.revision,request_id:uid(),card_id:cardId,answers,note:draft.notes[stage],issues:stage==="B"?[...draft.issues]:[],active_seconds:seconds,
          ...(data.presentation?{display:{language:display.language,version:data.presentation.version,sha256:data.presentation.sha256,renderer_revision:"visible-results-v3"}}:{})};
      }
      saveStatus("正在保存到本机…");
      try {
        data=await request(sessionPath("/save"),{method:"POST",body:JSON.stringify(pending)}); pending=null;
        saveStatus(`已保存 · ${new Date().toLocaleTimeString(display.language==="en"?"en-GB":"zh-CN",{hour:"2-digit",minute:"2-digit",second:"2-digit"})}`);
      } catch(error) {
        saveStatus("保存失败，尚未保存的输入仍在当前页面。",true); message(error.message); throw error;
      }
    }
  })().finally(()=>{saving=null;});
  return saving;
}
async function navigate(index) {try {await saveAll();current=index;render();StudyWorkspace.resetReadingPosition();window.scrollTo({top:65,behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});} catch{/* stay with unsaved input */}}
function render() {
  const stage=data.session.stage;
  const assignmentId=data.session.assignment_id;
  $("#session-code").textContent=`${data.session.profile.code} · ${data.meta.version_label}${assignmentId&&assignmentId!=="ALL"?` · ${assignmentId}`:""}`;
  $("#order-note").textContent=`${data.cards.length} 份材料顺序随机。没有标准答案。`;
  $("#stage-title").textContent=stage==="complete"?"评审已完成":singleRound()?"逐例评审 / 完整研究成果":stage==="A"?"阶段 A / 假设与方法":"阶段 B / 完整研究结果";
  $("#stage-help").textContent=stage==="complete"?(data.scoring?"感谢你的独立判断。答案、各能力均分及综合均分已保存，可查看与导出。":"旧版答案已保存，可查看与导出；不会自动改用新版计分规则。"):singleRound()?`每次阅读一个完整案例，完成六项评分后继续下一份。共 ${data.cards.length} 份，不再分两轮；提交前可回看和修改。`:stage==="A"?`每份先评 ${stageQuestions("A").length} 项能力；完成全部 ${data.cards.length} 份初评后统一显示实验结果。`:`初评已锁定。现在结合内外部证据，完成每份剩余 ${stageQuestions("B").length} 题。`;
  if(multiTopic() && stage!=="complete") $("#stage-help").textContent="每次阅读一份完整材料，完成能力评分后继续。提交前可回看和修改。";
  $("#steps").innerHTML=singleRound()?'<div class="step active"><b>1</b> 单轮 · 逐例完成</div>':`<div class="step ${stage==="A"?"active":""}"><b>A</b> 假设与设计</div><div class="step ${stage!=="A"?"active":""}"><b>B</b> 结果与贡献</div>`;
  renderMaterial(data.cards[current]); renderQuestions(data.cards[current]); refreshProgress();
}
function downloadJSON(value,filename) {
  const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:"application/json;charset=utf-8"}));
  const anchor=document.createElement("a");anchor.href=url;anchor.download=filename;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

$("#auth-form").addEventListener("submit",async event=>{
  event.preventDefault();try {const result=await request("/api/studies/auth",{method:"POST",body:JSON.stringify({password:$("#password").value})}); authToken=result.token;studyStorage().setItem("neurodiscoveryStudyToken",authToken);$("#password").value="";await loadConfig();}catch(error){message(error.message);}
});
$("#setup-form").addEventListener("submit",async event=>{
  event.preventDefault();$("#start").disabled=true;
  const payload=Object.fromEntries(new FormData(event.target));
  try {credentials=null;const result=await request(`${API}/sessions`,{method:"POST",body:JSON.stringify(payload)});credentials={id:result.session.id,token:result.session_token};delete result.session_token;storeCredentials();try{localStorage.setItem(PARTICIPANT_CODE_KEY,String(payload.code||""));}catch(_e){}initialize(result);}
  catch(error){message(error.message);}finally{$("#start").disabled=false;}
});
$("#resume").addEventListener("click",async()=>{credentials=readCredentials();try{initialize(await request(sessionPath()));}catch(error){message(error.message);}});
$("#separate").addEventListener("click",async()=>{if(await dialog("开始独立的新会话？","旧答案不会删除，但此浏览器会改记新会话的继续码。若需要返回旧会话，请先进入原会话保存继续码。","开始新会话")){$("#resume-banner").hidden=true;$("#setup-form").hidden=false;}});
$("#restore").addEventListener("click",async()=>{try{const value=JSON.parse($("#restore-code").value);if(!value.id||!value.token)throw new Error("继续码格式不正确。");credentials={id:value.id,token:value.token};const result=await request(sessionPath());storeCredentials();$("#restore-code").value="";initialize(result);}catch(error){message(error.message);}});
$("#previous").addEventListener("click",()=>navigate(Math.max(0,current-1)));
$("#next").addEventListener("click",()=>navigate(Math.min(data.cards.length-1,current+1)));
$("#advance").addEventListener("click",async()=>{
  try {
    await saveAll();const stage=data.session.stage;
    const missing=data.cards.find(card=>completeCount(card.id,stage)<stageQuestions(stage,card.id).length);
    if(missing){await navigate(data.cards.indexOf(missing));message("请完成这份材料的所有必答问题；确实无法判断时可选择对应选项。");return;}
    const yes=await dialog(stage==="A"?"锁定阶段 A 并显示结果？":"完成并锁定本次评审？",stage==="A"?`全部 ${data.cards.length} 份材料的初评将不可回改。随后显示全部内外部结果及反馈链。`:(data.scoring?"全部答案将按当前材料与计分版本保存并锁定。提交后可查看能力均分、综合均分及有效数量。":"旧版答案将保存并锁定；保留原有题目与计分含义。"),stage==="A"?"锁定并解锁结果":"完成评审");
    if(!yes)return;
    const result=await request(sessionPath(stage==="A"?"/reveal":"/submit"),{method:"POST",body:JSON.stringify({request_id:uid(),revision:data.session.revision})});initialize(result);saveStatus(stage==="A"?"初评已锁定 · 结果已解锁":"完成记录已保存");
  }catch(error){message(error.message);}
});
function rememberEvaluation() {
  try { EvaluationExport.remember('discovery',data.session.profile.code,{id:credentials.id,token:credentials.token}); }
  catch(error) { message(error.message); }
}
async function saveProgress() {
  if(!data)return;
  // Include reading time even when no answer changed since the previous save.
  if(data?.session.stage!=="complete" && activeSeconds && data?.cards[current]) dirty.add(data.cards[current].id);
  await saveAll(); rememberEvaluation(); saveStatus("已保存");
}
async function exportEvaluations() {
  const buttons=[$('#export'),$('#export-complete'),$('#export-saved')];buttons.forEach(button=>button.disabled=true);
  try {
    await saveProgress();
    const code=!$('#workspace').hidden?data.session.profile.code:$('#setup-form input[name=code]').value.trim();
    const result=await EvaluationExport.exportResults({code,token:authToken,language:display.language});
    if(!result?.canceled) message(result.downloadRequested?"已请求下载，请将 JSON 文件交给研究者。":`已导出：${result.path}`);
  } catch(error) { message(error.message); }
  finally { buttons.forEach(button=>button.disabled=false); }
}
$('#save-progress').addEventListener('click',()=>saveProgress().catch(error=>message(error.message)));
$('#export').addEventListener('click',exportEvaluations);
$('#export-complete').addEventListener('click',exportEvaluations);
$('#export-saved').addEventListener('click',exportEvaluations);
$("#resume-code").addEventListener("click",async()=>{try{await saveAll();await navigator.clipboard.writeText(JSON.stringify(credentials));message("继续码已复制，请存到安全位置，不要发在公开群或论文材料中。");}catch{message("无法复制。请保留本浏览器，可随时使用入口的“继续评审”。");}});
$("#pause").addEventListener("click",async()=>{try{await saveProgress();if(embedded){window.parent.postMessage({type:'neurodiscovery:close-study-workspace'},location.origin);return;}$("#workspace").hidden=true;$("#welcome").hidden=false;await loadConfig();window.scrollTo(0,0);message("已保存，可关闭页面。下次在本浏览器继续。");}catch{/* Do not pretend unsaved data was persisted. */}});
window.addEventListener("beforeunload",event=>{if(dirty.size||pending||saving){event.preventDefault();event.returnValue="";}});
["pointerdown","keydown","scroll"].forEach(name=>document.addEventListener(name,()=>{lastActivity=Date.now();},{passive:true,capture:name==='scroll'}));
setInterval(()=>{if(data && hostVisible && !$("#workspace").hidden && data.session.stage!=="complete" && document.visibilityState==="visible" && Date.now()-lastActivity<60000) activeSeconds++;},1000);
const params=new URLSearchParams(location.search);
const embedded=params.get('embedded')==='1' && window.parent!==window;
const display=DiscoveryI18n.mount(document,{initialLanguage:params.get('lang'),onLanguage(language){
  // Language changes are not answers. Store display provenance alongside the
  // next normal save without rebuilding or discarding any in-flight input.
  if(data && !$('#workspace').hidden && data.session.stage!=='complete') {
    dirty.add(data.cards[current].id);clearTimeout(saveTimer);saveTimer=setTimeout(()=>saveAll().catch(()=>{}),100);
  }
  // Reference notes ship bilingual strings (not catalog entries), so the
  // material re-renders on language change; reading position is preserved.
  if(data && !$('#workspace').hidden) {
    const pane=$('#material'),top=pane.scrollTop;
    renderMaterial(data.cards[current]);
    pane.scrollTop=top;
  }
  if(embedded)window.parent.postMessage({type:'neurodiscovery:set-study-language',language},location.origin);
}});
function hostAppearance(payload) {
  StudyWorkspace.applyAppearance(payload);
  // Match the extension study (study.html applyHostTextScale): zoom the whole
  // app container so layout widths and fixed-px fonts scale identically on
  // both study entries under the host textScale setting.
  if(payload&&payload.scale!==undefined&&typeof document!=='undefined'){
    const raw=payload.scale===null||payload.scale===''?1:Number(payload.scale);
    const scale=Number.isFinite(raw)?Math.min(1.5,Math.max(.8,Math.round(raw*10)/10)):1;
    const app=document.getElementById('app');
    if(app){app.style.zoom=String(scale);app.style.minWidth=`${960/scale}px`;}
  }
  if(payload.language)display.setLanguage(payload.language,{notify:false});
}
hostAppearance({theme:params.get('theme'),scale:params.get('textScale')||1});
window.addEventListener('message',event=>{
  if(!embedded || event.origin!==location.origin || event.source!==window.parent)return;
  const payload=event.data;if(!payload||typeof payload!=='object')return;
  if(['neurodiscovery:appearance','neurodiscovery:theme','neuroclaw:text-scale'].includes(payload.type))hostAppearance(payload);
  if(payload.type==='neurodiscovery:study-visibility') {
    hostVisible=payload.visible===true;
    if(!hostVisible)void saveAll().catch(()=>{});
  }
});
const SIDEBAR_WIDTH_KEY="neurodiscovery.discovery-study.sidebar-width.v1";
const sidebarDefaultWidth=()=>document.documentElement.dataset.embedded==="true"?164:180;
function sidebarWidthBounds() {
  const available=$(".review-grid")?.parentElement?.clientWidth||window.innerWidth;
  return {minimum:120,maximum:Math.max(200,Math.min(400,Math.floor(available*.32)))};
}
function setSidebarWidth(value,persist=true) {
  const grid=$(".review-grid"),handle=$("#sidebar-resizer");
  if(!grid||!handle)return sidebarDefaultWidth();
  const bounds=sidebarWidthBounds();
  const width=Math.round(Math.max(bounds.minimum,Math.min(bounds.maximum,Number(value)||sidebarDefaultWidth())));
  grid.style.setProperty("--sidebar-w",`${width}px`);
  handle.setAttribute("aria-valuemin",String(bounds.minimum));
  handle.setAttribute("aria-valuemax",String(bounds.maximum));
  handle.setAttribute("aria-valuenow",String(width));
  if(persist){try{localStorage.setItem(SIDEBAR_WIDTH_KEY,String(width));}catch(_err){}}
  return width;
}
function initializeSidebarResizer() {
  const grid=$(".review-grid"),handle=$("#sidebar-resizer");
  if(!grid||!handle||typeof grid.style?.setProperty!=="function")return;
  let stored=null; try{stored=Number(localStorage.getItem(SIDEBAR_WIDTH_KEY))||null;}catch(_err){}
  if(stored)setSidebarWidth(stored,false);
  let pointerId=null,startX=0,startWidth=sidebarDefaultWidth();
  const finish=event=>{
    if(pointerId===null||event.pointerId!==pointerId)return;
    if(handle.hasPointerCapture(pointerId))handle.releasePointerCapture(pointerId);
    pointerId=null;document.body.classList.remove("sidebar-resizing");
    setSidebarWidth($(".sidebar")?.getBoundingClientRect().width,true);
  };
  handle.addEventListener("pointerdown",event=>{
    if(event.button!==0)return;
    pointerId=event.pointerId;startX=event.clientX;
    startWidth=$(".sidebar")?.getBoundingClientRect().width||sidebarDefaultWidth();
    handle.setPointerCapture(pointerId);document.body.classList.add("sidebar-resizing");event.preventDefault();
  });
  handle.addEventListener("pointermove",event=>{
    if(pointerId===event.pointerId)setSidebarWidth(startWidth+(event.clientX-startX),false);
  });
  handle.addEventListener("pointerup",finish);handle.addEventListener("pointercancel",finish);
  handle.addEventListener("keydown",event=>{
    if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;
    const bounds=sidebarWidthBounds(),current=$(".sidebar")?.getBoundingClientRect().width||sidebarDefaultWidth();
    setSidebarWidth(event.key==="Home"?bounds.minimum:event.key==="End"?bounds.maximum:current+(event.key==="ArrowRight"?16:-16),true);
    event.preventDefault();
  });
  handle.addEventListener("dblclick",()=>{
    grid.style.removeProperty("--sidebar-w");
    try{localStorage.removeItem(SIDEBAR_WIDTH_KEY);}catch(_err){}
    handle.setAttribute("aria-valuenow",String(sidebarDefaultWidth()));
  });
}
$('#close-study').hidden=!embedded;
function closeChoice() {
  return new Promise(resolve=>{
    const modal=$("#close-dialog");
    const finish=value=>{modal.close();resolve(value);};
    $("#close-discard").onclick=()=>finish("discard");
    $("#close-save").onclick=()=>finish("save");
    modal.oncancel=event=>{event.preventDefault();finish("stay");};
    modal.showModal();
  });
}
$('#close-study').addEventListener('click',async()=>{
  const close=(discarded=false)=>window.parent.postMessage({type:'neurodiscovery:close-study-workspace',discarded},location.origin);
  const active=data && !$("#workspace").hidden && data.session.stage!=="complete";
  if(!active){close();return;}
  const choice=await closeChoice();
  if(choice==="stay")return;
  if(choice==="discard"){
    try{await request(sessionPath(),{method:"DELETE"});EvaluationExport.forget('discovery',credentials.id);}catch(error){message(error.message);return;}
    credentials=null;dirty.clear();try{localStorage.removeItem(STORAGE);}catch(_e){}
    close(true);return;
  }
  try{await saveProgress();close();}catch{/* Keep the page open with unsaved input. */}
});
loadConfig();
initializeSidebarResizer();
