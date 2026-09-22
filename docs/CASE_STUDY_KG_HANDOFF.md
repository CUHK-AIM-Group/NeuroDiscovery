# Case Study and KG handoff

Updated: 2026-08-10

This document is a short, context-free introduction for a new session. The
authoritative operational state is
[`neurooracle/data/full_v2/README.md`](../neurooracle/data/full_v2/README.md),
the machine-readable formal snapshot is
[`neurooracle/data/full_v2/CURRENT_STATE.json`](../neurooracle/data/full_v2/CURRENT_STATE.json),
and the active semantic-review rules are
[`neurooracle/src/CASE_STUDY_MEMBERSHIP_RUBRIC_V2.md`](../neurooracle/src/CASE_STUDY_MEMBERSHIP_RUBRIC_V2.md).

## 给其他会话的中文摘要

- 现在没有独立的 Case Study 3。原 Case 3 的 15 个子任务已经升格，和
  Case 1、Case 2 并列，合计 17 个正式 Case Study ID。
- `hindcasting` 是可用于全部 Case Study 的按时间验证协议，不是 Case
  Study ID；`general` 是所有保留 claim 共享的语料层，也不是 Case Study
  ID。
- claim 归属允许多选；paper 归属必须由该 paper 全部 claim 的归属并集
  计算。一个没有 Case Study 标签的 claim 仍保留在 general 中。
- Case 1 的新标准是：至少一种确诊疾病存在脑区、脑网络、神经系统或
  神经影像变化/关联；不再要求单篇论文自身跨三种疾病。跨诊断假设可以
  由多篇单病种研究组合支持。
- Case 2 采用与 Case 1 相同的“直接贡献”原则：claim 只要直接支持
  遗传/通路 -> 脑影像/神经生理，或者基线脑影像/神经生理 -> 后续临床/
  认知结局中的任一环节即可；不要求单篇论文完成整条中介链。
- 图谱正式归属字段只使用 `claim_case_study_ids` 和
  `paper_case_study_ids`；旧 Case 3 父子层级字段已经从正式文件移除。
- 当前正式 KG 为 252,739 篇唯一论文、312,376 个 claim。机器可读实时数字
  以 `CURRENT_STATE.json` 为准，不复制聊天记录中的旧统计。
- KG v3 收集期间，新 claim 使用 `2026-08-10.peer17.v2` 和
  `case_study_membership_contract.v4`；旧 claim 与旧 seal 暂不重映射。完成
  全部 v3 收集后，再对整图执行一次统一的新 epoch 重审。
- 通过当前 fail-closed seal、论文去重和原子 dry-run 的完整批次可以注入；
  未完成、未封印或局部人工标签不得直接写入正式 KG。
- `final_complete` 现在遵循内容寻址的不可变契约：证据、rubric 和 17-ID
  registry 未变化时，任何后续模型或会话都必须直接复用该结果，不得重审；
  正式投影会写入 `audit_key` 与 `decision_sha256`。只有证据或定义变化才
  能建立新的版本化 audit epoch，旧结果不得原地覆盖。
  当前 rubric SHA-256 为
  `fa2dca6d8e2d491e30f088810f00319feb2480906604030221a845756808fafe`，
  registry SHA-256 为
  `9124da847c91a753d457337845dfa31bb1f11a584adf5945fa615008305ae7fa`。

## What changed in the Case Study model

- The old standalone `case3` parent was removed. Its 15 former sub-tasks are
  now peer Case Studies alongside Case 1 and Case 2, for 17 formal IDs total.
- Hindcasting is no longer a Case Study. It is a validation protocol that can
  be applied to every Case Study when paper chronology and membership are
  known.
- `general` is the shared corpus, not a stored Case Study ID. A useful claim
  may have no Case Study label and still remain in the general corpus.
- Membership is non-exclusive: one claim may support several Case Studies, and
  a paper belongs to the union of the memberships of all of its claims.
- Case 1 no longer requires one paper to cover at least three diagnoses. Direct
  evidence of a neural or imaging change in one diagnosed disease is enough;
  cross-diagnostic hypotheses can be assembled across such papers.
- Case 2 now follows the same direct-contribution rule as Case 1. A claim may
  provide either the genetic/pathway -> neural link or the baseline neural ->
  later clinical/cognitive outcome link; one paper need not complete the chain.
- The 15 promoted scopes are `biomarker_discovery`, `disease_subtyping`,
  `progression_prediction`, `imaging_genetics`, `differential_diagnosis`,
  `drug_response_prediction`, `personalised_treatment`, `drug_repurposing`,
  `adverse_event_prediction`, `neuromodulation_target`,
  `functional_localization`, `cognitive_decoding`, `connectome_behavior`,
  `brain_age`, and `prognosis`.
- `drug_repurposing` remains a valid ID but is not a current collection/top-up
  target.

## What changed in the graph

- Canonical membership now uses only `claim_case_study_ids` and
  `paper_case_study_ids`; obsolete parent/sub-task scope fields were removed.
- Claim membership is assigned from direct evidence, then paper membership is
  recomputed as the union of its claims. Previous labels are not treated as
  truth.
- The current formal graph has 252,739 unique papers and 312,376 claims. The
  exact 17-scope coverage table is maintained in the authoritative README and
  `CURRENT_STATE.json`.
- During v3 collection, new claims use the v2 rubric/v4 seal while historical
  claims keep their immutable prior seals. One whole-graph re-audit is planned
  after collection, rather than repeatedly re-auditing during expansion.

## Instructions for another session

1. Read the three authoritative files linked at the top before changing Case
   Study membership or the formal KG.
2. Read `CURRENT_STATE.json` for live graph coverage; do not trust a copied
   progress or coverage number.
3. Treat labels as claim-level, evidence-based, and non-exclusive. Empty is a
   valid Case Study decision and means general-only.
4. Do not write `general`, `hindcasting`, or the removed `case3` ID into formal
   membership arrays.
5. Mutate the formal KG only through a complete, sealed, paper-deduplicated,
   dry-run-validated atomic injection.

The automated primary-review runner supports both `chat_completions` and
`responses` wire APIs. API output is still schema- and gate-validated before it
enters the re-audit ledger; it never writes directly to the formal KG.

Do not import the compact GPT-5.5 concurrency benchmarks created on 2026-08-03.
At 64 workers and 12 claims/request the proxy completed 768/768 claims at about
10,782 claims/hour with no transport failures, but replay against 384 immutable
human-finalized claims produced only 0.810 label micro-F1 under the benchmark
prompt. Later calibration, singleton, no-context, and xhigh variants remained at
approximately 0.824--0.852 and all failed the quality gate. The completed
five-specialist grouped replay was also worse: 180/192 valid claims, 0.795 label
micro-F1, 33.33% exact decisions, and 12 technical failures. The completed
benchmark directories contain `DO_NOT_IMPORT.json`, and the importer refuses
quarantined directories. Continue with the validated three-shard
`gpt-5.6-sol high` host-review route unless a replacement passes the frozen
human-gold replay first.

The shard merger must preserve an explicit JSON-boolean
`needs_secondary_review`. This was fixed before batch 14 was imported; the
core re-audit suite contains a regression test for both preservation and
non-boolean rejection.
