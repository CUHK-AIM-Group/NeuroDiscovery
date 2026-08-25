from __future__ import annotations

"""Mass batch: research_tooling tasks T331-T395 (65 tasks)."""

from neurobench_taskkit import body, std_eval

CAT = "research_tooling"


def _t(num, slug, title, desc, ins=None, cons=None, outs=None, ev=None):
    folder = f"T{num}_{slug}"
    return (num, slug, CAT, title,
            body(folder, desc, ins, cons, outs, ev, save_to=std_eval(folder)))


TASKS = []

# --- A. Academic searches (T01 pattern) on neuroimaging topics, T331-T342 ------
_SEARCH_TOPICS = [
    ("search_hippo_subfields", "hippocampal subfields episodic memory"),
    ("search_graph_connectome", "graph theory functional connectome"),
    ("search_dl_brain_age", "deep learning brain age prediction"),
    ("search_harmonization_fmri", "multi-site harmonization resting-state fMRI"),
    ("search_asl_aging", "arterial spin labeling perfusion aging"),
    ("search_qsm_iron", "quantitative susceptibility mapping iron neurodegeneration"),
    ("search_7t_laminar", "7T laminar fMRI cortical layers"),
    ("search_fetal_segmentation", "fetal brain MRI segmentation deep learning"),
    ("search_tms_connectivity", "TMS depression functional connectivity target"),
    ("search_sleep_fmri", "sleep functional MRI default mode"),
    ("search_epilepsy_network", "epilepsy network analysis EEG fMRI"),
    ("search_neurofeedback", "real-time fMRI neurofeedback clinical"),
]
for i, (slug, topic) in enumerate(_SEARCH_TOPICS):
    TASKS.append(_t(
        331 + i, slug, f"Academic Search: {topic.title()}",
        f'Search for the most recent papers related to **"{topic}"** from '
        "multiple academic platforms (arXiv, PubMed, Semantic Scholar), "
        "deduplicate across platforms, and save structured results.",
        ins=None,
        cons=["Time range: last 180 days.",
              "20 papers per platform (60 total minimum), newest first.",
              "Deduplicate by DOI/title across platforms.",
              "Tolerate partial platform failures without failing the run."],
        outs=["`results.json` (metadata + per-platform paper lists with "
              "title/authors/published/url/abstract/doi)",
              "`search_summary.md` (query, counts, date coverage)"]))

# --- B. Systematic multi-database searches with protocol, T343-T347 ---------------
_SYST = [
    ("syst_abcd_adolescent", "adolescent brain development ABCD study"),
    ("syst_gnn_alzheimer", "graph neural network Alzheimer classification MRI"),
    ("syst_taskfmri_individual", "task fMRI individual differences reliability"),
    ("syst_motion_correction", "head motion correction resting-state fMRI methods"),
    ("syst_functional_gradients", "connectopic mapping functional gradients cortex"),
]
for i, (slug, topic) in enumerate(_SYST):
    TASKS.append(_t(
        343 + i, slug, f"Systematic Search Protocol: {topic.title()}",
        f'Run a systematic-style multi-database search for "{topic}": '
        "document the query strings per database (PubMed, Web of Science "
        "alternatives like OpenAlex, Scopus-alternative Semantic Scholar), "
        "apply date/language filters, deduplicate, and emit the counts trail.",
        ins=None,
        cons=["Query strings documented verbatim per database.",
              "Last 5 years, English only; state filters explicitly.",
              "Counts trail: retrieved per DB -> deduplicated -> final."],
        outs=["`search_protocol.md` (queries + filters)",
              "`merged_results.csv`", "`counts_trail.json`"]))

# --- C. Citation / bibliometrics, T348-T353 --------------------------------------------
_BIB = [
    (348, "cocitation_network", "Co-Citation Network Analysis",
     "Build a co-citation network for a seed corpus: papers frequently cited "
     "together form clusters; identify the intellectual base of the topic.",
     ["Seed corpus from a provided reference list (DOIs).",
      "Threshold for co-citation edges documented (e.g. >= 3)."],
     ["`cocitation_graph.json`", "`cluster_summary.md`"]),
    (349, "bibliographic_coupling", "Bibliographic Coupling Map",
     "Compute bibliographic coupling among recent papers on a topic (shared "
     "references) and cluster them into research fronts.",
     ["Corpus from OpenAlex/Semantic Scholar query; query documented.",
      "Report the top coupled clusters with representative papers."],
     ["`coupling_edges.csv`", "`research_fronts.md`"]),
    (350, "author_collab_network", "Author Collaboration Network",
     "Build the co-authorship network around a seed author or paper set: "
     "nodes = authors, edges = co-authored papers, with community detection.",
     ["Cap at 500 authors; selection rule documented.",
      "Disambiguate authors by OpenAlex/ORCID where possible."],
     ["`collab_network.json`", "`top_collaborators.csv`"]),
    (351, "keyword_cooccurrence_map", "Keyword Co-Occurrence Map",
     "Extract author keywords from a corpus and build a keyword "
     "co-occurrence map with clusters, revealing topic structure.",
     ["Minimum keyword occurrence 5.", "Clusters labeled by their top "
      "keywords."],
     ["`keyword_network.json`", "`keyword_clusters.md`"]),
    (352, "backward_snowball", "Backward Snowball Sampling",
     "From a seed set of papers, perform backward snowballing: collect all "
     "references, screen by title/abstract criteria, and iterate one level.",
     ["Screening criteria file required as input.",
      "One iteration level; counts per stage reported."],
     ["`snowball_results.csv`", "`stage_counts.json`"]),
    (353, "forward_snowball", "Forward Snowball Sampling",
     "From a seed set of papers, perform forward snowballing: collect citing "
     "papers via OpenAlex/Semantic Scholar, screen, and iterate one level.",
     ["Citing-paper API source documented.",
      "One iteration level; counts per stage reported."],
     ["`forward_results.csv`", "`stage_counts.json`"]),
]
for num, slug, title, desc, cons, outs in _BIB:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["Seed paper list or query (required)",
                         "Screening criteria where applicable"],
                    cons=cons, outs=outs))

# --- D. Registries / grants / protocols, T354-T358 -------------------------------------------------
_REG = [
    (354, "clinicaltrials_search", "ClinicalTrials.gov Search: TMS Depression",
     "Search ClinicalTrials.gov for completed/ongoing trials on TMS for "
     "depression with connectivity-based targeting; extract structured "
     "trial data.",
     ["Use the ClinicalTrials.gov API v2.",
      "Fields: NCT, status, phase, enrollment, intervention, primary "
      "outcome, results availability."],
     ["`trials.csv`", "`trial_landscape.md`"]),
    (355, "nih_reporter_grants", "NIH RePORTER Grant Search",
     "Search NIH RePORTER for funded grants on connectomics/graph neural "
     "networks in the last 3 fiscal years; summarize funding patterns.",
     ["Use the RePORTER API.", "Summarize by institute, mechanism, "
      "amount."],
     ["`grants.csv`", "`funding_summary.md`"]),
    (356, "osf_preregistration_search", "OSF Preregistration Search",
     "Search OSF for preregistered studies on resting-state fMRI and "
     "depression; extract hypothesis, sample size plan, and analysis plan "
     "summaries.",
     ["OSF API or scrape with documented rate limits.",
      "Summaries structured per preregistration."],
     ["`preregistrations.csv`", "`prereg_summary.md`"]),
    (357, "protocols_io_search", "Protocols.io Method Search",
     "Search protocols.io for preprocessing protocols (fMRI/EEG) relevant "
     "to a given pipeline step; extract and compare parameter choices.",
     ["Compare at least 3 protocols if available.",
      "Parameter comparison as a table."],
     ["`protocols_found.csv`", "`parameter_comparison.md`"]),
    (358, "re3data_repository_search", "Research Data Repository Search",
     "Search re3data for repositories suitable for depositing a "
     "neuroimaging derivatives dataset; produce a shortlist with access, "
     "license, and size policies.",
     ["Shortlist criteria documented.", "At least 3 repositories "
      "compared."],
     ["`repository_shortlist.md`", "`policy_comparison.csv`"]),
]
for num, slug, title, desc, cons, outs in _REG:
    TASKS.append(_t(num, slug, title, desc, ins=None, cons=cons, outs=outs))

# --- E. Alerts / monitoring, T359-T363 ------------------------------------------------------------------
_ALERT = [
    (359, "biorxiv_rss_digest", "bioRxiv RSS Weekly Digest",
     "Build a weekly digest of new bioRxiv neuroscience preprints matching "
     "a keyword set: fetch RSS/collection feeds, filter, rank by relevance, "
     "format as Markdown.",
     ["Keyword set in a config file.", "Relevance ranking rule "
      "documented."],
     ["`digest_YYYYWW.md`", "`matched_papers.json`"]),
    (360, "pubmed_alert_digest", "PubMed Alert Digest Builder",
     "Emulate a PubMed alert: run a saved query for the last 7 days, "
     "diff against the previous week's results, and emit only new hits.",
     ["State persisted between runs (JSON).", "Diff logic by PMID."],
     ["`pubmed_digest.md`", "`seen_pmids.json`"]),
    (361, "scholar_author_watch", "Author Publication Watch",
     "Watch a list of key authors (OpenAlex IDs) for new publications and "
     "produce a digest of their latest papers with abstracts.",
     ["Author list file as input.", "Deduplicate against previous "
      "digest."],
     ["`author_digest.md`", "`author_state.json`"]),
    (362, "conference_deadline_tracker", "Conference Deadline Tracker",
     "Maintain a tracker of neuroimaging conference deadlines (OHBM, "
     "ISMRM, SfN, HBM-alphabet, MICCAI): scrape/curate dates, compute days "
     "remaining, emit an ordered table.",
     ["Sources documented per conference.",
      "Table sorted by abstract deadline."],
     ["`deadlines.md`", "`deadlines.json`"]),
    (363, "preprint_watchlist_diff", "Preprint Watchlist Diff",
     "Track specific arXiv/bioRxiv category queries over time: store the "
     "corpus weekly and report what entered and left the top-K by date.",
     ["Deterministic query; state stored between runs.",
      "Report in/out lists with links."],
     ["`watchlist_diff.md`", "`corpus_state.json`"]),
]
for num, slug, title, desc, cons, outs in _ALERT:
    TASKS.append(_t(num, slug, title, desc, ins=None, cons=cons, outs=outs))

# --- F. Screening / review support, T364-T369 ---------------------------------------------------------------
_SCR = [
    (364, "prisma_fulltext_retrieval", "Full-Text Retrieval for Screening",
     "For an included-paper list, attempt full-text PDF retrieval via "
     "Unpaywall/open-access endpoints; record OA status and retrieval "
     "success per paper.",
     ["Unpaywall API with email configured.",
      "Never bypass paywalls; OA only."],
     ["`fulltext_status.csv`", "`retrieved/` PDF list"]),
    (365, "data_extraction_table", "Data Extraction Table Builder",
     "From full-texts or abstracts of included papers, build the data "
     "extraction table for a meta-analysis: sample size, age, sex ratio, "
     "scanner, preprocessing, main finding.",
     ["Fields fixed in a schema file.", "Uncertain values marked, not "
      "invented."],
     ["`extraction_table.csv`", "`extraction_notes.md`"]),
    (366, "rob2_assessment_sheet", "Risk-of-Bias (RoB 2) Assessment Sheet",
     "Prepare a RoB 2 risk-of-bias assessment sheet for the included RCTs: "
     "domains, signaling questions, and a prefilled draft per paper with "
     "justifications.",
     ["RoB 2 structure followed exactly.",
      "Every judgment cites the supporting text span."],
     ["`rob2_assessments.csv`", "`rob2_justifications.md`"]),
    (367, "effect_size_extraction", "Effect Size Extraction for Meta-Analysis",
     "Extract effect sizes (or compute from reported statistics: t, F, p, "
     "means/SDs) from included papers into an analysis-ready table.",
     ["Conversion formulas documented.",
      "Direction conventions stated (sign of effect)."],
     ["`effect_sizes.csv`", "`conversion_log.md`"]),
    (368, "forest_plot_data_prep", "Forest Plot Data Preparation",
     "Prepare forest-plot-ready data: per-study effect + CI, weights under "
     "fixed and random effects (DerSimonian-Laird), heterogeneity "
     "statistics (Q, I^2).",
     ["Computed values verified against a reference implementation if "
      "available.", "Heterogeneity interpreted in one paragraph."],
     ["`forest_data.csv`", "`heterogeneity_report.md`"]),
    (369, "ale_coordinate_extraction", "ALE Coordinate Extraction",
     "Extract activation foci coordinates from papers for an ALE-style "
     "meta-analysis: normalize to MNI (document conversions from "
     "Talairach), format for GingerALE.",
     ["Conversion tool/params documented (icbm2tal or reverse).",
      "GingerALE input format validated."],
     ["`foci_ale.txt`", "`coordinate_log.csv`"]),
]
for num, slug, title, desc, cons, outs in _SCR:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["Included-paper list from screening (required)"],
                    cons=cons, outs=outs))

# --- G. Corpus analytics, T370-T375 -----------------------------------------------------------------------------
_ANAL = [
    (370, "topic_trend_by_year", "Topic Trend by Year",
     "Quantify publication trends for a topic: per-year counts via PubMed/"
     "OpenAlex queries, growth rate, and a text-rendered or PNG trend "
     "chart.",
     ["Queries identical across years (only date filter varies).",
      "Report CAGR over the window."],
     ["`trend_data.csv`", "`trend_chart.png`", "`trend_summary.md`"]),
    (371, "abstract_cluster_map", "Abstract Embedding Cluster Map",
     "Cluster a corpus of abstracts into thematic clusters (embedding + "
     "HDBSCAN or TF-IDF + k-means), label clusters, and produce a 2D map.",
     ["Method + parameters documented.", "Cluster labels from top "
      "terms."],
     ["`clusters.csv`", "`cluster_map.png`", "`cluster_labels.md`"]),
    (372, "literature_gap_matrix", "Literature Gap Matrix",
     "Build a topic x method matrix over a corpus (e.g. disorders x "
     "analysis methods) and highlight under-studied cells as candidate "
     "gaps.",
     ["Cell thresholds for 'studied' documented.",
      "Gap claims must cite cell counts."],
     ["`gap_matrix.csv`", "`gap_report.md`"]),
    (373, "survey_taxonomy_builder", "Survey Taxonomy Builder",
     "From a review corpus, propose a taxonomy for the survey paper: 3-5 "
     "top-level categories with inclusion rules, every corpus paper "
     "assigned.",
     ["Every paper assigned to exactly one leaf.",
      "Rules written so a second rater could apply them."],
     ["`taxonomy.md`", "`paper_assignments.csv`"]),
    (374, "influential_papers_topk", "Most Influential Papers (Field-Normalized)",
     "Rank corpus papers by field-normalized influence (citations per year "
     "or OpenAlex percentile) and produce an annotated top-20 list.",
     ["Normalization method documented.",
      "One-line significance note per paper."],
     ["`top20_papers.md`", "`influence_scores.csv`"]),
    (375, "venue_analysis_topic", "Venue Analysis for a Topic",
     "Analyze where papers on a topic publish: venue ranking, OA share, "
     "and mean citations per venue, to inform submission strategy.",
     ["Venue normalization (name variants merged).",
      "Include acceptance-relevant notes if data available."],
     ["`venue_ranking.csv`", "`venue_strategy.md`"]),
]
for num, slug, title, desc, cons, outs in _ANAL:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["Corpus file (paper list or query) (required)"],
                    cons=cons, outs=outs))

# --- H. Reference management, T376-T380 --------------------------------------------------------------------------
_REFM = [
    (376, "zotero_collection_audit", "Zotero Collection Audit",
     "Audit a Zotero export (or API collection): duplicate items, missing "
     "DOIs, missing PDFs, and incomplete metadata, with a fix list.",
     ["Duplicates by DOI then fuzzy title.",
      "Fix list grouped by issue type."],
     ["`zotero_audit.csv`", "`fix_list.md`"]),
    (377, "bibtex_hygiene_fix", "BibTeX Hygiene Repair",
     "Clean a .bib file: consistent entry keys, complete fields (via "
     "Crossref lookup), no duplicate entries, valid LaTeX escaping.",
     ["Crossref lookups for DOI-less entries documented.",
      "Before/after stats reported."],
     ["`references_clean.bib`", "`hygiene_report.md`"]),
    (378, "pdf_metadata_reconcile", "PDF Metadata Reconciliation",
     "Reconcile a folder of PDFs against the .bib: match by title, flag "
     "unmatched on both sides, and rename PDFs to the citation key scheme.",
     ["Matching threshold documented; low-confidence matches flagged.",
      "Renames as a plan first, then executed with log."],
     ["`match_report.csv`", "`rename_log.txt`"]),
    (379, "citation_style_conversion", "Citation Style Conversion",
     "Convert the manuscript bibliography to a different citation style "
     "(e.g. APA -> Vancouver) using CSL, verifying every entry renders.",
     ["CSL file used is recorded.", "Spot-check 10 entries against the "
      "style guide."],
     ["Converted bibliography", "`style_check.md`"]),
    (380, "reading_list_prioritizer", "Reading List Prioritizer",
     "Prioritize a backlog reading list: score papers by relevance to the "
     "current project abstract (provided) + recency + venue, output an "
     "ordered reading plan.",
     ["Scoring formula documented.",
      "Plan grouped into must-read / skim / archive."],
     ["`reading_plan.md`", "`scores.csv`"]),
]
for num, slug, title, desc, cons, outs in _REFM:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["Zotero export / .bib / PDF folder as applicable "
                         "(required)"],
                    cons=cons, outs=outs))

# --- I. Code / data availability audits, T381-T385 -----------------------------------------------------------------
_AVAIL = [
    (381, "papers_with_code_audit", "Code Availability Audit",
     "For a paper list, audit code availability: official repo found, "
     "license present, last commit recency, and reproducibility files "
     "(env, seeds).",
     ["Repo evidence via URL; no guessing.",
      "Availability classes: full / partial / none."],
     ["`code_availability.csv`", "`audit_summary.md`"]),
    (382, "github_repo_health_check", "GitHub Repo Health Check",
     "Health-check the GitHub repos linked from included papers: stars, "
     "maintenance status, issue response, CI presence, documentation "
     "quality.",
     ["Metrics via GitHub API.", "Health score formula documented."],
     ["`repo_health.csv`", "`health_notes.md`"]),
    (383, "dataset_citation_audit", "Dataset Citation Audit",
     "Audit whether papers using public datasets (HCP/ABIDE/ADNI/UKB) cite "
     "them correctly: required citations + acknowledgments present.",
     ["Required-citation lists per dataset compiled first.",
      "Verdict per paper: compliant / partial / missing."],
     ["`dataset_citation_audit.csv`", "`compliance_summary.md`"]),
    (384, "rrid_lookup_tools", "RRID Lookup for Methods Section",
     "Compile RRIDs (SciCrunch) for every software tool used in the "
     "project's methods section, producing a methods-ready resource table.",
     ["Tool list provided as input.",
      "Table: tool, version used, RRID, citation."],
     ["`resource_table.md`", "`rrid_lookup_log.json`"]),
    (385, "reproducibility_checklist", "Reproducibility Checklist Audit",
     "Audit included papers against a reproducibility checklist (data "
     "availability, code, seeds, environment, hyperparameters), producing "
     "per-paper scores.",
     ["Checklist items fixed in a YAML schema.",
      "Score = items met / total; evidence quoted per item."],
     ["`repro_scores.csv`", "`evidence_notes.md`"]),
]
for num, slug, title, desc, cons, outs in _AVAIL:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["Paper list or corpus (required)"],
                    cons=cons, outs=outs))

# --- J. Writing support, T386-T390 ---------------------------------------------------------------------------------------
_WRIT = [
    (386, "related_work_outline", "Related Work Outline from Corpus",
     "Draft a related-work outline from the corpus: thematic sections, "
     "papers assigned per section, and a one-sentence contrast to our "
     "approach per section.",
     ["Sections traceable to the taxonomy (or built ad hoc).",
      "No fabricated findings; contrasts cite paper claims."],
     ["`related_work_outline.md`", "`section_assignments.csv`"]),
    (387, "survey_comparison_table", "Survey Comparison Table Auto-Build",
     "Auto-build the survey comparison table: papers x fields (year, data, "
     "method, sample size, metric, result), with missing fields flagged.",
     ["Field schema provided or derived from 5 exemplars.",
      "Flag, never invent, missing values."],
     ["`comparison_table.csv`", "`table_notes.md`"]),
    (388, "plain_language_abstract", "Plain-Language Abstract Rewrite",
     "Rewrite the project abstract into a plain-language summary for a "
     "general audience (e.g. for ethics/press), keeping every claim "
     "faithful to the original.",
     ["~150 words; no jargon without explanation.",
      "Faithfulness: each sentence traceable to the original."],
     ["`plain_abstract.md`", "`claim_mapping.md`"]),
    (389, "graphical_abstract_brief", "Graphical Abstract Brief",
     "Write the design brief for a graphical abstract: key message, "
     "3-panel storyline, visual elements per panel, and text labels, ready "
     "to hand to a designer.",
     ["Follows the target journal's GA guidelines if provided.",
      "Labels short enough to render (<= 8 words each)."],
     ["`ga_brief.md`"]),
    (390, "reviewer_suggestion_list", "Reviewer Suggestion List",
     "Suggest qualified reviewers for the manuscript: candidate pool from "
     "cited authors + field experts, screened for conflicts (co-authorship, "
     "same institution).",
     ["Conflict screen documented per candidate.",
      "8-12 candidates with rationale and contact field placeholders."],
     ["`reviewer_suggestions.md`", "`conflict_screen.csv`"]),
]
for num, slug, title, desc, cons, outs in _WRIT:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["Project abstract/manuscript draft (required)",
                         "Corpus files as applicable"],
                    cons=cons, outs=outs))

# --- K. Meta-research, T391-T395 -----------------------------------------------------------------------------------------
_META = [
    (391, "journal_metrics_lookup", "Journal Metrics Lookup",
     "Compile journal metrics for a target-venue shortlist: JIF/CiteScore "
     "alternatives (SJR, SNIP), turnaround data if available, OA options "
     "and APCs.",
     ["Sources + access dates documented.",
      "Predatory-journal check included."],
     ["`venue_metrics.csv`", "`venue_recommendation.md`"]),
    (392, "oa_status_report", "Open-Access Status Report",
     "Report the OA status of every reference in the bibliography "
     "(Unpaywall): OA color, repository version available, and self-"
     "archiving options for closed ones.",
     ["Unpaywall data with timestamps.",
      "Summary: % OA by color."],
     ["`oa_status.csv`", "`oa_summary.md`"]),
    (393, "retraction_check", "Retraction / EoC Check of References",
     "Check every reference against retraction databases (Crossref "
     "retractions, Retraction Watch data): flag retracted or "
     "expression-of-concern items.",
     ["Database + date documented.",
      "Flagged items include recommended action."],
     ["`retraction_flags.csv`", "`actions.md`"]),
    (394, "preprint_published_matching", "Preprint-to-Published Matching",
     "Match preprints in the corpus to their published versions (Crossref/"
     "OpenAlex relation), updating citation records to the version of "
     "record.",
     ["Match evidence documented (relation field or title+authors).",
      "Unmatched preprints listed."],
     ["`version_matches.csv`", "`updated_references.bib`"]),
    (395, "funding_ack_mining", "Funding Acknowledgment Mining",
     "Mine funding acknowledgments from the corpus: funding agencies, "
     "grant numbers, and co-funding patterns relevant to our grant "
     "application.",
     ["Agency normalization (name variants merged).",
      "Grant-number validation per agency format."],
     ["`funding_mentions.csv`", "`funding_landscape.md`"]),
]
for num, slug, title, desc, cons, outs in _META:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["Bibliography/corpus file (required)"],
                    cons=cons, outs=outs))

assert len(TASKS) == 65, f"research batch must be 65 tasks, got {len(TASKS)}"
