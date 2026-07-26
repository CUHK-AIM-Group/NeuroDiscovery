const fs = require('fs');
const path = require('path');

const repo = path.resolve(__dirname, '..', '..');
const stage = process.env.GENMED_STAGE
  ? path.resolve(process.env.GENMED_STAGE)
  : path.join(repo, 'neurooracle', 'data', 'phase2_staging', 'general_neuromed_manual_20260610');
const queuePath = path.join(stage, 'abstracts_queue.jsonl');
const logDir = path.join(stage, 'logs');

const TARGET_NEW = Number(process.env.GENMED_SUPPLEMENT_TARGET || 5000);
const YEAR_START = Number(process.env.GENMED_YEAR_START || 1980);
const YEAR_END = Number(process.env.GENMED_YEAR_END || 2026);
const PER_QUERY = Number(process.env.GENMED_PER_QUERY || 500);
const EMAIL = process.env.NCBI_EMAIL || 'neuroclaw@example.com';
const API_KEY = process.env.NCBI_API_KEY || '1e72705978ad50249ffc129798ba3958f308';
const RUN_LABEL = process.env.GENMED_RUN_LABEL || `supplement_${YEAR_START}_${YEAR_END}`;
const CURATION_ROUND = process.env.GENMED_CURATION_ROUND || 'general_neuromed_1980_2026_balanced_supplement';

const DISEASES = [
  "Alzheimer's disease",
  "Parkinson's disease",
  'multiple sclerosis',
  'epilepsy',
  'stroke',
  'amyotrophic lateral sclerosis',
  'Huntington disease',
  'frontotemporal dementia',
  'Lewy body dementia',
  'vascular dementia',
  'migraine',
  'traumatic brain injury',
  'brain tumor',
  'glioma',
  'cerebral small vessel disease',
  'neuromyelitis optica',
  'autism spectrum disorder',
  'ADHD',
  'schizophrenia',
  'major depressive disorder',
  'bipolar disorder',
  'anxiety disorder',
  'obsessive-compulsive disorder',
  'post-traumatic stress disorder',
  'Tourette syndrome',
  'essential tremor',
  'dystonia',
  'cerebellar ataxia',
  'spinal cord injury',
  'encephalitis',
  'meningitis',
  'sleep disorder',
  'mild cognitive impairment',
  'normal aging',
];

const NEUROSCIENCE_QUERY_TERMS = [
  '"neuroimaging"', '"brain imaging"', '"magnetic resonance imaging"', '"MRI"',
  '"functional MRI"', '"fMRI"', '"diffusion tensor imaging"', '"DTI"',
  '"positron emission tomography"', '"PET"', '"FDG"', '"amyloid PET"', '"tau PET"',
  '"electroencephalography"', '"EEG"', '"magnetoencephalography"', '"MEG"',
  '"functional connectivity"', '"structural connectivity"', '"connectome"',
  '"brain network"', '"default mode network"', '"cortical thickness"',
  '"cortical volume"', '"gray matter"', '"grey matter"', '"white matter"',
  '"brain atrophy"', '"hippocampus"', '"amygdala"', '"thalamus"', '"striatum"',
  '"prefrontal cortex"', '"cingulate"', '"brain region"', '"neural circuit"',
  '"synaptic"', '"neurotransmitter"', '"dopamine"', '"serotonin"', '"glutamate"',
  '"GABA"', '"neuroinflammation"', '"microglia"', '"cerebrospinal fluid"',
  '"CSF"', '"neural marker"', '"brain biomarker"', '"cortical activation"',
  '"regional homogeneity"', '"ALFF"', '"ReHo"', '"fractional anisotropy"',
  '"mean diffusivity"', '"cerebral blood flow"', '"perfusion"', '"neuropathology"',
  '"postmortem"', '"cerebral cortex"', '"neuron"', '"neuronal"', '"neurodegeneration"',
  '"neuropsychological"', '"cognition"', '"cognitive"', '"receptor"', '"gene expression"',
];

const WEAK_TOPIC_TITLE_TERMS = [
  'appropriate use criteria',
  'clinical practice guideline',
  'practice guideline',
  'consensus statement',
  'recommendations',
  'expert review',
  'overview',
  'personalized management',
  'treatment alliance',
  'treatment adherence',
  'drug delivery',
  'case report',
  'editorial',
  'letter',
];

const CLINICAL_TRIAL_TITLE_TERMS = ['randomized clinical trial', 'clinical trial'];
const NEURO_MARKER_TITLE_TERMS = [
  'pet', 'mri', 'fmri', 'eeg', 'meg', 'dti', 'csf', 'biomarker',
  'neuroimaging', 'brain', 'cortical', 'hippocamp', 'synaptic', 'microglia',
];

const STRONG_NEUROSCIENCE_TEXT_TERMS = [
  'neuroimaging', 'brain imaging', 'magnetic resonance imaging', ' mri', 'fmri',
  'diffusion tensor', ' dti', 'positron emission tomography', ' pet', 'fdg',
  'amyloid pet', 'tau pet', 'electroencephalography', ' eeg',
  'magnetoencephalography', ' meg', 'functional connectivity',
  'structural connectivity', 'connectome', 'brain network', 'default mode network',
  'cortical thickness', 'cortical volume', 'gray matter', 'grey matter',
  'white matter', 'brain atrophy', 'hippocamp', 'amygdala', 'thalam', 'striat',
  'prefrontal cortex', 'cingulate', 'brain region', 'neural circuit', 'synaptic',
  'neurotransmitter', 'dopamine', 'serotonin', 'glutamate', 'gaba',
  'neuroinflammation', 'microglia', 'cerebrospinal fluid', ' csf',
  'neural marker', 'brain biomarker', 'regional homogeneity', 'alff', 'reho',
  'fractional anisotropy', 'mean diffusivity', 'cerebral blood flow', 'perfusion',
  'neuropathology', 'postmortem', 'cerebral cortex', 'neuron', 'neuronal',
  'neurodegeneration', 'neuropsychological', 'cognition', 'cognitive',
  'receptor', 'gene expression',
];

function readJsonl(file) {
  if (!fs.existsSync(file)) return [];
  return fs.readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function appendJsonl(file, rows) {
  if (!rows.length) return;
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.appendFileSync(file, rows.map((r) => JSON.stringify(r)).join('\n') + '\n', 'utf8');
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function pmidOf(row) {
  return String(row.pmid || row.pubmed_id || row.source_pmid || row.paper_id || row.source_paper?.pmid || row.metadata?.source_paper?.pmid || '').replace(/^PMID:/, '');
}

function decodeXml(s) {
  return String(s || '')
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

function firstDecoded(block, re) {
  const m = block.match(re);
  return m ? decodeXml(m[1]) : '';
}

function parseArticles(xml) {
  const blocks = Array.from(xml.matchAll(/<PubmedArticle\b[\s\S]*?<\/PubmedArticle>/g), (m) => m[0]);
  const out = [];
  for (const block of blocks) {
    const pmid = firstDecoded(block, /<PMID[^>]*>([\s\S]*?)<\/PMID>/);
    const title = firstDecoded(block, /<ArticleTitle[^>]*>([\s\S]*?)<\/ArticleTitle>/);
    const yearRaw = firstDecoded(block, /<PubDate>[\s\S]*?<Year>(\d{4})<\/Year>[\s\S]*?<\/PubDate>/);
    const year = yearRaw ? Number(yearRaw) : null;
    const journal = firstDecoded(block, /<Journal>[\s\S]*?<Title>([\s\S]*?)<\/Title>[\s\S]*?<\/Journal>/);
    const doi = firstDecoded(block, /<ArticleId[^>]+IdType="doi"[^>]*>([\s\S]*?)<\/ArticleId>/);
    const abstractParts = Array.from(block.matchAll(/<AbstractText\b[^>]*>([\s\S]*?)<\/AbstractText>/g))
      .map((m) => decodeXml(m[1]))
      .filter(Boolean);
    const abstract = abstractParts.join(' ');
    const authors = Array.from(block.matchAll(/<Author\b[\s\S]*?<\/Author>/g)).slice(0, 5).map((m) => {
      const a = m[0];
      const last = firstDecoded(a, /<LastName>([\s\S]*?)<\/LastName>/);
      const fore = firstDecoded(a, /<ForeName>([\s\S]*?)<\/ForeName>/);
      return [last, fore].filter(Boolean).join(' ');
    }).filter(Boolean).join(', ');
    if (pmid && title && abstract) {
      out.push({ pmid, doi, title, authors, year, journal, abstract });
    }
  }
  return out;
}

function pubmedOrTerms(terms) {
  return terms.map((t) => `${t}[Title/Abstract]`).join(' OR ');
}

function buildQuery(disease, year) {
  const neuroscience = pubmedOrTerms(NEUROSCIENCE_QUERY_TERMS);
  const weakTitle = WEAK_TOPIC_TITLE_TERMS.map((t) => `"${t}"[Title]`).join(' OR ');
  return `(${disease}[Title/Abstract]) AND (${neuroscience}) AND ${year}:${year}[pdat] NOT (${weakTitle})`;
}

function isRelevant(ref) {
  const title = (ref.title || '').toLowerCase();
  const text = `${title} ${ref.abstract || ''}`.toLowerCase();
  if (WEAK_TOPIC_TITLE_TERMS.some((term) => title.includes(term))) return false;
  if (CLINICAL_TRIAL_TITLE_TERMS.some((term) => title.includes(term)) &&
      !NEURO_MARKER_TITLE_TERMS.some((term) => title.includes(term))) {
    return false;
  }
  return STRONG_NEUROSCIENCE_TEXT_TERMS.some((term) => text.includes(term));
}

async function fetchJson(url, params, retries = 5) {
  const u = new URL(url);
  for (const [k, v] of Object.entries(params)) u.searchParams.set(k, String(v));
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const resp = await fetch(u);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return await resp.json();
    } catch (err) {
      const wait = 1000 * Math.pow(2, attempt);
      console.log(`PubMed JSON fetch failed (${err.message}); retrying in ${wait / 1000}s`);
      await sleep(wait);
    }
  }
  return null;
}

async function fetchTextPost(url, body, retries = 5) {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams(body).toString(),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return await resp.text();
    } catch (err) {
      const wait = 1000 * Math.pow(2, attempt);
      console.log(`PubMed XML fetch failed (${err.message}); retrying in ${wait / 1000}s`);
      await sleep(wait);
    }
  }
  return '';
}

async function searchPmids(query, retmax) {
  const data = await fetchJson('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi', {
    db: 'pubmed',
    term: query,
    retmax,
    sort: 'relevance',
    retmode: 'json',
    email: EMAIL,
    api_key: API_KEY,
  });
  const result = data?.esearchresult || {};
  return { pmids: result.idlist || [], totalHits: Number(result.count || 0) };
}

async function fetchArticles(pmids) {
  const out = [];
  for (let i = 0; i < pmids.length; i += 100) {
    const batch = pmids.slice(i, i + 100);
    const xml = await fetchTextPost('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', {
      db: 'pubmed',
      id: batch.join(','),
      rettype: 'xml',
      retmode: 'xml',
      email: EMAIL,
      api_key: API_KEY,
    });
    out.push(...parseArticles(xml));
    await sleep(120);
  }
  return out;
}

function planDiseaseYears(queue) {
  const counts = new Map();
  for (let year = YEAR_START; year <= YEAR_END; year++) counts.set(year, 0);
  for (const row of queue) {
    const year = Number(row.year || row.query_year || 0);
    if (year >= YEAR_START && year <= YEAR_END) counts.set(year, (counts.get(year) || 0) + 1);
  }
  const yearsByThinness = Array.from(counts).sort((a, b) => a[1] - b[1] || a[0] - b[0]).map(([year]) => year);
  const plan = [];
  for (const year of yearsByThinness) {
    for (const disease of DISEASES) plan.push({ disease, year });
  }
  return { plan, counts };
}

async function main() {
  fs.mkdirSync(stage, { recursive: true });
  fs.mkdirSync(logDir, { recursive: true });

  const queue = readJsonl(queuePath);
  const claims = readJsonl(path.join(stage, 'manual_claims.jsonl'));
  const noClaims = readJsonl(path.join(stage, 'manual_no_claims.jsonl'));
  const seenPmids = new Set([...queue, ...claims, ...noClaims].map(pmidOf).filter(Boolean));
  let nextIndex = queue.length + 1;
  let added = 0;

  const { plan, counts } = planDiseaseYears(queue);
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const logPath = path.join(logDir, `general_supplement_${RUN_LABEL}_${stamp}.jsonl`);
  const summaryPath = path.join(stage, `general_supplement_${RUN_LABEL}_summary.json`);

  console.log(`starting balanced supplement: queue=${queue.length}, unique_seen=${seenPmids.size}, target_new=${TARGET_NEW}`);
  for (const { disease, year } of plan) {
    if (added >= TARGET_NEW) break;
    const query = buildQuery(disease, year);
    const { pmids, totalHits } = await searchPmids(query, PER_QUERY);
    await sleep(140);
    const freshPmids = pmids.filter((p) => !seenPmids.has(String(p)));
    const articles = await fetchArticles(freshPmids);
    const rows = [];
    for (const ref of articles) {
      if (!ref.pmid || seenPmids.has(String(ref.pmid))) continue;
      seenPmids.add(String(ref.pmid));
      if (!isRelevant(ref)) continue;
      rows.push({
        batch_index: nextIndex++,
        queue_id: `GENMEDSUPP:${ref.pmid}`,
        pmid: String(ref.pmid),
        doi: ref.doi || '',
        title: ref.title || '',
        authors: ref.authors || '',
        year: ref.year || year,
        journal: ref.journal || '',
        disease_query: disease,
        query_year: year,
        query_total_hits: totalHits,
        abstract: ref.abstract || '',
        status: 'pending_manual_review',
        curation_round: CURATION_ROUND,
        prior_manual_processed: claims.some((r) => pmidOf(r) === String(ref.pmid)) || noClaims.some((r) => pmidOf(r) === String(ref.pmid)),
        created_at: new Date().toISOString(),
      });
      if (added + rows.length >= TARGET_NEW) break;
    }
    appendJsonl(queuePath, rows);
    added += rows.length;
    appendJsonl(logPath, [{
      disease,
      year,
      total_hits: totalHits,
      returned_pmids: pmids.length,
      fresh_pmids: freshPmids.length,
      queued: rows.length,
      added_total: added,
      timestamp: new Date().toISOString(),
    }]);
    if (rows.length || freshPmids.length) {
      console.log(`${year} ${disease}: +${rows.length} queued; fresh=${freshPmids.length}; total_added=${added}/${TARGET_NEW}`);
    }
    await sleep(220);
  }

  const finalQueue = readJsonl(queuePath);
  const byYearNew = {};
  for (const row of finalQueue.slice(queue.length)) {
    const year = String(row.year || row.query_year || 'unknown');
    byYearNew[year] = (byYearNew[year] || 0) + 1;
  }
  const summary = {
    created_at: new Date().toISOString(),
    objective: 'balanced PubMed supplement for general manual queue, 1980-present',
    stage,
    queue_path: queuePath,
    log_path: logPath,
    target_new: TARGET_NEW,
    added,
    queue_before: queue.length,
    queue_after: finalQueue.length,
    year_start: YEAR_START,
    year_end: YEAR_END,
    per_query: PER_QUERY,
    curation_round: CURATION_ROUND,
    previous_year_counts: Object.fromEntries(counts),
    new_records_by_year: Object.fromEntries(Object.entries(byYearNew).sort(([a], [b]) => Number(a) - Number(b))),
    unique_pmids_after: new Set(finalQueue.map(pmidOf).filter(Boolean)).size,
  };
  fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
