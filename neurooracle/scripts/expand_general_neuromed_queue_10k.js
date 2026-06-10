const fs = require('fs');
const path = require('path');

const repo = path.resolve(__dirname, '..', '..');
const stage = path.join(repo, 'neurooracle', 'data', 'phase2_staging', 'general_neuromed_manual_20260610');
const queuePath = path.join(stage, 'abstracts_queue.jsonl');
const manifestPath = path.join(stage, 'manifest_10k_expansion.json');
const logDir = path.join(stage, 'logs');

const TARGET = Number(process.env.GENMED_TARGET || 10000);
const COVER_ALL_DISEASES = String(process.env.GENMED_COVER_ALL || '').trim().toLowerCase() === '1'
  || String(process.env.GENMED_COVER_ALL || '').trim().toLowerCase() === 'true';
const YEAR_START = 2000;
const YEAR_END = 2026;
const PER_QUERY = Number(process.env.GENMED_PER_QUERY || 90);
const EMAIL = process.env.NCBI_EMAIL || 'neuroclaw@example.com';
const API_KEY = process.env.NCBI_API_KEY || '1e72705978ad50249ffc129798ba3958f308';

const DISEASES = [
  "Alzheimer's disease",
  "Parkinson's disease",
  "multiple sclerosis",
  "epilepsy",
  "stroke",
  "amyotrophic lateral sclerosis",
  "Huntington disease",
  "frontotemporal dementia",
  "Lewy body dementia",
  "vascular dementia",
  "migraine",
  "traumatic brain injury",
  "brain tumor",
  "glioma",
  "cerebral small vessel disease",
  "neuromyelitis optica",
  "autism spectrum disorder",
  "ADHD",
  "schizophrenia",
  "major depressive disorder",
  "bipolar disorder",
  "anxiety disorder",
  "obsessive-compulsive disorder",
  "post-traumatic stress disorder",
  "Tourette syndrome",
  "essential tremor",
  "dystonia",
  "cerebellar ataxia",
  "spinal cord injury",
  "encephalitis",
  "meningitis",
  "sleep disorder",
  "mild cognitive impairment",
  "normal aging",
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
  '"mean diffusivity"', '"cerebral blood flow"', '"perfusion"',
];

const WEAK_TOPIC_TITLE_TERMS = [
  'appropriate use criteria', 'clinical practice guideline', 'practice guideline',
  'consensus statement', 'recommendations', 'expert review', 'overview',
  'personalized management', 'treatment alliance', 'treatment adherence',
  'drug delivery',
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

function matchAll(text, re) {
  return Array.from(text.matchAll(re), (m) => m);
}

function firstDecoded(block, re) {
  const m = block.match(re);
  return m ? decodeXml(m[1]) : '';
}

function parseArticles(xml) {
  const blocks = matchAll(xml, /<PubmedArticle\b[\s\S]*?<\/PubmedArticle>/g).map((m) => m[0]);
  const out = [];
  for (const block of blocks) {
    const pmid = firstDecoded(block, /<PMID[^>]*>([\s\S]*?)<\/PMID>/);
    const title = firstDecoded(block, /<ArticleTitle[^>]*>([\s\S]*?)<\/ArticleTitle>/);
    const yearRaw = firstDecoded(block, /<PubDate>[\s\S]*?<Year>(\d{4})<\/Year>[\s\S]*?<\/PubDate>/);
    const year = yearRaw ? Number(yearRaw) : null;
    const journal = firstDecoded(block, /<Journal>[\s\S]*?<Title>([\s\S]*?)<\/Title>[\s\S]*?<\/Journal>/);
    const doi = firstDecoded(block, /<ArticleId[^>]+IdType="doi"[^>]*>([\s\S]*?)<\/ArticleId>/);
    const abstractParts = matchAll(block, /<AbstractText\b[^>]*>([\s\S]*?)<\/AbstractText>/g)
      .map((m) => decodeXml(m[1]))
      .filter(Boolean);
    const abstract = abstractParts.join(' ');
    const authors = matchAll(block, /<Author\b[\s\S]*?<\/Author>/g).slice(0, 5).map((m) => {
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

async function main() {
  fs.mkdirSync(stage, { recursive: true });
  fs.mkdirSync(logDir, { recursive: true });
  const queue = readJsonl(queuePath);
  const claims = readJsonl(path.join(stage, 'manual_claims.jsonl'));
  const seenPmids = new Set(queue.map((r) => String(r.pmid)).filter(Boolean));
  const injectedPmids = new Set(claims.map((r) => String(r.source_paper?.pmid || r.pmid || '')).filter(Boolean));
  let total = queue.length;
  let nextIndex = total + 1;
  const sweepLog = path.join(logDir, `search_10k_2000_2026_${new Date().toISOString().replace(/[:.]/g, '-')}.jsonl`);

  const manifest = {
    created_or_updated: new Date().toISOString(),
    target: TARGET,
    cover_all_diseases: COVER_ALL_DISEASES,
    year_start: YEAR_START,
    year_end: YEAR_END,
    papers_per_disease_year: PER_QUERY,
    diseases: DISEASES,
    mode: 'manual_abstract_queue_no_llm_extraction',
    search_profile: 'strict_neuroscience_major_neuro_diseases_10k',
    search_requires: 'disease plus neuroscience marker terms in title/abstract; weak clinical-management title terms excluded',
    already_injected_manual_claims: injectedPmids.size,
  };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n', 'utf8');

  console.log(`starting queue expansion: ${total}/${TARGET} queued, ${seenPmids.size} seen PMIDs, ${injectedPmids.size} already curated/injected`);
  for (const disease of DISEASES) {
    for (let year = YEAR_END; year >= YEAR_START; year--) {
      if (!COVER_ALL_DISEASES && total >= TARGET) {
        console.log(`target reached: ${total}/${TARGET}`);
        return;
      }
      const query = buildQuery(disease, year);
      const { pmids, totalHits } = await searchPmids(query, PER_QUERY);
      await sleep(140);
      const freshPmids = pmids.filter((p) => !seenPmids.has(String(p)));
      const articles = await fetchArticles(freshPmids);
      const batch = [];
      for (const ref of articles) {
        if (!ref.pmid || seenPmids.has(String(ref.pmid))) continue;
        seenPmids.add(String(ref.pmid));
        if (!isRelevant(ref)) continue;
        batch.push({
          batch_index: nextIndex++,
          queue_id: `GENMED10K:${ref.pmid}`,
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
          curation_round: 'general_neuromed_10k_expansion',
          prior_manual_injected: injectedPmids.has(String(ref.pmid)),
          created_at: new Date().toISOString(),
        });
      }
      appendJsonl(queuePath, batch);
      total += batch.length;
      appendJsonl(sweepLog, [{
        disease,
        year,
        total_hits: totalHits,
        returned_pmids: pmids.length,
        fresh_pmids: freshPmids.length,
        queued: batch.length,
        queue_total: total,
        timestamp: new Date().toISOString(),
      }]);
      console.log(`${disease} ${year}: +${batch.length} queued (${total}/${TARGET}); hits=${totalHits}`);
      await sleep(220);
    }
  }
  console.log(`finished sweep: ${total}/${TARGET}${COVER_ALL_DISEASES ? ' (cover-all mode)' : ''}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
