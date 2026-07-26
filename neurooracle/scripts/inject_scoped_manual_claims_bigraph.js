const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { parser } = require('stream-json');
const Assembler = require('stream-json/Assembler');

// Streaming graph loader: the JS injector's original JSON.parse(readFileSync) cannot
// load graphs > ~512MB (Node string limit). This assembles the same object via a token
// stream so the rest of the injection logic is byte-for-byte identical.
function loadGraphStreaming(file) {
  return new Promise((resolve, reject) => {
    const pipeline = fs.createReadStream(file).pipe(parser());
    const asm = Assembler.connectTo(pipeline);
    asm.on('done', (a) => resolve(a.current));
    pipeline.on('error', reject);
  });
}

// Streaming id-set reader for extracted_claims.jsonl (> 512MB, exceeds Node string limit).
// Scans raw bytes for newlines and JSON.parses each complete line, so no whole-file string
// is materialized and no multi-byte UTF-8 char is split across chunk boundaries.
function readJsonlIdSetStreaming(file) {
  const ids = new Set();
  if (!fs.existsSync(file)) return ids;
  const fd = fs.openSync(file, 'r');
  try {
    const CHUNK = 64 * 1024 * 1024;
    const buf = Buffer.allocUnsafe(CHUNK);
    let tail = Buffer.alloc(0);
    let bytes;
    const take = (line) => {
      if (!line) return;
      try { const o = JSON.parse(line); if (o && o.id) ids.add(o.id); } catch (e) { /* skip partial/garbage */ }
    };
    while ((bytes = fs.readSync(fd, buf, 0, CHUNK, null)) > 0) {
      const combined = Buffer.concat([tail, buf.subarray(0, bytes)]);
      let from = 0;
      for (let i = 0; i < combined.length; i += 1) {
        if (combined[i] === 0x0a) {
          take(combined.toString('utf8', from, i));
          from = i + 1;
        }
      }
      tail = combined.subarray(from);
    }
    if (tail.length) take(tail.toString('utf8'));
  } finally {
    fs.closeSync(fd);
  }
  return ids;
}

const repo = path.resolve(__dirname, '..', '..');
const defaultGraph = path.join(repo, 'neurooracle', 'data', 'full_v2', 'knowledge_graph.json');
const defaultExtracted = path.join(repo, 'neurooracle', 'data', 'full_v2', 'extracted_claims.jsonl');

const relationTypes = new Set([
  'is_a', 'part_of', 'has_part', 'causes', 'associated_with', 'predisposes',
  'treats', 'contraindicated_for', 'gene_associated_with_disease',
  'gene_associated_with_anatomy', 'gene_enriched_in_region', 'receptor_density_in',
  'protein_encoded_by', 'modulates', 'binds_to', 'projects_to', 'connects_to',
  'activates', 'coactivates', 'supported_by', 'contradicts', 'about', 'reduces',
  'increases', 'correlates_with', 'is_biomarker_of', 'is_risk_factor_for',
  'is_associated_with', 'predicts', 'mediates', 'inhibits', 'distinguishes',
  'supports_modality', 'provides_modality', 'evokes', 'decoded_from', 'elicits',
  'measures', 'assessed_in', 'affects_system', 'provides_signal_for',
  'is_indicated_for', 'is_treated_by', 'measured_in_modality', 'modality_provides',
  'is_assessed_by', 'has_adverse_effect', 'defines_region', 'measured_by_modality',
  'is_imaging_feature_of', 'has_imaging_feature',
]);

const predicateMap = new Map(Object.entries({
  does_not_associate_with: 'is_associated_with',
  not_associated_with: 'is_associated_with',
  does_not_predict: 'predicts',
  does_not_outperform: 'predicts',
  identifies: 'is_biomarker_of',
  indicates: 'is_biomarker_of',
  detects: 'is_biomarker_of',
  visualizes: 'is_biomarker_of',
  reveals: 'is_biomarker_of',
  maps: 'is_biomarker_of',
  characterizes: 'is_biomarker_of',
  captures: 'is_biomarker_of',
  measures: 'is_biomarker_of',
  contributes_to: 'causes',
  induces: 'causes',
  drives: 'causes',
  triggers: 'causes',
  disrupts: 'causes',
  accelerates: 'causes',
  supports: 'is_associated_with',
  recapitulates: 'is_associated_with',
  models: 'is_associated_with',
  harmonizes: 'is_associated_with',
  standardizes: 'is_associated_with',
  routes: 'is_associated_with',
  tracks: 'is_associated_with',
  confounds: 'is_associated_with',
  is_implicated_in: 'is_associated_with',
  are_associated_with: 'is_associated_with',
  are_spatially_associated_with: 'is_associated_with',
  is_contested_as: 'is_associated_with',
  is_concordant_with: 'is_associated_with',
  is_equivalent_to: 'is_associated_with',
  has_amyloid_dependent_and_independent_effects_on: 'modulates',
  modifies: 'modulates',
  promotes: 'modulates',
  restores: 'modulates',
  enables: 'predicts',
  improves: 'treats',
  protects_against: 'reduces',
  suppresses: 'inhibits',
  rules_in_or_out: 'distinguishes',
  is_increased_in: 'is_biomarker_of',
  changes_before: 'predicts',
}));

function parseArgs(argv) {
  const args = {
    claims: '',
    graph: defaultGraph,
    extracted: defaultExtracted,
    metadataCsv: [],
    metadataDir: [],
    defaultScope: [],
    injectionSource: 'scoped_manual_claims',
    dryRun: false,
    repairExisting: false,
    flattenOutput: '',
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--claims') args.claims = argv[++i];
    else if (arg === '--graph') args.graph = argv[++i];
    else if (arg === '--extracted') args.extracted = argv[++i];
    else if (arg === '--metadata-csv') args.metadataCsv.push(argv[++i]);
    else if (arg === '--metadata-dir') args.metadataDir.push(argv[++i]);
    else if (arg === '--default-scope') args.defaultScope.push(argv[++i]);
    else if (arg === '--injection-source') args.injectionSource = argv[++i];
    else if (arg === '--flatten-output') args.flattenOutput = argv[++i];
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '--repair-existing') args.repairExisting = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.claims) throw new Error('--claims is required');
  if (!args.defaultScope.length) args.defaultScope = ['general'];
  return args;
}

function readJsonl(file) {
  return fs.readFileSync(file, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`Invalid JSONL at ${file}:${index + 1}: ${error.message}`);
      }
    });
}

function writeGraphCompact(file, graph) {
  const fd = fs.openSync(file, 'w');
  try {
    fs.writeSync(fd, '{"metadata":');
    fs.writeSync(fd, JSON.stringify(graph.metadata || {}));
    fs.writeSync(fd, ',"concepts":{');
    let firstConcept = true;
    for (const [id, node] of Object.entries(graph.concepts || {})) {
      if (!firstConcept) fs.writeSync(fd, ',');
      firstConcept = false;
      fs.writeSync(fd, JSON.stringify(id));
      fs.writeSync(fd, ':');
      fs.writeSync(fd, JSON.stringify(node));
    }
    fs.writeSync(fd, '},"edges":[');
    let firstEdge = true;
    for (const edge of graph.edges || []) {
      if (!firstEdge) fs.writeSync(fd, ',');
      firstEdge = false;
      fs.writeSync(fd, JSON.stringify(edge));
    }
    fs.writeSync(fd, ']}\n');
  } finally {
    fs.closeSync(fd);
  }
}

function parseCsvLine(line) {
  const values = [];
  let current = '';
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (ch === ',' && !quoted) {
      values.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  values.push(current);
  return values;
}

function loadMetadataCsv(file, metadataByPaper) {
  const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean);
  if (!lines.length) return 0;
  const headers = parseCsvLine(lines[0]);
  let count = 0;
  for (const line of lines.slice(1)) {
    const values = parseCsvLine(line);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] || '';
    });
    const paperId = String(row.id || row.pmid || row.doi || row.key || '').trim();
    if (!paperId) continue;
    metadataByPaper.set(paperId, { ...(metadataByPaper.get(paperId) || {}), ...row });
    count += 1;
  }
  return count;
}

function loadMetadataDir(dir, metadataByPaper) {
  let count = 0;
  for (const name of fs.readdirSync(dir)) {
    if (!name.endsWith('.jsonl')) continue;
    if (!name.includes('enriched') && !name.includes('abstract_cache')) continue;
    for (const row of readJsonl(path.join(dir, name))) {
      const paperId = String(row.id || row.pmid || row.doi || row.key || '').trim();
      if (!paperId) continue;
      metadataByPaper.set(paperId, { ...(metadataByPaper.get(paperId) || {}), ...row });
      count += 1;
    }
  }
  return count;
}

function normalizePaperScope(scope) {
  const raw = Array.isArray(scope) ? scope : [scope];
  const out = [];
  for (const item of raw) {
    let value = String(item || '').trim().toLowerCase();
    if (!value) continue;
    if (['cs1', 'case_1', 'case-1', 'case study 1', 'case1_transdiagnostic'].includes(value)) value = 'case1';
    else if (['cs2', 'case_2', 'case-2', 'case study 2'].includes(value)) value = 'case2';
    else if (['cs3', 'case_3', 'case-3', 'case study 3', 'hindcasting'].includes(value)) value = 'case3';
    else if (['general_neuromed', 'manual_general', 'base', 'full_v2_base'].includes(value)) value = 'general';
    if (['general', 'case1', 'case2', 'case3'].includes(value) && !out.includes(value)) out.push(value);
  }
  if (!out.length) return [];
  return ['general', ...['case1', 'case2', 'case3'].filter((value) => out.includes(value))];
}

function stableAnchorId(name) {
  const normalized = String(name || '').trim().toLowerCase();
  const slug = normalized
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 96) || 'unnamed';
  const hash = crypto.createHash('sha1').update(normalized).digest('hex').slice(0, 12);
  return `CLM_CONCEPT:${slug}_${hash}`;
}

function stableClaimId(prefix, paperId, ordinal) {
  const safePaper = String(paperId || 'paper')
    .replace(/[^A-Za-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 80) || 'paper';
  return `CLM:${prefix}_${safePaper}_${String(ordinal).padStart(3, '0')}`;
}

function canonicalClaimId(value) {
  const id = String(value || '').trim();
  if (!id.toUpperCase().startsWith('MANUAL') || !id.includes(':')) return id;
  return `CLM:${crypto.createHash('sha256').update(id).digest('hex').slice(0, 16)}`;
}

function canonicalConceptId(value) {
  const id = String(value || '').trim();
  if (!id.toUpperCase().startsWith('MANUAL') || !id.includes(':')) return id;
  return `CLM_CONCEPT:${crypto.createHash('sha256').update(id).digest('hex').slice(0, 16)}`;
}

function confidence(value) {
  const text = String(value ?? '').toLowerCase();
  if (text === 'high') return 0.85;
  if (text === 'medium') return 0.65;
  if (text === 'low') return 0.45;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0.5;
}

function sourcePaperFrom(paperId, row, meta) {
  const source = row.source && typeof row.source === 'object' ? row.source : {};
  const nested = row.metadata && typeof row.metadata === 'object' ? row.metadata : {};
  const rawId = String(paperId || '').trim();
  const ref = {
    pmid: /^[0-9]+$/.test(rawId) ? rawId : '',
    doi: String(row.doi || source.doi || nested.doi || meta.doi || meta.abstract_source_doi || '').trim(),
    title: String(row.title || source.title || nested.title || meta.title || meta.abstract_cache_title || '').trim(),
    authors: String(row.authors || source.authors || nested.authors || meta.authors || '').trim(),
    year: row.year || source.year || nested.year || meta.year ? Number(row.year || source.year || nested.year || meta.year) : null,
    journal: String(row.journal || source.journal || nested.journal || meta.journal || '').trim(),
  };
  if (/^(BIO|MED)RXIV:/i.test(rawId) && !ref.doi) ref.doi = rawId.replace(/^[^:]+:/, '');
  if (/^DOI:/i.test(rawId) && !ref.doi) ref.doi = rawId.replace(/^DOI:/i, '');
  if (/^ARXIV:/i.test(rawId)) ref.arxiv_id = rawId.replace(/^ARXIV:/i, '');
  if (/^PMC/i.test(rawId)) ref.pmcid = rawId;
  if (!ref.pmid && !ref.doi && !ref.arxiv_id && !ref.pmcid) ref.pmid = rawId;
  return ref;
}

function typeDomains(typeRaw) {
  const type = String(typeRaw || '').toUpperCase();
  if (type.includes('DISEASE') || type.includes('SYNDROME')) return ['disease'];
  if (type.includes('DRUG') || type.includes('TREATMENT') || type.includes('INTERVENTION') || type.includes('THERAPEUTIC')) return ['drug'];
  if (type.includes('GENE')) return ['gene'];
  if (type.includes('NEUROTRANSMITTER')) return ['neurotransmitter', 'biomarker'];
  if (type.includes('COGNITIVE') || type.includes('TASK')) return ['cognitive_function'];
  if (type.includes('OUTCOME') || type.includes('PHENOTYPE') || type.includes('ENDPOINT') || type.includes('RISK')) return ['treatment_outcome'];
  if (type.includes('CONNECTIVITY') || type.includes('CIRCUIT') || type.includes('NETWORK')) return ['connectivity', 'biomarker'];
  if (type.includes('IMAGING') || type.includes('MRI') || type.includes('PET') || type.includes('EEG') || type.includes('MEG') || type.includes('ELECTROPHYSIOLOGY')) {
    return ['imaging_feature', 'biomarker'];
  }
  if (type.includes('BIOMARKER') || type.includes('MARKER') || type.includes('MOLECULAR') || type.includes('CELLULAR') || type.includes('MECHANISM') || type.includes('PATHOLOGY') || type.includes('PROCESS')) {
    return ['biomarker'];
  }
  return ['biomarker'];
}

function conceptNode(id, name, domainTags, sourceVocab, metadata = {}, definition = '') {
  return {
    id,
    preferred_name: name,
    semantic_types: [],
    domain_tags: Array.from(new Set(domainTags.filter(Boolean))),
    source_vocab: sourceVocab,
    definition,
    aliases: [],
    external_ids: {},
    atlas_mapping: null,
    metadata,
  };
}

function edgeKey(edge) {
  return `${edge.source_id}\u0000${edge.target_id}\u0000${edge.relation_type}\u0000${edge.metadata?.claim_id || ''}`;
}

function canonicalizeClaim(raw, injectionSource) {
  const claim = JSON.parse(JSON.stringify(raw));
  claim.id = canonicalClaimId(claim.id);
  claim.subject_id = canonicalConceptId(claim.subject_id);
  claim.object_id = canonicalConceptId(claim.object_id);
  const originalPredicate = String(claim.predicate || '').trim();
  let predicate = originalPredicate;
  let negated = Boolean(claim.negated);
  if (predicateMap.has(predicate)) predicate = predicateMap.get(predicate);
  if (originalPredicate.startsWith('does_not_') || originalPredicate.startsWith('not_')) negated = true;
  if (!relationTypes.has(predicate)) predicate = 'is_associated_with';
  claim.predicate = predicate;
  claim.negated = negated;
  claim.metadata = claim.metadata || {};
  claim.metadata.original_predicate = originalPredicate;
  claim.metadata.predicate_canonicalized = originalPredicate !== predicate;
  claim.metadata.kg_injection_source = claim.metadata.kg_injection_source || injectionSource;
  claim.metadata.kg_injected = true;
  return claim;
}

function flattenInputRows(rows, metadataByPaper, args) {
  const defaultScope = normalizePaperScope(args.defaultScope);
  const out = [];
  const claimIdPrefix = defaultScope.length === 1
    ? `${defaultScope[0]}_manual`
    : 'scoped_manual';

  for (const row of rows) {
    if (Array.isArray(row.claims)) {
      const paperId = String(row.paper_id || row.id || row.pmid || row.doi || '').trim();
      const paperMeta = metadataByPaper.get(paperId) || {};
      let ordinal = 0;
      for (const claim of row.claims) {
        ordinal += 1;
        const paperScope =
          normalizePaperScope(claim.paper_scope).length
            ? normalizePaperScope(claim.paper_scope)
            : normalizePaperScope(row.paper_scope).length
              ? normalizePaperScope(row.paper_scope)
              : defaultScope;
        out.push({
          id: claim.id || claim.claim_id || stableClaimId(claimIdPrefix, paperId, ordinal),
          subject_name: claim.subject_name || claim.subject || '',
          subject_type: claim.subject_type || '',
          predicate: claim.predicate || 'is_associated_with',
          object_name: claim.object_name || claim.object || '',
          object_type: claim.object_type || 'OUTCOME',
          negated: Boolean(claim.negated),
          confidence: confidence(claim.confidence),
          evidence: typeof claim.evidence === 'object'
            ? claim.evidence
            : {
                study_type: 'manual_curated_abstract',
                methodology: String(claim.evidence || row.curation_note || ''),
                replicability: 'single_study',
                direction: '',
              },
          source_paper: sourcePaperFrom(paperId, row, paperMeta),
          raw_text: claim.raw_text || String(claim.evidence || ''),
          paper_scope: paperScope,
          metadata: {
            ...(claim.metadata || {}),
            paper_id: paperId,
            paper_scope: paperScope,
            curation_scope: claim.curation_scope || row.curation_scope || `${paperScope.join('_')}_manual`,
            kg_injection_source: claim.kg_injection_source || args.injectionSource,
            case_study: paperScope.find((scope) => scope.startsWith('case')) || '',
            case_id: paperScope.find((scope) => scope.startsWith('case')) || '',
            manual_decision: row.decision || '',
            manual_curation_note: row.curation_note || '',
            task: paperMeta.task || '',
            priority_tier: paperMeta.priority_tier || '',
            priority_score: paperMeta.priority_score || '',
            source_runs: paperMeta.source_runs || '',
            sources: paperMeta.sources || '',
            labels: paperMeta.labels || '',
            claim_ordinal: ordinal,
            original_confidence: claim.confidence ?? '',
          },
        });
      }
      continue;
    }

    const paperId = String(row.paper_id || row.pmid || row.doi || row.source_paper?.pmid || row.source_paper?.doi || '').trim();
    const paperMeta = metadataByPaper.get(paperId) || {};
    const paperScope =
      normalizePaperScope(row.paper_scope).length
        ? normalizePaperScope(row.paper_scope)
        : normalizePaperScope(row.metadata?.paper_scope).length
          ? normalizePaperScope(row.metadata.paper_scope)
          : defaultScope;
    out.push({
      ...row,
      id: row.id || row.claim_id || stableClaimId(claimIdPrefix, paperId || row.subject_name || row.object_name, 1),
      subject_name: row.subject_name || row.subject || '',
      subject_type: row.subject_type || row.metadata?.subject_type || '',
      object_name: row.object_name || row.object || '',
      object_type: row.object_type || row.metadata?.object_type || 'OUTCOME',
      confidence: confidence(row.confidence),
      source_paper: row.source_paper || sourcePaperFrom(paperId, row, paperMeta),
      raw_text: row.raw_text || (typeof row.evidence === 'string' ? row.evidence : ''),
      paper_scope: paperScope,
      metadata: {
        ...(row.metadata || {}),
        paper_id: paperId || row.metadata?.paper_id || '',
        paper_scope: paperScope,
        kg_injection_source: row.metadata?.kg_injection_source || args.injectionSource,
      },
    });
  }
  return out;
}

function hasPaperId(ref) {
  return Boolean(ref && (ref.pmid || ref.doi || ref.pmcid || ref.arxiv_id));
}

function repairClaimNode(node) {
  const claim = node.metadata || {};
  const nested = claim.metadata && typeof claim.metadata === 'object' ? claim.metadata : {};
  const source = claim.source && typeof claim.source === 'object' ? claim.source : {};
  let sourceRepaired = false;
  if (!hasPaperId(claim.source_paper)) {
    claim.source_paper = {
      pmid: String(source.pmid || nested.pmid || claim.paper_id || '').trim(),
      doi: String(source.doi || nested.doi || '').trim(),
      title: String(source.title || nested.title || '').trim(),
      year: source.year || nested.year || null,
      journal: String(source.journal || nested.journal || '').trim(),
    };
    sourceRepaired = hasPaperId(claim.source_paper);
  }

  let scopeRepaired = false;
  if (!normalizePaperScope(claim.paper_scope).length) {
    const text = [
      claim.id,
      claim.claim_id,
      claim.curation_scope,
      claim.kg_injection_source,
      claim.case_study,
      claim.case_id,
      nested.curation_scope,
      nested.kg_injection_source,
      nested.case_study,
      nested.case_id,
      nested.staging_dataset,
    ].join(' ').toLowerCase();
    const scopes = [];
    if (text.includes('case1') || text.includes('cs1') || text.includes('transdiagnostic')) scopes.push('case1');
    if (text.includes('case2') || text.includes('cs2')) scopes.push('case2');
    if (text.includes('case3') || text.includes('cs3') || text.includes('hindcast')) scopes.push('case3');
    if (
      text.includes('general_neuromed') ||
      text.includes('manual_general') ||
      text.includes('strict_neuroscience') ||
      text.includes('genmed')
    ) {
      scopes.push('general');
    }
    claim.paper_scope = normalizePaperScope(scopes.length ? scopes : ['general']);
    scopeRepaired = true;
  }
  return { sourceRepaired, scopeRepaired };
}

function isClaimNode(node) {
  const metadata = node.metadata || {};
  return (
    String(node.id || '').startsWith('CLM:') ||
    (node.domain_tags || []).includes('claim') ||
    (metadata.source_paper && metadata.subject_name)
  );
}

function computeGraphStats(data) {
  const domains = {};
  const sources = {};
  const relations = {};
  for (const node of Object.values(data.concepts || {})) {
    for (const domain of node.domain_tags || []) domains[domain] = (domains[domain] || 0) + 1;
    if (node.source_vocab) sources[node.source_vocab] = (sources[node.source_vocab] || 0) + 1;
  }
  for (const edge of data.edges || []) {
    if (edge.relation_type) relations[edge.relation_type] = (relations[edge.relation_type] || 0) + 1;
  }
  return {
    n_concepts: Object.keys(data.concepts || {}).length,
    n_edges: (data.edges || []).length,
    domains,
    sources,
    relations,
  };
}

function scopeStats(data) {
  const paperSets = {
    general: new Set(),
    case1: new Set(),
    case2: new Set(),
    case3: new Set(),
    case_study_any: new Set(),
  };
  const claimCounts = { general: 0, case1: 0, case2: 0, case3: 0, case_study_any: 0 };
  let claimNodes = 0;
  let missingScope = 0;
  let claimsWithoutPaper = 0;
  for (const node of Object.values(data.concepts || {})) {
    if (!isClaimNode(node)) continue;
    claimNodes += 1;
    const claim = node.metadata || {};
    const scopes = normalizePaperScope(claim.paper_scope || node.paper_scope);
    if (!scopes.length) {
      missingScope += 1;
      continue;
    }
    const paper = claim.source_paper || {};
    const paperId = String(paper.pmid || paper.doi || paper.pmcid || paper.arxiv_id || claim.paper_id || claim.metadata?.paper_id || '').trim();
    if (!paperId) claimsWithoutPaper += 1;
    const hasCase = scopes.some((scope) => scope === 'case1' || scope === 'case2' || scope === 'case3');
    for (const scope of scopes) {
      claimCounts[scope] = (claimCounts[scope] || 0) + 1;
      if (paperId && paperSets[scope]) paperSets[scope].add(paperId);
    }
    if (hasCase) {
      claimCounts.case_study_any += 1;
      if (paperId) paperSets.case_study_any.add(paperId);
    }
  }
  const paperCounts = {};
  for (const [scope, papers] of Object.entries(paperSets)) paperCounts[scope] = papers.size;
  return { claimNodes, missingScope, claimsWithoutPaper, paperCounts, claimCounts };
}

function injectClaims(graph, claims, args) {
  graph.concepts = graph.concepts || {};
  graph.edges = graph.edges || [];
  const exactNameToId = new Map();
  for (const [id, node] of Object.entries(graph.concepts)) {
    const key = String(node.preferred_name || '').trim().toLowerCase();
    if (key && !exactNameToId.has(key)) exactNameToId.set(key, id);
  }
  const existingConcepts = new Set(Object.keys(graph.concepts));
  const existingEdges = new Set(graph.edges.map(edgeKey));
  const existingExtractedIds = readJsonlIdSetStreaming(args.extracted);

  const appendLines = [];
  const predicateChanges = {};
  let anchorsAdded = 0;
  let claimsAdded = 0;
  let edgesAdded = 0;
  let skippedExistingClaims = 0;
  let extractedAppended = 0;

  for (const raw of claims) {
    const claim = canonicalizeClaim(raw, args.injectionSource);
    claim.paper_scope = normalizePaperScope(claim.paper_scope || claim.metadata?.paper_scope || args.defaultScope);
    claim.metadata = claim.metadata || {};
    claim.metadata.paper_scope = normalizePaperScope(claim.metadata.paper_scope || claim.paper_scope);
    if (!claim.paper_scope.length) throw new Error(`Claim ${claim.id} is missing paper_scope`);
    if (!claim.subject_name || !claim.object_name) throw new Error(`Claim ${claim.id} is missing subject_name or object_name`);

    const subjectKey = String(claim.subject_name || '').trim().toLowerCase();
    const objectKey = String(claim.object_name || '').trim().toLowerCase();
    claim.subject_id = claim.subject_id || exactNameToId.get(subjectKey) || stableAnchorId(claim.subject_name);
    claim.object_id = claim.object_id || exactNameToId.get(objectKey) || stableAnchorId(claim.object_name);

    if (existingConcepts.has(claim.id)) {
      skippedExistingClaims += 1;
      continue;
    }

    for (const [role, idKey, nameKey, typeKey] of [
      ['subject', 'subject_id', 'subject_name', 'subject_type'],
      ['object', 'object_id', 'object_name', 'object_type'],
    ]) {
      const id = claim[idKey];
      if (!id || existingConcepts.has(id)) continue;
      const type = claim.metadata?.[typeKey] || claim[typeKey] || '';
      graph.concepts[id] = conceptNode(
        id,
        claim[nameKey] || id,
        typeDomains(type),
        'manual_claim_anchor',
        {
          anchor_role: role,
          atom_type: type,
          paper_scope: claim.paper_scope,
          curation_scope: claim.metadata?.curation_scope || '',
          staging_source: args.injectionSource,
        }
      );
      existingConcepts.add(id);
      anchorsAdded += 1;
    }

    graph.concepts[claim.id] = conceptNode(
      claim.id,
      `${claim.subject_name} ${claim.predicate} ${claim.object_name}`,
      ['claim'],
      'claim_extraction',
      claim,
      claim.raw_text || ''
    );
    existingConcepts.add(claim.id);
    claimsAdded += 1;

    for (const edge of [
      {
        source_id: claim.subject_id,
        target_id: claim.object_id,
        relation_type: claim.predicate,
        source: `claim:${claim.source_paper?.pmid || claim.source_paper?.doi || claim.id}`,
        confidence: claim.confidence,
        evidence_ref: claim.source_paper?.title || '',
        metadata: {
          claim_id: claim.id,
          negated: claim.negated,
          paper_scope: claim.paper_scope,
          original_predicate: claim.metadata?.original_predicate || claim.predicate,
        },
      },
      {
        source_id: claim.id,
        target_id: claim.subject_id,
        relation_type: 'about',
        source: 'claim_extraction',
        confidence: claim.confidence,
        evidence_ref: claim.source_paper?.title || '',
        metadata: { claim_id: claim.id, anchor_role: 'subject', paper_scope: claim.paper_scope },
      },
      {
        source_id: claim.id,
        target_id: claim.object_id,
        relation_type: 'about',
        source: 'claim_extraction',
        confidence: claim.confidence,
        evidence_ref: claim.source_paper?.title || '',
        metadata: { claim_id: claim.id, anchor_role: 'object', paper_scope: claim.paper_scope },
      },
    ]) {
      if (!edge.source_id || !edge.target_id || !existingConcepts.has(edge.source_id) || !existingConcepts.has(edge.target_id)) continue;
      const key = edgeKey(edge);
      if (existingEdges.has(key)) continue;
      graph.edges.push(edge);
      existingEdges.add(key);
      edgesAdded += 1;
    }

    const original = claim.metadata?.original_predicate || claim.predicate;
    if (original !== claim.predicate) {
      const key = `${original}->${claim.predicate}`;
      predicateChanges[key] = (predicateChanges[key] || 0) + 1;
    }
    if (!existingExtractedIds.has(claim.id)) {
      appendLines.push(JSON.stringify(claim));
      existingExtractedIds.add(claim.id);
      extractedAppended += 1;
    }
  }

  return {
    anchorsAdded,
    claimsAdded,
    edgesAdded,
    skippedExistingClaims,
    extractedAppended,
    appendLines,
    predicateChangeKinds: Object.keys(predicateChanges).length,
    predicateChangeTop: Object.entries(predicateChanges)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 30)
      .map(([change, count]) => ({ change, count })),
  };
}

async function main() {
  const args = parseArgs(process.argv);
  const metadataByPaper = new Map();
  let metadataRows = 0;
  for (const file of args.metadataCsv) metadataRows += loadMetadataCsv(file, metadataByPaper);
  for (const dir of args.metadataDir) metadataRows += loadMetadataDir(dir, metadataByPaper);

  const rawRows = readJsonl(args.claims);
  const flattenedClaims = flattenInputRows(rawRows, metadataByPaper, args);
  if (args.flattenOutput) {
    fs.writeFileSync(args.flattenOutput, flattenedClaims.map(JSON.stringify).join('\n') + '\n', 'utf8');
  }

  const graph = await loadGraphStreaming(args.graph);
  const beforeStats = computeGraphStats(graph);
  const injection = injectClaims(graph, flattenedClaims, args);

  let repairedSourcePaper = 0;
  let repairedScope = 0;
  if (args.repairExisting) {
    for (const node of Object.values(graph.concepts || {})) {
      if (!isClaimNode(node)) continue;
      const repaired = repairClaimNode(node);
      if (repaired.sourceRepaired) repairedSourcePaper += 1;
      if (repaired.scopeRepaired) repairedScope += 1;
    }
  }

  graph.metadata = graph.metadata || {};
  graph.metadata.stats = computeGraphStats(graph);
  graph.metadata.scoped_manual_claim_injection = {
    claims_file: path.relative(repo, args.claims).replace(/\\/g, '/'),
    injected_at: new Date().toISOString(),
    injection_source: args.injectionSource,
    dry_run: args.dryRun,
    raw_rows: rawRows.length,
    flattened_claims: flattenedClaims.length,
    repair_existing: args.repairExisting,
    repaired_source_paper: repairedSourcePaper,
    repaired_scope: repairedScope,
    ...Object.fromEntries(Object.entries(injection).filter(([key]) => key !== 'appendLines')),
  };

  const afterScopeStats = scopeStats(graph);
  const summary = {
    dryRun: args.dryRun,
    graph: args.graph,
    extracted: args.extracted,
    rawRows: rawRows.length,
    flattenedClaims: flattenedClaims.length,
    metadataRows,
    repairExisting: args.repairExisting,
    repairedSourcePaper,
    repairedScope,
    beforeStats: {
      n_concepts: beforeStats.n_concepts,
      n_edges: beforeStats.n_edges,
    },
    afterStats: {
      n_concepts: graph.metadata.stats.n_concepts,
      n_edges: graph.metadata.stats.n_edges,
    },
    injection: Object.fromEntries(Object.entries(injection).filter(([key]) => key !== 'appendLines')),
    scopeStats: afterScopeStats,
  };

  if (!args.dryRun) {
    const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, '').replace('T', '_');
    const graphBackup = `${args.graph}.bak_scoped_manual_${stamp}`;
    const extractedBackup = fs.existsSync(args.extracted) ? `${args.extracted}.bak_scoped_manual_${stamp}` : '';
    fs.copyFileSync(args.graph, graphBackup);
    if (extractedBackup) fs.copyFileSync(args.extracted, extractedBackup);
    writeGraphCompact(`${args.graph}.tmp`, graph);
    // Windows renameSync cannot overwrite an existing file (EPERM). The original is already
    // copied to graphBackup above, so move it aside (rename to a fresh name always works),
    // put the new graph in place, then drop the moved-aside original.
    const oldGraph = `${args.graph}.replacing_${stamp}`;
    if (fs.existsSync(args.graph)) fs.renameSync(args.graph, oldGraph);
    fs.renameSync(`${args.graph}.tmp`, args.graph);
    if (fs.existsSync(oldGraph)) fs.rmSync(oldGraph);
    if (injection.appendLines.length) {
      fs.appendFileSync(args.extracted, injection.appendLines.join('\n') + '\n', 'utf8');
    }
    summary.graphBackup = graphBackup;
    summary.extractedBackup = extractedBackup;
  }

  console.log(JSON.stringify(summary, null, 2));
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : error);
  process.exit(1);
});
