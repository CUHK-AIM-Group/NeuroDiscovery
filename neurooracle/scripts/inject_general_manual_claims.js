const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const repo = path.resolve(__dirname, '..', '..');
const stagingDir = path.join(repo, 'neurooracle', 'data', 'phase2_staging', 'general_neuromed_manual_20260610');
const defaultClaims = path.join(stagingDir, 'manual_claims.jsonl');
const defaultGraph = path.join(repo, 'neurooracle', 'data', 'cs_runs', 'phase2_case1_transdiagnostic_v1', 'knowledge_graph.json');
const defaultExtracted = path.join(repo, 'neurooracle', 'data', 'cs_runs', 'phase2_case1_transdiagnostic_v1', 'extracted_claims.jsonl');

function parseArgs(argv) {
  const args = {
    claims: defaultClaims,
    graph: defaultGraph,
    extracted: defaultExtracted,
    dryRun: false,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--dry-run') args.dryRun = true;
    else if (a === '--claims') args.claims = argv[++i];
    else if (a === '--graph') args.graph = argv[++i];
    else if (a === '--extracted') args.extracted = argv[++i];
    else throw new Error(`Unknown argument: ${a}`);
  }
  return args;
}

function readJsonl(file) {
  return fs.readFileSync(file, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
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

function normalizeClaimShape(raw, exactNameToId) {
  const claim = JSON.parse(JSON.stringify(raw));
  claim.id = claim.id || claim.claim_id;
  if (!claim.id) throw new Error('Manual claim is missing both id and claim_id.');

  claim.metadata = claim.metadata || {};
  claim.metadata.subject_type = claim.metadata.subject_type || claim.subject_type || '';
  claim.metadata.object_type = claim.metadata.object_type || claim.object_type || 'OUTCOME';
  claim.metadata.curation_scope = claim.metadata.curation_scope || 'general_neuromed_manual_strict_neuroscience';

  const subjectNameKey = String(claim.subject_name || '').trim().toLowerCase();
  const objectNameKey = String(claim.object_name || '').trim().toLowerCase();
  claim.subject_id = claim.subject_id || exactNameToId.get(subjectNameKey) || stableAnchorId(claim.subject_name);
  claim.object_id = claim.object_id || exactNameToId.get(objectNameKey) || stableAnchorId(claim.object_name);

  claim.source_paper = claim.source_paper || {
    pmid: claim.pmid || '',
    doi: claim.doi || '',
    title: claim.title || '',
    year: claim.year || null,
    journal: claim.journal || '',
  };
  claim.raw_text = claim.raw_text
    || (typeof claim.evidence === 'string' ? claim.evidence : '')
    || claim.evidence?.rationale
    || claim.evidence?.methodology
    || '';
  return claim;
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

function typeDomains(typeRaw) {
  const t = String(typeRaw || '').toUpperCase();
  if (t.includes('DISEASE') || t.includes('SYNDROME')) return ['disease'];
  if (t.includes('DRUG') || t.includes('TREATMENT') || t.includes('INTERVENTION') || t.includes('THERAPEUTIC')) return ['drug'];
  if (t.includes('GENE')) return ['gene'];
  if (t.includes('NEUROTRANSMITTER')) return ['neurotransmitter', 'biomarker'];
  if (t.includes('COGNITIVE') || t.includes('TASK')) return ['cognitive_function'];
  if (t.includes('OUTCOME') || t.includes('PHENOTYPE') || t.includes('ENDPOINT') || t.includes('RISK')) return ['treatment_outcome'];
  if (t.includes('CONNECTIVITY') || t.includes('CIRCUIT') || t.includes('NETWORK')) return ['connectivity', 'biomarker'];
  if (t.includes('IMAGING') || t.includes('MRI') || t.includes('PET') || t.includes('EEG') || t.includes('MEG') || t.includes('ELECTROPHYSIOLOGY')) {
    return ['imaging_feature', 'biomarker'];
  }
  if (t.includes('BIOMARKER') || t.includes('MARKER') || t.includes('MOLECULAR') || t.includes('CELLULAR') || t.includes('MECHANISM') || t.includes('PATHOLOGY') || t.includes('PROCESS')) {
    return ['biomarker'];
  }
  return ['biomarker'];
}

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
  restrains: 'inhibits',
  rules_in_or_out: 'distinguishes',
  is_increased_in: 'is_biomarker_of',
  changes_before: 'predicts',
}));

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

function canonicalizeClaim(raw) {
  const claim = JSON.parse(JSON.stringify(raw));
  const original = claim.predicate;
  let predicate = String(original || '').trim();
  let negated = Boolean(claim.negated);
  if (predicateMap.has(predicate)) {
    predicate = predicateMap.get(predicate);
  }
  if (String(original || '').startsWith('does_not_') || String(original || '').startsWith('not_')) {
    negated = true;
  }
  if (!relationTypes.has(predicate)) {
    predicate = 'is_associated_with';
  }
  claim.predicate = predicate;
  claim.negated = negated;
  claim.metadata = claim.metadata || {};
  claim.metadata.original_predicate = original;
  claim.metadata.predicate_canonicalized = original !== predicate;
  claim.metadata.kg_injection_source = 'general_neuromed_manual_20260610';
  claim.metadata.kg_injected = true;
  return claim;
}

function edgeKey(e) {
  return `${e.source_id}\u0000${e.target_id}\u0000${e.relation_type}\u0000${e.metadata?.claim_id || ''}`;
}

function computeStats(data) {
  const domains = {};
  const sources = {};
  const relations = {};
  for (const node of Object.values(data.concepts || {})) {
    for (const d of node.domain_tags || []) domains[d] = (domains[d] || 0) + 1;
    const s = node.source_vocab || '';
    if (s) sources[s] = (sources[s] || 0) + 1;
  }
  for (const edge of data.edges || []) {
    const r = edge.relation_type || '';
    if (r) relations[r] = (relations[r] || 0) + 1;
  }
  return {
    n_concepts: Object.keys(data.concepts || {}).length,
    n_edges: (data.edges || []).length,
    domains,
    sources,
    relations,
  };
}

function main() {
  const args = parseArgs(process.argv);
  const graph = JSON.parse(fs.readFileSync(args.graph, 'utf8'));
  graph.concepts = graph.concepts || {};
  graph.edges = graph.edges || [];

  const exactNameToId = new Map();
  for (const [id, node] of Object.entries(graph.concepts)) {
    const key = String(node.preferred_name || '').trim().toLowerCase();
    if (key && !exactNameToId.has(key)) exactNameToId.set(key, id);
  }

  const rawClaims = readJsonl(args.claims);
  const claims = rawClaims.map((claim) => canonicalizeClaim(normalizeClaimShape(claim, exactNameToId)));
  const existingConcepts = new Set(Object.keys(graph.concepts));
  const existingEdges = new Set(graph.edges.map(edgeKey));
  const existingExtractedIds = fs.existsSync(args.extracted)
    ? new Set(fs.readFileSync(args.extracted, 'utf8').split(/\r?\n/).filter(Boolean).map((line) => {
        try { return JSON.parse(line).id; } catch { return ''; }
      }).filter(Boolean))
    : new Set();

  let anchorsAdded = 0;
  let claimsAdded = 0;
  let edgesAdded = 0;
  let skippedExistingClaims = 0;
  let extractedAppended = 0;
  const predicateChanges = {};
  const appendLines = [];

  for (const claim of claims) {
    if (existingConcepts.has(claim.id)) {
      skippedExistingClaims += 1;
      continue;
    }

    for (const [which, idKey, nameKey, typeKey] of [
      ['subject', 'subject_id', 'subject_name', 'subject_type'],
      ['object', 'object_id', 'object_name', 'object_type'],
    ]) {
      const id = claim[idKey];
      if (!id) continue;
      if (!existingConcepts.has(id)) {
        const type = claim.metadata?.[typeKey] || '';
        graph.concepts[id] = conceptNode(
          id,
          claim[nameKey] || id,
          typeDomains(type),
          'manual_general_claim_anchor',
          {
            anchor_role: which,
            atom_type: type,
            curation_scope: claim.metadata?.curation_scope || '',
            staging_source: 'general_neuromed_manual_20260610',
          }
        );
        existingConcepts.add(id);
        anchorsAdded += 1;
      }
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
        source: `claim:${claim.source_paper?.pmid || claim.id}`,
        confidence: claim.confidence,
        evidence_ref: claim.source_paper?.title || '',
        metadata: {
          claim_id: claim.id,
          negated: claim.negated,
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
        metadata: { claim_id: claim.id, anchor_role: 'subject' },
      },
      {
        source_id: claim.id,
        target_id: claim.object_id,
        relation_type: 'about',
        source: 'claim_extraction',
        confidence: claim.confidence,
        evidence_ref: claim.source_paper?.title || '',
        metadata: { claim_id: claim.id, anchor_role: 'object' },
      },
    ]) {
      if (!edge.source_id || !edge.target_id || !existingConcepts.has(edge.source_id) || !existingConcepts.has(edge.target_id)) {
        continue;
      }
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

  const beforeStats = graph.metadata?.stats || {};
  const afterStats = computeStats(graph);
  graph.metadata = graph.metadata || {};
  graph.metadata.version = graph.metadata.version || '0.1';
  graph.metadata.created = new Date().toISOString();
  graph.metadata.stats = afterStats;
  graph.metadata.general_manual_claim_injection = {
    staging_dir: path.relative(repo, stagingDir).replace(/\\/g, '/'),
    claims_file: path.relative(repo, args.claims).replace(/\\/g, '/'),
    injected_at: new Date().toISOString(),
    dry_run: args.dryRun,
    raw_claims: rawClaims.length,
    claims_added: claimsAdded,
    anchors_added: anchorsAdded,
    edges_added: edgesAdded,
    skipped_existing_claims: skippedExistingClaims,
    extracted_claims_appended: extractedAppended,
    predicate_changes: predicateChanges,
  };

  const summary = {
    dryRun: args.dryRun,
    graph: args.graph,
    extracted: args.extracted,
    rawClaims: rawClaims.length,
    claimsAdded,
    anchorsAdded,
    edgesAdded,
    skippedExistingClaims,
    extractedAppended,
    beforeStats,
    afterStats,
    predicateChanges,
  };

  if (!args.dryRun) {
    const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, '').replace('T', '_');
    const graphBackup = `${args.graph}.bak_general_manual_${stamp}`;
    const extractedBackup = fs.existsSync(args.extracted) ? `${args.extracted}.bak_general_manual_${stamp}` : '';
    fs.copyFileSync(args.graph, graphBackup);
    if (extractedBackup) fs.copyFileSync(args.extracted, extractedBackup);
    const tmpGraph = `${args.graph}.tmp`;
    fs.writeFileSync(tmpGraph, JSON.stringify(graph, null, 2) + '\n', 'utf8');
    fs.renameSync(tmpGraph, args.graph);
    if (appendLines.length) {
      fs.appendFileSync(args.extracted, appendLines.join('\n') + '\n', 'utf8');
    }
    summary.graphBackup = graphBackup;
    summary.extractedBackup = extractedBackup;
  }

  console.log(JSON.stringify(summary, null, 2));
}

main();
