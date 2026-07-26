const fs = require('fs');
const path = require('path');

const repo = path.resolve(__dirname, '..', '..');
const defaultGraph = path.join(repo, 'neurooracle', 'data', 'full_v2', 'knowledge_graph.json');
const defaultOutDir = path.join(repo, 'neurooracle', 'data', 'reports', 'clm_standardization');

const CANONICAL_SOURCE_ALLOWLIST = new Set([
  'MeSH',
  'NeuroNames',
  'DisGeNET',
  'HPO',
  'HGNC',
  'ClinicalOutcomes',
  'MedDRA-SOC',
  'IndividualDataAnchor',
  'CognitiveAtlas',
  'ATC',
  'UKB-Showcase',
  'ADNI',
  'HCP',
  'NeuroClaw-IM',
  'NeuroClaw-GeneSet',
  'NeuroClaw-GM',
  'NeuroClaw-ManualCanonical',
  'experiment_infra',
  'visual_functional_roi',
]);

const GENERIC_NAMES = new Set([
  'abnormalities',
  'activity',
  'age',
  'analysis',
  'association',
  'biomarker',
  'biomarkers',
  'brain',
  'change',
  'changes',
  'clinical outcomes',
  'cognition',
  'cognitive impairment',
  'cognitive performance',
  'disease',
  'disorder',
  'function',
  'imaging',
  'intervention',
  'marker',
  'markers',
  'memory',
  'method',
  'model',
  'neuroimaging',
  'outcome',
  'outcomes',
  'patients',
  'performance',
  'risk',
  'score',
  'symptoms',
  'treatment',
]);

const ATOM_ORDER = [
  'disease',
  'drug',
  'imaging_marker',
  'gene_target',
  'cognitive_task',
  'outcome',
  'individual_data',
];

const DOMAIN_TO_ATOMS = {
  disease: ['disease'],
  drug: ['drug'],
  imaging_feature: ['imaging_marker'],
  connectivity: ['imaging_marker'],
  biomarker: ['imaging_marker'],
  neuroanatomy: ['imaging_marker'],
  gene: ['gene_target'],
  neurotransmitter: ['gene_target'],
  paradigm: ['cognitive_task'],
  cognitive_function: ['cognitive_task'],
  visual_stimulus: ['cognitive_task'],
  emotion: ['cognitive_task'],
  vigilance: ['cognitive_task'],
  treatment_outcome: ['outcome'],
  dataset_variable: ['outcome', 'individual_data'],
  individual_data_anchor: ['individual_data'],
};

function normalizeAtomTypes(values) {
  const raw = Array.isArray(values) ? values : values ? [values] : [];
  const out = new Set();
  for (const item of raw) {
    let value = String(item || '').trim().toLowerCase();
    if (!value) continue;
    if (value === 'imaging') value = 'imaging_marker';
    if (value === 'gene') value = 'gene_target';
    if (value === 'individual') value = 'individual_data';
    if (ATOM_ORDER.includes(value)) out.add(value);
  }
  return ATOM_ORDER.filter((atom) => out.has(atom));
}

function inferAtomTypes(node) {
  const out = new Set(normalizeAtomTypes(node.metadata?.atom_types || node.metadata?.atom_type));
  const id = String(node.id || '');
  if (id.startsWith('NCL_IMAGING:')) return ['imaging_marker'];
  if (id.startsWith('NCL_OUTCOME:')) return ['outcome'];
  if (id.startsWith('NCL_DISEASE:')) return ['disease'];
  if (id.startsWith('NCL_DRUG:')) return ['drug'];
  if (id.startsWith('NCL_GENE_TARGET:')) return ['gene_target'];
  if (id.startsWith('NCL_INDIVIDUAL:') || id.startsWith('NCL_COVARIATE:')) return ['individual_data'];
  if (id.startsWith('NCL_BIOMARKER:') || id.startsWith('NCL_METHOD:')) return ['imaging_marker'];
  for (const tag of node.domain_tags || []) {
    for (const atom of DOMAIN_TO_ATOMS[String(tag || '').trim()] || []) out.add(atom);
  }
  if (id.startsWith('IM:') || id.startsWith('NCL_IMAGING:')) out.add('imaging_marker');
  if (id.startsWith('GM:') || id.startsWith('GENESET:') || id.startsWith('NCL_GENE_TARGET:')) out.add('gene_target');
  if (id.startsWith('OUTCOME:') || id.startsWith('NCL_OUTCOME:')) out.add('outcome');
  if (id.startsWith('NCL_DISEASE:')) out.add('disease');
  if (id.startsWith('NCL_DRUG:')) out.add('drug');
  if (id.startsWith('NCL_INDIVIDUAL:') || id.startsWith('NCL_COVARIATE:')) out.add('individual_data');
  if (id.startsWith('COGAT_TASK:') || id.startsWith('COGAT_CONCEPT:')) out.add('cognitive_task');
  return ATOM_ORDER.filter((atom) => out.has(atom));
}

function parseArgs(argv) {
  const args = {
    graph: defaultGraph,
    outputDir: defaultOutDir,
    apply: false,
    applyTier: 'high',
    includeClaimExtractionCanonical: false,
    repairMalformedUnnamed: false,
    dedupeClmNormalized: false,
    manualConcepts: '',
    manualMappings: '',
    maxExamples: 50,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--graph') args.graph = argv[++i];
    else if (arg === '--output-dir') args.outputDir = argv[++i];
    else if (arg === '--apply') args.apply = true;
    else if (arg === '--apply-tier') args.applyTier = argv[++i];
    else if (arg === '--include-claim-extraction-canonical') args.includeClaimExtractionCanonical = true;
    else if (arg === '--repair-malformed-unnamed') args.repairMalformedUnnamed = true;
    else if (arg === '--dedupe-clm-normalized') args.dedupeClmNormalized = true;
    else if (arg === '--manual-concepts') args.manualConcepts = argv[++i];
    else if (arg === '--manual-mappings') args.manualMappings = argv[++i];
    else if (arg === '--max-examples') args.maxExamples = Number(argv[++i]);
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function conceptValues(concepts) {
  return Array.isArray(concepts) ? concepts : Object.values(concepts || {});
}

function getConcept(data, id) {
  if (Array.isArray(data.concepts)) return data.concepts.find((node) => node && node.id === id);
  return data.concepts?.[id];
}

function setConcept(data, id, node) {
  if (Array.isArray(data.concepts)) {
    const index = data.concepts.findIndex((item) => item && item.id === id);
    if (index >= 0) data.concepts[index] = node;
    else data.concepts.push(node);
    return;
  }
  data.concepts[id] = node;
}

function deleteConcept(data, id) {
  if (Array.isArray(data.concepts)) {
    const index = data.concepts.findIndex((item) => item && item.id === id);
    if (index >= 0) data.concepts.splice(index, 1);
    return;
  }
  delete data.concepts[id];
}

function normalizeName(value) {
  return String(value || '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/^CLM_CONCEPT:/i, '')
    .replace(/_/g, ' ')
    .replace(/\([^)]*\)/g, ' ')
    .replace(/[^a-zA-Z0-9]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
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

function canonicalNameFromClm(node) {
  const preferred = String(node.preferred_name || '').trim();
  if (preferred && preferred !== node.id) return preferred;
  const id = String(node.id || '');
  if (id.startsWith('CLM_CONCEPT:')) {
    return id.replace(/^CLM_CONCEPT:/, '').replace(/_/g, ' ');
  }
  return preferred;
}

function isClmConcept(node) {
  return String(node.id || '').startsWith('CLM_CONCEPT:');
}

function isClaimNode(node) {
  return (
    String(node.id || '').startsWith('CLM:') ||
    (node.domain_tags || []).includes('claim')
  );
}

function isCanonicalNode(node, includeClaimExtractionCanonical) {
  if (!node || !node.id) return false;
  if (isClmConcept(node) || isClaimNode(node)) return false;
  if (includeClaimExtractionCanonical) return true;
  return CANONICAL_SOURCE_ALLOWLIST.has(String(node.source_vocab || ''));
}

function isGeneric(name) {
  const normalized = normalizeName(name);
  return GENERIC_NAMES.has(normalized) || normalized.length < 3;
}

function pushIndex(index, key, node) {
  if (!key) return;
  if (!index.has(key)) index.set(key, []);
  index.get(key).push(node);
}

function buildCanonicalIndex(concepts, includeClaimExtractionCanonical) {
  const exact = new Map();
  const aliases = new Map();
  const normalized = new Map();
  let canonicalNodes = 0;

  for (const node of concepts) {
    if (!isCanonicalNode(node, includeClaimExtractionCanonical)) continue;
    canonicalNodes += 1;
    const preferred = String(node.preferred_name || '').trim();
    pushIndex(exact, preferred.toLowerCase(), node);
    pushIndex(normalized, normalizeName(preferred), node);
    for (const alias of node.aliases || []) {
      const aliasText = String(alias || '').trim();
      pushIndex(aliases, aliasText.toLowerCase(), node);
      pushIndex(normalized, normalizeName(aliasText), node);
    }
  }
  return { exact, aliases, normalized, canonicalNodes };
}

function candidateRecord(clmNode, target, matchType, confidence) {
  return {
    clm_id: clmNode.id,
    clm_name: canonicalNameFromClm(clmNode),
    clm_source_vocab: clmNode.source_vocab || '',
    clm_domain_tags: clmNode.domain_tags || [],
    target_id: target.id,
    target_name: target.preferred_name || '',
    target_source_vocab: target.source_vocab || '',
    target_domain_tags: target.domain_tags || [],
    match_type: matchType,
    confidence,
  };
}

function chooseCandidate(clmNode, index) {
  const name = canonicalNameFromClm(clmNode);
  const exactKey = String(name || '').trim().toLowerCase();
  const normKey = normalizeName(name);
  if (!name || /^unnamed($| )/.test(normKey)) {
    return { status: 'malformed_unnamed', name, candidates: [] };
  }
  if (isGeneric(name)) {
    return { status: 'generic_no_auto', name, candidates: [] };
  }

  const matchers = [
    ['exact_preferred_name', index.exact.get(exactKey) || [], 1.0],
    ['exact_alias', index.aliases.get(exactKey) || [], 0.97],
    ['normalized_name_or_alias', index.normalized.get(normKey) || [], 0.9],
  ];

  for (const [matchType, rawCandidates, confidence] of matchers) {
    const candidates = dedupeCandidates(rawCandidates);
    if (candidates.length === 1) {
      return {
        status: confidence >= 0.97 ? 'high_confidence' : 'medium_confidence',
        name,
        candidates: [candidateRecord(clmNode, candidates[0], matchType, confidence)],
      };
    }
    if (candidates.length > 1) {
      return {
        status: 'ambiguous',
        name,
        candidates: candidates.map((target) => candidateRecord(clmNode, target, matchType, confidence)),
      };
    }
  }
  return { status: 'no_match', name, candidates: [] };
}

function dedupeCandidates(nodes) {
  const seen = new Set();
  const out = [];
  for (const node of nodes || []) {
    if (!node || seen.has(node.id)) continue;
    seen.add(node.id);
    out.push(node);
  }
  return out;
}

function degreeMap(edges) {
  const degree = new Map();
  for (const edge of edges || []) {
    if (edge.source_id) degree.set(edge.source_id, (degree.get(edge.source_id) || 0) + 1);
    if (edge.target_id) degree.set(edge.target_id, (degree.get(edge.target_id) || 0) + 1);
  }
  return degree;
}

function edgeKey(edge) {
  return [
    edge.source_id || '',
    edge.target_id || '',
    edge.relation_type || '',
    edge.source || '',
    edge.metadata?.claim_id || '',
    edge.metadata?.anchor_role || '',
  ].join('\u0000');
}

function computeStats(data) {
  const domains = {};
  const sources = {};
  const relations = {};
  for (const node of conceptValues(data.concepts)) {
    for (const domain of node.domain_tags || []) domains[domain] = (domains[domain] || 0) + 1;
    const source = node.source_vocab || '';
    if (source) sources[source] = (sources[source] || 0) + 1;
  }
  for (const edge of data.edges || []) {
    const relation = edge.relation_type || '';
    if (relation) relations[relation] = (relations[relation] || 0) + 1;
  }
  return {
    n_concepts: conceptValues(data.concepts).length,
    n_edges: (data.edges || []).length,
    domains,
    sources,
    relations,
  };
}

function applyMappings(data, mappings) {
  const mapping = new Map(mappings.map((row) => [row.clm_id, row.target_id]));
  let edgeEndpointsRewritten = 0;
  let claimMetadataRewritten = 0;
  let removedDuplicateEdges = 0;
  let removedConcepts = 0;

  for (const edge of data.edges || []) {
    if (mapping.has(edge.source_id)) {
      edge.source_id = mapping.get(edge.source_id);
      edgeEndpointsRewritten += 1;
    }
    if (mapping.has(edge.target_id)) {
      edge.target_id = mapping.get(edge.target_id);
      edgeEndpointsRewritten += 1;
    }
  }

  const deduped = [];
  const seenEdges = new Set();
  for (const edge of data.edges || []) {
    const key = edgeKey(edge);
    if (seenEdges.has(key)) {
      removedDuplicateEdges += 1;
      continue;
    }
    seenEdges.add(key);
    deduped.push(edge);
  }
  data.edges = deduped;

  for (const node of conceptValues(data.concepts)) {
    if (!node || !node.metadata || !isClaimNode(node)) continue;
    const metadata = node.metadata;
    if (mapping.has(metadata.subject_id)) {
      metadata.original_subject_id = metadata.original_subject_id || metadata.subject_id;
      metadata.subject_id = mapping.get(metadata.subject_id);
      claimMetadataRewritten += 1;
    }
    if (mapping.has(metadata.object_id)) {
      metadata.original_object_id = metadata.original_object_id || metadata.object_id;
      metadata.object_id = mapping.get(metadata.object_id);
      claimMetadataRewritten += 1;
    }
  }

  for (const sourceId of mapping.keys()) {
    const sourceNode = getConcept(data, sourceId);
    const targetNode = getConcept(data, mapping.get(sourceId));
    if (sourceNode && targetNode) {
      targetNode.metadata = targetNode.metadata || {};
      targetNode.metadata.standardized_clm_sources = targetNode.metadata.standardized_clm_sources || [];
      targetNode.metadata.standardized_clm_sources.push({
        id: sourceNode.id,
        preferred_name: sourceNode.preferred_name,
        source_vocab: sourceNode.source_vocab,
        domain_tags: sourceNode.domain_tags || [],
      });
      setConcept(data, targetNode.id, targetNode);
    }
    deleteConcept(data, sourceId);
    removedConcepts += 1;
  }

  return {
    mappingsApplied: mapping.size,
    edgeEndpointsRewritten,
    claimMetadataRewritten,
    removedDuplicateEdges,
    removedConcepts,
  };
}

function readJsonOrJsonl(file) {
  if (!file) return [];
  const text = fs.readFileSync(file, 'utf8').trim();
  if (!text) return [];
  if (text.startsWith('[')) return JSON.parse(text);
  return text.split(/\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function loadManualMappings(file, data) {
  const rows = readJsonOrJsonl(file);
  const out = [];
  for (const row of rows) {
    const clmId = String(row.clm_id || row.source_id || '').trim();
    const targetId = String(row.target_id || '').trim();
    if (!clmId || !targetId) {
      throw new Error(`Manual mapping is missing clm_id or target_id: ${JSON.stringify(row)}`);
    }
    const sourceNode = getConcept(data, clmId);
    const targetNode = getConcept(data, targetId);
    if (!sourceNode) throw new Error(`Manual mapping source not found: ${clmId}`);
    if (!targetNode) throw new Error(`Manual mapping target not found: ${targetId}`);
    if (!isClmConcept(sourceNode)) throw new Error(`Manual mapping source is not CLM_CONCEPT: ${clmId}`);
    out.push({
      clm_id: clmId,
      clm_name: canonicalNameFromClm(sourceNode),
      clm_source_vocab: sourceNode.source_vocab || '',
      clm_domain_tags: sourceNode.domain_tags || [],
      target_id: targetId,
      target_name: targetNode.preferred_name || '',
      target_source_vocab: targetNode.source_vocab || '',
      target_domain_tags: targetNode.domain_tags || [],
      match_type: 'manual_curated',
      confidence: 0.99,
      rationale: String(row.rationale || '').trim(),
      curator_decision: String(row.decision || 'apply').trim(),
    });
  }
  return out;
}

function loadManualConcepts(file, data) {
  const rows = readJsonOrJsonl(file);
  const out = [];
  for (const row of rows) {
    const id = String(row.id || '').trim();
    const preferredName = String(row.preferred_name || '').trim();
    if (!id || !preferredName) {
      throw new Error(`Manual concept is missing id or preferred_name: ${JSON.stringify(row)}`);
    }
    if (isClmConcept({ id }) || isClaimNode({ id, domain_tags: row.domain_tags || [] })) {
      throw new Error(`Manual concept must be canonical, not CLM/claim: ${id}`);
    }
    const existing = getConcept(data, id) || {};
    const node = {
      id,
      preferred_name: preferredName,
      semantic_types: row.semantic_types || existing.semantic_types || [],
      domain_tags: row.domain_tags || existing.domain_tags || [],
      source_vocab: row.source_vocab || existing.source_vocab || 'NeuroClaw-ManualCanonical',
      definition: row.definition || existing.definition || '',
      aliases: row.aliases || existing.aliases || [],
      external_ids: row.external_ids || existing.external_ids || {},
      atlas_mapping: row.atlas_mapping ?? existing.atlas_mapping ?? null,
      metadata: {
        ...(existing.metadata || {}),
        ...(row.metadata || {}),
        manual_canonical: true,
      },
    };
    node.metadata.atom_types = inferAtomTypes(node);
    setConcept(data, id, node);
    out.push(node);
  }
  return out;
}

function stableAnchorId(name) {
  const normalized = String(name || '').trim().toLowerCase();
  const slug = normalized
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 96) || 'unnamed';
  let hash = 0;
  for (let i = 0; i < normalized.length; i += 1) {
    hash = ((hash << 5) - hash + normalized.charCodeAt(i)) | 0;
  }
  const suffix = Math.abs(hash).toString(16).padStart(8, '0').slice(0, 8);
  return `CLM_CONCEPT:${slug}_${suffix}`;
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
  if (type.includes('IMAGING') || type.includes('MRI') || type.includes('PET') || type.includes('EEG') || type.includes('MEG') || type.includes('ELECTROPHYSIOLOGY')) return ['imaging_feature', 'biomarker'];
  if (type.includes('BIOMARKER') || type.includes('MARKER') || type.includes('MOLECULAR') || type.includes('CELLULAR') || type.includes('MECHANISM') || type.includes('PATHOLOGY') || type.includes('PROCESS')) return ['biomarker'];
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

function findHighConfidenceTarget(name, index) {
  if (!name || isGeneric(name)) return '';
  const exactKey = String(name || '').trim().toLowerCase();
  const exact = dedupeCandidates(index.exact.get(exactKey) || []);
  if (exact.length === 1) return exact[0].id;
  const alias = dedupeCandidates(index.aliases.get(exactKey) || []);
  if (alias.length === 1) return alias[0].id;
  return '';
}

function ensureAnchor(data, name, type, role, scope, index) {
  const canonicalTarget = findHighConfidenceTarget(name, index);
  if (canonicalTarget) return canonicalTarget;
  const id = stableAnchorId(name);
  if (!getConcept(data, id)) {
    setConcept(data, id, conceptNode(
      id,
      name,
      typeDomains(type),
      'manual_claim_anchor_repair',
      {
        anchor_role: role,
        atom_type: type || '',
        paper_scope: scope || [],
        repair_source: 'malformed_unnamed_clm',
      }
    ));
  }
  return id;
}

function repairMalformedUnnamed(data, index) {
  const unnamedIds = new Set(
    conceptValues(data.concepts)
      .filter((node) => isClmConcept(node) && /^unnamed($| )/.test(normalizeName(canonicalNameFromClm(node))))
      .map((node) => node.id)
  );
  if (!unnamedIds.size) {
    return {
      unnamedNodes: 0,
      claimNodesRepaired: 0,
      semanticEdgesRewritten: 0,
      aboutEdgesRewritten: 0,
      anchorsCreatedOrReused: 0,
      unnamedNodesRemoved: 0,
    };
  }

  const claimRepairs = new Map();
  let anchorsCreatedOrReused = 0;
  for (const node of conceptValues(data.concepts)) {
    if (!isClaimNode(node) || !node.metadata) continue;
    const claim = node.metadata;
    const needsSubject = unnamedIds.has(claim.subject_id);
    const needsObject = unnamedIds.has(claim.object_id);
    if (!needsSubject && !needsObject) continue;
    const nestedSubject = claim.subject && typeof claim.subject === 'object' ? claim.subject : {};
    const nestedObject = claim.object && typeof claim.object === 'object' ? claim.object : {};
    const subjectName = String(claim.subject_name || nestedSubject.name || '').trim();
    const objectName = String(claim.object_name || nestedObject.name || '').trim();
    const subjectType = String(claim.subject_type || nestedSubject.type || claim.metadata?.subject_type || '').trim();
    const objectType = String(claim.object_type || nestedObject.type || claim.metadata?.object_type || '').trim();
    const scope = normalizePaperScope(claim.paper_scope || claim.metadata?.paper_scope || ['general']);
    if (!subjectName || !objectName) continue;

    const subjectId = needsSubject ? ensureAnchor(data, subjectName, subjectType, 'subject', scope, index) : claim.subject_id;
    const objectId = needsObject ? ensureAnchor(data, objectName, objectType, 'object', scope, index) : claim.object_id;
    anchorsCreatedOrReused += Number(needsSubject) + Number(needsObject);
    claim.original_subject_id = claim.original_subject_id || claim.subject_id;
    claim.original_object_id = claim.original_object_id || claim.object_id;
    claim.subject_id = subjectId;
    claim.object_id = objectId;
    claim.subject_name = subjectName;
    claim.object_name = objectName;
    claim.subject_type = subjectType;
    claim.object_type = objectType;
    claim.paper_scope = scope;
    node.preferred_name = `${subjectName} ${claim.predicate || ''} ${objectName}`.replace(/\s+/g, ' ').trim();
    claimRepairs.set(node.id, { subjectId, objectId });
  }

  let semanticEdgesRewritten = 0;
  let aboutEdgesRewritten = 0;
  for (const edge of data.edges || []) {
    const claimId = edge.metadata?.claim_id;
    const repair = claimRepairs.get(claimId);
    if (!repair) continue;
    if (edge.relation_type === 'about') {
      if (edge.metadata?.anchor_role === 'subject' && unnamedIds.has(edge.target_id)) {
        edge.target_id = repair.subjectId;
        aboutEdgesRewritten += 1;
      } else if (edge.metadata?.anchor_role === 'object' && unnamedIds.has(edge.target_id)) {
        edge.target_id = repair.objectId;
        aboutEdgesRewritten += 1;
      }
      continue;
    }
    if (unnamedIds.has(edge.source_id)) {
      edge.source_id = repair.subjectId;
      semanticEdgesRewritten += 1;
    }
    if (unnamedIds.has(edge.target_id)) {
      edge.target_id = repair.objectId;
      semanticEdgesRewritten += 1;
    }
  }

  const seenEdges = new Set();
  data.edges = (data.edges || []).filter((edge) => {
    const key = edgeKey(edge);
    if (seenEdges.has(key)) return false;
    seenEdges.add(key);
    return true;
  });

  let unnamedNodesRemoved = 0;
  const incident = new Set();
  for (const edge of data.edges || []) {
    if (unnamedIds.has(edge.source_id)) incident.add(edge.source_id);
    if (unnamedIds.has(edge.target_id)) incident.add(edge.target_id);
  }
  for (const id of unnamedIds) {
    if (!incident.has(id)) {
      deleteConcept(data, id);
      unnamedNodesRemoved += 1;
    }
  }

  return {
    unnamedNodes: unnamedIds.size,
    claimNodesRepaired: claimRepairs.size,
    semanticEdgesRewritten,
    aboutEdgesRewritten,
    anchorsCreatedOrReused,
    unnamedNodesRemoved,
  };
}

function clmRepresentativeScore(node) {
  const id = String(node.id || '');
  const source = String(node.source_vocab || '');
  const sourceRank = {
    claim_extraction: 6,
    replay_anchor_mint: 5,
    manual_claim_anchor: 4,
    manual_general_claim_anchor: 3,
    manual_claim_anchor_repair: 2,
  }[source] || 1;
  const cleanIdBonus = /_[0-9a-f]{8,12}$/i.test(id) ? 0 : 3;
  const typedPrefixPenalty = /^CLM_CONCEPT:[a-z_]+:/i.test(id) ? -1 : 0;
  return sourceRank + cleanIdBonus + typedPrefixPenalty;
}

function buildClmDedupeMappings(data) {
  const groups = new Map();
  for (const node of conceptValues(data.concepts)) {
    if (!isClmConcept(node)) continue;
    const name = canonicalNameFromClm(node);
    const normalized = normalizeName(name);
    if (!normalized || /^unnamed($| )/.test(normalized) || isGeneric(name)) continue;
    if (!groups.has(normalized)) groups.set(normalized, []);
    groups.get(normalized).push(node);
  }

  const mappings = [];
  const duplicateGroups = [];
  for (const [normalized, nodes] of groups.entries()) {
    if (nodes.length < 2) continue;
    const sorted = nodes.slice().sort((a, b) => {
      const scoreDiff = clmRepresentativeScore(b) - clmRepresentativeScore(a);
      if (scoreDiff) return scoreDiff;
      return String(a.id).length - String(b.id).length;
    });
    const target = sorted[0];
    duplicateGroups.push({
      normalized_name: normalized,
      target_id: target.id,
      target_name: canonicalNameFromClm(target),
      count: nodes.length,
      sources: nodes.map((node) => ({
        id: node.id,
        name: canonicalNameFromClm(node),
        source_vocab: node.source_vocab || '',
        domain_tags: node.domain_tags || [],
      })),
    });
    for (const source of sorted.slice(1)) {
      mappings.push({
        clm_id: source.id,
        clm_name: canonicalNameFromClm(source),
        clm_source_vocab: source.source_vocab || '',
        clm_domain_tags: source.domain_tags || [],
        target_id: target.id,
        target_name: canonicalNameFromClm(target),
        target_source_vocab: target.source_vocab || '',
        target_domain_tags: target.domain_tags || [],
        match_type: 'clm_normalized_duplicate',
        confidence: 0.95,
      });
    }
  }
  return { mappings, duplicateGroups };
}

function writeJsonl(file, rows) {
  fs.writeFileSync(file, rows.map((row) => JSON.stringify(row)).join('\n') + (rows.length ? '\n' : ''), 'utf8');
}

function main() {
  const args = parseArgs(process.argv);
  const data = JSON.parse(fs.readFileSync(args.graph, 'utf8'));
  const manualConcepts = loadManualConcepts(args.manualConcepts, data);
  const concepts = conceptValues(data.concepts);
  const clmNodes = concepts.filter(isClmConcept);
  const degree = degreeMap(data.edges || []);
  const index = buildCanonicalIndex(concepts, args.includeClaimExtractionCanonical);
  const manualMappings = loadManualMappings(args.manualMappings, data);

  const statusCounts = {};
  const highConfidence = [];
  const mediumConfidence = [];
  const ambiguous = [];
  const noMatch = [];
  const malformed = [];
  const generic = [];

  for (const node of clmNodes) {
    const decision = chooseCandidate(node, index);
    statusCounts[decision.status] = (statusCounts[decision.status] || 0) + 1;
    const base = {
      clm_id: node.id,
      clm_name: decision.name,
      clm_source_vocab: node.source_vocab || '',
      clm_domain_tags: node.domain_tags || [],
      degree: degree.get(node.id) || 0,
      status: decision.status,
    };
    if (decision.status === 'high_confidence') highConfidence.push({ ...base, ...decision.candidates[0] });
    else if (decision.status === 'medium_confidence') mediumConfidence.push({ ...base, ...decision.candidates[0] });
    else if (decision.status === 'ambiguous') {
      ambiguous.push({ ...base, candidates: decision.candidates.slice(0, 10) });
    } else if (decision.status === 'malformed_unnamed') malformed.push(base);
    else if (decision.status === 'generic_no_auto') generic.push(base);
    else noMatch.push(base);
  }

  const outDir = args.outputDir;
  fs.mkdirSync(outDir, { recursive: true });
  const clmDedupe = buildClmDedupeMappings(data);

  const plan = {
    graph: args.graph,
    apply: args.apply,
    applyTier: args.applyTier,
    includeClaimExtractionCanonical: args.includeClaimExtractionCanonical,
    repairMalformedUnnamed: args.repairMalformedUnnamed,
    dedupeClmNormalized: args.dedupeClmNormalized,
    manualConceptsFile: args.manualConcepts || null,
    manualMappingsFile: args.manualMappings || null,
    totalConcepts: concepts.length,
    clmConcepts: clmNodes.length,
    canonicalNodesIndexed: index.canonicalNodes,
    statusCounts,
    highConfidenceMappings: highConfidence.length,
    mediumConfidenceMappings: mediumConfidence.length,
    ambiguousMappings: ambiguous.length,
    manualConcepts: manualConcepts.length,
    manualMappings: manualMappings.length,
    malformedUnnamed: malformed.length,
    genericNoAuto: generic.length,
    noMatch: noMatch.length,
    clmDuplicateGroups: clmDedupe.duplicateGroups.length,
    clmDuplicateMappings: clmDedupe.mappings.length,
    outputDir: outDir,
  };

  const applyRows = args.applyTier === 'medium'
    ? highConfidence.concat(mediumConfidence)
    : highConfidence;

  let applySummary = null;
  if (args.apply) {
    applySummary = applyMappings(
      data,
      args.dedupeClmNormalized
        ? applyRows.concat(manualMappings, clmDedupe.mappings)
        : applyRows.concat(manualMappings)
    );
    if (args.repairMalformedUnnamed) {
      applySummary.malformedUnnamedRepair = repairMalformedUnnamed(data, index);
    }
    data.metadata = data.metadata || {};
    data.metadata.stats = computeStats(data);
    data.metadata.clm_concept_standardization = {
      ...plan,
      appliedAt: new Date().toISOString(),
      applySummary,
    };
    const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, '').replace('T', '_');
    const backup = `${args.graph}.bak_clm_standardize_${stamp}`;
    fs.copyFileSync(args.graph, backup);
    fs.writeFileSync(`${args.graph}.tmp`, JSON.stringify(data) + '\n', 'utf8');
    fs.renameSync(`${args.graph}.tmp`, args.graph);
    applySummary.backup = backup;
  }

  fs.writeFileSync(path.join(outDir, 'clm_standardization_summary.json'), JSON.stringify({ ...plan, applySummary }, null, 2) + '\n', 'utf8');
  writeJsonl(path.join(outDir, 'high_confidence_mappings.jsonl'), highConfidence);
  writeJsonl(path.join(outDir, 'medium_confidence_mappings.jsonl'), mediumConfidence);
  writeJsonl(path.join(outDir, 'ambiguous_mappings.jsonl'), ambiguous);
  writeJsonl(path.join(outDir, 'manual_concepts.jsonl'), manualConcepts);
  writeJsonl(path.join(outDir, 'manual_mappings.jsonl'), manualMappings);
  writeJsonl(path.join(outDir, 'malformed_unnamed.jsonl'), malformed);
  writeJsonl(path.join(outDir, 'generic_no_auto.jsonl'), generic);
  writeJsonl(path.join(outDir, 'no_match_examples.jsonl'), noMatch.slice(0, Math.max(0, args.maxExamples)));
  writeJsonl(path.join(outDir, 'clm_normalized_duplicate_mappings.jsonl'), clmDedupe.mappings);
  writeJsonl(path.join(outDir, 'clm_normalized_duplicate_groups.jsonl'), clmDedupe.duplicateGroups);

  console.log(JSON.stringify({ ...plan, applySummary }, null, 2));
}

main();
