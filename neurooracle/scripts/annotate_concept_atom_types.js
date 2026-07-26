const fs = require('fs');
const path = require('path');

const repo = path.resolve(__dirname, '..', '..');
const defaultGraph = path.join(repo, 'neurooracle', 'data', 'full_v2', 'knowledge_graph.json');

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

function parseArgs(argv) {
  const args = {
    graph: defaultGraph,
    output: '',
    apply: false,
    report: path.join(repo, 'neurooracle', 'data', 'reports', 'atom_type_annotation_summary.json'),
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--graph') args.graph = argv[++i];
    else if (arg === '--output') args.output = argv[++i];
    else if (arg === '--report') args.report = argv[++i];
    else if (arg === '--apply') args.apply = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function conceptValues(concepts) {
  return Array.isArray(concepts) ? concepts : Object.values(concepts || {});
}

function add(out, atom) {
  if (ATOM_ORDER.includes(atom) && !out.includes(atom)) out.push(atom);
}

function normalizeAtomTypes(values) {
  const raw = Array.isArray(values) ? values : values ? [values] : [];
  const out = [];
  for (const item of raw) {
    let value = String(item || '').trim().toLowerCase();
    if (!value) continue;
    if (value === 'imaging') value = 'imaging_marker';
    if (value === 'gene') value = 'gene_target';
    if (value === 'individual') value = 'individual_data';
    add(out, value);
  }
  return ATOM_ORDER.filter((atom) => out.includes(atom));
}

function inferAtomTypes(node) {
  const out = [];
  const id = String(node.id || '');
  const metadata = node.metadata && typeof node.metadata === 'object' ? node.metadata : {};

  if (id.startsWith('NCL_IMAGING:')) return ['imaging_marker'];
  if (id.startsWith('NCL_OUTCOME:')) return ['outcome'];
  if (id.startsWith('NCL_DISEASE:')) return ['disease'];
  if (id.startsWith('NCL_DRUG:')) return ['drug'];
  if (id.startsWith('NCL_GENE_TARGET:')) return ['gene_target'];
  if (id.startsWith('NCL_INDIVIDUAL:') || id.startsWith('NCL_COVARIATE:')) return ['individual_data'];
  if (id.startsWith('NCL_BIOMARKER:') || id.startsWith('NCL_METHOD:')) return ['imaging_marker'];

  for (const atom of normalizeAtomTypes(metadata.atom_types || metadata.atom_type)) {
    add(out, atom);
  }

  for (const tag of node.domain_tags || []) {
    for (const atom of DOMAIN_TO_ATOMS[String(tag || '').trim()] || []) {
      add(out, atom);
    }
  }

  if (id.startsWith('IM:') || id.startsWith('NCL_IMAGING:')) add(out, 'imaging_marker');
  if (id.startsWith('GM:') || id.startsWith('GENESET:') || id.startsWith('NCL_GENE_TARGET:')) add(out, 'gene_target');
  if (id.startsWith('OUTCOME:') || id.startsWith('NCL_OUTCOME:')) add(out, 'outcome');
  if (id.startsWith('NCL_DISEASE:')) add(out, 'disease');
  if (id.startsWith('NCL_DRUG:')) add(out, 'drug');
  if (id.startsWith('NCL_INDIVIDUAL:') || id.startsWith('NCL_COVARIATE:')) add(out, 'individual_data');
  if (id.startsWith('COGAT_TASK:') || id.startsWith('COGAT_CONCEPT:')) add(out, 'cognitive_task');

  return ATOM_ORDER.filter((atom) => out.includes(atom));
}

function updateStats(stats, key, inc = 1) {
  stats[key] = (stats[key] || 0) + inc;
}

function computeStats(data) {
  const concepts = conceptValues(data.concepts);
  const stats = {
    n_concepts: concepts.length,
    n_edges: (data.edges || []).length,
    domains: {},
    sources: {},
    relations: {},
  };
  for (const node of concepts) {
    for (const tag of node.domain_tags || ['unknown']) updateStats(stats.domains, tag);
    updateStats(stats.sources, node.source_vocab || 'unknown');
  }
  for (const edge of data.edges || []) {
    updateStats(stats.relations, edge.relation_type || 'unknown');
  }
  return stats;
}

function main() {
  const args = parseArgs(process.argv);
  const data = JSON.parse(fs.readFileSync(args.graph, 'utf8'));
  const concepts = conceptValues(data.concepts);

  const byAtom = Object.fromEntries(ATOM_ORDER.map((atom) => [atom, 0]));
  const bySource = {};
  const emptyBySource = {};
  let createdMetadata = 0;
  let changed = 0;
  let empty = 0;

  for (const node of concepts) {
    if (!node.metadata || typeof node.metadata !== 'object') {
      node.metadata = {};
      createdMetadata += 1;
    }
    const before = JSON.stringify(normalizeAtomTypes(node.metadata.atom_types || []));
    const atoms = inferAtomTypes(node);
    node.metadata.atom_types = atoms;
    const after = JSON.stringify(atoms);
    if (before !== after) changed += 1;
    if (!atoms.length) {
      empty += 1;
      updateStats(emptyBySource, node.source_vocab || 'unknown');
    }
    updateStats(bySource, node.source_vocab || 'unknown');
    for (const atom of atoms) updateStats(byAtom, atom);
  }

  data.metadata = data.metadata || {};
  data.metadata.stats = computeStats(data);
  data.metadata.atom_type_annotation = {
    annotated_at: new Date().toISOString(),
    source: path.basename(__filename),
    atom_order: ATOM_ORDER,
    domain_to_atoms: DOMAIN_TO_ATOMS,
    changed,
    empty,
  };

  const summary = {
    graph: args.graph,
    apply: args.apply,
    output: args.output || args.graph,
    concepts: concepts.length,
    changed,
    createdMetadata,
    emptyAtomTypes: empty,
    byAtom,
    bySource,
    emptyBySource,
  };

  fs.mkdirSync(path.dirname(args.report), { recursive: true });
  fs.writeFileSync(args.report, JSON.stringify(summary, null, 2) + '\n', 'utf8');

  if (args.apply) {
    const output = args.output || args.graph;
    const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, '').replace('T', '_');
    if (output === args.graph) {
      const backup = `${args.graph}.bak_atom_types_${stamp}`;
      fs.copyFileSync(args.graph, backup);
      summary.backup = backup;
    }
    fs.writeFileSync(`${output}.tmp`, JSON.stringify(data) + '\n', 'utf8');
    fs.renameSync(`${output}.tmp`, output);
    fs.writeFileSync(args.report, JSON.stringify(summary, null, 2) + '\n', 'utf8');
  }

  console.log(JSON.stringify(summary, null, 2));
}

main();
