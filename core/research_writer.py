"""Approval-gated, create-only research artifacts; never a completion verdict."""

import json
from pathlib import Path


WRITE_TOOL_NAME = 'write_research_file'
WRITE_TOOL = {'type': 'function', 'function': {
    'name': WRITE_TOOL_NAME,
    'description': 'Create a new UTF-8 research artifact without shell quoting. Requires write approval. '
                   'Never overwrites existing files: choose a revision path. For hypotheses supply a JSON list '
                   'of hypothesis, rationale, source_ids and prediction/limitations. In idea mode preserve '
                   'the validated chain, kg_triples, evidence_queries and revisions from generate_idea_hypotheses. For deliverable '
                   'supply a Markdown report such as IDEA.md. A saved draft is not accepted research.',
    'parameters': {'type': 'object', 'properties': {
        'path': {'type': 'string'}, 'kind': {'type': 'string', 'enum': ['hypotheses', 'deliverable']},
        'content': {'type': 'string', 'description': 'Actual complete file content, not a plan to write it.'},
    }, 'required': ['path', 'kind', 'content']},
}}


def write_research_file(workspace, arguments, cancel, *, require_chain=False):
    try:
        raw, kind, content = (arguments.get(key) for key in ('path', 'kind', 'content'))
        if not isinstance(raw, str) or not raw or not isinstance(content, str) or not content.strip() or len(content.encode('utf-8')) > 512 * 1024:
            raise ValueError('Provide a path and nonempty content of at most 512 KiB.')
        workspace = Path(workspace).resolve()
        target = (workspace / raw).resolve()
        relative = target.relative_to(workspace)
        if any(part.startswith('.') or part.lower() in {'skills', 'core', 'desktop', 'configs', 'node_modules', 'neurooracle', 'neurooracledata'} for part in relative.parts):
            raise ValueError('Use a research output path, not source code, protected data or runtime storage.')
        if target.name.upper() in {'AGENTS.MD', 'SOUL.MD', 'USER.MD', 'MEMORY.MD', 'SKILL.MD', 'README.MD'} or ':' in str(relative):
            raise ValueError('Instructions and configuration are not research output.')
        if kind == 'hypotheses':
            if target.suffix.lower() != '.json':
                raise ValueError('Candidate hypotheses must use a JSON file.')
            rows = json.loads(content)
            if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
                raise ValueError('Provide 1–100 candidates; for no defensible candidate write an evidence-gap report instead.')
            for row in rows:
                if not isinstance(row, dict) or not all(isinstance(row.get(key), str) and row[key].strip() for key in ('hypothesis', 'rationale')):
                    raise ValueError('Each candidate needs a hypothesis and source-based rationale.')
                sources = row.get('source_ids')
                if not isinstance(sources, list) or not sources or not all(isinstance(source, str) and source.strip() for source in sources):
                    raise ValueError('Each candidate needs source_ids; citation validity still requires review.')
                if require_chain or 'chain' in row:
                    from core.idea_hypotheses import validate_candidate_shape
                    validate_candidate_shape(row)
        elif kind != 'deliverable' or target.suffix.lower() != '.md':
            raise ValueError('Deliverables must be Markdown; hypotheses must be JSON.')
        if cancel.is_set():
            return {'success': False, 'executed': False, 'error_type': 'cancelled'}
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as handle:
            handle.write(content)
        return {'success': True, 'path': str(relative), 'kind': kind, 'bytes': target.stat().st_size,
                'message': f'Saved draft {relative}. Read it back and verify sources before requesting acceptance.',
                'scientifically_validated': False}
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        return {'success': False, 'executed': False, 'error_type': 'research_write_failed', 'error': str(exc)}
