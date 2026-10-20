"""Opt-in proposal checks. No proposal is executed or scientifically approved.

Uses the existing ToolSpec/dispatch pipeline. Diagnostics identify demonstrable
syntax errors separately from incomplete static coverage and domain review hints.
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

VERSION = 'neuroruntime-proposal-checks-20261013-r1'
TOOL_NAMES = ('describe_task_contract', 'validate_proposal')
MAX_TEXT = 500_000
REQUIRED_HEADINGS = ('Solution Thinking', 'Commands Or Code')


def digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def task_contract(task_text: str) -> dict:
    """Task-derived checklist, never a per-case answer or hidden grader oracle."""
    if not isinstance(task_text, str) or not task_text.strip() or len(task_text) > MAX_TEXT:
        raise ValueError('task_text_missing_or_too_large')
    low = task_text.casefold()
    requirements = [
        {'id': 'input_identity', 'rule': 'Name required inputs, subject IDs and output paths; validate inputs before acting.'},
        {'id': 'real_inputs', 'rule': 'Missing inputs stay missing. Simulated/test data must not replace requested real research results.'},
        {'id': 'completion_evidence', 'rule': 'Separate proposed, launched, running, completed and verified. Exit/terminal evidence and fresh outputs are required for execution claims.'},
        {'id': 'delivery', 'rule': 'Deliver complete proposed code/commands and explicit verification conditions, not print-only workflow placeholders.'},
    ]
    domains = []
    for domain, terms, rules in (
        ('environment', ('conda', 'environment', 'venv', '环境'), [
            ('environment_identity', 'Verify the actual environment/prefix and executable ownership; handle command failures before recording success.')]),
        ('freesurfer', ('freesurfer', 'recon-all'), [
            ('recon_status', 'Choose fresh import versus continuation from actual subject state. Capture real recon-all exit/launch status; stale files alone do not prove this run completed.')]),
        ('tracking', ('probtrack', 'bedpost', 'tractography', 'tracking', '纤维追踪'), [
            ('sampling_set', 'Validate complete theta/phi/f sampling sets and the exact bedpostx prefix and brain mask.'),
            ('coordinate_space', 'Declare seed/target/reference spaces and required transforms; do not assume voxel, world, native and standard spaces coincide.')]),
        ('bids', ('bids',), [('bids_identity', 'Preserve subject/session/modality identity and required sidecars; validate intended output layout.')]),
        ('learning', ('train', 'cross-validation', 'split', '训练', '交叉验证'), [
            ('split_independence', 'Keep subjects/sites/families separate across folds; fit transforms only within training data and save split identity.')]),
        ('statistics', ('statistic', 'p-value', '统计', 'mediation'), [
            ('statistical_design', 'Name the independent observation unit, covariates, multiple-test correction and missing-data handling.')]),
    ):
        if any(t in low for t in terms):
            domains.append(domain)
            requirements += [{'id': k, 'rule': v} for k, v in rules]
    return {'version': VERSION, 'task_sha256': digest(task_text), 'domains': domains,
            'requirements': requirements, 'workflow_executed': False,
            'scope': 'Task-derived public checklist; not hidden expected answers or scientific acceptance.'}


def code_blocks(text: str) -> tuple[list[dict], list[dict]]:
    """CommonMark-style matching fences; preserve longer fences around backticks."""
    blocks, errors = [], []
    opened = None
    body = []
    for number, line in enumerate(text.split('\n'), 1):
        probe = line[:-1] if line.endswith('\r') else line
        if opened is None:
            match = re.match(r'^ {0,3}(`{3,}|~{3,})([^\r\n]*)$', probe)
            if match:
                fence, info = match.groups()
                if fence[0] == '`' and '`' in info:
                    continue
                opened = (fence[0], len(fence), info.strip().split(' ', 1)[0].casefold(), number)
                body = []
        else:
            char, length, lang, start = opened
            if re.fullmatch(r' {0,3}' + re.escape(char) + '{' + str(length) + r',}[ \t]*', probe):
                blocks.append({'language': lang, 'code': '\n'.join(body), 'start_line': start + 1})
                opened = None
            else:
                body.append(line)
    if opened:
        errors.append({'code': 'unclosed_code_fence', 'severity': 'error', 'line': opened[3],
                       'message': 'The final code fence is unfinished; full-script syntax cannot be verified.'})
    return blocks, errors


def validate_proposal(task_text: str, proposal: str) -> dict:
    if not isinstance(proposal, str) or len(proposal) > MAX_TEXT:
        raise ValueError('proposal_not_text_or_too_large')
    contract = task_contract(task_text)
    blocks, findings = code_blocks(proposal)
    headings = [m.group(1).strip() for m in re.finditer(r'^##[ \t]+([^\r\n]+)[ \t]*$', proposal, re.M)]
    if headings != list(REQUIRED_HEADINGS):
        findings.append({'code': 'answer_sections', 'severity': 'error',
                         'message': 'Use exactly ## Solution Thinking then ## Commands Or Code.'})
    syntax = []
    for index, block in enumerate(blocks):
        lang, code = block['language'], block['code']
        if lang in ('python', 'py', 'python3'):
            try:
                ast.parse(code)
                syntax.append({'block': index, 'language': lang, 'status': 'parse_pass'})
            except SyntaxError as exc:
                findings.append({'code': 'python_syntax', 'severity': 'error', 'block': index,
                                 'line': block['start_line'] + (exc.lineno or 1) - 1, 'message': exc.msg})
                syntax.append({'block': index, 'language': lang, 'status': 'parse_fail'})
        else:
            syntax.append({'block': index, 'language': lang, 'status': 'not_parsed'})
    if not blocks:
        findings.append({'code': 'code_coverage', 'severity': 'review',
                         'message': 'No complete fenced code block was found. Plain commands and prose were not syntax validated.'})
    low = proposal.casefold()
    code = '\n'.join(b['code'] for b in blocks)
    def review(key, message):
        findings.append({'code': key, 'severity': 'review', 'message': message})
    if 'freesurfer' in contract['domains'] and 'recon-all' in low:
        if not re.search(r'(check\s*=\s*True|returncode|exit[_ ]?(?:status|code)|set\s+-[a-z]*e|\$\?|wait\b)', code, re.I):
            review('recon_exit_evidence', 'No explicit exit/failure handling detected. Verify launched/completed status from the actual process, not output-file existence.')
        if not re.search(r'(mtime|stat\(|start(?:ed)?[_ ]?(?:at|time)|fresh|本次|stale|newer)', proposal, re.I):
            review('fresh_output_evidence', 'No fresh-output/run-identity check detected. Old FreeSurfer outputs cannot establish current-run completion.')
        calls = re.search(r'(subprocess\.(?:run|Popen|check_call|check_output)|os\.system|^\s*recon-all\b)', code, re.M)
        if not calls and re.search(r'print\([^\n]*recon-all', code):
            findings.append({'code': 'print_only_workflow', 'severity': 'error',
                             'message': 'Code prints the recon-all command without a detected launch. A proposed launcher must implement a real command call.'})
    if 'tracking' in contract['domains']:
        for token, field in (('merged_th', 'theta'), ('merged_ph', 'phi'), ('merged_f', 'f')):
            if token not in low and not re.search(r'(sampling.set|samples?.*valid|validate.*samples?|采样)', proposal, re.I):
                review('sampling_' + field, 'Complete ' + field + ' sample-set validation was not detected; inspect all requested sampling components.')
        if not re.search(r'(transform|coordinate|affine|space|坐标|空间|xfm)', proposal, re.I):
            review('tracking_space', 'Seed/target/reference coordinate-space declaration or transformation is not detectable.')
    if 'environment' in contract['domains'] and re.search(r'(shutil\.which|which\b)', code):
        if not re.search(r'(is_relative_to|commonpath|relative_to|startswith\(.*prefix|prefix.*(?:assert|check)|resolve\(\).*prefix)', code, re.I | re.S):
            review('executable_prefix', 'Finding an executable is insufficient to prove it belongs to the requested environment; verify resolved prefix ownership.')
    if 'learning' in contract['domains'] and re.search(r'(fit_transform\(|\.fit\()', code):
        first_split = re.search(r'(train_test_split|GroupKFold|StratifiedGroupKFold|split\()', code)
        first_fit = re.search(r'(fit_transform\(|\.fit\()', code)
        if first_split and first_fit and first_fit.start() < first_split.start():
            review('fit_before_split', 'A fitted transform appears before a split in displayed code. Review training-only fitting and data leakage; static order alone is not a proof.')
    error_count = sum(f['severity'] == 'error' for f in findings)
    return {'success': True, 'version': VERSION, 'task_sha256': contract['task_sha256'],
            'proposal_sha256': digest(proposal), 'contract': contract, 'findings': findings,
            'syntax_blocks': syntax, 'blocking_error_count': error_count,
            'status': 'static_errors_detected' if error_count else 'no_detected_static_errors',
            'workflow_executed': False, 'scientific_correctness_verified': False,
            'coverage_limit': 'Python AST/fence/section checks plus conservative domain review hints; no shell execution, dataset access or full correctness proof.'}


def _task(session, args):
    bound = getattr(session, '_evaluation_task_text', '')
    if bound:
        return bound
    path = getattr(session, '_evaluation_task_file', None)
    if path:
        text = Path(path).read_text(encoding='utf-8')
        expected = getattr(session, '_evaluation_task_sha256', '')
        if expected and digest(text) != expected:
            raise ValueError('bound_task_text_changed')
        return text
    return args.get('task_text', '')


def _describe(session: Any, args: Mapping, workspace: Path):
    result = {'success': True, **task_contract(_task(session, args))}
    session._tool_events.append({'tool': TOOL_NAMES[0], 'executed': True, 'success': True, 'result': result})
    return result


def _validate(session: Any, args: Mapping, workspace: Path):
    result = validate_proposal(_task(session, args), args.get('proposal', ''))
    session._tool_events.append({'tool': TOOL_NAMES[1], 'executed': True, 'success': True, 'result': result})
    return result


def register_validation_tools():
    from core.tool_registry import ToolSpec, register
    optional_task = {'type': 'string', 'description': 'Current task text; omit when the host binds an immutable task.'}
    register(ToolSpec(TOOL_NAMES[0], 'Describe input, output and domain-specific completion checks for the bound task; no workflow is executed.',
                      {'type': 'object', 'properties': {'task_text': optional_task}}, _describe))
    register(ToolSpec(TOOL_NAMES[1], 'Inspect a complete proposed final answer for Python/fence/section errors and domain review hints. Use diagnostics for one bounded repair; this is not execution or scientific acceptance.',
                      {'type': 'object', 'properties': {'task_text': optional_task, 'proposal': {'type': 'string'}}, 'required': ['proposal']}, _validate))
