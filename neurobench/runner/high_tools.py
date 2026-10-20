"""Same public task context for all arms; opt-in domain consumers for B/D."""
import ast
import json
from pathlib import Path

from core.harness_validation import register_validation_tools
from core.mcp_stdio import HarnessHost
from core.tools_impl import SKILL_TOOL_NAMES, register_core_tools
from core import tool_registry

COMMON = [
    {'name': 'read_context_file', 'description': 'Read a public task/reference file. Use context/skills/<name>/SKILL.md for the same method library available to native CLIs.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0}}, 'required': ['path']}},
    {'name': 'check_python', 'description': 'Parse proposed Python code with ast.parse without executing it. This is a common static-check primitive.', 'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}}, 'required': ['code']}},
    {'name': 'save_proposed_file', 'description': 'Save proposed code or final answer in proposals/. This does not execute the workflow.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}}, 'required': ['path', 'content']}},
]
NR_NAMES = (*SKILL_TOOL_NAMES, 'describe_task_contract', 'validate_proposal')


class TaskTools:
    def __init__(self, workspace, task_text, *, enabled):
        self.workspace = Path(workspace).resolve()
        self.task_text = task_text
        self.enabled = enabled
        register_core_tools()
        if enabled:
            register_validation_tools()
        self.host = HarnessHost(self.workspace, skills_dir=self.workspace / 'context/skills')
        self.host._evaluation_task_text = task_text

    def schemas(self):
        rows = list(COMMON)
        if self.enabled:
            rows += [s['function'] for s in tool_registry.schemas(include=NR_NAMES)]
        return rows

    def call(self, name, args):
        if not isinstance(args, dict):
            return {'success': False, 'error': 'arguments_not_object'}
        if self.enabled and name in NR_NAMES:
            return tool_registry.dispatch(self.host, name, args, self.workspace)
        if name == 'read_context_file':
            path = (self.workspace / str(args.get('path', ''))).resolve()
            if not path.is_relative_to(self.workspace / 'context') or not path.is_file():
                return {'success': False, 'error': 'path_outside_public_context_or_missing'}
            text = path.read_text(encoding='utf-8')
            offset = int(args.get('offset', 0))
            if offset < 0:
                return {'success': False, 'error': 'negative_offset'}
            return {'success': True, 'path': str(path.relative_to(self.workspace)), 'content': text[offset:offset + 20000],
                    'next_offset': offset + 20000 if offset + 20000 < len(text) else None, 'workflow_executed': False}
        if name == 'check_python':
            code = args.get('code', '')
            if not isinstance(code, str) or len(code) > 500000:
                return {'success': False, 'error': 'code_not_text_or_too_large'}
            try:
                ast.parse(code)
                return {'success': True, 'syntax': 'parse_pass', 'workflow_executed': False}
            except SyntaxError as e:
                return {'success': True, 'syntax': 'parse_fail', 'line': e.lineno, 'message': e.msg, 'workflow_executed': False}
        if name == 'save_proposed_file':
            path = (self.workspace / 'proposals' / str(args.get('path', ''))).resolve()
            if not path.is_relative_to(self.workspace / 'proposals'):
                return {'success': False, 'error': 'path_outside_proposals'}
            content = args.get('content', '')
            if not isinstance(content, str) or len(content) > 500000:
                return {'success': False, 'error': 'content_not_text_or_too_large'}
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                history = self.workspace / 'proposal_history'
                history.mkdir(exist_ok=True)
                from neurobench.runner.artifacts import file_sha256, freeze_bytes
                freeze_bytes(history / (file_sha256(path) + '.txt'), path.read_bytes())
            path.write_text(content, encoding='utf-8')
            return {'success': True, 'path': str(path.relative_to(self.workspace)), 'workflow_executed': False}
        return {'success': False, 'error': 'tool_not_exposed_in_this_arm'}


BASE_PROMPT = '''Generate a complete proposed solution for the bound NeuroBench task.
This is a proposal-quality experiment. You may read the public context, save proposed files, and perform local static syntax/parameter checks. Do not execute the proposed research workflow, install environments, run training, access real datasets or fetch network resources. A proposal or static check does not mean the workflow ran.
All conditions receive the same task text and generic method library in context/skills. context/skill_catalog.json lists the exact method names and descriptions and can be read with a generic file tool. Missing actual inputs must be stated and checked by the proposed code; do not substitute synthetic research outcomes.
Return exactly two top-level sections: ## Solution Thinking then ## Commands Or Code. Preserve the actual requested stages, inputs, spaces and outputs. Include complete proposed commands or code and failure handling. No preamble or additional level-2 section.
'''

NR_PROMPT = '''NeuroRuntime provides compile_task, search_skills, read_skill_section, read_skill, describe_task_contract and validate_proposal. Organize public requirements and observed input metadata with compile_task when useful. Load relevant method sections or full guidance, adapt it, and check your draft with validate_proposal before final delivery. Correct detected errors within the task budget; at most one additional diagnostic feedback repair after a final draft. Review hints are not proof of failure or correctness. Reading skills and checking code do not execute the workflow.
'''


def prompt(task, enabled):
    return BASE_PROMPT + (NR_PROMPT if enabled else '') + '\n--- TASK ---\n' + task
