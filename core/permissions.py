"""Cooperative tool approval policy, not an operating-system sandbox."""
from pathlib import Path


PERMISSION_MODES = frozenset({'ask', 'risk', 'never', 'read_only'})


def tool_permission(mode: str, name: str, arguments: dict, workspace: Path) -> str:
    if mode not in PERMISSION_MODES or name == 'spawn_subagent':
        return 'deny'
    if name == 'finish_autoresearch':
        return 'allow'
    if name in {'search_skills', 'read_skill'}:
        return 'allow'  # Provider confines these reads to its named skill catalog.
    if name in {'generate_idea_hypotheses', 'rank_idea_hypotheses'}:
        return 'allow'  # Current accepted graph reads only; no model calls or writes.
    if name == 'record_research_note':
        try:
            target = (workspace / str(arguments.get('source_path') or '.')).resolve()
            return 'allow' if target.is_relative_to(workspace.resolve()) else 'deny'
        except (OSError, ValueError):
            return 'deny'
    if name in {'read_workspace_file', 'inspect_local_path', 'inspect_research_progress'}:
        try:
            target = (workspace / str(arguments.get('path') or '.')).resolve()
            return 'allow' if target.is_relative_to(workspace.resolve()) else 'deny'
        except (OSError, ValueError):
            return 'deny'
    if name in {'search_pubmed', 'write_research_file'}:
        if mode == 'read_only':
            return 'deny'
        return 'allow' if mode == 'never' else 'ask'
    if name != 'run_shell_command' or mode == 'read_only':
        return 'deny'
    if mode == 'never':
        return 'allow'
    if mode == 'risk':
        command = str(arguments.get('command') or '').strip()
        if command in {'pwd', 'cd', 'ver'}:
            return 'allow'
    return 'ask'
