"""Conservative probe recognition and Windows multiline Python transport."""

import re
import ast
from pathlib import Path


def is_readonly_probe(command: str) -> bool:
    command = re.sub(r'(?i)\b2>\s*nul\b', '', command)
    if re.search(r'[<>\r\n%^`()]|\$\(', command):
        return False
    parts = re.split(r'&&|\|\||[|;&]', command)
    for part in parts:
        part = part.strip()
        if not part:
            return False
        if re.fullmatch(r'(?:python(?:\.exe)?|"[^"]*python\.exe")\s+(?:-V|--version)', part, re.I):
            continue
        if not re.match(r'^(?:cd|pwd|dir|ls|echo|where|which|findstr|grep|head|tail|type|cat|ver)(?:\s|$)', part, re.I):
            return False
        if re.search(r'\b(?:--exec|--pre|--files-with-matches)\b', part, re.I):
            return False
    return bool(parts)


def multiline_python(command: str, cwd: Path):
    match = re.fullmatch(
        r'\s*(?:cd\s+(?:/d\s+)?(?P<cwd>"[^"\r\n]+"|[^&\r\n]+?)\s*&&\s*)?'
        r'(?P<exe>"[^"\r\n]+"|[^\s"&|]+)\s+-c\s+"(?P<code>.*)"\s*', command, re.I | re.S)
    if not match or '"' in match['code'] or Path(match['exe'].strip('"')).name.lower() not in {'python', 'python.exe', 'python3', 'python3.exe'}:
        raise ValueError('Unsupported multiline shell command. Write a UTF-8 script and execute it, or use read_workspace_file for text/JSON. No partial command was executed.')
    workdir = (cwd / match['cwd'].strip().strip('"')).resolve() if match['cwd'] else cwd
    return [match['exe'].strip('"'), '-c', match['code']], workdir


def is_python_listing_probe(command: str) -> bool:
    try:
        arguments, _ = multiline_python(command, Path.cwd())
        tree = ast.parse(arguments[-1])
    except (ValueError, SyntaxError, OSError):
        return False
    allowed_calls = {'print', 'len', 'sorted', 'list', 'str', 'glob.glob', 'glob.iglob',
                     'os.listdir', 'os.walk', 'os.path.getsize', 'os.path.basename', 'os.path.join'}
    has_listing = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name not in allowed_calls:
                return False
            has_listing = has_listing or name in {'glob.glob', 'glob.iglob', 'os.listdir', 'os.walk'}
        elif isinstance(node, ast.Import):
            if any(alias.name not in {'glob', 'os', 'json'} or alias.asname for alias in node.names):
                return False
        elif isinstance(node, (ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Lambda, ast.With, ast.Try, ast.Delete, ast.Await)):
            return False
    return has_listing
