#!/usr/bin/env python3
"""Validate that skill dependencies form a DAG (no circular dependencies)."""
import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None


def parse_front_matter(skill_md: Path) -> Optional[Dict]:
    """Parse YAML front-matter from a SKILL.md file."""
    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    end = text.find("---", 3)
    if end == -1:
        return None
    raw = text[3:end]
    if yaml:
        try:
            return yaml.safe_load(raw) or {}
        except Exception:
            return None
    # Fallback: manual parse for dependencies only
    meta = {}
    in_deps = False
    deps = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("name:"):
            meta["name"] = stripped.split(":", 1)[1].strip().strip('"').strip("'")
        elif stripped.startswith("dependencies:"):
            in_deps = True
        elif in_deps:
            if stripped.startswith("- "):
                deps.append(stripped[2:].strip().strip('"').strip("'"))
            elif stripped and not stripped.startswith("#"):
                in_deps = False
    if deps:
        meta["dependencies"] = deps
    return meta


def load_skill_graph(skills_dir: Path) -> Tuple[Dict[str, List[str]], Dict[str, Dict]]:
    """Load all skills and their dependencies into a graph."""
    graph: Dict[str, List[str]] = {}
    metadata: Dict[str, Dict] = {}

    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        meta = parse_front_matter(skill_md)
        if not meta or "name" not in meta:
            continue
        name = meta["name"]
        deps = meta.get("dependencies", [])
        if isinstance(deps, str):
            deps = [d.strip() for d in deps.split(",")]
        graph[name] = list(deps)
        metadata[name] = meta

    return graph, metadata


def find_cycles(graph: Dict[str, List[str]]) -> List[List[str]]:
    """Find all cycles in the dependency graph using DFS."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph}
    parent = {}
    cycles = []

    def dfs(u: str, path: List[str]) -> None:
        color[u] = GRAY
        path.append(u)
        for v in graph.get(u, []):
            if v not in color:
                continue
            if color[v] == GRAY:
                # Found a cycle
                idx = path.index(v)
                cycles.append(path[idx:] + [v])
            elif color[v] == WHITE:
                dfs(v, path)
        path.pop()
        color[u] = BLACK

    for v in graph:
        if color[v] == WHITE:
            dfs(v, [])

    return cycles


def find_missing_deps(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Find dependencies that reference non-existent skills."""
    all_skills = set(graph.keys())
    missing = {}
    for skill, deps in graph.items():
        bad = [d for d in deps if d not in all_skills]
        if bad:
            missing[skill] = bad
    return missing


def topological_sort(graph: Dict[str, List[str]]) -> List[str]:
    """Return topological order using Kahn's algorithm."""
    in_degree = defaultdict(int)
    for v in graph:
        in_degree.setdefault(v, 0)
        for dep in graph[v]:
            in_degree[dep] += 1  # dep depends on v, so dep has incoming edge

    # Actually: edge from dep -> skill (skill depends on dep)
    # So in_degree[skill] = number of its dependencies
    in_degree = {v: 0 for v in graph}
    reverse = defaultdict(list)
    for skill, deps in graph.items():
        in_degree[skill] = len(deps)
        for d in deps:
            reverse[d].append(skill)

    queue = deque(v for v in graph if in_degree[v] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for dependent in reverse.get(node, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    return order


def print_tree(graph: Dict[str, List[str]], root: str, indent: int = 0, visited: Optional[Set[str]] = None) -> None:
    """Print dependency tree from a root skill."""
    if visited is None:
        visited = set()
    prefix = "  " * indent
    if root in visited:
        print(f"{prefix}{root} (shared)")
        return
    visited.add(root)
    deps = graph.get(root, [])
    print(f"{prefix}{root}")
    for dep in deps:
        print_tree(graph, dep, indent + 1, visited)


def export_dot(graph: Dict[str, List[str]], output: Path) -> None:
    """Export dependency graph in DOT format."""
    lines = ["digraph skills {", '  rankdir=LR;', '  node [shape=box, style=filled, fillcolor="#e8f5e9"];']
    for skill, deps in graph.items():
        if not deps:
            lines.append(f'  "{skill}" [fillcolor="#fff3e0"];')
        for dep in deps:
            lines.append(f'  "{skill}" -> "{dep}";')
    lines.append("}")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDOT graph written to {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate skill dependency DAG.")
    parser.add_argument("--skills-dir", default="skills", help="Path to skills directory (default: skills)")
    parser.add_argument("--dot", help="Export dependency graph to DOT file")
    parser.add_argument("--tree", help="Print dependency tree for a given skill")
    parser.add_argument("--topo", action="store_true", help="Print topological order")
    args = parser.parse_args()

    skills_dir = Path(args.skills_dir).resolve()
    if not skills_dir.exists():
        print(f"[ERROR] Skills directory not found: {skills_dir}", file=sys.stderr)
        return 1

    graph, metadata = load_skill_graph(skills_dir)
    print(f"Loaded {len(graph)} skills")

    # Check missing dependencies
    missing = find_missing_deps(graph)
    if missing:
        print(f"\n[ERROR] {len(missing)} skill(s) have missing dependencies:")
        for skill, bad_deps in sorted(missing.items()):
            print(f"  {skill} -> {', '.join(bad_deps)}")
    else:
        print("  All dependencies reference existing skills")

    # Check cycles
    cycles = find_cycles(graph)
    if cycles:
        print(f"\n[ERROR] Found {len(cycles)} circular dependency cycle(s):")
        for i, cycle in enumerate(cycles, 1):
            print(f"  Cycle {i}: {' -> '.join(cycle)}")
        return 1
    else:
        print("  No circular dependencies (DAG is valid)")

    # Print topological order
    if args.topo:
        order = topological_sort(graph)
        print(f"\nTopological order ({len(order)} skills):")
        for i, skill in enumerate(order, 1):
            deps = graph.get(skill, [])
            dep_str = f" -> [{', '.join(deps)}]" if deps else " (leaf)"
            print(f"  {i}. {skill}{dep_str}")

    # Print dependency tree
    if args.tree:
        if args.tree not in graph:
            print(f"\n[ERROR] Skill '{args.tree}' not found", file=sys.stderr)
            return 1
        print(f"\nDependency tree for '{args.tree}':")
        print_tree(graph, args.tree)

    # Export DOT
    if args.dot:
        export_dot(graph, Path(args.dot).resolve())

    # Summary stats
    leaves = [s for s, d in graph.items() if not d]
    roots = [s for s in graph if not any(s in d for d in graph.values())]
    max_deps = max((len(d) for d in graph.values()), default=0)
    # Count how many skills depend on each skill
    depended_count: Dict[str, int] = defaultdict(int)
    for deps in graph.values():
        for d in deps:
            depended_count[d] += 1
    most_depended = max(depended_count.items(), key=lambda x: x[1]) if depended_count else None

    print(f"\nSummary:")
    print(f"  Total skills: {len(graph)}")
    print(f"  Leaf skills (no deps): {len(leaves)}")
    print(f"  Max dependency depth: {max_deps}")
    if most_depended:
        print(f"  Most depended-on: {most_depended[0]} (depended by {most_depended[1]} skills)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
