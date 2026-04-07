"""
NeuroClaw Skill Loader

Scans skills/*/SKILL.md, parses YAML front-matter, and returns a list of
skill descriptors.  Also discovers handler.js and Python handler files.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Extract key: value pairs from a YAML-style front-matter block."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    result: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip().strip('"')
    return result


class SkillLoader:
    """
    Discovers and registers NeuroClaw skills from a skills directory.

    Parameters
    ----------
    skills_dir : Path
        Directory containing one sub-folder per skill (each with SKILL.md).
    """

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = Path(skills_dir)

    def load_all(self) -> list[dict[str, Any]]:
        """
        Return a list of skill descriptor dicts, one per discovered skill.

        Each dict contains:
            name        : str — skill name from SKILL.md front-matter
            description : str — skill description
            path        : Path — skill directory
            handler     : Path | None — handler.js or handler.py if present
        """
        skills: list[dict[str, Any]] = []

        if not self.skills_dir.is_dir():
            return skills

        for skill_dir in sorted(self.skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            meta = _parse_frontmatter(skill_md.read_text())
            name = meta.get("name") or skill_dir.name
            description = meta.get("description", "")

            # Locate handler file
            handler: Path | None = None
            # handler.ts is intentionally excluded: TypeScript requires compilation
            # before execution and is not directly runnable by Node.js without a build step.
            for candidate in ("handler.js", "handler.py"):
                h = skill_dir / candidate
                if h.exists():
                    handler = h
                    break

            skills.append(
                {
                    "name": name,
                    "description": description,
                    "path": skill_dir,
                    "handler": handler,
                    "skill_md": skill_md,
                }
            )

        return skills

    def find(self, query: str) -> list[dict[str, Any]]:
        """
        Case-insensitive keyword search over skill names and descriptions.

        Returns skills ranked by number of matching keywords.
        """
        keywords = [kw.lower() for kw in query.split()]
        scored: list[tuple[int, dict]] = []

        for skill in self.load_all():
            haystack = (skill["name"] + " " + skill["description"]).lower()
            score = sum(1 for kw in keywords if kw in haystack)
            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored]
