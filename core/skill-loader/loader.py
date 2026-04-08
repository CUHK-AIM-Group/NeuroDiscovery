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


def _strip_frontmatter(text: str) -> str:
    """Return SKILL.md body without YAML front-matter."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return text
    return text[match.end():]


def _clean_line(line: str) -> str:
    """Normalize markdown-heavy lines into plain readable text."""
    out = line.strip()
    out = re.sub(r"`([^`]*)`", r"\1", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"\1", out)
    out = re.sub(r"\*([^*]+)\*", r"\1", out)
    out = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _first_meaningful_paragraph(md_text: str) -> str:
    """
    Extract a concise paragraph from SKILL.md body.

    Preference:
    1) Text under an Overview section
    2) First non-code, non-table, non-list paragraph in the file
    """
    body = _strip_frontmatter(md_text)
    lines = body.splitlines()

    def collect_paragraph(start_idx: int) -> str:
        in_code = False
        para: list[str] = []
        for i in range(start_idx, len(lines)):
            raw = lines[i]
            s = raw.strip()
            if s.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            if not s:
                if para:
                    break
                continue
            if s.startswith("#"):
                if para:
                    break
                continue
            if s.startswith(("-", "*", "|", ">")):
                if para:
                    break
                continue
            if re.match(r"^\d+[\.)]\s", s):
                if para:
                    break
                continue
            cleaned = _clean_line(s)
            if cleaned:
                para.append(cleaned)
        return " ".join(para).strip()

    for idx, raw in enumerate(lines):
        s = raw.strip().lower()
        if s.startswith("## overview") or s.startswith("# overview"):
            p = collect_paragraph(idx + 1)
            if p:
                return p

    for idx in range(len(lines)):
        p = collect_paragraph(idx)
        if p:
            return p

    return ""


def _summarize_skill_text(skill_name: str, md_text: str, fallback_desc: str) -> tuple[str, str]:
    """Create stable EN/ZH short summaries from SKILL.md content."""
    para = _first_meaningful_paragraph(md_text)
    en = para or fallback_desc.strip() or f"{skill_name} skill for specialized workflow support."
    en = re.sub(r"\s+", " ", en).strip()
    if len(en) > 220:
        en = en[:217].rstrip() + "..."

    zh = f"该技能主要用于：{en}"
    return en, zh


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

    def _candidate_skill_roots(self) -> list[Path]:
        """
        Return candidate skill roots in priority order.

        Priority:
        1) Current working directory convention: ./skills
        2) Explicit loader path provided by caller (self.skills_dir)

        This guarantees the default convention works without any path config:
            ./skills/<skill-name>/SKILL.md
        """
        candidates = [Path.cwd() / "skills", self.skills_dir]
        seen: set[Path] = set()
        ordered: list[Path] = []
        for c in candidates:
            p = c.resolve()
            if p in seen:
                continue
            seen.add(p)
            ordered.append(p)
        return ordered

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

        loaded_from_md: set[Path] = set()

        for root in self._candidate_skill_roots():
            if not root.is_dir():
                continue

            for skill_dir in sorted(root.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = (skill_dir / "SKILL.md").resolve()
                if not skill_md.exists() or skill_md in loaded_from_md:
                    continue

                skill_text = skill_md.read_text(encoding="utf-8")
                meta = _parse_frontmatter(skill_text)
                name = meta.get("name") or skill_dir.name
                description = meta.get("description", "")
                summary_en, summary_zh = _summarize_skill_text(name, skill_text, description)

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
                        "summary_en": summary_en,
                        "summary_zh": summary_zh,
                        "path": skill_dir,
                        "handler": handler,
                        "skill_md": skill_md,
                    }
                )
                loaded_from_md.add(skill_md)

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
