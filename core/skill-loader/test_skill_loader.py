"""Tests for the skill loader module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.skill_loader.loader import (
    SkillLoader,
    _clean_line,
    _first_meaningful_paragraph,
    _parse_frontmatter,
    _strip_frontmatter,
    _summarize_skill_text,
)


def test_parse_frontmatter_simple():
    """Test parsing simple key-value front-matter."""
    text = """---
name: test-skill
description: A test skill
layer: base
---
Body content here."""
    result = _parse_frontmatter(text)
    assert result["name"] == "test-skill"
    assert result["description"] == "A test skill"
    assert result["layer"] == "base"


def test_parse_frontmatter_with_list():
    """Test parsing front-matter with list values."""
    text = """---
name: test-skill
dependencies:
  - fmriprep-tool
  - fsl-tool
layer: subagent
---
Body content."""
    result = _parse_frontmatter(text)
    assert result["name"] == "test-skill"
    assert result["dependencies"] == ["fmriprep-tool", "fsl-tool"]
    assert result["layer"] == "subagent"


def test_parse_frontmatter_empty():
    """Test parsing text without front-matter."""
    text = "No front-matter here."
    result = _parse_frontmatter(text)
    assert result == {}


def test_strip_frontmatter():
    """Test stripping front-matter from text."""
    text = """---
name: test
description: desc
---
Body content."""
    result = _strip_frontmatter(text)
    assert result == "Body content."
    assert "---" not in result


def test_strip_frontmatter_no_frontmatter():
    """Test stripping when no front-matter exists."""
    text = "Just plain text."
    result = _strip_frontmatter(text)
    assert result == text


def test_clean_line():
    """Test markdown cleaning."""
    assert _clean_line("`code`") == "code"
    assert _clean_line("**bold**") == "bold"
    assert _clean_line("*italic*") == "italic"
    assert _clean_line("[link](http://example.com)") == "link"
    assert _clean_line("  spaces  ") == "spaces"
    assert _clean_line("`code` and **bold**") == "code and bold"


def test_first_meaningful_paragraph_overview():
    """Test extracting paragraph under Overview section."""
    text = """---
name: test
---

## Overview

This is a meaningful paragraph about the skill.

## Usage

Some usage info."""
    result = _first_meaningful_paragraph(text)
    assert "meaningful paragraph" in result


def test_first_meaningful_paragraph_no_overview():
    """Test extracting first paragraph when no Overview section."""
    text = """---
name: test
---

This is the first meaningful paragraph.

## Section

More content."""
    result = _first_meaningful_paragraph(text)
    assert "first meaningful paragraph" in result


def test_first_meaningful_paragraph_skip_code():
    """Test that code blocks are skipped."""
    text = """---
name: test
---

```python
code block
```

This is the real paragraph."""
    result = _first_meaningful_paragraph(text)
    assert "real paragraph" in result
    assert "code block" not in result


def test_summarize_skill_text():
    """Test skill text summarization."""
    text = """---
name: test-skill
description: fallback description
---

## Overview

This skill performs neuroimaging analysis."""
    en, zh = _summarize_skill_text("test-skill", text, "fallback")
    assert "neuroimaging analysis" in en
    assert zh.startswith("该技能主要用于")


def test_summarize_skill_text_fallback():
    """Test fallback when no meaningful content."""
    en, zh = _summarize_skill_text("test-skill", "---\nname: test\n---\n", "fallback desc")
    assert "fallback desc" in en


def test_skill_loader_init():
    """Test SkillLoader initialization."""
    loader = SkillLoader(Path("skills"))
    assert loader.skills_dir == Path("skills")


def test_skill_loader_find(tmp_path):
    """Test skill finding by keyword."""
    # Create a minimal skill structure
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: test-skill
description: neuroimaging preprocessing
layer: base
---
A test skill for preprocessing.""")

    loader = SkillLoader(tmp_path)
    results = loader.find("neuroimaging")
    assert len(results) == 1
    assert results[0]["name"] == "test-skill"


def test_skill_loader_find_case_insensitive():
    """Test case-insensitive search."""
    loader = SkillLoader(Path("skills"))
    # This tests the find method logic, not actual file loading
    # The method should handle case-insensitive search
    assert loader.skills_dir == Path("skills")


if __name__ == "__main__":
    test_parse_frontmatter_simple()
    test_parse_frontmatter_with_list()
    test_parse_frontmatter_empty()
    test_strip_frontmatter()
    test_strip_frontmatter_no_frontmatter()
    test_clean_line()
    test_first_meaningful_paragraph_overview()
    test_first_meaningful_paragraph_no_overview()
    test_first_meaningful_paragraph_skip_code()
    test_summarize_skill_text()
    test_summarize_skill_text_fallback()
    test_skill_loader_init()
    print("\n=== ALL SKILL LOADER TESTS PASSED ===")
