"""Tests for the DAG validation module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.skill_loader.validate_dag import (
    find_cycles,
    find_missing_deps,
    topological_sort,
)


def test_find_cycles_no_cycles():
    """Test that a DAG without cycles is detected correctly."""
    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": [],
    }
    cycles = find_cycles(graph)
    assert len(cycles) == 0


def test_find_cycles_simple_cycle():
    """Test detection of a simple cycle."""
    graph = {
        "A": ["B"],
        "B": ["C"],
        "C": ["A"],
    }
    cycles = find_cycles(graph)
    assert len(cycles) > 0
    # Check that the cycle contains A, B, C
    cycle_nodes = set()
    for cycle in cycles:
        cycle_nodes.update(cycle)
    assert "A" in cycle_nodes
    assert "B" in cycle_nodes
    assert "C" in cycle_nodes


def test_find_cycles_self_loop():
    """Test detection of a self-loop."""
    graph = {
        "A": ["A"],
    }
    cycles = find_cycles(graph)
    assert len(cycles) > 0


def test_find_cycles_complex():
    """Test detection of cycles in a more complex graph."""
    graph = {
        "A": ["B"],
        "B": ["C"],
        "C": ["D"],
        "D": ["B"],  # Creates cycle: B -> C -> D -> B
        "E": ["F"],
        "F": [],
    }
    cycles = find_cycles(graph)
    assert len(cycles) > 0


def test_find_missing_deps_no_missing():
    """Test when all dependencies exist."""
    graph = {
        "A": ["B", "C"],
        "B": ["C"],
        "C": [],
    }
    missing = find_missing_deps(graph)
    assert len(missing) == 0


def test_find_missing_deps_with_missing():
    """Test detection of missing dependencies."""
    graph = {
        "A": ["B", "nonexistent"],
        "B": ["C"],
        "C": [],
    }
    missing = find_missing_deps(graph)
    assert "A" in missing
    assert "nonexistent" in missing["A"]


def test_find_missing_deps_multiple_missing():
    """Test detection of multiple missing dependencies."""
    graph = {
        "skill-a": ["dep-1", "dep-2"],
        "skill-b": ["dep-3"],
        "dep-1": [],
    }
    missing = find_missing_deps(graph)
    assert "skill-a" in missing
    assert "dep-2" in missing["skill-a"]
    assert "skill-b" in missing
    assert "dep-3" in missing["skill-b"]


def test_topological_sort_linear():
    """Test topological sort on a linear chain."""
    graph = {
        "A": ["B"],
        "B": ["C"],
        "C": [],
    }
    order = topological_sort(graph)
    assert order.index("C") < order.index("B")
    assert order.index("B") < order.index("A")


def test_topological_sort_diamond():
    """Test topological sort on a diamond dependency."""
    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": [],
    }
    order = topological_sort(graph)
    # D must come before B and C
    assert order.index("D") < order.index("B")
    assert order.index("D") < order.index("C")
    # B and C must come before A
    assert order.index("B") < order.index("A")
    assert order.index("C") < order.index("A")


def test_topological_sort_independent():
    """Test topological sort with independent nodes."""
    graph = {
        "A": [],
        "B": [],
        "C": [],
    }
    order = topological_sort(graph)
    assert len(order) == 3
    assert set(order) == {"A", "B", "C"}


def test_topological_sort_empty():
    """Test topological sort on empty graph."""
    graph = {}
    order = topological_sort(graph)
    assert order == []


if __name__ == "__main__":
    test_find_cycles_no_cycles()
    test_find_cycles_simple_cycle()
    test_find_cycles_self_loop()
    test_find_cycles_complex()
    test_find_missing_deps_no_missing()
    test_find_missing_deps_with_missing()
    test_find_missing_deps_multiple_missing()
    test_topological_sort_linear()
    test_topological_sort_diamond()
    test_topological_sort_independent()
    test_topological_sort_empty()
    print("\n=== ALL DAG VALIDATION TESTS PASSED ===")
