"""Unit tests for the Reflexion module.

Tests the core functionality of ReflectionEntry, ReflectionStorage,
ReflectionRetriever, and ReflexionAgent.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from core.agent.reflexion import (
    ReflectionEntry,
    ReflectionStorage,
    ReflectionRetriever,
    ReflexionAgent,
)


class TestReflectionEntry(unittest.TestCase):
    """Test ReflectionEntry data structure."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        entry = ReflectionEntry(
            id="test-id",
            timestamp="2026-05-07T12:00:00Z",
            trigger_type="tool_failure",
            task_description="Test task",
            tool_events=[{"tool": "test", "success": False}],
            error_summary="Test error",
            root_cause_analysis="Root cause",
            alternative_approaches=["approach1", "approach2"],
            confidence_score=0.8,
            keywords=["test", "error"],
            related_skills=["test-skill"]
        )

        data = entry.to_dict()
        self.assertEqual(data["id"], "test-id")
        self.assertEqual(data["trigger_type"], "tool_failure")
        self.assertEqual(data["confidence_score"], 0.8)

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "test-id",
            "timestamp": "2026-05-07T12:00:00Z",
            "trigger_type": "task_summary",
            "task_description": "Test task",
            "tool_events": [],
            "error_summary": "",
            "root_cause_analysis": "Analysis",
            "alternative_approaches": [],
            "confidence_score": 0.9,
            "keywords": ["test"],
            "related_skills": []
        }

        entry = ReflectionEntry.from_dict(data)
        self.assertEqual(entry.id, "test-id")
        self.assertEqual(entry.trigger_type, "task_summary")
        self.assertEqual(entry.confidence_score, 0.9)


class TestReflectionStorage(unittest.TestCase):
    """Test ReflectionStorage persistence."""

    def setUp(self):
        """Create temporary storage for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "reflections.json"
        self.storage = ReflectionStorage(self.storage_path)

    def test_initialization(self):
        """Test storage file is created on initialization."""
        self.assertTrue(self.storage_path.exists())

        with open(self.storage_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["version"], "1.0")
        self.assertEqual(data["reflections"], [])

    def test_save_and_load(self):
        """Test saving and loading reflections."""
        entry = ReflectionEntry(
            id="test-1",
            timestamp="2026-05-07T12:00:00Z",
            trigger_type="tool_failure",
            task_description="Test",
            tool_events=[],
            error_summary="Error",
            root_cause_analysis="Cause",
            alternative_approaches=["alt1"],
            confidence_score=0.7,
            keywords=["test"],
            related_skills=[]
        )

        self.storage.save(entry)
        loaded = self.storage.load_all()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].id, "test-1")
        self.assertEqual(loaded[0].confidence_score, 0.7)

    def test_pruning(self):
        """Test automatic pruning when exceeding MAX_REFLECTIONS."""
        # Save 105 reflections (exceeds MAX_REFLECTIONS=100)
        for i in range(105):
            entry = ReflectionEntry(
                id=f"test-{i}",
                timestamp=f"2026-05-07T12:{i:02d}:00Z",
                trigger_type="tool_failure",
                task_description=f"Test {i}",
                tool_events=[],
                error_summary="",
                root_cause_analysis="",
                alternative_approaches=[],
                confidence_score=0.5,
                keywords=[],
                related_skills=[]
            )
            self.storage.save(entry)

        loaded = self.storage.load_all()
        self.assertEqual(len(loaded), 100)
        # Should keep the most recent 100
        self.assertEqual(loaded[0].id, "test-5")
        self.assertEqual(loaded[-1].id, "test-104")


class TestReflectionRetriever(unittest.TestCase):
    """Test ReflectionRetriever keyword matching."""

    def setUp(self):
        """Create storage with sample reflections."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "reflections.json"
        self.storage = ReflectionStorage(self.storage_path)
        self.retriever = ReflectionRetriever(self.storage)

        # Add sample reflections
        self.storage.save(ReflectionEntry(
            id="r1",
            timestamp="2026-05-07T12:00:00Z",
            trigger_type="tool_failure",
            task_description="fMRI preprocessing",
            tool_events=[],
            error_summary="FSL error",
            root_cause_analysis="FSLDIR not set",
            alternative_approaches=[],
            confidence_score=0.8,
            keywords=["fmri", "fsl", "preprocessing", "environment"],
            related_skills=["fmri-skill"]
        ))

        self.storage.save(ReflectionEntry(
            id="r2",
            timestamp="2026-05-07T12:01:00Z",
            trigger_type="task_summary",
            task_description="EEG analysis",
            tool_events=[],
            error_summary="",
            root_cause_analysis="MNE import failed",
            alternative_approaches=[],
            confidence_score=0.9,
            keywords=["eeg", "mne", "analysis"],
            related_skills=["eeg-skill"]
        ))

    def test_retrieve_similar(self):
        """Test keyword-based retrieval."""
        results = self.retriever.retrieve_similar(["fmri", "fsl"], top_k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "r1")

    def test_retrieve_no_match(self):
        """Test retrieval with no matching keywords."""
        results = self.retriever.retrieve_similar(["nonexistent"], top_k=5)
        self.assertEqual(len(results), 0)

    def test_retrieve_top_k(self):
        """Test top_k limiting."""
        # Add more reflections
        for i in range(5):
            self.storage.save(ReflectionEntry(
                id=f"r{i+3}",
                timestamp=f"2026-05-07T12:{i+2:02d}:00Z",
                trigger_type="tool_failure",
                task_description=f"Task {i}",
                tool_events=[],
                error_summary="",
                root_cause_analysis="",
                alternative_approaches=[],
                confidence_score=0.5,
                keywords=["common", "keyword"],
                related_skills=[]
            ))

        results = self.retriever.retrieve_similar(["common"], top_k=3)
        self.assertEqual(len(results), 3)


class TestReflexionAgent(unittest.TestCase):
    """Test ReflexionAgent integration."""

    def setUp(self):
        """Create mock LLM client and agent."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "reflections.json"

        # Mock LLM client
        self.mock_llm = Mock()
        self.agent = ReflexionAgent(self.mock_llm, self.storage_path)

    def test_reflect_on_failure(self):
        """Test immediate failure reflection."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "root_cause": "Missing environment variable",
            "should_retry": True,
            "retry_strategy": "Set FSLDIR before running",
            "confidence": 0.85
        })
        self.mock_llm.chat.completions.create.return_value = mock_response

        reflection = self.agent.reflect_on_failure(
            tool_name="run_shell_command",
            args={"command": "fslmaths ..."},
            error="FSLDIR not set",
            recent_events=[]
        )

        self.assertEqual(reflection["root_cause"], "Missing environment variable")
        self.assertTrue(reflection["should_retry"])
        self.assertEqual(reflection["confidence"], 0.85)

    def test_reflect_on_task(self):
        """Test summary reflection at task completion."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "root_cause_analysis": "Task completed successfully",
            "alternative_approaches": ["Use different tool", "Optimize parameters"],
            "confidence": 0.9,
            "keywords": "fmri, preprocessing, success"
        })
        self.mock_llm.chat.completions.create.return_value = mock_response

        entry = self.agent.reflect_on_task(
            task_desc="Run fMRI preprocessing",
            tool_events=[{"tool": "test", "success": True}],
            outcome="success"
        )

        self.assertEqual(entry.trigger_type, "task_summary")
        self.assertEqual(entry.root_cause_analysis, "Task completed successfully")
        self.assertEqual(len(entry.alternative_approaches), 2)
        self.assertIn("fmri", entry.keywords)

    def test_retrieve_relevant_reflections(self):
        """Test retrieval of relevant historical reflections."""
        # Add a reflection first
        self.agent._storage.save(ReflectionEntry(
            id="test",
            timestamp="2026-05-07T12:00:00Z",
            trigger_type="tool_failure",
            task_description="fMRI task",
            tool_events=[],
            error_summary="",
            root_cause_analysis="Analysis",
            alternative_approaches=[],
            confidence_score=0.8,
            keywords=["fmri", "preprocessing"],
            related_skills=[]
        ))

        results = self.agent.retrieve_relevant_reflections("Run fMRI preprocessing pipeline")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "test")


if __name__ == "__main__":
    unittest.main()
