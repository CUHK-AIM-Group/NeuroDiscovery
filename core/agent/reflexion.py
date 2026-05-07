"""Reflexion: Self-reflection mechanism for learning from execution failures.

This module implements the Reflexion framework for NeuroClaw, enabling the agent
to reflect on tool execution failures and complex tasks, store reflections
persistently, and retrieve relevant historical reflections for similar tasks.

Architecture:
    - ReflectionEntry: Data structure for a single reflection record
    - ReflectionStorage: JSON-based persistent storage manager
    - ReflectionRetriever: Keyword-based similarity retrieval
    - ReflexionAgent: Main coordinator for reflection generation and retrieval

Usage:
    from core.agent.reflexion import ReflexionAgent

    agent = ReflexionAgent(llm_client, storage_path)

    # Immediate reflection on tool failure
    reflection = agent.reflect_on_failure(
        tool_name="run_shell_command",
        args={"command": "fslmaths ..."},
        error="FSLDIR not set",
        recent_events=[...]
    )

    # Summary reflection at task completion
    entry = agent.reflect_on_task(
        task_desc="Run fMRI preprocessing",
        tool_events=[...],
        outcome="partial_failure"
    )

    # Retrieve relevant historical reflections
    reflections = agent.retrieve_relevant_reflections("fMRI preprocessing")
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Maximum number of reflections to keep in storage
MAX_REFLECTIONS = 100


@dataclass
class ReflectionEntry:
    """A single reflection record.

    Attributes:
        id: Unique identifier (UUID)
        timestamp: ISO 8601 timestamp
        trigger_type: "tool_failure" or "task_summary"
        task_description: Brief description of the task
        tool_events: List of relevant tool execution events
        error_summary: Error message summary (for tool_failure only)
        root_cause_analysis: LLM-generated root cause analysis
        alternative_approaches: List of suggested alternative strategies
        confidence_score: Confidence in the analysis [0.0, 1.0]
        keywords: Keywords for retrieval (lowercase)
        related_skills: Names of skills involved in the task
    """
    id: str
    timestamp: str
    trigger_type: str  # "tool_failure" | "task_summary"
    task_description: str
    tool_events: list[dict]
    error_summary: str
    root_cause_analysis: str
    alternative_approaches: list[str]
    confidence_score: float
    keywords: list[str]
    related_skills: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReflectionEntry:
        """Create from dictionary loaded from JSON."""
        return cls(**data)


class ReflectionStorage:
    """Persistent storage manager for reflections using JSON files.

    Storage format:
        {
            "version": "1.0",
            "reflections": [...]
        }

    The storage automatically prunes old reflections when the count exceeds
    MAX_REFLECTIONS, keeping only the most recent entries.
    """

    def __init__(self, storage_path: Path):
        """Initialize storage.

        Args:
            storage_path: Path to the JSON file (e.g., .neuroclaw_checkpoints/reflections.json)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize file if it doesn't exist
        if not self.storage_path.exists():
            self._write_storage({"version": "1.0", "reflections": []})

    def save(self, entry: ReflectionEntry) -> None:
        """Append a reflection entry to storage.

        Args:
            entry: The reflection entry to save
        """
        data = self._read_storage()
        data["reflections"].append(entry.to_dict())
        self._write_storage(data)
        self._prune_old()
        logger.info(f"Reflection saved: {entry.id} ({entry.trigger_type})")

    def load_all(self) -> list[ReflectionEntry]:
        """Load all reflection entries from storage.

        Returns:
            List of ReflectionEntry objects, ordered by timestamp (oldest first)
        """
        data = self._read_storage()
        return [ReflectionEntry.from_dict(r) for r in data["reflections"]]

    def _read_storage(self) -> dict[str, Any]:
        """Read the JSON storage file."""
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read reflections storage: {e}")
            return {"version": "1.0", "reflections": []}

    def _write_storage(self, data: dict[str, Any]) -> None:
        """Write to the JSON storage file."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to write reflections storage: {e}")

    def _prune_old(self) -> None:
        """Remove oldest reflections if count exceeds MAX_REFLECTIONS."""
        data = self._read_storage()
        reflections = data["reflections"]

        if len(reflections) > MAX_REFLECTIONS:
            # Keep only the most recent MAX_REFLECTIONS entries
            data["reflections"] = reflections[-MAX_REFLECTIONS:]
            self._write_storage(data)
            logger.info(f"Pruned reflections to {MAX_REFLECTIONS} entries")


class ReflectionRetriever:
    """Keyword-based retrieval for similar reflections.

    Uses simple keyword intersection matching to find relevant historical
    reflections. Future versions may use vector embeddings for semantic search.
    """

    def __init__(self, storage: ReflectionStorage):
        """Initialize retriever.

        Args:
            storage: The ReflectionStorage instance to retrieve from
        """
        self.storage = storage

    def retrieve_similar(
        self,
        keywords: list[str],
        top_k: int = 3
    ) -> list[ReflectionEntry]:
        """Retrieve reflections similar to the given keywords.

        Args:
            keywords: List of query keywords (will be lowercased)
            top_k: Maximum number of results to return

        Returns:
            List of ReflectionEntry objects, sorted by relevance (descending)
        """
        query_keywords = set(kw.lower() for kw in keywords)
        all_reflections = self.storage.load_all()

        # Score each reflection by keyword intersection size
        scored = []
        for reflection in all_reflections:
            reflection_keywords = set(reflection.keywords)
            intersection_size = len(query_keywords & reflection_keywords)
            if intersection_size > 0:
                scored.append((intersection_size, reflection))

        # Sort by score (descending) and return top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [reflection for _, reflection in scored[:top_k]]


class ReflexionAgent:
    """Main coordinator for reflection generation and retrieval.

    This agent uses an LLM to generate reflections on tool failures and task
    completions, stores them persistently, and retrieves relevant historical
    reflections for similar tasks.
    """

    def __init__(self, llm_client: Any, storage_path: Path):
        """Initialize the Reflexion agent.

        Args:
            llm_client: OpenAI-compatible LLM client (e.g., openai.OpenAI())
            storage_path: Path to the reflections JSON file
        """
        self.llm_client = llm_client
        self._storage = ReflectionStorage(storage_path)
        self._retriever = ReflectionRetriever(self._storage)

    def reflect_on_failure(
        self,
        tool_name: str,
        args: dict[str, Any],
        error: str,
        recent_events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate immediate reflection on a tool execution failure.

        This is a lightweight reflection used for retry decision-making.

        Args:
            tool_name: Name of the failed tool
            args: Arguments passed to the tool
            error: Error message from the tool
            recent_events: Last 3 tool events for context

        Returns:
            Dictionary with keys:
                - root_cause: 1-2 sentence analysis
                - should_retry: bool
                - retry_strategy: suggested strategy if should_retry
                - confidence: float [0.0, 1.0]
        """
        prompt = self._build_failure_prompt(tool_name, args, error, recent_events)

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",  # Use lightweight model for immediate reflection
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content or "{}"
            reflection = json.loads(content)

            # Validate and set defaults
            return {
                "root_cause": reflection.get("root_cause", "Unknown error"),
                "should_retry": reflection.get("should_retry", False),
                "retry_strategy": reflection.get("retry_strategy", ""),
                "confidence": float(reflection.get("confidence", 0.5))
            }

        except Exception as e:
            logger.error(f"Failed to generate failure reflection: {e}")
            return {
                "root_cause": f"Tool {tool_name} failed: {error}",
                "should_retry": False,
                "retry_strategy": "",
                "confidence": 0.0
            }

    def reflect_on_task(
        self,
        task_desc: str,
        tool_events: list[dict[str, Any]],
        outcome: str
    ) -> ReflectionEntry:
        """Generate summary reflection at task completion.

        This is a comprehensive reflection stored for long-term learning.

        Args:
            task_desc: Brief description of the task
            tool_events: Complete list of tool execution events
            outcome: "success" | "partial_failure" | "failure"

        Returns:
            ReflectionEntry object (also saved to storage)
        """
        prompt = self._build_summary_prompt(task_desc, tool_events, outcome)

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content or "{}"
            reflection_data = json.loads(content)

            # Extract error summary from tool events
            error_summary = ""
            for event in tool_events:
                if not event.get("success", True):
                    error_summary = event.get("result", {}).get("error", "Unknown error")
                    break

            # Extract related skills
            related_skills = []
            for event in tool_events:
                skills = event.get("skills_used", [])
                related_skills.extend(skills)
            related_skills = list(set(related_skills))  # Deduplicate

            # Create reflection entry
            entry = ReflectionEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                trigger_type="task_summary",
                task_description=task_desc[:200],  # Truncate to 200 chars
                tool_events=tool_events,
                error_summary=error_summary,
                root_cause_analysis=reflection_data.get("root_cause_analysis", ""),
                alternative_approaches=reflection_data.get("alternative_approaches", []),
                confidence_score=float(reflection_data.get("confidence", 0.5)),
                keywords=self._parse_keywords(reflection_data.get("keywords", "")),
                related_skills=related_skills
            )

            # Save to storage
            self._storage.save(entry)
            return entry

        except Exception as e:
            logger.error(f"Failed to generate task reflection: {e}")
            # Return a minimal entry on failure
            return ReflectionEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                trigger_type="task_summary",
                task_description=task_desc[:200],
                tool_events=tool_events,
                error_summary="Reflection generation failed",
                root_cause_analysis=f"Failed to generate reflection: {e}",
                alternative_approaches=[],
                confidence_score=0.0,
                keywords=[],
                related_skills=[]
            )

    def retrieve_relevant_reflections(
        self,
        task_desc: str,
        top_k: int = 3
    ) -> list[ReflectionEntry]:
        """Retrieve relevant historical reflections for a task.

        Args:
            task_desc: Description of the current task
            top_k: Maximum number of reflections to retrieve

        Returns:
            List of relevant ReflectionEntry objects
        """
        # Extract keywords from task description
        keywords = self._extract_keywords_from_text(task_desc)
        return self._retriever.retrieve_similar(keywords, top_k)

    def _build_failure_prompt(
        self,
        tool_name: str,
        args: dict[str, Any],
        error: str,
        recent_events: list[dict[str, Any]]
    ) -> str:
        """Build prompt for immediate failure reflection."""
        args_str = json.dumps(args, ensure_ascii=False, indent=2)
        events_str = json.dumps(recent_events, ensure_ascii=False, indent=2)

        return f"""You are analyzing a tool execution failure. Provide a concise reflection.

Tool: {tool_name}
Arguments: {args_str}
Error: {error}
Recent context (last 3 tool events): {events_str}

Analyze the failure and output JSON with these fields:
{{
  "root_cause": "1-2 sentence analysis of why this failed",
  "should_retry": true or false,
  "retry_strategy": "if should_retry is true, suggest how to retry (e.g., fix arguments, check environment)",
  "confidence": 0.0 to 1.0 (how confident are you in this analysis)
}}

Focus on actionable insights. Be concise."""

    def _build_summary_prompt(
        self,
        task_desc: str,
        tool_events: list[dict[str, Any]],
        outcome: str
    ) -> str:
        """Build prompt for summary reflection."""
        # Truncate tool events if too long
        events_str = json.dumps(tool_events[:10], ensure_ascii=False, indent=2)
        if len(tool_events) > 10:
            events_str += f"\n... ({len(tool_events) - 10} more events)"

        return f"""You are reflecting on a completed task execution. Analyze the full trajectory.

Task: {task_desc}
Outcome: {outcome}
Tool Events: {events_str}

Generate a structured reflection in JSON format:
{{
  "root_cause_analysis": "What went wrong (if any)? What worked well? 2-3 sentences.",
  "alternative_approaches": ["approach 1", "approach 2", "approach 3"],
  "confidence": 0.0 to 1.0 (confidence in execution quality),
  "keywords": "5-10 keywords for future retrieval, lowercase, comma-separated (e.g., fmri, preprocessing, fsl, environment)"
}}

Focus on:
1. Root causes of failures or inefficiencies
2. 2-3 concrete alternative strategies for similar tasks
3. Keywords that capture the task domain and key concepts"""

    def _extract_keywords_from_text(self, text: str) -> list[str]:
        """Extract keywords from text for retrieval.

        Simple heuristic: lowercase words, remove common stop words.
        """
        # Lowercase and split
        words = re.findall(r'\b\w+\b', text.lower())

        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "should", "could", "may", "might", "can", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they"
        }

        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Return unique keywords (preserve order)
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords[:15]  # Limit to 15 keywords

    def _parse_keywords(self, keywords_str: str) -> list[str]:
        """Parse comma-separated keywords string into list."""
        if not keywords_str:
            return []
        return [kw.strip().lower() for kw in keywords_str.split(",") if kw.strip()]
