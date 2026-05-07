"""Simple demonstration of Reflexion functionality.

This script demonstrates the basic usage of the Reflexion module
without requiring a full NeuroClaw agent setup.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

from core.agent.reflexion import (
    ReflectionEntry,
    ReflectionStorage,
    ReflectionRetriever,
    ReflexionAgent,
)


def demo_storage():
    """Demonstrate ReflectionStorage functionality."""
    print("=" * 60)
    print("Demo 1: ReflectionStorage - Save and Load")
    print("=" * 60)

    # Create temporary storage
    temp_dir = tempfile.mkdtemp()
    storage_path = Path(temp_dir) / "reflections.json"
    storage = ReflectionStorage(storage_path)

    # Create and save a reflection
    entry = ReflectionEntry(
        id="demo-1",
        timestamp="2026-05-07T12:00:00Z",
        trigger_type="tool_failure",
        task_description="Run fMRI preprocessing with FSL",
        tool_events=[
            {"tool": "run_shell_command", "command": "fslmaths ...", "success": False}
        ],
        error_summary="FSLDIR environment variable not set",
        root_cause_analysis="The FSL environment was not properly initialized before running fslmaths",
        alternative_approaches=[
            "Export FSLDIR before running FSL commands",
            "Use conda activate to ensure full environment setup",
            "Check neuroclaw_environment.json for correct FSL path"
        ],
        confidence_score=0.85,
        keywords=["fsl", "fmri", "environment", "preprocessing"],
        related_skills=["fmri-skill", "freesurfer-tool"]
    )

    storage.save(entry)
    print(f"✓ Saved reflection: {entry.id}")

    # Load and display
    loaded = storage.load_all()
    print(f"✓ Loaded {len(loaded)} reflection(s)")
    print(f"\nReflection details:")
    print(f"  Task: {loaded[0].task_description}")
    print(f"  Root cause: {loaded[0].root_cause_analysis}")
    print(f"  Confidence: {loaded[0].confidence_score}")
    print(f"  Keywords: {', '.join(loaded[0].keywords)}")
    print()


def demo_retrieval():
    """Demonstrate ReflectionRetriever functionality."""
    print("=" * 60)
    print("Demo 2: ReflectionRetriever - Keyword Search")
    print("=" * 60)

    # Create storage with multiple reflections
    temp_dir = tempfile.mkdtemp()
    storage_path = Path(temp_dir) / "reflections.json"
    storage = ReflectionStorage(storage_path)
    retriever = ReflectionRetriever(storage)

    # Add sample reflections
    reflections = [
        {
            "id": "r1",
            "task": "fMRI preprocessing",
            "keywords": ["fmri", "fsl", "preprocessing", "environment"],
            "analysis": "FSLDIR not set"
        },
        {
            "id": "r2",
            "task": "EEG analysis",
            "keywords": ["eeg", "mne", "analysis", "import"],
            "analysis": "MNE library not installed"
        },
        {
            "id": "r3",
            "task": "DTI tractography",
            "keywords": ["dti", "fsl", "tractography", "bedpostx"],
            "analysis": "Insufficient memory for bedpostx"
        }
    ]

    for r in reflections:
        entry = ReflectionEntry(
            id=r["id"],
            timestamp="2026-05-07T12:00:00Z",
            trigger_type="tool_failure",
            task_description=r["task"],
            tool_events=[],
            error_summary="",
            root_cause_analysis=r["analysis"],
            alternative_approaches=[],
            confidence_score=0.8,
            keywords=r["keywords"],
            related_skills=[]
        )
        storage.save(entry)

    print(f"✓ Added {len(reflections)} reflections to storage\n")

    # Test retrieval
    queries = [
        ["fmri", "fsl"],
        ["eeg"],
        ["fsl", "tractography"]
    ]

    for query in queries:
        results = retriever.retrieve_similar(query, top_k=3)
        print(f"Query: {query}")
        print(f"Found {len(results)} match(es):")
        for result in results:
            print(f"  - {result.task_description}: {result.root_cause_analysis}")
        print()


def demo_agent_mock():
    """Demonstrate ReflexionAgent with mocked LLM."""
    print("=" * 60)
    print("Demo 3: ReflexionAgent - Failure Reflection (Mocked)")
    print("=" * 60)

    # Create mock LLM
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps({
        "root_cause": "FSL environment variable FSLDIR is not set in the current shell session",
        "should_retry": True,
        "retry_strategy": "Export FSLDIR from neuroclaw_environment.json before executing FSL commands",
        "confidence": 0.9
    })
    mock_llm.chat.completions.create.return_value = mock_response

    # Create agent
    temp_dir = tempfile.mkdtemp()
    storage_path = Path(temp_dir) / "reflections.json"
    agent = ReflexionAgent(mock_llm, storage_path)

    # Simulate tool failure
    reflection = agent.reflect_on_failure(
        tool_name="run_shell_command",
        args={"command": "fslmaths input.nii.gz -mas mask.nii.gz output.nii.gz"},
        error="fslmaths: command not found (FSLDIR not set)",
        recent_events=[
            {"tool": "read_workspace_file", "success": True},
            {"tool": "run_shell_command", "success": False}
        ]
    )

    print("✓ Generated immediate reflection on tool failure\n")
    print(f"Root cause: {reflection['root_cause']}")
    print(f"Should retry: {reflection['should_retry']}")
    print(f"Retry strategy: {reflection['retry_strategy']}")
    print(f"Confidence: {reflection['confidence']}")
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Reflexion Module Demonstration")
    print("=" * 60 + "\n")

    try:
        demo_storage()
        demo_retrieval()
        demo_agent_mock()

        print("=" * 60)
        print("All demos completed successfully! ✓")
        print("=" * 60)
        print("\nReflexion is ready to use in NeuroClaw.")
        print("Try these commands in the agent REPL:")
        print("  /reflection list              - Show recent reflections")
        print("  /reflection search <keywords> - Search by keywords")
        print()

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
