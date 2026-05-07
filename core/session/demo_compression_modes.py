"""Demo script comparing stub mode vs LLM summary mode.

This script demonstrates the difference between simple stub compression
and LLM-generated semantic summaries.
"""

from unittest.mock import Mock
from core.session.manager import SessionManager


def demo_stub_mode():
    """Demonstrate stub mode compression."""
    print("=" * 70)
    print("Demo 1: Stub Mode (Default)")
    print("=" * 70)

    # Create SessionManager in stub mode
    env = {"llm_backend": {"provider": "openai"}}
    manager = SessionManager(env=env, compression_mode="stub")

    # Simulate a conversation with 25 messages
    history = [
        {"role": "system", "content": "You are NeuroClaw, a neuroscience research assistant."},
    ]

    # Add 25 user/assistant messages
    for i in range(1, 26):
        history.append({"role": "user", "content": f"Question {i}"})
        history.append({"role": "assistant", "content": f"Answer {i}"})

    print(f"\nBefore compression: {len(history)} messages")
    print(f"  - 1 system message")
    print(f"  - 25 user/assistant pairs (50 messages)")
    print(f"  - Total: 51 messages\n")

    # Compress
    manager.maybe_compress(history)

    print(f"After compression: {len(history)} messages")
    print(f"  - 1 system message")
    print(f"  - 1 summary stub")
    print(f"  - 20 recent messages (10 pairs)")
    print(f"  - Total: 22 messages\n")

    # Show the summary stub
    summary = history[1]  # Second message (after system)
    print("Summary stub content:")
    print(f"  Role: {summary['role']}")
    print(f"  Content: {summary['content']}\n")


def demo_llm_summary_mode():
    """Demonstrate LLM summary mode compression."""
    print("=" * 70)
    print("Demo 2: LLM Summary Mode")
    print("=" * 70)

    # Mock LLM client
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = (
        "Earlier discussion covered fMRI preprocessing with FSL. "
        "User encountered FSLDIR environment variable issues, which were resolved "
        "by exporting FSLDIR from neuroclaw_environment.json. "
        "Successfully ran fslmaths on input data."
    )
    mock_llm.chat.completions.create.return_value = mock_response

    # Create SessionManager in LLM summary mode
    env = {"llm_backend": {"provider": "openai"}}
    manager = SessionManager(
        env=env,
        llm_client=mock_llm,
        compression_mode="llm_summary"
    )

    # Simulate a realistic conversation
    history = [
        {"role": "system", "content": "You are NeuroClaw, a neuroscience research assistant."},
        {"role": "user", "content": "How do I run fMRI preprocessing with FSL?"},
        {"role": "assistant", "content": "You need to set FSLDIR environment variable first. Export it from neuroclaw_environment.json."},
        {"role": "user", "content": "I got 'FSLDIR not set' error when running fslmaths"},
        {"role": "assistant", "content": "Make sure to export FSLDIR before running FSL commands. Check your neuroclaw_environment.json."},
        {"role": "user", "content": "It works now! The preprocessing completed successfully."},
        {"role": "assistant", "content": "Great! Your fMRI data has been preprocessed with FSL."},
    ]

    # Add more messages to trigger compression
    for i in range(7, 26):
        history.append({"role": "user", "content": f"Follow-up question {i}"})
        history.append({"role": "assistant", "content": f"Follow-up answer {i}"})

    print(f"\nBefore compression: {len(history)} messages")
    print(f"  - 1 system message")
    print(f"  - 25 user/assistant pairs (50 messages)")
    print(f"  - Total: 51 messages\n")

    # Compress
    manager.maybe_compress(history)

    print(f"After compression: {len(history)} messages")
    print(f"  - 1 system message")
    print(f"  - 1 LLM-generated summary")
    print(f"  - 20 recent messages (10 pairs)")
    print(f"  - Total: 22 messages\n")

    # Show the LLM summary
    summary = history[1]  # Second message (after system)
    print("LLM-generated summary content:")
    print(f"  Role: {summary['role']}")
    print(f"  Content: {summary['content']}\n")

    # Show LLM call details
    print("LLM call details:")
    print(f"  Model: gpt-4o-mini")
    print(f"  Temperature: 0.3")
    print(f"  Max tokens: 200")
    print(f"  Estimated cost: ~$0.0003 per compression\n")


def demo_comparison():
    """Side-by-side comparison of both modes."""
    print("=" * 70)
    print("Demo 3: Side-by-Side Comparison")
    print("=" * 70)

    print("\n" + "─" * 70)
    print("Stub Mode Output:")
    print("─" * 70)
    print("[Context summary: 5 earlier message(s) compressed to save context space. "
          "Key topics covered in prior turns are available in the session checkpoint.]")

    print("\n" + "─" * 70)
    print("LLM Summary Mode Output:")
    print("─" * 70)
    print("[Context summary: Earlier discussion covered fMRI preprocessing with FSL. "
          "User encountered FSLDIR environment variable issues, which were resolved "
          "by exporting FSLDIR from neuroclaw_environment.json. "
          "Successfully ran fslmaths on input data.]")

    print("\n" + "─" * 70)
    print("Comparison:")
    print("─" * 70)
    print("| Feature              | Stub Mode          | LLM Summary Mode        |")
    print("|----------------------|--------------------|-------------------------|")
    print("| Semantic info        | ❌ No              | ✅ Yes                  |")
    print("| LLM call required    | ❌ No              | ✅ Yes                  |")
    print("| Cost per compression | $0                 | ~$0.0003                |")
    print("| Speed                | Instant            | ~1-2 seconds            |")
    print("| Failure risk         | None               | Low (auto-fallback)     |")
    print("| User experience      | Basic              | Rich                    |")
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("NeuroClaw Context Compression: Stub vs LLM Summary")
    print("=" * 70 + "\n")

    try:
        demo_stub_mode()
        print()
        demo_llm_summary_mode()
        print()
        demo_comparison()

        print("=" * 70)
        print("All demos completed successfully! ✓")
        print("=" * 70)
        print("\nTo enable LLM summary mode in your NeuroClaw installation:")
        print("1. Add to neuroclaw_environment.json:")
        print('   "compression_mode": "llm_summary"')
        print("\n2. Or keep default stub mode (zero cost)")
        print()

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
