#!/usr/bin/env python
"""
verify_day3_features.py — Quick verification script for Day 3 features.

Run this in a Python shell after starting the backend and completing the
4-turn dialogue in the frontend UI.

Usage:
  cd backend
  python
  >>> exec(open('scripts/verify_day3_features.py').read())
  >>> verify_conversation_memory("<session-id>")
  >>> verify_adaptive_difficulty("<session-id>")
  >>> verify_graph_deltas("<session-id>")
"""

from app.services import session_memory, knowledge_graph, difficulty_adapter


def verify_conversation_memory(session_id: str):
    """Verify that conversation history is stored and retrievable."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Conversation Memory")
    print("=" * 60)
    
    # Get history
    history = session_memory.get_history(session_id)
    print(f"\n✓ Session {session_id}: {len(history)} messages stored")
    
    if len(history) == 0:
        print("  ⚠ WARNING: No history found. Have you completed turns in the UI?")
        return
    
    # Verify structure
    for i, msg in enumerate(history):
        role = msg.get("role", "?")
        content_preview = msg.get("content", "")[:50]
        intent = msg.get("intent", "N/A") if role == "user" else "N/A"
        print(f"  [{i+1}] {role.upper():10} intent={intent:20} content='{content_preview}...'")
    
    # Format for prompt
    formatted = session_memory.format_history_for_prompt(session_id)
    print(f"\n✓ Formatted history for prompt ({len(formatted)} chars):")
    print(f"  {formatted[:100]}...")
    
    # OpenAI format
    openai_hist = session_memory.get_openai_history(session_id)
    print(f"\n✓ OpenAI format history: {len(openai_hist)} messages")
    for msg in openai_hist:
        assert "role" in msg and "content" in msg
        assert "intent" not in msg  # Metadata stripped
    print("  All messages have role + content only (no metadata)")


def verify_adaptive_difficulty(session_id: str):
    """Verify that escalation logic works correctly."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Adaptive Difficulty")
    print("=" * 60)
    
    # Get graph
    graph = knowledge_graph._session_graphs.get(session_id)
    if not graph:
        print("  ⚠ WARNING: No graph found for session")
        return
    
    # Check backpropagation mastery
    bp_mastery = graph.nodes.get("backpropagation", {}).get("mastery", 0.0)
    print(f"\n✓ Backpropagation mastery: {bp_mastery:.1%}")
    
    # Test escalation threshold
    THRESHOLD = 0.60
    if bp_mastery < THRESHOLD:
        print(f"  ✓ Mastery < {THRESHOLD:.0%}: No escalation (expected)")
        target = difficulty_adapter.get_escalation_target(session_id, "backpropagation")
        assert target is None, "Should not escalate below threshold"
    else:
        print(f"  ✓ Mastery >= {THRESHOLD:.0%}: Escalation check triggered")
        target = difficulty_adapter.get_escalation_target(session_id, "backpropagation")
        if target:
            print(f"    → Escalation target: {target}")
            note = difficulty_adapter.build_escalation_note(
                session_id, "backpropagation", target
            )
            print(f"    → Note preview: {note[:80]}...")
        else:
            print(f"    → No escalation target (all successors mastered)")


def verify_graph_deltas(session_id: str):
    """Verify that graph deltas include mastery_before information."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Graph Deltas")
    print("=" * 60)
    
    graph = knowledge_graph._session_graphs.get(session_id)
    if not graph:
        print("  ⚠ WARNING: No graph found for session")
        return
    
    # Get node history
    node_hist = knowledge_graph.get_node_history(session_id, "backpropagation")
    print(f"\n✓ Backpropagation interaction history: {len(node_hist)} entries")
    
    if len(node_hist) == 0:
        print("  ⚠ WARNING: No interaction history recorded")
        return
    
    for i, entry in enumerate(node_hist):
        before = entry.get("mastery_before", "?")
        after = entry.get("mastery_after", "?")
        intent = entry.get("intent", "?")
        preview = entry.get("preview", "")[:40]
        print(f"  [{i+1}] {intent:20} {before:.1%} → {after:.1%}  | {preview}...")
    
    # Verify deltas make sense
    prev_mastery = 0.0
    for entry in node_hist:
        before = entry.get("mastery_before", 0.0)
        after = entry.get("mastery_after", 0.0)
        assert before >= 0 and after <= 1, f"Invalid mastery values: {before} → {after}"
        # Either +0.15, +0.10, or -0.05 (concept_question, general, or wrong_answer)
        delta = after - before
        assert delta in (0.10, 0.15, -0.05, 0.0), f"Unexpected delta: {delta}"


def verify_content_availability(session_id: str):
    """Verify that nodes have content_available flags."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Content Availability Flags")
    print("=" * 60)
    
    import asyncio
    
    async def go():
        graph_data = await knowledge_graph.get_session_graph(session_id)
        print(f"\n✓ Graph has {len(graph_data.nodes)} nodes")
        
        available_count = 0
        unavailable_count = 0
        
        for node in graph_data.nodes:
            if node.content_available:
                available_count += 1
                marker = " ✓"
            else:
                unavailable_count += 1
                marker = " ✗"
            status = f"{marker} {node.label[:25]:25} content_available={node.content_available}"
            if node.mastery > 0:
                status += f" mastery={node.mastery:.0%}"
            print(f"  {status}")
        
        print(f"\n✓ Summary: {available_count} available, {unavailable_count} not yet available")
        assert available_count > 0, "No nodes with content available"
        assert unavailable_count > 0, "No nodes without content (should have some placeholders)"
    
    asyncio.run(go())


def verify_all(session_id: str):
    """Run all verification checks."""
    print("\n\n")
    print("╔" + "=" * 58 + "╗")
    print("║  DAY 3 VERIFICATION — Conversation Memory & Adaptive Difficulty  ║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        verify_conversation_memory(session_id)
        verify_adaptive_difficulty(session_id)
        verify_graph_deltas(session_id)
        verify_content_availability(session_id)
        
        print("\n\n")
        print("╔" + "=" * 58 + "╗")
        print("║  ✅ ALL VERIFICATIONS PASSED                                    ║")
        print("╚" + "=" * 58 + "╝")
    except Exception as e:
        print(f"\n\n✗ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Example usage
    session_id = input("Enter session ID (from sessionStorage): ").strip()
    if session_id:
        verify_all(session_id)
    else:
        print("No session ID provided")
