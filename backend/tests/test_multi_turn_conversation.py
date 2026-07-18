"""
tests/test_multi_turn_conversation.py — Multi-turn conversation flow with memory & adaptive difficulty.

This test verifies Day 3 core features:
  1. Conversation memory: Last 3-4 turns stored and retrieved per session
  2. History injection: Prior turns passed to prompt templates as context
  3. Adaptive difficulty: When mastery exceeds 0.6, escalation to adjacent concept is triggered
  4. Interaction history: Per-node interaction records for click-to-expand in frontend

This is the primary manual verification blueprint for the hackathon demo.
"""

import pytest
from app.services import (
    session_memory,
    knowledge_graph,
    difficulty_adapter,
)


def setup_function():
    """Clear state before each test to ensure isolation."""
    session_memory._histories.clear()
    knowledge_graph._session_graphs.clear()
    knowledge_graph._node_histories.clear()


class TestConversationMemory:
    """Test that conversation history is stored, retrieved, and formatted correctly."""

    def test_memory_stores_first_turn(self):
        """First turn is stored with user + assistant messages."""
        sid = "multi-turn-1"
        session_memory.add_turn(
            sid,
            user_message="What is backpropagation?",
            assistant_response="Let's think about how gradients flow...",
            intent="concept_question",
            response_type="socratic_question"
        )
        hist = session_memory.get_history(sid)
        assert len(hist) == 2
        assert hist[0]["role"] == "user"
        assert hist[1]["role"] == "assistant"

    def test_memory_stores_multiple_turns_sequentially(self):
        """Multiple turns accumulate correctly (FIFO with eviction)."""
        sid = "multi-turn-2"
        for i in range(3):
            session_memory.add_turn(
                sid,
                user_message=f"Question {i}",
                assistant_response=f"Response {i}",
                intent="concept_question",
                response_type="socratic_question"
            )
        hist = session_memory.get_history(sid)
        # 3 turns × 2 messages each = 6 messages
        assert len(hist) == 6
        # Verify order (newest last)
        assert "Question 0" in [m["content"] for m in hist]
        assert "Question 2" in [m["content"] for m in hist]

    def test_memory_evicts_old_turns_at_limit(self):
        """Exceeding MAX_TURNS (4) evicts oldest turn pair."""
        sid = "multi-turn-3"
        # Add 5 pairs (10 messages) when max is 4 pairs (8 messages)
        for i in range(5):
            session_memory.add_turn(
                sid,
                user_message=f"Q{i}",
                assistant_response=f"A{i}"
            )
        hist = session_memory.get_history(sid)
        # Should keep only last 4 pairs = 8 messages
        assert len(hist) == 8
        # Q0/A0 should be evicted
        assert all("Q0" not in m["content"] for m in hist)
        # Q4/A4 should be present
        assert any("Q4" in m["content"] for m in hist)

    def test_format_history_for_prompt_preserves_order(self):
        """format_history_for_prompt returns chronological dialogue string."""
        sid = "multi-turn-4"
        session_memory.add_turn(
            sid,
            user_message="What is the chain rule?",
            assistant_response="Think about how the derivative of a composite function works..."
        )
        session_memory.add_turn(
            sid,
            user_message="Is it like multiplying derivatives?",
            assistant_response="Exactly — you're on the right track..."
        )
        formatted = session_memory.format_history_for_prompt(sid)
        # Verify both turns are present and in order
        assert "Student: What is the chain rule?" in formatted
        assert "Tutor: Think about how the derivative" in formatted
        assert "Student: Is it like multiplying" in formatted
        assert "Tutor: Exactly" in formatted
        # Verify order (first question appears before second)
        assert formatted.find("chain rule") < formatted.find("multiplying")

    def test_get_openai_history_strips_metadata(self):
        """get_openai_history returns only role/content (safe for GPT API)."""
        sid = "multi-turn-5"
        session_memory.add_turn(
            sid,
            user_message="Q1",
            assistant_response="A1",
            intent="concept_question",
            response_type="socratic_question"
        )
        openai_hist = session_memory.get_openai_history(sid)
        assert len(openai_hist) == 2
        # Should only have role and content
        for msg in openai_hist:
            assert "role" in msg
            assert "content" in msg
            assert "intent" not in msg
            assert "response_type" not in msg


class TestAdaptiveDifficultyEscalation:
    """Test that mastery-based escalation triggers correctly."""

    def test_no_escalation_below_threshold(self):
        """Mastery < 0.60 does not trigger escalation."""
        sid = "escalate-1"
        g = knowledge_graph._get_or_create(sid)
        g.nodes["backpropagation"]["mastery"] = 0.50

        target = difficulty_adapter.get_escalation_target(sid, "backpropagation")
        assert target is None

    def test_escalation_triggered_at_threshold(self):
        """Mastery >= 0.60 triggers escalation to unmastered successor."""
        sid = "escalate-2"
        g = knowledge_graph._get_or_create(sid)
        g.nodes["backpropagation"]["mastery"] = 0.60  # At threshold

        target = difficulty_adapter.get_escalation_target(sid, "backpropagation")
        # backpropagation → gradient_descent via 'extends'
        assert target == "gradient_descent"

    def test_escalation_picks_lowest_mastery(self):
        """When multiple unmastered successors exist, pick the lowest mastery one."""
        sid = "escalate-3"
        g = knowledge_graph._get_or_create(sid)
        g.nodes["backpropagation"]["mastery"] = 0.75

        # Give gradient_descent some mastery
        g.nodes["gradient_descent"]["mastery"] = 0.30
        # But don't modify optimizers (stays at 0.0)

        target = difficulty_adapter.get_escalation_target(sid, "backpropagation")
        # Should pick gradient_descent (the 'extends' successor with lowest mastery)
        assert target == "gradient_descent"

    def test_escalation_note_format(self):
        """build_escalation_note returns a properly formatted string."""
        sid = "escalate-4"
        knowledge_graph._get_or_create(sid)
        note = difficulty_adapter.build_escalation_note(
            sid, "backpropagation", "gradient_descent"
        )
        assert "ESCALATION NOTE" in note
        assert "Backpropagation" in note
        assert "Gradient Descent" in note
        assert "bridge" in note.lower()


class TestInteractionHistoryPerNode:
    """Test that node interaction history is recorded for click-to-expand frontend panel."""

    @pytest.mark.asyncio
    async def test_interaction_recorded_on_update(self):
        """Each interaction updates node history with intent, mastery delta, response preview."""
        sid = "interact-1"
        g = knowledge_graph._get_or_create(sid)

        # Perform an interaction
        delta = await knowledge_graph.update_from_interaction(
            session_id=sid,
            concept_ids=["backpropagation"],
            intent="concept_question",
            response_type="socratic_question",
            response_text="Think about what the forward pass needs to save for the backward pass.",
        )

        # Verify node was updated
        assert len(delta.updated_nodes) == 1
        assert delta.updated_nodes[0].id == "backpropagation"

        # Verify interaction history was recorded
        history = knowledge_graph.get_node_history(sid, "backpropagation")
        assert len(history) == 1
        assert history[0]["intent"] == "concept_question"
        assert history[0]["response_type"] == "socratic_question"
        assert history[0]["mastery_before"] == 0.0
        assert history[0]["mastery_after"] == 0.15  # +0.15 for concept_question

    @pytest.mark.asyncio
    async def test_multiple_interactions_accumulate(self):
        """Multiple interactions on the same node create separate history entries."""
        sid = "interact-2"

        # First interaction
        await knowledge_graph.update_from_interaction(
            sid, ["backpropagation"], "concept_question", "socratic_question", "Q1"
        )

        # Second interaction
        await knowledge_graph.update_from_interaction(
            sid, ["backpropagation"], "concept_question", "socratic_question", "Q2"
        )

        history = knowledge_graph.get_node_history(sid, "backpropagation")
        assert len(history) == 2
        assert history[0]["mastery_before"] == 0.0
        assert history[0]["mastery_after"] == 0.15
        assert history[1]["mastery_before"] == 0.15
        assert history[1]["mastery_after"] == 0.30

    @pytest.mark.asyncio
    async def test_interaction_history_preview_truncation(self):
        """Long response text is truncated to 120 chars in history."""
        sid = "interact-3"
        long_response = "This is a very long response that should be truncated when stored in the interaction history because we want to keep token usage and storage minimal for the preview display in the frontend click-to-expand panel. " * 2

        await knowledge_graph.update_from_interaction(
            sid, ["backpropagation"], "concept_question", "socratic_question", long_response
        )

        history = knowledge_graph.get_node_history(sid, "backpropagation")
        preview = history[0]["preview"]
        assert len(preview) <= 123  # 120 + "..."
        assert preview.endswith("...")


class TestMultiTurnEndToEnd:
    """Integration test simulating a realistic 4-turn conversation with mastery progression."""

    @pytest.mark.asyncio
    async def test_four_turn_dialogue_with_mastery_progression_and_escalation(self):
        """
        Realistic scenario:
         Turn 1: Student asks about backpropagation → mastery 0 → 0.15
         Turn 2: Student asks follow-up → mastery 0.15 → 0.30 (uses prior dialogue)
         Turn 3: Student asks another question → mastery 0.30 → 0.45
         Turn 4: Student makes progress → mastery 0.45 → 0.60
         Turn 5: Escalation should trigger (mastery at 0.60)
        """
        sid = "e2e-1"

        # ── Turn 1 ────────────────────────────────────────────────────────
        q1 = "What is backpropagation?"
        a1 = "Let's think about what the forward pass needs to save..."

        session_memory.add_turn(sid, q1, a1, "concept_question", "socratic_question")
        await knowledge_graph.update_from_interaction(
            sid, ["backpropagation"], "concept_question", "socratic_question", a1
        )

        hist1 = session_memory.get_history(sid)
        assert len(hist1) == 2
        assert "What is backpropagation?" in [m["content"] for m in hist1]

        node1 = knowledge_graph._session_graphs[sid].nodes["backpropagation"]
        assert node1["mastery"] == 0.15
        assert node1["status"] == "learning"

        # ── Turn 2 ────────────────────────────────────────────────────────
        q2 = "So does it go layer by layer?"
        a2 = "Good instinct. Now, in what order do you think those layers are visited?"

        session_memory.add_turn(sid, q2, a2, "concept_question", "socratic_question")
        await knowledge_graph.update_from_interaction(
            sid, ["backpropagation"], "concept_question", "socratic_question", a2
        )

        hist2 = session_memory.get_history(sid)
        assert len(hist2) == 4
        # Both turns should be present
        assert any("What is backpropagation?" in m["content"] for m in hist2)
        assert any("layer by layer" in m["content"] for m in hist2)

        # Verify history can be formatted for prompt injection
        formatted = session_memory.format_history_for_prompt(sid)
        assert "Student: What is backpropagation?" in formatted
        assert "Tutor: Let's think about" in formatted
        assert "Student: So does it go layer by layer?" in formatted

        node2 = knowledge_graph._session_graphs[sid].nodes["backpropagation"]
        assert node2["mastery"] == 0.30
        assert node2["status"] == "learning"

        # ── Turn 3 ────────────────────────────────────────────────────────
        q3 = "In reverse order, right?"
        a3 = "Yes — starting from the output layer. Why do you think that's necessary?"

        session_memory.add_turn(sid, q3, a3, "concept_question", "socratic_question")
        await knowledge_graph.update_from_interaction(
            sid, ["backpropagation"], "concept_question", "socratic_question", a3
        )

        node3 = knowledge_graph._session_graphs[sid].nodes["backpropagation"]
        assert node3["mastery"] == 0.45
        assert node3["status"] == "practiced"

        # ── Turn 4 ────────────────────────────────────────────────────────
        q4 = "Because we need the derivatives from the next layer?"
        a4 = "Exactly! You're grasping the chain rule connection here..."

        session_memory.add_turn(sid, q4, a4, "concept_question", "socratic_question")
        await knowledge_graph.update_from_interaction(
            sid, ["backpropagation"], "concept_question", "socratic_question", a4
        )

        node4 = knowledge_graph._session_graphs[sid].nodes["backpropagation"]
        assert node4["mastery"] == 0.60
        assert node4["status"] == "practiced"

        # ── Turn 5: Escalation should trigger ────────────────────────────
        escalation_target = difficulty_adapter.get_escalation_target(sid, "backpropagation")
        assert escalation_target is not None
        assert escalation_target == "gradient_descent"

        escalation_note = difficulty_adapter.build_escalation_note(
            sid, "backpropagation", escalation_target
        )
        assert "ESCALATION NOTE" in escalation_note

        # ── Verify full session state ────────────────────────────────────
        # 4 turns = 8 messages stored
        final_hist = session_memory.get_history(sid)
        assert len(final_hist) == 8

        # Node history should have 4 entries (one per interaction)
        node_hist = knowledge_graph.get_node_history(sid, "backpropagation")
        assert len(node_hist) == 4
        # Each entry should show the mastery progression
        assert node_hist[0]["mastery_after"] == 0.15
        assert node_hist[1]["mastery_after"] == 0.30
        assert node_hist[2]["mastery_after"] == 0.45
        assert node_hist[3]["mastery_after"] == 0.60

    @pytest.mark.asyncio
    async def test_wrong_answer_decrements_mastery(self):
        """Wrong answer interaction decrements mastery (−0.05)."""
        sid = "e2e-2"

        # First, build up some mastery
        await knowledge_graph.update_from_interaction(
            sid, ["backpropagation"], "concept_question", "socratic_question", "Q1"
        )
        assert knowledge_graph._session_graphs[sid].nodes["backpropagation"]["mastery"] == 0.15

        # Then a wrong answer
        await knowledge_graph.update_from_interaction(
            sid, ["backpropagation"], "wrong_answer", "misconception_diagnosis", "Wrong answer"
        )
        mastery = knowledge_graph._session_graphs[sid].nodes["backpropagation"]["mastery"]
        assert mastery == 0.10  # 0.15 - 0.05

    @pytest.mark.asyncio
    async def test_content_available_flag_on_nodes(self):
        """Nodes have content_available flag (True for backpropagation, False for others)."""
        sid = "e2e-3"
        graph_data = await knowledge_graph.get_session_graph(sid)

        # Find backpropagation node
        bp_node = next((n for n in graph_data.nodes if n.id == "backpropagation"), None)
        assert bp_node is not None
        assert bp_node.content_available is True

        # Find a node without content
        cnn_node = next((n for n in graph_data.nodes if n.id == "cnn_basics"), None)
        assert cnn_node is not None
        assert cnn_node.content_available is False
