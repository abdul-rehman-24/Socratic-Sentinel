/**
 * useChat.js — Custom hook for managing chat state.
 *
 * Encapsulates: message list, loading state, session ID,
 * and the sendMessage action that calls the API client.
 * Components import this hook rather than managing API state directly.
 */

import { useState, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { sendChatMessage } from '../api/client';

// Generate a session ID once per browser session (persists on refresh via sessionStorage).
function getOrCreateSessionId() {
  const existing = sessionStorage.getItem('sentinel_session_id');
  if (existing) return existing;
  const newId = uuidv4();
  sessionStorage.setItem('sentinel_session_id', newId);
  return newId;
}

/**
 * @returns {{
 *   messages: Array,
 *   isLoading: boolean,
 *   sessionId: string,
 *   sendMessage: (text: string) => Promise<void>,
 *   graphDelta: object | null,
 * }}
 */
export function useChat() {
  const [sessionId] = useState(getOrCreateSessionId);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [graphDelta, setGraphDelta] = useState(null);

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || isLoading) return;

    // Optimistically add the user's message immediately
    const userMsg = {
      id: uuidv4(),
      role: 'user',
      text,
      timestamp: Date.now(),
    };
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const data = await sendChatMessage(sessionId, text, 'concept_question');

      // Extract mastery deltas from updated_nodes if available
      const mastery_deltas = data.knowledge_graph_delta?.updated_nodes?.map(node => ({
        label: node.label,
        mastery: node.mastery,
        mastery_before: node.mastery_before ?? 0,
      })) || [];

      const botMsg = {
        id: uuidv4(),
        role: 'bot',
        text: data.response_text,
        responseType: data.response_type,
        confidence: data.confidence,
        curriculumRefs: data.curriculum_refs,
        mastery_deltas: mastery_deltas,
        timestamp: Date.now(),
      };

      setMessages(prev => [...prev, botMsg]);
      setGraphDelta(data.knowledge_graph_delta);
    } catch (err) {
      const errMsg = {
        id: uuidv4(),
        role: 'bot',
        text: '⚠️ Could not reach the backend. Is the FastAPI server running on :8000?',
        responseType: 'error',
        confidence: null,
        timestamp: Date.now(),
      };
      setMessages(prev => [...prev, errMsg]);
      console.error('[useChat] API error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, isLoading]);

  return { messages, isLoading, sessionId, sendMessage, graphDelta };
}
