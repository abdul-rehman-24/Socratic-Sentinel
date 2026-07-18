/**
 * client.js — Axios API client for the Socratic Sentinel backend.
 *
 * Centralizes all backend communication. All components import from here
 * rather than calling fetch/axios directly. This makes it easy to swap
 * the base URL or add auth headers in one place.
 *
 * The Vite dev server proxies /api/* to http://localhost:8000,
 * so BASE_URL is just '' in development (no CORS needed).
 */

import axios from 'axios';

const apiClient = axios.create({
  baseURL: '',          // Vite proxy handles routing to :8000
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,       // 30s timeout — LLM calls can be slow
});

/**
 * POST /api/chat — Send a user message, get a Socratic response.
 *
 * @param {string} sessionId  - UUID for the current session
 * @param {string} message    - User's message text
 * @param {string} messageType - 'concept_question' | 'wrong_answer' | 'general'
 * @returns {Promise<ChatResponse>} - Full response object per API contract
 */
export async function sendChatMessage(sessionId, message, messageType = 'general') {
  const response = await apiClient.post('/api/chat', {
    session_id: sessionId,
    message,
    message_type: messageType,
  });
  return response.data;
}

/**
 * GET /api/graph/{sessionId} — Fetch the full knowledge graph for a session.
 *
 * Returns data in react-force-graph-2d format: { nodes: [...], links: [...] }
 * Assign directly to the ForceGraph2D `graphData` prop.
 *
 * @param {string} sessionId - UUID for the current session
 * @returns {Promise<GraphResponse>} - { session_id, graph: { nodes, links } }
 */
export async function fetchGraph(sessionId) {
  const response = await apiClient.get(`/api/graph/${sessionId}`);
  return response.data;
}

/**
 * GET /api/graph/{sessionId}/node/{conceptId} — Fetch detail for a specific node.
 *
 * Returns node metadata plus interaction history (for side panel expansion).
 *
 * @param {string} sessionId - UUID for the current session
 * @param {string} conceptId - Concept node ID
 * @returns {Promise<NodeDetailResponse>} - { session_id, concept_id, label, mastery, status, content_available, history }
 */
export async function fetchNodeDetail(sessionId, conceptId) {
  const response = await apiClient.get(`/api/graph/${sessionId}/node/${conceptId}`);
  return response.data;
}

export default apiClient;
