/**
 * useGraph.js — Custom hook for fetching and updating the knowledge graph.
 *
 * Fetches the initial graph on mount, then applies graph deltas received
 * from chat responses to keep the visualization live without a full refetch.
 */

import { useState, useEffect, useCallback } from 'react';
import { fetchGraph } from '../api/client';

const STATUS_COLORS = {
  not_started: '#2d2f3f',
  learning: '#6c63ff',
  practiced: '#00d4aa',
  mastered: '#22c55e',
};

/**
 * @param {string} sessionId - The current session ID
 * @param {object | null} graphDelta - Delta from the latest chat response
 * @returns {{ graphData: object, isLoading: boolean, error: string | null }}
 */
export function useGraph(sessionId, graphDelta) {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch full graph on session start
  useEffect(() => {
    if (!sessionId) return;

    let cancelled = false;
    setIsLoading(true);

    fetchGraph(sessionId)
      .then(data => {
        if (cancelled) return;
        // Enrich nodes with color for react-force-graph-2d
        const enriched = {
          ...data.graph,
          nodes: data.graph.nodes.map(n => ({
            ...n,
            color: STATUS_COLORS[n.status] || STATUS_COLORS.not_started,
          })),
        };
        setGraphData(enriched);
        setError(null);
      })
      .catch(err => {
        if (cancelled) return;
        setError('Could not load knowledge graph.');
        console.error('[useGraph] fetch error:', err);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => { cancelled = true; };
  }, [sessionId]);

  // Apply graph deltas from chat responses without full refetch
  useEffect(() => {
    if (!graphDelta?.updated_nodes?.length) return;

    setGraphData(prev => {
      const updatedMap = Object.fromEntries(
        graphDelta.updated_nodes.map(n => [n.id, n])
      );
      return {
        ...prev,
        nodes: prev.nodes.map(n =>
          updatedMap[n.id]
            ? {
                ...n,
                ...updatedMap[n.id],
                color: STATUS_COLORS[updatedMap[n.id].status] || n.color,
              }
            : n
        ),
      };
    });
  }, [graphDelta]);

  return { graphData, isLoading, error };
}
