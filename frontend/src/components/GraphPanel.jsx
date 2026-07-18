/**
 * GraphPanel.jsx — Knowledge graph visualization using react-force-graph-2d.
 *
 * Renders the per-session concept mastery graph.
 * Node color encodes mastery status (grey → violet → teal → green).
 * Node size scales with mastery level.
 * Nodes with content_available=false show reduced opacity and dashed border.
 *
 * Day 3: Click a node to expand a detail panel showing mastery %, interaction history.
 * Frontend listens to graph deltas from chat responses and updates live.
 */

import { useRef, useCallback, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { useGraph } from '../hooks/useGraph';
import { NodeDetailPanel } from './NodeDetailPanel';

const STATUS_COLORS = {
  not_started: '#2d2f3f',
  learning: '#6c63ff',
  practiced: '#00d4aa',
  mastered: '#22c55e',
};

/**
 * @param {{ sessionId: string, graphDelta: object | null }} props
 */
export function GraphPanel({ sessionId, graphDelta }) {
  const containerRef = useRef(null);
  const graphRef = useRef(null);
  const { graphData, isLoading, error } = useGraph(sessionId, graphDelta);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);

  const nodeCanvasObject = useCallback((node, ctx, globalScale) => {
    const mastery = node.mastery ?? 0;
    const radius = 5 + mastery * 6; // 5px base, up to 11px at full mastery
    const color = STATUS_COLORS[node.status] || STATUS_COLORS.not_started;
    const contentAvailable = node.content_available !== false;

    // Adjust opacity for nodes without content
    const baseFill = contentAvailable ? 1.0 : 0.4;
    ctx.globalAlpha = baseFill;

    // Node circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    // Dashed border for nodes without content, solid for those with content
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius + 1, 0, 2 * Math.PI);
    if (!contentAvailable) {
      ctx.setLineDash([2, 2]);
    }
    ctx.strokeStyle = `${color}aa`;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);

    // Glow ring at higher mastery
    if (mastery > 0.3) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius + 2, 0, 2 * Math.PI);
      ctx.strokeStyle = `${color}55`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Reset opacity before label
    ctx.globalAlpha = 1.0;

    const showLabel = node === selectedNode || node === hoveredNode;
    if (showLabel) {
      const label = node.label ?? node.id;
      const fontSize = Math.max(10, 14 / globalScale);
      ctx.font = `bold ${fontSize}px Inter, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#e8eaf0';
      ctx.fillText(label, node.x, node.y + radius + fontSize * 1.2);
    }
  }, [hoveredNode, selectedNode]);

  const handleNodeClick = useCallback((node) => {
    setSelectedNode(node);
  }, []);

  const handleNodeHover = useCallback((node) => {
    setHoveredNode(node);
  }, []);

  const handleCloseDetail = useCallback(() => {
    setSelectedNode(null);
  }, []);

  return (
    <div className="graph-panel">
      <div className="graph-panel__header">
        <div className="graph-panel__title">Knowledge Graph</div>
        {selectedNode && (
          <div className="graph-panel__node-badge">
            {selectedNode.label} · {Math.round((selectedNode.mastery ?? 0) * 100)}%
          </div>
        )}
      </div>

      <div className="graph-panel__container">
        <div className="graph-panel__canvas" ref={containerRef} id="knowledge-graph-canvas">
          {error && (
            <div className="empty-state">
              <div className="empty-state__icon">⚠️</div>
              <div className="empty-state__title">Graph unavailable</div>
              <div className="empty-state__subtitle">{error}</div>
            </div>
          )}

          {!error && (
            <>
              <ForceGraph2D
                ref={graphRef}
                graphData={graphData}
                nodeId="id"
                nodeCanvasObject={nodeCanvasObject}
                nodeCanvasObjectMode={() => 'replace'}
                linkColor={() => 'rgba(255,255,255,0.08)'}
                linkWidth={1}
                linkDirectionalArrowLength={4}
                linkDirectionalArrowRelPos={1}
                backgroundColor="#12141c"
                warmupTicks={50}
                cooldownTicks={200}
                width={380}
                height={
                  containerRef.current
                    ? containerRef.current.clientHeight
                    : 500
                }
                onNodeClick={handleNodeClick}
                onNodeHover={handleNodeHover}
              />

              {hoveredNode && hoveredNode !== selectedNode && (
                <div className="graph-tooltip" data-node-id={hoveredNode.id}>
                  <div className="graph-tooltip__title">{hoveredNode.label}</div>
                  <div className="graph-tooltip__meta">{hoveredNode.status.replace('_', ' ')} · {Math.round((hoveredNode.mastery ?? 0) * 100)}%</div>
                </div>
              )}
            </>
          )}
        </div>

        {selectedNode && (
          <NodeDetailPanel
            sessionId={sessionId}
            node={selectedNode}
            onClose={handleCloseDetail}
          />
        )}
      </div>

      <div className="graph-panel__legend">
        {Object.entries(STATUS_COLORS).map(([status, color]) => (
          <div key={status} className="legend-item">
            <div className="legend-dot" style={{ background: color }} />
            <span style={{ textTransform: 'capitalize' }}>{status.replace('_', ' ')}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
