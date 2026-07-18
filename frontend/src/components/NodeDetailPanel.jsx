/**
 * NodeDetailPanel.jsx — Expandable side panel showing concept node details.
 *
 * Shows:
 *   - Concept name and current mastery %
 *   - Status label (not started, learning, practiced, mastered)
 *   - Content availability indicator (with warning if no content)
 *   - Interaction history (what the student did on this concept)
 *
 * Fetches node detail from GET /api/graph/{sessionId}/node/{conceptId}
 * on mount and updates live when graph deltas come in.
 */

import { useEffect, useState, useCallback } from 'react';
import { fetchNodeDetail } from '../api/client';

/**
 * @param {{ sessionId: string, node: object, onClose: () => void }} props
 */
export function NodeDetailPanel({ sessionId, node, onClose }) {
  const [nodeDetail, setNodeDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!node?.id) {
      setError('No node selected');
      return;
    }

    setLoading(true);
    setError(null);

    fetchNodeDetail(sessionId, node.id)
      .then(data => {
        setNodeDetail(data);
      })
      .catch(err => {
        console.error('[NodeDetailPanel] fetch error:', err);
        setError('Could not load node details');
      })
      .finally(() => {
        setLoading(false);
      });
  }, [sessionId, node?.id]);

  // Update mastery from upstream node prop when graph deltas come in
  useEffect(() => {
    if (node && nodeDetail) {
      setNodeDetail(prev => ({
        ...prev,
        mastery: node.mastery,
        status: node.status,
      }));
    }
  }, [node?.mastery, node?.status]);

  const handleBackdropClick = useCallback((e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  }, [onClose]);

  if (loading) {
    return (
      <div className="node-detail-panel" onClick={handleBackdropClick}>
        <div className="node-detail-panel__content">
          <div className="loading-spinner">Loading...</div>
        </div>
      </div>
    );
  }

  if (error || !nodeDetail) {
    return (
      <div className="node-detail-panel" onClick={handleBackdropClick}>
        <div className="node-detail-panel__content">
          <div className="node-detail-panel__header">
            <button className="node-detail-panel__close" onClick={onClose}>✕</button>
          </div>
          <div className="error-message">{error}</div>
        </div>
      </div>
    );
  }

  const masteryPercent = Math.round((nodeDetail.mastery ?? 0) * 100);
  const contentStatusIcon = nodeDetail.content_available ? '✓' : '⚠';

  return (
    <div className="node-detail-panel" onClick={handleBackdropClick}>
      <div className="node-detail-panel__content">
        <div className="node-detail-panel__header">
          <div>
            <h3 className="node-detail-panel__title">{nodeDetail.label}</h3>
            <p className="node-detail-panel__subtitle">
              Mastery: <span className="mastery-badge">{masteryPercent}%</span>
            </p>
          </div>
          <button className="node-detail-panel__close" onClick={onClose}>✕</button>
        </div>

        <div className="node-detail-panel__status">
          <div className="status-item">
            <span className="status-label">Status:</span>
            <span className={`status-value status-${nodeDetail.status}`}>
              {nodeDetail.status.replace('_', ' ').toUpperCase()}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Content:</span>
            <span className={`status-value ${nodeDetail.content_available ? 'available' : 'unavailable'}`}>
              {contentStatusIcon} {nodeDetail.content_available ? 'Available' : 'Not Yet Available'}
            </span>
          </div>
        </div>

        {!nodeDetail.content_available && (
          <div className="warning-box">
            <div className="warning-box__icon">ℹ</div>
            <div className="warning-box__text">
              This concept is not yet fully covered in the curriculum. Focus on{' '}
              <strong>Backpropagation</strong> for now — it has comprehensive materials.
            </div>
          </div>
        )}

        <div className="node-detail-panel__section">
          <h4 className="node-detail-panel__section-title">
            Interaction History ({nodeDetail.interaction_count})
          </h4>
          {nodeDetail.history && nodeDetail.history.length > 0 ? (
            <div className="interaction-history">
              {nodeDetail.history.map((entry, idx) => (
                <div key={idx} className="history-entry">
                  <div className="history-entry__header">
                    <span className="history-intent">{entry.intent.toUpperCase()}</span>
                    <span className="history-delta">
                      {entry.mastery_before.toFixed(0)}% → {entry.mastery_after.toFixed(0)}%
                    </span>
                  </div>
                  <div className="history-entry__preview">{entry.preview}</div>
                </div>
              ))}
            </div>
          ) : (
            <p className="empty-history">No interactions yet on this concept.</p>
          )}
        </div>
      </div>
    </div>
  );
}
