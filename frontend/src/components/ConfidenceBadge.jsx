/**
 * ConfidenceBadge.jsx — Green/yellow/red grounding indicator.
 *
 * Displays a colored pill showing whether the bot's response is grounded
 * in curriculum content (GREEN), partially grounded (YELLOW), or
 * potentially hallucinated / out-of-scope (RED).
 *
 * Shows a tooltip with the full confidence explanation on hover.
 */

import { useState } from 'react';

const LEVEL_LABELS = {
  green: '● Grounded',
  yellow: '● Partial',
  red: '● Ungrounded',
};

/**
 * @param {{ confidence: { level: string, score: number, explanation: string } }} props
 */
export function ConfidenceBadge({ confidence }) {
  const [showTooltip, setShowTooltip] = useState(false);

  if (!confidence) return null;

  const { level, score, explanation } = confidence;

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <span
        id={`confidence-badge-${level}-${Math.floor(score * 100)}`}
        className={`confidence-badge confidence-badge--${level}`}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        role="status"
        aria-label={`Confidence: ${level} (${Math.round(score * 100)}%)`}
      >
        <span className="confidence-badge__dot" />
        {LEVEL_LABELS[level] || level}
        <span style={{ opacity: 0.7, marginLeft: 2 }}>{Math.round(score * 100)}%</span>
      </span>

      {showTooltip && (
        <div
          style={{
            position: 'absolute',
            bottom: 'calc(100% + 6px)',
            left: '50%',
            transform: 'translateX(-50%)',
            background: '#1a1d28',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '8px',
            padding: '8px 12px',
            fontSize: '11px',
            color: '#e8eaf0',
            whiteSpace: 'nowrap',
            maxWidth: '260px',
            whiteSpace: 'normal',
            zIndex: 100,
            lineHeight: 1.5,
            boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
          }}
        >
          {explanation}
        </div>
      )}
    </div>
  );
}
