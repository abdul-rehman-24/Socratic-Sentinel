/**
 * MessageBubble.jsx — Individual chat message with role, text, and metadata.
 *
 * Renders differently for user (right-aligned violet) vs bot (left-aligned dark card).
 * Bot messages include the ConfidenceBadge, response type label, and mastery deltas.
 */

import { ConfidenceBadge } from './ConfidenceBadge';
import { MasteryBadge } from './MasteryBadge';

const RESPONSE_TYPE_LABELS = {
  socratic_question: 'Socratic Question',
  misconception_diagnosis: 'Misconception Detected',
  out_of_scope: 'Out of Scope',
  general: 'Response',
  error: 'Error',
};

/**
 * @param {{ message: { role: string, text: string, confidence: object, responseType: string, mastery_deltas: Array } }} props
 */
export function MessageBubble({ message }) {
  const isUser = message.role === 'user';

  return (
    <div className={`message message--${isUser ? 'user' : 'bot'}`}>
      <div className="message__bubble">
        {message.text}
      </div>

      {!isUser && (
        <div className="message__meta">
          {message.responseType && (
            <span className="message__type-label">
              {RESPONSE_TYPE_LABELS[message.responseType] || message.responseType}
            </span>
          )}
          {message.confidence && (
            <ConfidenceBadge confidence={message.confidence} />
          )}
        </div>
      )}

      {!isUser && message.mastery_deltas && message.mastery_deltas.length > 0 && (
        <div className="message__mastery-deltas">
          {message.mastery_deltas.map((delta, idx) => (
            <MasteryBadge
              key={idx}
              conceptName={delta.label}
              masteryBefore={delta.mastery_before}
              masteryAfter={delta.mastery}
            />
          ))}
        </div>
      )}
    </div>
  );
}

