/**
 * MasteryBadge.jsx — Small inline badge showing mastery level change.
 *
 * Example: "Backpropagation: 45% → 58%"
 * Displayed after chat responses with mastery updates.
 */

/**
 * @param {{ conceptName: string, masteryBefore: number, masteryAfter: number }} props
 */
export function MasteryBadge({ conceptName, masteryBefore, masteryAfter }) {
  const beforePercent = Math.round(masteryBefore * 100);
  const afterPercent = Math.round(masteryAfter * 100);
  const isImprovement = masteryAfter > masteryBefore;
  const delta = Math.abs(afterPercent - beforePercent);

  return (
    <span className={`mastery-badge ${isImprovement ? 'improvement' : 'penalty'}`}>
      <span className="concept-name">{conceptName}</span>
      <span className="mastery-progression">
        {beforePercent}% → {afterPercent}%
      </span>
      <span className={`delta-icon ${isImprovement ? 'up' : 'down'}`}>
        {isImprovement ? '↑' : '↓'} {delta}%
      </span>
    </span>
  );
}
