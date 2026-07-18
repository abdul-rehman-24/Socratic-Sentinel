/**
 * ChatWindow.jsx — Scrollable list of chat messages.
 *
 * Renders all messages in order, auto-scrolls to the latest message,
 * and shows a loading indicator while the bot is responding.
 */

import { useEffect, useRef } from 'react';
import { MessageBubble } from './MessageBubble';

/**
 * @param {{ messages: Array, isLoading: boolean }} props
 */
export function ChatWindow({ messages, isLoading }) {
  const bottomRef = useRef(null);

  // Auto-scroll to bottom whenever messages update
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="chat-messages" role="log" aria-live="polite" aria-label="Chat messages">
      {messages.length === 0 && !isLoading && (
        <div className="empty-state">
          <div className="empty-state__icon">🧠</div>
          <div className="empty-state__title">Socratic Sentinel</div>
          <div className="empty-state__subtitle">
            Ask me anything about deep learning fundamentals. 
            I won't give you the answer — I'll help you find it yourself.
          </div>
        </div>
      )}

      {messages.map(msg => (
        <MessageBubble key={msg.id} message={msg} />
      ))}

      {isLoading && (
        <div className="message message--bot">
          <div className="message__bubble" style={{ padding: '8px 12px' }}>
            <div className="loading-dots" aria-label="Thinking...">
              <span /><span /><span />
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
