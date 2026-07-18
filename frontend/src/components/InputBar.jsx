/**
 * InputBar.jsx — Message input with send button.
 *
 * Supports Enter to send (Shift+Enter for newline),
 * auto-resize textarea, and disabled state during loading.
 */

import { useState, useRef, useCallback } from 'react';

/**
 * @param {{ onSend: (text: string) => void, isLoading: boolean }} props
 */
export function InputBar({ onSend, isLoading }) {
  const [value, setValue] = useState('');
  const textareaRef = useRef(null);

  const handleSend = useCallback(() => {
    const text = value.trim();
    if (!text || isLoading) return;
    onSend(text);
    setValue('');
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [value, isLoading, onSend]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleChange = useCallback((e) => {
    setValue(e.target.value);
    // Auto-resize textarea
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = 'auto';
      ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`;
    }
  }, []);

  return (
    <div className="input-bar">
      <textarea
        ref={textareaRef}
        id="chat-input"
        className="input-bar__textarea"
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder="Ask about backpropagation, CNNs, optimizers… (Enter to send)"
        disabled={isLoading}
        rows={1}
        aria-label="Type your message"
      />
      <button
        id="chat-send-btn"
        className="input-bar__send"
        onClick={handleSend}
        disabled={isLoading || !value.trim()}
        aria-label="Send message"
        title="Send (Enter)"
      >
        ↑
      </button>
    </div>
  );
}
