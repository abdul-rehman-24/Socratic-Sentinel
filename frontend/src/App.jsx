/**
 * App.jsx — Root application component.
 *
 * Two-column layout:
 * - Left: Chat panel (ChatWindow + InputBar)
 * - Right: GraphPanel (knowledge graph visualization)
 *
 * The useChat hook owns all chat state. graphDelta from chat responses
 * is passed down to GraphPanel to update node mastery live.
 */

import { useChat } from './hooks/useChat';
import { ChatWindow } from './components/ChatWindow';
import { InputBar } from './components/InputBar';
import { GraphPanel } from './components/GraphPanel';

export default function App() {
  const { messages, isLoading, sessionId, sendMessage, graphDelta } = useChat();

  return (
    <>
      {/* App Header */}
      <header className="app-header">
        <span style={{ fontSize: 20 }}>🧠</span>
        <h1>Socratic Sentinel</h1>
        <span className="header-subtitle">— Deep Learning Tutor</span>
        <span className="header-badge">Skeleton v0.1</span>
      </header>

      {/* Main Layout */}
      <div className="app-layout">
        {/* Left: Chat Panel */}
        <section className="chat-panel" aria-label="Chat interface">
          <ChatWindow messages={messages} isLoading={isLoading} />
          <InputBar onSend={sendMessage} isLoading={isLoading} />
        </section>

        {/* Right: Knowledge Graph */}
        <aside aria-label="Knowledge graph">
          <GraphPanel sessionId={sessionId} graphDelta={graphDelta} />
        </aside>
      </div>
    </>
  );
}
