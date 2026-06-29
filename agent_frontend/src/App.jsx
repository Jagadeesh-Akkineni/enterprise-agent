import { useState, useRef, useEffect } from 'react'

const API = 'http://localhost:8000'
const DEFAULT_EMAIL = 'user@enterprise.com'

function App() {
  const [sessions, setSessions]         = useState([])
  const [activeSession, setActiveSession] = useState(null)
  const [isPendingNewChat, setIsPendingNewChat] = useState(false) // new chat not yet saved
  const [messages, setMessages]         = useState([])
  const [input, setInput]               = useState('')
  const [isLoading, setIsLoading]       = useState(false)
  const [menuOpen, setMenuOpen]         = useState(null)   // session id whose ⋮ menu is open
  const [renamingId, setRenamingId]     = useState(null)   // session id being renamed inline
  const [renameValue, setRenameValue]   = useState('')
  const messagesEndRef = useRef(null)

  useEffect(() => { fetchSessions() }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Close dropdown when clicking anywhere outside it
  useEffect(() => {
    const close = () => setMenuOpen(null)
    document.addEventListener('click', close)
    return () => document.removeEventListener('click', close)
  }, [])

  // ─── API helpers ────────────────────────────────────────────────────────────

  async function fetchSessions() {
    try {
      const res = await fetch(`${API}/sessions?email=${DEFAULT_EMAIL}`)
      setSessions(await res.json())
    } catch { /* server not ready yet */ }
  }

  function startNewChat() {
    setActiveSession(null)
    setMessages([])
    setIsPendingNewChat(true)
    setMenuOpen(null)
  }

  function nameFromMessage(text) {
    const t = text.trim()
    if (t.length <= 45) return t
    const cut = t.slice(0, 45)
    const lastSpace = cut.lastIndexOf(' ')
    return (lastSpace > 20 ? cut.slice(0, lastSpace) : cut) + '...'
  }

  async function selectSession(session) {
    setIsPendingNewChat(false)
    setActiveSession(session)
    const res = await fetch(`${API}/sessions/${session.id}/messages`)
    setMessages(await res.json())
  }

  async function deleteSession(sessionId) {
    await fetch(`${API}/sessions/${sessionId}`, { method: 'DELETE' })
    setSessions(prev => prev.filter(s => s.id !== sessionId))
    if (activeSession?.id === sessionId) {
      setActiveSession(null)
      setMessages([])
    }
    setMenuOpen(null)
  }

  async function confirmRename(sessionId) {
    if (!renameValue.trim()) return
    const res = await fetch(`${API}/sessions/${sessionId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: renameValue.trim() }),
    })
    const updated = await res.json()
    setSessions(prev => prev.map(s => s.id === sessionId ? updated : s))
    if (activeSession?.id === sessionId) setActiveSession(updated)
    setRenamingId(null)
    setRenameValue('')
  }

  async function sendMessage(e) {
    e.preventDefault()
    if (!input.trim() || isLoading || (!activeSession && !isPendingNewChat)) return

    const userText = input.trim()
    setInput('')
    setMessages(prev => [...prev, {
      id: `tmp-${Date.now()}`, role: 'user', content: userText, citations: []
    }])
    setIsLoading(true)

    try {
      let sessionId = activeSession?.id

      if (isPendingNewChat) {
        // Create the session now, name derived from first message
        const res = await fetch(`${API}/sessions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: nameFromMessage(userText), email: DEFAULT_EMAIL }),
        })
        const newSession = await res.json()
        setSessions(prev => [newSession, ...prev])
        setActiveSession(newSession)
        setIsPendingNewChat(false)
        sessionId = newSession.id
      }

      const res = await fetch(`${API}/sessions/${sessionId}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userText }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, data.message])
    } catch {
      setMessages(prev => [...prev, {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        citations: [],
      }])
    } finally {
      setIsLoading(false)
    }
  }

  // ─── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="app-layout">

      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-top">
          <span className="sidebar-title">Enterprise Agent</span>
          <button className="new-chat-btn" onClick={startNewChat}>+ New Chat</button>
        </div>

        <div className="sessions-list">
          {sessions.length === 0 && (
            <p className="no-sessions">No chats yet. Start one!</p>
          )}
          {sessions.map(session => (
            <div
              key={session.id}
              className={`session-item ${activeSession?.id === session.id ? 'active' : ''}`}
              onClick={() => selectSession(session)}
            >
              {renamingId === session.id ? (
                <input
                  className="rename-input"
                  value={renameValue}
                  autoFocus
                  onClick={e => e.stopPropagation()}
                  onChange={e => setRenameValue(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter') confirmRename(session.id)
                    if (e.key === 'Escape') { setRenamingId(null); setRenameValue('') }
                  }}
                />
              ) : (
                <span className="session-name">{session.name}</span>
              )}

              {/* 3-dot menu */}
              <div
                className="menu-wrapper"
                onClick={e => e.stopPropagation()}
              >
                <button
                  className="dots-btn"
                  onClick={() => setMenuOpen(menuOpen === session.id ? null : session.id)}
                >⋮</button>

                {menuOpen === session.id && (
                  <div className="dropdown">
                    <button onClick={() => {
                      setRenamingId(session.id)
                      setRenameValue(session.name)
                      setMenuOpen(null)
                    }}>Rename</button>
                    <button
                      className="danger"
                      onClick={() => deleteSession(session.id)}
                    >Delete</button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </aside>

      {/* ── Chat area ── */}
      <div className="chat-container">
        {(activeSession || isPendingNewChat) ? (
          <>
            <header className="chat-header">
              <h1>{isPendingNewChat ? 'New Chat' : activeSession.name}</h1>
            </header>

            <main className="chat-messages">
              {messages.map((msg, idx) => (
                <div key={msg.id || idx} className={`message ${msg.role}`}>
                  <div className="message-bubble">{msg.content}</div>
                  {msg.citations?.length > 0 && (
                    <div className="sources-container">
                      {msg.citations.map((src, i) => (
                        <div key={i} className="source-item">
                          📄 {src.source_file} • {src.section_title}
                          {src.score ? ` (Score: ${src.score})` : ''}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}

              {isLoading && (
                <div className="message assistant">
                  <div className="message-bubble">
                    <div className="loading-dots">
                      <div className="dot" />
                      <div className="dot" />
                      <div className="dot" />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </main>

            <div className="chat-input-container">
              <form onSubmit={sendMessage} className="chat-input-form">
                <input
                  type="text"
                  className="chat-input"
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  placeholder="Ask a question about company policies..."
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  className="send-button"
                  disabled={isLoading || !input.trim()}
                >Send</button>
              </form>
            </div>
          </>
        ) : (
          <div className="empty-state">
            <h2>Enterprise Knowledge Assistant</h2>
            <p>Select a chat from the sidebar or start a new one</p>
            <button className="start-btn" onClick={startNewChat}>+ Start New Chat</button>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
