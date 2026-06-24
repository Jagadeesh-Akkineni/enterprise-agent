import { useState, useRef, useEffect } from 'react'

function App() {
  const [messages, setMessages] = useState([
    { role: 'agent', text: 'Hello! I am your Enterprise AI Assistant. Ask me anything about the company policies and documents.', sources: [] }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userText = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', text: userText }])
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userText })
      })

      if (!response.ok) {
        throw new Error('Network response was not ok')
      }

      const data = await response.json()
      
      setMessages(prev => [...prev, { 
        role: 'agent', 
        text: data.reply,
        sources: data.sources 
      }])
    } catch (error) {
      console.error('Error fetching chat:', error)
      setMessages(prev => [...prev, { 
        role: 'agent', 
        text: 'Sorry, I encountered an error while trying to reach the server. Please ensure the backend is running.',
        sources: []
      }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>Enterprise Agent</h1>
      </header>
      
      <main className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-bubble">
              {msg.text}
            </div>
            {msg.sources && msg.sources.length > 0 && (
              <div className="sources-container">
                {msg.sources.map((src, sIdx) => (
                  <div key={sIdx} className="source-item">
                    📄 {src.source_file} • {src.section_title} (Score: {src.score})
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="message agent">
            <div className="message-bubble">
              <div className="loading-dots">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      <div className="chat-input-container">
        <form onSubmit={handleSubmit} className="chat-input-form">
          <input
            type="text"
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            disabled={isLoading}
          />
          <button type="submit" className="send-button" disabled={isLoading || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  )
}

export default App
