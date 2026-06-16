import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// ── Citation footnotes ────────────────────────────────────────────────────────

function Footnotes({ sources }) {
  if (!sources || sources.length === 0) return null
  return (
    <div className="footnotes">
      {sources.map((s, i) => (
        <div key={s.id} className="footnote">
          <span className="fn-num">[{i + 1}]</span>
          <span className="fn-label">{s.display}</span>
          <span className="fn-id">{s.id}</span>
        </div>
      ))}
    </div>
  )
}

// Wrap [N] markers in the rendered markdown into styled superscript spans
function CitationRenderer({ children }) {
  if (typeof children !== 'string') return children
  const parts = children.split(/(\[\d+\])/)
  return parts.map((part, i) =>
    /^\[\d+\]$/.test(part)
      ? <sup key={i} className="cite-mark">{part}</sup>
      : part
  )
}

// ── Assistant message ─────────────────────────────────────────────────────────

function AssistantMessage({ content, sources }) {
  return (
    <div className="message assistant">
      <span className="role">Assistant</span>
      <div className="md-body">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{ p: ({ children }) => <p><CitationRenderer>{children}</CitationRenderer></p> }}
        >
          {content}
        </ReactMarkdown>
      </div>
      <Footnotes sources={sources} />
    </div>
  )
}

// ── Document sidebar ──────────────────────────────────────────────────────────

function DocItem({ doc }) {
  const [open, setOpen] = useState(false)
  const date = doc.ingested_at
    ? new Date(doc.ingested_at).toLocaleDateString()
    : 'pre-loaded'
  return (
    <div className="doc-item">
      <button className="doc-header" onClick={() => setOpen(o => !o)}>
        <span className="doc-name">{doc.display_name || doc.id}</span>
        <span className="doc-meta">{doc.chunk_count} chunks · {date}</span>
        <span className="doc-arrow">{open ? '▾' : '▸'}</span>
      </button>
      {open && doc.sections && doc.sections.length > 0 && (
        <ul className="doc-sections">
          {doc.sections.map(s => <li key={s}>{s}</li>)}
        </ul>
      )}
    </div>
  )
}

function Sidebar({ docs, onUpload, uploading, uploadStatus }) {
  const fileRef = useRef(null)

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>Documents</h2>
        <button
          className="upload-btn"
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
          title="Upload a PDF, HTML, or TXT document"
        >
          {uploading ? '…' : '+ Add'}
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.html,.htm,.txt,.md,.docx"
          style={{ display: 'none' }}
          onChange={e => {
            const f = e.target.files?.[0]
            if (f) onUpload(f)
            e.target.value = ''
          }}
        />
      </div>

      {uploadStatus && (
        <div className={`upload-status ${uploadStatus.type}`}>
          {uploadStatus.message}
        </div>
      )}

      <div className="doc-list">
        {docs.length === 0
          ? <p className="empty-docs">No documents loaded yet.</p>
          : docs.map(d => <DocItem key={d.id} doc={d} />)
        }
      </div>
    </aside>
  )
}

// ── Main app ──────────────────────────────────────────────────────────────────

export default function App() {
  const [messages,     setMessages]     = useState([])
  const [input,        setInput]        = useState('')
  const [loading,      setLoading]      = useState(false)
  const [docs,         setDocs]         = useState([])
  const [uploading,    setUploading]    = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [sidebarOpen,  setSidebarOpen]  = useState(true)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Load document list on mount
  useEffect(() => {
    fetch('/api/documents')
      .then(r => r.json())
      .then(d => setDocs(d.documents || []))
      .catch(() => {})
  }, [])

  // ── Send chat message ───────────────────────────────────────────────────────
  async function sendMessage() {
    const text = input.trim()
    if (!text || loading) return

    const newMessages = [...messages, { role: 'user', content: text }]
    setMessages(newMessages)
    setInput('')
    setLoading(true)

    try {
      const payload = newMessages.map(({ role, content }) => ({ role, content }))
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: payload }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setMessages([
        ...newMessages,
        { role: 'assistant', content: data.response, sources: data.sources || [] },
      ])
    } catch (err) {
      setMessages([
        ...newMessages,
        { role: 'assistant', content: `**Error:** ${err.message}`, sources: [] },
      ])
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // ── Upload document ─────────────────────────────────────────────────────────
  const handleUpload = useCallback(async (file) => {
    setUploading(true)
    setUploadStatus({ type: 'info', message: `Ingesting "${file.name}"…` })

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch('/api/ingest', { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }
      const result = await res.json()
      setUploadStatus({
        type: 'success',
        message: `✓ "${result.display_name}" ingested — ${result.chunk_count} chunks`,
      })
      // Refresh document list
      fetch('/api/documents')
        .then(r => r.json())
        .then(d => setDocs(d.documents || []))
        .catch(() => {})
    } catch (err) {
      setUploadStatus({ type: 'error', message: `✗ ${err.message}` })
    } finally {
      setUploading(false)
      setTimeout(() => setUploadStatus(null), 6000)
    }
  }, [])

  return (
    <div className="app-shell">
      {/* Sidebar toggle on small screens */}
      <button
        className="sidebar-toggle"
        onClick={() => setSidebarOpen(o => !o)}
        title="Toggle document panel"
      >
        ☰
      </button>

      {sidebarOpen && (
        <Sidebar
          docs={docs}
          onUpload={handleUpload}
          uploading={uploading}
          uploadStatus={uploadStatus}
        />
      )}

      <div className="chat-area">
        <div className="chat-header">
          <h1>Document Research Assistant</h1>
          <p className="subtitle">Ask questions about the loaded documents</p>
        </div>

        <div className="messages">
          {messages.map((m, i) =>
            m.role === 'assistant' ? (
              <AssistantMessage key={i} content={m.content} sources={m.sources} />
            ) : (
              <div key={i} className="message user">
                <span className="role">You</span>
                <p>{m.content}</p>
              </div>
            )
          )}

          {loading && (
            <div className="message assistant">
              <span className="role">Assistant</span>
              <p className="thinking">
                <span className="dot" /><span className="dot" /><span className="dot" />
              </p>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="input-row">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about the loaded documents…"
            rows={2}
          />
          <button onClick={sendMessage} disabled={loading}>
            {loading ? '…' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}
