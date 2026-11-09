import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react"
import {
  AlertTriangleIcon,
  BotIcon,
  CloseIcon,
  MessageIcon,
  SendIcon,
  SparklesIcon,
} from "../../components/Icons"
import {
  getErrorMessage,
  sendChatMessage,
  type ChatMessage,
} from "../../lib/api"

interface ChatBoxProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const SUGGESTIONS = [
  "What are common health insurance fraud patterns?",
  "How should I interpret the fraud score?",
  "What does the agent audit trail contain?",
]

function getChatErrorMessage(error: unknown): string {
  const message = getErrorMessage(error)
  if (/failed to generate response/i.test(message)) {
    return "The assistant service couldn't complete the request. Check that GEMINI_API_KEY is set on the backend, then try again."
  }
  return message
}

export function ChatBox({ open, onOpenChange }: ChatBoxProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [sending, setSending] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [lastPrompt, setLastPrompt] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fabRef = useRef<HTMLButtonElement>(null)
  const wasOpenRef = useRef(false)

  useEffect(() => {
    if (!open) return
    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches
    bottomRef.current?.scrollIntoView({
      behavior: reduceMotion ? "auto" : "smooth",
      block: "end",
    })
  }, [messages, sending, errorMessage, open])

  useEffect(() => {
    if (!open) return
    const timer = window.setTimeout(() => inputRef.current?.focus(), 180)
    return () => window.clearTimeout(timer)
  }, [open])

  useEffect(() => {
    if (open) {
      wasOpenRef.current = true
      return
    }
    if (!wasOpenRef.current) return
    wasOpenRef.current = false
    const timer = window.setTimeout(() => fabRef.current?.focus(), 80)
    return () => window.clearTimeout(timer)
  }, [open])

  useEffect(() => {
    if (!open) return
    const handleEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") onOpenChange(false)
    }
    document.addEventListener("keydown", handleEscape)
    return () => document.removeEventListener("keydown", handleEscape)
  }, [open, onOpenChange])

  useEffect(() => {
    const textarea = inputRef.current
    if (!textarea) return
    textarea.style.height = "auto"
    textarea.style.height = `${Math.min(
      Math.max(textarea.scrollHeight, 44),
      108,
    )}px`
  }, [input, open])

  const requestReply = async (text: string, history: ChatMessage[]) => {
    try {
      const reply = await sendChatMessage(text, history)
      setMessages((current) => [...current, { role: "assistant", content: reply }])
      setErrorMessage(null)
      setLastPrompt(null)
    } catch (requestError) {
      setErrorMessage(getChatErrorMessage(requestError))
      setLastPrompt(text)
    } finally {
      setSending(false)
    }
  }

  const send = async (override?: string) => {
    const text = (override ?? input).trim()
    if (!text || sending) return

    const history = messages.map(({ role, content }) => ({ role, content }))
    const userMessage: ChatMessage = { role: "user", content: text }

    setMessages((current) => [...current, userMessage])
    setInput("")
    setSending(true)
    setErrorMessage(null)
    setLastPrompt(null)

    await requestReply(text, history)
  }

  const retry = () => {
    if (!lastPrompt || sending) return

    const text = lastPrompt
    const history = messages.map(({ role, content }) => ({ role, content }))
    const last = history[history.length - 1]
    if (last?.role === "user" && last.content === text) history.pop()

    setSending(true)
    setErrorMessage(null)
    setLastPrompt(null)
    void requestReply(text, history)
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault()
      void send()
    }
  }

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    void send()
  }

  if (!open) {
    return (
      <button
        type="button"
        ref={fabRef}
        className="chat-fab"
        onClick={() => onOpenChange(true)}
        aria-label="Open the fraud detection assistant"
        title="Ask the fraud detection assistant"
      >
        <MessageIcon size={23} />
        <span className="chat-fab__ping" aria-hidden="true" />
      </button>
    )
  }

  return (
    <div
      className="chat-panel"
      role="dialog"
      aria-labelledby="chat-title"
      aria-describedby="chat-subtitle"
    >
      <header className="chat-panel__header">
        <div className="chat-panel__identity">
          <span className="chat-panel__avatar" aria-hidden="true">
            <SparklesIcon size={19} />
          </span>
          <div>
            <h2 id="chat-title">Fraud Intelligence Assistant</h2>
            <p id="chat-subtitle">Ask about claims, scores, or the audit trail.</p>
          </div>
        </div>
        <button
          type="button"
          className="chat-close"
          onClick={() => onOpenChange(false)}
          aria-label="Close the assistant"
        >
          <CloseIcon size={18} />
        </button>
      </header>

      <div className="chat-panel__body">
        <div className="chat-messages" aria-live="polite">
          {messages.length === 0 && !sending && (
            <div className="chat-welcome">
              <span className="chat-welcome__icon" aria-hidden="true">
                <SparklesIcon size={23} />
              </span>
              <h3>How can I help?</h3>
              <p>Ask about fraud patterns, risk scores, or the audit trail.</p>
              <div className="chat-suggestions">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    className="chat-suggestion"
                    onClick={() => void send(suggestion)}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              key={`${message.role}-${index}`}
              className={`chat-row chat-row--${message.role}`}
            >
              {message.role === "assistant" && (
                <span className="chat-avatar" aria-hidden="true">
                  <BotIcon size={14} />
                </span>
              )}
              <div className={`chat-bubble chat-bubble--${message.role}`}>
                {message.content}
              </div>
            </div>
          ))}

          {sending && (
            <div className="chat-row chat-row--assistant">
              <span className="chat-avatar" aria-hidden="true">
                <BotIcon size={14} />
              </span>
              <div
                className="chat-bubble chat-bubble--assistant chat-typing"
                aria-label="Assistant is typing"
              >
                <span />
                <span />
                <span />
              </div>
            </div>
          )}

          {errorMessage && (
            <div className="chat-error" role="alert">
              <AlertTriangleIcon size={17} />
              <div>
                <strong>Assistant unavailable</strong>
                <p>{errorMessage}</p>
                {lastPrompt && (
                  <button
                    type="button"
                    className="chat-retry"
                    onClick={retry}
                    disabled={sending}
                  >
                    Try again
                  </button>
                )}
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      <form className="chat-composer" onSubmit={handleSubmit}>
        <div className="chat-composer__field">
          <textarea
            ref={inputRef}
            rows={1}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about fraud patterns, scoring, or the audit trail…"
            aria-label="Message"
            disabled={sending}
          />
          <button
            type="submit"
            className="chat-send"
            disabled={sending || !input.trim()}
            aria-label="Send message"
          >
            <SendIcon size={18} />
          </button>
        </div>
        <p className="chat-composer__hint">
          Enter to send · Shift + Enter for a new line
        </p>
      </form>
    </div>
  )
}
