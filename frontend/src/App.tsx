import { useState } from "react"
import { AgenticDashboard } from "./agentic/pages/AgenticDashboard"
import { ChatBox } from "./agentic/components/ChatBox"
import { PipelinePreview } from "./components/PipelinePreview"
import { ThemeToggle } from "./components/ThemeToggle"
import {
  ArrowRightIcon,
  CheckCircleIcon,
  MessageIcon,
  ShieldIcon,
} from "./components/Icons"
import { useApiHealth } from "./lib/useApiHealth"
import { useScrollReveal } from "./lib/useScrollReveal"
import { useTheme } from "./lib/useTheme"

const HEALTH_COPY = {
  checking: {
    label: "Connecting",
    title: "Checking the backend connection…",
  },
  online: {
    label: "API online",
    title: "Backend connected at localhost:3000",
  },
  offline: {
    label: "API offline",
    title: "Backend unreachable — start the Node API on port 3000.",
  },
} as const

export function App() {
  const { theme, toggleTheme } = useTheme()
  const health = useApiHealth()
  const [chatOpen, setChatOpen] = useState(false)

  useScrollReveal()

  return (
    <div className="page">
      <a className="skip-link" href="#main">
        Skip to content
      </a>

      <div className="ambient" aria-hidden="true">
        <span className="ambient__orb ambient__orb--one" />
        <span className="ambient__orb ambient__orb--two" />
        <span className="ambient__grid" />
      </div>

      <header className="site-header">
        <div className="shell site-header__inner">
          <a className="brand" href="#overview" aria-label="Health Fraud Intelligence home">
            <span className="brand__mark">
              <ShieldIcon size={20} />
            </span>
            <span className="brand__text">
              <strong>Health Fraud</strong>
              <span>Intelligence</span>
            </span>
          </a>

          <nav className="site-nav" aria-label="Primary">
            <a href="#pipeline">Pipeline</a>
            <a href="#claim-analysis">Claim analysis</a>
          </nav>

          <div className="site-header__actions">
            <span
              className={`api-status api-status--${health}`}
              title={HEALTH_COPY[health].title}
              aria-label={HEALTH_COPY[health].title}
              role="status"
            >
              <span className="api-status__dot" />
              <span className="api-status__text">
                {HEALTH_COPY[health].label}
              </span>
            </span>
            <ThemeToggle theme={theme} onToggle={toggleTheme} />
          </div>
        </div>
      </header>

      <main id="main">
        <div className="shell">
          <section
            id="overview"
            className="hero"
            aria-labelledby="hero-title"
          >
            <div className="hero-copy">
              <p className="eyebrow">
                <span className="eyebrow__dot" />
                Agentic AI · Claim integrity
              </p>
              <h1 id="hero-title">
                Catch fraudulent claims <em>before</em> they&apos;re paid.
              </h1>
              <p className="hero-lede">
                Six coordinated AI agents inspect every claim — validating the
                data, scoring risk, explaining the signals, and leaving an
                audit-ready trail.
              </p>

              <div className="hero-actions">
                <a className="btn btn--primary" href="#claim-analysis">
                  Analyze a claim
                  <ArrowRightIcon size={17} />
                </a>
                <button
                  type="button"
                  className="btn btn--ghost"
                  onClick={() => setChatOpen(true)}
                >
                  <MessageIcon size={17} />
                  Ask the assistant
                </button>
              </div>

              <ul className="hero-points">
                <li>
                  <CheckCircleIcon size={15} />
                  Multi-agent review
                </li>
                <li>
                  <CheckCircleIcon size={15} />
                  Explainable scoring
                </li>
                <li>
                  <CheckCircleIcon size={15} />
                  Audit-ready output
                </li>
              </ul>
            </div>

            <PipelinePreview />
          </section>

          <AgenticDashboard />
        </div>
      </main>

      <footer className="site-footer">
        <div className="shell site-footer__inner">
          <div className="site-footer__brand">
            <ShieldIcon size={16} />
            <span>Health Fraud Intelligence</span>
          </div>
          <p>
            Mini-Project 2 · Scores are uncalibrated and support human triage.
          </p>
        </div>
      </footer>

      <ChatBox open={chatOpen} onOpenChange={setChatOpen} />
    </div>
  )
}
