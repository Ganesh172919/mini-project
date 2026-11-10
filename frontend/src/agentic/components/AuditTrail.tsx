import type { CSSProperties } from "react"
import {
  ActivityIcon,
  AgentGlyph,
  AlertTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
} from "../../components/Icons"
import type { AuditTrailEntry } from "../../lib/api"
import { normalizeFraudScore } from "../../lib/api"

interface AuditTrailProps {
  entries: AuditTrailEntry[]
}

const AGENT_LABELS: Record<string, string> = {
  ingestion: "Ingestion",
  validation: "Validation",
  "risk-scorer": "Risk Scorer",
  "xai-explainer": "XAI Explainer",
  decision: "Decision",
  audit: "Audit",
}

const AGENT_ICONS: Record<string, Parameters<typeof AgentGlyph>[0]["name"]> = {
  ingestion: "inbox",
  validation: "clipboard",
  "risk-scorer": "gauge",
  "xai-explainer": "sparkles",
  decision: "scale",
  audit: "scroll",
}

type Tone = "success" | "warning" | "danger" | "neutral"

function humanizeAgent(agent: string) {
  const key = agent.toLowerCase()
  if (AGENT_LABELS[key]) return AGENT_LABELS[key]
  return key
    .replace(/[-_]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase())
}

function statusTone(status: string): Tone {
  const normalized = status.toLowerCase()
  if (/(complete|success|done|passed|approved)/.test(normalized)) return "success"
  if (/(pending|running|processing|review)/.test(normalized)) return "warning"
  if (/(fail|error|reject|blocked)/.test(normalized)) return "danger"
  return "neutral"
}

function formatTimestamp(timestamp?: string) {
  if (!timestamp) return null
  const date = new Date(timestamp)
  if (Number.isNaN(date.getTime())) return timestamp
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  })
}

function formatDetailValue(value: unknown) {
  if (typeof value === "number") return String(value)
  if (typeof value === "string") return value
  if (typeof value === "boolean") return value ? "true" : "false"
  if (value === null) return "null"
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function StatusIcon({ tone }: { tone: Tone }) {
  if (tone === "danger") return <AlertTriangleIcon size={14} />
  if (tone === "warning") return <ClockIcon size={14} />
  return <CheckCircleIcon size={14} />
}

export function AuditTrail({ entries }: AuditTrailProps) {
  if (entries.length === 0) {
    return (
      <div className="audit-trail audit-trail--empty">
        <ActivityIcon size={16} />
        <span>No audit entries were returned for this run.</span>
      </div>
    )
  }

  return (
    <div className="audit-trail">
      <div className="audit-trail__header">
        <h4>Agent audit trail</h4>
        <span className="audit-trail__count">{entries.length} events</span>
      </div>
      <ol className="audit-trail__list">
        {entries.map((entry, index) => {
          const status = entry.status ?? "completed"
          const tone = statusTone(status)
          const timestamp = formatTimestamp(entry.timestamp)
          const details =
            entry.details && typeof entry.details === "object"
              ? Object.entries(entry.details)
              : []
          const score =
            typeof entry.score === "number"
              ? normalizeFraudScore(entry.score)
              : null

          return (
            <li
              key={`${entry.agent}-${index}`}
              className="audit-step"
              style={{ "--delay": `${index * 80}ms` } as CSSProperties}
            >
              <span className={`audit-step__marker audit-step__marker--${tone}`}>
                <AgentGlyph
                  name={AGENT_ICONS[entry.agent.toLowerCase()] ?? "sparkles"}
                  size={15}
                />
              </span>
              <div className="audit-step__body">
                <div className="audit-step__top">
                  <strong>{humanizeAgent(entry.agent)}</strong>
                  <span className={`audit-status audit-status--${tone}`}>
                    <StatusIcon tone={tone} />
                    {status}
                  </span>
                  {timestamp && (
                    <time className="audit-step__time">{timestamp}</time>
                  )}
                </div>
                {(score !== null || details.length > 0) && (
                  <div className="audit-step__details">
                    {score !== null && (
                      <span className="audit-chip">
                        <span>Fraud score</span>
                        <strong>{score.toFixed(1)}%</strong>
                      </span>
                    )}
                    {details.map(([key, value]) => (
                      <span className="audit-chip" key={key}>
                        <span>{key.replace(/[_-]+/g, " ")}</span>
                        <strong>{formatDetailValue(value)}</strong>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
