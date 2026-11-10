import type { CSSProperties } from "react"
import { AGENTS } from "../agentic/data/agents"
import { AgentGlyph, CheckCircleIcon, ShieldIcon } from "./Icons"

export function PipelinePreview() {
  return (
    <div
      className="pipeline-preview"
      aria-label="Preview of the six-agent claim review pipeline"
    >
      <div className="pipeline-preview__halo" aria-hidden="true" />
      <div className="pipeline-card">
        <div className="pipeline-card__head">
          <div>
            <p className="pipeline-card__eyebrow">Live pipeline</p>
            <h2>Claim triage flow</h2>
          </div>
          <span className="live-badge">
            <span className="status-dot status-dot--online" />
            Ready
          </span>
        </div>

        <ol className="pipeline-steps">
          {AGENTS.map((agent, index) => (
            <li
              key={agent.id}
              className="pipeline-step"
              style={{ "--delay": `${index * 90}ms` } as CSSProperties}
            >
              <span className="pipeline-step__icon">
                <AgentGlyph name={agent.icon} size={17} />
              </span>
              <span className="pipeline-step__copy">
                <strong>{agent.role}</strong>
                <small>{agent.description}</small>
              </span>
              <CheckCircleIcon className="pipeline-step__check" size={16} />
            </li>
          ))}
        </ol>

        <div className="pipeline-card__foot">
          <div className="pipeline-meter" aria-hidden="true">
            <span />
          </div>
          <p>
            <ShieldIcon size={14} />
            Audit-ready at every step
          </p>
        </div>
      </div>
    </div>
  )
}
