import type { CSSProperties } from "react"
import type { AgentDefinition } from "../data/agents"
import { AgentGlyph } from "../../components/Icons"

interface AgentCardProps {
  agent: AgentDefinition
  index: number
}

const STATUS_LABEL: Record<AgentDefinition["status"], string> = {
  active: "Active",
  idle: "Idle",
  error: "Error",
}

export function AgentCard({ agent, index }: AgentCardProps) {
  return (
    <article
      className="agent-card stagger"
      style={{ "--delay": `${index * 70}ms` } as CSSProperties}
    >
      <div className="agent-card__top">
        <span className="agent-card__icon">
          <AgentGlyph name={agent.icon} size={20} />
        </span>
        <span className={`agent-status agent-status--${agent.status}`}>
          <span className="agent-status__dot" />
          {STATUS_LABEL[agent.status]}
        </span>
      </div>
      <h3 className="agent-card__role">{agent.role}</h3>
      <p className="agent-card__description">{agent.description}</p>
      <div className="agent-card__footer">
        <span className="agent-card__id">{agent.id}</span>
        <span className="agent-card__chain" aria-hidden="true">
          <span />
        </span>
      </div>
    </article>
  )
}
