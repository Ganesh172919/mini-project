export function AgenticDashboard() {
  return (
    <section className="agentic-dashboard">
      <h2>Agentic AI Fraud Detection</h2>
      <div className="agent-status">
        <AgentCard agentId="agent-001" status="active" role="Ingestion" />
        <AgentCard agentId="agent-002" status="active" role="Validation" />
        <AgentCard agentId="agent-003" status="active" role="Risk Scorer" />
        <AgentCard agentId="agent-004" status="active" role="XAI Explainer" />
        <AgentCard agentId="agent-005" status="active" role="Decision" />
        <AgentCard agentId="agent-006" status="active" role="Audit" />
      </div>
      <ClaimInputForm />
    </section>
  )
}

interface AgentCardProps {
  agentId: string
  status: "active" | "idle" | "error"
  role: string
}

function AgentCard({ agentId, status, role }: AgentCardProps) {
  const statusColor = status === "active" ? "#22c55e" : status === "idle" ? "#f6e05e" : "#ef4444"
  return (
    <div className="agent-card" style={{ borderLeft: `4px solid ${statusColor}` }}>
      <div className="agent-info">
        <span className="agent-id">Agent {agentId}</span>
        <span className="agent-role">{role}</span>
      </div>
      <div className="agent-status-dot" style={{ backgroundColor: statusColor }} />
    </div>
  )
}