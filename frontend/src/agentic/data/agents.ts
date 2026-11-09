import type { AgentGlyphName } from "../../components/Icons"

export type AgentStatus = "active" | "idle" | "error"

export interface AgentDefinition {
  id: string
  role: string
  description: string
  icon: AgentGlyphName
  status: AgentStatus
}

export const AGENTS: AgentDefinition[] = [
  {
    id: "agent-001",
    role: "Ingestion",
    description: "Normalizes claim payloads and feature inputs.",
    icon: "inbox",
    status: "active",
  },
  {
    id: "agent-002",
    role: "Validation",
    description: "Checks completeness, schema, and data quality.",
    icon: "clipboard",
    status: "active",
  },
  {
    id: "agent-003",
    role: "Risk Scorer",
    description: "Scores fraud likelihood across the claim profile.",
    icon: "gauge",
    status: "active",
  },
  {
    id: "agent-004",
    role: "XAI Explainer",
    description: "Surfaces the signals behind each risk score.",
    icon: "sparkles",
    status: "active",
  },
  {
    id: "agent-005",
    role: "Decision",
    description: "Recommends approve, review, or escalate.",
    icon: "scale",
    status: "active",
  },
  {
    id: "agent-006",
    role: "Audit",
    description: "Records an immutable, review-ready trail.",
    icon: "scroll",
    status: "active",
  },
]
