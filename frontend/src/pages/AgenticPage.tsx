import React, { useState } from "react"
import type { AgenticResponse } from "../types"

const AGENTS = [
  { id: "agent-001", role: "Ingestion",    desc: "Parses and validates raw claim payload" },
  { id: "agent-002", role: "Validation",   desc: "Checks field ranges, dates, and schema" },
  { id: "agent-003", role: "Risk Scorer",  desc: "Computes fraud probability score" },
  { id: "agent-004", role: "XAI Explainer",desc: "Generates feature attribution explanations" },
  { id: "agent-005", role: "Decision",     desc: "Applies threshold policy and flags claim" },
  { id: "agent-006", role: "Audit",        desc: "Records chain-of-custody audit trail" },
]

const INITIAL_FORM = {
  claim_amount: "5000",
  patient_age: "45",
  patient_income: "60000",
  claim_type: "Routine",
  provider_specialty: "Cardiology",
  cluster: "1",
  submission_method: "Online",
}

export function AgenticPage() {
  const [form, setForm] = useState(INITIAL_FORM)
  const [result, setResult] = useState<AgenticResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const set = (k: keyof typeof INITIAL_FORM, v: string) =>
    setForm(f => ({ ...f, [k]: v }))

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch("/agentic/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          claim_amount: parseFloat(form.claim_amount),
          patient_age: parseInt(form.patient_age),
          patient_income: parseFloat(form.patient_income),
          claim_type: form.claim_type,
          provider_specialty: form.provider_specialty,
          cluster: parseInt(form.cluster),
          submission_method: form.submission_method,
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        setError(data?.error ?? "Agent pipeline failed")
        return
      }
      setResult(data)
    } catch {
      setError("Cannot reach agentic API. Make sure the Node.js backend is running on port 3000.")
    } finally {
      setLoading(false)
    }
  }

  const isFraud = result?.prediction === "Fraud"

  return (
    <div>
      <div className="section-heading">
        <div className="section-heading-icon">🤖</div>
        <h2>Agentic AI — Multi-Agent Pipeline</h2>
      </div>
      <p className="page-description">
        Six specialised agents process each claim sequentially, producing an
        inspectable chain-of-custody audit trail. (Approach 3 — Varshith's implementation)
      </p>

      {/* Agent pipeline display */}
      <div className="card" style={{ marginBottom: "1.5rem" }}>
        <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "1rem" }}>
          Agent Pipeline
        </div>
        <div className="agent-pipeline">
          {AGENTS.map((a, i) => (
            <div key={a.id} className="agent-step active" style={{ animationDelay: `${i * 60}ms` }}>
              <div className="step-dot step-dot-active" />
              <div className="step-role">{a.role}</div>
              <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", flex: 2 }}>{a.desc}</div>
              <div className="step-id">{a.id}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="predict-layout">
        {/* Form */}
        <div className="card">
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <div className="form-group">
                <label className="form-label" htmlFor="ag-claim-amount">Claim Amount (USD)</label>
                <input id="ag-claim-amount" className="form-input" type="number" min={0} step="0.01"
                  value={form.claim_amount}
                  onChange={e => set("claim_amount", e.target.value)}
                  required />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="ag-patient-age">Patient Age</label>
                <input id="ag-patient-age" className="form-input" type="number" min={0} max={120}
                  value={form.patient_age}
                  onChange={e => set("patient_age", e.target.value)}
                  required />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="ag-income">Patient Income (USD)</label>
                <input id="ag-income" className="form-input" type="number" min={0} step="0.01"
                  value={form.patient_income}
                  onChange={e => set("patient_income", e.target.value)}
                  required />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="ag-cluster">Risk Cluster (0–3)</label>
                <input id="ag-cluster" className="form-input" type="number" min={0} max={3}
                  value={form.cluster}
                  onChange={e => set("cluster", e.target.value)}
                  required />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="ag-claim-type">Claim Type</label>
                <select id="ag-claim-type" className="form-select" value={form.claim_type}
                  onChange={e => set("claim_type", e.target.value)}>
                  <option value="Emergency">Emergency</option>
                  <option value="Inpatient">Inpatient</option>
                  <option value="Outpatient">Outpatient</option>
                  <option value="Routine">Routine</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="ag-specialty">Provider Specialty</label>
                <select id="ag-specialty" className="form-select" value={form.provider_specialty}
                  onChange={e => set("provider_specialty", e.target.value)}>
                  <option value="Cardiology">Cardiology</option>
                  <option value="General Practice">General Practice</option>
                  <option value="Neurology">Neurology</option>
                  <option value="Orthopedics">Orthopedics</option>
                  <option value="Pediatrics">Pediatrics</option>
                </select>
              </div>

              <div className="form-group form-group-full">
                <label className="form-label" htmlFor="ag-submission">Submission Method</label>
                <select id="ag-submission" className="form-select" value={form.submission_method}
                  onChange={e => set("submission_method", e.target.value)}>
                  <option value="Online">Online</option>
                  <option value="Paper">Paper</option>
                  <option value="Phone">Phone</option>
                </select>
              </div>
            </div>

            {error && (
              <div className="notice notice-error" style={{ marginTop: "1rem" }}>
                <span>⚠️</span><span>{error}</span>
              </div>
            )}

            <button className="btn-primary" type="submit" disabled={loading}>
              {loading ? <><span className="spinner" /> Running Agents…</> : <>🤖 Analyse with AI Agents</>}
            </button>
          </form>
        </div>

        {/* Result */}
        {result ? (
          <div className="result-panel">
            <div className="result-verdict">
              <div className={`verdict-icon ${isFraud ? "verdict-fraud" : "verdict-legit"}`}>
                {isFraud ? "🚨" : "✅"}
              </div>
              <div>
                <div className="verdict-label">Agent Decision</div>
                <div className={`verdict-value ${isFraud ? "verdict-fraud-text" : "verdict-legit-text"}`}>
                  {result.prediction}
                </div>
              </div>
            </div>

            <div className="score-bar-wrap">
              <div className="score-bar-label">
                <span>Fraud Score</span>
                <span>{(result.fraud_score * 100).toFixed(1)}%</span>
              </div>
              <div className="score-bar-track">
                <div className={`score-bar-fill ${isFraud ? "score-bar-danger" : "score-bar-safe"}`}
                  style={{ width: `${result.fraud_score * 100}%` }} />
              </div>
            </div>

            <div className="result-meta" style={{ marginBottom: "1rem" }}>
              <div className="result-meta-row">
                <span className="result-meta-key">Model</span>
                <span className="result-meta-value">{result.model_name}</span>
              </div>
              <div className="result-meta-row">
                <span className="result-meta-key">Agent ID</span>
                <span className="result-meta-value">{result.agent_id}</span>
              </div>
              <div className="result-meta-row">
                <span className="result-meta-key">Features</span>
                <span className="result-meta-value">{result.feature_count}</span>
              </div>
            </div>

            <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.5rem" }}>
              Audit Trail
            </div>
            <div className="audit-trail">
              {result.audit_trail.map((entry, i) => (
                <div key={i} className="audit-entry">
                  <span className="audit-agent">[{entry.agent}]</span>{" "}
                  <span className="audit-status">{entry.status}</span>{" "}
                  <span className="audit-time">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                  {entry.details && Object.keys(entry.details).length > 0 && (
                    <span style={{ color: "var(--text-muted)", marginLeft: "0.5rem" }}>
                      {JSON.stringify(entry.details)}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="result-panel placeholder-center">
            <div className="placeholder-icon">🤖</div>
            <div className="placeholder-title">Awaiting agent run</div>
            <div className="placeholder-text">Submit a claim to trigger the six-agent pipeline.</div>
          </div>
        )}
      </div>
    </div>
  )
}
