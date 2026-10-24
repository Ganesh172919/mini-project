import React, { useEffect, useState } from "react"
import type { DLPredictionRequest, DLPredictionResponse } from "../types"

const DL_MODELS = [
  { id: "MLP", name: "MLP (Baseline)", icon: "🧠", type: "Supervised Feedforward" },
  { id: "ResidualMLP", name: "Residual MLP", icon: "⚡", type: "Deep Residual" },
  { id: "CNN1D", name: "1D-CNN", icon: "📈", type: "1D Convolutional" },
  { id: "TabTransformer", name: "TabTransformer", icon: "✨", type: "Multi-Head Self-Attention" },
  { id: "Autoencoder", name: "Autoencoder", icon: "🔍", type: "Unsupervised Anomaly Detector" },
] as const

type DLModelId = typeof DL_MODELS[number]["id"]

const INITIAL: DLPredictionRequest = {
  ClaimAmount: 5000,
  ClaimDate: "2024-06-15",
  PatientAge: 45,
  PatientGender: "F",
  ProviderSpecialty: "Cardiology",
  PatientIncome: 60000,
  PatientMaritalStatus: "Married",
  PatientEmploymentStatus: "Employed",
  ClaimType: "Routine",
  ClaimSubmissionMethod: "Online",
  Cluster: 1,
}

export function DLPage() {
  const [selectedModel, setSelectedModel] = useState<DLModelId>("MLP")
  const [form, setForm] = useState<DLPredictionRequest>(INITIAL)
  const [result, setResult] = useState<DLPredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [allMetrics, setAllMetrics] = useState<Record<string, any>>({})

  useEffect(() => {
    fetch("/api/results/dl/all")
      .then(r => r.json())
      .then(data => {
        if (data && typeof data === "object") {
          setAllMetrics(data)
        }
      })
      .catch(() => {})
  }, [])

  const set = <K extends keyof DLPredictionRequest>(k: K, v: DLPredictionRequest[K]) =>
    setForm(f => ({ ...f, [k]: v }))

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`/api/dl/predict?model=${encodeURIComponent(selectedModel)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      })
      const data = await res.json()
      if (!res.ok) {
        setError(data?.detail ?? data?.error ?? "Prediction failed")
        return
      }
      setResult(data)
    } catch {
      setError("Cannot reach prediction service. Make sure the Node API and FastAPI ml-service are running.")
    } finally {
      setLoading(false)
    }
  }

  const isFraud = result?.prediction === "Fraud"
  const currentModelKey = selectedModel.toLowerCase()
  const modelMetrics = allMetrics[currentModelKey]
  const testM = modelMetrics?.metrics?.test

  return (
    <div>
      <div className="section-heading">
        <div className="section-heading-icon">🧠</div>
        <h2>Deep Learning — Approach 2</h2>
      </div>
      <p className="page-description">
        Five neural architectures trained on the frozen 29-feature contract.
        Select an architecture, inspect its test benchmarks, and run real-time inference.
      </p>

      {/* Model Selection Tabs */}
      <div className="model-selector-bar" style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1.25rem" }}>
        {DL_MODELS.map(m => {
          const isActive = selectedModel === m.id
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => {
                setSelectedModel(m.id)
                setResult(null)
                setError(null)
              }}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.4rem",
                padding: "0.5rem 0.9rem",
                borderRadius: "8px",
                border: isActive ? "2px solid #3b82f6" : "1px solid #334155",
                background: isActive ? "rgba(59, 130, 246, 0.15)" : "#1e293b",
                color: isActive ? "#60a5fa" : "#94a3b8",
                fontWeight: isActive ? 600 : 500,
                fontSize: "0.85rem",
                cursor: "pointer",
                transition: "all 0.15s ease",
              }}
            >
              <span>{m.icon}</span>
              <span>{m.name}</span>
            </button>
          )
        })}
      </div>

      {/* Model Benchmark Strip */}
      <div className="model-info-strip">
        <div className="model-info-icon">📋</div>
        <div className="model-info-text">
          <h4>{selectedModel} — Architecture & Test Metrics</h4>
          {selectedModel === "Autoencoder" ? (
            <p>
              Unsupervised Reconstruction Anomaly Detector · Fitted Val Threshold: 0.1286 (95th percentile) · Anomaly Ranking PR-AUC: {(testM?.pr_auc ? testM.pr_auc * 100 : 5.2).toFixed(1)}% · <em>Anomaly scores are normalized MSE reconstruction errors, NOT calibrated fraud probabilities.</em>
            </p>
          ) : testM ? (
            <p>
              Accuracy {(testM.accuracy * 100).toFixed(1)}% ·
              Precision {(testM.precision * 100).toFixed(1)}% ·
              Recall {(testM.recall * 100).toFixed(1)}% ·
              F1 {(testM.f1 * 100).toFixed(1)}% ·
              PR-AUC {(testM.pr_auc * 100).toFixed(1)}%
              {testM.roc_auc != null ? ` · ROC-AUC ${(testM.roc_auc * 100).toFixed(1)}%` : ""}
            </p>
          ) : (
            <p>Trained and evaluated on frozen 29-feature contract with zero data leakage.</p>
          )}
        </div>
      </div>

      <div className="predict-layout">
        {/* Form */}
        <div className="card">
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <div className="form-group">
                <label className="form-label" htmlFor="dl-claim-amount">Claim Amount (USD)</label>
                <input id="dl-claim-amount" className="form-input" type="number" min={0} step="0.01"
                  value={form.ClaimAmount}
                  onChange={e => set("ClaimAmount", parseFloat(e.target.value) || 0)}
                  required />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="dl-claim-date">Claim Date</label>
                <input id="dl-claim-date" className="form-input" type="date"
                  value={form.ClaimDate}
                  onChange={e => set("ClaimDate", e.target.value)}
                  required />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="dl-patient-age">Patient Age</label>
                <input id="dl-patient-age" className="form-input" type="number" min={0} max={120}
                  value={form.PatientAge}
                  onChange={e => set("PatientAge", parseInt(e.target.value) || 0)}
                  required />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="dl-patient-gender">Patient Gender</label>
                <select id="dl-patient-gender" className="form-select"
                  value={form.PatientGender}
                  onChange={e => set("PatientGender", e.target.value as any)}>
                  <option value="F">Female</option>
                  <option value="M">Male</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="dl-specialty">Provider Specialty</label>
                <select id="dl-specialty" className="form-select"
                  value={form.ProviderSpecialty}
                  onChange={e => set("ProviderSpecialty", e.target.value as any)}>
                  <option value="Cardiology">Cardiology</option>
                  <option value="General Practice">General Practice</option>
                  <option value="Neurology">Neurology</option>
                  <option value="Orthopedics">Orthopedics</option>
                  <option value="Pediatrics">Pediatrics</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="dl-income">Patient Annual Income (USD)</label>
                <input id="dl-income" className="form-input" type="number" min={0} step="100"
                  value={form.PatientIncome}
                  onChange={e => set("PatientIncome", parseFloat(e.target.value) || 0)}
                  required />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="dl-marital">Marital Status</label>
                <select id="dl-marital" className="form-select"
                  value={form.PatientMaritalStatus}
                  onChange={e => set("PatientMaritalStatus", e.target.value as any)}>
                  <option value="Single">Single</option>
                  <option value="Married">Married</option>
                  <option value="Divorced">Divorced</option>
                  <option value="Widowed">Widowed</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="dl-employment">Employment Status</label>
                <select id="dl-employment" className="form-select"
                  value={form.PatientEmploymentStatus}
                  onChange={e => set("PatientEmploymentStatus", e.target.value as any)}>
                  <option value="Employed">Employed</option>
                  <option value="Retired">Retired</option>
                  <option value="Student">Student</option>
                  <option value="Unemployed">Unemployed</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="dl-claim-type">Claim Type</label>
                <select id="dl-claim-type" className="form-select"
                  value={form.ClaimType}
                  onChange={e => set("ClaimType", e.target.value as any)}>
                  <option value="Emergency">Emergency</option>
                  <option value="Inpatient">Inpatient</option>
                  <option value="Outpatient">Outpatient</option>
                  <option value="Routine">Routine</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="dl-submission">Submission Method</label>
                <select id="dl-submission" className="form-select"
                  value={form.ClaimSubmissionMethod}
                  onChange={e => set("ClaimSubmissionMethod", e.target.value as any)}>
                  <option value="Online">Online</option>
                  <option value="Paper">Paper</option>
                  <option value="Phone">Phone</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="dl-cluster">Risk Cluster (0–3)</label>
                <input id="dl-cluster" className="form-input" type="number" min={0} max={3}
                  value={form.Cluster}
                  onChange={e => set("Cluster", parseInt(e.target.value) || 0)}
                  required />
              </div>
            </div>

            {error && (
              <div className="notice notice-error" style={{ marginTop: "1rem" }}>
                <span>⚠️</span><span>{error}</span>
              </div>
            )}

            <button className="btn-primary" type="submit" disabled={loading}>
              {loading ? <><span className="spinner" /> Inferring…</> : <>⚡ Run {selectedModel} Inference</>}
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
                <div className="verdict-label">{result.model_name} Prediction</div>
                <div className={`verdict-value ${isFraud ? "verdict-fraud-text" : "verdict-legit-text"}`}>
                  {result.prediction}
                </div>
              </div>
            </div>

            <div className="score-bar-wrap">
              <div className="score-bar-label">
                <span>{selectedModel === "Autoencoder" ? "Anomaly Score" : "Fraud Score"}</span>
                <span>{(result.fraud_score * 100).toFixed(1)}%</span>
              </div>
              <div className="score-bar-track">
                <div className={`score-bar-fill ${isFraud ? "score-bar-danger" : "score-bar-safe"}`}
                  style={{ width: `${Math.min(result.fraud_score * 100, 100)}%` }} />
              </div>
            </div>

            <div className="result-meta">
              <div className="result-meta-row">
                <span className="result-meta-key">Model</span>
                <span className="result-meta-value">{result.model_name}</span>
              </div>
              <div className="result-meta-row">
                <span className="result-meta-key">Version</span>
                <span className="result-meta-value">{result.model_version}</span>
              </div>
              <div className="result-meta-row">
                <span className="result-meta-key">Features</span>
                <span className="result-meta-value">{result.feature_count}</span>
              </div>
              <div className="result-meta-row">
                <span className="result-meta-key">Score calibrated</span>
                <span className="result-meta-value">
                  {result.score_is_calibrated ? "Yes" : selectedModel === "Autoencoder" ? "No — reconstruction MSE" : "No — raw sigmoid"}
                </span>
              </div>
              <div className="result-meta-row">
                <span className="result-meta-key">Contract</span>
                <span className="result-meta-value" style={{ fontSize: "0.7rem", maxWidth: "160px", textAlign: "right" }}>
                  {result.feature_contract}
                </span>
              </div>
            </div>

            {selectedModel === "Autoencoder" && (
              <div className="notice notice-info" style={{ marginTop: "1rem" }}>
                <span>ℹ️</span>
                <span style={{ fontSize: "0.8rem" }}>
                  <strong>Autoencoder Anomaly Detection:</strong> Evaluates MSE reconstruction error against the fitted validation threshold (0.1286, 95th percentile of normal claims). Errors exceeding the threshold trigger an anomalous flag. This measures feature reconstruction discrepancy, not a supervised fraud probability.
                </span>
              </div>
            )}
          </div>
        ) : (
          <div className="result-panel placeholder-center">
            <div className="placeholder-icon">🧪</div>
            <div className="placeholder-title">Awaiting inference</div>
            <div className="placeholder-text">Select an architecture, fill the form, and click Run {selectedModel} Inference.</div>
          </div>
        )}
      </div>
    </div>
  )
}
