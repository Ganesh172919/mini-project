import React, { useEffect, useState } from "react"
import type { MLMetricsRow, DLMetrics } from "../types"

interface Props {
  onNavigate: (tab: string) => void
}

export function HomePage({ onNavigate }: Props) {
  const [mlData, setMlData] = useState<MLMetricsRow[] | null>(null)
  const [dlData, setDlData] = useState<DLMetrics | null>(null)
  const [loadErr, setLoadErr] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([
      fetch("/api/results/ml").then(r => r.json()),
      fetch("/api/results/dl").then(r => r.json()),
    ])
      .then(([ml, dl]) => { setMlData(ml); setDlData(dl) })
      .catch(() => setLoadErr("Backend is offline — start the Node.js API server to see live results."))
  }, [])

  const bestML = mlData?.reduce((b, r) => r.Test_F1 > b.Test_F1 ? r : b, mlData[0])

  return (
    <div>
      {/* Hero */}
      <div className="hero-banner">
        <div className="hero-badge">🩺 Mini-Project 2 · 2026</div>
        <h1 className="hero-title">Health Insurance Fraud<br />Intelligence Platform</h1>
        <p className="hero-sub">
          Multi-approach fraud detection combining Classical ML, Deep Learning MLP,
          and Agentic AI pipelines — trained on real insurance claim data.
        </p>
      </div>

      {/* Stats */}
      {mlData && dlData && (
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">5</div>
            <div className="stat-label">Classical ML Models</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">99.9%</div>
            <div className="stat-label">Best Test ROC-AUC</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{(dlData.metrics.test.f1 * 100).toFixed(1)}%</div>
            <div className="stat-label">MLP Test F1</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">29</div>
            <div className="stat-label">Feature Contract</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">6</div>
            <div className="stat-label">AI Agents</div>
          </div>
        </div>
      )}

      {loadErr && (
        <div className="notice notice-info">
          <span>ℹ️</span>
          <span>{loadErr}</span>
        </div>
      )}

      {/* Approaches */}
      <div className="section-heading">
        <div className="section-heading-icon">🔬</div>
        <h2>Three Approaches</h2>
      </div>

      <div className="approach-grid">
        <button className="approach-card" onClick={() => onNavigate("classical")}>
          <div className="approach-icon">📊</div>
          <div className="approach-title">Approach 1 — Classical ML</div>
          <p className="approach-desc">
            Logistic Regression, Random Forest, XGBoost, SVM, and Gradient Boosting
            trained on 29 engineered features. Best test PR-AUC: 1.000.
          </p>
          {bestML && (
            <div style={{ marginTop: "1rem", fontSize: "0.8rem", color: "var(--accent)" }}>
              🏆 Best model: {bestML.Model} — F1 {(bestML.Test_F1 * 100).toFixed(1)}%
            </div>
          )}
        </button>

        <button className="approach-card" onClick={() => onNavigate("dl")}>
          <div className="approach-icon">🧠</div>
          <div className="approach-title">Approach 2 — Deep Learning</div>
          <p className="approach-desc">
            Multi-Layer Perceptron with batch normalisation and dropout, trained on the
            same 29-feature contract. Real-time inference via FastAPI.
          </p>
          {dlData && (
            <div style={{ marginTop: "1rem", fontSize: "0.8rem", color: "var(--success)" }}>
              🏅 MLP Test F1 {(dlData.metrics.test.f1 * 100).toFixed(1)}% · PR-AUC {(dlData.metrics.test.pr_auc * 100).toFixed(1)}%
            </div>
          )}
        </button>

        <button className="approach-card" onClick={() => onNavigate("agentic")}>
          <div className="approach-icon">🤖</div>
          <div className="approach-title">Approach 3 — Agentic AI</div>
          <p className="approach-desc">
            Six-agent pipeline: Ingestion → Validation → Risk Scorer → XAI Explainer
            → Decision → Audit. Each agent produces an inspectable audit trail.
          </p>
        </button>
      </div>

      {/* Comparison table */}
      {mlData && dlData && (
        <>
          <div className="section-heading" style={{ marginTop: "2rem" }}>
            <div className="section-heading-icon">📈</div>
            <h2>Algorithm Benchmark (Test Set)</h2>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Type</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>ROC-AUC</th>
                  <th>PR-AUC</th>
                </tr>
              </thead>
              <tbody>
                {mlData.map(row => (
                  <tr key={row.Model}>
                    <td>{row.Model}</td>
                    <td><span className="model-badge badge-ml">Classical ML</span></td>
                    <td>{(row.Test_Accuracy * 100).toFixed(2)}%</td>
                    <td>{(row.Test_Precision * 100).toFixed(2)}%</td>
                    <td>{(row.Test_Recall * 100).toFixed(2)}%</td>
                    <td className={row.Test_F1 >= 0.987 ? "best-cell" : ""}>{(row.Test_F1 * 100).toFixed(2)}%</td>
                    <td>{(row.Test_ROC_AUC * 100).toFixed(2)}%</td>
                    <td className={row.Test_PR_AUC >= 0.999 ? "best-cell" : ""}>{(row.Test_PR_AUC * 100).toFixed(2)}%</td>
                  </tr>
                ))}
                {/* DL row */}
                <tr>
                  <td>MLP (Neural)</td>
                  <td><span className="model-badge badge-dl">Deep Learning</span></td>
                  <td>{(dlData.metrics.test.accuracy * 100).toFixed(2)}%</td>
                  <td>{(dlData.metrics.test.precision * 100).toFixed(2)}%</td>
                  <td>{(dlData.metrics.test.recall * 100).toFixed(2)}%</td>
                  <td>{(dlData.metrics.test.f1 * 100).toFixed(2)}%</td>
                  <td>{(dlData.metrics.test.roc_auc * 100).toFixed(2)}%</td>
                  <td>{(dlData.metrics.test.pr_auc * 100).toFixed(2)}%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
