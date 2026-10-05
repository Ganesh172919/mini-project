export function ClaimInputForm() {
  const [form, setForm] = React.useState({
    claim_amount: "",
    patient_age: "",
    patient_income: "",
    claim_type: "",
    provider_specialty: "",
    cluster: "",
    submission_method: "",
    diagnosis_codes: "",
    procedure_codes: "",
  })

  const [result, setResult] = React.useState<{
    prediction: string
    fraud_score: number
    audit_trail: any[]
  } | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setResult(null)
    const response = await fetch("/agentic/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form),
    })
    const data = await response.json()
    setResult(data)
  }

  return (
    <div className="input-form">
      <h3>Submit Claim for Fraud Analysis</h3>
      <form onSubmit={handleSubmit}>
        <div className="form-row">
          <label>Claim Amount (USD)</label>
          <input value={form.claim_amount} onChange={(e) => setForm({ ...form, claim_amount: e.target.value })} type="number" required />
        </div>
        <div className="form-row">
          <label>Patient Age</label>
          <input value={form.patient_age} onChange={(e) => setForm({ ...form, patient_age: e.target.value })} type="number" required />
        </div>
        <div className="form-row">
          <label>Patient Income (USD)</label>
          <input value={form.patient_income} onChange={(e) => setForm({ ...form, patient_income: e.target.value })} type="number" required />
        </div>
        <div className="form-row">
          <label>Claim Type</label>
          <select value={form.claim_type} onChange={(e) => setForm({ ...form, claim_type: e.target.value })} required>
            <option value="">Select type</option>
            <option value="medical_surgical">Medical/Surgical</option>
            <option value>diagnostic</option>
            <option value>pharmaceutical</option>
          </select>
        </div>
        <div className="form-row">
          <label>Provider Specialty</label>
          <select value={form.provider_specialty} onChange={(e) => setForm({ ...form, provider_specialty: e.target.value })} required>
            <option value="">Select specialty</option>
            <option value="cardiology">Cardiology</option>
            <option value="orthopedics">Orthopedics</option>
            <option value="general_practice">General Practice</option>
          </select>
        </div>
        <div className="form-row">
          <label>Risk Cluster</label>
          <input value={form.cluster} onChange={(e) => setForm({ ...form, cluster: e.target.value })} type="number" required />
        </div>
        <div className="form-row">
          <label>Submission Method</label>
          <select value={form.submission_method} onChange={(e) => setForm({ ...form, submission_method: e.target.value })} required>
            <option value="">Select method</option>
            <option value>electronic">Electronic</option>
            <option value>paper">Paper</option>
          </select>
        </div>
        <button type="submit" className="submit-btn">Analyze with AI Agents</button>
      </form>

      {result && (
        <div className="result-panel">
          <h3>Prediction Result</h3>
          <p>Prediction: <strong>{result.prediction}</strong></p>
          <p>Fraud Score: {result.fraud_score}%</p>
          <AgentAuditTrail auditTrail={result.audit_trail} />
        </div>
      )}
    </div>
  )
}

function AgentAuditTrail({ auditTrail }: { auditTrail: any[] }) {
  return (
    <div className="audit-trail">
      <h4>Agent Audit Trail</h4>
      <ul>
        {auditTrail.map((entry, idx) => (
          <li key={idx} className="audit-item">
            <strong>{entry.agent}:</strong> {entry.status}{' '}
            {entry.details && Object.keys(entry.details).length > 0 && (
              <span>
                {JSON.stringify(entry.details)}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}