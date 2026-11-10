import { useRef, useState, type FormEvent } from "react"
import {
  AlertTriangleIcon,
  CheckCircleIcon,
  DownloadIcon,
  FileCheckIcon,
} from "../../components/Icons"
import {
  getErrorMessage,
  normalizeFraudScore,
  predictClaim,
  type ClaimPayload,
  type PredictionResult,
} from "../../lib/api"
import { AuditTrail } from "../components/AuditTrail"
import { ScoreGauge } from "../components/ScoreGauge"

interface ClaimFormState {
  claim_amount: string
  patient_age: string
  patient_income: string
  claim_type: string
  provider_specialty: string
  cluster: string
  submission_method: string
  diagnosis_codes: string
  procedure_codes: string
}

const EMPTY_FORM: ClaimFormState = {
  claim_amount: "5000",
  patient_age: "45",
  patient_income: "60000",
  claim_type: "Routine",
  provider_specialty: "Cardiology",
  cluster: "1",
  submission_method: "Online",
  diagnosis_codes: "",
  procedure_codes: "",
}

function splitCodes(value: string): string[] {
  return value
    .split(/[,\s]+/)
    .map((item) => item.trim())
    .filter(Boolean)
}

function getRecommendation(isFraud: boolean, score: number): string {
  if (isFraud || score >= 70) return "Refer for human review"
  if (score >= 35) return "Request additional documentation"
  return "Low-risk triage signal"
}

export function ClaimInputForm() {
  const [form, setForm] = useState<ClaimFormState>(EMPTY_FORM)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [scoredAt, setScoredAt] = useState<Date | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const resultRef = useRef<HTMLElement>(null)

  const update = (key: keyof ClaimFormState, value: string) => {
    setForm((current) => ({ ...current, [key]: value }))
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (submitting) return

    const amount = Number(form.claim_amount)
    const age = Number(form.patient_age)
    const income = Number(form.patient_income)
    const cluster = Number(form.cluster)

    if (!Number.isFinite(amount) || amount <= 0) {
      setError("Enter a claim amount greater than zero.")
      return
    }
    if (!Number.isFinite(age) || age < 0 || age > 120) {
      setError("Enter a patient age between 0 and 120.")
      return
    }
    if (!Number.isFinite(income) || income < 0) {
      setError("Enter a patient income of zero or more.")
      return
    }
    if (!Number.isFinite(cluster) || cluster < 0 || cluster > 3) {
      setError("Enter a risk cluster between 0 and 3.")
      return
    }
    if (!form.claim_type || !form.provider_specialty || !form.submission_method) {
      setError("Select a claim type, provider specialty, and submission method.")
      return
    }

    const payload: ClaimPayload = {
      claim_amount: amount,
      patient_age: age,
      patient_income: income,
      claim_type: form.claim_type,
      provider_specialty: form.provider_specialty,
      cluster,
      submission_method: form.submission_method,
      diagnosis_codes: splitCodes(form.diagnosis_codes),
      procedure_codes: splitCodes(form.procedure_codes),
    }

    setSubmitting(true)
    setError(null)
    try {
      const prediction = await predictClaim(payload)
      setResult(prediction)
      setScoredAt(new Date())
      window.setTimeout(() => {
        resultRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" })
      }, 100)
    } catch (requestError) {
      setError(getErrorMessage(requestError))
    } finally {
      setSubmitting(false)
    }
  }

  const exportResult = () => {
    if (!result) return
    const blob = new Blob(
      [
        JSON.stringify(
          {
            ...result,
            exported_at: new Date().toISOString(),
            model_note:
              "Fraud score is uncalibrated and intended for human triage, not final adjudication.",
          },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    )
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = `claim-assessment-${new Date().toISOString().slice(0, 10)}.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const scorePercent = result ? normalizeFraudScore(result.fraud_score) : 0
  const isFraud = /^fraud/i.test(result?.prediction.trim() ?? "")

  return (
    <div className="claim-card">
      <div className="claim-card__head">
        <div>
          <p className="claim-card__eyebrow">New assessment</p>
          <h3>Submit a claim</h3>
        </div>
        <button type="button" className="btn btn--quiet" onClick={() => setForm(EMPTY_FORM)}>
          Reset
        </button>
      </div>

      <form className="claim-form" onSubmit={handleSubmit}>
        <div className="form-grid">
          <label className="form-row">
            Claim amount
            <input type="number" min="0" step="0.01" value={form.claim_amount} onChange={(e) => update("claim_amount", e.target.value)} />
          </label>
          <label className="form-row">
            Patient age
            <input type="number" min="0" max="120" value={form.patient_age} onChange={(e) => update("patient_age", e.target.value)} />
          </label>
          <label className="form-row">
            Patient income
            <input type="number" min="0" step="0.01" value={form.patient_income} onChange={(e) => update("patient_income", e.target.value)} />
          </label>
          <label className="form-row">
            Risk cluster
            <input type="number" min="0" max="3" value={form.cluster} onChange={(e) => update("cluster", e.target.value)} />
          </label>
          <label className="form-row">
            Claim type
            <select value={form.claim_type} onChange={(e) => update("claim_type", e.target.value)}>
              <option value="Emergency">Emergency</option>
              <option value="Inpatient">Inpatient</option>
              <option value="Outpatient">Outpatient</option>
              <option value="Routine">Routine</option>
            </select>
          </label>
          <label className="form-row">
            Provider specialty
            <select value={form.provider_specialty} onChange={(e) => update("provider_specialty", e.target.value)}>
              <option value="Cardiology">Cardiology</option>
              <option value="General Practice">General Practice</option>
              <option value="Neurology">Neurology</option>
              <option value="Orthopedics">Orthopedics</option>
              <option value="Pediatrics">Pediatrics</option>
            </select>
          </label>
          <label className="form-row">
            Submission method
            <select value={form.submission_method} onChange={(e) => update("submission_method", e.target.value)}>
              <option value="Online">Online</option>
              <option value="Paper">Paper</option>
              <option value="Phone">Phone</option>
            </select>
          </label>
          <label className="form-row">
            Diagnosis codes
            <input value={form.diagnosis_codes} onChange={(e) => update("diagnosis_codes", e.target.value)} placeholder="Optional, comma-separated" />
          </label>
          <label className="form-row">
            Procedure codes
            <input value={form.procedure_codes} onChange={(e) => update("procedure_codes", e.target.value)} placeholder="Optional, comma-separated" />
          </label>
        </div>

        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="btn btn--primary" type="submit" disabled={submitting}>
          {submitting ? "Running review..." : "Run claim review"}
        </button>
      </form>

      {result && (
        <section className={`result-panel result-panel--${isFraud ? "fraud" : "safe"}`} ref={resultRef}>
          <div className="result-panel__head">
            <div>
              <p className="result-panel__eyebrow"><FileCheckIcon size={14} /> Analysis complete</p>
              <h3>Claim assessment</h3>
            </div>
            <span className={`verdict verdict--${isFraud ? "fraud" : "safe"}`}>
              {isFraud ? <AlertTriangleIcon size={16} /> : <CheckCircleIcon size={16} />}
              {result.prediction}
            </span>
          </div>
          <div className="result-panel__main">
            <ScoreGauge value={scorePercent} isFraud={isFraud} />
            <div className="result-summary">
              <p className="result-summary__label">Recommended action</p>
              <p className="result-summary__action">{getRecommendation(isFraud, scorePercent)}</p>
              <dl className="meta-grid">
                <div><dt>Model</dt><dd>{result.model_name ?? "Agentic ensemble"}</dd></div>
                <div><dt>Features</dt><dd>{result.feature_count ?? "—"}</dd></div>
                <div><dt>Agent chain</dt><dd className="mono">{result.agent_id ?? "—"}</dd></div>
                <div><dt>Scored at</dt><dd>{scoredAt?.toLocaleTimeString() ?? "—"}</dd></div>
              </dl>
              <button type="button" className="btn btn--quiet" onClick={exportResult}>
                <DownloadIcon size={15} /> Export JSON
              </button>
            </div>
          </div>
          <AuditTrail entries={result.audit_trail ?? []} />
          <p className="result-note"><AlertTriangleIcon size={14} /> Fraud scores are uncalibrated triage signals.</p>
        </section>
      )}
    </div>
  )
}
