import {
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react"
import {
  AlertTriangleIcon,
  CheckCircleIcon,
  ChevronDownIcon,
  DownloadIcon,
  FileCheckIcon,
  FlaskIcon,
  LoaderIcon,
  ScanIcon,
  ServerIcon,
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
  claim_amount: "",
  patient_age: "",
  patient_income: "",
  claim_type: "",
  provider_specialty: "",
  cluster: "",
  submission_method: "",
  diagnosis_codes: "",
  procedure_codes: "",
}

const SAMPLE_FORM: ClaimFormState = {
  claim_amount: "18450",
  patient_age: "57",
  patient_income: "42000",
  claim_type: "medical_surgical",
  provider_specialty: "cardiology",
  cluster: "3",
  submission_method: "electronic",
  diagnosis_codes: "I25.10, E11.9",
  procedure_codes: "93458, 92928",
}

const CLAIM_TYPES = [
  { value: "medical_surgical", label: "Medical / surgical" },
  { value: "diagnostic", label: "Diagnostic" },
  { value: "pharmaceutical", label: "Pharmaceutical" },
  { value: "emergency", label: "Emergency" },
  { value: "behavioral_health", label: "Behavioral health" },
]

const SPECIALTIES = [
  { value: "cardiology", label: "Cardiology" },
  { value: "orthopedics", label: "Orthopedics" },
  { value: "general_practice", label: "General practice" },
  { value: "oncology", label: "Oncology" },
  { value: "neurology", label: "Neurology" },
  { value: "radiology", label: "Radiology" },
  { value: "dermatology", label: "Dermatology" },
  { value: "internal_medicine", label: "Internal medicine" },
]

const SUBMISSION_METHODS = [
  { value: "electronic", label: "Electronic" },
  { value: "paper", label: "Paper" },
]

function splitCodes(raw: string): string[] {
  return raw
    .split(/[,;\s]+/)
    .map((code) => code.trim())
    .filter(Boolean)
}

function getRecommendation(isFraud: boolean, score: number): string {
  if (isFraud && score >= 75) {
    return "Escalate to a fraud investigator and hold the claim before any payout."
  }
  if (isFraud) {
    return "Route this claim to a human reviewer for a closer look at the flagged signals."
  }
  if (score >= 40) {
    return "Proceed with standard checks — the signal is inconclusive and merits a second look."
  }
  return "Proceed with normal processing while routine monitoring stays in place."
}

interface FieldProps {
  id: string
  label: string
  hint?: string
  className?: string
  children: ReactNode
}

function Field({ id, label, hint, className, children }: FieldProps) {
  return (
    <div className={`field${className ? ` ${className}` : ""}`}>
      <label htmlFor={id}>
        {label}
        {hint && <span className="field__hint">{hint}</span>}
      </label>
      {children}
    </div>
  )
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

  const loadSample = () => {
    setForm(SAMPLE_FORM)
    setError(null)
    setResult(null)
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (submitting) return

    setError(null)

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
    if (!Number.isFinite(cluster) || cluster < 0 || cluster > 5) {
      setError("Enter a risk cluster between 0 and 5.")
      return
    }
    if (
      !form.claim_type ||
      !form.provider_specialty ||
      !form.submission_method
    ) {
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
    setResult(null)

    try {
      const prediction = await predictClaim(payload)
      setResult(prediction)
      setScoredAt(new Date())

      const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches
      window.setTimeout(() => {
        resultRef.current?.scrollIntoView({
          behavior: reduceMotion ? "auto" : "smooth",
          block: "nearest",
        })
      }, 140)
    } catch (requestError) {
      setError(getErrorMessage(requestError))
    } finally {
      setSubmitting(false)
    }
  }

  const exportResult = () => {
    if (!result) return
    const payload = {
      ...result,
      exported_at: new Date().toISOString(),
      model_note:
        "Fraud score is uncalibrated and intended for human triage, not final adjudication.",
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = `claim-assessment-${new Date()
      .toISOString()
      .slice(0, 10)}.json`
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 0)
  }

  const scorePercent = result ? normalizeFraudScore(result.fraud_score) : 0
  const verdict = result?.prediction.trim() ?? ""
  const isFraud = /^fraud/i.test(verdict)

  return (
    <div className="claim-card">
      <div className="claim-card__head">
        <div>
          <p className="claim-card__eyebrow">New assessment</p>
          <h3>Submit a claim</h3>
        </div>
        <span className="claim-card__badge">
          <ServerIcon size={14} />
          POST /agentic/predict
        </span>
      </div>

      <form className="claim-form" onSubmit={handleSubmit}>
        <fieldset className="form-section">
          <legend>Claim details</legend>
          <div className="form-grid">
            <Field id="claim-amount" label="Claim amount" hint="USD">
              <div className="input-shell">
                <span className="input-prefix" aria-hidden="true">
                  $
                </span>
                <input
                  id="claim-amount"
                  className="has-prefix"
                  type="number"
                  min="1"
                  step="0.01"
                  inputMode="decimal"
                  placeholder="18,450"
                  autoComplete="off"
                  required
                  value={form.claim_amount}
                  onChange={(event) => update("claim_amount", event.target.value)}
                />
              </div>
            </Field>

            <Field id="claim-type" label="Claim type">
              <div className="select-shell">
                <select
                  id="claim-type"
                  required
                  value={form.claim_type}
                  onChange={(event) => update("claim_type", event.target.value)}
                >
                  <option value="">Select type</option>
                  {CLAIM_TYPES.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <ChevronDownIcon className="select-chevron" size={16} />
              </div>
            </Field>

            <Field id="submission-method" label="Submission method">
              <div className="select-shell">
                <select
                  id="submission-method"
                  required
                  value={form.submission_method}
                  onChange={(event) =>
                    update("submission_method", event.target.value)
                  }
                >
                  <option value="">Select method</option>
                  {SUBMISSION_METHODS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <ChevronDownIcon className="select-chevron" size={16} />
              </div>
            </Field>

            <Field id="cluster" label="Risk cluster" hint="0–5">
              <input
                id="cluster"
                type="number"
                min="0"
                max="5"
                step="1"
                inputMode="numeric"
                placeholder="3"
                autoComplete="off"
                required
                value={form.cluster}
                onChange={(event) => update("cluster", event.target.value)}
              />
            </Field>
          </div>
        </fieldset>

        <fieldset className="form-section">
          <legend>Patient profile</legend>
          <div className="form-grid">
            <Field id="patient-age" label="Patient age" hint="Years">
              <input
                id="patient-age"
                type="number"
                min="0"
                max="120"
                step="1"
                inputMode="numeric"
                placeholder="57"
                autoComplete="off"
                required
                value={form.patient_age}
                onChange={(event) => update("patient_age", event.target.value)}
              />
            </Field>

            <Field id="patient-income" label="Patient income" hint="USD / year">
              <div className="input-shell">
                <span className="input-prefix" aria-hidden="true">
                  $
                </span>
                <input
                  id="patient-income"
                  className="has-prefix"
                  type="number"
                  min="0"
                  step="100"
                  inputMode="decimal"
                  placeholder="42,000"
                  autoComplete="off"
                  required
                  value={form.patient_income}
                  onChange={(event) =>
                    update("patient_income", event.target.value)
                  }
                />
              </div>
            </Field>
          </div>
        </fieldset>

        <fieldset className="form-section">
          <legend>Provider &amp; coding</legend>
          <div className="form-grid">
            <Field
              id="provider-specialty"
              label="Provider specialty"
              className="field--full"
            >
              <div className="select-shell">
                <select
                  id="provider-specialty"
                  required
                  value={form.provider_specialty}
                  onChange={(event) =>
                    update("provider_specialty", event.target.value)
                  }
                >
                  <option value="">Select specialty</option>
                  {SPECIALTIES.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <ChevronDownIcon className="select-chevron" size={16} />
              </div>
            </Field>

            <Field id="diagnosis-codes" label="Diagnosis codes" hint="Optional">
              <input
                id="diagnosis-codes"
                type="text"
                placeholder="I25.10, E11.9"
                autoComplete="off"
                value={form.diagnosis_codes}
                onChange={(event) =>
                  update("diagnosis_codes", event.target.value)
                }
              />
            </Field>

            <Field id="procedure-codes" label="Procedure codes" hint="Optional">
              <input
                id="procedure-codes"
                type="text"
                placeholder="93458, 92928"
                autoComplete="off"
                value={form.procedure_codes}
                onChange={(event) =>
                  update("procedure_codes", event.target.value)
                }
              />
            </Field>
          </div>
        </fieldset>

        <div className="form-actions">
          <button
            type="submit"
            className="btn btn--primary"
            disabled={submitting}
          >
            {submitting ? (
              <LoaderIcon className="spin" size={17} />
            ) : (
              <ScanIcon size={17} />
            )}
            {submitting ? "Running agents…" : "Analyze claim"}
          </button>
          <button
            type="button"
            className="btn btn--ghost"
            onClick={loadSample}
            disabled={submitting}
          >
            <FlaskIcon size={16} />
            Load sample claim
          </button>
          <span className="form-actions__note">Comma-separated codes supported</span>
        </div>
      </form>

      {submitting && (
        <div className="analysis-progress" role="status" aria-live="polite">
          <div className="analysis-progress__head">
            <LoaderIcon className="spin" size={16} />
            <span>Running the six-agent review…</span>
          </div>
          <div className="analysis-progress__bar" aria-hidden="true">
            <span />
          </div>
        </div>
      )}

      {error && (
        <div className="alert alert--danger" role="alert">
          <AlertTriangleIcon size={19} />
          <div>
            <strong>Analysis failed</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      {result && (
        <section
          className={`result-panel result-panel--${isFraud ? "fraud" : "safe"}`}
          ref={resultRef}
          aria-labelledby="result-title"
          aria-live="polite"
        >
          <div className="result-panel__glow" aria-hidden="true" />

          <div className="result-panel__head">
            <div>
              <p className="result-panel__eyebrow">
                <FileCheckIcon size={14} />
                Analysis complete
              </p>
              <h3 id="result-title">Claim assessment</h3>
            </div>
            <span
              className={`verdict verdict--${isFraud ? "fraud" : "safe"}`}
            >
              {isFraud ? (
                <AlertTriangleIcon size={16} />
              ) : (
                <CheckCircleIcon size={16} />
              )}
              {result.prediction}
            </span>
          </div>

          <div className="result-panel__main">
            <ScoreGauge value={scorePercent} isFraud={isFraud} />

            <div className="result-summary">
              <p className="result-summary__label">Recommended action</p>
              <p className="result-summary__action">
                {getRecommendation(isFraud, scorePercent)}
              </p>

              <dl className="meta-grid">
                <div>
                  <dt>Model</dt>
                  <dd>{result.model_name ?? "Agentic ensemble"}</dd>
                </div>
                <div>
                  <dt>Features</dt>
                  <dd>{result.feature_count ?? "—"}</dd>
                </div>
                <div>
                  <dt>Agent chain</dt>
                  <dd className="mono">{result.agent_id ?? "—"}</dd>
                </div>
                <div>
                  <dt>Scored at</dt>
                  <dd>
                    {scoredAt
                      ? scoredAt.toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                          second: "2-digit",
                        })
                      : "—"}
                  </dd>
                </div>
              </dl>

              <button
                type="button"
                className="btn btn--quiet result-export"
                onClick={exportResult}
              >
                <DownloadIcon size={15} />
                Export JSON
              </button>
            </div>
          </div>

          <AuditTrail entries={result.audit_trail ?? []} />

          <p className="result-note">
            <AlertTriangleIcon size={14} />
            Fraud scores are uncalibrated and intended to support human triage,
            not replace it.
          </p>
        </section>
      )}
    </div>
  )
}
