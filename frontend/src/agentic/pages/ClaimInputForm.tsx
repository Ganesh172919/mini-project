import React from "react"

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
        <div className="form-row">
          <label>Claim Type</label>
          <select value={form.claim_type} onChange={(e) => setForm({ ...form, claim_type: e.target.value })} required>
            <option value="">Select type</option>
            <option value="medical_surgical">Medical/Surgical</option>
            <option value="diagnostic">Diagnostic</option>
            <option value="pharmaceutical">Pharmaceutical</option>
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
        <div className="form-row">
          <label>Submission Method</label>
          <select value={form.submission_method} onChange={(e) => setForm({ ...form, submission_method: e.target.value })} required>
            <option value="">Select method</option>
            <option value="electronic">Electronic</option>
            <option value="paper">Paper</option>
          </select>
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
