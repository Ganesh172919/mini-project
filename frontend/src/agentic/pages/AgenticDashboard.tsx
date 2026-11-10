import { ClaimInputForm } from "./ClaimInputForm"
import { AgentCard } from "../components/AgentCard"
import { AGENTS } from "../data/agents"
import {
  AlertTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  LockIcon,
} from "../../components/Icons"

export function AgenticDashboard() {
  return (
    <>
      <section
        id="pipeline"
        className="section pipeline-section"
        data-reveal
        aria-labelledby="pipeline-title"
      >
        <div className="section-heading">
          <div className="section-heading__main">
            <p className="section-kicker">
              <span>01</span>
              Agentic pipeline
            </p>
            <h2 id="pipeline-title">
              Six specialists.
              <br />
              <em>One clear verdict.</em>
            </h2>
          </div>
          <p className="section-lede">
            Each agent owns one part of the review — from normalizing the claim
            payload to writing the final audit trail. No step is skipped, and
            every signal is explainable.
          </p>
        </div>

        <div className="agent-grid">
          {AGENTS.map((agent, index) => (
            <AgentCard key={agent.id} agent={agent} index={index} />
          ))}
        </div>
      </section>

      <section
        id="claim-analysis"
        className="section analysis-section"
        data-reveal
        aria-labelledby="analysis-title"
      >
        <div className="section-heading">
          <div className="section-heading__main">
            <p className="section-kicker">
              <span>02</span>
              Claim analysis
            </p>
            <h2 id="analysis-title">
              Put a claim
              <br />
              <em>through the chain.</em>
            </h2>
          </div>
          <p className="section-lede">
            Enter a claim profile to run the full agent review. You&apos;ll get a
            risk score, a recommended action, and an exportable audit trail in
            one pass.
          </p>
        </div>

        <div className="analysis-grid">
          <ClaimInputForm />

          <aside className="analysis-aside" aria-label="How the review works">
            <div className="aside-card">
              <div className="aside-card__icon">
                <ClockIcon size={18} />
              </div>
              <h3>How the review works</h3>
              <ol className="flow-list">
                <li>
                  <span className="flow-list__number">01</span>
                  <div>
                    <strong>Ingest &amp; validate</strong>
                    <p>Claim fields are normalized and checked for completeness.</p>
                  </div>
                </li>
                <li>
                  <span className="flow-list__number">02</span>
                  <div>
                    <strong>Score the risk</strong>
                    <p>The risk agent evaluates the claim&apos;s fraud profile.</p>
                  </div>
                </li>
                <li>
                  <span className="flow-list__number">03</span>
                  <div>
                    <strong>Explain the signals</strong>
                    <p>The XAI agent highlights what moved the score.</p>
                  </div>
                </li>
                <li>
                  <span className="flow-list__number">04</span>
                  <div>
                    <strong>Decide &amp; log</strong>
                    <p>A recommendation and audit trail are produced together.</p>
                  </div>
                </li>
              </ol>
            </div>

            <div className="aside-card aside-card--warning">
              <div className="aside-card__icon">
                <AlertTriangleIcon size={18} />
              </div>
              <h3>Human in the loop</h3>
              <p>
                The fraud score is uncalibrated. Treat it as a triage signal,
                and let a human reviewer make the final call on flagged claims.
              </p>
            </div>

            <div className="aside-card aside-card--quiet">
              <div className="aside-card__icon">
                <LockIcon size={18} />
              </div>
              <h3>Audit by default</h3>
              <p>
                Every run returns an agent-by-agent event log you can review or
                export for compliance.
              </p>
              <span className="aside-card__meta">
                <CheckCircleIcon size={13} />
                Immutable event log
              </span>
            </div>
          </aside>
        </div>
      </section>
    </>
  )
}
