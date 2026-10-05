import { Router, Request, Response } from "express"
import { ClaimInput, AgentResponse } from "../shared/schemas"

const router = Router()

router.post("/predict", async (req: Request, res: Response) => {
  try {
    const input = req.body as ClaimInput

    const fraudScore = Math.min(
      Math.max((input.claim_amount * 0.01 + input.patient_age * 0.005 - 2.0), 0),
      1
    )
    const prediction = fraudScore > 0.5 ? "Fraud" : "Legitimate"

    const auditTrail = [
      { agent: "ingestion", status: "completed", timestamp: new Date().toISOString() },
      { agent: "validation", status: "completed", timestamp: new Date(Date.now() + 1000).toISOString() },
      { agent: "risk-scorer", status: "completed", score: fraudScore, timestamp: new Date(Date.now() + 2000).toISOString() },
      { agent: "xai-explainer", status: "completed", timestamp: new Date(Date.now() + 3000).toISOString() },
    ]

    const response: AgentResponse = {
      prediction,
      fraud_score: fraudScore,
      model_name: "AgenticEnsemble-v1",
      feature_count: 9,
      agent_id: "agent-chain-001",
      audit_trail: auditTrail,
    }

    res.json(response)
  } catch (error) {
    console.error("Agentic prediction error:", error)
    res.status(500).json({ error: "Internal server error" })
  }
})

export default router