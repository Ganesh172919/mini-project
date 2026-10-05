export interface ClaimInput {
  claim_amount: number
  patient_age: number
  patient_income: number
  claim_type: string
  provider_specialty: string
  cluster: number
  submission_method: string
  diagnosis_codes?: string[]
  procedure_codes?: string[]
}

export interface AgentResponse {
  prediction: string
  fraud_score: number
  model_name: string
  feature_count: number
  agent_id: string
  audit_trail: Array<{
    agent: string
    status: string
    timestamp: string
    details?: any
  }>
}