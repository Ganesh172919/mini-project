// ── Shared type definitions for the frontend ────────────────────────────────

export interface DLPredictionRequest {
  ClaimAmount: number
  ClaimDate: string            // "YYYY-MM-DD"
  PatientAge: number
  PatientGender: "F" | "M"
  ProviderSpecialty: "Cardiology" | "General Practice" | "Neurology" | "Orthopedics" | "Pediatrics"
  PatientIncome: number
  PatientMaritalStatus: "Divorced" | "Married" | "Single" | "Widowed"
  PatientEmploymentStatus: "Employed" | "Retired" | "Student" | "Unemployed"
  ClaimType: "Emergency" | "Inpatient" | "Outpatient" | "Routine"
  ClaimSubmissionMethod: "Online" | "Paper" | "Phone"
  Cluster: number              // 0–3
}

export interface DLPredictionResponse {
  model_name: string
  model_version: string
  prediction: "Fraud" | "Legitimate"
  fraud_score: number
  score_is_calibrated: boolean
  feature_count: number
  feature_contract: string
  consumed_input_fields: string[]
  excluded_input_fields: string[]
}

export interface AgenticResponse {
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

export interface MLMetricsRow {
  Model: string
  Val_Accuracy: number
  Val_Precision: number
  Val_Recall: number
  Val_F1: number
  Val_ROC_AUC: number
  Val_PR_AUC: number
  Test_Accuracy: number
  Test_Precision: number
  Test_Recall: number
  Test_F1: number
  Test_ROC_AUC: number
  Test_PR_AUC: number
}

export interface DLMetrics {
  model: string
  primary_metric: string
  best_epoch: number
  feature_contract: string
  metrics: {
    val: {
      accuracy: number
      precision: number
      recall: number
      f1: number
      roc_auc: number
      pr_auc: number
    }
    test: {
      accuracy: number
      precision: number
      recall: number
      f1: number
      roc_auc: number
      pr_auc: number
    }
  }
}
