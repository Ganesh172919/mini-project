export type ChatRole = "user" | "assistant"

export interface ChatMessage {
  role: ChatRole
  content: string
}

export interface ClaimPayload {
  claim_amount: number
  patient_age: number
  patient_income: number
  claim_type: string
  provider_specialty: string
  cluster: number
  submission_method: string
  diagnosis_codes: string[]
  procedure_codes: string[]
}

export interface AuditTrailEntry {
  agent: string
  status?: string
  timestamp?: string
  score?: number
  details?: Record<string, unknown>
  [key: string]: unknown
}

export interface PredictionResult {
  prediction: string
  fraud_score: number
  model_name?: string
  feature_count?: number
  agent_id?: string
  audit_trail?: AuditTrailEntry[]
}

interface JsonRecord {
  [key: string]: unknown
}

async function postJson<T>(
  url: string,
  body: unknown,
  timeoutMs: number,
): Promise<T> {
  const controller = new AbortController()
  let timedOut = false
  const timer = window.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    })

    const text = await response.text()
    let data: unknown = null
    if (text) {
      try {
        data = JSON.parse(text)
      } catch {
        data = null
      }
    }

    if (!response.ok) {
      const message =
        data && typeof data === "object" && "error" in data
          ? String((data as JsonRecord).error)
          : `The API request failed with status ${response.status}.`
      throw new Error(message)
    }

    return data as T
  } catch (error) {
    if (timedOut) {
      throw new Error("The request timed out. Please try again.")
    }
    throw error
  } finally {
    window.clearTimeout(timer)
  }
}

export async function predictClaim(payload: ClaimPayload): Promise<PredictionResult> {
  const data = await postJson<unknown>("/agentic/predict", payload, 20_000)

  if (
    !data ||
    typeof data !== "object" ||
    !("prediction" in data) ||
    !("fraud_score" in data)
  ) {
    throw new Error("The prediction API returned an unexpected response.")
  }

  const record = data as JsonRecord
  return {
    ...(record as unknown as PredictionResult),
    prediction: String(record.prediction),
    fraud_score: Number(record.fraud_score),
    audit_trail: Array.isArray(record.audit_trail)
      ? (record.audit_trail as AuditTrailEntry[])
      : [],
  }
}

export async function sendChatMessage(
  message: string,
  history: ChatMessage[],
): Promise<string> {
  const data = await postJson<unknown>(
    "/api/chat",
    {
      message,
      history: history.map(({ role, content }) => ({ role, content })),
    },
    45_000,
  )

  if (!data || typeof data !== "object" || !("reply" in data)) {
    throw new Error("The assistant returned an unexpected response.")
  }

  const reply = String((data as JsonRecord).reply ?? "").trim()
  if (!reply) {
    throw new Error("The assistant returned an empty response.")
  }
  return reply
}

export async function checkApiHealth(): Promise<boolean> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), 5_000)

  try {
    const response = await fetch("/api/health", {
      signal: controller.signal,
      cache: "no-store",
    })
    return response.ok
  } catch {
    return false
  } finally {
    window.clearTimeout(timer)
  }
}

export function normalizeFraudScore(score: number): number {
  if (!Number.isFinite(score)) return 0
  const percent = score <= 1 ? score * 100 : score
  return Math.min(100, Math.max(0, percent))
}

export function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    if (
      error.name === "TypeError" &&
      /fetch|network|failed to fetch/i.test(error.message)
    ) {
      return "Cannot reach the API. Make sure the Node server is running on port 3000."
    }
    return error.message
  }
  return "Something went wrong. Please try again."
}
