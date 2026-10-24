/**
 * ML / DL proxy routes.
 *
 * POST /api/dl/predict   → FastAPI ml-service at ML_SERVICE_URL (default http://127.0.0.1:8000)
 * GET  /api/results/dl   → DL metrics JSON (mlp_metrics.json)
 * GET  /api/results/ml   → Classical ML metrics JSON (model_results.json)
 */

import { Router, Request, Response } from "express"
import path from "path"
import { readFileSync } from "fs"
import { fileURLToPath } from "url"

const router = Router()

const ML_SERVICE_URL = process.env.ML_SERVICE_URL ?? "http://127.0.0.1:8000"

// ─── DL prediction proxy ───────────────────────────────────────────────────
router.post("/dl/predict", async (req: Request, res: Response) => {
  const model = (req.query.model as string) || "MLP"
  try {
    const upstream = await fetch(`${ML_SERVICE_URL}/predict/dl?model=${encodeURIComponent(model)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    })
    const data = await upstream.json()
    if (!upstream.ok) {
      res.status(upstream.status).json(data)
      return
    }
    res.json(data)
  } catch (err: any) {
    console.error("DL proxy error:", err?.message)
    res.status(502).json({ error: "Deep learning service unavailable", detail: String(err?.message ?? err) })
  }
})

// ─── Classical ML prediction proxy ─────────────────────────────────────────
router.post("/ml/predict", async (req: Request, res: Response) => {
  const model = (req.query.model as string) || "RandomForest"
  try {
    const upstream = await fetch(`${ML_SERVICE_URL}/predict/ml?model=${encodeURIComponent(model)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    })
    const data = await upstream.json()
    if (!upstream.ok) {
      res.status(upstream.status).json(data)
      return
    }
    res.json(data)
  } catch (err: any) {
    console.error("Classical ML proxy error:", err?.message)
    res.status(502).json({ error: "ML service unavailable", detail: String(err?.message ?? err) })
  }
})

// ─── Static result files ────────────────────────────────────────────────────
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PROJECT_ROOT = path.resolve(__dirname, "../../../../")

const DL_METRIC_FILES: Record<string, string> = {
  mlp: "mlp_metrics.json",
  residualmlp: "residualmlp_metrics.json",
  cnn1d: "cnn1d_metrics.json",
  tabtransformer: "tabtransformer_metrics.json",
  autoencoder: "autoencoder_metrics.json",
}

router.get("/results/dl/all", (_req: Request, res: Response) => {
  try {
    const results: Record<string, any> = {}
    for (const [key, filename] of Object.entries(DL_METRIC_FILES)) {
      const p = path.join(PROJECT_ROOT, "Approch-2-DL", "results", filename)
      try {
        results[key] = JSON.parse(readFileSync(p, "utf-8"))
      } catch {
        // checkpoint/metrics not found for this model
      }
    }
    res.json(results)
  } catch {
    res.status(500).json({ error: "Failed to read DL metrics" })
  }
})

router.get("/results/dl", (req: Request, res: Response) => {
  const model = (req.query.model as string)?.toLowerCase() || "mlp"
  const filename = DL_METRIC_FILES[model] || "mlp_metrics.json"
  try {
    const p = path.join(PROJECT_ROOT, "Approch-2-DL", "results", filename)
    const data = JSON.parse(readFileSync(p, "utf-8"))
    res.json(data)
  } catch {
    res.status(404).json({ error: `DL metrics file not found for ${model}` })
  }
})

router.get("/results/ml", (_req: Request, res: Response) => {
  try {
    const p = path.join(PROJECT_ROOT, "Approch-1", "evaluation", "model_results.json")
    const data = JSON.parse(readFileSync(p, "utf-8"))
    res.json(data)
  } catch {
    res.status(404).json({ error: "ML results file not found" })
  }
})

export default router
