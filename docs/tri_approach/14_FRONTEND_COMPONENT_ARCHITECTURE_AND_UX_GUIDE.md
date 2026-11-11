# Volume 14 — Frontend Component Architecture, Routing & Offline Fallback Audit (`frontend/`)

> **Document Classification:** React 19 + TypeScript + Vite UI Architecture & Component Inventory  
> **Primary Ownership:** **M. Jagadeeswar (`23BDS033`)** (Core shell, `HomePage`, `ClassicalMLPage`, `DLPage`, `api.ts`, `types.ts`) & **B. Varshith (`23BDS011`)** (`AgenticPage`, `ChatBox`, and modular `src/agentic/` views/widgets).  
> **Primary Ground-Truth Sources:** [`frontend/package.json`](../../frontend/package.json), [`frontend/index.html`](../../frontend/index.html), [`frontend/vite.config.ts`](../../frontend/vite.config.ts), [`frontend/src/App.tsx`](../../frontend/src/App.tsx), [`frontend/src/types.ts`](../../frontend/src/types.ts), [`frontend/src/lib/api.ts`](../../frontend/src/lib/api.ts), [`frontend/src/lib/useApiHealth.ts`](../../frontend/src/lib/useApiHealth.ts), [`frontend/src/pages/*.tsx`](../../frontend/src/pages/HomePage.tsx), [`frontend/src/components/{PipelinePreview,ThemeToggle}.tsx`](../../frontend/src/components/PipelinePreview.tsx), and [`frontend/src/agentic/**`](../../frontend/src/agentic/data/agents.ts).

---

## 1. Frontend Stack, Tooling & Proxy Boundary (`frontend/`)

Tier 3 renders as a single-page application compiled cleanly with Vite + TypeScript and styled with Tailwind CSS (`frontend/package.json`, `22` lines). Key configuration parameters include:

* **Entry Container (`index.html`, `15` lines):** Mounts React root cleanly onto DOM `#root`.
* **Dev Server & Proxy (`vite.config.ts`, `19` lines):** Serves hot-reloaded frontend assets on `:5173` while proxying `/api/*` paths transparently toward the Tier 2 Express gateway (`http://127.0.0.1:3000`) so browser fetch calls inside `src/lib/api.ts` avoid cross-origin cookie/preflight friction during local multi-terminal startup.
* **Type System (`src/types.ts`, `82` lines):** Centralizes TypeScript claim payload interfaces, classical/DL prediction envelopes, agentic pipeline trace types, XAI importance records, and chat history structures consumed identically by `api.ts` and all mounted pages.

---

## 2. Live Component Graph vs. Unmounted Modules (`src/App.tsx`)

Inspecting imports inside [`frontend/src/App.tsx`](../../frontend/src/App.tsx) (`63` lines) establishes the **exact live runtime tree** versus standalone modular drafts preserved on disk inside `frontend/src/agentic/`, `src/components/`, and `src/lib/`:

```text
frontend/src/App.tsx (63L) — Root Navigation Shell [activeTab state]
 |
 +--> tab == "overview" ----> HomePage.tsx (167L)          [MOUNTED]
 |
 +--> tab == "ml" ----------> ClassicalMLPage.tsx (405L)   [MOUNTED]
 |                             |-- Queries :3000/api/ml/* via api.ts
 |                             `-- Fallback formula L92-98 (clamps 0.01 < $47,033!)
 |
 +--> tab == "dl" ----------> DLPage.tsx (349L)            [MOUNTED]
 |                             `-- Tests MLP, ResMLP, CNN1D, TabTransformer, AE
 |
 +--> tab == "agentic" -----> AgenticPage.tsx (247L)       [MOUNTED]
       |                       `-- Triggers POST /api/agentic/predict (:3000 predict.ts)
       +--------------------> ChatBox.tsx (309L)           [MOUNTED ONLY on agentic tab!]
                               `-- Queries POST /api/chat (gemini-3.8-flash)

UNMOUNTED ON-DISK MODULES (Not imported anywhere inside App.tsx):
 |-- frontend/src/agentic/pages/AgenticDashboard.tsx (142L)
 |-- frontend/src/agentic/pages/ClaimInputForm.tsx (260L)
 |-- frontend/src/agentic/components/AuditTrail.tsx (157L)
 |-- frontend/src/agentic/components/AgentCard.tsx (41L)
 |-- frontend/src/agentic/components/ScoreGauge.tsx (97L)
 |-- frontend/src/components/PipelinePreview.tsx (55L)
 |-- frontend/src/components/ThemeToggle.tsx (25L)
 `-- frontend/src/lib/{useApiHealth.ts (29L), useScrollReveal.ts, useTheme.ts}
```

### 2.1 Mounted Pages Functional Matrix

| Mounted Component | Lines | Primary User Flows Supported | Backend Endpoints Consumed (`api.ts`) | Offline / Fallback Behavior |
| :--- | :---: | :--- | :--- | :--- |
| **`HomePage.tsx`** | `167` | System overview, tri-approach navigation, quick architecture summary cards | Passive status presentation | Fully functional static/client render |
| **`ClassicalMLPage.tsx`** | `405` | Interactive claim input form, sample claim presets, single-model selector + all-5 Classical ML comparator | `POST /api/ml/predict/ml`, `/api/ml/predict` | Catches fetch failures and runs client fallback L92–98 (`0.01` floor under `\$47k`) |
| **`DLPage.tsx`** | `349` | Neural architecture inspector + live scoring across `MLP`, `ResidualMLP`, `CNN1D`, `TabTransformer`, `Autoencoder` | `POST /api/ml/predict/dl` | Handles API errors cleanly on unavailable ML tier |
| **`AgenticPage.tsx`** | `247` | Interactive agentic evaluation pipeline showing `4`-stage execution trace, score badge, & XAI bar breakdown | `POST /api/agentic/predict` | Receives Express `predict.ts` shadow engine (`AgenticEnsemble-v1`) + offline safeguard |
| **`ChatBox.tsx`** | `309` | Floating/docked natural-language SIU Q&A drawer (**rendered only when `activeTab === "agentic"`**) | `POST /api/chat` (`gemini-3.8-flash`) | Displays error/setup guidance if `GEMINI_API_KEY` is missing |

---

## 3. Forensic Inspection of Unmounted Modular Components

Prior documentation (`docs/05`) conflated mounted pages (`AgenticPage.tsx`) with sibling modules on disk and referenced missing filenames (`benchmarks.ts`, `ClaimFieldsForm.tsx`, `PredictionResultCard.tsx`). Direct file reads document the exact verified symbols actually stored on disk inside `frontend/src/`:

### 3.1 `frontend/src/agentic/` Unmounted Suite

* **`data/agents.ts` (`56` lines):** Exports `AGENTS` (`AgentDefinition[]`), defining `6` conceptual agent roles (`id`, `role`, `description`, `icon`, `status: "active" | "idle" | "error"`). Used by both `PipelinePreview.tsx` and `AgentCard.tsx`.
* **`pages/AgenticDashboard.tsx` (`142` lines) & `pages/ClaimInputForm.tsx` (`260` lines):** Alternative standalone dashboard + dedicated claim input component built independently before tab consolidation inside `App.tsx` + `AgenticPage.tsx`.
* **`components/AgentCard.tsx` (`41` lines):** Renders an individual agent card article with CSS animation stagger (`--delay: ${index * 70}ms`), `<AgentGlyph>`, role heading, description, agent ID footer, and status indicator badge (`Active` / `Idle` / `Error`).
* **`components/ScoreGauge.tsx` (`97` lines):** Animated SVG circular arc gauge (`RADIUS = 62`, `CIRCUMFERENCE = 2 * Math.PI * 62`) animating risk scores `[0, 100]` over `1,100 ms` with cubic ease-out (`1 - Math.pow(1 - progress, 3)` via `window.requestAnimationFrame`), respecting `@media (prefers-reduced-motion: reduce)` cleanly by jumping straight to target angle, with fraud-red (`#f4b26b → #d64b3b`) vs safe-green (`#7ad9a8 → #1d8a60`) SVG gradients.
* **`components/AuditTrail.tsx` (`157` lines):** Renders interactive inspection steps and cryptographic ID badges for historical workflow traces.

### 3.2 `frontend/src/components/` & `frontend/src/lib/` Unmounted Helpers

* **`components/PipelinePreview.tsx` (`55` lines):** Renders a stylized `"Claim triage flow"` card listing all `6` conceptual roles from `AGENTS` with `Ready` live pill badge, staggered entry delays (`index * 90 ms`), and footer `"Audit-ready at every step"` shield indicator.
* **`components/ThemeToggle.tsx` (`25` lines):** Accessible light/dark mode switch button accepting `{ theme, onToggle }` props matching `Theme` type from `useTheme.ts`.
* **`lib/useApiHealth.ts` (`29` lines):** React custom hook exporting `ApiHealth = "checking" | "online" | "offline"` that polls `checkApiHealth()` (`GET /api/health` on Express `:3000` -> `{ status: 'ok' }`) asynchronously every **`30,000 ms` (`30s`)** with strict cleanup unmount guards (`cancelled = true`, `clearTimeout`).

---

## 4. Network Layer (`src/lib/api.ts`, `172` lines)

`api.ts` centralizes all HTTP client communication toward `:3000`:

1. **`checkApiHealth()`:** Fetches Express health probe `healthRouter` (`{ status: 'ok' }`) and resolves boolean health status safely without throwing uncaught promise rejections on network refusal.
2. **ML / DL Dispatchers:** Send `PascalCase` typed payloads (`ClaimFormData`) to `/api/ml/predict`, `/api/ml/predict/ml`, and `/api/ml/predict/dl`, parsing FastAPI model response structures surfaced through Tier 2 `routes/ml.ts`.
3. **Agentic & Copilot Dispatchers:** Post claim metrics to `/api/agentic/predict` and chat prompts + prediction context to `/api/chat`.

---

## 5. Offline Fallback Formulas Across UI Pages vs. Ground Truth

For offline demonstration portability (`ADR-13`), two UI flows contain client-side or Tier 2 fallback math that behaves strikingly differently on real insurance claims:

| Surface & Source | Offline / Stub Formula | Activation Range on Workbook Claims (`\$100.12–\$9,997.20`) | Risk of Presentation Misinterpretation |
| :--- | :--- | :--- | :--- |
| **`ClassicalMLPage.tsx` L92–98** | `clamp((ClaimAmount * 0.00003) + (PatientAge * 0.001) - 1.5, 0.01, 0.99)` | Returns **`0.01` across `100%` of workbook claims** (`>\$47,033.33` needed to exit lower clamp!) | Demo audience testing offline without `:8000`/`:3000` thinks Classical ML predicts `1%` Legit on true frauds! |
| **`AgenticPage.tsx` / Node `predict.ts`** | `clamp((claim_amount * 0.01) + (patient_age * 0.005) - 2.0, 0.0, 1.0)` | Returns **`1.00` on almost all claims above `\$300`** (`\$260–\$300` saturates ceiling!) | Every realistic hospital bill (`\$4,776.80` avg legit) shows `100%` Fraud even when benign! |

**Best-Practice Operational Rule:** Always verify both Python `:8000` and Node `:3000` are running prior to interactive demos, or use Track B's pure-JS `submission/` browser bundle on `:8080` which executes real exported tree/neural arrays matching Python within `4.42e-7`!

---

## 6. UI Consolidation & Clean-Up Opportunity

Because `ScoreGauge.tsx` (`97` lines, reduced-motion aware cubic animation) and `AgentCard.tsx` (`41` lines) implement high-polish visual primitives currently sitting unmounted in `frontend/src/agentic/components/`, importing `ScoreGauge` directly into `ClassicalMLPage.tsx`, `DLPage.tsx`, and `AgenticPage.tsx` — while activating `useApiHealth.ts` (`30s` heartbeat banner) and `ThemeToggle.tsx` inside `App.tsx` — eliminates dead modular code and unifies UI polish with zero architectural rework.
