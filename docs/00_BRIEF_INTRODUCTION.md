# Brief Introduction: Health Insurance Fraud Intelligence Platform

**Project:** Medical Insurance Claim Fraud Detection (`Health Insurance Fraud Intelligence Platform`)  
**Institution:** Department of Computer Science and Engineering, Indian Institute of Information Technology (IIIT) Dharwad  
**Authors:** B. Varshith (`23BDS011`), J. Ganesh Kumar Reddy (`23BDS024`), M. Jagadeeswar (`23BDS033`)  
**Supervisor:** Prof. Ramesh Athe  

---

## 1. Executive Introduction

Medical insurance fraud imposes heavy financial losses on public and private healthcare payers while delaying claim payouts for genuine patients. Building practical fraud detection software requires solving five intertwined engineering challenges rather than fitting a single isolated classifier:

1. **Severe Class Imbalance:** In our audited 4,500-claim dataset (`Health Insurance Fraud Claims.xlsx`), only **270 claims (6.00%)** are fraudulent compared to **4,230 legitimate claims (94.00%)**.
2. **Mixed Structured Clinical & Billing Fields:** Inputs combine continuous monetary figures (`ClaimAmount`, `PatientIncome`), demographic fields (`PatientAge`), temporal claim attributes (`ClaimDate`), and clinical/channel categoricals (`ProviderSpecialty`, `ClaimType`, `ClaimSubmissionMethod`).
3. **Target Proxy & Post-Adjudication Leakage:** Uninspected datasets harbor downstream status variables (`ClaimStatus`) and upstream proxy groupings (`Cluster`) that artificially inflate random-split accuracy unless governed by explicit intake feature contracts.
4. **Precision-Recall Audit Cost Balance:** Standard accuracy and ROC-AUC understate false alarms on minority classes; investigators require **PR-AUC (Average Precision)** ranking precision paired with validation-calibrated decision cutoffs ($\tau^*$).
5. **Auditability & Deployment Flexibility:** Special Investigation Units (SIU) need interpretable scores, step-by-step audit traces, natural-language query support, and reproducible execution across both full-stack servers and standalone browsers.

Our **Health Insurance Fraud Intelligence Platform** addresses all five challenges by uniting **Classical Machine Learning (Approach 1)**, **Deep Learning (Approach 2)**, and **Agentic Explainable AI (Approach 3)** under a leakage-governed evaluation harness (**Track A + Track B**) and a modern full-stack web application (`React + Vite -> Express -> FastAPI`).

---

## 2. Brief Literature Survey Highlights

Our system design synthesizes empirical lessons across six pillars of fraud detection and tabular AI research:

* **Tree Ensembles in Healthcare Claims (`du Preez et al., 2025`; `Bauder & Khoshgoftaar, 2017`):** Systematic reviews of 137 healthcare fraud studies establish that Random Forests (`Breiman, 2001`) and gradient boosted decision trees (`XGBoost`, Chen & Guestrin, 2016) consistently excel on tabular insurance records containing nonlinear amount thresholds—while warning against random-only evaluation without chronological testing.
* **Tabular Deep Learning vs. Orthogonal Trees (`Johnson & Khoshgoftaar, 2019`; `Huang et al., 2020`; `Gorishniy et al., 2021`; `Grinsztajn et al., 2022`):** While `TabTransformer` self-attention over categorical embeddings and `ResidualMLP` skip connections narrow the neural-vs-tree gap, Grinsztajn et al. proved across 45 tabular benchmarks that tree ensembles remain highly resilient against uninformative features and sharp numeric step boundaries in ~10K-sample regimes.
* **Unsupervised Reconstruction Limits (`Sakurada & Yairi, 2014`; `Chalapathy & Chawla, 2019`):** Normal-trained autoencoders struggle when fraudulent observations occupy overlapping subspaces within legitimate distributions rather than lying on isolated manifold outliers.
* **Imbalance-Aware Evaluation (`Saito & Rehmsmeier, 2015`; `Chawla et al., 2002`):** On skewed fraud datasets, `PR-AUC` (`Average Precision`) reveals true precision degradation at high recall where ROC curves appear deceptively near `1.0`.
* **Leakage & Shortcut Learning Governance (`Kaufman et al., 2012`; `Geirhos et al., 2020`):** Predictive models frequently exploit dataset artifacts (`Cluster` concentration, hard billing floors) unless validated via progressive feature ablation and out-of-time splits.
* **Explainability & Agentic Workflows (`Lundberg & Lee, 2017`; `Kapadiya et al., 2022`; `Lewis et al., 2020`):** Modern enterprise SIU platforms pair model verdicts with local feature attribution (`SHAP`), tamper-evident audit logs, and LLM/RAG analyst assistants.

*(For full academic citations and synthesis matrices, see [docs/01_LITERATURE_SURVEY.md](./01_LITERATURE_SURVEY.md).)*

---

## 3. Core Innovations

1. **Leakage-First Dual-Track Feature Contract (`29f -> 28f -> 27f -> Temporal`):**
   We preserve strict backward compatibility in **Track A** (`29` encoded features, excluding unique IDs, free-text location, high-cardinality codes, and post-adjudication `ClaimStatus`), while introducing **Track B** (`submission/`) to audit `Cluster` leakage (`267/270` frauds concentrated in `Cluster 1`), measure `ClaimAmount` shortcut sensitivity (`>= $7,465.07`), and test strictly chronological future claims (`>= 2024-03-11`).
2. **Unified 10-Model Multi-Paradigm Benchmark + Documented Negative Result:**
   Evaluates **5 Classical ML** models (`LogisticRegression`, `RandomForest`, `XGBoost`, `SVM`, `HistGradientBoosting`) alongside **5 PyTorch Deep Learning** models (`MLP`, `ResidualMLP`, `CNN1D`, `TabTransformer`, and legitimate-only `Autoencoder`) on identical canonical splits. Publishing the `Autoencoder` negative result (`PR-AUC = 5.23%`, `F1 = 2.99%` at `MSE threshold = 0.1286`) rigorously validates supervised learning.
3. **Validation-Tuned Operating Thresholds ($\tau^*$) & 1,000-Replicate Bootstrap Confidence Intervals:**
   Replaces arbitrary `0.50` cutoffs with validation-selected thresholds maximizing `F1`, improving regularized neural/kernel F1 (e.g., `SVM Baseline` `83.52% -> 89.74%`, `ResidualMLP` `86.08% -> 92.86%`) and quantifying test uncertainty via 1,000 stratified bootstrap replicates.
4. **Dual-Path Production Serving (`3-Tier Web Stack` + `Zero-Server Browser Parity Lab`):**
   - **Path A (Enterprise Stack):** React 19 UI (`:5173`) $\rightarrow$ Express API Gateway (`:3000`) $\rightarrow$ FastAPI Python Service (`:8000`) serving live `.joblib` and `.pt` artifacts plus offline interactive fallbacks.
   - **Path B (`submission/` Pure-JS Lab):** Exports fitted scalars, encoders, trees, SVM RBF kernels, and PyTorch weights directly to JSON with **`1,350` preprocessing checks** (`0` mismatches) and **`10,800` float score comparisons** ($\max |\Delta| = 4.42 \times 10^{-7}$).
5. **Agentic Triage UI, Cryptographic XAI Scaffold & Live Gemini 2.0 Flash Assistant:**
   Visualizes a 6-role agentic workflow alongside the live Express `POST /agentic/predict` handler, `ScoreGauge` risk ring, filterable/exportable `AuditTrail` JSON, standalone `XAIExplainer` SHA-256 hashing (`Approach-3-Agentic-AI/`), and an integrated `gemini-2.0-flash` analyst drawer (`POST /api/chat`).

---

## 4. Implementation & Headline Results Summary

| Paradigm / Architecture | Track A (`29f` Baseline) Test `PR-AUC` | Track A Test `F1` / `ROC-AUC` | Track B (`28f` Intake, No Cluster) Test `AP` | Key Takeaway |
|---|---|---|---|---|
| **XGBoost** *(Classical ML)* | **100.00%** | `98.77%` / `100.00%` | **1.0000** | Top Track A & Chronological tie champion |
| **RandomForest** *(Classical ML)* | **99.94%** | `98.73%` / `99.996%` | **1.0000** | Track B Default Champion (`FP=0, FN=1`, $\tau^*=0.4167$) |
| **GradientBoosting / HGB** *(Classical ML)* | `97.65%` | `98.73%` / `98.47%` | `0.9765` | `1.0000 AP & F1` on chronological split (`>= 2024-03-11`) |
| **TabTransformer** *(Supervised DL)* | **99.11%** | `89.89%` / **99.94%** | — *(Track A)* | Best DL ranking quality (`40/40` fraud recall) |
| **CNN1D** *(Supervised DL)* | `98.82%` | **90.91%** / `99.92%` | `0.9667` | Strongest neural model under `Cluster`-free intake |
| **ResidualMLP / MLP** *(Supervised DL)* | `95.61%` / `95.38%` | `88.89%` / `90.70%` | `0.8765` / `0.8866` | Fast (`~0.001 ms/claim`), benefits from $\tau^*$ tuning |
| **SVM / LogisticRegression** *(Classical ML)* | `92.52%` / `91.86%` | `83.52%` / `76.92%` | `0.8181` / `0.8525` | Sensitive to non-orthogonal boundaries without `Cluster` |
| **Autoencoder** *(Unsupervised Neg. Control)* | **5.23%** | `2.99%` / `46.02%` | — *(Track A)* | Fails to isolate overlapping legitimate vs. fraud covariates |

---

## 5. Complete Documentation Index

* [00_BRIEF_INTRODUCTION.md](./00_BRIEF_INTRODUCTION.md) — Executive brief covering Introduction, Literature Survey, Innovation, and Implementation *(this file)*
* [01_LITERATURE_SURVEY.md](./01_LITERATURE_SURVEY.md) — Full thematic literature survey and research gap analysis
* [02_PROBLEM_FORMULATION.md](./02_PROBLEM_FORMULATION.md) — Mathematical formulation, leakage audit, and optimization protocol
* [03_INNOVATION_AND_IMPLEMENTATION.md](./03_INNOVATION_AND_IMPLEMENTATION.md) — Deep dive into innovations, model internals, and dual serving paths
* [04_FULL_PROJECT_DOCUMENTATION.md](./04_FULL_PROJECT_DOCUMENTATION.md) — Consolidated end-to-end engineering handbook
* [05_SYSTEM_ARCHITECTURE_FRONTEND_BACKEND_GUIDE.md](./05_SYSTEM_ARCHITECTURE_FRONTEND_BACKEND_GUIDE.md) — Detailed frontend UI/UX component reference, Express API gateway, FastAPI `ml-service`, and Agentic microservice guide
* [06_DATA_MODEL_TRAINING_AND_TESTING_GUIDE.md](./06_DATA_MODEL_TRAINING_AND_TESTING_GUIDE.md) — Complete data pipeline, Track B ablations, verification proofs, and operational runbook
