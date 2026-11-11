# Speaker Notes & Walkthrough Guide: Project Presentation (`Project_Presentation.pptx`)

**Duration Target:** 12–15 minutes (~40 seconds per slide)  
**Presenters:** B. Varshith (`23BDS011`), J. Ganesh Kumar Reddy (`23BDS024`), M. Jagadeeswar (`23BDS033`)

---

## Slide-by-Slide Talking Points

1. **Slide 1 — Title Slide:** Introduce the project title, team from IIIT Dharwad (CSE/DSAI), faculty guide Prof. Ramesh Athe, and core objective: uniting Classical ML, Deep Learning, and Agentic/XAI tooling under strict leakage governance.
2. **Slide 2 — Team Ownership:** Explicitly credit each module across Approaches 1, 2, and 3—showing how individual approaches integrate cleanly into two synchronized deployment paths rather than isolated notebooks.
3. **Slide 3 — Problem Motivation:** Explain why high raw accuracy (94%) is meaningless on 6% fraud prevalence, why upstream `Cluster` leakage and single-feature amount shortcuts deceive naive benchmarks, and why human auditors need explanations alongside risk scores.
4. **Slide 4 — Literature Survey:** Highlight how systematic reviews (`du Preez et al., 2025`), tabular benchmarks (`Grinsztajn et al., 2022`), imbalance evaluation (`Saito & Rehmsmeier, 2015`), leakage theory (`Kaufman et al., 2012`), and SHAP/RAG directly shaped our experimental protocol.
5. **Slide 5 — Dataset Audit:** Walk through Panel A showing 267 of 270 frauds packed into Cluster 1 (`24.27%`) and Panel B showing `PatientIncome` (44.00%) and `ClaimAmount` (43.16%) driving Random Forest decisions once `Cluster` is excluded.
6. **Slide 6 — Mathematical Formulation:** Contrast weighted BCE (`pos_weight = 15.67`) for supervised networks against legitimate-only MSE reconstruction loss (`tau_AE = 0.1286`), and introduce our 4 controlled contracts (`Baseline`, `Intake`, `No-Amount`, `Temporal`).
7. **Slide 7 — Unified System Architecture:** Trace flow from the immutable Excel corpus -> training-only `ColumnTransformer` -> Approaches 1–3 -> Full-Stack Path A (`React -> Node -> FastAPI`) and Zero-Server Browser Lab Path B (`:8080`).
8. **Slide 8 — Approach 1 (Classical ML):** Summarize the 5 algorithms and highlight why orthogonal tree splits outperform smooth hyperplanes (`LR`, `RBF SVM`) when categorical one-hot noise interacts with continuous billing boundaries.
9. **Slide 9 — Approach 2 (Deep Learning):** Present the 4 supervised PyTorch architectures (`MLP`, `ResidualMLP`, `CNN1D`, `TabTransformer`) plus our transparent negative Autoencoder study (`PR-AUC = 5.23%`, explaining manifold overlap).
10. **Slide 10 — Track A 10-Model Benchmark:** Compare all 10 models on the canonical 29-feature test set: `XGBoost` (100% PR-AUC), `RandomForest` (99.94%), `TabTransformer` leading neural models (99.11%), `CNN1D` (98.82% PR-AUC / 90.91% F1), and the `Autoencoder` contrast.
11. **Slide 11 — Approach 3 (Agentic AI & XAI):** Present an honest engineering audit distinguishing working runtime endpoints (`/agentic/predict`, Google Gemini 2.0 Flash `/api/chat`), UI/scaffold assets (6-role cards, SHA-256 `XAIExplainer`), and Phase-2 LangGraph/RAG bindings.
12. **Slide 12 — Track B Ablation Matrix:** Walk through the 4-experiment heatmap: point out tree stability on Cluster removal & future temporal rollout (`1.0000` PR-AUC), non-tree PR-AUC drop (`-6.6` to `-10.7`), and the sharp `No-Amount` drop (`0.1869–0.2404`).
13. **Slide 13 — Quantitative Ledger:** Share exact `validation_f1` thresholds and test error counts (`Intake RF`: 0 FP / 1 FN at `tau*=0.4167`; `XGBoost`: 1 FP / 0 FN at `tau*=0.2121`).
14. **Slide 14 — Champion Forensics & Parity Verification:** Show Row 1124 (ClaimAmount `$7,465.07`—lowest fraud amount in dataset, score `0.27`), bootstrap 95% F1 CI `[0.9610, 1.0000]`, and zero-mismatch Python-vs-Browser verification (`4.42e-7` max error across 10,800 scores).
15. **Slide 15 — Path A Full-Stack Web App:** Walk through the React (`:5173`) -> Express (`:3000`) -> FastAPI (`:8000`) microservice integration including resilient offline demos.
16. **Slide 16 — Path B Claim Fraud Lab:** Show the zero-server Claim Fraud Lab UI executing all 8 models in pure client-side JavaScript.
17. **Slide 17 — Innovations Summary:** Recap the 5 engineering & research contributions.
18. **Slide 18 — Deliverables Index:** Point evaluators to `docs/` (Literature Survey, Problem Formulation, Innovation & Implementation, Full Documentation) and `deliverables/` (IIIT Dharwad Report LaTeX+PDF, IEEE LaTeX+PDF+DOCX, and Slides).
19. **Slide 19 — Honest Limitations & Roadmap:** Address `ClaimAmount` reliance, uncalibrated scores, `/agentic/predict` live binding, and OCR/RAG/PostgreSQL phase-2 steps candidly.
20. **Slide 20 — Conclusion & Demo Transition:** Conclude cleanly and invite judges to view a live prediction & Gemini co-pilot demo.
