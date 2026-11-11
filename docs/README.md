# Documentation Hub: Health Insurance Fraud Intelligence Platform

Welcome to the unified documentation and deliverable index for the **Medical Insurance Claim Fraud Detection (`Health Insurance Fraud Intelligence Platform`)** project, developed at the **Department of Computer Science and Engineering, IIIT Dharwad** under the supervision of **Prof. Ramesh Athe**.

## Team & Module Contributions
- **B. Varshith (`23BDS011`)** — *Approach 3*: Agentic workflow architecture, `AgenticEnsemble-v1` route, 6-role React UI pipeline, `XAIExplainer` scaffold, and Gemini 2.0 Flash `/api/chat` integration.
- **J. Ganesh Kumar Reddy (`23BDS024`)** — *Approach 1 & Research Lab*: Dataset audit, leakage/shortcut isolation, 5 Classical ML models, `submission/` 4-experiment ablation suite, bootstrap CIs, and zero-server Claim Fraud Lab.
- **M. Jagadeeswar (`23BDS033`)** — *Approach 2 & Full-Stack Integration*: 5 PyTorch Deep Learning models (`MLP`, `ResidualMLP`, `CNN1D`, `TabTransformer`, `Autoencoder`), FastAPI `:8000` model server, Node/Express `:3000` gateway, and React `:5173` workbench.

---

## Core Technical Documentation (`docs/` & Compiled PDFs in `deliverables/`)
1. [**01_LITERATURE_SURVEY.md**](./01_LITERATURE_SURVEY.md) ([PDF](../deliverables/01_Literature_Survey.pdf)) — Thematic survey of 20 peer-reviewed & archival sources across healthcare fraud ML benchmarks, tabular Deep Learning vs. trees, autoencoders, PR-AUC evaluation under imbalance, data leakage / shortcut learning, SHAP XAI, and RAG / Agentic workflows.
2. [**02_PROBLEM_FORMULATION.md**](./02_PROBLEM_FORMULATION.md) ([PDF](../deliverables/02_Problem_Formulation.pdf)) — Mathematical formulation of raw dataset schema (`N=4,500`, `6.00%` fraud), upstream `Cluster` leakage audit, weighted BCE risk minimization, reconstruction MSE detection, validation-tuned threshold selection, and 4 feature/split contracts.
3. [**03_INNOVATION_AND_IMPLEMENTATION.md**](./03_INNOVATION_AND_IMPLEMENTATION.md) ([PDF](../deliverables/03_Innovation_and_Implementation.pdf)) — Five key architectural/methodological innovations, complete implementation blueprints for Approaches 1–3, Track A (3-tier full-stack) & Track B (zero-server browser parity `<4.42e-7`), and documented honest boundaries.
4. [**04_FULL_PROJECT_DOCUMENTATION.md**](./04_FULL_PROJECT_DOCUMENTATION.md) ([PDF](../deliverables/04_Full_Project_Documentation.pdf)) — Repository map, Track A (10-model) & Track B (32-run) benchmark ledger, local developer setup instructions, API request/response reference, and honest engineering discrepancy log.

---

## Submission Deliverables (`deliverables/`)
- **IIIT Dharwad Mini-Project Report (LaTeX + Template-Compliant Source + PDF):**
  - Source package built from `MiniProjectTemplate.zip`: [`deliverables/MiniProjectReport_LaTeX/MiniProjectReport.tex`](../deliverables/MiniProjectReport_LaTeX/MiniProjectReport.tex) + [`Myreferences.bib`](../deliverables/MiniProjectReport_LaTeX/Myreferences.bib) + figures/logo
  - Compiled readable PDF: [`deliverables/MiniProjectReport.pdf`](../deliverables/MiniProjectReport.pdf)
- **IEEE Conference Research Paper (LaTeX + PDF + DOCX):**
  - `IEEEtran` two-column source: [`deliverables/IEEE_Paper_LaTeX/ieee_paper.tex`](../deliverables/IEEE_Paper_LaTeX/ieee_paper.tex) + [`references.bib`](../deliverables/IEEE_Paper_LaTeX/references.bib)
  - Compiled two-column PDF: [`deliverables/IEEE_Paper.pdf`](../deliverables/IEEE_Paper.pdf)
  - Editable IEEE manuscript Word document: [`deliverables/IEEE_Paper.docx`](../deliverables/IEEE_Paper.docx)
- **Project Presentation & Speaker Notes:**
  - Full-project 20-slide widescreen deck: [`deliverables/Project_Presentation.pptx`](../deliverables/Project_Presentation.pptx) | [`deliverables/Project_Presentation.pdf`](../deliverables/Project_Presentation.pdf)
  - Companion speaker script & slide walkthrough: [`deliverables/Project_Presentation_Notes.md`](../deliverables/Project_Presentation_Notes.md)
- **Unified Figures (`deliverables/figures/`):**
  - `fig1_system_architecture.png` through `fig6_verification_and_error.png`
