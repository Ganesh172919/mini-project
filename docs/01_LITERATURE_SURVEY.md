# Literature Survey: Medical Insurance Claim Fraud Detection

**Project:** Health Insurance Fraud Intelligence Platform (`Medical Insurance Claim Fraud Detection`)  
**Institution:** Department of Computer Science and Engineering, Indian Institute of Information Technology (IIIT) Dharwad  
**Team:** B. Varshith (`23BDS011`), J. Ganesh Kumar Reddy (`23BDS024`), M. Jagadeeswar (`23BDS033`) — Supervised by **Prof. Ramesh Athe**

---

## 1. Scope and Taxonomy of Reviewed Literature

Healthcare expenditure loss from fraudulent, wasteful, and abusive claims poses a major challenge across public and private insurance programs (Bauder & Khoshgoftaar, 2017; du Preez et al., 2025). Unlike general binary classification tasks, medical claim fraud detection combines five coupled challenges: (i) **extreme class imbalance** (in our operational corpus, only 270 of 4,500 claims — 6.00% — are fraudulent), (ii) **mixed structured tabular and clinical categorical attributes**, (iii) **upstream label leakage and shortcut features**, (iv) **temporal concept drift**, and (v) **regulatory requirements for auditability and human-in-the-loop explainability**.

To situate our three-approach architecture—Classical Machine Learning (**Approach 1**), Deep Learning (**Approach 2**), and Agentic AI with Explainable AI (**Approach 3**)—we review foundational and recent literature across six thematic pillars:

1. Systematic Reviews & Classical Supervised Machine Learning on Healthcare Claims
2. Deep Learning for Tabular Data (MLP, ResNet, CNN1D, and Attention Transformers)
3. Unsupervised Anomaly Detection via Autoencoders
4. Class Imbalance Handling & Imbalance-Aware Evaluation Metrics
5. Data Leakage Governance & Shortcut Learning
6. Explainable AI (XAI), Retrieval-Augmented Generation (RAG), and Agentic Claim Workflows

---

## 2. Thematic Literature Review

### 2.1 Classical Supervised ML & Systematic Reviews in Healthcare Claims
* **du Preez, Bhattacharya, Beling, and Bowen (2025/2024)** (*Artificial Intelligence in Medicine*, Vol. 160, 103061, DOI: `10.1016/j.artmed.2024.103061`) conducted a systematic literature review examining **137 primary studies** on machine learning for healthcare claims fraud detection. They observed that tree-based ensemble learners consistently dominate structured claim tabular benchmarks, while identifying persistent methodological weaknesses across the literature: inconsistent validation protocols, reliance on random splits over chronological splits, and sparse evaluation of false-positive audit cost.
* **Bauder and Khoshgoftaar (2017)** (*IEEE ICMLA*, DOI: `10.1109/ICMLA.2017.00048`) evaluated supervised Medicare Part B fraud detection using tree ensembles and margin-based classifiers under realistic class skew, showing that Random Forest and gradient-boosted learners outperform single linear models when nonlinear thresholds interact with claim amounts and provider characteristics.
* **Kapadiya et al. (2022)** (*IEEE Access*, Vol. 10, pp. 79606–79627, DOI: `10.1109/ACCESS.2022.3194569`) reviewed AI-empowered healthcare insurance fraud detection architectures alongside immutable audit trails, underscoring the dual need for predictive sensitivity and tamper-evident audit logging across multi-party claims workflows.
* **Foundational Algorithms Applied in Track A & B:** **Random Forests** (Breiman, 2001, *Machine Learning*, 45(1), DOI: `10.1023/A:1010933404324`), **XGBoost** (Chen & Guestrin, 2016, *ACM SIGKDD*, DOI: `10.1145/2939672.2939785`), **Histogram-Based Gradient Boosting** (Ke et al., 2017, *NeurIPS* LightGBM paradigm), and **Support Vector Machines** (Cortes & Vapnik, 1995, *Machine Learning*, DOI: `10.1007/BF00994018`) provide complementary inductive biases ranging from linear decision boundaries and smooth RBF margins to axis-aligned orthogonal tree partitions.

### 2.2 Deep Learning for Tabular Claims Data
* **Johnson and Khoshgoftaar (2019)** (*Journal of Big Data*, Vol. 6, Article 63, DOI: `10.1186/s40537-019-0225-0`) benchmarked neural networks for Medicare claim fraud under severe class imbalance, demonstrating that deep neural architectures require tuned class weighting or threshold calibration to prevent minority-class collapse.
* **He et al. (2016)** (*IEEE CVPR*, DOI: `10.1109/CVPR.2016.90`) established identity skip connections (`Residual` blocks), which **Gorishniy, Rubachev, Khrulkov, and Babenko (2021)** (*NeurIPS*) adapted into tabular Residual MLPs (`ResNet`-style tabular networks) alongside Feature Tokenizer Transformers.
* **Huang, Khetan, Cvitkovic, and Karnin (2020)** (*arXiv:2012.06678*) proposed **TabTransformer**, applying multi-head self-attention to categorical feature embeddings before concatenating normalized continuous attributes. Our Approach 2 implements and benchmarks a dedicated PyTorch **TabTransformer** (reaching 99.11% PR-AUC and 99.94% ROC-AUC on the 29-feature contract), directly comparing it against `MLP`, `ResidualMLP`, and `CNN1D`.
* **Grinsztajn, Oyallon, and Varoquaux (2022)** (*NeurIPS*, *arXiv:2207.08815*) systematically studied 45 tabular benchmarks (~10K-sample regime) and explained why tree ensembles still frequently outperform neural networks on medium-sized tabular problems: (1) orthogonal tree splits fit non-smooth threshold functions naturally, (2) neural networks suffer more from uninformative or noisy columns, and (3) trees are invariant to monotonic scale transforms. Our project's empirical comparison (both in 29-feature and 28-feature `Cluster`-free intake regimes, where `RandomForest` and `XGBoost` achieve 1.0000 PR-AUC vs. 0.8765–0.9667 for `ResidualMLP`/`MLP`/`CNN1D`) provides concrete, reproducible evidence supporting Grinsztajn et al.'s findings.

### 2.3 Unsupervised Anomaly Detection via Autoencoders
* **Sakurada and Yairi (2014)** (*MLSDA Workshop*) introduced nonlinear autoencoder reconstruction error as an unsupervised anomaly detector capable of capturing nonlinear subspaces missed by PCA.
* **Chalapathy and Chawla (2019)** surveyed deep anomaly detection frameworks for fraud and cybersecurity, emphasizing a critical failure mode: when fraudulent instances lie inside the high-density support of legitimate covariates—or differ only via subtle boundary shifts rather than structural manifold breaks—normal-trained autoencoders can reconstruct anomalous points accurately and yield poor discrimination.
* **Empirical Relevance:** We explicitly tested this hypothesis in Approach 2 using a symmetric PyTorch Autoencoder trained only on legitimate training claims (`y = 0`), establishing its reconstruction MSE threshold (0.1286) via the 95th percentile of validation errors. It yielded a decisive **negative result** (`PR-AUC = 5.23%`, `ROC-AUC = 46.02%`, `F1 = 2.99%`), empirically demonstrating why supervised classification is essential on this dataset.

### 2.4 Class Imbalance & Evaluation Metrics
* **Chawla, Bowyer, Hall, and Kegelmeyer (2002)** (*JAIR*, Vol. 16, pp. 321–357, DOI: `10.1613/jair.953`) introduced **SMOTE** synthetic minority oversampling, while weighted binary cross-entropy (`pos_weight`) and cost-sensitive class weights offer leakage-safe in-training alternatives that avoid synthesizing invalid categorical combinations.
* **Saito and Rehmsmeier (2015)** (*PLOS ONE*, 10(3):e0118432, DOI: `10.1371/journal.pone.0118432`) proved mathematically and empirically that **Precision-Recall (PR) curves and Average Precision (AP)** are far more informative than ROC curves on imbalanced datasets because ROC false-positive rate dilutes false positives by the massive negative denominator ($4,230$ legitimate claims), masking high false-alarm operational overload. In our project, `PR-AUC` (`Average Precision`) is elevated to primary selection metric across both Track A and Track B, paired with validation-split `F1` threshold tuning rather than default `0.50` cutoffs.

### 2.5 Data Leakage & Shortcut Learning Governance
* **Kaufman, Rosset, Perlich, and Stitelman (2012)** (*ACM TKDD*, 6(4), DOI: `10.1145/2382577.2382579`) formalized data leakage as any training feature carrying information unavailable at prediction time or derived from target-correlated upstream processes, separating preprocessing-pipeline leakage from feature-level target proxy leakage.
* **Geirhos et al. (2020)** (*Nature Machine Intelligence*, DOI: `10.1038/s42256-020-00257-z`) defined **shortcut learning**—decision rules that generalize across in-distribution i.i.d. splits using spurious or dataset-construction artifacts rather than domain causal signals.
* **Project Discovery:** Auditing the 4,500-claim workbook revealed both risks: (1) upstream column `Cluster` (`{0, 1, 2, 3}`) concentrates **267 of 270 frauds (98.89%) inside Cluster 1** (24.27% fraud rate vs. 0.00% in Clusters 0 & 2 and 0.26% in Cluster 3), and (2) `ClaimAmount` lies strictly above `7,465.07` for all 270 frauds while legitimate claims span `100.12–9,997.20`. Removing `Cluster` (`Intake` 28-feature protocol) and ablating `ClaimAmount` (`No-Amount` 27-feature protocol, where AP collapses from `1.0000` to `0.1869–0.2404`) turns these literature theories into explicit experimental controls.

### 2.6 Explainable AI (XAI), RAG, and Agentic Claim Processing
* **Lundberg and Lee (2017)** (*NeurIPS*, pp. 4765–4774) unified feature attribution under Shapley values (**SHAP**), providing additive local explanations essential when Special Investigation Units (SIU) review flagged insurance claims.
* **Lewis et al. (2020)** (*NeurIPS*, pp. 9459–9474, *arXiv:2005.11401*) introduced **Retrieval-Augmented Generation (RAG)**, grounding LLM outputs in external non-parametric corpora to mitigate hallucination. Recent 2024–2025 enterprise architectures extend RAG into multi-agent claim triage (Document Intake, Policy Verification, Fraud Detection, and XAI Synthesizer agents). Our Approach 3 maps out this 6-role multi-agent UI design alongside a working Gemini `gemini-2.0-flash` `/api/chat` assistant, a deterministic 4-stage `/agentic/predict` route, and a SHA-256 audit + SHAP-approximation `XAIExplainer` scaffold—while explicitly documenting where live FastAPI/RAG wiring sits on the roadmap.

---

## 3. Comparative Synthesis Matrix

| Research Theme | Key References | Core Insight / Limitation in Literature | How Addressed in This Project |
|---|---|---|---|
| **Systematic Review & Classical Ensembles** | du Preez et al. (2025); Bauder & Khoshgoftaar (2017); Breiman (2001); Chen & Guestrin (2016) | Tree ensembles excel on tabular claims, yet studies often lack temporal splits and audit-cost evaluation. | Implemented 5 balanced/regularized classical models (`LR`, `RF`, `XGB`, `SVM`, `HGB`); benchmarked random and chronological splits. |
| **Tabular Deep Learning vs. Trees** | Johnson & Khoshgoftaar (2019); Huang et al. (2020); Gorishniy et al. (2021); Grinsztajn et al. (2022) | Contextual embeddings (`TabTransformer`) & residual blocks improve NN tabular accuracy, though trees retain sharp boundary edges. | Built and evaluated 4 supervised PyTorch models (`MLP`, `ResidualMLP`, `CNN1D`, `TabTransformer`); confirmed `TabTransformer` leads DL (`AP=0.9911`) while trees dominate intake (`AP=1.0000`). |
| **Unsupervised Anomaly Detection** | Sakurada & Yairi (2014); Chalapathy & Chawla (2019) | Autoencoders struggle when anomalies share global manifold support with normal examples. | Trained legitimate-only PyTorch Autoencoder; documented `AP=0.0523` / `F1=2.99%` negative result transparently. |
| **Imbalanced Evaluation & Leakage** | Saito & Rehmsmeier (2015); Kaufman et al. (2012); Geirhos et al. (2020) | `ROC-AUC` misleads under heavy skew; uninspected datasets harbor post-hoc proxy columns and single-feature shortcuts. | Adopted `PR-AUC` + val-tuned `F1`; audited `Cluster` & `ClaimAmount` via `Intake` (28f), `No-Amount` (27f), and `Temporal` experiments. |
| **XAI & Agentic Insurance AI** | Lundberg & Lee (2017); Lewis et al. (2020); Kapadiya et al. (2022) | SIU teams require factor attributions, natural-language policy Q&A, and tamper-evident audit hashes alongside raw scores. | Built 6-role Agentic UI, Gemini 2.0 Flash `/api/chat`, `AgenticEnsemble-v1` 4-event API, `XAIExplainer` SHA-256 log utility, and exact parity browser lab. |

---

## 4. Identified Research & Engineering Gaps Driving Our Design

1. **Lack of Multi-Paradigm Controlled Comparison:** Prior student/benchmark prototypes rarely compare Classical ML, 5 distinct Deep Learning paradigms (including Attention Transformers and Autoencoders), and Agentic workflows on a shared canonical split.
2. **Neglect of Upstream Feature Leakage:** Treating arbitrary dataset indices or upstream clustering fields (`Cluster`) as legitimate runtime intake features inflates accuracy without testing deployment realism.
3. **Hard-Coded `0.50` Probability Cutoffs vs. Cost-Aware Validation Tuning:** Using `0.50` across uncalibrated models produces misleading precision-recall tradeoffs; validation-optimized thresholds ($\tau^*$) plus explicit error inspection are necessary.
4. **Reproducibility & Deployment Portability:** Full-stack microservices (`React -> Node -> FastAPI`) offer enterprise flexibility, yet lack zero-dependency static auditability unless verified weights can also run identically in client-side browsers with measured numerical parity.
