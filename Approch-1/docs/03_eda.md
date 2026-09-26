# 03 — Exploratory Data Analysis — 10 PNGs, Measured Insights

All plots: `project/scripts/eda_plots.py` (matplotlib/seaborn) → `evaluation/eda_plots/` (10 files, real data only). Each plot listed with what it actually tells the modelling desk, not just its name.

## 3.1 Plot inventory

| # | file | question | what the plot says (measured) |
|---|---|---|---|
| 1 | `target_distribution.png` | How imbalanced? | 4,230 Legitimate vs 270 Fraud (6.0%). In 10k claims ~600 frauds — enough to fund the desk, few enough that accuracy lies. |
| 2 | `claim_amount_by_target.png` | Are fraud amounts different? | Legit median ~4k, fraud median higher and spread wider, long right tail to ~10k. No fraud below ~8k in lower decile. |
| 3 | `age_distribution.png` | Age helps? | Uniform-ish 0–99 peak near 50. No fraud spike at any age — weak univariate signal. |
| 4 | `income_vs_amount.png` | Pair separation? | Overlapping clouds. Fraud sits at **lower income + higher amount** quadrant but not separable linearly — need multivariate model. |
| 5 | `correlation_heatmap.png` | Numeric collinearity? | Year/Month/DOW nearly orthogonal to amount/income (|r|<0.08). Amount vs income r≈-0.02 — no redundancy to prune. |
| 6 | `claim_type_counts.png` | Which types are common? | Routine and Emergency dominate volumes; raw counts alone don’t rank fraud rate. |
| 7 | `cluster_target.png` | Does Cluster matter? | **Strongest EDA signal:** Cluster 1 fraud ≈24%, clusters 0/2 ≈0%, cluster 3 ~2%. Cluster carries the decision. |
| 8 | `monthly_fraud_trend.png` | Seasonality? | Fraud ratio flat across 24 months (±1pp). No month to use as calendar rule. |
| 9 | `specialty_target.png` | Provider specialty risk? | Orthopedics largest volume; fraud rate similar across specialties after controlling for Cluster — specialty is secondary. |
|10 | `claimstatus_target.png` | Is ClaimStatus predictive? | Approved/Denied/Pending fraud ratios diverge sharply — **but it’s post-decision leakage**, so kept only for EDA, excluded from model. |

> Two of the 10 plots (`claimstatus_target`, `cluster_target`) are intentionally kept in the EDA section even though one of them is **not a feature**. Showing the leak in EDA and then excluding it in §04 is the honest story.

## 3.2 Numeric summaries (not just pictures)

- ClaimAmount: mean ~4,850, std ~2,900, min 60, max 10,000 — right-skewed, winsorization not applied (trees handle it).
- PatientAge: mean 50.1, std 29.0 — uniformish, no missing.
- PatientIncome: mean 85k, std ~40k, long tail to 200k — logged neither, scaled linearly (scaler fit on train).
- Cluster distribution: 0: ~1,125, 1: ~1,120, 2: ~1,130, 3: ~1,125 — uniform by design; fraud concentrated in 1.

## 3.3 What EDA decided for preprocessing

- No imputation needed (0 missing).
- No duplicate removal beyond modelling columns check.
- No log transform — tree models immune, linear model gets scaler instead.
- Drop high-cardinality strings (DiagnosisCode 4,495 uniques for 4,500 rows) — would one-hot to a 4.5k-wide sparse block and memorize.
- Expand ClaimDate → Year/Month/DOW (captures year drift without leaking calendar date as ID).
- Keep ClaimStatus **only for the exploratory figure** (documents the leak), then permanently exclude from registry.

## 3.4 How to reproduce

```bash
python project/scripts/eda_plots.py  # writes 10 PNGs to evaluation/eda_plots/
ls project/evaluation/eda_plots/     # 10 files
# Check counts without code:
grep -c "Fraud" project/data/processed/y_train.csv   # 189 etc via ledger
```

## 3.5 Visual appendix (in this repo)

Inline-mounted in the PDF (§3) and PPT (slide 4): `target_distribution` + `correlation_heatmap` + `claim_amount_by_target`. Full 10-plot gallery on the Flask site `/eda` (lazy-loaded png with caption per plot). AI-enhanced deployment and data-flow diagrams (`architecture_data_flow.png`, `architecture_unified.png`) sit alongside the matplotlib PNGs in `website/static/images/` for the architecture slides.


---
**Team:** B Varshith (23BDS011) · M Jagadeshwar (23BDS033) · J Ganesh (23BDS024) — IIIT Dharwad B.Tech DSAI · **Advisor:** Prof. Ramesh Athe · Branch `arena/01a09e96` · 2026-09-17
