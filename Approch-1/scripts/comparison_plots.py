"""Generate comparison bar charts and feature importance plots."""
import pandas as pd
import numpy as np
import joblib
import os, matplotlib.pyplot as plt

results = pd.read_csv('project/evaluation/model_results.csv')
res_dir = 'project/evaluation'

# 1. Comparison bar chart: Val F1 and Test F1
fig, ax = plt.subplots(figsize=(10,6))
x = np.arange(len(results))
width = 0.35
bars1 = ax.bar(x - width/2, results['Val_F1'], width, label='Val F1', color='steelblue')
bars2 = ax.bar(x + width/2, results['Test_F1'], width, label='Test F1', color='coral')
ax.set_ylabel('F1 Score')
ax.set_title('Model Performance Comparison — F1 Score (Real Data Only)')
ax.set_xticks(x)
ax.set_xticklabels(results['Model'], rotation=30, ha='right')
ax.set_ylim(0.7, 1.05)
ax.legend()
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(res_dir, 'model_comparison_f1.png'), dpi=150)
plt.close()

# 2. ROC AUC comparison
fig, ax = plt.subplots(figsize=(10,6))
bars = ax.barh(results['Model'], results['Test_ROC_AUC'], color='seagreen')
ax.set_xlabel('Test ROC AUC')
ax.set_title('Test ROC AUC by Model')
ax.set_xlim(0.95, 1.01)
for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{width:.4f}', xy=(width, bar.get_y()+bar.get_height()/2),
                xytext=(3,0), textcoords="offset points", ha='left', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(res_dir, 'model_comparison_roc.png'), dpi=150)
plt.close()

# 3. Feature importance for Random Forest, XGBoost, GradientBoosting
with open('project/data/processed/feature_names.txt') as f:
    feature_names = f.read().strip().split('\n')

tree_models = {'RandomForest':'RandomForest', 'XGBoost':'XGBoost', 'GradientBoosting':'GradientBoosting'}
for fname, mname in tree_models.items():
    model = joblib.load(f'project/models/{mname}.joblib')
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        importances = model.named_steps['classifier'].feature_importances_
    else:
        # try underlying booster
        importances = model.get_booster().get_score(importance_type='gain') if hasattr(model, 'get_booster') else None
        if importances is not None:
            # xgboost booster score is dict with feature names as strings; map to index
            pass
    if importances is not None and isinstance(importances, np.ndarray):
        idx = np.argsort(importances)[-15:][::-1]
        plt.figure(figsize=(10,6))
        plt.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1], color='teal')
        plt.xlabel('Importance')
        plt.title(f'{mname} — Top 15 Feature Importances (Real Data)')
        plt.tight_layout()
        plt.savefig(os.path.join(res_dir, f'feature_importance_{mname}.png'), dpi=150)
        plt.close()

print("Comparison and feature importance plots saved.")
