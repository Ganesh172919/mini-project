"""
Train 5 ML models for Approach 1 using only existing data.
Outputs: models, confusion matrix PNGs, metrics CSV, ROC curves.
"""
import numpy as np
import pandas as pd
import joblib
import os, json, time, datetime, platform, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)
import matplotlib.pyplot as plt

# Load processed data
X_train = np.load('project/data/processed/X_train.npy')
y_train = np.load('project/data/processed/y_train.npy')
X_val = np.load('project/data/processed/X_val.npy')
y_val = np.load('project/data/processed/y_val.npy')
X_test = np.load('project/data/processed/X_test.npy')
y_test = np.load('project/data/processed/y_test.npy')

models_dir = 'project/models'
eval_dir = 'project/evaluation'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# Define models
models = {
    'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs', random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                             scale_pos_weight=2961/189, random_state=42, n_jobs=-1),
    'SVM': SVC(probability=True, class_weight='balanced', kernel='rbf', random_state=42),
    'GradientBoosting': HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
}

results = []
run_log = {'models': {}}
_start = time.time()

for name, model in models.items():
    print(f"\nTraining {name} ...")
    _t0 = time.time()
    model.fit(X_train, y_train)
    _fit_seconds = time.time() - _t0
    # Validation predictions
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_val)
    # Test predictions
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

    # Metrics on validation (primary comparison)
    metrics = {
        'Model': name,
        'Val_Accuracy': accuracy_score(y_val, y_val_pred),
        'Val_Precision': precision_score(y_val, y_val_pred, zero_division=0),
        'Val_Recall': recall_score(y_val, y_val_pred, zero_division=0),
        'Val_F1': f1_score(y_val, y_val_pred, zero_division=0),
        'Val_ROC_AUC': roc_auc_score(y_val, y_val_prob),
        'Val_PR_AUC': average_precision_score(y_val, y_val_prob),
        'Test_Accuracy': accuracy_score(y_test, y_test_pred),
        'Test_Precision': precision_score(y_test, y_test_pred, zero_division=0),
        'Test_Recall': recall_score(y_test, y_test_pred, zero_division=0),
        'Test_F1': f1_score(y_test, y_test_pred, zero_division=0),
        'Test_ROC_AUC': roc_auc_score(y_test, y_test_prob),
        'Test_PR_AUC': average_precision_score(y_test, y_test_prob)
    }
    results.append(metrics)

    # Save model
    _path = os.path.join(models_dir, f"{name}.joblib")
    joblib.dump(model, _path)
    _t_inf = time.time()
    _ = model.predict(X_test[:100])
    _predict_100_s = time.time() - _t_inf
    run_log['models'][name] = {
        'fit_seconds': round(_fit_seconds, 3),
        'predict_100_seconds': round(_predict_100_s, 4),
        'model_bytes': os.path.getsize(_path),
        'n_train': int(len(y_train)),
        'positive_class_weight': 'balanced',
        'test_metrics': {k: float(v) for k, v in metrics.items() if k != 'Model'}
    }
    print(f"  fit {_fit_seconds:.2f}s | predict 100 rows {_predict_100_s:.4f}s | "
          f"test F1 {metrics['Test_F1']:.4f} | test ROC-AUC {metrics['Test_ROC_AUC']:.4f}")

    # Confusion matrix PNG
    fig, ax = plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=ax, cmap='Blues', colorbar=True)
    ax.set_title(f'{name} — Test Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"cm_{name}.png"), dpi=150)
    plt.close()

    # ROC curve PNG
    fig, ax = plt.subplots(figsize=(6,5))
    RocCurveDisplay.from_predictions(y_test, y_test_prob, ax=ax, name=f'{name} (AUC={metrics["Test_ROC_AUC"]:.3f})')
    ax.set_title(f'{name} — ROC Curve (Test)')
    ax.plot([0,1],[0,1],'k--')
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"roc_{name}.png"), dpi=150)
    plt.close()

    # Precision-Recall PNG
    fig, ax = plt.subplots(figsize=(6,5))
    PrecisionRecallDisplay.from_predictions(y_test, y_test_prob, ax=ax, name=f'{name} (AP={metrics["Test_PR_AUC"]:.3f})')
    ax.set_title(f'{name} — PR Curve (Test)')
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"pr_{name}.png"), dpi=150)
    plt.close()

# Save aggregated results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(eval_dir, 'model_results.csv'), index=False)
results_df.to_json(os.path.join(eval_dir, 'model_results.json'), orient='records', indent=2)

# Best model by validation F1
best = results_df.loc[results_df['Val_F1'].idxmax(), 'Model']
with open(os.path.join(eval_dir, 'best_model.txt'), 'w') as f:
    f.write(best)

print("\nAll 5 models trained and saved.")
print(results_df[['Model','Val_Accuracy','Val_F1','Val_ROC_AUC','Test_F1','Test_ROC_AUC']].to_string(index=False))
print("Best model by Val F1:", best)

# ---- detailed run log -------------------------------------------------
logs_dir = 'project/logs'
os.makedirs(logs_dir, exist_ok=True)
run_log.update({
    'generated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'wall_clock_seconds': round(time.time() - _start, 2),
    'python': platform.python_version(),
    'scikit_learn': sklearn.__version__,
    'split': {'train': int(len(y_train)), 'val': int(len(y_val)), 'test': int(len(y_test)),
              'train_fraud': int(y_train.sum()), 'val_fraud': int(y_val.sum()), 'test_fraud': int(y_test.sum())},
    'n_features': int(X_train.shape[1]),
    'random_state': 42,
    'best_model_by_val_f1': best,
    'data_source': 'Health Insurance Fraud Claims.xlsx (repository dataset, no synthetic data)'
})
with open(os.path.join(logs_dir, 'training_metrics.json'), 'w', encoding='utf-8') as fh:
    json.dump(run_log, fh, indent=2)
print("\nDetailed training log written to", os.path.join(logs_dir, 'training_metrics.json'))
