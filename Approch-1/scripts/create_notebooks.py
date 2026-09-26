"""Generate 5 Jupyter notebooks for Approach 1 ML models."""
import json, os

models = {
    'LogisticRegression': 'LogisticRegression(max_iter=2000, class_weight="balanced")',
    'RandomForest': 'RandomForestClassifier(n_estimators=300, class_weight="balanced")',
    'XGBoost': 'XGBClassifier(scale_pos_weight=2961/189, eval_metric="logloss")',
    'SVM': 'SVC(probability=True, class_weight="balanced", kernel="rbf")',
    'GradientBoosting': 'HistGradientBoostingClassifier(class_weight="balanced")'
}

def make_nb(name, class_def, result_row):
    cells = [
        {"cell_type":"markdown","metadata":{},"source":[f"# {name}\n","Approach 1 — Classification Model\n","Real dataset only (4,500 claims). No synthetic data."]},
        {"cell_type":"markdown","metadata":{},"source":["## 1. Load Processed Data\n","Data lives in `project/data/processed/`: X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy. Preprocessor saved as `preprocessor.joblib`."]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["import numpy as np, joblib, pandas as pd\n","preprocessor = joblib.load('project/data/processed/preprocessor.joblib')\n","X_train = np.load('project/data/processed/X_train.npy')\n","y_train = np.load('project/data/processed/y_train.npy')\n","X_test = np.load('project/data/processed/X_test.npy')\n","y_test = np.load('project/data/processed/y_test.npy')"]},
        {"cell_type":"markdown","metadata":{},"source":[f"## 2. Model Definition\n","```python\n{class_def}\n```"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[f"from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay\n","model = {class_def}\n","model.fit(X_train, y_train)\n","y_pred = model.predict(X_test)\n","y_prob = model.predict_proba(X_test)[:,1] if hasattr(model,'predict_proba') else model.decision_function(X_test)"]},
        {"cell_type":"markdown","metadata":{},"source":["## 3. Real Results (from model_results.csv)\n",f"- **Val F1:** {result_row.get('Val_F1','N/A')}\n",f"- **Test F1:** {result_row.get('Test_F1','N/A')}\n",f"- **Test ROC-AUC:** {result_row.get('Test_ROC_AUC','N/A')}\n"]},
        {"cell_type":"markdown","metadata":{},"source":["## 4. Confusion Matrix Image\n","Saved at `project/evaluation/cm_{name}.png` (real test set)."]},
        {"cell_type":"markdown","metadata":{},"source":["## 5. Conclusion\n","This model was trained end-to-end on real data only. It participates in the full-stack Flask website at `/predict`. For production, validate on a future temporal cohort and aggregate high-cardinality codes into clinical groups (CCS/DRG)."]}
    ]
    nb = {"cells": cells, "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11.0"}},"nbformat":4,"nbformat_minor":4}
    with open(f'project/notebooks/{name}.ipynb','w') as f:
        json.dump(nb, f, indent=2)

# Load actual results to inject real numbers
import pandas as pd
res = pd.read_csv('project/evaluation/model_results.csv').set_index('Model').to_dict('index')
for name in models:
    make_nb(name, models[name], res[name])
print("5 notebooks created.")
