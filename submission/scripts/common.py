"""Shared submission contract. Raw workbook and historical artifacts stay intact."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'submission'
sys.path.insert(0, str(ROOT / 'Approch-2-DL' / 'scripts'))
NUMERIC = ['ClaimAmount', 'PatientAge', 'PatientIncome', 'ClaimYear', 'ClaimMonth', 'ClaimDayOfWeek', 'Cluster']
CATEGORICAL = ['PatientGender', 'ProviderSpecialty', 'PatientMaritalStatus', 'PatientEmploymentStatus', 'ClaimType', 'ClaimSubmissionMethod']
CLASSICAL = ['LogisticRegression', 'RandomForest', 'XGBoost', 'SVM', 'GradientBoosting']
NEURAL = ['MLP', 'ResidualMLP', 'CNN1D']

def frame(raw):
    d = raw.copy()
    dates = pd.to_datetime(d['ClaimDate'], errors='raise')
    d['ClaimYear'], d['ClaimMonth'], d['ClaimDayOfWeek'] = dates.dt.year, dates.dt.month, dates.dt.dayofweek
    return d

def metrics(y, score, threshold=.5):
    pred = np.asarray(score) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = dict(accuracy=float(accuracy_score(y, pred)), precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=float(np.clip(roc_auc_score(y, score),0,1)), ap=float(np.clip(average_precision_score(y, score),0,1)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=float(threshold))
    # Independently cross-check confusion-count formulae, including the zero-positive boundary.
    exact = [((tp+tn)/len(y)), tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0,
             2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0]
    assert np.allclose(exact, [result[k] for k in ['accuracy','precision','recall','f1']], atol=1e-12)
    return result

def select_threshold(y, score):
    p, r, thresholds = precision_recall_curve(y, score)
    f = np.divide(2*p[:-1]*r[:-1], p[:-1]+r[:-1], out=np.zeros_like(p[:-1]), where=(p[:-1]+r[:-1])>0)
    # Ties choose the larger threshold, reducing alerts at equal validation F1.
    return float(thresholds[np.flatnonzero(np.isclose(f, f.max(), rtol=0, atol=1e-12))[-1]])

def scores(model, x, neural=False):
    if neural:
        import torch
        model.eval()
        with torch.inference_mode():
            return torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).numpy()
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(x)[:, 1]
    return expit(model.decision_function(x))

def load_model(scenario, name):
    import joblib
    if name in CLASSICAL:
        return joblib.load(OUT / 'models' / scenario / f'{name}.joblib')
    import torch
    from submission_models import make_model
    checkpoint = torch.load(OUT / 'models' / scenario / f'{name}.pt', map_location='cpu', weights_only=True)
    model = make_model(name, checkpoint['input_dim'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model
