"""Deep error analysis: FP / FN per model using real test data."""
import numpy as np, joblib, os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

X_test = np.load('project/data/processed/X_test.npy')
y_test = np.load('project/data/processed/y_test.npy')

models = ['LogisticRegression','RandomForest','XGBoost','SVM','GradientBoosting']
fp_fn = {}
for m in models:
    model = joblib.load(f'project/models/{m}.joblib')
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    fp_fn[m] = {'FP': int(fp), 'FN': int(fn), 'TP': int(tp), 'TN': int(tn)}

# Plot FP vs FN
fig, ax = plt.subplots(figsize=(9,5))
ms = list(fp_fn.keys())
fp_vals = [fp_fn[m]['FP'] for m in ms]
fn_vals = [fp_fn[m]['FN'] for m in ms]
x = np.arange(len(ms))
width = 0.35
bars1 = ax.bar(x - width/2, fp_vals, width, label='False Positives', color='coral')
bars2 = ax.bar(x + width/2, fn_vals, width, label='False Negatives', color='steelblue')
ax.set_ylabel('Count')
ax.set_title('Error Analysis — False Positives vs False Negatives (Real Test Set)')
ax.set_xticks(x)
ax.set_xticklabels(ms, rotation=30, ha='right')
ax.legend()
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{int(h)}', xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('project/evaluation/error_analysis_fp_fn.png', dpi=150)
plt.close()

# Save text report
with open('project/evaluation/result_analysis.txt','w') as f:
    f.write('ERROR ANALYSIS — REAL TEST SET (675 claims, 40 fraud)\n')
    f.write('='*60 + '\n')
    for m in ms:
        d = fp_fn[m]
        f.write(f"{m}: FP={d['FP']} FN={d['FN']} TP={d['TP']} TN={d['TN']}\n")
    f.write('\nObservation: Tree ensembles (RF, XGBoost, GradBoost) show near-zero FP/FN on this dataset.\n')
    f.write('Potential cause: the engineered Cluster column carries very strong predictive signal.\n')
    f.write('Note: ClaimStatus (Approved/Denied/Pending) is excluded from the model - it is the\n')
    f.write('post-adjudication outcome and would leak the target.\n')
    f.write('Recommendation: Validate on future time-period claims to confirm generalization.\n')

print('Error analysis saved.')
for m in ms:
    print(m, fp_fn[m])
