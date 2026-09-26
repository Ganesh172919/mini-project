"""
Preprocessing script for Fraud Detection - Approach 1 (ML)
Uses only existing dataset (Health Insurance Fraud Claims.xlsx) — no synthetic data.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Paths
RAW_PATH = 'project/data/raw/Health Insurance Fraud Claims.xlsx'
OUT_DIR = 'project/data/processed'
os.makedirs(OUT_DIR, exist_ok=True)

# Load
df = pd.read_excel(RAW_PATH)

# Derive date features
df['ClaimDate'] = pd.to_datetime(df['ClaimDate'], errors='coerce')
df['ClaimYear'] = df['ClaimDate'].dt.year
df['ClaimMonth'] = df['ClaimDate'].dt.month
df['ClaimDayOfWeek'] = df['ClaimDate'].dt.dayofweek

# Drop high-cardinality ID/code columns (near-unique per claim — cause overfit)
drop_cols = ['ClaimID', 'PatientID', 'ProviderID', 'DiagnosisCode', 'ProcedureCode', 'ProviderLocation', 'ClaimDate']
df = df.drop(columns=drop_cols)

# Target encoding: Fraud = 1, Legitimate = 0
df['Target'] = df['ClaimLegitimacy'].map({'Fraud': 1, 'Legitimate': 0})
df = df.drop(columns=['ClaimLegitimacy'])

# Identify feature types
numeric_features = ['ClaimAmount', 'PatientAge', 'PatientIncome', 'ClaimYear', 'ClaimMonth', 'ClaimDayOfWeek', 'Cluster']
# LEAKAGE CONTROL (added during the final verification pass):
# `ClaimStatus` (Approved / Denied / Pending) is the insurer's post-adjudication
# outcome.  It is recorded *after* the fraud investigation the model is asked to
# perform, so using it as a predictor leaks the answer and inflates every metric
# (it produced near-perfect ROC AUC in the first run).  The column is kept in
# the exploratory analysis but removed from the model registry here, which
# brings Approach 1 onto the same 29-feature contract Approach 2 is scored on.
LEAKY_COLUMNS = ['ClaimStatus']
categorical_features = ['PatientGender', 'ProviderSpecialty', 'PatientMaritalStatus',
                        'PatientEmploymentStatus', 'ClaimType', 'ClaimSubmissionMethod']

# Remove the leaky post-adjudication column(s) from the model matrix itself so
# no downstream script can accidentally pick them up again.
df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])

X = df.drop(columns=['Target'])
y = df['Target']

# Stratified split: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# Save splits as CSV for transparency
X_train.to_csv(os.path.join(OUT_DIR, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(OUT_DIR, 'y_train.csv'), index=False)
X_val.to_csv(os.path.join(OUT_DIR, 'X_val.csv'), index=False)
y_val.to_csv(os.path.join(OUT_DIR, 'y_val.csv'), index=False)
X_test.to_csv(os.path.join(OUT_DIR, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(OUT_DIR, 'y_test.csv'), index=False)

# Preprocessing pipeline: one-hot for categoricals, scale numeric
preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

X_train_processed = preprocess.fit_transform(X_train)
X_val_processed = preprocess.transform(X_val)
X_test_processed = preprocess.transform(X_test)

# Save processed arrays
np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train_processed)
np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train.values)
np.save(os.path.join(OUT_DIR, 'X_val.npy'), X_val_processed)
np.save(os.path.join(OUT_DIR, 'y_val.npy'), y_val.values)
np.save(os.path.join(OUT_DIR, 'X_test.npy'), X_test_processed)
np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test.values)

# Save preprocessor
joblib.dump(preprocess, os.path.join(OUT_DIR, 'preprocessor.joblib'))

# Feature names for interpretability
feature_names = (numeric_features +
                 list(preprocess.named_transformers_['cat'].get_feature_names_out(categorical_features)))
with open(os.path.join(OUT_DIR, 'feature_names.txt'), 'w') as f:
    f.write('\n'.join(feature_names))

print("Preprocessing complete.")
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
print("Target distribution (train):", y_train.value_counts().to_dict())
