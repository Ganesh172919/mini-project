"""Generate EDA plots using real dataset only."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

OUT = 'project/evaluation/eda_plots'
os.makedirs(OUT, exist_ok=True)

df = pd.read_excel('project/data/raw/Health Insurance Fraud Claims.xlsx')
df['ClaimDate'] = pd.to_datetime(df['ClaimDate'])
df['ClaimMonth'] = df['ClaimDate'].dt.month

# 1. Target distribution
plt.figure()
sns.countplot(x='ClaimLegitimacy', data=df, palette='Set2')
plt.title('Claim Legitimacy Distribution (Real Data Only)')
plt.xlabel('Legitimacy')
plt.ylabel('Count')
for p in plt.gca().patches:
    plt.gca().annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'target_distribution.png'), dpi=150)
plt.close()

# 2. ClaimAmount by target
plt.figure()
sns.boxplot(x='ClaimLegitimacy', y='ClaimAmount', data=df, palette='Set2')
plt.title('Claim Amount by Legitimacy')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'claim_amount_by_target.png'), dpi=150)
plt.close()

# 3. Age distribution
plt.figure()
sns.histplot(df['PatientAge'], kde=True, bins=30, color='steelblue')
plt.title('Patient Age Distribution')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'age_distribution.png'), dpi=150)
plt.close()

# 4. Income vs ClaimAmount scatter (sample 800 points for visibility)
plt.figure()
sample = df.sample(min(800, len(df)), random_state=42)
sns.scatterplot(x='PatientIncome', y='ClaimAmount', hue='ClaimLegitimacy', data=sample, palette='deep', alpha=0.7)
plt.title('Income vs Claim Amount (Sample)')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'income_vs_amount.png'), dpi=150)
plt.close()

# 5. Correlation heatmap of numeric features
num_df = df[['ClaimAmount','PatientAge','PatientIncome','Cluster']].copy()
plt.figure()
sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Numeric Feature Correlation')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'correlation_heatmap.png'), dpi=150)
plt.close()

# 6. ClaimType counts
plt.figure()
sns.countplot(x='ClaimType', data=df, palette='pastel')
plt.title('Claim Type Counts')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'claim_type_counts.png'), dpi=150)
plt.close()

# 7. Cluster vs target
plt.figure()
sns.countplot(x='Cluster', hue='ClaimLegitimacy', data=df, palette='Set2')
plt.title('Cluster Distribution by Legitimacy')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'cluster_target.png'), dpi=150)
plt.close()

# 8. Monthly trend of fraud ratio
monthly = df.groupby('ClaimMonth')['ClaimLegitimacy'].apply(lambda x: (x=='Fraud').mean()).reset_index()
monthly.columns = ['Month', 'FraudRatio']
plt.figure()
sns.lineplot(x='Month', y='FraudRatio', data=monthly, marker='o', color='crimson')
plt.title('Monthly Fraud Ratio (Real Data)')
plt.ylabel('Fraud Ratio')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'monthly_fraud_trend.png'), dpi=150)
plt.close()

# 9. ProviderSpecialty by target
plt.figure()
sns.countplot(y='ProviderSpecialty', hue='ClaimLegitimacy', data=df, palette='Set2', order=df['ProviderSpecialty'].value_counts().index)
plt.title('Provider Specialty by Legitimacy')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'specialty_target.png'), dpi=150)
plt.close()

# 10. ClaimStatus by target
plt.figure()
sns.countplot(x='ClaimStatus', hue='ClaimLegitimacy', data=df, palette='Set2')
plt.title('Claim Status by Legitimacy')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'claimstatus_target.png'), dpi=150)
plt.close()

print("EDA plots saved to", OUT)
