"""Publication figures from the saved benchmark predictions."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
from common import OUT, ROOT, CLASSICAL, NEURAL

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
    d=pd.read_csv(OUT/'results/metrics.csv'); p=pd.read_csv(OUT/'results/predictions.csv')
    rows=d[(d.split=='test')&(d.policy=='validation_f1')]; order=CLASSICAL+NEURAL
    fig,ax=plt.subplots(figsize=(9,4.6)); y=np.arange(8)
    for i,(scenario,color,label) in enumerate([('baseline','#7a9898','Original 29 features'),('intake','#d65b32','Intake without Cluster')]):
        a=rows[rows.scenario==scenario].set_index('model').loc[order]
        ax.barh(y+(i-.5)*.34,a.ap,height=.33,color=color,label=label)
    ax.set(yticks=y,yticklabels=order,xlim=(0,1.05),xlabel='Test average precision');ax.invert_yaxis();ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,frameon=False,fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'figures/ap_comparison.png',dpi=200);plt.close(fig)
    selected=json.loads((OUT/'results/selection.json').read_text());name=selected['model']
    a=rows[(rows.scenario=='intake')&(rows.model==name)].iloc[0]; chosen=p[(p.scenario=='intake')&(p.model==name)&(p.split=='test')]
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    pr,rc,_=precision_recall_curve(chosen.y,chosen.score);axes[0].step(rc,pr,where='post',color='#d65b32');axes[0].axhline(chosen.y.mean(),color='#798b8b',ls='--',label='Fraud prevalence')
    axes[0].set(xlabel='Recall',ylabel='Precision',xlim=(0,1.03),ylim=(0,1.03),title=f'Intake {name}  AP {a.ap:.4f}');axes[0].legend(frameon=False,loc='lower left')
    fpr,tpr,_=roc_curve(chosen.y,chosen.score);axes[1].plot(fpr,tpr,color='#d65b32');axes[1].plot([0,1],[0,1],ls='--',color='#798b8b')
    axes[1].set(xlabel='False positive rate',ylabel='True positive rate',xlim=(0,1),ylim=(0,1.03),title=f'ROC AUC {a.roc_auc:.4f}')
    fig.tight_layout();fig.savefig(OUT/'figures/selected_curves.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(4.5,3.6));cm=np.array([[a.tn,a.fp],[a.fn,a.tp]],int);ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=24,color='white' if cm[i,j]>300 else '#182b32')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Below threshold','Flagged'],yticklabels=['Legitimate','Fraud'],xlabel='Predicted',ylabel='Actual',title=f'Intake {name} at {a.threshold:.4f}')
    fig.tight_layout();fig.savefig(OUT/'figures/confusion_matrix.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,3.7));vals=rows[rows.model==name].set_index('scenario').loc[['baseline','intake','no_amount','temporal']]
    bars=ax.bar(['Original','No Cluster','No Cluster\nor amount','Chronological\nno Cluster'],vals.ap,color=['#93a9a1','#d65b32','#d65b32','#355d64'])
    ax.set(ylabel='Test average precision',ylim=(0,1.13))
    for bar,value in zip(bars,vals.ap):ax.text(bar.get_x()+bar.get_width()/2,value+.03,f'{value:.4f}',ha='center')
    fig.tight_layout();fig.savefig(OUT/'figures/sensitivity.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.6))
    for name in NEURAL:
        h=json.loads((OUT/'results'/f'intake_{name}_training.json').read_text())['history']
        ax.plot([r['epoch'] for r in h],[r['val_ap'] for r in h],label=name)
    ax.set(xlabel='Training epoch',ylabel='Validation average precision',ylim=(0,1.03));ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'figures/neural_training.png',dpi=200);plt.close(fig)
    rf=joblib.load(OUT/'models/intake/RandomForest.joblib');prep=joblib.load(OUT/'models/intake/preprocessor.joblib')
    importance=sorted(zip(prep.get_feature_names_out(),rf.feature_importances_),key=lambda v:v[1],reverse=True)
    error=chosen[(chosen.score>=selected['threshold'])!=chosen.y.astype(bool)]
    raw=pd.read_excel(ROOT/'Health Insurance Fraud Claims.xlsx')
    error_rows=[]
    for row in error.to_dict('records'):
        claim=raw.iloc[int(row['source_row'])-2]
        error_rows.append(dict(source_row=int(row['source_row']),target=int(row['y']),score=row['score'],ClaimAmount=claim.ClaimAmount,PatientAge=int(claim.PatientAge),ClaimType=claim.ClaimType))
    selected_test={k:(None if pd.isna(v) else v) for k,v in a.to_dict().items()}
    result=dict(selected_test=selected_test,feature_importance=[dict(feature=k,importance=float(v)) for k,v in importance],errors=error_rows)
    (OUT/'results/analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('Five evidence figures and selected-model error analysis written.')

if __name__=='__main__':main()
