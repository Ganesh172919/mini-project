"""Export fitted parameters for browser inference. No surrogate or synthetic model."""
import json
import numpy as np
import pandas as pd
import joblib
from common import ROOT, OUT, CLASSICAL, NEURAL, CATEGORICAL, load_model, scores
from run_pipeline import save_json

def tree(t):
    values=t.value[:,0,:]
    p=values[:,1]/values.sum(axis=1)
    return dict(left=t.children_left.tolist(),right=t.children_right.tolist(),feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=p.tolist())

def export(name, model):
    if name=='LogisticRegression': return dict(type='linear',weights=model.coef_[0].tolist(),bias=float(model.intercept_[0]))
    if name=='RandomForest': return dict(type='forest',trees=[tree(t.tree_) for t in model.estimators_])
    if name=='SVM': return dict(type='svm',support=model.support_vectors_.tolist(),dual=model.dual_coef_[0].tolist(),bias=float(model.intercept_[0]),gamma=float(model._gamma))
    if name=='XGBoost':
        config=json.loads(model.get_booster().save_config())
        base=float(config['learner']['learner_model_param']['base_score'].strip('[]'))
        return dict(type='xgboost',base=float(np.log(base/(1-base))),trees=[json.loads(t) for t in model.get_booster().get_dump(dump_format='json')])
    if name=='GradientBoosting':
        result=[]
        for stage in model._predictors:
            nodes=stage[0].nodes
            result.append([{k:float(n[k]) if k in ['value','num_threshold'] else int(n[k]) for k in ['value','num_threshold','feature_idx','left','right','is_leaf']} for n in nodes])
        return dict(type='hist',base=float(model._baseline_prediction[0,0]),trees=result)
    return dict(type=name,state={k:v.detach().numpy().tolist() for k,v in model.state_dict().items()})

def main():
    dest=OUT/'website'/'dist'; (dest/'models').mkdir(parents=True,exist_ok=True)
    regs=json.loads((OUT/'results/registry.json').read_text())
    fixtures=[]; preprocessing=[]
    for scenario in ['baseline','intake']:
        prep=joblib.load(OUT/'models'/scenario/'preprocessor.joblib')
        nums=prep.transformers_[0][2]; cats=prep.transformers_[1][2]
        scale=prep.named_transformers_['num'].named_steps['standardscaler']
        enc=prep.named_transformers_['cat'].named_steps['onehotencoder']
        contract=dict(numeric=nums,categorical=cats,mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),categories=[v.tolist() for v in enc.categories_])
        save_json(dest/'models'/f'{scenario}_contract.json',contract)
        # Full held-out test parity fixtures, kept outside the published site.
        x=np.load(OUT/'data'/f'{scenario}_X_test.npy')
        raw=pd.read_excel(ROOT/'Health Insurance Fraud Claims.xlsx')
        manifest=pd.read_csv(OUT/'data'/f'{scenario}_split_manifest.csv')
        raw=raw.iloc[manifest[manifest.split=='test'].source_row.to_numpy()-2]
        cols=['ClaimAmount','ClaimDate','PatientAge','PatientIncome']+CATEGORICAL+(['Cluster'] if scenario=='baseline' else [])
        raw=raw[cols].copy();raw['ClaimDate']=raw.ClaimDate.dt.strftime('%Y-%m-%d')
        preprocessing.append(dict(scenario=scenario,raw=raw.to_dict('records'),expected=x.tolist()))
        for name in CLASSICAL+NEURAL:
            m=load_model(scenario,name)
            data=export(name,m)
            data.update(next(r for r in regs if r['model']==name and r['scenario']==scenario))
            save_json(dest/'models'/f'{scenario}_{name}.json',data)
            fixtures.append(dict(scenario=scenario,model=name,x=x.tolist(),expected=scores(m,x,name in NEURAL).tolist()))
    save_json(OUT/'.build'/'browser_fixtures.json',fixtures)
    save_json(OUT/'.build'/'preprocessing_fixtures.json',preprocessing)
    save_json(dest/'results.json',dict(audit=json.loads((OUT/'results/dataset_audit.json').read_text()),
        selection=json.loads((OUT/'results/selection.json').read_text()),registry=regs,
        metrics=json.loads((OUT/'results/metrics.json').read_text()),run=json.loads((OUT/'results/run_manifest.json').read_text())))
    print('Exported 16 fitted models, two preprocessing contracts and test parity fixtures.')

if __name__=='__main__':main()
