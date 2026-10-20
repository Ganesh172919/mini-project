"""Rebuild the two-approach benchmark, intake ablations and chronological stress test."""
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from common import ROOT, OUT, NUMERIC, CATEGORICAL, CLASSICAL, NEURAL, frame, scores, metrics, select_threshold
from submission_models import make_model

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def neural_fit(name, arrays, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    x, y = arrays['train']; vx, vy = arrays['val']
    model = make_model(name, x.shape[1])
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype('float32'))),
        batch_size=128, shuffle=True, generator=generator)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y)-y.sum())/y.sum(), dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best, stale, history, state, best_epoch = -1, 0, [], None, 0
    for epoch in range(1, 101):
        model.train(); loss_total = 0
        for bx, by in loader:
            opt.zero_grad(); loss = loss_fn(model(bx), by); loss.backward(); opt.step()
            loss_total += loss.item()*len(by)
        val = scores(model, vx, True)
        m = metrics(vy, val)
        history.append(dict(epoch=epoch, train_loss=loss_total/len(y), val_ap=m['ap'], val_f1=m['f1']))
        if m['ap'] > best + 1e-9:
            best, stale, state, best_epoch = m['ap'], 0, copy.deepcopy(model.state_dict()), epoch
        else:
            stale += 1
        if stale >= 15: break
    model.load_state_dict(state); model.eval()
    return model, dict(best_epoch=best_epoch, epochs=len(history), history=history, parameters=sum(p.numel() for p in model.parameters()))

def model_factory(y):
    return dict(LogisticRegression=LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        RandomForest=RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=2, random_state=42),
        XGBoost=XGBClassifier(n_estimators=100, max_depth=6, learning_rate=.3, scale_pos_weight=(len(y)-y.sum())/y.sum(), eval_metric='logloss', n_jobs=2, random_state=42),
        SVM=SVC(kernel='rbf', class_weight='balanced', probability=False, random_state=42),
        GradientBoosting=HistGradientBoostingClassifier(max_iter=100, class_weight='balanced', random_state=42))

def main():
    start = time.perf_counter()
    source = ROOT / 'Health Insurance Fraud Claims.xlsx'
    raw = pd.read_excel(source)
    y = raw.ClaimLegitimacy.map({'Fraud':1, 'Legitimate':0}).to_numpy()
    assert not pd.isna(y).any() and set(y)=={0,1}
    d = frame(raw); ids = np.arange(len(raw))
    train, temp = train_test_split(ids, test_size=.3, random_state=42, stratify=y)
    val, test = train_test_split(temp, test_size=.5, random_state=42, stratify=y[temp])
    random_split = dict(train=train, val=val, test=test)
    ordered = d.ClaimDate.sort_values().to_numpy()
    cut1, cut2 = ordered[int(.70*len(raw))-1], ordered[int(.85*len(raw))-1]
    temporal_split = dict(train=ids[d.ClaimDate<=cut1], val=ids[(d.ClaimDate>cut1)&(d.ClaimDate<=cut2)], test=ids[d.ClaimDate>cut2])
    audit = dict(rows=len(raw), columns=len(raw.columns), fraud=int(y.sum()), prevalence=float(y.mean()),
        missing=raw.isna().sum().astype(int).to_dict(), duplicate_rows=int(raw.duplicated().sum()), unique=raw.nunique().astype(int).to_dict(),
        cluster_crosstab=pd.crosstab(raw.Cluster, raw.ClaimLegitimacy).to_dict(),
        date_min=str(raw.ClaimDate.min().date()), date_max=str(raw.ClaimDate.max().date()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_copy_identical=source.read_bytes()==(ROOT/'Approch-1/data/raw'/source.name).read_bytes(),
        amount_by_class=raw.groupby('ClaimLegitimacy').ClaimAmount.agg(['min','max','mean']).to_dict(),
        split_sizes={}, split_fraud={}, temporal_cutoffs=[str(pd.Timestamp(cut1).date()),str(pd.Timestamp(cut2).date())])
    all_results = []; predictions = []; registry = []
    # Primary comparison and two pre-specified feature sensitivity tests. No test-based tuning.
    scenarios = [('baseline', [], random_split), ('intake', ['Cluster'], random_split),
                 ('no_amount', ['Cluster','ClaimAmount'], random_split), ('temporal', ['Cluster'], temporal_split)]
    for scenario, removed, splits in scenarios:
        print(f'\nSCENARIO {scenario}', flush=True)
        modeldir = OUT/'models'/scenario; modeldir.mkdir(parents=True, exist_ok=True)
        nums = [x for x in NUMERIC if x not in removed]
        prep = ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums),
            ('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),CATEGORICAL)])
        prep.fit(d.iloc[splits['train']])
        arrays = {s:(prep.transform(d.iloc[idx]).astype('float32'),y[idx]) for s,idx in splits.items()}
        joblib.dump(prep, modeldir/'preprocessor.joblib')
        feature_names = prep.get_feature_names_out().tolist()
        save_json(modeldir/'contract.json',dict(numeric=nums,categorical=CATEGORICAL,features=feature_names,
            feature_count=len(feature_names),removed=['ClaimStatus','ClaimID','PatientID','ProviderID','DiagnosisCode','ProcedureCode','ProviderLocation']+removed))
        audit['split_sizes'][scenario]={s:len(idx) for s,idx in splits.items()}
        audit['split_fraud'][scenario]={s:int(y[idx].sum()) for s,idx in splits.items()}
        for s,idx in splits.items():
            assert len(set(idx))==len(idx)
            np.save(OUT/'data'/f'{scenario}_X_{s}.npy',arrays[s][0]); np.save(OUT/'data'/f'{scenario}_y_{s}.npy',arrays[s][1])
        assert not (set(splits['train'])&set(splits['test'])) and not (set(splits['val'])&set(splits['test']))
        pd.DataFrame([dict(source_row=int(i)+2,claim_id=raw.ClaimID.iloc[i],split=s) for s,idx in splits.items() for i in idx]).to_csv(OUT/'data'/f'{scenario}_split_manifest.csv',index=False)
        if scenario=='baseline':
            audit['legacy_arrays_match']={s:bool(np.allclose(arrays[s][0],np.load(ROOT/'Approch-1/data/processed'/f'X_{s}.npy'),atol=1e-5)) for s in splits}
        models = model_factory(arrays['train'][1])
        for name in CLASSICAL+NEURAL:
            t = time.perf_counter(); neural=name in NEURAL
            if neural:
                model, training = neural_fit(name,arrays)
                torch.save(dict(state_dict=model.state_dict(),input_dim=len(feature_names),name=name),modeldir/f'{name}.pt')
                save_json(OUT/'results'/f'{scenario}_{name}_training.json',training)
            else:
                model=models[name]; model.fit(*arrays['train']); training={}
                joblib.dump(model,modeldir/f'{name}.joblib')
            fit_time = time.perf_counter()-t
            val_scores=scores(model,arrays['val'][0],neural)
            threshold=select_threshold(arrays['val'][1],val_scores)
            for s in ['val','test']:
                sc=scores(model,arrays[s][0],neural)
                for policy,th in [('default',.5),('validation_f1',threshold)]:
                    row=dict(scenario=scenario,model=name,approach=2 if neural else 1,split=s,policy=policy,**metrics(arrays[s][1],sc,th),train_seconds=fit_time,
                        feature_count=len(feature_names),best_epoch=training.get('best_epoch'))
                    all_results.append(row)
                predictions.extend([dict(scenario=scenario,model=name,split=s,source_row=int(idx)+2,y=int(label),score=float(v)) for idx,label,v in zip(splits[s],arrays[s][1],sc)])
            registry.append(dict(scenario=scenario,model=name,approach=2 if neural else 1,threshold=threshold,features=len(feature_names)))
            print(f'{name}: val AP={metrics(arrays["val"][1],val_scores)["ap"]:.5f}, threshold={threshold:.5f}, fit={fit_time:.2f}s',flush=True)
    results=pd.DataFrame(all_results); results.to_csv(OUT/'results'/'metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'results'/'predictions.csv',index=False)
    save_json(OUT/'results'/'metrics.json',all_results); save_json(OUT/'results'/'registry.json',registry)
    save_json(OUT/'results'/'dataset_audit.json',audit)
    # Select only within intake models using validation ranking. Test outcomes never enter selection.
    rank=results.query("scenario=='intake' and split=='val' and policy=='validation_f1'").copy()
    rank['ap_rank']=rank.ap.round(12)
    rank['priority']=rank.model.map({name:i for i,name in enumerate(CLASSICAL+NEURAL)})
    rank=rank.sort_values(['ap_rank','f1','priority'],ascending=[False,False,True])
    chosen=rank.iloc[0]
    save_json(OUT/'results'/'selection.json',dict(scenario='intake',model=chosen['model'],criterion='validation AP rounded to 12 decimals, then validation F1, then declared model order',threshold=chosen.threshold,
        validation_ap=chosen.ap,validation_f1=chosen.f1))
    # Conditional bootstrap uncertainty. Stratification fixes the observed test class counts.
    rng=np.random.default_rng(2026); p=pd.DataFrame(predictions).query("scenario=='intake' and split=='test' and model==@chosen.model")
    positives=np.flatnonzero(p.y.to_numpy()==1); negatives=np.flatnonzero(p.y.to_numpy()==0)
    boot=[]
    for _ in range(1000):
        ix=np.r_[rng.choice(positives,len(positives),True),rng.choice(negatives,len(negatives),True)]
        m=metrics(p.y.to_numpy()[ix],p.score.to_numpy()[ix],chosen.threshold); boot.append([m['ap'],m['f1'],m['recall']])
    save_json(OUT/'results'/'bootstrap.json',dict(model=chosen.model,replicates=1000,seed=2026,method='stratified test resampling at fixed trained model and fixed validation threshold',
        intervals={k:np.quantile(np.array(boot)[:,i],[.025,.975]).tolist() for i,k in enumerate(['ap','f1','recall'])}))
    save_json(OUT/'results'/'run_manifest.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),seed=42,python=platform.python_version(),
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost','torch','joblib']},seconds=time.perf_counter()-start,
        source_sha256=audit['source_sha256'],models=CLASSICAL+NEURAL,scenarios=[s[0] for s in scenarios],test_reuse='Historical random-split test has appeared in prior project results. All reported experiments are retrospective internal benchmarks.'))
    print('\nCOMPLETE',json.dumps(dict(chosen=chosen.model,seconds=time.perf_counter()-start)),flush=True)

if __name__=='__main__': main()
