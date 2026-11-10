"""Integration and evaluation checks with saved prediction evidence."""
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from fastapi.testclient import TestClient
from common import OUT, ROOT, CLASSICAL, NEURAL, metrics
from serve import app

def main():
    p=pd.read_csv(OUT/'results/predictions.csv'); reported=pd.read_csv(OUT/'results/metrics.csv')
    count=0
    for row in reported.to_dict('records'):
        sub=p[(p.scenario==row['scenario'])&(p.model==row['model'])&(p.split==row['split'])]
        calc=metrics(sub.y.to_numpy(),sub.score.to_numpy(),row['threshold'])
        for k,v in calc.items():assert np.isclose(v,row[k],rtol=0,atol=1e-10),(k,v,row[k])
        yy=sub.y.to_numpy();ss=sub.score.to_numpy();ix=np.argsort(-ss,kind='stable');ys=yy[ix];ss=ss[ix]
        ends=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1];tp=np.cumsum(ys)[ends];rec=tp/ys.sum();prec=tp/(ends+1)
        manual_ap=float(np.sum(np.diff(np.r_[0,rec])*prec))
        ranks=rankdata(sub.score.to_numpy());n1=yy.sum();n0=len(yy)-n1
        manual_auc=float((ranks[yy==1].sum()-n1*(n1+1)/2)/(n1*n0))
        assert np.isclose(manual_ap,row['ap'],atol=1e-10) and np.isclose(manual_auc,row['roc_auc'],atol=1e-10)
        count+=1
    examples=json.loads((OUT/'website/dist/examples.json').read_text())
    client=TestClient(app);assert client.get('/api/health').status_code==200
    api=[]
    for scenario in ['intake','baseline']:
        for name in CLASSICAL+NEURAL:
            for example in examples.values():
                claim=example.copy()
                if scenario=='intake':del claim['Cluster']
                payload=dict(scenario=scenario,model=name,claim=claim)
                response=client.post('/api/predict',json=payload);assert response.status_code==200,response.text
                result=response.json();assert 0<=result['score']<=1 and result['feature_count']==(28 if scenario=='intake' else 29)
                api.append(dict(scenario=scenario,model=name,score=result['score']))
    bad=examples['fraud'].copy();bad.pop('Cluster');bad['PatientAge']=-1
    assert client.post('/api/predict',json={'claim':bad}).status_code==422
    bad['PatientAge']=35;bad['ClaimLegitimacy']='Fraud'
    assert client.post('/api/predict',json={'claim':bad}).status_code==422
    bad.pop('ClaimLegitimacy');bad['ClaimDate']='2026-02-30'
    assert client.post('/api/predict',json={'claim':bad}).status_code==422
    assert client.post('/api/predict',json={'scenario':'baseline','claim':{k:v for k,v in examples['fraud'].items() if k!='Cluster'}}).status_code==422
    for path in ['/','/app.mjs','/engine.mjs','/results.json','/models/intake_MLP.json']:assert client.get(path).status_code==200,path
    result=dict(metric_rows_recomputed=count,api_prediction_requests=len(api),negative_input_tests=4,static_routes_checked=5,api_results=api)
    (OUT/'results/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='api_results'},indent=2))

if __name__=='__main__':main()
