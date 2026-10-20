"""Read-only compatibility audit of the pre-existing fitted artifacts."""
import json
import warnings
import joblib
import numpy as np
import torch
from common import ROOT, OUT, CLASSICAL, metrics
from dl_common import MLP

def main():
    x=np.load(ROOT/'Approch-1/data/processed/X_test.npy');y=np.load(ROOT/'Approch-1/data/processed/y_test.npy')
    results=[]
    for name in CLASSICAL:
        try:
            with warnings.catch_warnings(record=True) as caught:
                model=joblib.load(ROOT/'Approch-1/models'/f'{name}.joblib')
                score=model.predict_proba(x)[:,1]
            m=metrics(y,score)
            # Legacy SVC .predict and .predict_proba can disagree at 0.5.
            results.append(dict(model=name,status='evaluated',metrics=m,original_predict_fraud=int(model.predict(x).sum()),warnings=list(dict.fromkeys(str(w.message) for w in caught))))
        except Exception as e:results.append(dict(model=name,status='incompatible',error=str(e)))
    try:
        cp=torch.load(ROOT/'Approch-2-DL/models/mlp/mlp_best.pt',map_location='cpu',weights_only=False)
        model=MLP(cp['input_dim']);model.load_state_dict(cp['model_state_dict']);model.eval()
        with torch.inference_mode():score=torch.sigmoid(model(torch.tensor(x,dtype=torch.float32))).numpy()
        results.append(dict(model='MLP',status='evaluated',metrics=metrics(y,score)))
    except Exception as e:results.append(dict(model='MLP',status='incompatible',error=str(e)))
    (OUT/'results/legacy_artifact_audit.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
