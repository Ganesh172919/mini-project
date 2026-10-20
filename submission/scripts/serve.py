"""Same-origin local website and Python inference API for Approach 1 and Approach 2."""
import json
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from common import OUT, CLASSICAL, NEURAL, frame, load_model, scores

class Claim(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    ClaimAmount: float=Field(ge=0)
    ClaimDate: date
    PatientAge: int=Field(ge=0,le=120)
    PatientIncome: float=Field(ge=0)
    PatientGender: Literal['F','M']
    ProviderSpecialty: Literal['Cardiology','General Practice','Neurology','Orthopedics','Pediatrics']
    PatientMaritalStatus: Literal['Divorced','Married','Single','Widowed']
    PatientEmploymentStatus: Literal['Employed','Retired','Student','Unemployed']
    ClaimType: Literal['Emergency','Inpatient','Outpatient','Routine']
    ClaimSubmissionMethod: Literal['Online','Paper','Phone']
    Cluster: int|None=Field(default=None,ge=0,le=3)

class Request(BaseModel):
    model_config=ConfigDict(extra='forbid')
    scenario: Literal['intake','baseline']='intake'
    model: Literal['LogisticRegression','RandomForest','XGBoost','SVM','GradientBoosting','MLP','ResidualMLP','CNN1D']='RandomForest'
    policy: Literal['validation_f1','default']='validation_f1'
    claim: Claim

app=FastAPI(title='Claim Fraud Lab',version='1.0.0')

@lru_cache(maxsize=16)
def model(scenario,name):return load_model(scenario,name)
@lru_cache(maxsize=2)
def preprocessor(scenario):return joblib.load(OUT/'models'/scenario/'preprocessor.joblib')

@app.get('/api/health')
def health():return dict(status='ok',approaches=[1,2],models=CLASSICAL+NEURAL)

@app.post('/api/predict')
def predict(request:Request):
    if request.scenario=='baseline' and request.claim.Cluster is None:raise HTTPException(422,'Baseline requires Cluster.')
    if request.scenario=='intake' and request.claim.Cluster is not None:raise HTTPException(422,'Intake excludes Cluster. Omit this field.')
    rows=frame(pd.DataFrame([request.claim.model_dump()]))
    x=preprocessor(request.scenario).transform(rows).astype('float32')
    sc=float(scores(model(request.scenario,request.model),x,request.model in NEURAL)[0])
    regs=json.loads((OUT/'results/registry.json').read_text())
    threshold=.5 if request.policy=='default' else next(r['threshold'] for r in regs if r['scenario']==request.scenario and r['model']==request.model)
    return dict(model=request.model,scenario=request.scenario,score=sc,threshold=threshold,flag_for_review=sc>=threshold,score_is_calibrated=False,feature_count=x.shape[1])

app.mount('/',StaticFiles(directory=OUT/'website'/'dist',html=True),name='website')

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8080)
