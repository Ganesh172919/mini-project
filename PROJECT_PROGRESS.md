# Mini-Project 2 — Project Progress

## Reference Materials

- `Health Insurance Fraud Claims.xlsx`
- `Medical_Insurance_Claim_Fraud_Detection_Project_Statement.pdf`
- `Mini_Project_2_Full_Low_Level_Technical_Specification.docx`

These are project reference/source documents. They are not to be modified as part of implementation.

## Team Components

### Approach-1 — Classical ML

Owner: Ganesh  
Status: Existing implementation

`Approch-1/` is the existing classical ML reference baseline. It is owned by Ganesh and is not modified by the Deep Learning work.

### Approach-2 — Deep Learning

Owner: Jagadeeswar

Completed:

- Dedicated Python virtual environment
- Frozen 29-feature baseline contract
- PyTorch MLP
- Training pipeline
- Validation
- Test evaluation
- Checkpoint
- Metrics
- Evaluation plots
- Inference smoke test

Pending:

- Residual MLP
- 1D CNN
- TabTransformer
- Autoencoder
- Node → FastAPI integration

### Website / Common Integration

Owner: Jagadeeswar  
Status: **APPLICATION SKELETON ONLY**

Planned:

- React + TypeScript frontend
- Node.js + Express backend
- Python + FastAPI ML/DL service
- Unified prediction/investigation interface

The frontend placeholder, Node health endpoint, and FastAPI health endpoint are verified. React → Node prediction integration, Node → FastAPI integration, and the full website are not implemented yet.

### Agentic AI

Owner: Varshith  
Status: **NOT IMPLEMENTED YET**

The agentic component is a separate team responsibility and is not implemented or modified by this work.

## Current Architecture Plan

This is the **PLANNED** architecture, not the implemented architecture:

```text
React + TypeScript
        ↓
Node.js + Express
        ↓
Python + FastAPI
        ↓
Classical ML / Deep Learning / Agentic AI
```

This remains the planned application architecture. The currently implemented application path is only:

```text
Raw claim
        ↓
FastAPI /predict
        ↓
Frozen preprocessing contract + existing MLP
        ↓
PredictionResponse
```

## Current Overall Status

- Dataset/reference material: ✅
- Approach-1 existing implementation: ✅
- DL environment: ✅
- MLP: ✅
- MLP training/evaluation/checkpoint: ✅
- Application skeleton: ✅
- Frontend health/startup verification: ✅
- Node API health endpoint: ✅
- FastAPI health endpoint: ✅
- FastAPI MLP prediction endpoint: ✅
- Remaining DL models: ⏳
- Node → FastAPI prediction integration: ⏳
- React → Node prediction integration: ⏳
- Full website/dashboard: ⏳
- Model comparison UI: ⏳
- Explainability integration: ⏳
- Agentic AI: ⏳
- Final unified application: ⏳
