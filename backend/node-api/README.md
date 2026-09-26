# Node.js API

Browser-facing API gateway skeleton for Mini-Project 2.

## Current status

Only the health endpoint is implemented:

```text
GET /api/health
```

Expected response:

```json
{"status":"ok"}
```

Prediction, model orchestration, authentication, persistence, and Python-service integration are intentionally not implemented yet.

## Run

From the repository root:

```powershell
Push-Location backend/node-api
npm install
npm run build
npm start
Pop-Location
```

The default port is `3000`; set `PORT` to override it.
