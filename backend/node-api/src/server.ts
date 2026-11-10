import express from 'express'
import cors from 'cors'
import { healthRouter } from './routes/health.js'
import agenticRouter from './agentic/predict.js'
import mlRouter from './routes/ml.js'

const app = express()
const port = Number(process.env.PORT ?? 3000)

app.use(cors())
app.use(express.json())
app.use('/api/health', healthRouter)
app.use('/api', mlRouter)
app.use('/agentic', agenticRouter)

app.listen(port, () => {
  console.log(`Node API listening on http://localhost:${port}`)
  console.log('Routes: /api/health | /api/dl/predict | /api/results/dl | /api/results/ml | /agentic/predict')
})
