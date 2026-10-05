import express from 'express'
import { healthRouter } from './routes/health.js'
import agenticRouter from './agentic/predict.js'

const app = express()
const port = Number(process.env.PORT ?? 3000)

app.use(express.json())
app.use('/api/health', healthRouter)
app.use('/agentic', agenticRouter)

app.listen(port, () => {
  console.log(`Node API listening on http://localhost:${port}`)
  console.log('Agentic AI routes registered at /agentic/predict')
})
