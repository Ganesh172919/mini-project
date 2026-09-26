import express from 'express'
import { healthRouter } from './routes/health.js'

const app = express()
const port = Number(process.env.PORT ?? 3000)

app.use(express.json())
app.use('/api/health', healthRouter)

app.listen(port, () => {
  console.log(`Node API listening on http://localhost:${port}`)
})
