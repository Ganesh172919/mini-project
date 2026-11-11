import { Router } from "express"
import { GoogleGenerativeAI } from "@google/generative-ai"
import dotenv from "dotenv"
import path from "node:path"

const router = Router()

dotenv.config({
  path: path.resolve(process.cwd(), "../../Approach-3-Agentic-AI/.env"),
})

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY ?? "")
const model = genAI.getGenerativeModel({ model: "gemini-3.8-flash" })

router.post("/", async (req, res) => {
  try {
    const { message, history = [] } = req.body

    if (!message) {
      res.status(400).json({ error: "Message is required" })
      return
    }

    const chat = model.startChat({
      history: history.map((msg: { role: string; content: string }) => ({
        role: msg.role === "assistant" ? "model" : "user",
        parts: [{ text: msg.content }],
      })),
      systemInstruction: {
        role: "system",
        parts: [{
          text: "You are a helpful health insurance fraud detection assistant. Help users understand fraud patterns, explain predictions, and answer questions about health insurance claims. Keep responses concise.",
        }],
      },
    })

    const result = await chat.sendMessage(message)
    const reply = result.response.text()

    res.json({ reply })
  } catch (error: any) {
    console.error("Chat error:", error.message)
    res.status(500).json({ error: "Failed to generate response" })
  }
})

export default router
