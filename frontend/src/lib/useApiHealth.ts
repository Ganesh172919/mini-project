import { useEffect, useState } from "react"
import { checkApiHealth } from "./api"

export type ApiHealth = "checking" | "online" | "offline"

export function useApiHealth(): ApiHealth {
  const [status, setStatus] = useState<ApiHealth>("checking")

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined

    const poll = async () => {
      const healthy = await checkApiHealth()
      if (cancelled) return
      setStatus(healthy ? "online" : "offline")
      timer = window.setTimeout(poll, 30_000)
    }

    void poll()

    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])

  return status
}
