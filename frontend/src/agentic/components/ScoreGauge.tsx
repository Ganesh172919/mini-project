import { useEffect, useId, useState } from "react"

interface ScoreGaugeProps {
  value: number
  isFraud: boolean
}

const RADIUS = 62
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

function clamp(value: number) {
  return Math.min(100, Math.max(0, value))
}

export function ScoreGauge({ value, isFraud }: ScoreGaugeProps) {
  const [displayValue, setDisplayValue] = useState(0)
  const gradientId = useId().replace(/:/g, "")
  const target = clamp(value)

  useEffect(() => {
    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches

    if (reduceMotion) {
      setDisplayValue(target)
      return
    }

    let frame = 0
    const start = performance.now()
    const duration = 1_100

    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / duration)
      const eased = 1 - Math.pow(1 - progress, 3)
      setDisplayValue(target * eased)
      if (progress < 1) frame = window.requestAnimationFrame(tick)
    }

    frame = window.requestAnimationFrame(tick)
    return () => window.cancelAnimationFrame(frame)
  }, [target])

  const dashOffset = CIRCUMFERENCE * (1 - displayValue / 100)

  return (
    <div
      className={`score-gauge ${isFraud ? "score-gauge--fraud" : "score-gauge--safe"}`}
      role="img"
      aria-label={`Fraud risk score: ${Math.round(target)} percent`}
    >
      <svg viewBox="0 0 160 160" className="score-gauge__svg">
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop
              offset="0%"
              stopColor={isFraud ? "#f4b26b" : "#7ad9a8"}
            />
            <stop
              offset="100%"
              stopColor={isFraud ? "#d64b3b" : "#1d8a60"}
            />
          </linearGradient>
        </defs>
        <circle
          className="score-gauge__track"
          cx="80"
          cy="80"
          r={RADIUS}
          fill="none"
          strokeWidth="11"
        />
        <circle
          className="score-gauge__progress"
          cx="80"
          cy="80"
          r={RADIUS}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth="11"
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={dashOffset}
          transform="rotate(-90 80 80)"
        />
      </svg>
      <div className="score-gauge__center">
        <span className="score-gauge__value">
          {Math.round(displayValue)}
          <small>%</small>
        </span>
        <span className="score-gauge__label">Fraud risk</span>
      </div>
    </div>
  )
}
