import { useCallback, useEffect, useState } from "react"

export type Theme = "light" | "dark"

const STORAGE_KEY = "hfi-theme"

function getInitialTheme(): Theme {
  if (typeof window === "undefined") return "light"

  try {
    const stored = window.localStorage.getItem(STORAGE_KEY)
    if (stored === "light" || stored === "dark") return stored
  } catch {
    // Storage can be unavailable in private browsing contexts.
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light"
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme)

  useEffect(() => {
    const root = document.documentElement
    root.dataset.theme = theme
    root.style.colorScheme = theme

    try {
      window.localStorage.setItem(STORAGE_KEY, theme)
    } catch {
      // Persistence is a nice-to-have; never break the UI over it.
    }

    const themeColor = document.querySelector('meta[name="theme-color"]')
    themeColor?.setAttribute("content", theme === "dark" ? "#0b1411" : "#f3f1e9")
  }, [theme])

  const toggleTheme = useCallback(() => {
    setTheme((current) => (current === "dark" ? "light" : "dark"))
  }, [])

  return { theme, toggleTheme }
}
