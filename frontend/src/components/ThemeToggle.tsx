import type { Theme } from "../lib/useTheme"
import { MoonIcon, SunIcon } from "./Icons"

interface ThemeToggleProps {
  theme: Theme
  onToggle: () => void
}

export function ThemeToggle({ theme, onToggle }: ThemeToggleProps) {
  const isDark = theme === "dark"

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={onToggle}
      aria-label={`Switch to ${isDark ? "light" : "dark"} mode`}
      aria-pressed={isDark}
      title={`Switch to ${isDark ? "light" : "dark"} mode`}
    >
      <SunIcon className="theme-toggle__icon theme-toggle__icon--sun" size={18} />
      <MoonIcon className="theme-toggle__icon theme-toggle__icon--moon" size={18} />
    </button>
  )
}
