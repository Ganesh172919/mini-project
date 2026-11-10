import { useState } from "react"
import { HomePage } from "./pages/HomePage"
import { DLPage } from "./pages/DLPage"
import { ClassicalMLPage } from "./pages/ClassicalMLPage"
import { AgenticPage } from "./pages/AgenticPage"

type Tab = "home" | "classical" | "dl" | "agentic"

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: "home",      label: "Dashboard",    icon: "🏠" },
  { id: "classical", label: "Classical ML",  icon: "📊" },
  { id: "dl",        label: "Deep Learning", icon: "🧠" },
  { id: "agentic",   label: "Agentic AI",    icon: "🤖" },
]

export function App() {
  const [activeTab, setActiveTab] = useState<Tab>("home")

  return (
    <div className="app-shell">
      {/* Header */}
      <header className="app-header">
        <div className="header-inner">
          <a className="logo" href="#" onClick={() => setActiveTab("home")}>
            <div className="logo-icon">🩺</div>
            <div>
              <div className="logo-text">Fraud Intelligence</div>
              <div className="logo-sub">Mini-Project 2 · 2026</div>
            </div>
          </a>
          <nav className="nav-tabs" aria-label="Main navigation">
            {TABS.map(t => (
              <button
                key={t.id}
                id={`nav-${t.id}`}
                className={`nav-tab ${activeTab === t.id ? "active" : ""}`}
                onClick={() => setActiveTab(t.id)}
                aria-current={activeTab === t.id ? "page" : undefined}
              >
                {t.icon} {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Page content */}
      <main>
        <div className="page-content">
          {activeTab === "home"      && <HomePage onNavigate={tab => setActiveTab(tab as Tab)} />}
          {activeTab === "classical" && <ClassicalMLPage />}
          {activeTab === "dl"        && <DLPage />}
          {activeTab === "agentic"   && <AgenticPage />}
        </div>
      </main>
    </div>
  )
}
