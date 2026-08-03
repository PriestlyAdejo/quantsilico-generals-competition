import { Link, Navigate, Route, Routes } from "react-router-dom";
import OverviewPage from "./pages/OverviewPage";
import ArenaPage from "./pages/ArenaPage";
import ReplayPage from "./pages/ReplayPage";
import GenericPage from "./pages/GenericPage";

const NAV = [
  ["/", "Overview"],
  ["/arena", "Arena"],
  ["/replay", "Replay Lab"],
  ["/experiments", "Experiments"],
  ["/training", "Training"],
  ["/models", "Models"],
  ["/population", "Population"],
  ["/explainability", "Explainability"],
  ["/champion", "Champion"],
  ["/submission", "Submission"],
  ["/competition", "Competition"],
  ["/repository", "Repository"],
] as const;

export default function App() {
  return (
    <div className="shell">
      <header className="top">
        <div>
          <div className="brand">QuantSilico</div>
          <div className="sub">Generals Research Console</div>
        </div>
      </header>
      <nav className="nav" aria-label="Primary">
        {NAV.map(([to, label]) => (
          <Link key={to} to={to}>
            {label}
          </Link>
        ))}
      </nav>
      <main className="main">
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/arena" element={<ArenaPage />} />
          <Route path="/replay" element={<ReplayPage />} />
          <Route path="/experiments" element={<GenericPage title="Experiments" endpoint="/api/experiments" />} />
          <Route path="/training" element={<GenericPage title="Training" endpoint="/api/training" />} />
          <Route path="/models" element={<GenericPage title="Models" endpoint="/api/models" />} />
          <Route path="/population" element={<GenericPage title="Population" endpoint="/api/population" />} />
          <Route path="/explainability" element={<GenericPage title="Explainability" endpoint="/api/explainability" />} />
          <Route path="/champion" element={<GenericPage title="Champion" endpoint="/api/models" />} />
          <Route path="/submission" element={<GenericPage title="Submission" endpoint="/api/submission" />} />
          <Route path="/competition" element={<GenericPage title="Competition" endpoint="/api/competition" />} />
          <Route path="/repository" element={<GenericPage title="Repository" endpoint="/api/repository" />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
