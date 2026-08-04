import { Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/shell/AppShell";
import OverviewPage from "./pages/OverviewPage";
import ArenaPage from "./pages/ArenaPage";
import SubmissionPage from "./pages/SubmissionPage";
import ModelsPage from "./pages/ModelsPage";
import CompetitionPage from "./pages/CompetitionPage";
import TrainingPage from "./pages/TrainingPage";
import QualificationPage, {
  ChampionPage,
  DocumentationPage,
  EnvironmentLabPage,
  ExperimentsPage,
  ExplainabilityPage,
  NotFoundPage,
  PopulationPage,
  ReplayLabPage,
  RepositoryPage,
} from "./pages/MorePages";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/overview" replace />} />
        <Route path="overview" element={<OverviewPage />} />
        <Route path="arena" element={<ArenaPage />} />
        <Route path="environment-lab" element={<EnvironmentLabPage />} />
        <Route path="replay/:replayId?" element={<ReplayLabPage />} />
        <Route path="qualification/:candidateId?" element={<QualificationPage />} />
        <Route path="training/:runId?" element={<TrainingPage />} />
        <Route path="experiments/:experimentId?" element={<ExperimentsPage />} />
        <Route path="models/:modelId?" element={<ModelsPage />} />
        <Route path="population" element={<PopulationPage />} />
        <Route path="explainability/:decisionId?" element={<ExplainabilityPage />} />
        <Route path="champion" element={<ChampionPage />} />
        <Route path="submission" element={<SubmissionPage />} />
        <Route path="competition" element={<CompetitionPage />} />
        <Route path="repository" element={<RepositoryPage />} />
        <Route path="documentation" element={<DocumentationPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
