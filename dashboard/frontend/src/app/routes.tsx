import { RouteObject, Navigate } from "react-router";
import AppShell from "../components/shell/AppShell";
import OverviewPage from "../pages/OverviewPage";
import ArenaPage from "../pages/ArenaPage";
import EnvironmentLabPage from "../pages/EnvironmentLabPage";
import ReplayLabPage from "../pages/ReplayLabPage";
import QualificationPage from "../pages/QualificationPage";
import TrainingPage from "../pages/TrainingPage";
import ExperimentsPage from "../pages/ExperimentsPage";
import ModelsPage from "../pages/ModelsPage";
import PopulationPage from "../pages/PopulationPage";
import ExplainabilityPage from "../pages/ExplainabilityPage";
import ChampionPage from "../pages/ChampionPage";
import SubmissionPage from "../pages/SubmissionPage";
import CompetitionPage from "../pages/CompetitionPage";
import RepositoryPage from "../pages/RepositoryPage";
import DocumentationPage from "../pages/DocumentationPage";

export const routes: RouteObject[] = [
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/overview" replace /> },
      { path: "overview", element: <OverviewPage /> },
      { path: "arena", element: <ArenaPage /> },
      { path: "environment-lab", element: <EnvironmentLabPage /> },
      { path: "replay/:replayId?", element: <ReplayLabPage /> },
      { path: "qualification/:candidateId?", element: <QualificationPage /> },
      { path: "training/:runId?", element: <TrainingPage /> },
      { path: "experiments/:experimentId?", element: <ExperimentsPage /> },
      { path: "models/:modelId?", element: <ModelsPage /> },
      { path: "population", element: <PopulationPage /> },
      { path: "explainability/:decisionId?", element: <ExplainabilityPage /> },
      { path: "champion", element: <ChampionPage /> },
      { path: "submission", element: <SubmissionPage /> },
      { path: "competition", element: <CompetitionPage /> },
      { path: "repository", element: <RepositoryPage /> },
      { path: "documentation/:sectionId?", element: <DocumentationPage /> },
    ],
  },
];
