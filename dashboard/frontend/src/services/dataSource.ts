import { MatchConfig, MatchFrame, MatchRecord } from "../types/match";
import { ReplayRecord } from "../types/replay";
import { QualCandidate, QualificationSummary } from "../types/qualification";
import { TrainingBlockedState, TrainingRun, TrainingMetric } from "../types/training";
import { ModelRecord } from "../types/model";
import { PopulationEntry, PayoffMatrix } from "../types/population";
import { ExplanationRecord, CounterfactualRecord } from "../types/explanation";
import { OverviewRecord, ApplicationStatusRecord } from "../types/overview";
import { ExperimentRecord, ExperimentFilter, ExperimentComparison } from "../types/experiment";
import { ChampionWorkspace, PromotionChecklist } from "../types/champion";
import { SubmissionPipeline, SubmissionPackage, PackageAction } from "../types/submission";
import { PortalObservation, ManualSubmissionRecord } from "../types/competition";
import { RepositoryStatus, CommitRecord, CiRun, EnvironmentLock } from "../types/repository";
import { DocIndex, DocSection } from "../types/documentation";

export interface DataSource {
  /* Pass 1 — Overview (legacy) */
  getOverviewStats(): Promise<{
    branch: string;
    engineVersion: string;
    gpuStatus: string;
    championModel: string;
    currentPhase: string;
  }>;

  /* Pass 1 — Matches / Arena */
  listMatches(): Promise<MatchRecord[]>;
  getMatchById(id: string): Promise<MatchRecord | null>;
  createDemoMatch(config: MatchConfig): Promise<string>;
  appendMatchFrame(matchId: string, frame: MatchFrame): Promise<void>;
  completeDemoMatch(matchId: string, outcome: MatchRecord["outcome"]): Promise<void>;

  /* Pass 1 — Replays */
  listReplays(): Promise<ReplayRecord[]>;
  getReplayById(id: string): Promise<ReplayRecord | null>;
  createReplayFromMatch(matchId: string): Promise<string>;

  /* Pass 1 — Qualification */
  listCandidates(): Promise<QualCandidate[]>;
  getCandidateById(id: string): Promise<QualCandidate | null>;
  getQualSummary(): Promise<QualificationSummary>;

  /* Pass 1 — Training */
  getTrainingBlockedState(): Promise<TrainingBlockedState | null>;
  listTrainingRuns(): Promise<TrainingRun[]>;
  getTrainingRunById(id: string): Promise<TrainingRun | null>;
  startDemoTrainingRun(preset: TrainingRun["preset"]): Promise<string>;
  appendTrainingMetric(runId: string, metric: TrainingMetric): Promise<void>;

  /* Pass 1 — Models (legacy) */
  listModels(): Promise<ModelRecord[]>;
  getModelById(id: string): Promise<ModelRecord | null>;

  /* Pass 2 — Overview */
  getOverview(): Promise<OverviewRecord>;
  getApplicationStatus(): Promise<ApplicationStatusRecord>;

  /* Pass 2 — Experiments */
  listExperiments(filters?: ExperimentFilter): Promise<ExperimentRecord[]>;
  getExperimentById(id: string): Promise<ExperimentRecord | null>;
  compareExperiments(ids: string[]): Promise<ExperimentComparison>;

  /* Pass 2 — Models (extended) */
  compareModels(ids: string[]): Promise<ModelRecord[]>;
  getModelLineage(id: string): Promise<{ nodes: ModelRecord[]; edges: { from: string; to: string }[] }>;

  /* Pass 2 — Population */
  getPopulationSummary(): Promise<{ entries: PopulationEntry[]; matrix: PayoffMatrix; updatedAt: string }>;
  getPayoffMatrix(): Promise<PayoffMatrix>;

  /* Pass 2 — Explainability */
  listExplanations(): Promise<ExplanationRecord[]>;
  getExplanationById(id: string): Promise<ExplanationRecord | null>;
  getCounterfactuals(explanationId: string): Promise<CounterfactualRecord[]>;

  /* Pass 2 — Champion */
  getChampionWorkspace(): Promise<ChampionWorkspace>;
  getPromotionChecklist(): Promise<PromotionChecklist>;

  /* Pass 2 — Submission */
  getSubmissionPipeline(): Promise<SubmissionPipeline>;
  listPackages(): Promise<SubmissionPackage[]>;
  runDemoPackageAction(action: PackageAction): Promise<SubmissionPipeline>;

  /* Pass 2 — Competition */
  listPortalObservations(): Promise<PortalObservation[]>;
  listManualSubmissionRecords(): Promise<ManualSubmissionRecord[]>;

  /* Pass 2 — Repository */
  getRepositoryStatus(): Promise<RepositoryStatus>;
  listRecentCommits(): Promise<CommitRecord[]>;
  listCiRuns(): Promise<CiRun[]>;
  getEnvironmentLocks(): Promise<EnvironmentLock[]>;

  /* Pass 2 — Documentation */
  getDocumentationIndex(): Promise<DocIndex>;
  getDocumentationSection(id: string): Promise<DocSection | null>;
}
