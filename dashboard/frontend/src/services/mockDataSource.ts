import { DataSource } from "./dataSource";
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
import { SubmissionPipeline, SubmissionPackage, PackageAction, PipelineStep } from "../types/submission";
import { PortalObservation, ManualSubmissionRecord } from "../types/competition";
import { RepositoryStatus, CommitRecord, CiRun, EnvironmentLock } from "../types/repository";
import { DocIndex, DocSection } from "../types/documentation";
import { applicationStatus, overviewRecord } from "../data/fixtures/overview";
import { allCandidates, qualSummary } from "../data/fixtures/qualification";
import { trainingBlockedState, demoTrainingRun, completedRuns } from "../data/fixtures/training";
import { demoReplay } from "../data/fixtures/replays";
import { allExperiments } from "../data/fixtures/experiments";
import { allModels } from "../data/fixtures/models";
import { populationEntries, payoffMatrix } from "../data/fixtures/population";
import { demoExplanation, demoCounterfactuals } from "../data/fixtures/explanations";
import { championWorkspace, promotionChecklist } from "../data/fixtures/champion";
import { submissionPipeline, currentPackage } from "../data/fixtures/submission";
import { portalObservations, manualRecords } from "../data/fixtures/competition";
import { repoStatus, recentCommits, ciRuns, environmentLocks } from "../data/fixtures/repository";
import { docSections, docIndex } from "../data/fixtures/documentation";
import { generateBoard } from "../utils/gameBoard";

const delay = (ms = 120) => new Promise<void>(r => setTimeout(r, ms));

export class MockDataSource implements DataSource {
  private matches: Map<string, MatchRecord> = new Map();
  private replays: Map<string, ReplayRecord> = new Map([["replay-demo-001", demoReplay]]);
  private trainingRuns: Map<string, TrainingRun> = new Map([
    ...completedRuns.map(r => [r.id, r] as [string, TrainingRun]),
    [demoTrainingRun.id, { ...demoTrainingRun }],
  ]);
  private _pipeline: SubmissionPipeline = { ...submissionPipeline, steps: submissionPipeline.steps.map(s => ({ ...s })) };

  private assertDemo(id: string, entityLabel: string) {
    const candidate = allCandidates.find(c => c.id === id);
    const replay = this.replays.get(id);
    const match = this.matches.get(id);
    const run = this.trainingRuns.get(id);
    const entity = match ?? replay ?? run ?? candidate;
    if (entity && "kind" in entity && (entity.kind === "IMPORTED_PROJECT_EVIDENCE" || entity.kind === "OFFICIAL_PORTAL_OBSERVATION")) {
      throw new Error(`Cannot mutate ${entity.kind} record: ${entityLabel} ${id}`);
    }
  }

  /* Pass 1 — Overview (legacy) */
  async getOverviewStats() {
    await delay();
    return {
      branch: applicationStatus.branch,
      engineVersion: applicationStatus.engineSha.slice(0, 8),
      gpuStatus: applicationStatus.hardware,
      championModel: applicationStatus.currentChampion ?? "NOT RECORDED",
      currentPhase: applicationStatus.currentPhase,
    };
  }

  /* Pass 1 — Matches */
  async listMatches() {
    await delay();
    return [...this.matches.values()];
  }

  async getMatchById(id: string) {
    await delay();
    return this.matches.get(id) ?? null;
  }

  async createDemoMatch(config: MatchConfig): Promise<string> {
    await delay();
    const id = `match-demo-${Date.now()}`;
    const board = generateBoard(config.mapSize, config.mapSize, 0);
    const record: MatchRecord = {
      id, kind: "DEMO", config,
      frames: [{ turn: 0, board, p1Armies: 10, p2Armies: 10, p1Land: 5, p2Land: 5, events: ["Match started"] }],
      outcome: "in_progress", totalTurns: 0, startedAt: new Date().toISOString(),
    };
    this.matches.set(id, record);
    return id;
  }

  async appendMatchFrame(matchId: string, frame: MatchFrame): Promise<void> {
    this.assertDemo(matchId, "match");
    const match = this.matches.get(matchId);
    if (!match) throw new Error(`Match not found: ${matchId}`);
    match.frames.push(frame);
    match.totalTurns = frame.turn + 1;
  }

  async completeDemoMatch(matchId: string, outcome: MatchRecord["outcome"]): Promise<void> {
    this.assertDemo(matchId, "match");
    const match = this.matches.get(matchId);
    if (!match) throw new Error(`Match not found: ${matchId}`);
    match.outcome = outcome;
    match.completedAt = new Date().toISOString();
  }

  /* Pass 1 — Replays */
  async listReplays() {
    await delay();
    return [...this.replays.values()];
  }

  async getReplayById(id: string) {
    await delay();
    return this.replays.get(id) ?? null;
  }

  async createReplayFromMatch(matchId: string): Promise<string> {
    this.assertDemo(matchId, "match");
    const match = this.matches.get(matchId);
    if (!match) throw new Error(`Match not found: ${matchId}`);
    const replayId = `replay-demo-${Date.now()}`;
    const replay: ReplayRecord = {
      id: replayId, kind: "DEMO", matchId, config: match.config, frames: match.frames,
      events: [{ turn: 0, type: "army_move", label: "Match started", player: "player1" }],
      decisions: match.frames.map(f => ({
        turn: f.turn, srcRow: 1, srcCol: 1, dstRow: 2, dstCol: 1,
        armiesMoved: 3, policyLogit: 1.8, valueEstimate: 0.6, topKActions: [],
      })),
      outcome: match.outcome === "player1_win" ? "player1_win" : match.outcome === "player2_win" ? "player2_win" : "draw",
      totalTurns: match.totalTurns, createdAt: new Date().toISOString(),
      label: `Demo — ${match.config.player1} vs ${match.config.player2}`,
    };
    this.replays.set(replayId, replay);
    return replayId;
  }

  /* Pass 1 — Qualification */
  async listCandidates() { await delay(); return allCandidates; }
  async getCandidateById(id: string) { await delay(); return allCandidates.find(c => c.id === id) ?? null; }
  async getQualSummary(): Promise<QualificationSummary> { await delay(); return qualSummary; }

  /* Pass 1 — Training */
  async getTrainingBlockedState(): Promise<TrainingBlockedState | null> { await delay(); return trainingBlockedState; }
  async listTrainingRuns() { await delay(); return [...this.trainingRuns.values()]; }
  async getTrainingRunById(id: string) { await delay(); return this.trainingRuns.get(id) ?? null; }

  async startDemoTrainingRun(preset: TrainingRun["preset"]): Promise<string> {
    await delay();
    const id = `run-demo-${Date.now()}`;
    const run: TrainingRun = {
      id, kind: "DEMO", label: `DEMO — ${preset.toUpperCase()}`, preset, status: "running",
      totalSteps: preset === "smoke" ? 10_000 : 100_000,
      currentStep: 0, metrics: [], startedAt: new Date().toISOString(),
    };
    this.trainingRuns.set(id, run);
    return id;
  }

  async appendTrainingMetric(runId: string, metric: TrainingMetric): Promise<void> {
    this.assertDemo(runId, "training run");
    const run = this.trainingRuns.get(runId);
    if (!run) throw new Error(`Run not found: ${runId}`);
    run.metrics.push(metric);
    run.currentStep = metric.step;
  }

  /* Pass 1 — Models (legacy, returns Pass 2 data) */
  async listModels(): Promise<ModelRecord[]> { await delay(); return allModels; }
  async getModelById(id: string): Promise<ModelRecord | null> { await delay(); return allModels.find(m => m.id === id) ?? null; }

  /* Pass 2 — Overview */
  async getOverview(): Promise<OverviewRecord> { await delay(); return overviewRecord; }
  async getApplicationStatus(): Promise<ApplicationStatusRecord> { await delay(); return applicationStatus; }

  /* Pass 2 — Experiments */
  async listExperiments(filters?: ExperimentFilter): Promise<ExperimentRecord[]> {
    await delay();
    let results = allExperiments;
    if (filters?.kind) results = results.filter(e => e.kind === filters.kind);
    if (filters?.candidate) results = results.filter(e => e.candidate.includes(filters.candidate!));
    if (filters?.lifecycle) results = results.filter(e => e.lifecycle === filters.lifecycle);
    if (filters?.suite) results = results.filter(e => e.suite === filters.suite);
    return results;
  }

  async getExperimentById(id: string): Promise<ExperimentRecord | null> {
    await delay();
    return allExperiments.find(e => e.id === id) ?? null;
  }

  async compareExperiments(ids: string[]): Promise<ExperimentComparison> {
    await delay();
    const records = ids.map(id => allExperiments.find(e => e.id === id)).filter((e): e is ExperimentRecord => e !== undefined);
    return { ids, records };
  }

  /* Pass 2 — Models (extended) */
  async compareModels(ids: string[]): Promise<ModelRecord[]> {
    await delay();
    return ids.map(id => allModels.find(m => m.id === id)).filter((m): m is ModelRecord => m !== undefined);
  }

  async getModelLineage(id: string): Promise<{ nodes: ModelRecord[]; edges: { from: string; to: string }[] }> {
    await delay();
    const model = allModels.find(m => m.id === id);
    if (!model) return { nodes: [], edges: [] };
    const nodes = [model];
    const edges: { from: string; to: string }[] = [];
    if (model.parentId) {
      const parent = allModels.find(m => m.id === model.parentId);
      if (parent) { nodes.unshift(parent); edges.push({ from: parent.id, to: model.id }); }
    }
    return { nodes, edges };
  }

  /* Pass 2 — Population */
  async getPopulationSummary(): Promise<{ entries: PopulationEntry[]; matrix: PayoffMatrix; updatedAt: string }> {
    await delay();
    return { entries: populationEntries, matrix: payoffMatrix, updatedAt: payoffMatrix.updatedAt };
  }

  async getPayoffMatrix(): Promise<PayoffMatrix> { await delay(); return payoffMatrix; }

  /* Pass 2 — Explainability */
  async listExplanations(): Promise<ExplanationRecord[]> { await delay(); return [demoExplanation]; }
  async getExplanationById(id: string): Promise<ExplanationRecord | null> {
    await delay();
    return id === demoExplanation.id ? demoExplanation : null;
  }
  async getCounterfactuals(explanationId: string): Promise<CounterfactualRecord[]> {
    await delay();
    return explanationId === demoExplanation.id ? demoCounterfactuals : [];
  }

  /* Pass 2 — Champion */
  async getChampionWorkspace(): Promise<ChampionWorkspace> { await delay(); return championWorkspace; }
  async getPromotionChecklist(): Promise<PromotionChecklist> { await delay(); return promotionChecklist; }

  /* Pass 2 — Submission */
  async getSubmissionPipeline(): Promise<SubmissionPipeline> { await delay(); return { ...this._pipeline }; }
  async listPackages(): Promise<SubmissionPackage[]> { await delay(); return [currentPackage]; }

  async runDemoPackageAction(action: PackageAction): Promise<SubmissionPipeline> {
    await delay(600);
    const pipeline = this._pipeline;

    const stageOrder: SubmissionPipeline["currentStage"][] = [
      "CANDIDATE_SELECTED", "PACKAGE_BUILT", "WINDOWS_VALIDATED",
      "LINUX_VALIDATED", "UPLOAD_READY", "MANUALLY_SUBMITTED", "PORTAL_ACCEPTED", "QUALIFIED",
    ];

    const actionToStage: Record<PackageAction, SubmissionPipeline["currentStage"]> = {
      simulate_build: "PACKAGE_BUILT",
      simulate_validate_windows: "WINDOWS_VALIDATED",
      simulate_validate_linux: "LINUX_VALIDATED",
      mark_upload_ready: "UPLOAD_READY",
    };

    const targetStage = actionToStage[action];
    const targetIdx = stageOrder.indexOf(targetStage);

    pipeline.steps = pipeline.steps.map((step, i): PipelineStep => {
      if (i < targetIdx) return { ...step, status: "complete", completedAt: new Date().toISOString() };
      if (i === targetIdx) return { ...step, status: "active", completedAt: new Date().toISOString() };
      return { ...step, status: "pending" };
    });
    pipeline.currentStage = targetStage;
    pipeline.updatedAt = new Date().toISOString();

    return { ...pipeline };
  }

  /* Pass 2 — Competition */
  async listPortalObservations(): Promise<PortalObservation[]> { await delay(); return portalObservations; }
  async listManualSubmissionRecords(): Promise<ManualSubmissionRecord[]> { await delay(); return manualRecords; }

  /* Pass 2 — Repository */
  async getRepositoryStatus(): Promise<RepositoryStatus> { await delay(); return repoStatus; }
  async listRecentCommits(): Promise<CommitRecord[]> { await delay(); return recentCommits; }
  async listCiRuns(): Promise<CiRun[]> { await delay(); return ciRuns; }
  async getEnvironmentLocks(): Promise<EnvironmentLock[]> { await delay(); return environmentLocks; }

  /* Pass 2 — Documentation */
  async getDocumentationIndex(): Promise<DocIndex> { await delay(); return docIndex; }
  async getDocumentationSection(id: string): Promise<DocSection | null> {
    await delay();
    return docSections.find(s => s.id === id) ?? null;
  }
}
