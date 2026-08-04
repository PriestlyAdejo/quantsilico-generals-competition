import type { DataSource } from "./dataSource";
import { ApiError, CapabilityDisabledError } from "./apiErrors";
import { SCHEMA_VERSION, type WDL } from "../types/common";
import type { MatchConfig, MatchFrame, MatchRecord } from "../types/match";
import type { ReplayRecord } from "../types/replay";
import type { QualCandidate, QualificationSummary, Phase9QStep, StepStatus } from "../types/qualification";
import type { TrainingBlockedState, TrainingRun, TrainingMetric } from "../types/training";
import type { ModelArchitecture, ModelLifecycle, CompetitiveRole, DeliveryStatus, ModelRecord } from "../types/model";
import type { PopulationEntry, PayoffMatrix } from "../types/population";
import type { ExplanationRecord, CounterfactualRecord } from "../types/explanation";
import type { OverviewRecord, ApplicationStatusRecord } from "../types/overview";
import type { ExperimentRecord, ExperimentFilter, ExperimentComparison, ExperimentLifecycle, GateStatus } from "../types/experiment";
import type { ChampionWorkspace, PromotionChecklist, ChecklistStatus, ChecklistRow } from "../types/champion";
import type { SubmissionPipeline, SubmissionPackage, PackageAction, PackageStage, PipelineStep } from "../types/submission";
import type { PortalObservation, ManualSubmissionRecord } from "../types/competition";
import type { RepositoryStatus, CommitRecord, CiRun, EnvironmentLock } from "../types/repository";
import type { DocIndex, DocSection } from "../types/documentation";

type Json = Record<string, unknown>;

const EMPTY_WDL: WDL = { wins: 0, draws: 0, losses: 0 };
const KIND = "IMPORTED_PROJECT_EVIDENCE" as const;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.headers || {}),
      },
    });
  } catch {
    throw new ApiError("backend_unavailable", "BACKEND UNAVAILABLE");
  }

  if (res.status === 404) {
    throw new ApiError("not_found", "Record not found", 404);
  }
  if (!res.ok) {
    const detail = await res.text();
    throw new ApiError("http", detail || `HTTP ${res.status}`, res.status);
  }

  return (await res.json()) as T;
}

function str(v: unknown, fallback = ""): string {
  return typeof v === "string" ? v : fallback;
}

function num(v: unknown, fallback = 0): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

function asObj(v: unknown): Json {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Json) : {};
}

function disabled(method: string, reason: string): never {
  throw new CapabilityDisabledError(method, reason);
}

function mapArchitecture(raw: string): ModelArchitecture {
  if (raw === "heuristic") return "heuristic";
  if (raw.includes("mlp")) return "mlp_control";
  if (raw.includes("cnn")) return "recurrent_cnn";
  if (raw.includes("graph")) return "recurrent_graph_belief";
  return "mlp_control";
}

function mapLifecycle(raw: string): ModelLifecycle {
  const allowed: ModelLifecycle[] = [
    "SCAFFOLDED",
    "SMOKE_TESTED",
    "TRAINED",
    "EVALUATED",
    "REJECTED",
    "REJECTED_INCOMPATIBLE",
  ];
  return (allowed.includes(raw as ModelLifecycle) ? raw : "SCAFFOLDED") as ModelLifecycle;
}

function mapRole(raw: string): CompetitiveRole {
  const allowed: CompetitiveRole[] = ["BASELINE", "CHALLENGER", "CHAMPION", "NONE"];
  return (allowed.includes(raw as CompetitiveRole) ? raw : "NONE") as CompetitiveRole;
}

function mapDelivery(raw: string): DeliveryStatus {
  const allowed: DeliveryStatus[] = [
    "NOT_PACKAGED",
    "PACKAGED",
    "UPLOAD_READY",
    "SUBMITTED",
    "NOT_APPLICABLE",
    "NOT_RECORDED",
  ];
  return (allowed.includes(raw as DeliveryStatus) ? raw : "NOT_RECORDED") as DeliveryStatus;
}

function mapChecklistStatus(raw: unknown): ChecklistStatus {
  const s = String(raw ?? "NOT_EVALUATED").toUpperCase();
  if (s === "PASS" || s === "PASSED") return "PASS";
  if (s === "FAIL" || s === "FAILED") return "FAIL";
  if (s === "BLOCKED" || s === "NONE") return "BLOCKED";
  if (s === "PENDING") return "PENDING";
  return "NOT_EVALUATED";
}

function mapGateStatus(raw: unknown): GateStatus {
  const s = String(raw ?? "NOT_EVALUATED").toUpperCase();
  if (s === "PASS" || s === "PASSED") return "PASSED";
  if (s === "FAIL" || s === "FAILED") return "FAILED";
  if (s === "PENDING") return "PENDING";
  return "NOT_EVALUATED";
}

type Phase9QStateSteps = { step: Phase9QStep; status: StepStatus; completedAt?: string }[];

function phaseFromGates(board: Json): { current: Phase9QStep; steps: Phase9QStateSteps } {
  const order: Phase9QStep[] = [
    "screening",
    "development",
    "holdout",
    "package",
    "linux_parity",
    "upload_ready",
    "portal",
  ];
  const mapping: Record<Phase9QStep, string[]> = {
    screening: ["LEARNING_READINESS_GATE"],
    development: ["HEURISTIC_DEVELOPMENT_GATE"],
    holdout: ["PRE_PPO_SUBMISSION_GATE"],
    package: [],
    linux_parity: [],
    upload_ready: [],
    portal: ["PORTAL_SUBMISSION_GATE"],
  };
  const steps = order.map((step) => {
    const keys = mapping[step];
    let status: StepStatus = "pending";
    for (const key of keys) {
      const v = String(board[key] ?? "").toUpperCase();
      if (v === "PASS" || v === "PASSED" || v === "QUALIFIED") status = "complete";
      else if (v === "FAIL" || v === "FAILED") status = "failed";
      else if (v === "PENDING") status = "active";
    }
    if (keys.length === 0) status = "pending";
    return { step, status };
  });
  const active = steps.find((s) => s.status === "active" || s.status === "failed");
  const current = active?.step ?? (steps.every((s) => s.status === "complete") ? "portal" : "screening");
  return { current, steps };
}

function mapModel(row: Json): ModelRecord {
  const id = str(row.id, "unknown-model");
  const notes = Array.isArray(row.notes) ? row.notes.map(String).join("; ") : str(row.notes);
  return {
    id,
    kind: KIND,
    schemaVersion: SCHEMA_VERSION,
    name: id,
    architecture: mapArchitecture(str(row.architecture)),
    checkpoint: str(row.checkpoint, id),
    lifecycle: mapLifecycle(str(row.lifecycle)),
    role: mapRole(str(row.competitive_role ?? row.role, "NONE")),
    deliveryStatus: mapDelivery(str(row.delivery_status, "NOT_RECORDED")),
    parameters: num(row.parameters),
    trainingSteps: num(row.training_steps),
    wdl: EMPTY_WDL,
    promotionState: "NONE",
    blockerReason: str(row.blocker_reason) || undefined,
    createdAt: str(row.created_at, new Date().toISOString()),
    notes: notes || undefined,
  };
}

function mapExperiment(item: Json): ExperimentRecord {
  const data = asObj(item.data);
  const id = str(item.id, str(data.id, "experiment"));
  const kindRaw = str(data.kind ?? item.kind, "IMPORTED_PROJECT_EVIDENCE");
  const lifecycleRaw = str(data.lifecycle ?? data.status, "COMPLETE").toUpperCase();
  let lifecycle: ExperimentLifecycle = "COMPLETE";
  if (lifecycleRaw.includes("RUN")) lifecycle = "RUNNING";
  else if (lifecycleRaw.includes("FAIL")) lifecycle = "FAILED";
  else if (lifecycleRaw.includes("PLAN")) lifecycle = "PLANNED";
  else if (lifecycleRaw.includes("ARCH")) lifecycle = "ARCHIVED";

  return {
    id,
    kind: KIND,
    schemaVersion: SCHEMA_VERSION,
    label: str(data.label ?? data.title ?? id),
    candidate: str(data.candidate ?? data.agent ?? "NOT RECORDED"),
    opponent: str(data.opponent, "NOT RECORDED"),
    suite: str(data.suite ?? data.kind ?? kindRaw),
    lifecycle,
    wdl: EMPTY_WDL,
    discoveryGate: mapGateStatus(data.discovery_gate),
    developmentGate: mapGateStatus(data.development_gate),
    observedAt: str(data.observed_at ?? data.completed_at) || null,
    dateLabel: str(data.date_label ?? data.observed_at ?? "Evidence recorded"),
    notes: str(data.note ?? data.notes) || `Manifest kind: ${kindRaw}`,
  };
}

function emptyReplayStub(id: string, label: string): ReplayRecord {
  return {
    id,
    kind: KIND,
    matchId: id,
    config: {
      player1: "heuristic",
      player2: "heuristic",
      mapPreset: "standard",
      mapSize: 18,
      speedMultiplier: 1,
      label,
    },
    frames: [],
    events: [],
    decisions: [],
    outcome: "draw",
    totalTurns: 0,
    createdAt: new Date().toISOString(),
    label: `${label} — board frames not present in recorded evidence`,
  };
}

export class ApiDataSource implements DataSource {
  /* Pass 1 — Overview (legacy) */
  async getOverviewStats() {
    const o = await request<Json>("/api/overview");
    return {
      branch: str(o.branch),
      engineVersion: str(o.engine_commit).slice(0, 8),
      gpuStatus: "NOT RECORDED",
      championModel: str(o.learned_champion) || str(o.heuristic_baseline) || "NO LEARNED CHAMPION",
      currentPhase: str(o.research_phase, "NOT RECORDED"),
    };
  }

  /* Pass 1 — Matches / Arena */
  async listMatches(): Promise<MatchRecord[]> {
    return [];
  }

  async getMatchById(_id: string): Promise<MatchRecord | null> {
    return null;
  }

  async createDemoMatch(_config: MatchConfig): Promise<string> {
    return disabled(
      "createDemoMatch",
      "Demo arena simulation is disabled in API mode. Use Replay Lab for recorded boards or allowlisted match jobs via the backend.",
    );
  }

  async appendMatchFrame(_matchId: string, _frame: MatchFrame): Promise<void> {
    return disabled("appendMatchFrame", "Demo match frame mutation is disabled in API mode.");
  }

  async completeDemoMatch(_matchId: string, _outcome: MatchRecord["outcome"]): Promise<void> {
    return disabled("completeDemoMatch", "Demo match completion is disabled in API mode.");
  }

  /* Pass 1 — Replays */
  async listReplays(): Promise<ReplayRecord[]> {
    const data = await request<{ replays?: Json[] }>("/api/replays");
    return (data.replays ?? []).map((r) =>
      emptyReplayStub(str(r.id), str(r.name ?? r.id)),
    );
  }

  async getReplayById(id: string): Promise<ReplayRecord | null> {
    try {
      const raw = await request<Json>(`/api/replays/${encodeURIComponent(id)}`);
      const candidate = str(raw.candidate, "heuristic");
      const opponent = str(raw.opponent, "opponent");
      const turns = num(raw.turns);
      const winner = raw.winner;
      let outcome: ReplayRecord["outcome"] = "draw";
      if (winner === 0) outcome = "player1_win";
      else if (winner === 1) outcome = "player2_win";
      return {
        ...emptyReplayStub(id, `${candidate} vs ${opponent}`),
        totalTurns: turns,
        outcome,
        label: str(raw.privileged_label) || `${candidate} vs ${opponent} (seed ${String(raw.seed ?? "?")})`,
        config: {
          player1: "heuristic",
          player2: "heuristic",
          mapPreset: "standard",
          mapSize: 18,
          speedMultiplier: 1,
          label: `${candidate} vs ${opponent}`,
        },
      };
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) return null;
      throw e;
    }
  }

  async createReplayFromMatch(_matchId: string): Promise<string> {
    return disabled("createReplayFromMatch", "Demo replay creation is disabled in API mode.");
  }

  /* Pass 1 — Qualification */
  async listCandidates(): Promise<QualCandidate[]> {
    const data = await request<Json>("/api/qualification");
    const board = asObj(data.gates);
    const { current, steps } = phaseFromGates(board);
    const name = str(data.champion_until_promoted, "NOT RECORDED");
    const candidate: QualCandidate = {
      id: name,
      kind: KIND,
      name,
      checkpoint: name,
      phase9q: { currentStep: current, steps },
      screeningWDL: EMPTY_WDL,
      developmentWDL: EMPTY_WDL,
      discoveryRate: 0,
      terminalTurnP50: 0,
      terminalTurnP95: 0,
      submittedAt: new Date().toISOString(),
      notes: str(data.note) || undefined,
    };
    return name ? [candidate] : [];
  }

  async getCandidateById(id: string): Promise<QualCandidate | null> {
    const all = await this.listCandidates();
    return all.find((c) => c.id === id) ?? null;
  }

  async getQualSummary(): Promise<QualificationSummary> {
    const data = await request<Json>("/api/qualification");
    const board = asObj(data.gates);
    const { current } = phaseFromGates(board);
    const candidates = await this.listCandidates();
    return {
      totalCandidates: candidates.length,
      passed: 0,
      failed: 0,
      inProgress: candidates.length,
      phase9qCurrent: current,
      expanderRecord: EMPTY_WDL,
      discoveryRate: 0,
      conversionRate: 0,
    };
  }

  /* Pass 1 — Training */
  async getTrainingBlockedState(): Promise<TrainingBlockedState | null> {
    const data = await request<Json>("/api/training");
    if (data.launch_enabled === true) return null;
    return {
      reason: "Training launch is disabled in this console. Only recorded smoke/campaign evidence is shown.",
      gateFailedAt: new Date().toISOString(),
      requiredAction: str(asObj(data.labels).charts) || "Use recorded charts; do not launch INITIAL/OVERNIGHT from the dashboard.",
    };
  }

  async listTrainingRuns(): Promise<TrainingRun[]> {
    const data = await request<Json>("/api/training");
    const campaigns = Array.isArray(data.campaigns) ? data.campaigns : [];
    const charts = Array.isArray(data.charts) ? data.charts : [];
    const fromCampaigns: TrainingRun[] = campaigns.map((c) => {
      const row = asObj(c);
      const id = str(row.id, "campaign");
      return {
        id,
        kind: KIND,
        label: str(row.kind ?? id),
        preset: "dev",
        status: "complete",
        totalSteps: 0,
        currentStep: 0,
        metrics: [],
        blockedReason: str(row.stopped_reason) || undefined,
      };
    });
    const fromCharts: TrainingRun[] = charts
      .filter((c) => !fromCampaigns.some((r) => r.id === str(asObj(c).id)))
      .map((c) => {
        const row = asObj(c);
        const points = Array.isArray(row.points) ? row.points : [];
        const metrics: TrainingMetric[] = points.slice(0, 64).map((p, i) => {
          const pt = asObj(p);
          return {
            step: num(pt.step ?? pt.t, i),
            policyLoss: num(pt.policy_loss ?? pt.loss),
            valueLoss: num(pt.value_loss),
            entropy: num(pt.entropy),
            klDiv: 0,
            gradNorm: 0,
            reward: 0,
            winRate: 0,
            drawRate: 0,
            lossRate: 0,
            stepsPerSec: 0,
            gpuUtil: 0,
          };
        });
        return {
          id: str(row.id, `chart-${metrics.length}`),
          kind: KIND,
          label: str(row.title, str(row.id)),
          preset: "smoke",
          status: "complete" as const,
          totalSteps: metrics.length ? metrics[metrics.length - 1].step : 0,
          currentStep: metrics.length ? metrics[metrics.length - 1].step : 0,
          metrics,
        };
      });
    return [...fromCampaigns, ...fromCharts];
  }

  async getTrainingRunById(id: string): Promise<TrainingRun | null> {
    const runs = await this.listTrainingRuns();
    return runs.find((r) => r.id === id) ?? null;
  }

  async startDemoTrainingRun(_preset: TrainingRun["preset"]): Promise<string> {
    return disabled("startDemoTrainingRun", "Demo training launch is disabled in API mode (launch_enabled=false).");
  }

  async appendTrainingMetric(_runId: string, _metric: TrainingMetric): Promise<void> {
    return disabled("appendTrainingMetric", "Demo training metric mutation is disabled in API mode.");
  }

  /* Pass 1 — Models (legacy) */
  async listModels(): Promise<ModelRecord[]> {
    const data = await request<{ models?: Json[] }>("/api/models");
    return (data.models ?? []).map(mapModel);
  }

  async getModelById(id: string): Promise<ModelRecord | null> {
    const models = await this.listModels();
    return models.find((m) => m.id === id) ?? null;
  }

  /* Pass 2 — Overview */
  async getOverview(): Promise<OverviewRecord> {
    const o = await request<Json>("/api/overview");
    const metrics = asObj(o.metrics);
    const candidate = str(
      metrics.submitted_candidate ?? o.heuristic_baseline ?? o.candidate_identity,
      "NOT RECORDED",
    );
    const jobs = Array.isArray(o.active_jobs) ? o.active_jobs : [];
    const readiness = str(asObj(o.learning_smoke).readiness ?? metrics.learning_readiness);
    const ppoRaw = str(metrics.development_campaign);
    let ppoStatus: OverviewRecord["ppoStatus"] = "NOT_STARTED";
    if (ppoRaw) ppoStatus = "COMPLETE";

    const promotion = str(metrics.learned_promotion, "NONE");
    const blocker =
      promotion === "NONE" || promotion === "BLOCKED"
        ? `LEARNED PROMOTION: ${promotion || "NONE"}`
        : null;

    return {
      schemaVersion: SCHEMA_VERSION,
      id: "overview-api",
      kind: KIND,
      currentCandidate: typeof candidate === "string" ? candidate : str(asObj(candidate).candidate, "NOT RECORDED"),
      currentResult: EMPTY_WDL,
      discoveryRate: 0,
      conversionRate: 0,
      ppoStatus,
      blocker,
      wdlHistory: [],
      qualificationFunnel: [
        { stage: "Learning readiness", count: readiness === "PASS" ? 1 : 0 },
        { stage: "Submitted package", count: o.active_submitted_package ? 1 : 0 },
        { stage: "Learned champion", count: o.learned_champion ? 1 : 0 },
      ],
      experimentTimeline: [
        {
          id: "repo-commit",
          label: `${str(o.branch)} @ ${str(o.commit).slice(0, 8)}`,
          startedAt: new Date().toISOString(),
          completedAt: null,
          status: "running",
          dateLabel: str(o.research_phase),
        },
      ],
      activeJobs: jobs.map((j) => {
        const job = asObj(j);
        const state = str(job.state, "RUNNING").toUpperCase();
        return {
          id: str(job.id, "job"),
          label: str(job.label ?? job.job_type ?? job.id, "job"),
          progress: state === "RUNNING" || state === "QUEUED" ? 0.5 : 1,
          status:
            state === "FAILED"
              ? ("failed" as const)
              : state === "SUCCEEDED" || state === "COMPLETE"
                ? ("complete" as const)
                : ("running" as const),
        };
      }),
    };
  }

  async getApplicationStatus(): Promise<ApplicationStatusRecord> {
    const o = await request<Json>("/api/overview");
    const pkg = asObj(o.active_submitted_package);
    return {
      schemaVersion: SCHEMA_VERSION,
      id: "app-status-api",
      kind: KIND,
      branch: str(o.branch),
      engineSha: str(o.engine_commit),
      hardware: "NOT RECORDED",
      currentCandidate: str(pkg.candidate ?? o.heuristic_baseline, "NOT RECORDED"),
      currentSubmittedBaseline: str(pkg.candidate) || null,
      currentChampion: str(o.learned_champion) || null,
      currentPhase: str(o.research_phase, "NOT RECORDED"),
      updatedAt: new Date().toISOString(),
    };
  }

  /* Pass 2 — Experiments */
  async listExperiments(filters?: ExperimentFilter): Promise<ExperimentRecord[]> {
    const data = await request<{ experiments?: Json[] }>("/api/experiments");
    let rows = (data.experiments ?? []).map(mapExperiment);
    if (filters?.candidate) rows = rows.filter((r) => r.candidate.includes(filters.candidate!));
    if (filters?.opponent) rows = rows.filter((r) => r.opponent.includes(filters.opponent!));
    if (filters?.suite) rows = rows.filter((r) => r.suite.includes(filters.suite!));
    if (filters?.lifecycle) rows = rows.filter((r) => r.lifecycle === filters.lifecycle);
    if (filters?.kind) rows = rows.filter((r) => r.kind === filters.kind);
    return rows;
  }

  async getExperimentById(id: string): Promise<ExperimentRecord | null> {
    try {
      const item = await request<Json>(`/api/experiments/${encodeURIComponent(id)}`);
      return mapExperiment({ id: item.id, data: item.data, kind: asObj(item.data).kind });
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) return null;
      throw e;
    }
  }

  async compareExperiments(ids: string[]): Promise<ExperimentComparison> {
    const records: ExperimentRecord[] = [];
    for (const id of ids) {
      const r = await this.getExperimentById(id);
      if (r) records.push(r);
    }
    return { ids, records };
  }

  /* Pass 2 — Models (extended) */
  async compareModels(ids: string[]): Promise<ModelRecord[]> {
    const all = await this.listModels();
    return all.filter((m) => ids.includes(m.id));
  }

  async getModelLineage(id: string): Promise<{ nodes: ModelRecord[]; edges: { from: string; to: string }[] }> {
    const model = await this.getModelById(id);
    return { nodes: model ? [model] : [], edges: [] };
  }

  /* Pass 2 — Population */
  async getPopulationSummary(): Promise<{ entries: PopulationEntry[]; matrix: PayoffMatrix; updatedAt: string }> {
    const data = await request<Json>("/api/population");
    const matrix = await this.getPayoffMatrix();
    const labels = Array.isArray(data.population) ? data.population.map(String) : matrix.agents;
    const entries: PopulationEntry[] = labels.map((name, i) => ({
      id: name,
      kind: KIND,
      name,
      checkpoint: name,
      payoffs: matrix.matrix[i] ?? [],
      pfspWeight: 0,
      gamesPlayed: num(data.games_total),
      winRate: 0,
      isMainAgent: i === 0,
    }));
    return {
      entries,
      matrix,
      updatedAt: matrix.updatedAt,
    };
  }

  async getPayoffMatrix(): Promise<PayoffMatrix> {
    const data = await request<Json>("/api/population");
    const payoff = asObj(data.payoff_matrix);
    const agents = Array.isArray(payoff.labels)
      ? payoff.labels.map(String)
      : Array.isArray(data.population)
        ? data.population.map(String)
        : [];
    const rawMatrix = Array.isArray(payoff.matrix) ? payoff.matrix : [];
    const matrix: (number | null)[][] = rawMatrix.map((row) =>
      Array.isArray(row) ? row.map((v) => (typeof v === "number" ? v : null)) : [],
    );
    return {
      agents,
      matrix,
      suite: data.synthetic ? "DEMO_SUITE" : "PFSP_LATEST",
      updatedAt: new Date().toISOString(),
    };
  }

  /* Pass 2 — Explainability */
  async listExplanations(): Promise<ExplanationRecord[]> {
    const data = await request<Json>("/api/explainability");
    const rows = Array.isArray(data.explanations) ? data.explanations : [];
    return rows.map((raw, i) => {
      const row = asObj(raw);
      return {
        id: str(row.id, `explanation-${i}`),
        kind: KIND,
        matchId: str(row.match_id, "NOT RECORDED"),
        turn: num(row.turn),
        method: str(row.method ?? row.kind, "recorded"),
        saliencyMap: Array.isArray(row.saliency_map) ? (row.saliency_map as number[][]) : [],
        topFeatures: [],
        faithfulness: "NOT_EVALUATED" as const,
        notes: str(row.note ?? data.note) || undefined,
      };
    });
  }

  async getExplanationById(id: string): Promise<ExplanationRecord | null> {
    const all = await this.listExplanations();
    return all.find((e) => e.id === id) ?? null;
  }

  async getCounterfactuals(_explanationId: string): Promise<CounterfactualRecord[]> {
    return [];
  }

  /* Pass 2 — Champion */
  async getPromotionChecklist(): Promise<PromotionChecklist> {
    const data = await request<Json>("/api/champion");
    const checklist = asObj(data.promotion_checklist);
    const reasons = Array.isArray(checklist.reasons) ? checklist.reasons.map(String) : [];
    const rows: ChecklistRow[] = [
      {
        gate: "LEARNED_PROMOTION_GATE",
        status: mapChecklistStatus(checklist.LEARNED_PROMOTION_GATE),
        detail: reasons.join("; ") || str(data.learned_champion_note, "NO LEARNED CHAMPION"),
        blockerReason: checklist.blocked ? reasons[0] : undefined,
      },
      {
        gate: "competitive_evaluation",
        status: mapChecklistStatus(checklist.competitive_evaluation),
        detail: str(checklist.competitive_evaluation, "NOT_EVALUATED"),
      },
      {
        gate: "official_cpu_packaging",
        status: mapChecklistStatus(checklist.official_cpu_packaging),
        detail: str(checklist.official_cpu_packaging, "NOT_EVALUATED"),
      },
      {
        gate: "portal_ready",
        status: checklist.portal_ready ? "PASS" : "BLOCKED",
        detail: checklist.portal_ready ? "Ready" : "Portal package promotion blocked",
      },
    ];
    const overall: ChecklistStatus = rows.some((r) => r.status === "FAIL")
      ? "FAIL"
      : rows.some((r) => r.status === "BLOCKED")
        ? "BLOCKED"
        : rows.every((r) => r.status === "PASS")
          ? "PASS"
          : "PENDING";
    return {
      id: "promotion-checklist-api",
      kind: KIND,
      candidateId: str(data.heuristic_baseline ?? data.local_champion, "NOT RECORDED"),
      rows,
      overallStatus: overall,
      promotionAllowed: checklist.portal_ready === true && !checklist.blocked,
    };
  }

  async getChampionWorkspace(): Promise<ChampionWorkspace> {
    const data = await request<Json>("/api/champion");
    const checklist = await this.getPromotionChecklist();
    const pkg = asObj(data.active_submitted_package);
    return {
      id: "champion-workspace-api",
      kind: KIND,
      schemaVersion: SCHEMA_VERSION,
      currentChampion: str(data.learned_champion) || null,
      currentCandidate: str(data.local_champion ?? data.heuristic_baseline, "NOT RECORDED"),
      currentSubmittedBaseline: str(pkg.candidate) || null,
      checklist,
      updatedAt: new Date().toISOString(),
    };
  }

  /* Pass 2 — Submission */
  async getSubmissionPipeline(): Promise<SubmissionPipeline> {
    const data = await request<Json>("/api/submission");
    const pkg = asObj(data.package);
    const parity = asObj(data.linux_parity_report);
    const active: SubmissionPackage | null = pkg.candidate
      ? {
          id: str(pkg.package_id ?? pkg.candidate, "package"),
          kind: KIND,
          candidateName: str(pkg.candidate),
          checkpoint: str(pkg.candidate),
          sha256: str(pkg.sha256) || null,
          sizeBytes: typeof pkg.size_bytes === "number" ? pkg.size_bytes : null,
          builtAt: str(pkg.built_at) || null,
          validatedWindowsAt: null,
          validatedLinuxAt: parity.decision ? new Date().toISOString() : null,
          notes: str(data.upload_note) || undefined,
        }
      : null;

    const stages: PackageStage[] = [
      "CANDIDATE_SELECTED",
      "PACKAGE_BUILT",
      "WINDOWS_VALIDATED",
      "LINUX_VALIDATED",
      "UPLOAD_READY",
      "MANUALLY_SUBMITTED",
      "PORTAL_ACCEPTED",
      "QUALIFIED",
    ];
    const portal = asObj(data.active_portal_submission);
    const steps: PipelineStep[] = stages.map((stage) => {
      let status: PipelineStep["status"] = "pending";
      if (stage === "CANDIDATE_SELECTED" && active) status = "complete";
      if (stage === "PACKAGE_BUILT" && active?.sha256) status = "complete";
      if (stage === "LINUX_VALIDATED" && parity.decision) status = "complete";
      if (stage === "UPLOAD_READY") status = data.upload_enabled ? "active" : "blocked";
      if (stage === "MANUALLY_SUBMITTED" && portal.candidate) status = "complete";
      if (stage === "PORTAL_ACCEPTED" && portal.status) status = "complete";
      return {
        stage,
        label: stage.replaceAll("_", " "),
        status,
        blockerReason: stage === "UPLOAD_READY" && !data.upload_enabled ? str(data.upload_note) : undefined,
      };
    });
    const current =
      steps.find((s) => s.status === "active" || s.status === "blocked")?.stage ??
      (active ? "PACKAGE_BUILT" : "CANDIDATE_SELECTED");

    return {
      id: "submission-pipeline-api",
      kind: KIND,
      currentStage: current,
      steps,
      activePackage: active,
      updatedAt: new Date().toISOString(),
    };
  }

  async listPackages(): Promise<SubmissionPackage[]> {
    const pipeline = await this.getSubmissionPipeline();
    return pipeline.activePackage ? [pipeline.activePackage] : [];
  }

  async runDemoPackageAction(_action: PackageAction): Promise<SubmissionPipeline> {
    return disabled(
      "runDemoPackageAction",
      "Package simulation actions are disabled in API mode. Uploads remain manual by design.",
    );
  }

  /* Pass 2 — Competition */
  async listPortalObservations(): Promise<PortalObservation[]> {
    const data = await request<Json>("/api/competition");
    const active = asObj(data.active_submission);
    if (!active.candidate && !active.candidate_id) return [];
    return [
      {
        id: str(active.id ?? active.candidate ?? "portal-observation"),
        kind: "OFFICIAL_PORTAL_OBSERVATION",
        candidateName: str(active.candidate ?? active.candidate_id, "NOT RECORDED"),
        rank: typeof active.rank === "number" ? active.rank : undefined,
        score: typeof active.score === "number" ? active.score : undefined,
        observedAt: str(active.observed_at, new Date().toISOString()),
        notes: str(data.note) || undefined,
      },
    ];
  }

  async listManualSubmissionRecords(): Promise<ManualSubmissionRecord[]> {
    const data = await request<Json>("/api/competition");
    const snap = asObj(data.profile_snapshot);
    if (!snap.candidate && !Object.keys(snap).length) return [];
    return [
      {
        id: "manual-profile-snapshot",
        kind: "MANUALLY_RECORDED",
        candidateName: str(snap.candidate ?? snap.username, "NOT RECORDED"),
        submittedAt: str(snap.captured_at, new Date().toISOString()),
        method: "manual profile snapshot",
        notes: str(data.note) || undefined,
      },
    ];
  }

  /* Pass 2 — Repository */
  async getRepositoryStatus(): Promise<RepositoryStatus> {
    const data = await request<Json>("/api/repository");
    return {
      id: "repository-status-api",
      kind: KIND,
      schemaVersion: SCHEMA_VERSION,
      branch: str(data.branch),
      engineSha: str(data.engine_commit),
      hardware: "NOT RECORDED",
      linuxParityStatus: "skipped",
      testStatus: "skipped",
      packageStatus: "BUILT",
      updatedAt: new Date().toISOString(),
    };
  }

  async listRecentCommits(): Promise<CommitRecord[]> {
    const data = await request<Json>("/api/repository");
    const lines = Array.isArray(data.recent_commits) ? data.recent_commits.map(String) : [];
    return lines.map((line, i) => {
      const [sha, ...rest] = line.split(" ");
      return {
        id: `commit-${i}-${sha}`,
        kind: KIND,
        sha: sha || `unknown-${i}`,
        message: rest.join(" ") || line,
        author: "NOT RECORDED",
        committedAt: new Date().toISOString(),
        branch: str(data.branch),
      };
    });
  }

  async listCiRuns(): Promise<CiRun[]> {
    return [];
  }

  async getEnvironmentLocks(): Promise<EnvironmentLock[]> {
    return [];
  }

  /* Pass 2 — Documentation */
  async getDocumentationIndex(): Promise<DocIndex> {
    const data = await request<{ sections?: { id: string; title: string }[]; schema_version?: number }>(
      "/api/documentation",
    );
    return {
      schemaVersion: SCHEMA_VERSION,
      sections: (data.sections ?? []).map((s, order) => ({
        id: s.id,
        title: s.title,
        order,
      })),
    };
  }

  async getDocumentationSection(id: string): Promise<DocSection | null> {
    const index = await this.getDocumentationIndex();
    const entry = index.sections.find((s) => s.id === id);
    if (!entry) return null;
    return {
      id: entry.id,
      title: entry.title,
      order: entry.order,
      content:
        `Section «${entry.title}» — body text is not served by /api/documentation yet.\n` +
        `Use repository docs under docs/ and plans/ for authoritative guidance.`,
      tags: ["api"],
    };
  }
}
