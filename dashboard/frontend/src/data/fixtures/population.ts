import { PopulationEntry, PayoffMatrix } from "../../types/population";

export const populationEntries: PopulationEntry[] = [
  {
    id: "pop-heuristic-v2f",
    kind: "IMPORTED_PROJECT_EVIDENCE",
    name: "heuristic_v2f",
    checkpoint: "heuristic-v2f-evaluated",
    payoffs: [null, 0.62, 0.78, 0.91],
    pfspWeight: 0.42,
    gamesPlayed: 48,
    winRate: 0.44,
    isMainAgent: true,
  },
  {
    id: "pop-hunter",
    kind: "IMPORTED_PROJECT_EVIDENCE",
    name: "hunter_heuristic",
    checkpoint: "hunter-baseline",
    payoffs: [0.38, null, 0.55, 0.71],
    pfspWeight: 0.31,
    gamesPlayed: 48,
    winRate: 0.52,
    isMainAgent: false,
  },
  {
    id: "pop-random",
    kind: "DEMO",
    name: "legal_random",
    checkpoint: "random-demo",
    payoffs: [0.22, 0.45, null, 0.60],
    pfspWeight: 0.15,
    gamesPlayed: 20,
    winRate: 0.22,
    isMainAgent: false,
  },
  {
    id: "pop-expander",
    kind: "IMPORTED_PROJECT_EVIDENCE",
    name: "CNN-v3-Expander",
    checkpoint: "ckpt-expander-v3-step-480k",
    payoffs: [0.09, 0.29, 0.40, null],
    pfspWeight: 0.12,
    gamesPlayed: 30,
    winRate: 0.30,
    isMainAgent: false,
  },
];

export const payoffMatrix: PayoffMatrix = {
  agents: ["heuristic_v2f", "hunter_heuristic", "legal_random", "CNN-v3-Expander"],
  matrix: [
    [null, 0.62, 0.78, 0.91],
    [0.38, null, 0.55, 0.71],
    [0.22, 0.45, null, 0.60],
    [0.09, 0.29, 0.40, null],
  ],
  suite: "PFSP_LATEST",
  updatedAt: "2024-11-06T00:00:00.000Z",
};
