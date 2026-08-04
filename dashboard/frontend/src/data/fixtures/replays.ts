import { ReplayRecord } from "../../types/replay";
import { generateBoard } from "../../utils/gameBoard";

export const demoReplay: ReplayRecord = {
  id: "replay-demo-001",
  kind: "DEMO",
  matchId: "match-demo-001",
  config: {
    player1: "cnn_agent",
    player2: "heuristic",
    mapPreset: "standard",
    mapSize: 18,
    speedMultiplier: 1,
  },
  frames: Array.from({ length: 20 }, (_, i) => ({
    turn: i,
    board: generateBoard(18, 18, i),
    p1Armies: 20 + i * 2,
    p2Armies: 18 + i,
    p1Land: 10 + i,
    p2Land: 9 + i,
    events: i === 0 ? ["Match started"] : [],
  })),
  events: [
    { turn: 0,  type: "army_move",        label: "P1 opens center",       player: "player1" },
    { turn: 5,  type: "city_taken",        label: "P1 captures city C7",   player: "player1" },
    { turn: 12, type: "capture",           label: "P2 recaptures C7",      player: "player2" },
    { turn: 18, type: "general_captured",  label: "P1 captures P2 general",player: "player1" },
  ],
  decisions: Array.from({ length: 20 }, (_, i) => ({
    turn: i,
    srcRow: 3 + (i % 4),
    srcCol: 3 + (i % 4),
    dstRow: 4 + (i % 4),
    dstCol: 4 + (i % 4),
    armiesMoved: 5 + i,
    policyLogit: 2.1 + Math.sin(i) * 0.5,
    valueEstimate: 0.6 + i * 0.01,
    topKActions: [],
  })),
  outcome: "player1_win",
  totalTurns: 20,
  createdAt: new Date().toISOString(),
  label: "Demo match — CNN vs Heuristic",
};
