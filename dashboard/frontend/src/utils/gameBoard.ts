import { BoardState, CellState, CellTerrain, CellOwner } from "../types/match";

const seededRand = (seed: number) => {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
};

export function generateBoard(width: number, height: number, turn: number): BoardState {
  const rand = seededRand(width * height + turn * 7919);
  const cells: CellState[][] = [];

  for (let r = 0; r < height; r++) {
    const row: CellState[] = [];
    for (let c = 0; c < width; c++) {
      const rn = rand();
      let terrain: CellTerrain = "plain";
      if (rn < 0.08) terrain = "mountain";
      else if (rn < 0.14) terrain = "city";

      const rn2 = rand();
      let owner: CellOwner = "neutral";
      if (r < 4 && c < 4 && rn2 > 0.4) owner = "player1";
      else if (r >= height - 4 && c >= width - 4 && rn2 > 0.4) owner = "player2";

      if (r === 1 && c === 1) { terrain = "general"; owner = "player1"; }
      if (r === height - 2 && c === width - 2) { terrain = "general"; owner = "player2"; }

      const baseArmies = terrain === "city" ? 20 + Math.floor(rand() * 20) : 0;
      const owned = owner !== "neutral" ? Math.floor(turn * 0.5 + rand() * 8) + 1 : baseArmies;

      row.push({ terrain, owner, armies: owned, visible: owner === "player1" || rn2 > 0.55 });
    }
    cells.push(row);
  }

  return { width, height, cells, turn };
}

export function applyMove(
  board: BoardState,
  srcRow: number,
  srcCol: number,
  dstRow: number,
  dstCol: number,
): BoardState {
  const cells = board.cells.map(row => row.map(cell => ({ ...cell })));
  const src = cells[srcRow]?.[srcCol];
  const dst = cells[dstRow]?.[dstCol];
  if (!src || !dst || src.armies <= 1) return board;

  const moving = src.armies - 1;
  src.armies = 1;

  if (dst.owner === src.owner) {
    dst.armies += moving;
  } else if (dst.armies < moving) {
    dst.owner = src.owner;
    dst.armies = moving - dst.armies;
  } else {
    dst.armies -= moving;
  }

  return { ...board, cells };
}

export function coordinateLabel(row: number, col: number): string {
  return `${String.fromCharCode(65 + col)}${row + 1}`;
}
