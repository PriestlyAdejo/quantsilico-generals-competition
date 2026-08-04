# Figma → target mapping (integration)

| Figma path | Target | Action |
|---|---|---|
| `src/components/shell/*` | `dashboard/frontend/src/components/shell/*` | ADAPT (no Tailwind) |
| `src/app/navigation.ts` | `dashboard/frontend/src/navigation.ts` | PORT structure |
| `src/services/dataSource.ts` | `dashboard/frontend/src/data/DataSource.ts` | ADAPT + ApiDataSource |
| `src/services/mockDataSource.ts` | — | REJECT as production default |
| `src/data/fixtures/*` | — | REJECT |
| `src/styles/theme.css` tokens | `dashboard/frontend/src/styles/tokens.css` | PORT values |
| `src/styles/tailwind.css` | — | REJECT |
| `src/app/components/ui/*` (shadcn) | — | REJECT |
| `src/imports/*` | — | REJECT (review only; no production import) |
| `src/pages/*` | `dashboard/frontend/src/pages/*` | ADAPT to real APIs |
| `src/components/board/GeneralsBoard.tsx` | `dashboard/frontend/src/components/board/GeneralsBoard.tsx` | ADAPT + mapKey terrain identity |
| Figma `package.json` / Vite config | — | REJECT |

Extracted ZIP remains under `var/imports/` (gitignored). No production or test import from `var/imports/figma-console-extracted/`.
