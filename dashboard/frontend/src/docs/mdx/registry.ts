/**
 * Build-time MDX modules for required console docs.
 * Content remains authored under docs/console/; these modules exist so Vite's
 * @mdx-js/rollup pipeline is exercised and pages can import compiled MDX.
 *
 * Decorative diagrams / custom MDX components are intentionally out of scope.
 */
export { default as OverviewDoc } from "./overview.mdx";
export { default as GlossaryDoc } from "./glossary.mdx";
export { default as EnvOfficialDoc } from "./env-official.mdx";

export const MDX_SECTION_IDS = ["overview", "glossary", "env-official"] as const;
export type MdxSectionId = (typeof MDX_SECTION_IDS)[number];
