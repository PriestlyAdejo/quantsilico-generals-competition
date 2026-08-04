/// <reference types="vite/client" />
/// <reference types="@mdx-js/react" />

declare module "*.mdx" {
  import type { ComponentType } from "react";
  const MDXComponent: ComponentType<Record<string, unknown>>;
  export default MDXComponent;
}
