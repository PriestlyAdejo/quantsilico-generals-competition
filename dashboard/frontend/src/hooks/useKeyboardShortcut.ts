import { useEffect } from "react";

export function useKeyboardShortcut(
  keys: string[],
  callback: () => void,
  deps: unknown[] = [],
) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const isMac = navigator.platform.toUpperCase().includes("MAC");
      const modKey = isMac ? e.metaKey : e.ctrlKey;
      if (modKey && keys.includes(e.key.toLowerCase())) {
        e.preventDefault();
        callback();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

export function modKeyLabel(): string {
  const isMac = navigator.platform.toUpperCase().includes("MAC");
  return isMac ? "⌘" : "Ctrl";
}
