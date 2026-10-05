import { useRef, useSyncExternalStore } from "react";
import { HUB_MESSAGES, HubStore, type HubSnap } from "./hubcore";
import { studio } from "./studio";

// The Models page's link to the server's hub commands (src/phi_studio/hub_api.py). The store and its
// rules live in hubcore.ts, import-free so Node tests run them; this file wires it to the Studio link
// and to React. Its own store, so Hub replies never re-render the rest of Studio.

export * from "./hubcore";

export const hub = new HubStore({
  send: (msg) => studio.send(msg),
  localError: (message, fix) => studio.localError(message, fix),
  later: (fn) => { window.setTimeout(fn, 0); },
});
for (const type of HUB_MESSAGES) studio.onMessage(type, (m) => hub.receive(type, m));

function shallowEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null || Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => Object.is((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}

/** Same caching as useStudio and useTerminal: a selector returning a fresh object must not re-render
 * forever (React error #185). */
export function useHub<T>(select: (s: HubSnap) => T): T {
  const last = useRef<{ value: T } | null>(null);
  return useSyncExternalStore(hub.subscribe, () => {
    const value = select(hub.snap);
    if (last.current && shallowEqual(last.current.value, value)) return last.current.value;
    last.current = { value };
    return value;
  });
}
