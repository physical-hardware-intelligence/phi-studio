import { useRef, useSyncExternalStore } from "react";
import { studio } from "./studio";
import { terminal } from "./terminal";
import { TRAIN_MESSAGES, TrainStore, type TrainSnap } from "./traincore";

// The Train page's link to the server's train commands (src/phi_studio/train_api.py). The store and its
// rules live in traincore.ts, import-free so Node tests run them; this file wires it to the Studio link,
// the terminal and React.

export * from "./traincore";

export const train = new TrainStore({
  send: (msg) => studio.send(msg),
  run: (command) => terminal.run(command),
  now: () => new Date(),
});
for (const type of TRAIN_MESSAGES) studio.onMessage(type, (m) => train.receive(type, m));

function shallowEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null || Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => Object.is((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}

/** Same caching as useStudio: a selector returning a fresh object must not re-render forever (React #185). */
export function useTrain<T>(select: (s: TrainSnap) => T): T {
  const last = useRef<{ value: T } | null>(null);
  return useSyncExternalStore(train.subscribe, () => {
    const value = select(train.snap);
    if (last.current && shallowEqual(last.current.value, value)) return last.current.value;
    last.current = { value };
    return value;
  });
}
