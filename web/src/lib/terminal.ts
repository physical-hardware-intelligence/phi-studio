import { useRef, useSyncExternalStore } from "react";
import { studio } from "./studio";

// The terminal panel's link to the shell on this Mac (server: /terminal, terminal.py). Output arrives as raw
// bytes for xterm; status says whether a command holds the terminal and whether this window may type.

export interface Running { pgid: number; command: string }
export interface TermState {
  open: boolean; // the panel is showing
  link: "idle" | "connecting" | "open" | "closed";
  alive: boolean; // the shell is running
  running: Running | null; // the command in the foreground, null at the prompt
  typing: boolean; // this window has control, so it may type
  error: { message: string; fix: string } | null;
}

type Listener = () => void;

class TerminalLink {
  snap: TermState = { open: false, link: "idle", alive: false, running: null, typing: false, error: null };
  private listeners = new Set<Listener>();
  private output = new Set<(data: Uint8Array) => void>();
  private ws: WebSocket | null = null;
  private client: string | null = null;
  private queue: string[] = []; // sent once the socket opens: a Run clicked before the panel connected
  private retryTimer: number | null = null;

  constructor() {
    // WHY follow the main link: the server knows this window by the id it gave the main socket, and a new
    // main socket (a reconnect) means a new id, so the terminal socket must reopen with it.
    studio.subscribe(() => {
      const id = studio.snap.client;
      if (id !== this.client) {
        this.client = id;
        if (this.snap.open || this.ws) this.connect();
      }
    });
    // Stop (the button or Esc) also ends a LeRobot command running here: it may be driving the arms.
    // WHY not lerobot-calibrate: it never powers a motor, and Esc pressed by habit midway would leave the
    // motors reset to 0..4095 with no file saved.
    studio.onStop.push(() => {
      if (this.snap.running && /\blerobot-(?!calibrate\b)/.test(this.snap.running.command)) this.interrupt();
    });
  }

  subscribe = (fn: Listener): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  onOutput(fn: (data: Uint8Array) => void): () => void {
    this.output.add(fn);
    return () => this.output.delete(fn);
  }

  private set(patch: Partial<TermState>): void {
    this.snap = { ...this.snap, ...patch };
    for (const fn of this.listeners) fn();
  }

  setOpen(open: boolean): void {
    this.set({ open });
    if (open && !this.ws) this.connect();
  }

  connect(): void {
    if (this.retryTimer !== null) { window.clearTimeout(this.retryTimer); this.retryTimer = null; }
    const old = this.ws;
    if (old) { old.onopen = old.onmessage = old.onclose = null; old.close(); }
    this.ws = null;
    const tok = studio.token();
    if (!tok || !this.client) { this.set({ link: "connecting" }); return; } // waits for the main link's hello
    const ws = new WebSocket(`ws://${location.host}/terminal?token=${encodeURIComponent(tok)}&client=${encodeURIComponent(this.client)}`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    this.set({ link: "connecting", error: null });
    ws.onopen = () => {
      this.set({ link: "open" });
      for (const m of this.queue.splice(0)) ws.send(m);
    };
    ws.onmessage = (e) => {
      if (typeof e.data !== "string") {
        const bytes = new Uint8Array(e.data as ArrayBuffer);
        for (const fn of this.output) fn(bytes);
        return;
      }
      const m = JSON.parse(e.data);
      if (m.t === "status") this.set({ alive: m.alive, running: m.running ?? null, typing: m.typing });
      else if (m.t === "error") this.set({ error: { message: m.message, fix: m.fix ?? "" } });
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.set({ link: "closed", running: null, typing: false });
      if (this.snap.open) this.retryTimer = window.setTimeout(() => { this.retryTimer = null; this.connect(); }, 2000);
    };
  }

  private send(msg: object): void {
    const text = JSON.stringify(msg);
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(text);
    else { this.queue.push(text); if (!this.ws) this.connect(); }
  }

  /** Types a whole command and presses Enter, after opening the panel so the output is in view. */
  run(cmd: string): void {
    this.set({ error: null });
    this.setOpen(true);
    this.send({ t: "run", cmd });
  }
  input(data: string): void { this.send({ t: "in", d: data }); }
  interrupt(): void { this.send({ t: "interrupt" }); }
  restart(): void { this.set({ error: null }); this.send({ t: "restart" }); }
  resize(cols: number, rows: number): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ t: "resize", cols, rows }));
  }
  dismissError(): void { this.set({ error: null }); }
}

export const terminal = new TerminalLink();

/** "lerobot-find-port" for "/opt/.../python /opt/.../bin/lerobot-find-port": what the user typed, not how the
 * OS sees it. */
export function shortCommand(cmd: string): string {
  const parts = cmd.trim().split(/\s+/);
  if (parts.length > 1 && /(^|\/)python[\d.]*$/.test(parts[0])) parts.shift();
  if (parts[0]) parts[0] = parts[0].split("/").pop() ?? parts[0];
  return parts.join(" ");
}

/** Same caching as useStudio: a selector returning a fresh object must not re-render forever. */
export function useTerminal<T>(select: (s: TermState) => T): T {
  const last = useRef<{ value: T } | null>(null);
  return useSyncExternalStore(terminal.subscribe, () => {
    const value = select(terminal.snap);
    if (last.current && Object.is(last.current.value, value)) return last.current.value;
    last.current = { value };
    return value;
  });
}
