import { FitAddon } from "@xterm/addon-fit";
import { Terminal as XTerm } from "@xterm/xterm";
import "@xterm/xterm/css/xterm.css";
import { OctagonX, RotateCcw, SquareTerminal, X } from "lucide-react";
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { studio, useStudio, useTheme } from "../lib/studio";
import { shortCommand, terminal, useTerminal, type Guard } from "../lib/terminal";

const HEIGHT_KEY = "phi-studio-terminal-h";

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function xtermTheme() {
  return {
    background: cssVar("--bg-sunken"), foreground: cssVar("--text-primary"), cursor: cssVar("--text-primary"),
    cursorAccent: cssVar("--bg-sunken"), selectionBackground: cssVar("--bg-selected"),
  };
}

function savedHeight(): number {
  try { return Math.max(160, Number(localStorage.getItem(HEIGHT_KEY)) || 320); } catch { return 320; }
}

// The terminal panel along the bottom of every page: the user's own shell on this Mac, in the phi env, in the
// main checkout. Run buttons type their command here; the user can also type anything.
// WHY mounted once and hidden when closed: xterm keeps the screen and scrollback, and the socket stays open,
// so closing and reopening the panel loses nothing.
export function TerminalDock() {
  const open = useTerminal((s) => s.open);
  const [mounted, setMounted] = useState(open);
  useEffect(() => { if (open) setMounted(true); }, [open]);
  if (!mounted) return null;
  return <Dock open={open} />;
}

function Dock({ open }: { open: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const xterm = useRef<XTerm | null>(null);
  const fit = useRef<FitAddon | null>(null);
  const [height, setHeight] = useState(savedHeight);
  const running = useTerminal((s) => s.running);
  const typing = useTerminal((s) => s.typing);
  const alive = useTerminal((s) => s.alive);
  const link = useTerminal((s) => s.link);
  const error = useTerminal((s) => s.error);
  const guard = useTerminal((s) => s.guard);
  const control = useStudio((s) => s.control);
  const theme = useTheme();

  useEffect(() => {
    const term = new XTerm({
      fontFamily: cssVar("--font-mono") || "monospace", fontSize: 14, lineHeight: 1.25, cursorBlink: true,
      scrollback: 5000, theme: xtermTheme(), convertEol: false, allowProposedApi: false,
    });
    const f = new FitAddon();
    term.loadAddon(f);
    term.open(host.current!);
    xterm.current = term;
    fit.current = f;
    const offOut = terminal.onOutput((data) => term.write(data));
    const offIn = term.onData((d) => terminal.input(d));
    const offSize = term.onResize(({ cols, rows }) => terminal.resize(cols, rows));
    const ro = new ResizeObserver(() => { try { f.fit(); } catch { /* hidden: no size yet */ } });
    ro.observe(host.current!);
    terminal.connect();
    return () => { offOut(); offIn.dispose(); offSize.dispose(); ro.disconnect(); term.dispose(); };
  }, []);

  useEffect(() => { if (xterm.current) xterm.current.options.theme = xtermTheme(); }, [theme]);
  useEffect(() => { if (xterm.current) xterm.current.options.disableStdin = !typing; }, [typing]);
  useEffect(() => {
    if (!open) return;
    requestAnimationFrame(() => {
      try { fit.current?.fit(); } catch { /* not laid out yet */ }
      if (xterm.current) terminal.resize(xterm.current.cols, xterm.current.rows);
      xterm.current?.focus();
    });
  }, [open, height]);

  const drag = (e: ReactPointerEvent<HTMLDivElement>) => {
    const startY = e.clientY;
    const start = height;
    const move = (ev: PointerEvent) => setHeight(Math.min(Math.max(160, start + startY - ev.clientY), window.innerHeight - 160));
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      setHeight((h) => { try { localStorage.setItem(HEIGHT_KEY, String(h)); } catch { /* keep for this page */ } return h; });
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const status = link !== "open" ? "Connecting" : !alive ? "Shell stopped" : running ? `Running ${shortCommand(running.command) || "a command"}` : "Ready";
  return (
    <section className={`term-dock ${open ? "" : "is-hidden"}`} style={{ height }} aria-label="Terminal">
      <div className="term-grip" onPointerDown={drag} role="separator" aria-orientation="horizontal" aria-label="Resize the terminal" />
      <div className="term-head">
        <SquareTerminal aria-hidden />
        <span className="term-title">Terminal</span>
        <span className={`term-status ${running ? "is-busy" : ""}`} title={status}>{status}</span>
        {!control && <span className="term-note">View only. Take control to type.</span>}
        <span className="term-spacer" />
        <button className="btn btn-sm" onClick={() => terminal.interrupt()} disabled={!running || !typing} title="Send Ctrl-C to the running command">
          <OctagonX aria-hidden /> Stop command
        </button>
        <button className="btn btn-sm" onClick={() => terminal.restart()} disabled={!typing} title="Close this shell and open a new one">
          <RotateCcw aria-hidden /> Restart
        </button>
        <button className="btn btn-ghost btn-sm btn-icon" onClick={() => terminal.setOpen(false)} aria-label="Hide the terminal" title="Hide">
          <X aria-hidden />
        </button>
      </div>
      {error && (
        <div className="term-error" role="alert">
          <span><strong>{error.message}</strong> {error.fix}</span>
          {!control && error.message.includes("control") && (
            <button className="btn btn-sm" onClick={() => studio.send({ cmd: "take_control" })}>Take control</button>
          )}
          <button className="btn btn-ghost btn-sm btn-icon" onClick={() => terminal.dismissError()} aria-label="Dismiss"><X aria-hidden /></button>
        </div>
      )}
      {guard && <GuardBanner guard={guard} typing={typing} />}
      <div className="term-body" ref={host} />
    </section>
  );
}

// LeRobot asks "Press ENTER to use provided calibration file". ENTER writes that file into the arm's motors, so
// this says which arm and file, and whether that is that arm's own file (cmdcheck.py).
function GuardBanner({ guard, typing }: { guard: Guard; typing: boolean }) {
  const [sure, setSure] = useState(false);
  const file = guard.file?.split("/").pop() ?? "the file";
  if (guard.level === "ok") {
    return (
      <div className="term-guard tone-info" role="status">
        <span><strong>LeRobot asks for ENTER.</strong> ENTER writes {file} into the motors. {guard.message} Type c and ENTER
          instead to calibrate it again.</span>
        <button className="btn btn-ghost btn-sm btn-icon" onClick={() => terminal.dismissGuard()} aria-label="Dismiss"><X aria-hidden /></button>
      </div>
    );
  }
  return (
    <div className="term-guard tone-danger" role="alert">
      <span>
        <strong>{guard.held ? "Studio held your ENTER." : "Do not press ENTER yet."}</strong> {guard.message} {guard.fix}
        {" "}ENTER would write {file} into the arm's motors.
      </span>
      <span className="term-guard-actions">
        <button className="btn btn-sm" disabled={!typing} onClick={() => terminal.interrupt()}>Stop command</button>
        {/* WHY not for danger: c on a crossed port and id calibrates one arm into the other's file */}
        {guard.level === "unknown" && <button className="btn btn-sm" disabled={!typing} onClick={() => terminal.recalibrate()}>Calibrate it again (c)</button>}
        {guard.held && guard.level === "unknown" && (sure
          ? <button className="btn btn-sm btn-danger" disabled={!typing} onClick={() => terminal.allow()}>Yes, write {file}</button>
          : <button className="btn btn-ghost btn-sm" disabled={!typing} onClick={() => setSure(true)}>Write it anyway</button>)}
      </span>
    </div>
  );
}
