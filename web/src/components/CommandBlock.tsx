import { Check, Copy, Play } from "lucide-react";
import { useState } from "react";
import { setup, useSetup } from "../lib/setup";
import { useStudio } from "../lib/studio";
import { shortCommand, terminal, useTerminal } from "../lib/terminal";

// What each blank in a generated command asks for (rigspec.py writes them as <name>).
const BLANKS: Record<string, { label: string; hint: string; pattern?: RegExp }> = {
  hf_user: { label: "Hugging Face user", hint: "your user or org on the Hub", pattern: /^[A-Za-z0-9][\w.-]*$/ },
  dataset: { label: "Dataset name", hint: "letters, digits, - and _", pattern: /^[\w.-]+$/ },
  task: { label: "Task", hint: "what the robot should do, in words; the policy reads this" },
  recorded: { label: "Recorded dataset", hint: "the full name with the date that record printed", pattern: /^[\w.-]+$/ },
};
const label = (name: string) => BLANKS[name]?.label ?? name.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

/** The command with each <blank> filled. WHY escape ' only: the server quotes every argument that holds a
 * blank with shlex.quote, and "<" forces quoting, so a blank always sits inside single quotes. */
export function fill(cmd: string, values: Record<string, string>): string {
  return cmd.replace(/<([a-z_]+)>/g, (m, name: string) => (name in values ? values[name].replace(/'/g, "'\\''") : m));
}

/** One flag per line with a trailing backslash, the way LeRobot's docs print them. The copied text is the
 * same, and a shell runs it as one command. With `run`, a Run button types it into the terminal panel.
 * Blanks are shared by every command and kept across reloads (lib/setup.ts): a name typed for record is
 * already there for replay. The Hugging Face user starts as robot-config.yaml's dataset.hf_user. */
export function CommandBlock({ cmd, wrap = true, run = false }: { cmd: string; wrap?: boolean; run?: boolean }) {
  const blanks = [...new Set([...cmd.matchAll(/<([a-z_]+)>/g)].map((m) => m[1]))];
  const typed = useSetup((s) => s.blanks);
  const hfUser = useStudio((s) => s.files.index?.lerobot?.hf_user ?? null);
  const values: Record<string, string> = { ...(hfUser ? { hf_user: hfUser } : {}), ...typed };
  const filled = fill(cmd, values);
  const text = wrap ? filled.split(/ (?=--|'--)/).join(" \\\n  ") : filled;
  const [done, setDone] = useState(false);
  const control = useStudio((s) => s.control);
  const busy = useTerminal((s) => (s.running ? shortCommand(s.running.command) || "a command" : null));

  const missing = blanks.filter((b) => !values[b]?.trim() || (BLANKS[b]?.pattern && !BLANKS[b].pattern!.test(values[b].trim())));
  const needsPort = blanks.includes("port");
  const why = needsPort ? "Find the ports first: an arm has no port yet."
    : !control ? "Take control to run commands here."
    : busy ? `The terminal is still running ${busy}.`
    : missing.length ? `Fill in ${missing.map(label).join(", ")} first.` : "";

  return (
    <div className="cmd">
      {run && blanks.filter((b) => b !== "port").length > 0 && (
        <div className="cmd-blanks">
          {blanks.filter((b) => b !== "port").map((b) => (
            <label key={b} className="field cmd-blank">
              <span className="field-label">{label(b)}</span>
              <input className="input" value={values[b] ?? ""} placeholder={BLANKS[b]?.hint}
                onChange={(e) => setup.setBlank(b, e.target.value)} spellCheck={false} />
            </label>
          ))}
        </div>
      )}
      <div className="cmd-block">
        <pre><code>{text}</code></pre>
        <div className="cmd-actions">
          {run && (
            <button className="btn btn-sm" disabled={!!why} title={why || "Run in the terminal below"}
              onClick={() => terminal.run(filled)}>
              <Play aria-hidden /> Run
            </button>
          )}
          <button className="btn btn-ghost btn-sm btn-icon" aria-label="Copy command" title="Copy"
            onClick={() => { void navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500); }}>
            {done ? <Check aria-hidden /> : <Copy aria-hidden />}
          </button>
        </div>
      </div>
      {run && why && <p className="cmd-why-not">{why}</p>}
    </div>
  );
}
