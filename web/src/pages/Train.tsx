import * as Dialog from "@radix-ui/react-dialog";
import {
  CircleCheck, CircleDashed, CircleX, CornerDownRight, Download, LoaderCircle, Play, RefreshCw, Search, Send,
  Settings as SettingsIcon, Square, TriangleAlert, X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ComponentType, type MouseEvent, type ReactNode } from "react";
import { CommandBlock } from "../components/CommandBlock";
import { Notices } from "../components/Notices";
import { useStudio } from "../lib/studio";
import { shortCommand, useTerminal } from "../lib/terminal";
import {
  bytes, duration, settingName, stateWords, train, useTrain,
  type CheckRow, type CheckStatus, type Fit, type JobForm, type Point, type Run, type TrainNotice, type TrainSettings,
} from "../lib/train";
import "../styles/train.css";

const MAX_PARTS = 8; // the gpu QoS's MaxSubmitPU on Explorer (train.py MAX_PARTS)
const ENDED = new Set(["COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
  "BOOT_FAIL", "DEADLINE", "NOT SUBMITTED", "STOPPED"]);
// sacct can say "CANCELLED by 1234"; every other state is one word or one of Studio's own two-word states.
const ended = (state: string) => ENDED.has(state.startsWith("CANCELLED") ? "CANCELLED" : state);
const seconds = (t: string) => { const [h, m, x] = t.split(":").map(Number); return h * 3600 + m * 60 + x; };

// Train a LeRobot policy on this Mac (in Studio's terminal) or on the cluster (SLURM over ssh).
export function Train() {
  const link = useStudio((s) => s.link);
  const loaded = useTrain((s) => s.loaded);
  const ready = useTrain((s) => s.ready);
  const where = useTrain((s) => s.form.where);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => { if (link === "open") train.init(); }, [link]);
  // Job states every 30 s while this page is open; one ssh call, and none when no run is live.
  useEffect(() => {
    if (link !== "open") return;
    train.refresh();
    const t = window.setInterval(() => { if (!document.hidden) train.refresh(); }, 30_000);
    return () => window.clearInterval(t);
  }, [link]);

  if (!loaded) return <div className="page"><Notices /><p className="empty panel">Loading training state.</p></div>;
  if (!ready) {
    return (
      <div className="page">
        <Notices />
        <p className="empty panel">Training needs Studio's data folder. Start Studio with --data-dir.</p>
      </div>
    );
  }
  return (
    <div className="page train">
      <Notices />
      <TrainNoticeBox />
      <BackgroundBox />
      <DroppedBox />
      <section className="panel train-where">
        <div>
          <h2 className="panel-title">Where to train</h2>
          <p className="panel-sub">
            {where === "cluster"
              ? "Explorer's GPUs through SLURM. Studio writes the job script and submits it over ssh."
              : "This Mac's GPU (MPS), in the terminal panel. Fine for a short test; slow for a real run."}
          </p>
        </div>
        <div className="train-seg" role="radiogroup" aria-label="Where to train">
          {(["cluster", "mac"] as const).map((w) => (
            <button key={w} role="radio" aria-checked={where === w} className={`train-seg-btn ${where === w ? "is-on" : ""}`}
              onClick={() => train.setForm({ where: w })}>
              {w === "cluster" ? "Cluster" : "This Mac"}
            </button>
          ))}
        </div>
      </section>
      <div className="work-grid">
        <div className="col">
          <JobPanel />
          <ScriptPanel />
        </div>
        <div className="col">
          <PlanPanel />
        </div>
      </div>
      {where === "cluster" && <ClusterPanel />}
      <RunsPanel selected={selected} onSelect={setSelected} />
      {selected && <RunDetail id={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

// -- notices from train commands -----------------------------------------------------------------------
function TrainNoticeBox() {
  const n = useTrain((s) => s.notice);
  return n ? <NoticeView n={n} onDismiss={() => train.dismiss()} /> : null;
}

// A failed poll or refresh, apart from the notice above, so it never hides an sbatch failure.
function BackgroundBox() {
  const n = useTrain((s) => s.background);
  return n ? <NoticeView n={n} onDismiss={() => train.dismissBackground()} /> : null;
}

function DroppedBox() {
  const dropped = useTrain((s) => s.settingsDropped);
  const [hidden, setHidden] = useState("");
  const key = dropped.join(",");
  if (!dropped.length || hidden === key) return null;
  return (
    <NoticeView onDismiss={() => setHidden(key)} n={{
      id: 0, tone: "warn",
      message: `Studio put ${dropped.length === 1 ? "this saved setting" : "these saved settings"} back to the default, because the saved value did not pass its check: ${dropped.map(settingName).join(", ")}.`,
      fix: "Open Settings, check the values, and save.",
    }} />
  );
}

function NoticeView({ n, onDismiss }: { n: TrainNotice; onDismiss: () => void }) {
  const Icon = n.tone === "ok" ? CircleCheck : n.tone === "warn" ? TriangleAlert : CircleX;
  return (
    <div className={`notice tone-${n.tone === "ok" ? "ok" : n.tone}`} role={n.tone === "danger" ? "alert" : "status"}>
      <Icon className="notice-icon" aria-hidden />
      <div className="train-notice-body">
        <div className="notice-title">{n.message}</div>
        {n.fix && <div className="notice-fix">{n.fix}</div>}
        {(n.cmd || n.output) && <CommandOutput cmd={n.cmd ?? null} output={n.output ?? ""} />}
      </div>
      <div className="notice-actions">
        <button className="btn btn-ghost btn-sm btn-icon" aria-label="Dismiss" onClick={onDismiss}><X aria-hidden /></button>
      </div>
    </div>
  );
}

function CommandOutput({ cmd, output, rc }: { cmd: string | null; output: string; rc?: number }) {
  return (
    <details className="train-io">
      <summary>Command and output{rc !== undefined ? `, exit code ${rc}` : ""}</summary>
      {cmd && <pre className="train-pre"><code>{`$ ${cmd}`}</code></pre>}
      <pre className="train-pre"><code>{output || "(no output)"}</code></pre>
    </details>
  );
}

// -- the job form ----------------------------------------------------------------------------------------
function JobPanel() {
  const f = useTrain((s) => s.form);
  const policies = useTrain((s) => s.policies);
  const datasets = useTrain((s) => s.datasets);
  const errors = useTrain((s) => s.preview?.errors ?? NO_ERRORS);
  const partTime = useTrain((s) => s.settings?.time ?? "08:00:00");
  const link = useStudio((s) => s.link);
  const set = (patch: Partial<JobForm>) => train.setForm(patch);

  // The preview follows the form, a moment after typing stops.
  useEffect(() => {
    if (link !== "open") return;
    const t = window.setTimeout(() => train.preview(), 250);
    return () => window.clearTimeout(t);
  }, [f, link]);

  const types = policies?.policies ?? [];
  const chosen = types.find((p) => p.type === f.policy);
  const ds = datasets.find((d) => d.id === f.dataset);
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Training job</h2>
          <p className="panel-sub">{policies ? `Policy types from ${policies.source}.` : "Reading the policy types from LeRobot."}</p>
        </div>
      </div>
      <div className="panel-body">
        <fieldset className="form train-form">
          <Field label="Dataset" error={f.dataset ? errors.dataset : undefined} wide
            hint={ds ? `On this Mac: ${ds.episodes ?? "?"} episodes, ${ds.frames ?? "?"} frames at ${ds.fps ?? "?"} fps.`
              : f.where === "cluster" ? "A Hub dataset id. The cluster downloads it from the Hub, so push a dataset recorded here first."
                : "A dataset id. Datasets recorded on this Mac are in the list."}>
            <input className="input" list="train-datasets" value={f.dataset} placeholder="Parv-09/my_dataset" spellCheck={false}
              onChange={(e) => set({ dataset: e.target.value })} />
            <datalist id="train-datasets">{datasets.map((d) => <option key={d.id} value={d.id} />)}</datalist>
          </Field>
          <Field label="Policy type" error={errors.policy}
            hint={chosen ? extrasHint(chosen.extra, f.where === "mac" ? chosen.missing_here : []) : undefined}>
            <select className="select" value={f.policy} onChange={(e) => set({ policy: e.target.value })} disabled={!!f.pretrained}>
              {types.length === 0 && <option value={f.policy}>{f.policy}</option>}
              {types.map((p) => <option key={p.type} value={p.type}>{p.type}</option>)}
            </select>
          </Field>
          <Field label="Start from a Hub policy (optional)" error={errors.pretrained}
            hint="Fine-tune that model instead of starting a new one. Its type replaces the one above.">
            <input className="input" value={f.pretrained} placeholder="lerobot/smolvla_base" spellCheck={false}
              onChange={(e) => set({ pretrained: e.target.value })} />
          </Field>
          <Field label="Steps" error={errors.steps}>
            <input className="input num" type="number" min={1} value={f.steps} onChange={(e) => set({ steps: num(e.target.value) })} />
          </Field>
          <Field label="Batch size" error={errors.batch_size}>
            <input className="input num" type="number" min={1} value={f.batch_size} onChange={(e) => set({ batch_size: num(e.target.value) })} />
          </Field>
          <Field label="Save a checkpoint every" error={errors.save_freq} hint="Steps. A later part resumes from the newest one.">
            <input className="input num" type="number" min={1} value={f.save_freq} onChange={(e) => set({ save_freq: num(e.target.value) })} />
          </Field>
          <Field label="Run name" error={(f.name ? errors.name : undefined) ?? errors.stamp}
            hint={`Letters, digits, - and _. The run is saved as ${f.name || "<name>"}-${f.stamp}.`}>
            <input className="input" value={f.name} placeholder="act-cubes" spellCheck={false} onChange={(e) => set({ name: e.target.value })} />
          </Field>
          {f.where === "cluster" && (
            <Field label="Parts" error={errors.parts}
              hint={`Each part is one job of up to ${partTime}, the time per part in the settings. A longer run needs more parts, each resuming the last. Up to ${MAX_PARTS}.`}>
              <input className="input num" type="number" min={1} max={MAX_PARTS} value={f.parts} onChange={(e) => set({ parts: num(e.target.value) })} />
            </Field>
          )}
          <div className="field train-wide">
            <label className="switch">
              <input type="checkbox" checked={f.push} onChange={(e) => set({ push: e.target.checked })} />
              <span>Push the trained policy to the Hugging Face Hub</span>
            </label>
            {f.push && (
              <Field label="Hub id for the policy" error={errors.repo_id}
                hint="LeRobot pushes once, after the last step, with the token on the machine that trains.">
                <input className="input" value={f.repo_id} placeholder="Parv-09/act_cubes" spellCheck={false}
                  onChange={(e) => set({ repo_id: e.target.value })} />
              </Field>
            )}
            {!f.push && errors.repo_id && <span className="field-hint text-danger">{errors.repo_id}</span>}
            <label className="switch">
              <input type="checkbox" checked={f.wandb} onChange={(e) => set({ wandb: e.target.checked })} />
              <span>Log to Weights and Biases (needs a W&B login where it trains)</span>
            </label>
          </div>
        </fieldset>
      </div>
    </section>
  );
}

const NO_ERRORS: Record<string, string> = {};
const num = (v: string) => (v === "" ? 0 : Number(v));

function extrasHint(extra: string | null, missing: string[]): string {
  const ex = extra ? `Install with LeRobot's ${extra} extra.` : "Needs no extra beyond LeRobot's training packages.";
  return missing.length ? `${ex} Missing on this Mac: ${missing.join(", ")}.` : ex;
}

function Field({ label, hint, error, wide, children }: { label: string; hint?: string; error?: string; wide?: boolean; children: ReactNode }) {
  return (
    <label className={`field ${wide ? "train-wide" : ""}`}>
      <span className="field-label">{label}</span>
      {children}
      {error ? <span className="field-hint text-danger">{error}</span> : hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

// -- what will run --------------------------------------------------------------------------------------
function ScriptPanel() {
  const p = useTrain((s) => s.preview);
  const where = useTrain((s) => s.form.where);
  const blank = useTrain((s) => !s.form.dataset || !s.form.name);
  if (!p?.ok) {
    return (
      <section className="panel">
        <div className="panel-head"><h2 className="panel-title">{where === "cluster" ? "Job script" : "Command"}</h2></div>
        <p className="empty">{!p || blank ? "Fill in the dataset and the run name to see what will run." : "Fix the fields marked in red to see what will run."}</p>
      </section>
    );
  }
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">{where === "cluster" ? "Job script" : "Command"}</h2>
          <p className="panel-sub">
            {where === "cluster" ? `Written to ${p.run_dir}/job.sbatch and submitted once per part.`
              : `Runs in the terminal panel. Checkpoints go to ${p.output_dir}.`}
          </p>
        </div>
      </div>
      <div className="panel-body train-script">
        {where === "cluster" && (p.script
          ? <CommandBlock cmd={p.script} wrap={false} />
          : <p className="field-hint">The script needs your cluster user name for its folder. Run Check the cluster once.</p>)}
        {where === "cluster" && <p className="field-label">The lerobot-train command inside it</p>}
        {p.command && <CommandBlock cmd={p.command} />}
      </div>
    </section>
  );
}

function PlanPanel() {
  const f = useTrain((s) => s.form);
  const p = useTrain((s) => s.preview);
  const settings = useTrain((s) => s.settings);
  const limits = useTrain((s) => s.check.limits ?? null);
  const submitting = useTrain((s) => s.submitting || (!!s.preview?.run_id && s.busy.submit.includes(s.preview.run_id)));
  const starting = useTrain((s) => s.starting);
  const control = useStudio((s) => s.control);
  const link = useStudio((s) => s.link);
  const busy = useTerminal((s) => (s.running ? shortCommand(s.running.command) || "a command" : null));
  const [confirm, setConfirm] = useState(false);

  const plan = p?.ok ? p.plan : undefined;
  const cap = limits?.max_submit ?? MAX_PARTS;
  // The partition limit the last check read, if it read this partition.
  const maxTime = limits?.max_time && limits.partition === settings?.partition ? limits.max_time : null;
  const tooLong = !!(maxTime && settings && /^\d+:\d\d:\d\d$/.test(maxTime) && seconds(settings.time) > seconds(maxTime));
  const why = link !== "open" ? "Studio is not connected."
    : !control ? "Take control to start training from this window."
    : !f.dataset || !f.name ? "Fill in the dataset and the run name."
    : !p?.ok ? "Fix the fields marked in red first."
    : f.where === "mac" ? (busy ? `The terminal is still running ${busy}.` : starting ? "Starting." : "")
    : !settings?.remote_user ? "Run Check the cluster first, so Studio knows your user name."
    : tooLong ? `Partition ${settings?.partition} allows at most ${maxTime} per part, and the settings ask for ${settings?.time}. Lower the time per part in Settings.`
    : limits?.free != null && limits.free < f.parts
      ? `The last check found ${limits.free} free submit slot${limits.free === 1 ? "" : "s"} of ${cap}; this run needs ${f.parts}. Studio does not submit past your limit. Check again when jobs finish.`
    : submitting ? "Submitting." : "";

  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">{f.where === "cluster" ? "Submit" : "Run on this Mac"}</h2>
          <p className="panel-sub">{p?.ok && p.run_id ? <span className="ident">{p.run_id}</span> : "The run name and stamp."}</p>
        </div>
      </div>
      <div className="panel-body">
        {f.where === "cluster" && plan && (
          <dl className="kv train-kv">
            <dt>Parts</dt><dd className="num">{plan.parts} of up to {plan.part_time}</dd>
            <dt>GPU time, at most</dt><dd className="num">{plan.total_hours} h</dd>
            <dt>Submit slots used</dt><dd className="num">{plan.parts} of {cap}</dd>
            {limits?.free != null && <><dt>Free at the last check</dt><dd className="num">{limits.free}</dd></>}
          </dl>
        )}
        {f.where === "cluster" && plan?.fit && <FitNote fit={plan.fit} saveFreq={f.save_freq} />}
        {f.where === "cluster" && plan && plan.parts > 1 && (
          <p className="field-hint train-gap">
            Each part is its own job and counts toward your limit. A part waits for the one before it to end, then
            resumes from the newest checkpoint. Queue time between parts is not in the total.
          </p>
        )}
        {f.where === "mac" && (
          <p className="field-hint">Runs lerobot-train on MPS in the terminal panel. Stop it there with Ctrl-C or the Stop button.</p>
        )}
        <div className="form-actions train-gap">
          {why && <span className="field-hint">{why}</span>}
          {f.where === "cluster" ? (
            <button className="btn btn-primary" disabled={!!why} onClick={() => setConfirm(true)}>
              {submitting ? <LoaderCircle className="spin" aria-hidden /> : <Send aria-hidden />} Submit
            </button>
          ) : (
            <button className="btn btn-primary" disabled={!!why} onClick={() => train.startMac()}>
              {starting ? <LoaderCircle className="spin" aria-hidden /> : <Play aria-hidden />} Run on this Mac
            </button>
          )}
        </div>
      </div>
      <Dialog.Root open={confirm} onOpenChange={setConfirm}>
        <Dialog.Portal>
          <Dialog.Overlay className="dialog-overlay" />
          <Dialog.Content className="dialog dialog-wide" aria-describedby="train-submit-desc">
            <Dialog.Title className="dialog-title">Submit {plan?.parts ?? 0} job{plan && plan.parts > 1 ? "s" : ""} to {settings?.host}?</Dialog.Title>
            <div id="train-submit-desc" className="dialog-text">
              <p>
                Training {f.pretrained || f.policy} on {f.dataset} for {f.steps.toLocaleString()} steps. Studio writes
                the script to <span className="ident">{p?.run_dir}/job.sbatch</span> and runs sbatch once for each of
                these jobs, each asking for {settings?.gres} on {settings?.partition} for up to {plan?.part_time}:
              </p>
              <ul className="train-names">{plan?.names.map((n) => <li key={n} className="ident">{n}</li>)}</ul>
              <p>
                That uses {plan?.parts} of your {cap} submit slots
                {limits?.free != null ? `; ${limits.free} were free at the last check` : ""}. If sbatch reports an
                error, Studio stops and does not retry.
              </p>
            </div>
            <div className="dialog-buttons">
              <Dialog.Close asChild><button className="btn" autoFocus>Back</button></Dialog.Close>
              <button className="btn btn-primary" onClick={() => { train.submit(); setConfirm(false); }}>
                <Send aria-hidden /> Submit {plan?.parts ?? 0} job{plan && plan.parts > 1 ? "s" : ""}
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </section>
  );
}

/** Whether one checkpoint fits in one part. WHY it matters: a part that ends before its first checkpoint
 * saves nothing, and the next part would start from the same place. */
function FitNote({ fit, saveFreq }: { fit: Fit; saveFreq: number }) {
  if (fit.rate === null) {
    return (
      <p className="field-hint train-gap">
        Studio has not measured a step rate for this policy and batch size on the cluster yet, so it cannot tell
        whether one checkpoint ({saveFreq.toLocaleString()} steps) fits in one part. If a part saves no
        checkpoint, the job script cancels the parts still waiting and the log says why.
      </p>
    );
  }
  const rate = `${fit.rate.toFixed(2)} steps/s, measured on ${fit.from_run}`;
  if (fit.fits) {
    return <p className="field-hint train-gap">Estimate at {rate}: one checkpoint every {duration(fit.save_s)}, well inside a part of {duration(fit.part_s)}.</p>;
  }
  return (
    <div className="notice tone-warn train-gap" role="status">
      <TriangleAlert className="notice-icon" aria-hidden />
      <div className="train-notice-body">
        <div className="notice-title">One checkpoint may not fit in one part.</div>
        <div className="notice-fix">
          Estimate at {rate}: {saveFreq.toLocaleString()} steps take about {duration(fit.save_s)}, and a part is
          {" "}{duration(fit.part_s)}, some of it spent loading. A part that saves no checkpoint makes no progress.
          Save a checkpoint more often, or use a longer part. This is an estimate from one earlier run.
        </div>
      </div>
    </div>
  );
}

// -- the cluster check ----------------------------------------------------------------------------------
type Icon = ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
const LOOK: Record<CheckStatus, { icon: Icon; tone: string; label: string }> = {
  pass: { icon: CircleCheck, tone: "ok", label: "Passed" },
  warn: { icon: TriangleAlert, tone: "warn", label: "Warning" },
  fail: { icon: CircleX, tone: "danger", label: "Failed" },
  skip: { icon: CircleDashed, tone: "neutral", label: "Skipped" },
};

function ClusterPanel() {
  const check = useTrain((s) => s.check);
  const settings = useTrain((s) => s.settings);
  const control = useStudio((s) => s.control);
  const link = useStudio((s) => s.link);
  const [open, setOpen] = useState(false);
  const why = link !== "open" ? "Studio is not connected." : !control ? "Take control to check the cluster." : "";
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Check the cluster</h2>
          <p className="panel-sub">
            {settings ? <>ssh to <span className="ident">{settings.host}</span>{settings.remote_user ? <> as <span className="ident">{settings.remote_user}</span></> : null}. </> : null}
            Read-only, except that it makes the run folder.
          </p>
        </div>
      </div>
      <div className="panel-body train-check-actions">
        <button className="btn btn-primary btn-sm" disabled={!!why || check.running} title={why || undefined} onClick={() => train.runCheck()}>
          {check.running ? <LoaderCircle className="spin" aria-hidden /> : <Play aria-hidden />} {check.rows.length ? "Check again" : "Check"}
        </button>
        <button className="btn btn-sm" onClick={() => setOpen(true)}><SettingsIcon aria-hidden /> Settings</button>
        {check.at && !check.running && <span className="field-hint">Last run {new Date(check.at * 1000).toLocaleTimeString([], { hour12: false })}</span>}
      </div>
      {check.rows.length > 0 && (
        <ul className={`check-list ${check.running ? "is-running" : ""}`}>
          {check.rows.map((r) => <ClusterRow key={r.id} r={r} />)}
        </ul>
      )}
      <SettingsDialog open={open} onOpenChange={setOpen} />
    </section>
  );
}

function ClusterRow({ r }: { r: CheckRow }) {
  const { icon: I, tone, label } = LOOK[r.status];
  return (
    <li className={`check-row tone-${tone} is-${r.status}`}>
      <I className="check-icon" aria-hidden />
      <div className="check-text">
        <div className="check-title">{r.title}<span className="sr-only">: {label}</span></div>
        <div className="check-detail">{r.detail}</div>
        {(r.status === "fail" || r.status === "warn") && r.fix && (
          <div className="check-fix"><CornerDownRight aria-hidden /><span>{r.fix}</span></div>
        )}
        {r.runs.map((x, i) => <CommandOutput key={i} cmd={x.cmd} output={x.output} rc={x.rc} />)}
      </div>
    </li>
  );
}

const SETTING_FIELDS: Array<{ key: keyof TrainSettings; label: string; hint: string; wide?: boolean }> = [
  { key: "host", label: "ssh host", hint: "The alias in ~/.ssh/config. ssh must work with no password prompt." },
  { key: "remote_base", label: "Run folder on the cluster", hint: "Empty: /scratch/<your user>/phi-studio." },
  { key: "partition", label: "Partition", hint: "Where jobs run." },
  { key: "test_partition", label: "Test partition", hint: "Used to test the script when the partition above refuses even a test because you are at your submit limit." },
  { key: "gres", label: "GPUs", hint: "Such as gpu:1. Explorer's notes advise not pinning a GPU type." },
  { key: "time", label: "Time per part", hint: "At least 00:10:00, and no more than the partition's own limit, which Check the cluster reads and Submit checks again." },
  { key: "cpus", label: "CPUs", hint: "Also the number of data loader workers." },
  { key: "mem", label: "Memory", hint: "Such as 64G." },
  { key: "exclude", label: "Nodes to avoid", hint: "Comma separated. The default skips the 16 GB GPU and the slow hosts.", wide: true },
];

function SettingsDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const settings = useTrain((s) => s.settings);
  const defaults = useTrain((s) => s.defaults);
  const errors = useTrain((s) => s.settingsErrors);
  const saved = useTrain((s) => s.settingsSaved);
  const control = useStudio((s) => s.control);
  const [draft, setDraft] = useState<TrainSettings | null>(null);
  const opened = useRef(0);

  useEffect(() => { if (open && settings) { setDraft(settings); opened.current = Date.now(); } }, [open]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (open && saved > opened.current) onOpenChange(false); }, [saved]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!draft) return null;
  const set = (k: keyof TrainSettings, v: string) => setDraft({ ...draft, [k]: k === "cpus" ? num(v) : v });
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="dialog dialog-wide train-settings" aria-describedby="train-settings-desc">
          <Dialog.Title className="dialog-title">Cluster settings</Dialog.Title>
          <p id="train-settings-desc" className="dialog-text">
            Studio builds every remote command from these, after checking each against a pattern.
            {draft.remote_user ? <> Your cluster user, <span className="ident">{draft.remote_user}</span>, comes from the last check.</> : null}
          </p>
          <div className="train-settings-grid">
            {SETTING_FIELDS.map(({ key, label, hint, wide }) => (
              <Field key={key} label={label} hint={hint} error={errors[key]} wide={wide}>
                <input className="input" value={String(draft[key])} spellCheck={false} onChange={(e) => set(key, e.target.value)} />
              </Field>
            ))}
            <Field label="Environment setup" error={errors.env} wide
              hint="Lines that run before training: module load, source activate, source <file> or export NAME=value. Nothing else is allowed. The default is the setup earlier phi runs used.">
              <textarea className="textarea mono train-env" value={draft.env} spellCheck={false} rows={8} onChange={(e) => set("env", e.target.value)} />
            </Field>
          </div>
          {errors.form && <p className="field-hint text-danger">{errors.form}</p>}
          <div className="dialog-buttons">
            <button className="btn btn-ghost" disabled={!defaults} onClick={() => defaults && setDraft({ ...defaults, remote_user: draft.remote_user })}>Use defaults</button>
            <Dialog.Close asChild><button className="btn">Cancel</button></Dialog.Close>
            <button className="btn btn-primary" disabled={!control} title={control ? undefined : "Take control to change settings."}
              onClick={() => train.saveSettings(draft)}>Save</button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

// -- runs --------------------------------------------------------------------------------------------------
function RunsPanel({ selected, onSelect }: { selected: string | null; onSelect: (id: string | null) => void }) {
  const runs = useTrain((s) => s.runs);
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Runs</h2>
          <p className="panel-sub">Only runs Studio started. Studio never touches any other job.</p>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={() => train.refresh()}><RefreshCw aria-hidden /> Refresh</button>
      </div>
      {runs.length === 0 ? <p className="empty">No runs yet.</p> : (
        <div className="table-scroll">
          <table className="table train-runs">
            <thead>
              <tr><th>Run</th><th>Where</th><th>Policy</th><th>Dataset</th><th>State</th><th className="r">Steps</th><th>Started</th></tr>
            </thead>
            <tbody>
              {runs.map((r) => {
                const [word, tone] = stateWords(r.state);
                const step = r.progress?.step;
                return (
                  <tr key={r.id} className={`train-run ${selected === r.id ? "is-selected" : ""}`} aria-selected={selected === r.id}
                    tabIndex={0} onClick={() => onSelect(selected === r.id ? null : r.id)}
                    onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(selected === r.id ? null : r.id); } }}>
                    <td className="ident">{r.id}</td>
                    <td>{r.where === "cluster" ? `Cluster, ${r.jobs.length || r.parts_planned} part${(r.jobs.length || r.parts_planned) > 1 ? "s" : ""}` : "This Mac"}</td>
                    <td className="ident">{r.policy}</td>
                    <td className="ident ellipsis">{r.dataset}</td>
                    <td><span className={`badge tone-${tone}`}><span className="dot" />{word}</span></td>
                    <td className="r num">{step != null ? `${step.toLocaleString()} of ${r.steps.toLocaleString()}` : r.steps.toLocaleString()}</td>
                    <td className="num">{new Date(r.created * 1000).toLocaleString([], { hour12: false, month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function RunDetail({ id, onClose }: { id: string; onClose: () => void }) {
  const run = useTrain((s) => s.runs.find((r) => r.id === id) ?? null);
  const log = useTrain((s) => s.logs[id] ?? null);
  const control = useStudio((s) => s.control);
  const link = useStudio((s) => s.link);
  const changing = useTrain((s) => (s.busy.submit.includes(id) ? "Submitting." : s.busy.resubmit.includes(id) ? "Submitting the remaining parts."
    : s.busy.find.includes(id) ? "Looking for its jobs." : s.busy.cancel.includes(id) ? "Cancelling." : ""));
  const fetching = useTrain((s) => s.busy.fetch.includes(id));
  const [confirm, setConfirm] = useState(false);
  const [finish, setFinish] = useState(false);
  const tailRef = useRef<HTMLPreElement>(null);
  const state = run?.state ?? "";
  // WHY poll a completed run until its log has ended: the last piece of the log is read after the job ends.
  const live = !ended(state) || (state === "COMPLETED" && !log?.ended);

  // The log and chart: now, then every 5 s while this run is open and still going.
  useEffect(() => {
    if (link !== "open") return;
    train.poll(id);
    if (!live) return;
    const t = window.setInterval(() => { if (!document.hidden) train.poll(id); }, 5000);
    return () => window.clearInterval(t);
  }, [id, link, live]);
  useEffect(() => { const el = tailRef.current; if (el) el.scrollTop = el.scrollHeight; }, [log?.tail]);

  if (!run) return null;
  const [word, tone] = stateWords(run.state);
  const p = log ?? run.progress;
  const done = p ? Math.min(1, p.total ? p.step / p.total : 0) : 0;
  const liveParts = run.jobs.filter((j) => !ended(j.state));
  const why = !control ? "Take control to act on this run." : changing;
  const remaining = run.error && run.remaining_from != null
    ? Array.from({ length: run.parts_planned - run.remaining_from + 1 }, (_, i) => run.remaining_from! + i) : [];
  return (
    <section className="panel train-detail">
      <div className="panel-head">
        <div>
          <h2 className="panel-title"><span className="ident">{run.id}</span></h2>
          <p className="panel-sub">
            {run.policy} on <span className="ident">{run.dataset}</span>, batch {run.batch_size}, checkpoint every {run.save_freq.toLocaleString()} steps
          </p>
        </div>
        <div className="train-detail-actions">
          <span className={`badge tone-${tone}`}><span className="dot" />{word}</span>
          <button className="btn btn-ghost btn-sm btn-icon" aria-label="Close run" onClick={onClose}><X aria-hidden /></button>
        </div>
      </div>
      <div className="panel-body">
        <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={run.steps} aria-valuenow={p?.step ?? 0}
          aria-label="Training steps done"><span style={{ width: `${(done * 100).toFixed(1)}%` }} /></div>
        <div className="progress-label">
          <span className="num">{p ? `${p.step.toLocaleString()} of ${(p.total || run.steps).toLocaleString()} steps` : "No log read yet"}</span>
          <span className="num muted">
            {p?.ended ? "Training ended" : p?.rate ? `${p.rate.toFixed(2)} steps/s, about ${duration(p.eta_s)} left at this rate${run.where === "cluster" ? ", not counting queue time" : ""}` : ""}
          </span>
        </div>
        {log && log.points.length > 0 ? (
          <div className="train-charts">
            <LineChart points={log.points} field="loss" title="Loss" />
            <LineChart points={log.points} field="lr" title="Learning rate" />
          </div>
        ) : (
          <p className="field-hint train-gap">The charts fill in once lerobot-train logs its first steps, every {run.log_freq} steps.</p>
        )}

        {run.where === "cluster" && run.jobs.length > 0 && (
          <div className="table-scroll train-gap">
            <table className="table">
              <thead><tr><th>Part</th><th>Job id</th><th>State</th><th>Time used</th><th>SLURM's reason it waits</th><th>Exit code</th></tr></thead>
              <tbody>
                {run.jobs.map((j) => {
                  const [w, t] = stateWords(j.state);
                  return (
                    <tr key={j.id}>
                      <td className="num">{j.part}</td><td className="ident">{j.id}</td>
                      <td><span className={`badge tone-${t}`}><span className="dot" />{w}</span></td>
                      <td className="num">{j.elapsed || ""}</td><td className="ident">{j.reason || ""}</td><td className="num">{j.exit || ""}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        {p?.stalled && (
          <div className="notice tone-warn train-gap" role="status">
            <TriangleAlert className="notice-icon" aria-hidden />
            <div className="train-notice-body">
              <div className="notice-title">A part ended without saving a new checkpoint.</div>
              <div className="notice-fix">
                The next part would have started from the same place, so the job script cancelled the parts still
                waiting. Save a checkpoint more often, or use a longer part, then submit again.
              </div>
            </div>
          </div>
        )}
        {run.error && (
          <div className="notice tone-danger train-gap">
            <CircleX className="notice-icon" aria-hidden />
            <div className="train-notice-body">
              <div className="notice-title">sbatch failed for part {run.error.part}. Studio did not retry.</div>
              <div className="notice-fix">
                On Explorer a failed sbatch can still create the job, minutes later. Wait 3 minutes, then Look for it.
                {remaining.length > 0 && " Then you can submit the parts this run is missing; Studio looks for them again first."}
              </div>
              <CommandOutput cmd={run.error.cmd} output={run.error.output} />
              {remaining.length > 0 && (
                <div className="form-actions train-gap">
                  <button className="btn btn-sm" disabled={!!why} onClick={() => setFinish(true)}>
                    <Send aria-hidden /> Submit the remaining parts
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {log && log.tail.length > 0 && (
          <details className="train-io train-gap" open>
            <summary>Log, last {Math.min(log.tail.length, 200)} lines</summary>
            <pre ref={tailRef} className="train-pre train-tail"><code>{log.tail.join("\n")}</code></pre>
          </details>
        )}

        <div className="train-detail-foot">
          {run.where === "cluster" ? (
            <>
              <div className="train-where-to">
                {run.push ? <>LeRobot pushes the policy to <span className="ident">{run.repo_id}</span> after the last step.</>
                  : <>Checkpoints stay in <span className="ident">{run.remote_dir}/train</span> until you fetch one.</>}
                {run.fetched && <> Fetched to <span className="ident">{run.fetched.path}</span>.</>}
              </div>
              <div className="form-actions">
                {why && <span className="field-hint">{why}</span>}
                <button className="btn btn-sm" disabled={!!why || !run.remote_dir} onClick={() => train.find(run.id)}>
                  <Search aria-hidden /> Look for it
                </button>
                <button className="btn btn-sm" disabled={!control || fetching || run.fetch?.state === "running" || !run.remote_dir} onClick={() => train.fetch(run.id)}>
                  <Download aria-hidden /> Fetch newest checkpoint
                </button>
                <button className="btn btn-danger btn-sm" disabled={!!why || liveParts.length === 0} onClick={() => setConfirm(true)}>
                  <Square aria-hidden /> Cancel
                </button>
              </div>
              {run.fetch && <FetchProgress run={run} />}
            </>
          ) : (
            <div className="train-where-to">
              Checkpoints are saved in <span className="ident">{run.local_dir}/train/checkpoints</span>. Stop a run on this Mac
              from the terminal panel.
            </div>
          )}
        </div>
      </div>
      <Dialog.Root open={confirm} onOpenChange={setConfirm}>
        <Dialog.Portal>
          <Dialog.Overlay className="dialog-overlay" />
          <Dialog.Content className="dialog" aria-describedby="train-cancel-desc">
            <Dialog.Title className="dialog-title">Cancel this run?</Dialog.Title>
            <div id="train-cancel-desc" className="dialog-text">
              <p>Studio sends scancel for this run's parts that have not ended, and no other job:</p>
              <ul className="train-names">{liveParts.map((j) => <li key={j.id} className="ident">{j.id} ({j.name})</li>)}</ul>
              <p>Training stops. Checkpoints already saved stay on the cluster.</p>
            </div>
            <div className="dialog-buttons">
              <Dialog.Close asChild><button className="btn" autoFocus>Keep it running</button></Dialog.Close>
              <button className="btn btn-danger" onClick={() => { train.cancel(run.id); setConfirm(false); }}>Cancel {liveParts.length} job{liveParts.length > 1 ? "s" : ""}</button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      <Dialog.Root open={finish} onOpenChange={setFinish}>
        <Dialog.Portal>
          <Dialog.Overlay className="dialog-overlay" />
          <Dialog.Content className="dialog" aria-describedby="train-finish-desc">
            <Dialog.Title className="dialog-title">Submit the remaining parts?</Dialog.Title>
            <div id="train-finish-desc" className="dialog-text">
              <p>
                Studio first looks on {run.host} for jobs with this run's names, since the failed sbatch may have made one,
                and records any it finds. Then it runs sbatch once for each part still missing, with the same script in
                {" "}<span className="ident">{run.remote_dir}</span>, each after the one before it:
              </p>
              <ul className="train-names">{remaining.map((k) => <li key={k} className="ident">{run.id}-p{k}</li>)}</ul>
              <p>Each part resumes from the newest checkpoint. If sbatch reports an error again, Studio stops and does not retry.</p>
            </div>
            <div className="dialog-buttons">
              <Dialog.Close asChild><button className="btn" autoFocus>Back</button></Dialog.Close>
              <button className="btn btn-primary" onClick={() => { train.resubmit(run.id); setFinish(false); }}>
                <Send aria-hidden /> Submit {remaining.length} part{remaining.length > 1 ? "s" : ""}
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </section>
  );
}

function FetchProgress({ run }: { run: Run }) {
  const f = run.fetch!;
  const frac = f.total ? Math.min(1, f.bytes / f.total) : 0;
  return (
    <div className="train-gap">
      <div className="progress" role="progressbar" aria-label="Checkpoint copied" aria-valuemin={0} aria-valuemax={f.total} aria-valuenow={f.bytes}>
        <span style={{ width: `${(frac * 100).toFixed(1)}%` }} />
      </div>
      <div className="progress-label">
        <span className="num">{f.state === "done" ? `Copied ${bytes(f.bytes)}` : f.state === "error" ? "The copy failed" : `Copying ${bytes(f.bytes)} of ${bytes(f.total)}`}</span>
        <span className="ident muted">{f.path}</span>
      </div>
      {f.message && <p className={`field-hint ${f.state === "error" ? "text-danger" : ""}`}>{f.message}</p>}
    </div>
  );
}

// -- charts: one measure per chart, one y axis, SVG drawn here ------------------------------------------------
const H = 200;
const M = { l: 64, r: 16, t: 12, b: 28 };

function LineChart({ points, field, title }: { points: Point[]; field: "loss" | "lr"; title: string }) {
  const wrap = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(560);
  const [hover, setHover] = useState<number | null>(null);
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(240, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const pts = useMemo(() => points.filter((p) => typeof p[field] === "number" && Number.isFinite(p[field])) as Array<Point & Record<typeof field, number>>, [points, field]);
  const geo = useMemo(() => {
    if (pts.length === 0) return null;
    const xs = pts.map((p) => p.step), ys = pts.map((p) => p[field]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    // WHY a log scale for loss when it spans 20x or more: early steps are 10-100x the later loss, and a linear axis
    // flattens everything after them into the floor.
    const log = field === "loss" && y0 > 0 && y1 / y0 >= 20;
    if (!log && y0 === y1) { y0 -= Math.abs(y0) * 0.1 || 1; y1 += Math.abs(y1) * 0.1 || 1; }
    if (!log && field === "loss" && y0 > 0) y0 = 0;
    const f = (v: number) => (log ? Math.log10(v) : v);
    const fy0 = f(y0), fy1 = f(y1) === f(y0) ? f(y0) + 1 : f(y1);
    const iw = w - M.l - M.r, ih = H - M.t - M.b;
    const sx = (v: number) => M.l + (x1 === x0 ? iw / 2 : ((v - x0) / (x1 - x0)) * iw);
    const sy = (v: number) => M.t + ih - ((f(v) - fy0) / (fy1 - fy0)) * ih;
    const yt = log ? logTicks(y0, y1) : niceTicks(y0, y1, 4);
    const xt = niceTicks(x0, x1, Math.max(2, Math.floor(iw / 110)));
    const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.step).toFixed(1)},${sy(p[field]).toFixed(1)}`).join("");
    return { sx, sy, yt, xt, d, log, iw, ih };
  }, [pts, field, w]);

  const fmt = (v: number) => (v === 0 ? "0" : field === "lr" ? v.toExponential(1) : Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3));
  const first = pts[0], last = pts[pts.length - 1];
  const summary = first && last ? `${title} went from ${fmt(first[field])} at step ${first.step} to ${fmt(last[field])} at step ${last.step}.` : `${title}: no values yet.`;

  function onMove(e: MouseEvent<SVGRectElement>) {
    if (!geo || pts.length === 0) return;
    const box = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - box.left + M.l;
    let lo = 0, hi = pts.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (geo.sx(pts[mid].step) < x) lo = mid; else hi = mid; }
    setHover(Math.abs(geo.sx(pts[lo].step) - x) <= Math.abs(geo.sx(pts[hi].step) - x) ? lo : hi);
  }
  const hp = hover !== null ? pts[hover] : null;
  return (
    <figure className="train-chart">
      <figcaption className="train-chart-title">{title}{geo?.log ? <span className="faint"> (log scale)</span> : null}</figcaption>
      <div ref={wrap} className="train-chart-box">
        {geo ? (
          <svg width={w} height={H} role="img" aria-label={summary}>
            {geo.yt.map((v) => (
              <g key={`y${v}`}>
                <line className="chart-grid" x1={M.l} x2={w - M.r} y1={geo.sy(v)} y2={geo.sy(v)} />
                <text className="chart-tick" x={M.l - 8} y={geo.sy(v)} dy="0.32em" textAnchor="end">{fmt(v)}</text>
              </g>
            ))}
            {geo.xt.map((v) => (
              <text key={`x${v}`} className="chart-tick" x={geo.sx(v)} y={H - 8} textAnchor="middle">{stepLabel(v)}</text>
            ))}
            <line className="chart-axis" x1={M.l} x2={w - M.r} y1={H - M.b} y2={H - M.b} />
            <path className="chart-line" d={geo.d} />
            {pts.length === 1 && <circle className="chart-dot" cx={geo.sx(pts[0].step)} cy={geo.sy(pts[0][field])} r={4} />}
            {hp && (
              <g>
                <line className="chart-cross" x1={geo.sx(hp.step)} x2={geo.sx(hp.step)} y1={M.t} y2={H - M.b} />
                <circle className="chart-dot" cx={geo.sx(hp.step)} cy={geo.sy(hp[field])} r={4} />
              </g>
            )}
            <rect className="chart-hit" x={M.l} y={M.t} width={geo.iw} height={geo.ih}
              onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
          </svg>
        ) : <p className="empty">No {title.toLowerCase()} values in the log yet.</p>}
        {hp && geo && (
          <div className="chart-tip" style={{ left: Math.min(w - 170, Math.max(0, geo.sx(hp.step) + 10)), top: M.t }}>
            <div className="num">Step {hp.step.toLocaleString()}{hp.sure === false ? " (about)" : ""}</div>
            <div className="num chart-tip-value">{title} {fmt(hp[field])}</div>
          </div>
        )}
      </div>
    </figure>
  );
}

function stepLabel(v: number): string {
  return v >= 10_000 ? `${(v / 1000).toLocaleString()}K` : v.toLocaleString();
}

function niceTicks(a: number, b: number, n: number): number[] {
  if (a === b) return [a];
  const raw = (b - a) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

function logTicks(a: number, b: number): number[] {
  const out: number[] = [];
  for (let e = Math.ceil(Math.log10(a)); e <= Math.floor(Math.log10(b)); e++) out.push(10 ** e);
  return out.length >= 2 ? out : [a, b];
}
