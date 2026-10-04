import { ArrowLeft, Download, ExternalLink, Heart, LogIn, Play, RefreshCw, Search, TriangleAlert, X } from "lucide-react";
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { CommandBlock } from "../components/CommandBlock";
import {
  asRepoId, fmtBytes, fmtCount, fmtDate, hub, imageSize, useHub,
  type Detail, type LocalModel, type ModelInfo, type Progress as Job, type RigCamera, type SearchRow,
} from "../lib/hub";
import { useStudio } from "../lib/studio";
import "../styles/models.css";

// What the person last did here, kept while they move between pages.
const memory = {
  query: "",
  view: null as View,
  run: { task: "Pick up the red cube and place it in the box", duration: 60, strategy: "base", episodes: 10, dataset: "", upload: false },
};
type View = { kind: "hub"; repoId: string } | { kind: "run"; repoId: string; revision: string } | null;

const IMG = "observation.images.";
const camName = (key: string) => key.startsWith(IMG) ? key.slice(IMG.length) : key;

/** Text from the server with `code` spans, shown as code: camera keys, file names, commands. */
function Rich({ text }: { text: string }) {
  return <>{text.split("`").map((part, i) => (i % 2 ? <code key={i} className="ident">{part}</code> : <Fragment key={i}>{part}</Fragment>))}</>;
}

export function Models() {
  const [view, setViewState] = useState<View>(memory.view);
  const setView = (v: View) => { memory.view = v; setViewState(v); };
  useEffect(() => hub.open(), []);
  // A model the page remembers is read again, so the fit is checked against today's rig.
  useEffect(() => { if (memory.view?.kind === "hub") hub.inspect(memory.view.repoId, true); }, []);

  const open = (repoId: string) => { setView({ kind: "hub", repoId }); hub.inspect(repoId); };
  const run = (m: { repo_id: string; revision: string }) => setView({ kind: "run", repoId: m.repo_id, revision: m.revision });

  return (
    <div className="page models">
      <HubStatus />
      <div className="mdl-grid">
        <div className="col">
          <FindModel onOpen={open} selected={view?.repoId ?? null} />
          <OnThisMac onRun={run} onOpen={open} selected={view?.kind === "run" ? view : null} />
        </div>
        <div className="col">
          {view?.kind === "hub" && <ModelDetail repoId={view.repoId} onRun={run} onClose={() => { setView(null); hub.closeDetail(); }} />}
          {view?.kind === "run" && <RunOnArms repoId={view.repoId} revision={view.revision} onBack={() => open(view.repoId)} onClose={() => setView(null)} />}
          {!view && (
            <section className="panel">
              <p className="empty">Search for a policy, paste a Hugging Face link, or pick a model on this Mac.</p>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}

// -- Hub status ------------------------------------------------------------------------------------

function HubStatus() {
  const who = useHub((s) => s.who);
  const loading = useHub((s) => s.whoLoading);
  const signedIn = !!who?.user;
  return (
    <section className="panel mdl-status">
      <div className="mdl-status-row">
        <LogIn aria-hidden className="mdl-status-icon" />
        <div className="mdl-status-text">
          {!who ? <span className="muted">Checking your Hugging Face login...</span>
            : signedIn ? (
              <span>Signed in to Hugging Face as <b>{who.user}</b>
                {who.orgs.length > 0 && <span className="muted">, in {who.orgs.join(", ")}</span>}</span>
            ) : <span><b>Not signed in to Hugging Face.</b> <span className="muted">Public models work without it; private and gated ones need it.</span></span>}
          {who?.error && !signedIn && <span className="mdl-status-err"><Rich text={who.error} /></span>}
        </div>
        <button className="btn btn-sm" onClick={() => hub.checkLogin()} disabled={loading} title="Ask the Hub again">
          <RefreshCw aria-hidden className={loading ? "spin" : ""} /> {signedIn ? "Refresh" : "Check again"}
        </button>
      </div>
      {who && !signedIn && (
        <div className="mdl-login">
          <CommandBlock cmd={who.login_cmd} wrap={false} run />
          <p className="field-hint">You type the token into your own shell, and hf hides it. Studio never asks for, shows or stores it.</p>
        </div>
      )}
    </section>
  );
}

// -- search ----------------------------------------------------------------------------------------

function FindModel({ onOpen, selected }: { onOpen: (id: string) => void; selected: string | null }) {
  const [text, setText] = useState(memory.query);
  const search = useHub((s) => s.search);
  const searching = useHub((s) => s.searching);
  const timer = useRef<number | null>(null);

  const go = (q: string) => { if (timer.current) window.clearTimeout(timer.current); hub.search(q.trim()); };
  // hub.open() runs the first search (the most popular policies) once the link is up.
  useEffect(() => { hub.query = memory.query.trim(); }, []);
  useEffect(() => () => { if (timer.current) window.clearTimeout(timer.current); }, []);

  const change = (v: string) => {
    setText(v);
    memory.query = v;
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => { timer.current = null; hub.search(v.trim()); }, 300);
  };

  const results = search?.results ?? [];
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Find a model</h2>
          <p className="panel-sub">LeRobot policies on Hugging Face, most downloaded first</p>
        </div>
      </div>
      <div className="panel-body mdl-find">
        <label className="mdl-search">
          <Search aria-hidden />
          <span className="sr-only">Search words, a model id or a Hugging Face link</span>
          <input className="input" value={text} spellCheck={false} autoComplete="off"
            placeholder="Search words, owner/name, or a Hugging Face link"
            onChange={(e) => change(e.target.value)}
            onPaste={(e) => {
              const id = asRepoId(e.clipboardData.getData("text"));
              if (id) { e.preventDefault(); setText(id); memory.query = id; onOpen(id); go(id); }
            }}
            onKeyDown={(e) => {
              if (e.key !== "Enter") return;
              const id = asRepoId(text);
              if (id) onOpen(id);
              go(text);
            }} />
        </label>
        {searching !== null && <p className="field-hint">Searching...</p>}
        {search?.error && <p className="mdl-error"><Rich text={search.error} /></p>}
      </div>
      {search && !search.error && results.length === 0 && (
        <p className="empty">No LeRobot policy on the Hub matches "{search.query}".</p>
      )}
      {results.length > 0 && (
        <ul className="mdl-results" aria-label="Search results">
          {results.map((r) => <ResultRow key={r.repo_id} r={r} active={r.repo_id === selected} onOpen={onOpen} />)}
        </ul>
      )}
    </section>
  );
}

function ResultRow({ r, active, onOpen }: { r: SearchRow; active: boolean; onOpen: (id: string) => void }) {
  return (
    <li>
      <button className={`mdl-result ${active ? "is-active" : ""}`} onClick={() => onOpen(r.repo_id)}>
        <span className="mdl-result-name">{r.repo_id}</span>
        <span className="mdl-result-meta">
          {r.policy_type ? <span className="badge tone-info">{r.policy_type}</span> : <span className="faint">type not stated</span>}
          {r.gated && <span className="badge tone-warn">Gated</span>}
          <span title="Downloads"><Download aria-hidden /> {fmtCount(r.downloads)}</span>
          <span title="Likes"><Heart aria-hidden /> {fmtCount(r.likes)}</span>
          <span className="faint">Updated {fmtDate(r.last_modified)}</span>
        </span>
      </button>
    </li>
  );
}

// -- detail ----------------------------------------------------------------------------------------

function ModelDetail({ repoId, onRun, onClose }: { repoId: string; onRun: (m: { repo_id: string; revision: string }) => void; onClose: () => void }) {
  const detail = useHub((s) => s.detail);
  const inspecting = useHub((s) => s.inspecting);
  const local = useHub((s) => s.local);
  const loading = inspecting !== null || !detail;
  return (
    <section className="panel">
      <div className="panel-head">
        <div className="mdl-title">
          <h2 className="panel-title">{detail?.info?.repo_id ?? repoId}</h2>
          <a className="panel-sub" href={`https://huggingface.co/${detail?.info?.repo_id ?? repoId}`} target="_blank" rel="noreferrer noopener">
            Open on Hugging Face <ExternalLink aria-hidden className="mdl-ext" />
          </a>
        </div>
        <button className="btn btn-ghost btn-sm btn-icon" aria-label="Close" title="Close" onClick={onClose}><X aria-hidden /></button>
      </div>
      {loading ? <p className="empty">Reading the model's details from the Hub...</p>
        : detail.error ? <div className="panel-body"><p className="mdl-error"><Rich text={detail.error} /></p></div>
        : detail.info && <DetailBody d={detail} info={detail.info} local={local} onRun={onRun} />}
    </section>
  );
}

function DetailBody({ d, info, local, onRun }: { d: Detail; info: ModelInfo; local: LocalModel[] | null; onRun: (m: { repo_id: string; revision: string }) => void }) {
  const here = local?.find((m) => m.repo_id === info.repo_id && m.revision === info.revision && m.has_weights);
  // WHY: the cache keeps each downloaded commit, so an older one can sit here while the Hub has moved on.
  const older = here ? undefined : local?.find((m) => m.repo_id === info.repo_id && m.has_weights);
  const access = [info.private && "Private", info.gated && `Gated (${info.gated === "auto" ? "approved automatically" : "approved by the owner"})`]
    .filter(Boolean).join(", ") || "Public";
  return (
    <div className="panel-body mdl-detail">
      <dl className="mdl-kv">
        <dt>Policy type</dt><dd><code className="ident">{info.policy_type}</code></dd>
        <dt>Reads</dt>
        <dd>
          <ul className="mdl-inputs">
            {info.state_shape && <li>Joint positions, {info.state_shape.join("x")} numbers</li>}
            {Object.entries(info.cameras).map(([k, shape]) => (
              <li key={k}>Camera <code className="ident">{camName(k)}</code>, {imageSize(shape)}</li>
            ))}
            {Object.entries(info.input_features).filter(([, f]) => f.type === "ENV").map(([k]) => (
              <li key={k}>Simulator state <code className="ident">{k}</code></li>
            ))}
            {Object.keys(info.input_features).length === 0 && <li className="faint">Not listed in its config</li>}
          </ul>
        </dd>
        <dt>Outputs</dt><dd>{info.action_shape ? `Joint targets, ${info.action_shape.join("x")} numbers` : <span className="faint">Not listed in its config</span>}</dd>
        <dt>Size on disk</dt><dd>{fmtBytes(info.total_size)}{!info.has_weights && <span className="text-warn">, no weights file</span>}</dd>
        <dt>Licence</dt><dd>{info.card.license ?? <span className="faint">Not stated</span>}</dd>
        <dt>Last update</dt><dd>{fmtDate(info.last_modified)}</dd>
        <dt>Access</dt><dd>{access}</dd>
        {info.card.datasets.length > 0 && <><dt>Trained on</dt><dd>{info.card.datasets.join(", ")}</dd></>}
      </dl>
      <FitList title="Fit with this rig" fit={d} rig={d.rig} />
      <DownloadBox info={info} here={here} older={older} onRun={onRun} />
    </div>
  );
}

function FitList({ title, fit, rig }: { title: string; fit: { problems?: string[]; warnings?: string[]; checked?: boolean }; rig?: { error: string | null } | null }) {
  const problems = fit.problems ?? [], warnings = fit.warnings ?? [];
  return (
    <div className="mdl-fit">
      <h3 className="mdl-h3">{title}</h3>
      {rig?.error ? <p className="mdl-error"><Rich text={rig.error} /></p>
        : fit.checked === false ? <p className="faint">Studio could not read the rig, so it did not check the fit.</p>
        : problems.length === 0 ? <p className="text-ok">Fits this rig{warnings.length ? ", with notes below" : ""}.</p> : null}
      {problems.length > 0 && (
        <ul className="mdl-lines tone-danger">{problems.map((p) => <li key={p}><TriangleAlert aria-hidden /><span><Rich text={p} /></span></li>)}</ul>
      )}
      {warnings.length > 0 && (
        <ul className="mdl-lines tone-neutral">{warnings.map((w) => <li key={w}><span className="dot" aria-hidden /><span><Rich text={w} /></span></li>)}</ul>
      )}
    </div>
  );
}

function DownloadBox({ info, here, older, onRun }: { info: ModelInfo; here?: LocalModel; older?: LocalModel; onRun: (m: { repo_id: string; revision: string }) => void }) {
  const job = useHub((s) => s.download);
  const control = useStudio((s) => s.control);
  const mine = job && job.repo_id === info.repo_id && (job.revision === info.revision || job.revision === null);
  const busy = job && (job.state === "running" || job.state === "cancelling");
  if (here) {
    return (
      <div className="mdl-download">
        <p className="text-ok">On this Mac, {fmtBytes(here.size)}.</p>
        <button className="btn btn-primary" onClick={() => onRun(here)}><Play aria-hidden /> Run on the arms</button>
      </div>
    );
  }
  const why = !control ? "Take control to download from this window."
    : busy && !mine ? `Another download is running: ${job.repo_id}.`
    : !info.has_weights ? "This repo has no model.safetensors, so LeRobot cannot load it." : "";
  return (
    <div className="mdl-download">
      {older && (
        <p className="field-hint">
          An older version, <code className="ident">{older.revision.slice(0, 7)}</code>, downloaded {fmtDate(older.modified)}, is on this Mac.{" "}
          <button className="mdl-link" onClick={() => onRun(older)}>Run that one</button>, or download this version.
        </p>
      )}
      {mine && job ? <DownloadProgress job={job} /> : (
        <>
          <button className="btn btn-primary" disabled={!!why || !!busy} title={why || undefined} onClick={() => hub.download(info.repo_id, info.revision)}>
            <Download aria-hidden /> Download {fmtBytes(info.total_size)}
          </button>
          {why && <p className="field-hint">{why}</p>}
          <p className="field-hint">Into the Hugging Face cache, where LeRobot looks. Studio checks every file's size and checksum.</p>
        </>
      )}
    </div>
  );
}

function DownloadProgress({ job }: { job: Job }) {
  const control = useStudio((s) => s.control);
  const total = job.total ?? 0;
  const frac = total > 0 ? Math.min(1, job.done / total) : 0;
  const live = job.state === "running" || job.state === "cancelling";
  const label = job.state === "cancelling" ? "Cancelling..."
    : job.state === "running" ? (total > 0 && job.done >= total ? "Checking every file..." : "Downloading...")
    : job.state === "done" ? "Downloaded and checked." : job.message ?? job.state;
  return (
    <div className="mdl-progress">
      <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={total} aria-valuenow={job.done} aria-label={`Download of ${job.repo_id}`}>
        <span style={{ width: `${frac * 100}%` }} />
      </div>
      <div className="progress-label num">
        <span className={job.state === "error" ? "mdl-text-danger" : job.state === "done" ? "text-ok" : ""}><Rich text={label} /></span>
        <span className="faint">{fmtBytes(job.done)} of {fmtBytes(job.total)}</span>
      </div>
      {live && (
        <button className="btn btn-sm" disabled={!control || job.state === "cancelling"} title={control ? undefined : "Take control to cancel"} onClick={() => hub.cancel()}>
          <X aria-hidden /> Cancel
        </button>
      )}
    </div>
  );
}

// -- on this Mac -----------------------------------------------------------------------------------

function OnThisMac({ onRun, onOpen, selected }: { onRun: (m: LocalModel) => void; onOpen: (id: string) => void; selected: { repoId: string; revision: string } | null }) {
  const local = useHub((s) => s.local);
  const job = useHub((s) => s.download);
  const busy = job && (job.state === "running" || job.state === "cancelling");
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">On this Mac</h2>
          <p className="panel-sub">LeRobot policies in the Hugging Face cache, newest first</p>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={() => hub.loadLocal()}><RefreshCw aria-hidden /> Refresh</button>
      </div>
      {busy && job && <div className="panel-body mdl-busy"><span className="muted">Downloading {job.repo_id}</span><DownloadProgress job={job} /></div>}
      {!local ? <p className="empty">Looking in the cache...</p>
        : local.length === 0 ? <p className="empty">No LeRobot policy is downloaded yet.</p>
        : (
          <div className="table-scroll">
            <table className="table mdl-table">
              <thead><tr><th>Model</th><th>Type</th><th className="r">Size</th><th>Downloaded</th><th /></tr></thead>
              <tbody>
                {local.map((m) => (
                  <tr key={m.path} className={selected?.repoId === m.repo_id && selected.revision === m.revision ? "is-active" : ""}>
                    <td className="mdl-name-td">
                      <div className="mdl-cell-name">
                        <button className="mdl-link" onClick={() => onOpen(m.repo_id)} title="Show its details from the Hub">{m.repo_id}</button>
                        <span className="faint mono">{m.revision.slice(0, 7)}</span>
                      </div>
                    </td>
                    <td><code className="ident">{m.policy_type}</code></td>
                    <td className="r num">{fmtBytes(m.size)}</td>
                    <td>{fmtDate(m.modified)}</td>
                    <td className="r">
                      {m.has_weights
                        ? <button className="btn btn-sm" onClick={() => onRun(m)}><Play aria-hidden /> Run on the arms</button>
                        : <span className="text-warn" title="config.json is here but model.safetensors is not">Weights missing</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
    </section>
  );
}

// -- run on the arms -------------------------------------------------------------------------------

// lerobot-rollout's strategies (rollout/configs.py:53-189), in plain words.
const STRATEGIES: { id: string; label: string; what: string }[] = [
  { id: "base", label: "Just run it", what: "Runs the model for the time below and records nothing." },
  { id: "sentry", label: "Run and record everything", what: "Records the whole run into a dataset, saved in pieces as it goes." },
  { id: "highlight", label: "Run, and save moments when I press s", what: "Keeps the last 10 seconds in memory. Press s to save them and keep recording, s again to stop. With upload on, h uploads." },
  { id: "episodic", label: "Record episodes, like Record a dataset", what: "One episode at a time, with a reset between. Right arrow ends an episode, left arrow redoes it, Esc stops. The arm returns to its start pose between episodes." },
  { id: "dagger", label: "Run, and let me correct it with the leader", what: "Space pauses the model. Tab starts a correction, where you drive the follower with your leader arm. Each correction is saved as an episode." },
];

function RunOnArms({ repoId, revision, onBack, onClose }: { repoId: string; revision: string; onBack: () => void; onClose: () => void }) {
  const model = useHub((s) => s.local?.find((m) => m.repo_id === repoId && m.revision === revision) ?? null);
  const rig = useHub((s) => s.rig);
  const who = useHub((s) => s.who?.user ?? null);
  const reply = useHub((s) => s.rollout);
  const [form, setForm] = useState(memory.run);
  const set = (p: Partial<typeof form>) => { const f = { ...form, ...p }; memory.run = f; setForm(f); };
  const strategy = STRATEGIES.find((s) => s.id === form.strategy) ?? STRATEGIES[0];
  const records = strategy.id !== "base";

  // model camera key -> rig camera key; starts from the server's proposal
  const [cams, setCams] = useState<Record<string, string>>({});
  const modelCams = useMemo(() => Object.keys(model?.cameras ?? {}), [model]);
  const rigSig = (rig?.cameras ?? []).map((c) => `${c.key}:${c.usable}`).join(",");
  // WHY keyed on the path and the rig's cameras, not the objects: a fresh model list after a download
  // must not throw away a mapping the person is editing.
  useEffect(() => {
    if (!model) return;
    const back: Record<string, string> = {};
    for (const [r, m] of Object.entries(model.rename_map)) back[m] = r;
    const usable = new Set((rig?.cameras ?? []).filter((c) => c.usable).map((c) => c.key));
    setCams(Object.fromEntries(modelCams.map((k) => [k, back[k] ?? (usable.has(k) ? k : "")])));
  }, [model?.path, rigSig]); // eslint-disable-line react-hooks/exhaustive-deps

  const renameMap = useMemo(() => Object.fromEntries(Object.entries(cams).filter(([m, r]) => r && r !== m).map(([m, r]) => [r, m])), [cams]);
  const rigNames = new Set((rig?.cameras ?? []).map((c) => c.key));
  const needsMap = modelCams.some((k) => !rigNames.has(k)) || Object.keys(renameMap).length > 0;
  const owner = who ?? "<hf_user>";
  const dataset = records ? `${owner}/rollout_${form.dataset.trim()}` : null;

  useEffect(() => {
    if (!model?.has_weights) return;
    const t = window.setTimeout(() => hub.buildRollout({
      repo_id: repoId, revision, task: form.task, duration_s: Number(form.duration), strategy: form.strategy,
      episodes: Math.round(Number(form.episodes)), dataset_repo_id: records && form.dataset.trim() ? dataset : null,
      rename_map: renameMap, upload: form.upload,
    }), 150);
    return () => window.clearTimeout(t);
  }, [repoId, revision, form, renameMap, records, dataset, model?.has_weights]);
  useEffect(() => () => hub.clearRollout(), []);

  if (!model) {
    return (
      <section className="panel">
        <div className="panel-head"><h2 className="panel-title">Run on the arms</h2></div>
        <p className="empty">{repoId} at this revision is not on this Mac. Download it first.</p>
      </section>
    );
  }
  return (
    <section className="panel">
      <div className="panel-head">
        <div className="mdl-title">
          <h2 className="panel-title">Run on the arms</h2>
          <p className="panel-sub">{repoId} <span className="mono">{revision.slice(0, 7)}</span>, <code className="ident">{model.policy_type}</code></p>
        </div>
        <div className="mdl-head-actions">
          <button className="btn btn-ghost btn-sm" onClick={onBack}><ArrowLeft aria-hidden /> Details</button>
          <button className="btn btn-ghost btn-sm btn-icon" aria-label="Close" title="Close" onClick={onClose}><X aria-hidden /></button>
        </div>
      </div>
      <div className="panel-body form mdl-run">
        <label className="field">
          <span className="field-label">What to do</span>
          <select className="select" value={form.strategy} onChange={(e) => set({ strategy: e.target.value })}>
            {STRATEGIES.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
          </select>
          <span className="field-hint">{strategy.what}</span>
        </label>
        <label className="field">
          <span className="field-label">Task</span>
          <textarea className="textarea" rows={2} maxLength={300} value={form.task} onChange={(e) => set({ task: e.target.value })} />
          <span className="field-hint">The instruction the model reads. Models without language input, such as ACT, ignore it.</span>
        </label>
        <div className="mdl-row">
          <label className="field">
            <span className="field-label">{strategy.id === "episodic" ? "Each episode" : "Duration"}</span>
            <div className="input-suffix">
              <input className="input num" type="number" min={1} max={86400} step={1} value={form.duration} onChange={(e) => set({ duration: Number(e.target.value) })} />
              <span>seconds</span>
            </div>
          </label>
          {(strategy.id === "episodic" || strategy.id === "dagger") && (
            <label className="field">
              <span className="field-label">Episodes</span>
              <input className="input num mdl-narrow" type="number" min={1} max={1000} step={1} value={form.episodes} onChange={(e) => set({ episodes: Number(e.target.value) })} />
            </label>
          )}
        </div>
        {records && (
          <div className="field">
            <span className="field-label">Dataset name</span>
            <div className="mdl-dataset">
              <span className="mono">{owner}/rollout_</span>
              <input className="input" value={form.dataset} placeholder="pick_cube" spellCheck={false} aria-label="Dataset name after rollout_"
                onChange={(e) => set({ dataset: e.target.value.replace(/[^\w.-]/g, "") })} />
            </div>
            <span className="field-hint">lerobot-rollout only records into a name that starts with rollout_, and adds the date and time to it.</span>
            <label className="mdl-check">
              <input type="checkbox" checked={form.upload} onChange={(e) => set({ upload: e.target.checked })} />
              Upload the dataset to Hugging Face as it records (public unless your account makes new repos private)
            </label>
          </div>
        )}
        {needsMap && <CameraMap modelCams={model.cameras} rigCams={rig?.cameras ?? []} value={cams} onChange={setCams} />}
        <FitList title="Fit with this rig" fit={reply ?? { problems: model.problems, warnings: model.warnings }} rig={rig} />
        {reply?.error && <p className="mdl-error"><Rich text={reply.error} /></p>}
        {reply?.cmd && (
          <>
            <CommandBlock cmd={reply.cmd} run={(reply.problems ?? []).length === 0} />
            {(reply.problems ?? []).length > 0 && <p className="field-hint">Fix the problems above to run it from here.</p>}
          </>
        )}
        <p className="mdl-safety"><TriangleAlert aria-hidden /> Keep a hand near the follower's power. The arm moves on its own.</p>
      </div>
    </section>
  );
}

function CameraMap({ modelCams, rigCams, value, onChange }: {
  modelCams: Record<string, number[]>; rigCams: RigCamera[]; value: Record<string, string>; onChange: (v: Record<string, string>) => void;
}) {
  const used = new Map(Object.entries(value).filter(([, r]) => r).map(([m, r]) => [r, m]));
  return (
    <div className="field">
      <span className="field-label">Cameras</span>
      <span className="field-hint">The model's camera names differ from your rig's. Pick which rig camera feeds each one.</span>
      <div className="mdl-cammap">
        {Object.entries(modelCams).map(([key, shape]) => (
          <Fragment key={key}>
            <span><code className="ident">{camName(key)}</code> <span className="faint">{imageSize(shape)}</span></span>
            <select className="select" aria-label={`Rig camera for ${camName(key)}`} value={value[key] ?? ""}
              onChange={(e) => onChange({ ...value, [key]: e.target.value })}>
              <option value="">No camera</option>
              {rigCams.map((c) => (
                <option key={c.key} value={c.key} disabled={!c.usable || (used.has(c.key) && used.get(c.key) !== key)}>
                  {c.name}{c.size ? `, ${c.size}` : ""}{!c.usable ? " (no device in robot-config.yaml)" : used.has(c.key) && used.get(c.key) !== key ? " (in use)" : ""}
                </option>
              ))}
            </select>
          </Fragment>
        ))}
      </div>
    </div>
  );
}
