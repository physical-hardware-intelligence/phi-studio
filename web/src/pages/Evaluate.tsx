import * as Dialog from "@radix-ui/react-dialog";
import { Ban, Camera, Copy, Download, Eye, EyeOff, FileText, Flag, Import, Lock, Pencil, Plus, SkipForward, Trash2, Undo2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { ActionBar } from "../components/ActionBar";
import { CameraTile } from "../components/CameraTile";
import { Notices } from "../components/Notices";
import { Preflight, usePreflight } from "../components/Preflight";
import {
  AXES, blankCard, decisionText, draftProblem, milestoneLabel, pctOf, planOf, policyName, progressOf, stageFromKey,
  type Card, type EvalRec, type EvalState, type NextSlot, type PastEval, type Summary, type Trial,
} from "../lib/evalcore";
import { studio, useStudio } from "../lib/studio";

// Evals the way the labs that publish their protocols run them (Φ wiki, concepts/robot-policy-evaluation.md):
// a task card fixed before the first trial; every policy run once from each start condition, in a random order
// (a bundle); letters instead of names until the end; milestone scores; rig faults voided, not blamed on the
// policy; and numbers that say what they cannot show. The server owns the record and every number (evals.py).
export function Evaluate() {
  const evals = useStudio((s) => s.evals);
  useDownloads();
  if (!evals) {
    return (
      <div className="page">
        <Notices />
        <section className="panel"><p className="empty">Evals need a data directory. Start Studio with <span className="mono">--data-dir</span>.</p></section>
      </div>
    );
  }
  return evals.current ? <Session ev={evals} rec={evals.current} /> : <Plan ev={evals} />;
}

// The server sends an export to the window that asked; save it as a file.
function useDownloads() {
  useEffect(() => studio.onMessage("eval_export", (m: { text: string; name: string; format: string }) => {
    const url = URL.createObjectURL(new Blob([m.text], { type: m.format === "csv" ? "text/csv" : "text/markdown" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = m.name;
    a.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }), []);
}

const when = (t: number) => new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
const refUrl = (card: string, name: string) => `/api/evals/ref/${encodeURIComponent(card)}/${encodeURIComponent(name)}?token=${encodeURIComponent(studio.token())}`;

// == before an eval: cards, a new eval, past evals ===============================================================
function Plan({ ev }: { ev: EvalState }) {
  const [editing, setEditing] = useState<Card | null>(null);
  return (
    <div className="page evaluate">
      <Notices />
      <div className="ev-plan">
        <Cards ev={ev} onEdit={setEditing} />
        <NewEval ev={ev} />
      </div>
      <Past past={ev.past} dir={ev.dir} />
      {editing && <CardEditor card={editing} onClose={() => setEditing(null)} />}
    </div>
  );
}

function Cards({ ev, onEdit }: { ev: EvalState; onEdit: (c: Card) => void }) {
  const control = useStudio((s) => s.control);
  const [imported, setImported] = useState<string | null>(null);
  useEffect(() => studio.onMessage("eval_imported", (m: { kept: number; rows: number }) =>
    setImported(`Imported ${m.kept} of ${m.rows} rows as a past eval.`)), []);
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Task cards</h2>
          <p className="panel-sub">What success is, the milestones that earn credit, and where things start. An eval locks its card.</p>
        </div>
        <button className="btn btn-sm" disabled={!control} onClick={() => onEdit(blankCard())}><Plus aria-hidden /> New card</button>
      </div>
      {ev.cards.length === 0 ? <p className="empty">No cards yet. Make one for the task you want to measure.</p> : (
        <ul className="ev-cards">
          {ev.cards.map((c) => (
            <li key={c.id} className="ev-card">
              <div className="ev-card-text">
                <span className="ev-card-name">{c.name}{c.locked_at ? <Lock className="ev-lock" aria-label="Locked" /> : null}</span>
                <span className="faint t-sm">
                  {c.conditions.length} condition{c.conditions.length === 1 ? "" : "s"} · {c.rubric.length} milestones · {c.limit_s} s
                  {c.conditions.some((x) => x.axis !== "train") ? ` · tests ${[...new Set(c.conditions.map((x) => x.axis))].join(", ")}` : ""}
                </span>
              </div>
              <div className="ev-card-actions">
                <button className="btn btn-ghost btn-sm" onClick={() => onEdit(c)}>
                  {c.locked_at ? <><FileText aria-hidden /> View</> : <><Pencil aria-hidden /> Edit</>}
                </button>
                <button className="btn btn-ghost btn-sm" disabled={!control} onClick={() => studio.send({ cmd: "eval_card_dup", id: c.id })}
                  title="A copy you can change"><Copy aria-hidden /> Duplicate</button>
              </div>
            </li>
          ))}
        </ul>
      )}
      {ev.club_csv && (
        <div className="panel-body ev-import">
          <span className="field-hint">The club's scores from <span className="mono">phi.utils.eval_rollouts</span> can join the history as a past eval (not paired).</span>
          <button className="btn btn-sm" disabled={!control} onClick={() => studio.send({ cmd: "eval_import" })}><Import aria-hidden /> Import club scores</button>
          {imported && <span className="field-hint text-ok">{imported}</span>}
        </div>
      )}
    </section>
  );
}

function NewEval({ ev }: { ev: EvalState }) {
  const control = useStudio((s) => s.control);
  const policies = useStudio((s) => (s.rig?.policies ?? []).filter((p) => p.available));
  const [cardId, setCardId] = useState(ev.cards[0]?.id ?? "");
  const [chosen, setChosen] = useState<string[]>([]);
  const [reps, setReps] = useState(5);
  const [blind, setBlind] = useState(true);
  const [grouped, setGrouped] = useState(false);
  const [alpha, setAlpha] = useState(0.05);
  useEffect(() => { if (!ev.cards.some((c) => c.id === cardId)) setCardId(ev.cards[0]?.id ?? ""); }, [ev.cards, cardId]);
  const card = ev.cards.find((c) => c.id === cardId);
  const okReps = Number.isInteger(reps) && reps >= 1 && reps <= 50;
  const plan = card && okReps ? planOf(card.conditions.length, reps, Math.max(1, chosen.length), card.limit_s) : null;
  const why = !control ? "Another window has control" : !card ? "Make a task card first" : chosen.length === 0 ? "Pick at least one policy"
    : !okReps ? "1 to 50 bundles per condition" : null;
  const toggle = (id: string) => setChosen((c) => (c.includes(id) ? c.filter((x) => x !== id) : [...c, id].slice(0, 6)));
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">New eval</h2>
          <p className="panel-sub">Every policy runs once from each condition, in a random order, so drift and hard spots hit all of them alike.</p>
        </div>
      </div>
      <div className="panel-body form">
        <label className="field">
          <span className="field-label">Task card</span>
          <select className="select" value={cardId} onChange={(e) => setCardId(e.target.value)} disabled={!ev.cards.length}>
            {ev.cards.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
        </label>
        <fieldset className="field ev-fieldset">
          <legend className="field-label">Policies to compare</legend>
          {policies.length === 0 ? <span className="field-hint">No policy can run on this rig yet.</span> : (
            <div className="ev-picks">
              {policies.map((p) => (
                <label key={p.id} className={`ev-pick ${chosen.includes(p.id) ? "is-on" : ""}`}>
                  <input type="checkbox" checked={chosen.includes(p.id)} onChange={() => toggle(p.id)} /> {p.name}
                </label>
              ))}
            </div>
          )}
        </fieldset>
        <div className="ev-row">
          <label className="field">
            <span className="field-label">Bundles per condition</span>
            <input className="input num" type="number" min={1} max={50} value={reps} onChange={(e) => setReps(Number(e.target.value))} />
          </label>
          <label className="field">
            <span className="field-label">False-alarm rate</span>
            <select className="select" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))}>
              <option value={0.05}>5% (standard)</option>
              <option value={0.1}>10% (decides sooner)</option>
            </select>
          </label>
        </div>
        <label className="switch"><input type="checkbox" checked={blind} onChange={(e) => setBlind(e.target.checked)} />
          <span>Blind: show letters, not names, until the eval ends</span></label>
        <label className="switch"><input type="checkbox" checked={grouped} onChange={(e) => setGrouped(e.target.checked)} />
          <span>Keep each condition's bundles together (fewer scene changes, less protection from drift)</span></label>
        {plan && card && (
          <div className="ev-planbox">
            <div className="num"><b>{plan.trials}</b> trials · {plan.bundles} bundles · about {plan.minutes} min with 30 s resets</div>
            <div className="field-hint">
              {chosen.length < 2 ? "One policy: a success rate with its interval, no comparison."
                : plan.gap != null ? `With ${plan.bundles} bundles, only gaps of about ${Math.round(plan.gap * 100)} points or more (from 50%) can be shown. Smaller gaps need more bundles.`
                : `${plan.bundles} bundles cannot show any gap under 50 points. Add bundles.`}
              {card.conditions.some((c) => c.control) ? " Trained-on conditions run first: if a policy fails those, suspect the rig." : ""}
            </div>
          </div>
        )}
        <div className="form-actions">
          {why && <span className="field-hint">{why}</span>}
          <button className="btn btn-primary" disabled={why !== null}
            onClick={() => studio.send({ cmd: "eval_begin", card: cardId, policies: chosen, reps, blind, grouped, alpha })}>
            <Flag aria-hidden /> Start eval
          </button>
        </div>
      </div>
    </section>
  );
}

function Past({ past, dir }: { past: PastEval[]; dir: string }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Past evals</h2>
          <p className="panel-sub">Saved in <span className="mono">{dir}</span>. Policies sharing a letter were not shown to differ; imported scores were not run in bundles, so they get no letters.</p>
        </div>
      </div>
      {past.length === 0 ? <p className="empty">No evals yet.</p> : (
        <div className="table-scroll">
          <table className="table past">
            <thead><tr><th>Started</th><th>Card</th><th>Results</th><th className="r">Export</th></tr></thead>
            <tbody>
              {past.map((e) => (
                <tr key={e.id}>
                  <td className="num">{when(e.started_at)}</td>
                  <td>{e.name}{e.imported ? <span className="badge tone-neutral ev-tag">imported</span> : null}{e.schema === 1 ? <span className="badge tone-neutral ev-tag">first version</span> : null}</td>
                  <td>
                    {(e.results ?? [{ policy: e.policies[0], n: e.n ?? 0, k: e.k ?? 0, rate: e.rate ?? null, ci95: e.ci95 ?? [0, 1], progress: null }]).map((r) => (
                      <div key={r.policy} className="ev-result">
                        <span className="ellipsis" title={r.policy}>{r.policy}{"letters" in r && r.letters ? <span className="faint"> ({r.letters})</span> : null}</span>
                        <Interval k={r.k} n={r.n} lo={r.ci95[0]} hi={r.ci95[1]} compact />
                        {"by_axis" in r && r.by_axis && Object.keys(r.by_axis).length > 1 && (
                          <span className="faint t-sm ev-axisline">
                            {Object.entries(r.by_axis).map(([ax, [k, n]]) => `${ax} ${k}/${n}`).join(" · ")}
                          </span>
                        )}
                      </div>
                    ))}
                  </td>
                  <td className="r">
                    {e.schema === 2 && (
                      <span className="ev-export">
                        <button className="btn btn-ghost btn-sm" onClick={() => studio.send({ cmd: "eval_export", id: e.id, format: "md" })}><FileText aria-hidden /> Report</button>
                        <button className="btn btn-ghost btn-sm" onClick={() => studio.send({ cmd: "eval_export", id: e.id, format: "csv" })}><Download aria-hidden /> CSV</button>
                      </span>
                    )}
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

// -- the card editor --------------------------------------------------------------------------------------------------
function CardEditor({ card, onClose }: { card: Card; onClose: () => void }) {
  const control = useStudio((s) => s.control);
  const [d, setD] = useState<Card>(() => structuredClone(card));
  const locked = !!card.locked_at;
  useEffect(() => studio.onMessage("eval_card_saved", () => onClose()), [onClose]);
  const problem = draftProblem(d);
  const set = (patch: Partial<Card>) => setD({ ...d, ...patch });
  return (
    <Dialog.Root open onOpenChange={(o) => { if (!o) onClose(); }}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="dialog dialog-wide ev-editor" aria-describedby="ev-editor-desc">
          <Dialog.Title className="dialog-title">{locked ? "Task card (locked)" : card.id ? "Edit task card" : "New task card"}</Dialog.Title>
          <p id="ev-editor-desc" className="dialog-text">
            {locked ? "An eval used this card, so its rubric is fixed: results stay comparable. Duplicate it to change it."
              : "Write it before the first trial: what counts as success must not depend on what the policies do."}
          </p>
          <fieldset className="form ev-form" disabled={locked || !control}>
            <label className="field"><span className="field-label">Name</span>
              <input className="input" value={d.name} maxLength={80} onChange={(e) => set({ name: e.target.value })} placeholder="Cube to bin" /></label>
            <label className="field"><span className="field-label">Task, as the policy is told it</span>
              <input className="input" value={d.task} maxLength={300} onChange={(e) => set({ task: e.target.value })} placeholder="Put the red cube in the white bin" /></label>
            <label className="field"><span className="field-label">Success, in one sentence</span>
              <input className="input" value={d.success} maxLength={300} onChange={(e) => set({ success: e.target.value })} placeholder="The cube ends inside the bin and the gripper has let go" /></label>
            <label className="field ev-narrow"><span className="field-label">Time limit per trial (s)</span>
              <input className="input num" type="number" min={1} max={600} value={d.limit_s} onChange={(e) => set({ limit_s: Number(e.target.value) })} /></label>

            <div className="field">
              <span className="field-label">Milestones (credit for how far it got; the last one is success)</span>
              {d.rubric.map((m, i) => (
                <div key={i} className="ev-line">
                  <span className="num faint">{i + 1}</span>
                  <input className="input" value={m.label} maxLength={80} onChange={(e) => set({ rubric: d.rubric.map((x, j) => (j === i ? { ...x, label: e.target.value } : x)) })} />
                  <input className="input num ev-points" type="number" step={0.05} min={0.01} max={1} value={m.points}
                    onChange={(e) => set({ rubric: d.rubric.map((x, j) => (j === i ? { ...x, points: Number(e.target.value) } : x)) })} />
                  <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Remove milestone" onClick={() => set({ rubric: d.rubric.filter((_, j) => j !== i) })}><Trash2 aria-hidden /></button>
                </div>
              ))}
              {d.rubric.length < 8 && <button type="button" className="link-btn" onClick={() => set({ rubric: [...d.rubric, { label: "", points: 1 }] })}>Add a milestone</button>}
            </div>

            <label className="field"><span className="field-label">Failure tags, one per line (what went wrong)</span>
              <textarea className="textarea" rows={4} value={d.failures.join("\n")}
                onChange={(e) => set({ failures: e.target.value.split("\n").map((x) => x.trim()).filter(Boolean).slice(0, 16) })} /></label>

            <div className="field">
              <span className="field-label">Start conditions (where things start, and what each one tests)</span>
              {d.conditions.map((c, i) => (
                <div key={c.id || i} className="ev-line ev-cond">
                  <input className="input" value={c.label} maxLength={80} placeholder="Red cube at spot 3, bin on the left"
                    onChange={(e) => set({ conditions: d.conditions.map((x, j) => (j === i ? { ...x, label: e.target.value } : x)) })} />
                  <select className="select" value={c.axis} onChange={(e) => set({ conditions: d.conditions.map((x, j) => (j === i ? { ...x, axis: e.target.value as Card["conditions"][number]["axis"] } : x)) })}>
                    {AXES.map((a) => <option key={a} value={a}>{a}</option>)}
                  </select>
                  <label className="switch ev-control" title="A spot the policies trained on: run first, as a rig check">
                    <input type="checkbox" checked={c.control} onChange={(e) => set({ conditions: d.conditions.map((x, j) => (j === i ? { ...x, control: e.target.checked } : x)) })} />
                    <span>trained on</span>
                  </label>
                  <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Remove condition" onClick={() => set({ conditions: d.conditions.filter((_, j) => j !== i) })}><Trash2 aria-hidden /></button>
                </div>
              ))}
              {d.conditions.length < 40 && <button type="button" className="link-btn" onClick={() => set({ conditions: [...d.conditions, { id: "", label: "", axis: "train", control: false, refs: {} }] })}>Add a condition</button>}
              <span className="field-hint">A held-out position is interpolation, not a new object. Camera pose and table texture move real success the most.</span>
            </div>
          </fieldset>
          <div className="dialog-buttons">
            {!locked && problem && <span className="field-hint">{problem}</span>}
            <Dialog.Close asChild><button className="btn">{locked ? "Close" : "Cancel"}</button></Dialog.Close>
            {locked
              ? <button className="btn btn-primary" disabled={!control} onClick={() => studio.send({ cmd: "eval_card_dup", id: card.id })}><Copy aria-hidden /> Duplicate</button>
              : <button className="btn btn-primary" disabled={!control || problem !== null}
                  onClick={() => studio.send({ cmd: "eval_card_save", card: { ...d, conditions: d.conditions.map((c) => (c.id ? c : { ...c, id: undefined })) } })}>Save card</button>}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

// == during an eval ===================================================================================================
function Session({ ev, rec }: { ev: EvalState; rec: EvalRec }) {
  const run = useStudio((s) => s.telemetry?.policy ?? null);
  const pre = usePreflight(undefined);
  const next = ev.next;
  const prog = progressOf(rec);
  const blocked = ev.awaiting ? "Score the last trial first" : !next ? "Every trial is done: end the eval" : pre.block;
  const s = ev.summary;
  return (
    <div className="page evaluate">
      <Head rec={rec} prog={prog} />
      <ActionBar activity="policy" startCmd={{ cmd: "eval_run" }} canStart={blocked === null} startBlocked={blocked ?? undefined}
        startLabel={next ? `Run trial ${prog.done + 1}: ${policyName(rec, next.alias)}` : "All trials done"} />
      <Preflight items={pre.items} />
      <Notices />
      <div className="work-grid">
        <div className="col">
          {ev.awaiting && run ? <Score rec={rec} next={next} runId={run.run_id} seconds={run.episode_s} ended={run.ended} />
            : <NextTrial rec={rec} next={next} running={!!run?.running} seconds={run?.episode_s ?? 0} />}
          <Stage rec={rec} next={next} />
        </div>
        <div className="col rail">
          {s && <Results rec={rec} s={s} />}
          <Trials rec={rec} />
        </div>
      </div>
    </div>
  );
}

function Head({ rec, prog }: { rec: EvalRec; prog: { done: number; planned: number } }) {
  const control = useStudio((s) => s.control);
  const [confirm, setConfirm] = useState(false);
  const end = () => (prog.done < prog.planned ? setConfirm(true) : studio.send({ cmd: "eval_end" }));
  return (
    <section className="panel eval-summary ev-head">
      <div className="eval-what">
        <span className="t-cap faint">Evaluating{rec.blind ? " · blind" : ""}</span>
        <span className="eval-policy">{rec.card.name}</span>
        <span className="muted ellipsis" title={rec.card.success}>Success: {rec.card.success}</span>
      </div>
      <div className="eval-stat">
        <span className="t-cap faint">Trials</span>
        <span className="t-stat num">{prog.done}<span className="eval-of"> / {prog.planned}</span></span>
      </div>
      <div className="ev-progress" aria-hidden><span style={{ width: `${prog.planned ? (prog.done / prog.planned) * 100 : 0}%` }} /></div>
      <div className="ev-head-actions">
        <button className="btn btn-ghost btn-sm" disabled={!control} onClick={() => studio.send({ cmd: "eval_undo" })} title="Undo the last score, void or skip"><Undo2 aria-hidden /> Undo</button>
        <button className="btn btn-ghost btn-sm" onClick={() => studio.send({ cmd: "eval_export", format: "csv" })}><Download aria-hidden /> CSV</button>
        <button className={`btn btn-sm ${prog.done >= prog.planned ? "btn-primary" : ""}`} disabled={!control} onClick={end}>End eval</button>
      </div>
      <Dialog.Root open={confirm} onOpenChange={setConfirm}>
        <Dialog.Portal>
          <Dialog.Overlay className="dialog-overlay" />
          <Dialog.Content className="dialog" aria-describedby="ev-end-desc">
            <Dialog.Title className="dialog-title">End the eval at {prog.done} of {prog.planned} trials?</Dialog.Title>
            <p id="ev-end-desc" className="dialog-text">
              The record closes and the policies are named. Comparisons use only the bundles where both policies ran.
            </p>
            <div className="dialog-buttons">
              <Dialog.Close asChild><button className="btn" autoFocus>Keep going</button></Dialog.Close>
              <button className="btn btn-primary" onClick={() => { studio.send({ cmd: "eval_end" }); setConfirm(false); }}>End eval</button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </section>
  );
}

function NextTrial({ rec, next, running, seconds }: { rec: EvalRec; next: NextSlot | null; running: boolean; seconds: number }) {
  const control = useStudio((s) => s.control);
  const last = rec.trials[rec.trials.length - 1];
  const [kept, setKept] = useState<number | null>(null);
  const suspect = last && !last.void && kept !== last.n && "void" in last.suspect ? last.suspect.void : [];
  const cond = next?.condition;
  const prog = progressOf(rec);
  return (
    <section className={`panel judge ${running ? "is-open" : ""}`}>
      <div className="panel-head">
        <div>
          <h2 className="panel-title">{!next ? "All trials are done" : running ? `Trial ${prog.done + 1} is running` : `Next: trial ${prog.done + 1} of ${prog.planned}`}</h2>
          <p className="panel-sub">
            {!next ? "End the eval to name the policies and see who is better."
              : running ? `${seconds.toFixed(1)} of ${rec.card.limit_s} s. Score it when it ends.`
              : "Stage the scene as in the reference, put the arm at its start pose, then run the trial."}
          </p>
        </div>
        {next && !running && (
          <button className="btn btn-ghost btn-sm" disabled={!control} title="Give up on this trial: the scene cannot be staged or the policy cannot run"
            onClick={() => { const r = window.prompt("Why skip this trial?"); if (r?.trim()) studio.send({ cmd: "eval_skip", reason: r.trim() }); }}>
            <SkipForward aria-hidden /> Skip
          </button>
        )}
      </div>
      {next && cond && (
        <div className="panel-body ev-next">
          <div className="ev-who">{policyName(rec, next.alias)}</div>
          <div className="ev-where">
            <span className="strong">{cond.label}</span>
            <span className="badge tone-neutral">{cond.axis}</span>
            {cond.control && <span className="badge tone-info">trained on</span>}
            {next.repeat && <span className="badge tone-warn">run again: the last try was void</span>}
            <span className="faint t-sm">bundle {next.b} of {rec.schedule.length}</span>
          </div>
          {Object.keys(cond.refs).length === 0 && (
            <button className="btn btn-sm" disabled={!control} onClick={() => studio.send({ cmd: "eval_card_ref", id: rec.card.id, condition: cond.id })}
              title="Keep the cameras' current pictures as this condition's reference">
              <Camera aria-hidden /> Take reference pictures (stage the scene first)
            </button>
          )}
        </div>
      )}
      {suspect.length > 0 && (
        <div className="notice tone-warn ev-suspect" role="status">
          <div>
            <div className="notice-title">Trial {last.n} may be the rig's fault, not {policyName(rec, last.alias)}'s:</div>
            <ul className="ev-reasons">{suspect.map((r) => <li key={r}>{r}</li>)}</ul>
          </div>
          <div className="notice-actions">
            <button className="btn btn-sm" disabled={!control} onClick={() => studio.send({ cmd: "eval_void", n: last.n, reason: suspect.join("; ").slice(0, 200) })}><Ban aria-hidden /> Void and run again</button>
            <button className="btn btn-ghost btn-sm" onClick={() => setKept(last.n)}>Keep it</button>
          </div>
        </div>
      )}
    </section>
  );
}

function Score({ rec, next, runId, seconds, ended }: { rec: EvalRec; next: NextSlot | null; runId: string; seconds: number; ended: string | null }) {
  const control = useStudio((s) => s.control);
  const K = rec.card.rubric.length;
  const [stage, setStage] = useState<number | null>(null);
  const [fails, setFails] = useState<string[]>([]);
  const [note, setNote] = useState("");
  const sent = useRef({ id: "", at: 0 });
  useEffect(() => { setStage(null); setFails([]); setNote(""); }, [runId]);
  const save = () => {
    if (stage === null || (sent.current.id === runId && performance.now() - sent.current.at < 1000)) return;
    sent.current = { id: runId, at: performance.now() };
    // The server scores its own copy of the run; run_id says which run this window showed.
    studio.send({ cmd: "eval_score", stage, failures: stage === K ? [] : fails, note: note.trim(), run_id: runId });
  };
  // 0..K pick a milestone and Enter saves, unless the person is typing or a dialog is open.
  useEffect(() => {
    if (!control) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.metaKey || e.ctrlKey) return;
      if (e.target instanceof Element && e.target.closest('[role="dialog"], [role="alertdialog"]')) return;
      const st = stageFromKey(e.key, K);
      if (st !== null) setStage(st);
      if (e.key === "Enter") save();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }); // eslint-disable-line react-hooks/exhaustive-deps
  const cond = next?.condition;
  return (
    <section className="panel judge is-open">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">How far did {next ? policyName(rec, next.alias) : "it"} get?</h2>
          <p className="panel-sub">Ended after {seconds.toFixed(1)} s ({ended ?? "stopped"}){cond ? ` · ${cond.label}` : ""}. Score what the arm did, then reset the scene.</p>
        </div>
      </div>
      <div className="panel-body judge-body">
        <div className="ev-ms" role="radiogroup" aria-label="Highest milestone reached">
          {[0, ...rec.card.rubric.map((_, i) => i + 1)].map((i) => (
            <button key={i} role="radio" aria-checked={stage === i} className={`ev-ms-btn ${stage === i ? "is-on" : ""} ${i === K ? "is-success" : ""}`}
              disabled={!control} onClick={() => setStage(i)}>
              <span className="kbd">{i}</span>
              <span className="ev-ms-label">{milestoneLabel(rec.card, i)}</span>
              <span className="num faint">{i === 0 ? "0" : rec.card.rubric[i - 1].points}</span>
            </button>
          ))}
        </div>
        {stage !== null && stage < K && rec.card.failures.length > 0 && (
          <div className="ev-chips" role="group" aria-label="What went wrong">
            {rec.card.failures.map((f) => (
              <button key={f} className={`ev-chip ${fails.includes(f) ? "is-on" : ""}`} aria-pressed={fails.includes(f)} disabled={!control}
                onClick={() => setFails((x) => (x.includes(f) ? x.filter((y) => y !== f) : [...x, f]))}>{f}</button>
            ))}
          </div>
        )}
        <input className="input" placeholder="Note, optional: what happened" value={note} maxLength={2000} onChange={(e) => setNote(e.target.value)} />
        <div className="form-actions">
          <span className="field-hint">Keys: 0 to {K} pick, Enter saves.</span>
          <button className="btn btn-primary" disabled={!control || stage === null} onClick={save}>Save score</button>
        </div>
      </div>
    </section>
  );
}

// The live cameras, each with this condition's reference picture laid over it as a ghost (TRI's overlay):
// match the two and every trial starts the same way.
function Stage({ rec, next }: { rec: EvalRec; next: NextSlot | null }) {
  const cams = useStudio((s) => Object.keys(s.cameras));
  const [ghost, setGhost] = useState(true);
  const refs = next?.condition?.refs ?? {};
  // WHY a copy: sorting the selector's own array in place changes the store's cached value, and React
  // then sees a new snapshot on every read and renders forever (#185).
  const keys = cams.length ? [...cams].sort() : Object.keys(refs);
  const any = Object.keys(refs).length > 0;
  return (
    <section className="panel cams-panel">
      <div className="panel-head">
        <h2 className="panel-title">Stage the scene</h2>
        {any && (
          <button className="btn btn-ghost btn-sm" onClick={() => setGhost(!ghost)} aria-pressed={ghost}>
            {ghost ? <><EyeOff aria-hidden /> Hide reference</> : <><Eye aria-hidden /> Show reference</>}
          </button>
        )}
      </div>
      {!any && <p className="field-hint ev-pad">No reference pictures for this condition yet.</p>}
      <div className="cams">
        {keys.map((k) => (
          <div key={k} className="ev-cam">
            <CameraTile name={k} />
            {ghost && refs[k] && rec.card.id && <img className="ev-ghost" src={refUrl(rec.card.id, refs[k])} alt="" aria-hidden />}
          </div>
        ))}
      </div>
    </section>
  );
}

function Results({ rec, s }: { rec: EvalRec; s: Summary }) {
  const name = (a: string) => policyName(rec, a);
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Results so far</h2>
          <p className="panel-sub">{s.valid_trials} scored{s.voided ? ` · ${s.voided} void` : ""}{s.skipped ? ` · ${s.skipped} skipped` : ""}. Rates with 95% intervals.</p>
        </div>
      </div>
      <div className="panel-body ev-results">
        {s.order.map((a) => {
          const r = s.policies[a];
          const tops = Object.entries(r.failures).sort((x, y) => y[1] - x[1]).slice(0, 2);
          return (
            <div key={a} className="ev-res">
              <div className="ev-res-name">{name(a)}{s.letters ? <span className="faint"> ({s.letters[a]})</span> : null}</div>
              <Interval k={r.k} n={r.n} lo={r.ci95[0]} hi={r.ci95[1]} compact />
              <div className="faint t-sm">
                credit {r.progress == null ? "–" : r.progress.toFixed(2)}
                {r.median_success_s != null ? ` · ${r.median_success_s.toFixed(1)} s to succeed` : ""}
                {tops.length ? ` · ${tops.map(([f, n]) => `${f} ${n}`).join(", ")}` : ""}
              </div>
            </div>
          );
        })}
        {s.pairs.map((p) => <p key={p.a + p.b} className="ev-pair">{decisionText(p, name)}</p>)}
        {s.detectable_gap != null && s.pairs.length > 0 && (
          <p className="field-hint">This plan ({s.planned_bundles} bundles) can show gaps of about {Math.round(s.detectable_gap * 100)} points or more.</p>
        )}
        {Object.keys(s.policies).length > 0 && <AxisTable rec={rec} s={s} />}
      </div>
    </section>
  );
}

// Success by what each condition tests, never pooled: a policy can be fine on trained spots and lost on new ones.
function AxisTable({ rec, s }: { rec: EvalRec; s: Summary }) {
  const axes = [...new Set(s.order.flatMap((a) => Object.keys(s.policies[a].by_axis)))];
  if (axes.length < 2) return null;
  return (
    <table className="table ev-axes">
      <thead><tr><th>Tests</th>{s.order.map((a) => <th key={a} className="r">{policyName(rec, a)}</th>)}</tr></thead>
      <tbody>
        {axes.map((ax) => (
          <tr key={ax}>
            <td>{ax}</td>
            {s.order.map((a) => {
              const r = s.policies[a].by_axis[ax];
              return <td key={a} className="r num">{r ? `${r.k}/${r.n}` : "–"}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Trials({ rec }: { rec: EvalRec }) {
  const control = useStudio((s) => s.control);
  const conds = Object.fromEntries(rec.card.conditions.map((c) => [c.id, c]));
  const voidIt = (t: Trial) => {
    const r = window.prompt(`Why is trial ${t.n} void? It will run again and not count.`, "void" in t.suspect ? t.suspect.void.join("; ") : "");
    if (r?.trim()) studio.send({ cmd: "eval_void", n: t.n, reason: r.trim().slice(0, 200) });
  };
  return (
    <section className="panel">
      <div className="panel-head"><h2 className="panel-title">Trials</h2></div>
      {rec.trials.length === 0 ? <p className="empty">None scored yet.</p> : (
        <ol className="episodes">
          {[...rec.trials].reverse().map((t) => (
            <li key={t.n} className={`episode ev-trial ${t.void ? "is-void" : ""}`}>
              <span className="episode-n num">{t.n}</span>
              <span className={`badge tone-${t.void ? "neutral" : t.success ? "ok" : t.stage > 0 ? "warn" : "danger"}`}>
                {t.void ? "void" : t.success ? "success" : milestoneLabel(rec.card, t.stage)}
              </span>
              <span className="episode-time num faint">{policyName(rec, t.alias)}</span>
              <span className="episode-note">
                {conds[t.c]?.label ?? t.c}{t.failures.length ? ` · ${t.failures.join(", ")}` : ""}{t.note ? ` · ${t.note}` : ""}
                {t.void ? ` · void: ${t.void.reason}` : ""}
                {!t.void && "warn" in t.suspect && t.suspect.warn.length ? <span className="text-warn"> · {t.suspect.warn.join("; ")}</span> : null}
              </span>
              {!t.void && t.b != null && <button className="btn btn-ghost btn-sm ev-void" disabled={!control} onClick={() => voidIt(t)} title="The rig spoiled it: take it out and run it again">Void</button>}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

// The point estimate with its interval on a 0 to 100% track, so a wide interval looks wide.
function Interval({ k, n, lo, hi, compact }: { k: number; n: number; lo: number; hi: number; compact?: boolean }) {
  if (n === 0) return <span className="faint">{compact ? "none yet" : "No trials yet"}</span>;
  const rate = k / n;
  return (
    <div className={`interval ${compact ? "is-compact" : ""}`}>
      <span className={compact ? "num strong" : "t-stat num"}>{pctOf(rate)}</span>
      <div className="interval-track" aria-hidden>
        <span className="interval-band" style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }} />
        <span className="interval-dot" style={{ left: `${rate * 100}%` }} />
      </div>
      <span className="num muted">{k}/{n} · {pctOf(lo)}–{pctOf(hi)}</span>
    </div>
  );
}
