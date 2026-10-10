import { ArrowRight, Camera, Check, ChevronRight, CircleAlert, Pencil, Plug, Thermometer } from "lucide-react";
import { ArmsGlyph, HealthBar } from "../components/data/bits";
import { Notices } from "../components/Notices";
import { PhiMark } from "../components/PhiMark";
import { RigSwitch } from "../components/RigSwitch";
import { StatePill } from "../components/StatePill";
import { fmtAgo, fmtDuration, useNotes, useResource, type DatasetSummary } from "../lib/data";
import { SLOTS, shortPort, slotLabel, useRig, type Layout } from "../lib/rig";
import { go, studio, useStudio } from "../lib/studio";

// Home: this rig at a glance, the one thing to do next, and the latest data.
export function Home() {
  const rig = useRig();
  return (
    <div className="page home">
      <Notices />
      {!rig ? <div className="empty">Loading</div> : !rig.exists ? <Welcome /> : <RigCard />}
      <Recent />
    </div>
  );
}

function Welcome() {
  return (
    <section className="panel welcome">
      <PhiMark size={44} className="welcome-mark" />
      <h2 className="welcome-title">Set up your rig</h2>
      <p className="welcome-sub">Name it, find its arms, calibrate, check the cameras.</p>
      <button className="btn btn-primary" onClick={() => go("onboard")}>Start <ArrowRight aria-hidden /></button>
    </section>
  );
}

function NextAction() {
  const s = useStudio((x) => x.state);
  const identity = useStudio((x) => x.identity);
  const control = useStudio((x) => x.control);
  const link = useStudio((x) => x.link);
  const st = s?.state ?? "DISCONNECTED";
  if (link !== "open") return <button className="btn" disabled>Offline</button>;
  if (st === "DISCONNECTED") {
    return <button className="btn btn-primary" disabled={!control} onClick={() => studio.send({ cmd: "connect" })}><Plug aria-hidden />Connect</button>;
  }
  if (st === "CONNECTED") return <button className="btn" disabled>Connecting</button>;
  if (st === "IDENTIFIED") {
    return identity.every((a) => a.ok)
      ? <button className="btn btn-primary" disabled={!control} onClick={() => studio.send({ cmd: "confirm" })}><Check aria-hidden />Confirm arms</button>
      : <button className="btn btn-primary" onClick={() => go("calibrate")}>Calibrate<ChevronRight aria-hidden /></button>;
  }
  if (st === "CALIBRATING") return <button className="btn" onClick={() => go("calibrate")}>Calibrating<ChevronRight aria-hidden /></button>;
  return <button className="btn btn-primary" onClick={() => go("teleop")}>Teleop<ChevronRight aria-hidden /></button>;
}

function RigCard() {
  const rig = useRig()!;
  const mock = useStudio((s) => s.mock);
  const state = useStudio((s) => s.state);
  const link = useStudio((s) => s.link);
  const identity = useStudio((s) => s.identity);
  const tel = useStudio((s) => s.telemetry);
  const live = useStudio((s) => s.cameras);
  const control = useStudio((s) => s.control);
  const layout: Layout = rig.layout ?? "single";
  // WHY the session too: telemetry calls an arm online when its bus has not failed, which a disconnected rig
  // also satisfies. Green means read this session.
  const connected = !!state && !["DISCONNECTED", "CONNECTED"].includes(state.state);
  const arms = new Map(rig.arms.map((a) => [a.key, a]));
  return (
    <section className="panel rig">
      <div className="rig-head">
        <ArmsGlyph bimanual={layout === "bimanual"} />
        <div className="rig-name">
          <h2>{rig.name ?? (layout === "bimanual" ? "Bimanual rig" : "Rig")}</h2>
          <span className="faint t-sm">{layout === "bimanual" ? "Bimanual" : "Single arm"}{mock ? " · simulated arms" : ""}</span>
        </div>
        <span className="grow" />
        <StatePill s={state} link={link} />
        <NextAction />
      </div>
      <div className={`rig-arms n-${SLOTS[layout].length}`}>
        {SLOTS[layout].map((slot) => {
          const a = arms.get(slot);
          const id = identity.find((x) => x.name === slot);
          const t = tel?.arms[slot];
          const temp = t ? Math.max(0, ...Object.values(t.health).map((h) => h.temp)) : null;
          return (
            <div key={slot} className="arm-tile">
              <div className="arm-tile-head">
                <span className="strong">{slotLabel(slot)}</span>
                <span className={`dot tone-${connected && t?.online ? (t.torque ? "info" : "ok") : "neutral"}`} title={connected && t?.online ? (t.torque ? "Holding torque" : "Read") : "Not read yet"} />
              </div>
              <span className="mono faint" title={a?.port ?? undefined}>{a?.port ? shortPort(a.port) : "no port"}</span>
              <div className="arm-tile-foot">
                {a?.calibrated ? <span className="ok-text"><Check aria-hidden className="ico-inline" />calibrated</span>
                  : <span className="warn-text"><CircleAlert aria-hidden className="ico-inline" />no calibration</span>}
                {id && !id.ok && <span className="warn-text" title={id.match ?? undefined}>mismatch</span>}
                {temp !== null && temp > 0 && <span className="faint num"><Thermometer aria-hidden className="ico-inline" />{temp.toFixed(0)}°</span>}
              </div>
            </div>
          );
        })}
      </div>
      <div className="rig-foot">
        <span className="rig-cams">
          {rig.cameras.length ? rig.cameras.map((c) => (
            <span key={c.feature} className="cam-chip" title={`${c.feature} · camera ${c.source}`}>
              <i className={`dot tone-${live[c.key]?.online ? "ok" : "neutral"}`} /><Camera aria-hidden />{c.key}
            </span>
          )) : <span className="faint t-sm"><Camera aria-hidden className="ico-inline" />No cameras yet</span>}
        </span>
        <span className="grow" />
        {/* camcheck.py: macOS renumbers cameras at restarts; Studio checks at each start, and here on demand */}
        {!mock && rig.cameras.length > 0 && (
          <button className="btn btn-ghost btn-sm" disabled={!control} title={control ? "Look again at which camera is which" : "Take control first"}
            onClick={() => studio.send({ cmd: "camera_check" })}><Camera aria-hidden />Check cameras</button>
        )}
        <RigSwitch />
        <button className="btn btn-ghost btn-sm" onClick={() => go("onboard")}><Pencil aria-hidden />Edit rig</button>
      </div>
      {!!rig.problems.length && <div className="rig-problems">{rig.problems.slice(0, 3).map((p, i) => <div key={i} className="hrow"><CircleAlert aria-hidden className="sev sev-warn" /><span>{p}</span></div>)}</div>}
    </section>
  );
}

function Recent() {
  const { value } = useResource<{ datasets: DatasetSummary[] }>("/api/data/datasets");
  const open = useNotes(null, null)?.filter((n) => n.kind === "issue" && n.status === "open").length ?? 0;
  const rows = (value?.datasets ?? []).filter((d) => d.episodes > 0).slice(0, 4);
  if (!rows.length) return null;
  return (
    <section className="panel">
      <div className="panel-head">
        <h3 className="panel-title">Recent data</h3>
        <span className="row-gap">
          {open > 0 && <a className="badge tone-warn" href="#/issues">{open} open {open === 1 ? "issue" : "issues"}</a>}
          <a className="link-btn t-sm" href="#/data">All datasets</a>
        </span>
      </div>
      <div className="recent">
        {rows.map((d) => (
          <a key={d.id} className="recent-row" href={`#/data/${d.id}`}>
            <ArmsGlyph bimanual={d.bimanual} />
            <span className="recent-name">{d.repo_id.split("/").pop()}</span>
            <span className="faint num t-sm">{d.episodes} ep · {fmtDuration(d.duration_s)}</span>
            <span className="recent-health">{d.analysis ? <HealthBar counts={d.analysis.episodes_by_health} total={d.episodes} /> : null}</span>
            <span className="faint t-sm">{fmtAgo(d.modified)}</span>
          </a>
        ))}
      </div>
    </section>
  );
}
