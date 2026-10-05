// "Workspace points" on the 3D view page: capture a metric point cloud of the desk from a camera picture, and how
// to show it. The server does the work (src/phi_studio/recon_api.py); the clouds are drawn by scene/pointcloud.ts.
import { Camera, Download, RefreshCw, X } from "lucide-react";
import { useEffect } from "react";
import { label } from "../lib/labels";
import { type CloudInfo, type DatasetInfo, fitText, reconStore, type Refusal, SIZE, useRecon } from "../lib/recon";
import { followerSlots } from "../lib/scene";
import { useStudio } from "../lib/studio";

const mb = (b: number) => (b / 1e6).toFixed(0);

export function ReconPanel() {
  const status = useRecon((s) => s.status);
  const settings = useRecon((s) => s.settings);
  const datasets = useRecon((s) => s.datasets);
  const datasetsError = useRecon((s) => s.datasetsError);
  const clouds = useRecon((s) => s.clouds);
  const refused = useRecon((s) => s.refused);
  const asked = useRecon((s) => s.asked);
  const control = useStudio((s) => s.control);
  const link = useStudio((s) => s.link);
  const telemetry = useStudio((s) => s.telemetry);
  const rigArms = useStudio((s) => s.rig?.arms);

  useEffect(() => { reconStore.start(); }, []);
  useEffect(() => { if (link === "open") { reconStore.refresh(); reconStore.loadDatasets(); } }, [link]);
  // The live camera list and the download bar change on their own: ask again while either matters.
  useEffect(() => {
    if (link !== "open") return;
    const t = window.setInterval(() => reconStore.refresh(), 2000);
    return () => window.clearInterval(t);
  }, [link]);

  const ds = datasets?.find((d) => d.root === settings.root) ?? null;
  const key = ds && settings.key && ds.cameras.includes(settings.key) ? settings.key : ds?.cameras[0] ?? null;
  const live = status?.live ?? [];
  const liveKey = settings.liveKey && live.includes(settings.liveKey) ? settings.liveKey : live[0] ?? null;
  const cached = status?.model.cached ?? true;
  const dl = status?.download;
  const pairs = followerSlots(telemetry, rigArms).length;
  const cams = [...new Set([...Object.keys(clouds), ...Object.keys(refused)])].sort();
  const busy = (cam: string) => cam in asked || status?.running === cam || (status?.pending ?? []).includes(cam);

  const captureOne = () => {
    if (settings.source === "dataset") { if (ds && key) reconStore.captureDataset(key, ds); }
    else if (liveKey) reconStore.captureLive(liveKey);
  };
  const captureAll = () => {
    if (settings.source === "dataset") ds?.cameras.forEach((k) => reconStore.captureDataset(k, ds));
    else live.forEach((k) => reconStore.captureLive(k));
  };
  const canCapture = settings.source === "dataset" ? Boolean(ds && key) : Boolean(liveKey);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Workspace points</h2>
        <span className="panel-sub">A 3D picture of the desk from a camera</span>
      </div>
      <div className="panel-body scene-form">
        {!cached && (
          <div className="recon-model">
            <p>The depth model is not on this computer yet. Studio does not download it without asking.</p>
            {dl?.state === "running" ? (
              <>
                <div className="scene-bar recon-bar" role="progressbar" aria-valuemin={0} aria-valuemax={dl.total} aria-valuenow={dl.done}>
                  <div style={{ width: `${Math.min(100, (100 * dl.done) / Math.max(1, dl.total))}%` }} />
                </div>
                <div className="recon-row">
                  <span className="field-hint">{mb(dl.done)} of {mb(dl.total)} MB</span>
                  <button type="button" className="btn btn-sm" disabled={!control} onClick={() => reconStore.cancelDownload()}><X />Cancel</button>
                </div>
              </>
            ) : (
              <button type="button" className="btn btn-sm btn-primary" disabled={!control} onClick={() => reconStore.download()}>
                <Download />Download the depth model ({mb(status?.model.bytes ?? 99e6)} MB, {status?.model.license ?? "Apache-2.0"})
              </button>
            )}
            {!control && <span className="field-hint">Take control to download.</span>}
            {dl?.state === "failed" && <p className="recon-warn">{dl.error}</p>}
            {dl?.state === "cancelled" && <span className="field-hint">Download cancelled. Pressing it again picks up where it stopped.</span>}
          </div>
        )}

        <div className="scene-seg" role="group" aria-label="Where the picture comes from">
          <button type="button" aria-pressed={settings.source === "dataset"} onClick={() => reconStore.update({ source: "dataset" })}>Dataset frame</button>
          <button type="button" aria-pressed={settings.source === "live"} onClick={() => reconStore.update({ source: "live" })}>Live</button>
        </div>

        {settings.source === "dataset" ? (
          <DatasetPicker datasets={datasets} error={datasetsError} ds={ds} cameraKey={key} />
        ) : (
          <label className="field">
            <span className="field-label">Camera</span>
            {live.length ? (
              <select className="select" value={liveKey ?? ""} onChange={(e) => reconStore.update({ liveKey: e.target.value })}>
                {live.map((k) => <option key={k} value={k}>{label(k)}</option>)}
              </select>
            ) : (
              <span className="field-hint">No camera is streaming. Start teleop or camera align to capture live.</span>
            )}
          </label>
        )}

        <div className="recon-row">
          <button type="button" className="btn btn-sm btn-primary" disabled={!canCapture || link !== "open"} onClick={captureOne}><Camera />Capture scene</button>
          <button type="button" className="btn btn-sm" disabled={!canCapture || link !== "open"} onClick={captureAll}>Every camera</button>
        </div>
        {settings.source === "live" && liveKey && (
          <label className="scene-check">
            <input type="checkbox" checked={status?.keep === liveKey} onChange={(e) => reconStore.keep(e.target.checked ? liveKey : null)} />
            <span>Keep updating {label(liveKey)}<span className="field-hint"> (at most twice a second, one camera)</span></span>
          </label>
        )}
        {settings.source === "dataset" && pairs > 1 && (
          <span className="field-hint">A dataset frame is drawn for one arm with its base at the middle of the view.</span>
        )}

        {cams.map((cam) => (
          <LayerRow key={cam} cam={cam} cloud={clouds[cam]} refusal={refused[cam]} busy={busy(cam)} visible={settings.visible[cam] !== false} />
        ))}

        {cams.length > 0 && (
          <>
            <label className="field">
              <span className="field-label">Point size: {settings.size} mm</span>
              <input type="range" className="scene-slider" min={SIZE.min} max={SIZE.max} step={1} value={settings.size}
                onChange={(e) => reconStore.update({ size: Number(e.target.value) })} />
            </label>
            <label className="field">
              <span className="field-label">Opacity: {Math.round(settings.opacity * 100)}%</span>
              <input type="range" className="scene-slider" min={10} max={100} step={5} value={Math.round(settings.opacity * 100)}
                onChange={(e) => reconStore.update({ opacity: Number(e.target.value) / 100 })} />
            </label>
            <label className="scene-check">
              <input type="checkbox" checked={settings.hideArm} onChange={(e) => reconStore.update({ hideArm: e.target.checked })} />
              <span>Hide points on the arm<span className="field-hint"> (a box around each part of the drawn arm, plus 1.5 cm)</span></span>
            </label>
          </>
        )}
        <p className="field-hint">
          Depth from Depth Anything V2 Small. It gives depth only up to an unknown scale and offset, so Studio sets both from
          the table the arm stands on. The points are only as right as the camera's place and field of view in this view.
        </p>
      </div>
    </section>
  );
}

function DatasetPicker({ datasets, error, ds, cameraKey }: { datasets: DatasetInfo[] | null; error: string | null; ds: DatasetInfo | null; cameraKey: string | null }) {
  const settings = useRecon((s) => s.settings);
  if (error) return <p className="recon-warn">{error}</p>;
  if (!datasets) return <p className="field-hint">Looking for datasets on this computer.</p>;
  if (!datasets.length) return <p className="field-hint">No LeRobot datasets on this computer.</p>;
  const num = (v: string, max: number) => Math.max(0, Math.min(max, Math.floor(Number(v) || 0)));
  return (
    <>
      <label className="field">
        <span className="field-label">Dataset</span>
        <select className="select" value={ds?.root ?? ""} onChange={(e) => reconStore.update({ root: e.target.value, key: null })}>
          {!ds && <option value="">Pick one</option>}
          {datasets.map((d) => <option key={d.root} value={d.root}>{d.name}</option>)}
        </select>
        {ds?.note && <span className="field-hint">{ds.note}</span>}
      </label>
      {ds && (
        <>
          <div className="recon-grid">
            <label className="field">
              <span className="field-label">Episode</span>
              <input className="input num" inputMode="numeric" value={settings.episode} aria-label="Episode"
                onChange={(e) => reconStore.update({ episode: num(e.target.value, ds.episodes - 1) })} />
            </label>
            <label className="field">
              <span className="field-label">Frame</span>
              <input className="input num" inputMode="numeric" value={settings.frame} aria-label="Frame"
                onChange={(e) => reconStore.update({ frame: num(e.target.value, 100_000) })} />
            </label>
          </div>
          <label className="field">
            <span className="field-label">Camera</span>
            <select className="select" value={cameraKey ?? ""} onChange={(e) => reconStore.update({ key: e.target.value })}>
              {ds.cameras.map((k) => <option key={k} value={k}>{label(reconStore.sceneCamera(k, ds))}</option>)}
            </select>
          </label>
        </>
      )}
    </>
  );
}

function LayerRow({ cam, cloud, refusal, busy, visible }: { cam: string; cloud?: CloudInfo; refusal?: Refusal; busy: boolean; visible: boolean }) {
  const newer = refusal && (!cloud || refusal.at > cloud.at * 1000);
  const est = (newer ? refusal?.estimate : cloud?.estimate) ?? cloud?.estimate;
  return (
    <div className="scene-cam recon-layer">
      <div className="scene-cam-head">
        {cloud ? (
          <label className="scene-check">
            <input type="checkbox" checked={visible} onChange={(e) => reconStore.setVisible(cam, e.target.checked)} />
            <span className="strong">{label(cam)}</span>
          </label>
        ) : <span className="strong">{label(cam)}</span>}
        {busy && <span className="badge tone-info"><RefreshCw className="spin" />Working</span>}
        <button type="button" className="btn btn-sm btn-ghost btn-icon" title={`Remove the ${label(cam)} points`} aria-label={`Remove the ${label(cam)} points`}
          onClick={() => reconStore.clear(cam)}><X /></button>
      </div>
      {newer && refusal && <p className="recon-warn">{refusal.message}</p>}
      {cloud && (
        <>
          <span className={newer ? "field-hint" : "recon-fit"}>{newer ? "Showing the last cloud that worked. " : ""}{fitText(cloud.fit)}.</span>
          <span className="field-hint">
            {cloud.n.toLocaleString()} points from {cloud.source.kind === "dataset"
              ? `${cloud.source.name}, episode ${cloud.source.episode}, frame ${cloud.source.frame}`
              : "the live camera"}.
          </span>
        </>
      )}
      {newer && refusal?.fit && refusal.fit.s !== null && (
        <span className="field-hint">This try: {fitText(refusal.fit).toLowerCase()}.</span>
      )}
      {est && (
        <span className="field-hint">
          Field of view {est.fovy_deg}° vertical, {est.fov}. Placement: {est.placement}. Lens: {est.lens}.
        </span>
      )}
    </div>
  );
}
