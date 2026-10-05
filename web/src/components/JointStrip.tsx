// Each follower's joints in one line: the reading, and where it sits in the model's range. Amber near an
// end, red past it. The table with every caveat stays in the 3D view's settings (pages/Scene.tsx).
import { label } from "../lib/labels";
import { DEGREES, followerSlots, limitState, readArm, useScene } from "../lib/scene";
import { useStudio } from "../lib/studio";

const SHORT: Record<string, string> = {
  shoulder_pan: "pan", shoulder_lift: "lift", elbow_flex: "elbow", wrist_flex: "flex", wrist_roll: "roll", gripper: "grip",
};

export function JointStrip() {
  const model = useScene((s) => s.model);
  const units = useScene((s) => s.units);
  const telemetry = useStudio((s) => s.telemetry);
  const rigArms = useStudio((s) => s.rig?.arms);
  const slots = followerSlots(telemetry, rigArms);
  if (!model || !slots.length) return null;
  return (
    <section className="panel jstrip">
      {slots.map((s) => {
        const pos = telemetry?.arms[s.name]?.pos;
        const q = new Float64Array(6);
        const got = readArm(model, pos, units[s.name] ?? DEGREES, q);
        return (
          <div key={s.name} className="jstrip-row">
            <span className="jstrip-arm">{label(s.name)}</span>
            {model.joint_order.map((j, i) => {
              const v = pos?.[j];
              const has = typeof v === "number" && Number.isFinite(v);
              const lim = has && got === "ok" ? limitState(model, j, q[i]) : null;
              return (
                <div key={j} className={`jcell ${lim ? `is-${lim.state}` : ""}`} title={label(j)}>
                  <span className="jcell-name">{SHORT[j] ?? j}</span>
                  <span className="jcell-v num">{has ? `${v.toFixed(j === "gripper" ? 0 : 1)}${j === "gripper" ? "" : "°"}` : "–"}</span>
                  <span className="jcell-bar">{lim && <i style={{ left: `${Math.min(100, Math.max(0, lim.frac * 100))}%` }} />}</span>
                </div>
              );
            })}
          </div>
        );
      })}
    </section>
  );
}
