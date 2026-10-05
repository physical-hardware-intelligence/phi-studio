// Simulated arms or the real ones, without restarting Studio: the server stops the arm worker and starts one for
// robot-config.yaml (onboard_api.py rig_use). Only while no arm holds torque; a connected rig is disconnected first.
import { Cpu, FlaskConical } from "lucide-react";
import { useRig } from "../lib/rig";
import { studio, useStudio } from "../lib/studio";

const HOLDING = new Set(["ARMED", "MOVING", "STOPPED", "FAULT", "CALIBRATING"]);

export function RigSwitch({ primary = false }: { primary?: boolean }) {
  const mock = useStudio((s) => s.mock);
  const control = useStudio((s) => s.control);
  const st = useStudio((s) => s.state?.state ?? "DISCONNECTED");
  const rig = useRig();
  const ports = !!rig?.exists && rig.arms.length > 0 && rig.arms.every((a) => a.port);
  const why = !control ? "Take control first"
    : HOLDING.has(st) ? "Release torque and disconnect first"
    : mock && !ports ? "Every arm needs a port: edit the rig"
    : null;
  if (mock) {
    return (
      <button className={`btn btn-sm ${primary ? "btn-primary" : ""}`} disabled={why !== null}
        title={why ?? "Studio switches to the arms in robot-config.yaml"} onClick={() => studio.send({ cmd: "rig_use", hardware: true })}>
        <Cpu aria-hidden />Use real arms
      </button>
    );
  }
  return (
    <button className="btn btn-sm btn-ghost" disabled={why !== null} title={why ?? undefined}
      onClick={() => studio.send({ cmd: "rig_use", hardware: false })}>
      <FlaskConical aria-hidden />Use simulated arms
    </button>
  );
}
