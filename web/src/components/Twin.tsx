import { Box, Eye, EyeOff } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { TwinScene, type View } from "../lib/twin";

interface Props {
  arms: string[]; // "" for a single arm, or "left" and "right"
  onReady?: (scene: TwinScene) => void;
  legend?: { solid: string; ghost: string } | null;
  className?: string;
}

const VIEWS: { v: View; label: string; key: string }[] = [
  { v: "iso", label: "3/4", key: "1" }, { v: "front", label: "Front", key: "2" },
  { v: "side", label: "Side", key: "3" }, { v: "top", label: "Top", key: "4" },
];

/** The SO-101 digital twin. The parent drives it through the scene handed to onReady. */
export function Twin({ arms, onReady, legend = { solid: "Measured", ghost: "Commanded" }, className }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const [scene, setScene] = useState<TwinScene | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ghosts, setGhosts] = useState(true);
  const [view, setView] = useState<View>("iso");
  const ready = useRef(onReady);
  ready.current = onReady;
  const key = arms.join(",");

  useEffect(() => {
    if (!host.current) return;
    const s = new TwinScene(host.current);
    let live = true;
    s.init(key ? key.split(",") : [""]).then(
      () => { if (live) { setScene(s); ready.current?.(s); } },
      (e: unknown) => { if (live) setError(e instanceof Error ? e.message : String(e)); },
    );
    return () => { live = false; s.dispose(); setScene(null); };
  }, [key]);

  useEffect(() => { scene?.showGhosts(ghosts); }, [scene, ghosts]);

  return (
    <div className={`twin ${className ?? ""}`}>
      <div className="twin-host" ref={host} />
      {!scene && !error && <div className="twin-status"><Box aria-hidden className="spin-slow" />Loading the SO-101 model</div>}
      {error && <div className="twin-status tone-danger">{error}</div>}
      {scene && (
        <div className="twin-tools">
          <div className="seg" role="group" aria-label="Camera view">
            {VIEWS.map((x) => (
              <button key={x.v} className={`seg-btn ${view === x.v ? "is-on" : ""}`} title={`${x.label} view`}
                onClick={() => { setView(x.v); scene.view(x.v); }}>{x.label}</button>
            ))}
          </div>
          {legend && (
            <button className={`seg-btn twin-ghost-btn ${ghosts ? "is-on" : ""}`} onClick={() => setGhosts((g) => !g)}
              title={ghosts ? "Hide the commanded pose" : "Show the commanded pose"}>
              {ghosts ? <Eye aria-hidden /> : <EyeOff aria-hidden />}
            </button>
          )}
        </div>
      )}
      {scene && legend && (
        <div className="twin-legend" aria-hidden>
          <span><i className="sw sw-solid" />{legend.solid}</span>
          {ghosts && <span><i className="sw sw-ghost" />{legend.ghost}</span>}
        </div>
      )}
    </div>
  );
}
