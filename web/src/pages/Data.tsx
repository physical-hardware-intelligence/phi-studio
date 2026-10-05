import { useSubpath } from "../lib/studio";
import { DatasetView } from "./data/DatasetView";
import { Inspector } from "./data/Inspector";
import { Library } from "./data/Library";

// #/data                      every dataset on this Mac
// #/data/<dataset>            one dataset: episodes, distributions, workspace, health
// #/data/<dataset>/<episode>  one episode: cameras, 3D, timeline, signals, notes
export function Data() {
  const [ds, ep] = useSubpath();
  if (ds && ep !== undefined && /^\d+$/.test(ep)) return <Inspector id={ds} episode={Number(ep)} />;
  if (ds) return <DatasetView id={ds} />;
  return <Library />;
}
