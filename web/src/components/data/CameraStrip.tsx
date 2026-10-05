// The episode's cameras, locked to the playhead. A LeRobot v3 video file holds many episodes, so every
// time is offset by the episode's from_timestamp (lerobot datasets/dataset_reader.py:277-281).
import { VideoOff } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { videoUrl, type VideoSeg } from "../../lib/data";
import type { Player } from "../../lib/player";

interface Props { dataset: string; videos: Record<string, VideoSeg>; player: Player; fps: number }

export function CameraStrip({ dataset, videos, player, fps }: Props) {
  const keys = Object.keys(videos);
  const els = useRef<(HTMLVideoElement | null)[]>([]);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const segs = keys.map((k) => videos[k]);

  useEffect(() => {
    let lastFrame = -1;
    const target = (i: number) => segs[i].from + (player.frame() + 0.5) / fps; // a frame's middle: never the one before
    const sync = () => {
      const vs = els.current;
      if (player.playing) {
        vs.forEach((v, i) => {
          if (!v || v.readyState < 2) return;
          if (v.playbackRate !== player.rate) v.playbackRate = player.rate;
          if (v.paused) void v.play().catch(() => undefined);
          if (i > 0) { // followers: pull back into step with the master clock
            const want = segs[i].from + player.t;
            if (Math.abs(v.currentTime - want) > 0.12) v.currentTime = want;
          }
        });
      } else {
        const k = player.frame();
        vs.forEach((v, i) => {
          if (!v) return;
          if (!v.paused) v.pause();
          if (k !== lastFrame || Math.abs(v.currentTime - target(i)) > 0.5 / fps) v.currentTime = target(i);
        });
        lastFrame = k;
      }
    };
    player.clock = () => {
      const m = els.current[0];
      if (!m || m.paused || m.seeking || m.readyState < 3) return null;
      return m.currentTime - segs[0].from;
    };
    const off = player.on(sync);
    sync();
    return () => { off(); player.clock = null; els.current.forEach((v) => v?.pause()); };
  }, [player, fps, dataset, JSON.stringify(videos)]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!keys.length) return null;
  return (
    <>
      {keys.map((k, i) => (
        <figure key={`${dataset}/${k}/${videos[k].chunk}/${videos[k].file}`} className="vid">
          <video
            ref={(e) => { els.current[i] = e; }}
            src={videoUrl(dataset, k, videos[k])}
            muted playsInline preload="auto"
            onLoadedMetadata={(e) => { e.currentTarget.currentTime = videos[k].from + (player.frame() + 0.5) / fps; }}
            onError={() => setErrors((x) => ({ ...x, [k]: "This browser cannot play the video (AV1)." }))}
          />
          {errors[k] && <div className="vid-err"><VideoOff aria-hidden />{errors[k]}<span className="faint">Chrome plays it; so does Safari on an M3 or newer Mac.</span></div>}
          <figcaption className="vid-cap">{k.replace(/^observation\.images\./, "")}</figcaption>
        </figure>
      ))}
    </>
  );
}
