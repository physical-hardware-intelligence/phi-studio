import * as Tooltip from "@radix-ui/react-tooltip";
import { Suspense, lazy, useEffect, type ComponentType } from "react";
import { Boundary } from "./components/Boundary";
import { Sidebar } from "./components/Sidebar";
import { TerminalDock } from "./components/TerminalDock";
import { TopBar } from "./components/TopBar";
import { getTheme, setTheme, studio, useRoute, useStudio, type Route } from "./lib/studio";
import { Calibrate } from "./pages/Calibrate";
import { Checks } from "./pages/Checks";
import { Evaluate } from "./pages/Evaluate";
import { Files } from "./pages/Files";
import { Guide } from "./pages/Guide";
import { Overview } from "./pages/Overview";
import { Models } from "./pages/Models";
import { Policy } from "./pages/Policy";
import { Scene } from "./pages/Scene";
import { Setup } from "./pages/Setup";
import { Teleop } from "./pages/Teleop";
import { Train } from "./pages/Train";

// WHY lazy: the panel carries the markdown renderer, a third of the bundle, and most sessions never open it.
const AssistantPanel = lazy(() => import("./components/Assistant").then((m) => ({ default: m.AssistantPanel })));

const PAGES: Record<Route, { title: string; sub: string; el: ComponentType }> = {
  overview: { title: "Overview", sub: "Rig status, and what to do next", el: Overview },
  checks: { title: "Checks", sub: "Every read-only check of this Mac, the rig and Studio, in one run", el: Checks },
  calibrate: { title: "Calibrate", sub: "Write each arm's homing offsets and joint ranges", el: Calibrate },
  teleop: { title: "Teleoperate", sub: "Drive each follower with its leader", el: Teleop },
  policy: { title: "Run policy", sub: "Run a trained policy on the followers", el: Policy },
  evaluate: { title: "Evaluate", sub: "Judge episodes and measure the success rate", el: Evaluate },
  setup: { title: "Set up", sub: "Every step from a new rig to a recorded dataset, in order", el: Setup },
  scene: { title: "3D view", sub: "The arms as they move, in 3D", el: Scene },
  train: { title: "Train", sub: "Train a policy on this Mac or on the Northeastern cluster", el: Train },
  models: { title: "Models", sub: "Find a policy on Hugging Face, check it fits this rig, and download it", el: Models },
  files: { title: "Files", sub: "Serial ports, rig notes, calibration files and code on this Mac", el: Files },
  guide: { title: "Guide", sub: "How to set up a rig and use each page", el: Guide },
};

export function App() {
  const route = useRoute();
  const assist = useStudio((s) => s.assist.open);
  // Esc to stop is registered by studio.start(), outside React, so a render crash cannot remove it.
  // Cmd+J (Ctrl+J off the Mac) opens and closes Claude from anywhere.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "j") {
        e.preventDefault();
        studio.toggleAssistant();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => { setTheme(getTheme()); studio.start(); }, []);

  const page = PAGES[route];
  const Page = page.el;
  return (
    <Tooltip.Provider delayDuration={400}>
      <div className={`shell ${assist ? "has-assist" : ""}`}>
        <Sidebar />
        <div className="main">
          <TopBar route={route} title={page.title} sub={page.sub} onStop={() => studio.stop()} />
          <main className="content" key={route}>
            <Boundary what={`The ${page.title} page`}><Page /></Boundary>
          </main>
          <Boundary what="The terminal"><TerminalDock /></Boundary>
        </div>
        {assist && (
          <Boundary what="Claude" className="assist">
            <Suspense fallback={<aside className="assist" aria-busy />}><AssistantPanel /></Suspense>
          </Boundary>
        )}
      </div>
    </Tooltip.Provider>
  );
}
