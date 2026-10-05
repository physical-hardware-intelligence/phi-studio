import * as Tooltip from "@radix-ui/react-tooltip";
import { Suspense, lazy, useEffect, type ComponentType } from "react";
import { Boundary } from "./components/Boundary";
import { Sidebar } from "./components/Sidebar";
import { TerminalDock } from "./components/TerminalDock";
import { TopBar } from "./components/TopBar";
import { getTheme, setTheme, studio, useRoute, useStudio, type Route } from "./lib/studio";
import { Align } from "./pages/Align";
import { Calibrate } from "./pages/Calibrate";
import { Checks } from "./pages/Checks";
import { Evaluate } from "./pages/Evaluate";
import { Files } from "./pages/Files";
import { Guide } from "./pages/Guide";
import { Home } from "./pages/Home";
import { Onboard } from "./pages/Onboard";
import { Settings } from "./pages/Settings";
import { Models } from "./pages/Models";
import { Policy } from "./pages/Policy";
import { Record } from "./pages/Record";
import { Scene } from "./pages/Scene";
import { Setup } from "./pages/Setup";
import { Teleop } from "./pages/Teleop";
import { Train } from "./pages/Train";

// WHY lazy: the data pages carry three.js and the 3D model code; pages that never show them do not pay.
const Data = lazy(() => import("./pages/Data").then((m) => ({ default: m.Data })));
const Issues = lazy(() => import("./pages/Issues").then((m) => ({ default: m.Issues })));
// WHY lazy: the panel carries the markdown renderer, a third of the bundle, and most sessions never open it.
const AssistantPanel = lazy(() => import("./components/Assistant").then((m) => ({ default: m.AssistantPanel })));

const PAGES: Record<Route, { title: string; sub: string; el: ComponentType }> = {
  overview: { title: "Home", sub: "Your rig and your latest data", el: Home },
  teleop: { title: "Teleop", sub: "Drive the followers with the leaders", el: Teleop },
  scene: { title: "3D view", sub: "The arms, live", el: Scene },
  align: { title: "Align", sub: "Each camera back where the training data was recorded", el: Align },
  record: { title: "Record", sub: "Teleop episodes into a LeRobot dataset", el: Record },
  data: { title: "Datasets", sub: "Every recording on this Mac", el: Data },
  issues: { title: "Issues", sub: "Notes and excluded episodes", el: Issues },
  train: { title: "Train", sub: "On this Mac or the cluster", el: Train },
  models: { title: "Models", sub: "Policies from Hugging Face", el: Models },
  policy: { title: "Run", sub: "A policy on the arms", el: Policy },
  evaluate: { title: "Evaluate", sub: "Score episodes, get a success rate", el: Evaluate },
  settings: { title: "Settings", sub: "Rig, setup, calibration, checks, files", el: Settings },
  onboard: { title: "Rig setup", sub: "Name, arms, calibration, cameras", el: Onboard },
  setup: { title: "Advanced setup", sub: "Every LeRobot step, in order", el: Setup },
  calibrate: { title: "Calibrate", sub: "Homing offsets and joint ranges, by hand", el: Calibrate },
  checks: { title: "Checks", sub: "This Mac, the rig and Studio", el: Checks },
  files: { title: "Files", sub: "Ports, calibration files and code", el: Files },
  guide: { title: "Guide", sub: "How each page works", el: Guide },
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
            <Boundary what={`The ${page.title} page`}>
              <Suspense fallback={<div className="page"><div className="empty">Loading</div></div>}><Page /></Suspense>
            </Boundary>
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
