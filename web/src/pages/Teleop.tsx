import { ActionBar } from "../components/ActionBar";
import { ArmsPanel } from "../components/ArmsPanel";
import { JointTable } from "../components/JointTable";
import { MockPanel } from "../components/MockPanel";
import { Notices } from "../components/Notices";
import { ViewSwitch } from "../scene/ViewSwitch";

export function Teleop() {
  return (
    <div className="page teleop">
      <ActionBar activity="teleop" />
      <Notices />
      <div className="work-grid">
        <div className="col">
          <ViewSwitch page="teleop" />
          <JointTable />
        </div>
        <div className="col rail">
          <ArmsPanel />
          <MockPanel />
        </div>
      </div>
    </div>
  );
}
