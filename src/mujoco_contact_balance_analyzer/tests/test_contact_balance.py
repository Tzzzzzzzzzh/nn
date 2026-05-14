from __future__ import annotations
import sys, tempfile
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT))
from contact_balance import analyze, load_rows, parse_mjcf, run
def test_balance_action() -> None:
    rows=analyze(load_rows()); assert any(r["balance_action"]=="recover_posture" for r in rows)
    mjcf=parse_mjcf(); assert mjcf["sensor_count"] >= 5; assert mjcf["contact_geom_count"] == 4
def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        m=run(Path(tmp)); assert m["source"] == "MuJoCo MJCF model, cfrc_ext contact force log and sensordata export"; assert m["recover_posture_frames"]>0; assert (Path(tmp)/"mujoco_contact_balance_curve.png").exists(); assert (Path(tmp)/"mujoco_support_polygon_replay.png").exists()
if __name__=="__main__": test_balance_action(); test_exports(); print("mujoco_contact_balance_analyzer tests passed")
