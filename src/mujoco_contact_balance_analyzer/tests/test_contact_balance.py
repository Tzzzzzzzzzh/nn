from __future__ import annotations
import sys, tempfile
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT))
from contact_balance import analyze, load_rows, run
def test_balance_action() -> None:
    rows=analyze(load_rows()); assert any(r["balance_action"]=="recover_posture" for r in rows)
def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        m=run(Path(tmp)); assert m["recover_posture_frames"]>0; assert (Path(tmp)/"mujoco_contact_balance_curve.png").exists()
if __name__=="__main__": test_balance_action(); test_exports(); print("mujoco_contact_balance_analyzer tests passed")
