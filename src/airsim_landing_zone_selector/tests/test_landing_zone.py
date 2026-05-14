from __future__ import annotations
import sys, tempfile
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT))
from landing_zone import evaluate, load_grid, run
def test_selects_zone() -> None:
    rows=evaluate(load_grid()); assert rows[0]["decision"] in ("best","safe"); assert float(rows[0]["landing_score"])>.78
def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        m=run(Path(tmp)); assert m["safe_zones"]>0; assert (Path(tmp)/"airsim_depth_grid_landing.png").exists()
if __name__=="__main__": test_selects_zone(); test_exports(); print("airsim_landing_zone_selector tests passed")
