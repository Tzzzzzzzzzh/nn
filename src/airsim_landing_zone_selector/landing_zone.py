"""AirSim depth-grid landing zone selector."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).with_name("sample_data") / "airsim_depth_grid.csv"

def generate(path: Path = DATA) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(715); grid=8+0.08*rng.normal(size=(12,12))
    grid[2:5,2:5]-=2.2; grid[7:10,8:11]-=1.7; grid[6:9,3:6]+=0.05
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerows([[round(float(v),3) for v in row] for row in grid])

def load_grid(path: Path = DATA) -> np.ndarray:
    if not path.exists(): generate(path)
    with path.open(newline="", encoding="utf-8") as f:
        return np.array([[float(v) for v in row] for row in csv.reader(f)])

def evaluate(grid: np.ndarray, window: int = 3) -> list[dict[str, float | int | str]]:
    rows=[]
    for y in range(grid.shape[0]-window+1):
        for x in range(grid.shape[1]-window+1):
            patch=grid[y:y+window,x:x+window]
            flatness=float(np.std(patch)); clearance=float(np.mean(patch)); slope=float(np.max(patch)-np.min(patch))
            score=max(0,1-flatness/0.35)*0.45 + min(clearance/8.0,1.2)*0.35 + max(0,1-slope/0.8)*0.20
            label="best" if score>.92 else "safe" if score>.78 else "reject"
            rows.append({"x":x+1,"y":y+1,"clearance":round(clearance,3),"flatness":round(flatness,3),"slope":round(slope,3),"landing_score":round(score,4),"decision":label})
    return sorted(rows,key=lambda r:float(r["landing_score"]),reverse=True)

def plot(grid: np.ndarray, rows: list[dict[str,float|int|str]], output: Path) -> list[Path]:
    output.mkdir(parents=True,exist_ok=True); paths=[]; best=rows[0]
    p=output/"airsim_depth_grid_landing.png"; plt.figure(figsize=(6,5.4)); plt.imshow(grid,cmap="viridis"); plt.colorbar(label="depth/clearance"); plt.scatter([int(best["x"])+1],[int(best["y"])+1],c="red",s=120,label="selected zone"); plt.title("AirSim depth grid and selected landing zone"); plt.legend(); plt.tight_layout(); plt.savefig(p,dpi=180); plt.close(); paths.append(p)
    p=output/"airsim_landing_score_rank.png"; top=rows[:10]; plt.figure(figsize=(7.2,4.8)); plt.bar(range(len(top)),[float(r["landing_score"]) for r in top],color="#2f80ed"); plt.xticks(range(len(top)),[f'({r["x"]},{r["y"]})' for r in top],rotation=30); plt.ylabel("score"); plt.title("Top landing zone scores"); plt.tight_layout(); plt.savefig(p,dpi=180); plt.close(); paths.append(p); return paths

def run(output: Path) -> dict[str,object]:
    grid=load_grid(); rows=evaluate(grid); files=plot(grid,rows,output)
    csv_path=output/"airsim_landing_zone_scores.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    files.append(csv_path); best=rows[0]
    report={"source":"AirSim depth camera grid","grid_size":list(grid.shape),"candidate_zones":len(rows),"best_zone":[best["x"],best["y"]],"best_score":best["landing_score"],"safe_zones":sum(r["decision"]!="reject" for r in rows),"generated_files":[p.name for p in files]}
    (output/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8"); return report

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path("docs/pr_assets/airsim_landing_zone_selector")); args=parser.parse_args(); print(json.dumps(run(args.output),indent=2,ensure_ascii=False))
if __name__=="__main__": main()
