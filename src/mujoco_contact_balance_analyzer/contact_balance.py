"""MuJoCo contact force balance analyzer."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DATA=Path(__file__).with_name("sample_data")/"mujoco_contact_forces.csv"

def load_rows(path: Path=DATA) -> list[dict[str,float]]:
    with path.open(newline="",encoding="utf-8") as f: return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]

def analyze(rows: list[dict[str,float]]) -> list[dict[str,float|str]]:
    out=[]
    for r in rows:
        left=r["left_front_n"]+r["left_rear_n"]; right=r["right_front_n"]+r["right_rear_n"]; total=left+right
        imbalance=abs(left-right)/total; com_shift=np.hypot(r["com_x_m"],r["com_y_m"]); roll=abs(r["torso_roll_deg"])
        risk=0.42*min(imbalance/0.32,1.4)+0.32*min(com_shift/0.30,1.3)+0.26*min(roll/10,1.4)
        action="recover_posture" if risk>.72 else "adjust_support" if risk>.45 else "stable_walk"
        out.append({**r,"left_force_n":round(left,2),"right_force_n":round(right,2),"force_imbalance":round(imbalance,4),"balance_risk":round(float(risk),4),"balance_action":action})
    return out

def plot(rows: list[dict[str,float|str]], output: Path) -> list[Path]:
    output.mkdir(parents=True,exist_ok=True); t=[float(r["time_s"]) for r in rows]; risk=[float(r["balance_risk"]) for r in rows]; imb=[float(r["force_imbalance"]) for r in rows]; paths=[]
    p=output/"mujoco_contact_balance_curve.png"; plt.figure(figsize=(8,4.8)); plt.plot(t,risk,marker="o",label="balance risk",color="#eb5757"); plt.plot(t,imb,marker="s",label="force imbalance",color="#2f80ed"); plt.xlabel("time (s)"); plt.title("MuJoCo contact balance risk"); plt.grid(True,linestyle="--",alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(p,dpi=180); plt.close(); paths.append(p)
    p=output/"mujoco_force_distribution.png"; plt.figure(figsize=(7.5,4.8)); plt.plot(t,[float(r["left_force_n"]) for r in rows],label="left force"); plt.plot(t,[float(r["right_force_n"]) for r in rows],label="right force"); plt.xlabel("time (s)"); plt.ylabel("force (N)"); plt.title("Left-right contact force distribution"); plt.grid(True,linestyle="--",alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(p,dpi=180); plt.close(); paths.append(p); return paths

def run(output: Path) -> dict[str,object]:
    rows=analyze(load_rows()); files=plot(rows,output); csv_path=output/"mujoco_contact_balance_scores.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    files.append(csv_path); risk=np.array([float(r["balance_risk"]) for r in rows])
    report={"source":"MuJoCo foot contact force log","records":len(rows),"max_balance_risk":round(float(risk.max()),4),"recover_posture_frames":sum(r["balance_action"]=="recover_posture" for r in rows),"generated_files":[p.name for p in files]}
    (output/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8"); return report

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path("docs/pr_assets/mujoco_contact_balance_analyzer")); args=parser.parse_args(); print(json.dumps(run(args.output),indent=2,ensure_ascii=False))
if __name__=="__main__": main()
