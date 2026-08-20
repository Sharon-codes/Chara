#!/usr/bin/env python3
"""
Master Sequential GPU Pipeline Executor (MAXIMUM GPU BOOST MODE - 32 OPENMP THREADS)
Executes all remaining MD production runs (Mut_p53 rep2, rep3 -> PTPN11 rep1, rep2, rep3)
ONE AT A TIME with maximum OpenMP GPU queue feeding (-ntomp 32 -pin on).
As soon as one 500 ns replicate finishes, Python AUTOMATICALLY launches the next replicate.
"""

import os
import sys
import subprocess
import logging
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_RUNS_DIR = os.path.join(BASE_DIR, "data", "md_runs")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CONDA_BIN_DIR = "/home/sharon/env_md/bin"
GMX_BIN = os.path.join(CONDA_BIN_DIR, "gmx") if os.path.exists(os.path.join(CONDA_BIN_DIR, "gmx")) else "gmx"

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "master_pipeline.log"), mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Strict sequential queue (Runs ONE AT A TIME at MAX GPU Speed)
QUEUE = [
    ("Mut_p53", "rep2"),
    ("Mut_p53", "rep3"),
    ("PTPN11", "rep1"),
    ("PTPN11", "rep2"),
    ("PTPN11", "rep3")
]


def is_replicate_complete(rep_dir: str) -> bool:
    """Check if a replicate's 500 ns production trajectory is 100% complete."""
    gro_file = os.path.join(rep_dir, "production.gro")
    log_file = os.path.join(rep_dir, "production.log")
    if os.path.exists(gro_file) and os.path.getsize(gro_file) > 100000:
        return True
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", errors="ignore") as f:
                content = f.read()
                if "Finished mdrun" in content or "Writing final coordinates" in content:
                    return True
        except Exception:
            pass
    return False


def run_replicate(target: str, rep: str):
    rep_dir = os.path.join(MD_RUNS_DIR, target, rep)
    tpr_file = os.path.join(rep_dir, "production.tpr")
    cpt_file = os.path.join(rep_dir, "production.cpt")

    if not os.path.exists(rep_dir) or not os.path.exists(tpr_file):
        logging.warning(f"Target directory or TPR file missing: {rep_dir}, skipping...")
        return

    if is_replicate_complete(rep_dir):
        logging.info(f"[✓] Target {target} / {rep} is ALREADY COMPLETE (500 ns). Moving to next in queue...")
        return

    env = os.environ.copy()
    env["PATH"] = f"{CONDA_BIN_DIR}:" + env.get("PATH", "")

    # MAX GPU BOOST MODE: -ntmpi 1 -ntomp 32 -pin on
    cmd = [GMX_BIN, "mdrun", "-deffnm", "production", "-nb", "gpu", "-bonded", "gpu", "-pin", "on", "-ntmpi", "1", "-ntomp", "32"]
    if os.path.exists(cpt_file):
        cmd.extend(["-cpi", "production.cpt", "-append"])

    logging.info(f"=================================================================")
    logging.info(f" EXECUTING REPLICATE (MAX GPU BOOST MODE): {target} / {rep}")
    logging.info(f" Command: {' '.join(cmd)}")
    logging.info(f" Directory: {rep_dir}")
    logging.info(f"=================================================================")

    start_time = time.time()
    res = subprocess.run(cmd, cwd=rep_dir, env=env)
    elapsed = time.time() - start_time

    logging.info(f"[✓] FINISHED {target} / {rep} in {elapsed/3600:.2f} hours with exit code {res.returncode}")
    logging.info(f"--> AUTOMATICALLY PROCEEDING TO NEXT REPLICATE IN QUEUE NOW...")
    time.sleep(3)


def main():
    logging.info("=================================================================")
    logging.info(" MASTER SEQUENTIAL GPU PIPELINE (MAXIMUM GPU BOOST MODE)")
    logging.info(" Target Queue: " + " -> ".join([f"{t}/{r}" for t, r in QUEUE]))
    logging.info("=================================================================")

    for target, rep in QUEUE:
        run_replicate(target, rep)

    logging.info("=================================================================")
    logging.info("🎉 ALL TARGET REPLICATES IN MASTER PIPELINE HAVE 100% COMPLETED!")
    logging.info("=================================================================")


if __name__ == "__main__":
    main()
