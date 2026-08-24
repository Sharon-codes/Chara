#!/usr/bin/env python3
"""
High-Performance Single-Job GPU Auto-Scheduler Daemon (Maximum ns/day Speed Mode)
"""

import os
import sys
import time
import subprocess
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_RUNS_DIR = os.path.join(BASE_DIR, "data", "md_runs")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CONDA_BIN_DIR = "/home/sharon/env_md/bin"
GMX_BIN = os.path.join(CONDA_BIN_DIR, "gmx") if os.path.exists(os.path.join(CONDA_BIN_DIR, "gmx")) else "gmx"

TARGETS = ["KRAS_G12D", "cMYC_MAX", "Mut_p53", "PTPN11"]
REPLICATES = ["rep1", "rep2", "rep3"]
MAX_CONCURRENT_JOBS = 1

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "daemon_runner.log"), mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)


def get_active_mdrun_count() -> int:
    """Count how many gmx mdrun processes are currently running on the GPU."""
    count = 0
    try:
        res = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if ("mdrun" in line or "bin.AVX2_256/gmx" in line) and ("production" in line):
                if "grep" not in line and "daemon_runner" not in line:
                    count += 1
    except Exception as e:
        logging.error(f"Error checking ps aux: {e}")
    return count


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


def is_replicate_running(rep_dir: str) -> bool:
    """Check if a specific replicate directory currently has an active mdrun process."""
    try:
        res = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if rep_dir in line and "grep" not in line:
                return True
    except Exception:
        pass
    return False


def launch_replicate_mdrun(rep_dir: str, target: str, rep: str):
    """Launch GROMACS production mdrun with maximum GPU acceleration."""
    tpr_file = os.path.join(rep_dir, "production.tpr")
    cpt_file = os.path.join(rep_dir, "production.cpt")

    if not os.path.exists(tpr_file):
        logging.warning(f"production.tpr not found in {rep_dir}, skipping...")
        return

    env = os.environ.copy()
    env["PATH"] = f"{CONDA_BIN_DIR}:" + env.get("PATH", "")

    if os.path.exists(cpt_file):
        cmd = [GMX_BIN, "mdrun", "-deffnm", "production", "-nb", "gpu", "-bonded", "gpu", "-pin", "on", "-cpi", "production.cpt", "-append"]
    else:
        cmd = [GMX_BIN, "mdrun", "-deffnm", "production", "-nb", "gpu", "-bonded", "gpu", "-pin", "on"]

    logging.info(f"=== LAUNCHING GPU SINGLE-JOB RUN: {target} / {rep} ===")
    logging.info(f"Command: {' '.join(cmd)}")
    subprocess.Popen(cmd, cwd=rep_dir, env=env)


def run_daemon_loop():
    """Main continuous daemon loop supporting maximum speed single-job mode."""
    logging.info("Starting High-Performance Single-Job GPU Auto-Scheduler (Maximum Speed Mode)...")
    
    while True:
        try:
            active_count = get_active_mdrun_count()
            if active_count < MAX_CONCURRENT_JOBS:
                slots_needed = MAX_CONCURRENT_JOBS - active_count
                logging.info(f"Active GPU jobs: {active_count}/{MAX_CONCURRENT_JOBS}. Launching {slots_needed} job...")
                
                launched_count = 0
                for target in TARGETS:
                    for rep in REPLICATES:
                        rep_dir = os.path.join(MD_RUNS_DIR, target, rep)
                        if os.path.exists(rep_dir):
                            if not is_replicate_complete(rep_dir) and not is_replicate_running(rep_dir):
                                launch_replicate_mdrun(rep_dir, target, rep)
                                launched_count += 1
                                if launched_count >= slots_needed:
                                    break
                    if launched_count >= slots_needed:
                        break
            else:
                pass
        except Exception as e:
            logging.error(f"Error in daemon loop: {e}")

        time.sleep(20)


if __name__ == "__main__":
    run_daemon_loop()
