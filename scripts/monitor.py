#!/usr/bin/env python3
"""
Real-time Terminal Progress Dashboard for MARTINI 3 GROMACS MD Triplicate Runs
Parses production.log and em.log across data/md_runs/<Target>/<Replicate>/ to calculate:
- Active Pipeline Stage per Replicate
- Completed Simulated Nanoseconds (out of 500 ns target)
- Real-time Simulation Speed (ns/day)
- Accurate Target ETA (Hours/Minutes)
"""

import os
import sys
import time
import glob
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_RUNS_DIR = os.path.join(BASE_DIR, "data", "md_runs")
TARGETS = ["KRAS_G12D", "cMYC_MAX", "Mut_p53", "PTPN11"]
REPLICATES = ["rep1", "rep2", "rep3"]
TOTAL_NS = 500.0


def parse_replicate_status(target: str, rep: str) -> dict:
    rep_dir = os.path.join(MD_RUNS_DIR, target, rep)
    prod_log = os.path.join(rep_dir, "production.log")
    eq_log = os.path.join(rep_dir, "eq.log")
    target_dir = os.path.join(MD_RUNS_DIR, target)
    em_log = os.path.join(target_dir, "em.log")
    
    status = {
        "stage": "Pending",
        "ns_simulated": 0.0,
        "speed_ns_day": 0.0,
        "eta_str": "N/A",
        "progress_pct": 0.0,
        "details": "Initializing..."
    }

    if not os.path.exists(rep_dir):
        return status

    # Check Production MD
    if os.path.exists(prod_log):
        status["stage"] = "Production MD"
        try:
            with open(prod_log, "r", errors="ignore") as f:
                lines = f.readlines()

            step_re = re.compile(r"^\s*(\d+)\s+([\d\.]+)\s*$")
            perf_re = re.compile(r"Performance:\s+([\d\.]+)\s+ns/day")

            last_ns = 0.0
            last_speed = 0.0

            for l in lines:
                m_step = step_re.match(l)
                if m_step:
                    time_ps = float(m_step.group(2))
                    last_ns = time_ps / 1000.0
                
                m_perf = perf_re.search(l)
                if m_perf:
                    last_speed = float(m_perf.group(1))

            status["ns_simulated"] = last_ns
            status["progress_pct"] = min(100.0, (last_ns / TOTAL_NS) * 100.0)
            status["speed_ns_day"] = last_speed

            if last_speed > 0:
                remaining_ns = max(0.0, TOTAL_NS - last_ns)
                remaining_hours = (remaining_ns / last_speed) * 24.0
                h = int(remaining_hours)
                m = int((remaining_hours - h) * 60)
                status["eta_str"] = f"{h}h {m}m"
            elif last_ns > 0:
                status["eta_str"] = "Calculating..."
            else:
                status["eta_str"] = "Starting..."

            status["details"] = f"{last_ns:.2f} / 500 ns ({status['progress_pct']:.1f}%)"

        except Exception as e:
            status["details"] = f"Reading log..."
        return status

    # Check NPT Equilibration
    if os.path.exists(eq_log):
        status["stage"] = "NPT Equilibration"
        status["details"] = "1 ns NPT (10 fs)"
        status["eta_str"] = "~1 min"
        return status

    # Check Energy Minimization
    if os.path.exists(em_log):
        status["stage"] = "Energy Minimization"
        status["details"] = "Minimizing Target"
        status["eta_str"] = "~1-2 min"
        return status

    return status


def render_ascii_bar(pct: float, width: int = 20) -> str:
    filled = int(round((pct / 100.0) * width))
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}]"


def display_dashboard():
    os.system("clear" if os.name == "posix" else "cls")
    print("=" * 96)
    print("                GROMACS MARTINI 3 MD TRIPLICATE (N=3) REAL-TIME DASHBOARD                ")
    print("=" * 96)
    print(f"{'Target / Replicate':<18} | {'Current Stage':<21} | {'Progress / Details':<25} | {'Speed':<10} | {'ETA':<8}")
    print("-" * 96)

    for target in TARGETS:
        for rep in REPLICATES:
            st = parse_replicate_status(target, rep)
            t_rep_str = f"{target} / {rep}"
            stage_str = st["stage"]
            details_str = st["details"]
            speed_str = f"{st['speed_ns_day']:.1f} ns/d" if st['speed_ns_day'] > 0 else "---"
            eta_str = st["eta_str"]

            print(f"{t_rep_str:<18} | {stage_str:<21} | {details_str:<25} | {speed_str:<10} | {eta_str:<8}")

    print("=" * 96)
    print(" Status: Triplicate N=3 CUDA GPU pipeline active (nice -n 19). Press Ctrl+C to exit.")
    print("=" * 96)


def monitor_loop():
    try:
        while True:
            display_dashboard()
            time.sleep(15)
    except KeyboardInterrupt:
        print("\nExiting dashboard monitor. MD simulations continue in background.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        monitor_loop()
    else:
        display_dashboard()
