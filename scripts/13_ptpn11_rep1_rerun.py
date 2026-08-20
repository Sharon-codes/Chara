#!/usr/bin/env python3
"""
Auto-Rerun Script: PTPN11 / rep1 (Full 500 ns)
Waits for PTPN11 / rep3 to complete, then:
  1. Archives the failed 1.0 ns pilot run files
  2. Recompiles production.tpr from eq.cpt + unique gen_seed = -1
  3. Launches 500 ns production run solo on GPU (MAX BOOST MODE)
"""

import os
import subprocess
import logging
import time
import shutil

BASE_DIR = "/home/sharon/Desktop/Sharon"
REP1_DIR = os.path.join(BASE_DIR, "data", "md_runs", "PTPN11", "rep1")
REP3_DIR = os.path.join(BASE_DIR, "data", "md_runs", "PTPN11", "rep3")
GMX_BIN  = "/home/sharon/env_md/bin/gmx"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "ptpn11_rep1_rerun.log"), mode="a"),
        logging.StreamHandler()
    ]
)


def rep3_is_complete() -> bool:
    gro = os.path.join(REP3_DIR, "production.gro")
    log = os.path.join(REP3_DIR, "production.log")
    if os.path.exists(gro) and os.path.getsize(gro) > 100000:
        return True
    if os.path.exists(log):
        try:
            with open(log, errors="ignore") as f:
                content = f.read()
                if "Finished mdrun" in content:
                    return True
        except Exception:
            pass
    return False


def archive_pilot_run():
    archive_dir = os.path.join(REP1_DIR, "pilot_run_archive")
    os.makedirs(archive_dir, exist_ok=True)
    for fname in ["production.cpt", "production.edr", "production.log",
                  "production.xtc", "production.gro"]:
        src = os.path.join(REP1_DIR, fname)
        if os.path.exists(src):
            shutil.move(src, os.path.join(archive_dir, fname))
            logging.info(f"Archived: {fname} -> pilot_run_archive/")


def write_fresh_mdp():
    mdp_path = os.path.join(REP1_DIR, "production.mdp")
    mdp_content = """\
integrator              = md
dt                      = 0.020
nsteps                  = 25000000
nstxout-compressed      = 25000
nstlog                  = 5000
nstenergy               = 5000
cutoff-scheme           = Verlet
coulombtype             = reaction-field
rcoulomb                = 1.1
epsilon_r               = 15
epsilon_rf              = 0
vdw_type                = cutoff
vdw-modifier            = Potential-shift-verlet
rvdw                    = 1.1
tcoupl                  = v-rescale
tc-grps                 = Protein Non-Protein
tau_t                   = 1.0 1.0
ref_t                   = 310 310
pcoupl                  = c-rescale
pcoupltype              = isotropic
tau_p                   = 12.0
ref_p                   = 1.0
compressibility         = 3e-4
comm-grps               = Protein Non-Protein
gen_vel                 = yes
gen_temp                = 310
gen_seed                = -1
pbc                     = xyz
"""
    with open(mdp_path, "w") as f:
        f.write(mdp_content)
    logging.info(f"Written fresh production.mdp with gen_seed = -1")


def recompile_tpr():
    cmd = [GMX_BIN, "grompp",
           "-f", "production.mdp",
           "-c", "eq.cpt",
           "-p", "topol.top",
           "-o", "production.tpr",
           "-maxwarn", "5"]
    logging.info(f"Recompiling production.tpr: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REP1_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        # Fallback: try using eq.tpr as reference structure
        logging.warning("eq.cpt failed, trying with eq.tpr as reference...")
        cmd[4] = "eq.tpr"
        res = subprocess.run(cmd, cwd=REP1_DIR, capture_output=True, text=True)
    if res.returncode == 0:
        logging.info("[✓] production.tpr recompiled successfully!")
    else:
        logging.error(f"grompp failed: {res.stderr}")
        raise RuntimeError("grompp failed")


def launch_production():
    cmd = [GMX_BIN, "mdrun",
           "-deffnm", "production",
           "-nb", "gpu", "-bonded", "gpu",
           "-pin", "on", "-ntmpi", "1", "-ntomp", "32"]
    logging.info("=" * 65)
    logging.info(" LAUNCHING PTPN11 / rep1 RERUN — MAX GPU BOOST MODE")
    logging.info(f" Command: {' '.join(cmd)}")
    logging.info("=" * 65)
    res = subprocess.run(cmd, cwd=REP1_DIR)
    logging.info(f"[✓] PTPN11 / rep1 RERUN COMPLETE with exit code {res.returncode}")


def main():
    logging.info("=" * 65)
    logging.info(" PTPN11 / rep1 RERUN WATCHDOG ACTIVE — Waiting for rep3...")
    logging.info("=" * 65)

    # Poll every 60 seconds until rep3 is done
    while not rep3_is_complete():
        logging.info("rep3 still running... checking again in 60s")
        time.sleep(60)

    logging.info("[✓] PTPN11 / rep3 is COMPLETE! Initiating rep1 rerun now...")
    time.sleep(5)  # Brief pause for filesystem sync

    archive_pilot_run()
    write_fresh_mdp()
    recompile_tpr()
    launch_production()


if __name__ == "__main__":
    main()
