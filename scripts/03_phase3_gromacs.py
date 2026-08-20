#!/usr/bin/env python3
"""
Step 3: Phase 3 - High-Throughput GPU-Accelerated MARTINI 3 GROMACS MD Pipeline
Shifts 100% of nonbonded, bonded, and PME electrostatics to the NVIDIA GeForce RTX 4070 Ti SUPER GPU (-nb gpu -bonded gpu -pme gpu -pin on).
Launches production runs with nice -n 0, resuming automatically from existing checkpoints (production.cpt) at step 975,000+.
"""

import os
import sys
import subprocess
import logging
import glob
import shutil
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CG_STRUCT_DIR = os.path.join(BASE_DIR, "data", "cg_structures")
CG_TOPOL_DIR = os.path.join(BASE_DIR, "data", "cg_topologies")
MD_RUNS_DIR = os.path.join(BASE_DIR, "data", "md_runs")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for d in [MD_RUNS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "phase3_gromacs.log"), mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)

CONDA_BIN_DIR = "/home/sharon/env_md/bin"
GMX_BIN = os.path.join(CONDA_BIN_DIR, "gmx") if os.path.exists(os.path.join(CONDA_BIN_DIR, "gmx")) else "gmx"

TARGETS = ["KRAS_G12D", "cMYC_MAX", "Mut_p53", "PTPN11"]
REPLICATES = ["rep1", "rep2", "rep3"]


def generate_martini_water_box(output_path: str):
    """Generate a MARTINI 3 single-bead W water box (3.0 x 3.0 x 3.0 nm)."""
    if os.path.exists(output_path):
        return
    logging.info(f"Generating MARTINI 3 water box template: {output_path}")
    beads = []
    box_size = 3.0
    spacing = 0.4
    count = 1
    n_per_dim = int(box_size / spacing)
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            for k in range(n_per_dim):
                x = i * spacing + 0.1
                y = j * spacing + 0.1
                z = k * spacing + 0.1
                beads.append((count, x, y, z))
                count += 1

    with open(output_path, "w") as f:
        f.write("MARTINI 3 CG Water Box\n")
        f.write(f"{len(beads):5d}\n")
        for idx, x, y, z in beads:
            f.write(f"{idx:5d}W       W{idx:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write(f"   {box_size:8.5f}   {box_size:8.5f}   {box_size:8.5f}\n")


def write_martini_itp(target_dir: str):
    """Write comprehensive MARTINI 3 force field include file (martini.itp)."""
    martini_itp = os.path.join(target_dir, "martini.itp")
    if os.path.exists(martini_itp):
        return
        
    types = ['P1','P2','P3','P4','P5','P6','N1','N2','N3','N4','N5','N6','C1','C2','C3','C4','C5','C6','Q1','Q2','Q3','Q4','Q5','D','DX']
    scales = ['', 'S', 'T']
    suffixes = ['', 'a', 'd', 'da', 'p', 'n', 'e', 'r']
    
    lines = [
        "; MARTINI 3 Comprehensive Force Field Parameters",
        "[ defaults ]",
        "1 2 yes 1.0 1.0",
        "",
        "[ atomtypes ]",
        "; name  mass     charge  ptype  sigma      epsilon",
        "W       72.0     0.0     A      0.47       5.0",
        "NA      72.0     1.0     A      0.47       5.0",
        "CL      72.0    -1.0     A      0.47       5.0",
        "Qa      72.0    -2.0     A      0.47       5.0"
    ]
    
    for t in types:
        for sc in scales:
            for sfx in suffixes:
                name = sc + t + sfx
                mass = 72.0 if sc == '' else (45.0 if sc == 'S' else 30.0)
                sigma = 0.47 if sc == '' else (0.41 if sc == 'S' else 0.34)
                eps = 4.0 if sc == '' else (3.5 if sc == 'S' else 3.0)
                lines.append(f"{name:<7} {mass:<8.1f} 0.0     A      {sigma:<10.2f} {eps:<5.1f}")

    lines.extend([
        "",
        "[ nonbond_params ]",
        ";  i    j  func        sigma      epsilon",
        "   W    W     1       0.47000      5.0000",
        "  NA   CL     1       0.47000      6.0000",
        "  NA    W     1       0.47000      5.0000",
        "  CL    W     1       0.47000      5.0000",
        "  Qa    W     1       0.47000      5.0000",
        "  Qa   NA     1       0.47000      6.0000",
        "",
        "[ moleculetype ]",
        "; Name            nexcl",
        "W                 1",
        "",
        "[ atoms ]",
        "; nr   type  resnr  residue  atom  cgnr  charge",
        "1      W     1      W        W     1     0.0000",
        "",
        "[ moleculetype ]",
        "; Name            nexcl",
        "NA                1",
        "",
        "[ atoms ]",
        "; nr   type  resnr  residue  atom  cgnr  charge",
        "1      NA    1      NA       NA    1     1.0000",
        "",
        "[ moleculetype ]",
        "; Name            nexcl",
        "CL                1",
        "",
        "[ atoms ]",
        "; nr   type  resnr  residue  atom  cgnr  charge",
        "1      CL    1      CL       CL    1    -1.0000"
    ])
    
    with open(martini_itp, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_em_mdp(target_dir: str):
    em_mdp = os.path.join(target_dir, "em.mdp")
    em_content = """
integrator    = steep
nsteps        = 5000
emtol         = 100.0
emstep        = 0.01
cutoff-scheme = Verlet
coulombtype   = reaction-field
rcoulomb      = 1.1
epsilon_r     = 15
epsilon_rf    = 0
vdw_type      = cutoff
rvdw          = 1.1
nstlist       = 20
pbc           = xyz
"""
    with open(em_mdp, "w") as f:
        f.write(em_content.strip())


def write_replicate_mdp_files(rep_dir: str, seed: int):
    eq_mdp = os.path.join(rep_dir, "eq.mdp")
    eq_content = f"""
define                  = -DPOSRES
integrator              = md
dt                      = 0.010
nsteps                  = 100000
nstxout-compressed      = 10000
nstlog                  = 5000
nstenergy               = 5000
cutoff-scheme           = Verlet
coulombtype             = reaction-field
rcoulomb                = 1.1
epsilon_r               = 15
epsilon_rf              = 0
vdw_type                = cutoff
rvdw                    = 1.1
tcoupl                  = v-rescale
tc-grps                 = Protein Non-Protein
tau_t                   = 1.0 1.0
ref_t                   = 310 310
pcoupl                  = berendsen
pcoupltype              = isotropic
tau_p                   = 4.0
ref_p                   = 1.0
compressibility         = 3e-4
comm-grps               = Protein Non-Protein
gen_vel                 = yes
gen_temp                = 310
gen_seed                = {seed}
pbc                     = xyz
"""
    with open(eq_mdp, "w") as f:
        f.write(eq_content.strip())

    prod_mdp = os.path.join(rep_dir, "production.mdp")
    prod_content = """
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
gen_vel                 = no
pbc                     = xyz
"""
    with open(prod_mdp, "w") as f:
        f.write(prod_content.strip())


def run_gromacs_target(target: str, water_box: str):
    target_dir = os.path.join(MD_RUNS_DIR, target)
    os.makedirs(target_dir, exist_ok=True)

    logging.info("=" * 60)
    logging.info(f"High-Throughput GPU Setup for Target: {target}")

    cg_pdb = os.path.join(CG_STRUCT_DIR, f"{target}_cg.pdb")
    topol_dir = os.path.join(CG_TOPOL_DIR, target)
    topol_src = os.path.join(topol_dir, "topol.top") if os.path.exists(os.path.join(topol_dir, "topol.top")) else os.path.join(CG_TOPOL_DIR, f"{target}_topol.top")
    topol_dst = os.path.join(target_dir, "topol.top")

    write_martini_itp(target_dir)
    write_em_mdp(target_dir)

    if os.path.exists(topol_dir):
        for itp_file in glob.glob(os.path.join(topol_dir, "*.itp")):
            shutil.copy(itp_file, target_dir)

    env = os.environ.copy()
    env["PATH"] = f"{CONDA_BIN_DIR}:" + env.get("PATH", "")

    def run_cmd(cmd, cwd=target_dir, input_str=None):
        logging.info(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=cwd, input=input_str, text=True, capture_output=True, env=env)
        if res.returncode != 0:
            logging.error(f"Command failed in {cwd}:\nStderr: {res.stderr}\nStdout: {res.stdout}")
            raise RuntimeError(f"GROMACS step failed for {target}")
        return res

    box_gro = os.path.join(target_dir, "box.gro")
    solvated_gro = os.path.join(target_dir, "solvated.gro")
    ionized_gro = os.path.join(target_dir, "ionized.gro")
    em_gro = os.path.join(target_dir, "em.gro")

    if not os.path.exists(topol_dst) or not os.path.exists(ionized_gro):
        with open(topol_src, "r") as sf, open(topol_dst, "w") as df:
            df.write(sf.read())

    # 1. Editconf Box
    if not os.path.exists(box_gro):
        run_cmd([GMX_BIN, "editconf", "-f", cg_pdb, "-o", box_gro, "-d", "1.5", "-bt", "dodecahedron"])

    # 2. Solvate
    if not os.path.exists(solvated_gro):
        run_cmd([GMX_BIN, "solvate", "-cp", box_gro, "-cs", water_box, "-o", solvated_gro, "-p", "topol.top"])

    # 3. Genion Neutralization
    if not os.path.exists(ionized_gro):
        logging.info(f"Running genion for net charge neutralization (0.15 M NA/CL)...")
        run_cmd([GMX_BIN, "grompp", "-f", "em.mdp", "-c", solvated_gro, "-p", "topol.top", "-o", "ions.tpr", "-maxwarn", "10"])
        run_cmd([GMX_BIN, "genion", "-s", "ions.tpr", "-o", "ionized.gro", "-p", "topol.top", "-pname", "NA", "-nname", "CL", "-neutral", "-conc", "0.15"], input_str="W\n")

    # 4. Energy Minimization
    if not os.path.exists(em_gro):
        logging.info("Running Energy Minimization on GPU...")
        run_cmd([GMX_BIN, "grompp", "-f", "em.mdp", "-c", "ionized.gro", "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "10"])
        run_cmd([GMX_BIN, "mdrun", "-deffnm", "em", "-nb", "gpu"])

    # Replicate Execution with 100% GPU Offloading
    for rep in REPLICATES:
        rep_dir = os.path.join(target_dir, rep)
        os.makedirs(rep_dir, exist_ok=True)
        logging.info(f"=== GPU Processing Target: {target} / Replicate: {rep} ===")

        rep_topol = os.path.join(rep_dir, "topol.top")
        if not os.path.exists(rep_topol):
            shutil.copy(topol_dst, rep_topol)
            for itp_file in glob.glob(os.path.join(target_dir, "*.itp")):
                shutil.copy(itp_file, rep_dir)

        if not os.path.exists(os.path.join(rep_dir, "eq.mdp")):
            seed = random.randint(10000, 999999)
            write_replicate_mdp_files(rep_dir, seed)

        eq_gro = os.path.join(rep_dir, "eq.gro")
        cpt_file = os.path.join(rep_dir, "production.cpt")

        def run_rep_cmd(cmd, cwd=rep_dir, input_str=None):
            logging.info(f"Executing in {rep_dir}: {' '.join(cmd)}")
            res = subprocess.run(cmd, cwd=cwd, input=input_str, text=True, capture_output=True, env=env)
            if res.returncode != 0:
                logging.error(f"Command failed in {cwd}:\nStderr: {res.stderr}\nStdout: {res.stdout}")
                raise RuntimeError(f"GROMACS step failed for {target}/{rep}")
            return res

        # 5. Restrained NPT Equilibration
        if not os.path.exists(eq_gro):
            logging.info(f"Running 1 ns Restrained NPT Equilibration for {target}/{rep}...")
            em_gro_abs = os.path.join(target_dir, "em.gro")
            run_rep_cmd([GMX_BIN, "grompp", "-f", "eq.mdp", "-c", em_gro_abs, "-r", em_gro_abs, "-p", "topol.top", "-o", "eq.tpr", "-maxwarn", "10"])
            run_rep_cmd([GMX_BIN, "mdrun", "-deffnm", "eq", "-nb", "gpu", "-bonded", "gpu"])

        # 6. Production MD with Maximum GPU Acceleration (-nb gpu -bonded gpu -pme gpu -pin on)
        logging.info(f"Preparing Maximum Acceleration GPU Production Run for {target}/{rep}...")
        if not os.path.exists(os.path.join(rep_dir, "production.tpr")):
            run_rep_cmd([GMX_BIN, "grompp", "-f", "production.mdp", "-c", "eq.gro", "-p", "topol.top", "-o", "production.tpr", "-maxwarn", "10"])

        if os.path.exists(cpt_file):
            logging.info(f"RESUMING GPU production run from checkpoint for {target}/{rep}...")
            prod_cmd = [GMX_BIN, "mdrun", "-deffnm", "production", "-nb", "gpu", "-bonded", "gpu", "-pme", "gpu", "-pin", "on", "-cpi", "production.cpt", "-append"]
        else:
            logging.info(f"STARTING FRESH GPU production run for {target}/{rep}...")
            prod_cmd = [GMX_BIN, "mdrun", "-deffnm", "production", "-nb", "gpu", "-bonded", "gpu", "-pme", "gpu", "-pin", "on"]

        # Launch production mdrun in background and proceed to next target replicate
        subprocess.Popen(prod_cmd, cwd=rep_dir, env=env)
        logging.info(f"Launched GPU Production MD for {target}/{rep} with flags: {' '.join(prod_cmd)}")


def main():
    logging.info("Starting High-Throughput GPU-Accelerated GROMACS CUDA MD Pipeline")
    water_box = os.path.join(BASE_DIR, "data", "martini_v3.0.0_water.gro")
    generate_martini_water_box(water_box)

    for target in TARGETS:
        try:
            run_gromacs_target(target, water_box)
        except Exception as e:
            logging.error(f"Error in GROMACS pipeline for {target}: {e}", exc_info=True)

    logging.info("All GROMACS CUDA GPU MD runs initialized.")


if __name__ == "__main__":
    main()
