#!/usr/bin/env python3
"""
tools/chara_closure/03_dataset_and_literature.py
Generates:
- reports/chara_closure/DATASET_ACCESS_VERIFIED.csv
- reports/chara_closure/CLOSEST_WORK_CORRECTED.csv
- reports/chara_closure/CLAIM_CORRECTIONS_FINAL.csv
"""

import os
import pandas as pd

BASE_DIR = r"E:\Sharon"
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "chara_closure")
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_dataset_and_literature():
    print("=== [1/3] Generating DATASET_ACCESS_VERIFIED.csv ===")
    datasets = [
        {
            "dataset_name": "DESRES Anton BPTI",
            "primary_publication": "Shaw et al., Science 330, 341-346 (2010)",
            "primary_doi": "10.1126/science.1187409",
            "access_route": "Academic non-commercial research agreement on request to D. E. Shaw Research (deshawresearch.com). Not deposited in Zenodo/Dryad public open repositories.",
            "license_type": "Proprietary Academic Research Agreement",
            "simulation_architecture": "Anton specialized supercomputer, Desmond engine",
            "reported_trajectory_structure": "1 continuous trajectory of 1.031 ms (Condition A) + 1 trajectory of 106 us (Condition B).",
            "independent_replicate_count": 1,
            "eligibility_status": "VERIFIED_INELIGIBLE",
            "factual_reasons": "Fails multi-replicate requirement (1 continuous run under standard condition, not 5 independent replicates). Fails direct open download requirement (requires formal non-commercial request and review by D. E. Shaw Research)."
        },
        {
            "dataset_name": "MoDEL Database",
            "primary_publication": "Meyer et al., Structure 18, 1399-1409 (2010)",
            "primary_doi": "10.1016/j.str.2010.07.013",
            "access_route": "Public web repository (mmb.irbbarcelona.org/MoDEL/) and REST API. Creative Commons open access.",
            "license_type": "Creative Commons Open Access",
            "simulation_architecture": "GROMACS / NAMD / Amber CPU/GPU cluster simulations",
            "reported_trajectory_structure": "Single short trajectories (10–100 ns) per representative PDB fold across 1,700+ distinct protein structures.",
            "independent_replicate_count": 1,
            "eligibility_status": "VERIFIED_INELIGIBLE",
            "factual_reasons": "Fails multi-replicate benchmark criterion: MoDEL was constructed for broad structural coverage across the PDB fold universe, providing only single trajectories per structure rather than replicate ensembles. Note: Erroneous citation DOI 10.1093/bioinformatics/btw348 (a protein-protein network alignment paper) is corrected to 10.1016/j.str.2010.07.013."
        }
    ]
    df_datasets = pd.DataFrame(datasets)
    p_ds = os.path.join(REPORTS_DIR, "DATASET_ACCESS_VERIFIED.csv")
    df_datasets.to_csv(p_ds, index=False)
    print(f"  [OK] Saved {p_ds}")

    print("=== [2/3] Generating CLOSEST_WORK_CORRECTED.csv ===")
    prior_art = [
        {
            "study": "Differentiable Information Imbalance (DII)",
            "citation": "Wild et al., Nat. Commun. 16, 54 (2025)",
            "doi": "10.1038/s41467-024-55449-7",
            "published_status": "Published January 2, 2025 in Nature Communications",
            "core_objective": "Continuous optimization of feature weights minimizing information imbalance between full coordinates and selected low-dimensional representations.",
            "methodology": "Gradient-based continuous relaxation of asymmetric rank correlation, enabling automatic feature selection.",
            "relationship_to_chara": "DII solves generic continuous/sparse feature weighting for coordinate embeddings; Chara evaluates discrete residue-pair sensor placement for cross-run distance matrix reconstruction."
        },
        {
            "study": "Variational Approach for Markov Processes (VAMP / VAC)",
            "citation": "Wu & Noé, J. Chem. Phys. 153, 054117 (2020); Perez-Hernandez et al., J. Chem. Phys. 139, 015102 (2013)",
            "doi": "10.1063/5.0023804",
            "published_status": "Peer-reviewed journal publication",
            "core_objective": "Optimal kinetic feature extraction and lag-time dynamic propagation using variational score maximization.",
            "methodology": "Time-lagged independent component analysis (TICA) and VAMPnet deep neural dynamical representations.",
            "relationship_to_chara": "Wu & Noé optimize slow dynamical modes across lag times tau; Chara optimizes instantaneous sensor placement for geometric distance reconstruction."
        },
        {
            "study": "Sparse Sensor Placement Optimization",
            "citation": "Manohar et al., IEEE Trans. Signal Process. 66, 1729-1740 (2018)",
            "doi": "10.1109/TSP.2017.2786235",
            "published_status": "Peer-reviewed journal publication",
            "core_objective": "Data-driven sensor placement using QR factorization with column pivoting on tailored proper orthogonal decomposition bases.",
            "methodology": "Greedy QR pivoting on principal component modes for linear signal reconstruction.",
            "relationship_to_chara": "Direct conceptual foundation for Chara M4 (PCA/QR benchmark model). Established in signal processing prior to Chara."
        },
        {
            "study": "Information Imbalance Distance Metrics",
            "citation": "Glielmo et al., PNAS 118, e2102170118 (2021)",
            "doi": "10.1073/pnas.2102170118",
            "published_status": "Peer-reviewed journal publication",
            "core_objective": "Non-parametric comparison of distance metrics and coordinate representations in high-dimensional data.",
            "methodology": "Asymmetric relative distance ranking between alternative feature spaces.",
            "relationship_to_chara": "Direct conceptual foundation for Chara M5 (Rank Information Imbalance benchmark selector)."
        },
        {
            "study": "Cross-Simulation Feature Generalization & Replicate Validation",
            "citation": "Standard statistical learning literature (e.g. Hastie, Tibshirani, Friedman, Elements of Statistical Learning)",
            "doi": "10.1007/978-0-387-84858-7",
            "published_status": "Standard textbook / literature consensus",
            "core_objective": "Evaluating predictive models on independent data groups / hold-out clusters to prevent data leakage.",
            "methodology": "Group K-fold cross-validation across simulation runs.",
            "relationship_to_chara": "Holding out replicate trajectories is standard GroupKFold cross-validation; penalizing cross-replicate variance is a known regularizer."
        }
    ]
    df_art = pd.DataFrame(prior_art)
    p_art = os.path.join(REPORTS_DIR, "CLOSEST_WORK_CORRECTED.csv")
    df_art.to_csv(p_art, index=False)
    print(f"  [OK] Saved {p_art}")

    print("=== [3/3] Generating CLAIM_CORRECTIONS_FINAL.csv ===")
    claim_corrections = [
        {
            "claim_id": "CLM-01",
            "topic": "KRAS Target Mutation Status",
            "original_assertion": "KRAS_G12D simulation system represents oncogenic KRAS G12D mutant.",
            "evidence_status": "FACTUALLY_INCORRECT",
            "source_evidence": "PDB 4OBE contains wild-type KRAS (residues 1-169) with Glycine at canonical residue 12 (construct residue 13). Zero mutations occurred in structure or topology generation.",
            "supported_reconciliation": "System represents wild-type KRAS (PDB 4OBE fragment 1-169), not G12D mutant."
        },
        {
            "claim_id": "CLM-02",
            "topic": "p53 Target Mutation Status",
            "original_assertion": "Mut_p53 represents classic cancer hotspot mutant R273H.",
            "evidence_status": "FACTUALLY_INCORRECT",
            "source_evidence": "PDB 2J1X is a quintuple core-domain mutant (M133L/V203A/Y220C/N239Y/N268D) with 4 stabilizing framework mutations and 1 oncogenic mutation (Y220C). Residue 273 is wild-type Arginine.",
            "supported_reconciliation": "System represents quintuple core-domain mutant p53 (PDB 2J1X), not isolated R273H hotspot."
        },
        {
            "claim_id": "CLM-03",
            "topic": "cMYC/MAX Molecular Construct",
            "original_assertion": "cMYC/MAX simulation modeled the functional DNA-bound transcriptional complex.",
            "evidence_status": "FACTUALLY_INCORRECT",
            "source_evidence": "PDB 1NKP was coarse-grained using protein Chains E-H only (812 beads total); the double-stranded DNA duplex was entirely omitted from simulation.",
            "supported_reconciliation": "System represents protein-only heterotetrameric bHLH-LZ domain; DNA was omitted."
        },
        {
            "claim_id": "CLM-04",
            "topic": "PTPN11 Trajectory Continuity & Frame Fixing",
            "original_assertion": "PTPN11 Rep1 trajectory coordinate file was repaired to replace dropped frames.",
            "evidence_status": "FACTUALLY_INCORRECT",
            "source_evidence": "scripts/19_fix_ptpn11_rep1_frames.py interpolated scalar XVG analysis metrics only; binary trajectory coordinates (production.xtc) and compiled TPR were never modified.",
            "supported_reconciliation": "PTPN11 Rep1 contains 895 physical frames (5 dropped frames at 119.5 ns); binary coordinates were never interpolated."
        },
        {
            "claim_id": "CLM-05",
            "topic": "Force Field Parameter Provenance",
            "original_assertion": "Production simulations executed using standard official Martini 3.0.0 force field.",
            "evidence_status": "UNRESOLVED_COMPILED_PROVENANCE",
            "source_evidence": "scripts/03_phase3_gromacs.py write_martini_itp() generated 16 copies of a 604-atomtype simplified potential with uniform diagonal defaults and 6 overrides. However, direct binary inspection (gmx dump -s production.tpr) remains UNRESOLVED due to local dump tool unavailability.",
            "supported_reconciliation": "Source topologies define simplified 604-type potential, but compiled binary TPR status remains UNRESOLVED; simulations cannot be asserted to match official Martini 3.0.0 or proved to use simplified potential from binary evidence."
        },
        {
            "claim_id": "CLM-06",
            "topic": "Subgroup Selection Test Data Contamination",
            "original_assertion": "High-variance subgroup was identified purely from development simulations.",
            "evidence_status": "CONFIRMED_BUG_CORRECTED",
            "source_evidence": "code/02_correct_diagnostics.py looked for key 'targets' instead of 'Y', falling back silently to test run Y_true in all 12 folds. Correcting development variance loading altered top-quartile membership by 29.2% to 67.6%.",
            "supported_reconciliation": "Bug eliminated; development subgroup is now strictly loaded from pooled development frames with zero test data leakage."
        },
        {
            "claim_id": "CLM-07",
            "topic": "Mechanistic Basis of Reconstruction Skill",
            "original_assertion": "Positive reconstruction skill proves model accurately tracks dynamic molecular fluctuations.",
            "evidence_status": "UNSUPPORTED_CAUSAL_CLAIM",
            "source_evidence": "Population-moment decomposition proves 80.5% to 86.2% of MSE reduction across all folds arises from static temporal mean offset correction (Delta Bias^2), with dynamic variance reduction bounded below 0.0007 nm^2.",
            "supported_reconciliation": "Models capture static mean offsets between simulation runs; dynamic variance tracking contributes a minor fraction (<20%) of overall MSE reduction."
        },
        {
            "claim_id": "CLM-08",
            "topic": "Novelty of Cross-Simulation Holdout & Transfer Selection",
            "original_assertion": "Holding out replicate trajectories and penalizing cross-replicate transfer error is a novel discovery.",
            "evidence_status": "UNSUPPORTED_NOVELTY_CLAIM",
            "source_evidence": "Cross-validation across independent simulation runs (GroupKFold) and penalizing transfer variance are established practices in statistical learning and MD analysis literature.",
            "supported_reconciliation": "Method combines known components (GroupKFold across replicates, sparse sensor placement, variance penalization) for discrete distance reconstruction."
        }
    ]
    df_claims = pd.DataFrame(claim_corrections)
    p_cl = os.path.join(REPORTS_DIR, "CLAIM_CORRECTIONS_FINAL.csv")
    df_claims.to_csv(p_cl, index=False)
    print(f"  [OK] Saved {p_cl}")

if __name__ == "__main__":
    generate_dataset_and_literature()
