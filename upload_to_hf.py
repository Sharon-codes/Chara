#!/usr/bin/env python3
"""
upload_to_hf.py - Upload Chara Model to Hugging Face Hub
Usage:
    python upload_to_hf.py --token YOUR_HF_WRITE_TOKEN [--repo Sharon-codes/chara-survival]
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent
HF_FOLDER = ROOT / "hf_model_repo"

def main():
    parser = argparse.ArgumentParser(description="Upload Chara model to Hugging Face Hub")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face User Access Token (Write permission)")
    parser.add_argument("--repo", type=str, default="Sharon-codes/chara-survival", help="Target Hugging Face repository name (e.g. Sharon-codes/chara-survival)")
    args = parser.parse_args()

    token = args.token
    if not token:
        print("\n[!] Please provide your Hugging Face Write Token:")
        print("    You can create one for free at: https://huggingface.co/settings/tokens")
        token = input("Enter your Hugging Face token (hf_...): ").strip()

    if not token:
        print("[ERROR] No Hugging Face token provided. Aborting.")
        sys.exit(1)

    repo_id = args.repo
    print(f"\n[*] Target Repository: {repo_id}")
    print(f"[*] Uploading folder: {HF_FOLDER}")

    api = HfApi()

    try:
        # Create repository if it doesn't already exist
        print(f"[*] Ensuring repository '{repo_id}' exists...")
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"[+] Repository ready: https://huggingface.co/{repo_id}")

        # Upload folder
        print(f"[*] Uploading model artifacts and Model Card to Hugging Face Hub...")
        api.upload_folder(
            folder_path=str(HF_FOLDER),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Initial release of Chara Survival Model (4,337 Genes, 58 Active Biomarkers)"
        )
        print("\n" + "="*70)
        print(f"🎉 SUCCESS: Chara Model is now LIVE on Hugging Face!")
        print(f"👉 Model Hub URL: https://huggingface.co/{repo_id}")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Failed to upload to Hugging Face: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
