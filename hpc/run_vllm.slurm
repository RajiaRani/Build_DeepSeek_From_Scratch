#!/bin/bash
# =====================================================================
#  SLURM batch script to run vLLM on a university HPC GPU node.
#  Submit it with:   sbatch run_vllm.slurm
#  Watch it with:    squeue --me        and    tail -f vllm_<jobid>.out
#
#  EDIT the lines marked  <-- EDIT  to match your cluster.
#  Find your options with:  sinfo        (partitions)
#                           sacctmgr show assoc user=$USER   (accounts)
# =====================================================================

#SBATCH --job-name=vllm-demo
#SBATCH --partition=gpu            # USD gpu partition
#SBATCH --gres=gpu:volta:1         # a V100 (Volta). Do NOT use pascal (gpu001), too old for vLLM
#SBATCH --cpus-per-task=8          # CPU cores for data loading
#SBATCH --mem=32G                  # system RAM (not VRAM)
#SBATCH --time=01:00:00            # walltime HH:MM:SS (job is killed after this)
#SBATCH --output=vllm_%j.out       # stdout + stderr go to vllm_<jobid>.out
#SBATCH --account=rizk_lab         # your SLURM account

set -euo pipefail
echo "Job $SLURM_JOB_ID started on host $(hostname) at $(date)"

# --- 1. Load modules (USD cluster) ---
module purge
module load python/3.11.9-gcc-14.2.0 cuda/12.3

# --- 2. Activate the environment you created once (see HPC_SETUP.md) ---
source "$HOME/vllm-env/bin/activate"

# --- 3. Keep the Hugging Face model cache off your small home quota ---
# Point this at your scratch space if home is quota-limited.
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"

# --- 4. Show the GPU you were allocated, then run ---
nvidia-smi
python "$SLURM_SUBMIT_DIR/vllm_demo.py"

echo "Job $SLURM_JOB_ID finished at $(date)"
