#!/bin/bash
# Release fine-tuning recipe on SLURM (6 nodes x 4 GPUs, batch 8/GPU).
#
#   REPO=$PWD HF_HOME=... sbatch scripts/train_release.sbatch   # fine-tune from the release weights
#   INIT=/path/to/warm_start.ckpt sbatch scripts/train_release.sbatch
#   sbatch --nodes=2 scripts/train_release.sbatch experiment.training.max_steps=3000
#
# Extra arguments are passed to main.py as Hydra overrides. Requeue-safe: a preempted job
# resumes from <run dir>/checkpoints/last.ckpt (full state: weights + AdamW + step).
#SBATCH -J ugg_train
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=1400G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

REPO=${REPO:?set REPO=/path/to/UGG}  # not BASH_SOURCE: sbatch runs a spooled copy of this script
RUN_DIR=${RUN_DIR:-$REPO/outputs/ugg_train}
INIT=${INIT:-}          # weights-only warm start; empty = the release checkpoint (algorithm/ugg.yaml)
cd "$REPO" || exit 1
mkdir -p slurm_logs "$RUN_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ${CONDA_ENV:-ugg}
export HF_HOME=${HF_HOME:?set HF_HOME to the Hugging Face cache holding Wan2.2-TI2V-5B}
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_TIMEOUT_MS=3600000 NCCL_TIMEOUT=3600   # a full-state save can stall rank 0 > 10 min
[ -n "${FA3:-}" ] && export PYTHONPATH=$FA3${PYTHONPATH:+:$PYTHONPATH}   # optional FlashAttention-3 build dir

ARGS=()
LAST="$RUN_DIR/checkpoints/last.ckpt"
if [ -e "$LAST" ]; then
  ARGS+=("load=$LAST")                                   # resume (full state)
elif [ -n "$INIT" ]; then
  ARGS+=("algorithm.model.tuned_ckpt_path=$INIT")        # fresh run from other weights
fi

srun python -m main +name=${SLURM_JOB_NAME} \
  hydra.run.dir="$RUN_DIR" \
  experiment=ugg_train algorithm=ugg dataset=mixture_parallax14 \
  experiment.num_nodes=$SLURM_JOB_NUM_NODES \
  "${ARGS[@]}" "$@"
