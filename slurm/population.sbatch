#!/bin/zsh -l

# Case study 4: simulate the population, then the posterior samples of every star over
# MPI. Submit from the repo root:
#   cd slurm
#   mkdir -p logs
#   sbatch slurm/population.sbatch
#     or: sbatch slurm/population.sbatch --overwrite
# Resubmit the same command (without --overwrite) after a timeout: finished stars are
# skipped. Then run scripts/population-inference.ipynb for the population model and figures.

#SBATCH --job-name=cs4-population
#SBATCH --partition=cca
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail

source ~/.env
init_env_2026

# builtin: the cd function in the shell profile reads $VIRTUAL_ENV, which is unset here
# and fails under set -u. uv run finds the project's .venv without activating it.
builtin cd /mnt/home/apricewhelan/work/harv-paper

# One XLA thread pool per rank, sized to its allocation, so ranks don't contend.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"

# Deterministic, so rewriting the data on a resubmit doesn't change it.
uv run --no-sync python scripts/simulate_population.py
srun uv run --no-sync python scripts/population_posteriors.py --mpi "$@"
