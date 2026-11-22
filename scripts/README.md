# Scripts

The notebooks and scripts in this directory produce the results, figures, and
tables in the paper. This file describes how to run the two benchmark notebooks,
which need more care than the others.

## Benchmark notebooks

- `benchmarks-thejoker.ipynb` compares the rejection step of `harv` with The
  Joker on simulated RV data, against the number of prior samples $J$ and the
  number of observations $N$. It writes `paper/figures/bench-thejoker.pdf`.
- `benchmarks-ess.ipynb` measures how the cost of the rejection sampler grows
  with the signal-to-noise ratio of the data, for the standard RV
  parameterization and the standard and Thiele-Innes Gaia astrometry
  parameterizations. It writes `paper/figures/bench-ess.pdf`.

Both report times for one device: one CPU core or one GPU. We ran them on a
workstation with an Intel Xeon w5-3435X CPU and an NVIDIA RTX 6000 Ada GPU.

### Setup

On a Linux machine with an NVIDIA GPU, install the CUDA-enabled `jaxlib`:

```sh
uv sync --group cuda
uv run --group cuda python -c "import jax; print(jax.devices(), jax.devices('cpu'))"
```

The second command should print a `CudaDevice` and a `CpuDevice`. The `cuda`
group installs `jax[cuda13]`, which needs a driver that supports CUDA 13 or
newer (see the "CUDA Version" printed by `nvidia-smi`); for an older driver,
change the group to `jax[cuda12]` in `pyproject.toml`. Pass `--group cuda` to
every `uv` command, because a plain `uv sync` removes the CUDA packages. On a
machine with several GPUs, set `CUDA_VISIBLE_DEVICES=0`.

### Running the notebooks

Each notebook times both devices in one run: The Joker and `harv` on one CPU
core, and `harv` on one GPU if JAX finds one. The first cell limits JAX and the
linear algebra libraries to one thread and, on Linux, pins the process to one
CPU core. These settings only take effect before `jax` is imported, so restart
the kernel before rerunning a notebook. Run the benchmarks on Linux: macOS
cannot pin a process to a core, and there JAX spreads the work of `harv` over
several cores despite these settings (about 5.5 cores on a 16-core laptop), so
the CPU numbers are not for one core.

Run one benchmark at a time, so that the GPU is not shared. JAX reserves 75% of
the GPU memory when it starts, so a second JAX process on the same GPU (for
example an idle Jupyter kernel) prints `CUDA_ERROR_OUT_OF_MEMORY` messages while
it falls back to a smaller reservation; check with `nvidia-smi`.

To run them without an open session (for example inside `tmux`), from the top of
the repository:

```sh
uv run --group cuda --with nbconvert jupyter nbconvert --to notebook \
    --execute --inplace --ExecutePreprocessor.timeout=-1 scripts/benchmarks-thejoker.ipynb
uv run --group cuda --with nbconvert jupyter nbconvert --to notebook \
    --execute --inplace --ExecutePreprocessor.timeout=-1 scripts/benchmarks-ess.ipynb
```

`benchmarks-thejoker.ipynb` takes about 20 minutes, most of it The Joker at
large $N$ and the $10^6$ prior samples it draws for the evidence check. Its last
timing, The Joker at $N = 128$, takes about 7 minutes on its own, and
`nbconvert` writes no output until the end, so the run can look stuck near the
end. `benchmarks-ess.ipynb` takes a few hours; set `J_MAX = 10**6` for a quick
check.

### Results files and remaking the figures

Each notebook saves its measurements to a file in `data/`
(`data/benchmarks-thejoker.ecsv` and `data/benchmarks-ess.ecsv`). If the file
exists, the notebook skips the measurements and only remakes the printed summary
and the figure, so delete the file to rerun them, for example after changing the
grids or settings at the top of a notebook. To change the plots on another
machine, copy the results files there and run the notebook. Some numbers quoted
in the paper, such as the evidence check and the time to generate the prior
library, are only printed during the measurement run, so keep the outputs of
that run.
