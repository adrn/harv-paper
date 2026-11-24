"""Posterior samples of every star of Case study 4.

For each star of the simulated population (`simulate_population.py`), we run the
rejection sampler with one shared library of 10^7 prior samples and keep at most
M_max = 1024 samples. For stars with fewer than M_max samples, we rerun it with a shared
library of 10^8 prior samples, which is written to disk once and streamed from there. If
a star still has fewer than M_max samples, we continue with 4 chains of NUTS initialized
from the rejection samples (samples in a single period mode), and keep the MCMC samples
if R-hat < 1.2 for all parameters. Stars with samples in several period modes, or with
R-hat >= 1.2, go to the nested slice sampler from blackjax, following Case study 3
(`nested-sampling.ipynb`). ln Z_int and J_eff come from the largest rejection run.

The samples of each star go to `data/population/cache/stars/star-NNN.h5`, with a summary
of the procedure (outcome, evidence, timing) in the metadata. Stars that are done are
skipped when the script is run again; `--overwrite` deletes the cache and starts over.
`population-inference.ipynb` reads the cache with `read_posteriors()`.

    uv run python scripts/population_posteriors.py              # all stars, one process
    srun uv run python scripts/population_posteriors.py --mpi   # see slurm/population.sbatch
"""

import jax
import numpyro

# Enable float64 before importing harv, and 4 devices for 4 parallel MCMC chains:
jax.config.update("jax_enable_x64", True)
numpyro.set_host_device_count(4)

import argparse
import dataclasses
import shutil
import time
import warnings
from pathlib import Path

import astropy.table as at
import blackjax
import blackjax.ns.utils as nsu
import harv
import harv.models as hm
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from harv.samplers import Samples, make_prior_cache
from numpyro.diagnostics import summary
from numpyro.handlers import block
from numpyro.infer import Predictive
from numpyro.infer.util import (
    constrain_fn,
    log_likelihood,
    potential_energy,
    unconstrain_fn,
)
from unxt import Q

repo_path = Path(__file__).resolve().parents[1]
data_path = repo_path / "data" / "population"
cache_path = data_path / "cache"
prior_1e8_path = cache_path / "prior-1e8.h5"

M_max = 1024
J = 10_000_000

# The interim prior: uniform in ln P between 0.5 and 10^4 days, a normal distribution for
# the eccentricity with mean 0.1 and standard deviation 0.4 (truncated to [0, 1)), uniform
# in the phase and argument of pericenter, and the default Gaussian priors on the linear
# parameters with sigma_K0 = 30 km/s and sigma_v0 = 50 km/s.
sigma_K0, sigma_v0 = 30.0, 50.0  # km/s
prior = hm.StandardRV().default_prior(
    period_min=Q(0.5, "day"),
    period_max=Q(1e4, "day"),
    sigma_K0=Q(sigma_K0, "km/s"),
    sigma_v0=Q(sigma_v0, "km/s"),
    eccentricity=dist.TruncatedNormal(0.1, 0.4, low=0.0, high=1.0),
)
model = harv.RVModel()
# numpyro caches the normalization of a truncated distribution at its first evaluation
# outside of jit (here, in the nested sampler), and code compiled afterwards uses the
# cached value. That changes the rounding enough to change an MCMC run, so we fill the
# cache now: the result of each star then doesn't depend on which stars ran before it.
prior.nonlinear_priors["eccentricity"].log_prob(0.5)

# The summary of each star's procedure, stored in the metadata of its samples.
info_names = [
    "outcome",  # rejection 1e7, rejection 1e8, MCMC, or nested
    "ln_Z_int",  # from the largest rejection run
    "J_eff",  # from the largest rejection run
    "n_samples_1e7",
    "J_eff_1e7",
    "time_1e7",  # s, the rejection run with 10^7 prior samples
    "first",  # whether time_1e7 includes compilation
    "r_hat",  # largest R-hat of the MCMC run (NaN if none)
    "ln_Z_nested",  # NaN if no nested sampling
    "time_total",  # s
]


def read_data():
    """The table of the simulated stars, their RV data, and their numbers of epochs."""
    tbl = at.Table.read(data_path / "population.ecsv")
    n_epochs = np.asarray(tbl["n_epochs"])
    rv_data = [
        harv.RVData(
            time=Q(row["time"][:m], "day"),
            rv=Q(row["rv"][:m], "km/s"),
            rv_err=Q(row["rv_err"][:m], "km/s"),
        )
        for row, m in zip(tbl, n_epochs, strict=True)
    ]
    return tbl, rv_data, n_epochs


def star_path(n):
    return cache_path / "stars" / f"star-{n:03d}.h5"


def read_posteriors(n_stars):
    """The cached samples of stars 0 to n_stars - 1, and a table of their summaries."""
    missing = [n for n in range(n_stars) if not star_path(n).exists()]
    if missing:
        msg = f"no cached samples for stars {missing}: run population_posteriors.py"
        raise FileNotFoundError(msg)
    posteriors = [Samples.from_hdf5(star_path(n)) for n in range(n_stars)]
    info = at.Table(
        rows=[[s.metadata[k] for k in info_names] for s in posteriors],
        names=info_names,
    )
    return posteriors, info


def max_r_hat(s):
    # wrap_angles makes K positive; Samples stores the chains one after another.
    w = s.wrap_angles()
    stats = summary(
        {k: np.asarray(w[k].value).reshape(4, -1) for k in [*w.nonlinear, *w.linear]}
    )
    return np.nanmax([v["r_hat"] for v in stats.values()])


def nested_samples(data, key, units, n_live=10_000, num_inner_steps=20):
    """M_max posterior samples of one star from the nested slice sampler.

    As in Case study 3: we run the sampler on the numpyro model that harv builds, in
    numpyro's unconstrained space, replace n_live / 10 points per iteration, and stop
    when the evidence in the live points is below 10^-3 of the accumulated evidence. We
    then draw M_max samples from the weighted dead points, and for each we draw the
    linear parameters from their conditional posterior pdf, as the rejection sampler
    does. `units` are the units of the nonlinear parameters.
    """
    numpyro_model = model.numpyro_model(
        prior.nonlinear_priors, data, prior.linear_priors
    )
    prior_model = block(numpyro_model, hide=["ln_lik"])

    def constrain(u):
        return constrain_fn(prior_model, (), {}, u)

    def log_prior(u):
        return -potential_energy(prior_model, (), {}, u)

    def log_lik(u):
        return log_likelihood(numpyro_model, constrain(u), batch_ndims=0)["ln_lik"]

    algo = blackjax.nss(
        log_prior, log_lik, num_inner_steps=num_inner_steps, num_delete=n_live // 10
    )
    step = jax.jit(algo.step)
    init_key, sample_key, linear_key, key = jax.random.split(key, 4)
    draws = Predictive(prior_model, num_samples=n_live)(init_key)
    state = algo.init(jax.vmap(lambda p: unconstrain_fn(prior_model, (), {}, p))(draws))
    dead = []
    while state.integrator.logZ_live - state.integrator.logZ >= np.log(1e-3):
        key, subkey = jax.random.split(key)
        state, info = step(subkey, state)
        dead.append(info)
    final = nsu.finalise(state, dead)

    values = jax.vmap(constrain)(nsu.sample(sample_key, final, shape=M_max).position)
    nonlinear = {k: Q(values[k], unit) for k, unit in units.items()}
    linear = jax.vmap(
        lambda k, v: model.sample_conditional_linear(
            v, data, key=k, linear_priors=prior.linear_priors
        )
    )(jax.random.split(linear_key, M_max), nonlinear)
    ln_Z = jnp.logaddexp(state.integrator.logZ, state.integrator.logZ_live)
    return Samples(
        nonlinear=nonlinear,
        linear={k: Q(v, "km/s") for k, v in linear.items()},
        model_type="RVModel",
        metadata={"ln_Z_nested": float(ln_Z)},
    )


def star_posterior(n, data, prior_samples, sampler, mcmc, first):
    """Samples of star n, with the summary of the procedure as their metadata."""
    t0 = time.perf_counter()
    s = sampler.run_with_samples(
        data,
        prior_samples,
        key=jax.random.key(n + 1),
        max_posterior_samples=M_max,
        return_evidence_stats=True,
    )
    info = {
        "outcome": "rejection 1e7",
        "n_samples_1e7": s.n_samples,
        "J_eff_1e7": float(s.metadata["ln_Z_int_ess"]),
        "time_1e7": time.perf_counter() - t0,
        "first": first,
        "r_hat": np.nan,
        "ln_Z_nested": np.nan,
    }
    if s.n_samples < M_max:
        s = sampler.run_with_samples(
            data,
            prior_1e8_path,
            key=jax.random.key(1000 + n),
            max_posterior_samples=M_max,
            return_evidence_stats=True,
        )
        info["outcome"] = "rejection 1e8"
    info["ln_Z_int"] = float(s.metadata["ln_Z_int"])
    info["J_eff"] = float(s.metadata["ln_Z_int_ess"])

    if s.n_samples < M_max:
        converged = False
        if s.period_unimodal(data):
            m = mcmc.run(
                data,
                init_samples=s,
                key=jax.random.key(2000 + n),
                num_chains=4,
                num_warmup=1000,
                num_samples=M_max // 4,
                progress_bar=False,
            )
            info["r_hat"] = float(max_r_hat(m))
            converged = info["r_hat"] < 1.2
        if converged:
            s, info["outcome"] = m, "MCMC"
        else:
            units = {k: s[k].unit for k in s.nonlinear}
            s = nested_samples(data, jax.random.key(3000 + n), units)
            info["outcome"] = "nested"
            info["ln_Z_nested"] = s.metadata["ln_Z_nested"]
    info["time_total"] = time.perf_counter() - t0
    return dataclasses.replace(s, metadata=info)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mpi", action="store_true", help="split the stars over MPI ranks"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="delete the cache and start over"
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Under-resolved rejection run")

    comm, rank, size = None, 0, 1
    if args.mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()

    tbl, rv_data, n_epochs = read_data()
    todo = None
    if rank == 0:
        try:
            if args.overwrite and cache_path.exists():
                shutil.rmtree(cache_path)
            star_path(0).parent.mkdir(parents=True, exist_ok=True)
            if not prior_1e8_path.exists():
                t0 = time.perf_counter()
                tmp = prior_1e8_path.with_name(prior_1e8_path.name + ".tmp")
                make_prior_cache(prior, model, 10 * J, tmp, key=jax.random.key(10_000))
                tmp.replace(prior_1e8_path)
                print(
                    f"10^8 prior samples: {time.perf_counter() - t0:.0f} s", flush=True
                )
            # In order of the number of epochs, so that each rank compiles few times.
            order = np.argsort(n_epochs, kind="stable")
            todo = [int(n) for n in order if not star_path(n).exists()]
            print(f"{len(todo)} of {len(rv_data)} stars to run", flush=True)
        except BaseException:
            # The other ranks are waiting in bcast; end them too.
            if comm is not None:
                comm.Abort(1)
            raise
    if comm is not None:
        todo = comm.bcast(todo)
    mine = np.array_split(todo, size)[rank]
    if len(mine) == 0:
        return

    # The same 10^7 prior samples on every rank.
    prior_samples = prior.sample(J, key=jax.random.key(0), model=model)
    sampler = harv.RejectionSampler(prior, model)
    mcmc = harv.NumpyroSampler(prior, model)
    compiled = set()
    for n in mine:
        s = star_posterior(
            n, rv_data[n], prior_samples, sampler, mcmc, n_epochs[n] not in compiled
        )
        compiled.add(n_epochs[n])
        # Write under another name first, so a star killed mid-write is not "done".
        path = star_path(n)
        tmp = path.with_name(path.name + ".tmp")
        s.to_hdf5(tmp)
        tmp.replace(path)
        info = s.metadata
        P = np.median(s["period"].to_value("day"))
        print(
            f"rank {rank:3d}, star {n:3d} ({n_epochs[n]:2d} epochs): "
            f"{info['n_samples_1e7']} samples, J_eff {info['J_eff_1e7']:.3g} with 10^7 "
            f"-> {info['outcome']}; median P {P:.1f} d, true {tbl['period'][n]:.1f} d; "
            f"{info['time_total']:.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
