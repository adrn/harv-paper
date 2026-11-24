"""Simulated population for Case study 4.

Simulates the radial velocity data for Case study 4 (`population_posteriors.py` and
`population-inference.ipynb`): 256 stars where ~40% have a companion. We assume a survey
with a baseline of 8 years with log-uniform draws for the observation time separations
with an uncertainty of 0.5 km/s. For this sample, by construction we know the population
parameters, so we can compare our population inferences to the truth.

Writes `data/population/population.ecsv`, with one row per star (the data, the true
parameters, and an observed primary mass) and the population parameters in the table
metadata. All seeds are fixed, so rerunning writes the same file.
"""

import jax

# Enable float64 before importing harv:
jax.config.update("jax_enable_x64", True)

from pathlib import Path

import astropy.table as at
import astropy.units as u
import harv
import numpy as np
import numpyro.distributions as dist
from astropy.constants import G
from harv.kepler.orbits import rv_at_times
from scipy.stats import truncnorm
from unxt import Q

print(f"harv {harv.__version__}, jax {jax.__version__}")

data_path = Path(__file__).resolve().parents[1] / "data" / "population"
data_path.mkdir(parents=True, exist_ok=True)

# Population parameters. The orbital periods of the binaries follow a log-normal
# distribution (Raghavan et al. 2010) truncated to 5 to 1000 days, the eccentricities a
# normal distribution truncated to [0, 1], and the companion masses a log-normal
# distribution truncated to 0.1 to 1 Msun. The primary masses follow a normal
# distribution with mean 1 Msun and standard deviation 0.3 Msun, truncated to 0.5 to
# 1.5 Msun, and the orbits are randomly oriented. We also record an observed primary
# mass for each star, with an uncertainty of 0.05 Msun. Logarithms are base 10, periods
# are in days and masses in Msun.
seed = 0
n_stars, n_binaries, n_epochs_range = 256, 102, (3, 12)
baseline, min_separation = 8 * 365.0, 1.0  # day
rv_err, sigma_v0 = 0.5, 50.0  # km/s

pop = {
    "binary_fraction": n_binaries / n_stars,
    "mu_logP": 5.03,  # Raghavan et al. (2010)
    "sigma_logP": 2.28,
    "P_min": 5.0,
    "P_max": 1000.0,
    "mu_e": 0.4,
    "sigma_e": 0.15,
    "mu_logM2": float(np.log10(0.9)),
    "sigma_logM2": 0.3,
    "M2_min": 0.1,
    "M2_max": 1.0,
    "M1_mean": 1.0,
    "M1_std": 0.3,
    "M1_min": 0.5,
    "M1_max": 1.5,
    "M1_err": 0.05,
}

# Draw the stars. Each star is observed between 3 and 12 times (drawn uniformly). The
# separations between consecutive epochs are log-uniform between 1 day and the baseline,
# scaled so that the last epoch is 8 years after the first. We pad the times of each
# star with NaN to 12 epochs. A random subset of 102 stars (~40%) have a companion. We
# draw orbital parameters for every star and keep them only for the binaries.
rng = np.random.default_rng(seed)
n_epochs = rng.integers(*n_epochs_range, endpoint=True, size=n_stars)
time = np.full((n_stars, n_epochs_range[1]), np.nan)
for n, m in enumerate(n_epochs):
    separation = np.exp(
        rng.uniform(np.log(min_separation), np.log(baseline), size=m - 1)
    )
    t = np.concatenate([[0.0], np.cumsum(separation)])
    time[n, :m] = t * baseline / t[-1]

is_binary = np.zeros(n_stars, dtype=bool)
is_binary[rng.choice(n_stars, n_binaries, replace=False)] = True

M1 = truncnorm.rvs(
    (pop["M1_min"] - pop["M1_mean"]) / pop["M1_std"],
    (pop["M1_max"] - pop["M1_mean"]) / pop["M1_std"],
    loc=pop["M1_mean"],
    scale=pop["M1_std"],
    size=n_stars,
    random_state=rng,
)
logM2 = dist.TruncatedNormal(
    pop["mu_logM2"],
    pop["sigma_logM2"],
    low=np.log10(pop["M2_min"]),
    high=np.log10(pop["M2_max"]),
).sample(jax.random.key(987), (n_stars,))
logP = dist.TruncatedNormal(
    pop["mu_logP"],
    pop["sigma_logP"],
    low=np.log10(pop["P_min"]),
    high=np.log10(pop["P_max"]),
).sample(jax.random.key(42), (n_stars,))
M2, period = 10 ** np.asarray(logM2), 10 ** np.asarray(logP)
ecc = truncnorm.rvs(
    -pop["mu_e"] / pop["sigma_e"],
    (1 - pop["mu_e"]) / pop["sigma_e"],
    loc=pop["mu_e"],
    scale=pop["sigma_e"],
    size=n_stars,
    random_state=rng,
)
sin_i = np.sqrt(1.0 - rng.uniform(-1.0, 1.0, size=n_stars) ** 2)
arg_peri = rng.uniform(0.0, 2 * np.pi, size=n_stars)
time_peri = rng.uniform(0.0, period, size=n_stars)
v_sys = rng.normal(0.0, sigma_v0, size=n_stars)

K = (
    (2 * np.pi * G / (period * u.day)) ** (1 / 3)
    * M2
    * sin_i
    * u.Msun
    / ((M1 + M2) * u.Msun) ** (2 / 3)
    / np.sqrt(1 - ecc**2)
).to_value(u.km / u.s)
print(f"{is_binary.sum()} binaries ({is_binary.mean():.3f} of the stars)")
print(f"K of the binaries: {K[is_binary].min():.2f} to {K[is_binary].max():.2f} km/s")
print(f"{n_epochs.min()} to {n_epochs.max()} epochs per star")
print(
    f"separations between epochs: {np.nanmin(np.diff(time, axis=1)):.2f} to "
    f"{np.nanmax(np.diff(time, axis=1)):.0f} days"
)

# Radial velocities. The single stars have a constant velocity. We add Gaussian noise to
# all velocities, and then draw the observed primary masses.
rv = np.where(np.isnan(time), np.nan, v_sys[:, None])
for n in np.flatnonzero(is_binary):
    m = n_epochs[n]
    rv[n, :m] = rv_at_times(
        Q(time[n, :m], "day"),
        period=Q(period[n], "day"),
        eccentricity=ecc[n],
        time_peri=Q(time_peri[n], "day"),
        arg_peri=Q(arg_peri[n], "rad"),
        rv_semiamp=Q(K[n], "km/s"),
        v_sys=Q(v_sys[n], "km/s"),
    ).to_value("km/s")
rv += rng.normal(scale=rv_err, size=rv.shape)
M1_obs = M1 + rng.normal(scale=pop["M1_err"], size=n_stars)


# Save. The time, velocity and uncertainty columns are padded with NaN to 12 epochs, and
# `n_epochs` is the number of epochs of each star. The true orbital parameters are NaN
# for the single stars, and K = 0.
def binaries_only(x):
    return np.where(is_binary, x, np.nan)


tbl = at.Table(
    {
        "n_epochs": n_epochs,
        "time": time * u.day,
        "rv": rv * u.km / u.s,
        "rv_err": np.where(np.isnan(time), np.nan, rv_err) * u.km / u.s,
        "M1_obs": M1_obs * u.Msun,
        "is_binary": is_binary,
        "M1": M1 * u.Msun,
        "M2": binaries_only(M2) * u.Msun,
        "period": binaries_only(period) * u.day,
        "eccentricity": binaries_only(ecc),
        "sin_i": binaries_only(sin_i),
        "arg_peri": binaries_only(arg_peri) * u.rad,
        "time_peri": binaries_only(time_peri) * u.day,
        "K": np.where(is_binary, K, 0.0) * u.km / u.s,
        "v_sys": v_sys * u.km / u.s,
    },
    meta={**pop, "seed": seed, "sigma_v0": sigma_v0},
)
tbl.write(data_path / "population.ecsv", overwrite=True)
print(f"{len(tbl)} stars, {tbl['is_binary'].sum()} binaries")
