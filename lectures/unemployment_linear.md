---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# A Linear Model of Unemployment

```{include} _admonition/gpu.md
```

In addition to what's in Anaconda, this lecture needs the following libraries.

We first install `numpyro` and `jax`:

```{code-cell} ipython3
:tags: [hide-output]

!pip install numpyro jax
```

We also install `pandas_datareader`, which we use to download data from FRED:

```{code-cell} ipython3
:tags: [hide-output]

!pip install pandas_datareader
```

## Overview

This lecture fits a simple linear time series model to the US unemployment rate.

We met Bayesian estimation of such models in {doc}`ar1_bayes`, but there the data were simulated and the focus was on theory.

Here our aim is to apply Bayesian estimation to real data and work carefully through the results.

Along the way we ask whether unemployment is a **random walk** — a question that was the subject of considerable debate among macroeconomists in the 1980s and 1990s.

We will find that the linear model we fit here is too limited to capture some important features of the data.

That will motivate the nonlinear model we study in the sequel {doc}`unemployment_nonlinear`.

As in {doc}`ar1_bayes` and {doc}`bayes_nonconj`, we estimate by sampling posteriors with the NUTS sampler in [NumPyro](https://num.pyro.ai/en/stable/).

(See {doc}`bayes_nonconj` for a brief account of how NUTS works.)

Our plan is:

1. look at the monthly data,
2. fit a linear AR(1) model and find that it is *almost* a random walk,
3. ask whether it could really be a random walk — and give two reasons it is not, and
4. see how reconciling those reasons points toward a nonlinear model.

Let's start with some imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as web

import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
```

## The data

We use the US civilian unemployment rate, series `UNRATE` from FRED, monthly and seasonally adjusted.

```{code-cell} ipython3
start, end = dt.datetime(1948, 1, 1), dt.datetime(2024, 12, 31)
unrate = web.DataReader("UNRATE", "fred", start, end)["UNRATE"]
```

The COVID-19 spike of 2020 is an extreme outlier driven by events our models know nothing about, so we drop it.

We begin with the monthly series; later we will also form an annual version.

```{code-cell} ipython3
pre_covid = unrate[unrate.index < "2020-01-01"]
u_monthly = pre_covid.to_numpy()
print(f"{len(u_monthly)} monthly observations")
```

Here is the monthly series.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: US monthly unemployment rate, 1948–2019
    name: fig-unrate-monthly
---
fig, ax = plt.subplots()
ax.plot(pre_covid.index, u_monthly, lw=2)
ax.set_xlabel('year')
ax.set_ylabel('unemployment rate (%)')
plt.show()
```

As {numref}`fig-unrate-monthly` shows, the rate rises sharply in recessions and drifts down in recoveries, but it always stays inside a band — roughly 3% to 11% over the whole post-war period.

Any sensible model has to respect that band.

## A linear model of unemployment

In this section we set up a simple linear model, estimate it on the monthly data, and read off what the estimates say.

This is also where we practise the mechanics of applying Bayesian estimation to a real series.

### The model

We let unemployment be pulled back toward a normal level $\bar u$:

$$
u_{t+1} = \bar u + \phi\,(u_t - \bar u) + \varepsilon_{t+1},
\qquad \varepsilon_{t+1} \sim N(0, \sigma^2),
$$ (eq:linear)

with $0 \le \phi < 1$.

This is a linear AR(1) model, written so that $\bar u$ is the level the series reverts to and $\phi$ measures persistence.

The **random walk** is the special case $\phi = 1$, where the pull toward $\bar u$ vanishes.

The smaller $\phi$, the faster the series returns to $\bar u$ after a shock.

### Priors

We treat $\bar u$, $\phi$ and $\sigma$ as unknown and place weakly informative priors on them.

We give $\phi$ a uniform prior on $[0, 1)$ — this imposes stationarity (ruling out the explosive case) while letting the data decide how close to a random walk we are.

We center $\bar u$ on a plausible natural rate with a fairly wide normal prior, and give the shock scale $\sigma$ a half-normal prior.

We write the model as a NumPyro function: each `numpyro.sample` introduces a random variable, and the keyword `obs=` ties the last one to the data, supplying the likelihood.

```{code-cell} ipython3
def linear_model(u):
    ubar = numpyro.sample("ubar",  dist.Normal(5.5, 2.0))
    φ    = numpyro.sample("phi",   dist.Uniform(0.0, 1.0))
    σ    = numpyro.sample("sigma", dist.HalfNormal(1.0))
    μ = ubar + φ * (u[:-1] - ubar)
    numpyro.sample("u_obs", dist.Normal(μ, σ), obs=u[1:])
```

The vector `μ` holds the conditional means $\bar u + \phi(u_t - \bar u)$, and `observed=u[1:]` says that each next value is drawn from $N(\mu_t, \sigma^2)$ — so this one statement encodes the whole likelihood. (See {doc}`bayes_nonconj` for more on writing NumPyro models.)

### Estimation

We sample the posterior with NUTS, running four chains so that we can check convergence.

We use `chain_method="vectorized"`, which evaluates all chains together on a single device, so the same code runs unchanged on a CPU or a GPU.

```{code-cell} ipython3
def run_nuts(model, data, seed=0, num_warmup=1000, num_samples=2000, num_chains=4):
    "Sample a NumPyro model with the NUTS sampler."
    mcmc = MCMC(NUTS(model),
                num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="vectorized",
                progress_bar=False)
    mcmc.run(random.PRNGKey(seed), jnp.asarray(data))
    return mcmc
```

We fit the model to the monthly data.

```{code-cell} ipython3
:tags: [hide-output]

mcmc_monthly = run_nuts(linear_model, u_monthly)
```

Now we inspect the output.

```{code-cell} ipython3
mcmc_monthly.print_summary()
```

Each row summarizes the posterior for one parameter.

The `mean`, `median`, and `5.0%`/`95.0%` columns give the posterior mean, median, and a 90% credible interval; `std` is the posterior standard deviation.

The last two columns are convergence diagnostics: `n_eff` is the effective number of independent draws, and `r_hat` compares variation within and across chains — a value very close to $1.0$ means the chains agree and the sampler has converged.

Here `r_hat` is essentially one and `n_eff` is large, so we can trust the draws.

The number that matters for us is the posterior for $\phi$: its mass crowds right up against one.

In other words, at a monthly frequency, US unemployment is *almost* a random walk — the estimated pull back toward $\bar u$ is barely distinguishable from no pull at all.

### The hysteresis debate

A persistence of essentially one is exactly what one side of a long-running macroeconomics debate predicted.

The **natural rate hypothesis** {cite}`friedman1968role` holds that unemployment fluctuates around a stable equilibrium rate, so shocks fade away and the series is stationary — a $\phi$ below one.

The **hysteresis hypothesis** {cite}`blanchard_summers1986` holds the opposite: shocks to unemployment can be more or less permanent, so the series behaves like a random walk with no fixed level to return to — a $\phi$ equal to one.

Our monthly estimate sits right at the hysteresis boundary.

There is an irony here.

{cite:t}`nelson_plosser1982`, the study that launched the unit-root literature, examined fourteen US macroeconomic series and could reject the unit root for only *one* of them — the unemployment rate — even as the hysteresis literature was arguing the reverse for Europe.

## Is it a unit root?

The monthly fit cannot tell a unit root ($\phi = 1$) from a value just below it.

So could unemployment really be a random walk?

We give two reasons to think not: first the evidence from annual data, then a theoretical argument.

### Evidence from annual data

We form an annual series from end-of-year values and fit the same model to it.

```{code-cell} ipython3
u_annual = pre_covid.resample("YE").last().to_numpy()
print(f"{len(u_annual)} annual observations")
```

We fit the same model to this series.

```{code-cell} ipython3
:tags: [hide-output]

mcmc_annual = run_nuts(linear_model, u_annual)
```

Now we compare the posterior for $\phi$ across the two frequencies.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Posterior for the persistence parameter
    name: fig-phi-post
---
φ_m = np.asarray(mcmc_monthly.get_samples()["phi"])
φ_a = np.asarray(mcmc_annual.get_samples()["phi"])

fig, ax = plt.subplots()
ax.hist(φ_m, bins=50, density=True, alpha=0.6, label='monthly')
ax.hist(φ_a, bins=50, density=True, alpha=0.6, label='annual')
ax.set_xlabel('$\\phi$')
ax.legend()
plt.show()
```

{numref}`fig-phi-post` tells the story: the annual posterior for $\phi$ sits well below one, with clear reversion, while the monthly posterior is jammed against the boundary.

This is **not** a sign of different dynamics at the two frequencies.

A single linear AR(1) already predicts it: if the monthly persistence is $\phi$, then for end-of-year values the persistence is about $\phi^{12}$, and raising a number near one to the twelfth power pulls it appreciably below one — broadly in line with our annual estimate.

The reversion was there all along; month to month it is so slight as to be invisible, but over a year it accumulates into something we can clearly see and measure.

The annual data simply give us the statistical power to detect it.

We can put a number on the speeds using the **half-life** of a shock.

Suppose unemployment sits a gap $g$ above $\bar u$ and no further shocks arrive.

From {eq}`eq:linear` the gap is $\phi g$ next period, $\phi^2 g$ the period after, and $\phi^k g$ after $k$ periods — it shrinks by the factor $\phi$ every period, like radioactive decay.

The half-life is the number of periods $k$ for the gap to halve, so setting $\phi^k = \tfrac12$ and taking logs gives $k = \ln 0.5 / \ln \phi$.

```{code-cell} ipython3
def describe(mcmc, label, unit):
    p = mcmc.get_samples()
    φ = np.asarray(p["phi"])
    half_life = np.median(np.log(0.5) / np.log(φ))
    print(f"{label}: φ median {np.median(φ):.3f}, "
          f"half-life median {half_life:.0f} {unit}")

describe(mcmc_monthly, "monthly", "months")
describe(mcmc_annual, "annual ", "years")
```

At the annual frequency a shock decays with a half-life of only a few years — robust, visible reversion toward the natural rate.

At the monthly frequency the half-life runs to the better part of a decade, which is exactly why, over a few hundred months, the series looks like a random walk.

### A random walk would wander off

The second reason is decisive and needs no estimation.

Return to the possibility that $\phi = 1$ exactly, so the model is a pure random walk,

$$
u_{t+1} = u_t + \varepsilon_{t+1}, \qquad \varepsilon_{t+1} \sim N(0, \sigma^2).
$$

Then $u_t = u_0 + \sum_{s=1}^t \varepsilon_s$, so the variance grows without bound, $\operatorname{Var}(u_t) = t\sigma^2$.

The distribution spreads out forever, and eventually probability mass drains out of *every* bounded interval.

We can see this by simulating many random-walk paths and watching them fan out.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Random walks leave the observed range
    name: fig-rw-escape
---
rng = np.random.default_rng(0)
T = len(u_monthly)
σ_rw = np.diff(u_monthly).std()
paths = u_monthly[0] + np.cumsum(rng.normal(0, σ_rw, size=(400, T)), axis=1)

u_min, u_max = u_monthly.min(), u_monthly.max()

fig, ax = plt.subplots()
ax.plot(paths[:60].T, color='C0', lw=0.5, alpha=0.3)
ax.axhspan(u_min, u_max, color='C1', alpha=0.15, label='observed range')
ax.axhline(u_min, color='C1', ls='--', lw=1.5)
ax.axhline(u_max, color='C1', ls='--', lw=1.5)
ax.set_xlabel('months since 1948')
ax.set_ylabel('unemployment rate (%)')
ax.legend()
plt.show()
```

In {numref}`fig-rw-escape` the dashed lines mark the lowest and highest unemployment rates seen in the data, and the shaded region is the band between them.

The simulated paths fan out like $\sqrt{t}$ and quickly spread far beyond this band — into negative rates and rates above 15% — whereas actual unemployment has never left it.

A genuine random walk has no anchor, but unemployment clearly does, so it cannot literally be one.

## Reconciling the two views

We are left with a puzzle.

At a monthly frequency the data look like a random walk, and the persistence alone cannot rule one out.

Yet a literal random walk is impossible — it would wander out of the historical band — and at an annual frequency we can plainly see the series reverting.

A linear AR(1) does reconcile these facts: with $\phi$ a little below one it is stationary and bounded, while reverting only slowly.

But that reconciliation sits on a knife-edge.

The persistence has to be placed just below one — a value the monthly data cannot distinguish from a unit root {cite}`roed1997hysteresis` — and the model assumes the *same* gentle reversion at all times, however far unemployment strays.

A more robust reconciliation lets the reversion be **nonlinear**: the series can drift like a random walk in normal times while a firmer pull, switching on when it strays far from its normal level, keeps it from ever wandering away {cite}`kapetanios_shin_snell2003`.

We take up that nonlinear model in {doc}`unemployment_nonlinear`.

## Exercises

The lecture {doc}`ar1_turningpts` forecasts an AR(1) process by simulating future paths, and draws a careful distinction between a predictive distribution that **conditions on** fixed parameter values and one that **integrates over** their posterior uncertainty.

The following exercises apply that methodology to our fitted unemployment model.

```{exercise}
:label: unemp_lin_ex1

Using the fitted **annual** model, forecast unemployment over the next $H = 15$ years, starting from the last observed value, in two ways:

1. **plug-in**: hold the parameters at their posterior medians and simulate many future paths;
2. **extended**: for each future path, draw a fresh $(\bar u, \phi, \sigma)$ from the posterior.

Plot the 90% predictive band for each on the same axes and compare.

Which band is wider, and why?
```

```{solution-start} unemp_lin_ex1
:class: dropdown
```

We use the annual posterior draws and simulate forward from the last observation.

We work with the annual model because its persistence $\phi$ is far less certain than at the monthly frequency, so parameter uncertainty has more to say.

```{code-cell} ipython3
post = mcmc_annual.get_samples()
ubar_s = np.asarray(post["ubar"])
φ_s = np.asarray(post["phi"])
σ_s = np.asarray(post["sigma"])

def sim_future(u_last, ubar, φ, σ, H, rng):
    "Simulate H steps of the linear model forward from u_last."
    u = np.empty(H)
    prev = u_last
    for h in range(H):
        prev = ubar + φ * (prev - ubar) + rng.normal(0, σ)
        u[h] = prev
    return u

H, N = 15, 2000
u_last = u_annual[-1]
rng = np.random.default_rng(0)

# plug-in: parameters fixed at posterior medians
ub0, φ0, σ0 = np.median(ubar_s), np.median(φ_s), np.median(σ_s)
plug = np.array([sim_future(u_last, ub0, φ0, σ0, H, rng) for _ in range(N)])

# extended: a fresh posterior draw for each path
idx = rng.integers(0, len(φ_s), N)
ext = np.array([sim_future(u_last, ubar_s[i], φ_s[i], σ_s[i], H, rng)
                for i in idx])

fig, ax = plt.subplots()
horizon = np.arange(1, H + 1)
for data, c, lab in [(plug, 'C0', 'plug-in'), (ext, 'C1', 'extended')]:
    lo, hi = np.percentile(data, [5, 95], axis=0)
    ax.fill_between(horizon, lo, hi, alpha=0.3, color=c, label=lab)
ax.axhline(u_last, color='k', lw=0.5)
ax.set_xlabel('years ahead')
ax.set_ylabel('unemployment rate (%)')
ax.legend()
plt.show()
```

Both forecasts drift up from the low starting value toward the reversion level $\bar u$, with bands that widen as we look further ahead.

The extended band is clearly wider, because it adds uncertainty about the parameters on top of the uncertainty from future shocks — and here the persistence $\phi$ and the level $\bar u$ are both genuinely uncertain.

```{solution-end}
```

```{exercise}
:label: unemp_lin_ex2

Following Wecker (see {doc}`ar1_turningpts`), we can also form a predictive distribution for a **path statistic** — a nonlinear function of the whole future path.

Take the **maximum unemployment rate over the next $H = 8$ years** as the statistic: a simple measure of how bad the next several years might get.

Using the extended simulation (a posterior draw per path), compute this maximum for each path and plot its predictive distribution.

What is the posterior predictive probability that unemployment reaches at least $7\%$ — recession territory — at some point over the next eight years?
```

```{solution-start} unemp_lin_ex2
:class: dropdown
```

We reuse `sim_future` and the posterior draws from the previous exercise.

```{code-cell} ipython3
H2, M = 8, 5000
idx = rng.integers(0, len(φ_s), M)
peak = np.array([sim_future(u_last, ubar_s[i], φ_s[i], σ_s[i], H2, rng).max()
                 for i in idx])

fig, ax = plt.subplots()
ax.hist(peak, bins=50, density=True, alpha=0.6)
ax.axvline(u_last, color='C3', lw=2, label='current rate')
ax.set_xlabel('maximum unemployment over next 8 years (%)')
ax.legend()
plt.show()

prob = (peak > 7.0).mean()
print(f"P(unemployment reaches 7% within 8 years) = {prob:.2f}")
```

The predictive distribution summarizes the plausible "worst case" over the next several years, integrating over both shocks and parameter uncertainty.

Starting from a cyclical low, the model sees a substantial chance of a return to recession-level unemployment within the horizon.

```{solution-end}
```
