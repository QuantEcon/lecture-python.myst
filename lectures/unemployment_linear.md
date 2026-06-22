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

This lecture fits some simple time series models to the US unemployment rate.

Our aim is partly to practice Bayesian estimation on real data, and partly to find out what these simple models *cannot* do — which will motivate the richer model in the sequel {doc}`unemployment_nonlinear`.

We met Bayesian estimation of a first-order autoregression (AR(1)) in {doc}`ar1_bayes`, but there the data were simulated and the focus was on the initial condition.

Here the data are real, and we try to describe them with an AR(1) model.

We will see how well that description works and where it breaks down.

We'll also ask whether or not unemployment is a random walk, which is a special case of the AR(1) model.

That last question is not just a technical curiosity — it was the subject of a lively macroeconomics debate in the 1980s and 1990s.

The **natural rate hypothesis** {cite}`friedman1968role` holds that unemployment fluctuates around a stable equilibrium rate, so shocks fade away and the series is stationary.

The **hysteresis hypothesis** {cite}`blanchard_summers1986` holds the opposite — that shocks to unemployment can be more or less permanent, so the series behaves like a random walk with no fixed level to return to.

(Interestingly, the paper that launched the unit-root literature {cite}`nelson_plosser1982` found that, of fourteen US macroeconomic series, the unemployment rate was the *one* it could confidently call stationary — even as the hysteresis literature was arguing the reverse for Europe.)

We will see that this debate is genuinely hard to settle.

As in {doc}`ar1_bayes` and {doc}`bayes_nonconj` we sample posteriors with the NUTS sampler in [NumPyro](https://num.pyro.ai/en/stable/); see {doc}`bayes_nonconj` for a brief account of how NUTS works.

Our plan is:

1. look at the data,
2. consider the simplest model — a random walk — and see what works and what doesn't,
3. add mean reversion to get a linear model and estimate it,
4. ask whether the data really want a random walk after all, and
5. lay out what is unsatisfying about the linear model.

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

We download the whole post-war record.

```{code-cell} ipython3
start, end = dt.datetime(1948, 1, 1), dt.datetime(2024, 12, 31)
unrate = web.DataReader("UNRATE", "fred", start, end)["UNRATE"]
```

The COVID-19 spike of 2020 is an extreme outlier driven by events the model knows nothing about, so we drop it.

We keep two versions of the series: the monthly data, and an annual series formed from end-of-year values.

```{code-cell} ipython3
pre_covid = unrate[unrate.index < "2020-01-01"]
u_monthly = pre_covid.to_numpy()
u_annual = pre_covid.resample("YE").last().to_numpy()
print(f"monthly: {len(u_monthly)} obs, annual: {len(u_annual)} obs")
```

Here are the two series.

```{code-cell} ipython3
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
axL.plot(pre_covid.index, u_monthly, lw=2)
axL.set_title('monthly')
axL.set_xlabel('year')
axL.set_ylabel('unemployment rate (%)')
axR.plot(pre_covid.resample("YE").last().index, u_annual, lw=2)
axR.set_title('annual')
axR.set_xlabel('year')
plt.show()
```

The rate rises sharply in recessions and drifts down in recoveries, but it always stays inside a band — roughly 3% to 11% over the whole post-war period.

Any sensible model has to respect that band.

## A random walk?

The simplest dynamic model just says that next period equals this period plus a shock,

$$
u_{t+1} = u_t + \varepsilon_{t+1}, \qquad \varepsilon_{t+1} \sim N(0, \sigma^2).
$$

This is a **random walk**.

It is worth taking seriously because, as we will see, the data come surprisingly close to it.

But as a description of unemployment it has a fatal flaw.

Under a random walk, $u_t = u_0 + \sum_{s=1}^t \varepsilon_s$, so the variance of $u_t$ grows without bound: $\operatorname{Var}(u_t) = t\sigma^2$.

The distribution spreads out forever, and eventually probability mass drains out of *every* bounded interval.

We can see the spreading by simulating many random-walk paths and watching them fan out.

```{code-cell} ipython3
rng = np.random.default_rng(0)
T = len(u_monthly)
σ_rw = np.diff(u_monthly).std()
paths = u_monthly[0] + np.cumsum(rng.normal(0, σ_rw, size=(400, T)), axis=1)

fig, ax = plt.subplots()
ax.plot(paths[:60].T, color='C0', lw=0.5, alpha=0.3)
lo, hi = np.percentile(paths, [5, 95], axis=0)
ax.fill_between(np.arange(T), lo, hi, color='C0', alpha=0.2,
                label='90% band')
ax.plot(u_monthly, 'k', lw=2, label='observed')
ax.set_xlabel('months since 1948')
ax.set_ylabel('unemployment rate (%)')
ax.legend()
plt.show()
```

The simulated band widens like $\sqrt{t}$ and soon covers negative rates and rates above 15%, while the actual series stays in its narrow band.

A random walk has no anchor, but unemployment clearly has one.

## Adding mean reversion

To give the series an anchor we let it be pulled back toward a normal level $\bar u$,

$$
u_{t+1} = \bar u + \phi\,(u_t - \bar u) + \varepsilon_{t+1},
\qquad \varepsilon_{t+1} \sim N(0, \sigma^2),
$$ (eq:linear)

with $0 \le \phi < 1$.

This is a linear AR(1) model for unemployment, written so that $\bar u$ is the level it reverts to and $\phi$ measures persistence.

The random walk is the special case $\phi = 1$, where the pull vanishes.

The smaller $\phi$, the faster the series returns to $\bar u$ after a shock.

We give $\phi$ a uniform prior on $[0, 1)$, so we let the data decide how close to a random walk we are.

```{code-cell} ipython3
def linear_model(u):
    ubar = numpyro.sample("ubar",  dist.Normal(5.5, 2.0))
    φ    = numpyro.sample("phi",   dist.Uniform(0.0, 1.0))
    σ    = numpyro.sample("sigma", dist.HalfNormal(1.0))
    μ = ubar + φ * (u[:-1] - ubar)
    numpyro.sample("u_obs", dist.Normal(μ, σ), obs=u[1:])
```

We sample the posterior with NUTS, running four chains so we can check convergence.

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

We fit the model to the monthly data first.

```{code-cell} ipython3
:tags: [hide-output]

mcmc_monthly = run_nuts(linear_model, u_monthly)
```

The chains mix well, and here is the posterior summary.

```{code-cell} ipython3
mcmc_monthly.print_summary()
```

We then fit it to the annual data.

```{code-cell} ipython3
:tags: [hide-output]

mcmc_annual = run_nuts(linear_model, u_annual)
```

Here is the corresponding summary.

```{code-cell} ipython3
mcmc_annual.print_summary()
```

## Is it a random walk after all?

The persistence parameter $\phi$ tells the story, so we compare its posterior across the two frequencies.

```{code-cell} ipython3
φ_m = np.asarray(mcmc_monthly.get_samples()["phi"])
φ_a = np.asarray(mcmc_annual.get_samples()["phi"])

fig, ax = plt.subplots()
ax.hist(φ_m, bins=50, density=True, alpha=0.6, label='monthly')
ax.hist(φ_a, bins=50, density=True, alpha=0.6, label='annual')
ax.set_xlabel('$\\phi$')
ax.legend()
plt.show()
```

At the monthly frequency the posterior for $\phi$ crowds right up against one — the random-walk boundary.

In other words, month to month, US unemployment is *almost* a random walk: the pull back toward $\bar u$ is barely detectable.

Two summaries make the point.

The model has a stationary distribution $N\!\big(\bar u,\ \sigma^2/(1-\phi^2)\big)$, and a shock has a half-life of $\ln 0.5 / \ln \phi$ periods.

```{code-cell} ipython3
def describe(mcmc, label):
    p = mcmc.get_samples()
    φ, σ = np.asarray(p["phi"]), np.asarray(p["sigma"])
    half_life = np.log(0.5) / np.log(φ)
    stat_sd = σ / np.sqrt(1 - φ**2)
    print(f"{label}:")
    print(f"  φ          median {np.median(φ):.3f}  90% [{np.percentile(φ,5):.3f}, {np.percentile(φ,95):.3f}]")
    print(f"  half-life  median {np.median(half_life):.0f} periods")
    print(f"  stationary sd of u: median {np.median(stat_sd):.2f} pp")

describe(mcmc_monthly, "monthly")
describe(mcmc_annual, "annual")
```

At the monthly frequency the half-life of a shock runs to several years.

Over a sample of a few hundred months the series barely has time to revert, so it behaves, for practical purposes, like the random walk we just rejected.

At the annual frequency the picture is better: $\phi$ sits well below one and shocks die out within a few years.

The lesson is that whether unemployment "looks like a random walk" depends on how often you look.

```{note}
This is exactly the question behind the **natural rate versus hysteresis** debate of the 1980s and 1990s {cite}`friedman1968role,blanchard_summers1986`.

When $\phi$ is this close to one, the debate is hard to settle: in samples of the length we have, a true random walk and a slowly-reverting series look almost the same, so the statistical tests have little power to tell them apart {cite}`roed1997hysteresis`.

One way forward, which we take in {doc}`unemployment_nonlinear`, is to let the reversion be **nonlinear** — a series can look like a random walk to a linear test while reverting briskly once it strays far enough from its normal level {cite}`kapetanios_shin_snell2003`.
```

## Random walk, yet recurrent

The linear model is a clear improvement on the random walk — it has an anchor and a stationary distribution.

But at the frequencies we care about it is *barely* an improvement: $\phi$ sits so close to one that the anchor does almost no work, and over realistic horizons the series still behaves much like the random walk we rejected.

This leaves us with a genuine tension.

Read through a linear lens, the data look like a random walk.

But a random walk wanders off without limit, whereas unemployment has stayed in a narrow band for seventy-five years.

The linear model can reconcile these two facts, but only by setting $\phi$ a hair below one — a knife-edge value the data cannot distinguish from one, and one that imposes the same gentle reversion at all times.

A more robust reconciliation is to let the reversion be **nonlinear**.

The series can then drift like a random walk in normal times, while a restoring force that strengthens far from the natural rate keeps it from ever wandering away.

Recurrence then comes from the *shape* of the dynamics in the tails, not from a precise value of $\phi$ — and the series can be genuinely random-walk-like exactly where we usually observe it.

To set up what comes next, it helps to look at the restoring force directly, by plotting each one-step change against the gap that preceded it, with the fitted linear pull overlaid.

```{code-cell} ipython3
ubar_hat = np.median(np.asarray(mcmc_annual.get_samples()["ubar"]))
φ_hat = np.median(φ_a)
gap = u_annual[:-1] - ubar_hat
Δu = np.diff(u_annual)

fig, ax = plt.subplots()
ax.scatter(gap, Δu, alpha=0.6, label='annual data')
gg = np.linspace(gap.min(), gap.max(), 100)
ax.plot(gg, (φ_hat - 1) * gg, 'C1', lw=2, label='linear pull')
ax.axhline(0, color='k', lw=.5)
ax.axvline(0, color='k', lw=.5)
ax.set_xlabel('gap $u_t - \\bar u$')
ax.set_ylabel('change $\\Delta u$')
ax.legend()
plt.show()
```

The straight line is a reasonable summary of the average reversion in the data.

Whether a force that is **gentle near the center but firmer far from it** fits better — and whether the data can even tell — is the question we take up in {doc}`unemployment_nonlinear`.

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
