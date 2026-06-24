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

# Bayesian Estimation of Nonlinear Unemployment Dynamics

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

This lecture is a sequel to {doc}`unemployment_linear`.

There we fit linear models to the US unemployment rate and ran into a tension.

At a monthly frequency the series looks almost like a random walk.

Yet a random walk wanders off without limit, while unemployment stays in a band.

One way to accommodate both facts is a **nonlinear** model: one that drifts like a random walk in normal times, but whose pull strengthens far from the normal level, so the series is kept from ever wandering away.

The model is a one-dimensional stochastic difference equation — simple enough to understand completely, yet rich enough to be interesting.

We study its dynamics, estimate it from US data, and then ask whether the nonlinearity earns its keep.

For estimation we again use the No-U-Turn sampler (NUTS) in [NumPyro](https://num.pyro.ai/en/stable/); see {doc}`bayes_nonconj` for a brief account of how it works.

Our plan is:

1. write down the model and understand its restoring force,
2. study its dynamics — stability, mean reversion, and the stationary distribution,
3. estimate it from US unemployment data and see what the data can and cannot tell us, and
4. compare it head-to-head with the linear model.

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

## The model

Let $u_t$ denote the unemployment rate at time $t$, in percentage points.

We keep the mean-reverting form of the linear model but **cap how far the pull can carry the series**:

$$
u_{t+1} = \bar u + A\,\tanh\!\Big(\frac{\phi\,(u_t - \bar u)}{A}\Big) + \varepsilon_{t+1},
\qquad \varepsilon_{t+1} \sim N(0, \sigma^2),
$$ (eq:model)

with $0 \le \phi < 1$ and $A > 0$.

The model has four parameters:

| symbol | name | role |
| --- | --- | --- |
| $\bar u$ | natural rate | the level the series is pulled toward |
| $\phi$ | persistence | the slope near $\bar u$ — *the same $\phi$ as in the linear lecture* |
| $A$ | ceiling | half-width of the band the series is confined to |
| $\sigma$ | shock volatility | standard deviation of the shocks |

The map behind the deterministic part is

$$
g(u) = \bar u + A\,\tanh\!\big(\phi\,(u - \bar u)/A\big),
$$

an **S-shaped** function. Its two limits explain the two parameters.

**Near $\bar u$**, $\tanh(x) \approx x$, so

$$
g(u) \approx \bar u + \phi\,(u - \bar u),
$$

which is *exactly the linear AR(1) of {doc}`unemployment_linear`*, with the same persistence $\phi$.

**Far from $\bar u$**, $\tanh$ saturates, so $g(u) \to \bar u \pm A$: the map flattens to horizontal asymptotes, and the series is pulled firmly back toward the band $(\bar u - A,\ \bar u + A)$.

So the model behaves like the linear model in normal times but bends to keep unemployment bounded.

In fact the linear model is the limit $A \to \infty$ — remove the ceiling and the S-curve straightens into a line.

```{note}
This is a deliberately simple model, and the bending has a cost.

Because the map flattens, the implied one-step pull *grows* with the distance from the band — at extreme values the model snaps back hard, whereas real recoveries from deep recessions are gradual.

So the model will sometimes miss the data at the extremes.

What it buys in return is exactly what the narrative needs: random-walk-like behaviour in the normal band, together with a stationary distribution — probability mass never escapes to infinity.
```

## Dynamics

We now study how the model behaves, setting the shocks aside for a moment.

Our roadmap is: the deterministic skeleton and its 45-degree diagram, then the separate roles of $\phi$ and $A$, and finally the stationary distribution that emerges once the shocks return.

### The deterministic skeleton

Setting $\varepsilon_{t+1}$ to its mean of zero turns {eq}`eq:model` into the deterministic map $u_{t+1} = g(u_t)$.

The function below evaluates it.

```{code-cell} ipython3
def g(u, ubar, A, φ):
    "The deterministic skeleton of the nonlinear model."
    return ubar + A * np.tanh(φ * (u - ubar) / A)
```

A **45-degree diagram** plots $g$ against the line $u_{t+1} = u_t$; fixed points sit where they cross, and the slope of $g$ there tells us about stability.

The next cell draws the diagram for illustrative parameters and adds **cobweb** paths from two starting points.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The S-shaped map and cobweb paths
    name: fig-skeleton
---
ubar, A, φ = 5.0, 4.0, 0.95
grid = np.linspace(0, 12, 400)

fig, ax = plt.subplots()
ax.plot(grid, g(grid, ubar, A, φ), lw=2, label='$g(u)$')
ax.plot(grid, grid, 'k--', lw=1, label='45 degrees')
for a in (ubar - A, ubar + A):
    ax.axhline(a, color='C1', ls=':', lw=1)
ax.axvline(ubar, color='C2', ls=':', lw=1, label='$\\bar u$')

def cobweb(u0, n=40):
    xs, ys, u = [], [], u0
    for _ in range(n):
        xs += [u, u]
        ys += [u, g(u, ubar, A, φ)]
        u = g(u, ubar, A, φ)
    return xs, ys

for u0, c in [(11.0, 'C3'), (0.7, 'C4')]:
    xs, ys = cobweb(u0)
    ax.plot(xs, ys, c, lw=1, alpha=0.8)

ax.set_xlabel('$u_t$')
ax.set_ylabel('$u_{t+1}$')
ax.legend()
plt.show()
```

{numref}`fig-skeleton` shows the S-shape at work.

The map crosses the 45-degree line once, at $\bar u$, so $\bar u$ is the unique rest point.

Through the band the curve hugs the 45-degree line, so a cobweb there takes tiny steps — random-walk-like drift.

Outside the band the curve flattens toward the asymptotes $\bar u \pm A$ (the dotted lines), so a path that strays far is snapped back hard, as the cobweb from $u_0 = 11$ shows.

We can confirm the local picture analytically.

Differentiating $g$,

$$
g'(u) = \phi\,\operatorname{sech}^2\!\big(\phi(u-\bar u)/A\big),
\qquad\text{so}\qquad
g'(\bar u) = \phi .
$$

Near $\bar u$ the gap therefore evolves as $g_{t+1} \approx \phi\, g_t$ — a linear AR(1) with persistence $\phi$, exactly as in the linear lecture.

Far from $\bar u$ the slope falls to zero, which is the flattening that bounds the series.

### The roles of $\phi$ and $A$

The two parameters control two different things, and {numref}`fig-roles` separates them.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Persistence sets the centre, the ceiling sets the band
    name: fig-roles
---
grid = np.linspace(0, 12, 400)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

for φ_ in (0.6, 0.9, 1.0):
    axL.plot(grid, g(grid, 5.0, 4.0, φ_), lw=2, label=f'$\\phi={φ_}$')
axL.plot(grid, grid, 'k--', lw=1)
axL.set_title('vary $\\phi$ (fixed $A=4$)')
axL.set_xlabel('$u_t$')
axL.set_ylabel('$u_{t+1}$')
axL.legend()

for A_ in (2.0, 4.0, 8.0):
    axR.plot(grid, g(grid, 5.0, A_, 0.95), lw=2, label=f'$A={A_}$')
axR.plot(grid, grid, 'k--', lw=1)
axR.set_title('vary $A$ (fixed $\\phi=0.95$)')
axR.set_xlabel('$u_t$')
axR.legend()
plt.show()
```

The left panel varies $\phi$: it sets the **slope at the centre** — how strongly the series reverts in the normal band, from clear reversion ($\phi=0.6$) up to a random walk ($\phi=1$).

The right panel varies $A$: it sets the **ceiling** — the asymptotes $\bar u \pm A$, and hence how wide the band is and how sharply the map bends.

Notice in the right panel that the three curves are almost identical near $\bar u$ and only separate far out.

That is the key to estimation: near the centre the map depends only on $\phi$, so **$A$ can be learned only from large deviations** — the deep recessions.

### The stationary distribution

Now switch the shocks back on.

Because the map pulls every path back toward the band, and the shocks keep knocking it around, the process settles into a **stationary distribution**.

The function below simulates a path of {eq}`eq:model`.

```{code-cell} ipython3
def simulate(u0, T, ubar, A, φ, σ, rng):
    "Simulate a path of the nonlinear model from u0."
    u = np.empty(T)
    u[0] = u0
    ε = rng.normal(0, σ, T)
    for t in range(1, T):
        u[t] = ubar + A * np.tanh(φ * (u[t-1] - ubar) / A) + ε[t]
    return u
```

To see the stationarity, we run two long simulations from very different starting points and compare the distributions they visit.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Stationary distribution from two starting points
    name: fig-stationary
---
rng = np.random.default_rng(0)
T = 20_000
paths = {u0: simulate(u0, T, 5.0, 4.0, 0.97, 0.7, rng) for u0 in (2.0, 9.0)}

fig, ax = plt.subplots()
for u0, c in [(2.0, 'C0'), (9.0, 'C1')]:
    ax.hist(paths[u0][1000:], bins=60, density=True, alpha=0.5,
            label=f'start $u_0 = {u0}$')
ax.axvline(5.0, color='C2', ls=':', label='$\\bar u$')
ax.set_xlabel('$u$')
ax.legend()
plt.show()
```

The two histograms in {numref}`fig-stationary` lie almost on top of one another: the process forgets where it started and settles into the same distribution either way.

Even with persistence close to one, the ceiling keeps the distribution concentrated — probability mass does not escape to infinity, the failing of the random walk.

## The data

We use the same data as in {doc}`unemployment_linear`: the US unemployment rate (`UNRATE` from FRED), excluding the COVID-19 spike, both monthly and as an annual end-of-year series.

```{code-cell} ipython3
start, end = dt.datetime(1948, 1, 1), dt.datetime(2024, 12, 31)
unrate = web.DataReader("UNRATE", "fred", start, end)["UNRATE"]
pre_covid = unrate[unrate.index < "2020-01-01"]
u_monthly = pre_covid.to_numpy()
u_annual = pre_covid.resample("YE").last().to_numpy()
```

Recall from {doc}`unemployment_linear` that month to month the series barely moves, while year to year it makes large swings as recessions come and go.

That difference will again be central.

## Bayesian setup

Conditional on the parameters and the previous observation, {eq}`eq:model` makes each observation Gaussian, so the likelihood is a product of one-step normal densities and the posterior is proportional to likelihood times prior.

We use weakly informative priors.

```{code-cell} ipython3
def nonlinear_model(u):
    ubar = numpyro.sample("ubar",  dist.Normal(5.5, 2.0))
    φ    = numpyro.sample("phi",   dist.Uniform(0.0, 1.0))
    A    = numpyro.sample("A",     dist.HalfNormal(5.0))
    σ    = numpyro.sample("sigma", dist.HalfNormal(1.0))
    gap = u[:-1] - ubar
    μ = ubar + A * jnp.tanh(φ * gap / A)
    numpyro.sample("u_obs", dist.Normal(μ, σ), obs=u[1:])
```

The prior for $\phi$ is uniform on $[0,1)$, as in the linear lecture; $A$ gets a half-normal prior wide enough to cover bands a few percentage points either side of $\bar u$.

Before looking at the data, we check that these priors generate plausible paths.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Unemployment paths drawn from the prior
    name: fig-prior-pred
---
rng = np.random.default_rng(1)
T = len(u_annual)
fig, ax = plt.subplots()
for _ in range(30):
    ubar = rng.normal(5.5, 2.0)
    φ = rng.uniform(0.0, 1.0)
    A = abs(rng.normal(0, 5.0))
    σ = abs(rng.normal(0, 1.0))
    ax.plot(simulate(u_annual[0], T, ubar, A, φ, σ, rng), lw=1, alpha=0.5)
ax.set_xlabel('$t$')
ax.set_ylabel('unemployment rate (%)')
plt.show()
```

The prior paths in {numref}`fig-prior-pred` mostly stay in a plausible range — a few dip slightly below zero, a flaw we take up in the exercises — so we proceed.

We sample with NUTS, four chains, using `chain_method="vectorized"` so the same code runs on a CPU or a GPU.

```{code-cell} ipython3
def run_nuts(model, data, seed=0, num_warmup=1500, num_samples=4000, num_chains=4):
    "Sample a NumPyro model with the NUTS sampler."
    mcmc = MCMC(NUTS(model),
                num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="vectorized",
                progress_bar=False)
    mcmc.run(random.PRNGKey(seed), jnp.asarray(data))
    return mcmc
```

## Estimation

We fit the model at both frequencies and compare what the data reveal about $\phi$ and $A$.

```{code-cell} ipython3
:tags: [hide-output]

mcmc_monthly = run_nuts(nonlinear_model, u_monthly)
mcmc_annual = run_nuts(nonlinear_model, u_annual)
```

The two posteriors tell different stories, so we plot them side by side.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Posteriors for the persistence and the ceiling
    name: fig-phiA-post
---
def post(mcmc, name):
    return np.asarray(mcmc.get_samples()[name])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
axL.hist(post(mcmc_monthly, "phi"), bins=50, density=True, alpha=0.6, label='monthly')
axL.hist(post(mcmc_annual, "phi"), bins=50, density=True, alpha=0.6, label='annual')
axL.set_xlabel('$\\phi$ (persistence)')
axL.legend()
axR.hist(post(mcmc_monthly, "A"), bins=50, density=True, alpha=0.6, label='monthly')
axR.hist(post(mcmc_annual, "A"), bins=50, density=True, alpha=0.6, label='annual')
axR.set_xlabel('$A$ (ceiling)')
axR.legend()
plt.show()
```

The left panel of {numref}`fig-phiA-post` repeats the linear lecture's finding: monthly $\phi$ crowds against one (random-walk-like), while annual $\phi$ sits clearly below it.

The right panel is the new part.

At the monthly frequency the posterior for $A$ is pushed up as far as the prior allows: the monthly data see no curvature, so they cannot detect a ceiling and are content with the linear ($A\to\infty$) limit.

At the annual frequency $A$ settles to a finite value of a few percentage points: the deep recessions bend the curve, so the data prefer a genuine ceiling — though, as the wide posterior shows, only weakly.

This is the same lesson as before: the nonlinearity is identified only by the large-gap episodes, which are rare, so the data speak about it softly.

### Linear versus nonlinear

Is the nonlinear model actually better than the linear one?

To compare them on equal footing, we refit the linear model on the annual data.

```{code-cell} ipython3
def linear_model(u):
    ubar = numpyro.sample("ubar",  dist.Normal(5.5, 2.0))
    φ    = numpyro.sample("phi",   dist.Uniform(0.0, 1.0))
    σ    = numpyro.sample("sigma", dist.HalfNormal(1.0))
    μ = ubar + φ * (u[:-1] - ubar)
    numpyro.sample("u_obs", dist.Normal(μ, σ), obs=u[1:])
```

We sample it on the same data.

```{code-cell} ipython3
:tags: [hide-output]

mcmc_lin = run_nuts(linear_model, u_annual)
```

Now we overlay the two fitted maps $g(u)$, with 90% posterior bands, over the range the data visit.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Fitted maps, linear versus nonlinear
    name: fig-map-compare
---
uu = np.linspace(2, 12, 200)

an = mcmc_annual.get_samples()
ub_n, A_n, φ_n = (np.asarray(an[k]) for k in ("ubar", "A", "phi"))
maps_nl = ub_n[:, None] + A_n[:, None] * np.tanh(φ_n[:, None] * (uu[None, :] - ub_n[:, None]) / A_n[:, None])

al = mcmc_lin.get_samples()
ub_l, φ_l = np.asarray(al["ubar"]), np.asarray(al["phi"])
maps_lin = ub_l[:, None] + φ_l[:, None] * (uu[None, :] - ub_l[:, None])

fig, ax = plt.subplots()
for m, color, lab in [(maps_lin, 'C1', 'linear'), (maps_nl, 'C0', 'nonlinear')]:
    ax.plot(uu, np.median(m, 0), color=color, lw=2, label=lab)
    ax.fill_between(uu, np.percentile(m, 5, 0), np.percentile(m, 95, 0),
                    color=color, alpha=0.2)
ax.plot(uu, uu, 'k--', lw=1, label='45 degrees')
ax.set_xlabel('$u_t$')
ax.set_ylabel('$u_{t+1}$')
ax.legend()
plt.show()
```

In {numref}`fig-map-compare` the two fitted maps are close over the bulk of the data and their bands overlap: in ordinary times the linear model is hard to beat.

They part company only at the extremes, where the nonlinear map bends toward its ceiling while the linear one keeps going straight.

So the curvature earns its keep exactly where we argued it should, and essentially nowhere else: in the deep recessions, not in normal times.

This is an honest verdict rather than a triumphant one — within ordinary experience the two models barely differ, the annual data mildly prefer the bounded one, and the monthly data cannot tell at all.

### A posterior predictive check

A fitted model should be able to generate data that look like what we observed.

We draw parameters from the annual posterior, simulate a path from each, and compare the band of simulated paths to the data.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Posterior predictive paths against the data
    name: fig-ppc
---
σ_n = np.asarray(an["sigma"])
rng = np.random.default_rng(2)
T = len(u_annual)
idx = rng.integers(0, len(φ_n), 400)
sims = np.array([simulate(u_annual[0], T, ub_n[i], A_n[i], φ_n[i], σ_n[i], rng)
                 for i in idx])
lo, mid, hi = np.percentile(sims, [5, 50, 95], axis=0)

fig, ax = plt.subplots()
yrs = pre_covid.resample("YE").last().index.year
ax.fill_between(yrs, lo, hi, alpha=0.3, label='90% predictive band')
ax.plot(yrs, mid, lw=2, label='predictive median')
ax.plot(yrs, u_annual, 'k', lw=2, label='observed')
ax.set_xlabel('year')
ax.set_ylabel('unemployment rate (%)')
ax.legend()
plt.show()
```

The observed series in {numref}`fig-ppc` stays within the predictive band, so the model reproduces the broad swings of post-war unemployment.

It is not perfect — the model's IID shocks miss some of the smooth, multi-year persistence of real recoveries — but it captures the level, the spread, and the mean reversion well.

## Conclusion

We built a compact nonlinear model of unemployment and took it from theory to data.

Its dynamics are clean: an S-shaped map that runs along the 45-degree line in the normal band, so the series behaves like a random walk there, but flattens toward a ceiling $\bar u \pm A$ that keeps it from ever wandering away.

The persistence $\phi$ is the very parameter we estimated in {doc}`unemployment_linear`, and the linear model is the no-ceiling limit $A \to \infty$, so the two lectures fit together as one story.

The estimation taught a Bayesian lesson about identification: at a monthly frequency the data look linear and cannot detect the ceiling, while the annual recessions pull $A$ to a finite value — though only weakly, because the nonlinearity matters only at the extremes.

What the data can tell you depends on what the data have seen.

The model is deliberately simple, and its growing pull at the extremes is a real limitation — the exercises and the wider literature take up the natural refinements.

## Exercises

```{exercise}
:label: unemp_nl_ex1

Real unemployment rises quickly in recessions but falls slowly in recoveries — an asymmetry the symmetric S-curve cannot capture.

Modify the model so the ceiling is asymmetric: a larger $A$ above $\bar u$ than below it.

Refit on the annual data and report the posterior for the difference between the two ceilings. Is there evidence of asymmetry?
```

```{exercise}
:label: unemp_nl_ex2

The model allows a negative unemployment rate, even if the event is unlikely.

Build a version that respects the range exactly by modelling $x_t = \operatorname{logit}(u_t / 100)$ with the same S-shaped reversion, refit, and compare the implied dynamics with the model in the text.
```

```{exercise}
:label: unemp_nl_ex3

Reintroduce the COVID-19 observations you dropped.

Using the annual posterior fitted without them, compute the posterior predictive probability of an annual unemployment rate as high as the 2020 value.

How surprising is the COVID spike under the estimated model?
```
