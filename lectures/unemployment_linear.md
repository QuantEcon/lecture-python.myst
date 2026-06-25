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

This lecture applies Bayesian estimation to a real macroeconomic series: the US unemployment rate.

We met the machinery in {doc}`ar1_bayes`, but there the data were simulated and the focus was theoretical.

Here we work carefully through real data, and we organize the lecture around a question that divided macroeconomists in the 1980s and 1990s: is unemployment a *random walk*?

Our plan is to

1. lay out the historical debate behind that question,
2. argue that a literal random walk is impossible, so we fit a *stationary* model,
3. fit a linear AR(1) to monthly data and find that its persistence is very close to one,
4. contrast what the monthly and annual frequencies can tell us, and
5. close by noting an asymmetry the model cannot capture — which motivates the sequel, {doc}`unemployment_shocks`.

As in {doc}`ar1_bayes` and {doc}`bayes_nonconj`, we estimate by sampling posteriors with the NUTS sampler in [NumPyro](https://num.pyro.ai/en/stable/); see {doc}`bayes_nonconj` for a brief account of how NUTS works.

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

## The natural rate versus hysteresis

Whether unemployment is a random walk is not a dry statistical question — it sits on a fault line in macroeconomics.

The **natural rate hypothesis** {cite}`friedman1968role` holds that unemployment fluctuates around a stable equilibrium rate.

On this view shocks fade away and the series reverts to its normal level — in the language of the model below, a persistence $\phi$ below one.

The **hysteresis hypothesis** {cite}`blanchard_summers1986` holds the opposite: a shock to unemployment can be more or less permanent, because spells of joblessness erode skills and attachment to the labour force.

On this view there is no fixed level to return to, and the series behaves like a random walk — a persistence $\phi$ equal to one.

There is an irony in the history.

{cite:t}`nelson_plosser1982`, the study that launched the unit-root literature, examined fourteen US macroeconomic series and could reject the random walk for only *one* of them — the unemployment rate — even as the hysteresis literature was arguing the reverse for Europe.

The debate matters because the two views imply very different policy: if shocks are permanent, a deep recession leaves a lasting scar.

We bring Bayesian estimation to bear on it.

## The data

We use the US civilian unemployment rate, series `UNRATE` from FRED, monthly and seasonally adjusted.

```{code-cell} ipython3
start, end = dt.datetime(1948, 1, 1), dt.datetime(2024, 12, 31)
unrate = web.DataReader("UNRATE", "fred", start, end)["UNRATE"]
```

The COVID-19 spike of 2020 is an extreme outlier driven by events our models know nothing about, so we drop it.

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

Keep that band in mind: it is the first clue, and it tells against a literal random walk.

## A random walk would wander off

Before fitting anything, we can settle part of the question with a simple argument.

Suppose unemployment really were a pure random walk, $\phi = 1$:

$$
u_{t+1} = u_t + \varepsilon_{t+1}, \qquad \varepsilon_{t+1} \sim N(0, \sigma^2).
$$

Then $u_t = u_0 + \sum_{s=1}^t \varepsilon_s$, so its variance grows without bound, $\operatorname{Var}(u_t) = t\sigma^2$.

The distribution spreads out forever, and eventually probability mass leaves *every* bounded interval.

We can see this by simulating many random-walk paths, using the observed one-month changes to set the shock size, and watching them fan out.

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

In {numref}`fig-rw-escape` the dashed lines mark the lowest and highest unemployment rates ever seen in the data, and the shaded region is the band between them.

The simulated paths fan out like $\sqrt{t}$ and quickly spread far beyond this band, including into negative rates.

A random walk has no anchor, but unemployment clearly does — it has stayed inside a narrow band for seventy years.

So we can already rule out a *literal* random walk.

This does not settle the debate, though: the interesting question is whether unemployment is *almost* a random walk — persistent enough that shocks die away only very slowly.

For that we need to estimate.

## A linear model of unemployment

### The model

We let unemployment be pulled back toward a normal level $\bar u$:

$$
u_{t+1} = \bar u + \phi\,(u_t - \bar u) + \varepsilon_{t+1},
\qquad \varepsilon_{t+1} \sim N(0, \sigma^2),
$$ (eq:linear)

with $0 \le \phi < 1$.

This is a linear AR(1) model, written so that $\bar u$ is the level the series reverts to and $\phi$ measures persistence.

The closer $\phi$ is to one, the more slowly the series returns to $\bar u$ after a shock; the random walk is the limiting case $\phi = 1$.

### Priors

We treat $\bar u$, $\phi$ and $\sigma$ as unknown and place weakly informative priors on them.

We give $\phi$ a uniform prior on $[0, 1)$.

The upper endpoint is excluded deliberately: we argued above that unemployment is bounded, which rules out a unit root on economic grounds.

This is not assuming our answer — the prior still lets $\phi$ approach one as closely as the data demand, and we will see that it does exactly that.

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

The vector `μ` holds the conditional means $\bar u + \phi(u_t - \bar u)$, and `obs=u[1:]` says that each next value is drawn from $N(\mu_t, \sigma^2)$ — so this one statement encodes the whole likelihood. (See {doc}`bayes_nonconj` for more on writing NumPyro models.)

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

In other words, at a monthly frequency, US unemployment is *almost* a random walk.

This is the hysteresis boundary.

At this persistence the natural-rate and hysteresis views are nearly indistinguishable in the monthly data — the estimate is consistent with both, which is exactly why the unit-root tests of the era struggled to settle the debate {cite}`roed1997hysteresis`.

## Monthly versus annual

The monthly data leave $\phi$ pinned against one. Does a different frequency see more?

We form an annual series from end-of-year values and fit the same model to it.

```{code-cell} ipython3
u_annual = pre_covid.resample("YE").last().to_numpy()
print(f"{len(u_annual)} annual observations")
```

We fit the same model to this shorter series.

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

{numref}`fig-phi-post` tells the story: the annual posterior for $\phi$ sits well below one, with clear reversion, while the monthly posterior pushes up against the boundary.

This is not a contradiction.

If the monthly persistence is $\phi$, then for end-of-year values the persistence is about $\phi^{12}$, and raising a number near one to the twelfth power pulls it appreciably below one — broadly in line with our annual estimate.

The reversion was there in the monthly data all along; month to month it is too slight to see, but over a year it accumulates into something we can measure.

The lesson is about identification: *what the data can tell you depends on the frequency you look at*.

## What the linear model misses

So far the linear model has served us well. But look again at {numref}`fig-unrate-monthly`: unemployment shoots *up* quickly in recessions and drifts *down* slowly in recoveries.

This is an asymmetry, and our model has no room for it.

To see the problem, we compute the model's one-step residuals at the posterior median — the part of each month's change the model cannot explain — and compare them with the symmetric Gaussian the model assumes.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Model residuals are right-skewed
    name: fig-resid-skew
---
def skewness(x):
    x = x - x.mean()
    return (x**3).mean() / x.std()**3

med = {k: np.median(np.asarray(mcmc_monthly.get_samples()[k]))
       for k in ("ubar", "phi", "sigma")}
resid = u_monthly[1:] - (med["ubar"] + med["phi"] * (u_monthly[:-1] - med["ubar"]))

fig, ax = plt.subplots()
ax.hist(resid, bins=60, density=True, alpha=0.6, label='residuals')
grid = np.linspace(resid.min(), resid.max(), 200)
gauss = np.exp(-grid**2 / (2 * resid.std()**2)) / (resid.std() * np.sqrt(2 * np.pi))
ax.plot(grid, gauss, 'C1', lw=2, label='symmetric Gaussian')
ax.set_xlabel('one-month change not explained by the model')
ax.legend()
plt.show()

print(f"residual skewness = {skewness(resid):.2f}")
```

The residuals in {numref}`fig-resid-skew` depart from the Gaussian in two ways.

They are **heavy-tailed and sharply peaked**: far more tiny changes, and more large ones, than a bell curve allows.

And they are **right-skewed** (a skewness of about $0.4$) — the largest surprises sit on the upside, the recessions, with nothing comparable on the downside.

The symmetric Gaussian (orange) can match neither feature: it treats upward and downward shocks as equally likely and has no room for the occasional violent jump.

So our model is bound to misread the data, seeing the rare jump up and the long gentle slide down as the *same* kind of shock.

Capturing that asymmetry is the task of the next lecture, {doc}`unemployment_shocks`, where we model the shocks themselves rather than the reversion curve.

We will find there that allowing for asymmetric shocks barely changes the estimated persistence — so the near-unit-root we found here is robust, not an artifact of the Gaussian assumption.

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
