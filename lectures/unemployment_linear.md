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

This lecture applies Bayesian estimation to a linear AR(1) model of the US unemployment rate.

We met the machinery in {doc}`ar1_bayes`, but there the data were simulated and the focus was theoretical.

Here we work carefully through real data, organizing the lecture around a
question that divided macroeconomists in the 1980s and 1990s: is unemployment a
*random walk*?

In the process, we will observe an asymmetry the model cannot capture.

This will motivate a sequel lecture, titled {doc}`unemployment_shocks`.

As in {doc}`ar1_bayes` and {doc}`bayes_nonconj`, we estimate by sampling
posteriors with the NUTS sampler in [NumPyro](https://num.pyro.ai/en/stable/).

(See the {ref}`introduction to NUTS <nuts>` in {doc}`bayes_nonconj` for a brief account of how it works.)

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

In the early 1980s, {cite:t}`nelson_plosser1982` launched the unit-root literature.

They examined 14 macroeconomic time series and found that they could reject the random walk for only one of them.

Roughly speaking, this meant that, for most macroeconomic series, the effects of shocks looked permanent rather than transitory.

For unemployment, that view found expression in the **hysteresis hypothesis** of {cite}`blanchard_summers1986`.

This hypothesis states that a shock to unemployment can be more or less permanent, because spells of joblessness erode skills and attachment to the labor force.

In contrast, the **natural rate hypothesis** of {cite}`friedman1968role` holds that unemployment fluctuates around a
stable equilibrium rate --- and hence shocks are transitory rather than permanent.

The debate matters because the two views imply different policies.

In particular, if shocks are permanent, a deep recession leaves a lasting scar, encouraging remedial action.

Here we reexamine the question with Bayesian estimation.



```{note}
There is an irony in the history described above.

While {cite:t}`nelson_plosser1982` fueled the unit root debate, and hence the hysteresis hypothesis, the one macroeconomic series 
they rejected the unit root for was the unemployment rate.
```


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

(Keep that band in mind, since it connects back to the unit root debate, as we discuss below.)


## A linear model of unemployment

Let's now set up and estimate our model, using monthly data.

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

The upper endpoint is excluded deliberately, since doing so confines us to the *stationary* region. 

This is necessary, as we'll see below.

At the same time, the prior still lets $\phi$ approach one as closely as the data demand.

We center $\bar u$ on a plausible natural rate with a fairly wide normal prior, and give the shock scale $\sigma$ a half-normal prior.

We write the model as a NumPyro function: each `numpyro.sample` introduces a random variable, and the keyword `obs=` ties the last one to the data, supplying the likelihood.

```{code-cell} ipython3
def linear_model(u):
    ubar = numpyro.sample("ubar",  dist.Normal(5.5, 2.0))      # Natural rate prior
    φ    = numpyro.sample("phi",   dist.Uniform(0.0, 1.0))     # Persistence prior
    σ    = numpyro.sample("sigma", dist.HalfNormal(1.0))       # Volatility prior
    μ = ubar + φ * (u[:-1] - ubar)
    numpyro.sample("u_obs", dist.Normal(μ, σ), obs=u[1:])
```

The vector `μ` holds the conditional means $\bar u + \phi(u_t - \bar u)$, and `obs=u[1:]` says that each next value is drawn from $N(\mu_t, \sigma^2)$. 

This one statement encodes the whole likelihood. (See {doc}`bayes_nonconj` for more on writing NumPyro models.)

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
    mcmc.run(random.key(seed), jnp.asarray(data))
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

In particular, the mean and median are very close to one, while the standard deviation is very small.

In other words, at a monthly frequency, US unemployment is *almost* a random walk.

This is the hysteresis boundary.

Thus, we find that the natural-rate and hysteresis views are nearly indistinguishable in the monthly data — the estimate is consistent with both.

This is why the unit-root tests of the era struggled to settle the debate {cite}`roed1997hysteresis`.


## A random walk would wander off

While the estimation above seems to give reasonable support to the unit root hypothesis, there is a good reason to view it as false.

To see the argument, suppose that unemployment really is a pure random walk, with $\phi = 1$:

$$
u_{t+1} = u_t + \varepsilon_{t+1}, \qquad \varepsilon_{t+1} \sim N(0, \sigma^2).
$$

Then $u_t = u_0 + \sum_{s=1}^t \varepsilon_s$, so its variance grows without bound: $\operatorname{Var}(u_t) = t\sigma^2$.

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

So we can already rule out an exact random walk.

This will become even clearer when we examine annual data.



## Monthly versus annual

The monthly data leave $\phi$ pinned against one. 

Does a different frequency allow us to see something more?

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

{numref}`fig-phi-post` illustrates our main finding: the annual posterior for $\phi$ sits well below one, with clear reversion, while the monthly posterior pushes up against the boundary.

This is not a contradiction.

If the monthly persistence is $\phi$, then for end-of-year values the persistence is about $\phi^{12}$, and raising a number near one to the twelfth power pulls it appreciably below one — in line with our annual estimate.



## What the model misses

Looking again at {numref}`fig-unrate-monthly`, we notice that unemployment jumps up quickly in recessions and drifts down slowly in recoveries.

This is an *asymmetry* that our current model cannot replicate.

To see why, we look closely at the one part of the model that is random — the shocks.

We proceed in three steps: state what the model assumes about the shocks, recover them from the data, and compare the two.

### What the model assumes about the shocks


Given last period's rate $u_t$ and the parameters, the next rate is a deterministic conditional mean plus a shock:

$$
u_{t+1} = \underbrace{\bar u + \phi\,(u_t - \bar u)}_{\text{conditional mean}} + \varepsilon_{t+1},
\qquad \varepsilon_{t+1} \sim N(0, \sigma^2).
$$

Rearranging, the shock is just the gap between what happened and what the model expected:

$$
\varepsilon_{t+1} = u_{t+1} - \big(\bar u + \phi\,(u_t - \bar u)\big).
$$

The model makes a strong and testable claim about these shocks: they are independent draws from a *symmetric* normal distribution.

If that claim holds, the shocks we recover from the data should look like a bell curve.

If it fails, the way it fails will tell us what the model is missing.

### Recovering the shocks

We cannot read the shocks off directly, because we do not know the parameters $\bar u$ and $\phi$.

So we estimate them, plugging a single representative value in for each.

We use the posterior medians — a reasonable choice for a quick diagnostic.

```{code-cell} ipython3
med = {k: np.median(np.asarray(mcmc_monthly.get_samples()[k]))
       for k in ("ubar", "phi", "sigma")}
resid = u_monthly[1:] - (med["ubar"] + med["phi"] * (u_monthly[:-1] - med["ubar"]))
```

The `resid` array holds our estimated shocks, one for each month-to-month transition — the model's **residuals**.

The slicing is what lines up each month with the one before it.

Each entry of `resid` is $u_{t+1} - \big(\hat{\bar u} + \hat\phi\,(u_t - \hat{\bar u})\big)$, the model's one-step-ahead forecast error, with hats denoting the median estimates.

(Because our estimate $\hat\phi$ is so close to one, this is nearly just the monthly change $u_{t+1} - u_t$.)


### Comparing with the Gaussian

Now we ask whether these residuals look Gaussian.

We overlay a normal density whose standard deviation we set equal to that of the residuals themselves.

This is deliberate: the residuals already have mean near zero, so matching the variance makes the mean and the spread agree.

Anything left over is then a difference in *shape* — which is what we want to isolate.

One way to measure the shape is the **skewness**, the third standardized moment:

$$
\text{skew} = \frac{\frac1n \sum_i (\varepsilon_i - \bar\varepsilon)^3}{\Big(\frac1n \sum_i (\varepsilon_i - \bar\varepsilon)^2\Big)^{3/2}}.
$$

This measure is zero for any symmetric distribution and positive when the right tail is the longer one.

We now plot the residuals and compute their skewness:

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

They are *heavy-tailed* and *sharply peaked*: more tiny changes, and more large ones, than a bell curve allows.

Moreover, they are *right-skewed*, with the largest surprises on the upside (recessions).

The symmetric Gaussian (orange) can match neither feature: it treats upward and downward shocks as equally likely and has no room for the occasional very large jump.

So our model is bound to misread the data, seeing the rare jump up and the long gentle slide down as the *same* kind of shock.


### A note on the plug-in

A Bayesian purist would object that there is no single residual series here.

Every posterior draw of $(\bar u, \phi)$ implies its own, and we simply chose the medians.

That is fair, and the fully Bayesian version of this check — simulating whole datasets from the posterior and comparing a summary statistic — is what we do in {doc}`unemployment_shocks`.

This plug-in check is a quick preview of that fuller test.

Capturing the asymmetry is the task of that next lecture, where we model the shocks themselves rather than the reversion curve.

## Exercises

The lecture {doc}`ar1_turningpts` forecasts an AR(1) process by simulating future paths, considering both a predictive distribution that **conditions on** fixed parameter values and one that **integrates over** their posterior uncertainty.

The following exercises apply these ideas to our fitted unemployment model.

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

We work with the annual model because its persistence $\phi$ is far less certain
than at the monthly frequency, so parameter uncertainty has more to say.

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
