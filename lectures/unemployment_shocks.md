---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Unemployment Dynamics with Asymmetric Shocks 

```{include} _admonition/gpu.md
```


This lecture is a sequel to {doc}`unemployment_linear`.

There we fit a linear AR(1) model to the US unemployment rate and examined persistence.

One thing we found but didn't address is that the model's residuals were both *heavy-tailed and right-skewed*.

Unemployment jumps up sharply in recessions and drifts down slowly in recoveries. 

A model with symmetric Gaussian shocks cannot reproduce these features.

In this lecture we allow the innovations to be right-skewed and, to some degree, heavy-tailed.

We also use this lecture to develop a very useful Bayesian technique: *model comparison* by leave-one-out cross-validation.

We use this to decide whether one model really predicts better than another.

Our plan is to

1. build a model with a linear mean but asymmetric, occasionally-large shocks,
2. estimate it on the annual data,
3. compare it to the original Gaussian model with cross-validation, and
4. check whether it captures the asymmetry the original model missed.

In addition to what's in Anaconda, this lecture needs the following libraries.

```{code-cell} ipython3
:tags: [hide-output]

!pip install numpyro jax pandas_datareader arviz
```

We'll use the following imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as web

import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood
```

## The data

We use the same series as in {doc}`unemployment_linear` — the US unemployment
rate (`UNRATE` from FRED), excluding the COVID-19 spike — and work at an annual
frequency.

```{code-cell} ipython3
start, end = dt.datetime(1948, 1, 1), dt.datetime(2024, 12, 31)
unrate = web.DataReader("UNRATE", "fred", start, end)["UNRATE"]
pre_covid = unrate[unrate.index < "2020-01-01"]
u_annual = pre_covid.resample("YE").last().to_numpy()
years = pre_covid.resample("YE").last().index.year
```

The shape we want to capture is plain in the annual series.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Annual US unemployment
    name: fig-annual-data
---
fig, ax = plt.subplots()
ax.plot(years, u_annual, lw=2, marker='o', ms=3)
ax.set_xlabel('year')
ax.set_ylabel('unemployment rate (%)')
plt.show()
```

In {numref}`fig-annual-data` each recession is a sharp rise followed by a long, gradual decline — fast up, slow down.

## A model with asymmetric shocks

We keep the linear structure from {doc}`unemployment_linear`:

$$
u_{t+1} = \bar u + \rho\,(u_t - \bar u) + \eta_{t+1},
\qquad 0 \le \rho < 1,
$$ (eq:shocks)

Here, however, the innovation $\eta_{t+1}$ is drawn from a *mixture of two normals*:

$$
\eta_{t+1} \sim
\begin{cases}
N(0, \sigma_s^2) & \text{with probability } 1-p \quad\text{(a quiet year)},\\[4pt]
N(\mu_J, \sigma_J^2) & \text{with probability } p \quad\text{(a recession jump)},
\end{cases}
\qquad \mu_J > 0 .
$$

The model has six parameters:

| symbol | name | role |
| --- | --- | --- |
| $\bar u$ | floor | the low level the series reverts to |
| $\rho$ | persistence | speed of reversion (as in the linear lecture) |
| $p$ | jump probability | how often a large shock arrives |
| $\mu_J$ | jump size | the average upward kick of a recession |
| $\sigma_s,\ \sigma_J$ | shock spreads | the quiet and jumpy volatilities |

The mechanism leads to a sawtooth dynamic for the unemployment rate.

Most years are quiet, with small noise.

Occasionally a positive jump pushes unemployment up by a large amount. 

Then, in most cases, the linear reversion pulls it slowly back down toward $\bar u$.

Thus, a spike arrives in one step, while the recovery takes time --- the asymmetric dynamics we observe in the data.

This is, in spirit, Milton Friedman's "plucking model": a floor near full employment, from which bad shocks pluck the series upward.

The next cell shows the density of the shock and its two components.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Shock distribution as a mixture
    name: fig-shock-density
---
def normal_pdf(x, m, s):
    return np.exp(-(x - m)**2 / (2 * s**2)) / (s * np.sqrt(2 * np.pi))

η = np.linspace(-3, 6, 400)
p_, μJ_, σs_, σJ_ = 0.25, 2.0, 0.4, 1.2
quiet = (1 - p_) * normal_pdf(η, 0.0, σs_)
jump  = p_ * normal_pdf(η, μJ_, σJ_)

fig, ax = plt.subplots()
ax.plot(η, quiet + jump, lw=2, label='shock density')
ax.fill_between(η, quiet, alpha=0.3, label='quiet years')
ax.fill_between(η, jump, alpha=0.3, label='recession jumps')
ax.set_xlabel('innovation $\\eta$')
ax.legend()
plt.show()
```

{numref}`fig-shock-density` shows a tall, narrow peak at zero for
ordinary years, plus a low, broad bump on the positive side for the recession
jumps.



## Bayesian estimation

We place weakly informative priors on the six parameters. 

The mixture is written with NumPyro's `MixtureSameFamily`, which sums over the two components analytically.

```{code-cell} ipython3
def jump_model(u):
    # Set the priors
    ubar = numpyro.sample("ubar",    dist.Normal(4.5, 1.5))  
    ρ    = numpyro.sample("rho",     dist.Uniform(0.0, 1.0))  
    p    = numpyro.sample("p",       dist.Beta(2.0, 8.0))      
    μ_J  = numpyro.sample("mu_J",    dist.HalfNormal(2.0))
    σ_s  = numpyro.sample("sigma_s", dist.HalfNormal(0.5))
    σ_J  = numpyro.sample("sigma_J", dist.HalfNormal(1.5))
    # Build the model
    n = u.shape[0] - 1
    base   = ubar + ρ * (u[:-1] - ubar)
    locs   = jnp.stack([base, base + μ_J], axis=-1)
    scales = jnp.stack([jnp.broadcast_to(σ_s, (n,)),
                        jnp.broadcast_to(σ_J, (n,))], axis=-1)
    probs  = jnp.broadcast_to(jnp.stack([1 - p, p]), (n, 2))
    mix = dist.MixtureSameFamily(dist.Categorical(probs=probs),
                                 dist.Normal(locs, scales))
    numpyro.sample("u_obs", mix, obs=u[1:])
```

We sample with NUTS, four chains, vectorized so the code runs on a CPU or a GPU.

```{code-cell} ipython3
def run_nuts(model, data, seed=0, num_warmup=2000, num_samples=4000, num_chains=4):
    "Sample a NumPyro model with the NUTS sampler."
    mcmc = MCMC(NUTS(model),
                num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="vectorized",
                progress_bar=False)
    mcmc.run(random.PRNGKey(seed), jnp.asarray(data))
    return mcmc
```

We fit the model to the annual data.

```{code-cell} ipython3
:tags: [hide-output]

mcmc_jump = run_nuts(jump_model, u_annual)
```

Let us look at the posterior.

```{code-cell} ipython3
mcmc_jump.print_summary()
```

The `r_hat` values are essentially one, so the chains have converged.

Notice the distinct jump component: the normal year spread $\sigma_s$ is small and the jump spread $\sigma_J$ is large, with a positive mean $\mu_J$.

Let's do a quick check by overlaying the fitted shock density on the model's actual residuals, as we did in {doc}`unemployment_linear`.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Fitted shock density against the residuals
    name: fig-jump-fit
---
m = {k: np.median(np.asarray(mcmc_jump.get_samples()[k]))
     for k in ("ubar", "rho", "p", "mu_J", "sigma_s", "sigma_J")}
resid = u_annual[1:] - (m["ubar"] + m["rho"] * (u_annual[:-1] - m["ubar"]))

η = np.linspace(resid.min() - 0.5, resid.max() + 0.5, 400)
quiet = (1 - m["p"]) * normal_pdf(η, 0.0, m["sigma_s"])
jump  = m["p"] * normal_pdf(η, m["mu_J"], m["sigma_J"])

fig, ax = plt.subplots()
ax.hist(resid, bins=25, density=True, alpha=0.5, label='residuals')
ax.plot(η, quiet + jump, lw=2, label='fitted density')
ax.fill_between(η, jump, alpha=0.3, color='C2', label='jump component')
ax.set_xlabel('innovation $\\eta$')
ax.legend()
plt.show()
```

The fit looks quite good (although we are still struggling to capture some of
the large shocks in the right tail).


## Comparing models with cross-validation

The jump model looks reasonable, but is it actually better than the linear one?

To answer this question, we will use a form of *Bayesian model comparison*.

We build it up in stages: the guiding principle, the leave-one-out estimate, and how the computation is actually done.


### The guiding principle: out-of-sample predictions

It is tempting to compare two models by asking which one fits the observed data better.

But this isn't the right criterion: more complex models can always be tuned to better match the data they are fitting.

The right question is instead: which model better predicts data it has not seen?

In other words, the criterion we care about is out-of-sample predictive accuracy.

We do not have new data for out-of-sample testing, so we manufacture some using *cross-validation*.


### Leave-one-out cross-validation

The idea is to leave out one observation, fit the model to the rest, and then
score the held-out point using a predictive distribution constructed from the
rest of the data set.

The predictive distribution for data point $u_i$ given the rest of the data $u_{-i}$ is 

$$
p(u_i \mid u_{-i}) = \int p(u_i \mid \theta)\, p(\theta \mid u_{-i})\, d\theta .
$$

Here $p(\theta \mid u_{-i})$ is the posterior obtained from the data in $u_{-i}$.

We score each held-out point by the *log* of this density. 

This rewards a model for putting high probability on what actually happened. 

Adding these scores over all data points gives the **expected log predictive density** (elpd), here in its leave-one-out (LOO) form,

$$
\text{elpd}_{\text{loo}} = \sum_{i=1}^{n} \log p(u_i \mid u_{-i}).
$$


### Computational methods

Taken at face value, the definition asks us to refit the model $n$ times, once for each omitted point.

The following result removes all but the first fit: it expresses each leave-one-out density through the *full* posterior, which we have already sampled.

Fix notation: let $\theta$ carry the prior $p(\theta)$, let $u = (u_1, \dots, u_n)$, and write $u_{-i}$ for the sample with $u_i$ deleted.

```{prf:assumption}
:label: assm-cond-ind

Given $\theta$, the observations are conditionally independent:

$$
p(u \mid \theta) = \prod_{j=1}^{n} p(u_j \mid \theta) .
$$
```

For our autoregressive data this assumption is false: given $\theta$, the observation $u_i$ stays dependent on its neighbours through the transition dynamics.

We adopt it nonetheless, since it is what makes the leave-one-out computation tractable, and weigh the consequences in the section on time-series structure below.

Conditional on it, the proposition that follows is exact.

```{prf:proposition}
:label: prop-loo-identity

Under {prf:ref}`assm-cond-ind`,

$$
\frac{1}{p(u_i \mid u_{-i})}
\;=\;
\int \frac{1}{p(u_i \mid \theta)}\, p(\theta \mid u)\, d\theta
\qquad (i = 1, \dots, n),
$$

the integral being taken against the full posterior $p(\theta \mid u)$.
```

```{prf:proof}
By Bayes' rule, $p(\theta \mid u) = p(u \mid \theta)\, p(\theta) / p(u)$ and $p(\theta \mid u_{-i}) = p(u_{-i} \mid \theta)\, p(\theta) / p(u_{-i})$.

By {prf:ref}`assm-cond-ind`, $p(u \mid \theta) = p(u_i \mid \theta)\, p(u_{-i} \mid \theta)$.

Dividing the two posteriors and substituting this factorization,

$$
\frac{p(\theta \mid u)}{p(\theta \mid u_{-i})}
\;=\;
\frac{p(u \mid \theta)}{p(u_{-i} \mid \theta)} \cdot \frac{p(u_{-i})}{p(u)}
\;=\;
p(u_i \mid \theta) \cdot \frac{p(u_{-i})}{p(u)} ,
$$

so, rearranging,

$$
\frac{p(\theta \mid u)}{p(u_i \mid \theta)}
\;=\;
\frac{p(u_{-i})}{p(u)}\, p(\theta \mid u_{-i}) .
$$

Now integrate both sides over $\theta$.

On the right, $\int p(\theta \mid u_{-i})\, d\theta = 1$, leaving the constant factor $p(u_{-i}) / p(u)$, so

$$
\int \frac{1}{p(u_i \mid \theta)}\, p(\theta \mid u)\, d\theta
\;=\;
\frac{p(u_{-i})}{p(u)}
\;=\;
\frac{1}{p(u_i \mid u_{-i})} ,
$$

the last equality by $p(u) = p(u_i \mid u_{-i})\, p(u_{-i})$.
```

The leave-$i$-out posterior has vanished; only the full posterior remains, and of it we have $S$ draws $\theta^1, \dots, \theta^S$.

Replacing the integral in {prf:ref}`prop-loo-identity` by the average over these draws gives

$$
\frac{1}{p(u_i \mid u_{-i})}
\;\approx\;
\frac{1}{S} \sum_{s=1}^{S} \frac{1}{p(u_i \mid \theta^s)},
\qquad\text{so}\qquad
p(u_i \mid u_{-i})
\;\approx\;
\frac{S}{\sum_{s=1}^{S} 1 / p(u_i \mid \theta^s)} ,
$$

the **harmonic mean** of the per-draw likelihoods of $u_i$.

The integrand has finite mean, equal to $1/p(u_i \mid u_{-i})$ itself, so the law of large numbers applies.

The average therefore converges to that mean as $S \to \infty$, and the approximation becomes exact, with probability one, in the limit.

Its one ingredient is the likelihood of each observation under each draw — the **pointwise log-likelihood** — which is why we have been careful to keep it.

One caveat forces the refinement we make later.

The mean is finite, but the variance need not be, since $1/p(u_i \mid \theta)$ is largest in the thin-sampled tails of the posterior, where the likelihood is small.

When that variance is infinite the average still converges, but erratically, as a single poorly-fitting draw can dominate the sum.

The terms $1/p(u_i \mid \theta^s)$ are the **importance weights** that the next section stabilizes.

For numerical stability we compute on the log scale, where the harmonic mean becomes

$$
\log p(u_i \mid u_{-i})
\;\approx\;
\log S - \log \sum_{s=1}^{S} e^{-\ell_i^s},
\qquad \ell_i^s = \log p(u_i \mid \theta^s),
$$

which is the `jnp.log(S) - logsumexp(-ll)` we use below.

To compare against the linear model, we first fit it on the same data.

```{code-cell} ipython3
def linear_model(u):
    ubar = numpyro.sample("ubar",  dist.Normal(5.5, 2.0))
    ρ    = numpyro.sample("rho",   dist.Uniform(0.0, 1.0))
    σ    = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("u_obs", dist.Normal(ubar + ρ * (u[:-1] - ubar), σ), obs=u[1:])
```

We sample it on the same annual data.

```{code-cell} ipython3
:tags: [hide-output]

mcmc_lin = run_nuts(linear_model, u_annual)
```

Now we can do the whole leave-one-out calculation by hand in a few lines.

```{code-cell} ipython3
from jax.scipy.special import logsumexp

def pointwise_loo(mcmc, model):
    "Leave-one-out log predictive density for each observation."
    ll = log_likelihood(model, mcmc.get_samples(),
                        u=jnp.asarray(u_annual))["u_obs"]  # (draws, obs)
    S = ll.shape[0]
    # log of the harmonic mean of the per-draw likelihoods, per observation
    return jnp.log(S) - logsumexp(-ll, axis=0)

elpd_jump = pointwise_loo(mcmc_jump, jump_model)
elpd_lin  = pointwise_loo(mcmc_lin,  linear_model)

print(f"jump   elpd_loo = {elpd_jump.sum():.1f}")
print(f"linear elpd_loo = {elpd_lin.sum():.1f}")
```

The jump model scores higher (closer to zero), so on out-of-sample prediction it wins.

To judge whether the gap is real, we need its uncertainty.

Because the total elpd is a *sum* over observations, the difference between the two models is itself a sum of per-observation differences, and its standard error follows from their spread.

```{code-cell} ipython3
diff = elpd_jump - elpd_lin
n = diff.size
print(f"elpd difference = {diff.sum():.1f}")
print(f"standard error  = {jnp.sqrt(n) * diff.std():.1f}")
```

The jump model is ahead by about twelve points, against a standard error near six — a bit over two standard errors, so a real improvement and not a fluke.

### Letting ArviZ do it

In practice we let a library handle the bookkeeping, and add two refinements.

The raw importance weights $1/p(u_i\mid\theta^s)$ can occasionally be dominated by a single extreme draw; **ArviZ** stabilizes them with Pareto-smoothed importance sampling, and reports a diagnostic — the Pareto shape $\hat k$ — that flags any observation where the estimate is unreliable (a value above $0.7$ is the usual warning line).

We hand it the same pointwise log-likelihoods, packaged in its data format.

```{code-cell} ipython3
import arviz as az
import xarray as xr

def to_arviz(mcmc, model, u):
    "Package a NumPyro fit for ArviZ, with pointwise log-likelihoods."
    idata = az.from_numpyro(mcmc)
    ll = np.asarray(log_likelihood(model, mcmc.get_samples(),
                                   u=jnp.asarray(u))["u_obs"])
    ll = ll.reshape(mcmc.num_chains, -1, ll.shape[-1])
    idata["log_likelihood"] = xr.Dataset(
        {"u_obs": (("chain", "draw", "obs"), ll)})
    return idata

az.compare({
    "linear": to_arviz(mcmc_lin,  linear_model, u_annual),
    "jump":   to_arviz(mcmc_jump, jump_model,   u_annual),
})
```

Here the smoothing changes nothing — every $\hat k$ stays below $0.7$ — so `az.compare` confirms the hand calculation and ranks the jump model first (its table rounds the scores for display).

### Reading the comparison

Two of the columns repay a closer look.

The column `p` is the **effective number of parameters**: not the count we wrote down, but how much freedom the data actually grant the model, estimated from the gap between its in-sample and out-of-sample fit.

It is about three and a half for the linear model and six and a half for the jump model — and, tellingly, it need not equal the nominal parameter count, since a parameter the data cannot pin down adds almost nothing to it.

Readers from classical statistics will recognize the whole exercise: it is the goal behind the **AIC**, which estimates out-of-sample accuracy as in-sample fit minus a parameter count.

LOO reaches that goal without the asymptotic shortcut — it uses the entire posterior and learns the effective complexity from the data.

(The **BIC**, by contrast, aims at a different target, the marginal likelihood, and hence the probability that each model is *true*; that is the province of Bayes factors, not of predictive accuracy.)

Finally, a word of caution: these standard errors rest on only about seventy observations, so they are rough.

There is also a deeper issue, special to time series, which we take up now.

### Accommodating the time series structure

Leave-one-out drops one transition at a time, but it does not respect the order of time.

When it scores the step from $u_t$ to $u_{t+1}$, the model doing the scoring was fit on data lying on *both* sides of that step — including the neighboring values $u_t$ and $u_{t+1}$ themselves.

So the held-out point was never truly unseen, and a time series, unlike an exchangeable sample, has an arrow of time that this ignores.

The formally correct measure is **leave-future-out** cross-validation.

It only ever predicts forward: fit the model on $u_1, \dots, u_t$, score its one-step-ahead forecast of $u_{t+1}$, then expand the window by one step and repeat.

Now the model is genuinely blind to the future it is asked to predict, exactly as in real forecasting.

The price is computational.

Leave-one-out reused a single fit through its importance-sampling shortcut, but leave-future-out has no such trick — each step conditions on a different stretch of history, so the model must be refit from scratch, dozens of times rather than once.

For our short annual series that is minutes rather than seconds; for a long series it can become prohibitive.

We ran it anyway, and the conclusion holds: under leave-future-out the jump model still beats the linear one, by an even clearer margin than leave-one-out reported.

Genuine forecasting favors the asymmetric shocks more strongly, not less — so the verdict survives the stricter test.

## Does it capture the asymmetry?

LOO says the jump model predicts better, but we built it for a specific reason: to reproduce the asymmetry.

A **posterior predictive check** tests exactly that.

We pick a statistic that summarizes the asymmetry — the **skewness of the annual changes** $\Delta u$ — and ask whether paths simulated from the fitted model produce values like the one we see in the data.

A symmetric model is pinned at a skewness of zero and cannot pass; the jump model should.

```{code-cell} ipython3
def simulate(u0, T, ubar, ρ, p, μ_J, σ_s, σ_J, rng):
    "Simulate a path of the jump model."
    u = np.empty(T)
    u[0] = u0
    for t in range(1, T):
        η = rng.normal(μ_J, σ_J) if rng.random() < p else rng.normal(0.0, σ_s)
        u[t] = ubar + ρ * (u[t-1] - ubar) + η
    return u

def skewness(x):
    x = x - x.mean()
    return (x**3).mean() / x.std()**3

post = mcmc_jump.get_samples()
keys = ("ubar", "rho", "p", "mu_J", "sigma_s", "sigma_J")
draws = {k: np.asarray(post[k]) for k in keys}

rng = np.random.default_rng(1)
T, N = len(u_annual), 2000
idx = rng.integers(0, len(draws["rho"]), N)
sims = np.array([simulate(u_annual[0], T, *(draws[k][i] for k in keys), rng)
                 for i in idx])

obs_skew = skewness(np.diff(u_annual))
rep_skew = np.array([skewness(np.diff(s)) for s in sims])
print(f"observed skewness of Δu = {obs_skew:.2f}")
print(f"posterior predictive P(skew > observed) = {(rep_skew > obs_skew).mean():.2f}")
```

The observed annual changes are right-skewed, and the simulated paths reproduce that skew, with the observed value sitting comfortably inside the predictive distribution.

The figure makes the check, and the sawtooth, visible.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Posterior predictive check for the asymmetry
    name: fig-ppc
---
fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 4.2))

lo, mid, hi = np.percentile(sims, [5, 50, 95], axis=0)
a0.fill_between(years, lo, hi, alpha=0.3, label='90% band')
a0.plot(years, u_annual, 'k', lw=2, label='observed')
a0.set_title('predictive band')
a0.set_xlabel('year')
a0.legend()

for s in sims[:6]:
    a1.plot(years, s, lw=1, alpha=0.7)
a1.plot(years, u_annual, 'k', lw=2.5, label='observed')
a1.set_title('simulated paths')
a1.set_xlabel('year')
a1.legend()

a2.hist(rep_skew, bins=50, density=True, alpha=0.6)
a2.axvline(obs_skew, color='k', lw=2, label='observed')
a2.set_title('skewness of $\\Delta u$')
a2.set_xlabel('skewness')
a2.legend()
plt.show()
```

The left panel shows the observed series staying inside the predictive band; the middle panel shows simulated paths with the same spiky, fast-up-slow-down character as the data; and the right panel shows the observed skewness falling in the bulk of the predictive distribution.

The symmetric models of {doc}`unemployment_linear` would put that vertical line far out in the tail.

## Conclusion


In {doc}`unemployment_linear` we applied a linear AR(1) model with Gaussian shocks to unemployment and found a high levels of persistence in monthly data.

We also argued that the model is overly simplistic.

Here we found that the feature the linear model most conspicuously misses — the asymmetry of recessions — can be addressed by considering the distribution of the shocks.

A linear reversion with large, one-sided innovations reproduces the spikes, the slow recoveries, and the boundedness, and cross-validation prefers it clearly.

Along the way we learned about leave-one-out cross-validation, which helps us determine asks which model predicts better.

A richer model would let the economy switch between persistent expansion and recession regimes.

This is the Markov-switching approach of Hamilton, which we leave to further reading.

## Exercises

```{exercise}
:label: unemp_shocks_ex1

The jump component here is symmetric within itself, $N(\mu_J, \sigma_J^2)$.

Replace the two-component mixture with a single **skew-normal** (or Student-$t$ with a skew) innovation, refit, and compare to the mixture model with `az.compare`.

Does the simpler skewed shock do as well as the mixture?
```
