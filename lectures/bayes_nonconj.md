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

# Non-Conjugate Priors

```{include} _admonition/gpu.md
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install numpyro jax arviz
```

## Overview

This lecture is a sequel to {doc}`prob_meaning`.

In that lecture we adopted a **beta** prior for the unknown probability $\theta$ of a coin landing heads, together with a **binomial** likelihood.

That prior and likelihood form a **conjugate pair**: applying Bayes' law returns a posterior of the *same* family as the prior — again a beta distribution.

Conjugacy is convenient because it delivers a posterior in closed form.

But a person's prior beliefs are their own business, and in general they will not happen to be conjugate to the likelihood.

When the prior and likelihood are **not** conjugate, the posterior usually has no closed form, and we must approximate it numerically.

This lecture introduces two widely used ways to do that, both implemented in the probabilistic programming library [NumPyro](https://num.pyro.ai/en/stable/getting_started.html):

* **Markov chain Monte Carlo (MCMC)** — construct a Markov chain whose stationary distribution is the posterior, then sample from it. We use the **No-U-Turn Sampler (NUTS)**, a state-of-the-art form of Hamiltonian Monte Carlo.

* **Variational inference (VI)** — replace sampling with optimization: search within a tractable family of distributions for the member closest to the posterior.

Our plan is:

1. Confirm that MCMC reproduces the *conjugate* beta posterior that we can compute analytically — this validates the machinery on a problem whose answer we already know.
2. Replace the beta prior with several **non-conjugate** priors and approximate each posterior with MCMC.
3. Introduce variational inference and compare it with MCMC.

Let us start with some imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

import arviz as az
```

To draw posterior samples from several Markov chains in parallel, we tell NumPyro how many CPU devices to use.

```{code-cell} ipython3
numpyro.set_host_device_count(4)
```

## The coin-flipping model

As in {doc}`prob_meaning`, a coin lands heads ($Y=1$) with probability $\theta$ and tails ($Y=0$) with probability $1-\theta$.

If we flip the coin $n$ times, the number of heads $k$ has the **binomial** distribution

$$
p(k \mid \theta) = \binom{n}{k}\, \theta^k (1-\theta)^{n-k} .
$$

We treat $\theta$ as a random variable with a prior density $p(\theta)$, and we want the posterior

$$
p(\theta \mid k) \propto p(k \mid \theta)\, p(\theta) .
$$

### Generating data

We simulate a sequence of coin flips from a coin whose true (but unknown to the analyst) probability of heads is $\theta = 0.4$.

```{code-cell} ipython3
def simulate_coin_flips(θ=0.4, n=20, seed=1234):
    "Flip a coin n times; return an array of 0s (tails) and 1s (heads)."
    rng = np.random.default_rng(seed)
    return (rng.random(n) < θ).astype(int)

data = simulate_coin_flips()
k, n = int(data.sum()), len(data)
k, n
```

We deliberately use a **small** sample ($n = 20$).

The reason is that the prior matters most when data are scarce.

With a large sample the likelihood dominates and almost any reasonable prior leads to the same posterior — exactly the concentration we saw in {doc}`prob_meaning`.

A modest $n$ keeps the influence of the prior visible, which is what we want to study here.

### One model, many priors

In NumPyro a model is an ordinary Python function that uses `numpyro.sample` to declare random variables.

We write a *single* model that takes the prior distribution as an argument.

This lets us reuse it unchanged for every prior we consider — conjugate or not.

```{code-cell} ipython3
def binomial_model(prior, k, n):
    "Binomial likelihood with a caller-supplied prior on θ."
    θ = numpyro.sample("θ", prior)
    numpyro.sample("k", dist.Binomial(n, θ), obs=k)
```

The first `sample` statement draws $\theta$ from the prior; the second ties the observed count `k` to the binomial likelihood through `obs=k`.

We also write a small helper that runs NUTS and returns the fitted sampler.

```{code-cell} ipython3
def run_nuts(prior, k, n, seed=0, num_warmup=1000, num_samples=4000, num_chains=4):
    "Sample the posterior of θ with the NUTS sampler."
    mcmc = MCMC(
        NUTS(binomial_model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(seed), prior, k, n)
    return mcmc
```

## MCMC reproduces the conjugate posterior

Before trusting MCMC on hard problems, let us check it on an easy one.

With a $\text{Beta}(\alpha_0, \beta_0)$ prior the posterior is known analytically (see {doc}`prob_meaning`):

$$
\theta \mid k \sim \text{Beta}(\alpha_0 + k,\ \beta_0 + n - k) .
$$

We take $\alpha_0 = \beta_0 = 2$ and sample the posterior with NUTS.

```{code-cell} ipython3
α0, β0 = 2.0, 2.0
mcmc = run_nuts(dist.Beta(α0, β0), k, n)
```

Before looking at the posterior we check that the sampler converged.

ArviZ reads NumPyro's output directly and reports standard diagnostics.

```{code-cell} ipython3
idata = az.from_numpyro(mcmc)
az.summary(idata, var_names=["θ"])
```

The potential scale reduction factor `r_hat` is essentially $1.0$ and the effective sample sizes are large, both signs that the chains have mixed well.

The trace plot tells the same story: the four chains overlap and look like stationary noise.

```{code-cell} ipython3
az.plot_trace(idata, var_names=["θ"])
plt.tight_layout()
plt.show()
```

Now we compare the MCMC posterior with the analytical beta posterior.

```{code-cell} ipython3
θ_grid = np.linspace(0.001, 0.999, 500)
samples = np.asarray(mcmc.get_samples()["θ"])

fig, ax = plt.subplots()
ax.hist(samples, bins=50, density=True, alpha=0.4,
        label="MCMC posterior")
ax.plot(θ_grid, st.beta(α0 + k, β0 + n - k).pdf(θ_grid),
        'k-', lw=2, label="analytical posterior")
ax.plot(θ_grid, st.beta(α0, β0).pdf(θ_grid),
        'C1--', lw=2, label="prior")
ax.set_xlabel(r"$\theta$")
ax.legend()
plt.show()
```

The histogram of MCMC draws sits right on top of the analytical posterior density.

The sampler works, so we can rely on it for priors that have no closed-form posterior.

## Non-conjugate priors

We now keep the binomial likelihood and the same data, but replace the beta prior with priors that are **not** conjugate to it.

For each prior the recipe is identical:

1. describe the prior and build it as a NumPyro distribution,
2. pass it to `binomial_model` and run NUTS,
3. plot the prior against the resulting posterior.

The following helper draws a prior density and the posterior samples on the same axes.

```{code-cell} ipython3
def plot_prior_posterior(prior, samples, title=""):
    "Overlay a prior density and posterior MCMC draws for θ on [0, 1]."
    grid = jnp.linspace(0.001, 0.999, 500)
    prior_pdf = np.exp(np.asarray(prior.log_prob(grid)))

    fig, ax = plt.subplots()
    ax.hist(np.asarray(samples), bins=50, density=True, alpha=0.4,
            label="posterior (MCMC)")
    ax.plot(np.asarray(grid), prior_pdf, 'C1--', lw=2, label="prior")
    ax.set_xlabel(r"$\theta$")
    ax.set_xlim(0, 1)
    ax.legend()
    if title:
        ax.set_title(title)
    plt.show()
```

### A uniform prior

The simplest non-conjugate prior is **uniform**: the analyst regards every value of $\theta$ in some interval as equally likely.

A uniform prior on all of $[0, 1]$ expresses indifference.

Because its density is constant, the posterior is then proportional to the likelihood alone.

```{code-cell} ipython3
mcmc_flat = run_nuts(dist.Uniform(0.0, 1.0), k, n)
plot_prior_posterior(dist.Uniform(0.0, 1.0),
                     mcmc_flat.get_samples()["θ"],
                     title="flat uniform prior")
```

The posterior is centered near the sample frequency $k/n$, just as the likelihood is.

Now suppose instead that the analyst is convinced the coin favors heads, and places a uniform prior on $[0.5, 0.95]$.

This prior assigns *zero* density to the region around the true value $\theta = 0.4$.

```{code-cell} ipython3
mcmc_restr = run_nuts(dist.Uniform(0.5, 0.95), k, n)
plot_prior_posterior(dist.Uniform(0.5, 0.95),
                     mcmc_restr.get_samples()["θ"],
                     title="restrictive uniform prior")
```

The posterior cannot put mass where the prior is zero, so it piles up against the lower boundary $0.5$ — as close to the data as the prior permits.

This is a vivid warning: a prior that rules out the truth can never be overturned by data, no matter how much we collect.

### A truncated log-normal prior

A uniform prior is flat. A more realistic prior is smooth and asymmetric.

A convenient choice on $[0, 1]$ is a **truncated log-normal**: take $Z \sim N(\mu, \sigma)$ truncated to $Z \le 0$, and set $\theta = e^{Z}$, which then lies in $(0, 1]$.

NumPyro builds this by feeding a `TruncatedNormal` through an `ExpTransform`.

```{code-cell} ipython3
def truncated_lognormal(μ, σ):
    "Log-normal distribution truncated to the unit interval (0, 1]."
    base = dist.TruncatedNormal(loc=μ, scale=σ, low=-jnp.inf, high=0.0)
    return dist.TransformedDistribution(base, dist.transforms.ExpTransform())

prior_ln = truncated_lognormal(0.0, 1.0)
mcmc_ln = run_nuts(prior_ln, k, n)
plot_prior_posterior(prior_ln, mcmc_ln.get_samples()["θ"],
                     title="truncated log-normal prior")
```

The prior favors smaller values of $\theta$, but with $\sigma = 1$ it is diffuse, so the likelihood pulls the posterior toward the sample frequency.

We keep `mcmc_ln` — we will compare it with variational inference below.

### A truncated Laplace prior

Our final prior has a sharp, non-smooth peak.

A **Laplace** density $\propto e^{-|\theta - \mu| / b}$ has a kink at its center $\mu$, expressing a strong belief that $\theta$ sits near $\mu$ while still allowing for surprises in the tails.

We truncate it to $[0, 1]$ and center it at $0.5$.

```{code-cell} ipython3
def truncated_laplace(μ, b):
    "Laplace distribution truncated to the unit interval [0, 1]."
    return dist.TruncatedDistribution(dist.Laplace(μ, b), low=0.0, high=1.0)

prior_lp = truncated_laplace(0.5, 0.1)
mcmc_lp = run_nuts(prior_lp, k, n)
plot_prior_posterior(prior_lp, mcmc_lp.get_samples()["θ"],
                     title="truncated Laplace prior")
```

The spiked prior tugs the posterior toward $0.5$, away from the sample frequency near $0.4$.

The pull is gentle here because the prior, though peaked, is not very tight; with a smaller $b$ it would dominate the modest sample.

NUTS handles the kink in the prior without any special tuning — a practical advantage of gradient-based samplers paired with automatic differentiation.

## Variational inference

MCMC approximates the posterior by *sampling* from it.

**Variational inference (VI)** takes a different route: it turns posterior approximation into an *optimization* problem.

We restrict attention to a tractable family of densities $q_\phi(\theta)$ — the **guide** — indexed by parameters $\phi$, and we search for the member of that family closest to the posterior.

### The evidence lower bound

Let the prior be $p(\theta)$ and the likelihood be $p(Y \mid \theta)$.

By Bayes' rule,

$$
p(\theta \mid Y) = \frac{p(Y, \theta)}{p(Y)} = \frac{p(Y \mid \theta)\, p(\theta)}{p(Y)},
$$

where

$$
p(Y) = \int p(Y \mid \theta)\, p(\theta)\, d\theta .
$$ (eq:intchallenge)

The integral in {eq}`eq:intchallenge` is the troublesome one: in the non-conjugate case it has no closed form.

We measure the discrepancy between the guide $q_\phi(\theta)$ and the posterior with the **Kullback–Leibler (KL) divergence**

$$
D_{KL}\big(q_\phi(\theta)\ \|\ p(\theta \mid Y)\big)
= -\int q_\phi(\theta)\, \log \frac{p(\theta \mid Y)}{q_\phi(\theta)}\, d\theta ,
$$

and we choose $\phi$ to minimize it.

The KL divergence still involves the intractable posterior, but we can rearrange it. Using $p(\theta \mid Y) = p(\theta, Y) / p(Y)$,

$$
\begin{aligned}
D_{KL}\big(q_\phi \,\|\, p(\theta \mid Y)\big)
 & = -\int q_\phi(\theta)\, \log \frac{p(\theta, Y) / p(Y)}{q_\phi(\theta)}\, d\theta \\
 & = -\int q_\phi(\theta) \left[\log \frac{p(\theta, Y)}{q_\phi(\theta)} - \log p(Y)\right] d\theta \\
 & = -\int q_\phi(\theta)\, \log \frac{p(\theta, Y)}{q_\phi(\theta)}\, d\theta + \log p(Y) ,
\end{aligned}
$$

where the last line uses $\int q_\phi(\theta)\, d\theta = 1$. Rearranging,

$$
\log p(Y) = D_{KL}\big(q_\phi \,\|\, p(\theta \mid Y)\big)
            + \underbrace{\int q_\phi(\theta)\, \log \frac{p(\theta, Y)}{q_\phi(\theta)}\, d\theta}_{\text{ELBO}} .
$$

The marginal likelihood $\log p(Y)$ on the left does not depend on $\phi$.

Hence **minimizing** the KL divergence is equivalent to **maximizing** the second term, the **evidence lower bound (ELBO)**:

$$
\text{ELBO}(\phi) \equiv \int q_\phi(\theta)\, \log \frac{p(\theta, Y)}{q_\phi(\theta)}\, d\theta
= \mathbb{E}_{q_\phi(\theta)}\big[\log p(\theta, Y) - \log q_\phi(\theta)\big] .
$$ (eq:ELBO)

Because $D_{KL} \ge 0$, the ELBO is a lower bound on $\log p(Y)$ — hence its name.

Crucially, {eq}`eq:ELBO` involves only the *joint* density $p(\theta, Y) = p(Y \mid \theta)\, p(\theta)$, which we can evaluate, not the intractable normalizing constant $p(Y)$.

The expectation can be estimated by sampling from $q_\phi$, and $\phi$ improved by gradient ascent — this is **stochastic variational inference (SVI)**.

### Implementing SVI in NumPyro

We need a guide $q_\phi$.

The simplest choice is an **autoguide**: NumPyro inspects the model and automatically constructs a guide for us.

`AutoNormal` places an independent normal distribution on each latent variable, transformed to respect its support — here, to keep $\theta$ inside $(0, 1)$.

We apply SVI to the truncated log-normal model from above and maximize the ELBO with the Adam optimizer.

```{code-cell} ipython3
guide = AutoNormal(binomial_model)
optimizer = Adam(step_size=0.01)
svi = SVI(binomial_model, guide, optimizer, loss=Trace_ELBO())

svi_result = svi.run(random.PRNGKey(0), 5000, prior_ln, k, n, progress_bar=False)
```

SVI maximizes the ELBO; equivalently, it minimizes its negative, which is the reported loss.

A loss curve that flattens out indicates convergence.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(svi_result.losses)
ax.set_xlabel("step")
ax.set_ylabel("negative ELBO")
ax.set_title("SVI convergence")
plt.show()
```

### Comparing VI with MCMC

To assess the approximation, we draw samples from the fitted guide and compare them with the NUTS posterior for the same (log-normal-prior) model.

```{code-cell} ipython3
vi_samples = guide.sample_posterior(
    random.PRNGKey(1), svi_result.params, sample_shape=(4000,)
)["θ"]
nuts_samples = mcmc_ln.get_samples()["θ"]

fig, ax = plt.subplots()
ax.hist(np.asarray(nuts_samples), bins=50, density=True, alpha=0.4,
        label="MCMC (NUTS)")
ax.hist(np.asarray(vi_samples), bins=50, density=True, alpha=0.4,
        label="VI (AutoNormal)")
ax.set_xlabel(r"$\theta$")
ax.legend()
plt.show()
```

The two approximations broadly agree on the location and spread of the posterior.

They need not agree perfectly.

MCMC samples the true posterior (up to Monte Carlo error), whereas VI reports the best fit *within its guide family*.

A mean-field normal guide is symmetric on the transformed scale and can miss skewness or heavy tails in the true posterior.

The trade-off is one of cost against fidelity: VI replaces sampling with optimization and is often much faster in high dimensions, but it delivers an approximation whose quality is capped by the flexibility of the guide.

## Where to next

This lecture showed how to compute posteriors when prior and likelihood are not conjugate, using NUTS and stochastic variational inference in NumPyro.

The same tools carry over to richer models.

The lectures {doc}`ar1_bayes` and {doc}`ar1_turningpts` apply NumPyro to Bayesian estimation and forecasting of autoregressive time series, where the parameter is a vector and conjugate analysis is unavailable.
