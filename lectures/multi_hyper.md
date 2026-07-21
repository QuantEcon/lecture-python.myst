---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

(multi_hyper_v7)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Multivariate Hypergeometric Distribution

```{contents} Contents
:depth: 2
```

In addition to what's included in base Anaconda, we need to install JAX

```{code-cell} ipython3
:tags: [hide-output]

!pip install jax
```

## Overview

This lecture describes how an  administrator deployed a *multivariate hypergeometric distribution* in order to access the fairness of a procedure for awarding research grants.

In the lecture we'll learn about

* properties of the multivariate hypergeometric distribution
* first and second moments of  a multivariate hypergeometric distribution
* using a Monte Carlo simulation of a multivariate normal distribution to evaluate the quality of a normal approximation
* the administrator's problem and why the multivariate hypergeometric distribution is the right tool

## The administrator's problem

An administrator in charge of allocating research grants is in the following situation.

To help us forget details that are none of our business here and to protect the anonymity of the administrator and the subjects, we call
research proposals *balls* and continents of residence of authors of a proposal a *color*.

There are $K_i$ balls (proposals) of color $i$.

There are $c$ distinct colors (continents of residence).

Thus, $i = 1, 2, \ldots, c$

So there is a total of $N = \sum_{i=1}^c K_i$ balls.

All $N$ of these balls are placed in an urn.

Then $n$ balls are drawn randomly.

The selection procedure is supposed to be *color blind*  meaning that *ball quality*, a random variable that is supposed to be  independent of *ball color*, governs whether a ball is drawn.

Thus, the selection procedure is supposed randomly to draw $n$  balls from the urn.

The $n$ balls drawn represent  successful proposals and are  awarded research funds.

The remaining $N-n$ balls receive no research funds.

### Details of the awards procedure under study

Let $k_i$ be the number of balls of color $i$ that are drawn.

Things have to add up so $\sum_{i=1}^c k_i = n$.

Under the hypothesis that the selection process judges proposals on their quality and that quality is independent of continent of the author's continent of residence, the administrator views the outcome of the selection procedure as a random vector

$$
X = \begin{bmatrix} k_1 \cr k_2 \cr \vdots \cr k_c \end{bmatrix}.
$$

To evaluate whether the selection procedure is *color blind* the administrator wants to  study whether the particular realization of $X$ drawn can plausibly
be said to be a random draw from the probability distribution that is implied by the *color blind* hypothesis.

The appropriate probability distribution is the one described [here](https://en.wikipedia.org/wiki/Hypergeometric_distribution).

Let's now instantiate the administrator's problem, while continuing to use the colored balls metaphor.

The administrator has an urn with $N = 238$ balls.

157 balls are blue, 11 balls are green, 46 balls are yellow, and 24 balls are black.

So $(K_1, K_2, K_3, K_4) = (157 , 11 , 46 , 24)$ and $c = 4$.

15 balls are drawn without replacement.

So $n = 15$.

The administrator wants to know the probability distribution of outcomes

$$
X = \begin{bmatrix} k_1 \cr k_2 \cr \vdots \cr k_4 \end{bmatrix}.
$$

In particular, he wants to know whether a particular
outcome - in the form of a $4 \times 1$ vector of integers recording the
numbers of blue, green, yellow, and black balls, respectively, - contains
evidence against the hypothesis that the selection process is *fair*, which
here means  *color blind* and truly are random draws without replacement from
the population of $N$ balls.

The right tool for the administrator's job is the *multivariate hypergeometric distribution*.

### Multivariate hypergeometric distribution

Let's start with some imports.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from jax.scipy.special import gammaln
from scipy.stats import normaltest
from typing import NamedTuple

jax.config.update("jax_platform_name", "cpu")
```

To recapitulate, we assume there are in total $c$ types of objects in an urn.

If there are $K_{i}$ type $i$ object in the urn and we take
$n$ draws at random without replacement, then the numbers of type
$i$ objects in the sample $(k_{1},k_{2},\dots,k_{c})$
has the multivariate hypergeometric distribution.

Note again  that $N=\sum_{i=1}^{c} K_{i}$ is
the total number of objects in the urn and $n=\sum_{i=1}^{c}k_{i}$.

We use the following notation for *binomial coefficients*: ${m \choose q} = \frac{m!}{(m-q)!}$.

The multivariate hypergeometric distribution has the following properties:

*Probability mass function*:

$$
\Pr \{X_{i}=k_{i} \  \forall i\} =
 \frac {\prod _{i=1}^{c}{\binom {K_{i}}{k_{i}}}}{\binom {N}{n}}
$$

*Mean*:

$$
{\displaystyle \operatorname {E} (X_{i})=n{\frac {K_{i}}{N}}}
$$

*Variances and covariances*:

$$
{\displaystyle \operatorname {Var} (X_{i})=n{\frac {N-n}{N-1}}\;{\frac {K_{i}}{N}}\left(1-{\frac {K_{i}}{N}}\right)}
$$

$$
{\displaystyle \operatorname {Cov} (X_{i},X_{j})=-n{\frac {N-n}{N-1}}\;{\frac {K_{i}}{N}}{\frac {K_{j}}{N}}}
$$

We follow the same template used in other JAX lectures by building a small helper structure and standalone functions.

```{code-cell} ipython3
class UrnModel(NamedTuple):
    K: jnp.ndarray
    N: int
    c: int


def create_urn(K_arr):
    """Return an UrnModel containing totals for each color."""
    K_arr = jnp.asarray(K_arr, dtype=jnp.int32)
    N = int(jnp.sum(K_arr))
    c = int(K_arr.size)
    return UrnModel(K=K_arr, N=N, c=c)


def log_comb(n, k):
    """Compute log binomial coefficients using gammaln."""
    n = jnp.asarray(n)
    k = jnp.asarray(k)
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def pmf(urn, k_arr):
    """Evaluate the multivariate hypergeometric PMF."""
    K_arr, N = urn.K, urn.N
    k_arr = jnp.atleast_2d(jnp.asarray(k_arr, dtype=jnp.int32))
    n = jnp.sum(k_arr, axis=1)
    num_log = jnp.sum(log_comb(K_arr[None, :], k_arr), axis=1)
    denom_log = log_comb(N, n)
    return jnp.exp(num_log - denom_log)


def moments(urn, n):
    """Return the mean vector and covariance matrix."""
    K_arr, N = urn.K, urn.N
    p = K_arr / N
    μ = n * p
    factor = n * (N - n) / (N - 1)
    Σ = factor * (jnp.diag(p) - jnp.outer(p, p))
    return μ, Σ


def simulate(urn, n, size=1, seed=0):
    """Simulate multivariate hypergeometric draws."""
    K_arr, c = urn.K, urn.c
    n = int(n)
    size = int(size)
    key = jr.PRNGKey(seed)

    if size == 0:
        return jnp.zeros((0, c), dtype=jnp.int32)

    def draw_once(key):
        """Generate one draw via a lax.scan loop."""
        counts0 = jnp.zeros(c, dtype=jnp.int32)

        def body_fn(carry, _):
            counts, remaining, key = carry
            key, subkey = jr.split(key)
            probs = remaining / jnp.sum(remaining)
            u = jr.uniform(subkey)
            cut = jnp.cumsum(probs)
            idx = jnp.searchsorted(cut, u, side='right')
            counts = counts.at[idx].add(1)
            remaining = remaining.at[idx].add(-1)
            return (counts, remaining, key), None

        (counts, _, _), _ = jax.lax.scan(
            body_fn, (counts0, K_arr, key), None, length=n)
        return counts

    draw_once = jax.jit(draw_once)
    keys = jr.split(key, size)
    return jax.vmap(draw_once)(keys)
```

## Usage

### First example

Apply this to an example from
[wiki](https://en.wikipedia.org/wiki/Hypergeometric_distribution#Multivariate_hypergeometric_distribution):

Suppose there are 5 black, 10 white, and 15 red marbles in an urn. If
six marbles are chosen without replacement, the probability that exactly
two of each color are chosen is

$$
P(2{\text{ black}},2{\text{ white}},2{\text{ red}})={{{5 \choose 2}{10 \choose 2}{15 \choose 2}} \over {30 \choose 6}}=0.079575596816976
$$

```{code-cell} ipython3
K_arr = [5, 10, 15]
urn = create_urn(K_arr)
```

Now use `pmf` to compute the probability of the outcome $X = \begin{bmatrix} 2 & 2 & 2 \end{bmatrix}$.

```{code-cell} ipython3
k_arr = [2, 2, 2]
pmf(urn, k_arr)
```

We can use the code to compute probabilities of a list of possible outcomes by
constructing  a 2-dimensional
array `k_arr` and `pmf` will return an array of probabilities for
observing each case.

```{code-cell} ipython3
k_arr = [[2, 2, 2], [1, 3, 2]]
pmf(urn, k_arr)
```

Now let's compute the mean vector and variance-covariance matrix.

```{code-cell} ipython3
n = 6
μ, Σ = moments(urn, n)
```

```{code-cell} ipython3
μ
```

```{code-cell} ipython3
Σ
```

### Back to the administrator's problem

Now let's turn to the grant administrator's problem.

Here the array of
numbers of $i$ objects in the urn is
$\left(157, 11, 46, 24\right)$.

```{code-cell} ipython3
K_arr = [157, 11, 46, 24]
urn = create_urn(K_arr)
```

Let's compute the probability of the outcome $\left(10, 1, 4, 0 \right)$.

```{code-cell} ipython3
k_arr = [10, 1, 4, 0]
pmf(urn, k_arr)
```

We can compute probabilities of three possible outcomes by constructing a 3-dimensional array `k_arr` and applying `pmf`.

```{code-cell} ipython3
k_arr = [[5, 5, 4 ,1], [10, 1, 2, 2], [13, 0, 2, 0]]
pmf(urn, k_arr)
```

Now let's compute the mean and variance-covariance matrix of $X$ when $n=6$.

```{code-cell} ipython3
n = 6
μ, Σ = moments(urn, n)
```

```{code-cell} ipython3
μ
```

```{code-cell} ipython3
Σ
```

We can simulate a large sample and verify that sample means and covariances closely approximate the population means and covariances.

```{code-cell} ipython3
size = 200_000
sample = simulate(urn, n, size=size, seed=123)
```

```{code-cell} ipython3
jnp.mean(sample, axis=0)
```

```{code-cell} ipython3
jnp.cov(sample.T)
```

Evidently, the sample means and covariances approximate their population counterparts well.

### Quality of normal approximation

To judge the quality of a multivariate normal approximation to the multivariate hypergeometric distribution, we draw a large sample from a multivariate normal distribution with the mean vector  and covariance matrix for the corresponding multivariate hypergeometric distribution and compare the simulated distribution with the population multivariate hypergeometric distribution.

```{code-cell} ipython3
key_normal = jr.PRNGKey(0)
sample_normal = jr.multivariate_normal(
        key_normal, μ, Σ, shape=(size,))
```

```{code-cell} ipython3
def bivariate_normal(x, y, μ, Σ, i, j):

    μ_x, μ_y = μ[i], μ[j]
    σ_x, σ_y = jnp.sqrt(Σ[i, i]), jnp.sqrt(Σ[j, j])
    σ_xy = Σ[i, j]

    x_μ = x - μ_x
    y_μ = y - μ_y

    ρ = σ_xy / (σ_x * σ_y)
    z = x_μ**2 / σ_x**2 + y_μ**2 / σ_y**2 - 2 * ρ * x_μ * y_μ / (σ_x * σ_y)
    denom = 2 * jnp.pi * σ_x * σ_y * jnp.sqrt(1 - ρ**2)

    return jnp.exp(-z / (2 * (1 - ρ**2))) / denom
```

```{code-cell} ipython3
@partial(jax.jit, static_argnums=2)
def count(vec1, vec2, n):
    """Count joint frequencies of integer pairs using JAX bincount."""
    vec1 = vec1.astype(jnp.int32)
    vec2 = vec2.astype(jnp.int32)
    base = n + 1
    idx = vec1 * base + vec2
    counts = jnp.bincount(idx, length=base * base)
    return counts.reshape((base, base))
```

```{code-cell} ipython3
c = urn.c
fig, axs = plt.subplots(c, c, figsize=(14, 14))

# grids for ploting the bivariate Gaussian
x_grid = jnp.linspace(-2, n+1, 100)
y_grid = jnp.linspace(-2, n+1, 100)
X, Y = jnp.meshgrid(x_grid, y_grid)
bin_edges = list(range(0, n + 1))

for i in range(c):
    axs[i, i].hist(sample[:, i],
                bins=bin_edges, alpha=0.5,
                density=True, label='hypergeom')
    axs[i, i].hist(sample_normal[:, i],
                bins=bin_edges, alpha=0.5,
                density=True, label='normal')
    axs[i, i].legend()
    axs[i, i].set_title('$k_{' +str(i+1) +'}$')
    for j in range(c):
        if i == j:
            continue

        # bivariate Gaussian density function
        Z = np.asarray(bivariate_normal(X, Y, μ, Σ, i, j))
        cs = axs[i, j].contour(
            X, Y, Z, 4, colors="black", alpha=0.6)
        axs[i, j].clabel(cs, inline=1, fontsize=10)

        # empirical multivariate hypergeometric distrbution
        count_mat = count(
            sample[:, i], sample[:, j], n)
        axs[i, j].pcolor(
            count_mat.T/size, cmap='Blues')
        axs[i, j].set_title(
            '$(k_{' +str(i+1) +'}, k_{' + str(j+1) + '})$')

plt.show()
```

The diagonal graphs plot the marginal distributions of $k_i$ for
each $i$ using histograms.

Note the substantial differences between hypergeometric distribution and the approximating normal distribution.

The off-diagonal graphs plot the empirical joint distribution of
$k_i$ and $k_j$ for each pair $(i, j)$.

The darker the blue, the more data points are contained in the corresponding cell.  (Note that $k_i$ is on the x-axis and $k_j$ is on the y-axis).

The contour maps plot the bivariate Gaussian density function of $\left(k_i, k_j\right)$ with the population mean and covariance given by slices of $\mu$ and $\Sigma$ that we computed above.

Let's also test the normality for each $k_i$ using `scipy.stats.normaltest` that implements D’Agostino and Pearson's
test that combines skew and kurtosis to form an omnibus test of normality.

The null hypothesis is that the sample follows normal distribution.

> `normaltest` returns an array of p-values associated with tests for each $k_i$ sample.

```{code-cell} ipython3
test_multihyper = normaltest(sample)
test_multihyper.pvalue
```

As we can see, all the p-values are almost $0$ and the null hypothesis is soundly rejected.

By contrast, the sample from normal distribution does not reject the null hypothesis.

```{code-cell} ipython3
test_normal = normaltest(sample_normal)
test_normal.pvalue
```

The lesson to take away from this is that the normal approximation is imperfect.
