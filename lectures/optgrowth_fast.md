---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(optgrowth_fast)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`Optimal Growth II: Accelerating the Code with JAX <single: Optimal Growth II: Accelerating the Code with JAX>`

```{contents} Contents
:depth: 2
```

In addition to what is in Anaconda, this lecture needs an extra package.

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon jax
```

## Overview

{doc}`Previously <optgrowth>`, we studied a stochastic optimal
growth model with one representative agent.

We solved the model using dynamic programming.

In writing our code, we focused on clarity and flexibility.

These are important, but there's often a trade-off between flexibility and
speed.

The reason is that, when code is less flexible, we can exploit structure more
easily.

(This is true about algorithms and mathematical problems more generally:
more specific problems have more structure, which, with some thought, can be
exploited for better results.)

So, in this lecture, we are going to accept less flexibility while gaining
speed, using just-in-time (JIT) compilation in JAX to
accelerate our code.

Let's start with some imports:

```{code-cell} ipython
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple
import quantecon as qe

jax.config.update("jax_platform_name", "cpu")
```

## The model

The model is the same as in our {doc}`previous lecture <optgrowth>` on optimal growth.

We use log utility in the baseline case.

$$
u(c) = \ln(c)
$$

We continue to assume that

* $f(k) = k^{\alpha}$
* $\phi$ is the distribution of $\xi := \exp(\mu + s \zeta)$ where $\zeta$ is standard normal

We will once again use value function iteration to solve the model.

The algorithm is unchanged, but the implementation uses JAX.

As before, we will be able to compare with the true solutions

```{code-cell} python3
:load: _static/lecture_specific/optgrowth/cd_analytical.py
```

## Computation

We store primitives in a `NamedTuple` built for JAX and create a factory function to generate instances.

```{code-cell} python3
class OptimalGrowthModel(NamedTuple):
    α: float               # production parameter
    β: float               # discount factor
    μ: float               # shock location parameter
    s: float               # shock scale parameter
    γ: float               # CRRA parameter (γ = 1 gives log)
    y_grid: jnp.ndarray    # grid for output/income
    shocks: jnp.ndarray    # Monte Carlo draws of ξ
    c_grid_frac: jnp.ndarray  # fractional consumption grid in (0, 1)


def create_optgrowth_model(α=0.4,
                           β=0.96,
                           μ=0.0,
                           s=0.1,
                           γ=1.0,
                           grid_max=4.0,
                           grid_size=120,
                           shock_size=250,
                           c_grid_size=200,
                           seed=0):
    """Factory function to create an OptimalGrowthModel instance."""

    key = jax.random.PRNGKey(seed)
    y_grid = jnp.linspace(1e-5, grid_max, grid_size)
    z = jax.random.normal(key, (shock_size,))
    shocks = jnp.exp(μ + s * z)

    # Avoid endpoints 0 and 1 to keep feasibility and positivity.
    c_grid_frac = jnp.linspace(1e-6, 1.0 - 1e-6, c_grid_size)
    return OptimalGrowthModel(α=α, β=β, μ=μ, s=s, γ=γ,
                                 y_grid=y_grid, shocks=shocks,
                                 c_grid_frac=c_grid_frac)
```

We now implement the CRRA utility function, the Bellman operator and the value function iteration loop using JAX

```{code-cell} python3
@jax.jit
def u(c, γ):
    # CRRA utility with log at γ = 1.
    return jnp.where(jnp.isclose(γ, 1.0), 
            jnp.log(c), (c**(1.0 - γ) - 1.0) / (1.0 - γ))


@jax.jit
def T(v, model):
    """
    Bellman operator returning greedy policy and updated value.
    """
    α, β, γ, shocks = model.α, model.β, model.γ, model.shocks
    y_grid, c_grid_frac = model.y_grid, model.c_grid_frac

    # Interpolant for value function on the state grid.
    vf = lambda x: jnp.interp(x, y_grid, v)

    def solve_state(y):
        # Candidate consumptions scaled by income.
        c = c_grid_frac * y

        # Next income for each c and each shock.
        k = jnp.maximum(y - c, 1e-12)
        y_next = (k**α)[:, None] * shocks[None, :]

        # Expected continuation value via Monte Carlo.
        v_next = vf(y_next.reshape(-1)).reshape(
                c.shape[0], shocks.shape[0]).mean(axis=1)

        # Objective on the consumption grid.
        obj = u(c, γ) + β * v_next

        # Maximize over c-grid.
        idx = jnp.argmax(obj)

        c_star = c[idx]
        v_val = obj[idx]
        return c_star, v_val

    # Vectorize across states.
    c_star_vec, v_new_vec = jax.vmap(solve_state)(y_grid)
    return c_star_vec, v_new_vec


@jax.jit
def vfi(model, tol=1e-4, max_iter=1_000):
    """Iterate on the Bellman operator until convergence."""
    y_grid = model.y_grid
    v0 = u(y_grid, model.γ)

    def body(state):
        v, i, err = state
        _, v_new = T(v, model)
        err = jnp.max(jnp.abs(v_new - v))
        return v_new, i + 1, err

    def cond(state):
        _, i, err = state
        return (err > tol) & (i < max_iter)

    v_final, _, _ = jax.lax.while_loop(cond, body, (v0, 0, tol + 1.0))
    c_greedy, v_solution = T(v_final, model)
    return c_greedy, v_solution
```

Let us compute the approximate solution at the default parameters

```{code-cell} python3
og = create_optgrowth_model()

with qe.Timer(unit="milliseconds"):
    v_greedy = vfi(og)[0].block_until_ready()
```

Here is a plot of the resulting policy, compared with the true policy:

```{code-cell} python3
fig, ax = plt.subplots()

ax.plot(og.y_grid, v_greedy, lw=2, alpha=0.8, 
            label='approximate policy function')

ax.plot(og.y_grid, (1 - og.α * og.β) * og.y_grid, 
            'k--', lw=2, alpha=0.8, label='true policy function')

ax.legend()
plt.show()
```

Again, the fit is excellent --- this is as expected since we have not changed
the algorithm.

The maximal absolute deviation between the two policies is

```{code-cell} python3
np.max(np.abs(np.asarray(v_greedy) 
            - np.asarray((1 - og.α * og.β) * og.y_grid)))
```

## Exercises

```{exercise-start}
:label: ogfast_ex1
```

Time how long it takes to iterate with the Bellman operator 20 times, starting from initial condition $v(y) = u(y)$.

Use the default parameterization.
```{exercise-end}
```

```{solution-start} ogfast_ex1
:class: dropdown
```

Let's set up the initial condition.

```{code-cell} ipython3
v = u(og.y_grid, og.γ)
```

Here is the timing.

```{code-cell} ipython3
with qe.Timer(unit="milliseconds"):
    for _ in range(20):
        v = T(v, og)[1].block_until_ready()
```

Compared with our {ref}`timing <og_ex2>` for the non-compiled version of
value function iteration, the JIT-compiled code is usually an order of magnitude faster.

```{solution-end}
```

```{exercise-start}
:label: ogfast_ex2
```
Modify the optimal growth model to use the CRRA utility specification.

$$
u(c) = \frac{c^{1 - \gamma} } {1 - \gamma}
$$

Set `γ = 1.5` as the default value while maintaining other specifications.

Use the JAX implementation above and change only the utility parameter.

Compute an estimate of the optimal policy and plot it.

Compare visually with the same plot from the {ref}`analogous exercise <og_ex1>` in the first optimal growth lecture.

Compare execution time as well.
```{exercise-end}
```


```{solution-start} ogfast_ex2
:class: dropdown
```

Here is the CRRA variant using the same code path

```{code-cell} python3
og_crra = create_optgrowth_model(γ=1.5)
```

Let's solve and time the model

```{code-cell} python3
with qe.Timer(unit="milliseconds"):
    v_greedy = vfi(og_crra)[0].block_until_ready()
```

Here is a plot of the resulting policy

```{code-cell} python3
fig, ax = plt.subplots()

ax.plot(og_crra.y_grid, v_greedy, lw=2, alpha=0.6, 
            label='approximate policy function')

ax.legend(loc='lower right')
plt.show()
```

This matches the solution obtained in the non-jitted code in {ref}`the earlier exercise <og_ex1>`.

Execution time is an order of magnitude faster.

```{solution-end}
```


```{exercise-start}
:label: ogfast_ex3
```

In this exercise we return to the original log utility specification.

Once an optimal consumption policy $\sigma$ is given, income follows

$$
y_{t+1} = f(y_t - \sigma(y_t)) \xi_{t+1}
$$

The next figure shows a simulation of 100 elements of this sequence for three different discount factors and hence three different policies.

```{image} /_static/lecture_specific/optgrowth/solution_og_ex2.png
:align: center
```

In each sequence, the initial condition is $y_0 = 0.1$.

The discount factors are `discount_factors = (0.8, 0.9, 0.98)`.

We have also dialed down the shocks a bit with `s = 0.05`.

Other parameters match the log-linear model discussed earlier.

Notice that more patient agents typically have higher wealth.

Replicate the figure modulo randomness.

```{exercise-end}
```

```{solution-start} ogfast_ex3
:class: dropdown
```

Here is one solution.

```{code-cell} python3
import jax.random as jr

def simulate_og(σ_func, og_model, y0=0.1, ts_length=100, seed=0):
    """Compute a time series given consumption policy σ."""
    key = jr.PRNGKey(seed)
    ξ = jr.normal(key, (ts_length - 1,))
    y = np.empty(ts_length)
    y[0] = y0
    for t in range(ts_length - 1):
        y[t+1] = (y[t] - σ_func(y[t]))**og_model.α \
                    * np.exp(og_model.μ + og_model.s * ξ[t])
    return y
```

```{code-cell} python3
fig, ax = plt.subplots()

for β in (0.8, 0.9, 0.98):

    og_temp = create_optgrowth_model(β=β, s=0.05)
    v_greedy, v_solution = vfi(og_temp)

    # Define an optimal policy function
    σ_func = lambda x: np.interp(x, og_temp.y_grid, np.asarray(v_greedy))
    y = simulate_og(σ_func, og_temp)
    ax.plot(y, lw=2, alpha=0.6, label=rf'$\beta = {β}$')

ax.legend(loc='lower right')
plt.show()
```

```{solution-end}
```
