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

```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Job Search III: Fitted Value Function Iteration

```{contents} Contents
:depth: 2
```


## Overview

In this lecture we again study the {doc}`McCall job search model with separation <mccall_model_with_separation>`, but now with a continuous wage distribution.

While we already considered continuous wage distributions briefly in the
exercises of the {doc}`first job search lecture <mccall_model>`,
the change was relatively trivial in that case.

This is because we were able to reduce the problem to solving for a single
scalar value (the continuation value).

Here, with separation, the change is less trivial, since a continuous wage distribution leads to an uncountably infinite state space.

The infinite state space leads to additional challenges, particularly when it
comes to applying value function iteration (VFI).

These challenges will lead us to modify VFI by adding an interpolation step.

The combination of VFI and this interpolation step is called **fitted value function iteration** (fitted VFI).

Fitted VFI is very common in practice, so we will take some time to work through the details.

We will use the following imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from typing import NamedTuple
import quantecon as qe
```

## The algorithm

The model is the same as the McCall model with job separation that we {doc}`studied before <mccall_model_with_separation>`, except that the wage offer distribution is continuous.

We are going to start with the two Bellman equations we obtained for the model with job separation after {ref}`a simplifying transformation <ast_mcm>`.

Modified to accommodate continuous wage draws, they take the following form:

```{math}
:label: bell1mcmc

d = \int \max \left\{ v(w'), \,  u(c) + \beta d \right\} q(w') d w'
```

and

```{math}
:label: bell2mcmc

v(w) = u(w) + \beta
    \left[
        (1-\alpha)v(w) + \alpha d
    \right]
```

The unknowns here are the function $v$ and the scalar $d$.

The differences between these and the pair of Bellman equations we previously worked on are

1. In {eq}`bell1mcmc`, what used to be a sum over a finite number of wage values is an integral over an infinite set.
1. The function $v$ in {eq}`bell2mcmc` is defined over all $w \in \mathbb R_+$.

The function $q$ in {eq}`bell1mcmc` is the density of the wage offer distribution.

Its support is taken as equal to $\mathbb R_+$.

### Value function iteration

In theory, we should now proceed as follows:

1. Begin with a guess $v, d$ for the solutions to {eq}`bell1mcmc`--{eq}`bell2mcmc`.
1. Plug $v, d$ into the right hand side of {eq}`bell1mcmc`--{eq}`bell2mcmc` and
   compute the left hand side to obtain updates $v', d'$
1. Unless some stopping condition is satisfied, set $(v, d) = (v', d')$
   and go to step 2.

However, there is a problem we must confront before we implement this procedure:
The iterates of the value function can neither be calculated exactly nor stored on a computer.

To see the issue, consider {eq}`bell2mcmc`.

Even if $v$ is a known function, the only way to store its update $v'$
is to record its value $v'(w)$ for every $w \in \mathbb R_+$.

Clearly, this is impossible.

### Fitted value function iteration

What we will do instead is use **fitted value function iteration**.

The procedure is as follows:

Let a current guess $v$ be given.

Now we record the value of the function $v'$ at only
finitely many "grid" points $w_1 < w_2 < \cdots < w_I$ and then reconstruct $v'$ from this information when required.

More precisely, the algorithm will be

(fvi_alg)=
1. Begin with an array $\mathbf v$ representing the values of an initial guess of the value function on some grid points $\{w_i\}$.
1. Build a function $v$ on the state space $\mathbb R_+$ by interpolation or approximation, based on $\mathbf v$ and $\{ w_i\}$.
1. Obtain and record the samples of the updated function $v'(w_i)$ on each grid point $w_i$.
1. Unless some stopping condition is satisfied, take this as the new array and go to step 1.

How should we go about step 2?

This is a problem of function approximation, and there are many ways to approach it.

What's important here is that the function approximation scheme must not only
produce a good approximation to each $v$, but also that it combines well with the broader iteration algorithm described above.

One good choice from both respects is continuous piecewise linear interpolation.

This method

1. combines well with value function iteration (see, e.g.,
   {cite}`gordon1995stable` or {cite}`stachurski2008continuous`) and
1. preserves useful shape properties such as monotonicity and concavity/convexity.

Linear interpolation will be implemented using JAX's interpolation function `jnp.interp`.

The next figure illustrates piecewise linear interpolation of an arbitrary
function on grid points $0, 0.2, 0.4, 0.6, 0.8, 1$.

```{code-cell} python3
def f(x):
    y1 = 2 * jnp.cos(6 * x) + jnp.sin(14 * x)
    return y1 + 2.5

c_grid = jnp.linspace(0, 1, 6)
f_grid = jnp.linspace(0, 1, 150)

def Af(x):
    return jnp.interp(x, c_grid, f(c_grid))

fig, ax = plt.subplots()

ax.plot(f_grid, f(f_grid), 'b-', label='true function')
ax.plot(f_grid, Af(f_grid), 'g-', label='linear approximation')
ax.vlines(c_grid, c_grid * 0, f(c_grid), linestyle='dashed', alpha=0.5)

ax.legend(loc="upper center")

ax.set(xlim=(0, 1), ylim=(0, 6))
plt.show()
```

## Implementation

The first step is to build a JAX-compatible structure for the McCall model with separation and a continuous wage offer distribution.

We will take the utility function to be the log function for this application, with $u(c) = \ln c$.

We will adopt the lognormal distribution for wages, with $w = \exp(\mu + \sigma z)$
when $z$ is standard normal and $\mu, \sigma$ are parameters.

```{code-cell} python3
def lognormal_draws(n=1000, μ=2.5, σ=0.5, seed=1234):
    key = jax.random.PRNGKey(seed)
    z = jax.random.normal(key, (n,))
    w_draws = jnp.exp(μ + σ * z)
    return w_draws
```

Here's our model structure using a NamedTuple.

```{code-cell} python3
class McCallModelContinuous(NamedTuple):
    c: float              # unemployment compensation
    α: float              # job separation rate
    β: float              # discount factor
    w_grid: jnp.ndarray   # grid of points for fitted VFI
    w_draws: jnp.ndarray  # draws of wages for Monte Carlo

def create_mccall_model(c=1,
                        α=0.1,
                        β=0.96,
                        grid_min=1e-10,
                        grid_max=5,
                        grid_size=100,
                        μ=2.5,
                        σ=0.5,
                        mc_size=1000,
                        seed=1234,
                        w_draws=None):
    """Factory function to create a McCall model instance."""
    if w_draws is None:
        # Generate wage draws if not provided
        w_draws = lognormal_draws(n=mc_size, μ=μ, σ=σ, seed=seed)

    w_grid = jnp.linspace(grid_min, grid_max, grid_size)
    return McCallModelContinuous(c=c, α=α, β=β, w_grid=w_grid, w_draws=w_draws)

@jax.jit
def update(model, v, d):
    """Update value function and continuation value."""
    # Unpack model parameters
    c, α, β, w_grid, w_draws = model
    u = jnp.log
    
    # Interpolate array represented value function
    vf = lambda x: jnp.interp(x, w_grid, v)
    
    # Update d using Monte Carlo to evaluate integral
    d_new = jnp.mean(jnp.maximum(vf(w_draws), u(c) + β * d))
    
    # Update v
    v_new = u(w_grid) + β * ((1 - α) * v + α * d)
    
    return v_new, d_new
```

We then return the current iterate as an approximate solution.

```{code-cell} python3
@jax.jit
def solve_model(model, tol=1e-5, max_iter=2000):
    """
    Iterates to convergence on the Bellman equations

    * model is an instance of McCallModelContinuous
    """
    
    # Initial guesses
    v = jnp.ones_like(model.w_grid)
    d = 1.0
    
    def body_fun(state):
        v, d, i, error = state
        v_new, d_new = update(model, v, d)
        error_1 = jnp.max(jnp.abs(v_new - v))
        error_2 = jnp.abs(d_new - d)
        error = jnp.maximum(error_1, error_2)
        return v_new, d_new, i + 1, error
    
    def cond_fun(state):
        _, _, i, error = state
        return (error > tol) & (i < max_iter)
    
    initial_state = (v, d, 0, tol + 1)
    v_final, d_final, _, _ = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    
    return v_final, d_final
```

Here's a function `compute_reservation_wage` that takes an instance of `McCallModelContinuous`
and returns the associated reservation wage.

If $v(w) < h$ for all $w$, then the function returns `jnp.inf`

```{code-cell} python3
@jax.jit
def compute_reservation_wage(model):
    """
    Computes the reservation wage of an instance of the McCall model
    by finding the smallest w such that v(w) >= h.

    If no such w exists, then w_bar is set to inf.
    """
    c, α, β, w_grid, w_draws = model
    u = jnp.log
    
    v, d = solve_model(model)
    h = u(c) + β * d
    
    # Find the first wage where v(w) >= h
    indices = jnp.where(v >= h, size=1, fill_value=-1)
    w_bar = jnp.where(indices[0] >= 0, w_grid[indices[0]], jnp.inf)
    
    return w_bar
```

The exercises ask you to explore the solution and how it changes with parameters.

## Exercises

```{exercise}
:label: mfv_ex1

Use the code above to explore what happens to the reservation wage when the wage parameter $\mu$
changes.

Use the default parameters and $\mu$ in `μ_vals = jnp.linspace(0.0, 2.0, 15)`.

Is the impact on the reservation wage as you expected?
```

```{solution-start} mfv_ex1
:class: dropdown
```

Here is one solution

```{code-cell} python3
def compute_res_wage_given_μ(μ):
    model = create_mccall_model(μ=μ)
    w_bar = compute_reservation_wage(model)
    return w_bar

μ_vals = jnp.linspace(0.0, 2.0, 15)
w_bar_vals = jax.vmap(compute_res_wage_given_μ)(μ_vals)

fig, ax = plt.subplots()
ax.set(xlabel='mean', ylabel='reservation wage')
ax.plot(μ_vals, w_bar_vals, label=r'$\bar w$ as a function of $\mu$')
ax.legend()
plt.show()
```

Not surprisingly, the agent is more inclined to wait when the distribution of
offers shifts to the right.

```{solution-end}
```

```{exercise}
:label: mfv_ex2

Let us now consider how the agent responds to an increase in volatility.

To try to understand this, compute the reservation wage when the wage offer
distribution is uniform on $(m - s, m + s)$ and $s$ varies.

The idea here is that we are holding the mean constant and spreading the
support.

(This is a form of *mean-preserving spread*.)

Use `s_vals = jnp.linspace(1.0, 2.0, 15)` and `m = 2.0`.

State how you expect the reservation wage to vary with $s$.

Now compute it - is this as you expected?
```

```{solution-start} mfv_ex2
:class: dropdown
```

Here is one solution

```{code-cell} python3
def compute_res_wage_given_s(s, m=2.0, seed=1234):
    a, b = m - s, m + s
    key = jax.random.PRNGKey(seed)
    uniform_draws = jax.random.uniform(key, shape=(10_000,), minval=a, maxval=b)
    # Create model with default parameters but replace wage draws
    model = create_mccall_model(w_draws=uniform_draws)
    w_bar = compute_reservation_wage(model)
    return w_bar

s_vals = jnp.linspace(1.0, 2.0, 15)
# Use vmap with different seeds for each s value
seeds = jnp.arange(len(s_vals))
compute_vectorized = jax.vmap(compute_res_wage_given_s, in_axes=(0, None, 0))
w_bar_vals = compute_vectorized(s_vals, 2.0, seeds)

fig, ax = plt.subplots()
ax.set(xlabel='volatility', ylabel='reservation wage')
ax.plot(s_vals, w_bar_vals, label=r'$\bar w$ as a function of wage volatility')
ax.legend()
plt.show()
```

The reservation wage increases with volatility.

One might think that higher volatility would make the agent more inclined to
take a given offer, since doing so represents certainty and waiting represents
risk.

But job search is like holding an option: the worker is only exposed to upside risk (since, in a free market, no one can force them to take a bad offer).

More volatility means higher upside potential, which encourages the agent to wait.

```{solution-end}
```
