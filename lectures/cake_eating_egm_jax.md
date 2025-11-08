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

# {index}`Cake Eating VI: EGM with JAX <single: Cake Eating VI: EGM with JAX>`

```{contents} Contents
:depth: 2
```


## Overview

In this lecture, we'll implement the endogenous grid method (EGM) using JAX.

This lecture builds on {doc}`cake_eating_egm`, which introduced EGM using NumPy.

By converting to JAX, we can leverage fast linear algebra, hardware accelerators, and JIT compilation for improved performance.

We'll also use JAX's `vmap` function to fully vectorize the Coleman-Reffett operator.

Let's start with some standard imports:

```{code-cell} ipython
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import quantecon as qe
```

## Implementation

For details on the endogenous grid method, please see {doc}`cake_eating_egm`.

Here we focus on the JAX implementation.

We use the same setting as in {doc}`cake_eating_egm`:

* $u(c) = \ln c$,
* production is Cobb-Douglas, and
* the shocks are lognormal.

Here are the analytical solutions for comparison.

```{code-cell} python3
def v_star(x, α, β, μ):
    """
    True value function
    """
    c1 = jnp.log(1 - α * β) / (1 - β)
    c2 = (μ + α * jnp.log(α * β)) / (1 - α)
    c3 = 1 / (1 - β)
    c4 = 1 / (1 - α * β)
    return c1 + c2 * (c3 - c4) + c4 * jnp.log(x)

def σ_star(x, α, β):
    """
    True optimal policy
    """
    return (1 - α * β) * x
```

The `Model` class stores only the data (grids, shocks, and parameters).

Utility and production functions will be defined globally to work with JAX's JIT compiler.

```{code-cell} python3
from typing import NamedTuple, Callable

class Model(NamedTuple):
    β: float              # discount factor
    μ: float              # shock location parameter
    s: float              # shock scale parameter
    grid: jnp.ndarray     # state grid
    shocks: jnp.ndarray   # shock draws
    α: float              # production function parameter


def create_model(β: float = 0.96,
                 μ: float = 0.0,
                 s: float = 0.1,
                 grid_max: float = 4.0,
                 grid_size: int = 120,
                 shock_size: int = 250,
                 seed: int = 1234,
                 α: float = 0.4) -> Model:
    """
    Creates an instance of the cake eating model.
    """
    # Set up grid
    grid = jnp.linspace(1e-4, grid_max, grid_size)

    # Store shocks (with a seed, so results are reproducible)
    key = jax.random.PRNGKey(seed)
    shocks = jnp.exp(μ + s * jax.random.normal(key, shape=(shock_size,)))

    return Model(β=β, μ=μ, s=s, grid=grid, shocks=shocks, α=α)
```

Here's the Coleman-Reffett operator using EGM.

The key JAX feature here is `vmap`, which vectorizes the computation over the grid points.

```{code-cell} python3
def K(σ_array: jnp.ndarray, model: Model) -> jnp.ndarray:
    """
    The Coleman-Reffett operator using EGM

    """

    # Simplify names
    β, α = model.β, model.α
    grid, shocks = model.grid, model.shocks

    # Determine endogenous grid
    x = grid + σ_array  # x_i = k_i + c_i

    # Linear interpolation of policy using endogenous grid
    σ = lambda x_val: jnp.interp(x_val, x, σ_array)

    # Define function to compute consumption at a single grid point
    def compute_c(k):
        vals = u_prime(σ(f(k, α) * shocks)) * f_prime(k, α) * shocks
        return u_prime_inv(β * jnp.mean(vals))

    # Vectorize over grid using vmap
    compute_c_vectorized = jax.vmap(compute_c)
    c = compute_c_vectorized(grid)

    return c
```

We define utility and production functions globally.

Note that `f` and `f_prime` take `α` as an explicit argument, allowing them to work with JAX's functional programming model.

```{code-cell} python3
# Define utility and production functions with derivatives
u = lambda c: jnp.log(c)
u_prime = lambda c: 1 / c
u_prime_inv = lambda x: 1 / x
f = lambda k, α: k**α
f_prime = lambda k, α: α * k**(α - 1)
```

Now we create a model instance.

```{code-cell} python3
α = 0.4

model = create_model(α=α)
grid = model.grid
```

The solver uses JAX's `jax.lax.while_loop` for the iteration and is JIT-compiled for speed.

```{code-cell} python3
@jax.jit
def solve_model_time_iter(model: Model,
                          σ_init: jnp.ndarray,
                          tol: float = 1e-5,
                          max_iter: int = 1000) -> jnp.ndarray:
    """
    Solve the model using time iteration with EGM.
    """

    def condition(loop_state):
        i, σ, error = loop_state
        return (error > tol) & (i < max_iter)

    def body(loop_state):
        i, σ, error = loop_state
        σ_new = K(σ, model)
        error = jnp.max(jnp.abs(σ_new - σ))
        return i + 1, σ_new, error

    # Initialize loop state
    initial_state = (0, σ_init, tol + 1)

    # Run the loop
    i, σ, error = jax.lax.while_loop(condition, body, initial_state)

    return σ
```

We solve the model starting from an initial guess.

```{code-cell} python3
σ_init = jnp.copy(grid)
σ = solve_model_time_iter(model, σ_init)
```

Let's plot the resulting policy against the analytical solution.

```{code-cell} python3
x = grid + σ  # x_i = k_i + c_i

fig, ax = plt.subplots()

ax.plot(x, σ, lw=2,
        alpha=0.8, label='approximate policy function')

ax.plot(x, σ_star(x, model.α, model.β), 'k--',
        lw=2, alpha=0.8, label='true policy function')

ax.legend()
plt.show()
```

The fit is excellent.

```{code-cell} python3
jnp.max(jnp.abs(σ - σ_star(x, model.α, model.β)))
```

The JAX implementation is very fast thanks to JIT compilation and vectorization.

```{code-cell} python3
with qe.Timer():
    σ = solve_model_time_iter(model, σ_init)
```

This speed comes from:

* JIT compilation of the entire solver
* Vectorization via `vmap` in the Coleman-Reffett operator
* Use of `jax.lax.while_loop` instead of a Python loop
* Efficient JAX array operations throughout
