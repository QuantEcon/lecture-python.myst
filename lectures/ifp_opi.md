---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# The Income Fluctuation Problem II: Optimistic Policy Iteration


## Overview

In {doc}`ifp_discrete` we studied the income fluctuation problem and solved it using value function iteration (VFI).

In this lecture we'll solve the same problem using **optimistic policy iteration** (OPI), which is a faster alternative to VFI.

OPI combines elements of both value function iteration and policy iteration.

The algorithm can be found in [this book](https://dp.quantecon.org), where a PDF is freely available.

We will show that OPI provides significant speed improvements over standard VFI for the income fluctuation problem.

For details on the income fluctuation problem, see {doc}`ifp_discrete`.

In addition to Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon jax
```

We will use the following imports:

```{code-cell} ipython3
import quantecon as qe
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import NamedTuple
from time import time
```


We'll use 64 bit floats to gain extra precision.

```{code-cell} ipython3
jax.config.update("jax_enable_x64", True)
```

## Model and Primitives

The model and parameters are the same as in {doc}`ifp_discrete`.

We repeat the key elements here for convenience.

The household's problem is to maximize

$$
\mathbb{E} \, \sum_{t=0}^{\infty} \beta^t u(c_t)
$$

subject to

$$
    a_{t+1} + c_t \leq R a_t + y_t
$$

where $u(c) = c^{1-\gamma}/(1-\gamma)$.

Here's the model structure:

```{code-cell} ipython3
class Model(NamedTuple):
    β: float              # Discount factor
    R: float              # Gross interest rate
    γ: float              # CRRA parameter
    a_grid: jnp.ndarray   # Asset grid
    y_grid: jnp.ndarray   # Income grid
    Q: jnp.ndarray        # Markov matrix for income


def create_consumption_model(R=1.01,                    # Gross interest rate
                             β=0.98,                    # Discount factor
                             γ=2,                       # CRRA parameter
                             a_min=0.01,                # Min assets
                             a_max=5.0,                 # Max assets
                             a_size=150,                # Grid size
                             ρ=0.9, ν=0.1, y_size=100): # Income parameters
    """
    Creates an instance of the consumption-savings model.
    """
    a_grid = jnp.linspace(a_min, a_max, a_size)
    mc = qe.tauchen(n=y_size, rho=ρ, sigma=ν)
    y_grid, Q = jnp.exp(mc.state_values), jax.device_put(mc.P)
    return Model(β, R, γ, a_grid, y_grid, Q)
```

## Operators and Policies

We need to define several operators for implementing OPI.

First, the right hand side of the Bellman equation:

```{code-cell} ipython3
@jax.jit
def B(v, model):
    """
    A vectorized version of the right-hand side of the Bellman equation
    (before maximization), which is a 3D array representing

        B(a, y, a′) = u(Ra + y - a′) + β Σ_y′ v(a′, y′) Q(y, y′)

    for all (a, y, a′).
    """

    # Unpack
    β, R, γ, a_grid, y_grid, Q = model
    a_size, y_size = len(a_grid), len(y_grid)

    # Compute current rewards r(a, y, ap) as array r[i, j, ip]
    a  = jnp.reshape(a_grid, (a_size, 1, 1))    # a[i]   ->  a[i, j, ip]
    y  = jnp.reshape(y_grid, (1, y_size, 1))    # z[j]   ->  z[i, j, ip]
    ap = jnp.reshape(a_grid, (1, 1, a_size))    # ap[ip] -> ap[i, j, ip]
    c = R * a + y - ap

    # Calculate continuation rewards at all combinations of (a, y, ap)
    v = jnp.reshape(v, (1, 1, a_size, y_size))  # v[ip, jp] -> v[i, j, ip, jp]
    Q = jnp.reshape(Q, (1, y_size, 1, y_size))  # Q[j, jp]  -> Q[i, j, ip, jp]
    EV = jnp.sum(v * Q, axis=3)                 # sum over last index jp

    # Compute the right-hand side of the Bellman equation
    return jnp.where(c > 0, c**(1-γ)/(1-γ) + β * EV, -jnp.inf)
```

The Bellman operator:

```{code-cell} ipython3
@jax.jit
def T(v, model):
    "The Bellman operator."
    return jnp.max(B(v, model), axis=2)
```

The greedy policy:

```{code-cell} ipython3
@jax.jit
def get_greedy(v, model):
    "Computes a v-greedy policy, returned as a set of indices."
    return jnp.argmax(B(v, model), axis=2)
```

Now we define the policy operator $T_\sigma$, which is the Bellman operator with policy $\sigma$ fixed.

For a given policy $\sigma$, the policy operator is defined by

$$
    (T_\sigma v)(a, y) = u(Ra + y - \sigma(a, y)) + \beta \sum_{y'} v(\sigma(a, y), y') Q(y, y')
$$

```{code-cell} ipython3
def T_σ(v, σ, model, i, j):
    """
    The σ-policy operator for indices (i, j) -> (a, y).
    """
    β, R, γ, a_grid, y_grid, Q = model

    # Get values at current state
    a, y = a_grid[i], y_grid[j]
    # Get policy choice
    ap = a_grid[σ[i, j]]

    # Compute current reward
    c = R * a + y - ap
    r = jnp.where(c > 0, c**(1-γ)/(1-γ), -jnp.inf)

    # Compute expected value
    EV = jnp.sum(v[σ[i, j], :] * Q[j, :])

    return r + β * EV
```

Apply vmap to vectorize:

```{code-cell} ipython3
T_σ_1    = jax.vmap(T_σ,   in_axes=(None, None, None, None, 0))
T_σ_vmap = jax.vmap(T_σ_1, in_axes=(None, None, None, 0,    None))

@jax.jit
def T_σ_vec(v, σ, model):
    """Vectorized version of T_σ."""
    a_size, y_size = len(model.a_grid), len(model.y_grid)
    a_indices = jnp.arange(a_size)
    y_indices = jnp.arange(y_size)
    return T_σ_vmap(v, σ, model, a_indices, y_indices)
```

Now we need a function to apply the policy operator m times:

```{code-cell} ipython3
@jax.jit
def iterate_policy_operator(σ, v, m, model):
    """
    Apply the policy operator T_σ exactly m times to v.
    """
    def update(i, v):
        return T_σ_vec(v, σ, model)

    v = jax.lax.fori_loop(0, m, update, v)
    return v
```

## Value Function Iteration

For comparison, here's VFI from {doc}`ifp_discrete`:

```{code-cell} ipython3
@jax.jit
def value_function_iteration(model, tol=1e-5, max_iter=10_000):
    """
    Implements VFI using successive approximation.
    """
    def body_fun(k_v_err):
        k, v, error = k_v_err
        v_new = T(v, model)
        error = jnp.max(jnp.abs(v_new - v))
        return k + 1, v_new, error

    def cond_fun(k_v_err):
        k, v, error = k_v_err
        return jnp.logical_and(error > tol, k < max_iter)

    v_init = jnp.zeros((len(model.a_grid), len(model.y_grid)))
    k, v_star, error = jax.lax.while_loop(cond_fun, body_fun,
                                          (1, v_init, tol + 1))
    return v_star, get_greedy(v_star, model)
```

## Optimistic Policy Iteration

Now we implement OPI.

The algorithm alternates between

1. Performing $m$ policy operator iterations to update the value function
2. Computing a new greedy policy based on the updated value function

```{code-cell} ipython3
@jax.jit
def optimistic_policy_iteration(model, m=10, tol=1e-5, max_iter=10_000):
    """
    Implements optimistic policy iteration with step size m.

    Parameters:
    -----------
    model : Model
        The consumption-savings model
    m : int
        Number of policy operator iterations per step
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    """
    v_init = jnp.zeros((len(model.a_grid), len(model.y_grid)))

    def condition_function(inputs):
        i, v, error = inputs
        return jnp.logical_and(error > tol, i < max_iter)

    def update(inputs):
        i, v, error = inputs
        last_v = v
        σ = get_greedy(v, model)
        v = iterate_policy_operator(σ, v, m, model)
        error = jnp.max(jnp.abs(v - last_v))
        i += 1
        return i, v, error

    num_iter, v, error = jax.lax.while_loop(condition_function,
                                            update,
                                            (0, v_init, tol + 1))

    return v, get_greedy(v, model)
```

## Timing Comparison

Let's create a model and compare the performance of VFI and OPI.

```{code-cell} ipython3
model = create_consumption_model()
```

First, let's time VFI:

```{code-cell} ipython3
print("Starting VFI.")
start = time()
v_star_vfi, σ_star_vfi = value_function_iteration(model)
v_star_vfi.block_until_ready()
vfi_time_with_compile = time() - start
print(f"VFI completed in {vfi_time_with_compile:.2f} seconds.")
```

Run it again to eliminate compile time:

```{code-cell} ipython3
start = time()
v_star_vfi, σ_star_vfi = value_function_iteration(model)
v_star_vfi.block_until_ready()
vfi_time = time() - start
print(f"VFI completed in {vfi_time:.2f} seconds.")
```

Now let's time OPI with different values of m:

```{code-cell} ipython3
print("Starting OPI with m=10.")
start = time()
v_star_opi, σ_star_opi = optimistic_policy_iteration(model, m=10)
v_star_opi.block_until_ready()
opi_time_with_compile = time() - start
print(f"OPI completed in {opi_time_with_compile:.2f} seconds.")
```

Run it again:

```{code-cell} ipython3
start = time()
v_star_opi, σ_star_opi = optimistic_policy_iteration(model, m=10)
v_star_opi.block_until_ready()
opi_time = time() - start
print(f"OPI completed in {opi_time:.2f} seconds.")
```

Check that we get the same result:

```{code-cell} ipython3
print(f"Policies match: {jnp.allclose(σ_star_vfi, σ_star_opi)}")
```

Here's the speedup:

```{code-cell} ipython3
print(f"Speedup factor: {vfi_time / opi_time:.2f}")
```

Let's try different values of m to see how it affects performance:

```{code-cell} ipython3
m_vals = [1, 5, 10, 25, 50, 100, 200, 400]
opi_times = []

for m in m_vals:
    start = time()
    v_star, σ_star = optimistic_policy_iteration(model, m=m)
    v_star.block_until_ready()
    elapsed = time() - start
    opi_times.append(elapsed)
    print(f"OPI with m={m:3d} completed in {elapsed:.2f} seconds.")
```

Plot the results:

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(m_vals, opi_times, 'o-', label='OPI')
ax.axhline(vfi_time, linestyle='--', color='red', label='VFI')
ax.set_xlabel('m (policy steps per iteration)')
ax.set_ylabel('time (seconds)')
ax.legend()
ax.set_title('OPI execution time vs step size m')
plt.show()
```

The results show interesting behavior across different values of m:

* When m=1, OPI is actually slower than VFI, even though they should be mathematically equivalent. This is because the OPI implementation has overhead from computing the greedy policy and calling the policy operator, making it less efficient than the direct VFI approach for m=1.

* The optimal performance occurs around m=25-50, where OPI achieves roughly 3x speedup over VFI.

* For very large m (200, 400), performance degrades as we spend too much time iterating the policy operator before updating the policy.

This demonstrates that there's a "sweet spot" for the OPI step size m that balances between policy updates and value function iterations.

## Exercises

```{exercise}
:label: ifp_opi_ex1

Experiment with different parameter values for the income process ($\rho$ and $\nu$) and see how they affect the relative performance of VFI vs OPI.

Try:
* $\rho \in \{0.8, 0.9, 0.95\}$
* $\nu \in \{0.05, 0.1, 0.2\}$

For each combination, compute the speedup factor (VFI time / OPI time) and report your findings.
```

```{solution-start} ifp_opi_ex1
:class: dropdown
```

Here's one solution:

```{code-cell} ipython3
ρ_vals = [0.8, 0.9, 0.95]
ν_vals = [0.05, 0.1, 0.2]

results = []

for ρ in ρ_vals:
    for ν in ν_vals:
        print(f"\nTesting ρ={ρ}, ν={ν}")

        # Create model
        model = create_consumption_model(ρ=ρ, ν=ν)

        # Time VFI
        start = time()
        v_vfi, σ_vfi = value_function_iteration(model)
        v_vfi.block_until_ready()
        vfi_t = time() - start

        # Time OPI
        start = time()
        v_opi, σ_opi = optimistic_policy_iteration(model, m=10)
        v_opi.block_until_ready()
        opi_t = time() - start

        speedup = vfi_t / opi_t
        results.append((ρ, ν, speedup))
        print(f"  VFI: {vfi_t:.2f}s, OPI: {opi_t:.2f}s, Speedup: {speedup:.2f}x")

# Print summary
print("\nSummary of speedup factors:")
for ρ, ν, speedup in results:
    print(f"ρ={ρ}, ν={ν}: {speedup:.2f}x")
```

```{solution-end}
```
