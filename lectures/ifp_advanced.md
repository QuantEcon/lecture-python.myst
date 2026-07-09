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

# {index}`The Income Fluctuation Problem V: Stochastic Returns on Assets <single: The Income Fluctuation Problem V: Stochastic Returns on Assets>`

```{include} _admonition/gpu.md
```

```{contents} Contents
:depth: 2
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install quantecon
```

## Overview

In this lecture, we continue our study of the income fluctuation problem described in {doc}`ifp_egm`.

While the interest rate was previously taken to be fixed, we now allow
returns on assets to be state-dependent.

This matches the fact that most households with a positive level of assets
face some capital income risk.

It has been argued that modeling capital income risk is essential for
understanding the joint distribution of income and wealth (see, e.g.,
{cite}`benhabib2015` or {cite}`stachurski2019impossibility`).

Theoretical properties of the household savings model presented here are
analyzed in detail in {cite}`ma2020income`.

In terms of computation, we use a combination of time iteration and the
endogenous grid method to solve the model quickly and accurately.

We require the following imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe
import jax
import jax.numpy as jnp
from jax import vmap
from typing import NamedTuple
from functools import partial
```



## The Model

In this section we review the household problem and optimality results.

### Set Up

A household chooses a consumption-asset path $\{(c_t, a_t)\}$ to
maximize

```{math}
:label: trans_at

\mathbb E \left\{ \sum_{t=0}^\infty \beta^t u(c_t) \right\}
```

subject to

```{math}
:label: trans_at2

a_{t+1} = R_{t+1} (a_t - c_t) + Y_{t+1}
\; \text{ and } \;
0 \leq c_t \leq a_t,
```

with initial condition $(a_0, Z_0)=(a,z)$ treated as given.

The only difference from {doc}`ifp_egm_transient_shocks` is that $\{R_t\}_{t \geq 1}$, the gross rate of return on wealth, is allowed to be stochastic.

In particular, we assume that

```{math}
:label: eq:RY_func

    R_t = R(Z_t, \zeta_t)
      \quad \text{and} \quad
    Y_t = Y(Z_t, \eta_t),
```

where

* $R$ and $Y$ are time-invariant nonnegative functions,
* the innovation processes $\{\zeta_t\}$ and
  $\{\eta_t\}$ are IID and independent of each other, and
* $\{Z_t\}_{t \geq 0}$ is a Markov chain on a finite set $\mathsf Z$

Let $P$ represent the Markov matrix for the chain $\{Z_t\}_{t \geq 0}$.

In what follows, $\mathbb E_z \hat X$ means expectation of next period value $\hat X$ given current value $Z = z$.

### Assumptions

We need restrictions to ensure that the objective {eq}`trans_at` is finite and
the solution methods described below converge.

We also need to ensure that the present discounted value of wealth
does not grow too quickly.

When $\{R_t\}$ was constant we required that $\beta R < 1$.

Since it is now stochastic, we require (see {cite}`ma2020income`) that

```{math}
:label: fpbc2

\beta G_R < 1,
\quad \text{where} \quad
G_R := \lim_{n \to \infty}
\left(\mathbb E \prod_{t=1}^n R_t \right)^{1/n}
```

The value $G_R$ can be thought of as the long run (geometric) average
gross rate of return.

To simplify this lecture, we will *assume that the interest rate process is
IID*.

In that case, it is clear from the definition of $G_R$ that $G_R$ is just $\mathbb E R_t$.

We test the condition $\beta \mathbb E R_t < 1$ in the code below.

Finally, we impose some routine technical restrictions on non-financial income.

$$
\mathbb E \, Y_t < \infty \text{ and } \mathbb E \, u'(Y_t) < \infty
\label{a:y0}
$$

One relatively simple setting where all these restrictions are satisfied is
the IID and CRRA environment of {cite}`benhabib2015`.

### Optimality

Let the class of candidate consumption policies $\mathscr C$ be defined as in {doc}`ifp_egm`.

In {cite}`ma2020income` it is shown that, under the stated assumptions,

* any $\sigma \in \mathscr C$ satisfying the Euler equation is an
  optimal policy and
* exactly one such policy exists in $\mathscr C$.

In the present setting, the Euler equation takes the form

```{math}
:label: ifpa_euler

(u' \circ \sigma) (a, z) =
\max \left\{
           \beta \, \mathbb E_z \,\hat{R} \,
             (u' \circ \sigma)[\hat{R}(a - \sigma(a, z)) + \hat{Y}, \, \hat{Z}],
          \, u'(a)
       \right\}
```

(Intuition and derivation are similar to {doc}`ifp_egm`.)

We again solve the Euler equation using time iteration, iterating with a
Coleman--Reffett operator $K$ defined to match the Euler equation
{eq}`ifpa_euler`.

## Solution Algorithm

```{index} single: Optimal Savings; Computation
```

### A Time Iteration Operator

Our definition of the candidate class $\sigma \in \mathscr C$ of consumption
policies is the same as in {doc}`ifp_egm`.

For fixed $\sigma \in \mathscr C$ and $(a,z) \in \mathbf S$, the value
$K\sigma(a,z)$ of the function $K\sigma$ at $(a,z)$ is defined as the
$\xi \in (0,a]$ that solves

```{math}
:label: k_opr

u'(\xi) =
\max \left\{
          \beta \, \mathbb E_z \, \hat{R} \,
             (u' \circ \sigma)[\hat{R}(a - \xi) + \hat{Y}, \, \hat{Z}],
          \, u'(a)
       \right\}
```

The idea behind $K$ is that, as can be seen from the definitions,
$\sigma \in \mathscr C$ satisfies the Euler equation
if and only if $K\sigma(a, z) = \sigma(a, z)$ for all $(a, z) \in
\mathbf S$.

This means that fixed points of $K$ in $\mathscr C$ and optimal
consumption policies exactly coincide (see {cite}`ma2020income` for more details).

### Convergence Properties

As before, we pair $\mathscr C$ with the distance

$$
\rho(c,d)
:= \sup_{(a,z) \in \mathbf S}
          \left|
              \left(u' \circ c \right)(a,z) -
              \left(u' \circ d \right)(a,z)
          \right|,
$$

It can be shown that

1. $(\mathscr C, \rho)$ is a complete metric space,
1. there exists an integer $n$ such that $K^n$ is a contraction
   mapping on $(\mathscr C, \rho)$, and
1. The unique fixed point of $K$ in $\mathscr C$ is
   the unique optimal policy in $\mathscr C$.

We now have a clear path to successfully approximating the optimal policy:
choose some $\sigma \in \mathscr C$ and then iterate with $K$ until
convergence (as measured by the distance $\rho$).

### Using an Endogenous Grid

In the study of that model we found that it was possible to further
accelerate time iteration via the {doc}`endogenous grid method <os_egm>`.

We will use the same method here.

The methodology is the same as it was for the optimal growth model, with the
minor exception that we need to remember that consumption is not always
interior.

In particular, optimal consumption can be equal to assets when the level of
assets is low.

#### Finding Optimal Consumption

The endogenous grid method (EGM) calls for us to take a grid of *savings*
values $s_i$, where each such $s$ is interpreted as $s = a -
c$.

For the lowest grid point we take $s_0 = 0$.

For the corresponding $a_0, c_0$ pair we have $a_0 = c_0$.

This happens close to the origin, where assets are low and the household
consumes all that it can.

Although there are many solutions, the one we take is $a_0 = c_0 = 0$,
which pins down the policy at the origin, aiding interpolation.

For $s > 0$, we have, by definition, $c < a$, and hence
consumption is interior.

Hence the max component of {eq}`ifpa_euler` drops out, and we solve for

```{math}
:label: eqsifc2

c_i =
(u')^{-1}
\left\{
    \beta \, \mathbb E_z
    \hat R
    (u' \circ \sigma) \, [\hat R s_i + \hat Y, \, \hat Z]
\right\}
```

at each $s_i$.

#### Iterating

Once we have the pairs $\{s_i, c_i\}$, the endogenous asset grid is
obtained by $a_i = c_i + s_i$.

Also, we held $z \in \mathsf Z$ in the discussion above so we can pair
it with $a_i$.

An approximation of the policy $(a, z) \mapsto \sigma(a, z)$ can be
obtained by interpolating $\{a_i, c_i\}$ at each $z$.

In what follows, we use linear interpolation.


## Implementation

Here's the model as a `NamedTuple`.

```{code-cell} ipython3
class IFP(NamedTuple):
    """
    A NamedTuple that stores primitives for the income fluctuation
    problem, using JAX.
    """
    γ: float
    β: float
    P: jnp.ndarray
    a_r: float
    b_r: float
    a_y: float
    b_y: float
    s_grid: jnp.ndarray
    η_draws: jnp.ndarray
    ζ_draws: jnp.ndarray


def create_ifp(
        γ=1.5,                      # Utility parameter
        β=0.96,                     # Discount factor
        P=jnp.array([(0.9, 0.1),    # Default Markov chain for Z
                    (0.1, 0.9)]),
        a_r=0.16,                   # Volatility term in R shock
        b_r=0.0,                    # Mean shift R shock
        a_y=0.2,                    # Volatility term in Y shock
        b_y=0.5,                    # Mean shift Y shock
        shock_draw_size=100,        # For Monte Carlo
        grid_max=100,               # Exogenous grid max
        grid_size=100,              # Exogenous grid size
        seed=1234                   # Random seed
    ):
    """
    Create an instance of IFP with the given parameters.

    """
    # Test stability assuming {R_t} is IID and ln R ~ N(b_r, a_r)
    ER = np.exp(b_r + a_r**2 / 2)
    assert β * ER < 1, "Stability condition failed."

    # Generate random draws using JAX
    key = jax.random.PRNGKey(seed)
    subkey1, subkey2 = jax.random.split(key)
    η_draws = jax.random.normal(subkey1, (shock_draw_size,))
    ζ_draws = jax.random.normal(subkey2, (shock_draw_size,))
    s_grid = jnp.linspace(0, grid_max, grid_size)

    return IFP(
        γ, β, P, a_r, b_r, a_y, b_y, s_grid, η_draws, ζ_draws
    )


def u_prime(c, γ):
    """Marginal utility"""
    return c**(-γ)

def u_prime_inv(c, γ):
    """Inverse of marginal utility"""
    return c**(-1/γ)

def R(z, ζ, a_r, b_r):
    """Gross return on assets"""
    return jnp.exp(a_r * ζ + b_r)

def Y(z, η, a_y, b_y):
    """Labor income"""
    return jnp.exp(a_y * η + (z * b_y))
```

Here's the Coleman-Reffett operator using JAX:

```{code-cell} ipython3
def K(
        a_in: jnp.array,   # a_in[i, z] is an asset grid
        c_in: jnp.array,   # c_in[i, z] = consumption at a_in[i, z]
        ifp: IFP
    ):
    """
    The Coleman--Reffett operator for the income fluctuation problem,
    using the endogenous grid method with JAX.

    """

    # Extract parameters from ifp
    γ, β, P, a_r, b_r, a_y, b_y, s_grid, η_draws, ζ_draws = ifp
    n = len(P)

    def compute_expectation(s, z):
        def inner_expectation(z_hat):
            def compute_term(η, ζ):
                R_hat = R(z_hat, ζ, a_r, b_r)
                Y_hat = Y(z_hat, η, a_y, b_y)
                a_val = R_hat * s + Y_hat
                # Interpolate consumption
                c_interp = jnp.interp(a_val, a_in[:, z_hat], c_in[:, z_hat])
                mu = u_prime(c_interp, γ)
                return R_hat * mu
            # Vectorize over all shock combinations
            η_grid, ζ_grid = jnp.meshgrid(η_draws, ζ_draws, indexing='ij')
            terms = vmap(vmap(compute_term))(η_grid, ζ_grid)
            return P[z, z_hat] * jnp.mean(terms)
        # Sum over z_hat states
        Ez = jnp.sum(vmap(inner_expectation)(jnp.arange(n)))
        return u_prime_inv(β * Ez, γ)

    # Vectorize over s_grid and z
    compute_exp_v1 = vmap(compute_expectation, in_axes=(None, 0))
    compute_exp_v2 = vmap(compute_exp_v1,      in_axes=(0, None))
    c_out = compute_exp_v2(s_grid, jnp.arange(n))
    # Calculate endogenous asset grid
    a_out = s_grid[:, None] + c_out
    # Fix consumption-asset pair at (0, 0) 
    c_out = c_out.at[0, :].set(0)
    a_out = a_out.at[0, :].set(0)

    return a_out, c_out
```

The next function solves for an approximation of the optimal consumption policy
via time iteration using JAX:

```{code-cell} ipython3
@jax.jit
def solve_model(
        ifp: IFP,
        c_init: jnp.ndarray,  # Initial guess of σ on grid endogenous grid
        a_init: jnp.ndarray,  # Initial endogenous grid
        tol: float = 1e-5,
        max_iter: int = 1000
    ) -> jnp.ndarray:
    " Solve the model using time iteration with EGM. "

    def condition(loop_state):
        c_in, a_in, i, error = loop_state
        return (error > tol) & (i < max_iter)

    def body(loop_state):
        c_in, a_in, i, error = loop_state
        c_out, a_out = K(c_in, a_in, ifp)
        error = jnp.max(jnp.abs(c_out - c_in))
        i += 1
        return c_out, a_out, i, error

    i, error = 0, tol + 1
    initial_state = (c_init, a_init, i, error)
    final_loop_state = jax.lax.while_loop(condition, body, initial_state)
    c_out, a_out, i, error = final_loop_state

    return c_out, a_out
```

Now we can create an instance and solve the model using JAX:

```{code-cell} ipython3
ifp = create_ifp()
```

Set up the initial condition:

```{code-cell} ipython3
# Initial guess of σ = consume all assets
k = len(ifp.s_grid)
n = len(ifp.P)
σ_init = jnp.empty((k, n))
for z in range(n):
    σ_init = σ_init.at[:, z].set(ifp.s_grid)
a_init = σ_init.copy()
```

Let's generate an approximation solution with JAX:

```{code-cell} ipython3
a_star, σ_star = solve_model(ifp, a_init, σ_init)
```

Let's try it again with a timer.

```{code-cell} python3
with qe.Timer(precision=8):
    a_star, σ_star = solve_model(ifp, a_init, σ_init)
    a_star.block_until_ready()
```

## Simulation

Let's return to the default model and study the stationary distribution of assets.

Our plan is to run a large number of households forward for $T$ periods and then
histogram the cross-sectional distribution of assets.

Set `num_households=50_000, T=500`.

First we write a function to run a single household forward in time and record
the final value of assets.

The function takes a solution pair `c_vec`  and `a_vec`, understanding them
as representing an optimal policy associated with a given model `ifp`

```{code-cell} ipython3
def simulate_household(
        key, a_0, z_idx_0, c_vec, a_vec, ifp, T
    ):
    """
    Simulates a single household for T periods to approximate the stationary
    distribution of assets.

    - key is the state of the random number generator
    - ifp is an instance of IFP
    - c_vec, a_vec are the optimal consumption policy, endogenous grid for ifp

    """
    # Extract parameters from ifp
    γ, β, P, a_r, b_r, a_y, b_y, s_grid, η_draws, ζ_draws = ifp
    n_z = len(P)

    # Create interpolation function for consumption policy
    σ = lambda a, z_idx: jnp.interp(a, a_vec[:, z_idx], c_vec[:, z_idx])

    # Simulate forward T periods
    def update(t, state):
        a, z_idx = state
        # Draw next shock z' from P[z, z']
        current_key = jax.random.fold_in(key, 3*t)
        z_next_idx = jax.random.choice(current_key, n_z, p=P[z_idx]).astype(jnp.int32)
        # Draw η shock for income
        η_key = jax.random.fold_in(key, 3*t + 1)
        η = jax.random.normal(η_key)
        # Draw ζ shock for return
        ζ_key = jax.random.fold_in(key, 3*t + 2)
        ζ = jax.random.normal(ζ_key)
        # Compute stochastic return
        R_next = R(z_next_idx, ζ, a_r, b_r)
        # Compute income
        Y_next = Y(z_next_idx, η, a_y, b_y)
        # Update assets: a' = R' * (a - c) + Y'
        a_next = R_next * (a - σ(a, z_idx)) + Y_next
        # Return updated state
        return a_next, z_next_idx

    initial_state = a_0, z_idx_0
    final_state = jax.lax.fori_loop(0, T, update, initial_state)
    a_final, _ = final_state
    return a_final
```

Now we write a function to simulate many households in parallel.

```{code-cell} ipython3
@partial(jax.jit, static_argnums=(3, 4, 5))
def compute_asset_stationary(
        c_vec, a_vec, ifp, num_households=50_000, T=500, seed=1234
    ):
    """
    Simulates num_households households for T periods to approximate
    the stationary distribution of assets.

    Returns the final cross-section of asset holdings.

    - ifp is an instance of IFP
    - c_vec, a_vec are the optimal consumption policy and endogenous grid.

    """
    # Extract parameters from ifp
    γ, β, P, a_r, b_r, a_y, b_y, s_grid, η_draws, ζ_draws = ifp

    # Start with assets = savings_grid_max / 2
    a_0_vector = jnp.full(num_households, s_grid[-1] / 2)
    # Initialize the exogenous state of each household
    z_idx_0_vector = jnp.zeros(num_households).astype(jnp.int32)

    # Vectorize over many households
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, num_households)
    # Vectorize simulate_household in (key, a_0, z_idx_0)
    sim_all_households = jax.vmap(
        simulate_household, in_axes=(0, 0, 0, None, None, None, None)
    )
    assets = sim_all_households(keys, a_0_vector, z_idx_0_vector, c_vec, a_vec, ifp, T)

    return jnp.array(assets)
```

We'll need some inequality measures for visualization, so let's define them first:

```{code-cell} ipython3
def gini_coefficient(x):
    """
    Compute the Gini coefficient for array x.

    """
    x = jnp.asarray(x)
    n = len(x)
    x_sorted = jnp.sort(x)
    # Compute Gini coefficient
    cumsum = jnp.cumsum(x_sorted)
    a = (2 * jnp.sum((jnp.arange(1, n+1)) * x_sorted)) / (n * cumsum[-1])
    return a - (n + 1) / n


def top_share(
        x: jnp.array,   # array of wealth values
        p: float=0.01   # fraction of top households (default 0.01 for top 1%)
    ):
    """
    Compute the share of total wealth held by the top p fraction of households.

    """
    x = jnp.asarray(x)
    x_sorted = jnp.sort(x)
    # Number of households in top p%
    n_top = int(jnp.ceil(len(x) * p))
    # Wealth held by top p%
    wealth_top = jnp.sum(x_sorted[-n_top:])
    # Total wealth
    wealth_total = jnp.sum(x_sorted)
    return wealth_top / wealth_total
```

Now we call the function, generate the asset distribution and visualize it:

```{code-cell} ipython3
ifp = create_ifp()
# Extract parameters for initialization
s_grid = ifp.s_grid
n_z = len(ifp.P)
a_init = s_grid[:, None] * jnp.ones(n_z)
c_init = a_init
a_vec, c_vec = solve_model(ifp, a_init, c_init)
assets = compute_asset_stationary(c_vec, a_vec, ifp, num_households=200_000)

# Compute Gini coefficient for the plot
gini_plot = gini_coefficient(assets)

# Plot histogram of log wealth
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(jnp.log(assets), bins=40, alpha=0.5, density=True)
ax.set(xlabel='log assets', ylabel='density', title="Wealth Distribution")
plt.tight_layout()
plt.show()
```

The histogram shows the distribution of log wealth. 

Bearing in mind that we are looking at log values, the histogram suggests 
a long right tail of the distribution.

Below we examine this in more detail.



## Wealth Inequality

Lets' look at wealth inequality by computing some standard measures of this phenomenon.

We will also examine how inequality varies with the interest rate.


### Measuring Inequality

Let's print the Gini coefficient and the top 1% wealth share from our simulation:

```{code-cell} ipython3
gini = gini_coefficient(assets)
top1 = top_share(assets, p=0.01)

print(f"Gini coefficient: {gini:.4f}")
print(f"Top 1% wealth share: {top1:.4f}")
```

Recent numbers suggest that

* the Gini coefficient for wealth in the US is around 0.8
* the top 1% wealth share is over 0.3

Our model with stochastic returns generates a Gini coefficient close to the
empirical value, demonstrating that capital income risk is an important factor
in wealth inequality.

The top 1% wealth share is, however, too large.

Our model needs proper calibration and additional work -- we set these tasks aside for now.

## Exercises

```{exercise}
:label: ifp_advanced_ex1

Plot how the Gini coefficient varies with the volatility of returns on assets.

Specifically, compute the Gini coefficient for values of `a_r` ranging from 0.10 to 0.16 (use at least 5 different values) and plot the results.

What does this tell you about the relationship between capital income risk and wealth inequality?

```

```{solution-start} ifp_advanced_ex1
:class: dropdown
```

We loop over different values of `a_r`, solve the model for each, simulate the wealth distribution, and compute the Gini coefficient.

```{code-cell} ipython3
# Range of a_r values to explore
a_r_vals = np.linspace(0.10, 0.16, 5)
gini_vals = []

print("Computing Gini coefficients for different return volatilities...\n")

for a_r in a_r_vals:
    print(f"a_r = {a_r:.3f}...", end=" ")

    # Create model with this a_r value
    ifp_temp = create_ifp(a_r=a_r, grid_max=100)

    # Solve the model
    s_grid_temp = ifp_temp.s_grid
    n_z_temp = len(ifp_temp.P)
    a_init_temp = s_grid_temp[:, None] * jnp.ones(n_z_temp)
    c_init_temp = a_init_temp
    a_vec_temp, c_vec_temp = solve_model(
        ifp_temp, a_init_temp, c_init_temp
    )

    # Simulate households
    assets_temp = compute_asset_stationary(
        c_vec_temp, a_vec_temp, ifp_temp, num_households=200_000
    )

    # Compute Gini coefficient
    gini_temp = gini_coefficient(assets_temp)
    gini_vals.append(gini_temp)
    print(f"Gini = {gini_temp:.4f}")

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(a_r_vals, gini_vals, 'o-', linewidth=2, markersize=8)
ax.set(xlabel='Return volatility (a_r)',
       ylabel='Gini coefficient',
       title='Wealth Inequality vs Return Volatility')
ax.axhline(y=0.8, color='k', linestyle='--', linewidth=1,
           label='Empirical US Gini (~0.8)')
ax.legend()
plt.tight_layout()
plt.show()
```

The plot shows that wealth inequality (measured by the Gini coefficient) increases with return volatility.

This demonstrates that capital income risk is a key driver of wealth inequality.

When returns are more volatile, lucky households who experience sequences of
high returns accumulate substantially more wealth than unlucky households,
leading to greater inequality in the wealth distribution.


```{solution-end}
```

```{exercise}
:label: ifp_advanced_ex2

Plot how the Gini coefficient varies with the volatility of labor income.

Specifically, compute the Gini coefficient for values of `a_y` ranging from
0.125 to 0.20 and plot the results. Set `a_r=0.10` for this exercise.

What does this tell you about the relationship between labor income risk and
wealth inequality? Can we achieve the same rise in inequality by varying labor
income volatility as we can by varying return volatility?

```

```{solution-start} ifp_advanced_ex2
:class: dropdown
```

We loop over different values of `a_y`, solve the model for each, simulate the wealth distribution, and compute the Gini coefficient.

```{code-cell} ipython3
# Range of a_y values to explore
a_y_vals = np.linspace(0.125, 0.20, 5)
gini_vals_y = []

print("Computing Gini coefficients for different labor income volatilities...\n")

for a_y in a_y_vals:
    print(f"a_y = {a_y:.3f}...", end=" ")

    # Create model with this a_y value and a_r=0.10
    ifp_temp = create_ifp(a_y=a_y, a_r=0.10, grid_max=100)

    # Solve the model
    s_grid_temp = ifp_temp.s_grid
    n_z_temp = len(ifp_temp.P)
    a_init_temp = s_grid_temp[:, None] * jnp.ones(n_z_temp)
    c_init_temp = a_init_temp
    a_vec_temp, c_vec_temp = solve_model(
        ifp_temp, a_init_temp, c_init_temp
    )

    # Simulate households
    assets_temp = compute_asset_stationary(
        c_vec_temp, a_vec_temp, ifp_temp, num_households=200_000
    )

    # Compute Gini coefficient
    gini_temp = gini_coefficient(assets_temp)
    gini_vals_y.append(gini_temp)
    print(f"Gini = {gini_temp:.4f}")

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(a_y_vals, gini_vals_y, 'o-', linewidth=2, markersize=8, color='green')
ax.set(xlabel='Labor income volatility (a_y)',
       ylabel='Gini coefficient',
       title='Wealth Inequality vs Labor Income Volatility')
ax.axhline(y=0.8, color='k', linestyle='--', linewidth=1,
           label='Empirical US Gini (~0.8)')
ax.legend()
plt.tight_layout()
plt.show()
```

The plot shows that wealth inequality increases with labor income volatility, but the effect is much weaker than the effect of return volatility.

Comparing the two exercises:

- When return volatility (`a_r`) varies from 0.10 to 0.16, the Gini coefficient rises dramatically from around 0.20 to 0.79
- When labor income volatility (`a_y`) varies from 0.125 to 0.20, a similar amount in percentage terms, the Gini coefficient increases but by a much smaller amount

This suggests that capital income risk is a more important driver of wealth inequality than labor income risk.

The intuition is that wealth accumulation compounds over time: households who
experience favorable returns on their assets can reinvest those returns, leading
to exponential growth.

In contrast, labor income shocks, while they affect current consumption and
savings, do not have the same compounding effect on wealth accumulation.


```{solution-end}
```
