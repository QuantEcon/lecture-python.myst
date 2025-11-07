# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ```{raw} jupyter
# <div id="qe-notebook-header" align="right" style="text-align:right;">
#         <a href="https://quantecon.org/" title="quantecon.org">
#                 <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
#         </a>
# </div>
# ```
#
# # {index}`Cake Eating V: The Endogenous Grid Method <single: Cake Eating V: The Endogenous Grid Method>`
#
# ```{contents} Contents
# :depth: 2
# ```
#
#
# ## Overview
#
# Previously, we solved the stochastic cake eating problem using
#
# 1. {doc}`value function iteration <cake_eating_stochastic>`
# 1. {doc}`Euler equation based time iteration <cake_eating_time_iter>`
#
# We found time iteration to be significantly more accurate and efficient.
#
# In this lecture, we'll look at a clever twist on time iteration called the **endogenous grid method** (EGM).
#
# EGM is a numerical method for implementing policy iteration invented by [Chris Carroll](https://econ.jhu.edu/directory/christopher-carroll/).
#
# The original reference is {cite}`Carroll2006`.
#
# Let's start with some standard imports:

# %%
import matplotlib.pyplot as plt
import numpy as np


# %% [markdown]
# ## Key Idea
#
# Let's start by reminding ourselves of the theory and then see how the numerics fit in.
#
# ### Theory
#
# Take the model set out in {doc}`Cake Eating IV <cake_eating_time_iter>`, following the same terminology and notation.
#
# The Euler equation is
#
# ```{math}
# :label: egm_euler
#
# (u'\circ \sigma^*)(x)
# = \beta \int (u'\circ \sigma^*)(f(x - \sigma^*(x)) z) f'(x - \sigma^*(x)) z \phi(dz)
# ```
#
# As we saw, the Coleman-Reffett operator is a nonlinear operator $K$ engineered so that $\sigma^*$ is a fixed point of $K$.
#
# It takes as its argument a continuous strictly increasing consumption policy $\sigma \in \Sigma$.
#
# It returns a new function $K \sigma$,  where $(K \sigma)(x)$ is the $c \in (0, \infty)$ that solves
#
# ```{math}
# :label: egm_coledef
#
# u'(c)
# = \beta \int (u' \circ \sigma) (f(x - c) z ) f'(x - c) z \phi(dz)
# ```
#
# ### Exogenous Grid
#
# As discussed in {doc}`Cake Eating IV <cake_eating_time_iter>`, to implement the method on a computer, we need a numerical approximation.
#
# In particular, we represent a policy function by a set of values on a finite grid.
#
# The function itself is reconstructed from this representation when necessary, using interpolation or some other method.
#
# {doc}`Previously <cake_eating_time_iter>`, to obtain a finite representation of an updated consumption policy, we
#
# * fixed a grid of income points $\{x_i\}$
# * calculated the consumption value $c_i$ corresponding to each
#   $x_i$ using {eq}`egm_coledef` and a root-finding routine
#
# Each $c_i$ is then interpreted as the value of the function $K \sigma$ at $x_i$.
#
# Thus, with the points $\{x_i, c_i\}$ in hand, we can reconstruct $K \sigma$ via approximation.
#
# Iteration then continues...
#
# ### Endogenous Grid
#
# The method discussed above requires a root-finding routine to find the
# $c_i$ corresponding to a given income value $x_i$.
#
# Root-finding is costly because it typically involves a significant number of
# function evaluations.
#
# As pointed out by Carroll {cite}`Carroll2006`, we can avoid this if
# $x_i$ is chosen endogenously.
#
# The only assumption required is that $u'$ is invertible on $(0, \infty)$.
#
# Let $(u')^{-1}$ be the inverse function of $u'$.
#
# The idea is this:
#
# * First, we fix an *exogenous* grid $\{k_i\}$ for capital ($k = x - c$).
# * Then we obtain  $c_i$ via
#
# ```{math}
# :label: egm_getc
#
# c_i =
# (u')^{-1}
# \left\{
#     \beta \int (u' \circ \sigma) (f(k_i) z ) \, f'(k_i) \, z \, \phi(dz)
# \right\}
# ```
#
# * Finally, for each $c_i$ we set $x_i = c_i + k_i$.
#
# It is clear that each $(x_i, c_i)$ pair constructed in this manner satisfies {eq}`egm_coledef`.
#
# With the points $\{x_i, c_i\}$ in hand, we can reconstruct $K \sigma$ via approximation as before.
#
# The name EGM comes from the fact that the grid $\{x_i\}$ is  determined **endogenously**.
#
# ## Implementation
#
# As in {doc}`Cake Eating IV <cake_eating_time_iter>`, we will start with a simple setting
# where
#
# * $u(c) = \ln c$,
# * production is Cobb-Douglas, and
# * the shocks are lognormal.
#
# This will allow us to make comparisons with the analytical solutions

# %%
def v_star(x, α, β, μ):
    """
    True value function
    """
    c1 = np.log(1 - α * β) / (1 - β)
    c2 = (μ + α * np.log(α * β)) / (1 - α)
    c3 = 1 / (1 - β)
    c4 = 1 / (1 - α * β)
    return c1 + c2 * (c3 - c4) + c4 * np.log(x)

def σ_star(x, α, β):
    """
    True optimal policy
    """
    return (1 - α * β) * x


# %% [markdown]
# We reuse the `Model` structure from {doc}`Cake Eating IV <cake_eating_time_iter>`.

# %%
from typing import NamedTuple, Callable

class Model(NamedTuple):
    u: Callable        # utility function
    f: Callable        # production function
    β: float           # discount factor
    μ: float           # shock location parameter
    s: float           # shock scale parameter
    grid: np.ndarray   # state grid
    shocks: np.ndarray # shock draws
    α: float = 0.4     # production function parameter
    u_prime: Callable = None        # derivative of utility
    f_prime: Callable = None        # derivative of production
    u_prime_inv: Callable = None    # inverse of u_prime


def create_model(u: Callable,
                 f: Callable,
                 β: float = 0.96,
                 μ: float = 0.0,
                 s: float = 0.1,
                 grid_max: float = 4.0,
                 grid_size: int = 120,
                 shock_size: int = 250,
                 seed: int = 1234,
                 α: float = 0.4,
                 u_prime: Callable = None,
                 f_prime: Callable = None,
                 u_prime_inv: Callable = None) -> Model:
    """
    Creates an instance of the cake eating model.
    """
    # Set up grid
    grid = np.linspace(1e-4, grid_max, grid_size)

    # Store shocks (with a seed, so results are reproducible)
    np.random.seed(seed)
    shocks = np.exp(μ + s * np.random.randn(shock_size))

    return Model(u=u, f=f, β=β, μ=μ, s=s, grid=grid, shocks=shocks,
                 α=α, u_prime=u_prime, f_prime=f_prime, u_prime_inv=u_prime_inv)


# %% [markdown]
# ### The Operator
#
# Here's an implementation of $K$ using EGM as described above.

# %%
def K(σ_array: np.ndarray, model: Model) -> np.ndarray:
    """
    The Coleman-Reffett operator using EGM

    """

    # Simplify names
    f, β = model.f, model.β
    f_prime, u_prime = model.f_prime, model.u_prime
    u_prime_inv = model.u_prime_inv
    grid, shocks = model.grid, model.shocks

    # Determine endogenous grid
    x = grid + σ_array  # x_i = k_i + c_i

    # Linear interpolation of policy using endogenous grid
    σ = lambda x_val: np.interp(x_val, x, σ_array)

    # Allocate memory for new consumption array
    c = np.empty_like(grid)

    # Solve for updated consumption value
    for i, k in enumerate(grid):
        vals = u_prime(σ(f(k) * shocks)) * f_prime(k) * shocks
        c[i] = u_prime_inv(β * np.mean(vals))

    return c


# %% [markdown]
# Note the lack of any root-finding algorithm.
#
# ### Testing
#
# First we create an instance.

# %%
# Define utility and production functions with derivatives
α = 0.4
u = lambda c: np.log(c)
u_prime = lambda c: 1 / c
u_prime_inv = lambda x: 1 / x
f = lambda k: k**α
f_prime = lambda k: α * k**(α - 1)

model = create_model(u=u, f=f, α=α, u_prime=u_prime,
                     f_prime=f_prime, u_prime_inv=u_prime_inv)
grid = model.grid


# %% [markdown]
# Here's our solver routine:

# %%
def solve_model_time_iter(model: Model,
                          σ_init: np.ndarray,
                          tol: float = 1e-5,
                          max_iter: int = 1000,
                          verbose: bool = True) -> np.ndarray:
    """
    Solve the model using time iteration with EGM.
    """
    σ = σ_init
    error = tol + 1
    i = 0

    while error > tol and i < max_iter:
        σ_new = K(σ, model)
        error = np.max(np.abs(σ_new - σ))
        σ = σ_new
        i += 1
        if verbose:
            print(f"Iteration {i}, error = {error}")

    if i == max_iter:
        print("Warning: maximum iterations reached")

    return σ


# %% [markdown]
# Let's call it:

# %%
σ_init = np.copy(grid)
σ = solve_model_time_iter(model, σ_init)

# %% [markdown]
# Here is a plot of the resulting policy, compared with the true policy:

# %%
x = grid + σ  # x_i = k_i + c_i

fig, ax = plt.subplots()

ax.plot(x, σ, lw=2,
        alpha=0.8, label='approximate policy function')

ax.plot(x, σ_star(x, model.α, model.β), 'k--',
        lw=2, alpha=0.8, label='true policy function')

ax.legend()
plt.show()

# %% [markdown]
# The maximal absolute deviation between the two policies is

# %%
np.max(np.abs(σ - σ_star(x, model.α, model.β)))

# %% [markdown]
# How long does it take to converge?

# %%
# %%timeit -n 3 -r 1
σ = solve_model_time_iter(model, σ_init, verbose=False)

# %% [markdown]
# Relative to time iteration, which was already found to be highly efficient, EGM
# has managed to shave off still more run time without compromising accuracy.
#
# This is due to the lack of a numerical root-finding step.
#
# We can now solve the stochastic cake eating problem at given parameters extremely fast.
