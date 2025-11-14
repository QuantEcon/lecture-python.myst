# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
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
# # {index}`The Income Fluctuation Problem I: Basic Model <single: The Income Fluctuation Problem I: Basic Model>`
#
# ```{contents} Contents
# :depth: 2
# ```
#
# In addition to what's in Anaconda, this lecture will need the following libraries:

# %% tags=["hide-output"]
# !pip install quantecon

# %% [markdown]
# ## Overview
#
# In this lecture, we study an optimal savings problem for an infinitely lived consumer---the "common ancestor" described in {cite}`Ljungqvist2012`, section 1.3.
#
# This is an essential sub-problem for many representative macroeconomic models
#
# * {cite}`Aiyagari1994`
# * {cite}`Huggett1993`
# * etc.
#
# It is related to the decision problem in the {doc}`cake eating model <cake_eating_stochastic>` and yet differs in important ways.
#
# For example, the choice problem for the agent includes an additive income term that leads to an occasionally binding constraint.
#
# Moreover, shocks affecting the budget constraint are correlated, forcing us to
# track an extra state variable.
#
# To solve the model we will use the endogenous grid method, which we found to be {doc}`fast and accurate <cake_eating_egm_jax>` in our investigation of cake eating.
#
#
# We'll need the following imports:

# %%
import matplotlib.pyplot as plt
import numpy as np
from quantecon import MarkovChain
import jax
import jax.numpy as jnp
from typing import NamedTuple


# %% [markdown]
# ### References
#
# We skip most technical details but they can be found in {cite}`ma2020income`.
#
# Other references include {cite}`Deaton1991`, {cite}`DenHaan2010`,
# {cite}`Kuhn2013`, {cite}`Rabault2002`,  {cite}`Reiter2009`  and
# {cite}`SchechtmanEscudero1977`.
#
#
# ## The Optimal Savings Problem
#
# ```{index} single: Optimal Savings; Problem
# ```
#
# Let's write down the model and then discuss how to solve it.
#
# ### Set-Up
#
# Consider a household that chooses a state-contingent consumption plan $\{c_t\}_{t \geq 0}$ to maximize
#
# $$
# \mathbb{E} \, \sum_{t=0}^{\infty} \beta^t u(c_t)
# $$
#
# subject to
#
# ```{math}
# :label: eqst
#
# a_{t+1} = R (a_t - c_t) + Y_{t+1}
# \quad c_t \geq 0,
# \quad a_t \geq 0
# \quad t = 0, 1, \ldots
# ```
#
# Here
#
# * $\beta \in (0,1)$ is the discount factor
# * $a_t$ is asset holdings at time $t$, with borrowing constraint $a_t \geq 0$
# * $c_t$ is consumption
# * $Y_t$ is non-capital income (wages, unemployment compensation, etc.)
# * $R := 1 + r$, where $r > 0$ is the interest rate on savings
#
# The timing here is as follows:
#
# 1. At the start of period $t$, the household observes current asset holdings $a_t$.
# 1. The household chooses current consumption $c_t$.
# 1. Savings $(a_t - c_t)$ earn interest at rate $r$.
# 1. Labor income $Y_{t+1}$ is realized and time shifts to $t+1$.
#
# Non-capital income $Y_t$ is given by $Y_t = y(Z_t)$, where
#
# * $\{Z_t\}$ is an exogenous state process and
# * $y$ is a given function taking values in $\mathbb{R}_+$.
#
# As is common in the literature, we take $\{Z_t\}$ to be a finite state
# Markov chain taking values in $\mathsf Z$ with Markov matrix $\Pi$.
#
# We further assume that
#
# 1. $\beta R < 1$
# 1. $u$ is smooth, strictly increasing and strictly concave with $\lim_{c \to 0} u'(c) = \infty$ and $\lim_{c \to \infty} u'(c) = 0$
# 1. $y(z) = \exp(z)$ 
#
# The asset space is $\mathbb R_+$ and the state is the pair $(a,z) \in \mathsf S := \mathbb R_+ \times \mathsf Z$.
#
# A **feasible consumption path** from $(a,z) \in \mathsf S$ is a consumption
# sequence $\{c_t\}$ such that $\{c_t\}$ and its induced asset path $\{a_t\}$ satisfy
#
# 1. $(a_0, z_0) = (a, z)$
# 1. the feasibility constraints in {eq}`eqst`, and
# 1. measurability, which means that $c_t$ is a function of random
#    outcomes up to date $t$ but not after.
#
# The meaning of the third point is just that consumption at time $t$
# cannot be a function of outcomes are yet to be observed.
#
# In fact, for this problem, consumption can be chosen optimally by taking it to
# be contingent only on the current state.
#
# Optimality is defined below.
#
#
#
# ### Value Function and Euler Equation
#
# The **value function** $V \colon \mathsf S \to \mathbb{R}$ is defined by
#
# ```{math}
# :label: eqvf
#
# V(a, z) := \max \, \mathbb{E}
# \left\{
# \sum_{t=0}^{\infty} \beta^t u(c_t)
# \right\}
# ```
#
# where the maximization is overall feasible consumption paths from $(a,z)$.
#
# An **optimal consumption path** from $(a,z)$ is a feasible consumption path from $(a,z)$ that attains the supremum in {eq}`eqvf`.
#
# To pin down such paths we can use a version of the Euler equation, which in the present setting is
#
# ```{math}
# :label: ee00
#
#     u' (c_t) \geq \beta R \,  \mathbb{E}_t  u'(c_{t+1})
# ```
#
# and
#
# ```{math}
# :label: ee01
#
#     c_t < a_t
#     \; \implies \;
#     u' (c_t) = \beta R \,  \mathbb{E}_t  u'(c_{t+1})
# ```
#
# When $c_t = a_t$ we obviously have $u'(c_t) = u'(a_t)$,
#
# When $c_t$ hits the upper bound $a_t$, the
# strict inequality $u' (c_t) > \beta R \,  \mathbb{E}_t  u'(c_{t+1})$
# can occur because $c_t$ cannot increase sufficiently to attain equality.
#
# (The lower boundary case $c_t = 0$ never arises at the optimum because
# $u'(0) = \infty$.)
#
#
# ### Optimality Results
#
# As shown in {cite}`ma2020income`,
#
# 1. For each $(a,z) \in \mathsf S$, a unique optimal consumption path from $(a,z)$ exists
# 1. This path is the unique feasible path from $(a,z)$ satisfying the
#    Euler equations {eq}`ee00`-{eq}`ee01` and the transversality condition
#
# ```{math}
# :label: eqtv
#
# \lim_{t \to \infty} \beta^t \, \mathbb{E} \, [ u'(c_t) a_{t+1} ] = 0
# ```
#
# Moreover, there exists an **optimal consumption policy**
# $\sigma^* \colon \mathsf S \to \mathbb R_+$ such that the path
# from $(a,z)$ generated by
#
# $$
#     (a_0, z_0) = (a, z),
#     \quad
#     c_t = \sigma^*(a_t, Z_t)
#     \quad \text{and} \quad
#     a_{t+1} = R (a_t - c_t) + Y_{t+1}
# $$
#
# satisfies both the Euler equations {eq}`ee00`-{eq}`ee01` and {eq}`eqtv`, and hence is the unique optimal
# path from $(a,z)$.
#
# Thus, to solve the optimization problem, we need to compute the policy $\sigma^*$.
#
# (ifp_computation)=
# ## Computation
#
# ```{index} single: Optimal Savings; Computation
# ```
#
# We solve for the optimal consumption policy using time iteration and the
# endogenous grid method.
#
# Readers unfamiliar with the endogenous grid method should review the discussion
# in {doc}`cake_eating_egm`.
#
# ### Solution Method
#
# We rewrite {eq}`ee01` to make it a statement about functions rather than
# random variables:
#
#
# ```{math}
# :label: eqeul1
#
#     (u' \circ \sigma)  (a, z)
#     = \beta R \, \sum_{z'}  (u' \circ \sigma)
#             [R (a - \sigma(a, z)) + y(z'), \, z'] \Pi(z, z')
# ```
#
# Here
#
# * $(u' \circ \sigma)(s) := u'(\sigma(s))$,
# * primes indicate next period states (as well as derivatives), and
# * $\sigma$ is the unknown function.
#
# We aim to find a fixed point $\sigma$ of {eq}`eqeul1`.
#
# To do so we use the EGM.
#
# We begin with an exogenous grid $G = \{s_0, \ldots, s_{m-1}\}$ with $s_0 > 0$, where each $s_i$ represents savings.
#
# The relationship between current assets $a$, consumption $c$, and savings $s$ is
#
# $$
#     a = c + s
# $$
#
# and next period assets are given by
#
# $$
#     a' = R s + y(z').
# $$
#
# Fix a current guess of the policy function $\sigma$.
#
# For each savings level $s_i$ and current state $z_j$, we set
#
# $$
#     c_{ij} = (u')^{-1}
#         \left[
#             \beta R \, \sum_{z'}
#             u' [ \sigma(R s_i + y(z'), z') ] \Pi(z_j, z')
#         \right]
# $$
#
# and then obtain the endogenous grid of current assets via
#
# $$
#     a^e_{ij} = c_{ij} + s_i.
# $$
#
# Our next guess policy function, which we write as $K\sigma$, is the linear interpolation of
# $(a^e_{ij}, c_{ij})$ over $i$, for each $j$.
#
# (The number of one dimensional linear interpolations is equal to `len(z_grid)`.)
#
# For $a < a^e_{i0}$ (i.e., below the minimum endogenous grid point), the household consumes everything, so we set $(K \sigma)(a, z_j) = a$.
#
#
#
# ## Implementation
#
# ```{index} single: Optimal Savings; Programming Implementation
# ```
#
# We use the CRRA utility specification
#
# $$
#     u(c) = \frac{c^{1 - \gamma}} {1 - \gamma}
# $$
#
#
# ### Set Up
#
# Here we build a class called `IFP` that stores the model primitives.

# %%
class IFP(NamedTuple):
    R: float                  # Gross interest rate R = 1 + r
    β: float                  # Discount factor
    γ: float                  # Preference parameter
    Π: jnp.ndarray            # Markov matrix for exogenous shock
    z_grid: jnp.ndarray       # Markov state values for Z_t
    asset_grid: jnp.ndarray   # Exogenous asset grid


def create_ifp(r=0.01,
               β=0.98,
               γ=1.5,
               Π=((0.6, 0.4),
                  (0.05, 0.95)),
               z_grid=(0.0, 0.2),
               asset_grid_max=40,
               asset_grid_size=50):

    asset_grid = jnp.linspace(0, asset_grid_max, asset_grid_size)
    Π, z_grid = jnp.array(Π), jnp.array(z_grid)
    R = 1 + r
    assert R * β < 1, "Stability condition violated."
    return IFP(R=R, β=β, γ=γ, Π=Π, z_grid=z_grid, asset_grid=asset_grid)

# Set y(z) = exp(z)
y = jnp.exp

# %% [markdown]
# The exogenous state process $\{Z_t\}$ defaults to a two-state Markov chain
# with transition matrix $\Pi$.
#
# We define utility globally:

# %%
# Define utility function derivatives
u_prime = lambda c, γ: c**(-γ)
u_prime_inv = lambda c, γ: c**(-1/γ)


# %% [markdown]
# ### Solver
#
# Here is the operator $K$ that transforms current guess $\sigma$ into next period
# guess $K\sigma$.
#
# We understand $\sigma$ is an array of shape $(n_a, n_z)$, where $n_a$ and $n_z$
# are the respective grid sizes.
#
# The value `σ[i,j]` corresponds to $\sigma(a_i, z_j)$, where $a_i$ is a point on the asset grid.

# %%
def K(σ: jnp.ndarray, ifp: IFP) -> jnp.ndarray:
    """
    The Coleman-Reffett operator for the IFP model using the
    Endogenous Grid Method.

    This operator implements one iteration of the EGM algorithm to
    update the consumption policy function.

    Algorithm
    ---------
    The EGM works with a savings grid:
    1. Use exogenous savings grid s_i
    2. For each (s_i, z_j), compute next period assets a' = R*s_i + y(z')
    3. Given σ(a', z'), compute current consumption c that
       satisfies Euler equation
    4. Compute the endogenous current asset level a^e = c + s
    5. Interpolate back to asset grid to get σ_new(a, z)

    """
    R, β, γ, Π, z_grid, asset_grid = ifp
    n_a = len(asset_grid)
    n_z = len(z_grid)

    # Create savings grid (exogenous grid for EGM)
    # We use the asset grid as the savings grid
    savings_grid = asset_grid

    def compute_c_for_fixed_income_state(j):
        """
        Compute updated consumption policy for income state z_j.

        The savings_grid represents s (savings), where a' = R*s + y(z').

        """

        # For each savings level s_i, compute expected marginal utility
        # We need to evaluate σ at next period assets a' = R*s + y(z')

        # Compute next period assets for all (s_i, z') combinations
        # Shape: (n_a, n_z) where savings_grid has n_a points
        a_next_grid = R * savings_grid[:, None] + y(z_grid)

        # Interpolate to get consumption at each (a', z')
        # For each z', interpolate over the a' values
        def interp_for_z(z_idx):
            return jnp.interp(a_next_grid[:, z_idx], asset_grid, σ[:, z_idx])

        c_next_grid = jax.vmap(interp_for_z)(jnp.arange(n_z))  # Shape: (n_z, n_a)
        c_next_grid = c_next_grid.T  # Shape: (n_a, n_z)

        # Compute u'(c') for all points
        u_prime_next = u_prime(c_next_grid, γ)

        # Take expectation over z' for each s, given current state z_j
        expected_marginal = u_prime_next @ Π[j, :]  # Shape: (n_a,)

        # Use Euler equation to find today's consumption
        c_vals = u_prime_inv(β * R * expected_marginal, γ)

        # Compute endogenous grid of current assets: a = c + s
        a_endogenous = c_vals + savings_grid

        # Interpolate back to exogenous asset grid
        σ_new = jnp.interp(asset_grid, a_endogenous, c_vals)

        # For asset levels below the minimum endogenous grid point,
        # the household is constrained and consumes everything: c = a

        σ_new = jnp.where(asset_grid < a_endogenous[0],
                          asset_grid,
                          σ_new)

        return σ_new  #  Consumption over the asset grid given z[j]

    # Compute consumption over all income states using vmap
    c_vmap = jax.vmap(compute_c_for_fixed_income_state)
    σ_new = c_vmap(jnp.arange(n_z)) # Shape (n_z, n_a), one row per income state

    return σ_new.T  # Transpose to get (n_a, n_z)


# %%
@jax.jit
def solve_model(ifp: IFP,
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
        σ_new = K(σ, ifp)
        error = jnp.max(jnp.abs(σ_new - σ))
        return i + 1, σ_new, error

    initial_state = (0, σ_init, tol + 1)
    final_loop_state = jax.lax.while_loop(condition, body, initial_state)
    i, σ, error = final_loop_state

    return σ


# %% [markdown]
# ### Test run
#
# Let's road test the EGM code.

# %%
ifp = create_ifp()
R, β, γ, Π, z_grid, asset_grid = ifp
σ_init = asset_grid[:, None] * jnp.ones(len(z_grid))
σ_star = solve_model(ifp, σ_init)

# %% [markdown]
# Here's a plot of the optimal policy for each $z$ state

# %%
fig, ax = plt.subplots()
ax.plot(asset_grid, σ_star[:, 0], label='bad state')
ax.plot(asset_grid, σ_star[:, 1], label='good state')
ax.set(xlabel='assets', ylabel='consumption')
ax.legend()
plt.show()

# %% [markdown]
# To begin to understand the long run asset levels held by households under the default parameters, let's look at the
# 45 degree diagram showing the law of motion for assets under the optimal consumption policy.

# %%
ifp = create_ifp()
R, β, γ, Π, z_grid, asset_grid = ifp
σ_init = asset_grid[:, None] * jnp.ones(len(z_grid))
σ_star = solve_model(ifp, σ_init)
a = asset_grid

fig, ax = plt.subplots()
for z, lb in zip((0, 1), ('low income', 'high income')):
    ax.plot(a, R * (a - σ_star[:, z]) + y(z_grid[z]) , label=lb)

ax.plot(a, a, 'k--')
ax.set(xlabel='current assets', ylabel='next period assets')

ax.legend()
plt.show()


# %% [markdown]
# The unbroken lines show the update function for assets at each $z$, which is
#
# $$
#     a \mapsto R (a - \sigma^*(a, z)) + y(z')
# $$
#
# where we plot this for a particular realization $z' = z$.
#
# The dashed line is the 45 degree line.
#
# The figure suggests that the dynamics will be stable --- assets do not diverge
# even in the highest state.
#
# In fact there is a unique stationary distribution of assets that we can calculate by simulation -- we examine this below.
#
# * Can be proved via theorem 2 of {cite}`HopenhaynPrescott1992`.
# * It represents the long run dispersion of assets across households when households have idiosyncratic shocks.
#
#
#
# ### A Sanity Check
#
# One way to check our results is to
#
# * set labor income to zero in each state and
# * set the gross interest rate $R$ to unity.
#
# In this case, our income fluctuation problem is just a CRRA cake eating problem.
#
# Then the value function and optimal consumption policy are given by

# %%
def c_star(x, β, γ):
    return (1 - β ** (1/γ)) * x


def v_star(x, β, γ):
    return (1 - β**(1 / γ))**(-γ) * (x**(1-γ) / (1-γ))


# %% [markdown]
# Let's see if we match up:

# %%
ifp_cake_eating = create_ifp(r=0.0, z_grid=(-jnp.inf, -jnp.inf))
R, β, γ, Π, z_grid, asset_grid = ifp_cake_eating
σ_init = asset_grid[:, None] * jnp.ones(len(z_grid))
σ_star = solve_model(ifp_cake_eating, σ_init)

fig, ax = plt.subplots()
ax.plot(asset_grid, σ_star[:, 0], label='numerical')
ax.plot(asset_grid,
        c_star(asset_grid, ifp_cake_eating.β, ifp_cake_eating.γ),
        '--', label='analytical')
ax.set(xlabel='assets', ylabel='consumption')
ax.legend()
plt.show()

# %% [markdown]
# This looks pretty good.
#
#
# ## Exercises
#
# ```{exercise-start}
# :label: ifp_ex1
# ```
#
# Let's consider how the interest rate affects consumption.
#
# * Step `r` through `np.linspace(0, 0.016, 4)`.
# * Other than `r`, hold all parameters at their default values.
# * Plot consumption against assets for income shock fixed at the smallest value.
#
# Your figure should show that, for this model, higher interest rates boost
# suppress consumption (because they encourage more savings).
#
# ```{exercise-end}
# ```
#
# ```{solution-start} ifp_ex1
# :class: dropdown
# ```
#
# Here's one solution:

# %%
# With β=0.98, we need R*β < 1, so r < 0.0204
r_vals = np.linspace(0, 0.016, 4)

fig, ax = plt.subplots()
for r_val in r_vals:
    ifp = create_ifp(r=r_val)
    R, β, γ, Π, z_grid, asset_grid = ifp
    σ_init = asset_grid[:, None] * jnp.ones(len(z_grid))
    σ_star = solve_model(ifp, σ_init)
    ax.plot(asset_grid, σ_star[:, 0], label=f'$r = {r_val:.3f}$')

ax.set(xlabel='asset level', ylabel='consumption (low income)')
ax.legend()
plt.show()


# %% [markdown]
# ```{solution-end}
# ```
#
#
# ```{exercise-start}
# :label: ifp_ex2
# ```
#
# Let's approximate the stationary distribution by simulation.
#
# Run a large number of households forward for $T$ periods and then histogram the
# cross-sectional distribution of assets.
#
# Set `num_households=50_000, T=500`.
#
# ```{exercise-end}
# ```
#
# ```{solution-start} ifp_ex2
# :class: dropdown
# ```
#
# First we write a function to simulate many households in parallel using JAX.

# %%
def compute_asset_stationary(ifp, σ_star, num_households=50_000, T=500, seed=1234):
    """
    Simulates num_households households for T periods to approximate
    the stationary distribution of assets.

    By ergodicity, simulating many households for moderate time is equivalent
    to simulating one household for very long time, but parallelizes better.

    ifp is an instance of IFP
    σ_star is the optimal consumption policy
    """
    R, β, γ, Π, z_grid, asset_grid = ifp
    n_z = len(z_grid)

    # Create interpolation function for consumption policy
    σ_interp = lambda a, z_idx: jnp.interp(a, asset_grid, σ_star[:, z_idx])

    # Simulate one household forward
    def simulate_one_household(key):
        # Random initial state (both z and a)
        key1, key2, key3 = jax.random.split(key, 3)
        z_idx = jax.random.choice(key1, n_z)
        # Start with random assets drawn uniformly from [0, asset_grid_max/2]
        a = jax.random.uniform(key3, minval=0.0, maxval=asset_grid[-1]/2)

        # Simulate forward T periods
        def step(state, key_t):
            a_current, z_current = state
            # Consume based on current state
            c = σ_interp(a_current, z_current)
            # Draw next shock
            z_next = jax.random.choice(key_t, n_z, p=Π[z_current])
            # Update assets: a' = R*(a - c) + Y'
            z_val = z_grid[z_next]
            a_next = R * (a_current - c) + y(z_val)
            return (a_next, z_next), None

        keys = jax.random.split(key2, T)
        (a_final, _), _ = jax.lax.scan(step, (a, z_idx), keys)
        return a_final

    # Vectorize over many households
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, num_households)
    assets = jax.vmap(simulate_one_household)(keys)

    return np.array(assets)


# %% [markdown]
# Now we call the function, generate the asset distribution and histogram it:

# %%
ifp = create_ifp()
R, β, γ, Π, z_grid, asset_grid = ifp
σ_init = asset_grid[:, None] * jnp.ones(len(z_grid))
σ_star = solve_model(ifp, σ_init)
assets = compute_asset_stationary(ifp, σ_star)

fig, ax = plt.subplots()
ax.hist(assets, bins=20, alpha=0.5, density=True)
ax.set(xlabel='assets')
plt.show()

# %% [markdown]
# The shape of the asset distribution is unrealistic.
#
# Here it is left skewed when in reality it has a long right tail.
#
# In a {doc}`subsequent lecture <ifp_advanced>` we will rectify this by adding
# more realistic features to the model.
#
# ```{solution-end}
# ```
#
#
#
# ```{exercise-start}
# :label: ifp_ex3
# ```
#
# Following on from exercises 1 and 2, let's look at how savings and aggregate
# asset holdings vary with the interest rate
#
# ```{note}
# {cite}`Ljungqvist2012` section 18.6 can be consulted for more background on the topic treated in this exercise.
# ```
# For a given parameterization of the model, the mean of the stationary
# distribution of assets can be interpreted as aggregate capital in an economy
# with a unit mass of *ex-ante* identical households facing idiosyncratic
# shocks.
#
# Your task is to investigate how this measure of aggregate capital varies with
# the interest rate.
#
# Following tradition, put the price (i.e., interest rate) on the vertical axis.
#
# On the horizontal axis put aggregate capital, computed as the mean of the
# stationary distribution given the interest rate.
#
# ```{exercise-end}
# ```
#
# ```{solution-start} ifp_ex3
# :class: dropdown
# ```
#
# Here's one solution

# %%
M = 25
# With β=0.98, we need R*β < 1, so R < 1/0.98 ≈ 1.0204, thus r < 0.0204
r_vals = np.linspace(0, 0.015, M)
fig, ax = plt.subplots()

asset_mean = []
for r in r_vals:
    print(f'Solving model at r = {r}')
    ifp = create_ifp(r=r)
    R, β, γ, Π, z_grid, asset_grid = ifp
    σ_init = asset_grid[:, None] * jnp.ones(len(z_grid))
    σ_star = solve_model(ifp, σ_init)
    assets = compute_asset_stationary(ifp, σ_star, num_households=10_000, T=500)
    mean = np.mean(assets)
    asset_mean.append(mean)
    print(f'  Mean assets: {mean:.4f}')
ax.plot(r_vals, asset_mean)

ax.set(xlabel='interest rate', ylabel='capital')

plt.show()

# %% [markdown]
# As expected, aggregate savings increases with the interest rate.
#
# ```{solution-end}
# ```
