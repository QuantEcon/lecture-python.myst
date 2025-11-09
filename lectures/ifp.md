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

# {index}`The Income Fluctuation Problem I: Basic Model <single: The Income Fluctuation Problem I: Basic Model>`

```{contents} Contents
:depth: 2
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon
```

## Overview

In this lecture, we study an optimal savings problem for an infinitely lived consumer---the "common ancestor" described in {cite}`Ljungqvist2012`, section 1.3.

This is an essential sub-problem for many representative macroeconomic models

* {cite}`Aiyagari1994`
* {cite}`Huggett1993`
* etc.

It is related to the decision problem in the {doc}`cake eating model <cake_eating_stochastic>` and yet differs in important ways.

For example, the choice problem for the agent includes an additive income term that leads to an occasionally binding constraint.

Moreover, shocks affecting the budget constraint are correlated, forcing us to
track an extra state variable.

To solve the model we will use the endogenous grid method, which we found to be {doc}`fast and accurate <cake_eating_egm_jax>` in our investigation of cake eating.


We'll need the following imports:

```{code-cell} ipython
import matplotlib.pyplot as plt
import numpy as np
from quantecon import MarkovChain
import jax.numpy as jnp
```

### References

We skip most technical details but they can be found in {cite}`ma2020income`.

Other references include {cite}`Deaton1991`, {cite}`DenHaan2010`,
{cite}`Kuhn2013`, {cite}`Rabault2002`,  {cite}`Reiter2009`  and
{cite}`SchechtmanEscudero1977`.


## The Optimal Savings Problem

```{index} single: Optimal Savings; Problem
```

Let's write down the model and then discuss how to solve it.

### Set-Up

Consider a household that chooses a state-contingent consumption plan $\{c_t\}_{t \geq 0}$ to maximize

$$
\mathbb{E} \, \sum_{t=0}^{\infty} \beta^t u(c_t)
$$

subject to

```{math}
:label: eqst

a_{t+1} + c_t \leq  R a_t + Y_t
\quad c_t \geq 0,
\quad a_t \geq 0
\quad t = 0, 1, \ldots
```

Here

* $\beta \in (0,1)$ is the discount factor
* $a_t$ is asset holdings at time $t$, with borrowing constraint $a_t \geq 0$
* $c_t$ is consumption
* $Y_t$ is non-capital income (wages, unemployment compensation, etc.)
* $R := 1 + r$, where $r > 0$ is the interest rate on savings

The timing here is as follows:

1. At the start of period $t$, the household observes labor income $Y_t$ and financial assets $R a_t$ .
1. The household chooses current consumption $c_t$.
1. Time shifts to $t+1$ and the process repeats.

Non-capital income $Y_t$ is given by $Y_t = y(Z_t)$, where

* $\{Z_t\}$ is an exogeneous state process and
* $y$ is a given function taking values in $\mathbb{R}_+$.

As is common in the literature, we take $\{Z_t\}$ to be a finite state
Markov chain taking values in $\mathsf Z$ with Markov matrix $\Pi$.

We further assume that

1. $\beta R < 1$
1. $u$ is smooth, strictly increasing and strictly concave with $\lim_{c \to 0} u'(c) = \infty$ and $\lim_{c \to \infty} u'(c) = 0$
1. $y(z) = \exp(z)$ 

The asset space is $\mathbb R_+$ and the state is the pair $(a,z) \in \mathsf S := \mathbb R_+ \times \mathsf Z$.

A **feasible consumption path** from $(a,z) \in \mathsf S$ is a consumption
sequence $\{c_t\}$ such that $\{c_t\}$ and its induced asset path $\{a_t\}$ satisfy

1. $(a_0, z_0) = (a, z)$
1. the feasibility constraints in {eq}`eqst`, and
1. measurability, which means that $c_t$ is a function of random
   outcomes up to date $t$ but not after.

The meaning of the third point is just that consumption at time $t$
cannot be a function of outcomes are yet to be observed.

In fact, for this problem, consumption can be chosen optimally by taking it to
be contingent only on the current state.

Optimality is defined below.



### Value Function and Euler Equation

The **value function** $V \colon \mathsf S \to \mathbb{R}$ is defined by

```{math}
:label: eqvf

V(a, z) := \max \, \mathbb{E}
\left\{
\sum_{t=0}^{\infty} \beta^t u(c_t)
\right\}
```

where the maximization is overall feasible consumption paths from $(a,z)$.

An **optimal consumption path** from $(a,z)$ is a feasible consumption path from $(a,z)$ that attains the supremum in {eq}`eqvf`.

To pin down such paths we can use a version of the Euler equation, which in the present setting is

```{math}
:label: ee00

    u' (c_t) \geq \beta R \,  \mathbb{E}_t  u'(c_{t+1})
```

and

```{math}
:label: ee01

    c_t < a_t
    \; \implies \;
    u' (c_t) = \beta R \,  \mathbb{E}_t  u'(c_{t+1})
```

When $c_t = a_t$ we obviously have $u'(c_t) = u'(a_t)$,

When $c_t$ hits the upper bound $a_t$, the
strict inequality $u' (c_t) > \beta R \,  \mathbb{E}_t  u'(c_{t+1})$
can occur because $c_t$ cannot increase sufficiently to attain equality.

(The lower boundary case $c_t = 0$ never arises at the optimum because
$u'(0) = \infty$.)


### Optimality Results

As shown in {cite}`ma2020income`,

1. For each $(a,z) \in \mathsf S$, a unique optimal consumption path from $(a,z)$ exists
1. This path is the unique feasible path from $(a,z)$ satisfying the
   Euler equality {eq}`eqeul0` and the transversality condition

```{math}
:label: eqtv

\lim_{t \to \infty} \beta^t \, \mathbb{E} \, [ u'(c_t) a_{t+1} ] = 0
```

Moreover, there exists an **optimal consumption policy**
$\sigma^* \colon \mathsf S \to \mathbb R_+$ such that the path
from $(a,z)$ generated by

$$
    (a_0, z_0) = (a, z),
    \quad
    c_t = \sigma^*(a_t, Z_t)
    \quad \text{and} \quad
    a_{t+1} = R (a_t - c_t) + Y_{t+1}
$$

satisfies both {eq}`eqeul0` and {eq}`eqtv`, and hence is the unique optimal
path from $(a,z)$.

Thus, to solve the optimization problem, we need to compute the policy $\sigma^*$.

(ifp_computation)=
## Computation

```{index} single: Optimal Savings; Computation
```

We solve for the optimal consumption policy using time iteration and the
endogenous grid method.

Readers unfamiliar with the endogenous grid method should review the discussion
in {doc}`cake_eating_egm`.

### Solution Method

We rewrite {eq}`ee01` to make it a statement about functions rather than
random variables:


```{math}
:label: eqeul1

    (u' \circ \sigma)  (a, z)
    = \beta R \, \sum_{z'}  (u' \circ \sigma)
            [R a + y(z) - \sigma(a, z)), \, z'] \Pi(z, z')
```

Here

* $(u' \circ \sigma)(s) := u'(\sigma(s))$,
* primes indicate next period states (as well as derivatives), and
* $\sigma$ is the unknown function.

We aim to find a fixed point $\sigma$ of {eq}`eqeul1`.

To do so we use the EGM.

We begin with an exogeneous grid $G = \{a'_0, \ldots, a'_{m-1}\}$ with $a'_0 = 0$.

Fix a current guess of the policy function $\sigma$. 

For each $a'_i$ and $z_j$ we set

$$
    c_{ij} = (u')^{-1}
        \left[
            \beta R \, \sum_{z'}  
            u' [ \sigma(a'_i, z') ] \Pi(z_j, z')
        \right]
$$

and then $a^e_{ij}$ as the current asset level $a_t$ that solves the budget constraint
$a'_{ij} + c_{ij} =  R a_t + y(z_j)$.  

That is,

$$
    a^e_{ij} = \frac{1}{R} [a'_{ij} + c_{ij} - y(z_j)].  
$$

Our next guess policy function, which we write as $K\sigma$, is the linear interpolation of
$(a^e_{ij}, c_{ij})$ over $i$, for each $j$.

(The number of one dimensional linear interpolations is equal to `len(z_grid)`.)

For $a < a^e_{ij}$ we use the budget constraint to set $(K \sigma)(a, z_j) = Ra + y(z_j)$.



## Implementation

```{index} single: Optimal Savings; Programming Implementation
```

We use the CRRA utility specification

$$
    u(c) = \frac{c^{1 - \gamma}} {1 - \gamma}
$$


### Set Up

Here we build a class called `IFP` that stores the model primitives.

```{code-cell} python3
class IFP(NamedTuple):
    R: float,               # Interest rate 1 + r
    β: float,               # Discount factor
    γ: float,               # Preference parameter
    Π: jnp.array            # Markov matrix
    z_grid: jnp.array       # Markov state values for Z_t
    asset_grid: jnp.array   # Exogenous asset grid 
]

def create_ifp(r=0.01,
               β=0.96,
               γ=1.5,
               Π=((0.6, 0.4),
                 (0.05, 0.95)),
               z_grid=(0.0, 0.1),
               asset_grid_max=16,
               asset_grid_size=50):

    assert self.R * self.β < 1, "Stability condition violated."
    
    asset_grid = jnp.linspace(0, asset_grid_max, asset_grid_size)
    Π, z_grid = jnp.array(Π), jnp.array(z_grid)
    R = 1 + r
    return IFP(R=R, β=β, γ=γ, Π=Π, z_grid=z_grid, asset_grid=asset_grid)

# Set y(z) = exp(z)
y = jnp.exp
```

The exogeneous state process $\{Z_t\}$ defaults to a two-state Markov chain
with transition matrix $\Pi$.

We define utility globally:

```{code-cell} python3
# Define utility function derivatives
u_prime = lambda c, γ: c**(-γ)
u_prime_inv = lambda c, γ: c**(-1/γ)
```


### Solver

```{code-cell} python3
def K(σ: jnp.ndarray, ifp: IFP) -> jnp.ndarray:
    """
    The Coleman-Reffett operator for the IFP model using EGM

    """
    R, β, γ, Π, z_grid, asset_grid = ifp

    # Determine endogenous grid associated with consumption choices in σ_array
    ae = (1/R) (σ_array + a_grid - y(z_grid))

    # Linear interpolation of policy using endogenous grid.  
    def σ_interp(ap):
        return [jnp.interp(ap, ae[:, j], σ[:, j]) for j in range(len(z_grid))]

    # Define function to compute consumption at a single grid pair (a'_i, z_j)
    def compute_c(i, j):
        ap = ae[i]
        rhs = jnp.sum( u_prime(σ_interp(ap)) * Π[j, :] )
        return u_prime_inv(β * R * rhs)

    # Vectorize over grid using vmap
    compute_c_vectorized = jax.vmap(compute_c)
    next_σ_array = compute_c_vectorized(asset_grid, z_grid)

    return next_σ_array
```



```{code-cell} python3
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
        σ_new = K(σ, model)
        error = jnp.max(jnp.abs(σ_new - σ))
        return i + 1, σ_new, error

    # Initialize loop state
    initial_state = (0, σ_init, tol + 1)

    # Run the loop
    i, σ, error = jax.lax.while_loop(condition, body, initial_state)

    return σ
```

### Test run

Let's road test the EGM code.

```{code-cell} python3
ifp = IFP()
R, β, γ, Π, z_grid, asset_grid = ifp
σ_init = R * a_grid + z_grid
σ_star = solve_model(ifp, σ_init)
```

Here's a plot of the optimal policy for each $z$ state


```{code-cell} python3
fig, ax = plt.subplots()
ax.plot(a_grid, σ_star[:, 0], label='bad state')
ax.plot(a_grid, σ_star[:, 1], label='good state')
ax.set(xlabel='assets', ylabel='consumption')
ax.legend()
plt.show()
```



### A Sanity Check

One way to check our results is to

* set labor income to zero in each state and
* set the gross interest rate $R$ to unity.

In this case, our income fluctuation problem is just a CRRA cake eating problem.

Then the value function and optimal consumption policy are given by

```{code-cell} python3
def c_star(x, β, γ):
    return (1 - β ** (1/γ)) * x


def v_star(x, β, γ):
    return (1 - β**(1 / γ))**(-γ) * (x**(1-γ) / (1-γ))
```

Let's see if we match up:

```{code-cell} python3
ifp_cake_eating = IFP(r=0.0, z_grid=(-jnp.inf, -jnp.inf))
R, β, γ, Π, z_grid, asset_grid = ifp_cake_eating
σ_init = R * a_grid + z_grid
σ_star = solve_model_time_iter(ifp_cake_eating, σ_init)

fig, ax = plt.subplots()
ax.plot(a_grid, σ_star[:, 0], label='numerical')
ax.plot(a_grid, c_star(a_grid, ifp.β, ifp.γ), '--', label='analytical')
ax.set(xlabel='assets', ylabel='consumption')
ax.legend()
plt.show()
```


This looks pretty good.


## Exercises

```{exercise-start}
:label: ifp_ex1
```

Let's consider how the interest rate affects consumption.

* Step `r` through `np.linspace(0, 0.04, 4)`.
* Other than `r`, hold all parameters at their default values.
* Plot consumption against assets for income shock fixed at the smallest value.

Your figure should show that higher interest rates boost savings and suppress consumption.

```{exercise-end}
```

```{solution-start} ifp_ex1
:class: dropdown
```

Here's one solution:

```{code-cell} python3
r_vals = np.linspace(0, 0.04, 2.0)

fig, ax = plt.subplots()
for r_val in r_vals:
    ifp = IFP(r=r_val)
    σ_star = solve_model(ifp, σ_init)
    ax.plot(ifp.asset_grid, σ_star[:, 0], label=f'$r = {r_val:.3f}$')

ax.set(xlabel='asset level', ylabel='consumption (low income)')
ax.legend()
plt.show()
```

```{solution-end}
```


```{exercise-start}
:label: ifp_ex2
```

Now let's consider the long run asset levels held by households under the
default parameters.

The following figure is a 45 degree diagram showing the law of motion for assets when consumption is optimal

```{code-cell} python3
ifp = IFP()

σ_star = solve_model(ifp, σ_init)
a = ifp.asset_grid
R = ifp.R

fig, ax = plt.subplots()
for z, lb in zip((0, 1), ('low income', 'high income')):
    ax.plot(a, R * (a - σ_star[:, z]) + y(z) , label=lb)

ax.plot(a, a, 'k--')
ax.set(xlabel='current assets', ylabel='next period assets')

ax.legend()
plt.show()
```

The unbroken lines show the update function for assets at each $z$, which is

$$
a \mapsto R (a - \sigma^*(a, z)) + y(z)
$$

The dashed line is the 45 degree line.

We can see from the figure that the dynamics will be stable --- assets do not
diverge even in the highest state.

In fact there is a unique stationary distribution of assets that we can calculate by simulation

* Can be proved via theorem 2 of {cite}`HopenhaynPrescott1992`.
* It represents the long run dispersion of assets across households when households have idiosyncratic shocks.

Ergodicity is valid here, so stationary probabilities can be calculated by averaging over a single long time series.

Hence to approximate the stationary distribution we can simulate a long time
series for assets and histogram it.

Your task is to generate such a histogram.

* Use a single time series $\{a_t\}$ of length 500,000.
* Given the length of this time series, the initial condition $(a_0,
  z_0)$ will not matter.
* You might find it helpful to use the `MarkovChain` class from `quantecon`.

```{exercise-end}
```

```{solution-start} ifp_ex2
:class: dropdown
```

First we write a function to compute a long asset series.

```{code-cell} python3
def compute_asset_series(ifp, T=500_000, seed=1234):
    """
    Simulates a time series of length T for assets, given optimal
    savings behavior.

    ifp is an instance of IFP
    """
    P, y, R = ifp.P, ifp.y, ifp.R  # Simplify names

    # Solve for the optimal policy
    σ_star = solve_model_time_iter(ifp, σ_init, verbose=False)
    σ = lambda a, z: np.interp(a, ifp.asset_grid, σ_star[:, z])

    # Simulate the exogeneous state process
    mc = MarkovChain(P)
    z_seq = mc.simulate(T, random_state=seed)

    # Simulate the asset path
    a = np.zeros(T+1)
    for t in range(T):
        z = z_seq[t]
        a[t+1] = R * (a[t] - σ(a[t], z)) + y[z]
    return a
```

Now we call the function, generate the series and then histogram it:

```{code-cell} python3
ifp = IFP()
a = compute_asset_series(ifp)

fig, ax = plt.subplots()
ax.hist(a, bins=20, alpha=0.5, density=True)
ax.set(xlabel='assets')
plt.show()
```

The shape of the asset distribution is unrealistic.

Here it is left skewed when in reality it has a long right tail.

In a {doc}`subsequent lecture <ifp_advanced>` we will rectify this by adding
more realistic features to the model.

```{solution-end}
```

```{exercise-start}
:label: ifp_ex3
```

Following on from exercises 1 and 2, let's look at how savings and aggregate
asset holdings vary with the interest rate

```{note}
{cite}`Ljungqvist2012` section 18.6 can be consulted for more background on the topic treated in this exercise.
```
For a given parameterization of the model, the mean of the stationary
distribution of assets can be interpreted as aggregate capital in an economy
with a unit mass of *ex-ante* identical households facing idiosyncratic
shocks.

Your task is to investigate how this measure of aggregate capital varies with
the interest rate.

Following tradition, put the price (i.e., interest rate) on the vertical axis.

On the horizontal axis put aggregate capital, computed as the mean of the
stationary distribution given the interest rate.

```{exercise-end}
```

```{solution-start} ifp_ex3
:class: dropdown
```

Here's one solution

```{code-cell} python3
M = 25
r_vals = np.linspace(0, 0.02, M)
fig, ax = plt.subplots()

asset_mean = []
for r in r_vals:
    print(f'Solving model at r = {r}')
    ifp = IFP(r=r)
    mean = np.mean(compute_asset_series(ifp, T=250_000))
    asset_mean.append(mean)
ax.plot(asset_mean, r_vals)

ax.set(xlabel='capital', ylabel='interest rate')

plt.show()
```

As expected, aggregate savings increases with the interest rate.

```{solution-end}
```
