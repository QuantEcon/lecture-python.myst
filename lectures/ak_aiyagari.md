---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Prerequisite

Two quantecon lectures:

1. [Discrete State Dynamic Programming](https://python-advanced.quantecon.org/discrete_dp.html)
2. [Transitions in an Overlapping Generations Model](https://python.quantecon.org/ak2.html)

Optional: [The Aiyagari Model (with JAX)](https://jax.quantecon.org/aiyagari_jax.html)

```{code-cell} ipython3
:id: ac32ac26

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.scipy as jsp
import jax
```

+++ {"id": "ec922397", "user_expressions": []}

# Transitions in an AK-Aiyagari Model

## 1. Introduction

This lecture describes an  overlapping generations model with these features:


- Agents live many periods as in Auerbach and Kotlikoff (1987)  (AK)
- Agents receive idiosyncratic labor productivity shocks that cannot be fully insured as in  Aiyagari (1994)
- Government fiscal policy instruments include taxes, debt, and transfers as in  AK
- A competitive equilibrium determines prices and quantities

We use the framework to study:
- How fiscal policies affect different generations
- How market incompleteness promotes precautionary savings
- How life-cycle savings and buffer-stock savings motives interact
- Fiscal policies that redistribute resources across and within generations

## 2. Basic Settings

### Demographics and Time

- Time is discrete and is indexed by $t = 0, 1, 2, ...$
- Each agent lives for $J = 50$ periods and faces no mortality risk
- Age is indexed by $j = 0, 1, ..., 49$
- Population size  is fixed at $1/J$

### Individual State Variables

Agent $i$ of age $j$ at time t is characterized by:

1. Asset holdings $a_{i,j,t}$
2. Idiosyncratic labor productivity $γ_{i,j,t}$

The idiosyncratic labor productivity process follows a two-state Markov chain with:
- Values $γ_l, γ_h$
- Transition matrix $Π$
- Initial distribution for newborns $π = [0.5, 0.5]$

### Labor Supply

- An agent with productivity $γ_{i,j,t}$ supplies $l(j)γ_{i,j,t}$ efficiency units of labor
- $l(j)$ is a deterministic age-specific labor efficiency units profile
- An agent's effective  labor supply combines both the life-cycle efficiency profile  and the idiosyncratic stochastic process
### Initial Conditions

- Newborns start with zero assets: $a_{i,0,t} = 0$
- Initial idiosyncratic  productivityies are drawn from distribution $π$
- Agents leave no bequests and have  terminal value function $V_J(a) = 0$

## 3. Production

A representative firm operates a constant returns to scale Cobb-Douglas technology:

$$Y_t = Z_t K_t^\alpha L_t^{1-\alpha}$$

where:
- $K_t$ is aggregate capital
- $L_t$ is aggregate efficiency units of  labor
- $Z_t$ is total factor productivity
- $α$ is the capital share

## 4. Government

The government:

1. Issues one-period debt $D_t$
2. Collects flat-rate tax rate  $τ_t$ on labor and capital income
3. Implements age-specific lump-sum taxes/transfers $δ_{j,t}$
4. Makes government purchases $G_t$

### Government Budget Constraint

The government budget constraint at time $t$ is:

$$D_{t+1} - D_t = r_t D_t + G_t - T_t$$

where total tax revenues $T_t$ satisfy:

$$T_t = \tau_t w_t L_t + \tau_t r_t(D_t + K_t) + \sum_j \delta_{j,t}$$

## 5. Activities in Factor Markets

At each time $t ≥ 0$, agents supply labor and capital:

### Age-Specific Labor Supply
- Agents of age $j ∈ {0,1,...,J-1}$ supply labor according to:
  * Their deterministic age-efficiency profile $l(j)$
  * Their current idiosyncratic productivity shock $γ_{i,j,t}$
  * Total effective labor supply of $l(j)γ_{i,j,t}$ units
  * A competitive wage $w_t$ per effective unit of labor
  * A flat tax rate  $τ_t$ on labor earnings

### Asset Market Participation
- Agents of all ages $j ∈ {0,1,...,J-1}$ can:
  * Hold assets $a_{i,j,t}$ (subject to borrowing constraints)
  * Earn a risk-free one-period return $r_t$ on savings
  * Pay capital income taxes at a flat rate  $τ_t$
  * Receive/pay age-specific transfers $δ_{j,t}$

### Key Features
1. Lifecycle Patterns:
   - Labor productivity varies systematically with age according to $l(j)$
   - Asset holdings typically follow a lifecycle pattern
   - Age-specific fiscal transfers are described by  $δ_{j,t}$

2. Within-Cohort Heterogeneity:
   - Agents of the same age differ in:
     * Their asset holdings $a_{i,j,t}$ due to different histories of idiosyncratic productivity shocks
     * Their  productivities $γ_{i,j,t}$
     * Their consequent labor incomes and financial wealth

3. Cross-Cohort Interactions:
   - All cohorts participate together in factor markets
   - Asset supplies from all cohorts determine aggregate capital
   - Effective labor supplies from all cohorts determine aggregate labor
   - Equilibrium prices  reflect both lifecycle and re-distributional forces

## 6. Representative Firm's Problem

A representative firm chooses capital and effective labor to maximize profits:

$$\max_{K,L} Z_t K_t^\alpha L_t^{1-\alpha} - r_t K_t - w_t L_t$$

First-order necessary conditions yield:

$$w_t = (1-\alpha)Z_t(K_t/L_t)^\alpha$$
$$r_t = \alpha Z_t(K_t/L_t)^{\alpha-1}$$

## 7. Individual's Problems

### Value Function

The household's value functions satisfy the Bellman equations

$$V_{j,t}(a, \gamma) = \max_{c,a'} \{u(c) + \beta\mathbb{E}[V_{j+1,t+1}(a', \gamma')]\}$$

where maximization is subject to

$$c + a' = (1 + r_t(1-\tau_t))a + (1-\tau_t)w_t l(j)\gamma - \delta_{j,t}$$
$$c \geq 0$$

and the  terminal condition
$V_{J,t}(a, γ) = 0$

### Population Dynamics

The joint probability density function $μ_{j,t}(a,γ)$ of asset holdings and idiosyncratic labor evolves according to

1. For newborns $(j=0)$:
   $$μ_{0,t+1}(a',γ') = π(γ')\text{ if }a'=0\text{, }0\text{ otherwise}$$

2. For other cohorts:
   $$\mu_{j+1,t+1}(a',\gamma') = \int 1\{\sigma_{j,t}(a,\gamma)=a'\}\Pi(\gamma,\gamma')\mu_{j,t}(a,\gamma)d(a,\gamma)$$

where $σ_{j,t}(a,γ)$ is the optimal saving policy function.

## 8. Equilibrium

An equilibrium consists of:
1. Value functions $V_{j,t}$
2. Policy functions $σ_{j,t}$
3. Joint probability distributions $μ_{j,t}$
4. Prices $r_t, w_t$
5. Government policies $τ_t, D_t, δ_{j,t}, G_t$

that satisfy the following conditions

1. Given prices and government policies, value and policy functions solve  households' problems
2. Given prices, firms maximize profits
3. Government budget constraints are  satisfied
4. Markets clear:
   - Asset market: $K_t = \sum_j \int a \mu_{j,t}(a,\gamma)d(a,\gamma) - D_t$
   - Labor market: $L_t = \sum_j \int l(j)\gamma \mu_{j,t}(a,\gamma)d(a,\gamma)$
   
Relative to the AK model, our model adds
- Heterogeneity within generations due to productivity shocks
- A precautionary savings motive
- More re-distributional effects
- More complicated transition dynamics

+++ {"id": "SrLouZTftR5W", "user_expressions": []}

## Implementation

+++ {"id": "pZKvFG_D6bK-", "user_expressions": []}

Using tools in  [discrete state dynamic programming lecture](https://python-advanced.quantecon.org/discrete_dp.html), we solve our
AK-Aiyagari model by combining

* value function iteration with
* equilibrium price determination.

A reasonable  approach is  to nest a discrete DP solver inside an outer loop that searches for market-clearing prices.

For each candidate sequence  of prices (interest rates $r$ and wages $w$), we can solve individual households' dynamic programming problems using either value function iteration or policy function iteration to obtain optimal policy functions, then deduce associated stationary joint probability distributions of asset holdings and idiosyncratic labor efficiency units for each age cohort.

That would give us aggregate capital supply (from household savings) and labor supply (from the age-efficiency profile and productivity shocks).

We can then compare these with capital and labor demand from firms, compute deviations between factor market supplies and demands,
then  update our price guesses until we find market-clearing prices.

For transition dynamics, we want to compute  sequences of time-varying prices by

* using backward induction to compute  value and policy functions,
* forward iteration for the distributions of agents across states.

1. Outer Loop (Market Clearing)
   * Guess initial prices ($r_t, w_t$)
   * Iterate until asset and labor markets clear
   * Use firms' first-order necessary conditions to update prices

2. Inner Loop (Individual Dynamic Programming)
   * For each age cohort:
     - Discretize asset and productivity state space
     - Use value function iteration or policy iteration
     - Solve for optimal savings policies
     - Compute stationary distributions

3. Aggregation
   * Sum across individual states within each cohort
   * Sum across cohorts both
     - Aggregate capital supply, and
     - Aggregate effective labor supply
   * Take into account  population weights $1/J$

4. Transition Dynamics
   * Backward induction:
     - Start from final steady state
     - Solve sequence of value functions
   * Forward iteration:
     - Start from initial distribution
     - Track cohort distributions over time
   * Market clearing in each period:
     - Solve for price sequences
     - Update until all markets clear in all periods

+++ {"id": "a3ab2468-7977-4c66-804f-ce56189fa86a", "user_expressions": []}

We  start coding by defining helper functions that describe preferences, firms, and  government budget constraints.

```{code-cell} ipython3
:id: 18c57a2f

# ϕ, k_bar = 0.2, 2.
ϕ, k_bar = 0., 0.

@jax.jit
def V_bar(a):
    "Terminal value function depending on the asset holding."

    return - ϕ * (a - k_bar) ** 2
```

```{code-cell} ipython3
:id: 8c81e074

ν = 0.5

@jax.jit
def u(c):
    "Utility from consumption."

    return c ** (1 - ν) / (1 - ν)

l1, l2, l3 = 0.5, 0.05, -0.0008

@jax.jit
def l(j):
    "Age-specific wage profile."

    return l1 + l2 * j + l3 * j ** 2
```

+++ {"id": "9f18ac68-3362-4c7c-8c95-7071ee707851", "user_expressions": []}

Let's define a `Firm` namedtuple that  contains parameters governing the  production technology.

```{code-cell} ipython3
:id: a45f7b71

Firm = namedtuple("Firm", ("α", "Z"))

def create_firm(α=0.3, Z=1):

    return Firm(α=α, Z=Z)
```

```{code-cell} ipython3
:id: 0110c7c7

firm = create_firm()
```

+++ {"id": "ef0a9141-3d7b-4264-b701-6eb3ecca3818", "user_expressions": []}

The following helper functions describe  relationship between the aggregates ($K, L$) and the prices ($w, r$) that emerge from the representative  firm's first-order necessary conditions.

```{code-cell} ipython3
:id: e766a3ec

@jax.jit
def KL_to_r(K, L, firm):

    α, Z = firm

    return Z * α * (K / L) ** (α - 1)

@jax.jit
def KL_to_w(K, L, firm):

    α, Z = firm

    return Z * (1 - α) * (K / L) ** α
```

+++ {"id": "2a4fc9d5-5fa9-46ed-ab84-c5fd6495e1c5", "user_expressions": []}

We use a function `find_τ` to find  flat tax rates that balance the government budget constraint given other policy variables that include s debt levels, government spending, and transfers.

```{code-cell} ipython3
:id: 732ce8b4

@jax.jit
def find_τ(policy, price, aggs):

    D, D_next, G, δ = policy
    r, w = price
    K, L = aggs

    num = r * D + G - D_next + D - δ.sum(axis=-1)
    denom = w * L + r * (D + K)

    return num / denom
```

+++ {"id": "ca66a85d-6c03-4f1f-ac5a-252ed9d955ab", "user_expressions": []}

We also use a namedtuple `Household` to store all the relevant parameters that characterize  the household problems.

```{code-cell} ipython3
:id: 67cf6952

Household = namedtuple("Household", ("j_grid", "a_grid", "γ_grid",
                                     "Π", "β", "init_μ", "VJ"))

def create_household(
        a_min=0., a_max=10, a_size=200,
        Π=[[0.9, 0.1], [0.1, 0.9]],
        γ_grid=[0.5, 1.5],
        β=0.96, J=50
    ):

    j_grid = jnp.arange(J)

    a_grid = jnp.linspace(a_min, a_max, a_size)

    γ_grid, Π = map(jnp.array, (γ_grid, Π))
    γ_size = len(γ_grid)

    # population distribution of new borns
    init_μ = jnp.zeros((a_size * γ_size))

    # newborns are endowed with zero asset
    # equal probability of γ
    init_μ = init_μ.at[:γ_size].set(1 / γ_size)

    # terminal value
    # V_bar(a)
    VJ = jnp.empty(a_size * γ_size)
    for a_i in range(a_size):
        a = a_grid[a_i]
        VJ = VJ.at[a_i*γ_size:(a_i+1)*γ_size].set(V_bar(a))

    return Household(j_grid=j_grid, a_grid=a_grid, γ_grid=γ_grid,
                     Π=Π, β=β, init_μ=init_μ, VJ=VJ)
```

```{code-cell} ipython3
:id: fu8p40dCFy0G

hh = create_household()
```

+++ {"id": "d4896e9f-b22e-4115-8c41-3103274dab72", "user_expressions": []}

We  solve household optimization problems using discrete state dynamic programming tools.

Initial steps involve preparing rewards and transition matrices, $R$ and $Q$, for our  discretized Bellman equations.

```{code-cell} ipython3
:id: 7d9439b0

@jax.jit
def populate_Q(household):

    j_grid, a_grid, γ_grid, Π, β, init_μ, VJ = household

    num_state = a_grid.size * γ_grid.size
    num_action = a_grid.size

    Q = jsp.linalg.block_diag(*[Π]*a_grid.size)
    Q = Q.reshape((num_state, num_action, γ_grid.size))
    Q = jnp.tile(Q, a_grid.size).T

    return Q

@jax.jit
def populate_R(j, r, w, τ, δ, household):

    j_grid, a_grid, γ_grid, Π, β, init_μ, VJ = household

    num_state = a_grid.size * γ_grid.size
    num_action = a_grid.size

    a = jnp.reshape(a_grid, (a_grid.size, 1, 1))
    γ = jnp.reshape(γ_grid, (1, γ_grid.size, 1))
    ap = jnp.reshape(a_grid, (1, 1, a_grid.size))
    c = (1 + r*(1-τ)) * a + (1-τ) * w * l(j) * γ - δ[j] - ap

    return jnp.reshape(jnp.where(c > 0, u(c), -jnp.inf),
                      (num_state, num_action))
```

+++ {"id": "1a48bef2-a41f-4bfe-8c84-857588f55075", "user_expressions": []}

## Steady State Computation

+++ {"id": "8d359435-d678-4450-b906-9e83eba4d225", "user_expressions": []}

We first  compute  steady state.

+++ {"id": "211fe8d6-7fcd-40ea-af69-dad5fe8cccec", "user_expressions": []}

Given  guesses of prices and taxes, we can use backwards induction to solve for  value functions and optimal consumption and saving policies  at all  ages.

The function `backwards_opt` solve for optimal values by applying the discretized bellman operator backwards.

```{code-cell} ipython3
:id: ucF_5omDrBZw

@jax.jit
def backwards_opt(prices, taxes, household, Q):

    r, w = prices
    τ, δ = taxes

    j_grid, a_grid, γ_grid, Π, β, init_μ, VJ = household
    J = j_grid.size

    num_state = a_grid.size * γ_grid.size
    num_action = a_grid.size

    def bellman_operator_j(V_next, j):

        Rj = populate_R(j, r, w, τ, δ, household)
        vals = Rj + β * Q.dot(V_next)
        σ_j = jnp.argmax(vals, axis=1)
        V_j = vals[jnp.arange(num_state), σ_j]

        return V_j, (V_j, σ_j)

    js = jnp.arange(J-1, -1, -1)
    init_V = VJ

    _, outputs = jax.lax.scan(bellman_operator_j, init_V, js)
    V, σ = outputs
    V = V[::-1]
    σ = σ[::-1]

    return V, σ
```

```{code-cell} ipython3
:id: 6ea68dc5

r, w = 0.05, 1
τ, δ = 0.15, np.zeros(hh.j_grid.size)

Q = populate_Q(hh)
```

```{code-cell} ipython3
:id: lrfizzme3Ubi

V, σ = backwards_opt([r, w], [τ, δ], hh, Q)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: dL0bCx1PyJ5a
outputId: dbde5d66-eab9-4673-b6e2-fbc4338e25f9
---
%time backwards_opt([r, w], [τ, δ], hh, Q)
```

+++ {"id": "f1d86162-1095-4341-aa22-64790c62a05f", "user_expressions": []}

Given optimal consumption and saving choices by each cohorts, we can compute the stationary joint probability  distribution
of asset levels and idiosyncratic productivity levels in the steady state.

```{code-cell} ipython3
:id: PEkQYUu_1MKv

@jax.jit
def popu_dist(σ, household, Q):

    j_grid, a_grid, γ_grid, Π, β, init_μ, VJ = household

    J = hh.j_grid.size
    num_state = hh.a_grid.size * hh.γ_grid.size

    def update_popu_j(μ_j, j):

        Qσ = Q[jnp.arange(num_state), σ[j]]
        μ_next = μ_j @ Qσ

        return μ_next, μ_next

    js = jnp.arange(J-1)

    _, μ = jax.lax.scan(update_popu_j, init_μ, js)
    μ = jnp.concatenate([init_μ[jnp.newaxis], μ], axis=0)

    return μ
```

```{code-cell} ipython3
:id: baa7ed0a

μ = popu_dist(σ, hh, Q)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: fca544fe
outputId: a735b2d2-0bbe-4d50-e6ac-3c6d40704930
---
%time popu_dist(σ, hh, Q)
```

+++ {"id": "9bc310c2-5abe-4510-8159-13a7a1cff040", "user_expressions": []}

Here we plot the distribution over savings by each age group.

It makes sense  that  young cohorts enter the economy with no asset holdings, then  gradually accumulate assets as they age.

As they approach the end of life, they deplete their asset holdings -- they leave no bequests.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 490
id: 4ed11e8a
outputId: 107851f3-75b1-4bc4-ac1b-8008cc29eaa5
---
for j in [0, 5, 20, 45, 49]:
    plt.plot(hh.a_grid, jnp.sum(μ[j].reshape((hh.a_grid.size, hh.γ_grid.size)), axis=1), label=f'j={j}')

plt.legend()
plt.xlabel('a')

plt.title(r'marginal distribution over a, $\sum_\gamma μ_j(a, γ)$')
plt.xlim([0, 8])
plt.ylim([0, 0.1])

plt.show()
```

+++ {"id": "63f9ed1f-5195-4e8b-98d6-24703b9a9a23", "user_expressions": []}

From an  implied stationary population distribution, we can compute the aggregate labor supply $L$ and private savings $A$.

```{code-cell} ipython3
:id: 05813ff9

@jax.jit
def compute_aggregates(μ, household):

    j_grid, a_grid, γ_grid, Π, β, init_μ, VJ = household

    J, a_size, γ_size = j_grid.size, a_grid.size, γ_grid.size

    μ = μ.reshape((J, hh.a_grid.size, hh.γ_grid.size))

    # compute private savings
    a = a_grid.reshape((1, a_size, 1))
    A = (a * μ).sum() / J

    γ = γ_grid.reshape((1, 1, γ_size))
    lj = l(j_grid).reshape((J, 1, 1))
    L = (lj * γ * μ).sum() / J

    return A, L
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: b5a65f79
outputId: 5eb56b0c-bee1-45c7-af92-4ecbeabdd36c
---
A, L = compute_aggregates(μ, hh)
A, L
```

+++ {"id": "f29b3e57-5105-4696-ac09-eef997389977", "user_expressions": []}

The capital stock in this economy equals $A-D$.

```{code-cell} ipython3
:id: 8aa3d7d1

D = 0
K = A - D
```

+++ {"id": "0c29eab7-2a49-4013-8840-cd8eae50cada", "user_expressions": []}

The firm's optimality conditions imply  interest rate $r$ and wage rate $w$.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 8a73b7ce
outputId: 804cc186-c477-46e2-e83d-9cfc4c83ee1e
---
KL_to_r(K, L, firm), KL_to_w(K, L, firm)
```

+++ {"id": "db5dca01-d878-402e-a64f-f0ad95be7f86", "user_expressions": []}

The implied prices $(r,w)$ differ  from our guesses, so we must update our guesses and iterate until we find a fixed point.

This is our  outer loop.

```{code-cell} ipython3
:id: KcyKDWCX55v1

@jax.jit
def find_ss(household, firm, pol_target, Q, tol=1e-6, verbose=False):

    j_grid, a_grid, γ_grid, Π, β, init_μ, VJ = household
    J = j_grid.size
    num_state = a_grid.size * γ_grid.size

    D, G, δ = pol_target

    # initial guesses of prices
    r, w = 0.05, 1.

    # initial guess of τ
    τ = 0.15

    def cond_fn(state):

      V, σ, μ, K, L, r, w, τ, D, G, δ, r_old, w_old = state

      error = (r - r_old) ** 2 + (w - w_old) ** 2

      return error > tol

    def body_fn(state):

        V, σ, μ, K, L, r, w, τ, D, G, δ, r_old, w_old = state
        r_old, w_old, τ_old = r, w, τ

        # household optimal decisions and values
        V, σ = backwards_opt([r, w], [τ, δ], hh, Q)

        # compute the stationary distribution
        μ = popu_dist(σ, hh, Q)

        # compute aggregates
        A, L = compute_aggregates(μ, hh)
        K = A - D

        # update prices
        r, w = KL_to_r(K, L, firm), KL_to_w(K, L, firm)

        # find τ
        D_next = D
        τ = find_τ([D, D_next, G, δ],
                   [r, w],
                   [K, L])

        r = (r + r_old) / 2
        w = (w + w_old) / 2

        return V, σ, μ, K, L, r, w, τ, D, G, δ, r_old, w_old

    # initial state
    V = jnp.empty((J, num_state), dtype=float)
    σ = jnp.empty((J, num_state), dtype=int)
    μ = jnp.empty((J, num_state), dtype=float)
    K, L = 1., 1.
    initial_state = (V, σ, μ, K, L, r, w, τ, D, G, δ, r-1, w-1)
    V, σ, μ, K, L, r, w, τ, D, G, δ, _, _ = jax.lax.while_loop(cond_fn, body_fn, initial_state)

    return V, σ, μ, K, L, r, w, τ, D, G, δ
```

```{code-cell} ipython3
:id: 5f0b2495

ss1 = find_ss(hh, firm, [0, 0.1, np.zeros(hh.j_grid.size)], Q, verbose=True)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 50d23855
outputId: 817c7f88-5ab3-448a-fbca-28cf6023816c
---
%time find_ss(hh, firm, [0, 0.1, np.zeros(hh.j_grid.size)], Q);
```

```{code-cell} ipython3
:id: fc5a72f8

hh_out_ss1 = ss1[:3]
quant_ss1 = ss1[3:5]
price_ss1 = ss1[5:7]
policy_ss1 = ss1[7:11]
```

```{code-cell} ipython3
:id: 4a0f47b5-3039-4147-8764-d1922fd46487

# V, σ, μ
V_ss1, σ_ss1, μ_ss1 = hh_out_ss1
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 3c44de09
outputId: e325605a-2249-4279-d73d-274c23652447
---
# K, L
K_ss1, L_ss1 = quant_ss1

K_ss1, L_ss1
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 677d5d7f
outputId: 45e269b4-621f-4bdc-b5c3-aaa17b3148ef
---
# r, w
r_ss1, w_ss1 = price_ss1

r_ss1, w_ss1
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: f18eb0f1
outputId: 64da6e40-afb0-4004-f069-9c9ac639cceb
---
# τ, D, G, δ
τ_ss1, D_ss1, G_ss1, δ_ss1 = policy_ss1

τ_ss1, D_ss1, G_ss1, δ_ss1
```

+++ {"id": "417b7af6-6540-41ee-9566-205cd19c71dc", "user_expressions": []}

## Transition Dynamics

+++ {"id": "85256e73-8668-4710-bb4d-4c3e15f011d5", "user_expressions": []}

We  compute transition dynamics using a function `path_iteration`.

We iterate over guesses of prices and taxes in our outer loop.

In an inner loop, we  compute the optimal consumption and saving choices by each cohort $j$ in each time $t$, then find the implied  evolution of the joint distribution of assets and productivities.

We then  update our  guesses of prices and taxes given the aggregate labor supply and capital stock in the economy.

 * We use `solve_backwards` to solve for optimal saving choices given  price and tax sequences
 * We use `simulate_forward` to compute the  evolution of the joint distributions.

We require two steady states as inputs:

* 1. the initial steady state to provide the initial condition for `simulate_forward`,
* 2. the final steady state to provide continuation values for `solve_backwards`.

```{code-cell} ipython3
:id: DHTE2pzyuaOb

@jax.jit
def bellman_operator(prices, taxes, V_next, household, Q):

    r, w = prices
    τ, δ = taxes

    j_grid, a_grid, γ_grid, Π, β, init_μ, VJ = household
    J = j_grid.size

    num_state = a_grid.size * γ_grid.size
    num_action = a_grid.size

    def bellman_operator_j(j):
        Rj = populate_R(j, r, w, τ, δ, household)
        vals = Rj + β * Q.dot(V_next[j+1])
        σ_j = jnp.argmax(vals, axis=1)
        V_j = vals[jnp.arange(num_state), σ_j]

        return V_j, σ_j

    V, σ = jax.vmap(bellman_operator_j, (0,))(jnp.arange(J-1))

    # the last life stage
    j = J-1
    Rj = populate_R(j, r, w, τ, δ, household)
    vals = Rj + β * Q.dot(VJ)
    σ = jnp.concatenate([σ, jnp.argmax(vals, axis=1)[jnp.newaxis]])
    V = jnp.concatenate([V, vals[jnp.arange(num_state), σ[j]][jnp.newaxis]])

    return V, σ
```

```{code-cell} ipython3
:id: 979c35a7

@jax.jit
def solve_backwards(V_ss2, σ_ss2, household, firm, price_seq, pol_seq, Q):

    j_grid, a_grid, γ_grid, Π, β, init_μ, VJ = household
    J = j_grid.size
    num_state = a_grid.size * γ_grid.size

    τ_seq, D_seq, G_seq, δ_seq = pol_seq
    r_seq, w_seq = price_seq

    T = r_seq.size

    def solve_backwards_t(V_next, t):

        prices = (r_seq[t], w_seq[t])
        taxes = (τ_seq[t], δ_seq[t])
        V, σ = bellman_operator(prices, taxes, V_next, household, Q)

        return V, (V,σ)

    ts = jnp.arange(T-2, -1, -1)
    init_V = V_ss2

    _, outputs = jax.lax.scan(solve_backwards_t, init_V, ts)
    V_seq, σ_seq = outputs
    V_seq = V_seq[::-1]
    σ_seq = σ_seq[::-1]

    V_seq = jnp.concatenate([V_seq, V_ss2[jnp.newaxis]])
    σ_seq = jnp.concatenate([σ_seq, σ_ss2[jnp.newaxis]])

    return V_seq, σ_seq
```

```{code-cell} ipython3
:id: f9f42858

@jax.jit
def population_evolution(σt, μt, household, Q):

    j_grid, a_grid, γ_grid, Π, β, init_μ, VJ = household

    J = hh.j_grid.size
    num_state = hh.a_grid.size * hh.γ_grid.size

    def population_evolution_j(j):

        Qσ = Q[jnp.arange(num_state), σt[j]]
        μ_next = μt[j] @ Qσ

        return μ_next

    μ_next = jax.vmap(population_evolution_j, (0,))(jnp.arange(J-1))
    μ_next = jnp.concatenate([init_μ[jnp.newaxis], μ_next])

    return μ_next
```

```{code-cell} ipython3
:id: 6cfafb24

@jax.jit
def simulate_forwards(σ_seq, D_seq, μ_ss1, K_ss1, L_ss1, household, Q):

    j_grid, a_grid, γ_grid, Π, β, init_μ, VJ = household

    J, num_state = μ_ss1.shape

    T = σ_seq.shape[0]

    def simulate_forwards_t(μ, t):

        μ_next = population_evolution(σ_seq[t], μ, household, Q)

        A, L = compute_aggregates(μ_next, household)
        K = A - D_seq[t+1]

        return μ_next, (μ_next, K, L)

    ts = jnp.arange(T-1)
    init_μ = μ_ss1

    _, outputs = jax.lax.scan(simulate_forwards_t, init_μ, ts)
    μ_seq, K_seq, L_seq = outputs

    μ_seq = jnp.concatenate([μ_ss1[jnp.newaxis], μ_seq])
    K_seq = jnp.concatenate([K_ss1[jnp.newaxis], K_seq])
    L_seq = jnp.concatenate([L_ss1[jnp.newaxis], L_seq])

    return μ_seq, K_seq, L_seq
```

+++ {"id": "259e729a-acc0-4e08-b3bd-f9f8c25c1b53", "user_expressions": []}

The following pseudo code  describe the algorithm of path iteration.

+++ {"id": "APfwIqU7LTfY", "user_expressions": []}

```
Algorithm 1 AK-Aiyagari Transition Path Algorithm
1: procedure TRANSITION-PATH(ss₁, ss₂, T, D, G, δ)
2:    V₁, σ₁, μ₁ ← ss₁                           ▷ Initial steady state
3:    V₂, σ₂, μ₂ ← ss₂                           ▷ Final steady state
4:    r, w, τ ← initialize_prices(T)              ▷ Linear interpolation
5:    error ← ∞, i ← 0
6:    repeat
7:       i ← i + 1
8:       r_old, w_old, τ_old ← r, w, τ
9:       for t ∈ [T, 1] do                        ▷ Backward induction
10:          for j ∈ [0, J-1] do                  ▷ Age groups
11:             V[t,j] ← max_{a'} {u(c) + βE[V[t+1,j+1]]}
12:             σ[t,j] ← argmax_{a'} {u(c) + βE[V[t+1,j+1]]}
13:          end for
14:       end for
15:       for t ∈ [1, T] do                       ▷ Forward simulation
16:          μ[t] ← Γ(σ[t], μ[t-1])              ▷ Distribution evolution
17:          K[t] ← ∫a dμ[t] - D[t]              ▷ Aggregate capital
18:          L[t] ← ∫l(j)γ dμ[t]                 ▷ Aggregate labor
19:          r[t] ← αZ(K[t]/L[t])^(α-1)          ▷ Interest rate
20:          w[t] ← (1-α)Z(K[t]/L[t])^α          ▷ Wage rate
21:          τ[t] ← solve_budget(r[t],w[t],K[t],L[t],D[t],G[t])
22:       end for
23:       error ← ‖r - r_old‖ + ‖w - w_old‖ + ‖τ - τ_old‖
24:       r ← λr + (1-λ)r_old                    ▷ Price dampening
25:       w ← λw + (1-λ)w_old
26:       τ ← λτ + (1-λ)τ_old
27:    until error < ε or i > max_iter
28:    return V, σ, μ, r, w, τ
29: end procedure
```

```{code-cell} ipython3
:id: fb927e09

def path_iteration(ss1, ss2, pol_target, household, firm, Q, tol=1e-4, verbose=False):

    V_ss1, σ_ss1, μ_ss1 = ss1[:3]
    K_ss1, L_ss1 = ss1[3:5]
    r_ss1, w_ss1 = ss1[5:7]
    τ_ss1, D_ss1, G_ss1, δ_ss1 = ss1[7:11]

    V_ss2, σ_ss2, μ_ss2 = ss2[:3]
    K_ss2, L_ss2 = ss2[3:5]
    r_ss2, w_ss2 = ss2[5:7]
    τ_ss2, D_ss2, G_ss2, δ_ss2 = ss2[7:11]

    # the given policies: D, G, δ
    D_seq, G_seq, δ_seq = pol_target
    T = G_seq.shape[0]

    # initial guess price
    r_seq = jnp.linspace(0, 1, T) * (r_ss2 - r_ss1) + r_ss1
    w_seq = jnp.linspace(0, 1, T) * (w_ss2 - w_ss1) + w_ss1

    # initial guess τ=τ_ss1
    τ_seq = jnp.linspace(0, 1, T) * (τ_ss2 - τ_ss1) + τ_ss1

    error = 1
    num_iter = 0

    if verbose:
        fig, axs = plt.subplots(1, 3, figsize=(14, 3))
        axs[0].plot(jnp.arange(T), r_seq)
        axs[1].plot(jnp.arange(T), w_seq)
        axs[2].plot(jnp.arange(T), τ_seq, label=f'iter {num_iter}')

    while error > tol:

        r_old, w_old, τ_old = r_seq, w_seq, τ_seq

        pol_seq = (τ_seq, D_seq, G_seq, δ_seq)
        price_seq = (r_seq, w_seq)

        V_seq, σ_seq = solve_backwards(V_ss2, σ_ss2, hh, firm, price_seq, pol_seq, Q)
        μ_seq, K_seq, L_seq = simulate_forwards(σ_seq, D_seq, μ_ss1, K_ss1, L_ss1, household, Q)

        r_seq = KL_to_r(K_seq, L_seq, firm)
        w_seq = KL_to_w(K_seq, L_seq, firm)

        τ_seq = find_τ([D_seq[:-1], D_seq[1:], G_seq, δ_seq],
                           [r_seq, w_seq],
                           [K_seq, L_seq])

        error = jnp.sum((r_old - r_seq) ** 2) + jnp.sum((w_old - w_seq) ** 2) + jnp.sum((τ_old - τ_seq) ** 2)

        num_iter += 1
        if verbose:
            print(error)
            axs[0].plot(jnp.arange(T), r_seq)
            axs[1].plot(jnp.arange(T), w_seq)
            axs[2].plot(jnp.arange(T), τ_seq, label=f'iter {num_iter}')

        r_seq = (r_seq + r_old) / 2
        w_seq = (w_seq + w_old) / 2
        τ_seq = (τ_seq + τ_old) / 2

    if verbose:
        axs[0].set_xlabel('t')
        axs[1].set_xlabel('t')
        axs[2].set_xlabel('t')

        axs[0].set_title('r')
        axs[1].set_title('w')
        axs[2].set_title('τ')

        axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig('AK-Aiyagari convergence.png', dpi=600)

    return V_seq, σ_seq, μ_seq, K_seq, L_seq, r_seq, w_seq, τ_seq, D_seq, G_seq, δ_seq
```

+++ {"id": "67d2bc7b-9adc-4d37-a871-9c895e2ec6cb", "user_expressions": []}

We now have  tools for  computing  equilibrium transition dynamics ignited by fiscal policy reforms, in the spirit of the [AK lecture](https://python.quantecon.org/ak2.html).

+++ {"id": "NU0ptFQKJ-K-", "user_expressions": []}

## Experiment 1: Immediate Tax Cut

+++ {"id": "6db5bec6-08b7-4bd0-8fd5-c7a6be1ee030", "user_expressions": []}

Assume  that the government cuts the tax rate immediately balances its  budget by issuing debt.

1. at $t=0$, the government unexpectedly announces an immediate tax cut
2. from $t=0$ to $19$, the government  issues debt, so debt  $D_{t+1}$  increases linearly for $20$ periods
3. the government sets a target for its new debt level  $D_{20} =D_0 + 1 = \bar{D} + 1$
4. government spending $\bar{G}$ and transfers $\bar{\delta}_j$ remain constant
5. the government  adjust $\tau_t$ to balance the budget along the transition

We want to compute the  equilibrium transition path.

+++ {"id": "07f5d7c1-3ce9-4b5a-a554-7a1ed21ed598", "user_expressions": []}

Our first step is to prepare appropriate policy variable arrays `D_seq`, `G_seq`, `δ_seq`

We'll compute a `τ_seq`  that balances government budgets.

```{code-cell} ipython3
:id: 11c6a6ce

T = 150

D_seq = jnp.ones(T+1) * D_ss1
D_seq = D_seq.at[:21].set(D_ss1 + jnp.linspace(0, 1, 21))
D_seq = D_seq.at[21:].set(D_seq[20])

G_seq = jnp.ones(T) * G_ss1

δ_seq = jnp.repeat(δ_ss1, T).reshape((T, δ_ss1.size))
```

+++ {"id": "87b67456-1830-4d9c-937c-a8c5eb5572c2", "user_expressions": []}

In order to iterate the path, we need to first find its destination, which is the new steady state under the new fiscal policy.

```{code-cell} ipython3
:id: 8ab3a414

ss2 = find_ss(hh, firm, [D_seq[-1], G_seq[-1], δ_seq[-1]], Q)
```

+++ {"id": "4e24949c-a4d7-44f6-b7e0-baa2f0f102b6", "user_expressions": []}

We can use `path_iteration` to find  equilibrium transition dynamics.

Setting the key argument `verbose=True` tells  the function `path_iteration` to display  convergence information.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 455
id: d46bb77c
outputId: 08d81f40-8ce2-474e-ecbe-4bef44dcd20e
---
paths = path_iteration(ss1, ss2, [D_seq, G_seq, δ_seq], hh, firm, Q, verbose=True)
```

+++ {"id": "d6ee740a-1b02-413b-8bb5-1ea99a6ec729", "user_expressions": []}

After successfully computing transition dynamics, let's study them.

```{code-cell} ipython3
:id: 3a7b3524

V_seq, σ_seq, μ_seq = paths[:3]
K_seq, L_seq = paths[3:5]
r_seq, w_seq = paths[5:7]
τ_seq, D_seq, G_seq, δ_seq = paths[7:11]
```

```{code-cell} ipython3
:id: 5494ec30

ap = hh.a_grid[σ_seq[0]]
```

```{code-cell} ipython3
:id: 95f148cf

j = jnp.reshape(hh.j_grid, (hh.j_grid.size, 1, 1))
lj = l(j)
a = jnp.reshape(hh.a_grid, (1, hh.a_grid.size, 1))
γ = jnp.reshape(hh.γ_grid, (1, 1, hh.γ_grid.size))
```

```{code-cell} ipython3
:id: 06892ec2

t = 0

ap = hh.a_grid[σ_seq[t]]
δ = δ_seq[t].reshape((hh.j_grid.size, 1, 1))

inc = (1 + r_seq[t]*(1-τ_seq[t])) * a + (1-τ_seq[t]) * w_seq[t] * lj * γ - δ
inc = inc.reshape((hh.j_grid.size, hh.a_grid.size * hh.γ_grid.size))

c = inc - ap

c_mean0 = (c * μ_seq[t]).sum(axis=1)
```

+++ {"id": "2a973cca-db35-49b5-98f5-6427f65ae3e5", "user_expressions": []}

We care about how such policy change impacts consumption levels of  cohorts at different  times.

We can study  age-specific average consumption levels.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 489
id: 61ae59d5
outputId: 0573b9cc-cc8d-4b96-9342-32ea0e13019a
---
for t in [1, 10, 20, 50, 149]:

    ap = hh.a_grid[σ_seq[t]]
    δ = δ_seq[t].reshape((hh.j_grid.size, 1, 1))

    inc = (1 + r_seq[t]*(1-τ_seq[t])) * a + (1-τ_seq[t]) * w_seq[t] * lj * γ - δ
    inc = inc.reshape((hh.j_grid.size, hh.a_grid.size * hh.γ_grid.size))

    c = inc - ap

    c_mean = (c * μ_seq[t]).sum(axis=1)

    plt.plot(range(hh.j_grid.size), c_mean-c_mean0, label=f't={t}')

plt.legend()
plt.xlabel(r'j')
plt.title(r'Δmean(C(j))')
```

+++ {"id": "d8e841b8-1b16-4a4c-a156-396559e0550a", "user_expressions": []}

To summarize the transition, we can plot paths as we did in [AK lecture](https://python.quantecon.org/ak2.html#experiment-1-tax-cut).

Unlike the AK setup, we no longer have representative old and young agents.

Instead we have agents from 50 cohorts coexisting simultaneously.

To get a counterpart of AK lectures we can construct two age groups with equal size, setting a threshold at age 25.

```{code-cell} ipython3
:id: 1b7fe4e7

ap = hh.a_grid[σ_ss1]
J = hh.j_grid.size
δ = δ_ss1.reshape((hh.j_grid.size, 1, 1))

inc = (1 + r_ss1*(1-τ_ss1)) * a + (1-τ_ss1) * w_ss1 * lj * γ - δ
inc = inc.reshape((hh.j_grid.size, hh.a_grid.size * hh.γ_grid.size))

c = inc - ap

Cy_ss1 = (c[:J//2] * μ_ss1[:J//2]).sum() / (J // 2)
Co_ss1 = (c[J//2:] * μ_ss1[J//2:]).sum() / (J // 2)
```

```{code-cell} ipython3
:id: b912bdf1

T = σ_seq.shape[0]
J = σ_seq.shape[1]

Cy_seq = np.empty(T)
Co_seq = np.empty(T)

for t in range(T):
    ap = hh.a_grid[σ_seq[t]]
    δ = δ_seq[t].reshape((hh.j_grid.size, 1, 1))

    inc = (1 + r_seq[t]*(1-τ_seq[t])) * a + (1-τ_seq[t]) * w_seq[t] * lj * γ - δ
    inc = inc.reshape((hh.j_grid.size, hh.a_grid.size * hh.γ_grid.size))

    c = inc - ap

    Cy_seq[t] = (c[:J//2] * μ_seq[t, :J//2]).sum() / (J // 2)
    Co_seq[t] = (c[J//2:] * μ_seq[t, J//2:]).sum() / (J // 2)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 855
id: cf48686f
outputId: c163452c-b4cf-4494-df41-98f9ac74a316
---
fig, axs = plt.subplots(3, 3, figsize=(14, 10))

# Cy (j=0-24)
axs[0, 0].plot(Cy_seq)
axs[0, 0].hlines(Cy_ss1, 0, T, color='r', linestyle='--')
axs[0, 0].set_title('Cy (j < 25)')

# Cy (j=25-49)
axs[0, 1].plot(Co_seq)
axs[0, 1].hlines(Co_ss1, 0, T, color='r', linestyle='--')
axs[0, 1].set_title(r'Co (j $\geq$ 25)')

names = ['K', 'L', 'r', 'w', 'τ', 'D', 'G']
for i in range(len(names)):
    i_var = i + 3
    i_axes = i + 2

    row_i = i_axes // 3
    col_i = i_axes % 3

    axs[row_i, col_i].plot(paths[i_var])
    axs[row_i, col_i].hlines(ss1[i_var], 0, T, color='r', linestyle='--')
    axs[row_i, col_i].set_title(names[i])

# ylims
axs[1, 0].set_ylim([ss1[4]-0.1, ss1[4]+0.1])
axs[2, 2].set_ylim([ss1[9]-0.1, ss1[9]+0.1])

plt.show()
```

+++ {"id": "336ced39-80b5-47c3-8a67-1472d27a9221", "user_expressions": []}

To look into the evolution of consumption distribution over age in more detail, let's compute the mean and variance of consumption conditional on age in each time $t$.

```{code-cell} ipython3
:id: 9b84b7ca-bc79-45b8-890a-1771d29dfb11

Cmean_seq = np.empty((T, J))
Cvar_seq = np.empty((T, J))

for t in range(T):
    ap = hh.a_grid[σ_seq[t]]
    δ = δ_seq[t].reshape((hh.j_grid.size, 1, 1))

    inc = (1 + r_seq[t]*(1-τ_seq[t])) * a + (1-τ_seq[t]) * w_seq[t] * lj * γ - δ
    inc = inc.reshape((hh.j_grid.size, hh.a_grid.size * hh.γ_grid.size))

    c = inc - ap

    Cmean_seq[t] = (c * μ_seq[t]).sum(axis=1)
    Cvar_seq[t] = ((c - Cmean_seq[t].reshape((J, 1))) ** 2 * μ_seq[t]).sum(axis=1)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 681
id: 9c6c6a49-6f29-4b90-b171-a2cfba2f2d43
outputId: 4fc2f07e-4a10-4413-b0c8-f8135ea6219d
---
J_seq, T_range = np.meshgrid(np.arange(J), np.arange(T))

fig = plt.figure(figsize=[20, 20])

# plot the consumption mean over age and time
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(T_range, J_seq, Cmean_seq, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax1.set_title(r"Mean of consumption")
ax1.set_xlabel(r"t")
ax1.set_ylabel(r"j")

# plot the consumption variance over age and time
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(T_range, J_seq, Cvar_seq, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax2.set_title(r"Variance of consumption")
ax2.set_xlabel(r"t")
ax2.set_ylabel(r"j")

plt.show()
```

+++ {"id": "48756aed-3bcc-41f8-8489-548f46445565", "user_expressions": []}

## Experiment 2: Preannounced Tax Cut

+++ {"id": "fa340801-bf90-4d55-97bb-9314055a67ba", "user_expressions": []}



Instead of implementing a tax rate cut immediately as it did in Experiment 1, now the government announces a tax rate cut at time $0$ but  delays implementing it for  20 periods.

+++ {"id": "09929aa0-533d-478c-b33a-95870d871e32", "user_expressions": []}

We will use the same key toolkit `path_iteration`.

We only need to specify  `D_seq` appropriately.

```{code-cell} ipython3
:id: 7562ddf6

T = 150

D_t = 20
D_seq = jnp.ones(T+1) * D_ss1
D_seq = D_seq.at[D_t:D_t+21].set(D_ss1 + jnp.linspace(0, 1, 21))
D_seq = D_seq.at[D_t+21:].set(D_seq[D_t+20])

G_seq = jnp.ones(T) * G_ss1

δ_seq = jnp.repeat(δ_ss1, T).reshape((T, δ_ss1.size))
```

```{code-cell} ipython3
:id: d100dd45

ss2 = find_ss(hh, firm, [D_seq[-1], G_seq[-1], δ_seq[-1]], Q)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 437
id: c1e2cf31
outputId: 5ec48f9b-76c8-4723-c468-86e612124147
---
paths = path_iteration(ss1, ss2, [D_seq, G_seq, δ_seq], hh, firm, Q, verbose=True)
```

```{code-cell} ipython3
:id: dfec98f5

V_seq, σ_seq, μ_seq = paths[:3]
K_seq, L_seq = paths[3:5]
r_seq, w_seq = paths[5:7]
τ_seq, D_seq, G_seq, δ_seq = paths[7:11]
```

```{code-cell} ipython3
:id: e84bb57c

T = σ_seq.shape[0]
J = σ_seq.shape[1]

Cy_seq = np.empty(T)
Co_seq = np.empty(T)

for t in range(T):
    ap = hh.a_grid[σ_seq[t]]
    δ = δ_seq[t].reshape((hh.j_grid.size, 1, 1))

    inc = (1 + r_seq[t]*(1-τ_seq[t])) * a + (1-τ_seq[t]) * w_seq[t] * lj * γ - δ
    inc = inc.reshape((hh.j_grid.size, hh.a_grid.size * hh.γ_grid.size))

    c = inc - ap

    Cy_seq[t] = (c[:J//2] * μ_seq[t, :J//2]).sum() / (J // 2)
    Co_seq[t] = (c[J//2:] * μ_seq[t, J//2:]).sum() / (J // 2)
```

+++ {"id": "7743c581-e0a4-41b9-9558-1be5718eac87", "user_expressions": []}

Below we plot the transition paths of the economy.

Notice how prices and quantities  respond to the foreseen tax rate increase.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 855
id: e2407bcc
outputId: c6e8f769-94d1-4410-fb2e-fda6d3a412e4
---
fig, axs = plt.subplots(3, 3, figsize=(14, 10))

# Cy (j=0-24)
axs[0, 0].plot(Cy_seq)
axs[0, 0].hlines(Cy_ss1, 0, T, color='r', linestyle='--')
# axs[0, 0].vlines(D_t-1, Cy_seq.min()*0.95, Cy_seq.max()*1.05, color='k', linestyle='--', linewidth=0.5)
axs[0, 0].set_title('Cy (j < 25)')

# Cy (j=25-49)
axs[0, 1].plot(Co_seq)
axs[0, 1].hlines(Co_ss1, 0, T, color='r', linestyle='--')
# axs[0, 1].vlines(D_t-1, Co_seq.min()*0.95, Co_seq.max()*1.05, color='k', linestyle='--', linewidth=0.5)
axs[0, 1].set_title(r'Co (j $\geq$ 25)')

names = ['K', 'L', 'r', 'w', 'τ', 'D', 'G']
for i in range(len(names)):
    i_var = i + 3
    i_axes = i + 2

    row_i = i_axes // 3
    col_i = i_axes % 3

    axs[row_i, col_i].plot(paths[i_var])
    axs[row_i, col_i].hlines(ss1[i_var], 0, T, color='r', linestyle='--')
#     axs[row_i, col_i].vlines(D_t-1, paths[i_var].min()*0.95, paths[i_var].max()*1.05, color='k', linestyle='--', linewidth=0.5)
    axs[row_i, col_i].set_title(names[i])

# ylims
axs[1, 0].set_ylim([ss1[4]-0.1, ss1[4]+0.1])
axs[2, 2].set_ylim([ss1[9]-0.1, ss1[9]+0.1])

plt.show()
```

+++ {"id": "d48df239-c4d4-455e-890a-a1a3034fea2f", "user_expressions": []}

Let's zoom in and look at how the capital stock  responds to the future tax cut.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 489
id: 00cbf90b
outputId: eb54401c-6b5b-4b9d-8505-d2f0942f76c0
---
# K
i_var = 3

plt.plot(paths[i_var][:25])
plt.hlines(ss1[i_var], 0, 25, color='r', linestyle='--')
plt.vlines(20, 6, 7, color='k', linestyle='--', linewidth=0.5)
plt.text(17, 6.56, r'tax cut')
plt.ylim([6.52, 6.65])
plt.title("K")
plt.xlabel("t")
```

```{code-cell} ipython3
:id: ca1b3484-4fac-415d-82bb-d011f2e349cf

Cmean_seq = np.empty((T, J))
Cvar_seq = np.empty((T, J))

for t in range(T):
    ap = hh.a_grid[σ_seq[t]]
    δ = δ_seq[t].reshape((hh.j_grid.size, 1, 1))

    inc = (1 + r_seq[t]*(1-τ_seq[t])) * a + (1-τ_seq[t]) * w_seq[t] * lj * γ - δ
    inc = inc.reshape((hh.j_grid.size, hh.a_grid.size * hh.γ_grid.size))

    c = inc - ap

    Cmean_seq[t] = (c * μ_seq[t]).sum(axis=1)
    Cvar_seq[t] = ((c - Cmean_seq[t].reshape((J, 1))) ** 2 * μ_seq[t]).sum(axis=1)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 681
id: 035dbce4-d586-4c3c-83c9-a791cfb3310f
outputId: 6c7d9091-5eed-4902-fcac-341a7fb929bd
---
J_seq, T_range = np.meshgrid(np.arange(J), np.arange(T))

fig = plt.figure(figsize=[20, 20])

# plot the consumption mean over age and time
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(T_range, J_seq, Cmean_seq, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax1.set_title(r"Mean of consumption")
ax1.set_xlabel(r"t")
ax1.set_ylabel(r"j")

# plot the consumption variance over age and time
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(T_range, J_seq, Cvar_seq, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax2.set_title(r"Variance of consumption")
ax2.set_xlabel(r"t")
ax2.set_ylabel(r"j")

plt.show()
```

```{code-cell} ipython3
:id: 94996077-9966-442f-b1ce-9bd93fb850d4


```
