---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(mccall_with_sep_markov)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

+++

# Job Search III: Search with Separation and Markov Wages

```{include} _admonition/gpu.md
```

```{index} single: An Introduction to Job Search
```

```{contents} Contents
:depth: 2
```

This lecture builds on the job search model with separation presented in the
{doc}`previous lecture <mccall_model_with_separation>`.

The key difference is that wage offers now follow a {doc}`Markov chain <finite_markov>` rather than
being IID.

This modification adds persistence to the wage offer process, meaning that
today's wage offer provides information about tomorrow's offer.

This feature makes the model more realistic, as labor market conditions tend to
exhibit serial correlation over time.

In addition to what's in Anaconda, this lecture will need the following
libraries

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon jax
```

We use the following imports:

```{code-cell} ipython3
from quantecon.markov import tauchen
import jax.numpy as jnp
import jax
from jax import lax
from typing import NamedTuple
import matplotlib.pyplot as plt
from functools import partial
```

## Model setup

The setting is as follows:

- Each unemployed agent receives a wage offer $w$ from a finite set $\mathbb W$
- Wage offers follow a Markov chain with transition matrix $P$
- Jobs terminate with probability $\alpha$ each period (separation rate)
- Unemployed workers receive compensation $c$ per period
- Future payoffs are discounted by factor $\beta \in (0,1)$

### Decision problem

When unemployed and receiving wage offer $w$, the agent chooses between:

1. Accept offer $w$: Become employed at wage $w$
2. Reject offer: Remain unemployed, receive $c$, get new offer next period

The wage updates are as follows:

* If an unemployed agent rejects offer $w$, then their next offer is drawn from $P(w, \cdot)$
* If an employed agent loses a job in which they were paid wage $w$, then their next offer is drawn from $P(w, \cdot)$

### The wage offer process

To construct the wage offer process we start with an AR1 process.

$$
    X_{t+1} = \rho X_t + \nu Z_{t+1}
$$

where $\{Z_t\}$ is IID and standard normal.


Below we will always choose $\rho \in (0, 1)$.

This means that the wage process will be positively correlated: the higher the current
wage offer, the more likely we are to get a high offer tomorrow.

To go from the AR1 process to the wage offer process, we set $W_t = \exp(X_t)$.

Actually, in practice, we approximate this wage process as follows:

* discretize the AR1 process using {ref}`Tauchen's method <fm_ex3>` and
* take the exponential of the resulting wage offer values.




### Value functions

We let

- $v_u(w)$ be the value of being unemployed when current wage offer is $w$
- $v_e(w)$ be the value of being employed at wage $w$

The Bellman equations are obvious modifications of the {doc}`IID case <mccall_model_with_separation>`.

The only change is that expectations for next period are computed using the transition matrix $P$ conditioned on current wage $w$, instead of being drawn independently from $q$.

The unemployed worker's value function satisfies the Bellman equation

$$
    v_u(w) = \max
        \left\{
            v_e(w), u(c) + \beta \sum_{w'} v_u(w') P(w,w')
        \right\}
$$

The employed worker's value function satisfies the Bellman equation

$$
    v_e(w) = 
    u(w) + \beta
    \left[
        \alpha \sum_{w'} v_u(w') P(w,w') + (1-\alpha) v_e(w)
    \right]
$$

As a matter of notation, given a function $h$ assigning values to wages, it is common to set

$$
    (Ph)(w) = \sum_{w'} h(w') P(w,w')
$$

(To understand this expression, think of $P$ as a matrix, $h$ as a column vector, and $w$ as a row index.)

With this notation, the Bellman equations become

$$
    v_u(w) = \max\{v_e(w), u(c) + \beta (P v_u)(w)\}
$$

and

$$
    v_e(w) = 
    u(w) + \beta
    \left[
        \alpha (P v_u)(w) + (1-\alpha) v_e(w)
    \right]
$$

+++

### Optimal policy

Once we have the solutions $v_e$ and $v_u$ to these Bellman equations, we can compute the optimal policy: accept at current wage offer $w$ if 

$$
    v_e(w) \geq u(c) + \beta (P v_u)(w)
$$

The optimal policy turns out to be a reservation wage strategy: accept all wages above some threshold.

+++


## Code

Let's now implement the model.

### Set up

The default utility function is a CRRA utility function

```{code-cell} ipython3
def u(x, γ):
    return (x**(1 - γ) - 1) / (1 - γ)
```

Let's set up a `Model` class to store information needed to solve the model.

We include `P_cumsum`, the row-wise cumulative sum of the transition matrix, to
optimize simulation -- the details are explained below.

```{code-cell} ipython3
class Model(NamedTuple):
    n: int
    w_vals: jnp.ndarray
    P: jnp.ndarray
    P_cumsum: jnp.ndarray  
    β: float
    c: float
    α: float
    γ: float
```

The next function holds default values and creates a `Model` instance:

```{code-cell} ipython3
def create_js_with_sep_model(
        n: int = 200,          # wage grid size
        ρ: float = 0.9,        # wage persistence
        ν: float = 0.2,        # wage volatility
        β: float = 0.96,       # discount factor
        α: float = 0.05,       # separation rate
        c: float = 1.0,        # unemployment compensation
        γ: float = 1.5         # utility parameter
    ) -> Model:
    """
    Creates an instance of the job search model with separation.

    """
    mc = tauchen(n, ρ, ν)
    w_vals, P = jnp.exp(jnp.array(mc.state_values)), jnp.array(mc.P)
    P_cumsum = jnp.cumsum(P, axis=1)
    return Model(n, w_vals, P, P_cumsum, β, c, α, γ)
```


### Solution: first pass

Let's put together a (not very efficient) routine for calculating the
reservation wage.

(We will think carefully about efficiency below.)

It works by starting with guesses for $v_e$ and $v_u$ and iterating to
convergence.

Here are Bellman operators that update $v_u$ and $v_e$ respectively.


```{code-cell} ipython3
def T_u(model, v_u, v_e):
    """
    Apply the unemployment Bellman update rule and return new guess of v_u.

    """
    n, w_vals, P, P_cumsum, β, c, α, γ = model
    h = u(c, γ) + β * P @ v_u 
    v_u_new = jnp.maximum(v_e, h)
    return v_u_new
```

```{code-cell} ipython3
def T_e(model, v_u, v_e):
    """
    Apply the employment Bellman update rule and return new guess of v_e.

    """
    n, w_vals, P, P_cumsum, β, c, α, γ = model
    v_e_new = u(w_vals, γ) + β * ((1 - α) * v_e + α * P @ v_u)
    return v_e_new
```

Here's a routine to iterate to convergence and then compute the reservation
wage.

```{code-cell} ipython3
def solve_model_first_pass(
        model: Model,           # instance containing default parameters
        v_u_init: jnp.ndarray,  # initial condition for v_u
        v_e_init: jnp.ndarray,  # initial condition for v_e
        tol: float=1e-6,        # error tolerance
        max_iter: int=1_000,    # maximum number of iterations for loop
    ):
    n, w_vals, P, P_cumsum, β, c, α, γ = model
    i = 0
    error = tol + 1 
    v_u = v_u_init
    v_e = v_e_init
    
    while i < max_iter and error > tol:
        v_u_next = T_u(model, v_u, v_e)
        v_e_next = T_e(model, v_u, v_e)
        error_u = jnp.max(jnp.abs(v_u_next - v_u))
        error_e = jnp.max(jnp.abs(v_e_next - v_e))
        error = jnp.maximum(error_u, error_e)
        v_u = v_u_next
        v_e = v_e_next
        i += 1

    # Compute accept and reject values
    continuation_values = u(c, γ) + β * P @ v_u

    # Find where acceptance becomes optimal
    accept_indices = v_e >= continuation_values
    first_accept_idx = jnp.argmax(accept_indices)  # index of first True

    # If no acceptance (all False), return infinity
    # Otherwise return the wage at the first acceptance index
    w_bar = jnp.where(
        jnp.any(accept_indices), w_vals[first_accept_idx], jnp.inf
    )
    return v_u, v_e, w_bar
```


### Road test

Let's solve the model:

```{code-cell} ipython3
model = create_js_with_sep_model()
n, w_vals, P, P_cumsum, β, c, α, γ = model
v_u_init = jnp.zeros(n)
v_e_init = jnp.zeros(n)
v_u, v_e, w_bar_first = solve_model_first_pass(model, v_u_init, v_e_init)
```

Next we compute the continuation values.

```{code-cell} ipython3
h = u(c, γ) + β * P @ v_u
```

Let's plot our results.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(w_vals, h, 'g-', linewidth=2, 
        label="continuation value function $h$")
ax.plot(w_vals, v_e, 'b-', linewidth=2, 
        label="employment value function $v_e$")
ax.legend(frameon=False)
ax.set_xlabel(r"$w$")
plt.show()
```

The reservation wage is at the intersection of $v_e$, and the continuation value
function, which is the value of rejecting.


## Improving efficiency

The solution method described above works fine but we can do much better.

First, we use the employed worker's Bellman equation to express
$v_e$ in terms of $Pv_u$

$$
    v_e(w) = 
    \frac{1}{1-\beta(1-\alpha)} \cdot (u(w) + \alpha\beta(Pv_u)(w))
$$

Next we substitute into the unemployed agent's Bellman equation to get

+++

$$
    v_u(w) = 
    \max
    \left\{
        \frac{1}{1-\beta(1-\alpha)} \cdot (u(w) + \alpha\beta(Pv_u)(w)),
        u(c) + \beta(Pv_u)(w)
    \right\}
$$

Then we use value function iteration to solve for $v_u$.

With $v_u$ in hand, we can recover $v_e$ through the equations above and
then compute the reservation wage.

Here's the new Bellman operator for the unemployed worker's value function:

```{code-cell} ipython3
def T(v: jnp.ndarray, model: Model) -> jnp.ndarray:
    """
    The Bellman operator for v_u.

    """
    n, w_vals, P, P_cumsum, β, c, α, γ = model
    d = 1 / (1 - β * (1 - α))
    v_e = d * (u(w_vals, γ) + α * β * P @ v)
    h = u(c, γ) + β * P @ v
    return jnp.maximum(v_e, h)
```

Here's a routine for value function iteration.

```{code-cell} ipython3
@jax.jit
def vfi(
        model: Model,
        tolerance: float = 1e-6,   # Error tolerance
        max_iter: int = 100_000,   # Max iteration bound
    ):

    v_init = jnp.zeros(model.w_vals.shape)

    def cond(loop_state):
        v, error, i = loop_state
        return (error > tolerance) & (i <= max_iter)

    def update(loop_state):
        v, error, i = loop_state
        v_new = T(v, model)
        error = jnp.max(jnp.abs(v_new - v))
        new_loop_state = v_new, error, i + 1
        return new_loop_state

    initial_state = (v_init, tolerance + 1, 1)
    final_loop_state = lax.while_loop(cond, update, initial_state)
    v_final, error, i = final_loop_state

    return v_final
```

Here is a routine that computes the reservation wage from the value function.

```{code-cell} ipython3
@jax.jit
def get_reservation_wage(v: jnp.ndarray, model: Model) -> float:
    """
    Calculate the reservation wage from the unemployed agents 
    value function v := v_u.

    The reservation wage is the lowest wage w where accepting (v_e(w))
    is at least as good as rejecting (u(c) + β(Pv_u)(w)).

    """
    n, w_vals, P, P_cumsum, β, c, α, γ = model

    # Compute accept and reject values
    d = 1 / (1 - β * (1 - α))
    v_e = d * (u(w_vals, γ) + α * β * P @ v)
    continuation_values = u(c, γ) + β * P @ v

    # Find where acceptance becomes optimal
    accept_indices = v_e >= continuation_values
    first_accept_idx = jnp.argmax(accept_indices)  # index of first True

    # If no acceptance (all False), return infinity
    # Otherwise return the wage at the first acceptance index
    return jnp.where(jnp.any(accept_indices), w_vals[first_accept_idx], jnp.inf)
```


Let's solve the model using our new method:

```{code-cell} ipython3
model = create_js_with_sep_model()
n, w_vals, P, P_cumsum, β, c, α, γ = model
v_u = vfi(model)
w_bar = get_reservation_wage(v_u, model)
```

Let's verify that both methods produce the same reservation wage:

```{code-cell} ipython3
print(f"Reservation wage (first method):  {w_bar_first:.6f}")
print(f"Reservation wage (second method): {w_bar:.6f}")
print(f"Difference: {abs(w_bar - w_bar_first):.2e}")
```

Next we compute some related quantities for plotting.

```{code-cell} ipython3
d = 1 / (1 - β * (1 - α))
v_e = d * (u(w_vals, γ) + α * β * P @ v_u)
h = u(c, γ) + β * P @ v_u
```

Let's plot our results.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(w_vals, h, 'g-', linewidth=2, 
        label="continuation value function $h$")
ax.plot(w_vals, v_e, 'b-', linewidth=2, 
        label="employment value function $v_e$")
ax.legend(frameon=False)
ax.set_xlabel(r"$w$")
plt.show()
```

The result is the same as before but we only iterate on one array --- and also
our JAX code is more efficient.


## Sensitivity analysis

Let's examine how reservation wages change with the separation rate.

```{code-cell} ipython3
α_vals: jnp.ndarray = jnp.linspace(0.0, 1.0, 10)

w_bar_vec = []
for α in α_vals:
    model = create_js_with_sep_model(α=α)
    v_u = vfi(model)
    w_bar = get_reservation_wage(v_u, model)
    w_bar_vec.append(w_bar)

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(
    α_vals, w_bar_vec, linewidth=2, alpha=0.6, label="reservation wage"
)
ax.legend(frameon=False)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$w$")
plt.show()
```

Can you provide an intuitive economic story behind the outcome that you see in this figure?

+++

## Employment simulation

Now let's simulate the employment dynamics of a single agent under the optimal policy.

Note that, when simulating the Markov chain for wage offers, we need to draw from the distribution in each
row of $P$ many times.

To do this, we use the inverse
transform method: draw a uniform random variable and find where it falls in the
cumulative distribution.

This is implemented via `jnp.searchsorted` on the precomputed cumulative sum
`P_cumsum`, which is much faster than recomputing the cumulative sum each time.

The function `update_agent` advances the agent's state by one period.

The agent's state is a pair $(S_t, W_t)$, where $S_t$ is employment status (0 if
unemployed, 1 if employed) and $W_t$ is 

* their current wage offer, if unemployed, or
* their current wage, if employed. 

```{code-cell} ipython3
def update_agent(key, status, wage_idx, model, w_bar):
    """
    Updates an agent's employment status and current wage.

    Parameters:
    - key: JAX random key
    - status: Current employment status (0 or 1)
    - wage_idx: Current wage, recorded as an array index
    - model: Model instance
    - w_bar: Reservation wage

    """
    n, w_vals, P, P_cumsum, β, c, α, γ = model

    key1, key2 = jax.random.split(key)
    # Use precomputed cumulative sum for efficient sampling
    # via the inverse transform method.
    new_wage_idx = jnp.searchsorted(
        P_cumsum[wage_idx, :], jax.random.uniform(key1)
    )
    separation_occurs = jax.random.uniform(key2) < α
    # Accept if current wage meets or exceeds reservation wage
    accepts = w_vals[wage_idx] >= w_bar

    # If employed: status = 1 if no separation, 0 if separation
    # If unemployed: status = 1 if accepts, 0 if rejects
    next_status = jnp.where(
        status,
        1 - separation_occurs.astype(jnp.int32),  # employed path
        accepts.astype(jnp.int32)                 # unemployed path
    )

    # If employed: wage = current if no separation, new if separation
    # If unemployed: wage = current if accepts, new if rejects
    next_wage = jnp.where(
        status,
        jnp.where(separation_occurs, new_wage_idx, wage_idx),  # employed path
        jnp.where(accepts, wage_idx, new_wage_idx)             # unemployed path
    )

    return next_status, next_wage
```

Here's a function to simulate the employment path of a single agent.

```{code-cell} ipython3
def simulate_employment_path(
        model: Model,     # Model details
        w_bar: float,    # Reservation wage
        T: int = 2_000,   # Simulation length
        seed: int = 42    # Set seed for simulation
    ):
    """
    Simulate employment path for T periods starting from unemployment.

    """
    key = jax.random.PRNGKey(seed)
    # Unpack model
    n, w_vals, P, P_cumsum, β, c, α, γ = model

    # Initial conditions
    status = 0
    wage_idx = 0

    wage_path = []
    status_path = []

    for t in range(T):
        wage_path.append(w_vals[wage_idx])
        status_path.append(status)

        key, subkey = jax.random.split(key)
        status, wage_idx = update_agent(
            subkey, status, wage_idx, model, w_bar
        )

    return jnp.array(wage_path), jnp.array(status_path)
```

Let's create a comprehensive plot of the employment simulation:

```{code-cell} ipython3
model = create_js_with_sep_model()

# Calculate reservation wage for plotting
v_u = vfi(model)
w_bar = get_reservation_wage(v_u, model)

wage_path, employment_status = simulate_employment_path(model, w_bar)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))

# Plot employment status
ax1.plot(employment_status, 'b-', alpha=0.7, linewidth=1)
ax1.fill_between(
    range(len(employment_status)), employment_status, alpha=0.3, color='blue'
)
ax1.set_ylabel('employment status')
ax1.set_title('Employment path (0=unemployed, 1=employed)')
ax1.set_xticks((0, 1))
ax1.set_ylim(-0.1, 1.1)

# Plot wage path with employment status coloring
ax2.plot(wage_path, 'b-', alpha=0.7, linewidth=1)
ax2.axhline(y=w_bar, color='black', linestyle='--', alpha=0.8,
           label=f'Reservation wage: {w_bar:.2f}')
ax2.set_xlabel('time')
ax2.set_ylabel('wage')
ax2.set_title('Wage path (actual and offers)')
ax2.legend()

# Plot cumulative fraction of time unemployed
unemployed_indicator = (employment_status == 0).astype(int)
cumulative_unemployment = (
    jnp.cumsum(unemployed_indicator) /
    jnp.arange(1, len(employment_status) + 1)
)

ax3.plot(cumulative_unemployment, 'r-', alpha=0.8, linewidth=2)
ax3.axhline(y=jnp.mean(unemployed_indicator), color='black',
            linestyle='--', alpha=0.7,
            label=f'Final rate: {jnp.mean(unemployed_indicator):.3f}')
ax3.set_xlabel('time')
ax3.set_ylabel('cumulative unemployment rate')
ax3.set_title('Cumulative fraction of time spent unemployed')
ax3.legend()
ax3.set_ylim(0, 1)

plt.tight_layout()
plt.show()
```

The simulation helps to visualize outcomes associated with this model.

The agent follows a reservation wage strategy.

Often the agent loses her job and immediately takes another job at a different
wage.

This is because she uses the wage $w$ from her last job to draw a new wage offer
via $P(w, \cdot)$, and positive correlation means that a high current $w$ is
often leads a high new draw.

+++

## Ergodic property

Below we examine cross-sectional unemployment.

In particular, we will look at the unemployment rate in a cross-sectional
simulation and compare it to the time-average unemployment rate, which is the
fraction of time an agent spends unemployed over a long time series.

We will see that these two values are approximately equal -- in fact they are
exactly equal in the limit.

The reason is that the process $(S_t, W_t)$, where

- $S_t$ is the employment status and
- $W_t$ is the wage 

is Markovian, since the next pair depends only on the current pair and iid
randomness, and ergodic. 

Ergodicity holds as a result of irreducibility.

Indeed, from any (status, wage) pair, an agent can eventually reach any other (status, wage) pair.

This holds because:

- Unemployed agents can become employed by accepting offers
- Employed agents can become unemployed through separation (probability $\alpha$)
- The wage process can transition between all wage states (because $P$ is itself irreducible)

These properties ensure the chain is ergodic with a unique stationary distribution $\pi$ over states $(s, w)$.

For an ergodic Markov chain, the ergodic theorem guarantees that time averages = cross-sectional averages.

In particular, the fraction of time a single agent spends unemployed (across all
wage states) converges to the cross-sectional unemployment rate:

$$
    \lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\{S_t = \text{unemployed}\} = \sum_{w=1}^{n} \pi(\text{unemployed}, w)
$$

This holds regardless of initial conditions -- provided that we burn in the
cross-sectional distribution (run it forward in time from a given initial cross
section in order to remove the influence of that initial condition).

As a result, we can study steady-state unemployment either by:

- Following one agent for a long time (time average), or
- Observing many agents at a single point in time (cross-sectional average)

Often the second approach is better for our purposes, since it's easier to parallelize.

+++

## Cross-sectional analysis

Now let's simulate many agents simultaneously to examine the cross-sectional unemployment rate.

To do this efficiently, we need a different approach than `simulate_employment_path` defined above.

The key differences are:

- `simulate_employment_path` records the entire history (all T periods) for a single agent, which is useful for visualization but memory-intensive
- The new function `sim_agent` below only tracks and returns the final state, which is all we need for cross-sectional statistics
- `sim_agent` uses `lax.fori_loop` instead of a Python loop, making it JIT-compilable and suitable for vectorization across many agents

We first define a function that simulates a single agent forward T time steps:

```{code-cell} ipython3
@jax.jit
def sim_agent(key, initial_status, initial_wage_idx, model, w_bar, T):
    """
    Simulate a single agent forward T time steps using lax.fori_loop.

    Uses fold_in to generate a new key at each time step.

    Parameters:
    - key: JAX random key for this agent
    - initial_status: Initial employment status (0 or 1)
    - initial_wage_idx: Initial wage index
    - model: Model instance
    - w_bar: Reservation wage
    - T: Number of time periods to simulate

    Returns:
    - final_status: Employment status after T periods
    - final_wage_idx: Wage index after T periods
    """
    def update(t, loop_state):
        status, wage_idx = loop_state
        step_key = jax.random.fold_in(key, t)
        status, wage_idx = update_agent(step_key, status, wage_idx, model, w_bar)
        return status, wage_idx

    initial_loop_state = (initial_status, initial_wage_idx)
    final_loop_state = lax.fori_loop(0, T, update, initial_loop_state)
    final_status, final_wage_idx = final_loop_state
    return final_status, final_wage_idx


# Create vectorized version of sim_agent to process multiple agents in parallel
sim_agents_vmap = jax.vmap(sim_agent, in_axes=(0, 0, 0, None, None, None))


def simulate_cross_section(
        model: Model,               # Model instance with parameters
        n_agents: int = 100_000,    # Number of agents to simulate
        T: int = 200,               # Length of burn-in
        seed: int = 42              # For reproducibility
    ) -> float:
    """
    Simulate cross-section of agents and return unemployment rate.

    This approach:
    1. Generates n_agents random keys
    2. Calls sim_agent for each agent (vectorized via vmap)
    3. Collects the final states to produce the cross-section

    Returns the cross-sectional unemployment rate.
    """
    key = jax.random.PRNGKey(seed)

    # Solve for optimal reservation wage
    v_u = vfi(model)
    w_bar = get_reservation_wage(v_u, model)

    # Initialize arrays
    initial_wage_indices = jnp.zeros(n_agents, dtype=jnp.int32)
    initial_status_vec = jnp.zeros(n_agents, dtype=jnp.int32)

    # Generate n_agents random keys
    agent_keys = jax.random.split(key, n_agents)

    # Simulate each agent forward T steps (vectorized)
    final_status, final_wage_idx = sim_agents_vmap(
        agent_keys, initial_status_vec, initial_wage_indices, model, w_bar, T
    )

    unemployment_rate = 1 - jnp.mean(final_status)
    return unemployment_rate
```

This function generates a histogram showing the distribution of employment status across many agents:

```{code-cell} ipython3
def plot_cross_sectional_unemployment(
        model: Model,
        t_snapshot: int = 200,    # Time of cross-sectional snapshot
        n_agents: int = 20_000    # Number of agents to simulate
    ):
    """
    Generate histogram of cross-sectional unemployment at a specific time.

    """
    # Get final employment state directly
    key = jax.random.PRNGKey(42)
    v_u = vfi(model)
    w_bar = get_reservation_wage(v_u, model)

    # Initialize arrays
    initial_wage_indices = jnp.zeros(n_agents, dtype=jnp.int32)
    initial_status_vec = jnp.zeros(n_agents, dtype=jnp.int32)

    # Generate n_agents random keys
    agent_keys = jax.random.split(key, n_agents)

    # Simulate each agent forward T steps (vectorized)
    final_status, _ = sim_agents_vmap(
        agent_keys, initial_status_vec, initial_wage_indices, model, w_bar, t_snapshot
    )

    # Calculate unemployment rate
    unemployment_rate = 1 - jnp.mean(final_status)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot histogram as density (bars sum to 1)
    weights = jnp.ones_like(final_status) / len(final_status)
    ax.hist(final_status, bins=[-0.5, 0.5, 1.5],
            alpha=0.7, color='blue', edgecolor='black',
            density=True, weights=weights)

    ax.set_xlabel('employment status (0=unemployed, 1=employed)')
    ax.set_ylabel('density')
    ax.set_title(f'Cross-sectional distribution at t={t_snapshot}, ' +
                 f'unemployment rate = {unemployment_rate:.3f}')
    ax.set_xticks([0, 1])

    plt.tight_layout()
    plt.show()
```

Now let's compare the time-average unemployment rate (from a single agent's long
simulation) with the cross-sectional unemployment rate (from many agents at a
single point in time).

We claimed above that these numbers will be approximately equal in large
samples, due to ergodicity.

Let's see if that's true.

```{code-cell} ipython3
model = create_js_with_sep_model()
cross_sectional_unemp = simulate_cross_section(
    model, n_agents=20_000, T=200
)

time_avg_unemp = jnp.mean(unemployed_indicator)
print(f"Time-average unemployment rate (single agent): "
      f"{time_avg_unemp:.4f}")
print(f"Cross-sectional unemployment rate (at t=200): "
      f"{cross_sectional_unemp:.4f}")
print(f"Difference: {abs(time_avg_unemp - cross_sectional_unemp):.4f}")
```

Indeed, they are very close.

Now let's visualize the cross-sectional distribution:

```{code-cell} ipython3
plot_cross_sectional_unemployment(model)
```

## Lower unemployment compensation (c=0.5)

What happens to the cross-sectional unemployment rate with lower unemployment compensation?

```{code-cell} ipython3
model_low_c = create_js_with_sep_model(c=0.5)
plot_cross_sectional_unemployment(model_low_c)
```

## Exercises

```{exercise-start}
:label: mmwsm_ex1
```

Create a plot that investigates more carefully how the steady state cross-sectional unemployment rate
changes with unemployment compensation.

Try a range of values for unemployment compensation `c`, such as `c = 0.2, 0.4, 0.6, 0.8, 1.0`.
For each value, compute the steady-state cross-sectional unemployment rate and plot it against `c`.

What relationship do you observe between unemployment compensation and the unemployment rate?

```{exercise-end}
```

```{solution-start} mmwsm_ex1
:class: dropdown
```

We compute the steady-state unemployment rate for different values of unemployment compensation:

```{code-cell} ipython3
c_values = 1.0, 0.8, 0.6, 0.4, 0.2
rates = []
for c in c_values:
    model = create_js_with_sep_model(c=c)
    unemployment_rate = simulate_cross_section(model)
    rates.append(unemployment_rate)

fig, ax = plt.subplots()
ax.plot(
    c_values, rates, alpha=0.8,
    linewidth=1.5, label='Steady-state unemployment rate'
)
ax.set_xlabel('unemployment compensation (c)')
ax.set_ylabel('unemployment rate')
ax.legend(frameon=False)
plt.show()
```

```{solution-end}
```
