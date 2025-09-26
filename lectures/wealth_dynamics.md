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

# Wealth Distribution Dynamics

```{contents} Contents
:depth: 2
```

```{seealso}
A version of this lecture using [JAX](https://github.com/jax-ml/jax) is {doc}`available here <jax:wealth_dynamics>`
```

```{admonition} GPU
:class: warning

This lecture includes implementation using JAX and can be accelerated with GPUs.

For Google Colab users, to enable GPU, go to Runtime → Change runtime type → Hardware accelerator → GPU.

For local users, see the [JAX installation guide](https://github.com/google/jax#installation) for local GPU installation instructions.
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon
```

## Overview

This notebook gives an introduction to wealth distribution dynamics, with a
focus on

* modeling and computing the wealth distribution via simulation,
* measures of inequality such as the Lorenz curve and Gini coefficient, and
* how inequality is affected by the properties of wage income and returns on assets.

One interesting property of the wealth distribution we discuss is {index}`Pareto tails`.

The wealth distribution in many countries exhibits a Pareto tail

* See {doc}`this lecture <intro:heavy_tails>` for a definition.
* For a review of the empirical evidence, see, for example, {cite}`benhabib2018skewed`.

This is consistent with high concentration of wealth amongst the richest households.

It also gives us a way to quantify such concentration, in terms of the tail index.

One question of interest is whether or not we can replicate Pareto tails from a relatively simple model.

### A note on assumptions

The evolution of wealth for any given household depends on their
savings behavior.

Modeling such behavior will form an important part of this lecture series.

However, in this particular lecture, we will be content with rather ad hoc (but plausible) savings rules.

We do this to more easily explore the implications of different specifications of income dynamics and investment returns.

At the same time, all of the techniques discussed here can be plugged into models that use optimization to obtain savings rules.

We will use the following imports.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe
import jax
import jax.numpy as jnp
import jax.random as jr
from collections import namedtuple
```

## Lorenz curves and the Gini coefficient

Before we investigate wealth dynamics, we briefly review some measures of
inequality.

### Lorenz curves

One popular graphical measure of inequality is the {index}`Lorenz curve`.

The package [QuantEcon.py](https://github.com/QuantEcon/QuantEcon.py), already imported above, contains a function to compute Lorenz curves.

To illustrate, suppose that

```{code-cell} ipython3
n = 10_000                      # size of sample
w = np.exp(np.random.randn(n))  # lognormal draws
```

is data representing the wealth of 10,000 households.

We can compute and plot the Lorenz curve as follows:

```{code-cell} ipython3
f_vals, l_vals = qe.lorenz_curve(w)

fig, ax = plt.subplots()
ax.plot(f_vals, l_vals, label='Lorenz curve, lognormal sample')
ax.plot(f_vals, f_vals, label='Lorenz curve, equality')
ax.legend()
plt.show()
```

This curve can be understood as follows: if point $(x,y)$ lies on the curve, it means that, collectively, the bottom $(100x)\%$ of the population holds $(100y)\%$ of the wealth.

The "equality" line is the 45 degree line (which might not be exactly 45
degrees in the figure, depending on the aspect ratio).

A sample that produces this line exhibits perfect equality.

The other line in the figure is the Lorenz curve for the lognormal sample, which deviates significantly from perfect equality.

For example, the bottom 80% of the population holds around 40% of total wealth.

Here is another example, which shows how the Lorenz curve shifts as the
underlying distribution changes.

We generate 10,000 observations using the Pareto distribution with a range of
parameters, and then compute the Lorenz curve corresponding to each set of
observations.

```{code-cell} ipython3
a_vals = (1, 2, 5)              # Pareto tail index
n = 10_000                      # size of each sample
fig, ax = plt.subplots()
for a in a_vals:
    u = np.random.uniform(size=n)
    y = u**(-1/a)               # distributed as Pareto with tail index a
    f_vals, l_vals = qe.lorenz_curve(y)
    ax.plot(f_vals, l_vals, label=f'$a = {a}$')
ax.plot(f_vals, f_vals, label='equality')
ax.legend()
plt.show()
```

You can see that, as the tail parameter of the Pareto distribution increases, inequality decreases.

This is to be expected, because a higher tail index implies less weight in the tail of the Pareto distribution.

### The Gini coefficient

The definition and interpretation of the Gini coefficient can be found on the corresponding [Wikipedia page](https://en.wikipedia.org/wiki/Gini_coefficient).

A value of 0 indicates perfect equality (corresponding the case where the
Lorenz curve matches the 45 degree line) and a value of 1 indicates complete
inequality (all wealth held by the richest household).

The [QuantEcon.py](https://github.com/QuantEcon/QuantEcon.py) library contains a function to calculate the Gini coefficient.

We can test it on the Weibull distribution with parameter $a$, where the Gini coefficient is known to be

$$
G = 1 - 2^{-1/a}
$$

Let's see if the Gini coefficient computed from a simulated sample matches
this at each fixed value of $a$.

```{code-cell} ipython3
a_vals = range(1, 20)
ginis = []
ginis_theoretical = []
n = 100

fig, ax = plt.subplots()
for a in a_vals:
    y = np.random.weibull(a, size=n)
    ginis.append(qe.gini_coefficient(y))
    ginis_theoretical.append(1 - 2**(-1/a))
ax.plot(a_vals, ginis, label='estimated gini coefficient')
ax.plot(a_vals, ginis_theoretical, label='theoretical gini coefficient')
ax.legend()
ax.set_xlabel("Weibull parameter $a$")
ax.set_ylabel("Gini coefficient")
plt.show()
```

The simulation shows that the fit is good.

## A model of wealth dynamics

Having discussed inequality measures, let us now turn to wealth dynamics.

The model we will study is

```{math}
:label: wealth_dynam_ah

w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
```

where

- $w_t$ is wealth at time $t$ for a given household,
- $r_t$ is the rate of return of financial assets,
- $y_t$ is current non-financial (e.g., labor) income and
- $s(w_t)$ is current wealth net of consumption

Letting $\{z_t\}$ be a correlated state process of the form

$$
z_{t+1} = a z_t + b + \sigma_z \epsilon_{t+1}
$$

we’ll assume that

$$
R_t := 1 + r_t = c_r \exp(z_t) + \exp(\mu_r + \sigma_r \xi_t)
$$

and

$$
y_t = c_y \exp(z_t) + \exp(\mu_y + \sigma_y \zeta_t)
$$

Here $\{ (\epsilon_t, \xi_t, \zeta_t) \}$ is IID and standard
normal in $\mathbb R^3$.

The value of $c_r$ should be close to zero, since rates of return
on assets do not exhibit large trends.

When we simulate a population of households, we will assume all shocks are idiosyncratic (i.e.,  specific to individual households and independent across them).

Regarding the savings function $s$, our default model will be

```{math}
:label: sav_ah

s(w) = s_0 w \cdot \mathbb 1\{w \geq \hat w\}
```

where $s_0$ is a positive constant.

Thus, for $w < \hat w$, the household saves nothing. For
$w \geq \bar w$, the household saves a fraction $s_0$ of
their wealth.

We are using something akin to a fixed savings rate model, while
acknowledging that low wealth households tend to save very little.

## Implementation

We use a NamedTuple to store the model parameters.

```{code-cell} ipython3
# Create a namedtuple to hold model parameters
WealthDynamics = namedtuple('WealthDynamics', [
    'w_hat',   # savings parameter
    's_0',     # savings parameter
    'c_y',     # labor income parameter
    'μ_y',     # labor income parameter
    'σ_y',     # labor income parameter
    'c_r',     # rate of return parameter
    'μ_r',     # rate of return parameter
    'σ_r',     # rate of return parameter
    'a',       # aggregate shock parameter
    'b',       # aggregate shock parameter
    'σ_z',     # aggregate shock parameter
    'z_mean',  # mean of z process
    'z_var',   # variance of z process
    'y_mean',  # mean of y process
    'R_mean'   # mean of R process
])
```

Here's a factory function to create WealthDynamics instances with computed
stationary moments and stability checks.

```{code-cell} ipython3
def create_wealth_dynamics(w_hat=1.0,
                          s_0=0.75,
                          c_y=1.0,
                          μ_y=1.0,
                          σ_y=0.2,
                          c_r=0.05,
                          μ_r=0.1,
                          σ_r=0.5,
                          a=0.5,
                          b=0.0,
                          σ_z=0.1):
    """
    Factory function to create a WealthDynamics instance.
    """
    # Record stationary moments
    z_mean = b / (1 - a)
    z_var = σ_z**2 / (1 - a**2)
    exp_z_mean = jnp.exp(z_mean + z_var / 2)
    R_mean = c_r * exp_z_mean + jnp.exp(μ_r + σ_r**2 / 2)
    y_mean = c_y * exp_z_mean + jnp.exp(μ_y + σ_y**2 / 2)
    
    # Test stability condition
    α = R_mean * s_0
    if α >= 1:
        raise ValueError("Stability condition failed.")
    
    return WealthDynamics(
        w_hat=w_hat, s_0=s_0, c_y=c_y, μ_y=μ_y, σ_y=σ_y,
        c_r=c_r, μ_r=μ_r, σ_r=σ_r, a=a, b=b, σ_z=σ_z,
        z_mean=z_mean, z_var=z_var, y_mean=y_mean, R_mean=R_mean
    )
```

Here are pure functions for updating wealth dynamics.

```{code-cell} ipython3
def update_states(wdy, w, z, key):
    """
    Update one period, given current wealth w and persistent state z.
    Returns new wealth and new persistent state.
    """
    key1, key2, key3 = jr.split(key, 3)
    
    # Update persistent state
    zp = wdy.a * z + wdy.b + wdy.σ_z * jr.normal(key1)
    
    # Generate income
    y = wdy.c_y * jnp.exp(zp) + jnp.exp(wdy.μ_y + wdy.σ_y * jr.normal(key2))
    
    # Update wealth
    wp = y
    wp = jnp.where(w >= wdy.w_hat,
                   wp + (wdy.c_r * jnp.exp(zp) + 
                         jnp.exp(wdy.μ_r + wdy.σ_r * jr.normal(key3))) * wdy.s_0 * w,
                   wp)
    
    return wp, zp
```

Here's a function to simulate the time series of wealth for individual households.

```{code-cell} ipython3
def wealth_time_series(wdy, w_0, n, key):
    """
    Generate a single time series of length n for wealth given
    initial value w_0.

    The initial persistent state z_0 for each household is drawn from
    the stationary distribution of the AR(1) process.

        * wdy: an instance of WealthDynamics
        * w_0: scalar
        * n: int
        * key: PRNGKey for random number generation
    """
    
    def scan_fn(carry, x):
        w, z, key = carry
        key, subkey = jr.split(key)
        w_new, z_new = update_states(wdy, w, z, subkey)
        return (w_new, z_new, key), w_new
    
    # Initialize
    key1, key2 = jr.split(key)
    z_0 = wdy.z_mean + jnp.sqrt(wdy.z_var) * jr.normal(key1)
    
    # Use scan to generate time series
    _, w_series = jax.lax.scan(scan_fn, (w_0, z_0, key2), jnp.arange(n-1))
    
    # Prepend initial value
    return jnp.concatenate([jnp.array([w_0]), w_series])

# JIT compile for performance
wealth_time_series = jax.jit(wealth_time_series)
```

Now here's a function to simulate a cross section of households forward in time.

Note the use of vectorization for parallel computation.

```{code-cell} ipython3
def update_cross_section(wdy, w_distribution, shift_length=500, key=None):
    """
    Shifts a cross-section of households forward in time

    * wdy: an instance of WealthDynamics
    * w_distribution: array_like, represents current cross-section
    * shift_length: number of periods to simulate forward
    * key: PRNGKey for random number generation

    Takes a current distribution of wealth values as w_distribution
    and updates each w_t in w_distribution to w_{t+j}, where
    j = shift_length.

    Returns the new distribution.
    """
    if key is None:
        key = jr.PRNGKey(42)
    
    num_households = len(w_distribution)
    
    def update_single_household(w_0, key):
        """Update a single household's wealth over shift_length periods"""
        
        def scan_fn(carry, x):
            w, z, subkey = carry
            subkey, new_key = jr.split(subkey)
            w_new, z_new = update_states(wdy, w, z, new_key)
            return (w_new, z_new, subkey), None
        
        # Initialize household's persistent state
        init_key, sim_key = jr.split(key)
        z_0 = wdy.z_mean + jnp.sqrt(wdy.z_var) * jr.normal(init_key)
        
        # Simulate forward
        (final_w, _, _), _ = jax.lax.scan(
            scan_fn, (w_0, z_0, sim_key), jnp.arange(shift_length)
        )
        return final_w
    
    # Vectorize over households
    keys = jr.split(key, num_households)
    update_household_vec = jax.vmap(update_single_household)
    
    return update_household_vec(w_distribution, keys)

# JIT compile for performance
update_cross_section = jax.jit(update_cross_section, static_argnums=(2,))
```

Parallelization is very effective in the function above because the time path
of each household can be calculated independently once the path for the
aggregate state is known.

## Applications

Let's try simulating the model at different parameter values and investigate
the implications for the wealth distribution.

### Time series

Let's look at the wealth dynamics of an individual household.

```{code-cell} ipython3
wdy = create_wealth_dynamics()
ts_length = 200
key = jr.PRNGKey(42)
w = wealth_time_series(wdy, wdy.y_mean, ts_length, key)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(w)
plt.show()
```

Notice the large spikes in wealth over time.

Such spikes are similar to what we observed in time series when {doc}`we studied Kesten processes <kesten_processes>`.

### Inequality measures

Let's look at how inequality varies with returns on financial assets.

The next function generates a cross section and then computes the Lorenz
curve and Gini coefficient.

```{code-cell} ipython3
def generate_lorenz_and_gini(wdy, num_households=100_000, T=500, key=None):
    """
    Generate the Lorenz curve data and gini coefficient corresponding to a
    WealthDynamics model by simulating num_households forward to time T.
    """
    if key is None:
        key = jr.PRNGKey(42)
    
    ψ_0 = jnp.full(num_households, wdy.y_mean)
    ψ_star = update_cross_section(wdy, ψ_0, shift_length=T, key=key)
    
    # Convert to numpy for QuantEcon functions
    ψ_star_np = np.array(ψ_star)
    return qe.gini_coefficient(ψ_star_np), qe.lorenz_curve(ψ_star_np)
```

Now we investigate how the Lorenz curves associated with the wealth distribution change as return to savings varies.

The code below plots Lorenz curves for three different values of $\mu_r$.

If you are running this yourself, note that it will take one or two minutes to execute.

This is unavoidable because we are executing a CPU intensive task.

In fact the code, which is JIT compiled and parallelized, runs extremely fast relative to the number of computations.

```{code-cell} ipython3
with qe.Timer():
    fig, ax = plt.subplots()
    μ_r_vals = (0.0, 0.025, 0.05)
    gini_vals = []
    key = jr.PRNGKey(42)

    for i, μ_r in enumerate(μ_r_vals):
        wdy = create_wealth_dynamics(μ_r=μ_r)
        key, subkey = jr.split(key)
        gv, (f_vals, l_vals) = generate_lorenz_and_gini(wdy, key=subkey)
        ax.plot(f_vals, l_vals, label=fr'$\psi^*$ at $\mu_r = {μ_r:0.2}$')
        gini_vals.append(gv)

    ax.plot(f_vals, f_vals, label='equality')
    ax.legend(loc="upper left")
    plt.show()
```

The Lorenz curve shifts downwards as returns on financial income rise, indicating a rise in inequality.

We will look at this again via the Gini coefficient immediately below, but
first consider the following image of our system resources when the code above
is executing:

Since the code is both efficiently JIT compiled and fully parallelized, it's
close to impossible to make this sequence of tasks run faster without changing
hardware.

Now let's check the Gini coefficient.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(μ_r_vals, gini_vals, label='gini coefficient')
ax.set_xlabel(r"$\mu_r$")
ax.legend()
plt.show()
```

Once again, we see that inequality increases as returns on financial income
rise.

Let's finish this section by investigating what happens when we change the
volatility term $\sigma_r$ in financial returns.


```{code-cell} ipython3
with qe.Timer():
    fig, ax = plt.subplots()
    σ_r_vals = (0.35, 0.45, 0.52)
    gini_vals = []
    key = jr.PRNGKey(42)

    for σ_r in σ_r_vals:
        wdy = create_wealth_dynamics(σ_r=σ_r)
        key, subkey = jr.split(key)
        gv, (f_vals, l_vals) = generate_lorenz_and_gini(wdy, key=subkey)
        ax.plot(f_vals, l_vals, label=fr'$\psi^*$ at $\sigma_r = {σ_r:0.2}$')
        gini_vals.append(gv)

    ax.plot(f_vals, f_vals, label='equality')
    ax.legend(loc="upper left")
    plt.show()
```

We see that greater volatility has the effect of increasing inequality in this model.

## Exercises

```{exercise}
:label: wd_ex1

For a wealth or income distribution with Pareto tail, a higher tail index suggests lower inequality.

Indeed, it is possible to prove that the Gini coefficient of the Pareto
distribution with tail index $a$ is $1/(2a - 1)$.

To the extent that you can, confirm this by simulation.

In particular, generate a plot of the Gini coefficient against the tail index
using both the theoretical value just given and the value computed from a sample via `qe.gini_coefficient`.

For the values of the tail index, use `a_vals = np.linspace(1, 10, 25)`.

Use sample of size 1,000 for each $a$ and the sampling method for generating Pareto draws employed in the discussion of Lorenz curves for the Pareto distribution.

To the extent that you can, interpret the monotone relationship between the
Gini index and $a$.
```

```{solution-start} wd_ex1
:class: dropdown
```

Here is one solution, which produces a good match between theory and
simulation.

```{code-cell} ipython3
a_vals = np.linspace(1, 10, 25)  # Pareto tail index
ginis = np.empty_like(a_vals)

n = 1000                         # size of each sample
fig, ax = plt.subplots()
for i, a in enumerate(a_vals):
    y = np.random.uniform(size=n)**(-1/a)
    ginis[i] = qe.gini_coefficient(y)
ax.plot(a_vals, ginis, label='sampled')
ax.plot(a_vals, 1/(2*a_vals - 1), label='theoretical')
ax.legend()
plt.show()
```

In general, for a Pareto distribution, a higher tail index implies less weight
in the right hand tail.

This means less extreme values for wealth and hence more equality.

More equality translates to a lower Gini index.

```{solution-end}
```

```{exercise-start}
:label: wd_ex2
```

The wealth process {eq}`wealth_dynam_ah` is similar to a {doc}`Kesten process <kesten_processes>`.

This is because, according to {eq}`sav_ah`, savings is constant for all wealth levels above $\hat w$.

When savings is constant, the wealth process has the same quasi-linear
structure as a Kesten process, with multiplicative and additive shocks.

The Kesten--Goldie theorem tells us that Kesten processes have Pareto tails under a range of parameterizations.

The theorem does not directly apply here, since savings is not always constant and since the multiplicative and additive terms in {eq}`wealth_dynam_ah` are not IID.

At the same time, given the similarities, perhaps Pareto tails will arise.

To test this, run a simulation that generates a cross-section of wealth and
generate a rank-size plot.

If you like, you can use the function `rank_size` from the `quantecon` library (documentation [here](https://quanteconpy.readthedocs.io/en/latest/tools/inequality.html#quantecon.inequality.rank_size)).

In viewing the plot, remember that Pareto tails generate a straight line.  Is
this what you see?

For sample size and initial conditions, use

```{code-cell} ipython3
num_households = 250_000
T = 500                                        # shift forward T periods
wdy = create_wealth_dynamics()
ψ_0 = jnp.full(num_households, wdy.y_mean)    # initial distribution
key = jr.PRNGKey(42)
```

```{exercise-end}
```

```{solution-start} wd_ex2
:class: dropdown
```

First let's generate the distribution:

```{code-cell} ipython3
num_households = 250_000
T = 500  # how far to shift forward in time
wdy = create_wealth_dynamics()
ψ_0 = jnp.full(num_households, wdy.y_mean)
key = jr.PRNGKey(42)

ψ_star = update_cross_section(wdy, ψ_0, shift_length=T, key=key)
```

Now let's see the rank-size plot:

```{code-cell} ipython3
fig, ax = plt.subplots()

# Convert to numpy for QuantEcon function
ψ_star_np = np.array(ψ_star)
rank_data, size_data = qe.rank_size(ψ_star_np, c=0.001)
ax.loglog(rank_data, size_data, 'o', markersize=3.0, alpha=0.5)
ax.set_xlabel("log rank")
ax.set_ylabel("log size")

plt.show()
```

```{solution-end}
```
