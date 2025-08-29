---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Forecasting  an AR(1) Process

```{code-cell} ipython3
:tags: [hide-output]

!pip install arviz pymc
```

This lecture describes methods for forecasting statistics that are functions of future values of a univariate autogressive process.  

The methods are designed to take into account two possible sources of uncertainty about these statistics:

- random shocks that impinge of the transition law

- uncertainty about the parameter values of the AR(1) process

We consider two sorts of statistics:

- prospective values $y_{t+j}$ of a random process $\{y_t\}$ that is governed by the AR(1) process

- sample path properties that are defined as non-linear functions of future values $\{y_{t+j}\}_{j \geq 1}$ at time $t$

**Sample path properties** are things like "time to next turning point" or "time to next recession".

To investigate sample path properties we'll use a simulation procedure recommended by Wecker {cite}`wecker1979predicting`.

To acknowledge uncertainty about parameters, we'll deploy `pymc` to construct a Bayesian joint posterior distribution for unknown parameters.

Let's start with some imports.

```{code-cell} ipython3
import numpy as np
import arviz as az
import pymc as pmc
import matplotlib.pyplot as plt
import seaborn as sns

# jax
import jax.random as random
import jax.numpy as jnp

# numpyro
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

sns.set_style('white')
colors = sns.color_palette()
key = random.PRNGKey(0)
```

## A Univariate First-Order Autoregressive Process

Consider the univariate AR(1) model: 

$$ 
y_{t+1} = \rho y_t + \sigma \epsilon_{t+1}, \quad t \geq 0 
$$ (ar1-tp-eq1) 

where the scalars $\rho$ and $\sigma$ satisfy $|\rho| < 1$ and $\sigma > 0$; 
$\{\epsilon_{t+1}\}$ is a sequence of i.i.d. normal random variables with mean $0$ and variance $1$. 

The initial condition $y_{0}$ is a known number. 

Equation {eq}`ar1-tp-eq1` implies that for $t \geq 0$, the conditional density of $y_{t+1}$ is

$$
f(y_{t+1} | y_{t}; \rho, \sigma) \sim {\mathcal N}(\rho y_{t}, \sigma^2) \
$$ (ar1-tp-eq2)

Further, equation {eq}`ar1-tp-eq1` also implies that for $t \geq 0$, the conditional density of $y_{t+j}$ for $j \geq 1$ is

$$
f(y_{t+j} | y_{t}; \rho, \sigma) \sim {\mathcal N}\left(\rho^j y_{t}, \sigma^2 \frac{1 - \rho^{2j}}{1 - \rho^2} \right) 
$$ (ar1-tp-eq3)

The predictive distribution {eq}`ar1-tp-eq3` that assumes that the parameters $\rho, \sigma$ are known, which we express
by conditioning on them.

We also want to compute a  predictive distribution that does not condition on $\rho,\sigma$ but instead takes account of our uncertainty about them.

We form this predictive distribution by integrating {eq}`ar1-tp-eq3` with respect to a joint posterior distribution $\pi_t(\rho,\sigma | y^t)$ 
that conditions on an observed history $y^t = \{y_s\}_{s=0}^t$:

$$ 
\begin{aligned}
f(y_{t+j} | y^t)  
= \int f(y_{t+j} | y^t, \rho, \sigma) \pi_t(\rho,\sigma | y^t ) d \rho d \sigma
= \int f(y_{t+j} | y_t, \rho, \sigma) \pi_t(\rho,\sigma | y^t ) d \rho d \sigma
\end{aligned}
$$ (ar1-tp-eq4)

Predictive distribution {eq}`ar1-tp-eq3` assumes that parameters $(\rho,\sigma)$ are known. 

Predictive distribution {eq}`ar1-tp-eq4` assumes that parameters $(\rho,\sigma)$ are uncertain, but have known probability distribution $\pi_t(\rho,\sigma | y^t )$. Notice the second equality follows that $\{y_t\}$ is a AR(1) process when $\(\rho, \sigma\)$ are given.  

We also want to compute some  predictive distributions of "sample path statistics" that might  include, for example

- the time until the next "recession",
- the minimum value of $Y$ over the next 8 periods,
- "severe recession", and
- the time until the next turning point (positive or negative).

To accomplish that for situations in which we are uncertain about parameter values, we shall extend Wecker's {cite}`wecker1979predicting` approach in the following way.

- first simulate an initial path of length $T_0$;
- for a given prior, draw a sample of size $N$ from the posterior joint distribution of parameters $\left(\rho,\sigma\right)$ after observing the initial path;
- for each draw $n=0,1,...,N$, simulate a "future path" of length $T_1$ with parameters $\left(\rho_n,\sigma_n\right)$ and compute our three "sample path statistics";
- finally, plot the desired statistics from the $N$ samples as an empirical distribution.

## Implementation

First, we'll simulate a  sample path from which to launch our forecasts.  

In addition to plotting the sample path, under the assumption that the true parameter values are known,
we'll plot $.9$ and $.95$ coverage intervals using conditional distribution
{eq}`ar1-tp-eq3` described above. 

We'll also plot a bunch of samples of sequences of future values and watch where they fall relative to the coverage interval.  

```{code-cell} ipython3
class AR1(NamedTuple):
    """
    Represents a univariate first-order autoregressive (AR(1)) process.

    Parameters
    ----------
    rho : float
        Autoregressive coefficient, must satisfy |rho| < 1 for stationarity.
    sigma : float
        Standard deviation of the error term.
    y0 : float
        Initial value of the process at time t=0.
    T0 : int, optional
        Length of the initial observed path (default is 100).
    T1 : int, optional
        Length of the future path to simulate (default is 100).
    """
    rho:float
    sigma:float
    y0: float
    T0: int=100
    T1: int=100


def AR1_simulate(ar1: AR1, T, subkey):
    """
    Simulate a realization of the AR(1) process for a given number of periods.

    Parameters
    ----------
    ar1 : AR1
        AR1 named tuple containing process parameters (rho, sigma, y0).
    T : int
        Number of time steps to simulate.
    subkey : jax.random.PRNGKey
        JAX random key for generating random noise.

    Returns
    -------
    y : jax.numpy.ndarray
        Simulated path of the AR(1) process of length T.
    """
    # ...function code...
    rho, sigma, y0 = ar.rho, ar1.sigma, ar1.y0
    # Allocate space and draw epsilons
    y = jnp.empty(T)
    eps = ar1.sigma * random.normal(subkey, (T,))

    # Initial condition and step forward
    y.at[0].set(y0)
    for t in range(1, T):
        y.at[t].set(ar1.rho * y[t-1] + eps[t])
        
    return jnp.asarray(y)


def plot_path(ar1, initial_path, key, ax):
    """Plot the initial path and the preceding predictive densities"""
    rho, sigma, T0, T1 = ar1.rho, ar1.sigma, ar1.T0, ar1.T1
    
    # Compute moments and confidence intervals
    y_T0 = initial_path[-1]
    center = jnp.array([rho**j * y_T0 for j in range(T1)])
    vars = jnp.array(
        [sigma**2 * (1 - rho**(2 * j)) / (1 - rho**2) for j in range(T1)]
        )

    y_upper_c95 = center + 1.96 * jnp.sqrt(vars)
    y_lower_c95 = center - 1.96 * jnp.sqrt(vars)

    y_upper_c90 = center + 1.65 * jnp.sqrt(vars)
    y_lower_c90 = center - 1.65 * jnp.sqrt(vars)

    # Plot
    ax.plot(jnp.arange(-T0 + 1, 1), initial_path)
    ax.set_xlim([-T0, T1])
    ax.axvline(0, linestyle='--', alpha=.4, color='k', lw=1)

    # Simulate 10 future paths
    subkeys = random.split(key, num=10)
    for i in range(10):
        ar1_n = AR1(rho=rho, sigma=sigma, y0=y_T0)
        y_future = AR1_simulate(ar1, T1, subkeys[i])
        ax.plot(jnp.arange(T1), y_future, color='grey', alpha=.5)
    
    # Plot 90% CI
    ax.fill_between(
        jnp.arange(T1), y_upper_c95, y_lower_c95, alpha=.3, label='95% CI'
        )
    ax.fill_between(
        jnp.arange(T1), y_upper_c90, y_lower_c90, alpha=.35, label='90% CI'
        )
    ax.plot(jnp.arange(T1), center, color='red', alpha=.7, label='expectation')
    ax.legend(fontsize=12)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
        Initial path and predictive densities
    name: fig_path
---
ar1 = AR1(rho=0.9, sigma=1, y0=10, T0=100, T1=100)

# Simulate
initial_path = AR1_simulate(ar1, ar1.T0, key)

# Plot
fig, ax = plt.subplots(1, 1)
plot_path(ar1, initial_path, key, ax)
plt.show()
```

As functions of forecast horizon, the coverage intervals have shapes like those described in 
https://python.quantecon.org/perm_income_cons.html


## Predictive Distributions of Path Properties

Wecker {cite}`wecker1979predicting` proposed using simulation techniques to characterize  predictive distribution of some statistics that are  non-linear functions of $y$. 

He called these functions "path properties" to contrast them with properties of single data points.

He studied two special  prospective path properties of a given series $\{y_t\}$. 

The first was **time until the next turning point**.

* he defined a **"turning point"** to be the date of the  second of two successive declines in  $y$. 

To examine this statistic, let $Z$ be an indicator process

$$
Z_t(\omega) :=  
\begin{cases} 
\ 1 & \text{if } Y_t(\omega)< Y_{t-1}(\omega)< Y_{t-2}(\omega) \geq Y_{t-3}(\omega) \\
\ 0 & \text{otherwise}
\end{cases} 
$$

Here $\omega \in Omega$ is a sequence of events, and $Y_t: \Omega \rightarrow R$ gives $y_t$ according to $\omega$ and the AR(1) process.

Then the random variable **time until the next turning point**  is defined as the following *stopping time* with respect to $Z$:

$$
W_t(\omega):= \inf \{ k\geq 1 \mid Z_{t+k}(\omega) = 1\}
$$

Wecker  {cite}`wecker1979predicting` also studied **the minimum value of $Y$ over the next 8 quarters**
which can be defined as the random variable.

$$ 
M_t(\omega) := \min \{ Y_{t+1}(\omega); Y_{t+2}(\omega); \dots; Y_{t+8}(\omega)\}
$$

It is interesting to study yet another possible concept of a **turning point**.

Thus, let

$$
T_t(\omega) := 
\begin{cases}
\ 1 & \text{if } Y_{t-2}(\omega)> Y_{t-1}(\omega) > Y_{t}(\omega) \ \text{and } \ Y_{t}(\omega) < Y_{t+1}(\omega) < Y_{t+2}(\omega) \\
\ -1 & \text{if } Y_{t-2}(\omega)< Y_{t-1}(\omega) < Y_{t}(\omega) \ \text{and } \ Y_{t}(\omega) > Y_{t+1}(\omega) > Y_{t+2}(\omega) \\
\ 0 & \text{otherwise}
\end{cases}
$$

Define a **positive turning point today or tomorrow** statistic as 

$$
P_t(\omega) := 
\begin{cases}
\ 1 & \text{if } T_t(\omega)=1 \ \text{or} \ T_{t+1}(\omega)=1 \\
\ 0 & \text{otherwise}
\end{cases}
$$

This is designed to express the event

- "after one or two decrease(s), $Y$ will grow for two consecutive quarters" 

Following {cite}`wecker1979predicting`, we can use simulations to calculate  probabilities of $P_t$ and $N_t$ for each period $t$. 

## A Wecker-Like Algorithm

The procedure consists of the following steps: 

* index a sample path by $\omega_i$ 

* from a given date $t$, simulate $I$ sample paths of length $N$ 

$$
Y(\omega_i) = \left\{ Y_{t+1}(\omega_i), Y_{t+2}(\omega_i), \dots, Y_{t+N}(\omega_i)\right\}_{i=1}^I
$$

* for each path $\omega_i$, compute the associated value of $W_t(\omega_i), W_{t+1}(\omega_i), \dots , W_{t+N}$

* consider the sets $\{W_t(\omega_i)\}^{I}_{i=1}, \ \{W_{t+1}(\omega_i)\}^{I}_{i=1}, \ \dots, \ \{W_{t+N}(\omega_i)\}^{I}_{i=1}$ as samples from the predictive distributions $f(W_{t+1} \mid y_t, y_{t-1}, \dots , y_0)$, $f(W_{t+2} \mid y_t, y_{t-1}, \dots , y_0)$, $\dots$, $f(W_{t+N} \mid y_t, y_{t-1}, \dots , y_0)$.


## Using Simulations to Approximate a Posterior Distribution

The next code cells use `pymc` to compute the time $t$ posterior distribution of $\rho, \sigma$.

Note that in defining the likelihood function, we choose to condition on the initial value $y_0$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
        Posterior distributions (rho, sigma)
    name: fig_post
---
def draw_from_posterior(sample, size=1e5, bins=20, dis_plot=1):
    """Draw a sample of size from the posterior distribution.""" 
    # Start with priors
    rho = numpyro.sample('rho', dist.Uniform(-1, 1))  # Assume stable rho
    sigma = numpyro.sample('sigma', dis.HalfNormal(sqrt(10)))
    
    # Define likelihood recursively
    for t in range(1, len(sample))
        # Expectation of y_t
        mu = rho * sample[t-1]

        # Likelihood of the actual realization.
        numpyro.sample(f'y_{t}', dist.Normal(mu, sigma), obs=sample[t])

    # Compute posterior distribution of parameters
    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernek, 
        num_warmup=5000, 
        num_sample=size, 
        progress_bar=False)
    mcmc.run(random.PRNGKey(0), sample=sample)

    post_sample = {
        'rho': mcmc.get_samples()['rho'],
        'sigma': mcmc.get_samples()['sigma']
    }

    if dis_plot==1:
        sns.displot(
            post_sample, kde=True, stat="density", bins=bins, height=5, aspect=1.5
            )
    # TODO:Check plot and return
    return post_sample


post_samples = draw_from_posterior(initial_path)
```

The graphs above portray posterior distributions.

## Calculating Sample Path Statistics

Our next step is to prepare Python code to compute our sample path statistics.

These statistics were originally defined as random variables with respect to $\omega$, but here we use $\{Y_t\}$ as the argument because $\omega$ is implicit.

Also, these two kinds of definitions are equivalent because $\omega$ determins path statistics only through $\{Y_t\}$

```{code-cell} ipython3
# Define statistics as functions of y, the realized path values.
def next_recession(y):
    n = y.shape[0] - 3
    z = jnp.zeros(n, dtype=int)
    
    for i in range(n):
        z.at[i].set(int(y[i] <= y[i+1] and y[i+1] > y[i+2] and y[i+2] > y[i+3]))

    if jnp.any(z) == False:
        return 500
    else:
        return jnp.where(z == 1)[0][0] + 1


def next_severe_recession(y):
    z = jnp.diff(y)
    n = z.shape[0]
    
    sr = (z < -.02).astype(int)
    indices = jnp.where(sr == 1)[0]

    if len(indices) == 0:
        return T1
    else:
        return indices[0] + 1


def minimum_value_8q(y):
    return jnp.min(y[:8])


def next_turning_point(y):
    """
    Suppose that y is of length 6
    
        y_{t-2}, y_{t-1}, y_{t}, y_{t+1}, y_{t+2}, y_{t+3}
    
    that is sufficient for determining the value of P/N    
    """
    
    n = jnp.asarray(y).shape[0] - 4
    T = jnp.zeros(n, dtype=int)
    
    for i in range(n):
        if ((y[i] > y[i+1]) and (y[i+1] > y[i+2]) and 
            (y[i+2] < y[i+3]) and (y[i+3] < y[i+4])):
            T[i] = 1
        elif ((y[i] < y[i+1]) and (y[i+1] < y[i+2]) and 
            (y[i+2] > y[i+3]) and (y[i+3] > y[i+4])):
            T[i] = -1
    
    up_turn = jnp.where(T == 1)[0][0] + 1 if (1 in T) == True else T1
    down_turn = jnp.where(T == -1)[0][0] + 1 if (-1 in T) == True else T1

    return up_turn, down_turn
```

## Original Wecker Method

Now we apply Wecker's original method by simulating future paths and compute predictive distributions, conditioning on the true parameters associated with the data-generating model.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
        Distributions of statistics by Wecker's method
    name: fig_wecker
---
def plot_Wecker(ar1: AR1, initial_path, N, ax):
    """
    Plot the predictive distributions from "pure" Wecker's method.

    Parameters
    ----------
    ar1 : AR1
        An AR1 named tuple containing the process parameters (rho, sigma, T0, T1).
    initial_path : array-like
        The initial observed path of the AR(1) process.
    N : int
        Number of future sample paths to simulate for predictive distributions.
    """
    rho, sigma, T0, T1 = ar1.rho, ar1.sigma, ar1.T0, ar1.T1
    # Store outcomes
    next_reces = jnp.zeros(N)
    severe_rec = jnp.zeros(N)
    min_vals = jnp.zeros(N)
    next_up_turn, next_down_turn = jnp.zeros(N), jnp.zeros(N)
    
    # Compute confidence intervals
    y_T0 = initial_path[-1]
    center = jnp.array([rho**j * yT0 for j in range(T1)])
    vars = jnp.array(
        [sigma**2 * (1 - rho**(2 * j)) / (1 - rho**2) for j in range(T1)]
        )
    y_upper_c95 = center + 1.96 * jnp.sqrt(vars)
    y_lower_c95 = center - 1.96 * jnp.sqrt(vars)

    y_upper_c90 = center + 1.65 * jnp.sqrt(vars)
    y_lower_c90 = center - 1.65 * jnp.sqrt(vars)

    # Plot
    ax[0, 0].set_title("Initial path and predictive densities", fontsize=15)
    ax[0, 0].plot(jnp.arange(-T0 + 1, 1), initial_path)
    ax[0, 0].set_xlim([-T0, T1])
    ax[0, 0].axvline(0, linestyle='--', alpha=.4, color='k', lw=1)

    # Plot 90% CI
    ax[0, 0].fill_between(jnp.arange(T1), y_upper_c95, y_lower_c95, alpha=.3)
    ax[0, 0].fill_between(jnp.arange(T1), y_upper_c90, y_lower_c90, alpha=.35)
    ax[0, 0].plot(jnp.arange(T1), center, color='red', alpha=.7)

    # Simulate future paths
    subkeys = random.split(key, num=N)
    for n in range(N):
        ar1_n = AR1(rho=rho, sigma=sigma, y0=y_T0)
        sim_path = AR1_simulate(ar1_n, T1, subkeys[n])
        next_reces[n] = next_recession(jnp.hstack([initial_path[-3:-1], sim_path]))
        severe_rec[n] = next_severe_recession(sim_path)
        min_vals[n] = minimum_value_8q(sim_path)
        next_up_turn[n], next_down_turn[n] = next_turning_point(sim_path)

        if n%(N/10) == 0:
            ax[0, 0].plot(jnp.arange(T1), sim_path, color='gray', alpha=.3, lw=1)
            
    # Return next_up_turn, next_down_turn
    sns.histplot(
        next_reces, kde=True, stat='density', 
        ax=ax[0, 1], alpha=.8, label='True parameters'
        )
    ax[0, 1].set_title(
        "Predictive distribution (time until the next recession)",
        fontsize=13
        )

    sns.histplot(
        severe_rec, kde=False, stat='density', 
        ax=ax[1, 0], binwidth=0.9, alpha=.7, label='True parameters'
        )
    ax[1, 0].set_title(
        r"Predictive distribution (stopping time of growth$<-2\%$)",
        fontsize=13
        )

    sns.histplot(
        min_vals, kde=True, stat='density', 
        ax=ax[1, 1], alpha=.8, label='True parameters'
        )
    ax[1, 1].set_title(
        "Predictive distribution (minimum value in next 8 periods)",
        fontsize=13
        )

    sns.histplot(
        next_up_turn, kde=True, stat='density', 
        ax=ax[2, 0], alpha=.8, label='True parameters'
        )
    ax[2, 0].set_title(
        "Predictive distribution (time until the next positive turn)", 
        fontsize=13
        )

    sns.histplot(
        next_down_turn, kde=True, stat='density', 
        ax=ax[2, 1], alpha=.8, label='True parameters'
        )
    ax[2, 1].set_title(
        "Predictive distribution (time until the next negative turn)",
        fontsize=13
        )


fig, ax = plt.subplots(3, 2)
plot_Wecker(ar1, initial_path, 1000)
plt.show()
```

## Extended Wecker Method

Now we apply we apply our  "extended" Wecker method based on  predictive densities of $y$ defined by
{eq}`ar1-tp-eq4` that acknowledge posterior uncertainty in the parameters $\rho, \sigma$.

To approximate  the intergration on the right side of {eq}`ar1-tp-eq4`, we  repeatedly draw parameters from the joint posterior distribution each time we simulate a sequence of future values from model {eq}`ar1-tp-eq1`.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
        Distributions of statistics by extended Wecker's method
    name: fig_extend_wecker
---
def plot_extended_Wecker(ar1, post_samples, initial_path, N, ax):
    """
    Plot the extended Wecker's predictive distribution
    """
    rho, sigma, T0, T1 = ar1.rho, ar1.sigma, ar1.T0, ar1.T1
    # Select a sample
    index = random.choice(
        key, jnp.arange(len(post_samples['rho'])), (N + 1,), replace=False
        )
    rho_sample = post_samples['rho'][index]
    sigma_sample = post_samples['sigma'][index]

    # Store outcomes
    next_reces = jnp.zeros(N)
    severe_rec = jnp.zeros(N)
    min_vals = jnp.zeros(N)
    next_up_turn, next_down_turn = jnp.zeros(N), jnp.zeros(N)

    # Plot
    ax[0, 0].set_title(
        "Initial path and future paths simulated from posterior draws", 
        fontsize=15
        )
    ax[0, 0].plot(jnp.arange(-T0 + 1, 1), initial_path)
    ax[0, 0].set_xlim([-T0, T1])
    ax[0, 0].axvline(0, linestyle='--', alpha=.4, color='k', lw=1)

    # Simulate future paths
    subkeys = random.split(key, num=N)
    for n in range(N):
        ar1_n = AR1(
            rho=rho_sample[n], sigma=sigma_sample[n], y0= initial_path[-1]
            )
        sim_path = AR1_simulate(ar1_n, T1, subkeys[n])
        next_reces[n] = next_recession(
            jnp.hstack([initial_path[-3:-1], sim_path])
            )
        severe_rec[n] = next_severe_recession(sim_path)
        min_vals[n] = minimum_value_8q(sim_path)
        next_up_turn[n], next_down_turn[n] = next_turning_point(sim_path)

        if n % (N / 10) == 0:
            ax[0, 0].plot(jnp.arange(T1), sim_path, color='gray', alpha=.3, lw=1)
        
    # Return next_up_turn, next_down_turn
    sns.histplot(
        next_reces, kde=True, stat='density',
        ax=ax[0, 1], alpha=.6, color=colors[1], 
        label='Sampling from posterior'
    )
    ax[0, 1].set_title(
        "Predictive distribution (time until the next recession)", 
        fontsize=13)

    sns.histplot(
        severe_rec, kde=False, stat='density', 
        ax=ax[1, 0], binwidth=.9, alpha=.6, color=colors[1], 
        label='Sampling from posterior'
        )
    ax[1, 0].set_title(
        r"Predictive distribution (stopping time of growth$<-2\%$)", 
        fontsize=13
        )

    sns.histplot(
        min_vals, kde=True, stat='density', 
        ax=ax[1, 1], alpha=.6, color=colors[1], 
        label='Sampling from posterior'
        )
    ax[1, 1].set_title(
        "Predictive distribution (minimum value in the next 8 periods)", 
        fontsize=13)

    sns.histplot(
        next_up_turn, kde=True, stat='density', 
        ax=ax[2, 0], alpha=.6, color=colors[1], 
        label='Sampling from posterior')
    ax[2, 0].set_title(
        "Predictive distribution of time until the next positive turn", 
        fontsize=13
        )

    sns.histplot(
        next_down_turn, kde=True, stat='density', 
        ax=ax[2, 1], alpha=.6, color=colors[1], 
        label='Sampling from posterior'
        )
    ax[2, 1].set_title(
        "Predictive distribution of time until the next negative turn", 
        fontsize=13
        )

fig, ax = plt.subplots(3, 2)
plot_extended_Wecker(ar1, post_samples, initial_path, 1000, ax)
plt.show()
```

## Comparison

Finally, we plot both the original Wecker method and the extended method with parameter values drawn from the posterior together to compare the differences that emerge from pretending to know parameter values when they are actually uncertain.  

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
        Comparison between two methods
    name: fig_wecker
---
fig, ax = plt.subplots(3, 2)
plot_Wecker(ar1, initial_path, 1000, ax)
ax[0, 0].clear()
plot_extended_Wecker(ar1, post_samples, initial_path, 1000, ax)
plt.legend()
plt.show()
```
