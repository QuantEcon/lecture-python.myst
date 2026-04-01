---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Organization Capital

```{index} single: Organization Capital
```

## Overview

This lecture describes a theory of **organization capital** proposed by
{cite}`Prescott_Visscher_1980`.

Prescott and Visscher define organization capital as information that a firm accumulates
about its employees, teams, and production processes.

This information is an **asset** to the firm because it affects the production possibility set
and is produced jointly with output.

Costs of adjusting the stock of organization capital constrain the firm's growth rate,
providing an explanation for

1. why firm growth rates are independent of firm size (Gibrat's Law)
1. why adjustment costs for rapid growth arise endogenously rather than being assumed

The paper offers three examples of organization capital:

* **Personnel information**: knowledge about the match between workers and tasks
* **Team information**: knowledge about how well groups of workers mesh
* **Firm-specific human capital**: skills of employees enhanced by on-the-job training

In each case, the investment possibilities lead firms to grow at a common rate,
yielding constant returns to scale together with increasing costs of rapid size adjustment.

```{note}
The theory is related to ideas of {cite}`Coase_1937` and {cite}`Williamson_1975` about the nature of the firm.
Prescott and Visscher stress the firm's role as a storehouse of information and argue that
incentives within the firm  are created for efficient accumulation and use of that information.
```

Let's start with some imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
```

## The Basic Idea

The firm is a storehouse of information.

Within the firm, incentives are created for the efficient accumulation and use of that information.

Prescott and Visscher exploit this concept to explain certain facts about firm growth and
size distribution.

The key insight: the process by which information is accumulated naturally leads to

1. **constant returns to scale**, and
2. **increasing costs to rapid firm size adjustment**

Constant returns to scale explain the absence of an observed unique optimum firm size
(see {cite}`Stigler_1958`).

Without costs of adjustment, the pattern of investment
by firms in the face of a change in market demand would exhibit
discontinuities we do not observe.

Further, without a cost penalty to rapid growth, the first firm to
discover a previously untapped market would preempt competition by
usurping all profitable investments as they appear, thus implying
monopoly more prevalent than it is.


## Personnel Information as Organization Capital

```{index} single: Organization Capital; Personnel Information
```

The first example of organization capital is information about the
match between workers and tasks.

### Setup

Workers have different sets of skills and talents.

A variable $\theta$ measures the aptitude of a worker for a particular kind of work.

* Workers with high $\theta$ have comparative advantage in tasks requiring repeated attention to detail
* Workers with low $\theta$ have comparative advantage in work requiring broadly defined duties

The population distribution of $\theta$ is normal with mean zero and precision (inverse of variance) $\pi$:

$$
\theta \sim N(0, 1/\pi)
$$

When a worker is hired from the labor pool, neither the worker nor the employer knows $\theta$.
Both know only the population distribution.

### Three Tasks

If $q$ units of output are produced, assume:

* $\varphi_1 q$ workers are assigned to **task 1** (screening)
* $\varphi q$ workers are assigned to **task 2**
* the remaining workers are assigned to **task 3**

where $\varphi_1 + 2\varphi = 1$.

```{note}
The fixed coefficients technology requires a constant ratio between the number of
personnel in jobs 2 and 3 and the number assigned to job 1.
```

For task 1, the screening task, per unit cost of production is **invariant** to the $\theta$-values of the individuals assigned.

However, the larger a worker's $\theta$, the larger is his product in task 2 relative to
his product in task 3.

Consequently:

* a worker with a highly positive $\theta$ is much better suited for task 2
* a worker with a highly negative $\theta$ is much better suited for task 3

### Bayesian Learning

Performance in tasks 2 or 3 cannot be observed at the individual level.

But information about a worker's $\theta$-value can be obtained from observing
performance in task 1, the screening task.

The expert supervising the apprentice determines a value of $z$ each period:

$$
z_{it} = \theta_i + \epsilon_{it}
$$ (eq:signal)

where $\epsilon_{it} \sim N(0, 1)$ are independently distributed over both workers $i$ and periods $t$.

After $n$ observations on a worker in the screening job, the **posterior distribution** of $\theta$ is normal with

**posterior mean:**

$$
m = \frac{1}{\pi + n} \sum_{k=1}^{n} z_k
$$ (eq:post_mean)

**posterior precision:**

$$
h = \pi + n
$$ (eq:post_prec)

Knowledge of an individual is thus completely characterized by the pair $(m, h)$.

```{code-cell} ipython3
def bayesian_update(z_observations, prior_precision):
    """
    Compute posterior mean and precision after observing signals.

    Parameters
    ----------
    z_observations : array_like
        Observed signals z_1, ..., z_n
    prior_precision : float
        Precision π of the prior distribution

    Returns
    -------
    m : float
        Posterior mean
    h : float
        Posterior precision
    """
    n = len(z_observations)
    h = prior_precision + n
    m = np.sum(z_observations) / h
    return m, h
```

Let's visualize how the posterior evolves as we observe a worker whose true $\theta = 0.8$:

```{code-cell} ipython3
np.random.seed(42)

# True worker type
theta_true = 0.8

# Prior precision
pi = 1.0

# Generate signals
T = 20
epsilons = np.random.randn(T)
z_signals = theta_true + epsilons

# Track posterior evolution
posterior_means = []
posterior_stds = []

for n in range(1, T + 1):
    m, h = bayesian_update(z_signals[:n], pi)
    posterior_means.append(m)
    posterior_stds.append(1 / np.sqrt(h))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot posterior mean convergence
ax = axes[0]
ax.plot(range(1, T + 1), posterior_means, 'b-o', markersize=4,
        label='Posterior mean $m$')
ax.axhline(theta_true, color='r', linestyle='--',
           label=fr'True $\theta = {theta_true}$')
ax.set_xlabel('Number of observations $n$')
ax.set_ylabel('Posterior mean $m$')
ax.set_title('Convergence of Posterior Mean')
ax.legend()

# Plot posterior standard deviation
ax = axes[1]
ax.plot(range(1, T + 1), posterior_stds, 'g-o', markersize=4,
        label='Posterior std $1/\sqrt{h}$')
ax.set_xlabel('Number of observations $n$')
ax.set_ylabel('Posterior standard deviation')
ax.set_title('Shrinking Posterior Uncertainty')
ax.legend()

plt.tight_layout()
plt.show()
```

As the number of screening observations $n$ increases, the posterior mean converges
to the true $\theta$, and the posterior uncertainty shrinks at rate $1/\sqrt{n}$.

### Per Unit Costs of Production

Under the nonsequential assignment rule, employees with the greatest seniority
are assigned to jobs 2 and 3, while newer employees remain in the screening task.

Workers with $m > 0$ are assigned to task 2, and those with $m \leq 0$ to task 3.

Per unit costs of production, assuming this assignment after $n$ screening periods, are:

$$
c(n) = c_1 + c_2 + c_3 - E\{\theta \mid m > 0\} + E\{\theta \mid m \leq 0\}
$$ (eq:unit_cost)

Because $m$ is normally distributed, evaluation of the conditional expectation in
{eq}`eq:unit_cost` yields per unit costs as a function of $n$:

$$
c(n) = c - 0.7978 \frac{n}{\pi(\pi + n)}
$$ (eq:cost_n)

where $c = c_1 + c_2 + c_3$ and $0.7978 = 2 \int_0^{\infty} \frac{t}{\sqrt{2\pi}} e^{-t^2/2} dt$.

```{note}
The constant $0.7978 \approx \sqrt{2/\pi}$ is twice the mean of the half-normal distribution.
It arises from computing $E[\theta \mid m > 0] - E[\theta \mid m \leq 0]$ for a normal distribution.
```

The function $c(n)$ decreases at a **decreasing rate** in $n$: more screening observations
reduce costs but with diminishing returns.

```{code-cell} ipython3
def cost_per_unit(n_vals, pi, c_bar=1.0):
    """
    Per unit cost of production as a function of screening periods n.

    Parameters
    ----------
    n_vals : array_like
        Number of screening periods
    pi : float
        Prior precision
    c_bar : float
        Base cost c = c1 + c2 + c3

    Returns
    -------
    costs : array
        Per unit costs c(n)
    """
    n_vals = np.asarray(n_vals, dtype=float)
    return c_bar - 0.7978 * n_vals / (pi * (pi + n_vals))


fig, ax = plt.subplots(figsize=(10, 6))

n_vals = np.linspace(0.1, 50, 200)

for pi in [0.5, 1.0, 2.0, 5.0]:
    costs = cost_per_unit(n_vals, pi)
    ax.plot(n_vals, costs, label=fr'$\pi = {pi}$')

ax.set_xlabel('Screening periods $n$')
ax.set_ylabel('Per unit cost $c(n)$')
ax.set_title('Per Unit Costs Decrease with Screening Time')
ax.legend()
ax.set_xlim(0, 50)
plt.tight_layout()
plt.show()
```

The figure shows that:

* costs decrease with more screening time $n$
* the decrease is at a declining rate (diminishing returns to screening)
* for smaller prior precision $\pi$ (more initial uncertainty about worker types), the gains from screening are larger

This diminishing-returns structure is the source of the **increasing costs of rapid adjustment**.


### Growth Rate and Screening Time

The greater the growth rate, the smaller must be $n$ --- the time spent in the screening
task before assignment to job 2 or 3.

If $\gamma$ is the growth rate of output and $\rho$ is the quit rate, and $y_i$ is the current number
of vintage $i$ employees, then

$$
(1 + \gamma) y_{i+1} = (1 - \rho) y_i
$$

Letting $\xi = (1 - \rho)/(1 + \gamma)$, from the above $y_i = \xi^i y_0$.

For the fixed coefficients technology, the fraction of present personnel with vintage
greater than $n$ must equal $2\varphi / (\varphi_1 + 2\varphi)$, which gives:

$$
\xi^{n+1} = \frac{2\varphi}{\varphi_1 + 2\varphi}
$$ (eq:cutoff)

Solving for $n$ as a function of $\gamma$:

$$
n(\gamma) = \frac{\log(2\varphi) - \log(\varphi_1 + 2\varphi)}{\log(1 - \rho) - \log(1 + \gamma)} - 1 \quad \text{for } \gamma > -\rho
$$ (eq:n_gamma)

```{code-cell} ipython3
def screening_time(gamma, rho, phi1, phi):
    """
    Compute the screening time n as a function of growth rate γ.

    Parameters
    ----------
    gamma : array_like
        Growth rate of output
    rho : float
        Quit rate
    phi1 : float
        Fraction of workers in task 1 per unit output
    phi : float
        Fraction of workers in each of tasks 2, 3 per unit output

    Returns
    -------
    n : array
        Screening periods before assignment
    """
    gamma = np.asarray(gamma, dtype=float)
    numerator = np.log(2 * phi) - np.log(phi1 + 2 * phi)
    denominator = np.log(1 - rho) - np.log(1 + gamma)
    return numerator / denominator - 1


# Parameters
rho = 0.1       # quit rate
phi1 = 0.5      # fraction in screening
phi = 0.25      # fraction in each of tasks 2, 3

gamma_vals = np.linspace(-0.05, 0.30, 200)

# Filter valid range: γ > -ρ and ensure n > 0
valid = gamma_vals > -rho
gamma_valid = gamma_vals[valid]
n_vals = screening_time(gamma_valid, rho, phi1, phi)
# Only keep non-negative n
mask = n_vals > 0
gamma_plot = gamma_valid[mask]
n_plot = n_vals[mask]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(gamma_plot, n_plot, 'b-', linewidth=2)
ax.set_xlabel(r'Growth rate $\gamma$')
ax.set_ylabel(r'Screening periods $n(\gamma)$')
ax.set_title('Faster Growth Means Less Screening Time')
ax.set_xlim(gamma_plot[0], gamma_plot[-1])
plt.tight_layout()
plt.show()
```

The figure shows the key trade-off: **faster growth forces shorter screening periods**.

When growth is rapid, new workers must be promoted from the screening task to
productive tasks more quickly, so less information is gathered about each worker
before assignment.


### Combined Effect: Growth Rate and Per Unit Costs

Composing the functions $c(n)$ and $n(\gamma)$ reveals how per unit costs depend on the
growth rate:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))

pi = 1.0
c_bar = 1.0

# Compute per unit costs as function of growth rate
n_of_gamma = screening_time(gamma_plot, rho, phi1, phi)
costs_of_gamma = cost_per_unit(n_of_gamma, pi, c_bar)

ax.plot(gamma_plot, costs_of_gamma, 'r-', linewidth=2)
ax.set_xlabel(r'Growth rate $\gamma$')
ax.set_ylabel(r'Per unit cost $c(n(\gamma))$')
ax.set_title('Per Unit Costs Increase with Growth Rate')
ax.set_xlim(gamma_plot[0], gamma_plot[-1])
plt.tight_layout()
plt.show()
```

This establishes the key result: **increasing costs of rapid adjustment arise endogenously**
from the trade-off between screening and growth.

The faster the firm grows, the less time it has to screen workers, the poorer the
match between workers and tasks, and the higher the per unit production costs.


## Industry Equilibrium

```{index} single: Organization Capital; Industry Equilibrium
```

Firm growth rates are independent of firm size in this model because the
mathematical structure of the technology constraint is the same as that
considered in {cite}`lucas1967adjustment`, except that the stock of organization capital
is a vector rather than a scalar.

The technology set facing price-taking firms is a **convex cone**: there are
constant returns to scale.

Constant returns and internal adjustment costs, along with some costs of
transferring capital between firms, yield an optimum rate of firm growth
**independent of the firm's size** --- this is Gibrat's Law.

The bounded, downward-sloping, inverse industry demand function is

$$
P_t = p(Q_t, u_t)
$$

where $Q_t$ is the sum of output over all firms and $u_t$ is a demand shock
subject to a stationary Markov process.

Prescott and Visscher show that a competitive equilibrium exists using the
framework of {cite}`Lucas_Prescott_1971`.

The discounted consumer surplus to be maximized is

$$
\sum_{t=0}^{\infty} \beta^t \left\{ \int_0^{Q_t} p(y, u_t) dy - Bw - Q_t \sum_i (A_{i2t} + A_{i3t}) c(i) / \sum_i (A_{i2t} + A_{i3t}) \right\}
$$ (eq:surplus)

where $A_{i2t}, A_{i3t}$, and $B$ are obtained by summing $a_{i2t}$, $a_{i3t}$, and $b$,
respectively, over all firms in the industry.


### Key Property: Growth Rates Independent of Size

If two firms have organization capital vectors $\underline{k}$ that are proportional at a point in time,
they will be proportional in all future periods.

That is, **growth rates are independent of firm size**.

```{code-cell} ipython3
def simulate_firm_growth(T, gamma, rho, q0, seed=42):
    """
    Simulate firm output growth with constant growth rate
    and stochastic quit turnover.

    Parameters
    ----------
    T : int
        Number of periods
    gamma : float
        Equilibrium growth rate
    rho : float
        Quit rate
    q0 : float
        Initial output
    seed : int
        Random seed

    Returns
    -------
    output : array
        Firm output path
    """
    rng = np.random.default_rng(seed)
    output = np.zeros(T)
    output[0] = q0
    for t in range(1, T):
        # Stochastic growth around equilibrium rate
        shock = rng.normal(0, 0.02)
        output[t] = output[t-1] * (1 + gamma + shock)
    return output


T = 50
gamma_eq = 0.05  # equilibrium growth rate
rho = 0.1

# Simulate firms of different initial sizes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Level plots
ax = axes[0]
for q0, label in [(10, 'Small firm'), (50, 'Medium firm'),
                   (200, 'Large firm')]:
    output = simulate_firm_growth(T, gamma_eq, rho, q0,
                                  seed=int(q0))
    ax.plot(range(T), output, label=f'{label} ($q_0={q0}$)')
ax.set_xlabel('Period')
ax.set_ylabel('Output $q_t$')
ax.set_title('Firm Output Levels')
ax.legend()

# Log plots (growth rates)
ax = axes[1]
for q0, label in [(10, 'Small firm'), (50, 'Medium firm'),
                   (200, 'Large firm')]:
    output = simulate_firm_growth(T, gamma_eq, rho, q0,
                                  seed=int(q0))
    ax.plot(range(T), np.log(output), label=f'{label} ($q_0={q0}$)')
ax.set_xlabel('Period')
ax.set_ylabel(r'$\log(q_t)$')
ax.set_title('Log Output (Parallel = Equal Growth Rates)')
ax.legend()

plt.tight_layout()
plt.show()
```

The right panel shows that all firms grow at the same rate regardless of initial size ---
the log output paths are parallel.

This is **Gibrat's Law**: growth rates are independent of firm size.

## Bayesian Screening Simulation

```{index} single: Organization Capital; Bayesian Screening
```

Let's simulate the full screening and assignment process for a single firm.

We draw workers from the population, observe their signals in the screening task,
and then assign them to the appropriate productive task based on the posterior mean.

```{code-cell} ipython3
def simulate_screening(n_workers, n_screen, pi, seed=123):
    """
    Simulate the screening and assignment of workers.

    Parameters
    ----------
    n_workers : int
        Number of workers to screen
    n_screen : int
        Number of screening periods per worker
    pi : float
        Prior precision of θ distribution
    seed : int
        Random seed

    Returns
    -------
    results : dict
        Dictionary with θ values, posterior means,
        assignments, and misassignment rate
    """
    rng = np.random.default_rng(seed)

    # Draw true worker types
    theta = rng.normal(0, 1/np.sqrt(pi), n_workers)

    # Generate screening signals
    signals = (theta[:, None]
               + rng.normal(0, 1, (n_workers, n_screen)))

    # Compute posterior means after screening
    posterior_means = signals.sum(axis=1) / (pi + n_screen)

    # Assign workers: m > 0 → task 2, m ≤ 0 → task 3
    assignment = np.where(posterior_means > 0, 2, 3)

    # Correct assignment based on true θ
    correct_assignment = np.where(theta > 0, 2, 3)

    # Misassignment rate
    misassignment_rate = np.mean(assignment != correct_assignment)

    return {
        'theta': theta,
        'posterior_means': posterior_means,
        'assignment': assignment,
        'correct_assignment': correct_assignment,
        'misassignment_rate': misassignment_rate
    }


pi = 1.0
n_workers = 5000
screening_periods = [1, 3, 5, 10, 20, 50]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

misassignment_rates = []

for idx, n_screen in enumerate(screening_periods):
    results = simulate_screening(n_workers, n_screen, pi)
    misassignment_rates.append(results['misassignment_rate'])

    ax = axes[idx]
    theta = results['theta']
    m = results['posterior_means']

    # Color by whether assignment matches true type
    correct = results['assignment'] == results['correct_assignment']
    ax.scatter(theta[correct], m[correct], alpha=0.1, s=5,
               color='blue', label='Correct')
    ax.scatter(theta[~correct], m[~correct], alpha=0.3, s=5,
               color='red', label='Misassigned')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    mis = results['misassignment_rate']
    ax.set_title(f'$n = {n_screen}$, misassign = {mis:.1%}')
    ax.set_xlabel(r'True $\theta$')
    ax.set_ylabel('Posterior mean $m$')
    if idx == 0:
        ax.legend(markerscale=5, loc='upper left')

plt.tight_layout()
plt.show()
```

Red dots are workers who are **misassigned** --- placed in the wrong productive task
because the posterior mean had the wrong sign relative to their true $\theta$.

As $n$ increases:
* The posterior mean $m$ becomes more strongly correlated with $\theta$
* Misassignment rates fall

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))

n_range = np.arange(1, 51)
mis_rates = []
for n_screen in n_range:
    results = simulate_screening(n_workers, n_screen, pi)
    mis_rates.append(results['misassignment_rate'])

ax.plot(n_range, mis_rates, 'b-o', markersize=3)
ax.set_xlabel('Screening periods $n$')
ax.set_ylabel('Misassignment rate')
ax.set_title('Misassignment Rate Decreases with Screening Time')
plt.tight_layout()
plt.show()
```

This confirms the theoretical prediction: the cost savings from better assignment
exhibit **diminishing returns** in the screening time $n$.

## Team Information

```{index} single: Organization Capital; Team Information
```

Personnel information need not be valuable only because it facilitates the matching of
workers to tasks.

Another equally valuable use of personnel information is in the **matching of workers to workers**.

What is important to performance in many activities within the firm is not just
the aptitude of an individual assigned to a task, but also how well the
characteristics of the individual mesh with those of others performing related duties.

### Structure

Suppose workers are grouped into teams, and team $i$ assigned to a screening task
has an observed productivity indicator

$$
z_{it} = \theta_i + \epsilon_{it}
$$

where:
* $\theta_i$ is a deterministic component directly related to how well team workers are paired
* $\epsilon_{it} \sim N(0, 1)$ are i.i.d. stochastic components

The $\theta$ from all possible teams are approximately independently and normally distributed
$N(\mu, 1/\pi)$.

After $n$ observations on team $i$, the posterior distribution on $\theta_i$ is normal with

$$
m = \mu + \frac{1}{\pi + n} \sum_{k=1}^{n} (z_k - \mu)
$$

and precision $h = \pi + n$.

If dissolution of a team also dissolves the accrued information, the team information
model has the **same mathematical structure** as the personnel information model.

```{code-cell} ipython3
def simulate_team_screening(n_teams, n_screen, pi, mu=0.5,
                            seed=456):
    """
    Simulate team screening with Bayesian updating.

    Parameters
    ----------
    n_teams : int
        Number of teams to screen
    n_screen : int
        Number of screening periods
    pi : float
        Prior precision
    mu : float
        Prior mean of team quality
    seed : int
        Random seed

    Returns
    -------
    results : dict
    """
    rng = np.random.default_rng(seed)

    # True team qualities
    theta = rng.normal(mu, 1/np.sqrt(pi), n_teams)

    # Generate signals
    signals = (theta[:, None]
               + rng.normal(0, 1, (n_teams, n_screen)))

    # Posterior means
    z_bar = signals.mean(axis=1)
    post_means = mu + n_screen * (z_bar - mu) / (pi + n_screen)
    post_prec = pi + n_screen

    return {
        'theta': theta,
        'posterior_means': post_means,
        'posterior_precision': post_prec
    }


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, n_screen in enumerate([1, 5, 20]):
    results = simulate_team_screening(500, n_screen, pi=1.0, mu=0.5)

    ax = axes[idx]
    ax.scatter(results['theta'], results['posterior_means'],
               alpha=0.4, s=10)
    lims = [-1.5, 2.5]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='45° line')
    ax.set_xlabel(r'True team quality $\theta$')
    ax.set_ylabel('Posterior mean $m$')
    ax.set_title(f'$n = {n_screen}$ screening periods')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()
```

As with individual screening, more observations improve the precision of team quality
estimates.
Rapid growth forces fewer observations before team assignments must be finalized, leading
to higher costs.


## Firm-Specific Human Capital

```{index} single: Organization Capital; Human Capital
```

The third example: organization capital consists of the **human capital** of the firm's employees.

The capacity of the organization to function effectively as a production unit is
determined largely by the level and meshing of the skills of the employees.

```{note}
The case for the human capital of employees being part of the capital stock of the firm
is well established (see {cite}`Becker_1975`). Productivity in the future depends on levels
of human capital in the future, but to acquire human capital for the future, a sacrifice
in real resources is required in the present.
```

The key features are:

* Output and skill enhancement are **joint products** resulting from the combination of
  labor inputs possessing different skill levels

* Experienced and inexperienced workers are combined in one of several available technical
  processes to generate the firm's product, and in the process, the overall competence
  of the work force is improved

* The transformation frontier between current output and future human capital is
  **concave** and linearly homogeneous

This gives the technology set the structure of a closed convex cone with a vertex at the
origin --- sufficient for optimal proportional growth by firms.

### Concave Transformation Frontier

```{code-cell} ipython3
def transformation_frontier(q, alpha=0.7):
    """
    Concave transformation frontier between current output
    and future human capital increment.

    Parameters
    ----------
    q : array_like
        Current output (fraction of capacity)
    alpha : float
        Concavity parameter

    Returns
    -------
    hk : array
        Future human capital increment
    """
    q = np.asarray(q, dtype=float)
    return (1 - q**alpha)**(1/alpha)


fig, ax = plt.subplots(figsize=(8, 8))

q_vals = np.linspace(0, 1, 200)

for alpha in [0.5, 0.7, 1.0, 1.5]:
    hk = transformation_frontier(q_vals, alpha)
    ax.plot(q_vals, hk,
            label=fr'$\alpha = {alpha}$', linewidth=2)

ax.set_xlabel('Current output $q$ (fraction of capacity)')
ax.set_ylabel('Future human capital increment $\\Delta h$')
ax.set_title('Concave Transformation Frontier')
ax.legend()
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
```

The concavity of the transformation frontier means that moving from an extremely
unbalanced bundle of production and learning activity to a more balanced bundle
entails little sacrifice.

But a workday consisting primarily of learning also has diminishing returns,
creating the cost of rapid adjustment.


## Costs of Transferring Organization Capital

```{index} single: Organization Capital; Transfer Costs
```

If there were no cost to transferring organization capital from one firm to another,
the model would not place constraints on the firm's growth rate.

Firms could then merge, divest, or pirate each other's personnel without a cost penalty
and thus produce a pattern of growth not restricted by the model.

Organization capital is **not** costlessly moved, however:

1. **Moving is disruptive**: relocating from one locale to another is disruptive to both
   employee and family

2. **Information is firm-specific**: the information set that makes a person productive
   in one organization may not make that person as productive in another, even if both
   firms produce identical output

   * Facility with a computer system at one firm
   * Knowing whom to ask when problems arise
   * Rapport with buyers or sellers

These are types of organization capital in one firm that **cannot be transferred costlessly**
to another.


## Summary and Implications

The Prescott-Visscher model provides a unified framework in which:

* The firm exists as an entity because it is an efficient structure for accumulating,
  storing, and using information

* **Constant returns to scale** arise because once the best combinations of worker types
  are discovered, nothing prevents the firm from replicating those combinations with
  proportional gains in product

* **Increasing adjustment costs** arise endogenously from the trade-off between
  current production and investment in organization capital

* **Gibrat's Law** --- growth rates independent of firm size --- is a natural implication

* Large firms should have growth rates that display **less variance** than small firms
  because large firms are essentially portfolios of smaller production units

```{code-cell} ipython3
# Illustrate the variance reduction in growth rates for large vs small firms
def simulate_growth_rate_distribution(n_firms, n_subunits, gamma,
                                      sigma, T=100, seed=789):
    """
    Simulate growth rate distributions for firms of different sizes.

    Parameters
    ----------
    n_firms : int
        Number of firms to simulate
    n_subunits : int
        Number of independent subunits per firm
    gamma : float
        Mean growth rate
    sigma : float
        Std dev of growth rate per subunit
    T : int
        Number of periods
    seed : int
        Random seed

    Returns
    -------
    growth_rates : array
        Realized growth rates for each firm
    """
    rng = np.random.default_rng(seed)
    # Each firm's growth is average of n_subunit growth rates
    subunit_growth = rng.normal(gamma, sigma,
                                (n_firms, n_subunits, T))
    firm_growth = subunit_growth.mean(axis=1)  # average across subunits
    # Return time-averaged growth rate for each firm
    return firm_growth.mean(axis=1)


fig, ax = plt.subplots(figsize=(10, 6))

sizes = {'Small (1 unit)': 1,
         'Medium (5 units)': 5,
         'Large (20 units)': 20}

gamma = 0.05
sigma = 0.10

for label, n_sub in sizes.items():
    rates = simulate_growth_rate_distribution(
        2000, n_sub, gamma, sigma)
    ax.hist(rates, bins=50, alpha=0.5, density=True,
            label=f'{label}: std={rates.std():.4f}')

ax.set_xlabel('Average growth rate')
ax.set_ylabel('Density')
ax.set_title('Growth Rate Distributions by Firm Size')
ax.legend()
ax.axvline(gamma, color='k', linestyle='--',
           label=r'$\gamma$', alpha=0.5)
plt.tight_layout()
plt.show()
```

The figure shows that although all firms have the **same mean growth rate** (Gibrat's Law),
large firms display **less variance** in realized growth rates because they are effectively
portfolios of independent subunits.

This is consistent with the empirical findings of {cite}`Mansfield_1962` and {cite}`Hymer_Pashigian_1962`.

The essence of the Prescott-Visscher theory is that the nature of the firm is tied to
**organization capital**.

What distinguishes the firm from other relationships is that it is a structure within which
agents have the incentive to acquire and reveal information in a manner that is less
costly than in possible alternative institutions.

