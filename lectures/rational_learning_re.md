---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(rational_learning_re)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Rational Learning and Rational Expectations

```{contents} Contents
:depth: 2
```

## Overview

This lecture explores a classic question in economic theory: can agents **learn** their way to a rational expectations equilibrium?

{cite:t}`BrayKreps1987` examine this question in a rigorously specified model.

In a rational expectations equilibrium, agents use market prices to make inferences about other agents' private information.

Each agent knows the **statistical relationship** between prices and the underlying payoff-relevant variables and that relationship is **correct** given the equilibrium.

But this raises a question: where does that knowledge come from?

The **rational learning** approach studied by Bray and Kreps asks whether agents who start with uncertainty about the equilibrium price function can, over time, learn it from observations of past prices.

The key findings are:

* In every rational learning model, posterior assessments converge because they are bounded martingales.
* In the paper's benchmark example, the uninformed agent learns the informed agent's risk tolerance.
* Correct learning requires identification, smooth equilibrium price maps, and positive prior probability for the true model.

This lecture presents the Bray–Kreps framework, explains their benchmark example, and provides Python code for a simplified Bayesian learning illustration.


We focus on {cite:t}`BrayKreps1987`, published in *Arrow and the Ascent of Modern Economic Theory*, which synthesizes earlier work by {cite:t}`Bray1982`, {cite:t}`BraySavin1984`, and the rational expectations literature of {cite:t}`Radner1979`, {cite:t}`grossman1976`, and {cite:t}`Jordan1982`.

The local PDF version is the June 1981 Stanford Research Paper version of the same work.

Let's start with the necessary imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## The economy

### Agents and assets

The paper's example is an infinitely repeated version of the information model in {cite:t}`GrossmanStiglitz1980`.

Each date is economically disconnected from the others, so agents start each period afresh.

There are two types of agents and two assets:

* A **safe asset** with net return normalized to zero.
* A **risky asset** endowed one unit per agent and traded at date $t$ at spot price $p_t$.

At each date $t = 0, 1, 2, \ldots$ the risky asset yields a gross return $r_t$ at date $t+1/2$.

An informed signal $s_t$ satisfies

$$
r_t = s_t + \epsilon_t,
\qquad
\epsilon_t \sim \mathcal N(0,\sigma^2),
$$

where $\{s_t\}$ and $\{\epsilon_t\}$ are IID normal sequences and are mutually independent.

There are two representative agents:

* **Agent $I$ (informed)** observes $s_t$ before trading at date $t$.
* **Agent $U$ (uninformed)** observes the equilibrium price $p_t$ but not $s_t$ before trading.

Both agents observe the previous return before current trading.

### Preferences

Agent $n \in \{I, U\}$ has constant absolute risk tolerance $\theta^n$.

If agent $n$ holds $x^n$ units of the risky asset and $y^n$ units of the safe asset between dates $t$ and $t+1/2$, period utility is

$$
-\exp\left[-\frac{x^n r_t + y^n}{\theta^n}\right].
$$

Thus $1/\theta^n$ is the coefficient of absolute risk aversion.

Given the signal $s_t$, the informed agent's demand is

$$
x^I_t
=
\frac{\theta^I}{\sigma^2}(s_t - p_t).
$$ (eq:bk-informed-demand)

Because each of the two agents is endowed with one unit of the risky asset, market clearing is

$$
x^I_t + x^U_t = 2.
$$

### Rational expectations equilibrium

If all agents knew $s_t$, agent $n$ would demand

$$
x^n_t = \frac{\theta^n}{\sigma^2}(s_t - p_t).
$$

With $N$ agents and total risky-asset supply $N$, market clearing gives the **full communication price**

$$
p_t
=
s_t
-
\frac{N\sigma^2}{\sum_{n=1}^N \theta^n}.
$$ (eq:bk-full-communication-price)

Thus if $\sum_n \theta^n$ is known, the price fully reveals $s_t$.

Following {cite:t}`Radner1979`, Bray and Kreps call this a full communication rational expectations equilibrium.

The paper's learning problem starts when $\theta^I$ is unknown to agent $U$.

Agent $U$ knows $\sigma^2$ and $\theta^U$, and starts with a prior density over $\theta^I$ on an interval $[a,b] \subset (0,\infty)$.

At a date when agent $U$ has posterior density $f$ over $\theta^I$, his own trade reveals $x^I_t=2-x^U_t$ through market clearing.

Combining this inferred $x^I_t$ with {eq}`eq:bk-informed-demand`, each candidate $\theta^I$ implies

$$
s_t
=
p_t
+
\frac{\sigma^2 x^I_t}{\theta^I}.
$$ (eq:bk-signal-implied)

After trading, agent $U$ observes $r_t$.

Bayes' rule then updates the posterior over $\theta^I$ using the normal density of the signal implied by {eq}`eq:bk-signal-implied` conditional on the realized return.

This is the main object learned in Bray and Kreps' benchmark example.

They emphasize that the equilibrium can be defined recursively, but closed-form prices are "out of the question" even in this simple case.

## A simplified Gaussian illustration

The code below is a pedagogical simplification of the Bayesian consistency logic.

Instead of solving the full Bray--Kreps equilibrium with a posterior over risk tolerance, it studies a linear observation model

$$
p_t = b r_t,
$$

where the single unknown coefficient $b$ plays the role of an identifiable structural parameter.

The point is to illustrate how Bayesian posteriors concentrate when the likelihood is correctly specified and the true parameter is identified by observations.

## The simplified learning model

### Setup

Agent $U$ **does not know** the equilibrium price function.

Specifically, $U$ does not know $b^*$.

However, $U$ does know:
* The distribution of $r_t$: $r_t \sim \mathcal{N}(0, \sigma^2)$ IID.
* That the price function is **linear**: $p_t = a + b r_t$ for some unknown $b$.
* The value of $a = 0$.

So $U$'s task is to learn the single parameter $b$ from observations of prices and (eventually) returns.

### Observing the signal

At date $t$, agent $U$ observes $p_t$.

The signal $U$ extracts is the return implied by the price:

$$
\hat{r}_t = \frac{p_t}{b_{t-1}}
$$

where $b_{t-1}$ is $U$'s current estimate of $b^*$.

After date $t$ trading and before date $t+1$, $U$ observes $r_t$ (the actual return is revealed, say through dividend payments).

### Bayesian updating

Agent $U$ begins with a **prior** distribution on $b$:

$$
b \sim \mathcal{N}(\mu_0, v_0)
$$

Given past data $(r_1, p_1), \ldots, (r_{t-1}, p_{t-1})$, agent $U$'s posterior on $b$ at date $t$ is

$$
b \mid \text{data} \sim \mathcal{N}(\mu_t, v_t)
$$

The posterior is updated using Bayes' rule.

Since $p_t = b \cdot r_t$ (with $a = 0$), each pair $(r_s, p_s)$ provides the observation $p_s = b \cdot r_s$, i.e., a noisy linear measurement of $b$.

For a Gaussian prior and Gaussian likelihood, the posterior updates as:

$$
v_t^{-1} = v_0^{-1} + \frac{1}{\sigma^2} \sum_{s=1}^{t} r_s^2
$$ (eq:posterior_precision)

$$
\mu_t = v_t \left( v_0^{-1} \mu_0 + \frac{1}{\sigma^2} \sum_{s=1}^{t} r_s p_s \right)
$$ (eq:posterior_mean)


Equations {eq}`eq:posterior_precision` and {eq}`eq:posterior_mean` follow from the standard Gaussian linear regression posterior.

Each observation $(r_s, p_s)$ with $p_s = b r_s + 0$ is treated as a noisy signal of $b$ with signal-to-noise ratio $r_s^2 / \sigma^2$.


### The simplified convergence result

For the simplified Gaussian model, standard Bayesian linear regression implies the following result.

**Proposition:** *For any prior $(\mu_0, v_0)$ with $v_0 < \infty$, as $t \to \infty$:*

$$
\mu_t \xrightarrow{a.s.} b^*, \qquad v_t \xrightarrow{a.s.} 0
$$

*That is, agent $U$'s posterior distribution on $b$ converges almost surely to a point mass at the true equilibrium value $b^*$.*

This statement is included to make the simulation transparent.

The formal propositions in {cite:t}`BrayKreps1987` are more general martingale convergence results for posterior assessments, and they are discussed below.

The intuition is straightforward:

* Each period adds a new observation $(r_t, p_t)$ with information content proportional to $r_t^2$.
* Since $r_t$ is IID with $E[r_t^2] = \sigma^2 > 0$, the cumulative information $\sum_{s=1}^t r_s^2 \to \infty$ by the law of large numbers.
* Therefore the posterior precision $v_t^{-1} \to \infty$, which means $v_t \to 0$.
* Since the observations are generated by the true $b^*$, the posterior mean $\mu_t$ converges to $b^*$.

The proof follows from standard results on Bayesian consistency for correctly specified Gaussian linear models.

## Simulating Bayesian learning

We now implement the Bayesian learning dynamics and verify convergence numerically.

### Parameters

```{code-cell} ipython3
# True equilibrium parameters
b_true = 2.0        # true b* in the REE

# Distribution of fundamentals
σ2 = 1.0            # variance of r_t

# Prior on b
μ_0 = 0.5           # prior mean (misspecified, true is 2.0)
v_0 = 2.0           # prior variance (diffuse)

# Simulation settings
T = 300             # time periods
N = 200             # number of Monte Carlo paths

np.random.seed(42)
```

### Bayesian updating function

```{code-cell} ipython3
def simulate_bayesian_learning(b_true, σ2, μ_0, v_0, T, N):
    """
    Simulate Bayesian learning of the REE slope parameter b*.

    Parameters
    ----------
    b_true : true equilibrium slope
    σ2     : variance of fundamentals r_t
    μ_0    : prior mean on b
    v_0    : prior variance on b
    T      : number of time periods
    N      : number of Monte Carlo paths

    Returns
    -------
    μ_paths : array (N, T) of posterior means over time
    v_paths : array (N, T) of posterior variances over time
    """
    # Draw fundamentals r_t for all paths
    r = np.random.normal(0, np.sqrt(σ2), size=(N, T))

    # Equilibrium prices: p_t = b_true * r_t
    p = b_true * r

    # Arrays to store posterior parameters
    μ_paths = np.empty((N, T))
    v_paths = np.empty((N, T))

    for i in range(N):
        # Initialize prior
        precision = 1.0 / v_0
        weighted_sum = μ_0 / v_0

        for t in range(T):
            # Each observation: p_s = b * r_s  =>  b = p_s / r_s (when r_s != 0)
            # Likelihood contribution: precision += r_s^2 / σ2
            #                          weighted_sum += r_s * p_s / σ2
            precision += r[i, t]**2 / σ2
            weighted_sum += r[i, t] * p[i, t] / σ2

            v_t = 1.0 / precision
            μ_t = v_t * weighted_sum

            μ_paths[i, t] = μ_t
            v_paths[i, t] = v_t

    return μ_paths, v_paths
```

### Running the simulation

```{code-cell} ipython3
μ_paths, v_paths = simulate_bayesian_learning(
    b_true, σ2, μ_0, v_0, T, N
)
```

### Plotting results

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: posterior learning paths
    name: fig-rle-posterior-learning
  image:
    alt: Posterior mean and posterior variance paths over time
---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

t_range = np.arange(1, T + 1)

# --- Left panel: posterior means ---
ax = axes[0]
for i in range(min(30, N)):
    ax.plot(t_range, μ_paths[i, :], color='steelblue', alpha=0.2, lw=2)

ax.plot(t_range, np.mean(μ_paths, axis=0), color='navy', lw=2,
        label='cross-path average')
ax.axhline(b_true, color='red', ls='--', lw=2, label=f'$b^* = {b_true}$')
ax.axhline(μ_0, color='gray', ls=':', lw=2, label=f'prior mean $= {μ_0}$')
ax.set_xlabel('$t$')
ax.set_ylabel('posterior mean $\\mu_t$')
ax.legend()

# --- Right panel: posterior variances ---
ax = axes[1]
for i in range(min(30, N)):
    ax.plot(t_range, v_paths[i, :], color='darkorange', alpha=0.2, lw=2)

ax.plot(t_range, np.mean(v_paths, axis=0), color='saddlebrown', lw=2,
        label='cross-path average')

# Theoretical rate: v_t ≈ σ2 / (t * σ2) = 1/t for large t
ax.plot(t_range, 1.0 / t_range, color='black', ls='--', lw=2,
        label='$1/t$ (theory)')
ax.set_xlabel('$t$')
ax.set_ylabel('posterior variance $v_t$')
ax.legend()

plt.tight_layout()
plt.show()
```

The left panel shows that regardless of the (misspecified) prior mean, agent $U$'s posterior mean converges to the true equilibrium value $b^* = 2$.

The right panel confirms that the posterior variance vanishes at rate $1/t$, consistent with the formula in {eq}`eq:posterior_precision`.

## Demand and equilibrium

To connect the learning story to market equilibrium, we can track how agent $U$'s **equilibrium demand** for the risky asset evolves.

Given $U$'s current beliefs about $b$ (summarized by $\mu_t$), $U$ estimates $r_t \approx p_t / \mu_t$ and formulates demand:

$$
x^U_t(\mu_t) = \frac{\theta^U}{\sigma^2} \cdot \left(\frac{p_t}{\mu_t} - p_t\right)
$$

As $\mu_t \to b^*$, this demand function converges to the demand implied by the rational expectations equilibrium.

The following code computes the demand trajectories.

```{code-cell} ipython3
def compute_demand(μ_t, p_t, σ2=1.0, θ_U=0.5):
    """
    Compute agent U's demand for the risky asset given beliefs μ_t.

    x^U = (θ_U / σ2) * (r_hat - p_t)
    where r_hat = p_t / μ_t is U's signal extraction.
    """
    r_hat = p_t / μ_t
    return (θ_U / σ2) * (r_hat - p_t)

# Single representative path
i_rep = 0
r_rep = np.random.normal(0, np.sqrt(σ2), T)
p_rep = b_true * r_rep

demand_path = np.array([
    compute_demand(μ_paths[i_rep, t], p_rep[t])
    for t in range(T)
])

# REE demand (what U would demand knowing b*)
demand_ree = np.array([
    compute_demand(b_true, p_rep[t])
    for t in range(T)
])

```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: demand convergence
    name: fig-rle-demand-convergence
  image:
    alt: Learning demand and rational expectations demand over time
---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_range, demand_path, color='steelblue', alpha=0.7,
        lw=2, label="$x^U_t$ (learning)")
ax.plot(t_range, demand_ree, color='red', ls='--', lw=2,
        label="$x^U_t$ (REE)")
ax.set_xlabel('$t$')
ax.set_ylabel("agent $U$'s demand $x^U_t$")
ax.legend()
plt.tight_layout()
plt.show()
```

## Two toy extensions

The next two simulations are not in Bray and Kreps.

They are included as small numerical illustrations of themes that appear in the paper: identification and feedback from beliefs to prices.

### 1. Two possible parameters

First suppose the simplified linear model can be generated by one of two possible values of $b^*$.

If the data identify which value is operating, Bayesian learning separates the two cases.

The following code illustrates this point with a mixture prior.

```{code-cell} ipython3
def simulate_two_parameters(b_values, σ2, T, N, seed=0):
    """
    Simulate learning when the prior is spread over two possible parameter values.
    Nature draws the true value from b_values.
    """
    rng = np.random.default_rng(seed)
    b_true_draw = rng.choice(b_values, size=N)

    μ_paths_all = np.empty((N, T))

    for i in range(N):
        b_i = b_true_draw[i]
        r = rng.normal(0, np.sqrt(σ2), T)
        p = b_i * r

        # Diffuse prior centered between the two equilibria
        μ_prior = np.mean(b_values)
        prec_prior = 1.0 / 4.0
        w_sum = μ_prior * prec_prior
        prec = prec_prior

        for t in range(T):
            prec += r[t]**2 / σ2
            w_sum += r[t] * p[t] / σ2
            μ_paths_all[i, t] = w_sum / prec

    return μ_paths_all, b_true_draw

b_values = [1.0, 3.0]
μ_two, b_drawn = simulate_two_parameters(b_values, σ2=1.0, T=200, N=300)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: two-parameter learning
    name: fig-rle-two-parameters
  image:
    alt: Posterior mean paths converging to two possible parameter values
---
fig, ax = plt.subplots(figsize=(10, 5))

colors = {b_values[0]: 'steelblue', b_values[1]: 'darkorange'}
for i in range(len(b_drawn)):
    c = colors[b_drawn[i]]
    ax.plot(np.arange(1, 201), μ_two[i, :], color=c, alpha=0.1, lw=2)

for bv, c in colors.items():
    ax.axhline(bv, color=c, ls='--', lw=2, label=f'$b^* = {bv}$')

ax.set_xlabel('$t$')
ax.set_ylabel('posterior mean $\\mu_t$')
ax.legend()
plt.tight_layout()
plt.show()
```

As expected, agent $U$ learns the **correct** equilibrium as long as the model is correctly specified and the true equilibrium generates the data.

The paper's non-identification example is different: with two informed agents, prices can reveal only the sum of their risk tolerances.

### 2. A self-referential price rule

The next toy model lets the price at date $t$ depend directly on agent $U$'s current belief $\mu_t$.

But $\mu_t$ is updated based on past prices.

This creates a **self-referential** system: beliefs drive prices, and prices update beliefs.

This is a deliberately simple stand-in for the paper's warning that learning changes behavior, which changes the data that agents observe.

The formal Bray--Kreps model handles this by making the whole price process part of a grand rational expectations equilibrium over an expanded state space.

```{code-cell} ipython3
def simulate_self_referential(b_true, σ2, μ_0, v_0, T, N,
                              α_demand=0.5):
    """
    Simulate the self-referential learning model where prices depend on
    current beliefs μ_t.

    p_t = b_true * r_t + α_demand * (μ_t - b_true) * r_t

    This captures the idea that as U's beliefs deviate from b*, the
    equilibrium price is distorted.
    """
    rng = np.random.default_rng(10)
    r_all = rng.normal(0, np.sqrt(σ2), (N, T))

    μ_paths_sr = np.empty((N, T))
    p_paths_sr = np.empty((N, T))

    for i in range(N):
        prec = 1.0 / v_0
        w_sum = μ_0 / v_0
        μ_t = μ_0

        for t in range(T):
            r_t = r_all[i, t]
            # Price is partly driven by current beliefs
            p_t = b_true * r_t + α_demand * (μ_t - b_true) * r_t

            # Update beliefs with this price
            prec += r_t**2 / σ2
            w_sum += r_t * p_t / σ2
            μ_t = w_sum / prec

            μ_paths_sr[i, t] = μ_t
            p_paths_sr[i, t] = p_t

    return μ_paths_sr, p_paths_sr

μ_sr, p_sr = simulate_self_referential(
    b_true, σ2, μ_0, v_0, T=200, N=100, α_demand=0.3
)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: self-referential learning
    name: fig-rle-self-referential
  image:
    alt: Self-referential posterior means and price paths over time
---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for i in range(30):
    ax.plot(np.arange(1, 201), μ_sr[i, :], color='steelblue', alpha=0.2, lw=2)
ax.plot(np.arange(1, 201), np.mean(μ_sr, axis=0), color='navy', lw=2,
        label='average $\\mu_t$')
ax.axhline(b_true, color='red', ls='--', lw=2, label=f'$b^* = {b_true}$')
ax.set_xlabel('$t$')
ax.set_ylabel('$\\mu_t$')
ax.legend()

ax = axes[1]
for i in range(30):
    ax.plot(np.arange(1, 201), p_sr[i, :], color='darkorange', alpha=0.15, lw=2)
ax.plot(np.arange(1, 201), np.mean(np.abs(p_sr), axis=0), color='saddlebrown', lw=2,
        label='average $|p_t|$')
ax.set_xlabel('$t$')
ax.set_ylabel('$p_t$')
ax.legend()

plt.tight_layout()
plt.show()
```

## Convergence of beliefs

Section 3 of {cite:t}`BrayKreps1987` proves two general convergence results.

Let $\Omega$ be the underlying state space, and let $H_t^n(p)$ be the information generated for agent $n$ by private information and observed equilibrium prices up to date $t$.

For any event $A$, the posterior assessment

$$
P^n(A \mid H_t^n(p))
$$

is a bounded martingale in $t$.

Their Proposition 1 is therefore

$$
P^n(A \mid H_t^n(p))
\xrightarrow{a.s.}
P^n(A \mid H_\infty^n(p)),
\qquad
H_\infty^n(p)=\bigvee_{t \geq 0} H_t^n(p).
$$

This is convergence of posterior assessments, not yet convergence to the truth.

If two agents' priors are mutually singular, the almost-sure statements need not hold on a common objective-probability set.

If their priors have the same null sets, simultaneous convergence is obtained outside a common null set.

Their Proposition 2 strengthens the result from events to whole posterior distributions.

When the parameter space is a complete separable metric space with its Borel sigma-field, regular posterior measures over that parameter space converge weakly almost surely.

Thus rational Bayesian learning always produces a limiting posterior, but additional identification assumptions are needed to say that the limiting posterior is correct.

## Identification in the Section 2 example

Section 4 returns to the two-agent example in which agent $U$ is uncertain about $\theta^I$.

Let $F_t$ be agent $U$'s posterior distribution over $\theta^I$ after observing the previous price, allocation, and return data.

By weak convergence of posteriors, $F_t$ converges almost surely to a limiting distribution $F_\infty$.

Bray and Kreps then show why this limiting posterior must be a point mass at the true $\theta^I$ in their example.

The argument has three parts.

First, because current equilibrium demand is continuous in the posterior distribution, prices converge to a limiting price functional

$$
p_\infty(s_t; F_\infty, \theta^I, \theta^U).
$$

Second, since the signals are IID, the empirical distribution of observed prices converges to the distribution of this limiting price functional.

Third, in this example that limiting price distribution is stochastically decreasing in $\theta^I$ when $F_\infty$ and $\theta^U$ are fixed.

Therefore the long-run distribution of prices identifies the true value of $\theta^I$.

This is the paper's concrete route from convergence of beliefs to convergence to correct beliefs.

It relies on smoothness, ergodicity, and identification, rather than on martingale convergence alone.

## Obstacles to convergence

While the positive convergence results are elegant, {cite:t}`BrayKreps1987` are careful to document when learning **fails** to produce convergence to REE.

### Obstacle 1: price maps might not settle down

The step from weak convergence of posteriors to convergence of prices requires smoothness of the equilibrium price functional.

Bray and Kreps stress that this can be hard, because small changes in a price function can produce large changes in the information communicated by prices.

Thus martingale convergence of beliefs does not by itself guarantee that the economy settles into a stationary price relation.

### Obstacle 2: prices might not identify the full parameter

Even if prices settle down, the long-run distribution of prices need not identify every structural parameter.

The paper gives a simple variant with two informed agents whose risk tolerances $\theta^{I1}$ and $\theta^{I2}$ are both unknown to the uninformed agent.

In that case, prices reveal only the sum $\theta^{I1}+\theta^{I2}$.

The uninformed agent cannot disentangle the two risk tolerances from price data alone.

For decisions in that example, learning the sum is enough, but it is not learning the full state.

### Obstacle 3: the truth might be outside the model

Section 5 compares the paper's rational-learning model with an example of {cite:t}`BlumeEasley1982`.

In that example, agents can converge to an incorrect model because the true stable price relation has zero prior probability under the models they entertain.

Bray and Kreps argue that this cannot occur in their rational-learning formulation except on a prior-null event.

The reason is that rational learning puts the possible price relations generated by the expanded state space inside the Bayesian model from the start.

## Learning *within* versus learning *about* a rational expectations equilibrium

One of the deepest conceptual contributions of {cite:t}`BrayKreps1987` is a distinction they draw in their concluding section between two fundamentally different notions of learning in a rational expectations context.

### The distinction

**Learning *within* a rational expectations equilibrium** is the subject of this lecture.

The phrase refers to Bayesian inference that takes place *inside* a correctly specified model of the economy.

In Bray and Kreps' rational-learning formulation, agents are uncertain about parameters such as other agents' risk tolerances.

But for every possible parameter realization, they are assumed to know the equilibrium price and allocation maps.

Their Bayesian learning model is therefore a large rational expectations equilibrium over an expanded state space.

This is why the martingale convergence theorem can be applied so cleanly.

**Learning *about* a rational expectations equilibrium** is a quite different enterprise.

Here agents do not begin with the equilibrium map already embedded in their model.

Instead, they try to infer the price-state relation from data generated while beliefs and behavior are changing.

This is the original problem mentioned at the start of the paper: learning changes behavior, and behavior changes the price-state relation being learned.

### Why rational learning has limited reach

Bray and Kreps call the expanded-state-space formulation natural but also identify its main flaw.

It avoids the question of how agents learn the relation between prices and states by assuming that agents already know the equilibrium for every possible economy in the state space.

In their conclusion, they say that their results do not satisfactorily answer the question "How does a rational expectations equilibrium come about?"

The reason is not that Bayesian convergence is false.

The reason is that the Bayesian agents must have extraordinary insight into the structure of the economy and the implied probabilities of events.

This is why the paper is useful both as a benchmark and as a warning.

It gives sharp restrictions on what rational learning can imply, but it does not provide a plausible behavioral story for attaining rational expectations.

### The role of "irrational" learning algorithms

This explains why the literature on learning *about* rational expectations equilibria --- going back to {cite:t}`Bray1982` and {cite:t}`BraySavin1984`, and extended in the influential work of {cite:t}`MarcetSargent1989jet` --- tends to rely on **ordinary least squares (OLS)** or other adaptive algorithms rather than Bayes' rule.

```{note}
{cite:t}`MarcetSargent1989jet` use some theorems about stochastic approximation to extend some of Bray and Savin's results to other settings.
```

In those models, agents estimate perceived laws of motion from observed data and update the estimates as new observations arrive.

Such rules are computationally tractable and can converge in important examples.

But they are **"irrational"** in Bray and Kreps' specific sense.

An agent who already understood the full equilibrium model would not generally use those rules as the Bayesian optimum.

The attraction of these rules is precisely that they ask a different question.

They ask whether agents using standard statistical procedures on the data generated by the model could eventually learn to form rational expectations.

Bray and Kreps are skeptical that rational Bayesian learning is behaviorally plausible, but they also use it to discipline adaptive learning stories.

Their proposed discipline is that a stationary limiting equilibrium should not leave agents' beliefs systematically contradicted by observations.

In the long run, they argue, equilibrium expectations must either keep changing or become rational.

There is a fundamental **epistemic tension** at the heart of learning about rational expectations equilibria:

* A fully rational (Bayesian, correctly specified) learner can only apply Bayes' rule to a model whose structure is *already known* but the structure of the REE is exactly what the agent is trying to learn.
* A learner who uses an adaptive algorithm (OLS, least-mean-squares, etc.) can potentially converge to the REE, but only by using a rule that cannot be derived from Bayesian rationality applied to a correctly specified model.

The Bray--Kreps rational-learning model avoids this tension by assumption: agent $U$ knows how each possible risk tolerance would map histories into equilibrium prices and trades.

The simplified Gaussian code example avoids it even more directly by replacing the equilibrium calculation with a fixed linear observation equation.

Both devices make Bayesian consistency transparent, but both sidestep the deeper difficulty of learning *about* an REE from scratch.


## Summary

This lecture has discussed ideas from {cite:t}`BrayKreps1987`:

1. **Rational learning** is modeled by expanding the state space to include unknown structural parameters such as risk tolerances.

2. **Posterior assessments converge** because conditional probabilities form bounded martingales.

3. **Posterior measures converge weakly** under standard topological assumptions on the parameter space.

4. **Correct learning** requires more than martingale convergence, because the limiting price distribution must identify the true parameter.

5. **In the paper's two-agent example**, the uninformed agent learns the informed agent's risk tolerance because the limiting price distribution is monotone in that parameter.

6. **Identification can fail** when prices reveal only a composite parameter, such as the sum of two informed agents' risk tolerances.

7. **Misspecification matters** because a stable price relation outside the learner's prior support cannot be learned by Bayes' rule.

8. **The simplified Gaussian simulation** illustrates posterior concentration in a fixed correctly specified model, not the full Bray--Kreps equilibrium calculation.

The broader message of Bray and Kreps is that while the mathematics of Bayesian learning is powerful, its application to learning *about* rational expectations equilibria is subtle and the conditions under which learning succeeds are more restrictive than they might appear.


## Exercises

```{exercise}
:label: rle_ex1

**Posterior Precision Growth**

In the Bayesian learning model above, the posterior precision is

$$
v_t^{-1} = v_0^{-1} + \frac{1}{\sigma^2} \sum_{s=1}^{t} r_s^2
$$

(a) Show that $v_t \to 0$ almost surely as $t \to \infty$, using the law of large numbers.

(b) What is the approximate rate of decay of $v_t$? That is, what does $t \cdot v_t$ converge to?

(c) Write Python code to verify your answer for $\sigma^2 = 1$ and a single simulated path of $T = 500$ periods.
```

```{solution-start} rle_ex1
:class: dropdown
```

**(a)** By the strong law of large numbers, since $r_s \sim \mathcal{N}(0, \sigma^2)$ IID with $E[r_s^2] = \sigma^2$:

$$
\frac{1}{t} \sum_{s=1}^t r_s^2 \xrightarrow{a.s.} \sigma^2 > 0
$$

Therefore

$$
\frac{1}{t} v_t^{-1} = \frac{v_0^{-1}}{t} + \frac{1}{\sigma^2} \cdot \frac{1}{t} \sum_{s=1}^t r_s^2 \xrightarrow{a.s.} \sigma^2 / \sigma^2 = 1
$$

So $v_t^{-1} \sim t$ and $v_t \to 0$ almost surely.

**(b)** From the above, $t \cdot v_t^{-1} \to 1$ implies $t \cdot v_t \to 1 / 1 = 1 / \sigma^2 \cdot \sigma^2 = 1$ when $\sigma^2 = 1$.

More precisely, $t \cdot v_t \to \sigma^2 / \sigma^2 = 1$ (since $v_t \approx \sigma^2 / (t \sigma^2) = 1/t$ for large $t$ when $\sigma^2 = 1$).

So $t \cdot v_t \to 1$ (when $\sigma^2 = 1$).

**(c)**

```{code-cell} ipython3
σ2_ex = 1.0
T_ex = 500
v0_ex = 2.0

np.random.seed(7)
r_ex = np.random.normal(0, np.sqrt(σ2_ex), T_ex)

precisions = np.empty(T_ex)
prec = 1.0 / v0_ex
for t in range(T_ex):
    prec += r_ex[t]**2 / σ2_ex
    precisions[t] = prec

v_t_ex = 1.0 / precisions

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(np.arange(1, T_ex + 1), v_t_ex, lw=2, label='$v_t$')
axes[0].plot(np.arange(1, T_ex + 1), 1.0 / np.arange(1, T_ex + 1),
             '--', lw=2, label='$1/t$')
axes[0].set_xlabel('$t$')
axes[0].set_ylabel('$v_t$')
axes[0].set_title('Posterior Variance Decay')
axes[0].legend()

axes[1].plot(np.arange(1, T_ex + 1),
             np.arange(1, T_ex + 1) * v_t_ex, lw=2, label='$t \\cdot v_t$')
axes[1].axhline(1.0, color='red', ls='--', lw=2, label='limit = 1')
axes[1].set_xlabel('$t$')
axes[1].set_ylabel('$t \\cdot v_t$')
axes[1].set_title('Normalized Variance Converges to 1')
axes[1].legend()

plt.tight_layout()
plt.show()
```

```{solution-end}
```

```{exercise}
:label: rle_ex2

**Effect of Prior Misspecification**

Suppose agent $U$ starts with a prior mean $\mu_0$ far from the true value $b^* = 2$.

(a) Simulate 100 paths of $T = 400$ periods for each of $\mu_0 \in \{-3, 0, 1, 3, 5\}$ and plot the average posterior mean across paths for each $\mu_0$.

(b) Does the prior mean affect the **rate** at which the posterior mean converges to $b^*$?

(c) Does the prior **variance** $v_0$ affect the rate? Verify by comparing $v_0 \in \{0.1, 1.0, 10.0\}$ with fixed $\mu_0 = 0$.
```

```{solution-start} rle_ex2
:class: dropdown
```

```{code-cell} ipython3
b_true_ex = 2.0
σ2_ex = 1.0
T_ex = 400
N_ex = 100
t_range_ex = np.arange(1, T_ex + 1)

# (a) and (b): different prior means
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for μ0 in [-3, 0, 1, 3, 5]:
    μ_p, _ = simulate_bayesian_learning(
        b_true_ex, σ2_ex, μ0, v_0=1.0, T=T_ex, N=N_ex
    )
    ax.plot(t_range_ex, np.mean(μ_p, axis=0), lw=2,
            label=f'$\\mu_0 = {μ0}$')

ax.axhline(b_true_ex, color='black', ls='--', lw=2,
           label=f'$b^* = {b_true_ex}$')
ax.set_xlabel('$t$')
ax.set_ylabel('$E[\\mu_t]$')
ax.set_title('Effect of Prior Mean on Convergence')
ax.legend(fontsize=8)

# (c): different prior variances
ax = axes[1]
for v0 in [0.1, 1.0, 10.0]:
    μ_p, _ = simulate_bayesian_learning(
        b_true_ex, σ2_ex, μ_0=0.0, v_0=v0, T=T_ex, N=N_ex
    )
    ax.plot(t_range_ex, np.mean(μ_p, axis=0), lw=2,
            label=f'$v_0 = {v0}$')

ax.axhline(b_true_ex, color='black', ls='--', lw=2,
           label=f'$b^* = {b_true_ex}$')
ax.set_xlabel('$t$')
ax.set_ylabel('$E[\\mu_t]$')
ax.set_title('Effect of Prior Variance on Convergence')
ax.legend()

plt.tight_layout()
plt.show()

print("Observations:")
print("(b) Prior mean affects the initial level but not the long-run rate.")
print("    All paths converge to b* = 2 at the same asymptotic rate.")
print("(c) A tighter prior (small v_0) slows initial adaptation but all")
print("    converge; a diffuse prior adapts quickly early on.")
```

```{solution-end}
```

```{exercise}
:label: rle_ex3

**Convergence with Non-Standard Fundamentals**

The convergence proof relies on $E[r_t^2] = \sigma^2 > 0$.

(a) Suppose $r_t$ follows a **mixture distribution**: with probability $0.5$ it equals $0$, and with probability $0.5$ it is drawn from $\mathcal{N}(0, 2\sigma^2)$.
Show that $E[r_t^2] = \sigma^2 > 0$ still holds, so convergence is guaranteed.

(b) Simulate $T = 500$ periods with $\sigma^2 = 1$ and $b^* = 2$ using this mixture distribution for $r_t$.
Plot the posterior mean and variance over time for 50 paths.

(c) Compare the speed of convergence to the Gaussian case.
Why does the mixture distribution slow convergence even though $E[r_t^2]$ is the same?
```

```{solution-start} rle_ex3
:class: dropdown
```

**(a)** Let $Z \sim \mathcal{N}(0, 2\sigma^2)$.
Then

$$
E[r_t^2] = 0.5 \cdot 0^2 + 0.5 \cdot E[Z^2] = 0.5 \cdot 2\sigma^2 = \sigma^2
$$

So $E[r_t^2] = \sigma^2 > 0$ and the strong law of large numbers guarantees $\sum_{s=1}^t r_s^2 / t \to \sigma^2$, ensuring convergence.

**(b) and (c)**

```{code-cell} ipython3
def simulate_learning_mixture(b_true, σ2, μ_0, v_0, T, N):
    """
    Simulate Bayesian learning with mixture fundamentals:
    r_t = 0 with prob 0.5, else N(0, 2*σ2) with prob 0.5.
    """
    rng = np.random.default_rng(42)

    μ_paths = np.empty((N, T))
    v_paths = np.empty((N, T))

    for i in range(N):
        prec = 1.0 / v_0
        w_sum = μ_0 / v_0

        for t in range(T):
            # Draw from mixture
            if rng.random() < 0.5:
                r_t = 0.0
            else:
                r_t = rng.normal(0, np.sqrt(2 * σ2))

            p_t = b_true * r_t

            prec += r_t**2 / σ2
            w_sum += r_t * p_t / σ2

            v_t = 1.0 / prec
            μ_t = v_t * w_sum

            μ_paths[i, t] = μ_t
            v_paths[i, t] = v_t

    return μ_paths, v_paths

σ2_ex = 1.0
T_ex = 500
N_ex = 50

# Gaussian case
μ_gauss, v_gauss = simulate_bayesian_learning(
    b_true=2.0, σ2=σ2_ex, μ_0=0.5, v_0=2.0, T=T_ex, N=N_ex
)

# Mixture case
μ_mix, v_mix = simulate_learning_mixture(
    b_true=2.0, σ2=σ2_ex, μ_0=0.5, v_0=2.0, T=T_ex, N=N_ex
)

t_range_ex = np.arange(1, T_ex + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(t_range_ex, np.mean(μ_gauss, axis=0), label='Gaussian $r_t$',
        color='steelblue', lw=2)
ax.plot(t_range_ex, np.mean(μ_mix, axis=0), label='Mixture $r_t$',
        color='darkorange', lw=2)
ax.axhline(2.0, color='red', ls='--', lw=2, label='$b^* = 2$')
ax.set_xlabel('$t$')
ax.set_ylabel('$E[\\mu_t]$')
ax.set_title('Posterior Mean: Gaussian vs Mixture')
ax.legend()

ax = axes[1]
ax.plot(t_range_ex, np.mean(v_gauss, axis=0), label='Gaussian $r_t$',
        color='steelblue', lw=2)
ax.plot(t_range_ex, np.mean(v_mix, axis=0), label='Mixture $r_t$',
        color='darkorange', lw=2)
ax.set_xlabel('$t$')
ax.set_ylabel('$E[v_t]$')
ax.set_title('Posterior Variance: Gaussian vs Mixture')
ax.legend()

plt.tight_layout()
plt.show()

print("The mixture distribution slows convergence because periods with r_t = 0")
print("provide NO information about b* (the observation p_t = 0 is uninformative).")
print("Even though E[r_t^2] = sigma^2, the variance of r_t^2 is larger under the")
print("mixture, leading to noisier information accumulation.")
```

```{solution-end}
```
