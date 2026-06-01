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

This lecture explores a classic question in economic theory: can agents *learn* their way to a rational expectations equilibrium?

The starting point is {cite:t}`BrayKreps1987`, which gives a rigorous model of Bayesian learning inside a rational expectations equilibrium.

In a rational expectations equilibrium, agents use market prices to make inferences about other agents' private information.

Each agent knows the *statistical relationship* between prices and the underlying payoff-relevant variables and that relationship is *correct* given the equilibrium.

But this raises a question: where does that knowledge come from?

The **rational learning** approach asks whether agents who start with uncertainty about the equilibrium price function can, over time, learn it from observations of past prices.

The key findings are:

* In every rational learning model, posterior assessments converge because they are bounded martingales.
* In the benchmark example, the uninformed agent learns the informed agent's risk tolerance.
* Correct learning requires identification, smooth equilibrium price maps, and positive prior probability for the true model.

This lecture presents the framework, explains the benchmark example, and provides Python code that solves the full equilibrium with rational learning.


The discussion also connects to earlier work by {cite:t}`Bray1982`, {cite:t}`BraySavin1984`, and the rational expectations literature of {cite:t}`Radner1979`, {cite:t}`grossman1976`, and {cite:t}`Jordan1982`.

Let's start with the following imports

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## The economy

### Agents and assets

The benchmark example is an infinitely repeated version of the information model in {cite:t}`GrossmanStiglitz1980`.

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

With $N$ agents and total risky-asset supply $N$, market clearing gives the **full communication equilibrium price**

$$
p_t
=
s_t
-
\frac{N\sigma^2}{\sum_{n=1}^N \theta^n}.
$$ (eq:bk-full-communication-price)

Thus if $\sum_n \theta^n$ is known, the price fully reveals $s_t$.

Following {cite:t}`Radner1979`, this is called a full communication rational expectations equilibrium.

Suppose that $\theta^I$ is unknown to agent $U$.

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

This is the main object learned in the benchmark example.

Even in this simple case, the equilibrium can be defined recursively but closed-form prices are unavailable.

## The rational learning equilibrium

The model has two pieces that interact at each date.

The first is the within-period equilibrium given the uninformed agent's current posterior on $\theta^I$.

The second is the Bayesian update of that posterior after the period closes.

### Uninformed demand given beliefs

Suppose at date $t$ agent $U$ has posterior density $f_t$ on $\theta^I$ supported on $[a, b]$.

Suppose the equilibrium informed trade and price are $X^I$ and $p$.

From {eq}`eq:bk-signal-implied`, conditional on $\theta^I$, agent $U$ infers $s_t = \sigma^2 X^I/\theta^I + p$.

Marginalising over $\theta^I \sim f_t$ and combining with $r_t = s_t + \epsilon_t$ where $\epsilon_t \sim \mathcal N(0,\sigma^2)$ gives the implied conditional distribution of $r_t$.

Because CARA preferences have no wealth effects, agent $U$'s problem reduces to

$$
\max_{x^U}\,
E\!\left[-\exp\!\left(-\tfrac{x^U(r_t - p)}{\theta^U}\right)\right],
$$

where the expectation integrates over $\theta^I \sim f_t$ and $\epsilon_t$.

Integrating out $\epsilon_t$ first and then $\theta^I$ yields

$$
E[u^U]
=
-\exp\!\left(\frac{(x^U)^2 \sigma^2}{2(\theta^U)^2}\right)
\int_a^b f_t(\theta)\,
\exp\!\left(-\frac{x^U \sigma^2 X^I}{\theta\,\theta^U}\right)
d\theta.
$$

The first-order condition rearranges to

$$
\frac{x^U}{\theta^U}
=
X^I \;
\frac{\int_a^b \theta^{-1} f_t(\theta)\,\exp\!\big(-x^U \sigma^2 X^I/(\theta\theta^U)\big)\,d\theta}
     {\int_a^b f_t(\theta)\,\exp\!\big(-x^U \sigma^2 X^I/(\theta\theta^U)\big)\,d\theta}.
$$ (eq:bk-foc)

The right-hand side is $X^I$ multiplied by a tilted expectation of $1/\theta^I$ under a weighting that depends on $x^U$ itself.

Equation {eq}`eq:bk-foc` implicitly defines $x^U(X^I; f_t)$, the uninformed agent's optimal demand at conjectured informed trade $X^I$ and posterior $f_t$.

The optimum does not depend separately on $p$, because the distribution of $r_t - p$ implied by the posterior depends only on $X^I$.

### Market clearing

Market clearing $X^I + x^U(X^I; f_t) = 2$ pins down the equilibrium informed trade $X^I_t$ as a function of beliefs alone.

Plugging $X^I_t$ into {eq}`eq:bk-informed-demand` recovers the equilibrium price

$$
p_t = s_t - \frac{\sigma^2 X^I_t}{\theta^I}.
$$ (eq:bk-price)

When $f_t$ collapses to a point mass at the true $\theta^I$, equation {eq}`eq:bk-foc` simplifies to $x^U/\theta^U = X^I/\theta^I$, and market clearing gives the full-communication allocation

$$
X^I_t = \frac{2\theta^I}{\theta^I + \theta^U},
\qquad
x^U_t = \frac{2\theta^U}{\theta^I + \theta^U}.
$$ (eq:bk-full-info-trade)

This is the CARA-Normal benchmark we will use to check the simulation.

### Bayesian update

After trading, agent $U$ observes $(p_t, x^U_t, r_t)$.

Market clearing gives $X^I_t = 2 - x^U_t$, and equation {eq}`eq:bk-signal-implied` assigns a candidate $s_t(\theta) = \sigma^2 X^I_t/\theta + p_t$ to each $\theta$.

Since $s_t \sim \mathcal N(\mu_s, \tau^2)$ independently of $\epsilon_t \sim \mathcal N(0,\sigma^2)$, the conditional density of $s_t$ given $r_t$ is Gaussian:

$$
g(s\mid r)
=
\phi\!\left(s;\, \frac{\sigma^2 \mu_s + \tau^2 r}{\sigma^2 + \tau^2},\,
                  \frac{\sigma^2 \tau^2}{\sigma^2 + \tau^2}\right),
$$

where $\phi(\cdot; m, v)$ denotes the Normal density with mean $m$ and variance $v$.

Bayes' rule then produces the posterior update

$$
f_{t+1}(\theta)
\propto
f_t(\theta)\;
g\!\left(\frac{\sigma^2 X^I_t}{\theta} + p_t \,\Big|\, r_t\right).
$$ (eq:bk-bayes)

This is the rule we simulate below.

## Computing the equilibrium

We discretise the support $[a,b]$ of $\theta^I$ on a fine grid and represent $f_t$ as a vector of density values.

There are three computational primitives.

* `uninformed_demand` solves the FOC in {eq}`eq:bk-foc` for $x^U(X^I; f)$ by root-finding.
* `equilibrium_XI` solves market clearing $X^I + x^U(X^I; f) = 2$ for $X^I_t$.
* `bayes_update` applies {eq}`eq:bk-bayes` and renormalises.

```{code-cell} ipython3
from scipy.optimize import brentq
```

```{code-cell} ipython3
def uninformed_demand(XI, f, θ_grid, θ_U, σ2):
    """
    Solve the FOC for the uninformed agent's demand x^U, given
    a conjectured informed trade XI and posterior density f.
    """
    with np.errstate(divide='ignore'):
        log_f = np.log(f)            # -inf where f == 0 is fine

    def foc(xU):
        z = xU * σ2 * XI / (θ_grid * θ_U)
        log_w = log_f - z
        M = log_w.max()
        w = np.exp(log_w - M)        # bounded in [0, 1], max value = 1
        num = np.sum(w / θ_grid)
        den = np.sum(w)
        return xU / θ_U - XI * num / den

    return brentq(foc, -20.0, 20.0, xtol=1e-10)
```

```{code-cell} ipython3
def equilibrium_XI(f, θ_grid, θ_U, σ2):
    """
    Solve market clearing X^I + x^U(X^I; f) = 2 for the
    equilibrium informed trade.
    """
    def mc(XI):
        return XI + uninformed_demand(XI, f, θ_grid, θ_U, σ2) - 2.0

    return brentq(mc, 1e-4, 4.0, xtol=1e-10)
```

```{code-cell} ipython3
def bayes_update(f, θ_grid, p_t, xU_t, r_t, σ2, τ2, μ_s):
    """
    Bayesian update of the posterior on θ^I given date-t observations.
    """
    XI = 2.0 - xU_t
    s_mean = (σ2 * μ_s + τ2 * r_t) / (σ2 + τ2)
    s_var  = σ2 * τ2 / (σ2 + τ2)
    s_implied = σ2 * XI / θ_grid + p_t

    log_like = -0.5 * (s_implied - s_mean)**2 / s_var
    log_like -= log_like.max()           # log-shift for stability
    f_new = f * np.exp(log_like)
    dθ = θ_grid[1] - θ_grid[0]
    f_new /= np.sum(f_new) * dθ
    return f_new
```

The simulation loop chains $(s_t, \epsilon_t)$ shocks through these three functions.

```{code-cell} ipython3
def simulate(θ_I_true, θ_U, σ2, μ_s, τ2,
             a, b, n_grid, T, prior=None, seed=42):
    """
    Simulate T periods of the Bray-Kreps rational-learning equilibrium.
    """
    rng = np.random.default_rng(seed)
    θ_grid = np.linspace(a, b, n_grid)
    dθ = θ_grid[1] - θ_grid[0]

    if prior is None:
        f = np.ones(n_grid) / (b - a)
    else:
        f = prior(θ_grid)
        f /= np.sum(f) * dθ

    s_seq   = rng.normal(μ_s, np.sqrt(τ2), T)
    eps_seq = rng.normal(0.0, np.sqrt(σ2), T)

    XI_path   = np.empty(T)
    p_path    = np.empty(T)
    r_path    = np.empty(T)
    post_mean = np.empty(T + 1)
    post_var  = np.empty(T + 1)
    post_mean[0] = np.sum(θ_grid * f) * dθ
    post_var[0]  = np.sum((θ_grid - post_mean[0])**2 * f) * dθ

    snap_times = {0, 5, 20, 50, 100, T}
    snapshots = {0: f.copy()}

    for t in range(T):
        XI  = equilibrium_XI(f, θ_grid, θ_U, σ2)
        xU  = 2.0 - XI
        p_t = s_seq[t] - σ2 * XI / θ_I_true
        r_t = s_seq[t] + eps_seq[t]
        f   = bayes_update(f, θ_grid, p_t, xU, r_t, σ2, τ2, μ_s)

        XI_path[t]   = XI
        p_path[t]    = p_t
        r_path[t]    = r_t
        post_mean[t + 1] = np.sum(θ_grid * f) * dθ
        post_var[t + 1]  = np.sum(
            (θ_grid - post_mean[t + 1])**2 * f
        ) * dθ
        if (t + 1) in snap_times:
            snapshots[t + 1] = f.copy()

    return dict(
        θ_grid=θ_grid,
        snapshots=snapshots,
        XI_path=XI_path,
        p_path=p_path,
        r_path=r_path,
        post_mean=post_mean,
        post_var=post_var,
    )
```

## Posterior concentration

We run the simulation with a uniform prior on $[0.5, 4]$ and true $\theta^I = 2$.

```{code-cell} ipython3
params = dict(
    θ_I_true=2.0,
    θ_U=1.0,
    σ2=1.0,
    μ_s=1.0,
    τ2=1.0,
    a=0.5,
    b=4.0,
    n_grid=300,
    T=200,
    seed=42,
)

res = simulate(**params)
```

The first picture shows snapshots of the posterior density at selected dates.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: posterior density over $\theta^I$ at selected dates
    name: fig-rle-posterior-density
  image:
    alt: Posterior density on theta^I concentrating around the true value
---
fig, ax = plt.subplots(figsize=(10, 5))
for t, ft in sorted(res['snapshots'].items()):
    ax.plot(res['θ_grid'], ft, lw=2, label=f't = {t}')
ax.axvline(params['θ_I_true'], color='black', ls='--', lw=1.5,
           label=r'$\theta^I_{\rm true}$')
ax.set_xlabel(r'$\theta^I$')
ax.set_ylabel('posterior density')
ax.legend()
plt.tight_layout()
plt.show()
```

The posterior tightens around $\theta^I_{\rm true} = 2$ as price and return data accumulate.

The next picture tracks the posterior mean and variance.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: posterior mean and variance over time
    name: fig-rle-posterior-moments
  image:
    alt: Posterior mean of theta^I converging to the true value and posterior variance vanishing
---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(np.arange(params['T'] + 1), res['post_mean'], lw=2)
ax.axhline(params['θ_I_true'], color='red', ls='--', lw=2,
           label=r'$\theta^I_{\rm true}$')
ax.set_xlabel('$t$')
ax.set_ylabel(r'$E_t[\theta^I]$')
ax.legend()

ax = axes[1]
ax.plot(np.arange(params['T'] + 1), res['post_var'], lw=2)
ax.set_xlabel('$t$')
ax.set_ylabel(r'${\rm Var}_t[\theta^I]$')

plt.tight_layout()
plt.show()
```

The posterior mean converges to the truth and the posterior variance vanishes.

This is the concrete manifestation of weak convergence of posteriors to a point mass at $\theta^I_{\rm true}$, which we describe in general terms below.

## Equilibrium trades and prices

The equilibrium informed trade $X^I_t$ depends only on $f_t$, not directly on $s_t$ or $\theta^I$.

As $f_t$ tightens around $\theta^I_{\rm true}$, $X^I_t$ approaches the full-information benchmark in {eq}`eq:bk-full-info-trade`.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: equilibrium trade and prices over time
    name: fig-rle-trade-price
  image:
    alt: Equilibrium informed trade X^I_t and price p_t over time
---
XI_full = 2 * params['θ_I_true'] / (params['θ_I_true'] + params['θ_U'])
p_mean_full = params['μ_s'] - params['σ2'] * XI_full / params['θ_I_true']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(res['XI_path'], lw=2, label='$X^I_t$ (learning)')
ax.axhline(XI_full, color='red', ls='--', lw=2,
           label='$X^I$ (full info)')
ax.set_xlabel('$t$')
ax.set_ylabel('$X^I_t$')
ax.legend()

ax = axes[1]
ax.plot(res['p_path'], lw=1.5, alpha=0.7, label='$p_t$ (learning)')
ax.axhline(p_mean_full, color='red', ls='--', lw=2,
           label='$E[p_t]$ (full info)')
ax.set_xlabel('$t$')
ax.set_ylabel('$p_t$')
ax.legend()

plt.tight_layout()
plt.show()
```

The left panel shows $X^I_t$ approaching the full-information allocation as beliefs concentrate.

The right panel shows the price path, which fluctuates because $p_t$ inherits the variation in $s_t$.


## Convergence of posterior assessments

The general theory in {cite:t}`BrayKreps1987` gives two convergence results.

Let $\Omega$ be the underlying state space, and let $H_t^n(p)$ be the information generated for agent $n$ by private information and observed equilibrium prices up to date $t$.

For any event $A$, the posterior assessment $P^n(A \mid H_t^n(p))$ is a bounded martingale in $t$.

The first convergence result is therefore an application of the martingale convergence theorem.

```{prf:proposition}
:label: prop-bk-event-convergence

For any event $A$,

$$
P^n(A \mid H_t^n(p))
\xrightarrow{a.s.}
P^n(A \mid H_\infty^n(p)),
\qquad
H_\infty^n(p)=\bigvee_{t \geq 0} H_t^n(p).
$$
```

This is convergence of posterior assessments, not yet convergence to "correct beliefs".

If two agents' priors are mutually singular, the almost-sure statements need not hold on a common objective-probability set.

If the priors have identical null sets, simultaneous convergence holds outside a common null set.

The second result sharpens the convergence from events to entire posterior distributions.

```{prf:proposition}
:label: prop-bk-measure-convergence

When the parameter space $\Theta$ is a complete separable metric space whose
Borel $\sigma$-field makes it a Borel space, fixed regular versions of the
conditional probabilities $P_t^n$ converge weakly $P^n$-a.s. to a regular
version $P_\infty^n$.
```

Thus rational Bayesian learning always produces a limiting posterior, but additional regularity is needed to ensure the limiting posterior assesses the truth correctly.

## Sharpening the convergence result

Now return to the two-agent example in which agent $U$ is uncertain about $\theta^I$.

Let $F_t$ be agent $U$'s posterior distribution over $\theta^I$ after observing the previous price, allocation, and return data.

By weak convergence of posteriors, $F_t$ converges almost surely to a limiting distribution $F_\infty$.

In the benchmark two-agent example, this limiting posterior must be a point mass at the true $\theta^I$.

The argument has three parts.

First, because current equilibrium demand is continuous in the posterior distribution, prices converge to a limiting price functional

$$
p_\infty(s_t; F_\infty, \theta^I, \theta^U).
$$

Second, since the signals are IID, the empirical distribution of observed prices converges to the distribution of this limiting price functional.

Third, in this example that limiting price distribution is stochastically decreasing in $\theta^I$ when $F_\infty$ and $\theta^U$ are fixed.

Therefore the long-run distribution of prices identifies the true value of $\theta^I$.

This is the concrete route from convergence of posterior assessments to convergence to the "correct beliefs".

It relies on smoothness, ergodicity, and identification, rather than on martingale convergence alone.

## Obstacles to convergence

While the positive convergence results are elegant, the same framework also shows when learning can *fail* to produce convergence to REE.

### Obstacle 1: price maps might not settle down

The step from weak convergence of posteriors to convergence of prices requires smoothness of the equilibrium price functional.

This can be hard, because small changes in a price function can produce large changes in the information communicated by prices.

Thus martingale convergence of beliefs does not by itself guarantee that the economy settles into a stationary price relation.

### Obstacle 2: prices might not identify the full parameter

Even if prices settle down, the long-run distribution of prices need not identify every structural parameter.

A simple variant has two informed agents whose risk tolerances $\theta^{I1}$ and $\theta^{I2}$ are both unknown to the uninformed agent.

In that case, prices reveal only the sum $\theta^{I1}+\theta^{I2}$.

The uninformed agent cannot disentangle the two risk tolerances from price data alone.

For decisions in that example, learning the sum is enough, but it is not learning the full state.

### Obstacle 3: the truth might be outside the model

An example of {cite:t}`BlumeEasley1982` illustrates a related misspecification problem.

In that example, agents can converge to an incorrect model because the true stable price relation has zero prior probability under the models they entertain.

In the rational-learning formulation, this kind of failure can occur only on a prior-null event.

The reason is that rational learning puts the possible price relations generated by the expanded state space inside the Bayesian model from the start.

## Learning *within* versus learning *about* a rational expectations equilibrium

One of the deepest conceptual points in {cite:t}`BrayKreps1987` is a distinction between two fundamentally different notions of learning in a rational expectations context.

### The distinction

**Learning *within* a rational expectations equilibrium** is the subject of this lecture.

The phrase refers to Bayesian inference that takes place *inside* a correctly specified model of the economy.

In the rational-learning formulation, agents are uncertain about parameters such as other agents' risk tolerances.

But for every possible parameter realization, they are assumed to know the equilibrium price and allocation maps.

The Bayesian learning model is therefore a large rational expectations equilibrium over an expanded state space.

This is why the martingale convergence theorem can be applied so cleanly.

**Learning *about* a rational expectations equilibrium** is a quite different enterprise.

Here agents do not begin with the equilibrium map already embedded in their model.

Instead, they try to infer the price-state relation from data generated while beliefs and behavior are changing.

This is the original problem that motivated the analysis: learning changes behavior, and behavior changes the price-state relation being learned.

### Why rational learning has limited value

The expanded-state-space formulation is natural, but it has a main flaw.

It avoids the question of how agents learn the relation between prices and states by assuming that agents already know the equilibrium for every possible economy in the state space.

It does not satisfactorily answer the question "How does a rational expectations equilibrium come about?"

The reason is not that Bayesian convergence is false.

The reason is that the Bayesian agents must have extraordinary insight into the structure of the economy and the implied probabilities of events.

This is why the framework is useful both as a benchmark and as a warning.

It gives sharp restrictions on what rational learning can imply, but it does not provide a plausible behavioral story for attaining rational expectations.

### The role of "irrational" learning algorithms

This explains why the literature on learning *about* rational expectations equilibria --- going back to {cite:t}`Bray1982` and {cite:t}`BraySavin1984`, and extended in the influential work of {cite:t}`MarcetSargent1989jet` --- tends to rely on **ordinary least squares (OLS)** or other adaptive algorithms rather than Bayes' rule.

```{note}
{cite:t}`MarcetSargent1989jet` use some theorems about stochastic approximation to extend some of Bray and Savin's results to other settings.
```

In those models, agents estimate perceived laws of motion from observed data and update the estimates as new observations arrive.

Such rules are computationally tractable and can converge in important examples.

But they are *"irrational"* in the specific sense used here.

An agent who already understood the full equilibrium model would not generally use those rules as the Bayesian optimum.

The attraction of these rules is precisely that they ask a different question.

They ask whether agents using standard statistical procedures on the data generated by the model could eventually learn to form rational expectations.

Rational Bayesian learning is demanding as a behavioral assumption, but it also disciplines adaptive learning stories.

The proposed discipline is that a stationary limiting equilibrium should not leave agents' beliefs systematically contradicted by observations.

In the long run, equilibrium expectations must either keep changing or become rational.

There is a fundamental tension at the heart of learning about rational expectations equilibria:

* A fully rational (Bayesian, correctly specified) learner can only apply Bayes' rule to a model whose structure is *already known*, but the structure of the REE is exactly what the agent is trying to learn.
* A learner who uses an adaptive algorithm (OLS, least-mean-squares, etc.) can potentially converge to the REE, but only by using a rule that cannot be derived from Bayesian rationality applied to a correctly specified model.

The rational-learning formulation avoids this tension by assumption: agent $U$ knows how each possible risk tolerance would map histories into equilibrium prices and trades.

The full equilibrium simulation above embeds exactly that knowledge, since `equilibrium_XI` is recomputed from $f_t$ at every date.

The device makes Bayesian consistency transparent, but it still sidesteps the deeper difficulty of learning *about* an REE from scratch.


## Summary

This lecture has discussed rational learning in the sense of {cite:t}`BrayKreps1987`:

1. **Rational learning** is modeled by expanding the state space to include unknown structural parameters such as risk tolerances.

2. **Posterior assessments converge** because conditional probabilities form bounded martingales.

3. **Posterior measures converge weakly** under standard topological assumptions on the parameter space.

4. **Correct learning** requires more than martingale convergence, because the limiting price distribution must identify the true parameter.

5. **In the two-agent example**, the uninformed agent learns the informed agent's risk tolerance because the limiting price distribution is monotone in that parameter.

6. **Identification can fail** when prices reveal only a composite parameter, such as the sum of two informed agents' risk tolerances.

7. **Misspecification matters** because a stable price relation outside the learner's prior support cannot be learned by Bayes' rule.

8. **The full simulation** above solves the within-period equilibrium from the posterior at every date and shows the posterior on $\theta^I$ collapsing to a point mass at $\theta^I_{\rm true}$.

The broader message is that while the mathematics of Bayesian learning is powerful, its application to learning *about* rational expectations equilibria is subtle and the conditions under which learning succeeds are more restrictive than they might appear.


## Exercises

```{exercise}
:label: rle_ex1

**Off-centre prior**

The baseline simulation uses a uniform prior on $\theta^I \in [0.5, 4]$.

(a) Re-run the simulation with a prior whose mass sits *above* the true value, for example

```
prior = lambda θ: (θ - 0.5)**3 * (4 - θ)
```

which peaks near $\theta = 3.1$.

(b) Plot the posterior mean over time alongside the uniform-prior baseline.

(c) Does the posterior eventually concentrate on $\theta^I_{\rm true}$, and how does the speed compare?
```

```{solution-start} rle_ex1
:class: dropdown
```

```{code-cell} ipython3
res_uniform = simulate(**params)

params_biased = dict(params)
params_biased['prior'] = lambda θ: (θ - 0.5)**3 * (4 - θ)
res_biased = simulate(**params_biased)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(res_uniform['post_mean'], lw=2, label='uniform prior')
ax.plot(res_biased['post_mean'], lw=2, label='off-centre prior')
ax.axhline(params['θ_I_true'], color='black', ls='--',
           label=r'$\theta^I_{\rm true}$')
ax.set_xlabel('$t$')
ax.set_ylabel(r'$E_t[\theta^I]$')
ax.legend()
plt.tight_layout()
plt.show()
```

The off-centre prior starts the posterior mean well above $\theta^I_{\rm true} = 2$, but Bayesian updating drives it down to the truth.

This is the rational-learning convergence result in action: any prior that puts positive density on $\theta^I_{\rm true}$ eventually concentrates around it.

```{solution-end}
```

```{exercise}
:label: rle_ex2

**Speed of learning across $\theta^I$**

Information from one period about $\theta^I$ comes through the implied signal

$$
s_t(\theta) = \frac{\sigma^2 X^I_t}{\theta} + p_t.
$$

The sensitivity $|\partial s_t/\partial \theta| = \sigma^2 X^I_t/\theta^2$ depends on the level of $\theta^I_{\rm true}$ through $X^I_t$ and $\theta^{-2}$.

(a) Run the simulation for $\theta^I_{\rm true} \in \{0.8, 2.0, 3.5\}$, holding everything else at the baseline.

(b) Plot the posterior variance on a log scale for each case.

(c) Which value of $\theta^I_{\rm true}$ yields the fastest concentration, and does the result match the sensitivity formula above?
```

```{solution-start} rle_ex2
:class: dropdown
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 5))
for θ_val in [0.8, 2.0, 3.5]:
    params_θ = dict(params)
    params_θ['θ_I_true'] = θ_val
    res_θ = simulate(**params_θ)
    ax.semilogy(res_θ['post_var'], lw=2,
                label=fr'$\theta^I_{{\rm true}} = {θ_val}$')
ax.set_xlabel('$t$')
ax.set_ylabel(r'${\rm Var}_t[\theta^I]$ (log scale)')
ax.legend()
plt.tight_layout()
plt.show()
```

The smallest $\theta^I_{\rm true}$ produces the steepest decline in posterior variance.

The reason is that the sensitivity $\sigma^2 X^I_t/\theta^2$ scales as $\theta^{-2}$ for fixed $X^I_t$, so the same noise level conveys much more information about $\theta^I$ when $\theta^I$ is small.

The asymmetry is a feature of the geometry of the equilibrium map, not of the learning rule itself.

```{solution-end}
```

```{exercise}
:label: rle_ex3

**Effect of return noise**

Larger $\sigma^2$ widens the conditional density of $s_t$ given $r_t$, which one might guess slows learning.

But $\sigma^2$ also scales the price intercept in {eq}`eq:bk-price`, so price dispersion across candidate $\theta$ grows with $\sigma^2$.

(a) Run the simulation with $\sigma^2 \in \{0.25, 1.0, 4.0\}$, keeping $\tau^2 = 1$ fixed.

(b) Plot the posterior variance on a log scale for each $\sigma^2$.

(c) Which effect dominates? Explain in terms of the signal-to-noise ratio for inferring $\theta^I$ from the price.
```

```{solution-start} rle_ex3
:class: dropdown
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 5))
for σ2_val in [0.25, 1.0, 4.0]:
    params_σ = dict(params)
    params_σ['σ2'] = σ2_val
    res_σ = simulate(**params_σ)
    ax.semilogy(res_σ['post_var'], lw=2,
                label=fr'$\sigma^2 = {σ2_val}$')
ax.set_xlabel('$t$')
ax.set_ylabel(r'${\rm Var}_t[\theta^I]$ (log scale)')
ax.legend()
plt.tight_layout()
plt.show()
```

The posterior variance falls *faster* for larger $\sigma^2$.

The reason is visible in the price equation $p_t = s_t - \sigma^2 X^I_t/\theta^I$: the price gap between two candidate $\theta$ values grows linearly with $\sigma^2$, while the conditional variance of the implied signal $g(s\mid r)$ is bounded above by $\tau^2$.

The Grossman-Stiglitz-style trade thus becomes more revealing about $\theta^I$ as the return shock $\epsilon_t$ becomes more volatile, even though each return is individually noisier.

```{solution-end}
```
