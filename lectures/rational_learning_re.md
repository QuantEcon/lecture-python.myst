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

This lecture explores an important question in economic theory: can agents *learn* their way to a rational expectations equilibrium?

If they can, then the rational expectations equilibrium can be justified as a dynamic attractor for learning processes.

The starting point is {cite:t}`BrayKreps1987`, which gives a rigorous model of Bayesian learning inside a rational expectations equilibrium.

In a rational expectations equilibrium, agents use market prices to make inferences about other agents' private information.

Each agent knows the *statistical relationship* between prices and the underlying payoff-relevant variables and that relationship is *correct* given the equilibrium.

But this raises a question: where does that knowledge come from?

The **rational learning** approach asks whether agents who start with uncertainty about the equilibrium price function can, over time, learn it from observations of past prices.

This lecture develops that idea through an asset-market model.

The aim is to see what rational learning can explain, and where its limits
appear, before turning to the computational illustration.

The discussion also connects to earlier work by {cite:t}`Bray1982`, {cite:t}`BraySavin1984`, and the rational expectations literature of {cite:t}`Radner1979`, {cite:t}`grossman1976`, and {cite:t}`Jordan1982`.

Let's start with the following imports

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
```

## The economy

Let's start with a simple asset-market model that captures the key features of rational learning.

The example is an infinitely repeated version of the information model in {cite:t}`GrossmanStiglitz1980`.


### Agents and assets

Each date is economically disconnected from the others, so agents start each period afresh.

There are two types of agents and two assets:

* A **safe asset** with net return normalized to zero.
* A **risky asset** endowed one unit per agent and traded at date $t$ at spot price $p_t$.

At each date $t = 0, 1, 2, \ldots$ the risky asset yields a gross return $r_t$ at date $t+1/2$.

An informed signal $s_t$ satisfies

$$
r_t = s_t + \epsilon_t,
\qquad
s_t \sim \mathcal N(\mu_s, \tau^2),
\qquad
\epsilon_t \sim \mathcal N(0,\sigma^2),
$$

where $\{s_t\}$ and $\{\epsilon_t\}$ are IID normal sequences and are mutually independent.

Common knowledge of the prior moments $(\mu_s, \tau^2)$ is what makes the price observation informative about $\theta^I$, as we will see.

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

Suppose now that $\theta^I$ is unknown to agent $U$.

Following {cite:t}`BrayKreps1987`, we treat this uncertainty by *expanding the state space*: we let the unknown parameter $\theta^I$ become a coordinate of the state, alongside the per-period shocks $(s_t, \epsilon_t)$.

Formally, the state space is $\Omega = \Theta \times \Phi^\infty$, where $\Theta = [a,b]$ supports the unknown $\theta^I$ and $\Phi$ supports each $(s_t, \epsilon_t)$.

Agent $U$ knows $\sigma^2$ and $\theta^U$, and starts with a prior density over $\theta^I$ on $[a,b]$.

This expansion is what turns the learning problem into Bayesian inference inside a single rational expectations equilibrium on $\Omega$.

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

This is the main object learned in the two-agent example.

Even in this simple case, the equilibrium can be defined recursively but closed-form prices are unavailable.

## The rational learning equilibrium

The model has two pieces that interact at each date.

The first is the within-period equilibrium given the uninformed agent's current posterior on $\theta^I$.

The second is the Bayesian update of that posterior after the period closes.

### Uninformed demand given beliefs

Suppose at date $t$ agent $U$ has posterior density $f_t$ on $\theta^I$, supported on $[a, b]$.

Suppose the equilibrium price is $p$ and the equilibrium informed trade $X^I = 2 - x^U$ has been inferred from market clearing.

Conditional on $\theta$, equation {eq}`eq:bk-signal-implied` pins down the signal as $s_t(\theta) = \sigma^2 X^I/\theta + p$.

Two sources of information about $\theta$ are therefore present at the start of date $t$ trading: the carried-over posterior $f_t(\theta)$ and the Gaussian prior $\phi_s(\cdot;\mu_s,\tau^2)$ on $s_t$ that values some implied signals as more plausible than others.

Bayes' rule combines them into the *intra-period* posterior

$$
f_t^{(p, X^I)}(\theta)
\propto
f_t(\theta)\,
\phi_s\!\left(\frac{\sigma^2 X^I}{\theta} + p\, ;\, \mu_s, \tau^2\right),
$$ (eq:bk-intra-posterior)

which is the posterior on $\theta^I$ that the agent actually uses to forecast $r_t$ before $r_t$ is observed.

Conditional on a candidate value $\theta$, the excess payoff on one unit of the risky asset is

$$
r_t - p
=
\frac{\sigma^2 X^I}{\theta} + \epsilon_t,
\qquad \epsilon_t \sim \mathcal N(0,\sigma^2).
$$

Because CARA preferences have no wealth effects, agent $U$'s problem reduces to

$$
\max_{x^U}\,
E[u^U(x^U, r_t, p)],
\qquad
u^U(x^U, r_t, p)
=
-\exp\!\left(-\frac{x^U(r_t-p)}{\theta^U}\right),
$$

where the expectation integrates over $\theta^I \sim f_t^{(p, X^I)}$ and $\epsilon_t$.

Substituting the conditional excess payoff and using the normal moment-generating formula gives

$$
E[u^U]
=
-\exp\!\left(\frac{(x^U)^2 \sigma^2}{2(\theta^U)^2}\right)
\int_a^b
f_t(\theta)\,
\phi_s\!\left(\tfrac{\sigma^2 X^I}{\theta} + p; \mu_s, \tau^2\right)
\exp\!\left(-\tfrac{x^U \sigma^2 X^I}{\theta\,\theta^U}\right)
d\theta,
$$

up to a $\theta$-independent constant absorbed in normalisation.

Define the tilted weight

$$
w(\theta;\, p, X^I, x^U)
=
f_t(\theta)\,
\phi_s\!\left(\tfrac{\sigma^2 X^I}{\theta} + p; \mu_s, \tau^2\right)
\exp\!\left(-\tfrac{x^U \sigma^2 X^I}{\theta\,\theta^U}\right).
$$ (eq:bk-weight)

The first-order condition rearranges to

$$
\frac{x^U}{\theta^U}
=
X^I \;
\frac{\int_a^b \theta^{-1}\, w(\theta;\, p, X^I, x^U)\, d\theta}
     {\int_a^b w(\theta;\, p, X^I, x^U)\, d\theta}.
$$ (eq:bk-foc)

The right-hand side is $X^I$ multiplied by a tilted expectation of $1/\theta^I$ under the weighting in {eq}`eq:bk-weight`.

Equation {eq}`eq:bk-foc` implicitly defines $x^U(p, X^I; f_t)$, the uninformed agent's optimal demand at observed price $p$, conjectured informed trade $X^I$, and prior posterior $f_t$.

Dependence on $p$ enters through the prior weight $\phi_s$: at higher prices, candidate values of $\theta$ that imply $s_t$ above the prior mean become less plausible, so the agent's demand schedule slopes downward in $p$ as expected.

### Market clearing

Equilibrium requires that the informed and uninformed demands sum to the total endowment.

Substituting {eq}`eq:bk-informed-demand` and the implicit function $x^U(p, X^I; f_t)$, the equilibrium $(p_t, X^I_t)$ satisfies the two equations

$$
X^I_t = \frac{\theta^I}{\sigma^2}(s_t - p_t),
\qquad
X^I_t + x^U(p_t, X^I_t; f_t) = 2.
$$ (eq:bk-mc)

Eliminating $X^I_t$ between the two leaves a single root-finding problem for $p_t$.

Combining the two equations, the equilibrium price has the form

$$
p_t = s_t - \frac{\sigma^2 X^I_t}{\theta^I}.
$$ (eq:bk-price)

When $f_t$ collapses to a point mass at the true $\theta^I$, equation {eq}`eq:bk-foc` simplifies to $x^U/\theta^U = X^I/\theta^I$, and market clearing gives the full-communication allocation

$$
X^I_t = \frac{2\theta^I}{\theta^I + \theta^U},
\qquad
x^U_t = \frac{2\theta^U}{\theta^I + \theta^U}.
$$ (eq:bk-full-info-trade)

This is the full-communication allocation we will use to check the simulation.

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

The three computational primitives are:

* `uninformed_demand` solves the FOC in {eq}`eq:bk-foc` for $x^U(p, X^I; f)$ by root-finding.
* `equilibrium_price` solves the market-clearing system {eq}`eq:bk-mc` for $p_t$.
* `bayes_update` applies {eq}`eq:bk-bayes` and renormalises.

```{code-cell} ipython3
def uninformed_demand(p, XI, f, θ_grid, θ_U, σ2, μ_s, τ2):
    """
    Solve the FOC for x^U(p, X^I; f), the uninformed
    agent's optimal demand given observed price p, conjectured
    informed trade XI, and carried-over posterior density f.
    """
    with np.errstate(divide='ignore'):
        log_f = np.log(f)
    s_implied = σ2 * XI / θ_grid + p
    log_phi_s = -0.5 * (s_implied - μ_s)**2 / τ2  # prior weight on s_t

    def foc(xU):
        z = xU * σ2 * XI / (θ_grid * θ_U)
        log_w = log_f + log_phi_s - z
        M = log_w.max()
        w = np.exp(log_w - M)
        num = np.sum(w / θ_grid)
        den = np.sum(w)
        return xU / θ_U - XI * num / den

    return brentq(foc, -50.0, 50.0, xtol=1e-10)
```

```{code-cell} ipython3
def equilibrium_price(s_t, θ_I_true, f, θ_grid, θ_U, σ2, μ_s, τ2):
    """
    Solve the market-clearing system for the equilibrium
    price p_t given signal s_t, true informed risk tolerance
    θ_I_true, and posterior f.
    """
    def mc_residual(p):
        XI = θ_I_true * (s_t - p) / σ2
        xU = uninformed_demand(p, XI, f, θ_grid, θ_U, σ2, μ_s, τ2)
        return XI + xU - 2.0

    return brentq(mc_residual, s_t - 10.0, s_t, xtol=1e-8)
```

```{code-cell} ipython3
def bayes_update(f, θ_grid, p_t, xU_t, r_t, σ2, τ2, μ_s):
    """
    Bayesian update of the posterior on θ^I given the date-t
    observations (p_t, x^U_t, r_t).
    """
    XI = 2.0 - xU_t
    s_mean = (σ2 * μ_s + τ2 * r_t) / (σ2 + τ2)
    s_var = σ2 * τ2 / (σ2 + τ2)
    s_implied = σ2 * XI / θ_grid + p_t

    log_like = -0.5 * (s_implied - s_mean)**2 / s_var
    log_like -= log_like.max()  # log-shift for stability
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

    s_seq = rng.normal(μ_s, np.sqrt(τ2), T)
    eps_seq = rng.normal(0.0, np.sqrt(σ2), T)

    XI_path = np.empty(T)
    p_path = np.empty(T)
    r_path = np.empty(T)
    post_mean = np.empty(T + 1)
    post_var = np.empty(T + 1)
    post_mean[0] = np.sum(θ_grid * f) * dθ
    post_var[0] = np.sum((θ_grid - post_mean[0])**2 * f) * dθ

    snap_times = {0, 5, 20, 50, 100, T}
    snapshots = {0: f.copy()}

    for t in range(T):
        p_t = equilibrium_price(
            s_seq[t], θ_I_true, f, θ_grid, θ_U, σ2, μ_s, τ2
        )
        XI = θ_I_true * (s_seq[t] - p_t) / σ2
        xU = 2.0 - XI
        r_t = s_seq[t] + eps_seq[t]
        f = bayes_update(f, θ_grid, p_t, xU, r_t, σ2, τ2, μ_s)

        XI_path[t] = XI
        p_path[t] = p_t
        r_path[t] = r_t
        post_mean[t + 1] = np.sum(θ_grid * f) * dθ
        post_var[t + 1] = np.sum(
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

The equilibrium informed trade $X^I_t$ depends on the current signal $s_t$, on $\theta^I_{\rm true}$, and on the carried-over posterior $f_t$, all through the market-clearing system {eq}`eq:bk-mc`.

As $f_t$ tightens around $\theta^I_{\rm true}$, the average $X^I_t$ approaches the full-communication allocation in {eq}`eq:bk-full-info-trade`.

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

The simulation suggests three empirical facts about this equilibrium.

The posterior density on $\theta^I$ concentrates around the true value, the posterior variance vanishes, and the equilibrium informed trade $X^I_t$ converges to its full-information benchmark.

The next sections ask what general theorems guarantee these outcomes and which assumptions they rely on.

The plan is to first state the two convergence theorems of {cite:t}`BrayKreps1987` for the abstract rational-learning model, then specialize to the two-agent example to identify the hypotheses that imply concentration on the true $\theta^I$, and finally explain when those hypotheses can fail.

## Convergence of posterior assessments

Let $(\Omega, \mathcal F)$ be a measurable space carrying the equilibrium.

In the two-agent example, $\Omega = \Theta \times \Phi^{\infty}$, where $\Theta = [a,b] \times \{\theta^U\}$ collects the structural parameters, $\Phi$ collects the per-period shocks $(s_t, \epsilon_t)$, and $\mathcal F$ is the product Borel $\sigma$-field.

Agent $n$ enters date $0$ with a prior probability measure $P^n$ on $(\Omega, \mathcal F)$.

Let $G_t^n \subseteq \mathcal F$ denote the $\sigma$-field generated by agent $n$'s private information through date $t$, and let

$$
H_t^n(p)
=
G_t^n \vee \sigma(p_0, p_1, \dots, p_t)
$$

be the $\sigma$-field that adds observation of equilibrium prices through date $t$.

The tail $\sigma$-field is

$$
H_\infty^n(p)
=
\bigvee_{t \ge 0} H_t^n(p).
$$

The first result, due to {cite:t}`BrayKreps1987`, states that the conditional probability of any event converges almost surely.

```{prf:proposition}
:label: prop-bk-event-convergence

Fix an agent $n$ and an event $A \in \mathcal F$.

The process $M_t = E^n[\mathbf 1_A \mid H_t^n(p)]$ is a $P^n$-bounded martingale with respect to $(H_t^n(p))_{t \ge 0}$, and

$$
\lim_{t\to\infty}
E^n[\mathbf 1_A \mid H_t^n(p)]
=
E^n[\mathbf 1_A \mid H_\infty^n(p)],
\qquad P^n\text{-a.s.}
$$
```

The proof is the bounded martingale convergence theorem, with $M_t \in [0,1]$ supplying the uniform integrability needed for the limit identification.

{prf:ref}`prop-bk-event-convergence` is convergence of posterior assessments, not convergence to "correct" beliefs.

Two qualifications are worth stating.

The "a.s." statement is relative to agent $n$'s own prior $P^n$, so if two priors $P^n$ and $P^{n'}$ are mutually singular, the conclusion need not hold simultaneously on a common $P$-positive event.

If the priors share a common null collection, simultaneous convergence holds outside a common null set.

The second result in {cite:t}`BrayKreps1987` sharpens convergence from individual events to the entire posterior measure on the parameter space, given a topological assumption on $\Theta$.

```{prf:assumption}
:label: assum-bk-borel

The parameter space $\Theta$ is a complete separable metric (Polish) space, and the Borel $\sigma$-field on $\Theta$ generated by its open sets makes $(\Theta, \mathcal B(\Theta))$ a Borel space.
```

In the two-agent example $\Theta = [a,b]$ trivially satisfies this assumption.

Under {prf:ref}`assum-bk-borel` one can fix regular versions of the conditional probabilities: maps

$$
P_t^n: \Omega \to \mathcal P(\Theta),
\qquad
\omega \mapsto P_t^n(\omega),
$$

such that for each measurable $A \subseteq \Theta$, $\omega \mapsto P_t^n(\omega)(A)$ is a version of $E^n[\mathbf 1_{A \times \Phi^\infty} \mid H_t^n(p)](\omega)$, and $P_t^n(\omega) \in \mathcal P(\Theta)$ is a probability measure $P^n$-a.s.

The sharpened convergence result says these regular versions converge weakly almost surely.

```{prf:proposition}
:label: prop-bk-measure-convergence

Under {prf:ref}`assum-bk-borel`, the regular versions $P_t^n$ converge weakly to a regular version $P_\infty^n$, $P^n$-a.s.

Equivalently, for $P^n$-a.e. $\omega$ and every bounded continuous $f: \Theta \to \mathbb R$,

$$
\int_\Theta f \, dP_t^n(\omega)
\xrightarrow[t \to \infty]{}
\int_\Theta f \, dP_\infty^n(\omega).
$$
```

The proof in {cite:t}`BrayKreps1987` applies {prf:ref}`prop-bk-event-convergence` to a countable disjoint partition of $\Theta$ by $1/k$-balls, which exists because $\Theta$ is Polish, and then invokes the Portmanteau characterisation of weak convergence on bounded continuous functions.

Rational Bayesian learning therefore always produces a limiting posterior measure.

But {prf:ref}`prop-bk-measure-convergence` alone does not pin down what that limit is, and additional structure is needed before the limit assesses the truth correctly.

## Sharpening the convergence result

We now return to the two-agent example and identify hypotheses under which $P_\infty^U$ is a point mass at the true $\theta^I$.

Write $F_t$ for the CDF of agent $U$'s posterior on $\theta^I$ at date $t$ after observing $(r_{t-1}, p_{t-1}, x^U_{t-1})$ and all earlier data.

{prf:ref}`prop-bk-measure-convergence` yields a random CDF $F_\infty$ such that $F_t$ converges weakly to $F_\infty$, $P^U$-a.s.

Three hypotheses sharpen this to concentration on the truth, corresponding to the three steps in {cite:t}`BrayKreps1987`.

```{prf:assumption}
:label: assum-bk-continuity

The equilibrium uninformed demand $x^U(p, F)$ is continuous in $F$ with respect to weak convergence, uniformly in $p$ on a $P^U$-full-measure set of prices.
```

```{prf:assumption}
:label: assum-bk-identification

For fixed $\theta^U$ and limiting posterior $F_\infty$, the marginal distribution of the limiting price functional $p_\infty(\,\cdot\,; F_\infty, \theta^I, \theta^U)$ is strictly monotone in $\theta^I$ in the first-order-stochastic-dominance order.

That is, $\theta^I \neq \theta^{I\,\prime}$ implies $p_\infty(s; F_\infty, \theta^I, \theta^U)$ and $p_\infty(s; F_\infty, \theta^{I\,\prime}, \theta^U)$ have distinct CDFs when $s$ is drawn from its marginal distribution.
```

In the lecture's CARA-Normal setup, {prf:ref}`assum-bk-continuity` holds because the FOC {eq}`eq:bk-foc` defines $x^U$ as a continuous functional of $F$ under weak convergence through bounded integrals, and {prf:ref}`assum-bk-identification` holds because the equilibrium price has the form $p_t = s_t - \sigma^2 X^I_t / \theta^I$ with $X^I_t > 0$ on a full-measure set.

The IID assumption on $\{s_t\}$, already part of the model, supplies the ergodicity used in step 2 below.

Under these three assumptions and the IID signal sequence, the limiting posterior in the two-agent example concentrates on the truth.

```{prf:proposition}
:label: prop-bk-sharpening

Suppose $\theta^I_{\rm true} \in [a,b]$ and the prior $f_0$ puts positive density in every neighbourhood of $\theta^I_{\rm true}$.

Under {prf:ref}`assum-bk-borel`, {prf:ref}`assum-bk-continuity`, and {prf:ref}`assum-bk-identification`, and given the IID signal sequence $\{s_t\}$, the limiting posterior on $\theta^I$ satisfies

$$
F_\infty
=
\delta_{\theta^I_{\rm true}}
\qquad P^U\text{-a.s.}
$$
```

The proof has three steps.

*Step 1: price functional convergence.*

{prf:ref}`assum-bk-continuity` and the weak convergence $F_t \Rightarrow F_\infty$ from {prf:ref}`prop-bk-measure-convergence` imply that equilibrium demands $x^U(p, F_t)$ converge to $x^U(p, F_\infty)$.

Combining with market clearing and the price equation {eq}`eq:bk-price` gives $p_t - p_\infty(s_t; F_\infty, \theta^I, \theta^U) \to 0$ on a $P^U$-full-measure set.

*Step 2: the limit price distribution is observable.*

Since the deviation $p_t - p_\infty(s_t; F_\infty, \theta^I, \theta^U) \to 0$ almost surely and $\{s_t\}$ is IID, the empirical distribution of observed prices has the same limit as the empirical distribution of the limiting price functional.

The latter equals the distribution of $p_\infty(s; F_\infty, \theta^I, \theta^U)$ for $s \sim \mathcal N(\mu_s, \tau^2)$, and that limit is $H_\infty^U(p)$-measurable as a long-run frequency of an observable sequence.

*Step 3: identification.*

{prf:ref}`assum-bk-identification` makes the marginal distribution of $p_\infty$ a strictly monotone function of $\theta^I$ given $(F_\infty, \theta^U)$.

Combined with step 2, this means $\theta^I$ is itself $H_\infty^U(p)$-measurable, so for any subinterval $[c,d] \subseteq [a,b]$ the limiting posterior satisfies $P_\infty^U(\theta^I \in [c,d]) = \mathbf 1_{\{\theta^I_{\rm true} \in [c,d]\}}$.

Combining steps 1, 2, and 3 yields $F_\infty = \delta_{\theta^I_{\rm true}}$.

The numerical simulation above is consistent with this result.

The posterior density on $\theta^I$ collapses to a spike at $\theta^I_{\rm true}=2$, and the equilibrium informed trade $X^I_t$ converges to the full-information value $2\theta^I_{\rm true}/(\theta^I_{\rm true} + \theta^U)$.

Hence, the path connecting {prf:ref}`prop-bk-event-convergence` (martingale convergence) to {prf:ref}`prop-bk-sharpening` (concentration on the truth) depends on three model-specific ingredients: continuity, ergodicity, and identification.

## Obstacles to convergence

It is natural to ask when these ingredients can fail, and what the consequences are for learning.

### Obstacle 1: failure of continuity

If {prf:ref}`assum-bk-continuity` fails, step 1 of the proof breaks.

When the equilibrium price functional is discontinuous in $F$, small changes in beliefs can produce large changes in the information content of prices, and weak convergence of beliefs need not imply convergence of prices.

{cite:t}`BrayKreps1987` flag this as the most delicate step in their argument.

Continuity of $x^U(p, F)$ in $F$ is automatic in this lecture because the FOC integrates a bounded continuous function against $F$, but verifying it in richer market structures often requires non-trivial regularity arguments.

### Obstacle 2: failure of identification

If {prf:ref}`assum-bk-identification` fails, step 3 breaks even when steps 1 and 2 succeed.

Consider a variant with two informed agents and risk tolerances $\theta^{I1}, \theta^{I2}$ both unknown to the uninformed agent.

With three agents each endowed with one unit of the risky asset, the full-communication formula {eq}`eq:bk-full-communication-price` gives

$$
p_t
=
s_t
-
\frac{3\sigma^2}{\theta^{I1} + \theta^{I2} + \theta^U},
$$

which depends on $(\theta^{I1}, \theta^{I2})$ only through the sum $\theta^{I1}+\theta^{I2}$.

{prf:ref}`prop-bk-measure-convergence` still applies, but $F_\infty$ is supported on the level set

$$
\{(\theta_1, \theta_2) \in [a,b]^2 : \theta_1 + \theta_2 = \theta^{I1}_{\rm true} + \theta^{I2}_{\rm true}\},
$$

not on the singleton $\{(\theta^{I1}_{\rm true},\theta^{I2}_{\rm true})\}$.

Convergence occurs, but to a manifold of observationally equivalent parameter values rather than to the truth.

### Obstacle 3: misspecification

A separate obstacle arises if the true pricing relation lies outside the agent's prior support.

{cite:t}`BlumeEasley1982` give a stylised version of this obstacle, and {doc}`likelihood_ratio_process_2` develops the Blume-Easley heterogeneous-beliefs model in this lecture series.

Each agent entertains two competing models $\psi_n^0$ and $\psi_n^1$ over $(I_t, p_t)$, and an equilibrium can exist in which agents assign asymptotic probability one to a model that places zero probability on the actually-observed price relation.

In strict rational learning the agent's prior must be supported on Bayesian-consistent models in the expanded state space, so this failure can occur only on a $P^U$-null event.

Rational learning embeds every candidate pricing relation in the prior from date zero, so any candidate with positive prior weight cannot be dominated by one with zero prior weight no matter what the data say.

## Learning within versus learning about a rational expectations equilibrium

The framework above points to an important conceptual distinction in {cite:t}`BrayKreps1987`.

### The distinction

Learning *within* a rational expectations equilibrium is the topic of this lecture.

It is Bayesian inference inside a correctly specified model: {prf:ref}`assum-bk-borel`, {prf:ref}`assum-bk-continuity`, and {prf:ref}`assum-bk-identification` all hold, and the prior puts positive weight on the truth.

Agent $U$ is uncertain about $\theta^I$, but for every candidate value he already knows the equilibrium price and allocation maps.

The expanded-state-space formulation $\Omega = \Theta \times \Phi^\infty$ embeds a rational expectations equilibrium on the larger space, and inference reduces to conditional probability over $\Theta$.

Learning *about* a rational expectations equilibrium is a fundamentally different exercise.

The agent does not begin with the equilibrium map embedded in his probability model.

Instead he must infer the price-state relation from data generated while his own beliefs and behavior co-evolve with the data.

### The trade-off

The two notions sit on opposite sides of a precise trade-off.

A correctly-specified Bayesian learner enjoys the convergence guarantees in {prf:ref}`prop-bk-event-convergence` and {prf:ref}`prop-bk-measure-convergence`, but only because the equilibrium has been built into the prior from date zero.

An adaptive learner who treats the price-state relation as something to be estimated can hope to discover it from data, but the estimator he uses cannot be derived from Bayes' rule applied to a correctly specified model.

Bayesian rational learning can update among equilibrium maps already included in the agent's prior, but it does not explain how agents come to obtain those maps in the first place.

The literature on learning *about* rational expectations equilibria, beginning with {cite:t}`Bray1982` and {cite:t}`BraySavin1984` and extended by {cite:t}`MarcetSargent1989jet`, takes the second side of the trade-off and replaces Bayes' rule with **ordinary least squares** or related recursive estimators.

The companion lecture {doc}`ls_learning` develops this least-squares-learning framework in self-referential models and traces the resulting dynamics through the associated ordinary differential equation.

Those rules are computationally tractable and converge in important examples, but they are *not* Bayesian-optimal under any correctly specified prior.

## Summary

This lecture implemented the rational-learning equilibrium of {cite:t}`BrayKreps1987`.

Posterior assessments converge by bounded martingale convergence ({prf:ref}`prop-bk-event-convergence`), and posterior measures converge weakly under a Polish-Borel assumption ({prf:ref}`prop-bk-measure-convergence`).

Concentration on the truth additionally requires continuity ({prf:ref}`assum-bk-continuity`), ergodicity, and identification ({prf:ref}`assum-bk-identification`); each obstacle above is a failure of one of these.

The simulation confirms both conclusions: the posterior on $\theta^I$ collapses to $\theta^I_{\rm true}$ and the equilibrium informed trade reaches its full-information value.

Rational learning describes the limits of Bayesian inference *given* the equilibrium structure; adaptive learning, in {doc}`ls_learning`, describes how that structure can be learned in the first place.


## Exercises

````{exercise}
:label: rle_ex1

*Off-center prior*

The baseline simulation uses a uniform prior on $\theta^I \in [0.5, 4]$.

1. Re-run the simulation with a prior whose mass sits *above* the true value, for example

```python
prior = lambda θ: (θ - 0.5)**3 * (4 - θ)
```

which peaks near $\theta = 3.1$.

2. Plot the posterior mean over time alongside the uniform-prior baseline.

3. Does the posterior eventually concentrate on $\theta^I_{\rm true}$, and how does the speed compare?
````

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
ax.plot(res_biased['post_mean'], lw=2, label='off-center prior')
ax.axhline(params['θ_I_true'], color='black', ls='--',
           label=r'$\theta^I_{\rm true}$')
ax.set_xlabel('$t$')
ax.set_ylabel(r'$E_t[\theta^I]$')
ax.legend()
plt.tight_layout()
plt.show()
```

The off-center prior starts the posterior mean well above $\theta^I_{\rm true} = 2$, but Bayesian updating drives it down to the truth.

This is the rational-learning convergence result in action: any prior that puts positive density on $\theta^I_{\rm true}$ eventually concentrates around it.

```{solution-end}
```

```{exercise}
:label: rle_ex2

*Speed of learning across $\theta^I$*

Information from one period about $\theta^I$ comes through the implied signal

$$
s_t(\theta) = \frac{\sigma^2 X^I_t}{\theta} + p_t.
$$

The sensitivity $|\partial s_t/\partial \theta| = \sigma^2 X^I_t/\theta^2$ depends on the level of $\theta^I_{\rm true}$ through $X^I_t$ and $\theta^{-2}$.

1. Run the simulation for $\theta^I_{\rm true} \in \{0.8, 2.0, 3.5\}$, holding everything else at the baseline.

2. Plot the posterior variance on a log scale for each case.

3. Which value of $\theta^I_{\rm true}$ yields the fastest concentration, and does the result match the sensitivity formula above?
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

```{solution-end}
```

```{exercise}
:label: rle_ex3

*Effect of return noise*

Larger $\sigma^2$ widens the conditional density of $s_t$ given $r_t$, which one might guess slows learning.

But $\sigma^2$ also scales the price intercept in {eq}`eq:bk-price`, so price dispersion across candidate $\theta$ grows with $\sigma^2$.

1. Run the simulation with $\sigma^2 \in \{0.25, 1.0, 4.0\}$, keeping $\tau^2 = 1$ fixed.

2. Plot the posterior variance on a log scale for each $\sigma^2$.

3. Which effect dominates? Explain in terms of the signal-to-noise ratio for inferring $\theta^I$ from the price.
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
