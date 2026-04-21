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

(information_market_equilibrium)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Information and Market Equilibrium

```{contents} Contents
:depth: 2
```

## Overview

This lecture studies two questions about the **informational role of prices** posed and
answered by {cite:t}`kihlstrom_mirman1975`.

1. *When do prices transmit inside information?*   
   - An informed insider observes a private
   signal correlated with an unknown state of the world and adjusts demand accordingly.
   - Equilibrium prices shift. 
   - Under what conditions can an outside observer *infer* the
   insider's private signal from the equilibrium price?

2. *Do Bayesian price expectations converge?*  
   - In a stationary stochastic exchange
   economy, an uninformed observer uses the history of market prices and Bayes' Law to form
   expectations about the economy's structure.  
   - Do those expectations eventually
   agree with those of a fully informed observer?

Kihlstrom and Mirman's answers rely on two classical ideas from statistics:

- **Blackwell sufficiency**: a random variable $\tilde{y}$ is said to be *sufficient* for a random variable
  $\tilde{y}'$ with respect to an unknown state if knowing $\tilde{y}$ gives all the
  information about the state that $\tilde{y}'$ contains.
- **Bayesian consistency**: as the sample grows, a Bayesian statistician's posterior probability distribution concentrates on the true
  parameter value, even when the underlying economic structure is not globally identified from prices alone.

Important findings of {cite:t}`kihlstrom_mirman1975` are:

- Equilibrium prices transmit inside information *if and only if* the map from the
  insider's posterior distribution to the equilibrium price vector is invertible
  (one-to-one).
- For a two-state pure exchange economy with CES preferences, invertibility holds whenever the
  elasticity of substitution $\sigma \neq 1$.  
  - With Cobb-Douglas preferences ($\sigma = 1$)
  the equilibrium price is independent of the insider's posterior, so information is never
  transmitted.
- In the dynamic economy, as information accumulates, Bayesian price expectations converge to **rational expectations**, even when the deep structure of the economy is not identified.

```{note}
{cite:t}`kihlstrom_mirman1975` use the terms "reduced form" and "structural" models in a
way that careful econometricians do. 

Reduced-form and structural models come in pairs. 

To each structure or structural model
there is a reduced form, or collection of reduced forms, underlying different possible regressions.
```

The lecture is organized as follows.

1. Set up the static two-commodity model and define equilibrium.
2. State the price-revelation theorem (Theorem 1 of the paper) and the invertibility
   conditions (Theorem 2).
3. Illustrate invertibility and its failure with numerical examples using CES and
   Cobb-Douglas preferences.
4. Introduce the dynamic stochastic economy and derive the Bayesian convergence result.
5. Simulate Bayesian learning from price observations.

This lecture builds on ideas in {doc}`blackwell_kihlstrom` and {doc}`likelihood_bayes`.

## Setup

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm
```

## A two-commodity economy with an informed insider

### Preferences, endowments, and the unknown state

The economy has two goods. 

Good 2 is the numeraire (price normalized to 1); good 1 trades
at price $p > 0$.

An unknown parameter $\bar{a}$ affects the value of good 1. 

Agent $i$'s expected utility
from a bundle $(x_1^i, x_2^i)$ is

$$
U^i(x_1^i, x_2^i)
  = \sum_{s=1}^{S} u^i(a_s x_1^i,\, x_2^i)\, PR^i(\bar{a} = a_s),
$$

where $PR^i$ is agent $i$'s subjective probability distribution over the finite state space
$A = \{a_1, \ldots, a_S\}$.

Each agent starts with an endowment $w^i$ of good 2 and a share $\theta^i$ of the
representative firm.

The firm's profit $\pi$ is determined by profit maximization.

Agent
$i$'s budget constraint is

$$
p x_1^i + x_2^i = w^i + \theta^i \pi.
$$

Agents maximize expected utility subject to their budget constraints.

A **competitive
equilibrium** is a price $\hat{p}$ that clears both markets simultaneously.

### The informed agent's problem

Suppose **agent 1** (the insider) observes a private signal $\tilde{y}$ correlated with
$\bar{a}$ before trading.

Upon observing $\tilde{y} = y$, agent 1 updates their prior
$\mu = PR^1$ to a **posterior** $\mu_y = (\mu_{y1}, \ldots, \mu_{yS})$ via Bayes' rule:

$$
\mu_{ys} = PR(\bar{a} = a_s \mid \tilde{y} = y).
$$

Because agent 1's demand depends on $\mu_y$, the new equilibrium price satisfies

$$
\hat{p} = p(\mu_y).
$$

Outside observers who see $\hat{p}$ but not $\tilde{y}$ can try to *back out* the
insider's posterior from the price.

This is possible when the map $\mu \mapsto p(\mu)$
is **invertible** on the relevant domain.

(price_revelation_theorem)=
## Price revelation

### Blackwell sufficiency

The price variable $p(\mu_{\tilde{y}})$ *accurately transmits* the insider's private
information if observing the equilibrium price is just as informative about $\bar{a}$ as
observing the signal $\tilde{y}$ directly.

In Blackwell's language ({cite}`blackwell1951` and {cite}`blackwell1953`), this means
$p(\mu_{\tilde{y}})$ is **sufficient** for $\tilde{y}$.

```{prf:definition} Sufficiency
:label: ime_def_sufficiency

A random variable $\tilde{y}$ is *sufficient* for $\tilde{y}'$ (with
respect to $\bar{a}$) if there exists a conditional distribution $PR(y' \mid y)$,
**independent of** $\bar{a}$, such that

$$
\phi'_a(y') = \sum_{y \in Y} PR(y' \mid y)\, \phi_a(y)
\quad \text{for all } a \text{ and all } y',
$$

where $\phi_a(y) = PR(\tilde{y} = y \mid \bar{a} = a)$.

Thus, once $\tilde{y}$ is known, $\tilde{y}'$ provides no additional information
about $\bar{a}$.
```

```{prf:lemma} Posterior Sufficiency
:label: ime_lemma_posterior_sufficiency

({cite:t}`kihlstrom_mirman1975`) The posterior distribution $\mu_{\tilde{y}}$
is sufficient for $\tilde{y}$.
```

```{prf:proof} (Sketch)
The posterior $\mu_{\tilde{y}}$ satisfies

$$
PR(\bar{a} = a_s \mid \mu_{\tilde{y}} = \mu_y,\; \tilde{y} = y) = \mu_{ys}
  = PR(\bar{a} = a_s \mid \mu_{\tilde{y}} = \mu_y).
$$

Because the posterior itself *encodes* what $\tilde{y}$ says about $\bar{a}$, observing
$\tilde{y}$ directly would add no information.
```

```{prf:theorem} Price Revelation
:label: ime_theorem_price_revelation

In the economy described above, the price
random variable $p(\mu_{\tilde{y}})$ is sufficient for $\tilde{y}$ **if and only if** the
function $p(PR^1)$ is **invertible** on the set

$$
P \equiv \bigl\{\, p(\mu_y) : y \in Y,\;
  PR(\tilde{y} = y) = \sum_{a \in A} \phi_a(y)\,\mu(a) > 0 \bigr\}.
$$
```

The "only if" direction follows because if $p$ were not one-to-one, two different posteriors
would generate the same price; an observer could not distinguish them, so the price would
not transmit all information that resides in the signal.

### Two interpretations

#### Insider trading in a stock market

Good 1 is a risky asset with random return $\bar{a}$; good 2 is "money".

An insider's demand reveals private information about the return.

If the invertibility condition holds, outside observers can read the insider's signal from
the equilibrium stock price.

#### Price as a quality signal

Good 1 has uncertain quality $\bar{a}$.

Experienced consumers (who have sampled the good) observe a signal correlated with quality
and buy accordingly.

Uninformed consumers can infer quality from the market price, provided invertibility holds.

(invertibility_conditions)=
## Invertibility and the elasticity of substitution

When does $p(PR^1)$ fail to be invertible?

Theorem 2 of {cite:t}`kihlstrom_mirman1975`
shows that for a two-state economy ($S = 2$), the answer turns on the **elasticity of
substitution** $\sigma$ of agent 1's utility function.

### The two-state first-order condition

With $S = 2$ and $\mu = (q,\, 1-q)$, the first-order condition for agent 1's demand
(equation (12a) in the paper) reduces to

$$
p(q) = \frac{\alpha_1 q + \alpha_2 (1-q)}{\beta_1 q + \beta_2 (1-q)},
$$

where

$$
\alpha_s = a_s\, u^1_1(a_s x_1,\, x_2), \qquad
\beta_s  = u^1_2(a_s x_1,\, x_2), \qquad s = 1, 2.
$$

The equilibrium consumption $(x_1, x_2)$ itself depends on $p$, so this is an implicit
equation in $p$.

```{prf:theorem} Invertibility Conditions
:label: ime_theorem_invertibility_conditions

Assume $u^1$ is quasi-concave and
homothetic with continuous first partials. Assume agent 1 always consumes positive
quantities of both goods. For $S = 2$:

- If $\sigma < 1$ for all feasible allocations, $p(PR^1)$ is **invertible** on $P$.
- If $\sigma > 1$ for all feasible allocations, $p(PR^1)$ is **invertible** on $P$.
- If $u^1$ is **Cobb-Douglas** ($\sigma = 1$), $p(PR^1)$ is **constant** on $P$
  (no information is transmitted).
```

Thus, when $\sigma = 1$ the income and substitution effects exactly cancel,
making agent 1's demand for good 1 independent of information about $\bar{a}$.

So the market price cannot reveal that information.

### CES utility

For concreteness we work with the **constant-elasticity-of-substitution** (CES) utility
function

$$
u(c_1, c_2) = \bigl(c_1^{\rho} + c_2^{\rho}\bigr)^{1/\rho}, \qquad \rho \in (-\infty,0) \cup (0,1),
$$

whose elasticity of substitution is $\sigma = 1/(1-\rho)$.

- $\rho \to 0$: Cobb-Douglas ($\sigma = 1$).
- $\rho < 0$: $\sigma < 1$ (complements).
- $0 < \rho < 1$: $\sigma > 1$ (substitutes).

Pertinent partial derivatives are

$$
u_1(c_1,c_2) = \bigl(c_1^\rho + c_2^\rho\bigr)^{1/\rho - 1}\, c_1^{\rho-1}, \qquad
u_2(c_1,c_2) = \bigl(c_1^\rho + c_2^\rho\bigr)^{1/\rho - 1}\, c_2^{\rho-1}.
$$

### Equilibrium price as a function of the posterior

We focus on agent 1 as the *only* informed trader who absorbs one unit of good 1 at
equilibrium (i.e., $x_1 = 1$).

Agent 1's budget constraint then reduces to
$x_2 = W^1 - p$, and the equilibrium price is the unique $p \in (0, W^1)$ satisfying
the first-order condition

$$
p \bigl[q\, u_2(a_1,\, W^1-p) + (1-q)\, u_2(a_2,\, W^1-p)\bigr]
= q\, a_1\, u_1(a_1,\, W^1-p) + (1-q)\, a_2\, u_1(a_2,\, W^1-p).
$$

For Cobb-Douglas utility ($\sigma = 1$), the first-order condition becomes $p = W^1 - p$,
giving $p^* = W^1/2$ regardless of the posterior $q$—confirming that no information
is transmitted through the price in the Cobb-Douglas case.

We compute first-order conditions numerically below.

```{code-cell} ipython3
def ces_derivatives(c1, c2, rho):
    """
    Returns (u1, u2) for u(c1,c2) = (c1^rho + c2^rho)^(1/rho).
    Uses Cobb-Douglas limit for |rho| < 1e-4 to avoid numerical overflow.
    """
    if abs(rho) < 1e-4:
        # Cobb-Douglas limit  u = sqrt(c1*c2)
        u1 = 0.5 * np.sqrt(c2 / c1)
        u2 = 0.5 * np.sqrt(c1 / c2)
    else:
        common = (c1**rho + c2**rho)**(1/rho - 1)
        u1 = common * c1**(rho - 1)
        u2 = common * c2**(rho - 1)
    return u1, u2


def eq_price(q, a1, a2, W1, rho):
    """
    Solve for the equilibrium price when the informed agent absorbs one unit
    of good 1.  With x1 = 1 and budget constraint x2 = W1 - p, the FOC

        p [q u2(a1, x2) + (1-q) u2(a2, x2)] = q a1 u1(a1, x2) + (1-q) a2 u1(a2, x2)

    has a unique root p* in (0, W1).

    Parameters
    ----------
    q   : posterior probability on state 1 (high state)
    a1  : state-1 productivity value  (a1 > a2)
    a2  : state-2 productivity value
    W1  : informed agent's wealth
    rho : CES parameter  (rho=0 → Cobb-Douglas; analytical p* = W1/2)

    Returns
    -------
    p_star : equilibrium price, or nan if solver fails
    """
    def residual(p):
        x2 = W1 - p          # x1 = 1 absorbed at equilibrium
        u1_s1, u2_s1 = ces_derivatives(a1, x2, rho)
        u1_s2, u2_s2 = ces_derivatives(a2, x2, rho)
        lhs = p * (q * u2_s1 + (1 - q) * u2_s2)
        rhs = q * a1 * u1_s1 + (1 - q) * a2 * u1_s2
        return lhs - rhs

    try:
        return brentq(residual, 1e-6, W1 - 1e-6, xtol=1e-10)
    except ValueError:
        return np.nan
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: equilibrium price vs posterior
    name: fig-eq-price-posterior
---
# ── Economy parameters ──────────────────────────────────────────────────────
a1, a2 = 2.0, 0.5     # state values (a1 > a2)
W1     = 4.0           # informed agent's wealth; equilibrium x2 = W1 - p

# Posterior grid
q_grid = np.linspace(0.05, 0.95, 200)

# rho values to compare: complements (<0), Cobb-Douglas (=0), substitutes (>0)
rho_values = [-0.5, 0.0, 0.5]
rho_labels = [r"$\rho = -0.5$  ($\sigma = 0.67$, complements)",
              r"$\rho = 0$  ($\sigma = 1$, Cobb-Douglas)",
              r"$\rho = 0.5$  ($\sigma = 2$, substitutes)"]
colors     = ["steelblue", "crimson", "forestgreen"]

fig, ax = plt.subplots(figsize=(8, 5))

for rho, label, color in zip(rho_values, rho_labels, colors):
    prices = [eq_price(q, a1, a2, W1, rho) for q in q_grid]
    ax.plot(q_grid, prices, label=label, color=color, lw=2)

ax.set_xlabel(r"posterior probability $q = \Pr(\bar{a} = a_1)$", fontsize=12)
ax.set_ylabel("equilibrium price $p^*(q)$", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

The plot confirms {prf:ref}`ime_theorem_invertibility_conditions`.

- **CES with $\sigma \neq 1$**: the equilibrium price is **strictly monotone** in $q$.
  An outside observer who knows the equilibrium map $p^*(\cdot)$ can uniquely invert the
  price to recover $q$—inside information is fully transmitted.
- **Cobb-Douglas ($\sigma = 1$)**: the price is *flat* in $q$—information is never
  transmitted through the market.

```{code-cell} ipython3
# Verify that rho=0 (exact Cobb-Douglas) gives a flat line
p_cd = [eq_price(q, a1, a2, W1, rho=0.0) for q in q_grid]

print(f"Cobb-Douglas (rho=0): min p* = {min(p_cd):.6f}, "
      f"max p* = {max(p_cd):.6f}, "
      f"range = {max(p_cd)-min(p_cd):.2e}")
print(f"Analytical CD price  = W1/2 = {W1/2:.6f}")
```

Every entry equals $W^1/2 = 2.0$ exactly, confirming analytically that the Cobb-Douglas
equilibrium price is independent of $q$ and of the state values $a_1, a_2$.

(price_monotonicity)=
### Why monotonicity depends on $\sigma$

The derivative $\partial p / \partial q$ has the sign of $\alpha_1 \beta_2 - \alpha_2 \beta_1$
(from differentiating the FOC formula).

Using

$$
\frac{\alpha_s}{\beta_s}
  = \frac{a_s\, u_1(a_s x_1, x_2)}{u_2(a_s x_1, x_2)}
  = a_s^{(\sigma-1)/\sigma}\,\Bigl(\frac{x_2}{x_1}\Bigr)^{1/\sigma},
$$

one can show

$$
\frac{\partial}{\partial a}\,\frac{\alpha}{\beta}
  = \frac{(\sigma - 1)}{\sigma}\, a^{-1/\sigma}\,
    \Bigl(\frac{x_2}{x_1}\Bigr)^{1/\sigma}.
$$

This is positive when $\sigma > 1$, negative when $\sigma < 1$, and **zero when $\sigma = 1$**
(Cobb-Douglas).

The vanishing derivative means the marginal rate of substitution is
independent of $a_s$, so the informed agent's demand—and hence the equilibrium price—does
not respond to changes in beliefs.

Let us visualize the ratio $\alpha_s / \beta_s$ as a function of $a_s$ for different
values of $\sigma$:

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: marginal rate of substitution
    name: fig-mrs-alpha-beta
---
a_vals = np.linspace(0.3, 3.0, 300)
x1_fix, x2_fix = 1.0, 1.0   # fix consumption bundle for illustration

fig, ax = plt.subplots(figsize=(7, 4))
for rho, color in zip([-0.5, -1e-6, 0.5], ["steelblue", "crimson", "forestgreen"]):
    sigma = 1 / (1 - rho) if abs(rho) > 1e-8 else 1.0
    ratios = []
    for a in a_vals:
        u1, u2 = ces_derivatives(a * x1_fix, x2_fix, rho)
        ratios.append(a * u1 / u2)
    ax.plot(a_vals, ratios,
            label=rf"$\sigma = {sigma:.2f}$", color=color, lw=2)

ax.set_xlabel(r"state value $a_s$", fontsize=12)
ax.set_ylabel(r"$\alpha_s / \beta_s = a_s u_1 / u_2$", fontsize=12)
ax.axhline(y=1.0, color="black", lw=0.8, ls="--")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

When $\sigma = 1$ (red line) the ratio is constant across all $a_s$ values—information
about the state has no effect on the marginal rate of substitution.

For $\sigma < 1$ the
ratio is decreasing in $a_s$, and for $\sigma > 1$ it is increasing, making the
equilibrium price strictly monotone in the posterior $q$ in both cases.

(bayesian_price_expectations)=
## Bayesian price expectations in a dynamic economy

We now turn to a question addressed in Section 3 of {cite:t}`kihlstrom_mirman1975`.

### A stochastic exchange economy

Time is discrete: $t = 1, 2, \ldots$

In each period $t$:

1. Consumer $i$ receives a random endowment $\omega_i^t$.
2. Markets open; competitive prices $p^t = p(\omega^t)$ clear all markets.
3. Consumers trade and consume.

The endowment vectors $\{\tilde{\omega}^t\}$ are **i.i.d.** with density
$f(\omega^t \mid \lambda)$, where $\lambda = (\lambda_1, \ldots, \lambda_n)$ is a
**structural parameter vector** that is *fixed but unknown*.

The equilibrium price at time $t$ is a deterministic function of $\omega^t$, so
$\{p^t\}$ is also i.i.d. with density

$$
g(p^t \mid \lambda) = \int f(\omega^t \mid \lambda)\,
  \mathbf{1}\bigl[p(\omega^t) = p^t\bigr]\, d\omega^t.
$$

Following econometric convention, {cite:t}`kihlstrom_mirman1975` call $g(p \mid \lambda)$
the **reduced form** and $f(\omega \mid \lambda)$ the **structure**.

### The identification problem

Because the map $\omega \mapsto p(\omega)$ is many-to-one, observing prices loses
information relative to observing endowments.

In particular, it may be impossible to
recover $\lambda$ from $g(p \mid \lambda)$ even with infinite price data.

To handle this, partition $\Lambda$ into equivalence classes $\mu$ such that
$\lambda \in \mu$ and $\lambda' \in \mu$ whenever $g(p \mid \lambda) = g(p \mid \lambda')$
for all $p$.

The equivalence class $\mu$ containing the true $\lambda$ is the **reduced
form** (with respect to data on prices).

An observer who knows the infinite price history learns
$\mu$ but not necessarily $\lambda$.

### Bayesian updating

An uninformed observer begins with a prior $h(\lambda)$ over $\lambda \in \Lambda$.

After observing the price sequence $(p^1, \ldots, p^t)$, the observer's Bayesian
posterior is

$$
h(\lambda \mid p^1, \ldots, p^t)
  = \frac{h(\lambda)\, \prod_{\tau=1}^{t} g(p^\tau \mid \lambda)}
         {\displaystyle\sum_{\lambda' \in \Lambda}
           h(\lambda')\, \prod_{\tau=1}^{t} g(p^\tau \mid \lambda')}.
$$

At time $t$, the observer's price expectations for the next period are

$$
g(p^{t+1} \mid p^1, \ldots, p^t)
  = \sum_{\lambda \in \Lambda} g(p^{t+1} \mid \lambda)\,
    h(\lambda \mid p^1, \ldots, p^t).
$$

### The convergence theorem

```{prf:theorem} Bayesian Convergence
:label: ime_theorem_bayesian_convergence

Let $\bar\lambda$ be the true
structural parameter and $\bar\mu$ the reduced form that contains $\bar\lambda$.

Then

$$
\lim_{t \to \infty} h(\mu \mid p^1, \ldots, p^t)
  = \begin{cases} 1 & \text{if } \mu = \bar\mu, \\ 0 & \text{otherwise,} \end{cases}
$$

with probability one.

Consequently,

$$
\lim_{t \to \infty} g(p^{t+1} \mid p^1, \ldots, p^t) = g(p \mid \bar\mu),
$$

which equals the rational-expectations price distribution for a fully informed observer.
```

Establishing convergence relies on appealing to the **Bayesian consistency** result of {cite:t}`degroot1962`: as
long as $g(\cdot \mid \mu)$ and $g(\cdot \mid \mu')$ generate mutually singular measures
(which holds here generically), the posterior concentrates on the true reduced form.

Price observers converge to **rational expectations** even if they never identify the
underlying structure $\bar\lambda$.

The reduced form $g(p \mid \bar\mu)$ statistical model is used to form equilibrium price
expectations, and the Bayesian observer learns the reduced form from prices alone.

(bayesian_simulation)=
## Simulating Bayesian learning from prices

We illustrate the theorem with a two-state example.

Two possible reduced forms $\mu_1$ and $\mu_2$ generate prices
$p^t \sim N(\bar{p}_i, \sigma_p^2)$ for $i = 1, 2$ respectively.

The observer knows the two possible price distributions (the reduced forms) but not which
one governs the data.

This is a standard **Bayesian model selection** problem.

With a prior $h_0$ on $\mu_1$ and the observed price $p^t$, the posterior weight on $\mu_1$
after period $t$ is

$$
h_t = \frac{h_{t-1}\, g(p^t \mid \mu_1)}{h_{t-1}\, g(p^t \mid \mu_1)
      + (1-h_{t-1})\, g(p^t \mid \mu_2)}.
$$

```{code-cell} ipython3
def simulate_bayesian_learning(p_bar_true, p_bar_alt, sigma_p, T, h0, n_paths,
                                seed=42):
    """
    Simulate Bayesian learning about which price distribution is true.

    Parameters
    ----------
    p_bar_true : mean of the true reduced form
    p_bar_alt  : mean of the alternative reduced form
    sigma_p    : common standard deviation of price distributions
    T          : number of periods
    h0         : initial prior probability on the true model
    n_paths    : number of simulation paths
    seed       : random seed

    Returns
    -------
    h_paths : array of shape (n_paths, T+1) with posterior beliefs on true model
    """
    rng = np.random.default_rng(seed)
    h_paths = np.zeros((n_paths, T + 1))
    h_paths[:, 0] = h0

    for path in range(n_paths):
        h = h0
        prices = rng.normal(p_bar_true, sigma_p, size=T)
        for t, p in enumerate(prices):
            g_true  = norm.pdf(p, loc=p_bar_true, scale=sigma_p)
            g_alt   = norm.pdf(p, loc=p_bar_alt,  scale=sigma_p)
            denom   = h * g_true + (1 - h) * g_alt
            h       = h * g_true / denom
            h_paths[path, t + 1] = h

    return h_paths


def plot_bayesian_learning(h_paths, p_bar_true, p_bar_alt, ax):
    """Plot posterior beliefs over time."""
    T = h_paths.shape[1] - 1
    t_grid = np.arange(T + 1)

    for path in h_paths:
        ax.plot(t_grid, path, alpha=0.25, lw=0.8, color="steelblue")

    median_path = np.median(h_paths, axis=0)
    ax.plot(t_grid, median_path, color="navy", lw=2, label="median posterior")

    ax.axhline(y=1.0, color="black", ls="--", lw=1.2, label="true model weight = 1")
    ax.set_xlabel("period $t$", fontsize=12)
    ax.set_ylabel(r"$h_t$ = posterior weight on true model", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.08)
    ax.grid(alpha=0.3)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: bayesian learning across paths
    name: fig-bayesian-learning
---
T       = 300
h0      = 0.5     # diffuse prior
n_paths = 40
sigma_p = 0.4

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Case 1: distinct reduced forms (easy to learn)
p_bar_true, p_bar_alt = 2.0, 1.2
h_paths = simulate_bayesian_learning(p_bar_true, p_bar_alt, sigma_p, T, h0, n_paths)
plot_bayesian_learning(h_paths, p_bar_true, p_bar_alt, axes[0])

# Case 2: similar reduced forms (harder to learn)
p_bar_true, p_bar_alt = 2.0, 1.8
h_paths_hard = simulate_bayesian_learning(p_bar_true, p_bar_alt, sigma_p, T, h0, n_paths)
plot_bayesian_learning(h_paths_hard, p_bar_true, p_bar_alt, axes[1])

plt.tight_layout()
plt.show()
```

In both panels the posterior weight on the true model converges to 1 with probability one,
though convergence is slower when the two price distributions are similar (right panel).

### Price expectations vs. rational expectations

We now verify that the observer's price expectations converge to the rational-expectations
distribution $g(p \mid \bar\mu)$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: price distribution convergence
    name: fig-price-convergence
---
def price_expectation(h_t, p_bar_true, p_bar_alt, p_grid):
    """
    Compute the observer's predictive price density at posterior weight h_t.
    Mixture: h_t * N(p_bar_true, ...) + (1-h_t) * N(p_bar_alt, ...)
    """
    return (h_t * norm.pdf(p_grid, loc=p_bar_true, scale=sigma_p)
            + (1 - h_t) * norm.pdf(p_grid, loc=p_bar_alt, scale=sigma_p))


p_bar_true, p_bar_alt = 2.0, 1.2
sigma_p = 0.4
T_long  = 1000
n_paths = 1
h_paths_long = simulate_bayesian_learning(
    p_bar_true, p_bar_alt, sigma_p, T_long, h0=0.5, n_paths=n_paths, seed=7
)

p_grid = np.linspace(0.0, 3.5, 300)
re_density = norm.pdf(p_grid, loc=p_bar_true, scale=sigma_p)

fig, ax = plt.subplots(figsize=(8, 5))
snapshots = [0, 10, 50, 200, T_long]
palette   = plt.cm.Blues(np.linspace(0.3, 1.0, len(snapshots)))

for t_snap, col in zip(snapshots, palette):
    h_t = h_paths_long[0, t_snap]
    dens = price_expectation(h_t, p_bar_true, p_bar_alt, p_grid)
    ax.plot(p_grid, dens, color=col, lw=2,
            label=rf"$t = {t_snap}$, $h_t = {h_t:.3f}$")

ax.plot(p_grid, re_density, "k--", lw=2,
        label=r"rational expectations $g(p \mid \bar\mu)$")
ax.set_xlabel("price $p$", fontsize=12)
ax.set_ylabel("density", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

The sequence of predictive densities (shades of blue) converges to the rational-expectations
density (dashed black line) as experience accumulates.

This illustrates {prf:ref}`ime_theorem_bayesian_convergence`.

(km_extension_nonidentification)=
### Learning the reduced form without identifying the structure

The convergence result is particularly striking because the observer converges to
*rational expectations* even when the underlying **structure** $\lambda$ is
*not identified* by prices.

To illustrate this, consider a case with *three* possible structures
$\lambda^{(1)}, \lambda^{(2)}, \lambda^{(3)}$ but only *two* reduced forms
$\mu_1 = \{\lambda^{(1)}, \lambda^{(2)}\}$ and $\mu_2 = \{\lambda^{(3)}\}$
(because $\lambda^{(1)}$ and $\lambda^{(2)}$ generate the same price distribution).

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: learning with non-identification
    name: fig-nonidentification
---
def simulate_learning_3struct(T, h0_vec, p_bar_vec, sigma_p, true_idx, n_paths, seed=0):
    """
    Bayesian learning with 3 structures, 2 reduced forms.
    h0_vec  : length-3 array of initial prior weights on each structure
    p_bar_vec: length-3 array of price means for each structure
               (structures 0 and 1 share the same reduced form if p_bar_vec[0]==p_bar_vec[1])
    true_idx: index (0,1,2) of the true structure
    Returns  : array (n_paths, T+1, 3) posterior weights on each structure
    """
    rng = np.random.default_rng(seed)
    h_paths = np.zeros((n_paths, T + 1, 3))
    h_paths[:, 0, :] = h0_vec

    for path in range(n_paths):
        h = np.array(h0_vec, dtype=float)
        prices = rng.normal(p_bar_vec[true_idx], sigma_p, size=T)
        for t, p in enumerate(prices):
            likelihoods = norm.pdf(p, loc=p_bar_vec, scale=sigma_p)
            h = h * likelihoods
            h /= h.sum()
            h_paths[path, t + 1, :] = h

    return h_paths


# Structures 0 and 1 have the same reduced form (same price mean)
p_bar_vec = np.array([2.0, 2.0, 1.2])
h0_vec    = np.array([1/3, 1/3, 1/3])
sigma_p   = 0.4
T         = 400
true_idx  = 0             # True structure is 0 (indistinguishable from 1)

h_paths_3 = simulate_learning_3struct(T, h0_vec, p_bar_vec, sigma_p, true_idx, n_paths=30)
t_grid = np.arange(T + 1)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
struct_labels = [r"$\lambda^{(1)}$",
                 r"$\lambda^{(2)}$ (same reduced form as $\lambda^{(1)}$)",
                 r"$\lambda^{(3)}$"]

for k, (ax, label) in enumerate(zip(axes, struct_labels)):
    for path in h_paths_3:
        ax.plot(t_grid, path[:, k], alpha=0.25, lw=0.8, color="steelblue")
    ax.plot(t_grid, np.median(h_paths_3[:, :, k], axis=0),
            color="navy", lw=2, label=f"median weight on {label}")
    ax.set_xlabel("period $t$", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

axes[0].set_ylabel("posterior weight", fontsize=11)
plt.tight_layout()
plt.show()
```

The observer correctly rules out $\lambda^{(3)}$ (the wrong reduced form) with probability
one, but cannot distinguish $\lambda^{(1)}$ from $\lambda^{(2)}$ because they generate an
identical price distribution.

Nevertheless, the observer's **price expectations** converge
to rational expectations because both structures imply the same reduced form $\bar\mu$.

## Exercises

```{exercise}
:label: km_ex1

**Invertibility with CARA preferences.**  Consider a two-state economy ($a_1 = 2$,
$a_2 = 0.5$) where the informed agent has **CARA** (constant absolute risk aversion)
preferences over portfolio wealth:

$$
u(W) = -e^{-\gamma W}, \quad W = x_2 + \bar{a}\, x_1.
$$

The agent chooses $x_1$ to maximize

$$
q\,u(W_1) + (1-q)\,u(W_2), \quad W_s = w - p\,x_1 + a_s\,x_1,
$$

subject to the budget constraint $p\,x_1 + x_2 = w$.  Total supply of good 1 is $X_1 = 1$.

1. Derive the first-order condition for the informed agent's optimal $x_1$.

1. Use the market-clearing condition $x_1 = 1$ (the informed agent absorbs the entire
supply) to obtain an implicit equation for the equilibrium price $p^*(q)$.  Solve it
numerically for $q \in (0,1)$ and several values of $\gamma$.

1. Show numerically that $p^*(q)$ is monotone in $q$, so the invertibility condition
holds.  Explain intuitively why CARA preferences always lead to an invertible price map
(the elasticity of substitution of portfolio utility is $\sigma = \infty$).
```

```{solution-start} km_ex1
:class: dropdown
```

**1. First-order condition.**

Define $W_s = w + (a_s - p)\,x_1$ for $s=1,2$.  The FOC is

$$
q\,(a_1 - p)\,\gamma\, e^{-\gamma W_1}
= (1-q)\,(p - a_2)\,\gamma\, e^{-\gamma W_2},
$$

or equivalently (dividing by $\gamma$ and rearranging)

$$
q\,(a_1 - p)\, e^{-\gamma(a_1-p) x_1}
  = (1-q)\,(p - a_2)\, e^{\gamma(p-a_2) x_1}.
$$

**2. Market-clearing equilibrium price.**

Setting $x_1 = 1$ (all supply absorbed by informed agent), the equation becomes
a scalar root-finding problem in $p$:

$$
F(p;\,q,\gamma) \equiv
  q\,(a_1-p)\,e^{-\gamma(a_1-p)} - (1-q)\,(p-a_2)\,e^{\gamma(p-a_2)} = 0.
$$

```{code-cell} ipython3
from scipy.optimize import brentq

def F_cara(p, q, a1, a2, gamma, x1=1.0):
    """Residual of CARA market-clearing condition."""
    return (q * (a1-p) * np.exp(-gamma*(a1-p)*x1)
            - (1-q) * (p-a2) * np.exp(gamma*(p-a2)*x1))

a1, a2  = 2.0, 0.5
q_grid  = np.linspace(0.05, 0.95, 200)
gammas  = [0.5, 1.0, 2.0, 5.0]
colors_sol = plt.cm.plasma(np.linspace(0.15, 0.85, len(gammas)))

fig, ax = plt.subplots(figsize=(8, 5))
for gamma, color in zip(gammas, colors_sol):
    p_eq = [brentq(F_cara, a2, a1,
                   args=(q, a1, a2, gamma))
            for q in q_grid]
    ax.plot(q_grid, p_eq, lw=2, color=color,
            label=rf"$\gamma = {gamma}$")

ax.set_xlabel(r"posterior $q = \Pr(\bar a = a_1)$", fontsize=12)
ax.set_ylabel("equilibrium price $p^*(q)$", fontsize=12)
ax.set_title("CARA preferences: equilibrium prices", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**3. Invertibility for CARA.**

The price is strictly increasing in $q$ for every $\gamma > 0$.  Intuitively, portfolio
utility $u(x_2 + \bar{a}\,x_1)$ treats the two goods as **perfect substitutes** in
creating wealth, giving an elasticity of substitution $\sigma = \infty \neq 1$. By
{prf:ref}`ime_theorem_invertibility_conditions`, the price map is therefore always invertible.

```{solution-end}
```

```{exercise}
:label: km_ex2

In the Bayesian learning simulation, the speed of
convergence to rational expectations is determined by the **Kullback-Leibler divergence**
between the two reduced forms.

The KL divergence from $g(\cdot \mid \mu_2)$ to $g(\cdot \mid \mu_1)$, for two normal
distributions with means $\bar{p}_1$ and $\bar{p}_2$ and common variance $\sigma_p^2$, is

$$
D_{KL}(\mu_1 \| \mu_2) = \frac{(\bar{p}_1 - \bar{p}_2)^2}{2\sigma_p^2}.
$$

1. For the "easy" case ($\bar{p}_1 = 2.0$, $\bar{p}_2 = 1.2$) and the "hard" case
($\bar{p}_1 = 2.0$, $\bar{p}_2 = 1.8$), compute $D_{KL}$ for $\sigma_p = 0.4$.

1. Re-run the simulations from the lecture for both cases with $n=100$ paths.  For each
path compute the first period $T_{0.99}$ at which $h_t \geq 0.99$.  Plot histograms of
$T_{0.99}$ for both cases.

1. How does the median $T_{0.99}$ scale with $D_{KL}$?  Verify numerically that
roughly $T_{0.99} \approx C / D_{KL}$ for some constant $C$.
```

```{solution-start} km_ex2
:class: dropdown
```

```{code-cell} ipython3
sigma_p = 0.4

def kl_normal(p1, p2, sigma):
    """KL divergence between N(p1,sigma^2) and N(p2,sigma^2)."""
    return (p1 - p2)**2 / (2 * sigma**2)

cases = [("Easy",  2.0, 1.2), ("Hard", 2.0, 1.8)]
for name, p1, p2 in cases:
    kl = kl_normal(p1, p2, sigma_p)
    print(f"{name} case: D_KL = {kl:.4f}")

n_paths = 100

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, p1, p2) in zip(axes, cases):
    kl = kl_normal(p1, p2, sigma_p)
    paths = simulate_bayesian_learning(p1, p2, sigma_p, T=2000,
                                       h0=0.5, n_paths=n_paths, seed=42)
    # First period where posterior >= 0.99
    T99 = []
    for path in paths:
        idx = np.where(path >= 0.99)[0]
        T99.append(idx[0] if len(idx) > 0 else 2001)

    median_T = np.median(T99)
    ax.hist(T99, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(median_T, color="crimson", lw=2,
               label=fr"Median $T_{{0.99}} = {median_T:.0f}$")
    ax.set_title(
        f"{name}: $D_{{KL}} = {kl:.4f}$,  "
        fr"$C/D_{{KL}} \approx {median_T*kl:.1f}$",
        fontsize=11
    )
    ax.set_xlabel(r"$T_{0.99}$", fontsize=12)
    ax.set_ylabel("count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

The median $T_{0.99}$ scales as approximately $C/D_{KL}$, confirming that learning is
faster when the two reduced forms are more easily distinguished (large $D_{KL}$).

```{solution-end}
```

```{exercise}
:label: km_ex3

**Failure of invertibility—counterexample for $S > 2$.**  The paper constructs a
counterexample showing that for $S = 3$ states, even if the elasticity of substitution
of $u^1$ is everywhere greater than one, $p(PR^1)$ need **not** be invertible.

Consider the marginal rate of substitution for the portfolio utility
$u^1(a_s x_1 + x_2)$ (infinite elasticity of substitution) and three states
$a_1 > a_2 > a_3$.  The MRS is

$$
m(\mu)
= \frac{a_1\beta_1\mu(a_1) + a_2\beta_2\mu(a_2) + a_3\beta_3\mu(a_3)}
       {\beta_1\mu(a_1) + \beta_2\mu(a_2) + \beta_3\mu(a_3)},
$$

where $\beta_s = u^{1\prime}(a_s x_1 + x_2)$.

1. For the parameterization used by {cite:t}`kihlstrom_mirman1975`—let
$\mu(a_3) = q$, $\mu(a_2) = r$, $\mu(a_1) = 1-r-q$—write $m$ as a function of $(q, r)$.
Compute $\partial m / \partial r$ and show that its sign depends on
$\beta_1\beta_2(a_1-a_2)$ and $\beta_2\beta_3(a_2-a_3)$.

1. Choose $a_1 = 3$, $a_2 = 2$, $a_3 = 0.5$ and $u'(c) = c^{-\gamma}$ (CRRA with risk
aversion $\gamma$).  Fix $x_1 = 1$, $x_2 = 0.5$.  For $\gamma = 2$, verify numerically
that $\partial m/\partial r$ changes sign (i.e., $m$ is *not* globally monotone in $r$),
giving a counterexample to invertibility.

1. Explain why this non-monotonicity does *not* arise in the two-state case $S = 2$.
```

```{solution-start} km_ex3
:class: dropdown
```

**1.** Rewrite the MRS with $\mu_1 = 1-r-q$:

$$
m(q,r) = \frac{a_1\beta_1(1-r-q) + a_2\beta_2 r + a_3\beta_3 q}
               {\beta_1(1-r-q) + \beta_2 r + \beta_3 q}.
$$

Differentiating using the quotient rule (denominator $D$):

$$
\frac{\partial m}{\partial r}
= \frac{(a_2\beta_2 - a_1\beta_1)D - (a_1\beta_1(1-r-q)+a_2\beta_2 r+a_3\beta_3 q)(\beta_2-\beta_1)}{D^2}.
$$

After simplification this reduces to a signed combination of
$\beta_1\beta_2(a_1-a_2)({\cdot})$ and $\beta_2\beta_3(a_2-a_3)({\cdot})$ terms
whose sign is parameter-dependent.

**2. Numerical verification.**

```{code-cell} ipython3
def mrs_3state(q, r, a1, a2, a3, x1, x2, gamma):
    """MRS with mu(a3)=q, mu(a2)=r, mu(a1)=1-r-q, portfolio utility u'(c)=c^{-gamma}."""
    mu1, mu2, mu3 = 1 - r - q, r, q
    beta1 = (a1 * x1 + x2)**(-gamma)
    beta2 = (a2 * x1 + x2)**(-gamma)
    beta3 = (a3 * x1 + x2)**(-gamma)
    num = a1*beta1*mu1 + a2*beta2*mu2 + a3*beta3*mu3
    den = beta1*mu1 + beta2*mu2 + beta3*mu3
    return num / den

a1, a2, a3  = 3.0, 2.0, 0.5
x1, x2      = 1.0, 0.5
gamma       = 2.0
q_fix       = 0.1       # fix q, vary r
r_grid      = np.linspace(0.05, 0.80, 200)

# Filter valid (q+r <= 1)
r_valid = r_grid[r_grid + q_fix <= 0.95]
m_vals  = [mrs_3state(q_fix, r, a1, a2, a3, x1, x2, gamma) for r in r_valid]
dm_dr   = np.gradient(m_vals, r_valid)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(r_valid, m_vals, color="steelblue", lw=2)
axes[0].set_xlabel(r"$r = \mu(a_2)$", fontsize=12)
axes[0].set_ylabel(r"$m(q, r)$ — MRS", fontsize=12)
axes[0].set_title(fr"MRS is non-monotone in $r$ (CRRA $\gamma={gamma}$)", fontsize=12)
axes[0].grid(alpha=0.3)

axes[1].plot(r_valid, dm_dr, color="crimson", lw=2)
axes[1].axhline(0, color="black", lw=1, ls="--")
axes[1].set_xlabel(r"$r = \mu(a_2)$", fontsize=12)
axes[1].set_ylabel(r"$\partial m / \partial r$", fontsize=12)
axes[1].set_title("Derivative changes sign — non-invertibility for $S=3$", fontsize=12)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("Sign changes in dm/dr:",
      np.sum(np.diff(np.sign(dm_dr)) != 0))
```

The derivative $\partial m / \partial r$ changes sign, confirming that the MRS (and hence
the equilibrium price) is **not** monotone in $r$ for $S = 3$.

**3.** In the two-state case $S = 2$, the prior is parameterized by a single scalar $q$
and the MRS is a function of $q$ alone.  One can show directly that $\partial m / \partial q$
has a definite sign determined entirely by whether $a_1 > a_2$ and whether
$\sigma > 1$ or $\sigma < 1$ hold—there is no room for sign changes.  With three states,
the two-dimensional prior $(q, r)$ allows richer interactions between $\beta_s$ values that
can reverse the sign of the derivative.

```{solution-end}
```

```{exercise}
:label: km_ex4

{prf:ref}`ime_theorem_bayesian_convergence`
assumes the true
distribution $g(\cdot \mid \bar\lambda)$ is in the support of the prior (i.e.,
$h(\bar\lambda) > 0$).  Investigate what happens when the true model is **not** in the
prior support.

1. Simulate $T = 1,000$ periods of prices from $N(2.0, 0.4^2)$ but use a prior that
    places equal weight on two *wrong* models: $N(1.5, 0.4^2)$ and $N(2.5, 0.4^2)$.

    - Plot the posterior weight on each model over time.

2. Show that the **predictive** (mixture) price distribution converges to the *closest*
    model in KL divergence terms—which by symmetry is the equal mixture, with mean 2.0.

    - Verify this numerically by computing the predictive mean over time.

3. Relate this finding to the Bayesian consistency literature: when is the limit
    distribution a good approximation to the true distribution even under misspecification?
```

```{solution-start} km_ex4
:class: dropdown
```

```{code-cell} ipython3
def simulate_misspecified(T, p_bar_true, p_bar_wrong, sigma_p, h0, n_paths, seed=0):
    """
    Misspecified Bayesian learning: two wrong models with means p_bar_wrong[0,1].
    True model has mean p_bar_true (not in prior support).
    Returns (n_paths, T+1, 2) array of posterior weights.
    """
    rng = np.random.default_rng(seed)
    h_paths = np.zeros((n_paths, T + 1, 2))
    h_paths[:, 0, :] = h0

    for path in range(n_paths):
        h = np.array(h0, dtype=float)
        prices = rng.normal(p_bar_true, sigma_p, size=T)
        for t, price in enumerate(prices):
            likes = norm.pdf(price, loc=p_bar_wrong, scale=sigma_p)
            h = h * likes
            h /= h.sum()
            h_paths[path, t + 1, :] = h

    return h_paths


T        = 1000
p_true   = 2.0
p_wrong  = np.array([1.5, 2.5])
sigma_p  = 0.4
h0       = np.array([0.5, 0.5])
n_paths  = 30

h_misspec = simulate_misspecified(T, p_true, p_wrong, sigma_p, h0, n_paths)

t_grid = np.arange(T + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, k, label in zip(axes, [0, 1], [r"$N(1.5, \sigma^2)$", r"$N(2.5, \sigma^2)$"]):
    for path in h_misspec:
        ax.plot(t_grid, path[:, k], alpha=0.2, lw=0.8, color="steelblue")
    ax.plot(t_grid, np.median(h_misspec[:, :, k], axis=0),
            color="navy", lw=2, label="median")
    ax.axhline(0.5, color="crimson", lw=1.5, ls="--", label="0.5 (symmetric limit)")
    ax.set_title(f"Posterior weight on {label}", fontsize=11)
    ax.set_xlabel("period $t$", fontsize=11)
    ax.set_ylabel("posterior weight", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Predictive mean = h[:,0]*1.5 + h[:,1]*2.5
pred_mean = np.median(
    h_misspec[:, :, 0] * p_wrong[0] + h_misspec[:, :, 1] * p_wrong[1], axis=0
)
print(f"True mean: {p_true}")
print(f"Predictive mean at T={T}: {pred_mean[-1]:.4f}")
print("(Symmetry implies equal weight on 1.5 and 2.5 → predictive mean = 2.0)")
```

By symmetry, the two wrong models are equidistant from the true distribution in KL
divergence. 

The posterior therefore converges to the 50-50 mixture, and the predictive mean
converges to $0.5 \times 1.5 + 0.5 \times 2.5 = 2.0$—coinciding with the true mean
despite misspecification. 

This is an instance of the general result that under
misspecification, Bayesian posteriors converge to the distribution in the model class that
minimizes KL divergence from the model actually generating the data.

```{solution-end}
```
