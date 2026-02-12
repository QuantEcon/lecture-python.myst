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

(doubts_or_variability)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Doubts or Variability?

```{contents} Contents
:depth: 2
```

> *No one has found risk aversion parameters of 50 or 100 in the diversification of
> individual portfolios, in the level of insurance deductibles, in the wage premiums
> associated with occupations with high earnings risk, or in the revenues raised by
> state-operated lotteries. It
> would be good to have the equity premium resolved, but I think we need to look beyond high
> estimates of risk aversion to do it.* -- Robert Lucas Jr., January 10, 2003

## Overview

{cite:t}`Tall2000` showed that a recursive preference specification could match the equity premium and the risk-free rate puzzle simultaneously.

But matching required setting the risk-aversion coefficient $\gamma$ to around 50 for a random-walk consumption model and around 75 for a trend-stationary model --- exactly the range that provoked Lucas's skepticism.

{cite:t}`BHS_2009` ask whether those large $\gamma$ values really measure aversion to atemporal risk, or whether they instead measure the agent's doubts about the underlying probability model.

Their answer --- and the theme of this lecture --- is that much of what looks like "risk aversion" can be reinterpreted as **model uncertainty**.

The same recursion that defines Tallarini's risk-sensitive agent is observationally equivalent to a max--min recursion in which the agent fears that the probability model governing consumption growth may be wrong.

Under this reading, the parameter that looked like extreme risk aversion instead measures concern about **misspecification**.

They show that modest amounts of model uncertainty can substitute for large amounts of risk aversion
in terms of choices and effects on asset prices.

This reinterpretation changes the welfare question that asset prices answer: do large risk premia measure the benefits from reducing well-understood aggregate fluctuations, or the benefits from reducing doubts about the underlying model?

We start with the Hansen--Jagannathan bound, then specify the statistical environment, lay out four related preference specifications and their relationships, and finally revisit Tallarini's calibration using detection-error probabilities.

This lecture draws on ideas and techniques that appear in

- {ref}`Asset Pricing: Finite State Models <mass>` where we introduce stochastic discount factors.
- {ref}`Likelihood Ratio Processes <likelihood_ratio_process>` where we develop the likelihood-ratio machinery that reappears here as the worst-case distortion $\hat g$.


In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]
!pip install pandas-datareader
```

We use the following imports:

```{code-cell} ipython3
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from scipy.stats import norm
from scipy.optimize import brentq
```

We also set up calibration inputs and compute the covariance matrix of equity and risk-free returns from reported moments.

```{code-cell} ipython3
β = 0.995
T = 235  

# Table 2 parameters
rw = dict(μ=0.00495, σ_ε=0.0050)
ts = dict(μ=0.00418, σ_ε=0.0050, ρ=0.980, ζ=-4.48)

# Table 1 moments, converted from percent to decimals
r_e_mean, r_e_std = 0.0227, 0.0768
r_f_mean, r_f_std = 0.0032, 0.0061
r_excess_std = 0.0767

R_mean = np.array([1.0 + r_e_mean, 1.0 + r_f_mean])  
cov_erf = (r_e_std**2 + r_f_std**2 - r_excess_std**2) / 2.0
Σ_R = np.array(
    [
        [r_e_std**2, cov_erf],
        [cov_erf, r_f_std**2],
    ]
)
Σ_R_inv = np.linalg.inv(Σ_R)
```

## The equity premium and risk-free rate puzzles

### Pricing kernel and the risk-free rate

In this section, we review a few key concepts from {ref}`Asset Pricing: Finite State Models <mass>`.

A random variable $m_{t+1}$ is said to be a **stochastic discount factor** if it satisfies the following equation for the time-$t$ price $p_t$ of a one-period payoff $y_{t+1}$:

```{math}
:label: bhs_pricing_eq
p_t = E_t(m_{t+1}  y_{t+1}),
```

where $E_t$ denotes the mathematical expectation conditioned on date-$t$ information.

For time-separable CRRA preferences with discount factor $\beta$ and coefficient of relative risk aversion $\gamma$, the marginal rate of substitution gives

```{math}
:label: bhs_crra_sdf
m_{t+1} = \beta \left(\frac{C_{t+1}}{C_t}\right)^{-\gamma},
```

where $C_t$ is consumption at time $t$.

Setting $y_{t+1} = 1$ (a risk-free bond) in {eq}`bhs_pricing_eq` yields the reciprocal of the gross one-period risk-free rate:

```{math}
:label: bhs_riskfree
\frac{1}{R_t^f} = E_t[m_{t+1}] = E_t \left[\beta\left(\frac{C_{t+1}}{C_t}\right)^{-\gamma}\right].
```

### The Hansen--Jagannathan bound

Let $R_{t+1}^e$ denote the gross return on a risky asset (e.g., the market portfolio) and $R_{t+1}^f$ the gross return on a one-period risk-free bond.

The **excess return** is

$$
\xi_{t+1} = R_{t+1}^e - R_{t+1}^f.
$$

An excess return is the payoff on a zero-cost portfolio that is long one dollar of the risky asset and short one dollar of the risk-free bond.

Because the portfolio costs nothing to enter, its price is $p_t = 0$, so {eq}`bhs_pricing_eq` implies

$$
0 = E_t[m_{t+1} \xi_{t+1}].
$$

We can decompose the expectation of a product into a covariance plus a product of expectations:

$$
E_t[m_{t+1} \xi_{t+1}]
=
\operatorname{cov}_t(m_{t+1},\xi_{t+1}) + E_t[m_{t+1}] E_t[\xi_{t+1}],
$$

where $\operatorname{cov}_t$ denotes the conditional covariance and $\sigma_t$ will denote the conditional standard deviation.

Setting the left-hand side to zero and solving for the expected excess return gives

$$
E_t[\xi_{t+1}] = -\frac{\operatorname{cov}_t(m_{t+1}, \xi_{t+1})}{E_t[m_{t+1}]}.
$$

Taking absolute values and applying the **Cauchy--Schwarz inequality** $|\operatorname{cov}(X,Y)| \leq \sigma(X) \sigma(Y)$ yields

```{math}
:label: bhs_hj_bound
\frac{|E_t[\xi_{t+1}]|}{\sigma_t(\xi_{t+1})}
\leq
\frac{\sigma_t(m_{t+1})}{E_t[m_{t+1}]}.
```

The left-hand side of {eq}`bhs_hj_bound` is the **Sharpe ratio**: the expected excess return per unit of return volatility.

The right-hand side, $\sigma_t(m)/E_t(m)$, is the **market price of risk**: the maximum Sharpe ratio attainable in the market.

The bound says that the Sharpe ratio of any asset cannot exceed the market price of risk.

#### Unconditional version

The bound {eq}`bhs_hj_bound` is stated in conditional terms.

An unconditional counterpart considers a vector of $n$ gross returns $R_{t+1}$ (e.g., equity and risk-free) with unconditional mean $E(R)$ and covariance matrix $\Sigma_R$:

```{math}
:label: bhs_hj_unconditional
\sigma(m)
\geq
\sqrt{b^\top \Sigma_R^{-1} b},
\qquad
b = \mathbf{1} - E(m) E(R).
```

In {ref}`Exercise 1 <dov_ex1>`, we will revisit and verify this unconditional version of the HJ bound.

Below we implement a function that computes the right-hand side of {eq}`bhs_hj_unconditional` for any given value of $E(m)$.

```{code-cell} ipython3
def hj_std_bound(E_m):
    b = np.ones(2) - E_m * R_mean
    var_lb = b @ Σ_R_inv @ b
    return np.sqrt(np.maximum(var_lb, 0.0))
```


### The puzzle

To reconcile formula {eq}`bhs_crra_sdf` with measures of the market price of risk extracted from data on asset returns and prices (like those in Table 1 below) requires a value of $\gamma$ so high that it provokes skepticism --- this is the **equity premium puzzle**.

But the puzzle has a second dimension.

High values of $\gamma$ that deliver enough volatility $\sigma(m)$ also push the reciprocal of the risk-free rate $E(m)$ down, and therefore away from the Hansen--Jagannathan bounds.

This is the **risk-free rate puzzle** of {cite:t}`Weil_1989`.

{cite:t}`Tall2000` showed that recursive preferences with IES $= 1$ can clear the HJ bar while avoiding the risk-free rate puzzle.

The figure below reproduces Tallarini's key diagnostic.

We present this figure before developing the underlying theory because it motivates much of the subsequent analysis.

The closed-form expressions for the Epstein--Zin SDF moments used in the plot are derived in {ref}`Exercise 2 <dov_ex2>`.

The code below implements those expressions and the corresponding CRRA moments.

```{code-cell} ipython3
def moments_type1_rw(γ):
    μ, σ = rw["μ"], rw["σ_ε"]
    E_m = β * np.exp(-μ + 0.5 * σ**2 * (2.0 * γ - 1.0))
    var_log_m = (σ * γ) ** 2
    mpr = np.sqrt(np.exp(var_log_m) - 1.0)
    return E_m, mpr


def moments_type1_ts(γ):
    μ, σ, ρ = ts["μ"], ts["σ_ε"], ts["ρ"]
    mean_term = 1.0 - (2.0 * (1.0 - β) * (1.0 - γ)) / (1.0 - β * ρ) \
                + (1.0 - ρ) / (1.0 + ρ)
    E_m = β * np.exp(-μ + 0.5 * σ**2 * mean_term)
    var_term = (((1.0 - β) * (1.0 - γ)) / (1.0 - β * ρ) - 1.0) ** 2 \
                + (1.0 - ρ) / (1.0 + ρ)
    var_log_m = σ**2 * var_term
    mpr = np.sqrt(np.exp(var_log_m) - 1.0)
    return E_m, mpr


def moments_crra_rw(γ):
    μ, σ = rw["μ"], rw["σ_ε"]
    var_log_m = (γ * σ) ** 2
    mean_log_m = np.log(β) - γ * μ
    E_m = np.exp(mean_log_m + 0.5 * var_log_m)
    mpr = np.sqrt(np.exp(var_log_m) - 1.0)
    return E_m, mpr
```

For each value of $\gamma \in \{1, 5, 10, \ldots, 51\}$, we plot the implied $(E(m),\sigma(m))$ pair for three specifications: time-separable CRRA (crosses), Epstein--Zin preferences with random-walk consumption (circles), and Epstein--Zin preferences with trend-stationary consumption (pluses).


```{code-cell} ipython3
---
mystnb:
  figure:
    caption: SDF moments and Hansen-Jagannathan bound
    name: fig-bhs-1
---
γ_grid = np.arange(1, 55, 5)

Em_rw = np.array([moments_type1_rw(γ)[0] for γ in γ_grid])
σ_m_rw = np.array([moments_type1_rw(γ)[0] * moments_type1_rw(γ)[1] for γ in γ_grid])

Em_ts = np.array([moments_type1_ts(γ)[0] for γ in γ_grid])
σ_m_ts = np.array([moments_type1_ts(γ)[0] * moments_type1_ts(γ)[1] for γ in γ_grid])

Em_crra = np.array([moments_crra_rw(γ)[0] for γ in γ_grid])
σ_m_crra = np.array([moments_crra_rw(γ)[0] * moments_crra_rw(γ)[1] for γ in γ_grid])

Em_grid = np.linspace(0.8, 1.01, 1000)
HJ_std = np.array([hj_std_bound(x) for x in Em_grid])

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Em_grid, HJ_std, lw=2, color="black",
                            label="Hansen-Jagannathan bound")
ax.plot(Em_rw, σ_m_rw, "o", lw=2,
                            label="Epstein-Zin, random walk")
ax.plot(Em_ts, σ_m_ts, "+", lw=2,
                            label="Epstein-Zin, trend stationary")
ax.plot(Em_crra, σ_m_crra, "x", lw=2,
                            label="time-separable CRRA")

ax.set_xlabel(r"$E(m)$")
ax.set_ylabel(r"$\sigma(m)$")
ax.legend(frameon=False)
ax.set_xlim(0.8, 1.01)
ax.set_ylim(0.0, 0.42)

plt.tight_layout()
plt.show()
```

The crosses show that as $\gamma$ rises, $\sigma(m)/E(m)$ grows but $E(m)$ falls well below the range consistent with the observed risk-free rate.

This is the risk-free-rate puzzle of {cite:t}`Weil_1989`.

The circles and pluses show Tallarini's solution.

Recursive utility with IES $= 1$ pushes volatility upward while keeping $E(m)$ roughly constant near $1/(1+r^f)$.

For the random-walk model, the bound is reached around $\gamma = 50$; for the trend-stationary model, around $\gamma = 75$.

The quantitative achievement is significant, but Lucas's challenge remains: what microeconomic evidence supports $\gamma = 50$?

{cite:t}`BHS_2009` argue that the large $\gamma$ values are not really about risk aversion, but instead reflect the agent's doubts about the underlying probability model.

## The choice setting

To develop this reinterpretation, we first need to formalize the setting we are working in.

### Shocks and consumption plans

We formulate the analysis in terms of a general class of consumption plans.

Let $x_t$ be an $n \times 1$ state vector and $\varepsilon_{t+1}$ an $m \times 1$ shock.

A consumption plan belongs to the set $\mathcal{C}(A, B, H; x_0)$ if it admits the recursive representation

```{math}
:label: bhs_state_space
x_{t+1} = A x_t + B \varepsilon_{t+1},
\qquad
c_t = H x_t,
```

where the eigenvalues of $A$ are bounded in modulus by $1/\sqrt{\beta}$.

The time-$t$ consumption can therefore be written as

```{math}
c_t = H \left(B\varepsilon_t + AB\varepsilon_{t-1} + \cdots + A^{t-1}B\varepsilon_1\right) + HA^t x_0.
```

The equivalence theorems and Bellman equations below hold for arbitrary plans in $\mathcal{C}(A,B,H;x_0)$.

The random-walk and trend-stationary models below are two special cases we focus on.

### Consumption dynamics

Let $c_t = \log C_t$ be log consumption.

The *geometric-random-walk* specification is

```{math}
c_{t+1} = c_t + \mu + \sigma_\varepsilon \varepsilon_{t+1}, \qquad \varepsilon_{t+1} \sim \mathcal{N}(0, 1).
```

Iterating forward yields

```{math}
c_t = c_0 + t\mu + \sigma_\varepsilon(\varepsilon_t + \varepsilon_{t-1} + \cdots + \varepsilon_1),
\qquad
t \geq 1.
```

The *geometric-trend-stationary* specification can be written as a deterministic trend plus a stationary AR(1) component:

```{math}
c_t = \zeta + \mu t + z_t,
\qquad
z_{t+1} = \rho z_t + \sigma_\varepsilon \varepsilon_{t+1},
\qquad
\varepsilon_{t+1} \sim \mathcal{N}(0, 1).
```

With $z_0 = c_0 - \zeta$, this implies the representation

```{math}
c_t
=
\rho^t c_0 + \mu t + (1-\rho^t)\zeta
+
\sigma_\varepsilon(\varepsilon_t + \rho \varepsilon_{t-1} + \cdots + \rho^{t-1}\varepsilon_1),
\qquad
t \geq 1.
```

Equivalently, defining the detrended series $\tilde c_t := c_t - \mu t$,

```{math}
\tilde c_{t+1} - \zeta = \rho(\tilde c_t - \zeta) + \sigma_\varepsilon \varepsilon_{t+1}.
```

The estimated parameters are $(\mu, \sigma_\varepsilon)$ for the random walk and $(\mu, \sigma_\varepsilon, \rho, \zeta)$ for the trend-stationary case.

Below we record these parameters and moments in the paper's tables for later reference.

```{code-cell} ipython3
print("Table 2 parameters")
print(f"random walk: μ={rw['μ']:.5f}, σ_ε={rw['σ_ε']:.5f}")
print(
    f"trend stationary: μ={ts['μ']:.5f}, σ_ε={ts['σ_ε']:.5f}, "
    f"ρ={ts['ρ']:.3f}, ζ={ts['ζ']:.2f}"
)
print()
print("Table 1 moments")
print(f"E[r_e]={r_e_mean:.4f}, std[r_e]={r_e_std:.4f}")
print(f"E[r_f]={r_f_mean:.4f}, std[r_f]={r_f_std:.4f}")
print(f"std[r_e-r_f]={r_excess_std:.4f}")
```

(pref_equiv)=
## Preferences, distortions, and detection


### Overview of agents I, II, III, and IV

We compare four preference specifications over consumption plans $C^\infty \in \mathcal{C}$.

*Type I agent (Kreps--Porteus--Epstein--Zin--Tallarini)* with
- a discount factor $\beta \in (0,1)$;
- an intertemporal elasticity of substitution fixed at $1$;
- a risk-aversion parameter $\gamma \geq 1$; and
- an approximating conditional density $\pi(\cdot)$ for shocks and its implied joint distribution $\Pi_\infty(\cdot \mid x_0)$.

*Type II agent (multiplier preferences)* with
- $\beta \in (0,1)$;
- IES $=1$;
- unit risk aversion;
- an approximating model $\Pi_\infty(\cdot \mid x_0)$; and
- a penalty parameter $\theta > 0$ that discourages probability distortions using relative entropy.

*Type III agent (constraint preferences)* with
- $\beta \in (0,1)$;
- IES $=1$;
- unit risk aversion;
- an approximating model $\Pi_\infty(\cdot \mid x_0)$; and
- a bound $\eta$ on discounted relative entropy.

*Type IV agent (pessimistic ex post Bayesian)* with
- $\beta \in (0,1)$;
- IES $=1$;
- unit risk aversion; and
- a single pessimistic joint distribution $\hat\Pi_\infty(\cdot \mid x_0, \theta)$ induced by the type II worst-case distortion.


We will introduce two sets of equivalence results.

Types I and II are observationally equivalent in the strong sense that they have identical preferences over $\mathcal{C}$.

Types III and IV are observationally equivalent in a weaker but still useful sense: for the particular endowment process taken as given, they deliver the same worst-case pricing implications as a type II agent.

We now formalize each of the four agent types and develop the equivalence results that connect them.

For each of the four types, we will derive a Bellman equation that characterizes the agent's value function and stochastic discount factor.

The stochastic discount factor for all four types takes the form

$$
m_{t+1} = \beta \frac{\partial U_{t+1}/\partial c_{t+1}}{\partial U_t/\partial c_t} \hat g_{t+1},
$$

where $\hat g_{t+1}$ is a likelihood-ratio distortion that we will define in each case.


Along the way, we introduce the likelihood-ratio distortion that appears in the stochastic discount factor and develop the detection-error probability that serves as our new calibration device.

### Type I: Kreps--Porteus--Epstein--Zin--Tallarini preferences

The general Epstein--Zin--Weil specification aggregates current consumption and a certainty equivalent of future utility using a CES function:

```{math}
:label: bhs_ez_general
V_t = \left[(1-\beta) C_t^{\rho} + \beta \mathcal{R}_t(V_{t+1})^{\rho}\right]^{1/\rho},
\qquad
\rho := 1 - \frac{1}{\psi},
```

where $\psi > 0$ is the intertemporal elasticity of substitution and the certainty equivalent uses the risk-aversion parameter $\gamma \geq 1$:

```{math}
:label: bhs_certainty_equiv
\mathcal{R}_t(V_{t+1})
=
\left(E_t\left[V_{t+1}^{1-\gamma}\right]\right)^{\frac{1}{1-\gamma}}.
```

```{note}
For readers interested in a general class of aggregators and certainty equivalents, see Section
7.3 of {cite:t}`Sargent_Stachurski_2025`.
```

Let $\psi = 1$, so $\rho \to 0$.

In this limit the CES aggregator reduces to

$$
V_t = C_t^{1-\beta} \cdot \mathcal{R}_t(V_{t+1})^{\beta}.
$$

Taking logs and expanding the certainty equivalent {eq}`bhs_certainty_equiv` gives the *type I recursion*:

```{math}
:label: bhs_type1_recursion
\log V_t
=
(1-\beta)c_t
+
\frac{\beta}{1-\gamma}
\log E_t\left[(V_{t+1})^{1-\gamma}\right].
```

A key intermediate step is to define the transformed continuation value

```{math}
:label: bhs_Ut_def
U_t \equiv \frac{\log V_t}{1-\beta}
```

and the robustness parameter

```{math}
:label: bhs_theta_def
\theta = \frac{-1}{(1-\beta)(1-\gamma)}.
```

Substituting into {eq}`bhs_type1_recursion` yields the **risk-sensitive recursion** ({ref}`Exercise 3 <dov_ex3>` asks you to verify this step)

```{math}
:label: bhs_risk_sensitive
U_t = c_t - \beta\theta \log E_t\left[\exp\left(\frac{-U_{t+1}}{\theta}\right)\right].
```

When $\gamma = 1$ (equivalently $\theta = +\infty$), the $\log E \exp$ term reduces to $E_t U_{t+1}$ and the recursion becomes standard discounted expected log utility: $U_t = c_t + \beta E_t U_{t+1}$.

For consumption plans in $\mathcal{C}(A, B, H; x_0)$, the recursion {eq}`bhs_risk_sensitive` implies the Bellman equation

```{math}
:label: bhs_bellman_type1
U(x) = c - \beta\theta \log \int \exp\left[\frac{-U(Ax + B\varepsilon)}{\theta}\right] \pi(\varepsilon)d\varepsilon.
```

#### Deriving the stochastic discount factor

The stochastic discount factor is the intertemporal marginal rate of substitution: the ratio of marginal utilities of the consumption good at dates $t+1$ and $t$.

Since $c_t$ enters {eq}`bhs_risk_sensitive` linearly, $\partial U_t / \partial c_t = 1$.

Converting from log consumption to the consumption good gives $\partial U_t / \partial C_t = 1/C_t$.

A perturbation to $c_{t+1}$ in a particular state affects $U_t$ through the $\log E_t \exp$ term.

Differentiating {eq}`bhs_risk_sensitive`:

$$
\frac{\partial U_t}{\partial c_{t+1}}
=
-\beta\theta
\frac{\exp(-U_{t+1}/\theta)  (-1/\theta)}{E_t[\exp(-U_{t+1}/\theta)]}
\underbrace{\frac{\partial U_{t+1}}{\partial c_{t+1}}}_{=1}
=
\beta \frac{\exp(-U_{t+1}/\theta)}{E_t[\exp(-U_{t+1}/\theta)]}.
$$

Converting to consumption levels gives
$\partial U_t / \partial C_{t+1} = \beta \frac{\exp(-U_{t+1}/\theta)}{E_t[\exp(-U_{t+1}/\theta)]} \frac{1}{C_{t+1}}$.

The ratio of these marginal utilities gives the SDF:

```{math}
:label: bhs_sdf_Ut
m_{t+1}
=
\frac{\partial U_t / \partial C_{t+1}}{\partial U_t / \partial C_t}
=
\beta \frac{C_t}{C_{t+1}}
\frac{\exp(-U_{t+1}/\theta)}{E_t[\exp(-U_{t+1}/\theta)]}.
```


The second factor is the likelihood-ratio distortion $\hat g_{t+1}$: an exponential tilt that overweights states where the continuation value $U_{t+1}$ is low.


### Type II: multiplier preferences

We now turn to the type II (multiplier) agent.

Before writing down the preferences, we introduce the machinery of martingale likelihood ratios used to formalize model distortions.

The tools in this section build on {ref}`Likelihood Ratio Processes <likelihood_ratio_process>`, which develops properties of likelihood ratios in detail, and {ref}`Divergence Measures <divergence_measures>`, which covers relative entropy.


#### Martingale likelihood ratios

Consider a nonnegative martingale $G_t$ with $E(G_t \mid x_0) = 1$.

Its one-step increments

```{math}
g_{t+1} = \frac{G_{t+1}}{G_t},
\qquad
E_t[g_{t+1}] = 1,
\quad
g_{t+1} \geq 0,
\qquad
G_0 = 1,
```

define distorted conditional expectations: $\tilde E_t[b_{t+1}] = E_t[g_{t+1}b_{t+1}]$.

The conditional relative entropy of the distortion is $E_t[g_{t+1}\log g_{t+1}]$, and the discounted entropy over the entire path is $\beta E\bigl[\sum_{t=0}^{\infty} \beta^t G_tE_t(g_{t+1}\log g_{t+1})\big|x_0\bigr]$.


A type II agent's *multiplier* preference ordering over consumption plans $C^\infty \in \mathcal{C}(A,B,H;x_0)$ is defined by

```{math}
:label: bhs_type2_objective
\min_{\{g_{t+1}\}}
\sum_{t=0}^{\infty} E\left\{\beta^t G_t
\left[c_t + \beta\theta E_t\left(g_{t+1}\log g_{t+1}\right)\right]
\Big| x_0\right\},
```

where $G_{t+1} = g_{t+1}G_t$, $E_t[g_{t+1}] = 1$, $g_{t+1} \geq 0$, and $G_0 = 1$.

The parameter $\theta > 0$ penalizes the relative entropy of probability distortions.

The value function satisfies the Bellman equation

```{math}
:label: bhs_bellman_type2
W(x)
=
c + \min_{g(\varepsilon) \geq 0}
\beta \int \bigl[g(\varepsilon) W(Ax + B\varepsilon)
+ \theta g(\varepsilon)\log g(\varepsilon)\bigr] \pi(\varepsilon) d\varepsilon
```

subject to $\int g(\varepsilon) \pi(\varepsilon) d\varepsilon = 1$.

Inside the integral, $g(\varepsilon) W(Ax + B\varepsilon)$ is the continuation value under the distorted model $g\pi$, while $\theta g(\varepsilon)\log g(\varepsilon)$ is the entropy penalty that makes large departures from the approximating model $\pi$ costly.

The minimizer is ({ref}`Exercise 4 <dov_ex4>` derives this and verifies the equivalence $W \equiv U$)

```{math}
:label: bhs_ghat
\hat g_{t+1}
=
\frac{\exp \bigl(-W(Ax_t + B\varepsilon_{t+1})/\theta\bigr)}{E_t \left[\exp \bigl(-W(Ax_t + B\varepsilon_{t+1})/\theta\bigr)\right]}.
```

The fact that $g(\varepsilon)$ multiplies both the continuation value $W$ and the entropy penalty is the key structural feature that makes $\hat g$ a likelihood ratio.


Substituting {eq}`bhs_ghat` back into {eq}`bhs_bellman_type2` gives

$$W(x) = c - \beta\theta \log \int \exp \left[\frac{-W(Ax + B\varepsilon)}{\theta}\right]\pi(\varepsilon) d\varepsilon,$$

which is identical to {eq}`bhs_bellman_type1`.

Therefore $W(x) \equiv U(x)$, establishing that *types I and II are observationally equivalent* over elements of $\mathcal{C}(A,B,H;x_0)$.

The mapping between parameters is

```{math}
\theta = \left[(1-\beta)(\gamma - 1)\right]^{-1}.
```

```{code-cell} ipython3
def θ_from_γ(γ, β=β):
    if γ <= 1:
        return np.inf
    return 1.0 / ((1.0 - β) * (γ - 1.0))


def γ_from_θ(θ, β=β):
    if np.isinf(θ):
        return 1.0
    return 1.0 + 1.0 / ((1.0 - β) * θ)
```

### Type III: constraint preferences

Type III (constraint) preferences replace the entropy penalty with a hard bound.

The agent minimizes expected discounted log consumption under the worst-case model, subject to a cap $\eta$ on discounted relative entropy:

```{math}
J(x_0)
=
\min_{\{g_{t+1}\}}
\sum_{t=0}^{\infty} E \left[\beta^t G_t c_t \Big|  x_0\right]
```

subject to $G_{t+1} = g_{t+1}G_t$, $E_t[g_{t+1}] = 1$, $g_{t+1} \geq 0$, $G_0 = 1$, and

```{math}
\beta E \left[\sum_{t=0}^{\infty} \beta^t G_t E_t\left(g_{t+1}\log g_{t+1}\right)\Big|x_0\right] \leq \eta.
```

The Lagrangian for the type III problem is

$$
\mathcal{L}
=
\sum_{t=0}^{\infty} E\left[\beta^t G_t c_t \Big| x_0\right]
+
\theta \left[
\beta E \left(\sum_{t=0}^{\infty} \beta^t G_t E_t(g_{t+1}\log g_{t+1})\Big| x_0 \right) - \eta
\right],
$$

where $\theta \ge 0$ is the multiplier on the entropy constraint.

Collecting terms inside the expectation gives

$$
\mathcal{L}
=
\sum_{t=0}^{\infty} E \left \{\beta^t G_t
\left[c_t + \beta \theta E_t(g_{t+1}\log g_{t+1})\right]
\Big| x_0\right\} - \theta\eta,
$$

which, apart from the constant $-\theta\eta$, has the same structure as the type II objective {eq}`bhs_type2_objective`.

The first-order condition for $g_{t+1}$ is therefore identical, and the optimal distortion is the same $\hat g_{t+1}$ as in {eq}`bhs_ghat` for the $\theta$ that makes the entropy constraint bind.

The SDF is again $m_{t+1} = \beta(C_t/C_{t+1})\hat g_{t+1}$.

For the particular $A, B, H$ and $\theta$ used to derive the worst-case joint distribution $\hat\Pi_\infty$, the shadow prices of uncertain claims for a type III agent match those of a type II agent.

### Type IV: ex post Bayesian

Type IV is an ordinary expected-utility agent with log preferences evaluated under a single pessimistic probability model $\hat\Pi_\infty$:

```{math}
\hat E_0 \sum_{t=0}^{\infty} \beta^t c_t.
```

$\hat E_0$ denotes expectation under the pessimistic model $\hat\Pi_\infty$.

The joint distribution $\hat\Pi_\infty(\cdot \mid x_0, \theta)$ is the one associated with the type II agent's worst-case distortion.

Under $\hat\Pi_\infty$ the agent has log utility, so the Euler equation for any gross return $R_{t+1}$ is

$$
1 = \hat E_t \left[\beta \frac{C_t}{C_{t+1}} R_{t+1}\right].
$$

To express this in terms of the approximating model $\Pi_\infty$, apply a change of measure using the one-step likelihood ratio $\hat g_{t+1} = d\hat\Pi / d\Pi$:

$$
1 = E_t\left[\hat g_{t+1} \cdot \beta \frac{C_t}{C_{t+1}} R_{t+1}\right]
= E_t\left[m_{t+1} R_{t+1}\right],
$$

so the effective SDF under the approximating model is $m_{t+1} = \beta(C_t/C_{t+1})\hat g_{t+1}$.

For the particular $A, B, H$ and $\theta$ used to construct $\hat\Pi_\infty$, the type IV value function equals $J(x)$ from type III.

### Stochastic discount factor

As we have shown for each of the four agent types, the stochastic discount factor can be written compactly as

```{math}
:label: bhs_sdf
m_{t+1}
=
\beta \frac{C_t}{C_{t+1}} \hat g_{t+1}.
```

The distortion $\hat g_{t+1}$ is a likelihood ratio between the approximating and worst-case one-step models.

With log utility, $C_t/C_{t+1} = \exp(-(c_{t+1}-c_t))$ is the usual intertemporal marginal rate of substitution.

Robustness multiplies that term by $\hat g_{t+1}$, so uncertainty aversion enters pricing only through the distortion.

For constraint preferences, the worst-case distortion is the same as for multiplier preferences with the $\theta$ that makes the entropy constraint bind.

For the ex post Bayesian, the distortion is a change of measure from the approximating model to the pessimistic model.

### Value function decomposition

Substituting the minimizing $\hat g$ back into the Bellman equation {eq}`bhs_bellman_type2` yields a revealing decomposition of the type II value function:

```{math}
:label: bhs_W_decomp_bellman
W(x) = c + \beta \int \bigl[\hat g(\varepsilon) W(Ax + B\varepsilon) + \theta \hat g(\varepsilon)\log \hat g(\varepsilon)\bigr] \pi(\varepsilon)d\varepsilon.
```

Define two components:

```{math}
:label: bhs_J_recursion
J(x) = c + \beta \int \hat g(\varepsilon) J(Ax + B\varepsilon) \pi(\varepsilon)d\varepsilon,
```

```{math}
:label: bhs_N_recursion
N(x) = \beta \int \hat g(\varepsilon)\bigl[\log \hat g(\varepsilon) + N(Ax + B\varepsilon)\bigr] \pi(\varepsilon)d\varepsilon.
```

Then $W(x) = J(x) + \theta N(x)$.

Here $J(x_t) = \hat E_t \sum_{j=0}^{\infty} \beta^j c_{t+j}$ is expected discounted log consumption under the *worst-case* model.

$J$ is the value function for both the type III and the type IV agent: the type III agent maximizes expected utility subject to an entropy constraint, and once the worst-case model is determined, the resulting value is expected discounted consumption under that model; the type IV agent uses the same worst-case model as a fixed belief, so evaluates the same expectation.

The term $N(x)$ is discounted continuation entropy: it measures the total information cost of the probability distortion from date $t$ onward.

This decomposition will be important for the welfare calculations in {ref}`the welfare section <welfare_experiments>` below, where it explains why type III uncertainty compensation is twice that of type II.

### Gaussian mean-shift distortions

The preceding results hold for general distortions $\hat g$.
We now specialize to the Gaussian case that underlies our two consumption models.

Under both models, the shock is $\varepsilon_{t+1} \sim \mathcal{N}(0,1)$.

As we verify in the next subsection, the value function $W$ is linear in the state, so the exponent in the worst-case distortion {eq}`bhs_ghat` is linear in $\varepsilon_{t+1}$.

Exponentially tilting a Gaussian by a linear function produces another Gaussian with the same variance but a shifted mean.

The worst-case model therefore keeps the variance at one but shifts the mean of $\varepsilon_{t+1}$ to some $w < 0$.

The resulting likelihood ratio is ({ref}`Exercise 5 <dov_ex5>` verifies its properties)

```{math}
:label: bhs_ghat_gaussian
\hat g_{t+1}
=
\exp\left(w \varepsilon_{t+1} - \frac{1}{2}w^2\right),
\qquad
E_t[\hat g_{t+1}] = 1.
```

Hence $\log \hat g_{t+1}$ is normal with mean $-w^2/2$ and variance $w^2$, and

```{math}
\operatorname{std}(\hat g_{t+1}) = \sqrt{e^{w^2}-1}.
```

The mean shift $w$ is determined by how strongly each shock $\varepsilon_{t+1}$ affects continuation value.
From {eq}`bhs_ghat`, the worst-case distortion puts $\hat g \propto \exp(-W(x_{t+1})/\theta)$.
If $W(x_{t+1})$ loads on $\varepsilon_{t+1}$ with coefficient $\lambda$, then the Gaussian mean shift is $w = -\lambda/\theta$.

By guessing linear value functions and matching coefficients in the Bellman equation ({ref}`Exercise 11 <dov_ex11>` works out both cases), we obtain the worst-case mean shifts

```{math}
:label: bhs_w_formulas
w_{rw}(\theta) = -\frac{\sigma_\varepsilon}{(1-\beta)\theta},
\qquad
w_{ts}(\theta) = -\frac{\sigma_\varepsilon}{(1-\rho\beta)\theta}.
```

The denominator $(1-\beta)$ in the random-walk case is replaced by $(1-\beta\rho)$ in the trend-stationary case: because the AR(1) component is persistent, each shock has a larger effect on continuation utility.

```{code-cell} ipython3
def w_from_θ(θ, model):
    if np.isinf(θ):
        return 0.0
    if model == "rw":
        return -rw["σ_ε"] / ((1.0 - β) * θ)
    if model == "ts":
        return -ts["σ_ε"] / ((1.0 - β * ts["ρ"]) * θ)
    raise ValueError("model must be 'rw' or 'ts'")
```

### Discounted entropy

When the approximating and worst-case conditional densities are $\mathcal{N}(0,1)$ and $\mathcal{N}(w,1)$, conditional relative entropy is

```{math}
:label: bhs_conditional_entropy
E_t[\hat g_{t+1}\log \hat g_{t+1}] = \frac{1}{2}w(\theta)^2.
```

Because the distortion is i.i.d., the discounted entropy recursion {eq}`bhs_N_recursion` reduces to $N = \beta(\frac{1}{2}w^2 + N)$, giving discounted entropy

```{math}
:label: bhs_eta_formula
\eta = \frac{\beta}{2(1-\beta)} w(\theta)^2.
```

```{code-cell} ipython3
def η_from_θ(θ, model):
    w = w_from_θ(θ, model)
    return β * w**2 / (2.0 * (1.0 - β))
```

This formula provides a mapping between $\theta$ and $\eta$ that aligns multiplier and constraint preferences along an exogenous endowment process.

In the {ref}`detection-error section <detection_error_section>` below, we show that it is more natural to hold $\eta$ (or equivalently the detection-error probability $p$) fixed rather than $\theta$ when comparing across consumption models.

### Value functions for random-walk consumption

We now solve the recursions {eq}`bhs_W_decomp_bellman`, {eq}`bhs_J_recursion`, and {eq}`bhs_N_recursion` in closed form for the random-walk model, where $W$ is the type II (multiplier) value function, $J$ is the type III/IV value function, and $N$ is discounted continuation entropy.

Substituting $w_{rw}(\theta) = -\sigma_\varepsilon / [(1-\beta)\theta]$ from {eq}`bhs_w_formulas` into {eq}`bhs_eta_formula` gives

```{math}
:label: bhs_N_rw
N(x) = \frac{\beta\sigma_\varepsilon^2}{2(1-\beta)^3\theta^2}.
```

For $W$, we guess $W(x_t) = \frac{1}{1-\beta}[c_t + d]$ for some constant $d$ and verify it in the risk-sensitive Bellman equation {eq}`bhs_bellman_type1`.

Under the random walk, $W(x_{t+1}) = \frac{1}{1-\beta}[c_t + \mu + \sigma_\varepsilon\varepsilon_{t+1} + d]$, so $-W(x_{t+1})/\theta$ is affine in the standard normal $\varepsilon_{t+1}$.

Using the fact that $\log E[e^Z] = \mu_Z + \frac{1}{2}\sigma_Z^2$ for a normal random variable $Z$, the Bellman equation {eq}`bhs_bellman_type1` reduces to a constant-matching condition that pins down $d$ ({ref}`Exercise 9 <dov_ex9>` works through the algebra):

```{math}
:label: bhs_W_rw
W(x_t) = \frac{1}{1-\beta}\left[c_t + \frac{\beta}{1-\beta}\left(\mu - \frac{\sigma_\varepsilon^2}{2(1-\beta)\theta}\right)\right].
```

Using $W = J + \theta N$, the type III/IV value function is

```{math}
:label: bhs_J_rw
J(x_t) = W(x_t) - \theta N(x_t) = \frac{1}{1-\beta}\left[c_t + \frac{\beta}{1-\beta}\left(\mu - \frac{\sigma_\varepsilon^2}{(1-\beta)\theta}\right)\right].
```

The coefficient on $\sigma_\varepsilon^2/[(1-\beta)\theta]$ doubles from $\tfrac{1}{2}$ in $W$ to $1$ in $J$ because $W$ includes the entropy "rebate" $\theta N$ that partially offsets the pessimistic tilt, while $J$ evaluates consumption purely under the worst-case model.

This difference propagates directly into the welfare calculations below.

(detection_error_section)=
## A new calibration language: detection-error probabilities

The preceding section derived SDF moments, value functions, and worst-case distortions as functions of $\gamma$ (or equivalently $\theta$).

But if $\gamma$ should not be calibrated by introspection about atemporal gambles, what replaces it?

The answer proposed by {cite:t}`BHS_2009` is a statistical test that asks how easily one could distinguish the approximating model from its worst-case alternative.

### Likelihood-ratio testing and detection errors

Let $L_T$ be the log likelihood ratio between the worst-case and approximating models based on a sample of length $T$.

Define

```{math}
p_A = \Pr_A(L_T < 0),
\qquad
p_B = \Pr_B(L_T > 0),
```

where $\Pr_A$ and $\Pr_B$ denote probabilities under the approximating and worst-case models.

Then $p(\theta^{-1}) = \frac{1}{2}(p_A + p_B)$ is the average probability of choosing the wrong model.

Fix a sample size $T$ (here 235 quarters, matching the postwar US data used in the paper).

For a given $\theta$, compute the worst-case model and ask: if a Bayesian ran a likelihood-ratio test to distinguish the approximating model from the worst-case model, what fraction of the time would she make an error?

That fraction is the **detection-error probability** $p(\theta^{-1})$.

A high $p$ (near 0.5) means the two models are nearly indistinguishable, so the consumer's fear is hard to rule out.

A low $p$ means the worst-case model is easy to reject and the robustness concern is less compelling.

### Market price of model uncertainty

The **market price of model uncertainty** (MPU) is the conditional standard deviation of the distortion:

```{math}
:label: bhs_mpu_formula
\text{MPU}
=
\operatorname{std}(\hat g_{t+1})
=
\sqrt{e^{w(\theta)^2}-1}
\approx |w(\theta)|.
```

In the Gaussian mean-shift setting, $L_T$ is normal with mean $\pm \tfrac{1}{2}w^2T$ and variance $w^2T$, so the detection-error probability has the closed form ({ref}`Exercise 6 <dov_ex6>` derives this)

```{math}
:label: bhs_detection_formula
p(\theta^{-1})
=
\frac{1}{2}\left(p_A + p_B\right),
```

```{math}
:label: bhs_detection_closed
p(\theta^{-1}) = \Phi \left(-\frac{|w(\theta)|\sqrt{T}}{2}\right).
```

```{code-cell} ipython3
def detection_probability(θ, model):
    w = abs(w_from_θ(θ, model))
    return norm.cdf(-0.5 * w * np.sqrt(T))


def θ_from_detection_probability(p, model):
    if p >= 0.5:
        return np.inf
    w_abs = -2.0 * norm.ppf(p) / np.sqrt(T)
    if model == "rw":
        return rw["σ_ε"] / ((1.0 - β) * w_abs)
    if model == "ts":
        return ts["σ_ε"] / ((1.0 - β * ts["ρ"]) * w_abs)
    raise ValueError("model must be 'rw' or 'ts'")
```

### Interpreting the calibration objects

We now summarize the chain of mappings that connects preference parameters to statistical distinguishability.

The parameter $\theta$ indexes how expensive it is for the minimizing player to distort the approximating model.

A small $\theta$ means a cheap distortion and therefore stronger robustness concerns.

The associated $\gamma = 1 + \left[(1-\beta)\theta\right]^{-1}$ can be large even when we do not want to interpret behavior as extreme atemporal risk aversion.

The distortion magnitude $|w(\theta)|$ is a direct measure of how pessimistically the agent tilts one-step probabilities.

Detection error probability $p(\theta^{-1})$ translates that tilt into a statistical statement about finite-sample distinguishability.

High $p(\theta^{-1})$ means the two models are hard to distinguish.

Low $p(\theta^{-1})$ means they are easier to distinguish.

This mapping bridges econometric identification and preference calibration.

Finally, recall from {eq}`bhs_eta_formula` that discounted entropy is $\eta = \frac{\beta}{2(1-\beta)}w(\theta)^2$.

This tells us that when the distortion is a Gaussian mean shift, discounted entropy is proportional to the squared market price of model uncertainty.

### Detection probabilities across the two models

The left panel below plots $p(\theta^{-1})$ against $\theta^{-1}$ for the two consumption specifications.

Notice that the same numerical $\theta$ corresponds to very different detection probabilities across models, because baseline dynamics differ.

The right panel resolves this by plotting detection probabilities against discounted relative entropy $\eta$, which normalizes the statistical distance.

Indexed by $\eta$, the two curves coincide.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Detection probabilities across two models
    name: fig-bhs-2
---
θ_inv_grid = np.linspace(0.0, 1.8, 400)
θ_grid = np.full_like(θ_inv_grid, np.inf)
mask_θ = θ_inv_grid > 0.0
θ_grid[mask_θ] = 1.0 / θ_inv_grid[mask_θ]

p_rw = np.array([detection_probability(θ, "rw") for θ in θ_grid])
p_ts = np.array([detection_probability(θ, "ts") for θ in θ_grid])

η_rw = np.array([η_from_θ(θ, "rw") for θ in θ_grid])
η_ts = np.array([η_from_θ(θ, "ts") for θ in θ_grid])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(θ_inv_grid, 100.0 * p_rw, lw=2, label="random walk")
axes[0].plot(θ_inv_grid, 100.0 * p_ts, lw=2, label="trend stationary")
axes[0].set_xlabel(r"$\theta^{-1}$")
axes[0].set_ylabel("detection error probability (percent)")
axes[0].legend(frameon=False)

axes[1].plot(η_rw, 100.0 * p_rw, lw=2, label="random walk")
axes[1].plot(η_ts, 100.0 * p_ts, lw=2, ls="--", label="trend stationary")
axes[1].set_xlabel(r"discounted entropy $\eta$")
axes[1].set_ylabel("detection error probability (percent)")
axes[1].set_xlim(0.0, 10)
axes[1].legend(frameon=False)

plt.tight_layout()
plt.show()
```

Detection-error probabilities (or equivalently, discounted entropy) therefore provide the right cross-model yardstick.

Holding $\theta$ fixed when switching from a random walk to a trend-stationary specification implicitly changes how much misspecification the consumer fears.

Holding $\eta$ or $p$ fixed keeps the statistical difficulty of detecting misspecification constant.

The explicit mapping that equates discounted entropy across models is ({ref}`Exercise 7 <dov_ex7>` derives it):

```{math}
:label: bhs_theta_cross_model
\theta_{\text{TS}}
=
\left(\frac{\sigma_\varepsilon^{\text{TS}}}{\sigma_\varepsilon^{\text{RW}}}\right)
\frac{1-\beta}{1-\rho\beta} \theta_{\text{RW}}.
```

At our calibration $\sigma_\varepsilon^{\text{TS}} = \sigma_\varepsilon^{\text{RW}}$, this simplifies to $\theta_{\text{TS}} = \frac{1-\beta}{1-\rho\beta}\theta_{\text{RW}}$.

Because $\rho = 0.98$ and $\beta = 0.995$, the ratio $(1-\beta)/(1-\rho\beta)$ is much less than one, so holding entropy fixed requires a substantially smaller $\theta$ (stronger robustness) for the trend-stationary model than for the random walk.

## Detection probabilities unify the two models

We now redraw Tallarini's figure using detection-error probabilities.

For each detection-error probability $p(\theta^{-1}) = 0.50, 0.45, \ldots, 0.01$, invert to find the model-specific $\theta$, convert to $\gamma$, and plot the implied $(E(m), \sigma(m))$ pair.

```{code-cell} ipython3
p_points = np.array(
    [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.01])

θ_rw_points = np.array(
    [θ_from_detection_probability(p, "rw") for p in p_points])
θ_ts_points = np.array(
    [θ_from_detection_probability(p, "ts") for p in p_points])

γ_rw_points = np.array([γ_from_θ(θ) for θ in θ_rw_points])
γ_ts_points = np.array([γ_from_θ(θ) for θ in θ_ts_points])

Em_rw_p = np.array(
    [moments_type1_rw(γ)[0] for γ in γ_rw_points])
σ_m_rw_p = np.array(
    [moments_type1_rw(γ)[0] * moments_type1_rw(γ)[1] for γ in γ_rw_points])
Em_ts_p = np.array(
    [moments_type1_ts(γ)[0] for γ in γ_ts_points])
σ_m_ts_p = np.array(
    [moments_type1_ts(γ)[0] * moments_type1_ts(γ)[1] for γ in γ_ts_points])

print("p      γ_rw      γ_ts")
for p, g1, g2 in zip(p_points, γ_rw_points, γ_ts_points):
    print(f"{p:>4.2f} {g1:>9.2f} {g2:>9.2f}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Pricing loci from common detectability
    name: fig-bhs-3
---
from scipy.optimize import brentq

# Empirical Sharpe ratio — the minimum of the HJ bound curve
sharpe = (r_e_mean - r_f_mean) / r_excess_std

def sharpe_gap(p, model):
    """Market price of risk minus Sharpe ratio, as a function of p."""
    if p >= 0.5:
        return -sharpe
    θ = θ_from_detection_probability(p, model)
    γ = γ_from_θ(θ)
    _, mpr = moments_type1_rw(γ) if model == "rw" else moments_type1_ts(γ)
    return mpr - sharpe

p_hj_rw = brentq(sharpe_gap, 1e-4, 0.49, args=("rw",))
p_hj_ts = brentq(sharpe_gap, 1e-4, 0.49, args=("ts",))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Em_rw_p, σ_m_rw_p, "o",
            label="random walk")
ax.plot(Em_ts_p, σ_m_ts_p, "+", markersize=12,
            label="trend stationary")
ax.plot(Em_grid, HJ_std, lw=2,
            color="black", label="Hansen-Jagannathan bound")

# Mark p where each model's market price of risk reaches the Sharpe ratio
for p_hj, model, color, name, marker in [
    (p_hj_rw, "rw", "C0", "RW", "o"),
    (p_hj_ts, "ts", "C1", "TS", "+"),
]:
    θ_hj = θ_from_detection_probability(p_hj, model)
    γ_hj = γ_from_θ(θ_hj)
    Em_hj, mpr_hj = (moments_type1_rw(γ_hj) if model == "rw"
                      else moments_type1_ts(γ_hj))
    σ_m_hj = Em_hj * mpr_hj
    ax.axhline(σ_m_hj, ls="--", lw=1, color=color,
               label=f"{name} reaches bound at $p = {p_hj:.3f}$")
    if model == "ts":
        ax.plot(Em_hj, σ_m_hj, marker, markersize=12, color=color)
    else:
        ax.plot(Em_hj, σ_m_hj, marker, color=color)

ax.set_xlabel(r"$E(m)$")
ax.set_ylabel(r"$\sigma(m)$")
ax.legend(frameon=False)
ax.set_xlim(0.96, 1.05)
ax.set_ylim(0.0, 0.34)

plt.tight_layout()
plt.show()
```

The result is striking: the random-walk and trend-stationary loci nearly coincide.

Recall that under Tallarini's $\gamma$-calibration, reaching the Hansen--Jagannathan bound required $\gamma \approx 50$ for the random walk but $\gamma \approx 75$ for the trend-stationary model --- very different numbers for the "same" preference parameter.

Under detection-error calibration, both models reach the bound at the same detectability level.

The apparent model dependence was an artifact of using $\gamma$ as a cross-model yardstick.

Once we measure robustness concerns in units of statistical detectability, the two consumption specifications tell the same story: a representative consumer with moderate, difficult-to-dismiss fears about model misspecification behaves as though she has very high risk aversion.

The following figure brings together the two key ideas of this section: a small one-step density shift that is hard to detect (left panel) compounds into a large gap in expected log consumption (right panel).

At $p = 0.03$ both models share the same innovation mean shift $w$, and the left panel shows that the approximating and worst-case one-step densities nearly coincide.

The right panel reveals the cumulative consequence: a per-period shift that is virtually undetectable compounds into a large gap in expected log consumption, especially under random-walk dynamics where each shock has a permanent effect.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Small one-step density shift (left) produces large cumulative
      consumption gap (right) at detection-error probability $p = 0.03$ with $T = 240$ quarters
    name: fig-bhs-fear
---
p_star = 0.03
θ_star = θ_from_detection_probability(p_star, "rw")
w_star = w_from_θ(θ_star, "rw") 
σ_ε = rw["σ_ε"]
ρ = ts["ρ"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ε = np.linspace(-4.5, 4.5, 500)
f0 = norm.pdf(ε, 0, 1)
fw = norm.pdf(ε, w_star, 1)

ax1.fill_between(ε, f0, alpha=0.15, color='k')
ax1.plot(ε, f0, 'k', lw=2.5,
         label=r'Approximating $\mathcal{N}(0, 1)$')
ax1.fill_between(ε, fw, alpha=0.15, color='C3')
ax1.plot(ε, fw, 'C3', lw=2, ls='--',
         label=f'Worst case $\mathcal{{N}}({w_star:.2f},1)$')

peak = norm.pdf(0, 0, 1)
ax1.annotate('', xy=(w_star, 0.55 * peak), xytext=(0, 0.55 * peak),
             arrowprops=dict(arrowstyle='->', color='C3', lw=1.8))
ax1.text(w_star / 2, 0.59 * peak, f'$w = {w_star:.2f}$',
         ha='center', fontsize=11, color='C3')

ax1.set_xlabel(r'$\varepsilon_{t+1}$')
ax1.set_ylabel('Density')
ax1.legend(frameon=False)

quarters = np.arange(0, 241)
years = quarters / 4

gap_rw = 100 * σ_ε * w_star * quarters
gap_ts = 100 * σ_ε * w_star * (1 - ρ**quarters) / (1 - ρ)

ax2.plot(years, gap_rw, 'C0', lw=2.5, label='Random walk')
ax2.plot(years, gap_ts, 'C1', lw=2.5, label='Trend stationary')
ax2.fill_between(years, gap_rw, alpha=0.1, color='C0')
ax2.fill_between(years, gap_ts, alpha=0.1, color='C1')
ax2.axhline(0, color='k', lw=0.5, alpha=0.3)

# Endpoint labels
ax2.text(61, gap_rw[-1], f'{gap_rw[-1]:.1f}%',
         fontsize=10, color='C0', va='center')
ax2.text(61, gap_ts[-1], f'{gap_ts[-1]:.1f}%',
         fontsize=10, color='C1', va='center')

ax2.set_xlabel('Years')
ax2.set_ylabel('Gap in expected log consumption (%)')
ax2.legend(frameon=False, loc='lower left')
ax2.set_xlim(0, 68)

plt.tight_layout()
plt.show()
```

The next figure decomposes the log SDF into two additive components.

Taking logs of the SDF {eq}`bhs_sdf` gives

$$
\log m_{t+1}
=
\underbrace{\log \beta - \Delta c_{t+1}}_{\text{log-utility intertemporal MRS}}
+
\underbrace{\log \hat g_{t+1}}_{\text{worst-case distortion}}.
$$

Under the random-walk model, $\Delta c_{t+1} = \mu + \sigma_\varepsilon \varepsilon_{t+1}$, and the Gaussian distortion {eq}`bhs_ghat_gaussian` gives $\log \hat g_{t+1} = w \varepsilon_{t+1} - \tfrac{1}{2}w^2$.

Substituting, we can write

$$
\log m_{t+1}
=
\bigl(\log\beta - \mu - \tfrac{1}{2}w^2\bigr)
-
(\sigma_\varepsilon - w)\varepsilon_{t+1},
$$

so the slope of $\log m_{t+1}$ in $\varepsilon_{t+1}$ is $\sigma_\varepsilon - w$.

Since $w < 0$, the distortion steepens the SDF relative to what log utility alone would deliver.

In the figure below, the intertemporal marginal rate of substitution (IMRS) is nearly flat: at postwar calibrated volatility ($\sigma_\varepsilon = 0.005$), it contributes almost nothing to the pricing kernel's slope.

The distortion accounts for virtually all of the SDF volatility --- what looks like extreme risk aversion ($\gamma \approx 34$) is really log utility plus moderate fears of model misspecification.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Doubts or variability? Decomposition of the robust SDF into
      log-utility IMRS and worst-case distortion at $p = 0.10$"
    name: fig-bhs-sdf-decomp
---
θ_cal = θ_from_detection_probability(0.10, "rw")
γ_cal = γ_from_θ(θ_cal)
w_cal = w_from_θ(θ_cal, "rw")

μ_c, σ_c = rw["μ"], rw["σ_ε"]
Δc = np.linspace(μ_c - 3.5 * σ_c, μ_c + 3.5 * σ_c, 300)
ε = (Δc - μ_c) / σ_c

log_imrs = np.log(β) - Δc
log_ghat = w_cal * ε - 0.5 * w_cal**2
log_sdf = log_imrs + log_ghat

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(100 * Δc, log_imrs, 'C1', lw=2,
        label=r'IMRS: $\log\beta - \Delta c$')
ax.plot(100 * Δc, log_ghat, 'C3', lw=2, ls='--',
        label=r'Distortion: $\log\hat{g}$')
ax.plot(100 * Δc, log_sdf, 'k', lw=2,
        label=r'SDF: $\log m = \log\mathrm{IMRS} + \log\hat{g}$')
ax.axhline(0, color='k', lw=0.5, alpha=0.3)
ax.set_xlabel(r'Consumption growth $\Delta c_{t+1}$ (%)')
ax.set_ylabel('Log SDF component')
ax.legend(frameon=False, fontsize=10, loc='upper right')

plt.show()
```

(welfare_experiments)=
## What do risk premia measure?

{cite:t}`Lucas_2003` asked how much consumption a representative consumer would sacrifice to eliminate aggregate fluctuations.

His answer rested on the assumption that the consumer knows the data-generating process.

The robust reinterpretation introduces a second, distinct thought experiment.

Instead of eliminating all randomness, suppose we keep randomness but remove the consumer's fear of model misspecification (set $\theta = \infty$).

How much would she pay for that relief?

Formally, we seek a permanent proportional reduction $c_0 - c_0^k$ in initial log consumption that leaves an agent of type $k$ indifferent between the original risky plan and a deterministic certainty-equivalent path.

Because utility is log and the consumption process is Gaussian, these compensations are available in closed form.

### The certainty equivalent path

The point of comparison is the deterministic path with the same mean level of consumption as the stochastic plan:

```{math}
:label: bhs_ce_path
c_{t+1}^{ce} - c_t^{ce} = \mu + \tfrac{1}{2}\sigma_\varepsilon^2.
```

The additional $\tfrac{1}{2}\sigma_\varepsilon^2$ term is a Jensen's inequality correction: $E[C_t] = E[e^{c_t}] = \exp(c_0 + t\mu + \tfrac{1}{2}t\sigma_\varepsilon^2)$, so {eq}`bhs_ce_path` matches the mean *level* of consumption at every date.

### Compensating variations from the value functions

We use the closed-form value functions derived earlier: {eq}`bhs_W_rw` for the type I/II value function $W$ and {eq}`bhs_J_rw` for the type III/IV value function $J$.

For the certainty-equivalent path {eq}`bhs_ce_path`, there is no risk and no model uncertainty ($\theta = \infty$, so $\hat g = 1$), so the value function reduces to discounted expected log utility.

With $c_t^{ce} = c_0^J + t(\mu + \tfrac{1}{2}\sigma_\varepsilon^2)$, we have

$$
U^{ce}(c_0^J)
= \sum_{t=0}^{\infty}\beta^t c_t^{ce}
= \sum_{t=0}^{\infty}\beta^t \bigl[c_0^J + t(\mu + \tfrac{1}{2}\sigma_\varepsilon^2)\bigr]
= \frac{c_0^J}{1-\beta} + \frac{\beta(\mu + \tfrac{1}{2}\sigma_\varepsilon^2)}{(1-\beta)^2},
$$

where we used $\sum_{t \geq 0}\beta^t = \frac{1}{1-\beta}$ and $\sum_{t \geq 0}t\beta^t = \frac{\beta}{(1-\beta)^2}$. 

Factoring gives

$$
U^{ce}(c_0^J) = \frac{1}{1-\beta}\left[c_0^J + \frac{\beta}{1-\beta}\left(\mu + \tfrac{1}{2}\sigma_\varepsilon^2\right)\right].
$$

### Type I (Epstein--Zin) compensation

Setting $U^{ce}(c_0^I) = W(x_0)$ from {eq}`bhs_W_rw`:

$$
\frac{1}{1-\beta}\left[c_0^I + \frac{\beta}{1-\beta}\left(\mu + \tfrac{1}{2}\sigma_\varepsilon^2\right)\right]
=
\frac{1}{1-\beta}\left[c_0 + \frac{\beta}{1-\beta}\left(\mu - \frac{\sigma_\varepsilon^2}{2(1-\beta)\theta}\right)\right].
$$

Multiplying both sides by $(1-\beta)$ and cancelling the common $\frac{\beta\mu}{1-\beta}$ terms gives

$$
c_0^I + \frac{\beta\sigma_\varepsilon^2}{2(1-\beta)}
=
c_0 - \frac{\beta\sigma_\varepsilon^2}{2(1-\beta)^2\theta}.
$$

Solving for $c_0 - c_0^I$:

```{math}
:label: bhs_comp_type1
c_0 - c_0^I
=
\frac{\beta\sigma_\varepsilon^2}{2(1-\beta)}\left(1 + \frac{1}{(1-\beta)\theta}\right)
=
\frac{\beta\sigma_\varepsilon^2\gamma}{2(1-\beta)},
```

where the last step uses $\gamma = 1 + [(1-\beta)\theta]^{-1}$.

### Type II (multiplier) decomposition

Because $W \equiv U$, we have $c_0^{II} = c_0^I$ and the total compensation is the same.

However, the interpretation differs: we can now decompose it into **risk** and **model uncertainty** components.

A type II agent with $\theta = \infty$ (no model uncertainty) has log preferences and requires

```{math}
:label: bhs_type2_rw_decomp
\Delta c_0^{risk}
=
\frac{\beta \sigma_\varepsilon^2}{2(1-\beta)},
\qquad
\Delta c_0^{uncertainty}
=
\frac{\beta \sigma_\varepsilon^2}{2(1-\beta)^2\theta}.
```

The risk term $\Delta c_0^{risk}$ is Lucas's cost of business cycles: at postwar consumption volatility ($\sigma_\varepsilon \approx 0.005$), it is small.

The uncertainty term $\Delta c_0^{uncertainty}$ is the additional compensation a type II agent requires for facing model misspecification. It can be first-order whenever the detection-error probability is moderate, because $\theta$ appears in the denominator.

### Type III (constraint) compensation

For a type III agent, we set $U^{ce}(c_0^{III}) = J(x_0)$ using the value function $J$ from {eq}`bhs_J_rw`:

$$
\frac{1}{1-\beta}\left[c_0^{III} + \frac{\beta}{1-\beta}\left(\mu + \tfrac{1}{2}\sigma_\varepsilon^2\right)\right]
=
\frac{1}{1-\beta}\left[c_0 + \frac{\beta}{1-\beta}\left(\mu - \frac{\sigma_\varepsilon^2}{(1-\beta)\theta}\right)\right].
$$

Following the same algebra as for type I but with the doubled uncertainty correction in $J$:

$$
c_0 - c_0^{III}
=
\frac{\beta\sigma_\varepsilon^2}{2(1-\beta)} + \frac{\beta\sigma_\varepsilon^2}{(1-\beta)^2\theta}.
$$

Using $\frac{1}{(1-\beta)\theta} = \gamma - 1$, this simplifies to

```{math}
:label: bhs_type3_rw_decomp
c_0 - c_0^{III}
=
\frac{\beta\sigma_\varepsilon^2}{2(1-\beta)}(2\gamma - 1).
```

The risk component is the same $\frac{\beta\sigma_\varepsilon^2}{2(1-\beta)}$ as before. The uncertainty component alone is

$$
c_0^{III}(r) - c_0^{III}
=
\frac{\beta\sigma_\varepsilon^2}{(1-\beta)^2\theta},
$$

which is *twice* the type II uncertainty compensation {eq}`bhs_type2_rw_decomp`.
The factor of two traces back to the difference between $W$ and $J$ noted after {eq}`bhs_J_rw`: the entropy rebate $\theta N$ in $W = J + \theta N$ partially offsets the pessimistic tilt for the type II agent, but not for the type III agent who evaluates consumption purely under the worst-case model.

### Type IV (ex post Bayesian) compensation

A type IV agent believes the pessimistic model, so the perceived drift is $\tilde\mu = \mu - \sigma_\varepsilon^2/[(1-\beta)\theta]$.
The compensation for moving to the certainty-equivalent path is the same as {eq}`bhs_type3_rw_decomp`, because this agent ranks plans using the same value function $J$.

### Comparison with a risky but free-of-model-uncertainty path

The certainty equivalents above compare a risky plan to a deterministic path, eliminating both risk and uncertainty simultaneously.

We now describe an alternative measure that isolates compensation for model uncertainty by keeping risk intact.

We compare two situations with identical risky consumption for all dates $t \geq 1$.

All compensation for model uncertainty is concentrated in an adjustment to date-zero consumption.

Specifically, we seek $c_0^{II}(u)$ that makes a type II agent indifferent between:

1. Facing the stochastic plan under $\theta < \infty$ (fear of model misspecification), consuming $c_0$ at date zero.
2. Facing the **same** stochastic plan under $\theta = \infty$ (no fear of misspecification), but consuming only $c_0^{II}(u) < c_0$ at date zero.

In both cases, continuation consumptions $c_t$ for $t \geq 1$ are generated by the random walk starting from the **same** $c_0$.

For the type II agent under $\theta < \infty$, the total value is $W(c_0)$ from {eq}`bhs_W_rw`.

For the agent liberated from model uncertainty ($\theta = \infty$), the value is

$$
c_0^{II}(u) + \beta E\left[V^{\log}(c_1)\right],
$$

where $V^{\log}(c_t) = \frac{1}{1-\beta} \left[c_t + \frac{\beta\mu}{1-\beta}\right]$ is the log-utility value function and $c_1 = c_0 + \mu + \sigma_\varepsilon \varepsilon_1$.

Since $c_1$ is built from $c_0$ (not $c_0^{II}(u)$), the continuation is

$$
\beta E\left[V^{\log}(c_1)\right]
= \frac{\beta}{1-\beta} E\left[c_1 + \frac{\beta\mu}{1-\beta}\right]
= \frac{\beta}{1-\beta}\left[c_0 + \mu + \frac{\beta\mu}{1-\beta}\right]
= \frac{\beta}{1-\beta}\left[c_0 + \frac{\mu}{1-\beta}\right],
$$

where we used $E[c_1] = c_0 + \mu$ (the noise term has zero mean). Expanding gives

$$
\beta E\left[V^{\log}(c_1)\right]
= \frac{\beta c_0}{1-\beta} + \frac{\beta\mu}{(1-\beta)^2}.
$$

Setting $W(c_0)$ equal to the liberation value and simplifying:

$$
\frac{c_0}{1-\beta} + \frac{\beta\mu}{(1-\beta)^2} - \frac{\beta\sigma_\varepsilon^2}{2(1-\beta)^3\theta}
=
c_0^{II}(u) + \frac{\beta c_0}{1-\beta} + \frac{\beta\mu}{(1-\beta)^2}.
$$

Because $\frac{c_0}{1-\beta} - \frac{\beta c_0}{1-\beta} = c_0$, solving for the compensation gives

```{math}
:label: bhs_comp_type2u
c_0 - c_0^{II}(u) = \frac{\beta\sigma_\varepsilon^2}{2(1-\beta)^3\theta} = \frac{\beta\sigma_\varepsilon^2(\gamma - 1)}{2(1-\beta)^2}.
```

This is $\frac{1}{1-\beta}$ times the uncertainty compensation $\Delta c_0^{\text{uncertainty}}$ from {eq}`bhs_type2_rw_decomp`.

The multiplicative factor $\frac{1}{1-\beta}$ arises because all compensation is concentrated in a single period: adjusting $c_0$ alone must offset the cumulative loss in continuation value that the uncertainty penalty imposes in every future period.

An analogous calculation for a **type III** agent, using $J(c_0)$ from {eq}`bhs_J_rw`, gives

```{math}
:label: bhs_comp_type3u
c_0 - c_0^{III}(u) = \frac{\beta\sigma_\varepsilon^2}{(1-\beta)^3\theta} = \frac{\beta\sigma_\varepsilon^2(\gamma - 1)}{(1-\beta)^2},
```

which is $\frac{1}{1-\beta}$ times the type III uncertainty compensation and **twice** the type II compensation {eq}`bhs_comp_type2u`, again reflecting the absence of the entropy rebate in $J$.

### Summary of welfare compensations (random walk)

The following table collects all compensating variations for the random walk model.

| Agent | Compensation | Formula | Measures |
|:------|:-------------|:--------|:---------|
| I, II | $c_0 - c_0^{II}$ | $\frac{\beta\sigma_\varepsilon^2\gamma}{2(1-\beta)}$ | risk + uncertainty (vs. deterministic) |
| II | $c_0 - c_0^{II}(r)$ | $\frac{\beta\sigma_\varepsilon^2}{2(1-\beta)}$ | risk only (vs. deterministic) |
| II | $c_0^{II}(r) - c_0^{II}$ | $\frac{\beta\sigma_\varepsilon^2}{2(1-\beta)^2\theta}$ | uncertainty only (vs. deterministic) |
| II | $c_0 - c_0^{II}(u)$ | $\frac{\beta\sigma_\varepsilon^2}{2(1-\beta)^3\theta}$ | uncertainty only (vs. risky path) |
| III | $c_0 - c_0^{III}$ | $\frac{\beta\sigma_\varepsilon^2(2\gamma-1)}{2(1-\beta)}$ | risk + uncertainty (vs. deterministic) |
| III | $c_0^{III}(r) - c_0^{III}$ | $\frac{\beta\sigma_\varepsilon^2}{(1-\beta)^2\theta}$ | uncertainty only (vs. deterministic) |
| III | $c_0 - c_0^{III}(u)$ | $\frac{\beta\sigma_\varepsilon^2}{(1-\beta)^3\theta}$ | uncertainty only (vs. risky path) |

The "vs. deterministic" rows use the certainty-equivalent path {eq}`bhs_ce_path` as a benchmark; the "vs. risky path" rows use the risky-but-uncertainty-free comparison of {eq}`bhs_comp_type2u`--{eq}`bhs_comp_type3u`.

### Trend-stationary formulas

For the trend-stationary model, the denominators $(1-\beta)$ in the uncertainty terms are replaced by $(1-\beta\rho)$, and the risk terms involve $(1-\beta\rho^2)$:

$$
\Delta c_0^{risk,ts} = \frac{\beta\sigma_\varepsilon^2}{2(1-\beta\rho^2)},
\qquad
\Delta c_0^{unc,ts,II} = \frac{\beta\sigma_\varepsilon^2}{2(1-\beta\rho)^2\theta},
\qquad
\Delta c_0^{unc,ts,III} = \frac{\beta\sigma_\varepsilon^2}{(1-\beta\rho)^2\theta}.
$$

The qualitative message is the same: the risk component is negligible, and the model-uncertainty component dominates.

## Visualizing the welfare decomposition

We set $\beta = 0.995$ and calibrate $\theta$ so that $p(\theta^{-1}) = 0.10$, a conservative detection-error level.

```{code-cell} ipython3
p_star = 0.10
θ_star = θ_from_detection_probability(p_star, "rw")
γ_star = γ_from_θ(θ_star)
w_star = w_from_θ(θ_star, "rw")

# Type II compensations, random walk model
comp_risk_only = β * rw["σ_ε"]**2 / (2.0 * (1.0 - β))
comp_risk_unc = comp_risk_only + β * rw["σ_ε"]**2 / (2.0 * (1.0 - β)**2 * θ_star)

# Two useful decompositions in levels
risk_only_pct = 100.0 * (np.exp(comp_risk_only) - 1.0)
risk_unc_pct = 100.0 * (np.exp(comp_risk_unc) - 1.0)
uncertainty_only_pct = 100.0 * (np.exp(comp_risk_unc - comp_risk_only) - 1.0)

print(f"p*={p_star:.2f}, θ*={θ_star:.4f}, γ*={γ_star:.2f}, w*={w_star:.4f}")
print(f"risk only compensation (log units): {comp_risk_only:.6f}")
print(f"risk + uncertainty compensation (log units): {comp_risk_unc:.6f}")
print(f"risk only compensation (percent): {risk_only_pct:.3f}%")
print(f"risk + uncertainty compensation (percent): {risk_unc_pct:.3f}%")
print(f"uncertainty component alone (percent): {uncertainty_only_pct:.3f}%")

h = 250
t = np.arange(h + 1)

# Baseline approximating model fan
mean_base = rw["μ"] * t
std_base = rw["σ_ε"] * np.sqrt(t)

# Certainty equivalent line from Eq. (47), shifted by compensating variations
certainty_slope = rw["μ"] + 0.5 * rw["σ_ε"]**2
ce_risk = -comp_risk_only + certainty_slope * t
ce_risk_unc = -comp_risk_unc + certainty_slope * t

# Alternative models from the ambiguity set in panel B
mean_low = (rw["μ"] + rw["σ_ε"] * w_star) * t
mean_high = (rw["μ"] - rw["σ_ε"] * w_star) * t
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Certainty equivalents under robustness
    name: fig-bhs-4
---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Panel A
ax = axes[0]
ax.fill_between(t, mean_base - std_base, mean_base + std_base, alpha=0.25, color="tab:blue")
ax.plot(t, ce_risk_unc, lw=2, ls="--", color="black", label="certainty equivalent: risk + uncertainty")
ax.plot(t, ce_risk, lw=2, color="tab:orange", label="certainty equivalent: risk only")
ax.plot(t, mean_base, lw=2, color="tab:blue", label="approximating-model mean")
ax.set_xlabel("quarters")
ax.set_ylabel("log consumption")
ax.legend(frameon=False, fontsize=8, loc="upper left")

# Panel B
ax = axes[1]
ax.fill_between(t, mean_base - std_base, mean_base + std_base, alpha=0.20, color="tab:blue")
ax.fill_between(t, mean_low - std_base, mean_low + std_base, alpha=0.20, color="tab:red")
ax.fill_between(t, mean_high - std_base, mean_high + std_base, alpha=0.20, color="tab:green")
ax.plot(t, ce_risk_unc, lw=2, ls="--", color="black", label="certainty equivalent: risk + uncertainty")
ax.plot(t, mean_base, lw=2, color="tab:blue", label="approximating-model mean")
ax.plot(t, mean_low, lw=2, color="tab:red", label="worst-case-leaning mean")
ax.plot(t, mean_high, lw=2, color="tab:green", label="best-case-leaning mean")
ax.set_xlabel("quarters")
ax.set_ylabel("log consumption")
ax.legend(frameon=False, fontsize=8, loc="upper left")

plt.tight_layout()
plt.show()
```

The left panel illustrates the elimination of model uncertainty and risk for a type II agent.

The shaded fan shows a one-standard-deviation band for the $j$-step-ahead conditional distribution of $c_t$ under the calibrated random-walk model.

The dashed line $c^{II}$ shows the certainty-equivalent path whose date-zero consumption is reduced by $c_0 - c_0^{II}$, making the type II agent indifferent between this deterministic trajectory and the stochastic plan; it compensates for bearing both risk and model ambiguity.

The solid line $c^r$ shows the certainty equivalent for a type II agent without model uncertainty ($\theta = \infty$), initialized at $c_0 - c_0^{II}(r)$.
At postwar calibrated values this gap is small, so $c^r$ sits just below the center of the fan.

Consistent with {cite:t}`Lucas_2003`, the welfare gains from eliminating well-understood risk are very small.

The large welfare gains found by {cite:t}`Tall2000` can be reinterpreted as arising not from reducing risk, but from reducing model uncertainty.

The right panel shows the set of nearby models that the robust consumer guards against.

Each shaded fan depicts a one-standard-deviation band for a different model in the ambiguity set.

The models are statistically close to the baseline --- their detection-error probability is $p = 0.10$ --- but imply very different long-run consumption levels.

The consumer's caution against such alternatives accounts for the large certainty-equivalent gap in the left panel.

## How large are the welfare gains from resolving model uncertainty?

A type III (constraint-preference) agent evaluates the worst model inside an entropy ball of radius $\eta$.

As $\eta$ grows, the set of plausible misspecifications expands and the welfare cost of confronting model uncertainty rises.

Because $\eta$ is not directly interpretable, we instead index these costs by the associated detection-error probability $p(\eta)$.

The figure below plots compensation for removing model uncertainty, measured as a proportion of consumption, against $p(\eta)$.

```{code-cell} ipython3
η_grid = np.linspace(0.0, 5.0, 300)

# Use w and η relation, then convert to θ model by model
w_abs_grid = np.sqrt(2.0 * (1.0 - β) * η_grid / β)

θ_rw_from_η = np.full_like(w_abs_grid, np.inf)
θ_ts_from_η = np.full_like(w_abs_grid, np.inf)
mask_w = w_abs_grid > 0.0
θ_rw_from_η[mask_w] = rw["σ_ε"] / ((1.0 - β) * w_abs_grid[mask_w])
θ_ts_from_η[mask_w] = ts["σ_ε"] / ((1.0 - β * ts["ρ"]) * w_abs_grid[mask_w])

# Type III uncertainty terms from Table 3
gain_rw = np.where(
    np.isinf(θ_rw_from_η),
    0.0,
    β * rw["σ_ε"]**2 / ((1.0 - β)**2 * θ_rw_from_η),
)
gain_ts = np.where(
    np.isinf(θ_ts_from_η),
    0.0,
    β * ts["σ_ε"]**2 / ((1.0 - β * ts["ρ"])**2 * θ_ts_from_η),
)

# Convert log compensation to percent of initial consumption in levels
gain_rw_pct = 100.0 * (np.exp(gain_rw) - 1.0)
gain_ts_pct = 100.0 * (np.exp(gain_ts) - 1.0)

# Detection error probabilities implied by η (common across RW/TS for the Gaussian mean-shift case)
p_eta_pct = 100.0 * norm.cdf(-0.5 * w_abs_grid * np.sqrt(T))
order = np.argsort(p_eta_pct)
p_plot = p_eta_pct[order]
gain_rw_plot = gain_rw_pct[order]
gain_ts_plot = gain_ts_pct[order]
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Type III uncertainty compensation curve
    name: fig-bhs-5
---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(p_plot, gain_rw_plot, lw=2, label="RW type III")
ax.plot(p_plot, gain_ts_plot, lw=2, label="TS type III")
ax.set_xlabel(r"detection error probability $p(\eta)$ (percent)")
ax.set_ylabel("proportion of consumption (percent)")
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
```

The random-walk model implies somewhat larger costs than the trend-stationary model at the same detection-error probability, but both curves greatly exceed the classic Lucas cost of business cycles.

To put these magnitudes in perspective, Lucas estimated that eliminating all aggregate consumption risk is worth roughly 0.05% of consumption.

At detection-error probabilities of 10--20%, the model-uncertainty compensation alone runs to several percent of consumption.

Under the robust reading, the large risk premia that Tallarini matched with high $\gamma$ are compensations for bearing model uncertainty, and the implied welfare gains from resolving that uncertainty are correspondingly large.

The following contour plot shows how type II (multiplier) compensation varies over a two-dimensional parameter space: the detection-error probability $p$ and the consumption volatility $\sigma_\varepsilon$.

The star marks the calibrated point ($p = 0.10$, $\sigma_\varepsilon = 0.5\%$).

At the calibrated volatility, moving left (lower $p$, stronger robustness concerns) increases compensation dramatically, while the classic risk-only cost (the $p = 50\%$ edge) remains negligible.

Comparing the two panels shows that the random-walk model generates much larger welfare costs than the trend-stationary model at the same ($p$, $\sigma_\varepsilon$), because permanent shocks compound the worst-case drift indefinitely.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Type II compensation across detection-error probability and
      consumption volatility
    name: fig-bhs-contour
---
p_grid = np.linspace(0.02, 0.49, 300)
σ_grid = np.linspace(0.001, 0.015, 300)
P, Σ = np.meshgrid(p_grid, σ_grid)

W_abs = -2 * norm.ppf(P) / np.sqrt(T)

# RW: total type II = βσ²γ / [2(1-β)] 
Γ_rw = 1 + W_abs / Σ
comp_rw = 100 * (np.exp(β * Σ**2 * Γ_rw / (2 * (1 - β))) - 1)

# TS: risk + uncertainty 
ρ_val = ts["ρ"]
risk_ts = β * Σ**2 / (2 * (1 - β * ρ_val**2))
unc_ts = β * Σ * W_abs / (2 * (1 - β * ρ_val))
comp_ts = 100 * (np.exp(risk_ts + unc_ts) - 1)

levels = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for ax, comp, title in [(ax1, comp_rw, 'Random walk'),
                         (ax2, comp_ts, 'Trend stationary')]:
    cf = ax.contourf(100 * P, 100 * Σ, comp, levels=levels,
                     cmap='Blues', extend='both')
    cs = ax.contour(100 * P, 100 * Σ, comp, levels=levels,
                    colors='k', linewidths=0.5)
    ax.clabel(cs, fmt='%g%%', fontsize=8)
    ax.plot(10, 0.5, 'x', markersize=14, color='w',
            mec='k', mew=1, zorder=5)
    ax.set_xlabel(r'Detection-error probability $p$ (%)')
    ax.set_title(title)

ax1.set_ylabel(r'Consumption volatility $\sigma_\varepsilon$ (%)')

plt.tight_layout()
plt.show()
```

## Why doesn't learning eliminate these fears?

A natural objection is: if the consumer has 235 quarters of data, why can't she learn the true drift well enough to dismiss the worst-case model?

The answer is that the drift is a low-frequency feature of the data.

Estimating the mean of a random walk to the precision needed to reject small but economically meaningful shifts requires far more data than estimating volatility.

The following figure makes this point concrete.

Consumption is measured as real personal consumption expenditures on nondurable goods and services, deflated by its implicit chain price deflator, and expressed in per-capita terms using the civilian noninstitutional population aged 16+.

We construct real per-capita nondurables-plus-services consumption from four FRED series:

| FRED series | Description |
| --- | --- |
| `PCND` | Nominal PCE: nondurable goods (billions of \$, SAAR, quarterly) |
| `PCESV` | Nominal PCE: services (billions of \$, SAAR, quarterly) |
| `DPCERD3Q086SBEA` | PCE implicit price deflator (index 2017 $= 100$, quarterly) |
| `CNP16OV` | Civilian noninstitutional population, 16+ (thousands, monthly) |

We use nominal rather than chained-dollar components because chained-dollar series are not additive: chain-weighted indices update their base-period expenditure weights every period, so components deflated with different price changes do not sum to the separately chained aggregate. 

Adding nominal series and deflating the sum with a single price index avoids this problem.

The processing pipeline is:

1. Add nominal nondurables and services: $C_t^{nom} = C_t^{nd} + C_t^{sv}$.
2. Deflate by the PCE price index: $C_t^{real} = C_t^{nom} / (P_t / 100)$.
3. Convert to per-capita: divide by the quarterly average of the monthly population series.
4. Compute log consumption: $c_t = \log C_t^{real,pc}$.

When we plot *levels* of log consumption, we align the time index to 1948Q1--2006Q4, which yields $T+1 = 236$ quarterly observations.

```{code-cell} ipython3
start_date = dt.datetime(1947, 1, 1)
end_date = dt.datetime(2007, 1, 1)


def _read_fred_series(series_id, start_date, end_date):
    series = web.DataReader(series_id, "fred", start_date, end_date)[series_id]
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if series.empty:
        raise ValueError(f"FRED series '{series_id}' returned no data in sample window")
    return series


# Fetch nominal PCE components, deflator, and population from FRED
nom_nd = _read_fred_series("PCND", start_date, end_date)        # quarterly, 1947–
nom_sv = _read_fred_series("PCESV", start_date, end_date)       # quarterly, 1947–
defl = _read_fred_series("DPCERD3Q086SBEA", start_date, end_date)  # quarterly, 1947–
pop_m = _read_fred_series("CNP16OV", start_date, end_date)      # monthly, 1948–

# Step 1: add nominal nondurables + services (nominal $ are additive)
nom_total = nom_nd + nom_sv

# Step 2: deflate by PCE implicit price deflator (index 2017=100)
real_total = nom_total / (defl / 100.0)

# Step 3: convert to per-capita (population is monthly, so average to quarterly)
pop_q = pop_m.resample("QS").mean()
real_pc = (real_total / pop_q).dropna()

# Restrict to sample period 1948Q1–2006Q4
real_pc = real_pc.loc["1948-01-01":"2006-12-31"].dropna()

if real_pc.empty:
    raise RuntimeError("FRED returned no usable observations after alignment/filtering")

# Step 4: log consumption
log_c_data = np.log(real_pc.to_numpy(dtype=float).reshape(-1))
years_data = (real_pc.index.year + (real_pc.index.month - 1) / 12.0).to_numpy(dtype=float)

print(f"Fetched {len(log_c_data)} quarterly observations from FRED")
print(f"Sample: {years_data[0]:.1f} – {years_data[-1] + 0.25:.1f}")
print(f"Observations: {len(log_c_data)}")
```

We can verify Table 2 by computing sample moments of log consumption growth from our FRED data:

```{code-cell} ipython3
# Growth rates: 1948Q2 to 2006Q4 (T = 235 quarters)
diff_c = np.diff(log_c_data)

μ_hat = diff_c.mean()
σ_hat = diff_c.std(ddof=1)

print("Sample estimates from FRED data vs Table 2:")
print(f"  μ   = {μ_hat:.5f}   (Table 2 RW: {rw['μ']:.5f})")
print(f"  σ_ε = {σ_hat:.4f}    (Table 2: {rw['σ_ε']:.4f})")
print(f"  T   = {len(diff_c)} quarters")
```

```{code-cell} ipython3
p_fig6 = 0.20

# Figure 6 overlays deterministic lines on the loaded consumption data.
# Use sample-estimated RW moments to avoid data-vintage drift mismatches.
rw_fig6 = dict(μ=μ_hat, σ_ε=σ_hat)
w_fig6 = 2.0 * norm.ppf(p_fig6) / np.sqrt(T)

c = log_c_data
years = years_data

t6 = np.arange(T + 1)
μ_approx = rw_fig6["μ"]
μ_worst = rw_fig6["μ"] + rw_fig6["σ_ε"] * w_fig6

# Match BHS Figure 6 visual construction by fitting intercepts separately
# while holding the two drifts fixed.
a_approx = (c - μ_approx * t6).mean()
a_worst = (c - μ_worst * t6).mean()
line_approx = a_approx + μ_approx * t6
line_worst = a_worst + μ_worst * t6

p_right = np.linspace(0.01, 0.50, 500)
w_right = 2.0 * norm.ppf(p_right) / np.sqrt(T)
μ_worst_right = rw_fig6["μ"] + rw_fig6["σ_ε"] * w_right

μ_se = rw_fig6["σ_ε"] / np.sqrt(T)
upper_band = rw_fig6["μ"] + 2.0 * μ_se
lower_band = rw_fig6["μ"] - 2.0 * μ_se
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Drift distortion and sampling uncertainty
    name: fig-bhs-6
---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(years, c, lw=2, color="tab:blue", label="log consumption")
ax.plot(years, line_approx, lw=2, ls="--", color="black", label="approximating model")
ax.plot(
    years,
    line_worst,
    lw=2,
    ls=":",
    color="black",
    label=rf"wc model $p(\theta^{{-1}})={p_fig6:.1f}$",
)
ax.set_xlabel("year")
ax.set_ylabel("log consumption")
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax = axes[1]
ax.plot(
    100.0 * p_right,
    1_000.0 * μ_worst_right,
    lw=2,
    color="tab:red",
    label=r"$\mu + \sigma_\varepsilon w(\theta)$",
)
ax.axhline(1_000.0 * rw_fig6["μ"], lw=2, color="black", label=r"$\hat\mu$")
ax.axhline(1_000.0 * upper_band, lw=2, ls="--", color="gray", label=r"$\hat\mu \pm 2\hat s.e.$")
ax.axhline(1_000.0 * lower_band, lw=2, ls="--", color="gray")
ax.set_xlabel("detection error probability (percent)")
ax.set_ylabel(r"mean consumption growth ($\times 10^{-3}$)")
ax.legend(frameon=False, fontsize=8)
ax.set_xlim(0.0, 50.0)

plt.tight_layout()
plt.show()
```

In the left panel, postwar U.S. log consumption is shown alongside two deterministic trend lines: the approximating-model drift $\mu$ and the worst-case drift $\mu + \sigma_\varepsilon w(\theta)$ for $p(\theta^{-1}) = 0.20$.

The two trends are close enough that, even with decades of data, it is hard to distinguish them by eye.

In the right panel, as the detection-error probability rises (models become harder to tell apart), the worst-case mean growth rate moves back toward $\hat\mu$.

The dashed gray lines mark a two-standard-error band around the maximum-likelihood estimate of $\mu$.

Even at detection probabilities in the 5--20% range, the worst-case drift remains inside (or very near) this confidence band.

Drift distortions that are economically large --- large enough to generate substantial model-uncertainty premia --- are statistically small relative to sampling uncertainty in $\hat\mu$.

Robustness concerns persist despite long histories precisely because the low-frequency features that matter most for pricing are the hardest to estimate precisely.

## Concluding remarks

The title of this lecture poses a question: are large risk premia prices of **variability** (atemporal risk aversion) or prices of **doubts** (model uncertainty)?

The analysis above shows that the answer cannot be settled by asset-pricing data alone, because the two interpretations are observationally equivalent.

But the choice of interpretation matters for the conclusions we draw.

Under the risk-aversion reading, high Sharpe ratios imply that consumers would pay a great deal to smooth known aggregate consumption fluctuations.

Under the robustness reading, those same Sharpe ratios tell us consumers would pay a great deal to resolve uncertainty about which probability model governs consumption growth.

Three features of the analysis support the robustness reading:

1. Detection-error probabilities provide a more stable calibration language than $\gamma$: the two consumption models that required very different $\gamma$ values to match the data yield nearly identical pricing implications when indexed by detectability.
2. The welfare gains implied by asset prices decompose overwhelmingly into a model-uncertainty component, with the pure risk component remaining small --- consistent with Lucas's original finding.
3. The drift distortions that drive pricing are small enough to hide inside standard-error bands, so finite-sample learning cannot eliminate the consumer's fears.

Whether one ultimately prefers the risk or the uncertainty interpretation, the framework clarifies that the question is not about the size of risk premia but about the economic object those premia measure.

## Exercises

The following exercises ask you to fill in several derivation steps.

```{exercise}
:label: dov_ex1

Let $R_{t+1}$ be an $n \times 1$ vector of gross returns with unconditional mean $E(R)$ and covariance matrix $\Sigma_R$.

Let $m_{t+1}$ be a stochastic discount factor satisfying $\mathbf{1} = E[m_{t+1} R_{t+1}]$.

1. Use the covariance decomposition $E[mR] = E[m] E[R] + \operatorname{cov}(m,R)$ to show that $\operatorname{cov}(m,R) = \mathbf{1} - E[m] E[R] =: b$.
2. For a portfolio with weight vector $\alpha$ and return $R^p = \alpha^\top R$, show that $\operatorname{cov}(m, R^p) = \alpha^\top b$.
3. Apply the Cauchy--Schwarz inequality to the pair $(m, R^p)$ to obtain $|\alpha^\top b| \leq \sigma(m)\sqrt{\alpha^\top \Sigma_R\alpha}$.
4. Maximize the ratio $|\alpha^\top b|/\sqrt{\alpha^\top \Sigma_R \alpha}$ over $\alpha$ and show that the maximum is $\sqrt{b^\top \Sigma_R^{-1} b}$, attained at $\alpha^\star = \Sigma_R^{-1}b$.
5. Conclude that $\sigma(m) \geq \sqrt{b^\top \Sigma_R^{-1} b}$, which is {eq}`bhs_hj_unconditional`.
```

```{solution-start} dov_ex1
:class: dropdown
```

**Part 1.** From $\mathbf{1} = E[mR] = E[m] E[R] + \operatorname{cov}(m,R)$, rearranging gives $\operatorname{cov}(m,R) = \mathbf{1} - E[m] E[R]= b$.

**Part 2.** The portfolio return is $R^p = \alpha^\top R$, so

$$
\operatorname{cov}(m, R^p) = \operatorname{cov}(m, \alpha^\top R) = \alpha^\top \operatorname{cov}(m, R) = \alpha^\top b.
$$

**Part 3.** 
Applying the Cauchy--Schwarz inequality to $(m, R^p)$:

$$
|\alpha^\top b| = |\operatorname{cov}(m, R^p)| \leq \sigma(m) \sigma(R^p) = \sigma(m) \sqrt{\alpha^\top \Sigma_R \alpha}.
$$

**Part 4.** Rearranging Part 3 gives

$$
\frac{|\alpha^\top b|}{\sqrt{\alpha^\top \Sigma_R \alpha}} \leq \sigma(m).
$$

To maximize the left-hand side over $\alpha$, define the $\Sigma_R$-inner product $\langle u, v \rangle_{\Sigma} = u^\top \Sigma_R v$.

Inserting $I = \Sigma_R \Sigma_R^{-1}$ gives

$$
\alpha^\top b
= \alpha^\top (\Sigma_R \Sigma_R^{-1}) b
= (\alpha^\top \Sigma_R)(\Sigma_R^{-1} b)
= \langle \alpha, \Sigma_R^{-1}b \rangle_{\Sigma}.
$$

Cauchy--Schwarz in this inner product gives

$$
|\langle \alpha, \Sigma_R^{-1}b \rangle_{\Sigma}|
\leq
\sqrt{\langle \alpha, \alpha \rangle_{\Sigma}}\sqrt{\langle \Sigma_R^{-1}b, \Sigma_R^{-1}b \rangle_{\Sigma}}
=
\sqrt{\alpha^\top \Sigma_R \alpha} \sqrt{b^\top \Sigma_R^{-1} b},
$$

with equality when $\alpha \propto \Sigma_R^{-1} b$.

Substituting $\alpha^\star = \Sigma_R^{-1} b$ verifies

$$
\max_\alpha \frac{|\alpha^\top b|}{\sqrt{\alpha^\top \Sigma_R \alpha}} = \sqrt{b^\top \Sigma_R^{-1} b}.
$$

**Part 5.** Combining Parts 3 and 4 gives $\sigma(m) \geq \sqrt{b^\top \Sigma_R^{-1} b}$, which is {eq}`bhs_hj_unconditional`.

```{solution-end}
```

```{exercise}
:label: dov_ex2

Combine the SDF representation {eq}`bhs_sdf` with the random-walk consumption dynamics and the Gaussian mean-shift distortion to derive closed-form SDF moments.

1. Show that $\log m_{t+1}$ is normally distributed under the approximating model and compute its mean and variance in terms of $(\beta,\mu,\sigma_\varepsilon,w)$.
2. Use lognormal moments to derive expressions for $E[m]$ and $\sigma(m)/E[m]$.
3. Use the parameter mapping $\theta = [(1-\beta)(\gamma-1)]^{-1}$ and the associated $w$ to obtain closed-form expressions for the random-walk model.
4. Explain why $E[m]$ stays roughly constant while $\sigma(m)/E[m]$ grows linearly with $\gamma$.
```

```{solution-start} dov_ex2
:class: dropdown
```

Under the random walk,

$$
c_{t+1}-c_t=\mu+\sigma_\varepsilon \varepsilon_{t+1}

$$
with $\varepsilon_{t+1}\sim\mathcal{N}(0,1)$ under the approximating model.

Using {eq}`bhs_sdf` and the Gaussian distortion

$$
\hat g_{t+1}=\exp \left(w\varepsilon_{t+1}-\tfrac{1}{2}w^2\right),

$$
we get

$$
m_{t+1}
=
\beta \exp \left(-(c_{t+1}-c_t)\right)\hat g_{t+1}
=
\beta \exp \left(-\mu-\sigma_\varepsilon\varepsilon_{t+1}\right)\exp \left(w\varepsilon_{t+1}-\frac{1}{2}w^2\right).
$$

Therefore

$$
\log m_{t+1}
=
\log\beta-\mu-\frac{1}{2}w^2 + (w-\sigma_\varepsilon)\varepsilon_{t+1},

$$
which is normal with mean

$$
E[\log m]=\log\beta-\mu-\tfrac{1}{2}w^2

$$
and variance

$$
\operatorname{Var}(\log m)=(w-\sigma_\varepsilon)^2.
$$

For a lognormal random variable,

$$
E[m]=\exp(E[\log m]+\tfrac{1}{2}\operatorname{Var}(\log m))

$$
and

$$
\sigma(m)/E[m]=\sqrt{e^{\operatorname{Var}(\log m)}-1}.

$$
Hence

$$
E[m]
=
\beta\exp\left(
-\mu-\frac{1}{2}w^2+\frac{1}{2}(w-\sigma_\varepsilon)^2
\right)
=
\beta\exp\left(-\mu+\frac{\sigma_\varepsilon^2}{2}-\sigma_\varepsilon w\right),

$$
and

$$
\frac{\sigma(m)}{E[m]}
=
\sqrt{\exp\left((w-\sigma_\varepsilon)^2\right)-1}.
$$

Now use $w_{\text{RW}}(\theta)=-\sigma_\varepsilon/[(1-\beta)\theta]$ from {eq}`bhs_w_formulas` and
$\theta=[(1-\beta)(\gamma-1)]^{-1}$ to get $w=-\sigma_\varepsilon(\gamma-1)$.
Then

$$
-\sigma_\varepsilon w=\sigma_\varepsilon^2(\gamma-1)

$$
and

$$
(w-\sigma_\varepsilon)^2 = (-\sigma_\varepsilon\gamma)^2=\sigma_\varepsilon^2\gamma^2.

$$
Substituting gives the closed-form expressions for the random-walk model:

```{math}
:label: bhs_Em_rw
E[m] = \beta \exp\left[-\mu + \frac{\sigma_\varepsilon^2}{2}(2\gamma - 1)\right],
```

```{math}
:label: bhs_sigma_rw
\frac{\sigma(m)}{E[m]} = \sqrt{\exp\left(\sigma_\varepsilon^2 \gamma^2\right) - 1}.
```

Notice that in {eq}`bhs_Em_rw`, because $\sigma_\varepsilon$ is small ($\approx 0.005$), the term $\frac{\sigma_\varepsilon^2}{2}(2\gamma-1)$ grows slowly with $\gamma$, keeping $E[m]$ roughly constant near $1/(1+r^f)$.

Meanwhile {eq}`bhs_sigma_rw` shows that $\sigma(m)/E[m] \approx \sigma_\varepsilon \gamma$ grows linearly with $\gamma$.

This is how Epstein--Zin preferences push volatility toward the HJ bound without distorting the risk-free rate.

An analogous calculation for the trend-stationary model yields:

```{math}
:label: bhs_Em_ts
E[m] = \beta \exp\left[-\mu + \frac{\sigma_\varepsilon^2}{2}\left(1 - \frac{2(1-\beta)(1-\gamma)}{1-\beta\rho} + \frac{1-\rho}{1+\rho}\right)\right],
```

```{math}
:label: bhs_sigma_ts
\frac{\sigma(m)}{E[m]} = \sqrt{\exp\left[\sigma_\varepsilon^2\left(\left(\frac{(1-\beta)(1-\gamma)}{1-\beta\rho} - 1\right)^{2} + \frac{1-\rho}{1+\rho}\right)\right] - 1}.
```

```{solution-end}
```

```{exercise}
:label: dov_ex3

Starting from the type I recursion {eq}`bhs_type1_recursion` and the definitions of $U_t$ and $\theta$ in {eq}`bhs_Ut_def`--{eq}`bhs_theta_def`, derive the risk-sensitive recursion {eq}`bhs_risk_sensitive`.

Verify that as $\gamma \to 1$ (equivalently $\theta \to \infty$), the recursion converges to standard discounted expected log utility $U_t = c_t + \beta E_t U_{t+1}$.
```

```{solution-start} dov_ex3
:class: dropdown
```

Start from the type I recursion {eq}`bhs_type1_recursion` and write

$$
(V_{t+1})^{1-\gamma} = \exp\bigl((1-\gamma)\log V_{t+1}\bigr).
$$

Using $\log V_t = (1-\beta)U_t$ from {eq}`bhs_Ut_def`, we obtain

$$
(1-\beta)U_t
=
(1-\beta)c_t
+
\frac{\beta}{1-\gamma}\log E_t\left[\exp\bigl((1-\gamma)(1-\beta)U_{t+1}\bigr)\right].
$$

Divide by $(1-\beta)$ and use {eq}`bhs_theta_def`,

$$
\theta = -\bigl[(1-\beta)(1-\gamma)\bigr]^{-1}.
$$

Then $(1-\gamma)(1-\beta)=-1/\theta$ and $\beta/[(1-\beta)(1-\gamma)]=-\beta\theta$, so

$$
U_t
=
c_t - \beta\theta \log E_t \left[\exp \left(-\frac{U_{t+1}}{\theta}\right)\right],
$$

which is {eq}`bhs_risk_sensitive`.

For $\theta\to\infty$ (equivalently $\gamma\to 1$), use the expansion

$$
\exp(-U_{t+1}/\theta)=1-U_{t+1}/\theta+o(1/\theta).
$$

Taking expectations,

$$
E_t[\exp(-U_{t+1}/\theta)] = 1 - E_t[U_{t+1}]/\theta + o(1/\theta).
$$

Applying $\log(1+x) = x + o(x)$ with $x = -E_t[U_{t+1}]/\theta + o(1/\theta)$,

$$
\log E_t[\exp(-U_{t+1}/\theta)]
=
-E_t[U_{t+1}]/\theta + o(1/\theta),
$$

so $-\theta\log E_t[\exp(-U_{t+1}/\theta)] \to E_t[U_{t+1}]$ as 
$\theta\to\infty$ and the recursion converges to

$$
U_t = c_t + \beta E_t U_{t+1}.
$$

```{solution-end}
```

```{exercise}
:label: dov_ex4

Consider the type II Bellman equation {eq}`bhs_bellman_type2`.

1. Use a Lagrange multiplier to impose the normalization constraint $\int g(\varepsilon) \pi(\varepsilon) d\varepsilon = 1$.
2. Derive the first-order condition for $g(\varepsilon)$ and show that the minimizer is the exponential tilt in {eq}`bhs_ghat`.
3. Substitute your minimizing $g$ back into {eq}`bhs_bellman_type2` to recover the risk-sensitive Bellman equation {eq}`bhs_bellman_type1`.

Conclude that $W(x) \equiv U(x)$ for consumption plans in $\mathcal{C}(A,B,H;x_0)$.
```

```{solution-start} dov_ex4
:class: dropdown
```

Fix $x$ and write $W'(\varepsilon) := W(Ax + B\varepsilon)$ for short.

Form the Lagrangian

$$
\mathcal{L}[g,\lambda]
=
\beta \int \Bigl[g(\varepsilon)W'(\varepsilon) + \theta g(\varepsilon)\log g(\varepsilon)\Bigr]\pi(\varepsilon)d\varepsilon
+
\lambda\left(\int g(\varepsilon)\pi(\varepsilon) d\varepsilon - 1\right).
$$

The pointwise first-order condition for $g(\varepsilon)$ is

$$
0
=
\frac{\partial \mathcal{L}}{\partial g(\varepsilon)}
=
\beta\Bigl[W'(\varepsilon) + \theta(1+\log g(\varepsilon))\Bigr]\pi(\varepsilon)
+
\lambda\pi(\varepsilon),
$$

so (dividing by $\beta\pi(\varepsilon)$)

$$
\log g(\varepsilon)
=
-\frac{W'(\varepsilon)}{\theta} - 1 - \frac{\lambda}{\beta\theta}.
$$

Exponentiating yields $g(\varepsilon)=K\exp(-W'(\varepsilon)/\theta)$ where $K = \exp(-1 - \lambda/(\beta\theta))$ is a constant that does not depend on $\varepsilon$.

To pin down $K$, impose the normalization $\int g(\varepsilon)\pi(\varepsilon)d\varepsilon=1$:

$$
1 = K \int \exp \left(-\frac{W(Ax+B\varepsilon)}{\theta}\right)\pi(\varepsilon) d\varepsilon,
$$

so

$$
K^{-1}
=
\int \exp\left(-\frac{W(Ax+B\varepsilon)}{\theta}\right)\pi(\varepsilon) d\varepsilon.
$$

Substituting $K^{-1}$ into the denominator of $g = K\exp(-W'/\theta)$ gives the minimizer:

$$
g^*(\varepsilon)
=
\frac{\exp\left(-W(Ax+B\varepsilon)/\theta\right)}{
    \int \exp\left(-W(Ax+B\tilde\varepsilon)/\theta\right)\pi(\tilde\varepsilon) d\tilde\varepsilon}.
$$

This has exactly the same form as the distortion $\hat g_{t+1} = \exp(-U_{t+1}/\theta)/E_t[\exp(-U_{t+1}/\theta)]$ that appears in the type I SDF {eq}`bhs_sdf_Ut`, with $W$ in place of $U$.

Once we verify below that $W \equiv U$, the minimizer $g^*$ and the SDF distortion $\hat g$ coincide, which is {eq}`bhs_ghat`.

To substitute back, define

$$
Z(x):=\int \exp(-W(Ax+B\varepsilon)/\theta)\pi(\varepsilon) d\varepsilon.
$$

Then $\hat g(\varepsilon)=\exp(-W(Ax+B\varepsilon)/\theta)/Z(x)$ and

$$
\log\hat g(\varepsilon)=-W(Ax+B\varepsilon)/\theta-\log Z(x).
$$

Hence

$$
\int \Bigl[\hat g(\varepsilon)W(Ax+B\varepsilon) + \theta \hat g(\varepsilon)\log \hat g(\varepsilon)\Bigr]\pi(\varepsilon) d\varepsilon
=
-\theta\log Z(x),
$$

because the $W$ terms cancel and $\int \hat g \pi = 1$.

Plugging this into {eq}`bhs_bellman_type2` gives

$$
W(x)
=
c-\beta\theta\log Z(x)
=
c-\beta\theta \log \int \exp\left(-\frac{W(Ax+B\varepsilon)}{\theta}\right)\pi(\varepsilon) d\varepsilon,
$$

which is {eq}`bhs_bellman_type1`. Therefore $W(x)\equiv U(x)$.

```{solution-end}
```

```{exercise}
:label: dov_ex5

Let $\varepsilon \sim \mathcal{N}(0,1)$ under the approximating model and define

$$
\hat g(\varepsilon) = \exp\left(w\varepsilon - \frac{1}{2}w^2\right)
$$

as in the Gaussian mean-shift section.

1. Show that $E[\hat g(\varepsilon)] = 1$.

2. Show that for any bounded measurable function $f$,

$$
E[\hat g(\varepsilon) f(\varepsilon)]
$$

equals the expectation of $f$ under $\mathcal{N}(w,1)$.

3. Compute the mean and variance of $\log \hat g(\varepsilon)$ and use these to derive

$$
\operatorname{std}(\hat g) = \sqrt{e^{w^2}-1}.
$$

4. Compute the conditional relative entropy $E[\hat g\log \hat g]$ and verify that it equals $\tfrac{1}{2}w^2$.
```

```{solution-start} dov_ex5
:class: dropdown
```

1. Using the moment generating function of a standard normal,

$$
E[\hat g(\varepsilon)]
=
e^{-w^2/2}E[e^{w\varepsilon}]
=
e^{-w^2/2}e^{w^2/2}
=
1.
$$

2. Let $\varphi(\varepsilon) = (2\pi)^{-1/2}e^{-\varepsilon^2/2}$ be the $\mathcal{N}(0,1)$ density.

Then

$$
\hat g(\varepsilon)\varphi(\varepsilon)
=
\frac{1}{\sqrt{2\pi}}
\exp\left(w\varepsilon-\frac{1}{2}w^2-\frac{1}{2}\varepsilon^2\right)
=
\frac{1}{\sqrt{2\pi}}
\exp\left(-\frac{1}{2}(\varepsilon-w)^2\right),
$$

which is the $\mathcal{N}(w,1)$ density.

Therefore, for bounded measurable $f$,

$$
E[\hat g(\varepsilon)f(\varepsilon)]
=
\int f(\varepsilon)\hat g(\varepsilon)\varphi(\varepsilon)d\varepsilon
$$

equals the expectation of $f$ under $\mathcal{N}(w,1)$.

3. Since $\log \hat g(\varepsilon) = w\varepsilon - \tfrac{1}{2}w^2$ and $\varepsilon\sim\mathcal{N}(0,1)$,

$$
E[\log \hat g] = -\frac{1}{2}w^2,
\qquad
\operatorname{Var}(\log \hat g)=w^2.

$$
Moreover, $\operatorname{Var}(\hat g)=E[\hat g^2]-1$ because $E[\hat g]=1$.

Now

$$
E[\hat g^2]
=
E\left[\exp\left(2w\varepsilon - w^2\right)\right]
=
e^{-w^2}E[e^{2w\varepsilon}]
=
e^{-w^2}e^{(2w)^2/2}
=
e^{w^2},

$$
so $\operatorname{std}(\hat g)=\sqrt{e^{w^2}-1}$.

4. Using part 2 with $f(\varepsilon)=\log \hat g(\varepsilon)=w\varepsilon-\tfrac{1}{2}w^2$,

$$
E[\hat g\log \hat g]
=
E_{\mathcal{N}(w,1)}\left[w\varepsilon-\frac{1}{2}w^2\right]
=
w\cdot E_{\mathcal{N}(w,1)}[\varepsilon]-\frac{1}{2}w^2
=
w^2-\frac{1}{2}w^2
=
\frac{1}{2}w^2.
$$

```{solution-end}
```

```{exercise}
:label: dov_ex6

In the Gaussian mean-shift setting of {ref}`Exercise 5 <dov_ex5>`, let $L_T$ be the log likelihood ratio between the worst-case and approximating models based on $T$ observations.

1. Show that $L_T$ is normal under each model.
2. Compute its mean and variance under the approximating and worst-case models.
3. Using the definition of detection-error probability in {eq}`bhs_detection_formula`, derive the closed-form expression {eq}`bhs_detection_closed`.
```

```{solution-start} dov_ex6
:class: dropdown
```

Let the approximating model be $\varepsilon_i \sim \mathcal{N}(0,1)$ and the worst-case model be $\varepsilon_i \sim \mathcal{N}(w,1)$, i.i.d. for $i=1,\ldots,T$.

Take the log likelihood ratio in the direction that matches the definitions in the text:

$$
L_T
=
\log \frac{\prod_{i=1}^T \varphi(\varepsilon_i)}{\prod_{i=1}^T \varphi(\varepsilon_i-w)}
=
\sum_{i=1}^T \ell(\varepsilon_i),

$$
where $\varphi$ is the $\mathcal{N}(0,1)$ density and

$$
\ell(\varepsilon)
=
\log \varphi(\varepsilon) - \log \varphi(\varepsilon-w)
=
-\frac{1}{2}\Bigl[\varepsilon^2-(\varepsilon-w)^2\Bigr]
=
-w\varepsilon + \frac{1}{2}w^2.
$$

Therefore

$$
L_T = -w\sum_{i=1}^T \varepsilon_i + \tfrac{1}{2}w^2T.
$$

Under the approximating model, $\sum_{i=1}^T \varepsilon_i \sim \mathcal{N}(0,T)$, so

$$
L_T \sim \mathcal{N}\left(\frac{1}{2}w^2T, w^2T\right).
$$

Under the worst-case model, $\sum_{i=1}^T \varepsilon_i \sim \mathcal{N}(wT,T)$, so

$$
L_T \sim \mathcal{N}\left(-\frac{1}{2}w^2T, w^2T\right).
$$

Now

$$
p_A = \Pr_A(L_T<0)
=
\Phi\left(\frac{0-\frac{1}{2}w^2T}{|w|\sqrt{T}}\right)
=
\Phi\left(-\frac{|w|\sqrt{T}}{2}\right),
$$

and

$$
p_B = \Pr_B(L_T>0)
=
1-\Phi\left(\frac{0-(-\frac{1}{2}w^2T)}{|w|\sqrt{T}}\right)
=
1-\Phi\left(\frac{|w|\sqrt{T}}{2}\right)
=
\Phi\left(-\frac{|w|\sqrt{T}}{2}\right).
$$

Therefore

$$
p(\theta^{-1})=\tfrac{1}{2}(p_A+p_B)=\Phi\left(-\tfrac{|w|\sqrt{T}}{2}\right),

$$
which is {eq}`bhs_detection_closed`.

```{solution-end}
```

```{exercise}
:label: dov_ex7

Using the formulas for $w(\theta)$ in {eq}`bhs_w_formulas` and the definition of discounted entropy

$$
\eta = \frac{\beta}{1-\beta}\cdot \frac{w(\theta)^2}{2},
$$

show that holding $\eta$ fixed across the random-walk and trend-stationary consumption specifications implies the mapping {eq}`bhs_theta_cross_model`.

Specialize your result to the case $\sigma_\varepsilon^{\text{TS}} = \sigma_\varepsilon^{\text{RW}}$ and interpret the role of $\rho$.
```

```{solution-start} dov_ex7
:class: dropdown
```

Because $\eta$ depends on $\theta$ only through $w(\theta)^2$, holding $\eta$ fixed across models is equivalent to holding $|w(\theta)|$ fixed.

Using {eq}`bhs_w_formulas`,

$$
|w_{\text{RW}}(\theta_{\text{RW}})|
=
\frac{\sigma_\varepsilon^{\text{RW}}}{(1-\beta)\theta_{\text{RW}}},
\qquad
|w_{\text{TS}}(\theta_{\text{TS}})|
=
\frac{\sigma_\varepsilon^{\text{TS}}}{(1-\beta\rho)\theta_{\text{TS}}}.
$$

Equating these magnitudes and solving for $\theta_{\text{TS}}$ gives

$$
\theta_{\text{TS}}
=
\left(\frac{\sigma_\varepsilon^{\text{TS}}}{\sigma_\varepsilon^{\text{RW}}}\right)
\frac{1-\beta}{1-\beta\rho}\theta_{\text{RW}},
$$

which is {eq}`bhs_theta_cross_model`.

If $\sigma_\varepsilon^{\text{TS}}=\sigma_\varepsilon^{\text{RW}}$, then

$$
\theta_{\text{TS}}=\frac{1-\beta}{1-\beta\rho}\theta_{\text{RW}}.
$$

Since $\rho\in(0,1)$ implies $1-\beta\rho > 1-\beta$, the ratio $(1-\beta)/(1-\beta\rho)$ is less than one.

To hold entropy fixed, the trend-stationary model therefore requires a smaller $\theta$ (i.e., a cheaper distortion and stronger robustness) than the random-walk model.

```{solution-end}
```

```{exercise}
:label: dov_ex8

For type II (multiplier) preferences under random-walk consumption growth, derive the compensating-variation formulas in {eq}`bhs_type2_rw_decomp`.

In particular, derive

1. the **risk** term by comparing the stochastic economy to a deterministic consumption path with the same mean level of consumption (Lucas's thought experiment), and
2. the **uncertainty** term by comparing a type II agent with parameter $\theta$ to the expected-utility case $\theta=\infty$, holding the stochastic environment fixed.
```

```{solution-start} dov_ex8
:class: dropdown
```

Write the random walk as

$$
c_t = c_0 + t\mu + \sigma_\varepsilon\sum_{j=1}^t \varepsilon_j
$$

with $\varepsilon_j\stackrel{iid}{\sim}\mathcal{N}(0,1)$.

**Risk term.**
The mean level of consumption is

$$
E[C_t]=E[e^{c_t}]=\exp(c_0+t\mu+\tfrac{1}{2}t\sigma_\varepsilon^2),
$$

so the deterministic path with the same mean levels is

$$
\bar c_t = c_0 + t(\mu+\tfrac{1}{2}\sigma_\varepsilon^2).
$$

Under expected log utility ($\theta=\infty$), discounted expected utility is

$$
\sum_{t\geq 0}\beta^t E[c_t]
=
\frac{c_0}{1-\beta} + \frac{\beta\mu}{(1-\beta)^2},

$$
while for the deterministic mean-level path it is

$$
\sum_{t\geq 0}\beta^t \bar c_t
=
\frac{c_0}{1-\beta} + \frac{\beta(\mu+\tfrac{1}{2}\sigma_\varepsilon^2)}{(1-\beta)^2}.
$$

If we reduce initial consumption by $\Delta c_0^{risk}$ (so $\bar c_t$ shifts down by $\Delta c_0^{risk}$ for all $t$), utility falls by $\Delta c_0^{risk}/(1-\beta)$.
Equating the two utilities gives

$$
\frac{\Delta c_0^{risk}}{1-\beta}
=
\frac{\beta(\tfrac{1}{2}\sigma_\varepsilon^2)}{(1-\beta)^2}
\quad\Rightarrow\quad
\Delta c_0^{risk}=\frac{\beta\sigma_\varepsilon^2}{2(1-\beta)}.
$$

**Uncertainty term.**
For type II multiplier preferences, the minimizing distortion is a Gaussian mean shift with parameter $w$ and per-period relative entropy $\tfrac{1}{2}w^2$.
Under the distorted model, $E[\varepsilon]=w$, so

$$
E[c_t]=c_0+t(\mu+\sigma_\varepsilon w).
$$

Plugging this into the type II objective (and using $E_t[g\log g]=\tfrac{1}{2}w^2$) gives the discounted objective as a function of $w$:

$$
J(w)
=
\sum_{t\geq 0}\beta^t\Bigl(c_0+t(\mu+\sigma_\varepsilon w)\Bigr)
+
\sum_{t\geq 0}\beta^{t+1}\theta\cdot\frac{w^2}{2}.
$$

Using $\sum_{t\ge0}\beta^t=1/(1-\beta)$ and $\sum_{t\ge0}t\beta^t=\beta/(1-\beta)^2$,

$$
J(w)
=
\frac{c_0}{1-\beta}
+
\frac{\beta(\mu+\sigma_\varepsilon w)}{(1-\beta)^2}
+
\frac{\beta\theta}{1-\beta}\cdot\frac{w^2}{2}.
$$

Minimizing over $w$ yields

$$
0=\frac{\partial J}{\partial w}
=
\frac{\beta\sigma_\varepsilon}{(1-\beta)^2}
+
\frac{\beta\theta}{1-\beta}w
\quad\Rightarrow\quad
w^*=-\frac{\sigma_\varepsilon}{(1-\beta)\theta},
$$

which matches {eq}`bhs_w_formulas`.

Substituting $w^*$ back in gives

$$
J(w^*)
=
\frac{c_0}{1-\beta}
+
\frac{\beta\mu}{(1-\beta)^2}
-\frac{\beta\sigma_\varepsilon^2}{2(1-\beta)^3\theta}.
$$

When $\theta=\infty$ (no model uncertainty), the last term disappears.
Thus the utility gain from removing model uncertainty at fixed $(\mu,\sigma_\varepsilon)$ is

$$
\beta\sigma_\varepsilon^2/[2(1-\beta)^3\theta].
$$

To offset this by a permanent upward shift in initial log consumption, we need

$$
\Delta c_0^{uncertainty}/(1-\beta)=\beta\sigma_\varepsilon^2/[2(1-\beta)^3\theta],
$$

so

$$
\Delta c_0^{uncertainty}
=
\frac{\beta\sigma_\varepsilon^2}{2(1-\beta)^2\theta}.
$$

Together these reproduce {eq}`bhs_type2_rw_decomp`.

```{solution-end}
```

```{exercise}
:label: dov_ex9

Verify the closed-form value function {eq}`bhs_W_rw` for the random-walk model by substituting a guess of the form $W(x_t) = \frac{1}{1-\beta}[c_t + d]$ into the risk-sensitive Bellman equation {eq}`bhs_bellman_type1`.

1. Under the random walk $c_{t+1} = c_t + \mu + \sigma_\varepsilon \varepsilon_{t+1}$, show that $W(Ax_t + B\varepsilon) = \frac{1}{1-\beta}[c_t + \mu + \sigma_\varepsilon\varepsilon_{t+1} + d]$.
2. Substitute into the $\log E\exp$ term, using the fact that for $Z \sim \mathcal{N}(\mu_Z, \sigma_Z^2)$ we have $\log E[\exp(Z)] = \mu_Z + \frac{1}{2}\sigma_Z^2$.
3. Solve for $d$ and confirm that it matches {eq}`bhs_W_rw`.
```

```{solution-start} dov_ex9
:class: dropdown
```

**Part 1.** Under the random walk, $c_{t+1} = c_t + \mu + \sigma_\varepsilon\varepsilon_{t+1}$. Substituting the guess $W(x) = \frac{1}{1-\beta}[Hx + d]$ with $Hx_t = c_t$:

$$
W(Ax_t + B\varepsilon_{t+1}) = \frac{1}{1-\beta}\bigl[c_t + \mu + \sigma_\varepsilon\varepsilon_{t+1} + d\bigr].
$$

**Part 2.** The Bellman equation {eq}`bhs_bellman_type1` requires computing

$$
-\beta\theta\log E_t\left[\exp\left(\frac{-W(Ax_t + B\varepsilon_{t+1})}{\theta}\right)\right].
$$

Substituting the guess:

$$
\frac{-W(Ax_t + B\varepsilon_{t+1})}{\theta}
=
\frac{-1}{(1-\beta)\theta}\bigl[c_t + \mu + d + \sigma_\varepsilon\varepsilon_{t+1}\bigr].
$$

This is an affine function of the standard normal $\varepsilon_{t+1}$, so the argument of the $\log E\exp$ is normal with

$$
\mu_Z = \frac{-(c_t + \mu + d)}{(1-\beta)\theta},
\qquad
\sigma_Z^2 = \frac{\sigma_\varepsilon^2}{(1-\beta)^2\theta^2}.
$$

Using $\log E[e^Z] = \mu_Z + \frac{1}{2}\sigma_Z^2$:

$$
-\beta\theta\left[\frac{-(c_t + \mu + d)}{(1-\beta)\theta} + \frac{\sigma_\varepsilon^2}{2(1-\beta)^2\theta^2}\right]
=
\frac{\beta}{1-\beta}\left[c_t + \mu + d - \frac{\sigma_\varepsilon^2}{2(1-\beta)\theta}\right].
$$

**Part 3.** The Bellman equation becomes

$$
\frac{1}{1-\beta}[c_t + d]
=
c_t + \frac{\beta}{1-\beta}\left[c_t + \mu + d - \frac{\sigma_\varepsilon^2}{2(1-\beta)\theta}\right].
$$

Expanding the right-hand side:

$$
c_t + \frac{\beta c_t}{1-\beta} + \frac{\beta(\mu + d)}{1-\beta} - \frac{\beta\sigma_\varepsilon^2}{2(1-\beta)^2\theta}
=
\frac{c_t}{1-\beta} + \frac{\beta(\mu + d)}{1-\beta} - \frac{\beta\sigma_\varepsilon^2}{2(1-\beta)^2\theta}.
$$

Equating both sides and cancelling $\frac{c_t}{1-\beta}$:

$$
\frac{d}{1-\beta} = \frac{\beta(\mu + d)}{1-\beta} - \frac{\beta\sigma_\varepsilon^2}{2(1-\beta)^2\theta}.
$$

Solving: $d - \beta d = \beta\mu - \frac{\beta\sigma_\varepsilon^2}{2(1-\beta)\theta}$, so

$$
d = \frac{\beta}{1-\beta}\left(\mu - \frac{\sigma_\varepsilon^2}{2(1-\beta)\theta}\right),
$$

which matches {eq}`bhs_W_rw`.

```{solution-end}
```

```{exercise}
:label: dov_ex10

Derive the trend-stationary risk compensation stated in the lecture.

For the trend-stationary model with $\tilde c_{t+1} - \zeta = \rho(\tilde c_t - \zeta) + \sigma_\varepsilon\varepsilon_{t+1}$, where $\tilde c_t = c_t - \mu t$, compute the risk compensation $\Delta c_0^{risk,ts}$ by comparing expected log utility under the stochastic plan to the deterministic certainty-equivalent path, and show that

$$
\Delta c_0^{risk,ts} = \frac{\beta\sigma_\varepsilon^2}{2(1-\beta\rho^2)}.
$$

*Hint:* You will need $\operatorname{Var}(z_t) = \sigma_\varepsilon^2(1 + \rho^2 + \cdots + \rho^{2(t-1)})$ and the formula $\sum_{t \geq 1}\beta^t \sum_{j=0}^{t-1}\rho^{2j} = \frac{\beta}{(1-\beta)(1-\beta\rho^2)}$.
```

```{solution-start} dov_ex10
:class: dropdown
```

Under the trend-stationary model with $z_0 = 0$, $c_t = c_0 + \mu t + z_t$ and $E[c_t] = c_0 + \mu t$ (since $E[z_t] = 0$).

The deterministic certainty-equivalent path matches $E[C_t] = \exp(c_0 + \mu t + \frac{1}{2}\operatorname{Var}(z_t))$, so its log is $c_0^{ce} + \mu t + \frac{1}{2}\operatorname{Var}(z_t)$.

Under expected log utility ($\theta = \infty$), the value of the stochastic plan is

$$
\sum_{t \geq 0}\beta^t E[c_t] = \frac{c_0}{1-\beta} + \frac{\beta\mu}{(1-\beta)^2}.
$$

The value of the certainty-equivalent path (matching mean levels) starting from $c_0 - \Delta c_0^{risk}$ is

$$
\sum_{t \geq 0}\beta^t \bigl[c_0 - \Delta c_0^{risk} + \mu t + \tfrac{1}{2}\operatorname{Var}(z_t)\bigr].
$$

Since $\operatorname{Var}(z_t) = \sigma_\varepsilon^2 \sum_{j=0}^{t-1}\rho^{2j}$, the extra term sums to

$$
\sum_{t \geq 1}\beta^t \cdot \frac{\sigma_\varepsilon^2}{2}\sum_{j=0}^{t-1}\rho^{2j}
= \frac{\sigma_\varepsilon^2}{2}\cdot\frac{\beta}{(1-\beta)(1-\beta\rho^2)}.
$$

Equating values and solving:

$$
\frac{\Delta c_0^{risk}}{1-\beta} = \frac{\beta\sigma_\varepsilon^2}{2(1-\beta)(1-\beta\rho^2)}
\quad\Rightarrow\quad
\Delta c_0^{risk,ts} = \frac{\beta\sigma_\varepsilon^2}{2(1-\beta\rho^2)}.
$$

The uncertainty compensation follows from the value function: $\Delta c_0^{unc,ts,II} = \frac{\beta\sigma_\varepsilon^2}{2(1-\beta\rho)^2\theta}$, with the $(1-\beta)$ factors replaced by $(1-\beta\rho)$ because the worst-case mean shift scales with $1/(1-\beta\rho)$ rather than $1/(1-\beta)$.

```{solution-end}
```

```{exercise}
:label: dov_ex11

Derive the worst-case mean shifts {eq}`bhs_w_formulas` for both consumption models.

Recall that the worst-case distortion {eq}`bhs_ghat` has $\hat g \propto \exp(-W(x_{t+1})/\theta)$.

When $W$ is linear in the state, the exponent is linear in $\varepsilon_{t+1}$, and the Gaussian mean shift is $w = -\lambda/\theta$ where $\lambda$ is the coefficient on $\varepsilon_{t+1}$ in $W(x_{t+1})$.

1. Random-walk model: Guess $W(x_t) = \frac{1}{1-\beta}[c_t + d]$. Using $c_{t+1} = c_t + \mu + \sigma_\varepsilon\varepsilon_{t+1}$, find $\lambda$ and show that $w = -\sigma_\varepsilon/[(1-\beta)\theta]$.

2. Trend-stationary model: Write $z_t = \tilde c_t - \zeta$ and guess $W(x_t) = \frac{1}{1-\beta}[c_t + \alpha_1 z_t + \alpha_0]$. Show that:
   - The coefficient on $\varepsilon_{t+1}$ in $W(x_{t+1})$ is $(1+\alpha_1)\sigma_\varepsilon/(1-\beta)$.
   - Matching coefficients on $z_t$ in the Bellman equation gives $\alpha_1 = \beta(\rho-1)/(1-\beta\rho)$.
   - Therefore $1+\alpha_1 = (1-\beta)/(1-\beta\rho)$ and $w = -\sigma_\varepsilon/[(1-\beta\rho)\theta]$.
```

```{solution-start} dov_ex11
:class: dropdown
```

**Part 1.**
Under the guess $W(x_t) = \frac{1}{1-\beta}[c_t + d]$ and $c_{t+1} = c_t + \mu + \sigma_\varepsilon\varepsilon_{t+1}$,

$$
W(x_{t+1}) = \frac{1}{1-\beta}[c_t + \mu + \sigma_\varepsilon\varepsilon_{t+1} + d].
$$

The coefficient on $\varepsilon_{t+1}$ is $\lambda = \sigma_\varepsilon/(1-\beta)$, so $w = -\lambda/\theta = -\sigma_\varepsilon/[(1-\beta)\theta]$.

**Part 2.**
Under the guess $W(x_t) = \frac{1}{1-\beta}[c_t + \alpha_1 z_t + \alpha_0]$ with $c_{t+1} = c_t + \mu + (\rho-1)z_t + \sigma_\varepsilon\varepsilon_{t+1}$ and $z_{t+1} = \rho z_t + \sigma_\varepsilon\varepsilon_{t+1}$,

$$
W(x_{t+1}) = \tfrac{1}{1-\beta}\bigl[c_t + \mu + (\rho-1)z_t + \sigma_\varepsilon\varepsilon_{t+1} + \alpha_1(\rho z_t + \sigma_\varepsilon\varepsilon_{t+1}) + \alpha_0\bigr].
$$

The coefficient on $\varepsilon_{t+1}$ is $(1+\alpha_1)\sigma_\varepsilon/(1-\beta)$.

To find $\alpha_1$, substitute the guess into the Bellman equation.

The factors of $\frac{1}{1-\beta}$ cancel on both sides, and matching coefficients on $z_t$ gives

$$
\alpha_1 = \beta\bigl[(\rho-1) + \alpha_1\rho\bigr]
\quad\Rightarrow\quad
\alpha_1(1-\beta\rho) = \beta(\rho-1)
\quad\Rightarrow\quad
\alpha_1 = \frac{\beta(\rho-1)}{1-\beta\rho}.
$$

Therefore

$$
1+\alpha_1 = \frac{1-\beta\rho + \beta(\rho-1)}{1-\beta\rho} = \frac{1-\beta}{1-\beta\rho},
$$

and the coefficient on $\varepsilon_{t+1}$ becomes $(1+\alpha_1)\sigma_\varepsilon/(1-\beta) = \sigma_\varepsilon/(1-\beta\rho)$, giving $w = -\sigma_\varepsilon/[(1-\beta\rho)\theta]$.

```{solution-end}
```
