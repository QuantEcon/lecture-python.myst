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

# Survival and Long-Run Dynamics under Recursive Preferences

```{index} single: Survival; Recursive Preferences
```

## Overview

This lecture describes a theory of **long-run survival** of agents with heterogeneous beliefs
developed by {cite}`Borovicka2020`.

The classical **market selection hypothesis** asserts that agents with incorrect beliefs
will be driven from the market in the long run --- they will lose all of their wealth
to agents with more accurate beliefs.

This result was established rigorously by {cite}`Sandroni2000` and {cite}`Blume_Easley2006`
for economies in which agents have **separable** (CRRA) preferences.

{cite}`Borovicka2020` shows that when agents have **recursive preferences**
of the {cite}`Epstein_Zin1989` type, the market selection hypothesis can fail:
agents with incorrect beliefs can survive and even prosper in the long run.

The key insight is that recursive preferences **disentangle** risk aversion from the
intertemporal elasticity of substitution (IES), and this separation opens new channels
through which agents with incorrect beliefs can accumulate wealth.

Three survival channels emerge:

1. **Risk premium channel**: a more optimistic agent earns a higher expected logarithmic
   return on her portfolio by holding a larger share of risky assets
2. **Speculative volatility channel**: speculative portfolio positions generate volatile
   returns that penalize survival through a Jensen's inequality effect
3. **Saving channel**: under high IES, an agent who believes her portfolio has a high
   expected return responds by saving more, which can help her outsave extinction

Under separable CRRA preferences, only the first two channels operate, and they ensure
that the agent with more accurate beliefs always dominates.
With recursive preferences, the saving channel can tip the balance in favor of an agent
whose beliefs are less accurate.

```{note}
The paper builds on the continuous-time recursive utility formulation of {cite}`Duffie_Epstein1992a`,
using the planner's problem approach of {cite}`Dumas_Uppal_Wang2000`.
Important foundations for the market selection hypothesis were laid by
{cite}`DeLong_etal1991` and {cite}`Blume_Easley1992`.
```

Let's start with some imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import brentq
```

## Environment

The economy is populated by two types of infinitely lived agents, $n \in \{1, 2\}$,
who have identical recursive preferences but **differ in their beliefs** about the
distribution of future aggregate endowment.

### Aggregate endowment

Aggregate endowment $Y$ follows a geometric Brownian motion under the true probability
measure $P$:

$$
d \log Y_t = \mu_Y dt + \sigma_Y dW_t, \quad Y_0 > 0
$$ (eq:endowment)

where $W$ is a standard Brownian motion, $\mu_Y$ is the drift, and $\sigma_Y > 0$ is the
volatility.

### Heterogeneous beliefs

Agent $n$ perceives the drift of aggregate endowment to be $\mu_Y + \omega^n \sigma_Y$ instead
of $\mu_Y$.

The parameter $\omega^n$ captures the degree of **optimism** ($\omega^n > 0$) or **pessimism**
($\omega^n < 0$) of agent $n$.

Formally, agent $n$'s subjective probability measure $Q^n$ is defined by the
Radon-Nikodým derivative

$$
M_t^n = \frac{dQ^n}{dP}\bigg|_t = \exp\left(-\frac{1}{2} |\omega^n|^2 t + \omega^n W_t\right)
$$ (eq:radon_nikodym)

Under her own measure $Q^n$, agent $n$ believes that $W_t^n = W_t - \omega^n t$ is a
Brownian motion, so that

$$
d \log Y_t = (\mu_Y + \omega^n \sigma_Y) dt + \sigma_Y dW_t^n
$$

Agent $n$ with $\omega^n > 0$ is **optimistic** about the growth rate of aggregate endowment;
agent $n$ with $\omega^n < 0$ is **pessimistic**.

### Recursive preferences

Both agents have Duffie-Epstein-Zin recursive preferences characterized by three
parameters:

* $\gamma > 0$: coefficient of relative risk aversion (CRRA)
* $\rho^{-1} > 0$: intertemporal elasticity of substitution (IES)
* $\beta > 0$: time-preference rate

The felicity function for these preferences is

$$
F(C, \nu) = \beta \frac{C^{1-\gamma}}{1-\gamma} \cdot \frac{(1-\gamma) - (1-\rho)\nu / \beta}{\rho - \gamma}
$$ (eq:felicity)

where $\nu$ is the endogenous discount rate.

```{note}
When $\gamma = \rho$, preferences reduce to the standard separable CRRA case.
The disentanglement of risk aversion $\gamma$ from the inverse IES $\rho$ is the key
feature that drives the new survival results.
```

## Planner's Problem

Following {cite}`Dumas_Uppal_Wang2000`, we study optimal allocations using a social
planner who maximizes a weighted average of the two agents' continuation values.

The planner assigns consumption shares $z^1$ and $z^2 = 1 - z^1$ to the two agents
and chooses discount rate processes $\nu^n$ for each agent.

### Modified discount factors

It is convenient to incorporate the belief distortions into modified discount factor
processes $\tilde{\lambda}^n = \lambda^n M^n$, where $\lambda^n$ is the standard discount factor.

The modified discount factor evolves as

$$
d \log \tilde{\lambda}_t^n = -\left(\nu_t^n + \frac{1}{2}(\omega^n)^2\right) dt + \omega^n dW_t
$$ (eq:modified_discount)

### State variable: Pareto share

The key state variable is the **Pareto share** of agent 1:

$$
\upsilon = \frac{\tilde{\lambda}^1}{\tilde{\lambda}^1 + \tilde{\lambda}^2} \in (0, 1)
$$ (eq:pareto_share)

This single scalar captures the relative weight of agent 1 in the planner's allocation.

The dynamics of the log-odds ratio $\vartheta = \log(\upsilon / (1-\upsilon))$ are

$$
d\vartheta_t = \underbrace{\left[\nu_t^2 + \frac{1}{2}(\omega^2)^2 - \nu_t^1 - \frac{1}{2}(\omega^1)^2\right]}_{m_{\vartheta}(\upsilon_t)} dt + (\omega^1 - \omega^2) dW_t
$$ (eq:log_odds)

The drift $m_\vartheta(\upsilon)$ determines the long-run behavior of the Pareto share.

### HJB equation

The planner's value function takes the form
$J(\tilde{\lambda}_t, Y_t) = (\tilde{\lambda}_t^1 + \tilde{\lambda}_t^2) Y_t^{1-\gamma} \tilde{J}(\upsilon_t)$,
where $\tilde{J}(\upsilon)$ solves a nonlinear ODE:

$$
0 = \sup_{(z^1,z^2,\nu^1,\nu^2)} \left\{ \upsilon F(z^1, \nu^1) + (1-\upsilon) F(z^2, \nu^2) + \mathcal{L} \tilde{J}(\upsilon) \right\}
$$ (eq:hjb)

subject to $z^1 + z^2 \leq 1$, where $\mathcal{L}$ is a second-order differential operator
that captures the drift and diffusion of the state variables.

The boundary conditions are $\tilde{J}(0) = V^2$ and $\tilde{J}(1) = V^1$, where $V^n$ is the
value in a homogeneous economy populated only by agent $n$.


## Survival Conditions

The central result of the paper characterizes survival in terms of the boundary behavior
of the drift $m_\vartheta(\upsilon)$.

```{prf:proposition}
:label: survival_conditions

Define the following repelling conditions (i) and (ii) and their attracting
counterparts (i') and (ii'):

$$
\text{(i)} \lim_{\upsilon \searrow 0} m_\vartheta(\upsilon) > 0, \qquad
\text{(i')} \lim_{\upsilon \searrow 0} m_\vartheta(\upsilon) < 0
$$

$$
\text{(ii)} \lim_{\upsilon \nearrow 1} m_\vartheta(\upsilon) < 0, \qquad
\text{(ii')} \lim_{\upsilon \nearrow 1} m_\vartheta(\upsilon) > 0
$$

Then:

**(a)** If (i) and (ii) hold, both agents survive under $P$.

**(b)** If (i) and (ii') hold, agent 1 dominates in the long run under $P$.

**(c)** If (i') and (ii) hold, agent 2 dominates in the long run under $P$.

**(d)** If (i') and (ii') hold, each agent dominates with strictly positive probability.
```

The proof uses the Feller classification of boundary behavior for diffusion processes,
as discussed in {cite}`Karlin_Taylor1981`.

The intuition is straightforward: condition (i) says that when agent 1's share is
nearly zero, there is a force pushing it back up; condition (ii) says that when agent 1's
share is nearly one, there is a force pushing it back down.
When both forces are present, the Pareto share is recurrent and both agents survive.

## Wealth Dynamics Decomposition

The survival conditions can be expressed in terms of equilibrium wealth dynamics.
When agent 1 becomes negligible ($\upsilon \searrow 0$), equilibrium prices converge to
those in a homogeneous economy populated by agent 2.

The difference in logarithmic wealth growth rates decomposes as

$$
\lim_{\upsilon \searrow 0} [m_A^1(\upsilon) - m_A^2(\upsilon)]
= \underbrace{\lim_{\upsilon \searrow 0} [m_R^1(\upsilon) - m_R^2(\upsilon)]}_{\text{portfolio returns}}
+ \underbrace{\lim_{\upsilon \searrow 0} [(y^2(\upsilon))^{-1} - (y^1(\upsilon))^{-1}]}_{\text{consumption rates}}
$$ (eq:wealth_decomp)

### Portfolio returns

The difference in expected logarithmic portfolio returns at the boundary is

$$
\lim_{\upsilon \searrow 0} [m_R^1 - m_R^2] = \underbrace{\frac{\omega^1 - \omega^2}{\gamma} \cdot \sigma_Y}_{\text{difference in portfolios}}
\cdot \underbrace{(\gamma \sigma_Y^2 - \omega^2 \sigma_Y)}_{\text{risk premium}}
- \underbrace{\frac{1}{2}\left(\frac{\omega^1 - \omega^2}{\gamma}\right)^2}_{\text{volatility penalty}}
$$ (eq:portfolio_returns)

This depends **only** on risk aversion $\gamma$, not on the IES.

### Consumption rates

The difference in consumption rates at the boundary is

$$
\lim_{\upsilon \searrow 0} [(y^2)^{-1} - (y^1)^{-1}]
= \frac{1-\rho}{\rho} \left[(\omega^1 - \omega^2)\sigma_Y + \frac{(\omega^1 - \omega^2)^2}{2\gamma}\right]
$$ (eq:consumption_rates)

This depends on $\rho$ (and hence the IES) but enters **only** through the consumption-saving
decision.

```{code-cell} ipython3
def portfolio_return_diff(omega1, omega2, gamma, sigma_y):
    """
    Difference in expected log portfolio returns at boundary v → 0.

    Parameters
    ----------
    omega1 : float
        Belief distortion of agent 1
    omega2 : float
        Belief distortion of agent 2
    gamma : float
        Risk aversion
    sigma_y : float
        Endowment volatility

    Returns
    -------
    float
        Difference in log portfolio returns, decomposed into
        (risk_premium_effect, volatility_penalty)
    """
    delta_omega = omega1 - omega2
    portfolio_diff = delta_omega / gamma
    risk_premium = gamma * sigma_y**2 - omega2 * sigma_y
    risk_premium_effect = portfolio_diff * risk_premium * sigma_y
    # Correct formula from the paper:
    # (ω1-ω2)/γ * σ_y * (γσ_y² - ω²σ_y) - (1/2)((ω1-ω2)/γ + ω1-ω2)²
    # Simplify using Prop 3.4
    diff_portfolios = delta_omega / gamma
    rp = gamma * sigma_y - omega2
    volatility_penalty = 0.5 * (delta_omega * sigma_y / gamma
                                 + delta_omega)**2
    total = diff_portfolios * sigma_y * rp - volatility_penalty
    return total


def consumption_rate_diff(omega1, omega2, gamma, rho, sigma_y):
    """
    Difference in consumption rates at boundary v → 0.

    Parameters
    ----------
    omega1, omega2 : float
        Belief distortions
    gamma : float
        Risk aversion
    rho : float
        Inverse of IES
    sigma_y : float
        Endowment volatility

    Returns
    -------
    float
    """
    delta_omega = omega1 - omega2
    subjective_return_diff = (delta_omega * sigma_y
                               + delta_omega**2 / (2 * gamma))
    return (1 - rho) / rho * subjective_return_diff


def survival_drift(omega1, omega2, gamma, rho, sigma_y):
    """
    Drift m_ϑ at boundary v → 0, determining survival of agent 1.

    Positive drift means agent 1 survives (repelling boundary).

    Parameters
    ----------
    omega1, omega2 : float
        Belief distortions of agents 1 and 2
    gamma : float
        Risk aversion
    rho : float
        Inverse of IES
    sigma_y : float
        Endowment volatility

    Returns
    -------
    float
        Drift at v = 0
    """
    pr = portfolio_return_diff(omega1, omega2, gamma, sigma_y)
    cr = consumption_rate_diff(omega1, omega2, gamma, rho, sigma_y)
    return gamma * (pr + cr)
```

## Survival Regions

A central contribution of {cite}`Borovicka2020` is the characterization of
**survival regions** in the $(\gamma, \rho)$ parameter space.

Under separable CRRA preferences ($\gamma = \rho$), the agent with more accurate beliefs
always dominates --- this is the market selection hypothesis.

Under recursive preferences, all four survival outcomes from {prf:ref}`survival_conditions`
can occur.

Let us compute and plot the survival regions for different levels of belief distortion,
following Figure 2 of {cite}`Borovicka2020`.

We focus on the case where agent 2 has correct beliefs ($\omega^2 = 0$) and agent 1
has distorted beliefs.

```{code-cell} ipython3
def compute_survival_boundary(omega1, omega2, sigma_y,
                               gamma_range, boundary='lower'):
    """
    Compute the boundary curve in (γ, ρ) space where survival
    condition holds with equality.

    For boundary='lower' (v → 0): drift at v=0 = 0, giving
    condition for agent 1's survival.
    For boundary='upper' (v → 1): drift at v=1 = 0, giving
    condition for agent 2's survival (symmetric).

    Returns ρ as function of γ along the boundary.
    """
    rho_boundary = []

    if boundary == 'lower':
        # Agent 1 survival: drift at v→0 = 0
        # portfolio_returns + consumption_rate_diff = 0
        for gamma in gamma_range:
            pr = portfolio_return_diff(omega1, omega2, gamma,
                                        sigma_y)
            delta_omega = omega1 - omega2
            subj_ret = (delta_omega * sigma_y
                        + delta_omega**2 / (2 * gamma))
            if abs(subj_ret) < 1e-15:
                rho_boundary.append(np.nan)
                continue
            # pr + (1-ρ)/ρ * subj_ret = 0
            # pr*ρ + subj_ret - ρ*subj_ret = 0
            # ρ(pr - subj_ret) = -subj_ret
            # ρ = -subj_ret / (pr - subj_ret)
            #   = subj_ret / (subj_ret - pr)
            denom = subj_ret - pr
            if abs(denom) < 1e-15:
                rho_boundary.append(np.nan)
            else:
                rho_val = subj_ret / denom
                rho_boundary.append(rho_val)
    else:
        # Agent 2 survival: drift at v→1 = 0 (symmetric)
        for gamma in gamma_range:
            pr = portfolio_return_diff(omega2, omega1, gamma,
                                        sigma_y)
            delta_omega = omega2 - omega1
            subj_ret = (delta_omega * sigma_y
                        + delta_omega**2 / (2 * gamma))
            if abs(subj_ret) < 1e-15:
                rho_boundary.append(np.nan)
                continue
            denom = subj_ret - pr
            if abs(denom) < 1e-15:
                rho_boundary.append(np.nan)
            else:
                rho_val = subj_ret / denom
                rho_boundary.append(rho_val)

    return np.array(rho_boundary)
```

```{code-cell} ipython3
sigma_y = 0.02

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Four cases of belief distortion
cases = [
    (0.25, 0, r'$\omega^1 = 0.25$ (moderate optimism)'),
    (1.0, 0, r'$\omega^1 = 1.0$ (strong optimism)'),
    (5.0, 0, r'$\omega^1 \to \infty$ (extreme optimism / $\sigma_Y \to 0$)'),
    (-0.5, 0, r'$\omega^1 = -0.5$ (pessimism)')
]

gamma_range = np.linspace(0.1, 30, 500)

for idx, (omega1, omega2, title) in enumerate(cases):
    ax = axes[idx // 2][idx % 2]

    # Compute boundaries
    rho_lower = compute_survival_boundary(omega1, omega2, sigma_y,
                                           gamma_range,
                                           boundary='lower')
    rho_upper = compute_survival_boundary(omega1, omega2, sigma_y,
                                           gamma_range,
                                           boundary='upper')

    # Clean up invalid values
    rho_lower = np.clip(rho_lower, 0.01, 30)
    rho_upper = np.clip(rho_upper, 0.01, 30)

    # Plot boundaries
    ax.plot(gamma_range, rho_lower, 'b--', linewidth=2,
            label=r'Agent 1 survival boundary')
    ax.plot(gamma_range, rho_upper, 'r-', linewidth=2,
            label=r'Agent 2 survival boundary')

    # CRRA diagonal
    ax.plot(gamma_range, gamma_range, 'k:', linewidth=1,
            label=r'CRRA ($\gamma = \rho$)')

    # Label regions
    ax.set_xlabel(r'Risk aversion $\gamma$', fontsize=12)
    ax.set_ylabel(r'Inverse of IES $\rho$', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.legend(fontsize=9, loc='upper left')

plt.tight_layout()
plt.show()
```

The **shaded region** between the two boundaries corresponds to parameter combinations
where both agents coexist in the long run --- a **nondegenerate stationary distribution**
of wealth exists.

Key observations:

* Along the CRRA diagonal ($\gamma = \rho$, dotted line), the agent with more accurate
  beliefs always dominates, confirming {cite}`Sandroni2000` and {cite}`Blume_Easley2006`

* The coexistence region lies in the empirically relevant part of the parameter space
  where $\gamma > \rho$ (i.e., risk aversion exceeds the inverse of IES)

* As optimism increases, the coexistence region expands

* A pessimistic agent can survive only when IES is sufficiently high and risk aversion
  is not too large


## Three Survival Channels

Let us now visualize the contribution of each survival channel to the total survival
drift, varying one parameter at a time.

```{code-cell} ipython3
def decompose_survival(omega1, omega2, gamma_vals, rho, sigma_y):
    """
    Decompose survival drift into three channels.

    Returns arrays for:
    - risk premium channel
    - volatility penalty
    - saving channel
    """
    delta_omega = omega1 - omega2

    risk_premium_ch = np.zeros_like(gamma_vals)
    vol_penalty_ch = np.zeros_like(gamma_vals)
    saving_ch = np.zeros_like(gamma_vals)

    for i, gamma in enumerate(gamma_vals):
        # Portfolio difference × risk premium
        diff_port = delta_omega / gamma
        rp = gamma * sigma_y - omega2
        risk_premium_ch[i] = diff_port * sigma_y * rp

        # Volatility penalty (always negative for survival)
        vol_penalty_ch[i] = -0.5 * (delta_omega * sigma_y / gamma
                                      + delta_omega)**2

        # Saving channel
        subj_ret = (delta_omega * sigma_y
                    + delta_omega**2 / (2 * gamma))
        saving_ch[i] = (1 - rho) / rho * subj_ret

    total = risk_premium_ch + vol_penalty_ch + saving_ch
    return risk_premium_ch, vol_penalty_ch, saving_ch, total


# Parameters
sigma_y = 0.02
omega1 = 0.25
omega2 = 0.0  # correct beliefs
rho = 0.67     # IES = 1.5

gamma_vals = np.linspace(0.5, 25, 300)

rp, vp, sc, total = decompose_survival(omega1, omega2, gamma_vals,
                                         rho, sigma_y)

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(gamma_vals, rp, 'b-', linewidth=2,
        label='Risk premium channel')
ax.plot(gamma_vals, vp, 'r--', linewidth=2,
        label='Volatility penalty')
ax.plot(gamma_vals, sc, 'g-.', linewidth=2,
        label='Saving channel')
ax.plot(gamma_vals, total, 'k-', linewidth=3,
        label='Total survival drift')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel(r'Risk aversion $\gamma$', fontsize=13)
ax.set_ylabel('Contribution to survival drift', fontsize=13)
ax.set_title(
    rf'Decomposition of survival channels ($\omega^1={omega1}$, '
    rf'$\omega^2={omega2}$, IES$={1/rho:.1f}$, '
    rf'$\sigma_Y={sigma_y}$)',
    fontsize=13
)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()
```

The figure reveals the distinct roles of the three channels:

* The **volatility penalty** (red dashed) is dominant at low risk aversion --- speculative
  portfolios generate volatile returns that hurt the incorrect agent

* The **risk premium channel** (blue) increases with risk aversion --- the more optimistic
  agent earns a higher return by holding more of the risky asset

* The **saving channel** (green) provides a constant positive lift when IES $> 1$ ---
  the optimistic agent saves more in response to her perceived high returns


## Varying IES

The intertemporal elasticity of substitution plays a critical role in survival outcomes.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

gamma_fixed = 10.0
omega1 = 0.25
omega2 = 0.0
sigma_y = 0.02

ies_values = [0.5, 1.0, 1.5]
ies_labels = ['IES = 0.5 (inelastic)', 'IES = 1.0 (log)',
              'IES = 1.5 (elastic)']

gamma_range = np.linspace(0.5, 25, 300)

for idx, (ies, label) in enumerate(zip(ies_values, ies_labels)):
    rho = 1.0 / ies

    rp, vp, sc, total = decompose_survival(omega1, omega2,
                                             gamma_range,
                                             rho, sigma_y)

    ax = axes[idx]
    ax.plot(gamma_range, rp, 'b-', linewidth=2,
            label='Risk premium')
    ax.plot(gamma_range, vp, 'r--', linewidth=2,
            label='Volatility penalty')
    ax.plot(gamma_range, sc, 'g-.', linewidth=2,
            label='Saving channel')
    ax.plot(gamma_range, total, 'k-', linewidth=3,
            label='Total')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(r'Risk aversion $\gamma$', fontsize=12)
    ax.set_ylabel('Contribution', fontsize=12)
    ax.set_title(label, fontsize=13)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.show()
```

Key insights:

* When **IES $= 1$** (center), the saving channel vanishes: consumption-wealth ratios
  are constant and equal to $\beta$, as in the logarithmic case.
  Only the risk premium and volatility channels matter.

* When **IES $> 1$** (right), the saving channel is positive for the optimistic agent:
  she perceives high expected returns and responds by saving more, helping her survive.

* When **IES $< 1$** (left), the saving channel reverses direction: higher perceived
  returns lead to *lower* saving (the income effect dominates), hurting the
  optimistic agent's survival.


## Asymptotic Results

{cite}`Borovicka2020` establishes four key asymptotic results:

**(a) Near risk neutrality** ($\gamma \searrow 0$): each agent dominates with strictly positive
probability.
Low risk aversion encourages speculative portfolio positions.
The volatile returns create a diverging force --- one agent must eventually become
extinct, but which one depends on the realized path.

**(b) High risk aversion** ($\gamma \nearrow \infty$): the relatively more optimistic agent
always dominates.
The risk premium channel dominates, and the pessimistic agent pays too high a price
for insurance.

**(c) High IES** ($\rho \searrow 0$): the relatively more optimistic agent always survives.
The saving channel is strong enough to prevent her extinction.
Whether the pessimistic agent also survives depends on risk aversion.

**(d) Low IES** ($\rho \nearrow \infty$): a nondegenerate long-run equilibrium cannot exist.
Inelastic preferences cause the saving channel to work against survival of the
small agent, regardless of identity.

```{code-cell} ipython3
# Illustrate asymptotic result (a): near risk neutrality
fig, ax = plt.subplots(figsize=(10, 6))

omega1 = 0.25
omega2 = 0.0
sigma_y = 0.02
rho = 0.67  # IES = 1.5

# Show drift at both boundaries as function of gamma
gamma_vals = np.linspace(0.05, 5, 300)

drift_v0 = np.array([survival_drift(omega1, omega2, g, rho, sigma_y)
                      for g in gamma_vals])
# Drift at v=1 by swapping agents
drift_v1 = np.array([survival_drift(omega2, omega1, g, rho, sigma_y)
                      for g in gamma_vals])

ax.plot(gamma_vals, drift_v0, 'b-', linewidth=2,
        label=r'Drift at $\upsilon \to 0$ (agent 1 survival)')
ax.plot(gamma_vals, -drift_v1, 'r--', linewidth=2,
        label=r'Drift at $\upsilon \to 1$ (agent 2 survival)')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel(r'Risk aversion $\gamma$', fontsize=13)
ax.set_ylabel('Survival drift', fontsize=13)
ax.set_title('Boundary drifts as risk aversion varies', fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()
```


## The Separable Case: CRRA Benchmark

Under separable CRRA preferences ($\gamma = \rho$), the dynamics of the log-odds
Pareto share $\vartheta$ become a Brownian motion with constant drift:

$$
d\vartheta_t = \frac{1}{2}\left[(\omega^2)^2 - (\omega^1)^2\right] dt + (\omega^1 - \omega^2) dW_t
$$

The drift does not depend on the state $\upsilon$ and is determined entirely by the
**relative entropy** (Kullback-Leibler divergence) of the agents' beliefs:
$\frac{1}{2}|\omega^n|^2$.

The agent with small $|\omega^n|$ --- more accurate beliefs --- always dominates.

```{code-cell} ipython3
def simulate_crra_pareto(omega1, omega2, T, dt, n_paths, seed=42):
    """
    Simulate Pareto share dynamics under CRRA (γ = ρ).

    Parameters
    ----------
    omega1, omega2 : float
        Belief distortions
    T : float
        Time horizon
    dt : float
        Time step
    n_paths : int
        Number of sample paths
    seed : int
        Random seed

    Returns
    -------
    t_grid : array
        Time grid
    v_paths : array
        Pareto share paths, shape (n_paths, n_steps)
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    t_grid = np.linspace(0, T, n_steps + 1)

    # Drift and vol of log-odds
    drift = 0.5 * (omega2**2 - omega1**2)
    vol = omega1 - omega2

    # Initial log-odds (v0 = 0.5 -> theta0 = 0)
    theta = np.zeros((n_paths, n_steps + 1))
    dW = rng.normal(0, np.sqrt(dt), (n_paths, n_steps))

    for t in range(n_steps):
        theta[:, t+1] = theta[:, t] + drift * dt + vol * dW[:, t]

    # Convert to Pareto share
    v_paths = 1.0 / (1.0 + np.exp(-theta))

    return t_grid, v_paths


# Simulate
T = 200
dt = 0.01
n_paths = 50

omega1 = 0.1   # slightly optimistic (incorrect)
omega2 = 0.0   # correct beliefs

t_grid, v_paths = simulate_crra_pareto(omega1, omega2, T, dt,
                                        n_paths)

fig, ax = plt.subplots(figsize=(12, 6))

for i in range(min(20, n_paths)):
    ax.plot(t_grid, v_paths[i], alpha=0.3, linewidth=0.5)

ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Time $t$', fontsize=13)
ax.set_ylabel(r'Pareto share $\upsilon_t$', fontsize=13)
ax.set_title(
    rf'CRRA case ($\gamma = \rho$): Agent 2 (correct, $\omega^2=0$) '
    rf'dominates over agent 1 ($\omega^1={omega1}$)',
    fontsize=13
)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
```

Under separable preferences, agent 2 (with correct beliefs) always drives agent 1's
Pareto share to zero.


## Economy with Constant Aggregate Endowment

An illuminating special case arises when aggregate endowment is constant
($\mu_Y = \sigma_Y = 0$).
In this economy, agents trade purely for **speculative** motives.

The survival results do not depend on $\mu_Y$ or $\sigma_Y$ independently but only on
the ratio $\omega^n / \sigma_Y$.
The limit $\sigma_Y \to 0$ with $\omega^1 \neq 0$ thus isolates the saving channel.

In this case:

* The risk premium is zero (no aggregate risk)
* The speculative volatility channel is present but muted at high risk aversion
* The saving channel alone can generate survival of the incorrect agent when IES $> 1$

```{code-cell} ipython3
# Show survival regions for the limiting case ω/σ_y → ∞
# (equivalent to σ_y → 0 or ω → ∞)

fig, ax = plt.subplots(figsize=(10, 8))

gamma_range = np.linspace(0.1, 30, 500)

# In the limit, survival of agent 1 requires IES > 1
# i.e., ρ < 1
ax.axhline(1.0, color='b', linestyle='--', linewidth=2,
           label=r'Agent 1 survival: $\rho < 1$ (IES $> 1$)')

# Agent 2 always survives (correct beliefs, no risk premium cost)
# The boundary is the CRRA line
ax.plot(gamma_range, gamma_range, 'k:', linewidth=1,
        label=r'CRRA ($\gamma = \rho$)')

# Shade coexistence region
ax.fill_between(gamma_range, 0, np.minimum(1.0, gamma_range),
                alpha=0.2, color='green',
                label='Both agents survive')
ax.fill_between(gamma_range, np.minimum(1.0, gamma_range),
                np.ones_like(gamma_range),
                where=gamma_range > 1,
                alpha=0.2, color='blue',
                label='Both survive (above CRRA, below ρ=1)')

ax.set_xlabel(r'Risk aversion $\gamma$', fontsize=13)
ax.set_ylabel(r'Inverse of IES $\rho$', fontsize=13)
ax.set_title(
    r'Survival regions: $\sigma_Y \to 0$ (pure speculation)',
    fontsize=13)
ax.set_xlim(0, 30)
ax.set_ylim(0, 10)
ax.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.show()
```

In the economy without aggregate risk, IES $> 1$ is sufficient for the incorrect agent
to survive when risk aversion is sufficiently high.
This is the pure saving channel at work.


## Asset Pricing Implications

{cite}`Borovicka2020` also shows that as the Pareto share of one agent becomes negligible,
current asset prices converge to those in a homogeneous economy populated by the
large agent.

### Prices at the boundary

As $\upsilon \searrow 0$ (agent 2 dominates):

**Risk-free rate:**

$$
\lim_{\upsilon \searrow 0} r(\upsilon) = \beta + \rho \mu_Y + \omega^2 \sigma_Y
+ \frac{1}{2}(1-\gamma)\sigma_Y^2 - \frac{1}{2}\gamma \sigma_Y^2
$$ (eq:riskfree)

**Wealth-consumption ratio:**

$$
\lim_{\upsilon \searrow 0} y(\upsilon) = \left[\beta - (1-\rho)\left(\mu_Y
+ \omega^2 \sigma_Y + \frac{1}{2}(1-\gamma)\sigma_Y^2\right)\right]^{-1}
$$ (eq:wc_ratio)

### Portfolio choice of the negligible agent

The small agent's portfolio share in the risky asset converges to

$$
\lim_{\upsilon \searrow 0} \pi^1(\upsilon) = 1 + \frac{\omega^1 - \omega^2}{\gamma \sigma_Y}
$$ (eq:portfolio)

An optimistic agent ($\omega^1 > \omega^2$) holds a **leveraged** position ($\pi^1 > 1$).

A pessimistic agent ($\omega^1 < \omega^2$) **shorts** the risky asset when
$\omega^1 - \omega^2 < -\gamma \sigma_Y$.

```{code-cell} ipython3
# Portfolio share of agent 1 as function of belief distortion
fig, ax = plt.subplots(figsize=(10, 6))

omega2 = 0.0
sigma_y = 0.02

omega1_range = np.linspace(-0.5, 1.0, 300)

for gamma in [2, 5, 10, 20]:
    pi1 = 1 + (omega1_range - omega2) / (gamma * sigma_y)
    ax.plot(omega1_range, pi1, linewidth=2,
            label=rf'$\gamma = {gamma}$')

ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
ax.axhline(0.0, color='gray', linestyle=':', alpha=0.5)
ax.axvline(0.0, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel(r'Belief distortion $\omega^1$', fontsize=13)
ax.set_ylabel(r'Portfolio share $\pi^1$', fontsize=13)
ax.set_title(
    'Portfolio share of negligible agent in risky asset',
    fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()
```

Key observations:

* At $\omega^1 = 0$ (correct beliefs), the agent holds the market portfolio ($\pi^1 = 1$)

* Higher risk aversion reduces the speculative position toward the market portfolio

* A pessimistic agent with low risk aversion may take a large short position,
  generating the volatile returns needed for the saving channel to operate


## Optimistic versus Pessimistic Distortions

A striking feature of the model is that optimistic and pessimistic belief distortions
have **asymmetric** effects on survival.

An optimistic agent ($\omega^1 > 0$) benefits from both the risk premium channel
(she holds more of the risky asset and earns the risk premium) and the saving
channel (she perceives high returns and saves more under IES $> 1$).

A pessimistic agent ($\omega^1 < 0$) is disadvantaged by the risk premium channel
(she holds less risky asset and foregoes the premium).
However, she can potentially survive through the saving channel if she shorts the
risky asset aggressively enough to perceive a high expected return on her own portfolio.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sigma_y = 0.02
omega2 = 0.0
rho = 0.67  # IES = 1.5
gamma_range = np.linspace(0.5, 25, 300)

# Optimistic agent
ax = axes[0]
for omega1 in [0.1, 0.25, 0.5, 1.0]:
    rp, vp, sc, total = decompose_survival(omega1, omega2,
                                             gamma_range,
                                             rho, sigma_y)
    ax.plot(gamma_range, total, linewidth=2,
            label=rf'$\omega^1 = {omega1}$')

ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel(r'Risk aversion $\gamma$', fontsize=12)
ax.set_ylabel('Total survival drift', fontsize=12)
ax.set_title('Optimistic agent 1', fontsize=13)
ax.legend(fontsize=10)

# Pessimistic agent
ax = axes[1]
for omega1 in [-0.1, -0.25, -0.5, -1.0]:
    rp, vp, sc, total = decompose_survival(omega1, omega2,
                                             gamma_range,
                                             rho, sigma_y)
    ax.plot(gamma_range, total, linewidth=2,
            label=rf'$\omega^1 = {omega1}$')

ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel(r'Risk aversion $\gamma$', fontsize=12)
ax.set_ylabel('Total survival drift', fontsize=12)
ax.set_title('Pessimistic agent 1', fontsize=13)
ax.legend(fontsize=10)

plt.tight_layout()
plt.show()
```

For the optimistic agent (left), survival drift turns positive at moderate risk
aversion and stays positive.

For the pessimistic agent (right), survival drift is negative for high risk aversion
and becomes positive only at intermediate risk aversion levels --- and only when the
belief distortion is large enough to induce an aggressive short position.



## Long-Run Consumption Distribution

When both agents survive, the stationary distribution of consumption shares provides
information about the typical wealth allocation.

{cite}`Borovicka2020` shows that when agent $n$ survives, she attains an
arbitrarily large consumption share $z^n \in (0, 1)$ with probability one at some
future date.

Let us simulate the Pareto share dynamics in a simplified model to illustrate
the ergodic behavior.

```{code-cell} ipython3
def simulate_pareto_share(omega1, omega2, gamma, rho, sigma_y,
                           beta, T, dt, n_paths=20, seed=42):
    """
    Simulate Pareto share dynamics with state-dependent drift.

    This uses a simplified approximation where the endogenous
    discount rate difference is computed from the boundary formulas.

    Parameters
    ----------
    omega1, omega2 : float
        Belief distortions
    gamma, rho : float
        Preference parameters
    sigma_y : float
        Endowment volatility
    beta : float
        Time preference
    T : float
        Time horizon
    dt : float
        Time step
    n_paths : int
        Number of paths
    seed : int
        Random seed

    Returns
    -------
    t_grid, v_paths : arrays
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    t_grid = np.linspace(0, T, n_steps + 1)

    vol_theta = omega1 - omega2

    # Compute boundary drifts for interpolation
    drift_at_0 = survival_drift(omega1, omega2, gamma, rho, sigma_y)
    drift_at_1 = -survival_drift(omega2, omega1, gamma, rho, sigma_y)

    theta = np.zeros((n_paths, n_steps + 1))
    dW = rng.normal(0, np.sqrt(dt), (n_paths, n_steps))

    for t in range(n_steps):
        v = 1.0 / (1.0 + np.exp(-theta[:, t]))
        # Interpolate drift between boundaries
        # Simple linear interpolation
        drift = drift_at_0 * (1 - v) + drift_at_1 * v
        theta[:, t+1] = (theta[:, t]
                          + drift * dt
                          + vol_theta * dW[:, t])

    v_paths = 1.0 / (1.0 + np.exp(-theta))
    return t_grid, v_paths


# Parameters for coexistence region
omega1 = 0.25
omega2 = 0.0
gamma = 10.0
rho = 0.67    # IES = 1.5
sigma_y = 0.02
beta = 0.05

T = 500
dt = 0.05

t_grid, v_paths = simulate_pareto_share(
    omega1, omega2, gamma, rho, sigma_y, beta, T, dt,
    n_paths=50, seed=42
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sample paths
ax = axes[0]
for i in range(20):
    ax.plot(t_grid, v_paths[i], alpha=0.3, linewidth=0.5)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel(r'Pareto share $\upsilon_t$', fontsize=12)
ax.set_title('Sample paths of Pareto share\n'
             r'($\gamma=10$, IES$=1.5$, $\omega^1=0.25$, '
             r'$\omega^2=0$)',
             fontsize=12)
ax.set_ylim(0, 1)

# Histogram of final values (approximate stationary distribution)
ax = axes[1]
# Use last half of a very long simulation
t_grid_long, v_long = simulate_pareto_share(
    omega1, omega2, gamma, rho, sigma_y, beta,
    T=2000, dt=0.05, n_paths=5, seed=123
)
# Pool observations from second half
v_stationary = v_long[:, v_long.shape[1]//2:].flatten()
ax.hist(v_stationary, bins=80, density=True, alpha=0.7,
        color='steelblue', edgecolor='white')
ax.set_xlabel(r'Pareto share $\upsilon$', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Approximate stationary distribution', fontsize=12)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.show()
```

When both agents survive, the Pareto share fluctuates persistently across the full
interval $(0, 1)$.
This means the incorrect agent periodically commands a substantial share of
aggregate consumption.

## Summary

{cite}`Borovicka2020` overturns the classical market selection hypothesis by showing
that under recursive preferences of the Epstein-Zin type, agents with incorrect
beliefs can survive --- and even thrive --- in the long run.

The key findings are:

1. **Three channels** determine survival: the risk premium channel, the speculative
   volatility channel, and the saving channel. Only the first two operate under
   separable CRRA preferences.

2. **IES matters**: When IES $> 1$, the saving channel helps agents with distorted beliefs
   outsave extinction. When IES $< 1$, it works against them.

3. **Coexistence is generic**: For empirically relevant parameter values
   ($\gamma > \rho$, IES $> 1$), nondegenerate stationary wealth distributions exist.

4. **Optimism vs. pessimism**: Optimistic agents benefit from both the risk premium and
   saving channels. Pessimistic agents can survive only through aggressive shorting
   combined with high IES.

5. **Price impact**: A surviving agent with currently negligible wealth has no impact on
   current prices, but will affect prices in the future when her wealth share recovers.

These results have important implications for asset pricing.
Models that feature agents with heterogeneous beliefs and recursive preferences can
generate persistent heterogeneity and endogenous fluctuations in the wealth
distribution, enriching the dynamics of equilibrium asset prices, risk premia,
and trading volume.
