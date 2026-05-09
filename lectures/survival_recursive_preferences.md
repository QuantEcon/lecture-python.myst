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

(survival_recursive_preferences)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Survival and Long-Run Dynamics under Recursive Preferences

```{index} single: Survival; Recursive Preferences
```

```{contents} Contents
:depth: 2
```

## Overview

This lecture studies the theory of long-run survival in {cite:t}`Borovicka2020`.

The classical **market selection hypothesis** says that agents with less accurate beliefs are driven out of the market in the long run.

This result was established rigorously by {cite:t}`Sandroni2000Markets` and {cite:t}`Blume_Easley2006` for economies with separable CRRA preferences.

Borovicka shows that the conclusion can fail under Epstein-Zin recursive preferences.

With recursive preferences, agents with distorted beliefs can survive and can even dominate.

The key mechanism is that recursive preferences separate risk aversion from the intertemporal elasticity of substitution.

That separation creates three channels that matter for survival:

1. The *risk premium channel* rewards the more optimistic agent for holding more of the risky asset.
1. The *speculative volatility channel* penalizes aggressive positions through log-return volatility.
1. The *saving channel* changes consumption and saving decisions when the IES differs from one.

Under separable preferences, only the first two channels remain.

Under recursive preferences, the saving channel can overturn market selection.

```{note}
The paper builds on the continuous-time recursive utility formulation of {cite:t}`Duffie_Epstein1992a`,
using the planner's problem approach of {cite:t}`Dumas_Uppal_Wang2000`.

Important foundations for the market selection hypothesis were laid by
{cite:t}`DeLong_etal1991` and {cite:t}`Blume_Easley1992`.
```

We start with some imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## Environment

The economy contains two infinitely lived agents, indexed by $n \in \{1, 2\}$.

The agents have identical recursive preferences but different beliefs about aggregate endowment growth.

We write Borovička's belief distortions $u^n$ as $\omega^n$.

### Aggregate endowment

Under the true probability measure $P$, aggregate endowment satisfies

$$
d \log Y_t = \mu_Y dt + \sigma_Y dW_t, \quad Y_0 > 0
$$ (eq:srp_endowment)

where $W$ is a standard Brownian motion, $\mu_Y$ is the drift, and $\sigma_Y > 0$ is the volatility.

### Heterogeneous beliefs

Agent $n$ believes that the drift is $\mu_Y + \omega^n \sigma_Y$ instead of $\mu_Y$.

The parameter $\omega^n$ measures optimism when $\omega^n > 0$ and pessimism when $\omega^n < 0$.

Agent $n$'s subjective probability measure $Q^n$ is defined by the Radon–Nikodym derivative

$$
M_t^n = \frac{dQ^n}{dP}\bigg|_t = \exp\left(-\frac{1}{2} |\omega^n|^2 t + \omega^n W_t\right)
$$ (eq:radon_nikodym)

Under $Q^n$, the process $W_t^n = W_t - \omega^n t$ is a Brownian motion, and agent $n$ perceives

$$
d \log Y_t = (\mu_Y + \omega^n \sigma_Y) dt + \sigma_Y dW_t^n .
$$

An agent with $\omega^n > 0$ is optimistic about endowment growth, while an agent with $\omega^n < 0$ is pessimistic.

### Recursive preferences

Both agents have Epstein-Zin recursive preferences.

We use $\gamma > 0$ for relative risk aversion, $\rho > 0$ for the inverse of the IES, and $\beta > 0$ for the time-preference rate.

The Duffie-Epstein-Zin felicity function is

$$
F(C, \nu)
= \beta \frac{C^{1-\gamma}}{1-\gamma}
\cdot
\left(\frac{(1-\gamma) - (1-\rho)\nu / \beta}{\rho - \gamma}\right)^{(\gamma - \rho)/(1-\rho)}
$$ (eq:felicity)

where $\nu$ is the endogenous discount rate.

```{note}
In discrete time, Epstein-Zin preferences aggregate current consumption with a certainty equivalent of future utility via a CES aggregator (see {doc}`advanced:doubts_or_variability`).

In continuous time there is no "next-period $V_{t+1}$," so {cite:t}`Duffie_Epstein1992a` recast the recursion as a felicity function $F(C,\nu)$ that depends on the agent's own continuation-value rate $\nu$.

The two formulations encode the same separation of risk aversion $\gamma$ from the inverse IES $\rho$.

When $\gamma = \rho$, preferences reduce to the standard separable CRRA case.
```

## Planner's problem

Following {cite:t}`Dumas_Uppal_Wang2000`, we study equilibrium allocations through a social planner's problem.

The planner chooses consumption shares $z^1$ and $z^2 = 1 - z^1$ and discount-rate processes $\nu^n$ for the two agents.

### Modified discount factors

It is convenient to absorb belief distortions into the modified discount factors $\tilde{\lambda}^n = \lambda^n M^n$, where $M^n$ is the Radon-Nikodym derivative {eq}`eq:radon_nikodym`.

These processes satisfy

$$
d \log \tilde{\lambda}_t^n
= -\left(\nu_t^n + \frac{1}{2} (\omega^n)^2\right) dt + \omega^n dW_t .
$$ (eq:modified_discount)

```{exercise}
:label: ex_modified_discount

Derive {eq}`eq:modified_discount`.

*Hint:* Use $\log \tilde{\lambda}^n = \log \lambda^n + \log M^n$. The Pareto weight $\lambda^n$ evolves as $d\log \lambda_t^n = -\nu_t^n \, dt$, and $\log M_t^n$ is given by {eq}`eq:radon_nikodym`.
```

```{solution-start} ex_modified_discount
:class: dropdown
```

From the definition $\tilde{\lambda}^n = \lambda^n M^n$, we have

$$
\log \tilde{\lambda}_t^n = \log \lambda_t^n + \log M_t^n.
$$

The Pareto weight satisfies $d\log \lambda_t^n = -\nu_t^n \, dt$.

From {eq}`eq:radon_nikodym`, $\log M_t^n = -\frac{1}{2}|\omega^n|^2 t + \omega^n W_t$, so

$$
d \log M_t^n = -\tfrac{1}{2}(\omega^n)^2 \, dt + \omega^n \, dW_t.
$$

Adding the two:

$$
d \log \tilde{\lambda}_t^n = -\nu_t^n \, dt - \tfrac{1}{2}(\omega^n)^2 \, dt + \omega^n \, dW_t = -\left(\nu_t^n + \tfrac{1}{2}(\omega^n)^2\right) dt + \omega^n \, dW_t.
$$

```{solution-end}
```

### State variable: Pareto share

The key state variable is the Pareto share of agent 1:

$$
\upsilon = \frac{\tilde{\lambda}^1}{\tilde{\lambda}^1 + \tilde{\lambda}^2} \in (0, 1)
$$ (eq:pareto_share)

It captures the relative weight of agent 1 in the planner's allocation.

Define the log-odds ratio $\vartheta = \log(\upsilon / (1 - \upsilon))$.

Its dynamics are

$$
d\vartheta_t = \underbrace{\left[\nu_t^2 + \frac{1}{2}(\omega^2)^2 - \nu_t^1 - \frac{1}{2}(\omega^1)^2\right]}_{m_{\vartheta}(\upsilon_t)} dt + (\omega^1 - \omega^2) dW_t
$$ (eq:log_odds)

The drift $m_\vartheta(\upsilon)$ determines the long-run behavior of the Pareto share.

```{exercise}
:label: ex_log_odds

Derive {eq}`eq:log_odds` from {eq}`eq:modified_discount` and the definition $\vartheta = \log(\upsilon/(1-\upsilon))$.

*Hint:* First show that $\vartheta = \log \tilde{\lambda}^1 - \log \tilde{\lambda}^2$, then subtract the two SDEs.
```

```{solution-start} ex_log_odds
:class: dropdown
```

Since $\upsilon = \tilde{\lambda}^1 / (\tilde{\lambda}^1 + \tilde{\lambda}^2)$, we have $1 - \upsilon = \tilde{\lambda}^2 / (\tilde{\lambda}^1 + \tilde{\lambda}^2)$, so

$$
\vartheta = \log\frac{\upsilon}{1-\upsilon} = \log \tilde{\lambda}^1 - \log \tilde{\lambda}^2.
$$

From {eq}`eq:modified_discount`, the two log-discount-factor SDEs are

$$
d\log \tilde{\lambda}^1_t = -\left(\nu_t^1 + \tfrac{1}{2}(\omega^1)^2\right)dt + \omega^1 dW_t,
$$

$$
d\log \tilde{\lambda}^2_t = -\left(\nu_t^2 + \tfrac{1}{2}(\omega^2)^2\right)dt + \omega^2 dW_t.
$$

Subtracting the second from the first:

$$
d\vartheta_t = \left[\nu_t^2 + \tfrac{1}{2}(\omega^2)^2 - \nu_t^1 - \tfrac{1}{2}(\omega^1)^2\right]dt + (\omega^1 - \omega^2)dW_t.
$$

```{solution-end}
```

### HJB equation

Homotheticity reduces the planner's problem to a nonlinear ODE in the single state variable $\upsilon$.

Because each agent's utility is homogeneous of degree $1-\gamma$ in consumption, the planner's value function factors as $J(\upsilon, Y) = \tilde{J}(\upsilon) \cdot Y^{1-\gamma}/(1-\gamma)$, eliminating $Y$ as a state variable.

#### From discrete to continuous time

In discrete time, a planner maximizes a weighted sum of agents' utilities by choosing allocations at each date.

The Bellman equation is

$$
\tilde{J}(\upsilon) = \max_{z^1, z^2} \left\{ \upsilon \, u(z^1) + (1-\upsilon) \, u(z^2) + \beta \, \mathbb{E}\left[\tilde{J}(\upsilon')\right] \right\}.
$$

In continuous time, the period length shrinks to $dt$.

The "flow payoff" over $[t, t+dt)$ becomes $\left[\upsilon F(z^1, \nu^1) + (1-\upsilon)F(z^2, \nu^2)\right] dt$, where $F$ is the Duffie-Epstein-Zin felicity {eq}`eq:felicity`.

The expected change in the value function over $dt$ is captured by the **infinitesimal generator** $\mathcal{L}$.

For a diffusion $d\upsilon = m \, dt + s \, dW$, Itô's lemma gives

$$
\mathcal{L}\tilde{J}(\upsilon) = m(\upsilon)\,\tilde{J}'(\upsilon) + \tfrac{1}{2} s(\upsilon)^2 \, \tilde{J}''(\upsilon),
$$

where $m$ and $s$ are the drift and diffusion of the Pareto share.

This is the continuous-time analogue of $\beta \, \mathbb{E}[\tilde{J}(\upsilon')] - \tilde{J}(\upsilon)$: it measures how the value function drifts and fluctuates as $\upsilon$ evolves.

Setting flow payoff plus expected capital gain equal to zero gives the schematic HJB equation:

$$
0 = \sup_{(z^1,z^2,\nu^1,\nu^2)} \left\{ \upsilon F(z^1, \nu^1) + (1-\upsilon) F(z^2, \nu^2) + \mathcal{L} \tilde{J}(\upsilon) \right\}
$$ (eq:hjb_sketch)

subject to $z^1 + z^2 \leq 1$.

#### Exact reduced ODE

Proposition 2.3 of {cite:t}`Borovicka2020` gives the exact HJB equation after substituting the homogeneity reduction $J(\tilde{\lambda}, Y) = (\tilde{\lambda}^1 + \tilde{\lambda}^2) Y^{1-\gamma} \tilde{J}(\upsilon)$ and the dynamics of $\upsilon$ and $Y$:

$$
0 = \sup_{(z^1, z^2, \nu^1, \nu^2)} \;
\upsilon \, F(z^1, \nu^1) + (1 - \upsilon) \, F(z^2, \nu^2)
$$ (eq:hjb)

$$
+ \left[
-\upsilon \nu^1 - (1-\upsilon)\nu^2
+ \bigl(\upsilon \omega^1 + (1-\upsilon)\omega^2\bigr)(1-\gamma)\sigma_Y
+ (1-\gamma)\mu_Y
+ \tfrac{1}{2}(1-\gamma)^2 \sigma_Y^2
\right] \tilde{J}(\upsilon)
$$

$$
+ \upsilon(1-\upsilon)
\left[\nu^2 - \nu^1 + (\omega^1 - \omega^2)(1-\gamma)\sigma_Y\right]
\tilde{J}'(\upsilon)
$$

$$
+ \tfrac{1}{2}\upsilon^2(1-\upsilon)^2 (\omega^1 - \omega^2)^2 \,
\tilde{J}''(\upsilon)
$$

subject to $z^1 + z^2 \leq 1$.

The first line is the flow payoff from the two agents' felicity functions.

The second line multiplies $\tilde{J}(\upsilon)$ by a term that combines the agents' discount rates, belief-weighted endowment drift, and a variance correction -- these arise from absorbing the $Y^{1-\gamma}$ factor via Itô's lemma.

The third line multiplies $\tilde{J}'(\upsilon)$ by the drift of the Pareto share, which depends on the difference in discount rates and the belief-weighted response to endowment risk.

The fourth line multiplies $\tilde{J}''(\upsilon)$ by the squared diffusion of the Pareto share.

The boundary conditions are $\tilde{J}(0) = \tilde{V}^2$ and $\tilde{J}(1) = \tilde{V}^1$, where $\tilde{V}^n$ is the continuation value in the homogeneous economy populated by agent $n$ alone.

This is the continuous-time counterpart of the discrete-time planner's problem in {cite:t}`Blume_Easley2006` (see also {doc}`likelihood_ratio_process_2`).


## Survival conditions

The central result characterizes survival by the boundary behavior of $m_\vartheta(\upsilon)$.

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

*(a)* If (i) and (ii) hold, both agents survive under $P$.

*(b)* If (i) and (ii') hold, agent 1 dominates in the long run under $P$.

*(c)* If (i') and (ii) hold, agent 2 dominates in the long run under $P$.

*(d)* If (i') and (ii') hold, each agent dominates with strictly positive probability.
```

The proof uses the Feller classification of boundary behavior for diffusions, as in {cite:t}`Karlin_Taylor1981`.

Condition (i) says that when agent 1 is close to extinction, there is a force pushing her share back up.

Condition (ii) says that when agent 1 is close to absorbing the whole economy, there is a force pushing her share back down.

When both forces are present, the Pareto share is recurrent and both agents survive.

## Wealth dynamics decomposition

We now rewrite the survival conditions from {prf:ref}`survival_conditions` in terms of equilibrium wealth dynamics.

Agent 1 survives near extinction if and only if her wealth grows faster than agent 2's when she is negligibly small.

When $\upsilon \searrow 0$, prices are set entirely by agent 2, as if the economy were homogeneous.

Agent 1 is a price-taker in agent 2's economy.

Let $m_A^n(\upsilon)$ denote the expected log growth rate of agent $n$'s wealth.

The difference decomposes into two channels:

$$
\lim_{\upsilon \searrow 0} [m_A^1(\upsilon) - m_A^2(\upsilon)]
= \underbrace{\lim_{\upsilon \searrow 0} [m_R^1(\upsilon) - m_R^2(\upsilon)]}_{\text{portfolio returns}}
+ \underbrace{\lim_{\upsilon \searrow 0} [(y^2(\upsilon))^{-1} - (y^1(\upsilon))^{-1}]}_{\text{consumption-wealth ratios}}
$$ (eq:wealth_decomp)

The first term measures how much faster agent 1's portfolio grows.

The second measures how much less agent 1 consumes out of wealth -- a lower consumption-wealth ratio means more saving and faster wealth accumulation.

When this total difference is positive, agent 1 survives; when negative, she shrinks toward extinction.

```{exercise}
:label: ex_wealth_decomp

Derive {eq}`eq:wealth_decomp`.

Let $A^n$ denote agent $n$'s wealth and $C^n$ her consumption. 

The budget constraint is $dA^n = A^n dR^n - C^n dt$, where $dR^n$ is the return on agent $n$'s portfolio. 

Define the consumption-wealth ratio $c^n = C^n / A^n = (y^n)^{-1}$.

Show that $d\log A^n = m_R^n \, dt - (y^n)^{-1} dt + \ldots$, so the difference in expected log wealth growth is $m_A^1 - m_A^2 = (m_R^1 - m_R^2) + [(y^2)^{-1} - (y^1)^{-1}]$.
```

```{solution-start} ex_wealth_decomp
:class: dropdown
```

Dividing the budget constraint by $A^n$:

$$
\frac{dA^n}{A^n} = dR^n - (y^n)^{-1} dt.
$$

By Itô's lemma, $d\log A^n = \frac{dA^n}{A^n} - \frac{1}{2}\left(\frac{dA^n}{A^n}\right)^2$.

Write $dR^n = m_R^n \, dt + \sigma_R^n \, dW$ (the portfolio return under $P$). 

Then

$$
d\log A^n = \left(m_R^n - (y^n)^{-1} - \tfrac{1}{2}(\sigma_R^n)^2\right) dt + \sigma_R^n \, dW.
$$

Taking the difference for agents 1 and 2:

$$
m_A^1 - m_A^2 = (m_R^1 - m_R^2) + \left[(y^2)^{-1} - (y^1)^{-1}\right] - \tfrac{1}{2}\left[(\sigma_R^1)^2 - (\sigma_R^2)^2\right].
$$

The volatility terms $\tfrac{1}{2}[(\sigma_R^1)^2 - (\sigma_R^2)^2]$ are absorbed into $m_R^1 - m_R^2$ when we define $m_R^n$ as the expected log portfolio return (i.e., the drift of $\log R^n$ rather than the arithmetic return), giving {eq}`eq:wealth_decomp`.

```{solution-end}
```

### Portfolio returns

At the boundary $\upsilon \searrow 0$, the difference in expected log portfolio returns is

$$
\lim_{\upsilon \searrow 0} [m_R^1 - m_R^2]
= \underbrace{\frac{\omega^1 - \omega^2}{\gamma \sigma_Y}}_{\text{difference in risky shares}}
\cdot \underbrace{(\gamma \sigma_Y^2 - \omega^2 \sigma_Y)}_{\text{risk premium}}
- \underbrace{\frac{\omega^1 - \omega^2}{\gamma}
\left(\sigma_Y + \frac{\omega^1 - \omega^2}{2\gamma}\right)}_{\text{volatility term}}
$$ (eq:portfolio_returns)

An optimistic agent ($\omega^1 > \omega^2$) overweights the risky asset by $(\omega^1 - \omega^2)/(\gamma \sigma_Y)$ relative to agent 2 and earns the equity risk premium on that extra exposure.

The subtracted *volatility penalty* reflects the cost of holding a more extreme portfolio: higher variance of log returns drags down expected log wealth growth.

This term depends on risk aversion $\gamma$ but not on the IES, because portfolio choice is determined by risk aversion alone.

```{exercise}
:label: ex_portfolio_returns

Derive {eq}`eq:portfolio_returns`.

At the boundary $\upsilon \searrow 0$, agent $n$'s optimal risky-asset share is $\pi^n = 1 + (\omega^n - \omega^2)/(\gamma \sigma_Y)$ (see {eq}`eq:portfolio`). 

Let $\bar{\mu}_R = \mu_Y + \gamma \sigma_Y^2 - \omega^2 \sigma_Y$ denote the expected return on the risky asset under $P$, and $r$ the risk-free rate.

The continuously rebalanced portfolio has expected log return $m_R^n = r + \pi^n(\bar{\mu}_R - r) - \frac{1}{2}(\pi^n)^2 \sigma_Y^2$.

Compute $m_R^1 - m_R^2$ and simplify.
```

```{solution-start} ex_portfolio_returns
:class: dropdown
```

Using $m_R^n = r + \pi^n(\bar{\mu}_R - r) - \frac{1}{2}(\pi^n)^2 \sigma_Y^2$, the difference is

$$
m_R^1 - m_R^2 = (\pi^1 - \pi^2)(\bar{\mu}_R - r) - \tfrac{1}{2}[(\pi^1)^2 - (\pi^2)^2]\sigma_Y^2.
$$

The difference in risky shares is $\pi^1 - \pi^2 = (\omega^1 - \omega^2)/(\gamma \sigma_Y)$.

The arithmetic equity premium is $\bar{\mu}_R - r = \gamma \sigma_Y^2 - \omega^2 \sigma_Y$, so:

$$
(\pi^1 - \pi^2)(\bar{\mu}_R - r) = \frac{\omega^1 - \omega^2}{\gamma \sigma_Y} \cdot (\gamma \sigma_Y^2 - \omega^2 \sigma_Y).
$$

For the volatility term, write $(\pi^1)^2 - (\pi^2)^2 = (\pi^1 - \pi^2)(\pi^1 + \pi^2)$ and note $\pi^1 + \pi^2 = 2 + (\omega^1 + \omega^2 - 2\omega^2)/(\gamma \sigma_Y)$. 

After simplification:

$$
\tfrac{1}{2}[(\pi^1)^2 - (\pi^2)^2]\sigma_Y^2 = \frac{\omega^1 - \omega^2}{\gamma}\left(\sigma_Y + \frac{\omega^1 - \omega^2}{2\gamma}\right).
$$

Combining the two pieces gives {eq}`eq:portfolio_returns`.

```{solution-end}
```

### Consumption-wealth ratios

The difference in consumption-wealth ratios at the boundary is

$$
\lim_{\upsilon \searrow 0} [(y^2)^{-1} - (y^1)^{-1}]
= \frac{1-\rho}{\rho} \left[(\omega^1 - \omega^2)\sigma_Y + \frac{(\omega^1 - \omega^2)^2}{2\gamma}\right]
$$ (eq:consumption_rates)

The term in brackets is the difference in *subjective* expected portfolio returns -- what agent 1 believes she earns relative to agent 2.

The factor $(1-\rho)/\rho$ translates this perceived return advantage into a saving response.

- When IES $> 1$ ($\rho < 1$), the factor is positive: a higher perceived return makes the agent save more, because the substitution effect dominates the income effect.
- When IES $< 1$ ($\rho > 1$), the factor is negative: the income effect dominates and the agent saves less, working against survival.
- When IES $= 1$ ($\rho = 1$), the two effects cancel and the saving channel vanishes entirely.

This is the channel through which recursive preferences alter survival outcomes by separating $\gamma$ from $\rho$.

```{exercise}
:label: ex_consumption_wealth

Derive {eq}`eq:consumption_rates`.

In the homogeneous economy populated by agent 2, the consumption-wealth ratio is $(y(0))^{-1} = \beta - (1-\rho)\mu_V^2$, where $\mu_V^2$ is agent 2's expected log return on wealth. 

Agent 1, as a negligible price-taker, has consumption-wealth ratio $(y^1)^{-1} = \beta - (1-\rho)\mu_V^1$, where $\mu_V^1$ is her own expected log return.

Use $(y^2)^{-1} - (y^1)^{-1} = (1-\rho)(\mu_V^1 - \mu_V^2)$ and express $\mu_V^1 - \mu_V^2$ in terms of agent 1's *subjective* expected excess return.

*Hint:* Under agent 1's beliefs, her portfolio earns an extra $(\omega^1 - \omega^2)\sigma_Y + (\omega^1 - \omega^2)^2/(2\gamma)$ in expected log returns relative to agent 2's portfolio.
```

```{solution-start} ex_consumption_wealth
:class: dropdown
```

The consumption-wealth ratio for agent $n$ satisfies $(y^n)^{-1} = \beta - (1-\rho)\mu_V^n$, where $\mu_V^n$ is the expected log return on agent $n$'s wealth under her own subjective measure.

Taking the difference:

$$
(y^2)^{-1} - (y^1)^{-1} = (1-\rho)(\mu_V^1 - \mu_V^2).
$$

Agent 1's subjective expected log portfolio return exceeds agent 2's by the amount she believes she gains from tilting toward the risky asset.

Her extra risky share is $\pi^1 - 1 = (\omega^1 - \omega^2)/(\gamma\sigma_Y)$, and under her subjective measure $Q^1$ the risky asset's expected excess log return is $(\gamma\sigma_Y^2 + (\omega^1 - \omega^2)\sigma_Y - \omega^2\sigma_Y) - r - \frac{1}{2}\sigma_Y^2$.

After simplification, the subjective expected log return difference is

$$
\mu_V^1 - \mu_V^2 = (\omega^1 - \omega^2)\sigma_Y + \frac{(\omega^1 - \omega^2)^2}{2\gamma}.
$$

Substituting and dividing through by $\rho$ (from the relationship between $(y^n)^{-1}$ and $\beta$):

$$
(y^2)^{-1} - (y^1)^{-1} = \frac{1-\rho}{\rho}\left[(\omega^1 - \omega^2)\sigma_Y + \frac{(\omega^1 - \omega^2)^2}{2\gamma}\right].
$$

```{solution-end}
```

### Two comparative statics

Survival depends on $\gamma$, $\rho$, and the signal-to-noise ratios $\omega^1 / \sigma_Y$ and $\omega^2 / \sigma_Y$, not on $\omega^1$, $\omega^2$, and $\sigma_Y$ separately.

The survival conditions do not depend on $\beta$ or $\mu_Y$, which affect the level of consumption and prices but not relative wealth dynamics at the boundary.

```{code-cell} ipython3
def portfolio_return_diff(ω_1, ω_2, γ, σ_y):
    """
    Difference in expected log portfolio returns at the boundary.
    """
    Δω = ω_1 - ω_2
    risky_share_diff = Δω / (γ * σ_y)
    risk_premium = γ * σ_y**2 - ω_2 * σ_y
    volatility_term = (Δω / γ) * (σ_y + 0.5 * Δω / γ)
    return risky_share_diff * risk_premium - volatility_term


def saving_channel(ω_1, ω_2, γ, ρ, σ_y):
    """
    Difference in consumption-wealth ratios at the boundary.
    """
    Δω = ω_1 - ω_2
    subjective_return_diff = Δω * σ_y + Δω**2 / (2 * γ)
    return (1 - ρ) / ρ * subjective_return_diff


def boundary_drift(ω_1, ω_2, γ, ρ, σ_y):
    """
    Boundary drift m_ϑ when agent 1 becomes negligible.

    Positive drift means agent 1 survives (repelling boundary).
    """
    return γ * (
        portfolio_return_diff(ω_1, ω_2, γ, σ_y)
        + saving_channel(ω_1, ω_2, γ, ρ, σ_y)
    )
```

## Survival regions

A central contribution of {cite:t}`Borovicka2020` is the characterization of survival regions in the $(\gamma, \rho)$ plane.

Under separable preferences, $\gamma = \rho$, the agent with more accurate beliefs always dominates.

Under recursive preferences, all four outcomes in {prf:ref}`survival_conditions` can occur.

Figure 2 in the paper studies the case where agent 2 has correct beliefs, so $\omega^2 = 0$.

The next cell follows that figure.

```{code-cell} ipython3
def compute_survival_boundary(ω_1, ω_2, σ_y, γ_grid, boundary="lower"):
    """
    Compute the curve in (γ, ρ) space where the boundary drift is zero.

    For boundary='lower', agent 1 is the small agent.
    For boundary='upper', agent 2 is the small agent.
    """
    ρ_boundary = []

    if boundary == "lower":
        small_agent = (ω_1, ω_2)
    else:
        small_agent = (ω_2, ω_1)

    ω_small, ω_large = small_agent

    for γ in γ_grid:
        pr = portfolio_return_diff(ω_small, ω_large, γ, σ_y)
        Δω = ω_small - ω_large
        subj_ret = Δω * σ_y + Δω**2 / (2 * γ)

        if abs(subj_ret) < 1e-14:
            ρ_boundary.append(np.nan)
            continue

        denom = subj_ret - pr
        if abs(denom) < 1e-14:
            ρ_boundary.append(np.nan)
        else:
            ρ_boundary.append(subj_ret / denom)

    return np.asarray(ρ_boundary)


def compute_limit_boundary(γ_grid, boundary="lower"):
    """
    Boundary curves for the limit |ω_1| / σ_y -> ∞.

    This is equivalent to the constant-endowment case discussed in the paper.
    """
    if boundary == "lower":
        return γ_grid / (1 + γ_grid)

    ρ = np.full_like(γ_grid, np.nan, dtype=float)
    mask = γ_grid < 1
    ρ[mask] = γ_grid[mask] / (1 - γ_grid[mask])
    return ρ
```

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Survival regions corresponding to Figure 2 in Borovicka (2020)
    name: fig-survival-regions
---
σ_y = 0.02
γ_vals = np.linspace(0.01, 6.0, 500)
ρ_vals = np.linspace(0.01, 2.0, 400)
G, R = np.meshgrid(γ_vals, ρ_vals)

panel_specs = [
    ("finite", 0.10, r"$\omega^1 = 0.10$"),
    ("finite", 0.20, r"$\omega^1 = 0.20$"),
    ("limit", None, r"$|\omega^1| / \sigma_Y \to \infty$"),
    ("finite", -0.25, r"$\omega^1 = -0.25$"),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)

for idx, (case, value, label) in enumerate(panel_specs):
    ax = axes.flat[idx]

    if case == "limit":
        ρ_1 = compute_limit_boundary(γ_vals, boundary="lower")
        ρ_2 = compute_limit_boundary(γ_vals, boundary="upper")
        # Limit boundary drifts: use closed-form expressions
        # m0 > 0 (agent 1 survives) when ρ < γ/(1+γ)
        m0 = G - (1 + G) * R
        # m1 < 0 (agent 2 survives) when ρ < γ/(1-γ) for γ<1, always for γ>=1
        m1 = (1 - G) * R - G
    else:
        ρ_1 = compute_survival_boundary(value, 0.0, σ_y, γ_vals,
                                        boundary="lower")
        ρ_2 = compute_survival_boundary(value, 0.0, σ_y, γ_vals,
                                        boundary="upper")
        # Evaluate boundary drifts on the grid
        m0 = boundary_drift(value, 0.0, G, R, σ_y)
        m1 = -boundary_drift(0.0, value, G, R, σ_y)

    # Classify all four regions
    both = (m0 > 0) & (m1 < 0)
    ag1_dom = (m0 > 0) & (m1 > 0)
    ag2_dom = (m0 < 0) & (m1 < 0)
    either = (m0 < 0) & (m1 > 0)

    # Shade coexistence region
    ax.contourf(G, R, both.astype(float), levels=[0.5, 1.5],
                colors=["C2"], alpha=0.18)
    if idx == 0:
        ax.fill_between([], [], color="C2", alpha=0.18,
                        label="both survive")

    # Plot boundary curves
    ax.contour(G, R, m0, levels=[0], colors=["C0"],
               linestyles="--", linewidths=2)
    ax.contour(G, R, m1, levels=[0], colors=["C3"],
               linestyles="-", linewidths=2)
    if idx == 0:
        ax.plot([], [], "--", color="C0", lw=2, label="agent 1 boundary")
        ax.plot([], [], "-", color="C3", lw=2, label="agent 2 boundary")

    ax.plot(
        γ_vals, γ_vals, ":", color="black", lw=2,
        label=r"$\gamma = \rho$" if idx == 0 else None
    )

    tkw = dict(ha="center", va="center", style="italic", color="0.15")
    if case == "finite" and value == 0.10:
        ax.text(0.31, 1.05, "either agent dominates", rotation=90,
                fontsize=10, **tkw)
        ax.text(1.8, 1.55, "agent 2\ndominates", fontsize=11, **tkw)
        ax.text(3.5, 0.75, "both\nsurvive", fontsize=11, **tkw)
        if ag1_dom.any():
            ax.text(5.0, 0.25, "agent 1\ndominates", fontsize=11, **tkw)
    elif case == "finite" and value == 0.20:
        ax.text(0.31, 1.05, "either agent dominates", rotation=90,
                fontsize=10, **tkw)
        ax.text(2.5, 1.55, "agent 2\ndominates", fontsize=11, **tkw)
        ax.text(3.8, 0.55, "both\nsurvive", fontsize=11, **tkw)
        if ag1_dom.any():
            ax.text(5.2, 0.08, "agent 1\ndominates", fontsize=9, **tkw)
    elif case == "limit":
        ax.text(0.31, 1.05, "either agent dominates", rotation=90,
                fontsize=10, **tkw)
        ax.text(3.0, 1.40, "agent 2\ndominates", fontsize=11, **tkw)
        ax.text(3.5, 0.30, "both\nsurvive", fontsize=11, **tkw)
    elif case == "finite" and value == -0.25:
        ax.text(0.31, 1.05, "either agent dominates", rotation=90,
                fontsize=10, **tkw)
        ax.text(3.5, 1.20, "agent 2\ndominates", fontsize=11, **tkw)
        ax.text(2.5, 0.18, "both\nsurvive", fontsize=11, **tkw)

    ax.set_title(label, fontsize=12)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 2)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\rho$")

axes[0, 0].legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()
```

Each panel plots two curves in the $(\gamma, \rho)$ plane for a different value of agent 1's belief distortion $\omega^1$ (agent 2 has correct beliefs, $\omega^2 = 0$).

- The dashed curve (blue) is where the boundary drift at $\upsilon = 0$ equals zero -- condition (i) in {prf:ref}`survival_conditions`.
- The solid curve (red) is where the boundary drift at $\upsilon = 1$ equals zero -- condition (ii).
- The shaded region between the two curves is where both agents survive.
- The dotted diagonal $\gamma = \rho$ is the separable CRRA case, along which the agent with more accurate beliefs always dominates.

Moderate optimism ($\omega^1 = 0.10$) produces a wide coexistence region that extends across most of the $\gamma$ range.

Stronger optimism ($\omega^1 = 0.20$) narrows the region: the agent 2 boundary shifts out of the plotted range for moderate and large $\gamma$, shrinking the set of $(\gamma, \rho)$ pairs where both agents coexist.

In the limit $|\omega^1|/\sigma_Y \to \infty$ (bottom-left), the boundaries simplify to closed-form expressions.

The coexistence region narrows but extends to large $\gamma$ values below the agent 2 boundary curve.

Pessimistic distortions ($\omega^1 = -0.25$, bottom-right) can also survive, but only in a much narrower part of the parameter space.

## Three survival channels

The decomposition above can be visualized directly.

```{code-cell} ipython3
def decompose_survival(ω_1, ω_2, γ_grid, ρ, σ_y):
    """
    Decompose the wealth-growth differential in proposition 3.4.
    """
    Δω = ω_1 - ω_2
    risk_premium_term = Δω * (γ_grid * σ_y - ω_2) / γ_grid
    volatility_term = -(Δω / γ_grid) * (σ_y + 0.5 * Δω / γ_grid)
    saving_term = (1 - ρ) / ρ * (Δω * σ_y + Δω**2 / (2 * γ_grid))
    total = risk_premium_term + volatility_term + saving_term
    return risk_premium_term, volatility_term, saving_term, total


ω_1 = 0.25
ω_2 = 0.0
ρ = 0.67
σ_y = 0.02
γ_grid = np.linspace(0.5, 15.0, 300)

risk_term, vol_term, save_term, total = decompose_survival(
    ω_1, ω_2, γ_grid, ρ, σ_y
)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(γ_grid, risk_term, color="C0", lw=2, label="risk premium term")
ax.plot(γ_grid, vol_term, "--", color="C3", lw=2, label="volatility term")
ax.plot(γ_grid, save_term, "-.", color="C2", lw=2, label="saving term")
ax.plot(γ_grid, total, color="black", lw=2, label="total")
ax.axhline(0, color="gray", lw=1)
ax.set_xlabel(r"risk aversion $\gamma$")
ax.set_ylabel("contribution to wealth-growth differential")
ax.legend()
plt.tight_layout()
plt.show()
```

This figure decomposes the boundary drift at $\upsilon = 0$ into three terms for an optimistic agent ($\omega^1 = {0.25}$, $\omega^2 = 0$) with IES $= 1/\rho \approx 1.49$ and $\sigma_Y = 0.02$.

- The risk premium term (blue) is positive throughout because the optimistic agent overweights the risky asset and earns the equity premium.
- The volatility term (red dashed) is negative and large at low $\gamma$, reflecting the cost of holding a volatile portfolio.
- The saving term (green dash-dot) is positive when IES $> 1$ because the optimistic agent perceives a high return on wealth and saves more aggressively.
- The total (black) crosses zero at the critical $\gamma$ below which the volatility penalty dominates and the agent cannot survive.

## Varying the IES

The sign of the saving term is pinned down by the IES.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Boundary decomposition for different IES values
    name: fig-survival-ies-panels
---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

ω_1 = 0.25
ω_2 = 0.0
σ_y = 0.02
γ_grid = np.linspace(0.5, 25.0, 300)

ies_values = [0.5, 1.0, 1.5]

for idx, ies in enumerate(ies_values):
    ρ = 1.0 / ies
    risk_term, vol_term, save_term, total = decompose_survival(
        ω_1, ω_2, γ_grid, ρ, σ_y
    )

    ax = axes[idx]
    ax.plot(γ_grid, risk_term, color="C0", lw=2, label="risk premium")
    ax.plot(γ_grid, vol_term, "--", color="C3", lw=2, label="volatility")
    ax.plot(γ_grid, save_term, "-.", color="C2", lw=2, label="saving")
    ax.plot(γ_grid, total, color="black", lw=2, label="total")
    ax.axhline(0, color="gray", lw=1)
    ax.set_title(f"IES = {ies:.1f}", fontsize=12)
    ax.set_xlabel(r"risk aversion $\gamma$")
    ax.set_ylabel("contribution")

axes[0].legend(fontsize=9)
plt.tight_layout()
plt.show()
```

Each panel shows the same three-term decomposition as the previous figure, but now for three different values of the IES ($\omega^1 = 0.25$, $\omega^2 = 0$, $\sigma_Y = 0.02$).

- Left panel (IES $= 0.5$): the saving term is negative, so the optimistic agent actually saves less, working against survival.
- Center panel (IES $= 1.0$): the saving term vanishes entirely, so only the portfolio return and volatility channels remain. 

    - This eliminates the saving channel but does not by itself reproduce the full separable CRRA benchmark, which requires $\gamma = \rho$ (i.e., IES $= 1/\gamma$), not merely $\rho = 1$.
- Right panel (IES $= 1.5$): the saving term is positive and shifts the total drift upward, expanding the range of $\gamma$ values for which the optimistic agent survives.

## Asymptotic results

Borovicka derives several useful asymptotic results.

1. As $\gamma \searrow 0$, each agent dominates with strictly positive probability.
1. As $\gamma \nearrow \infty$, the relatively more optimistic agent dominates.
1. As $\rho \searrow 0$, the relatively more optimistic agent always survives.
   - The relatively more pessimistic agent can also survive when risk aversion is sufficiently low.
1. As $\rho \nearrow \infty$, a nondegenerate long-run equilibrium cannot exist.

The next figure illustrates the first result by plotting both boundary drifts as $\gamma$ becomes small.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Boundary drifts for small risk aversion
    name: fig-boundary-drifts-small-gamma
---
ω_1 = 0.25
ω_2 = 0.0
ρ = 0.67
σ_y = 0.02
γ_grid = np.linspace(0.05, 5.0, 300)

drift_at_0 = np.array([boundary_drift(ω_1, ω_2, γ, ρ, σ_y) for γ in γ_grid])
drift_at_1 = np.array([-boundary_drift(ω_2, ω_1, γ, ρ, σ_y) for γ in γ_grid])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(γ_grid, drift_at_0, color="C0", lw=2, label=r"$\upsilon \to 0$")
ax.plot(γ_grid, drift_at_1, "--", color="C3", lw=2, label=r"$\upsilon \to 1$")
ax.axhline(0, color="gray", lw=1)
ax.set_xlabel(r"risk aversion $\gamma$")
ax.set_ylabel("boundary drift")
ax.legend()
plt.tight_layout()
plt.show()
```

This figure plots the two boundary drifts as a function of $\gamma$ ($\omega^1 = 0.25$, $\omega^2 = 0$, IES $\approx 1.49$).

- The solid blue curve is the drift $m_\vartheta$ at $\upsilon \to 0$ (agent 1 near extinction); coexistence requires this to be positive (condition (i)).
- The dashed red curve is the drift $m_\vartheta$ at $\upsilon \to 1$ (agent 2 near extinction); coexistence requires this to be negative (condition (ii)).

The figure illustrates asymptotic result 1.

For small $\gamma$, the blue curve is negative and the red curve is positive.

Both boundaries are attracting: near $\upsilon = 0$ the negative drift pulls $\upsilon$ toward 0, and near $\upsilon = 1$ the positive drift pushes $\upsilon$ toward 1.

This is outcome (d) in {prf:ref}`survival_conditions`: neither boundary is repelling, so whichever agent happens to get ahead early will dominate, with each agent having strictly positive probability of dominance depending on the realized Brownian path.

As $\gamma$ increases past roughly 1, the blue curve crosses zero and becomes positive while the red curve stays negative.

Now both boundaries are repelling and we enter the coexistence region -- outcome (a).

## The separable case

When $\gamma = \rho$, the model collapses to the separable CRRA benchmark.

In that case, the log-odds process becomes

$$
d\vartheta_t = \frac{1}{2}\left[(\omega^2)^2 - (\omega^1)^2\right] dt + (\omega^1 - \omega^2) dW_t .
$$

The drift is constant and depends only on the relative entropy of the two belief distortions.

The agent with the smaller $|\omega^n|$ dominates under $P$.

If the two agents have equal magnitudes of belief distortions, neither becomes extinct almost surely, but no nondegenerate stationary wealth distribution exists.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Pareto-share paths in the separable benchmark
    name: fig-crra-pareto-paths
---
def simulate_crra_pareto(ω_1, ω_2, T, dt, n_paths, seed=42):
    """
    Simulate Pareto-share dynamics in the separable benchmark.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    t_grid = np.linspace(0, T, n_steps + 1)

    drift = 0.5 * (ω_2**2 - ω_1**2)
    volatility = ω_1 - ω_2

    θ = np.zeros((n_paths, n_steps + 1))
    dW = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))

    for t in range(n_steps):
        θ[:, t + 1] = θ[:, t] + drift * dt + volatility * dW[:, t]

    υ_paths = 1.0 / (1.0 + np.exp(-θ))
    return t_grid, υ_paths


ω_1 = 0.10
ω_2 = 0.0
t_grid, υ_paths = simulate_crra_pareto(ω_1, ω_2, T=200, dt=0.01, n_paths=50)

fig, ax = plt.subplots(figsize=(11, 5))

for i in range(20):
    ax.plot(t_grid, υ_paths[i], color="C0", alpha=0.25, lw=1)

ax.axhline(0.5, color="gray", linestyle=":", lw=1)
ax.set_xlabel("time")
ax.set_ylabel(r"Pareto share $\upsilon_t$")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
```

This figure simulates 20 sample paths of the Pareto share $\upsilon_t$ under separable CRRA preferences ($\gamma = \rho$) with $\omega^1 = 0.10$ and $\omega^2 = 0$.

Agent 2 has correct beliefs, so the log-odds drift is negative and all paths trend toward $\upsilon = 0$.

Agent 1 is driven to extinction -- the classical market-selection result of {cite:t}`Blume_Easley2006`.

## Asset pricing implications

As one agent becomes negligible, current prices converge to those of the homogeneous economy populated by the large agent.

When agent 2 is the large agent, Proposition 5.1 in {cite:t}`Borovicka2020` implies

$$
\lim_{\upsilon \searrow 0} r(\upsilon)
= \beta + \rho \left(\mu_Y + \omega^2 \sigma_Y
+ \frac{1}{2} (1 - \gamma) \sigma_Y^2\right)
- \frac{1}{2} \gamma \sigma_Y^2
$$ (eq:riskfree)

and

$$
\lim_{\upsilon \searrow 0} y(\upsilon)
= \left[
\beta - (1 - \rho)
\left(
\mu_Y + \omega^2 \sigma_Y + \frac{1}{2} (1 - \gamma) \sigma_Y^2
\right)
\right]^{-1} .
$$ (eq:wc_ratio)

The aggregate wealth dynamics also converge to those of the homogeneous economy:

$$
\lim_{\upsilon \searrow 0} m_A(\upsilon) = \mu_Y,
\qquad
\lim_{\upsilon \searrow 0} \sigma_A(\upsilon) = \sigma_Y .
$$

Proposition 5.3 then gives the negligible agent's own consumption-saving and portfolio choices.

Her consumption-wealth ratio converges to

$$
\lim_{\upsilon \searrow 0} (y^1(\upsilon))^{-1}
= (y(0))^{-1}
- \frac{1-\rho}{\rho}
\left[
(\omega^1 - \omega^2)\sigma_Y
+ \frac{(\omega^1 - \omega^2)^2}{2 \gamma}
\right] .
$$

The small agent's risky-asset share converges to

$$
\lim_{\upsilon \searrow 0} \pi^1(\upsilon)
= 1 + \frac{\omega^1 - \omega^2}{\gamma \sigma_Y} .
$$ (eq:portfolio)

Hence optimism implies leverage, while sufficiently strong pessimism implies shorting.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Limiting risky-asset shares of the small agent
    name: fig-limiting-portfolio-shares
---
ω_2 = 0.0
σ_y = 0.02
ω_grid = np.linspace(-0.5, 1.0, 300)

fig, ax = plt.subplots(figsize=(10, 5))

for γ in [2, 5, 10, 20]:
    π_1 = 1 + (ω_grid - ω_2) / (γ * σ_y)
    ax.plot(ω_grid, π_1, lw=2, label=rf"$\gamma = {γ}$")

ax.axhline(1.0, color="gray", linestyle=":", lw=1)
ax.axhline(0.0, color="gray", linestyle=":", lw=1)
ax.axvline(0.0, color="gray", linestyle=":", lw=1)
ax.set_xlabel(r"belief distortion $\omega^1$")
ax.set_ylabel(r"risky share $\pi^1$")
ax.legend()
plt.tight_layout()
plt.show()
```

This figure plots the limiting risky-asset share $\pi^1$ of the negligible agent as a function of her belief distortion $\omega^1$ ($\omega^2 = 0$, $\sigma_Y = 0.02$), for four levels of risk aversion.

At $\omega^1 = 0$ the agent agrees with agent 2 and holds the market portfolio ($\pi^1 = 1$).

Optimism ($\omega^1 > 0$) leads to leverage ($\pi^1 > 1$), while sufficient pessimism ($\omega^1 < 0$) leads to shorting ($\pi^1 < 0$).

Higher risk aversion compresses these deviations toward one.

## Optimistic and pessimistic distortions

Optimistic and pessimistic beliefs affect survival asymmetrically.

An optimistic agent benefits from the risk premium term and, when IES $> 1$, from the saving term as well.

A pessimistic agent gives up the risk premium and can survive only if the saving effect is strong enough to offset that loss.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Total boundary drift for optimistic and pessimistic distortions
    name: fig-optimistic-pessimistic-drifts
---
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

σ_y = 0.02
ω_2 = 0.0
ρ = 0.67
γ_grid = np.linspace(0.5, 25.0, 300)

ax = axes[0]
for ω_1 in [0.1, 0.25, 0.5, 1.0]:
    _, _, _, total = decompose_survival(ω_1, ω_2, γ_grid, ρ, σ_y)
    ax.plot(γ_grid, total, lw=2, label=rf"$\omega^1 = {ω_1}$")
ax.axhline(0, color="gray", lw=1)
ax.set_title("optimistic", fontsize=12)
ax.set_xlabel(r"risk aversion $\gamma$")
ax.set_ylabel("boundary drift")
ax.legend(fontsize=9)

ax = axes[1]
for ω_1 in [-0.1, -0.25, -0.5, -1.0]:
    _, _, _, total = decompose_survival(ω_1, ω_2, γ_grid, ρ, σ_y)
    ax.plot(γ_grid, total, lw=2, label=rf"$\omega^1 = {ω_1}$")
ax.axhline(0, color="gray", lw=1)
ax.set_title("pessimistic", fontsize=12)
ax.set_xlabel(r"risk aversion $\gamma$")
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()
```

Both panels plot the total boundary drift at $\upsilon = 0$ as a function of $\gamma$ (IES $\approx 1.49$, $\omega^2 = 0$).

Where the curve is positive, agent 1 survives near extinction.

- Left panel (optimistic agent): larger $\omega^1$ means a bigger bet on the risky asset, so the volatility penalty dominates at low $\gamma$ but the drift turns positive once $\gamma$ is large enough.
- Right panel (pessimistic agent): a pessimistic agent gives up the risk premium by underweighting the risky asset, so the drift is negative for most of the parameter space and survival requires saving motives strong enough to offset the portfolio losses.

## Long-run consumption distribution

When both agents survive, the Pareto share keeps moving across the whole interval $(0, 1)$.

The next simulation is only a toy approximation.

It interpolates the drift between its two boundary values, so it illustrates the recurrence logic without solving the full equilibrium ODE.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: A toy stationary Pareto-share simulation
    name: fig-toy-stationary-pareto-share
---
def simulate_pareto_share_toy(ω_1, ω_2, γ, ρ, σ_y, T, dt, n_paths=20, seed=42):
    """
    Simulate a toy Pareto-share process by interpolating boundary drifts.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    t_grid = np.linspace(0, T, n_steps + 1)

    volatility = ω_1 - ω_2
    m_0 = boundary_drift(ω_1, ω_2, γ, ρ, σ_y)
    m_1 = -boundary_drift(ω_2, ω_1, γ, ρ, σ_y)

    θ = np.zeros((n_paths, n_steps + 1))
    dW = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))

    for t in range(n_steps):
        υ = 1.0 / (1.0 + np.exp(-θ[:, t]))
        drift = m_0 * (1 - υ) + m_1 * υ
        θ[:, t + 1] = θ[:, t] + drift * dt + volatility * dW[:, t]

    υ_paths = 1.0 / (1.0 + np.exp(-θ))
    return t_grid, υ_paths


ω_1 = 0.25
ω_2 = 0.0
γ = 5.0
ρ = 0.67
σ_y = 0.02

t_grid, υ_paths = simulate_pareto_share_toy(
    ω_1, ω_2, γ, ρ, σ_y, T=500, dt=0.05, n_paths=50, seed=42
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for i in range(20):
    ax.plot(t_grid, υ_paths[i], color="C0", alpha=0.25, lw=1)
ax.axhline(0.5, color="gray", linestyle=":", lw=1)
ax.set_title("sample paths", fontsize=12)
ax.set_xlabel("time")
ax.set_ylabel(r"Pareto share $\upsilon_t$")
ax.set_ylim(0, 1)

ax = axes[1]
_, υ_long = simulate_pareto_share_toy(
    ω_1, ω_2, γ, ρ, σ_y, T=2000, dt=0.05, n_paths=5, seed=123
)
υ_stationary = υ_long[:, υ_long.shape[1] // 2:].ravel()
ax.hist(υ_stationary, bins=80, density=True, color="steelblue",
        edgecolor="white", alpha=0.7)
ax.set_title("approximate stationary density", fontsize=12)
ax.set_xlabel(r"Pareto share $\upsilon$")
ax.set_ylabel("density")
ax.set_xlim(0, 1)

plt.tight_layout()
plt.show()
```

The left panel shows 20 sample paths of the Pareto share $\upsilon_t$ under parameters inside the coexistence region ($\omega^1 = 0.25$, $\omega^2 = 0$, $\gamma = 5$, IES $\approx 1.49$).

Unlike the separable case in {numref}`fig-crra-pareto-paths`, the paths do not drift to zero -- they repeatedly visit a wide range of values, bouncing between the two repelling boundaries.

The right panel approximates the stationary density by pooling the second half of longer simulations.

The interior mode is consistent with neither agent being driven to extinction.

However, this toy interpolation only illustrates the recurrence logic; it does not reproduce the quantitative stationary consumption-share density in Figure 4 of {cite:t}`Borovicka2020`, which requires solving the full interior equilibrium ODE.

## Summary

Recursive preferences weaken the classical market-selection result.

The portfolio return channel still rewards more optimistic beliefs.

The volatility channel still penalizes aggressive positions.

But when IES $> 1$, the saving channel can be strong enough to keep a distorted-belief agent alive.

This is why recursive-preference economies can support stationary long-run wealth distributions with persistent heterogeneity in beliefs and portfolio positions.
