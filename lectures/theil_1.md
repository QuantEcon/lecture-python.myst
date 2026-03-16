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

(certainty_equiv_robustness)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Certainty Equivalence 

```{index} single: Certainty Equivalence; Robustness
```

```{index} single: LQ Control; Permanent Income
```

```{contents} Contents
:depth: 2
```


In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install quantecon
```


## The Central Problem of Empirical Economics

The papers collected in {cite}`lucas1981rational` address a single overarching question: given observations on an agent's behavior in a particular economic environment, what can we infer about how that behavior **would have differed** had the environment been altered? This is the problem of policy-invariant structural inference.

The difficulty is immediate. Observations arise under one environment; we wish to predict behavior under another. Unless we understand *why* the agent behaves as he does—that is, unless we recover the deep objectives that rationalize observed decisions—estimated behavioral relationships are silent on this question.

---

## A Formal Setup

Consider a single decision maker whose situation at date $t$ is fully described by two state variables $(x_t, z_t)$.

**The environment** $z_t \in S_1$ is selected by "nature" and evolves exogenously according to

```{math}
:label: eq:z_transition
z_{t+1} = f(z_t,\, \epsilon_t),
```

where the innovations $\epsilon_t \in \mathcal{E}$ are i.i.d. draws from a fixed c.d.f. $\Phi(\cdot) : \mathcal{E} \to [0,1]$. The function $f : S_1 \times \mathcal{E} \to S_1$ is called the **decision maker's environment**.

**The endogenous state** $x_t \in S_2$ is under partial control of the agent. Each period the agent selects an action $u_t \in U$. A fixed technology $g : S_1 \times S_2 \times U \to S_2$ governs the transition

```{math}
:label: eq:x_transition
x_{t+1} = g(z_t,\, x_t,\, u_t).
```

**The decision rule** $h : S_1 \times S_2 \to U$ maps the agent's current situation into an action:

```{math}
:label: eq:decision_rule
u_t = h(z_t,\, x_t).
```

The econometrician observes (some or all of) the process $\{z_t, x_t, u_t\}$, the joint motion of which is determined by {eq}`eq:z_transition`, {eq}`eq:x_transition`, and {eq}`eq:decision_rule`.

---

## The Lucas Critique: Why Estimated Rules Are Not Enough

Suppose we have estimated $f$, $g$, and $h$ from a long time series generated under a fixed environment $f_0$. This gives us $h_0 = T(f_0)$, where $T$ is the (unknown) functional mapping environments into optimal decision rules. But this single estimate, however precise, **reveals nothing** about how $T(f)$ varies with $f$.

Policy evaluation requires knowledge of the entire map $f \mapsto T(f)$. Under an environment change $f_0 \to f_1$, agents will in general revise their decision rules $h_0 \to h_1 = T(f_1)$, rendering the estimated rule $h_0$ invalid for forecasting behavior under $f_1$.

The only nonexperimental path forward is to recover the **return function** $V$ from which $h$ is derived as the solution to an optimization problem, and then re-solve that problem under the counterfactual environment $f_1$.

---

##  An Optimization Problem

Assume the agent selects $h$ to maximize the expected discounted sum of current-period returns $V : S_1 \times S_2 \times U \to \mathbb{R}$:

```{math}
:label: eq:objective
E_0\!\left\{\sum_{t=0}^{\infty} \beta^t\, V(z_t,\, x_t,\, u_t)\right\}, \qquad 0 < \beta < 1,
```

given initial conditions $(z_0, x_0)$, the environment $f$, and the technology $g$. Here $E_0\{\cdot\}$ denotes expectation conditional on $(z_0, x_0)$ with respect to the distribution of $\{z_1, z_2, \ldots\}$ induced by {eq}`eq:z_transition`.

In principle, knowledge of $V$ (together with $g$ and $f$) allows one to compute $h = T(f)$ theoretically and hence to trace out $T(f)$ for any counterfactual $f$. The empirical question is whether $V$ can itself be recovered from observations on $\{f, g, h\}$—a problem of structural identification that, at this level of generality, is formidably difficult.

:::{note}
The decision rule is in general a functional $h = T(f, g, V)$. The dependence on $g$ and $V$ is suppressed in the main text but made explicit when needed.
:::

---

## A Linear-Quadratic Specialization and Certainty Equivalence

Progress at the level of generality of Section 4 requires restricting the primitives. The most productive restriction, exploited in the bulk of the volume, imposes **quadratic** $V$ and **linear** $g$, which forces $h$ to be linear. Beyond computational tractability, this specialization delivers a striking structural result: the **certainty equivalence** theorem of Simon {cite}`simon1956dynamic`  and Theil {cite}`theil1957note`. 

###  The Composite Decomposition of $h$

Under quadratic $V$ and linear $g$, the optimal decision rule $h$ decomposes into two components applied in sequence.

**Step 1 — Forecasting.** Define the infinite sequence of optimal point forecasts of all current and future states of nature:

```{math}
:label: eq:forecast_sequence
\tilde{z}_t \;=\; \bigl(z_t,\;\; {}_{t+1}z_t^e,\;\; {}_{t+2}z_t^e,\;\ldots\bigr) \;\in\; S_1^\infty,
```

where ${}_{t+j}z_t^e$ denotes the least-mean-squared-error forecast of $z_{t+j}$ formed at time $t$. The optimal forecast sequence is a (generally nonlinear) function of the current state:

```{math}
:label: eq:forecast_rule
\tilde{z}_t = h_2(z_t).
```

The function $h_2 : S_1 \to S_1^\infty$ depends entirely on the environment $(f, \Phi)$ and is obtained as the solution to a **pure forecasting problem**, with no reference to preferences or technology.

**Step 2 — Optimization.** Given the forecast sequence $\tilde{z}_t$, the optimal action is a **linear** function of $\tilde{z}_t$ and $x_t$:

```{math}
:label: eq:optimization_rule
u_t = h_1(\tilde{z}_t,\, x_t).
```

The function $h_1 : S_1^\infty \times S_2 \to U$ depends entirely on preferences $(V)$ and technology $(g)$ but **not** on the stochastic environment $(f, \Phi)$.

The full decision rule is therefore the **composite**:

```{math}
:label: eq:composite_rule
\boxed{h(z_t, x_t) \;=\; h_1\!\bigl[h_2(z_t),\; x_t\bigr].}
```

###  The Separation Principle

{eq}`eq:composite_rule` embodies a clean **separation** of the two sources of dependence in $h$:

| Component | Depends on | Independent of |
|-----------|-----------|----------------|
| $h_1$ (optimization) | $V$, $g$ | $f$, $\Phi$ |
| $h_2$ (forecasting)  | $f$, $\Phi$ | $V$, $g$ |

Since policy analysis concerns changes in $f$, and since $h_1$ is invariant to $f$, the policy analyst need only re-solve the forecasting problem $h_2 = S(f)$ under the new environment, keeping $h_1$ fixed. The relationship of original interest, $h = T(f)$, then follows directly from {eq}`eq:composite_rule`.

###  Certainty Equivalence and Perfect Foresight

The name "certainty equivalence" reflects a further implication of the LQ structure: the function $h_1$ can be derived as if the agent **knew the future path $z_{t+1}, z_{t+2}, \ldots$ with certainty** — i.e., by solving the deterministic problem in which $\tilde{z}_t$ is treated as the realized path rather than a forecast. The stochasticity of the environment affects actions only through the forecast $\tilde{z}_t$; conditional on $\tilde{z}_t$, the optimization problem is deterministic.

This means the LQ problem decouples into:

 *  **Dynamic optimization under perfect foresight** — solve for $h_1$ from $(V, g)$ by treating $\tilde{z}_t$ as known. This is a standard deterministic LQ regulator problem and is independent of the environment $(f, \Phi)$.

 *  **Optimal linear prediction** — solve for $h_2 = S(f)$ from $(f, \Phi)$ using least-squares forecasting theory. If $f$ is itself linear, $h_2$ is also linear and reduces to a standard Kalman/Wiener prediction formula.

###  Cross-Equation Restrictions

A hallmark of the rational expectations hypothesis as it appears in this framework is that it ties together what would otherwise be free parameters in different equations. The requirement that $\tilde{z}_t = h_2(z_t) = S(f)(z_t)$ — i.e., that agents' forecasts be *optimal* with respect to the *actual* law of motion $f$ — imposes **cross-equation restrictions** between the parameters of the forecasting rule $h_2$ and the parameters of the environment $f$. These restrictions, rather than any conditions on distributed lags within a single equation, are the operative empirical content of rational expectations.

---

##  A Trouble with  Ad Hoc Expectations 

Prior practice, exemplified by the adaptive expectations mechanisms of Friedman {cite}`Friedman1956` and Cagan {cite}`Cagan`, directly postulated a particular form of {eq}`eq:forecast_rule`:

```{math}
:label: eq:adaptive_expectations
\theta_t^e = \lambda \sum_{i=0}^{\infty} (1-\lambda)^i\, \theta_{t-i}, \qquad 0 < \lambda < 1,
```

treating the coefficient $\lambda$ as a free parameter to be estimated from data, with no reference to the underlying environment $f$.

The deficiency is not that {eq}`eq:adaptive_expectations` is a distributed lag — linear forecasting rules are perfectly acceptable simplifications. The deficiency is that the **coefficients** of the distributed lag are left unrestricted by theory. The mapping $h_2 = S(f)$ shows that optimal forecasting coefficients are *determined* by $f$: when $f$ changes, $h_2$ changes, and so does $h$. An estimated $\lambda$ calibrated under $f_0$ is therefore non-structural and will give incorrect predictions whenever $f$ is altered. This is the econometric content of the critique that Muth's paper delivers.

Rational expectations equates the subjective distribution that agents use in forming $\tilde{z}_t$ to the objective distribution $f$ that actually generates the data, thereby closing the model and eliminating free parameters in $h_2$.
