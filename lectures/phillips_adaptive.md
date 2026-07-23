---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(phillips_adaptive)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Adaptive Expectations and the Phelps Problem

```{contents} Contents
:depth: 2
```

In addition to what's in Anaconda, this lecture will use the following library:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

## Overview

> They cannot look out far.
> They cannot look in deep.
> But when was that ever a bar
> To any watch they keep?
>
> -- Robert Frost

This lecture continues the study of Phillips curve tradeoffs begun in {doc}`phillips_credibility`.

It follows chapter 5 of {cite}`Sargent1999`.

We describe

* the Cagan-Friedman adaptive expectations hypothesis,
* how {cite}`Phelps1967` used it to formulate a government control problem, and
* the role of adaptive expectations in early econometric tests of the natural-rate hypothesis.

The key object is the **Phelps problem**: a government that is rational solves an optimal control problem while the public forecasts inflation with a fixed, mechanical adaptive rule.

Unlike the one-period model of {doc}`phillips_credibility`, the government now takes into account that the economy lasts more than one period, and that today's inflation shapes tomorrow's expectations.

This intertemporal link can improve outcomes and, in a limiting case, even sustain the Ramsey outcome.

The Phelps problem is a linear-quadratic (LQ) control problem, so we solve it with the {doc}`LQ control <lqcontrol>` tools from QuantEcon.

Let's start with some imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe
```

## Adaptive expectations

{cite}`Phelps1967` formulated a control problem for a natural-rate model.

He dropped rationality for the public but not for the government, and assigned the public a particular mechanical forecasting rule that is known to the government.

The public uses the adaptive expectations scheme of Milton Friedman and {cite}`Cagan`:

```{math}
:label: pa_adaptive

x_t - x_{t-1} = (1 - \lambda)(y_{t-1} - x_{t-1}), \qquad \lambda \in (0, 1),
```

where $x_t$ is the public's expectation of inflation and $y_t$ is inflation.

Rearranging, $x_t = \lambda x_{t-1} + (1 - \lambda) y_{t-1}$, so expected inflation is a geometric distributed lag of past inflation,

```{math}
:label: pa_geom

x_t = (1 - \lambda) \sum_{i=1}^{\infty} \lambda^{i-1} y_{t-i} .
```

Notice that {eq}`pa_adaptive` is a *constant gain* version of the least squares learning algorithm from {doc}`phillips_credibility`, with the constant $(1 - \lambda)$ playing the role that the decreasing gain $t^{-1}$ played there.

Equation {eq}`pa_geom` possesses an **induction property**: if the government keeps repeating a constant $y_t = \tilde y$ policy, eventually the public comes to set $x_t \approx \tilde y$.

The weights $(1 - \lambda)\lambda^{i-1}$ sum to one, so a permanently maintained inflation rate is eventually expected.

Let's confirm the induction property numerically.

```{code-cell} ipython3
def adaptive_forecast(y, λ, x0=0.0):
    "Simulate x_t = λ x_{t-1} + (1-λ) y_{t-1}."
    T = len(y)
    x = np.empty(T)
    x[0] = x0
    for t in range(1, T):
        x[t] = λ * x[t - 1] + (1 - λ) * y[t - 1]
    return x

T = 60
y_const = np.full(T, 10.0)                 # a constant inflation policy

fig, ax = plt.subplots(figsize=(9, 4.5))
for λ in [0.7, 0.9]:
    x = adaptive_forecast(y_const, λ)
    ax.plot(x, lw=1.5, label=rf'$\lambda = {λ}$')
ax.axhline(10.0, color='k', ls='--', lw=1, label='policy $\\tilde y$')
ax.set_xlabel('$t$')
ax.set_ylabel('expected inflation $x_t$')
ax.legend()
plt.show()
```

Under a constant policy the public's expectation converges to the policy, more slowly the larger is $\lambda$.

Solow and Tobin exploited this induction property when they tested the natural-rate hypothesis, as we discuss below.

## The Phelps problem

The economy repeats forever, and the government evaluates outcome sequences with the discounted criterion

```{math}
:label: pa_criterion

V^g = (1 - \delta) \sum_{t=1}^{\infty} \delta^{t-1} p(U_t, y_t),
\qquad p(U_t, y_t) = - \frac{1}{2}(U_t^2 + y_t^2),
\qquad \delta \in (0, 1] .
```

When $\delta = 1$ we interpret {eq}`pa_criterion` in the limit-of-means (Cesàro) sense.

The government maximizes {eq}`pa_criterion` by choice of a rule for inflation $y_t$, subject to the adaptive expectations scheme {eq}`pa_adaptive` and the expectations-augmented Phillips curve

```{math}
:label: pa_phillips

U_t = U^* - \theta(y_t - x_t) .
```

Because expected inflation $x_t$ is predetermined at the start of period $t$, it is the only endogenous state variable.

The government's problem is therefore a discounted LQ control problem with

* state $s_t = \begin{bmatrix} 1 & x_t \end{bmatrix}'$,
* control $y_t$, and
* transition $x_{t+1} = \lambda x_t + (1 - \lambda) y_t$.

### Casting the problem in LQ form

Write $U_t = a' s_t - \theta y_t$ with $a = \begin{bmatrix} U^* & \theta \end{bmatrix}'$.

The per-period loss $\tfrac{1}{2}(U_t^2 + y_t^2)$ then equals

$$
\frac{1}{2}\left[ s_t'(a a') s_t + (\theta^2 + 1) y_t^2 - 2 \theta \, y_t \, (a' s_t) \right] .
$$

Matching this to the QuantEcon LQ loss $s_t' R s_t + y_t' Q y_t + 2 y_t' N s_t$, and matching the transition to $s_{t+1} = A s_t + B y_t$, gives

$$
R = \tfrac{1}{2} a a', \quad
Q = \tfrac{1}{2}(\theta^2 + 1), \quad
N = -\tfrac{1}{2}\theta\, a', \quad
A = \begin{bmatrix} 1 & 0 \\ 0 & \lambda \end{bmatrix}, \quad
B = \begin{bmatrix} 0 \\ 1 - \lambda \end{bmatrix} .
$$

The discount factor is $\beta = \delta$; the scaling $(1 - \delta)$ in {eq}`pa_criterion` does not affect the optimal policy.

```{code-cell} ipython3
class PhelpsProblem:
    """
    The Phelps optimal-control problem: a rational government facing a
    public that forecasts inflation adaptively with parameter λ.
    """

    def __init__(self, θ=1.0, U_star=5.0, λ=0.7, δ=0.96):
        self.θ, self.U_star, self.λ, self.δ = θ, U_star, λ, δ

        a = np.array([[U_star], [θ]])
        R = 0.5 * (a @ a.T)
        Q = np.array([[0.5 * (θ**2 + 1)]])
        N = -0.5 * θ * a.T
        A = np.array([[1.0, 0.0], [0.0, λ]])
        B = np.array([[0.0], [1 - λ]])

        # δ = 1 (limit of means) is handled as the limit δ → 1
        β = min(δ, 1 - 1e-7)
        self.lq = qe.LQ(Q, R, A, B, N=N, beta=β)
        P, F, d = self.lq.stationary_values()

        # optimal rule y_t = f1 + f2 x_t
        self.f1, self.f2 = -F[0, 0], -F[0, 1]

    def simulate(self, x0=12.0, T=60):
        "Disinflation path (U_t, y_t) starting from expectation x0."
        θ, U_star, λ = self.θ, self.U_star, self.λ
        U, y, x = np.empty(T), np.empty(T), x0
        for t in range(T):
            y[t] = self.f1 + self.f2 * x
            U[t] = U_star - θ * (y[t] - x)
            x = λ * x + (1 - λ) * y[t]
        return U, y
```

The optimal rule takes the form $y_t = f_1 + f_2 x_t$ with $f_1 \neq 0$ and $f_2 \neq 1$.

These inequalities reflect that the public does not use an optimal forecasting rule; if instead $f_1 = 0$ and $f_2 = 1$, we would have $y_t = x_t$ for all histories.

```{code-cell} ipython3
pp = PhelpsProblem(θ=1.0, U_star=5.0, λ=0.7, δ=0.96)
print(f"optimal rule:  y_t = {pp.f1:.3f} + {pp.f2:.3f} x_t")
```

### A proposition

The reason the Phelps problem is interesting is the following result.

```{prf:proposition} δ = 1 eventually sustains Ramsey
:label: pa_prop

In the absence of discounting ($\delta = 1$), the government drives $y_t$ to $0$, the Ramsey outcome.
```

When $\delta = 1$, $\lambda$ governs the speed of convergence to the Ramsey outcome.

When $\delta < 1$, the limit point of $y_t$ depends on a comparison of $\lambda$ with $\delta$.

For $\lambda < \delta$ and $\delta$ close to $1$, the government's policy eventually approximates the Ramsey outcome.

The public's expectations are wrong along the transition path but are correct in the steady state, by virtue of the induction property.

## Disinflation paths

We now reproduce the disinflation experiments of chapter 5 of {cite}`Sargent1999`.

Set $\theta = 1$ and $U^* = 5$, and start the government's problem from late-1970s initial conditions $x_{-1} = y_{-1} = 12$, which imply $U = U^* = 5$.

The following tables record paths of unemployment $U$ and inflation $y$ at selected lags, for two discount factors $\delta \in \{0.96, 1\}$ and two adaptation parameters $\lambda \in \{0.7, 0.9\}$.

```{code-cell} ipython3
def disinflation_table(δ, lags=(1, 5, 20, 50)):
    rows = []
    for λ in [0.7, 0.9]:
        U, y = PhelpsProblem(λ=λ, δ=δ).simulate(x0=12.0, T=max(lags) + 1)
        for lag in lags:
            rows.append((λ, lag, U[lag - 1], y[lag - 1]))
    return rows

for δ in [0.96, 1.0]:
    print(f"\n δ = {δ}")
    print(f" {'λ':>4} {'lag':>4} {'U':>7} {'y':>7}")
    for λ, lag, U, y in disinflation_table(δ):
        print(f" {λ:>4} {lag:>4} {U:>7.1f} {y:>7.1f}")
```

For each parameter setting the government engineers a major recession and immediately brings inflation down more than halfway toward its eventual limiting value.

In the discounted case ($\delta = 0.96$), inflation settles at a positive level, and the government accepts a longer but milder recession when $\lambda = 0.9$ than when $\lambda = 0.7$.

In the undiscounted case ($\delta = 1$), inflation is driven all the way to the Ramsey value of zero, more slowly the larger is $\lambda$.

Let's plot the full disinflation paths.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for δ, ax in zip([0.96, 1.0], axes):
    for λ in [0.7, 0.9]:
        U, y = PhelpsProblem(λ=λ, δ=δ).simulate(x0=12.0, T=60)
        ax.plot(y, lw=1.5, label=rf'$\lambda = {λ}$')
    ax.set_title(rf'$\delta = {δ}$')
    ax.set_xlabel('$t$')
    ax.set_ylabel('inflation $y_t$')
    ax.legend()

plt.tight_layout()
plt.show()
```

The undiscounted problem drives inflation to the Ramsey outcome; the discounted problem stops short of it.

## The general Phelps problem

For the government's control problem, what matters is the *reduced form* of the Phillips curve, not the underlying structure that identifies $x_t$.

It is useful to state a more general version of the Phelps problem in which the government's model is a reduced-form distributed lag Phillips curve.

Define the vectors

$$
X_{U,t} = \begin{bmatrix} U_{t-1} & \cdots & U_{t-m_U} \end{bmatrix}',
\qquad
X_{y,t} = \begin{bmatrix} y_{t-1} & \cdots & y_{t-m_y} \end{bmatrix}',
$$

and the state vector $X_t = \begin{bmatrix} X_{U,t}' & X_{y,t}' & 1 \end{bmatrix}'$, which collects information dated $t-1$ and earlier.

We can write two reduced-form Phillips curves that differ only in their *direction of fit*:

$$
\text{Classical:} \quad U_t = \gamma' X_{C,t} + \varepsilon_{C,t},
\qquad X_{C,t} = \begin{bmatrix} y_t & X_{t-1}' \end{bmatrix}',
$$

$$
\text{Keynesian:} \quad y_t = \beta' X_{K,t} + \varepsilon_{K,t},
\qquad X_{K,t} = \begin{bmatrix} U_t & X_{t-1}' \end{bmatrix}' .
$$

The subscripts $C$ and $K$ stand for *Classical* (regress $U$ on $y$) and *Keynesian* (regress $y$ on $U$).

The general **Phelps problem** is to choose a control law $\hat y_t = h X_{t-1}$ to maximize the expected value of {eq}`pa_criterion` subject to the government's believed Phillips curve and to $y_t = \hat y_t + v_{2t}$, where $v_{2t}$ is a control error.

This induces a mapping from the government's beliefs $\gamma$ to its decision rule $h$:

```{math}
:label: pa_map

h = h(\gamma) .
```

The specific problem solved above is the special case in which $\gamma$ is restricted by substituting the adaptive expectations hypothesis {eq}`pa_adaptive` into the Phillips curve {eq}`pa_phillips`.

Once that substitution is made, the state variable $x_t$ disappears from view.

These objects — $\gamma$, $\beta$, $h(\gamma)$, and the two directions of fit — are exactly the ingredients we will need to define **self-confirming equilibria** in {doc}`phillips_self_confirming`.

```{note}
The **induction hypothesis** is the restriction that, in the Keynesian Phillips curve $y_t = \beta' X_{K,t} + \varepsilon_{K,t}$, the weights on lagged $y$'s sum to unity (equivalently, in the classical form the weights on current and lagged $y$'s sum to zero). Under adaptive expectations this holds because the weights in {eq}`pa_geom` sum to one.
```

## Testing the natural-rate hypothesis

Robert Solow and James Tobin {cite}`Solow1968,Tobin1968` exploited the induction hypothesis to test the natural-rate hypothesis.

Substituting the geometric distributed lag {eq}`pa_geom` into an inverted Phillips curve gives

```{math}
:label: pa_invphill

y_t = (1 - \lambda) \sum_{i=1}^{\infty} \lambda^{i-1} y_{t-i}
      + \theta^{-1}(U^* - U_t) .
```

They proposed to test the natural-rate hypothesis by running the regression

```{math}
:label: pa_tobin

y_t = b_0 + b_1 (1 - \lambda) \sum_{i=1}^{\infty} \lambda^{i-1} y_{t-i}
      + b_2 U_t + \varepsilon_t ,
```

and interpreting a finding of $b_1 < 1$ as evidence of a long-run tradeoff between inflation and unemployment of slope $b_1 - 1$.

Early implementations found $b_1 < 1$ and so rejected the natural-rate hypothesis in favor of a long-run tradeoff.

{cite}`KingWatson1994` and others later argued that the pattern of rejections and non-rejections is consistent with the tendency of inflation to have a unit root after the 1960s but not before, so the unit-sum restriction $b_1 = 1$ is *compatible* with rational expectations when $y_t$ has a unit root.

From the viewpoint of Phelps's control problem, it is incidental whether the natural-rate hypothesis holds: the Phelps problem imparts interesting dynamics to the inflation-unemployment choice whether or not $b_1 = 1$.

## Sacrifice ratios and the subversion of the Phelps model

Despite its encouraging implications for sustaining the Ramsey outcome under the induction hypothesis, Phelps's control problem carries a tattered past.

In the Phelps problem, for a fixed $\delta < 1$ it is always possible to find a $\lambda$ close enough to 1 that high inflationary expectations will make a government want to avoid disinflating.

In the late 1970s, models with long expectations-adjustment lags were used to recommend *against* reducing inflation.

Large *sacrifice ratios* — estimated amounts of foregone GDP required to bring inflation down one percentage point — circulated widely.

The lesson to carry forward is that activating the induction hypothesis can eventually lead to better outcomes, but in the form that Phelps, Tobin, and Solow used it, the induction hypothesis retreats from rational expectations.

In {doc}`phillips_misspecified` and {doc}`phillips_self_confirming` we impute more symmetry to the government and the public, applying updating schemes like {eq}`pa_adaptive` to functions rather than to numbers, and we turn $\lambda$ from a free parameter into an equilibrium outcome.

The LQ Phelps problem solved here returns again in {doc}`phillips_learning`, {doc}`phillips_escaping_nash`, and {doc}`phillips_priors`, where a learning government re-solves it every period — and where activating the induction hypothesis is exactly what triggers the Volcker-like stabilizations.

## Exercises

```{exercise-start}
:label: pa_ex1
```

The proposition above states that as $\delta \to 1$ the government drives inflation to the Ramsey value $0$.

For $\lambda = 0.8$, compute the limiting inflation rate $y_\infty$ (the value that $y_t$ settles at after a long simulation) for a grid of discount factors $\delta \in \{0.90, 0.92, \ldots, 0.99\}$.

Plot $y_\infty$ against $\delta$ and confirm that it declines toward zero as $\delta \to 1$.

```{exercise-end}
```

```{solution-start} pa_ex1
:class: dropdown
```

```{code-cell} ipython3
δ_grid = np.arange(0.90, 0.995, 0.01)
y_inf = []
for δ in δ_grid:
    U, y = PhelpsProblem(λ=0.8, δ=δ).simulate(x0=12.0, T=400)
    y_inf.append(y[-1])

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(δ_grid, y_inf, 'o-')
ax.set_xlabel(r'discount factor $\delta$')
ax.set_ylabel(r'limiting inflation $y_\infty$')
plt.show()
```

As $\delta$ rises toward one, the government becomes willing to accept the transitional recession needed to reap the long-run benefit of low expected inflation, so the limiting inflation rate falls toward the Ramsey value of zero.

```{solution-end}
```

```{exercise-start}
:label: pa_ex2
```

Verify the induction property of the optimal policy directly.

Take the discounted problem with $\delta = 0.96$ and $\lambda = 0.7$, and simulate a long disinflation path.

Check that in the steady state the public's expectation $x_t$ equals actual inflation $y_t$ (so the public is *not* fooled in the limit), even though expectations are wrong along the transition.

```{exercise-end}
```

```{solution-start} pa_ex2
:class: dropdown
```

```{code-cell} ipython3
pp = PhelpsProblem(θ=1.0, U_star=5.0, λ=0.7, δ=0.96)
U, y = pp.simulate(x0=12.0, T=200)

# reconstruct the expectation path implied by the adaptive rule
x = adaptive_forecast(np.concatenate([[12.0], y]), λ=0.7, x0=12.0)[1:]

print(f"steady-state inflation   y_∞ = {y[-1]:.4f}")
print(f"steady-state expectation x_∞ = {x[-1]:.4f}")
print(f"gap                          = {y[-1] - x[-1]:.2e}")
```

In the steady state expected and actual inflation coincide, confirming that the induction property makes the public's forecast correct in the limit.

```{solution-end}
```
