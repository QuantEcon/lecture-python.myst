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

(phillips_misspecified)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Optimal Misspecified Beliefs

```{contents} Contents
:depth: 2
```

## Overview

This lecture continues the study of Phillips curve tradeoffs.

It follows chapter 6 of {cite}`Sargent1999`.

We describe three conceptual issues that recur throughout this suite of lectures:

1. how to formulate equilibria in which agents share a common *misspecified* least squares forecasting model,
2. how expectations can contribute independent dynamics within equilibria, and
3. how the classic adaptive expectations scheme can use second moments to approximate a first moment.

To expose these issues we temporarily set aside the Phillips curve and work with {cite}`Bray1982`'s simple model of the price of a single good, a workhorse for studying bounded rationality.

We alter Bray's model to illustrate an equilibrium concept that merges aspects of rational and adaptive expectations in a new way, and that we apply to the Phillips curve in {doc}`phillips_self_confirming`.

The focus is on **market equilibrium with optimal but misspecified forecasts**.

* *Optimal* means the free parameters of the forecasting scheme are chosen by (nonlinear) least squares.
* *Misspecified* means the forecasting model is wrong in functional form.

A distinctive feature is that the true model *depends on* how the agents' model is misspecified: agents' beliefs affect their behavior, which shapes the data they then fit.

We work in the frequency domain, so let's import what we need:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
```

## An experiment in Bray's lab

Following {cite}`Bray1982`, assume that

```{math}
:label: pm_bray

p_t = a + b \, p_{t+1}^e + u_t ,
```

where $u_t$ is i.i.d. with mean zero and variance $\sigma_u^2$, $a > 0$, $b \in (0, 1)$, $p_t$ is the market price, and $p_{t+1}^e$ is the market's expectation of next period's price.

The rational expectations equilibrium has $p_{t+1}^e = \frac{a}{1-b}$ and $p_t = \frac{a}{1-b} + u_t$.

Bray posited that $p_{t+1}^e$ is the empirical average of past prices, and showed that when $0 < b < 1$ this decreasing-gain scheme converges almost surely to the rational expectation $\frac{a}{1-b}$.

During the transition, the state variable $p_t^e$ contributes dynamics and makes the price serially correlated, but these dynamics are transitory: at the rational expectations equilibrium, $p_t$ is a constant plus a serially uncorrelated shock.

### Constant-gain adaptive expectations

To let expectations impart *persistent* serial correlation, we depart from Bray and assume that the market has **constant-gain** adaptive expectations,

```{math}
:label: pm_bray2

p_{t+1}^e = C p_t + (1 - C) p_t^e, \qquad |C| < 1 .
```

Bray's scheme replaces $C$ by $\frac{1}{t}$, so that $p_{t+1}^e$ becomes a sample average.

Fixing $C$ instead *discounts* past observations, which arrests convergence to rational expectations and prevents $p_{t+1}^e$ from converging to a constant.

Written as a distributed lag,

```{math}
:label: pm_bray3

p_{t+1}^e = \frac{C}{1 - (1 - C) L} p_t ,
```

where $L$ is the lag operator.

Equation {eq}`pm_bray2` would be the linear least squares forecast if the price followed the integrated moving-average process

```{math}
:label: pm_brayper

p_t = p_{t-1} + \epsilon_t - (1 - C)\epsilon_{t-1} ,
```

so the market perceives the price as composed of purely permanent and transitory components.

### The actual law of motion

Substituting the belief {eq}`pm_bray3` into {eq}`pm_bray` shows that when the market forecasts this way, its actions make the *actual* law of motion for the price

```{math}
:label: pm_bray4

p_t = \frac{a}{1 - b} + \frac{1}{1 - bC}
      \left[ \frac{1 - (1 - C)L}{1 - \frac{1 - C}{1 - bC} L} \right] u_t
    = \nu + f(L) u_t ,
```

where $\nu = \frac{a}{1-b}$ and $f(L)$ is defined to match.

The price has mean $\nu$ and spectral density

$$
F(\omega) = f(e^{i\omega}) f(e^{-i\omega}) \, \sigma_u^2, \qquad \omega \in [-\pi, \pi] .
$$

Notice that $F$ depends on $C$ through $f$.

Let's encode the true process.

```{code-cell} ipython3
class BrayModel:
    """
    Bray's price model with constant-gain adaptive expectations.

    The perceived law of motion is an IMA(1,1) with unit root; to keep its
    spectral density well defined we approximate the unit root by a root ρ
    slightly below one, following Sargent (1999).
    """

    def __init__(self, a=1.0, b=0.5, σ_u=1.0, ρ=0.995, N=1024):
        self.a, self.b, self.σ_u, self.ρ, self.N = a, b, σ_u, ρ, N
        self.ν = a / (1 - b)
        ω = 2 * np.pi * np.arange(N) / N
        self.ω = ω
        self.z = np.exp(1j * ω)

    def true_spectrum(self, C):
        "Spectral density F(ω) of the actual price process, given belief C."
        b, z = self.b, self.z
        φ = (1 - C) / (1 - b * C)
        scale = 1 / (1 - b * C)
        f = scale * (1 - (1 - C) * z) / (1 - φ * z)
        return np.abs(f)**2 * self.σ_u**2

    def approx_spectrum(self, c, σ_ε2=1.0):
        "Spectral density G(ω) of the agent's approximating IMA model."
        g = (1 - (1 - c) * self.z) / (1 - self.ρ * self.z)
        return np.abs(g)**2 * σ_ε2
```

## Optimal misspecification

Two facts about the actual law {eq}`pm_bray4` motivate an equilibrium restriction on $C$:

1. Given that the price obeys {eq}`pm_bray4`, the true linear least squares one-step forecasting rule is *not* a geometric distributed lag like {eq}`pm_bray2`.
2. Even restricting the forecast to the form {eq}`pm_bray2`, the *best* such rule would make $C$ solve a forecast-error-minimization problem, so $C$ is an outcome, not a free parameter.

A rational expectations equilibrium would repair both features.

Following {cite}`Bray1982` we soften the equilibrium concept: we leave feature 1 untouched (agents keep the wrong functional form) while fixing feature 2 (they choose the best parameter within that form).

Think of putting a single individual into a market where everyone else (the "representative agent") uses $C$, so the price obeys {eq}`pm_bray4`.

The individual chooses $c$ to fit the best model of the form

```{math}
:label: pm_bray6

p_t = \frac{1 - (1 - c) L}{1 - L}\epsilon_t = g(L)\epsilon_t ,
```

by minimizing the one-step-ahead forecast error variance.

Because $g(L)$ has a unit root, its DC gain is infinite; this is exactly how the perceived model uses a unit root to *fit the constant mean* $\nu$.

Numerically we replace the unit root by a root $\rho$ slightly below one.

```{prf:definition} Best-estimate map
:label: pm_bmap

Given $C$ and the consequent price process {eq}`pm_bray4`, the individual's best forecast parameter $c = B(C)$ is the nonlinear least squares estimator of $c$ in {eq}`pm_bray6`, where the data are generated by {eq}`pm_bray4`.
```

Following the frequency-domain method of Hansen and Sargent {cite}`HansenSargent1993`, the best approximating $(c, \sigma_\epsilon^2)$ minimizes

```{math}
:label: pm_criterion

A(c, \sigma_\epsilon^2) = \frac{1}{N}\sum_{j=0}^{N-1}
\left\{ \log G(\omega_j, c) + \frac{F(\omega_j)}{G(\omega_j, c)} \right\}
+ \frac{\nu^2}{G(0)} ,
```

where $\omega_j = \frac{2\pi j}{N}$, and the term $\frac{\nu^2}{G(0)}$ makes the approximating model use its near-unit-root to fit the mean.

Concentrating out $\sigma_\epsilon^2$ leaves a one-dimensional minimization over $c$.

```{prf:definition} Equilibrium under forecast misspecification
:label: pm_equilibrium

An equilibrium under forecast misspecification is a fixed point $C = B(C)$.
```

At such a fixed point the representative agent is representative: the single individual's best parameter equals the one everyone uses.

```{code-cell} ipython3
def best_estimate(model, C):
    "The best-estimate map c = B(C)."
    F = model.true_spectrum(C)
    z, ν, N = model.z, model.ν, model.N

    def neg_profile(c):
        H = np.abs((1 - (1 - c) * z) / (1 - model.ρ * z))**2   # |g|^2
        σ_ε2 = np.mean(F / H) + ν**2 / H[0]                    # concentrated
        return np.log(σ_ε2) + np.mean(np.log(H))               # profiled criterion

    res = minimize_scalar(neg_profile, bounds=(1e-4, 0.99), method='bounded')
    return res.x

def solve_equilibrium(model, C0=0.3, tol=1e-10, maxit=500):
    "Iterate the best-estimate map to a fixed point."
    C = C0
    for _ in range(maxit):
        C_new = best_estimate(model, C)
        if abs(C_new - C) < tol:
            break
        C = C_new
    return C_new
```

```{code-cell} ipython3
bray = BrayModel(a=1.0, b=0.5, σ_u=1.0)
C_star = solve_equilibrium(bray)
print(f"equilibrium belief   C = {C_star:.4f}")
```

For these parameters the equilibrium belief is $C \approx 0.08$, reproducing the value reported in chapter 6 of {cite}`Sargent1999`.

Let's also report the *actual* one-step-ahead forecast error standard deviation that agents incur by using their misspecified model.

```{code-cell} ipython3
def fitted_sigma2(model, C, c):
    "Concentrated innovation variance σ_ε^2 of the approximating model."
    F = model.true_spectrum(C)
    H = np.abs((1 - (1 - c) * model.z) / (1 - model.ρ * model.z))**2
    return np.mean(F / H) + model.ν**2 / H[0]

c_star = best_estimate(bray, C_star)
σ_bar = np.sqrt(fitted_sigma2(bray, C_star, c_star))
print(f"actual one-step forecast error std  σ̄_ε = {σ_bar:.4f}")
```

## Comparing the true and forecasting models

For the equilibrium $C$, we plot the equilibrium spectral densities of the true and approximating models.

```{code-cell} ipython3
F = bray.true_spectrum(C_star)
σ_ε2 = fitted_sigma2(bray, C_star, c_star)
G = bray.approx_spectrum(c_star, σ_ε2)

half = bray.N // 2
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(bray.ω[:half], np.log(F[:half]), 'C0', label='true model')
ax.plot(bray.ω[:half], np.log(G[:half]), 'C1--', label='forecasting model')
ax.set_xlabel(r'angular frequency $\omega$')
ax.set_ylabel('log spectral density')
ax.legend()
plt.show()
```

In minimizing {eq}`pm_criterion`, the approximating model uses its near-unit-root to fit the mean.

The large gap between the spectral densities at low frequencies reflects how the approximating model fits a *first* moment (the mean $\nu$) with features of *second* moments (a spike in the spectral density at frequency zero).

The true spectral density decreases sharply with frequency — Granger's "typical spectral shape" {cite}`Granger1966` — revealing substantial positive serial correlation in the price, because agents' belief that the price is subject to permanent shocks makes shocks persist.

### Impulse responses

We compare the impulse response functions of the two models by feeding a unit shock through each moving-average representation.

```{code-cell} ipython3
def impulse_response(num_roots, den_roots, T=25):
    "IRF of (1 - num L)/(1 - den L): coefficients of the ratio of lag polys."
    h = np.empty(T)
    h[0] = 1.0
    for k in range(1, T):
        h[k] = den_roots * h[k - 1]
    h[1:] -= num_roots * h[:-1]           # apply the numerator (1 - num L)
    return h

φ = (1 - C_star) / (1 - bray.b * C_star)
scale = 1 / (1 - bray.b * C_star)
irf_true = scale * impulse_response(1 - C_star, φ)          # f(L)
irf_approx = impulse_response(1 - c_star, bray.ρ)           # g(L)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(irf_true, 'C0o-', ms=4, label='true model')
ax.plot(irf_approx, 'C1s--', ms=4, label='approximating model')
ax.set_xlabel('lag')
ax.set_ylabel('response')
ax.legend()
plt.show()
```

The impulse response of the true model affirms the serial correlation in the price.

The approximating model tends to under-predict the short-term consequences of a shock while over-predicting the long-term ones: its near-unit-root produces a response that does not die out.

## Lessons

The agents in this model are **boundedly rational**: *rational* describes their use of least squares, and *bounded* describes their model misspecification.

Under rational expectations there is only one model in play.

Under bounded rationality there must be at least two: the one used by the boundedly rational agents, and the true one.

These mutually influence each other — the boundedly rational agents use their model to approximate the true one, and the true one reflects the decisions of the agents — and both differ from the rational expectations model.

The peculiar way that the adaptive expectations model uses a unit root to mimic a constant foreshadows a version of the Phillips curve model, developed in {doc}`phillips_self_confirming`, that will help vindicate econometric policy evaluation.

This same trick — using a unit root to approximate a constant — turns out to be the engine of the *escape dynamics* of {doc}`phillips_learning` and {doc}`phillips_escaping_nash`, where a learning government's estimated Phillips curve drifts toward the induction hypothesis and, believing it, cuts inflation toward Ramsey.

## Exercises

```{exercise-start}
:label: pmis_ex1
```

The equilibrium belief $C$ depends on the feedback parameter $b$ in {eq}`pm_bray`.

Compute and plot the equilibrium $C$ as a function of $b$ over a grid $b \in \{0.1, 0.2, \ldots, 0.8\}$, holding $a = 1$ and $\sigma_u = 1$ fixed.

How does stronger expectational feedback (larger $b$) affect the equilibrium amount of discounting of past data?

```{exercise-end}
```

```{solution-start} pmis_ex1
:class: dropdown
```

```{code-cell} ipython3
b_grid = np.arange(0.1, 0.85, 0.1)
C_of_b = [solve_equilibrium(BrayModel(a=1.0, b=b, σ_u=1.0)) for b in b_grid]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(b_grid, C_of_b, 'o-')
ax.set_xlabel('feedback parameter $b$')
ax.set_ylabel('equilibrium belief $C$')
plt.show()
```

Stronger feedback raises the equilibrium gain $C$: agents put more weight on recent observations, so the price process they generate is less persistent than it would otherwise be.

```{solution-end}
```

```{exercise-start}
:label: pmis_ex2
```

Verify that the equilibrium is a genuine fixed point by plotting the best-estimate map $c = B(C)$ against the 45-degree line, and mark the fixed point.

```{exercise-end}
```

```{solution-start} pmis_ex2
:class: dropdown
```

```{code-cell} ipython3
C_grid = np.linspace(0.02, 0.4, 25)
B_vals = [best_estimate(bray, C) for C in C_grid]

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(C_grid, B_vals, 'C0', label='$B(C)$')
ax.plot(C_grid, C_grid, 'k--', lw=1, label='45 degrees')
ax.plot(C_star, C_star, 'ko')
ax.annotate('equilibrium', (C_star, C_star),
            (C_star + 0.05, C_star - 0.03))
ax.set_xlabel('$C$')
ax.set_ylabel('$B(C)$')
ax.legend()
plt.show()
```

The best-estimate map crosses the 45-degree line at the equilibrium belief, confirming $C = B(C)$.

```{solution-end}
```
