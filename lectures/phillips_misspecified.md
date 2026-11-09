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

```{index} single: Phillips Curve; Optimal Misspecified Beliefs
```

```{contents} Contents
:depth: 2
```

## Overview

This lecture continues the study of Phillips curve tradeoffs.

It follows chapter 6 of {cite:t}`Sargent1999`.

In {doc}`phillips_adaptive` the public forecast inflation with a fixed adaptive rule whose
parameter $\lambda$ we simply chose.

That was unsatisfying in a way the *vindication* story of {doc}`phillips_two_stories` cannot
afford: a free parameter describing expectations is exactly what rational expectations was meant
to eliminate.

Here we take the first step toward earning that parameter, by letting agents choose it to fit
the data their own beliefs generate.

We describe three conceptual issues that recur throughout this suite of lectures:

1. how to formulate equilibria in which agents share a common *misspecified* least squares forecasting model,
2. how expectations can contribute independent dynamics within equilibria, and
3. how the classic adaptive expectations scheme can use second moments to approximate a first moment.

To expose these issues we temporarily set aside the Phillips curve and work with {cite:t}`Bray1982`'s simple model of the price of a single good, a workhorse for studying bounded rationality.

{ref}`br_bray` presents Bray's model as the relaxation algorithm with a gain of $1/t$, and {doc}`ls_learning` studies least squares learning in it.

We alter Bray's model to illustrate an equilibrium concept that merges aspects of rational and adaptive expectations in a new way, and that we apply to the Phillips curve in {doc}`phillips_self_confirming`.

The focus is on **market equilibrium with optimal but misspecified forecasts**.

* *Optimal* means the free parameters of the forecasting scheme are chosen by (nonlinear) least squares.
* *Misspecified* means the forecasting model is wrong in functional form, or uses too small an information set.

A distinctive feature is that the true model *depends on* how the agents' model is misspecified: agents' beliefs affect their behavior, which shapes the data they then fit.

We will also see that the equilibrium depends on a detail of the agents' model, a root $\rho$ that they place just below one, and that the forecasting gain it delivers vanishes as $\rho \to 1$.

We work in the frequency domain, so let's import what we need:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter
```

## An experiment in Bray's lab

Following {cite:t}`Bray1982`, assume that

```{math}
:label: pm_bray

p_t = a + b \, p_{t+1}^e + u_t ,
```

where $u_t$ is IID with mean zero and variance $\sigma_u^2$, $a > 0$, $b \in (0, 1)$, $p_t$ is the market price, and $p_{t+1}^e$ is the market's expectation of next period's price.

In the notation of {ref}`br_bray`, Bray's $b$ is the feedback coefficient $h'$.

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

This is the answer to {cite:t}`Muth1960`'s inverse optimal-prediction problem in {doc}`bounded_rationality`, with Muth's moving-average coefficient equal to $1 - C$.

Equivalently, the perceived model is a *local-level* model, a random walk observed with white noise.

For it, {eq}`pm_bray2` is the steady-state Kalman filter of {ref}`kalman_convergence`, and the gain $C$ is tied to the ratio $q/r$ of the variance of the random-walk innovation to that of the noise by $q/r = C^2/(1 - C)$ ({ref}`pmis_ex4`).

Muth asked for which process a given smoothing rule is optimal.

Here the process is generated by the rule, and agents choose the rule to fit the process.

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

    ρ is the root of the agents' perceived ARMA(1,1) model, which stands in
    for the unit root of the integrated moving-average model (see below).
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
        "Spectral density G(ω) of the agent's approximating model."
        g = (1 - (1 - c) * self.z) / (1 - self.ρ * self.z)
        return np.abs(g)**2 * σ_ε2
```

## Optimal misspecification

Two facts about the actual law {eq}`pm_bray4` motivate an equilibrium restriction on $C$:

1. Given that the price obeys {eq}`pm_bray4`, the true linear least squares one-step forecasting rule is *not* a geometric distributed lag like {eq}`pm_bray2`.
2. Even restricting the forecast to the form {eq}`pm_bray2`, the *best* such rule would make $C$ solve a forecast-error-minimization problem, so $C$ is an outcome, not a free parameter.

A rational expectations equilibrium would repair both features.

Chapter 6 of {cite:t}`Sargent1999` softens the equilibrium concept, proceeding "in the spirit (but not the letter)" of {cite:t}`Lucas_Prescott_1971` and {cite:t}`Brock1972`: we leave feature 1 untouched (agents keep the wrong functional form) while fixing feature 2 (they choose the best parameter within that form).

Think of putting a single individual into a market where everyone else (the "representative agent") uses $C$, so the price obeys {eq}`pm_bray4`.

This "big $C$, little $c$" device is the "big $Y$, little $y$" trick of {doc}`rational_expectations`, applied to a forecasting rule rather than to a decision rule.

The individual chooses $c$ to fit the best model of the form

```{math}
:label: pm_bray6

p_t = \frac{1 - (1 - c) L}{1 - L}\epsilon_t ,
```

by minimizing the one-step-ahead forecast error variance.

### The perceived model and its root $\rho$

A model with the exact unit root of {eq}`pm_bray6` cannot be fitted to data that have a mean, and its spectral density is infinite at frequency zero.

Following {cite:t}`Sargent1999` (chapter 6, footnotes 4 and 5), the agents' perceived model replaces the unit root by a root $\rho$ slightly below one:

```{math}
:label: pm_rhomodel

p_t = \frac{1 - (1 - c) L}{1 - \rho L}\epsilon_t = g(L)\epsilon_t ,
\qquad 0 < \rho < 1 .
```

This is a stationary, mean-zero ARMA(1,1) model.

It has no constant, so it must use the near-unit-root persistence of $1/(1 - \rho L)$ to fit the nonzero mean $\nu$ of the price.

We treat $\rho$ as a primitive of the perceived model, part of what agents believe, not as a numerical device: the equilibrium depends on it, as {ref}`pm_rho` shows.

```{prf:definition} Best-estimate map
:label: pm_bmap

Given $C$ and the consequent price process {eq}`pm_bray4`, the individual's best forecast parameter $c = B(C)$ is the nonlinear least squares estimator of $c$ in {eq}`pm_rhomodel`, where the data are generated by {eq}`pm_bray4`.
```

Inverting {eq}`pm_rhomodel` gives the perceived model's one-step-ahead forecast error $\epsilon_t = g(L)^{-1} p_t$.

Applied to the actual price, this error has mean $\nu / g(1) = \nu (1 - \rho)/c$ and a variance that we can compute from the spectral density $F$.

Following the frequency-domain method of Hansen and Sargent {cite}`HansenSargent1993`, the best approximating $(c, \sigma_\epsilon^2)$ minimizes

```{math}
:label: pm_criterion

A(c, \sigma_\epsilon^2) = \frac{1}{N}\sum_{j=0}^{N-1}
\left\{ \log G(\omega_j, c) + \frac{F(\omega_j)}{G(\omega_j, c)} \right\}
+ \frac{\nu^2}{G(0)} ,
```

where $\omega_j = \frac{2\pi j}{N}$ and $G(\omega, c) = |g(e^{i\omega})|^2 \sigma_\epsilon^2$ is the spectral density of the perceived model.

The term $\frac{\nu^2}{G(0)} = \frac{\nu^2 (1 - \rho)^2}{c^2 \sigma_\epsilon^2}$ charges the perceived model for the squared mean of its forecast errors; this is how the model is made to use its near-unit root to fit the mean.

Concentrating out $\sigma_\epsilon^2$ sets it equal to the mean square forecast error

```{math}
:label: pm_sigma2

\sigma_\epsilon^2(c) = \frac{1}{N}\sum_{j=0}^{N-1} \frac{F(\omega_j)}{|g(e^{i\omega_j})|^2}
+ \frac{\nu^2 (1 - \rho)^2}{c^2} .
```

The average of $\log |g(e^{i\omega})|^2$ over frequencies is zero, by Kolmogorov's formula for an invertible filter with $g(0) = 1$.

So minimizing {eq}`pm_criterion` over $c$ is the same as minimizing $\sigma_\epsilon^2(c)$: the agent chooses the $c$ that minimizes the mean square one-step-ahead forecast error, computed under his model {eq}`pm_rhomodel`.

The mean term shows where $\rho$ enters: a larger $c$ shrinks the squared mean $\nu^2 (1 - \rho)^2/c^2$, so the closer $\rho$ is to one, the weaker the pull toward a large $c$.

```{prf:definition} Equilibrium under forecast misspecification
:label: pm_equilibrium

An equilibrium under forecast misspecification is a fixed point $C = B(C)$.
```

At such a fixed point the representative agent is representative: the single individual's best parameter equals the one everyone uses.

```{code-cell} ipython3
def forecast_mse(F, ν, c, ρ, z):
    """
    Mean square one-step forecast error σ_ε^2(c) of the perceived model
    g(L) = (1 - (1-c)L)/(1 - ρL), when the true process has mean ν and
    spectral density F on the grid z = exp(iω).  ρ = 1 is allowed.
    """
    W = np.abs((1 - ρ * z) / (1 - (1 - c) * z))**2     # |1/g(e^{iω})|^2
    return np.mean(F * W) + ν**2 * W[0]                # variance + squared mean

def best_estimate(model, C):
    "The best-estimate map c = B(C)."
    F = model.true_spectrum(C)
    res = minimize_scalar(lambda c: forecast_mse(F, model.ν, c, model.ρ, model.z),
                          bounds=(1e-4, 0.99), method='bounded',
                          options={'xatol': 1e-10})
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

Now we solve for the equilibrium of the baseline model.

```{code-cell} ipython3
bray = BrayModel(a=1.0, b=0.5, σ_u=1.0, ρ=0.995)
C_star = solve_equilibrium(bray)
print(f"equilibrium belief   C = {C_star:.4f}")
```

For these parameters the equilibrium belief is $C \approx 0.08$, reproducing the value reported in chapter 6 of {cite:t}`Sargent1999`.

### Forecast errors

Three standard deviations of one-step-ahead forecast errors describe the equilibrium:

* $\bar\sigma_\epsilon$, the root mean square forecast error of the perceived model {eq}`pm_rhomodel` at $c = C$, including the mean of its errors: the square root of the minimized {eq}`pm_sigma2`.
* the root mean square error of the rule {eq}`pm_bray2` that agents actually use. Its weights sum to one, so its errors have mean zero; it is $\sigma_\epsilon$ computed with $\rho = 1$.
* $\sigma_u / (1 - bC)$, the standard deviation of the innovation in the true process {eq}`pm_bray4`, the smallest one-step error that any forecaster who knew the true law could attain.

```{code-cell} ipython3
F_star = bray.true_spectrum(C_star)
c_star = best_estimate(bray, C_star)

σ_bar = np.sqrt(forecast_mse(F_star, bray.ν, c_star, bray.ρ, bray.z))
σ_rule = np.sqrt(forecast_mse(F_star, bray.ν, C_star, 1.0, bray.z))
σ_true = bray.σ_u / (1 - bray.b * C_star)

print(f"perceived model (ρ = {bray.ρ}):    {σ_bar:.4f}")
print(f"rule actually used (ρ = 1):     {σ_rule:.4f}")
print(f"true innovation σ_u/(1 - bC):   {σ_true:.4f}")
```

The first number is the $\bar\sigma_\epsilon$ that chapter 6 of {cite:t}`Sargent1999` reports.

The book gives $1.0208$ for the last one, which is the square root of the value printed here.

The forecasts that agents actually make are only slightly worse than the best attainable ones.

(pm_concept)=
### What kind of equilibrium is this?

The fixed point $C = B(C)$ has several names in the later literature, each emphasizing a different feature.

* It is a **restricted-perceptions equilibrium** ({cite:t}`EvansHonkapohja2001`, chapter 13): forecasts are optimal within a restricted class of models that excludes the truth.
* For given $C$, the best estimate $c$ is a **pseudo-true value** in the sense of {cite:t}`White1982`, the parameter of a misspecified model that comes closest to the truth. Unlike in White's setting, here the truth itself depends on the agents' parameter, through $C$.
* Up to constants, criterion {eq}`pm_criterion` is the frequency-domain (Whittle) form of the Kullback–Leibler divergence of the perceived model from the true process. So the fixed point is also a **Berk–Nash equilibrium** in the sense of {cite:t}`EspondaPouzo2016`, with the forecasting rule as the agents' action.

{doc}`learning_approximation` computes the same kind of object, which it calls an approximate equilibrium, for a saving rule restricted to a polynomial family.

There the agents' fit is a least squares projection onto a basis; here it is a nonlinear fit of an ARMA model across the whole spectrum.

Because $B$ works like a best-response map, we can ask whether agents who nudge their gain toward the best estimate would find the equilibrium.

The associated differential equation $dC/d\tau = B(C) - C$ is the E-stability condition of {ref}`br_one_recursion`: the fixed point is E-stable when $B'(C^*) < 1$.

```{code-cell} ipython3
h = 1e-4
B_prime = (best_estimate(bray, C_star + h) - best_estimate(bray, C_star - h)) / (2 * h)
print(f"B'(C*) = {B_prime:.3f}")
```

Here $0 < B'(C^*) < 1$, so the equilibrium is E-stable, and iterating $B$ converges to it monotonically, which is how `solve_equilibrium` found it.

Why don't the agents eventually discover that their model is wrong?

{cite:t}`KalaiLehrer1993Nash` applied a merging theorem of {cite:t}`blackwell1962` to establish conditions under which forecasts must eventually merge with rational expectations.

The present model shuts off that mechanism at the outset: the agents' model puts zero probability on events that have positive probability under the truth, violating the absolute continuity condition that merging requires.

With enough data, appropriate statistical tests would eventually tell agents how to improve their forecasts.

{cite:t}`Kreps1998` argues that sensible agents would eventually abandon a misspecified model that the data reject; {ref}`pmis_ex3` estimates how much data that would take here.

The answer lies in how far apart the true and approximating spectral densities are, which we now study.

## Comparing the true and forecasting models

For the equilibrium $C$, we plot the equilibrium spectral densities of the true and approximating models.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Spectral densities of the true price process and of the agents' approximating model
    name: fig-mis-spectra
---
G = bray.approx_spectrum(c_star, σ_bar**2)

half = bray.N // 2
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(bray.ω[:half], np.log(F_star[:half]), 'C0', label='true model', lw=2)
ax.plot(bray.ω[:half], np.log(G[:half]), 'C1--', label='forecasting model', lw=2)
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

The true process {eq}`pm_bray4` and the perceived model {eq}`pm_rhomodel` are driven by different shocks, $u_t$ and $\epsilon_t$.

To compare like with like, we normalize the true response by its impact effect $f(0) = 1/(1 - bC)$, so that both models respond to a unit one-step-ahead forecast error.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Impulse responses of the true and approximating models to a unit one-step-ahead forecast error
    name: fig-mis-irf
---
def impulse_response(num, den, T=25):
    "IRF of (1 - num L)/(1 - den L)."
    h = np.empty(T)
    h[0] = 1.0
    for k in range(1, T):
        h[k] = den * h[k - 1]
    h[1:] -= num * h[:-1]           # apply the numerator (1 - num L)
    return h

φ = (1 - C_star) / (1 - bray.b * C_star)
irf_true = impulse_response(1 - C_star, φ)              # f(L) / f(0)
irf_approx = impulse_response(1 - c_star, bray.ρ)       # g(L)

lags = np.arange(len(irf_true))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, start in zip(axes, [0, 1]):
    ax.plot(lags[start:], irf_true[start:], 'C0o-', ms=4, label='true model', lw=2)
    ax.plot(lags[start:], irf_approx[start:], 'C1s--', ms=4,
            label='approximating model', lw=2)
    ax.set_xlabel('lag')
    ax.set_ylabel('response')
axes[0].set_title('all lags')
axes[1].set_title('lags 1 and beyond')
axes[0].legend()
plt.tight_layout()
plt.show()

print(f"response at lag 1:  true {irf_true[1]:.3f},  approximating {irf_approx[1]:.3f}")
```

The impulse response of the true model affirms the serial correlation in the price.

From lag one on, the true response lies below the approximating one: the approximating model over-predicts the effects of a shock at every horizon, and its near unit root makes those effects die out much more slowly, at rate $\rho$ rather than $\phi = (1 - C)/(1 - bC)$.

Chapter 6 of {cite:t}`Sargent1999` says that the approximating model under-predicts short-term consequences; that statement rests on comparing impact effects without the normalization, and so reflects only the different scales of the two shocks.

(pm_rho)=
## How the equilibrium depends on $\rho$

Every number so far was computed with the perceived root $\rho = 0.995$.

The next cell recomputes the equilibrium for values of $\rho$ closer to one, and then for the exact unit root $\rho = 1$ of the rule {eq}`pm_bray2` itself.

We use a finer frequency grid, because the perceived spectral density has a spike at frequency zero whose width is about $1 - \rho$.

```{code-cell} ipython3
print(f"{'ρ':>7} {'C*':>8} {'weight sum':>11} {'RMSE of rule':>13}")
for ρ in [0.98, 0.99, 0.995, 0.999, 0.9995]:
    m = BrayModel(a=1.0, b=0.5, σ_u=1.0, ρ=ρ, N=2**14)
    C = solve_equilibrium(m, C0=0.1)
    rmse = np.sqrt(forecast_mse(m.true_spectrum(C), m.ν, C, 1.0, m.z))
    print(f"{ρ:>7} {C:>8.4f} {1 - (1 - ρ) / C:>11.3f} {rmse:>13.4f}")

m1 = BrayModel(a=1.0, b=0.5, σ_u=1.0, ρ=1.0, N=2**14)
C_grid = np.linspace(0.005, 0.4, 40)
ratio = max(best_estimate(m1, C) / C for C in C_grid)
print(f"\nρ = 1: largest value of B(C)/C on the grid = {ratio:.3f}")
print(f"ρ = 1: best c given C = {C_star:.4f} is {best_estimate(m1, C_star):.4f}")
```

As $\rho$ rises toward one, the equilibrium gain $C^*$ falls toward zero.

With the exact unit root, $B(C) < C$ for every $C$ on the grid, so iterating $B$ drives $C$ to zero, and the only fixed point is $C = 0$.

That is the rational expectations equilibrium: with $C = 0$ expectations stop moving, and the price is $p_t = \nu + u_t$.

Accordingly, the root mean square error of the rule in the last column falls toward $\sigma_u = 1$ as $\rho \to 1$.

The mechanism is the mean term in {eq}`pm_sigma2`.

A perceived model with $\rho < 1$ forecasts with weights that sum to $1 - (1 - \rho)/c < 1$ (the column "weight sum"), so its errors have mean $\nu(1 - \rho)/c$.

A larger $c$ shrinks that bias, and this pull toward a large $c$ is what keeps $C^*$ away from zero; as $\rho \to 1$ the bias, and the pull, vanish.

There is a tension in the construction.

The price {eq}`pm_bray4` is generated by the rule {eq}`pm_bray2`, whose weights sum to one, but the gain is chosen by fitting the $\rho$-model.

An agent who chose $c$ to minimize the actual mean square error of the rule he uses would, in effect, set $\rho = 1$, and the last line of output shows that he would pick a gain well below $C^*$ ({ref}`pmis_ex2` confirms this by simulation).

So the equilibrium with $C^* > 0$, and the persistent serial correlation that comes with it, exists for each fixed $\rho < 1$ and disappears as $\rho \to 1$.

The issue goes back to footnotes 4 and 5 of chapter 6 of {cite:t}`Sargent1999`, where $\rho < 1$ is introduced to keep the spectral density well defined.

It matters again in {doc}`phillips_self_confirming`, where the better-than-Nash outcomes of an equilibrium with misspecified beliefs rest on it.

## Lessons

The agents in this model are **boundedly rational**, in the sense of {doc}`bounded_rationality`: *rational* describes their use of least squares, and *bounded* describes their model misspecification.

Under rational expectations there is only one model in play.

Under bounded rationality there must be at least two: the one used by the boundedly rational agents, and the true one.

These mutually influence each other — the boundedly rational agents use their model to approximate the true one, and the true one reflects the decisions of the agents — and both differ from the rational expectations model.

The peculiar way that the adaptive expectations model uses a unit root to mimic a constant foreshadows a version of the Phillips curve model, developed in {doc}`phillips_self_confirming`, that will help vindicate econometric policy evaluation, and whose good outcomes also depend on $\rho < 1$.

The same trick — using a unit root to approximate a constant — drives the *escape dynamics* of {doc}`phillips_learning`, where a learning government's estimated distributed-lag Phillips curve drifts toward the induction hypothesis and, believing it, cuts inflation toward Ramsey.

In the static model of {doc}`phillips_escaping_nash`, an escape instead makes the perceived Phillips curve vertical, which trivially satisfies the induction property.

## Exercises

```{exercise-start}
:label: pmis_ex1
```

The equilibrium belief $C$ depends on the feedback parameter $b$ in {eq}`pm_bray`.

1. Compute and plot the equilibrium $C$ as a function of $b \in \{0.1, 0.2, \ldots, 0.8\}$, holding $a = 1$ and $\sigma_u = 1$ fixed.
2. With $a$ fixed, a change in $b$ also changes the mean $\nu = a/(1 - b)$ that the perceived model must fit. Repeat the calculation holding $\nu = 2$ fixed, by setting $a = 2(1 - b)$.
3. How much of the effect of $b$ works through the mean? How does the persistence $\phi = (1 - C)/(1 - bC)$ of the actual price change with $b$?

```{exercise-end}
```

```{solution-start} pmis_ex1
:class: dropdown
```

```{code-cell} ipython3
b_grid = np.arange(0.1, 0.85, 0.1)
C_a1 = np.array([solve_equilibrium(BrayModel(a=1.0, b=b)) for b in b_grid])
C_ν2 = np.array([solve_equilibrium(BrayModel(a=2 * (1 - b), b=b)) for b in b_grid])
φ_a1 = (1 - C_a1) / (1 - b_grid * C_a1)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(b_grid, C_a1, 'o-', lw=2, label=r'$a = 1$ (mean $\nu$ rises with $b$)')
ax.plot(b_grid, C_ν2, 's--', lw=2, label=r'$\nu = 2$ held fixed')
ax.set_xlabel('feedback parameter $b$')
ax.set_ylabel('equilibrium belief $C$')
ax.legend()
plt.show()

print(f"a = 1:   C from {C_a1[0]:.3f} to {C_a1[-1]:.3f},  "
      f"ν from {1 / (1 - b_grid[0]):.2f} to {1 / (1 - b_grid[-1]):.2f}")
print(f"ν = 2:   C from {C_ν2[0]:.3f} to {C_ν2[-1]:.3f}")
print(f"persistence φ ranges from {φ_a1.min():.3f} to {φ_a1.max():.3f}")
```

With $a = 1$, stronger feedback raises the equilibrium gain a great deal, but it also raises the mean $\nu$ that the perceived model must fit.

Holding $\nu$ fixed, the gain rises much less.

Most of the effect of $b$ therefore works through the mean: a larger mean makes the bias $\nu(1 - \rho)/c$ of the perceived model costlier, so agents choose a larger gain.

The persistence $\phi$ of the actual price barely changes, so stronger feedback does not make the price process noticeably less persistent.

```{solution-end}
```

```{exercise-start}
:label: pmis_ex2
```

The best-estimate map is computed in the frequency domain, under the perceived model {eq}`pm_rhomodel`.

Check what the equilibrium implies in the time domain.

Simulate the market {eq}`pm_bray`–{eq}`pm_bray2` with $C = C^*$ for one million periods.

For a grid of values of $c$, compute the mean square error of the forecast $p^e_{t+1}(c) = c\, p_t + (1 - c) p^e_t(c)$ of $p_{t+1}$.

Which $c$ minimizes it?

Compare it with $C^*$ and with the exact-unit-root best estimate printed in {ref}`pm_rho`.

```{exercise-end}
```

```{solution-start} pmis_ex2
:class: dropdown
```

Substituting {eq}`pm_bray2` into {eq}`pm_bray` gives $p_t = [a + b(1 - C) p^e_t + u_t]/(1 - bC)$, and then $p^e_{t+1} = \phi\, p^e_t + C(a + u_t)/(1 - bC)$ with $\phi = (1 - C)/(1 - bC)$.

```{code-cell} ipython3
rng = np.random.default_rng(0)
T, burn = 1_000_000, 10_000
a, b, C = bray.a, bray.b, C_star
u = rng.standard_normal(T)

φ = (1 - C) / (1 - b * C)
pe_next = lfilter([C / (1 - b * C)], [1, -φ], a + u, zi=[φ * bray.ν])[0]
pe = np.concatenate([[bray.ν], pe_next[:-1]])                 # p^e_t
p = (a + b * (1 - C) * pe + u) / (1 - b * C)

def rmse(c):
    "Root mean square error of the forecast c p_t + (1-c) p^e_t of p_{t+1}."
    f = lfilter([c], [1, -(1 - c)], p, zi=[(1 - c) * bray.ν])[0]
    return np.sqrt(np.mean((p[burn + 1:] - f[burn:-1])**2))

c_grid = np.linspace(0.005, 0.15, 59)
errors = np.array([rmse(c) for c in c_grid])

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(c_grid, errors, lw=2)
ax.axvline(C_star, color='k', ls='--', lw=1, label='$C^*$')
ax.set_xlabel('gain $c$')
ax.set_ylabel('RMSE of the forecast')
ax.legend()
plt.show()

print(f"RMSE-minimizing c = {c_grid[errors.argmin()]:.3f},  C* = {C_star:.4f}")
```

The gain that minimizes the actual forecast error is far below $C^*$ and close to the exact-unit-root best estimate of {ref}`pm_rho`.

An individual who judged the rule {eq}`pm_bray2` by its actual forecast errors would not choose $C^*$.

$C^*$ is a best response only within the perceived model {eq}`pm_rhomodel`, whose mean penalty does not apply to the rule that agents actually use, because that rule's errors have mean zero.

```{solution-end}
```

```{exercise-start}
:label: pmis_ex3
```

How much data would agents need to detect that their model is wrong?

For Gaussian processes, the expected log-likelihood ratio per observation of the true model {eq}`pm_bray4` against the perceived model {eq}`pm_rhomodel` (at $c = B(C^*)$ and the fitted $\bar\sigma_\epsilon^2$) is the Kullback–Leibler divergence rate.

1. Using the concentrated criterion and Kolmogorov's formula, show that this rate equals $\log(\bar\sigma_\epsilon / \sigma_{\rm true})$, with $\sigma_{\rm true} = \sigma_u/(1 - bC^*)$.
2. Compute it, and the number of observations it takes for the expected log-likelihood ratio to reach 3 (a likelihood ratio of about 20).
3. Confirm the rate by simulating the log-likelihood ratio.

```{exercise-end}
```

```{solution-start} pmis_ex3
:class: dropdown
```

The Kullback–Leibler divergence rate of a mean-zero Gaussian model with spectral density $G$ from a Gaussian process with mean $\nu$ and spectral density $F$ is

$$
\frac{1}{2}\left\{ \frac{1}{N}\sum_j \left[\frac{F(\omega_j)}{G(\omega_j)} - 1 - \log\frac{F(\omega_j)}{G(\omega_j)}\right] + \frac{\nu^2}{G(0)} \right\} .
$$

At the fitted $\bar\sigma_\epsilon^2$, the terms $\frac{1}{N}\sum_j F/G + \nu^2/G(0)$ add up to one.

By Kolmogorov's formula, the average of $\log G$ is $\log \bar\sigma_\epsilon^2$ and the average of $\log F$ is $\log \sigma_{\rm true}^2$, the log one-step prediction variance of the true process.

So the rate is $\frac{1}{2}\log(\bar\sigma_\epsilon^2/\sigma_{\rm true}^2) = \log(\bar\sigma_\epsilon/\sigma_{\rm true})$.

```{code-cell} ipython3
kl = np.log(σ_bar / σ_true)
print(f"KL rate = {kl:.4f} nats per observation")
print(f"observations for an expected log-likelihood ratio of 3: {3 / kl:.0f}")

# simulate: true one-step errors are u_t/(1-bC); the perceived model's are g(L)^{-1} p_t
rng = np.random.default_rng(1)
T, burn = 200_000, 10_000
a, b, C = bray.a, bray.b, C_star
u = rng.standard_normal(T)
φ = (1 - C) / (1 - b * C)
pe_next = lfilter([C / (1 - b * C)], [1, -φ], a + u, zi=[φ * bray.ν])[0]
pe = np.concatenate([[bray.ν], pe_next[:-1]])
p = (a + b * (1 - C) * pe + u) / (1 - b * C)

e_true = u / (1 - b * C)
e_model = lfilter([1, -bray.ρ], [1, -(1 - c_star)], p)
llr = (-np.log(σ_true) - e_true**2 / (2 * σ_true**2)) \
    - (-np.log(σ_bar) - e_model**2 / (2 * σ_bar**2))
print(f"simulated mean log-likelihood ratio = {llr[burn:].mean():.4f}")
```

The divergence is about $0.015$ nats per observation, so it takes roughly two hundred observations before the data favor the true model by a likelihood ratio of about 20, and that is for a statistician who already knows which alternative to test.

```{solution-end}
```

```{exercise-start}
:label: pmis_ex4
```

{cite:t}`Muth1960` showed that the rule {eq}`pm_bray2` is optimal for a random walk observed with noise.

Consider the local-level model $p_t = \mu_t + \eta_t$, $\mu_t = \mu_{t-1} + w_t$, with $\mathbb{V}[w_t] = q$ and $\mathbb{V}[\eta_t] = r$.

1. Show that the steady-state Kalman gain $K$ for this model satisfies $q/r = K^2/(1 - K)$, and that the Kalman filter's forecast of $p_{t+1}$ then has the form {eq}`pm_bray2` with $C = K$.
2. Set $q/r = C^{*2}/(1 - C^*)$, iterate the Riccati equation of {ref}`kalman_convergence` to its fixed point, and confirm that $K = C^*$.
3. What does the implied signal-to-noise ratio say about how agents perceive the price?

```{exercise-end}
```

```{solution-start} pmis_ex4
:class: dropdown
```

With $A = G = 1$, $Q = q$, and $R = r$, the Riccati equation of {doc}`kalman` for the one-step-ahead prediction variance $\Sigma$ of $\mu_t$ is $\Sigma_{t+1} = \Sigma_t - \Sigma_t^2/(\Sigma_t + r) + q$.

At its fixed point, $\Sigma r/(\Sigma + r) = \Sigma - q$, and the gain is $K = \Sigma/(\Sigma + r)$.

Hence $(1 - K)\Sigma = \Sigma - q$, so $\Sigma = q/K$, and substituting into $K = \Sigma/(\Sigma + r)$ gives $q/r = K^2/(1 - K)$.

The forecast of $p_{t+1}$ is the forecast of $\mu_{t+1}$, which equals the filtered estimate of $\mu_t$: $\hat\mu_{t+1} = \hat\mu_t + K(p_t - \hat\mu_t) = K p_t + (1 - K)\hat\mu_t$, which is {eq}`pm_bray2` with $C = K$.

```{code-cell} ipython3
r = 1.0
q = r * C_star**2 / (1 - C_star)
Σ = 1.0
for _ in range(10_000):
    Σ = Σ - Σ**2 / (Σ + r) + q
K = Σ / (Σ + r)
print(f"q/r = {q / r:.5f},  steady-state Kalman gain K = {K:.4f},  C* = {C_star:.4f}")
```

The equilibrium beliefs imply a signal-to-noise ratio of less than one percent.

Agents perceive the price as almost entirely transitory noise around a level that drifts very slowly, and that slowly drifting level is how their model mimics the constant mean $\nu$.

```{solution-end}
```
