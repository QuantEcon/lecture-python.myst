---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(chow_business_cycles)=

```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# The Acceleration Principle and the Nature of Business Cycles

```{contents} Contents
:depth: 2
```

## Overview

This lecture studies two classic papers by Gregory Chow on business cycles in linear dynamic models:

- {cite}`Chow1968`: why acceleration-type investment behavior matters for oscillations, and how to read stochastic dynamics through autocovariances and spectral densities
- {cite}`ChowLevitan1969`: how those tools look when applied to a calibrated macroeconometric model of the U.S. economy

These papers sit right at the intersection of three themes in this lecture series:

- The multiplier–accelerator mechanism in {doc}`samuelson`
- Linear stochastic difference equations and autocovariances in {doc}`linear_models`
- Eigenmodes of multivariate dynamics in {doc}`var_dmd`
- Fourier ideas in {doc}`eig_circulant` (and, for empirical estimation, the advanced lecture [Estimation of Spectra](https://python-advanced.quantecon.org/estspec.html#))

We will keep coming back to three ideas:

- In deterministic models, oscillations correspond to complex eigenvalues of a transition matrix.
- In stochastic models, a "cycle" shows up as a local peak in a (univariate) spectral density.
- Spectral peaks depend on eigenvalues, but also on how shocks enter (the covariance matrix $V$) and on how observables load on eigenmodes.

## A linear system with shocks

Both papers analyze (or reduce to) a first-order linear stochastic system

```{math}
:label: chow_var1

y_t = A y_{t-1} + u_t,
\qquad
\mathbb E[u_t] = 0,
\qquad
\mathbb E[u_t u_t^\top] = V,
\qquad
\mathbb E[u_t u_{t-k}^\top] = 0 \ (k \neq 0).
```

When the eigenvalues of $A$ are strictly inside the unit circle, the process is (covariance) stationary and its autocovariances exist.

In the notation of {doc}`linear_models`, this is the same stability condition that guarantees a unique solution to a discrete Lyapunov equation.

Define the lag-$k$ autocovariance matrices

```{math}
:label: chow_autocov_def

\Gamma_k := \mathbb E[y_t y_{t-k}^\top] .
```

Standard calculations (also derived in {cite}`Chow1968`) give the recursion

```{math}
:label: chow_autocov_rec

\Gamma_k = A \Gamma_{k-1}, \quad k \ge 1,
\qquad\text{and}\qquad
\Gamma_0 = A \Gamma_0 A^\top + V.
```

The second equation is the discrete Lyapunov equation for $\Gamma_0$.

## From autocovariances to spectra

Chow’s key step is to translate the autocovariance sequence $\{\Gamma_k\}$ into a frequency-domain object.

The **spectral density matrix** is the Fourier transform of $\Gamma_k$:

```{math}
:label: chow_spectral_def

F(\omega) := \frac{1}{2\pi} \sum_{k=-\infty}^{\infty} \Gamma_k e^{-i \omega k},
\qquad \omega \in [0, \pi].
```

For the VAR(1) system {eq}`chow_var1`, this sum has a closed form

```{math}
:label: chow_spectral_closed

F(\omega)
= \frac{1}{2\pi}
\left(I - A e^{-i\omega}\right)^{-1}
V
\left(I - A^\top e^{i\omega}\right)^{-1}.
```

Intuitively, $F(\omega)$ tells us how much variation in $y_t$ is associated with cycles of (angular) frequency $\omega$.

The corresponding cycle length is

```{math}
:label: chow_period

T(\omega) = \frac{2\pi}{\omega}.
```

The advanced lecture {doc}`advanced:estspec` explains how to estimate $F(\omega)$ from data.

Here we focus on the model-implied spectrum.

We will use the following imports and helper functions throughout the lecture.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

def spectral_density_var1(A, V, ω_grid):
    """Spectral density matrix for VAR(1): y_t = A y_{t-1} + u_t."""
    A, V = np.asarray(A), np.asarray(V)
    n = A.shape[0]
    I = np.eye(n)
    F = np.empty((len(ω_grid), n, n), dtype=complex)
    for k, ω in enumerate(ω_grid):
        H = np.linalg.inv(I - np.exp(-1j * ω) * A)
        F[k] = (H @ V @ H.conj().T) / (2 * np.pi)
    return F

def spectrum_of_linear_combination(F, b):
    """Spectrum of x_t = b'y_t given the spectral matrix F(ω)."""
    b = np.asarray(b).reshape(-1, 1)
    return np.array([np.real((b.T @ F[k] @ b).item()) for k in range(F.shape[0])])

def simulate_var1(A, V, T, burn=200, seed=1234):
    """Simulate y_t = A y_{t-1} + u_t with u_t ~ N(0, V)."""
    rng = np.random.default_rng(seed)
    A, V = np.asarray(A), np.asarray(V)
    n = A.shape[0]
    chol = np.linalg.cholesky(V)
    y = np.zeros((T + burn, n))
    for t in range(1, T + burn):
        y[t] = A @ y[t - 1] + chol @ rng.standard_normal(n)
    return y[burn:]

def sample_autocorrelation(x, max_lag):
    """Sample autocorrelation of a 1d array from lag 0 to max_lag."""
    x = np.asarray(x)
    x = x - x.mean()
    denom = np.dot(x, x)
    acf = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        acf[k] = np.dot(x[:-k] if k else x, x[k:]) / denom
    return acf
```

## Deterministic propagation and acceleration

Chow {cite}`Chow1968` begins with a clean deterministic question:

> If you build a macro model using only standard demand equations with simple distributed lags, can the system generate sustained oscillations without acceleration?

He shows that, under natural sign restrictions, the answer is no.

### A demand system without acceleration

Consider a system where each component $y_{it}$ responds to aggregate output $Y_t$ and its own lag:

```{math}
:label: chow_simple_demand

y_{it} = a_i Y_t + b_i y_{i,t-1},
\qquad
Y_t = \sum_i y_{it},
\qquad
a_i > 0,\; b_i > 0.
```

Chow shows that the implied transition matrix has real characteristic roots, and that if $\sum_i a_i < 1$ these roots are also positive.

In that case, solutions are linear combinations of decaying exponentials without persistent sign-switching components, so there are no “business-cycle-like” oscillations driven purely by internal propagation.

### What acceleration changes

For investment (and some durables), Chow argues that a more relevant starting point is a *stock adjustment* equation (demand for a stock), e.g.

```{math}
:label: chow_stock_adj

s_{it} = \alpha_i Y_t + \beta_i s_{i,t-1}.
```

If flow investment is proportional to the change in the desired stock, differencing introduces terms in $\Delta Y_t$.

That "acceleration" structure creates negative coefficients (in lagged levels), which makes complex roots possible.

This connects directly to {doc}`samuelson`, where acceleration is the key ingredient that can generate damped or persistent oscillations in a deterministic second-order difference equation.

To see the mechanism with minimal algebra, take the multiplier–accelerator law of motion

```{math}
Y_t = c Y_{t-1} + v (Y_{t-1} - Y_{t-2}),
```

and rewrite it as a first-order system in $(Y_t, Y_{t-1})$.

```{code-cell} ipython3
def samuelson_transition(c, v):
    return np.array([[c + v, -v], [1.0, 0.0]])

c = 0.6
v_values = (0.0, 0.8)
A_list = [samuelson_transition(c, v) for v in v_values]

for v, A in zip(v_values, A_list):
    eig = np.linalg.eigvals(A)
    print(f"v={v:.1f}, eigenvalues={eig}")

# impulse responses from a one-time unit shock in Y
T = 40
s0 = np.array([1.0, 0.0])
irfs = []
for A in A_list:
    s = s0.copy()
    path = np.empty(T + 1)
    for t in range(T + 1):
        path[t] = s[0]
        s = A @ s
    irfs.append(path)

# model-implied spectra for the stochastic version with shocks in the Y equation
freq = np.linspace(1e-4, 0.5, 2500)  # cycles/period
ω_grid = 2 * np.pi * freq
V = np.array([[1.0, 0.0], [0.0, 0.0]])

spectra = []
for A in A_list:
    F = spectral_density_var1(A, V, ω_grid)
    f11 = np.real(F[:, 0, 0])
    spectra.append(f11 / np.trapz(f11, freq))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(T + 1), irfs[0], lw=1.8, label="no acceleration")
axes[0].plot(range(T + 1), irfs[1], lw=1.8, label="with acceleration")
axes[0].axhline(0.0, lw=0.8)
axes[0].set_xlabel("time")
axes[0].set_ylabel(r"$Y_t$")
axes[0].legend(frameon=False)

axes[1].plot(freq, spectra[0], lw=1.8, label="no acceleration")
axes[1].plot(freq, spectra[1], lw=1.8, label="with acceleration")
axes[1].set_xlabel(r"frequency $\omega/2\pi$")
axes[1].set_ylabel("normalized spectrum")
axes[1].set_xlim([0.0, 0.5])
axes[1].legend(frameon=False)

plt.tight_layout()
plt.show()
```

The left panel shows that acceleration creates oscillatory impulse responses.

The right panel shows the corresponding spectral signature: a peak at interior frequencies.

### How the accelerator shifts the spectral peak

As we increase the accelerator $v$, the complex eigenvalues rotate further from the real axis, shifting the spectral peak to higher frequencies.

```{code-cell} ipython3
v_grid = np.linspace(0.2, 1.2, 6)
c = 0.6
freq_fine = np.linspace(1e-4, 0.5, 2000)
ω_fine = 2 * np.pi * freq_fine
V_acc = np.array([[1.0, 0.0], [0.0, 0.0]])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for v in v_grid:
    A = samuelson_transition(c, v)
    eig = np.linalg.eigvals(A)
    F = spectral_density_var1(A, V_acc, ω_fine)
    f11 = np.real(F[:, 0, 0])
    f11_norm = f11 / np.trapz(f11, freq_fine)

    # plot eigenvalues
    axes[0].scatter(eig.real, eig.imag, s=40, label=f'$v={v:.1f}$')

    # plot spectrum
    axes[1].plot(freq_fine, f11_norm, lw=1.5, label=f'$v={v:.1f}$')

# unit circle
θ_circle = np.linspace(0, 2*np.pi, 100)
axes[0].plot(np.cos(θ_circle), np.sin(θ_circle), 'k--', lw=0.8)
axes[0].set_xlabel('real part')
axes[0].set_ylabel('imaginary part')
axes[0].set_aspect('equal')
axes[0].legend(frameon=False, fontsize=8)

axes[1].set_xlabel(r'frequency $\omega/2\pi$')
axes[1].set_ylabel('normalized spectrum')
axes[1].set_xlim([0, 0.5])
axes[1].legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.show()
```

Larger $v$ pushes the eigenvalues further off the real axis, shifting the spectral peak to higher frequencies.

When $v$ is large enough that eigenvalues leave the unit circle, the system becomes explosive.

## Spectral peaks are not just eigenvalues

With shocks, the deterministic question ("does the system oscillate?") becomes: at which cycle lengths does the variance of $y_t$ concentrate?

In this lecture, a "cycle" means a local peak in a univariate spectrum $f_{ii}(\omega)$.

Chow's point in {cite}`Chow1968` is that eigenvalues help interpret spectra, but they do not determine peaks by themselves.

Two extra ingredients matter:

- how shocks load on the eigenmodes (the covariance matrix $V$),
- how the variable of interest mixes those modes.

The next simulations isolate these effects.

### Complex roots: a peak and an oscillating autocorrelation

Take a stable “rotation–contraction” matrix

```{math}
:label: chow_rot

A = r
\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix},
\qquad 0 < r < 1,
```

whose eigenvalues are $r e^{\pm i\theta}$.

When $r$ is close to 1, the spectrum shows a pronounced peak near $\omega \approx \theta$.

```{code-cell} ipython3
def rotation_contraction(r, θ):
    c, s = np.cos(θ), np.sin(θ)
    return r * np.array([[c, -s], [s, c]])

θ = np.pi / 3
r_values = (0.95, 0.4)
ω_grid = np.linspace(1e-3, np.pi - 1e-3, 800)
V = np.eye(2)

acfs = []
spectra = []
for r in r_values:
    A = rotation_contraction(r, θ)

    y = simulate_var1(A, V, T=5000, burn=500, seed=1234)
    acfs.append(sample_autocorrelation(y[:, 0], 40))

    F = spectral_density_var1(A, V, ω_grid)
    spectra.append(np.real(F[:, 0, 0]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for r, acf in zip(r_values, acfs):
    axes[0].plot(range(len(acf)), acf, lw=1.8, label=fr"$r={r}$")
axes[0].axhline(0.0, lw=0.8)
axes[0].set_xlabel("lag")
axes[0].set_ylabel("autocorrelation")
axes[0].legend(frameon=False)

for r, f11 in zip(r_values, spectra):
    axes[1].plot(ω_grid / np.pi, f11, lw=1.8, label=fr"$r={r}$")
axes[1].axvline(θ / np.pi, ls="--", lw=1.0, label=r"$\theta/\pi$")
axes[1].set_xlabel(r"frequency $\omega/\pi$")
axes[1].set_ylabel(r"$f_{11}(\omega)$")
axes[1].legend(frameon=False)

plt.tight_layout()
plt.show()
```

When $r$ is close to 1, the autocorrelation oscillates slowly and the spectrum has a sharp peak near $\theta$.

When $r$ is smaller, oscillations die out quickly and the spectrum is flatter.

### How shock structure shapes the spectrum

Even with the same transition matrix, different shock covariance structures produce different spectral shapes.

Here we fix $r = 0.9$ and vary the correlation between the two shocks.

```{code-cell} ipython3
r_fixed = 0.9
A_fixed = rotation_contraction(r_fixed, θ)
corr_values = [-0.9, 0.0, 0.9]

fig, ax = plt.subplots(figsize=(9, 4))
for corr in corr_values:
    V_corr = np.array([[1.0, corr], [corr, 1.0]])
    F = spectral_density_var1(A_fixed, V_corr, ω_grid)
    f11 = np.real(F[:, 0, 0])
    f11_norm = f11 / np.trapz(f11, ω_grid / np.pi)
    ax.plot(ω_grid / np.pi, f11_norm, lw=1.8, label=fr'$\rho = {corr}$')

ax.axvline(θ / np.pi, ls='--', lw=1.0, color='gray')
ax.set_xlabel(r'frequency $\omega/\pi$')
ax.set_ylabel('normalized spectrum')
ax.legend(frameon=False)
plt.show()
```

The peak location is unchanged, but the peak height depends on the shock correlation.

This illustrates that eigenvalues alone do not determine the full spectral shape.

### Complex roots: an oscillatory mode can be hidden

Complex roots are not sufficient for a visible peak in the spectrum of every observed series.

Even if the state vector contains an oscillatory mode, a variable can be dominated by a non-oscillatory component.

The next example combines a rotation–contraction block with a very persistent real root, and then looks at a mixture that is dominated by the persistent component.

```{code-cell} ipython3
A_osc = rotation_contraction(0.95, θ)
A = np.block([
    [A_osc, np.zeros((2, 1))],
    [np.zeros((1, 2)), np.array([[0.99]])]
])

# shocks hit the persistent component much more strongly
V = np.diag([1.0, 1.0, 50.0])

ω_grid_big = np.linspace(1e-3, np.pi - 1e-3, 1200)
F = spectral_density_var1(A, V, ω_grid_big)

x_grid = ω_grid_big / np.pi
f_y1 = np.real(F[:, 0, 0])

b = np.array([0.05, 0.0, 1.0])
f_mix = spectrum_of_linear_combination(F, b)

f_y1_norm = f_y1 / np.trapz(f_y1, x_grid)
f_mix_norm = f_mix / np.trapz(f_mix, x_grid)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(x_grid, f_y1_norm, lw=1.8, label=r"$y_1$")
ax.plot(x_grid, f_mix_norm, lw=1.8, label=r"$x = 0.05\,y_1 + y_3$")
ax.set_xlabel(r"frequency $\omega/\pi$")
ax.set_ylabel("normalized spectrum")
ax.legend(frameon=False)
plt.show()
```

Here the oscillatory mode is still present (the $y_1$ spectrum peaks away from zero), but the mixture $x$ is dominated by the near-unit root and hence by very low frequencies.

### Real roots: a peak from mixing shocks

Chow also constructs examples where all roots are real and positive yet a linear combination displays a local spectral peak.

The mechanism is that cross-correlation in shocks can generate cyclical-looking behavior.

Here is a close analog of Chow’s two-root illustration.

```{code-cell} ipython3
A = np.diag([0.1, 0.9])
V = np.array([[1.0, 0.8], [0.8, 1.0]])
b = np.array([1.0, -0.01])

F = spectral_density_var1(A, V, ω_grid)
f_x = spectrum_of_linear_combination(F, b)
imax = np.argmax(f_x)
ω_star = ω_grid[imax]
period_star = 2 * np.pi / ω_star

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(ω_grid / np.pi, f_x)
ax.scatter([ω_star / np.pi], [f_x[imax]], zorder=3)
ax.set_xlabel(r"frequency $\omega/\pi$")
ax.set_ylabel(r"$f_x(\omega)$")
plt.show()
print(f"peak period ≈ {period_star:.1f}")
```

The lesson is the same as Chow’s: in multivariate stochastic systems, “cycle-like” spectra are shaped not only by eigenvalues, but also by how shocks enter ($V$) and how variables combine (the analogue of Chow’s eigenvector matrix).

## A calibrated model in the frequency domain

Chow and Levitan {cite}`ChowLevitan1969` use the frequency-domain objects from {cite}`Chow1968` to study a calibrated annual macroeconometric model.

They work with five annual aggregates

- $y_1 = C$ (consumption),
- $y_2 = I_1$ (equipment plus inventories),
- $y_3 = I_2$ (construction),
- $y_4 = R_a$ (long rate),
- $y_5 = Y_1 = C + I_1 + I_2$ (private-domestic gnp),

and add $y_6 = y_{1,t-1}$ to rewrite the original system in first-order form.

Throughout this section, frequency is measured in cycles per year, $f = \omega/2\pi \in [0, 1/2]$.

Following the paper, we normalize each spectrum to have area 1 over $[0, 1/2]$ so plots compare shape rather than scale.

Our goal is to reconstruct the transition matrix $A$ and then compute and interpret the model-implied spectra, gains/coherences, and phase differences.

### The cycle subsystem

The paper starts from a reduced form with exogenous inputs,

```{math}
:label: chow_reduced_full

y_t = A y_{t-1} + C x_t + u_t.
```

To study cycles, they remove the deterministic component attributable to $x_t$ and focus on the zero-mean subsystem

```{math}
:label: chow_cycle_system

y_t = A y_{t-1} + u_t.
```

For second moments, the only additional ingredient is the covariance matrix $V = \mathbb E[u_t u_t^\top]$.

Chow and Levitan compute it from structural parameters via

```{math}
:label: chow_v_from_structural

V = M^{-1} \Sigma (M^{-1})^\top
```

where $\Sigma$ is the covariance of structural residuals and $M$ is the matrix of contemporaneous structural coefficients.

Here we take $A$ and $V$ as given and ask what they imply for spectra and cross-spectra.

### Reported shock covariance

Chow and Levitan report the $6 \times 6$ reduced-form shock covariance matrix $V$ (scaled by $10^{-7}$):

```{math}
:label: chow_V_matrix

V = \begin{bmatrix}
8.250 & 7.290 & 2.137 & 2.277 & 17.68 & 0 \\
7.290 & 7.135 & 1.992 & 2.165 & 16.42 & 0 \\
2.137 & 1.992 & 0.618 & 0.451 & 4.746 & 0 \\
2.277 & 2.165 & 0.451 & 1.511 & 4.895 & 0 \\
17.68 & 16.42 & 4.746 & 4.895 & 38.84 & 0 \\
0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}.
```

The sixth row and column are zeros because $y_6$ is an identity (lagged $y_1$).

### Reported eigenvalues

The transition matrix $A$ has six characteristic roots:

```{math}
:label: chow_eigenvalues

\begin{aligned}
\lambda_1 &= 0.9999725, \quad \lambda_2 = 0.9999064, \quad \lambda_3 = 0.4838, \\
\lambda_4 &= 0.0761 + 0.1125i, \quad \lambda_5 = 0.0761 - 0.1125i, \quad \lambda_6 = -0.00004142.
\end{aligned}
```

Two roots are near unity because two structural equations are in first differences.

One root ($\lambda_6$) is theoretically zero because of the identity $y_5 = y_1 + y_2 + y_3$.

The complex conjugate pair $\lambda_{4,5}$ has modulus $|\lambda_4| = \sqrt{0.0761^2 + 0.1125^2} \approx 0.136$.

### Reported eigenvectors

The right eigenvector matrix $B$ (columns are eigenvectors corresponding to $\lambda_1, \ldots, \lambda_6$):

```{math}
:label: chow_B_matrix

B = \begin{bmatrix}
-0.008 & 1.143 & 0.320 & 0.283+0.581i & 0.283-0.581i & 0.000 \\
-0.000 & 0.013 & -0.586 & -2.151+0.742i & -2.151-0.742i & 2.241 \\
-0.001 & 0.078 & 0.889 & -0.215+0.135i & -0.215-0.135i & 0.270 \\
1.024 & 0.271 & 0.069 & -0.231+0.163i & -0.231-0.163i & 0.307 \\
-0.009 & 1.235 & 0.623 & -2.082+1.468i & -2.082-1.468i & 2.766 \\
-0.008 & 1.143 & 0.662 & 4.772+0.714i & 4.772-0.714i & -4.399
\end{bmatrix}.
```

Together, $V$, $\{\lambda_i\}$, and $B$ are sufficient to compute all spectral and cross-spectral densities.

### Reconstructing $A$ and computing $F(\omega)$

The paper reports $(\lambda, B, V)$, which is enough to reconstruct
$A = B \, \mathrm{diag}(\lambda_1,\dots,\lambda_6)\, B^{-1}$ and then compute the model-implied spectral objects.

```{code-cell} ipython3
λ = np.array([
    0.9999725, 0.9999064, 0.4838,
    0.0761 + 0.1125j, 0.0761 - 0.1125j, -0.00004142
], dtype=complex)

B = np.array([
    [-0.008, 1.143, 0.320, 0.283+0.581j, 0.283-0.581j, 0.000],
    [-0.000, 0.013, -0.586, -2.151+0.742j, -2.151-0.742j, 2.241],
    [-0.001, 0.078, 0.889, -0.215+0.135j, -0.215-0.135j, 0.270],
    [1.024, 0.271, 0.069, -0.231+0.163j, -0.231-0.163j, 0.307],
    [-0.009, 1.235, 0.623, -2.082+1.468j, -2.082-1.468j, 2.766],
    [-0.008, 1.143, 0.662, 4.772+0.714j, 4.772-0.714j, -4.399]
], dtype=complex)

V = np.array([
    [8.250, 7.290, 2.137, 2.277, 17.68, 0],
    [7.290, 7.135, 1.992, 2.165, 16.42, 0],
    [2.137, 1.992, 0.618, 0.451, 4.746, 0],
    [2.277, 2.165, 0.451, 1.511, 4.895, 0],
    [17.68, 16.42, 4.746, 4.895, 38.84, 0],
    [0, 0, 0, 0, 0, 0]
]) * 1e-7

D_λ = np.diag(λ)
A_chow = B @ D_λ @ np.linalg.inv(B)
A_chow = np.real(A_chow)  # drop tiny imaginary parts from reported rounding
print("eigenvalues of reconstructed A:")
print(np.linalg.eigvals(A_chow).round(6))
```

### Canonical coordinates

Chow's canonical transformation uses $z_t = B^{-1} y_t$, giving dynamics $z_t = D_\lambda z_{t-1} + e_t$.

An algebraic detail: the closed form for $F(\omega)$ uses $A^\top$ (real transpose) rather than a conjugate transpose.

Accordingly, the canonical shock covariance is

```{math}
W = B^{-1} V (B^{-1})^\top.
```

```{code-cell} ipython3
B_inv = np.linalg.inv(B)
W = B_inv @ V @ B_inv.T
print("diagonal of W:")
print(np.diag(W).round(10))
```

### Spectral density via eigendecomposition

Chow's closed-form formula for the spectral density matrix is

```{math}
:label: chow_spectral_eigen

F(\omega)
= B \left[ \frac{w_{ij}}{(1 - \lambda_i e^{-i\omega})(1 - \lambda_j e^{i\omega})} \right] B^\top,
```

where $w_{ij}$ are elements of the canonical shock covariance $W$.

```{code-cell} ipython3
def spectral_density_chow(λ, B, W, ω_grid):
    """Spectral density via Chow's eigendecomposition formula."""
    p = len(λ)
    F = np.zeros((len(ω_grid), p, p), dtype=complex)
    for k, ω in enumerate(ω_grid):
        F_star = np.zeros((p, p), dtype=complex)
        for i in range(p):
            for j in range(p):
                denom = (1 - λ[i] * np.exp(-1j * ω)) * (1 - λ[j] * np.exp(1j * ω))
                F_star[i, j] = W[i, j] / denom
        F[k] = B @ F_star @ B.T
    return F / (2 * np.pi)

freq = np.linspace(1e-4, 0.5, 5000)     # cycles/year in [0, 1/2]
ω_grid = 2 * np.pi * freq               # radians in [0, π]
F_chow = spectral_density_chow(λ, B, W, ω_grid)
```

### Where is variance concentrated?

Normalizing each spectrum to have unit area over $[0, 1/2]$ lets us compare shapes rather than scales.

```{code-cell} ipython3
variable_names = ['$C$', '$I_1$', '$I_2$', '$R_a$', '$Y_1$']
freq_ticks = [1/18, 1/9, 1/6, 1/4, 1/3, 1/2]
freq_labels = [r'$\frac{1}{18}$', r'$\frac{1}{9}$', r'$\frac{1}{6}$',
               r'$\frac{1}{4}$', r'$\frac{1}{3}$', r'$\frac{1}{2}$']

def paper_frequency_axis(ax):
    ax.set_xlim([0.0, 0.5])
    ax.set_xticks(freq_ticks)
    ax.set_xticklabels(freq_labels)
    ax.set_xlabel(r'frequency $\omega/2\pi$')

# Normalized spectra (areas set to 1)
S = np.real(np.diagonal(F_chow, axis1=1, axis2=2))[:, :5]  # y1..y5
areas = np.trapz(S, freq, axis=0)
S_norm = S / areas
mask = freq >= 0.0

fig, axes = plt.subplots(1, 2, figsize=(10, 6))

# Figure I.1: consumption (log scale)
axes[0].plot(freq[mask], S_norm[mask, 0], lw=1.8)
axes[0].set_yscale('log')
paper_frequency_axis(axes[0])
axes[0].set_ylabel(r'normalized $f_{11}(\omega)$')

# Figure I.2: equipment + inventories (log scale)
axes[1].plot(freq[mask], S_norm[mask, 1], lw=1.8)
axes[1].set_yscale('log')
paper_frequency_axis(axes[1])
axes[1].set_ylabel(r'normalized $f_{22}(\omega)$')

plt.tight_layout()
plt.show()

i_peak = np.argmax(S_norm[mask, 1])
f_peak = freq[mask][i_peak]
print(f"Peak within [1/18, 1/2]: frequency ≈ {f_peak:.3f} cycles/year, period ≈ {1/f_peak:.2f} years.")
```

Both spectra are dominated by very low frequencies, reflecting the near-unit eigenvalues.

This is the "typical spectral shape" of macroeconomic time series.

(These patterns match Figures I.1–I.2 of {cite}`ChowLevitan1969`.)

### How variables move together across frequencies

Beyond univariate spectra, we can ask how pairs of variables covary at each frequency.

The **cross-spectrum** $f_{ij}(\omega) = c_{ij}(\omega) - i \cdot q_{ij}(\omega)$ decomposes into the cospectrum $c_{ij}$ and the quadrature spectrum $q_{ij}$.

The **cross-amplitude** is $g_{ij}(\omega) = |f_{ij}(\omega)| = \sqrt{c_{ij}^2 + q_{ij}^2}$.

The **squared coherence** measures linear association at frequency $\omega$:

```{math}
:label: chow_coherence

R^2_{ij}(\omega) = \frac{|f_{ij}(\omega)|^2}{f_{ii}(\omega) f_{jj}(\omega)} \in [0, 1].
```

The **gain** is the frequency-response coefficient when regressing $y_i$ on $y_j$:

```{math}
:label: chow_gain

G_{ij}(\omega) = \frac{|f_{ij}(\omega)|}{f_{jj}(\omega)}.
```

The **phase** captures lead-lag relationships (in radians):

```{math}
:label: chow_phase

\Delta_{ij}(\omega) = \tan^{-1}\left( \frac{q_{ij}(\omega)}{c_{ij}(\omega)} \right).
```

```{code-cell} ipython3
def cross_spectral_measures(F, i, j):
    """Compute coherence, gain (y_i on y_j), and phase between variables i and j."""
    f_ij = F[:, i, j]
    f_ii, f_jj = np.real(F[:, i, i]), np.real(F[:, j, j])
    g_ij = np.abs(f_ij)
    coherence = (g_ij**2) / (f_ii * f_jj)
    gain = g_ij / f_jj
    phase = np.arctan2(-np.imag(f_ij), np.real(f_ij))
    return coherence, gain, phase
```

We now plot gain and coherence as in Figures II.1-II.3 of {cite}`ChowLevitan1969`.

```{code-cell} ipython3
gnp_idx = 4

fig, axes = plt.subplots(1, 3, figsize=(14, 6))

for idx, var_idx in enumerate([0, 1, 2]):
    coherence, gain, phase = cross_spectral_measures(F_chow, var_idx, gnp_idx)
    ax = axes[idx]

    ax.plot(freq[mask], coherence[mask],
            lw=1.8, label=rf'$R^2_{{{var_idx+1}5}}(\omega)$')
    ax.plot(freq[mask], gain[mask],
            lw=1.8, label=rf'$G_{{{var_idx+1}5}}(\omega)$')

    paper_frequency_axis(ax)
    ax.set_ylim([0, 1.0])
    ax.set_ylabel('gain, coherence')
    ax.legend(frameon=False, loc='best')

plt.tight_layout()
plt.show()
```

Coherence is high at low frequencies for all three components, meaning long-run movements track output closely.

Gains differ: consumption smooths (gain below 1), while investment responds more strongly at higher frequencies.

(These patterns match Figures II.1-II.3 of {cite}`ChowLevitan1969`.)

### Lead-lag relationships

The phase tells us which variable leads at each frequency.

Positive phase means output leads the component; negative phase means the component leads output.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 6))

labels = [r'$\psi_{15}(\omega)/2\pi$', r'$\psi_{25}(\omega)/2\pi$',
          r'$\psi_{35}(\omega)/2\pi$', r'$\psi_{45}(\omega)/2\pi$']

for var_idx in range(4):
    coherence, gain, phase = cross_spectral_measures(F_chow, var_idx, gnp_idx)
    phase_cycles = phase / (2 * np.pi)
    ax.plot(freq[mask], phase_cycles[mask], lw=1.8, label=labels[var_idx])

ax.axhline(0, lw=0.8)
paper_frequency_axis(ax)
ax.set_ylabel('phase difference in cycles')
ax.set_ylim([-0.25, 0.25])
ax.set_yticks([-0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25])
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.show()
```

At business-cycle frequencies, consumption tends to lag output while equipment and inventories tend to lead.

The interest rate is roughly coincident.

(This matches Figure III of {cite}`ChowLevitan1969`.)

### Building blocks of spectral shape

Each eigenvalue contributes a characteristic spectral shape through the **scalar kernel**

```{math}
:label: chow_scalar_kernel

g_i(\omega) = \frac{1 - |\lambda_i|^2}{|1 - \lambda_i e^{-i\omega}|^2} = \frac{1 - |\lambda_i|^2}{1 + |\lambda_i|^2 - 2 \text{Re}(\lambda_i) \cos\omega + 2 \text{Im}(\lambda_i) \sin\omega}.
```

For real $\lambda_i$, this simplifies to

```{math}
g_i(\omega) = \frac{1 - \lambda_i^2}{1 + \lambda_i^2 - 2\lambda_i \cos\omega}.
```

Each observable spectral density is a linear combination of these kernels (plus cross-terms).

```{code-cell} ipython3
def scalar_kernel(λ_i, ω_grid):
    """Chow's scalar spectral kernel g_i(ω)."""
    λ_i = complex(λ_i)
    mod_sq = np.abs(λ_i)**2
    return np.array([(1 - mod_sq) / np.abs(1 - λ_i * np.exp(-1j * ω))**2 for ω in ω_grid])

fig, ax = plt.subplots(figsize=(10, 5))
for i, λ_i in enumerate(λ[:4]):
    if np.abs(λ_i) > 0.01:
        g_i = scalar_kernel(λ_i, ω_grid)
        label = f'$\\lambda_{i+1}$ = {λ_i:.4f}' if np.isreal(λ_i) else f'$\\lambda_{i+1}$ = {λ_i:.3f}'
        ax.semilogy(freq, g_i, label=label, lw=1.5)
ax.set_xlabel(r'frequency $\omega/2\pi$')
ax.set_ylabel('$g_i(\\omega)$')
ax.set_xlim([1/18, 0.5])
ax.set_xticks(freq_ticks)
ax.set_xticklabels(freq_labels)
ax.legend(frameon=False)
plt.show()
```

Near-unit eigenvalues produce kernels sharply peaked at low frequencies.

Smaller eigenvalues produce flatter kernels.

The complex pair ($\lambda_{4,5}$) has such small modulus that its kernel is nearly flat.

### Why the spectra look the way they do

The two near-unit eigenvalues generate strong low-frequency power.

The moderate eigenvalue ($\lambda_3 \approx 0.48$) contributes a flatter component.

The complex pair has small modulus ($|\lambda_{4,5}| \approx 0.136$), so it cannot generate a pronounced interior peak.

The near-zero eigenvalue reflects the accounting identity $Y_1 = C + I_1 + I_2$.

This illustrates Chow's message: eigenvalues guide intuition, but observed spectra also depend on how shocks excite the modes and how observables combine them.

### Summary

The calibrated model reveals three patterns: (1) most variance sits at very low frequencies due to near-unit eigenvalues; (2) consumption smooths while investment amplifies high-frequency movements; (3) consumption lags output at business-cycle frequencies while investment leads.

## Wrap-up

Chow {cite}`Chow1968` emphasizes two complementary diagnostics for linear macro models: how eigenvalues shape deterministic propagation, and how spectra summarize stochastic dynamics.

Chow and Levitan {cite}`ChowLevitan1969` then show what these objects look like in a calibrated system: strong low-frequency power, frequency-dependent gains/coherences, and lead–lag relations that vary with the cycle length.

To connect this to data, pair the model-implied objects here with the advanced lecture [Estimation of Spectra](https://python-advanced.quantecon.org/estspec.html#).

## A structural view of acceleration

Chow {cite}`Chow1968` provides a structural interpretation of how acceleration enters the model.

The starting point is a stock-adjustment demand for capital:

```{math}
:label: chow_stock_adj_struct

s_{it} = a_i Y_t + b_i s_{i,t-1}
```

where $s_{it}$ is the desired stock of capital type $i$, $Y_t$ is aggregate output, and $(a_i, b_i)$ are parameters.

Net investment is the stock change:

```{math}
:label: chow_net_inv

y^n_{it} = \Delta s_{it} = a_i \Delta Y_t + b_i y^n_{i,t-1}.
```

For gross investment with depreciation rate $\delta_i$:

```{math}
:label: chow_gross_inv

y_{it} = a_i [Y_t - (1-\delta_i) Y_{t-1}] + b_i y_{i,t-1}.
```

The parameters $(a_i, b_i, \delta_i)$ are the key "acceleration equation" parameters.

The term $a_i \Delta Y_t$ is the acceleration effect: investment responds to *changes* in output, not just levels.

This creates negative coefficients on lagged output levels, which in turn makes complex roots (and hence oscillatory components) possible in the characteristic equation.

## Exercises

```{exercise}
:label: chow_cycles_ex1

In the rotation-contraction example, fix $\theta$ and vary $r$ in a grid between $0.2$ and $0.99$.

1. For each $r$, compute the frequency $\omega^*(r)$ that maximizes $f_{11}(\omega)$.
2. Plot $\omega^*(r)$ and the implied peak period $2\pi/\omega^*(r)$ as functions of $r$.

How does the peak location behave as $r \uparrow 1$?
```

```{solution-start} chow_cycles_ex1
:class: dropdown
```

```{code-cell} ipython3
r_grid = np.linspace(0.2, 0.99, 50)
θ = np.pi / 3
ω_grid_ex = np.linspace(1e-3, np.pi - 1e-3, 1000)
V_ex = np.eye(2)

ω_star = np.zeros(len(r_grid))
period_star = np.zeros(len(r_grid))
for idx, r in enumerate(r_grid):
    A_ex = rotation_contraction(r, θ)
    F_ex = spectral_density_var1(A_ex, V_ex, ω_grid_ex)
    f11 = np.real(F_ex[:, 0, 0])
    i_max = np.argmax(f11)
    ω_star[idx] = ω_grid_ex[i_max]
    period_star[idx] = 2 * np.pi / ω_star[idx]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(r_grid, ω_star / np.pi, lw=1.8)
axes[0].axhline(θ / np.pi, ls='--', lw=1.0, label=r'$\theta/\pi$')
axes[0].set_xlabel('$r$')
axes[0].set_ylabel(r'$\omega^*/\pi$')
axes[0].legend(frameon=False)

axes[1].plot(r_grid, period_star, lw=1.8)
axes[1].axhline(2 * np.pi / θ, ls='--', lw=1.0, label=r'$2\pi/\theta$')
axes[1].set_xlabel('$r$')
axes[1].set_ylabel('peak period')
axes[1].legend(frameon=False)
plt.tight_layout()
plt.show()
```

As $r \uparrow 1$, the peak frequency converges to $\theta$ (the argument of the complex eigenvalue).

This confirms Chow's insight: when the modulus is close to 1, the spectral peak aligns with the eigenvalue frequency.

```{solution-end}
```

```{exercise}
:label: chow_cycles_ex2

In the "real roots but a peak" example, hold $A$ fixed and vary the shock correlation (the off-diagonal entry of $V$) between $0$ and $0.99$.

When does the interior-frequency peak appear, and how does its location change?
```

```{solution-start} chow_cycles_ex2
:class: dropdown
```

```{code-cell} ipython3
A_ex2 = np.diag([0.1, 0.9])
b_ex2 = np.array([1.0, -0.01])
corr_grid = np.linspace(0, 0.99, 50)
peak_periods = []
for corr in corr_grid:
    V_ex2 = np.array([[1.0, corr], [corr, 1.0]])
    F_ex2 = spectral_density_var1(A_ex2, V_ex2, ω_grid_ex)
    f_x = spectrum_of_linear_combination(F_ex2, b_ex2)
    i_max = np.argmax(f_x)
    if 5 < i_max < len(ω_grid_ex) - 5:
        peak_periods.append(2 * np.pi / ω_grid_ex[i_max])
    else:
        peak_periods.append(np.nan)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(corr_grid, peak_periods, marker='o', lw=1.8, markersize=4)
ax.set_xlabel('shock correlation')
ax.set_ylabel('peak period')
plt.show()

threshold_idx = np.where(~np.isnan(peak_periods))[0]
if len(threshold_idx) > 0:
    print(f"interior peak appears when correlation ≥ {corr_grid[threshold_idx[0]]:.2f}")
```

The interior peak appears only when the shock correlation exceeds a threshold.

This illustrates Chow's point that spectral peaks depend on the full system structure, not just eigenvalues.

```{solution-end}
```

```{exercise}
:label: chow_cycles_ex3

Using the calibrated Chow-Levitan (1969) parameters, compute the autocovariance matrices $\Gamma_0, \Gamma_1, \ldots, \Gamma_{10}$ using:

1. The recursion $\Gamma_k = A \Gamma_{k-1}$ with $\Gamma_0$ from the Lyapunov equation.
2. Chow's eigendecomposition formula $\Gamma_k = B D_\lambda^k \Gamma_0^* B^\top$ where $\Gamma_0^*$ is the canonical covariance.

Verify that both methods give the same result.
```

```{solution-start} chow_cycles_ex3
:class: dropdown
```

```{code-cell} ipython3
from scipy.linalg import solve_discrete_lyapunov

Γ_0_lyap = solve_discrete_lyapunov(A_chow, V)
Γ_recursion = [Γ_0_lyap]
for k in range(1, 11):
    Γ_recursion.append(A_chow @ Γ_recursion[-1])

p = len(λ)
Γ_0_star = np.zeros((p, p), dtype=complex)
for i in range(p):
    for j in range(p):
        Γ_0_star[i, j] = W[i, j] / (1 - λ[i] * λ[j])

Γ_eigen = []
for k in range(11):
    D_k = np.diag(λ**k)
    Γ_eigen.append(np.real(B @ D_k @ Γ_0_star @ B.T))

print("Comparison of Γ_5 (first 3x3 block):")
print("\nRecursion method:")
print(np.real(Γ_recursion[5][:3, :3]).round(10))
print("\nEigendecomposition method:")
print(Γ_eigen[5][:3, :3].round(10))
print("\nMax absolute difference:", np.max(np.abs(np.real(Γ_recursion[5]) - Γ_eigen[5])))
```

Both methods produce essentially identical results, up to numerical precision.

```{solution-end}
```

```{exercise}
:label: chow_cycles_ex4

Modify the Chow-Levitan model by changing $\lambda_3$ from $0.4838$ to $0.95$.

1. Recompute the spectral densities.
2. How does this change affect the spectral shape for each variable?
3. What economic interpretation might correspond to this parameter change?
```

```{solution-start} chow_cycles_ex4
:class: dropdown
```

```{code-cell} ipython3
λ_modified = λ.copy()
λ_modified[2] = 0.95
F_mod = spectral_density_chow(λ_modified, B, W, ω_grid)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
var_labels = ["consumption", "equipment + inventories", "construction", "long rate", "output"]
for i in range(5):
    f_orig = np.real(F_chow[:, i, i])
    f_mod = np.real(F_mod[:, i, i])
    f_orig_norm = f_orig / np.trapz(f_orig, freq)
    f_mod_norm = f_mod / np.trapz(f_mod, freq)
    axes[i].semilogy(freq, f_orig_norm, lw=1.5, label=r"original ($\lambda_3=0.48$)")
    axes[i].semilogy(freq, f_mod_norm, lw=1.5, ls="--", label=r"modified ($\lambda_3=0.95$)")
    paper_frequency_axis(axes[i])
    axes[i].set_ylabel(rf"normalized $f_{{{i+1}{i+1}}}(\omega)$")
    axes[i].text(0.03, 0.08, var_labels[i], transform=axes[i].transAxes)
    axes[i].legend(frameon=False, fontsize=8)
axes[5].axis('off')
plt.tight_layout()
plt.show()
```

Increasing $\lambda_3$ from 0.48 to 0.95 adds more persistence to the system.

The spectral densities show increased power at low frequencies.

Economically, this could correspond to stronger persistence in the propagation of shocks—perhaps due to slower adjustment speeds in investment or consumption behavior.

```{solution-end}
```
