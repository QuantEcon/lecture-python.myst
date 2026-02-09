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

(sargent_measurement_models)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Two Models of Measurements and the Investment Accelerator

```{contents} Contents
:depth: 2
```

## Overview

{cite:t}`Sargent1989` studies what happens to an econometrician's
inferences about economic dynamics when observed data are contaminated
by measurement error.

The setting is a {doc}`permanent income <perm_income>` economy in which the
investment accelerator, the mechanism studied in {doc}`samuelson` and
{doc}`chow_business_cycles`, drives business cycle fluctuations.

We specify a {doc}`linear state space model <linear_models>` for the
true economy and then consider two ways of extracting information from
noisy measurements:

- In Model 1, the data collecting agency simply reports
  raw (noisy) observations.
- In Model 2, the agency applies an optimal
  {doc}`Kalman filter <kalman>` to the noisy data and
  reports least-squares estimates of the true variables.

The two models produce different Wold representations and
forecast-error-variance decompositions, even though they describe
the same underlying economy.

In this lecture we follow {cite:t}`Sargent1989` and study how
alternative measurement schemes change empirical implications.

We start with imports and helper functions used throughout.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from IPython.display import Latex

np.set_printoptions(precision=3, suppress=True)

def df_to_latex_matrix(df, label=''):
    """Convert DataFrame to LaTeX matrix."""
    lines = [r'\begin{bmatrix}']

    for idx, row in df.iterrows():
        row_str = ' & '.join(
          [f'{v:.4f}' if isinstance(v, (int, float)) 
            else str(v) for v in row]) + r' \\'
        lines.append(row_str)

    lines.append(r'\end{bmatrix}')

    if label:
        return '$' + label + ' = ' + '\n'.join(lines) + '$'
    else:
        return '$' + '\n'.join(lines) + '$'

def df_to_latex_array(df):
    """Convert DataFrame to LaTeX array."""
    n_rows, n_cols = df.shape

    # Build column format (centered columns)
    col_format = 'c' * (n_cols + 1)  # +1 for index

    # Start array
    lines = [r'\begin{array}{' + col_format + '}']

    # Header row
    header = ' & '.join([''] + [str(c) for c in df.columns]) + r' \\'
    lines.append(header)
    lines.append(r'\hline')

    # Data rows
    for idx, row in df.iterrows():
        row_str = str(idx) + ' & ' + ' & '.join(
          [f'{v:.3f}' if isinstance(v, (int, float)) else str(v) 
          for v in row]) + r' \\'
        lines.append(row_str)

    lines.append(r'\end{array}')

    return '$' + '\n'.join(lines) + '$'
```

## Classical formulation

Before moving to state-space methods, {cite:t}`Sargent1989` formulates
both measurement models in classical Wold form.

This setup separates:

- The law of motion for true economic variables.
- The law of motion for measurement errors.
- The map from these two objects to observables used by an econometrician.

Let the true data be

```{math}
:label: classical_true_wold
Z_t = c_Z(L)\,\varepsilon_t^Z, \qquad
E\varepsilon_t^Z {\varepsilon_t^Z}' = I.
```

In Model 1 (raw reports), the agency observes and reports

```{math}
:label: classical_model1_meas
z_t = Z_t + v_t, \qquad
v_t = c_v(L)\,\varepsilon_t^v, \qquad
E(Z_t v_s') = 0\ \forall t,s.
```

Then measured data have Wold representation

```{math}
:label: classical_model1_wold
z_t = c_z(L)\,\varepsilon_t,
```

with spectral factorization

```{math}
:label: classical_model1_factor
c_z(s)c_z(s^{-1})' = c_Z(s)c_Z(s^{-1})' + c_v(s)c_v(s^{-1})'.
```

In Model 2 (filtered reports), the agency reports

```{math}
:label: classical_model2_report
\tilde z_t = E[Z_t \mid z_t, z_{t-1}, \ldots] = h(L) z_t,
```

where

```{math}
:label: classical_model2_filter
h(L)
= \Big[
    c_Z(L)c_Z(L^{-1})'
    \big(c_z(L^{-1})'\big)^{-1}
  \Big]_+ c_z(L)^{-1},
```

and $[\cdot]_+$ keeps only nonnegative powers of $L$.

Filtered reports satisfy

```{math}
:label: classical_model2_wold
\tilde z_t = c_{\tilde z}(L)\,a_t,
```

with

```{math}
:label: classical_model2_factor
c_{\tilde z}(s)c_{\tilde z}(s^{-1})'
= h(s)c_z(s)c_z(s^{-1})'h(s^{-1})'.
```

These two data-generation schemes imply different Gaussian likelihood
functions.

In the rest of the lecture, we switch to a recursive state-space
representation because it makes these objects easy to compute.

## The economic model

The true economy is a linear-quadratic version of a stochastic
optimal growth model (see also {doc}`perm_income`).

A social planner maximizes

```{math}
:label: planner_obj
E \sum_{t=0}^{\infty} \beta^t \left( u_0 + u_1 c_t - \frac{u_2}{2} c_t^2 \right)
```

subject to the technology

```{math}
:label: tech_constraint
c_t + k_{t+1} = f k_t + \theta_t, \qquad \beta f^2 > 1,
```

where $c_t$ is consumption, $k_t$ is the capital stock,
$f$ is the gross rate of return on capital,
and $\theta_t$ is an endowment or technology shock following

```{math}
:label: shock_process
a(L)\,\theta_t = \varepsilon_t,
```

with $a(L) = 1 - a_1 L - a_2 L^2 - \cdots - a_r L^r$ having all roots
outside the unit circle.

### Optimal decision rule

The solution can be represented by the optimal decision rule
for $c_t$:

```{math}
:label: opt_decision
c_t = \frac{-\alpha}{f-1}
      + \left(1 - \frac{1}{\beta f^2}\right)
        \frac{L - f^{-1} a(f^{-1})^{-1} a(L)}{L - f^{-1}}\,\theta_t
      + f k_t,
\qquad
k_{t+1} = f k_t + \theta_t - c_t,
```

where $\alpha = u_1[1-(\beta f)^{-1}]/u_2$.

Equations {eq}`shock_process` and {eq}`opt_decision` exhibit the
cross-equation restrictions characteristic of rational expectations
models.

### Net income and the accelerator

Define net output or national income as

```{math}
:label: net_income
y_{nt} = (f-1)k_t + \theta_t.
```

Note that {eq}`tech_constraint` and {eq}`net_income` imply
$(k_{t+1} - k_t) + c_t = y_{nt}$.

To obtain both a version of {cite:t}`Friedman1956`'s geometric
distributed lag consumption function and a distributed lag
accelerator, we impose two assumptions:

1. $a(L) = 1$, so that $\theta_t$ is white noise.
2. $\beta f = 1$, so the rate of return on capital equals the rate
   of time preference.

Assumption 1 is crucial for the strict form of the accelerator.

Relaxing it to allow serially correlated $\theta_t$ preserves an
accelerator in a broad sense but loses the sharp geometric-lag
form of {eq}`accelerator`.

Adding a second shock breaks the one-index structure entirely
and can generate nontrivial Granger causality even without
measurement error.

The accelerator projection is also not invariant under
interventions that alter predictable components of income.

Assumption 2 is less important, affecting only various constants.

Under both assumptions, {eq}`opt_decision` simplifies to

```{math}
:label: simple_crule
c_t = (1-f^{-1})\,\theta_t + (f-1)\,k_t.
```

When {eq}`simple_crule`, {eq}`net_income`, and
{eq}`tech_constraint` are combined, the optimal plan satisfies

```{math}
:label: friedman_consumption
c_t = \left(\frac{1-\beta}{1-\beta L}\right) y_{nt},
```

```{math}
:label: accelerator
k_{t+1} - k_t = f^{-1} \left(\frac{1-L}{1-\beta L}\right) y_{nt},
```

```{math}
:label: income_process
y_{nt} = \theta_t + (1-\beta)(\theta_{t-1} + \theta_{t-2} + \cdots).
```

Equation {eq}`friedman_consumption` is Friedman's consumption
model: consumption is a geometric distributed lag of income,
with the decay coefficient $\beta$ equal to the discount factor.

Equation {eq}`accelerator` is the distributed lag accelerator:
investment is a geometric distributed lag of the first difference
of income.

This is the same mechanism that {cite:t}`Chow1968` documented
empirically (see {doc}`chow_business_cycles`).

Equation {eq}`income_process` says that $y_{nt}$ is an IMA(1,1)
process with innovation $\theta_t$.

As {cite:t}`Muth1960` showed, such a process is optimally forecast
via a geometric distributed lag or "adaptive expectations" scheme.

### The accelerator puzzle

When all variables are measured accurately and are driven by
the single shock $\theta_t$, the spectral density of
$(c_t,\, k_{t+1}-k_t,\, y_{nt})$ has rank one at all frequencies.

Each variable is an invertible one-sided distributed lag of the
same white noise, so no variable Granger-causes any other.

Empirically, however, measures of output Granger-cause investment
but not vice versa.

{cite:t}`Sargent1989` shows that measurement error can resolve
this puzzle.

To illustrate, suppose first that output $y_{nt}$ is measured
perfectly while consumption and capital are each polluted by
serially correlated measurement errors $v_{ct}$ and $v_{kt}$
orthogonal to $\theta_t$.

Let $\bar c_t$ and $\bar k_{t+1} - \bar k_t$ denote the measured
series.  Then

```{math}
:label: meas_consumption
\bar c_t = \left(\frac{1-\beta}{1-\beta L}\right) y_{nt} + v_{ct},
```

```{math}
:label: meas_investment
\bar k_{t+1} - \bar k_t
  = \beta\left(\frac{1-L}{1-\beta L}\right) y_{nt}
  + (v_{k,t+1} - v_{kt}),
```

```{math}
:label: income_process_ma
y_{nt} = \theta_t + (1-\beta)(\theta_{t-1} + \theta_{t-2} + \cdots).
```

In this case income Granger-causes consumption and investment
but is not Granger-caused by them.

In the numerical example below, $y_{nt}$ is also measured
with error: the agency reports $\bar y_{nt} = y_{nt} + v_{yt}$,
where $v_{yt}$ follows an AR(1) process orthogonal to $\theta_t$.

When every series is corrupted by measurement error, every measured
variable Granger-causes every other.

The strength of Granger causality depends on the relative
signal-to-noise ratios.

In a one-common-index model like this one ($\theta_t$ is the
common index), the best-measured variable extends the most
Granger causality to the others.

This mechanism drives the numerical results below.

## State-space formulation

We now map the economic model and the measurement process into
a recursive state-space framework.

Set $f = 1.05$ and $\theta_t \sim \mathcal{N}(0, 1)$.

Define the state and observable vectors

```{math}
x_t = \begin{bmatrix} k_t \\ \theta_t \end{bmatrix},
\qquad
z_t = \begin{bmatrix} y_{nt} \\ c_t \\ \Delta k_t \end{bmatrix},
```

so that the true economy follows the state-space system

```{math}
:label: true_ss
\begin{aligned}
x_{t+1} &= A x_t + \varepsilon_t, \\
z_t &= C x_t.
\end{aligned}
```

where $\varepsilon_t = \begin{bmatrix} 0 \\ \theta_t \end{bmatrix}$ has
covariance $E \varepsilon_t \varepsilon_t^\top = Q$ and the matrices are

```{math}
A = \begin{bmatrix}
1 & f^{-1} \\
0 & 0
\end{bmatrix},
\qquad
C = \begin{bmatrix}
f-1 & 1 \\
f-1 & 1-f^{-1} \\
0   & f^{-1}
\end{bmatrix},
\qquad
Q = \begin{bmatrix}
0 & 0 \\
0 & 1
\end{bmatrix}.
```

$Q$ is singular because there is only one source of randomness
$\theta_t$; the capital stock $k_t$ evolves deterministically
given $\theta_t$.

### Measurement errors

The econometrician does not observe $z_t$ directly but instead
sees $\bar z_t = z_t + v_t$, where $v_t$ is a vector of measurement
errors.

Measurement errors follow an AR(1) process

```{math}
:label: meas_error_ar1
v_{t+1} = D v_t + \eta_t,
```

where $\eta_t$ is a vector white noise with
$E \eta_t \eta_t^\top = \Sigma_\eta$ and
$E \varepsilon_t v_s^\top = 0$ for all $t, s$.

The parameters are

```{math}
D = \operatorname{diag}(0.6, 0.7, 0.3),
\qquad
\sigma_\eta = (0.05, 0.035, 0.65),
```

so the unconditional covariance of $v_t$ is

```{math}
R = \operatorname{diag}\!\left(\frac{\sigma_{\eta,i}^2}{1 - \rho_i^2}\right).
```

The innovation variances are smallest for consumption
($\sigma_\eta = 0.035$), next for income ($\sigma_\eta = 0.05$),
and largest for investment ($\sigma_\eta = 0.65$).

As in {cite:t}`Sargent1989`, what matters for Granger-causality
asymmetries is the overall measurement quality in the full system:
output is relatively well measured while investment is relatively
poorly measured.

```{code-cell} ipython3
f = 1.05
β = 1 / f

A = np.array([
    [1.0, 1.0 / f],
    [0.0, 0.0]
])

C = np.array([
    [f - 1.0, 1.0],
    [f - 1.0, 1.0 - 1.0 / f],
    [0.0, 1.0 / f]
])

Q = np.array([
    [0.0, 0.0],
    [0.0, 1.0]
])

ρ = np.array([0.6, 0.7, 0.3])
D = np.diag(ρ)

# Innovation std. devs of η_t
σ_η = np.array([0.05, 0.035, 0.65])
Σ_η = np.diag(σ_η**2)

# Unconditional covariance of measurement errors v_t
R = np.diag((σ_η / np.sqrt(1.0 - ρ**2))**2)

print(f"f = {f},  β = 1/f = {β:.6f}")
print()
display(Latex(df_to_latex_matrix(pd.DataFrame(A), 'A')))
display(Latex(df_to_latex_matrix(pd.DataFrame(C), 'C')))
display(Latex(df_to_latex_matrix(pd.DataFrame(D), 'D')))
```

## Kalman filter

Both models require a steady-state {doc}`Kalman filter <kalman>`.

The function below iterates on the Riccati equation until convergence,
returning the Kalman gain $K$, the state covariance $S$, and the
innovation covariance $V$

```{code-cell} ipython3
def steady_state_kalman(A, C_obs, Q, R, W=None, tol=1e-13, max_iter=200_000):
    """
    Solve steady-state Kalman equations for
        x_{t+1} = A x_t + w_{t+1}
        y_t     = C_obs x_t + v_t
    with cov(w)=Q, cov(v)=R, cov(w,v)=W.
    """
    n = A.shape[0]
    m = C_obs.shape[0]
    if W is None:
        W = np.zeros((n, m))

    S = Q.copy()
    for _ in range(max_iter):
        V = C_obs @ S @ C_obs.T + R
        K = (A @ S @ C_obs.T + W) @ np.linalg.inv(V)
        S_new = Q + A @ S @ A.T - K @ V @ K.T

        if np.max(np.abs(S_new - S)) < tol:
            S = S_new
            break
        S = S_new

    V = C_obs @ S @ C_obs.T + R
    K = (A @ S @ C_obs.T + W) @ np.linalg.inv(V)
    return K, S, V
```

(true-impulse-responses)=
## True impulse responses

Before introducing measurement error, we compute the impulse response of
the true system to a unit shock $\theta_0 = 1$.

The response shows the investment accelerator clearly: the full impact on
net income $y_n$ occurs at lag 0, while consumption adjusts by only
$1 - f^{-1} \approx 0.048$ and investment absorbs the remainder.

From lag 1 onward the economy is in its new steady state

```{code-cell} ipython3
def table2_irf(A, C, n_lags=6):
    x = np.array([0.0, 1.0])  # k_0 = 0, theta_0 = 1
    rows = []
    for j in range(n_lags):
        y_n, c, d_k = C @ x
        rows.append([y_n, c, d_k])
        x = A @ x
    return pd.DataFrame(rows, columns=[r'y_n', r'c', r'\Delta k'],
                         index=pd.Index(range(n_lags), name='lag'))

table2 = table2_irf(A, C, n_lags=6)
display(Latex(df_to_latex_array(table2)))
```

## Model 1 (raw measurements)

Model 1 is a classical errors-in-variables model: the data collecting
agency simply reports the error-corrupted data $\bar z_t = z_t + v_t$
that it collects, making no attempt to adjust for measurement errors.

Because the measurement errors $v_t$ are serially correlated,
the standard Kalman filter with white-noise measurement error
cannot be applied directly to $\bar z_t = C x_t + v_t$.

An alternative is to augment the state vector with the
measurement-error AR components (see Appendix B of
{cite:t}`Sargent1989`).

Here we take the quasi-differencing route, which reduces the
system to one with serially uncorrelated observation noise.

Substituting $\bar z_t = C x_t + v_t$, $x_{t+1} = A x_t + \varepsilon_t$,
and $v_{t+1} = D v_t + \eta_t$ into $\bar z_{t+1} - D \bar z_t$ gives

```{math}
:label: model1_obs
\bar z_{t+1} - D \bar z_t = \bar C\, x_t + C \varepsilon_t + \eta_t,
```

where $\bar C = CA - DC$.

The composite observation noise in {eq}`model1_obs` is
$\bar\nu_t = C\varepsilon_t + \eta_t$, which is serially uncorrelated.

Its covariance, and the cross-covariance between the state noise
$\varepsilon_t$ and $\bar\nu_t$, are

```{math}
:label: model1_covs
R_1 = C Q C^\top + R, \qquad W_1 = Q C^\top.
```

The system $\{x_{t+1} = A x_t + \varepsilon_t,\;
\bar z_{t+1} - D\bar z_t = \bar C x_t + \bar\nu_t\}$
with $\text{cov}(\varepsilon_t)=Q$, $\text{cov}(\bar\nu_t)=R_1$, and
$\text{cov}(\varepsilon_t, \bar\nu_t)=W_1$ now has serially uncorrelated
errors, so the standard {doc}`Kalman filter <kalman>` applies.

The steady-state Kalman filter yields the **innovations representation**

```{math}
:label: model1_innov
\begin{aligned}
\hat x_{t+1} &= A \hat x_t + K_1 u_t, \\
\bar z_{t+1} - D\bar z_t &= \bar C \hat x_t + u_t.
\end{aligned}
```

where $u_t = (\bar z_{t+1} - D\bar z_t) -
E[\bar z_{t+1} - D\bar z_t \mid \bar z_t, \bar z_{t-1}, \ldots]$
is the innovation process, $K_1$ is the Kalman gain, and
$V_1 = \bar C S_1 \bar C^\top + R_1$ is the innovation covariance matrix
(with $S_1 = E[(x_t - \hat x_t)(x_t - \hat x_t)^\top]$ the steady-state
state estimation error covariance).

To compute the innovations $\{u_t\}$ recursively from the data
$\{\bar z_t\}$, it is useful to represent {eq}`model1_innov` as

```{math}
:label: model1_recursion
\begin{aligned}
\hat x_{t+1} &= (A - K_1 \bar C)\,\hat x_t + K_1 \bar z_t, \\
u_t &= -\bar C\,\hat x_t + \bar z_t.
\end{aligned}
```

where $\bar z_t := \bar z_{t+1} - D\bar z_t$ is the quasi-differenced
observation.

Given an initial $\hat x_0$, equation {eq}`model1_recursion` generates
the innovation sequence, from which the Gaussian log-likelihood
of a sample $\{\bar z_t,\, t=0,\ldots,T\}$ is

```{math}
:label: model1_loglik
\mathcal{L}^* = -T\ln 2\pi - \tfrac{1}{2}T\ln|V_1|
  - \tfrac{1}{2}\sum_{t=0}^{T-1} u_t' V_1^{-1} u_t.
```

```{code-cell} ipython3
C_bar = C @ A - D @ C
R1 = C @ Q @ C.T + R
W1 = Q @ C.T

K1, S1, V1 = steady_state_kalman(A, C_bar, Q, R1, W1)
```

### Wold representation for measured data

With the innovations representation {eq}`model1_innov` in hand, we can
derive a Wold moving-average representation for the measured data
$\bar z_t$.

From {eq}`model1_innov` and the quasi-differencing definition, the
measured data satisfy

```{math}
:label: model1_wold
\bar z_{t+1} = (I - DL)^{-1}\bigl[\bar C(I - AL)^{-1}K_1 L + I\bigr] u_t,
```

where $L$ is the lag operator.

To compute the Wold coefficients numerically, define the augmented state

```{math}
r_t = \begin{bmatrix} \hat x_{t-1} \\ \bar z_{t-1} \end{bmatrix},
```

with dynamics

```{math}
r_{t+1} = F_1 r_t + G_1 u_t,
\qquad
\bar z_t = H_1 r_t + u_t,
```

where

```{math}
F_1 =
\begin{bmatrix}
A & 0 \\
\bar C & D
\end{bmatrix},
\quad
G_1 =
\begin{bmatrix}
K_1 \\
I
\end{bmatrix},
\quad
H_1 = [\bar C \;\; D].
```

The Wold coefficients are then $\psi_0 = I$ and
$\psi_j = H_1 F_1^{j-1} G_1$ for $j \geq 1$.

```{code-cell} ipython3
F1 = np.block([
    [A, np.zeros((2, 3))],
    [C_bar, D]
])
G1 = np.vstack([K1, np.eye(3)])
H1 = np.hstack([C_bar, D])


def measured_wold_coeffs(F, G, H, n_terms=25):
    psi = [np.eye(3)]
    Fpow = np.eye(F.shape[0])
    for _ in range(1, n_terms):
        psi.append(H @ Fpow @ G)
        Fpow = Fpow @ F
    return psi


def fev_contributions(psi, V, n_horizons=20):
    """
    Returns contrib[var, shock, h-1] = contribution at horizon h.
    """
    P = linalg.cholesky(V, lower=True)
    out = np.zeros((3, 3, n_horizons))
    for h in range(1, n_horizons + 1):
        acc = np.zeros((3, 3))
        for j in range(h):
            T = psi[j] @ P
            acc += T**2
        out[:, :, h - 1] = acc
    return out


psi1 = measured_wold_coeffs(F1, G1, H1, n_terms=40)
resp1 = np.array([psi1[j] @ linalg.cholesky(V1, lower=True) for j in range(14)])
decomp1 = fev_contributions(psi1, V1, n_horizons=20)
```

### Forecast-error-variance decomposition

To measure the relative importance of each innovation, we decompose
the $j$-step-ahead forecast-error variance of each measured variable.

Write $\bar z_{t+j} - E_t \bar z_{t+j} = \sum_{i=0}^{j-1} \psi_i u_{t+j-i}$.

Let $P$ be the lower-triangular Cholesky factor of $V_1$ so that the
orthogonalized innovations are $e_t = P^{-1} u_t$.

Then the contribution of orthogonalized innovation $k$ to the
$j$-step-ahead variance of variable $m$ is
$\sum_{i=0}^{j-1} (\psi_i P)_{mk}^2$.

The table below shows the cumulative contribution of each orthogonalized
innovation to the forecast-error variance of $y_n$, $c$, and $\Delta k$
at horizons 1 through 20.

Each panel fixes one orthogonalized innovation and reports its
cumulative contribution to each variable's forecast-error variance.

Rows are forecast horizons and columns are forecasted variables.

```{code-cell} ipython3
horizons = np.arange(1, 21)
labels = [r'y_n', r'c', r'\Delta k']

def fev_table(decomp, shock_idx, horizons):
    return pd.DataFrame(
        np.round(decomp[:, shock_idx, :].T, 4),
        columns=labels,
        index=pd.Index(horizons, name='j')
    )
```

```{code-cell} ipython3
shock_titles = [r'\text{A. Innovation in } y_n',
                r'\text{B. Innovation in } c',
                r'\text{C. Innovation in } \Delta k']

parts = []
for i, title in enumerate(shock_titles):
    arr = df_to_latex_array(fev_table(decomp1, i, horizons)).strip('$')
    parts.append(r'\begin{array}{c} ' + title + r' \\ ' + arr + r' \end{array}')

display(Latex('$' + r' \quad '.join(parts) + '$'))
```

The income innovation accounts for substantial proportions of
forecast-error variance in all three variables, while the consumption and
investment innovations contribute mainly to their own variances.

This is a **Granger causality** pattern: income appears to
Granger-cause consumption and investment, but not vice versa.

This matches the paper's message that, in a one-common-index model,
the relatively best measured series has the strongest predictive content.

The covariance matrix of the innovations is not diagonal, but the
eigenvalues are well separated


```{code-cell} ipython3
print('Covariance matrix of innovations:')
df_v1 = pd.DataFrame(np.round(V1, 4), index=labels, columns=labels)
display(Latex(df_to_latex_matrix(df_v1)))
```

The first eigenvalue is much larger than the others, consistent with
the presence of a dominant common shock $\theta_t$

```{code-cell} ipython3
print('Eigenvalues of covariance matrix:')
print(np.sort(np.linalg.eigvalsh(V1))[::-1].round(4))
```

### Wold impulse responses

The Wold impulse responses are reported using orthogonalized
innovations (Cholesky factorization of $V_1$ with ordering
$y_n$, $c$, $\Delta k$).

Under this identification, lag-0 responses reflect both
contemporaneous covariance and the Cholesky ordering.

```{code-cell} ipython3
lags = np.arange(14)

def wold_response_table(resp, shock_idx, lags):
    return pd.DataFrame(
        np.round(resp[:, :, shock_idx], 4),
        columns=labels,
        index=pd.Index(lags, name='j')
    )
```

```{code-cell} ipython3
wold_titles = [r'\text{A. Response to } y_n \text{ innovation}',
               r'\text{B. Response to } c \text{ innovation}',
               r'\text{C. Response to } \Delta k \text{ innovation}']

parts = []
for i, title in enumerate(wold_titles):
    arr = df_to_latex_array(wold_response_table(resp1, i, lags)).strip('$')
    parts.append(r'\begin{array}{c} ' + title + r' \\ ' + arr + r' \end{array}')

display(Latex('$' + r' \quad '.join(parts) + '$'))
```

At impact, the first orthogonalized innovation
loads on all three measured variables.

At subsequent lags the income innovation generates persistent
responses in all three variables because, being the best-measured
series, its innovation is dominated by the true permanent shock
$\theta_t$.

The consumption and investment innovations produce responses that
decay according to the AR(1) structure of their respective
measurement errors ($\rho_c = 0.7$, $\rho_{\Delta k} = 0.3$),
with little spillover to other variables.

## Model 2 (filtered measurements)

Model 2 corresponds to a data collecting agency that, instead of
reporting raw error-corrupted data, applies an optimal filter
to construct least-squares estimates of the true variables.

This is a natural model for agencies that seasonally adjust
data (one-sided filtering of current and past observations) or
publish preliminary, revised, and final estimates of the same
variable (successive conditional expectations as more data
accumulate).

Specifically, the agency uses the Kalman filter from Model 1 to form
$\hat x_t = E[x_t \mid \bar z_t, \bar z_{t-1}, \ldots]$ and reports
filtered estimates

```{math}
\tilde z_t = G \hat x_t,
```

where $G = C$ is a selection matrix.

### State-space for filtered data

From the innovations representation {eq}`model1_innov`, the state
$\hat x_t$ evolves as

```{math}
:label: model2_state
\hat x_{t+1} = A \hat x_t + K_1 u_t.
```

The reported filtered data are then

```{math}
:label: model2_obs
\tilde z_t = C \hat x_t + \eta_t,
```

where $\eta_t$ is a type 2 white-noise measurement error process
("typos") with presumably very small covariance matrix $R_2$.

The state noise in {eq}`model2_state` is $K_1 u_t$, which has covariance

```{math}
:label: model2_Q
Q_2 = K_1 V_1 K_1^\top.
```

The covariance matrix of the joint noise is

```{math}
E \begin{bmatrix} K_1 u_t \\ \eta_t \end{bmatrix}
  \begin{bmatrix} K_1 u_t \\ \eta_t \end{bmatrix}^\top
= \begin{bmatrix} Q_2 & 0 \\ 0 & R_2 \end{bmatrix}.
```

Since $R_2$ is close to or equal to zero (the filtered data have
negligible additional noise), we approximate it with a small
regularization term $R_2 = \epsilon I$ to keep the Kalman filter
numerically well-conditioned.

A second Kalman filter applied to {eq}`model2_state`--{eq}`model2_obs`
yields a second innovations representation

```{math}
:label: model2_innov
\begin{aligned}
\check{x}_{t+1} &= A \check{x}_t + K_2 a_t, \\
\tilde z_t &= C \check{x}_t + a_t.
\end{aligned}
```

where $a_t$ is the innovation process for the filtered data with
covariance $V_2 = C S_2 C^\top + R_2$.

To compute the innovations $\{a_t\}$ from observations on
$\tilde z_t$, use

```{math}
:label: model2_recursion
\begin{aligned}
\check{x}_{t+1} &= (A - K_2 C)\,\check{x}_t + K_2 \tilde z_t, \\
a_t &= -C\,\check{x}_t + \tilde z_t.
\end{aligned}
```

The Gaussian log-likelihood for a sample of $T$ observations
$\{\tilde z_t\}$ is then

```{math}
:label: model2_loglik
\mathcal{L}^{**} = -T\ln 2\pi - \tfrac{1}{2}T\ln|V_2|
  - \tfrac{1}{2}\sum_{t=0}^{T-1} a_t' V_2^{-1} a_t.
```

Computing {eq}`model2_loglik` requires both the first Kalman filter
(to form $\hat x_t$ and $u_t$) and the second Kalman filter
(to form $\check{x}_t$ and $a_t$).

In effect, the econometrician must retrace the steps that the agency
used to synthesize the filtered data.

### Wold representation for filtered data

The Wold moving-average representation for $\tilde z_t$ is

```{math}
:label: model2_wold
\tilde z_t = \bigl[C(I - AL)^{-1} K_2 L + I\bigr] a_t,
```

with coefficients $\psi_0 = I$ and $\psi_j = C A^{j-1} K_2$ for
$j \geq 1$.

Note that this is simpler than the Model 1 Wold
representation {eq}`model1_wold` because there is no quasi-differencing
to undo

```{code-cell} ipython3
Q2 = K1 @ V1 @ K1.T
ε = 1e-6

K2, S2, V2 = steady_state_kalman(A, C, Q2, ε * np.eye(3))


def filtered_wold_coeffs(A, C, K, n_terms=25):
    psi = [np.eye(3)]
    Apow = np.eye(2)
    for _ in range(1, n_terms):
        psi.append(C @ Apow @ K)
        Apow = Apow @ A
    return psi


psi2 = filtered_wold_coeffs(A, C, K2, n_terms=40)
resp2 = np.array([psi2[j] @ linalg.cholesky(V2, lower=True) for j in range(14)])
decomp2 = fev_contributions(psi2, V2, n_horizons=20)
```

### Forecast-error-variance decomposition

Because the filtered data are nearly noiseless, the innovation
covariance $V_2$ is close to singular with one dominant eigenvalue.

This means the filtered economy is driven by essentially one shock,
just like the true economy

```{code-cell} ipython3
parts = []
for i, title in enumerate(shock_titles):
    arr = df_to_latex_array(fev_table(decomp2, i, horizons)).strip('$')
    parts.append(r'\begin{array}{c} ' + title + r' \\ ' + arr + r' \end{array}')

display(Latex('$' + r' \quad '.join(parts) + '$'))
```

In Model 2, the first innovation accounts for virtually all forecast-error
variance, just as in the true economy where the single structural shock
$\theta_t$ drives everything.

The second and third innovations contribute negligibly.

This confirms that filtering strips away the measurement noise that created
the appearance of multiple independent sources of variation in Model 1.

We invite readers to compare this table to the one for the true impulse responses in the {ref}`true-impulse-responses` section above.

The numbers are essentially the same.

The covariance matrix and eigenvalues of the Model 2 innovations are

```{code-cell} ipython3
print('Covariance matrix of innovations:')
df_v2 = pd.DataFrame(np.round(V2, 4), index=labels, columns=labels)
display(Latex(df_to_latex_matrix(df_v2)))
```

```{code-cell} ipython3
print('Eigenvalues of covariance matrix:')
print(np.sort(np.linalg.eigvalsh(V2))[::-1].round(4))
```

As {cite:t}`Sargent1989` emphasizes, the two models of measurement
produce quite different inferences about the economy's dynamics despite
sharing identical underlying parameters.



### Wold impulse responses

We again use orthogonalized Wold responses with a Cholesky
decomposition of $V_2$ ordered as $y_n$, $c$, $\Delta k$.

```{code-cell} ipython3
parts = []
for i, title in enumerate(wold_titles):
    arr = df_to_latex_array(
      wold_response_table(resp2, i, lags)).strip('$')
    parts.append(
      r'\begin{array}{c} ' + title + r' \\ ' + arr + r' \end{array}')

display(Latex('$' + r' \quad '.join(parts) + '$'))
```

The income innovation in Model 2 produces responses that closely
approximate the true impulse response function from the structural
shock $\theta_t$ (compare with the table in the
{ref}`true-impulse-responses` section above).

The consumption and investment innovations produce responses
that are orders of magnitude smaller, confirming that the filtered
data are driven by essentially one shock.

Unlike Model 1, the filtered data from Model 2
*cannot* reproduce the apparent Granger causality pattern that the
accelerator literature has documented empirically.


## Simulation

The tables above characterize population moments of the two models.

We now simulate 80 periods of true, measured, and filtered data
to compare population implications with finite-sample behavior.

```{code-cell} ipython3
def simulate_series(seed=7909, T=80, k0=10.0):
    """
    Simulate true, measured, and filtered series.
    """
    rng = np.random.default_rng(seed)

    # True state/observables
    θ = rng.normal(0.0, 1.0, size=T)
    k = np.empty(T + 1)
    k[0] = k0

    y = np.empty(T)
    c = np.empty(T)
    dk = np.empty(T)

    for t in range(T):
        x_t = np.array([k[t], θ[t]])
        y[t], c[t], dk[t] = C @ x_t
        k[t + 1] = k[t] + (1.0 / f) * θ[t]

    # Measured data with AR(1) errors
    v_prev = np.zeros(3)
    v = np.empty((T, 3))
    for t in range(T):
        η_t = rng.multivariate_normal(np.zeros(3), Σ_η)
        v_prev = D @ v_prev + η_t
        v[t] = v_prev

    z_meas = np.column_stack([y, c, dk]) + v

    # Filtered data via Model 1 transformed filter
    xhat_prev = np.array([k0, 0.0])
    z_prev = np.zeros(3)
    z_filt = np.empty((T, 3))
    k_filt = np.empty(T)

    for t in range(T):
        z_bar_t = z_meas[t] - D @ z_prev
        u_t = z_bar_t - C_bar @ xhat_prev
        xhat_t = A @ xhat_prev + K1 @ u_t

        z_filt[t] = C @ xhat_t
        k_filt[t] = xhat_t[0]

        xhat_prev = xhat_t
        z_prev = z_meas[t]

    out = {
        "y_true": y, "c_true": c, "dk_true": dk, "k_true": k[:-1],
        "y_meas": z_meas[:, 0], "c_meas": z_meas[:, 1], 
        "dk_meas": z_meas[:, 2],
        "y_filt": z_filt[:, 0], "c_filt": z_filt[:, 1], 
        "dk_filt": z_filt[:, 2], "k_filt": k_filt
    }
    return out


sim = simulate_series(seed=7909, T=80, k0=10.0)
```

```{code-cell} ipython3
def plot_true_vs_other(t, true_series, other_series, 
                                  other_label, ylabel=""):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, true_series, lw=2, color="black", label="true")
    ax.plot(t, other_series, lw=2, ls="--", 
                          color="#1f77b4", label=other_label)
    ax.set_xlabel("time", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc="best", fontsize=11, frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()


t = np.arange(1, 81)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: True and measured consumption
    name: fig-true-measured-consumption
  image:
    alt: True and measured consumption plotted over 80 time periods
---
plot_true_vs_other(t, sim["c_true"], sim["c_meas"], 
                                    "measured", ylabel="consumption")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: True and measured investment
    name: fig-true-measured-investment
  image:
    alt: True and measured investment plotted over 80 time periods
---
plot_true_vs_other(t, sim["dk_true"], sim["dk_meas"], 
                                    "measured", ylabel="investment")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: True and measured income
    name: fig-true-measured-income
  image:
    alt: True and measured income plotted over 80 time periods
---
plot_true_vs_other(t, sim["y_true"], sim["y_meas"], 
                                    "measured", ylabel="income")
```

Investment is distorted the most because its measurement error
has the largest innovation variance ($\sigma_\eta = 0.65$),
while income is distorted the least ($\sigma_\eta = 0.05$).

The Kalman-filtered estimates from Model 1 remove much of the
measurement noise and track the truth closely.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: True and filtered consumption
    name: fig-true-filtered-consumption
  image:
    alt: True and filtered consumption plotted over 80 time periods
---
plot_true_vs_other(t, sim["c_true"], sim["c_filt"], 
                                    "filtered", ylabel="consumption")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: True and filtered investment
    name: fig-true-filtered-investment
  image:
    alt: True and filtered investment plotted over 80 time periods
---
plot_true_vs_other(t, sim["dk_true"], sim["dk_filt"], 
                                    "filtered", ylabel="investment")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: True and filtered income
    name: fig-true-filtered-income
  image:
    alt: True and filtered income plotted over 80 time periods
---
plot_true_vs_other(t, sim["y_true"], sim["y_filt"], 
                                    "filtered", ylabel="income")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: True and filtered capital stock
    name: fig-true-filtered-capital
  image:
    alt: True and filtered capital stock plotted over 80 time periods
---
plot_true_vs_other(t, sim["k_true"], sim["k_filt"], 
                                    "filtered", ylabel="capital stock")
```

In the true model the national income identity
$c_t + \Delta k_t = y_{n,t}$ holds exactly.

Independent measurement errors break this accounting identity
in the measured data.

The Kalman filter approximately restores it.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "National income identity residual: measured (left) vs. filtered (right)"
    name: fig-identity-residual
  image:
    alt: National income identity residual for measured and filtered data side by side
---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(t, sim["c_meas"] + sim["dk_meas"] - sim["y_meas"], lw=2)
ax1.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax1.set_xlabel("time", fontsize=12)
ax1.set_ylabel("residual", fontsize=12)
ax1.set_title(r'Measured: $c_t + \Delta k_t - y_{n,t}$', fontsize=13)

ax2.plot(t, sim["c_filt"] + sim["dk_filt"] - sim["y_filt"], lw=2)
ax2.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax2.set_xlabel("time", fontsize=12)
ax2.set_ylabel("residual", fontsize=12)
ax2.set_title(r'Filtered: $c_t + \Delta k_t - y_{n,t}$', fontsize=13)

plt.tight_layout()
plt.show()
```

For each variable $w \in \{c, \Delta k, y_n\}$ we compute the
covariance and correlation matrices among its true, measured, and
filtered versions.

Each matrix has the structure

```{math}
\begin{bmatrix}
\text{var}(w^{\text{true}}) & \text{cov}(w^{\text{true}}, w^{\text{meas}}) & \text{cov}(w^{\text{true}}, w^{\text{filt}}) \\
\cdot & \text{var}(w^{\text{meas}}) & \text{cov}(w^{\text{meas}}, w^{\text{filt}}) \\
\cdot & \cdot & \text{var}(w^{\text{filt}})
\end{bmatrix}.
```

The key entries are the off-diagonal terms linking true to measured
(distortion from noise) and true to filtered (recovery by the Kalman
filter).

```{code-cell} ipython3
def cov_corr_three(a, b, c):
    X = np.vstack([a, b, c])
    return np.cov(X), np.corrcoef(X)

def matrix_df(mat, labels):
    return pd.DataFrame(np.round(mat, 4), index=labels, columns=labels)

cov_c, corr_c = cov_corr_three(
                sim["c_true"], sim["c_meas"], sim["c_filt"])
cov_i, corr_i = cov_corr_three(
                sim["dk_true"], sim["dk_meas"], sim["dk_filt"])
cov_y, corr_y = cov_corr_three(
                sim["y_true"], sim["y_meas"], sim["y_filt"])
cov_k = np.cov(np.vstack([sim["k_true"], sim["k_filt"]]))
corr_k = np.corrcoef(np.vstack([sim["k_true"], sim["k_filt"]]))

tmf_labels = ['true', 'measured', 'filtered']
tf_labels = ['true', 'filtered']
```

For consumption, measurement error inflates the variance of measured
consumption relative to the truth, as the diagonal of the covariance
matrix shows

```{code-cell} ipython3
display(Latex(df_to_latex_matrix(matrix_df(cov_c, tmf_labels))))
```

The correlation matrix confirms that the filtered series recovers the
true series almost perfectly 

```{code-cell} ipython3
display(Latex(df_to_latex_matrix(matrix_df(corr_c, tmf_labels))))
```

For investment, measurement error creates the most variance inflation here

```{code-cell} ipython3
display(Latex(df_to_latex_matrix(matrix_df(cov_i, tmf_labels))))
```

Despite this, the true-filtered correlation remains high,
demonstrating the filter's effectiveness even with severe noise

```{code-cell} ipython3
display(Latex(df_to_latex_matrix(matrix_df(corr_i, tmf_labels))))
```

Income has the smallest measurement error ($\sigma_\eta = 0.05$),
so measured and true covariances are nearly identical

```{code-cell} ipython3
display(Latex(df_to_latex_matrix(matrix_df(cov_y, tmf_labels))))
```

The correlation matrix shows that both measured and filtered series
track the truth very closely

```{code-cell} ipython3
display(Latex(df_to_latex_matrix(matrix_df(corr_y, tmf_labels))))
```

The capital stock is never directly observed, yet
the covariance matrix shows that the filter recovers it with very
high accuracy

```{code-cell} ipython3
display(Latex(df_to_latex_matrix(matrix_df(cov_k, tf_labels))))
```

The near-unity correlation confirms this

```{code-cell} ipython3
display(Latex(df_to_latex_matrix(matrix_df(corr_k, tf_labels))))
```

## Summary

{cite}`Sargent1989` shows how measurement error alters an
econometrician's view of a permanent income economy driven by
the investment accelerator.

The Wold representations and variance decompositions of Model 1
(raw measurements) and Model 2 (filtered measurements) differ
substantially, even though the underlying economy is the same.

Measurement error can reshape inferences about which shocks
drive which variables.

Model 1 reproduces the **Granger causality** pattern documented in
the empirical accelerator literature: income appears to Granger-cause
consumption and investment, a result {cite:t}`Sargent1989` attributes
to measurement error and signal extraction in raw reported data.

Model 2, working with filtered data, attributes nearly all variance
to the single structural shock $\theta_t$ and *cannot* reproduce
the Granger causality pattern.

The {doc}`Kalman filter <kalman>` effectively strips measurement
noise from the data, so the filtered series track the truth closely.

Raw measurement error breaks the national income accounting identity,
but the near-zero residual shows that the filter approximately
restores it.
