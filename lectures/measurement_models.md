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

"Rational expectations econometrics" aims to interpret economic time
series in terms of objects that are meaningful to economists, namely,
parameters describing preferences, technologies, information sets,
endowments, and equilibrium concepts.

When fully worked out, rational expectations models typically deliver
a well-defined mapping from these economically interpretable parameters
to the moments of the time series determined by the model.

If accurate observations on these time series are available, one can
use that mapping to implement parameter estimation methods based
either on the likelihood function or on the method of moments.

However, if only error-ridden data exist for the variables of interest,
then more steps are needed to extract parameter estimates.

In effect, we require a model of the data reporting agency, one that
is workable enough that we can determine the mapping induced jointly
by the dynamic economic model and the measurement process to the
probability law for the measured data.

The model chosen for the data collection agency is an aspect of an
econometric specification that can make big differences in inferences
about the economic structure.

{cite:t}`Sargent1989` describes two alternative models of data generation
in a {doc}`permanent income <perm_income>` economy in which the
investment accelerator, the mechanism studied in {doc}`samuelson` and
{doc}`chow_business_cycles`, drives business cycle fluctuations.

- In Model 1, the data collecting agency simply reports the
  error-ridden data that it collects.
- In Model 2, although it collects error-ridden data that satisfy
  a classical errors-in-variables model, the data collecting agency
  filters the data and reports the best estimates that it possibly can.

Although the two models have the same "deep parameters," they produce
quite different sets of restrictions on the data.

In this lecture we follow {cite:t}`Sargent1989` and study how
alternative measurement schemes change empirical implications.

We start with imports and helper functions used throughout

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

Equation {eq}`income_process` states that the first difference of disposable income is a
first-order moving average process with innovation equal to the innovation of the endowment shock $\theta_t$.

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

When each measured series is corrupted by measurement error, every
measured variable will in general Granger-cause every other.

The strength of this Granger causality, as measured by decompositions
of $j$-step-ahead prediction error variances, depends on the relative
variances of the measurement errors.

In this case, each observed series mixes the common signal $\theta_t$
with idiosyncratic measurement noise. 

A series with lower measurement
error variance tracks $\theta_t$ more closely, so its innovations
contain more information about future values of the other series.

Accordingly, in a forecast-error-variance decomposition, shocks to
better-measured series account for a larger share of other variables'
$j$-step-ahead prediction errors.

In a one-common-index model like this one ($\theta_t$ is the common
index), better-measured variables extend more Granger causality to
less well measured series than vice versa.

This asymmetry drives the numerical results we observe soon.

### State-space formulation

Let's map the economic model and the measurement process into
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

```{code-cell} ipython3
# Baseline structural matrices for the true economy
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
```

(true-impulse-responses)=
### True impulse responses

Before introducing measurement error, we compute the impulse response of
the true system to a unit shock $\theta_0 = 1$.

This benchmark clarifies what changes when we later switch from
true variables to reported variables.

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

## Measurement errors

Let's add the measurement layer that generates reported data.

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

As in {cite:t}`Sargent1989` and our discussion above, what matters for Granger-causality
asymmetries is the overall measurement quality in the full system:
output is relatively well measured while investment is relatively
poorly measured.

```{code-cell} ipython3
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

We will analyze the two reporting schemes separately, but first we need a solver for the steady-state Kalman gain and error covariances.

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

With structural matrices and tools we need in place, we now follow
{cite:t}`Sargent1989`'s two reporting schemes in sequence.

## A Classical Model of Measurements Initially Collected by an Agency

A data collecting agency observes a noise-corrupted version of $z_t$, namely

```{math}
:label: model1_obs
\bar z_t = C x_t + v_t.
```

We refer to this as *Model 1*: the agency collects noisy
data and reports them without filtering.

To represent the second moments of the $\bar z_t$ process, it is
convenient to obtain its population vector autoregression.

The error vector in the vector autoregression is the
innovation to $\bar z_t$ and can be taken to be the white noise in a Wold
moving average representation, which can be obtained by "inverting"
the autoregressive representation.

The population vector autoregression, and how it depends on the
parameters of the state-space system and the measurement error process,
carries insights about how to interpret estimated vector
autoregressions for $\bar z_t$.

Constructing the vector autoregression is also useful as an
intermediate step in computing the likelihood of a sample of
$\bar z_t$'s as a function of the free parameters
$\{A, C, D, Q, R\}$.

The particular method that will be used to construct the vector
autoregressive representation also proves useful as an intermediate
step in constructing a model of an optimal reporting agency.

We use recursive (Kalman filtering) methods to obtain the
vector autoregression for $\bar z_t$.

### Quasi-differencing

Because the measurement errors $v_t$ are serially correlated,
the standard Kalman filter with white-noise measurement error
cannot be applied directly to $\bar z_t = C x_t + v_t$.

An alternative is to augment the state vector with the
measurement-error AR components (see Appendix B of
{cite:t}`Sargent1989`).

Here we take the quasi-differencing route described in
{cite:t}`Sargent1989`, which reduces the
system to one with serially uncorrelated observation noise.

Define

```{math}
:label: model1_qd
\tilde z_t = \bar z_{t+1} - D \bar z_t, \qquad
\bar\nu_t = C \varepsilon_t + \eta_t, \qquad
\bar C = CA - DC.
```

Then the state-space system {eq}`true_ss`, the measurement error
process {eq}`meas_error_ar1`, and the observation equation {eq}`model1_obs`
imply the state-space system

```{math}
:label: model1_transformed
\begin{aligned}
x_{t+1} &= A x_t + \varepsilon_t, \\
\tilde z_t &= \bar C\, x_t + \bar\nu_t,
\end{aligned}
```

where $(\varepsilon_t, \bar\nu_t)$ is a white noise process with

```{math}
:label: model1_covs
E \begin{bmatrix} \varepsilon_t \end{bmatrix}
\begin{bmatrix} \varepsilon_t' & \bar\nu_t' \end{bmatrix}
= \begin{bmatrix} Q & W_1 \\ W_1' & R_1 \end{bmatrix},
\qquad
R_1 = C Q C^\top + R, \quad W_1 = Q C^\top.
```

System {eq}`model1_transformed` with covariances {eq}`model1_covs` is
characterized by the five matrices
$[A, \bar C, Q, R_1, W_1]$.

### Innovations representation

Associated with {eq}`model1_transformed` and {eq}`model1_covs` is the
**innovations representation** for $\tilde z_t$,

```{math}
:label: model1_innov
\begin{aligned}
\hat x_{t+1} &= A \hat x_t + K_1 u_t, \\
\tilde z_t &= \bar C \hat x_t + u_t,
\end{aligned}
```

where

```{math}
:label: model1_innov_defs
\begin{aligned}
\hat x_t &= E[x_t \mid \tilde z_{t-1}, \tilde z_{t-2}, \ldots, \hat x_0]
         = E[x_t \mid \bar z_t, \bar z_{t-1}, \ldots], \\
u_t &= \tilde z_t - E[\tilde z_t \mid \tilde z_{t-1}, \tilde z_{t-2}, \ldots]
     = \bar z_{t+1} - E[\bar z_{t+1} \mid \bar z_t, \bar z_{t-1}, \ldots],
\end{aligned}
```

$[K_1, S_1]$ are computed from the steady-state Kalman filter applied to
$[A, \bar C, Q, R_1, W_1]$, and

```{math}
:label: model1_S1
S_1 = E[(x_t - \hat x_t)(x_t - \hat x_t)^\top].
```

From {eq}`model1_innov_defs`, $u_t$ is the innovation process for the
$\bar z_t$ process.

### Wold representation

System {eq}`model1_innov` and definition {eq}`model1_qd` can be used to
obtain a Wold vector moving average representation for the $\bar z_t$ process:

```{math}
:label: model1_wold
\bar z_{t+1} = (I - DL)^{-1}\bigl[\bar C(I - AL)^{-1}K_1 L + I\bigr] u_t,
```

where $L$ is the lag operator.

From {eq}`model1_transformed` and {eq}`model1_innov` the innovation
covariance is

```{math}
:label: model1_V1
V_1 = E\, u_t u_t^\top = \bar C\, S_1\, \bar C^\top + R_1.
```

Below we compute $K_1$, $S_1$, and $V_1$ numerically 

```{code-cell} ipython3
C_bar = C @ A - D @ C
R1 = C @ Q @ C.T + R
W1 = Q @ C.T

K1, S1, V1 = steady_state_kalman(A, C_bar, Q, R1, W1)
```


### Computing the Wold coefficients

To compute the Wold coefficients in {eq}`model1_wold` numerically,
define the augmented state

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

### Gaussian likelihood

The Gaussian log-likelihood function for a sample
$\{\bar z_t,\, t=0,\ldots,T\}$, conditioned on an initial state estimate
$\hat x_0$, can be represented as

```{math}
:label: model1_loglik
\mathcal{L}^* = -T\ln 2\pi - \tfrac{1}{2}T\ln|V_1|
  - \tfrac{1}{2}\sum_{t=0}^{T-1} u_t' V_1^{-1} u_t,
```

where $u_t$ is a function of $\{\bar z_t\}$ defined by
{eq}`model1_recursion` below.

To use {eq}`model1_innov` to compute $\{u_t\}$, it is useful to
represent it as

```{math}
:label: model1_recursion
\begin{aligned}
\hat x_{t+1} &= (A - K_1 \bar C)\,\hat x_t + K_1 \tilde z_t, \\
u_t &= -\bar C\,\hat x_t + \tilde z_t,
\end{aligned}
```

where $\tilde z_t = \bar z_{t+1} - D\bar z_t$ is the quasi-differenced
observation.

Given $\hat x_0$, equation {eq}`model1_recursion` can be used recursively
to compute a $\{u_t\}$ process.

Equations {eq}`model1_loglik` and {eq}`model1_recursion` give the
likelihood function of a sample of error-corrupted data
$\{\bar z_t\}$.

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

Let's look at the the covariance matrix of the innovations

```{code-cell} ipython3
print('Covariance matrix of innovations:')
df_v1 = pd.DataFrame(np.round(V1, 4), index=labels, columns=labels)
display(Latex(df_to_latex_matrix(df_v1)))
```

The covariance matrix of the innovations is not diagonal, but the
eigenvalues are well separated as shown below


```{code-cell} ipython3
print('Eigenvalues of covariance matrix:')
print(np.sort(np.linalg.eigvalsh(V1))[::-1].round(4))
```

The first eigenvalue is much larger than the others, consistent with
the presence of a dominant common shock $\theta_t$

### Wold impulse responses

The Wold impulse responses are reported using orthogonalized
innovations (Cholesky factorization of $V_1$ with ordering
$y_n$, $c$, $\Delta k$).

Under this method, lag-0 responses reflect both
contemporaneous covariance and the Cholesky ordering.

We first define a helper function to format the Wold responses as a LaTeX array

```{code-cell} ipython3
lags = np.arange(14)

def wold_response_table(resp, shock_idx, lags):
    return pd.DataFrame(
        np.round(resp[:, :, shock_idx], 4),
        columns=labels,
        index=pd.Index(lags, name='j')
    )
```

Now we report the Wold responses to each orthogonalized innovation in a single table with three panels

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

## A Model of Optimal Estimates Reported by an Agency

Suppose that instead of reporting the error-corrupted data $\bar z_t$,
the data collecting agency reports linear least-squares projections of
the true data on a history of the error-corrupted data.

This model provides a possible way of interpreting two features of
the data-reporting process.

- *seasonal adjustment*: if the components of $v_t$ have
strong seasonals, the optimal filter will assume a shape that can be
interpreted partly in terms of a seasonal adjustment filter, one that
is one-sided in current and past $\bar z_t$'s.

- *data revisions*: if $z_t$ contains current and lagged
values of some variable of interest, then the model simultaneously
determines "preliminary," "revised," and "final" estimates as
successive conditional expectations based on progressively longer
histories of error-ridden observations.

To make this operational, we impute to the reporting agency a model of
the joint process generating the true data and the measurement errors.

We assume that the reporting agency has "rational expectations": it
knows the economic and measurement structure leading to
{eq}`model1_transformed`--{eq}`model1_covs`.

To prepare its estimates, the reporting agency itself computes the
Kalman filter to obtain the innovations representation {eq}`model1_innov`.

Rather than reporting the error-corrupted data $\bar z_t$, the agency
reports $\tilde z_t = G \hat x_t$, where $G$ is a "selection matrix,"
possibly equal to $C$, for the data reported by the agency.

The data $G \hat x_t = E[G x_t \mid \bar z_t, \bar z_{t-1}, \ldots, \hat x_0]$.

The state-space representation for the reported data is then

```{math}
:label: model2_state
\begin{aligned}
\hat x_{t+1} &= A \hat x_t + K_1 u_t, \\
\tilde z_t &= G \hat x_t,
\end{aligned}
```

where the first line of {eq}`model2_state` is from the innovations
representation {eq}`model1_innov`.

Note that $u_t$ is the innovation to $\bar z_{t+1}$ and is *not* the
innovation to $\tilde z_t$.

To obtain a Wold representation for $\tilde z_t$ and the likelihood
function for a sample of $\tilde z_t$ requires that we obtain an
innovations representation for {eq}`model2_state`.

### Innovations representation for filtered data

To add a little generality to {eq}`model2_state` we amend it to the system

```{math}
:label: model2_obs
\begin{aligned}
\hat x_{t+1} &= A \hat x_t + K_1 u_t, \\
\tilde z_t &= G \hat x_t + \eta_t,
\end{aligned}
```

where $\eta_t$ is a type 2 white-noise measurement error process
("typos") with presumably very small covariance matrix $R_2$.

The covariance matrix of the joint noise is

```{math}
:label: model2_Q
E \begin{bmatrix} K_1 u_t \\ \eta_t \end{bmatrix}
  \begin{bmatrix} K_1 u_t \\ \eta_t \end{bmatrix}^\top
= \begin{bmatrix} Q_2 & 0 \\ 0 & R_2 \end{bmatrix},
```

where $Q_2 = K_1 V_1 K_1^\top$.

If $R_2$ is singular, it is necessary to adjust the Kalman filtering
formulas by using transformations that induce a "reduced order observer."

In practice, we approximate a zero $R_2$ matrix with the matrix
$\epsilon I$ for a small $\epsilon > 0$ to keep the Kalman filter
numerically well-conditioned.

For system {eq}`model2_obs` and {eq}`model2_Q`, an innovations
representation is

```{math}
:label: model2_innov
\begin{aligned}
\check{x}_{t+1} &= A \check{x}_t + K_2 a_t, \\
\tilde z_t &= G \check{x}_t + a_t,
\end{aligned}
```

where

```{math}
:label: model2_innov_defs
\begin{aligned}
a_t &= \tilde z_t - E[\tilde z_t \mid \tilde z_{t-1}, \tilde z_{t-2}, \ldots], \\
\check{x}_t &= E[\hat x_t \mid \tilde z_{t-1}, \tilde z_{t-2}, \ldots, \check{x}_0], \\
S_2 &= E[(\hat x_t - \check{x}_t)(\hat x_t - \check{x}_t)^\top], \\
[K_2, S_2] &= \text{kelmanfilter}(A, G, Q_2, R_2, 0).
\end{aligned}
```

Thus $\{a_t\}$ is the innovation process for the reported data
$\tilde z_t$, with innovation covariance

```{math}
:label: model2_V2
V_2 = E\, a_t a_t^\top = G\, S_2\, G^\top + R_2.
```

### Wold representation

A Wold moving average representation for $\tilde z_t$ is found from
{eq}`model2_innov` to be

```{math}
:label: model2_wold
\tilde z_t = \bigl[G(I - AL)^{-1} K_2 L + I\bigr] a_t,
```

with coefficients $\psi_0 = I$ and $\psi_j = G A^{j-1} K_2$ for
$j \geq 1$.

Note that this is simpler than the Model 1 Wold
representation {eq}`model1_wold` because there is no quasi-differencing
to undo.

### Gaussian likelihood

When a method analogous to Model 1 is used, a Gaussian log-likelihood
for $\tilde z_t$ can be computed by first computing an $\{a_t\}$ sequence
from observations on $\tilde z_t$ by using

```{math}
:label: model2_recursion
\begin{aligned}
\check{x}_{t+1} &= (A - K_2 G)\,\check{x}_t + K_2 \tilde z_t, \\
a_t &= -G\,\check{x}_t + \tilde z_t.
\end{aligned}
```

The likelihood function for a sample of $T$ observations
$\{\tilde z_t\}$ is then

```{math}
:label: model2_loglik
\mathcal{L}^{**} = -T\ln 2\pi - \tfrac{1}{2}T\ln|V_2|
  - \tfrac{1}{2}\sum_{t=0}^{T-1} a_t' V_2^{-1} a_t.
```

Note that relative to computing the likelihood function
{eq}`model1_loglik` for the error-corrupted data, computing the
likelihood function for the optimally filtered data requires more
calculations.

Both likelihood functions require that the Kalman filter
{eq}`model1_innov_defs` be computed, while the likelihood function for
the filtered data requires that the Kalman filter
{eq}`model2_innov_defs` also be computed.

In effect, in order to interpret and use the filtered data reported by
the agency, it is necessary to retrace the steps that the agency used
to synthesize those data.

The Kalman filter {eq}`model1_innov_defs` is supposed to be formed by
the agency.

The agency need not use Kalman filter {eq}`model2_innov_defs` because
it does not need the Wold representation for the filtered data.

In our parameterization $G = C$.

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
resp2 = np.array(
  [psi2[j] @ linalg.cholesky(V2, lower=True) for j in range(14)])
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
shock $\theta_t$.

Readers can compare the left table with the table in the
{ref}`true-impulse-responses` section above.

The numbers are essentially the same.

The consumption and investment innovations produce responses
that are orders of magnitude smaller, confirming that the filtered
data are driven by essentially one shock.

Unlike Model 1, the filtered data from Model 2
*cannot* reproduce the apparent Granger causality pattern that the
accelerator literature has documented empirically.


Hence, at the population level, the two measurement models imply different
empirical stories even though they share the same structural economy.

- In Model 1 (raw data), measurement noise creates multiple innovations
  and an apparent Granger-causality pattern.
- In Model 2 (filtered data), innovations collapse back to essentially
  one dominant shock, mirroring the true one-index economy.

Let's verify these implications in a finite sample simulation.

## Simulation

The tables above characterize population moments of the two models.

Let's simulate 80 periods of true, measured, and filtered data
to compare population implications with finite-sample behavior.

First, we define a function to simulate the true economy, generate measured data with AR(1) measurement errors, and apply the Model 1 Kalman filter to produce filtered estimates

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

We use the following helper function to plot the true series against either the measured or filtered series

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

Let's first compare the true series with the measured series to see how measurement errors distort the data

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


For the filtered series, we expect the Kalman filter to recover the true series more closely by stripping away measurement noise


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

Indeed, Kalman-filtered estimates from Model 1 remove much of the
measurement noise and track the truth closely.

In the true model the national income identity
$c_t + \Delta k_t = y_{n,t}$ holds exactly.

Independent measurement errors break this accounting identity
in the measured data.

The Kalman filter approximately restores it.

The following figure confirms this by showing the residual $c_t + \Delta k_t - y_{n,t}$ for
both measured and filtered data

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

As we have predicted, the residual for the measured data is large and volatile, while the residual for the filtered data is numerically 0.

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
