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

Sargent specifies a {doc}`linear state space model <linear_models>` for the
true economy and then considers two ways of extracting information from
noisy measurements:

- Model 1 applies a {doc}`Kalman filter <kalman>` directly to
  raw (noisy) observations.
- Model 2 first filters the data to remove measurement error,
  then computes dynamics from the filtered series.

The two models produce different Wold representations and
forecast-error-variance decompositions, even though they describe
the same underlying economy.

In this lecture we reproduce the analysis from {cite}`Sargent1989`
while studying the underlying mechanisms in the paper.

We use the following imports and precision settings for tables:

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from IPython.display import Latex

np.set_printoptions(precision=4, suppress=True)

def df_to_latex_matrix(df, label=''):
    """Convert DataFrame to LaTeX matrix (for math matrices)."""
    lines = [r'\begin{bmatrix}']

    for idx, row in df.iterrows():
        row_str = ' & '.join([f'{v:.4f}' if isinstance(v, (int, float)) else str(v) for v in row]) + r' \\'
        lines.append(row_str)

    lines.append(r'\end{bmatrix}')

    if label:
        return '$' + label + ' = ' + '\n'.join(lines) + '$'
    else:
        return '$' + '\n'.join(lines) + '$'

def df_to_latex_array(df):
    """Convert DataFrame to LaTeX array (for tables with headers)."""
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
        row_str = str(idx) + ' & ' + ' & '.join([f'{v:.4f}' if isinstance(v, (int, float)) else str(v) for v in row]) + r' \\'
        lines.append(row_str)

    lines.append(r'\end{array}')

    return '$' + '\n'.join(lines) + '$'
```

## Model Setup

The true economy is a version of the permanent income model
(see {doc}`perm_income`) in which a representative consumer
chooses consumption $c_t$ and capital accumulation $\Delta k_t$
to maximize expected discounted utility subject to a budget
constraint.

Assume that the discount factor satisfies $\beta f = 1$ and that the
productivity shock $\theta_t$ is white noise.

The optimal decision rules reduce the true system to

```{math}
\begin{aligned}
k_{t+1} &= k_t + f^{-1}\theta_t, \\
y_{n,t} &= (f-1)k_t + \theta_t, \\
c_t &= (f-1)k_t + (1-f^{-1})\theta_t, \\
\Delta k_t &= f^{-1}\theta_t.
\end{aligned}
```

with $f = 1.05$ and $\theta_t \sim \mathcal{N}(0, 1)$.

Here $k_t$ is capital, $y_{n,t}$ is national income, $c_t$ is consumption,
and $\Delta k_t$ is net investment.

Notice the investment accelerator at work: because $\Delta k_t = f^{-1}\theta_t$,
investment responds only to the innovation $\theta_t$, not to the level of
capital. 

This is the same mechanism that {cite:t}`Chow1968` documented
empirically (see {doc}`chow_business_cycles`).

We can cast this as a {doc}`linear state space model <linear_models>` by
defining state and observable vectors

```{math}
x_t = \begin{bmatrix} k_t \\ \theta_t \end{bmatrix},
\qquad
z_t = \begin{bmatrix} y_{n,t} \\ c_t \\ \Delta k_t \end{bmatrix},
```

so that the true economy follows the state-space system

```{math}
:label: true_ss
x_{t+1} = A x_t + \varepsilon_t, \qquad z_t = C x_t,
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

Note that $Q$ is singular because only the second component of $x_t$
(the productivity shock $\theta_t$) receives an innovation; the
capital stock $k_t$ evolves deterministically given $\theta_t$.

The econometrician does not observe $z_t$ directly but instead
sees $\bar z_t = z_t + v_t$, where $v_t$ is a vector of measurement
errors.

Measurement errors follow an AR(1) process:

```{math}
:label: meas_error_ar1
v_t = D v_{t-1} + \eta_t,
```

where $\eta_t$ is a vector white noise with
$E \eta_t \eta_t^\top = \Sigma_\eta$ and
$E \varepsilon_t v_s^\top = 0$ for all $t, s$
(measurement errors are orthogonal to the true state innovations).

The autoregressive matrix and innovation standard deviations are

```{math}
D = \operatorname{diag}(0.6, 0.7, 0.3),
\qquad
\sigma_\eta = (0.05, 0.035, 0.65),
```

so the unconditional covariance of $v_t$ is

```{math}
R = \operatorname{diag}\!\left(\frac{\sigma_{\eta,i}^2}{1 - \rho_i^2}\right).
```

The measurement errors are ordered from smallest to largest innovation
variance: income is measured most accurately ($\sigma_\eta = 0.05$),
consumption next ($\sigma_\eta = 0.035$), and investment least
accurately ($\sigma_\eta = 0.65$).
This ordering is central to the results below.

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

# Innovation std. devs
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

## Kalman Filter

Both models require a steady-state {doc}`Kalman filter <kalman>`.

The function below iterates on the Riccati equation until convergence,
returning the Kalman gain $K$, the state covariance $S$, and the
innovation covariance $V$.

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
## True Impulse Responses

Before introducing measurement error, we verify the impulse response of
the true system to a unit shock $\theta_0 = 1$.

The response shows the investment accelerator clearly: the full impact on
net income $y_n$ occurs at lag 0, while consumption adjusts by only
$1 - f^{-1} \approx 0.048$ and investment absorbs the remainder.

From lag 1 onward the economy is in its new steady state.

```{code-cell} ipython3
def table2_irf(A, C, n_lags=6):
    x = np.array([0.0, 1.0])  # k_0 = 0, theta_0 = 1
    rows = []
    for j in range(n_lags):
        y_n, c, d_k = C @ x
        rows.append([j, y_n, c, d_k])
        x = A @ x
    return np.array(rows)

rep_table2 = table2_irf(A, C, n_lags=6)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(rep_table2[:, 0], rep_table2[:, 1], 'o-', label=r'$y_n$', lw=2.5, markersize=7)
ax.plot(rep_table2[:, 0], rep_table2[:, 2], 's-', label=r'$c$', lw=2.5, markersize=7)
ax.plot(rep_table2[:, 0], rep_table2[:, 3], '^-', label=r'$\Delta k$', lw=2.5, markersize=7)
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('Response', fontsize=12)
ax.set_title(r'True impulse response to unit shock $\theta_0 = 1$', fontsize=13)
ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

## Model 1 (Raw Measurements)

Model 1 is a classical errors-in-variables model: the data collecting
agency simply reports the error-corrupted data $\bar z_t = z_t + v_t$
that it collects, making no attempt to adjust for measurement errors.

Because the measurement errors $v_t$ are serially correlated (AR(1)),
we cannot directly apply the Kalman filter to
$\bar z_t = C x_t + v_t$.
Following {cite:t}`Sargent1989` (Section III.B), we quasi-difference the
observation equation.

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
R_1 = C Q C^\top + \Sigma_\eta, \qquad W_1 = Q C^\top.
```

The system $\{x_{t+1} = A x_t + \varepsilon_t,\;
\bar z_{t+1} - D\bar z_t = \bar C x_t + \bar\nu_t\}$
with $\text{cov}(\varepsilon_t)=Q$, $\text{cov}(\bar\nu_t)=R_1$, and
$\text{cov}(\varepsilon_t, \bar\nu_t)=W_1$ now has serially uncorrelated
errors, so the standard {doc}`Kalman filter <kalman>` applies.

The steady-state Kalman filter yields the **innovations representation**

```{math}
:label: model1_innov
\hat x_{t+1} = A \hat x_t + K_1 u_t, \qquad
\bar z_{t+1} - D\bar z_t = \bar C \hat x_t + u_t,
```

where $u_t = (\bar z_{t+1} - D\bar z_t) -
E[\bar z_{t+1} - D\bar z_t \mid \bar z_t, \bar z_{t-1}, \ldots]$
is the innovation process, $K_1$ is the Kalman gain, and
$V_1 = \bar C S_1 \bar C^\top + R_1$ is the innovation covariance matrix
(with $S_1 = E[(x_t - \hat x_t)(x_t - \hat x_t)^\top]$ the steady-state
state estimation error covariance).

```{code-cell} ipython3
C_bar = C @ A - D @ C
R1 = C @ Q @ C.T + Σ_η
W1 = Q @ C.T

K1, S1, V1 = steady_state_kalman(A, C_bar, Q, R1, W1)
```

### Wold representation for measured data

With the innovations representation {eq}`model1_innov` in hand, we can
derive a Wold moving-average representation for the measured data
$\bar z_t$.

From {eq}`model1_innov` and the quasi-differencing definition, the
measured data satisfy (see eq. 19 of {cite:t}`Sargent1989`)

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

Each panel below shows the cumulative contribution of one orthogonalized
innovation to the forecast-error variance of $y_n$, $c$, and $\Delta k$
at horizons 1 through 20.

```{code-cell} ipython3
horizons = np.arange(1, 21)
cols = [r'y_n', r'c', r'\Delta k']

def fev_table(decomp, shock_idx, horizons):
    return pd.DataFrame(
        np.round(decomp[:, shock_idx, :].T, 4),
        columns=cols,
        index=pd.Index(horizons, name='Horizon')
    )
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for i, (shock_name, ax) in enumerate(zip([r'Innovation 1 ($y_n$)', r'Innovation 2 ($c$)', r'Innovation 3 ($\Delta k$)'], axes)):
    fev_data = decomp1[:, i, :]
    ax.plot(horizons, fev_data[0, :], label=r'$y_n$', lw=2.5)
    ax.plot(horizons, fev_data[1, :], label=r'$c$', lw=2.5)
    ax.plot(horizons, fev_data[2, :], label=r'$\Delta k$', lw=2.5)
    ax.set_xlabel('Horizon', fontsize=12)
    ax.set_ylabel('Contribution to FEV', fontsize=12)
    ax.set_title(shock_name, fontsize=13)
    ax.legend(loc='best', fontsize=10, frameon=True, shadow=True)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

These plots replicate Table 3 of {cite:t}`Sargent1989`.
The income innovation accounts for substantial proportions of
forecast-error variance in all three variables, while the consumption and
investment innovations contribute mainly to their own variances.
This is a **Granger causality** pattern: income appears to
Granger-cause consumption and investment, but not vice versa.
The pattern arises because income is the best-measured variable
($\sigma_\eta = 0.05$), so its innovation carries the most
information about the underlying structural shock $\theta_t$.

The innovation covariance matrix $V_1$ is:

```{code-cell} ipython3
labels = [r'y_n', r'c', r'\Delta k']
df_v1 = pd.DataFrame(np.round(V1, 4), index=labels, columns=labels)
display(Latex(df_to_latex_matrix(df_v1)))
```

### Wold impulse responses

The orthogonalized Wold impulse responses $\psi_j P$ show how the
measured variables respond at lag $j$ to a one-standard-deviation
orthogonalized innovation.  We plot lags 0 through 13.

```{code-cell} ipython3
lags = np.arange(14)

def wold_response_table(resp, shock_idx, lags):
    return pd.DataFrame(
        np.round(resp[:, :, shock_idx], 4),
        columns=cols,
        index=pd.Index(lags, name='Lag')
    )
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for i, (shock_name, ax) in enumerate(zip([r'Innovation in $y_n$', r'Innovation in $c$', r'Innovation in $\Delta k$'], axes)):
    ax.plot(lags, resp1[:, 0, i], label=r'$y_n$', lw=2.5)
    ax.plot(lags, resp1[:, 1, i], label=r'$c$', lw=2.5)
    ax.plot(lags, resp1[:, 2, i], label=r'$\Delta k$', lw=2.5)
    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Response', fontsize=12)
    ax.set_title(shock_name, fontsize=13)
    ax.legend(loc='best', fontsize=10, frameon=True, shadow=True)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

These plots replicate Table 4 of {cite:t}`Sargent1989`.
An income innovation generates persistent responses in all variables
because, being the best-measured series, its innovation is dominated
by the true permanent shock $\theta_t$, which permanently raises the
capital stock and hence steady-state consumption and income.
A consumption innovation produces smaller, decaying responses
that reflect the AR(1) structure of its measurement error ($\rho = 0.7$).
An investment innovation has a large initial impact on investment itself,
consistent with the high measurement error variance ($\sigma_\eta = 0.65$),
but the effect dies out quickly.

## Model 2 (Filtered Measurements)

Model 2 corresponds to a data collecting agency that, instead of
reporting raw error-corrupted data, applies an optimal filter
to construct least-squares estimates of the true variables.

Specifically, the agency uses the Kalman filter from Model 1 to form
$\hat x_t = E[x_t \mid \bar z_{t-1}, \bar z_{t-2}, \ldots]$ and reports
filtered estimates

```{math}
\tilde z_t = G \hat x_t,
```

where $G = C$ is a selection matrix
(see eq. 23 of {cite:t}`Sargent1989`).

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
(see eq. 25 of {cite:t}`Sargent1989`)

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
\hat{\hat x}_{t+1} = A \hat{\hat x}_t + K_2 a_t,
\qquad
\tilde z_t = C \hat{\hat x}_t + a_t,
```

where $a_t$ is the innovation process for the filtered data with
covariance $V_2 = C S_2 C^\top + R_2$.

### Wold representation for filtered data

The Wold moving-average representation for $\tilde z_t$ is
(see eq. 29 of {cite:t}`Sargent1989`)

```{math}
:label: model2_wold
\tilde z_t = \bigl[C(I - AL)^{-1} K_2 L + I\bigr] a_t,
```

with coefficients $\psi_0 = I$ and $\psi_j = C A^{j-1} K_2$ for
$j \geq 1$.  Note that this is simpler than the Model 1 Wold
representation {eq}`model1_wold` because there is no quasi-differencing
to undo.

```{code-cell} ipython3
Q2 = K1 @ V1 @ K1.T
ε = 1e-7

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
just like the true economy.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for i, (shock_name, ax) in enumerate(zip([r'Innovation 1 ($y_n$)', r'Innovation 2 ($c$) $\times 10^3$', r'Innovation 3 ($\Delta k$) $\times 10^6$'], axes)):
    scale = 1 if i == 0 else (1e3 if i == 1 else 1e6)
    fev_data = decomp2[:, i, :] * scale
    ax.plot(horizons, fev_data[0, :], label=r'$y_n$', lw=2.5)
    ax.plot(horizons, fev_data[1, :], label=r'$c$', lw=2.5)
    ax.plot(horizons, fev_data[2, :], label=r'$\Delta k$', lw=2.5)
    ax.set_xlabel('Horizon', fontsize=12)
    ax.set_ylabel('Contribution to FEV', fontsize=12)
    ax.set_title(shock_name, fontsize=13)
    ax.legend(loc='best', fontsize=10, frameon=True, shadow=True)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

These plots replicate Table 5 of {cite:t}`Sargent1989`.
In Model 2, the first innovation accounts for virtually all forecast-error
variance, just as in the true economy where the single structural shock
$\theta_t$ drives everything.
The second and third innovations contribute negligibly (note the scaling
factors of $10^3$ and $10^6$ required to make them visible).
This confirms that filtering strips away the measurement noise that created
the appearance of multiple independent sources of variation in Model 1.

The innovation covariance matrix $V_2$ for Model 2 is:

```{code-cell} ipython3
df_v2 = pd.DataFrame(np.round(V2, 4), index=labels, columns=labels)
display(Latex(df_to_latex_matrix(df_v2)))
```

### Wold impulse responses

The following plots show the orthogonalized Wold impulse responses for Model 2.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for i, (shock_name, scale) in enumerate(zip([r'Innovation in $y_n$', r'Innovation in $c$ $\times 10^3$', r'Innovation in $\Delta k$ $\times 10^3$'],
                                             [1, 1e3, 1e3])):
    ax = axes[i]
    ax.plot(lags, resp2[:, 0, i] * scale, label=r'$y_n$', lw=2.5)
    ax.plot(lags, resp2[:, 1, i] * scale, label=r'$c$', lw=2.5)
    ax.plot(lags, resp2[:, 2, i] * scale, label=r'$\Delta k$', lw=2.5)
    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Response', fontsize=12)
    ax.set_title(shock_name, fontsize=13)
    ax.legend(loc='best', fontsize=10, frameon=True, shadow=True)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

These plots replicate Table 6 of {cite:t}`Sargent1989`.
The income innovation in Model 2 produces responses that closely
approximate the true impulse response function from the structural
shock $\theta_t$ (compare with the figure in the
{ref}`true-impulse-responses` section above).
The consumption and investment innovations produce responses
that are orders of magnitude smaller (note the $10^3$ scaling),
confirming that the filtered data are driven by essentially one shock.

A key implication: unlike Model 1, the filtered data from Model 2
**cannot** reproduce the apparent Granger causality pattern that the
accelerator literature has documented empirically.
As {cite:t}`Sargent1989` emphasizes, the two models of measurement
produce quite different inferences about the economy's dynamics despite
sharing identical deep parameters.

## Simulation

The tables above characterize population moments of the two models.

To see how the models perform on a finite sample, Sargent simulates
80 periods of true, measured, and filtered data and reports
covariance and correlation matrices together with time-series plots.

We replicate these objects below.

```{code-cell} ipython3
def simulate_series(seed=7909, T=80, k0=10.0):
    """
    Simulate true, measured, and filtered series for Figures 1--9.
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
        "y_meas": z_meas[:, 0], "c_meas": z_meas[:, 1], "dk_meas": z_meas[:, 2],
        "y_filt": z_filt[:, 0], "c_filt": z_filt[:, 1], "dk_filt": z_filt[:, 2], "k_filt": k_filt
    }
    return out


sim = simulate_series(seed=7909, T=80, k0=10.0)
```

```{code-cell} ipython3
def plot_true_vs_other(t, true_series, other_series, other_label, ylabel=""):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, true_series, lw=2.5, color="black", label="true")
    ax.plot(t, other_series, lw=2.5, ls="--", color="#1f77b4", label=other_label)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(ylabel.capitalize(), fontsize=12)
    ax.legend(loc="best", fontsize=11, frameon=True, shadow=True)
    ax.grid(alpha=0.3)
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
plot_true_vs_other(t, sim["c_true"], sim["c_meas"], "measured", ylabel="consumption")
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
plot_true_vs_other(t, sim["dk_true"], sim["dk_meas"], "measured", ylabel="investment")
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
plot_true_vs_other(t, sim["y_true"], sim["y_meas"], "measured", ylabel="income")
```

The first three figures replicate Figures 1--3 of {cite:t}`Sargent1989`.
Investment is distorted the most because its measurement error
has the largest innovation variance ($\sigma_\eta = 0.65$),
while income is distorted the least ($\sigma_\eta = 0.05$).

The next four figures (Figures 4--7 in the paper) compare
true series with the Kalman-filtered estimates from Model 1.
The filter removes much of the measurement noise, recovering
series that track the truth closely.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: True and filtered consumption
    name: fig-true-filtered-consumption
  image:
    alt: True and filtered consumption plotted over 80 time periods
---
plot_true_vs_other(t, sim["c_true"], sim["c_filt"], "filtered", ylabel="consumption")
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
plot_true_vs_other(t, sim["dk_true"], sim["dk_filt"], "filtered", ylabel="investment")
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
plot_true_vs_other(t, sim["y_true"], sim["y_filt"], "filtered", ylabel="income")
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
plot_true_vs_other(t, sim["k_true"], sim["k_filt"], "filtered", ylabel="capital stock")
```

The following figure plots the national income identity residual
$c_t + \Delta k_t - y_{n,t}$ for both measured and filtered data
(Figures 8--9 of {cite:t}`Sargent1989`).

In the true model this identity holds exactly.
For measured data the residual is non-zero because
independent measurement errors break the accounting identity.
For filtered data the Kalman filter approximately restores the identity.

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

ax1.plot(t, sim["c_meas"] + sim["dk_meas"] - sim["y_meas"], color="#d62728", lw=2.5)
ax1.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax1.set_xlabel("Time", fontsize=12)
ax1.set_ylabel("Residual", fontsize=12)
ax1.set_title(r'Measured: $c_t + \Delta k_t - y_{n,t}$', fontsize=13)
ax1.grid(alpha=0.3)

ax2.plot(t, sim["c_filt"] + sim["dk_filt"] - sim["y_filt"], color="#2ca02c", lw=2.5)
ax2.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax2.set_xlabel("Time", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title(r'Filtered: $c_t + \Delta k_t - y_{n,t}$', fontsize=13)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

The following covariance and correlation matrices replicate Table 7
of {cite:t}`Sargent1989`.
For each variable we report the $3 \times 3$ covariance and correlation
matrices among the true, measured, and filtered versions.

High correlations between true and filtered series confirm that the
Kalman filter removes most measurement noise.
Lower correlations between true and measured series quantify how much
information is lost by using raw data.

```{code-cell} ipython3
def cov_corr_three(a, b, c):
    X = np.vstack([a, b, c])
    return np.cov(X), np.corrcoef(X)

def matrix_df(mat, labels):
    return pd.DataFrame(np.round(mat, 4), index=labels, columns=labels)

cov_c, corr_c = cov_corr_three(sim["c_true"], sim["c_meas"], sim["c_filt"])
cov_i, corr_i = cov_corr_three(sim["dk_true"], sim["dk_meas"], sim["dk_filt"])
cov_y, corr_y = cov_corr_three(sim["y_true"], sim["y_meas"], sim["y_filt"])
cov_k = np.cov(np.vstack([sim["k_true"], sim["k_filt"]]))
corr_k = np.corrcoef(np.vstack([sim["k_true"], sim["k_filt"]]))

tmf_labels = ['true', 'measured', 'filtered']
tf_labels = ['true', 'filtered']
```

**Consumption** -- Measurement error inflates variance, but the filtered
series recovers a variance close to the truth.
The true-filtered correlation exceeds 0.99.

```{code-cell} ipython3
display(Latex(df_to_latex_matrix(matrix_df(cov_c, tmf_labels))))
display(Latex(df_to_latex_matrix(matrix_df(corr_c, tmf_labels))))
```

**Investment** -- Because $\sigma_\eta = 0.65$ is large, measurement error
creates the most variance inflation here.
Despite this, the true-filtered correlation remains high,
demonstrating the filter's effectiveness even with severe noise.

```{code-cell} ipython3
display(Latex(df_to_latex_matrix(matrix_df(cov_i, tmf_labels))))
display(Latex(df_to_latex_matrix(matrix_df(corr_i, tmf_labels))))
```

**Income** -- Income has the smallest measurement error, so measured
and true variances are close.  True-filtered correlations are very high.

```{code-cell} ipython3
display(Latex(df_to_latex_matrix(matrix_df(cov_y, tmf_labels))))
display(Latex(df_to_latex_matrix(matrix_df(corr_y, tmf_labels))))
```

**Capital stock** -- The capital stock is never directly observed, yet
the filter recovers it with very high accuracy.

```{code-cell} ipython3
display(Latex(df_to_latex_matrix(matrix_df(cov_k, tf_labels))))
display(Latex(df_to_latex_matrix(matrix_df(corr_k, tf_labels))))
```

## Summary

This lecture reproduced the analysis in {cite}`Sargent1989`,
which studies how measurement error alters an econometrician's view
of a permanent income economy driven by the investment accelerator.

Several lessons emerge:

* The Wold representations and variance decompositions of Model 1 (raw
  measurements) and Model 2 (filtered measurements) are quite different,
  even though the underlying economy is the same.

* Measurement error is not a second-order issue: it can
  reshape inferences about which shocks drive which variables.

* Model 1 reproduces the **Granger causality** pattern documented in the
  empirical accelerator literature -- income appears to Granger-cause
  consumption and investment -- but this pattern is an artifact of
  measurement error ordering, not of the structural model.

* Model 2, working with filtered data, attributes nearly all variance to
  the single structural shock $\theta_t$ and **cannot** reproduce the
  Granger causality pattern.

* The {doc}`Kalman filter <kalman>` effectively strips measurement noise
  from the data: the filtered series track the truth closely, and the
  near-zero residual shows that the filter approximately restores the
  national income accounting identity that raw measurement error breaks.

These results connect to broader themes in this lecture series:
the role of {doc}`linear state space models <linear_models>` in
representing economic dynamics, the power of {doc}`Kalman filtering <kalman>`
for signal extraction, and the importance of the investment accelerator
for understanding business cycles ({doc}`samuelson`,
{doc}`chow_business_cycles`).

## References

* {cite}`Sargent1989`
