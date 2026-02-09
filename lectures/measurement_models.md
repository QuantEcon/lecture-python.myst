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

In this lecture we reproduce all numbered tables and figures from
{cite}`Sargent1989` while studying the underlying mechanisms in the paper.

We use the following imports and precision settings for tables:

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

np.set_printoptions(precision=4, suppress=True)
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

and matrices

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
\end{bmatrix}.
```

The econometrician does not observe $z_t$ directly but instead
sees $\bar z_t = z_t + v_t$, where $v_t$ is a vector of measurement
errors.

Measurement errors are AR(1):

```{math}
v_t = D v_{t-1} + \eta_t,
```

with diagonal

```{math}
D = \operatorname{diag}(0.6, 0.7, 0.3),
```

and innovation standard deviations $(0.05, 0.035, 0.65)$.

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

# Innovation std. devs shown in Table 1
σ_η = np.array([0.05, 0.035, 0.65])
Σ_η = np.diag(σ_η**2)

# Unconditional covariance of measurement errors v_t
R = np.diag((σ_η / np.sqrt(1.0 - ρ**2))**2)

print(f"f = {f},  β = 1/f = {β:.6f}")
print("\nA ="); display(pd.DataFrame(A))
print("C ="); display(pd.DataFrame(C))
print("D ="); display(pd.DataFrame(D))
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

## Table 2: True Impulse Responses

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

pd.DataFrame(
    np.round(rep_table2[:, 1:], 4),
    columns=[r'$y_n$', r'$c$', r'$\Delta k$'],
    index=pd.Index(range(6), name='Lag')
)
```

## Model 1 (Raw Measurements): Tables 3 and 4

Model 1 treats the raw measured series $\bar z_t$ as the observables and
applies a Kalman filter to extract the state.

Because the measurement errors $v_t$ are serially correlated, Sargent
quasi-differences the observation equation to obtain an innovation form
with serially uncorrelated errors.

The transformed observation equation is

```{math}
\bar z_t - D \bar z_{t-1} = (CA - DC)x_{t-1} + C w_t + \eta_t.
```

Hence

```{math}
\bar C = CA - DC, \quad R_1 = CQC^\top + R, \quad W_1 = QC^\top.
```

```{code-cell} ipython3
C_bar = C @ A - D @ C
R1 = C @ Q @ C.T + R
W1 = Q @ C.T

K1, S1, V1 = steady_state_kalman(A, C_bar, Q, R1, W1)
```

With the Kalman gain in hand, we can derive the Wold moving-average
representation for the measured data.

This representation tells us how measured $y_n$, $c$, and $\Delta k$
respond over time to the orthogonalized innovations in the
innovation covariance matrix $V_1$.

To recover the Wold representation, define the augmented state

```{math}
r_t = \begin{bmatrix} \hat x_{t-1} \\ z_{t-1} \end{bmatrix},
```

with dynamics

```{math}
r_{t+1} = F_1 r_t + G_1 u_t,
\qquad
z_t = H_1 r_t + u_t,
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

Table 3 reports the forecast-error-variance decomposition for Model 1.

Each panel shows the cumulative contribution of one orthogonalized
innovation to the forecast-error variance of $y_n$, $c$, and $\Delta k$
at horizons 1 through 20.

```{code-cell} ipython3
horizons = np.arange(1, 21)
cols = [r'$y_n$', r'$c$', r'$\Delta k$']

def fev_table(decomp, shock_idx, horizons):
    return pd.DataFrame(
        np.round(decomp[:, shock_idx, :].T, 4),
        columns=cols,
        index=pd.Index(horizons, name='Horizon')
    )

print("Table 3A: Contribution of innovation 1")
display(fev_table(decomp1, 0, horizons))

print("Table 3B: Contribution of innovation 2")
display(fev_table(decomp1, 1, horizons))

print("Table 3C: Contribution of innovation 3")
display(fev_table(decomp1, 2, horizons))
```

The innovation covariance matrix $V_1$ is:

```{code-cell} ipython3
labels = [r'$y_n$', r'$c$', r'$\Delta k$']
pd.DataFrame(np.round(V1, 4), index=labels, columns=labels)
```

Table 4 reports the orthogonalized Wold impulse responses for Model 1
at lags 0 through 13.

```{code-cell} ipython3
lags = np.arange(14)

def wold_response_table(resp, shock_idx, lags):
    return pd.DataFrame(
        np.round(resp[:, :, shock_idx], 4),
        columns=cols,
        index=pd.Index(lags, name='Lag')
    )

print("Table 4A: Response to innovation in y_n")
display(wold_response_table(resp1, 0, lags))

print("Table 4B: Response to innovation in c")
display(wold_response_table(resp1, 1, lags))

print("Table 4C: Response to innovation in Δk")
display(wold_response_table(resp1, 2, lags))
```

## Model 2 (Filtered Measurements): Tables 5 and 6

Model 2 takes a different approach: instead of working with the raw data,
the econometrician first applies the Kalman filter from Model 1 to
strip out measurement error and then treats the filtered estimates
$\hat z_t = C \hat x_t$ as if they were the true observations.

A second Kalman filter is then applied to the filtered series.

The state noise covariance for this second filter is

```{math}
Q_2 = K_1 V_1 K_1^\top,
```

We solve a second Kalman system with tiny measurement noise to regularize the
near-singular covariance matrix.

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

Table 5 is the analogue of Table 3 for Model 2.

Because the filtered data are nearly noiseless, the second and third
innovations contribute very little to forecast-error variance.

```{code-cell} ipython3
print("Table 5A: Contribution of innovation 1")
display(fev_table(decomp2, 0, horizons))

print("Table 5B: Contribution of innovation 2 (×10³)")
display(pd.DataFrame(
    np.round(decomp2[:, 1, :].T * 1e3, 4),
    columns=cols,
    index=pd.Index(horizons, name='Horizon')
))

print("Table 5C: Contribution of innovation 3 (×10⁶)")
display(pd.DataFrame(
    np.round(decomp2[:, 2, :].T * 1e6, 4),
    columns=cols,
    index=pd.Index(horizons, name='Horizon')
))
```

The innovation covariance matrix $V_2$ for Model 2 is:

```{code-cell} ipython3
pd.DataFrame(np.round(V2, 4), index=labels, columns=labels)
```

Table 6 reports the orthogonalized Wold impulse responses for Model 2.

```{code-cell} ipython3
print("Table 6A: Response to innovation in y_n")
display(wold_response_table(resp2, 0, lags))

print("Table 6B: Response to innovation in c")
display(wold_response_table(resp2, 1, lags))

print("Table 6C: Response to innovation in Δk (×10³)")
display(pd.DataFrame(
    np.round(resp2[:, :, 2] * 1e3, 4),
    columns=cols,
    index=pd.Index(lags, name='Lag')
))
```

## Simulation: Figures 1 through 9 and Table 7

The tables above characterize population moments of the two models.

To see how the models perform on a finite sample, Sargent simulates
80 periods of true, measured, and filtered data and reports
covariance and correlation matrices (Table 7) together with
time-series plots (Figures 1 through 9).

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
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(t, true_series, lw=2, color="black", label="true")
    ax.plot(t, other_series, lw=2, ls="--", color="#1f77b4", label=other_label)
    ax.set_xlabel("time", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(loc="best")
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

Figures 1 through 3 show how measurement error distorts each series.

Investment (Figure 2) is hit hardest because its measurement error
has the largest innovation variance ($\sigma_\eta = 0.65$).

Figures 4 through 7 compare the true series with the Kalman-filtered
estimates from Model 1.

The filter removes much of the measurement
noise, recovering series that track the truth closely.

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

Figures 8 and 9 plot the national income identity residual
$c_t + \Delta k_t - y_{n,t}$.

In the true model this identity holds exactly.

For measured data (Figure 8) the residual is non-zero because
independent measurement errors break the accounting identity.

For filtered data (Figure 9) the Kalman filter approximately
restores the identity.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Measured consumption plus investment minus income
    name: fig-measured-identity-residual
  image:
    alt: National income identity residual for measured data over 80 time periods
---
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(t, sim["c_meas"] + sim["dk_meas"] - sim["y_meas"], color="#d62728", lw=2)
ax.set_xlabel("time", fontsize=11)
ax.set_ylabel("residual", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Filtered consumption plus investment minus income
    name: fig-filtered-identity-residual
  image:
    alt: National income identity residual for filtered data over 80 time periods
---
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(t, sim["c_filt"] + sim["dk_filt"] - sim["y_filt"], color="#2ca02c", lw=2)
ax.set_xlabel("time", fontsize=11)
ax.set_ylabel("residual", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

Table 7 reports covariance and correlation matrices among the true,
measured, and filtered versions of each variable.

High correlations between true and filtered series confirm that the
Kalman filter does a good job of removing measurement noise.

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

print("Table 7A: Covariance matrix of consumption")
display(matrix_df(cov_c, tmf_labels))

print("Table 7B: Correlation matrix of consumption")
display(matrix_df(corr_c, tmf_labels))

print("Table 7C: Covariance matrix of investment")
display(matrix_df(cov_i, tmf_labels))

print("Table 7D: Correlation matrix of investment")
display(matrix_df(corr_i, tmf_labels))

print("Table 7E: Covariance matrix of income")
display(matrix_df(cov_y, tmf_labels))

print("Table 7F: Correlation matrix of income")
display(matrix_df(corr_y, tmf_labels))

print("Table 7G: Covariance matrix of capital")
display(matrix_df(cov_k, tf_labels))

print("Table 7H: Correlation matrix of capital")
display(matrix_df(corr_k, tf_labels))
```

## Summary

This lecture reproduced the tables and figures in {cite}`Sargent1989`,
which studies how measurement error alters an econometrician's view
of a permanent income economy driven by the investment accelerator.

Several lessons emerge:

* The Wold representations and variance decompositions of Model 1 (raw
  measurements) and Model 2 (filtered measurements) are quite different,
  even though the underlying economy is the same.

* Measurement error is not a second-order issue: it can
  reshape inferences about which shocks drive which variables.

* The {doc}`Kalman filter <kalman>` effectively strips measurement noise
  from the data.

* The filtered series track the truth closely
  (Figures 4 through 7), and the near-zero residual in Figure 9 shows that
  the filter approximately restores the national income accounting
  identity that raw measurement error breaks (Figure 8).

* The forecast-error-variance decompositions (Tables 3 and 5) reveal
  that Model 1 attributes substantial variance to measurement noise
  innovations, while Model 2, working with cleaned data, attributes
  nearly all variance to the single structural shock $\theta_t$.

These results connect to broader themes in this lecture series:
the role of {doc}`linear state space models <linear_models>` in
representing economic dynamics, the power of {doc}`Kalman filtering <kalman>`
for signal extraction, and the importance of the investment accelerator
for understanding business cycles ({doc}`samuelson`,
{doc}`chow_business_cycles`).

## References

* {cite}`Sargent1989`
