---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

(hansen_singleton)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Estimating Euler Equations by GMM and Maximum Likelihood

```{index} single: Asset Pricing; Estimating Euler Equations
```

```{contents} Contents
:depth: 2
```

## Overview

This lecture presents a unified treatment of the consumption-based Euler equation using the GMM approach of {cite}`hansen1982generalized` and the likelihood approach of {cite}`hansen1983stochastic`.

We cover:

- the consumption CRRA Euler equation and its stochastic discount factor representation
- GMM estimation using lagged instruments and HAC covariance, with Hansen's $J$ test for overidentifying restrictions
- a lognormal triangular-system likelihood that imposes stronger distributional assumptions for efficiency gains
- simulation exercises verifying that both estimators recover known parameters
- empirical estimation on monthly FRED consumption and CRSP stock-return data

In addition to what comes with Anaconda, this lecture requires `pandas-datareader`.

```{code-cell} ipython3
:tags: [hide-output]

!pip install pandas-datareader
```

Let's start by importing packages and defining a helper for displaying tables in LaTeX format

```{code-cell} ipython3
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Math
from numba import njit
from pandas_datareader import data as web
from scipy import stats
from scipy.linalg import LinAlgError, cholesky, solve_triangular
from scipy.optimize import minimize
from statsmodels.sandbox.regression import gmm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf


def display_table(df, title=None, fmt=None):
    """
    Display a DataFrame as a LaTeX array using IPython Math.
    """
    if fmt is None:
        fmt = {}
    formatted = df.copy()
    for col in formatted.columns:
        if col in fmt:
            formatted[col] = formatted[col].apply(
                lambda x: fmt[col].format(x) if np.isfinite(x) else str(x))
    idx_header = r"\text{" + df.index.name + "}" if df.index.name else ""
    columns = " & ".join(
        [idx_header] + [r"\text{" + c + "}" if "\\" not in c
         else c for c in formatted.columns])
    rows = r" \\".join(
        [" & ".join([str(idx)] + [str(v) for v in row])
         for idx, row in zip(formatted.index, formatted.values)])
    align = "r" + "c" * len(formatted.columns)
    latex = rf"""\begin{{array}}{{{align}}}
{columns} \\
\hline
{rows}
\end{{array}}"""
    if title:
        latex = rf"\textbf{{{title}}}" + r"\\" + "\n" + latex
    display(Math(latex))
```


## Economic environment

We begin with a representative consumer who solves

```{math}
:label: hs82-problem

\max E_0 \sum_{t=0}^{\infty} \beta^t u(C_t)
```

The consumer faces an intertemporal budget constraint with portfolio choice over traded assets.

Let's specialize to CRRA preferences:

```{math}
:label: hs82-crra

u(C_t) = \frac{C_t^{1-\gamma}}{1-\gamma}
```

and then we obtain the first-order condition for any traded gross return $R_{t+1}^i$:

```{math}
:label: hs82-euler

E_t\!\left[\beta \left(\frac{C_{t+1}}{C_t}\right)^{-\gamma} R_{t+1}^i\right] = 1.
```

The **stochastic discount factor** is $M_{t+1}(\theta) = \beta (C_{t+1}/C_t)^{-\gamma}$ with parameter vector $\theta = (\gamma, \beta)$.

In this notation the Euler equation becomes $E_t[M_{t+1}(\theta) R_{t+1}^i - 1] = 0$.

## From conditional to unconditional moments

Let $z_t$ denote any vector measurable with respect to the time-$t$ information set.

Multiplying the Euler error by $z_t$ and taking unconditional expectations gives

```{math}
:label: hs82-uncond

E\!\left[\left(M_{t+1}(\theta_0)R_{t+1}^i - 1\right) \otimes z_t\right] = 0.
```

Equation {eq}`hs82-uncond` is our central object because it converts a rational-expectations condition into estimable moment restrictions.

For one return and $p$ lags of instruments, we use

```{math}
:label: hs82-instruments

z_t = \left[1, R_t, g_t, R_{t-1}, g_{t-1}, \ldots, R_{t-p+1}, g_{t-p+1}\right]^\top,
```

where $g_t = C_t / C_{t-1}$ is gross consumption growth.

The code below encodes the orthogonality condition and lagged instrument vector exactly as in equations {eq}`hs82-uncond` and {eq}`hs82-instruments`.

```{code-cell} ipython3
def euler_error(params, exog):
    """
    Compute Euler-equation pricing errors for (γ, β).
    """
    γ, β = params
    gross_return = exog[:, 0]
    gross_cons_growth = exog[:, 1]
    return β * gross_cons_growth ** (-γ) * gross_return - 1.0
```

Next, a helper aligns outcomes and lagged instruments for nonlinear IV-GMM.

```{code-cell} ipython3
def build_gmm_arrays(data, n_lags):
    """
    Build endog, exog, and instruments for nonlinear IV-GMM.
    """
    if n_lags < 1:
        raise ValueError("n_lags must be at least one.")
    if data.shape[0] <= n_lags:
        raise ValueError("Sample size must exceed n_lags.")

    t_obs = data.shape[0]
    exog = data[n_lags:, :]
    endog = np.zeros(exog.shape[0])
    n_obs = t_obs - n_lags
    n_instr = 2 * n_lags + 1

    instruments = np.empty((n_obs, n_instr))
    instruments[:, 0] = 1.0

    for j in range(n_lags):
        left = 2 * j + 1
        right = left + 2
        instruments[:, left:right] = data[n_lags - 1 - j : t_obs - 1 - j, :]

    return endog, exog, instruments
```

## Data

The data pipeline has two parts: simulated Euler-consistent series and observed FRED/CRSP data.

```{code-cell} ipython3
FRED_CODES = {
    "population_16plus": "CNP16OV",
    "cons_nd_real_index": "DNDGRA3M086SBEA",
    "cons_nd_price_index": "DNDGRG3M086SBEA",
}

def to_month_end(index):
    """
    Convert a date index to month-end timestamps.
    """
    return pd.PeriodIndex(pd.DatetimeIndex(index), freq="M").to_timestamp("M")
```

Let's now define the simulation block that produces synthetic return-growth pairs satisfying the model by construction.

```{code-cell} ipython3
@njit
def _ar1_simulate(mu_c, phi_c, sigma_c, shocks_c, total_n):
    """
    Simulate AR(1) log consumption growth (JIT-compiled inner loop).
    """
    delta_c = np.empty(total_n)
    delta_c[0] = mu_c
    for t in range(1, total_n):
        delta_c[t] = mu_c * (1.0 - phi_c) + phi_c * delta_c[t - 1] + sigma_c * shocks_c[t]
    return delta_c


def simulate_euler_sample(
    n_obs,
    γ_true=0.8,
    β_true=0.993,
    seed=1234,
):
    """
    Simulate [gross real return, gross consumption growth] from an Euler-consistent DGP.
    """
    rng = np.random.default_rng(seed)
    mu_c = 0.0015
    sigma_c = 0.006
    phi_c = 0.4
    sigma_eta = 0.25
    burn_in = 200

    total_n = n_obs + burn_in
    shocks_c = rng.standard_normal(total_n)
    delta_c = _ar1_simulate(mu_c, phi_c, sigma_c, shocks_c, total_n)

    cons_growth = np.exp(delta_c[burn_in:])
    sdf = β_true * cons_growth ** (-γ_true)

    eta = sigma_eta * rng.standard_normal(n_obs)
    eta = np.clip(eta, -0.95, None)
    gross_return = (1.0 + eta) / sdf
    gross_return = np.maximum(gross_return, 1e-6)

    return np.column_stack([gross_return, cons_growth])
```

Next, a loader merges FRED consumption series with stock-return data from a CSV file.

```{code-cell} ipython3
def load_hs_monthly_data(
    csv_path="stock_prices_aggregate_return_full.csv",
    start="1984-01-01",
    end="2019-12-01",
):
    """
    Build monthly gross real return and gross consumption-growth series.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    start_ts = pd.Timestamp(start).to_period("M").to_timestamp("M")
    end_ts = pd.Timestamp(end).to_period("M").to_timestamp("M")

    fred = web.DataReader(list(FRED_CODES.values()), "fred", start_ts, end_ts)
    fred = fred.rename(columns={v: k for k, v in FRED_CODES.items()})
    fred.index = to_month_end(fred.index)
    fred["consumption_per_capita"] = (
        fred["cons_nd_real_index"] / fred["population_16plus"]
    )
    fred["gross_cons_growth"] = (
        fred["consumption_per_capita"] / fred["consumption_per_capita"].shift(1)
    )
    fred["gross_inflation_nd"] = (
        fred["cons_nd_price_index"] / fred["cons_nd_price_index"].shift(1)
    )

    stocks = pd.read_csv(csv_path)
    stocks.columns = [col.strip().lower() for col in stocks.columns]

    date_col = "date" if "date" in stocks.columns else stocks.columns[0]
    ret_col = "vwretd" if "vwretd" in stocks.columns else stocks.columns[1]

    stocks[date_col] = pd.to_datetime(stocks[date_col].astype(str), format="%Y%m%d")
    stocks = stocks.set_index(date_col).sort_index()
    stocks.index = to_month_end(stocks.index)
    stocks["gross_nom_return"] = 1.0 + stocks[ret_col]

    out = fred.join(stocks[["gross_nom_return"]], how="inner")
    out["gross_real_return"] = out["gross_nom_return"] / out["gross_inflation_nd"]
    out = out.loc[start_ts:end_ts].dropna()

    required_cols = [
        "gross_real_return",
        "gross_cons_growth",
        "gross_inflation_nd",
        "consumption_per_capita",
    ]
    return out[required_cols].copy()
```

A thin wrapper then packages the merged frame into the exact array used by our estimators.

```{code-cell} ipython3
def get_estimation_data(
    csv_path="stock_prices_aggregate_return_full.csv",
    start="1984-01-01",
    end="2019-12-01",
):
    """
    Return (dataframe, array, source_label) using observed data.
    """
    frame = load_hs_monthly_data(csv_path=csv_path, start=start, end=end)
    data = frame[["gross_real_return", "gross_cons_growth"]].to_numpy()
    source = "FRED + stock CSV"
    return frame, data, source
```

## GMM criterion and asymptotic theory

Define **moment vectors** $m_t(\theta)$ and sample moments

```{math}
:label: hs82-sample-moments

g_T(\theta) = \frac{1}{T} \sum_{t=1}^T m_t(\theta).
```

Estimation proceeds by minimizing

```{math}
:label: hs82-criterion

\hat\theta = \arg\min_\theta g_T(\theta)^\top W_T g_T(\theta)
```

for a positive-definite weighting matrix $W_T$.

Under standard regularity conditions and identification, we have

```{math}
:label: hs82-asymptotic

\sqrt{T}(\hat\theta-\theta_0) \Rightarrow N\!\left(0, (D^\top S^{-1}D)^{-1}\right),
```

where $D = E[\partial m_t(\theta_0)/\partial\theta^\top]$ and $S$ is the long-run covariance matrix of $m_t(\theta_0)$.

Efficiency requires setting $W_T \to S^{-1}$, which motivates two-step or iterated GMM.

If we have more moments than parameters, the minimized criterion implies Hansen's $J$ test:

```{math}
:label: hs82-jtest

J_T = T\, g_T(\hat\theta)^\top \hat S^{-1} g_T(\hat\theta) \Rightarrow \chi^2_{q-k},
```

where $q$ is the number of moments and $k$ is the number of free parameters.

## HAC and lagged instruments

When we include multiple lags as instruments, $m_t(\theta_0)$ typically has serial correlation even if Euler errors are conditionally mean zero, so $\hat S$ must be heteroskedasticity and autocorrelation consistent (HAC).

The implementation below uses a Newey-West kernel with bandwidth equal to the lag length.

```{code-cell} ipython3
def estimate_gmm(
    data,
    n_lags,
    start_params=None,
    use_hac=True,
    maxiter=2,
):
    """
    Estimate Euler-equation parameters with nonlinear IV-GMM.
    """
    if start_params is None:
        start_params = np.array([1.0, 0.99])

    endog, exog, instruments = build_gmm_arrays(data, n_lags)
    model = gmm.NonlinearIVGMM(endog, exog, instruments, euler_error)

    if use_hac:
        result = model.fit(
            start_params=start_params,
            maxiter=maxiter,
            optim_method="bfgs",
            optim_args={"disp": False},
            weights_method="hac",
            wargs={"maxlag": n_lags},
        )
    else:
        result = model.fit(
            start_params=start_params,
            maxiter=maxiter,
            optim_method="bfgs",
            optim_args={"disp": False},
        )

    return result
```

Because lag choice affects overidentification and finite-sample behavior, it is worth reporting estimates across several lag lengths.

```{code-cell} ipython3
def run_gmm_by_lag(
    data,
    lags=(2, 4, 6),
    use_hac=True,
):
    """
    Estimate GMM models by lag length and return a summary table.
    """
    rows = []
    results = {}

    for lag in lags:
        res = estimate_gmm(data, n_lags=lag, use_hac=use_hac)
        results[lag] = res
        j_stat, j_pval, _ = res.jtest()
        rows.append(
            {
                "n_lags": lag,
                "γ_hat": res.params[0],
                "se_γ": res.bse[0],
                "β_hat": res.params[1],
                "se_β": res.bse[1],
                "j_stat": j_stat,
                "j_pval": j_pval,
                "n_obs": int(res.nobs),
            }
        )

    table = pd.DataFrame(rows).set_index("n_lags")
    return table, results
```

Since weak identification often appears as flat criterion regions, it is also useful to compute the objective over a parameter grid.

```{code-cell} ipython3
def gmm_objective_surface(
    data,
    n_lags=2,
    γ_grid=None,
    β_grid=None,
):
    """
    Compute identity-weighted GMM objective on a parameter grid.
    """
    _, exog, instruments = build_gmm_arrays(data, n_lags)

    if γ_grid is None:
        γ_grid = np.linspace(-1.0, 8.0, 70)
    if β_grid is None:
        β_grid = np.linspace(0.96, 1.02, 70)

    objective = np.empty((len(β_grid), len(γ_grid)))

    for i, β_val in enumerate(β_grid):
        for j, γ_val in enumerate(γ_grid):
            err = euler_error(np.array([γ_val, β_val]), exog)
            moments = (err[:, None] * instruments).mean(axis=0)
            objective[i, j] = moments @ moments

    return γ_grid, β_grid, objective
```

## Lognormal transformation

The moment restrictions remain the economic core in {cite}`hansen1983stochastic`, and conditional lognormality then lets us construct a full likelihood around the same Euler equation.

Define $X_{t+1}=\log(c_{t+1}/c_t)$, $R_{i,t+1}=\log r_{i,t+1}$, and

```{math}
:label: hs83-u-def

U_{i,t+1}=\beta\left(\frac{c_{t+1}}{c_t}\right)^{\alpha}r_{i,t+1}.
```

Then $\log U_{i,t+1}=\alpha X_{t+1}+R_{i,t+1}+\log\beta$.

Under our joint lognormality assumption, conditional normality implies

```{math}
:label: hs83-lognormal-identity

E_t[U_{i,t+1}] = \exp\left(E_t[\log U_{i,t+1}] + \tfrac{1}{2}\operatorname{Var}_t(\log U_{i,t+1})\right).
```

Imposing $E_t[U_{i,t+1}]=1$ yields

```{math}
:label: hs83-v-it

V_{i,t+1}=\alpha X_{t+1}+R_{i,t+1}+\log\beta+\frac{\sigma_i^2}{2},
\quad E_t[V_{i,t+1}]=0,
```

where $\sigma_i^2=\operatorname{Var}_t(\log U_{i,t+1})$ is constant under conditional homoskedasticity.

Equation {eq}`hs83-v-it` implies the conditional mean restriction

```{math}
:label: hs83-cond-mean

E_t[R_{i,t+1}] = -\alpha E_t[X_{t+1}] - \log\beta - \frac{\sigma_i^2}{2}.
```

Let's implement the log transformation in equations {eq}`hs83-u-def` to {eq}`hs83-cond-mean` with the helper below.

```{code-cell} ipython3
def to_mle_array(data):
    """
    Convert [gross return, gross consumption growth] to [log consumption growth, log return].
    """
    valid = (data[:, 0] > 0.0) & (data[:, 1] > 0.0)
    return np.column_stack([np.log(data[valid, 1]), np.log(data[valid, 0])])
```

## Triangular restricted system

In the single-return case, write $Y_t=(X_t,R_t)^\top$.

The predictable component of $X_t$ is parameterized as

```{math}
:label: hs83-x-forecast

E(X_t\mid\psi_{t-1})=a(L)^\top Y_{t-1}+\mu_x,
```

where $a(L)$ is a finite lag polynomial in past $(X,R)$.

Combining {eq}`hs83-cond-mean` and {eq}`hs83-x-forecast` gives the triangular system

```{math}
:label: hs83-triangular

A_0Y_t=A_1(L)Y_{t-1}+\mu+V_t,
```

with

```{math}
:label: hs83-a0a1

A_0=\begin{bmatrix}1&0\\\alpha&1\end{bmatrix},
\quad
A_1(L)=\begin{bmatrix}a(L)^\top\\0\end{bmatrix},
\quad
\mu=\begin{bmatrix}\mu_x\\-\log\beta-\sigma_R^2/2\end{bmatrix}.
```

Because $A_0$ is unit lower triangular, the Jacobian of the transformation from innovations to observables is one.

This property makes the Gaussian likelihood straightforward once we compute residuals from {eq}`hs83-triangular`.

## Likelihood ingredients

To build the likelihood, we need lagged data stacks and a mapping from parameters to the matrices in {eq}`hs83-a0a1`.

```{code-cell} ipython3
def build_lagged_data(data, n_lags):
    """
    Build Y_t and lag stacks [Y_{t-1}, ..., Y_{t-p}] for bivariate data.
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must be T x 2.")
    if data.shape[0] <= n_lags:
        raise ValueError("Sample size must exceed n_lags.")

    t_obs = data.shape[0]
    n_obs = t_obs - n_lags
    y_t = data[n_lags:, :]
    y_lag = np.empty((n_obs, 2 * n_lags))

    for lag in range(1, n_lags + 1):
        y_lag[:, 2 * (lag - 1) : 2 * lag] = data[n_lags - lag : t_obs - lag, :]

    return y_t, y_lag
```

Next, we validate and unpack the parameter vector while enforcing feasibility conditions.

```{code-cell} ipython3
def unpack_parameters(params, n_lags):
    """
    Validate and unpack parameter vector.
    """
    if len(params) != 6 + 2 * n_lags:
        return None

    α, β, sigma_x, sigma_r, cov_xr, mu_x = params[:6]
    a_lags = params[6:]

    tol = 1e-8
    if not (α < 1.0):
        return None
    if not (tol < β < 1.0 - tol):
        return None
    if not (sigma_x > tol and sigma_r > tol):
        return None

    sigma = np.array(
        [
            [sigma_x ** 2, cov_xr],
            [cov_xr, sigma_r ** 2],
        ]
    )

    try:
        if np.linalg.det(sigma) <= tol:
            return None
        cholesky(sigma, lower=True)
    except (LinAlgError, ValueError):
        return None

    return {
        "α": np.array(α),
        "β": np.array(β),
        "sigma_x": np.array(sigma_x),
        "sigma_r": np.array(sigma_r),
        "cov_xr": np.array(cov_xr),
        "mu_x": np.array(mu_x),
        "a_lags": a_lags,
    }
```

The next step maps parameters and lagged data into triangular-system residuals.

```{code-cell} ipython3
def triangular_residuals(
    params,
    y_t,
    y_lag,
    n_lags,
):
    """
    Compute V_t implied by A0 Y_t - A1 Y_{t-1} - mu.
    """
    parsed = unpack_parameters(params, n_lags)
    if parsed is None:
        return None

    α = float(parsed["α"])
    β = float(parsed["β"])
    sigma_r = float(parsed["sigma_r"])
    mu_x = float(parsed["mu_x"])
    a_lags = np.asarray(parsed["a_lags"])

    a0 = np.array([[1.0, 0.0], [α, 1.0]])
    a1 = np.zeros((2, 2 * n_lags))
    a1[0, :] = a_lags
    mu = np.array([mu_x, -np.log(β) - 0.5 * sigma_r ** 2])

    resid = y_t @ a0.T - y_lag @ a1.T - mu[None, :]
    if np.any(np.abs(resid) > 1e10):
        return None
    return resid
```

The same triangular structure also gives us a direct simulation recursion for controlled experiments.

```{code-cell} ipython3
def simulate_triangular_var(
    params,
    n_obs,
    n_lags,
    burn_in=200,
    seed=None,
):
    """
    Simulate [log consumption growth, log return] from the triangular model.
    """
    if seed is not None:
        np.random.seed(seed)

    if len(params) != 6 + 2 * n_lags:
        raise ValueError("Parameter vector length must be 6 + 2 * n_lags.")

    α, β, sigma_x, sigma_r, cov_xr, mu_x = params[:6]
    a_lags = params[6:]

    sigma = np.array(
        [
            [sigma_x ** 2, cov_xr],
            [cov_xr, sigma_r ** 2],
        ]
    )

    eigvals = np.linalg.eigvals(sigma)
    if np.min(eigvals) <= 0.0:
        sigma += np.eye(2) * 1e-6

    a0 = np.array([[1.0, 0.0], [α, 1.0]])
    a1 = np.zeros((2, 2 * n_lags))
    a1[0, :] = a_lags
    mu = np.array([mu_x, -np.log(β) - 0.5 * sigma_r ** 2])

    total_n = n_obs + burn_in
    y = np.zeros((total_n, 2))

    for t in range(n_lags, total_n):
        lag_stack = []
        for lag in range(1, n_lags + 1):
            lag_stack.append(y[t - lag, :])
        lag_vec = np.concatenate(lag_stack)
        shock = np.random.multivariate_normal(np.zeros(2), sigma)
        y[t, :] = np.linalg.solve(a0, a1 @ lag_vec + mu + shock)

    return y[burn_in:, :]
```

Next, we encode the Gaussian log-likelihood implied by the residual covariance matrix.

```{code-cell} ipython3
def log_likelihood_mle(
    params,
    y_t,
    y_lag,
    n_lags,
    include_const=True,
):
    """
    Evaluate Gaussian log-likelihood for the restricted triangular system.
    """
    parsed = unpack_parameters(params, n_lags)
    if parsed is None:
        return -np.inf

    resid = triangular_residuals(params, y_t, y_lag, n_lags)
    if resid is None:
        return -np.inf

    sigma_x = float(parsed["sigma_x"])
    sigma_r = float(parsed["sigma_r"])
    cov_xr = float(parsed["cov_xr"])

    sigma = np.array(
        [
            [sigma_x ** 2, cov_xr],
            [cov_xr, sigma_r ** 2],
        ]
    )

    try:
        chol = cholesky(sigma, lower=True)
        log_det = 2.0 * np.sum(np.log(np.diag(chol) + 1e-16))
        std_resid = solve_triangular(chol, resid.T, lower=True).T
        quad_form = np.sum(std_resid ** 2)
    except (LinAlgError, ValueError):
        return -np.inf

    sample_size = y_t.shape[0]
    ll = -0.5 * sample_size * log_det - 0.5 * quad_form
    if include_const:
        ll -= sample_size * np.log(2.0 * np.pi)

    if np.isnan(ll) or np.isinf(ll):
        return -np.inf
    return float(ll)
```

To optimize numerically, let's wrap the log-likelihood as a minimization objective.

```{code-cell} ipython3
def negative_log_likelihood(
    params,
    y_t,
    y_lag,
    n_lags,
):
    """
    Return negative log-likelihood for minimization.
    """
    ll = log_likelihood_mle(params, y_t, y_lag, n_lags, include_const=False)
    if np.isfinite(ll):
        return -ll
    return 1e20
```

Let's set parameter bounds before generating multiple data-driven starting values for multi-start optimization.

```{code-cell} ipython3
def parameter_bounds(n_lags):
    """
    Bounds for optimization.
    """
    bounds = [
        (-5.0, 1.0),
        (1e-8, 1.0 - 1e-8),
        (1e-8, None),
        (1e-8, None),
        (None, None),
        (None, None),
    ]
    bounds += [(-0.99, 0.99)] * (2 * n_lags)
    return bounds
```

Several perturbed starting vectors help local solvers escape poor initializations.

```{code-cell} ipython3
def starting_values(data, n_lags, n_starts=8):
    """
    Generate multiple data-driven starting values.
    """
    rng = np.random.default_rng(123)
    starts = []
    n_params = 6 + 2 * n_lags

    base = np.zeros(n_params)
    base[0] = -0.5
    base[1] = 0.99
    base[2] = max(float(np.std(data[:, 0])), 1e-3)
    base[3] = max(float(np.std(data[:, 1])), 1e-3)
    base[4] = float(np.cov(data.T)[0, 1])
    base[5] = float(np.mean(data[:, 0]))
    base[6:] = 0.1
    starts.append(base.copy())

    for _ in range(n_starts - 1):
        trial = base.copy()
        trial[:2] += rng.normal(0.0, 0.25, 2)
        trial[2:6] *= 1.0 + rng.normal(0.0, 0.1, 4)
        trial[6:] += rng.normal(0.0, 0.05, 2 * n_lags)
        trial[1] = np.clip(trial[1], 1e-6, 1.0 - 1e-6)
        trial[2] = max(trial[2], 1e-6)
        trial[3] = max(trial[3], 1e-6)
        starts.append(trial)

    return starts
```

Standard errors come from a numerical Hessian of the negative log-likelihood.

```{code-cell} ipython3
def numerical_standard_errors(
    params,
    y_t,
    y_lag,
    n_lags,
    step=1e-5,
):
    """
    Compute standard errors from a numerical Hessian.
    """
    n = len(params)
    hess = np.zeros((n, n))
    f0 = negative_log_likelihood(params, y_t, y_lag, n_lags)

    if not np.isfinite(f0):
        return np.full(n, np.nan)

    for i in range(n):
        hi = step * max(1.0, abs(params[i]))
        for j in range(i, n):
            hj = step * max(1.0, abs(params[j]))
            if i == j:
                p_plus = params.copy()
                p_minus = params.copy()
                p_plus[i] += hi
                p_minus[i] -= hi
                f_plus = negative_log_likelihood(p_plus, y_t, y_lag, n_lags)
                f_minus = negative_log_likelihood(p_minus, y_t, y_lag, n_lags)
                hess[i, i] = (f_plus - 2.0 * f0 + f_minus) / (hi ** 2)
            else:
                p_pp = params.copy()
                p_pm = params.copy()
                p_mp = params.copy()
                p_mm = params.copy()
                p_pp[i] += hi
                p_pp[j] += hj
                p_pm[i] += hi
                p_pm[j] -= hj
                p_mp[i] -= hi
                p_mp[j] += hj
                p_mm[i] -= hi
                p_mm[j] -= hj

                f_pp = negative_log_likelihood(p_pp, y_t, y_lag, n_lags)
                f_pm = negative_log_likelihood(p_pm, y_t, y_lag, n_lags)
                f_mp = negative_log_likelihood(p_mp, y_t, y_lag, n_lags)
                f_mm = negative_log_likelihood(p_mm, y_t, y_lag, n_lags)

                h_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * hi * hj)
                hess[i, j] = h_ij
                hess[j, i] = h_ij

    try:
        eigvals = np.linalg.eigvalsh(hess)
        if np.min(eigvals) < 1e-10:
            hess += np.eye(n) * 1e-6
        cov = np.linalg.inv(hess)
    except (LinAlgError, ValueError):
        cov = np.linalg.pinv(hess)

    diagonal = np.diag(cov)
    return np.sqrt(np.where(diagonal > 0.0, diagonal, np.nan))
```

Let's combine the pieces in a multi-start MLE estimator that returns parameters, fit criteria, and residuals.

```{code-cell} ipython3
def estimate_mle(data, n_lags, verbose=False):
    """
    Estimate the restricted triangular model by multi-start local optimization.
    """
    y_t, y_lag = build_lagged_data(data, n_lags)
    bounds = parameter_bounds(n_lags)
    starts = starting_values(data, n_lags)

    best_result = None
    best_ll = -np.inf

    for i, x0 in enumerate(starts):
        try:
            result = minimize(
                negative_log_likelihood,
                x0=x0,
                args=(y_t, y_lag, n_lags),
                method="L-BFGS-B",
                bounds=bounds,
            )
        except Exception:
            continue

        if result.success and np.isfinite(result.fun):
            ll_val = log_likelihood_mle(result.x, y_t, y_lag, n_lags)
            if ll_val > best_ll:
                best_ll = ll_val
                best_result = result
                if verbose:
                    print(f"start={i}, loglike={ll_val:.2f}")

    n_params = 6 + 2 * n_lags

    if best_result is None:
        return {
            "params": np.full(n_params, np.nan),
            "se": np.full(n_params, np.nan),
            "loglike": -np.inf,
            "aic": np.inf,
            "bic": np.inf,
            "converged": False,
            "residuals": None,
        }

    params = best_result.x
    se = numerical_standard_errors(params, y_t, y_lag, n_lags)
    resid = triangular_residuals(params, y_t, y_lag, n_lags)
    ll_val = log_likelihood_mle(params, y_t, y_lag, n_lags)

    return {
        "params": params,
        "se": se,
        "loglike": ll_val,
        "aic": -2.0 * ll_val + 2.0 * n_params,
        "bic": -2.0 * ll_val + n_params * np.log(y_t.shape[0]),
        "converged": bool(best_result.success),
        "residuals": resid,
    }
```

Residual diagnostics below summarize normality and serial-correlation checks.

```{code-cell} ipython3
def residual_diagnostics(resid):
    """
    Compute basic residual diagnostics.
    """
    out = {}

    for i, label in enumerate(["consumption", "return"]):
        jb_stat, jb_pval = stats.jarque_bera(resid[:, i])
        out[f"{label}_jb_stat"] = float(jb_stat)
        out[f"{label}_jb_pval"] = float(jb_pval)
        out[f"{label}_dw"] = float(durbin_watson(resid[:, i]))

    return out
```

Finally, a lag-loop wrapper runs MLE across several instrument lengths.

```{code-cell} ipython3
def run_mle_by_lag(
    data,
    lags=(2, 4, 6),
    verbose=False,
):
    """
    Estimate restricted MLE models by lag length.
    """
    rows = []
    fits = {}

    for lag in lags:
        fit = estimate_mle(data, n_lags=lag, verbose=verbose)
        fits[lag] = fit

        rows.append(
            {
                "n_lags": lag,
                "α_hat": fit["params"][0],
                "se_α": fit["se"][0],
                "β_hat": fit["params"][1],
                "se_β": fit["se"][1],
                "loglike": fit["loglike"],
                "aic": fit["aic"],
                "bic": fit["bic"],
            }
        )

    table = pd.DataFrame(rows).set_index("n_lags")
    return table, fits
```

## Identification and tests

In this system, α links predictable variation in returns to predictable variation in consumption growth through cross-equation restrictions.

Here, β shifts the return-equation intercept through $-\log\beta$.

Likelihood-ratio tests compare the restricted system against less restricted alternatives.

The efficiency gain comes at the cost of sensitivity to violations of lognormality and conditional homoskedasticity.

For nested specifications, let's use

```{math}
:label: hs83-lr-test

LR = 2(\ell_u - \ell_r) \Rightarrow \chi^2_d,
```

where $\ell_u$ and $\ell_r$ denote unrestricted and restricted log-likelihood values.

This test applies whenever both fitted models converge.

```{code-cell} ipython3
def likelihood_ratio_test(
    fit_restricted,
    fit_unrestricted,
    df_diff,
):
    """
    Compare nested specifications with an LR test.
    """
    if not (fit_restricted["converged"] and fit_unrestricted["converged"]):
        return {"lr_stat": np.nan, "p_value": np.nan}

    lr_stat = 2.0 * (fit_unrestricted["loglike"] - fit_restricted["loglike"])
    p_value = 1.0 - stats.chi2.cdf(lr_stat, df=df_diff)
    return {"lr_stat": float(lr_stat), "p_value": float(p_value)}
```

## Simulation and finite-sample behavior

As a sanity check, let's first apply GMM to simulated data with known true parameters.

```{code-cell} ipython3
γ_true = 2.0
β_true = 0.995
sim_data = simulate_euler_sample(
    n_obs=700,
    γ_true=γ_true,
    β_true=β_true,
    seed=42,
)

print(f"Simulation sample size: {sim_data.shape[0]}")
print(f"True γ: {γ_true:.3f}")
print(f"True β: {β_true:.3f}")
print(f"Mean net return: {(sim_data[:, 0].mean() - 1.0) * 100:.3f}%")
print(f"Mean net consumption growth: {(sim_data[:, 1].mean() - 1.0) * 100:.3f}%")
```

The printed lines confirm sample size and restate the true parameters we want to recover.

The two mean growth figures verify that simulated moments look economically plausible before estimation.

Let's now estimate GMM across lag lengths and display a compact results table.

```{code-cell} ipython3
sim_table, sim_results = run_gmm_by_lag(sim_data, lags=(2, 4, 6), use_hac=True)
sim_pretty = sim_table.rename(columns={
    "γ_hat": r"\hat{\gamma}", "se_γ": r"\mathrm{se}(\hat{\gamma})",
    "β_hat": r"\hat{\beta}", "se_β": r"\mathrm{se}(\hat{\beta})",
    "j_stat": "J", "j_pval": "p(J)", "n_obs": "T",
})
display_table(sim_pretty, title="GMM Simulation Results", fmt={
    r"\hat{\gamma}": "{:.4f}", r"\mathrm{se}(\hat{\gamma})": "{:.4f}",
    r"\hat{\beta}": "{:.4f}", r"\mathrm{se}(\hat{\beta})": "{:.4f}",
    "J": "{:.3f}", "p(J)": "{:.3f}", "T": "{:.0f}",
})
```

From this table, we see that GMM recovers the true γ and β across lag specifications.

Small $J$ statistics with large $p$-values indicate non-rejection of the simulated moment conditions.

Let's run a Monte Carlo exercise to visualize the sampling distribution of the estimates and the $J$ statistic.

```{code-cell} ipython3
n_rep = 500
estimates = []
j_stats = []

for rep in range(n_rep):
    rep_data = simulate_euler_sample(
        n_obs=900,
        γ_true=γ_true,
        β_true=β_true,
        seed=rep,
    )
    rep_res = estimate_gmm(rep_data, n_lags=2, use_hac=True, maxiter=2)
    estimates.append(rep_res.params)
    j_stats.append(rep_res.jval)

estimates = np.asarray(estimates)
j_stats = np.asarray(j_stats)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(estimates[:, 0], bins=20, edgecolor="white")
axes[0].axvline(γ_true, color="red", ls="--", lw=2)
axes[0].set_xlabel(r"$\hat{\gamma}$")
axes[1].hist(estimates[:, 1], bins=20, edgecolor="white")
axes[1].axvline(β_true, color="red", ls="--", lw=2)
axes[1].set_xlabel(r"$\hat{\beta}$")

df_j = 2 * 2 + 1 - 2
axes[2].hist(j_stats, bins=20, density=True, edgecolor="white")
grid = np.linspace(0.0, max(j_stats.max(), 1.0), 200)
axes[2].plot(grid, stats.chi2.pdf(grid, df_j), "r-", lw=2)
axes[2].set_xlabel("j-statistic")
plt.tight_layout()
plt.show()
```

The histograms show $\hat\gamma$ and $\hat\beta$ centered near their true values, marked by red dashed lines.

The $J$ histogram tracks the overlaid $\chi^2$ density reasonably well, supporting the asymptotic approximation in this sample size.

## Empirical estimation

Let's load the empirical sample and print its key moments before fitting models.

```{code-cell} ipython3
emp_frame, emp_data, source = get_estimation_data(
    csv_path="stock_prices_aggregate_return_full.csv",
    start="1984-01-01",
    end="2019-12-01",
)

print(f"Data source: {source}")
print(f"Sample size: {len(emp_data)}")
print(f"Mean net real return: {(emp_data[:, 0].mean() - 1.0) * 100:.3f}%")
print(f"Std net real return: {emp_data[:, 0].std() * 100:.3f}%")
print(f"Mean net consumption growth: {(emp_data[:, 1].mean() - 1.0) * 100:.3f}%")
print(f"Std net consumption growth: {emp_data[:, 1].std() * 100:.3f}%")
print(f"Correlation: {np.corrcoef(emp_data[:, 0], emp_data[:, 1])[0, 1]:.4f}")
```

These summary lines tell us the sample span, return and consumption volatility, and the contemporaneous correlation we ask the Euler model to explain.

Next, let's estimate GMM with HAC and HC0 covariance choices and compare the resulting tables.

```{code-cell} ipython3
gmm_hac_table, gmm_hac_results = run_gmm_by_lag(emp_data, lags=(2, 4, 6), use_hac=True)
gmm_hc0_table, _ = run_gmm_by_lag(emp_data, lags=(2, 4, 6), use_hac=False)

gmm_hac_pretty = gmm_hac_table.rename(columns={
    "γ_hat": r"\hat{\gamma}", "se_γ": r"\mathrm{se}(\hat{\gamma})",
    "β_hat": r"\hat{\beta}", "se_β": r"\mathrm{se}(\hat{\beta})",
    "j_stat": "J", "j_pval": "p(J)", "n_obs": "T",
})
gmm_hc0_pretty = gmm_hc0_table[["se_γ", "se_β", "j_stat", "j_pval"]].rename(columns={
    "se_γ": r"\mathrm{se}(\hat{\gamma})", "se_β": r"\mathrm{se}(\hat{\beta})",
    "j_stat": "J", "j_pval": "p(J)",
})

gmm_fmt = {
    r"\hat{\gamma}": "{:.4f}", r"\mathrm{se}(\hat{\gamma})": "{:.4f}",
    r"\hat{\beta}": "{:.4f}", r"\mathrm{se}(\hat{\beta})": "{:.4f}",
    "J": "{:.3f}", "p(J)": "{:.3f}", "T": "{:.0f}",
}
display_table(gmm_hac_pretty, title="GMM Estimates (HAC Covariance)", fmt=gmm_fmt)
display_table(gmm_hc0_pretty, title="GMM Estimates (HC0 Covariance)", fmt={
    r"\mathrm{se}(\hat{\gamma})": "{:.4f}", r"\mathrm{se}(\hat{\beta})": "{:.4f}",
    "J": "{:.3f}", "p(J)": "{:.3f}",
})
```

Low and imprecise γ estimates typically appear here, mirroring the equity-premium puzzle.

The $J$-test $p$-values serve as a direct check on whether overidentifying restrictions are rejected at standard levels.

By comparing HAC and HC0 columns, we can see how much serial-correlation correction changes inference.

Let's inspect pricing errors and their autocorrelation structure to diagnose fit beyond summary statistics.

```{code-cell} ipython3
lag_diag = 2
res_diag = gmm_hac_results[lag_diag]
_, exog_diag, _ = build_gmm_arrays(emp_data, n_lags=lag_diag)
errors = euler_error(res_diag.params, exog_diag)
acf_vals = acf(errors, nlags=12, fft=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(errors, lw=2)
axes[0].axhline(0.0, color="black", lw=2)
axes[0].set_xlabel("observation")
axes[0].set_ylabel("euler error")

axes[1].hist(errors, bins=30, density=True, edgecolor="white")
axes[1].set_xlabel("euler error")
axes[1].set_ylabel("density")

axes[2].bar(range(len(acf_vals)), acf_vals)
bound = 1.96 / np.sqrt(len(errors))
axes[2].axhline(bound, color="red", ls="--", lw=2)
axes[2].axhline(-bound, color="red", ls="--", lw=2)
axes[2].set_xlabel("lag")
axes[2].set_ylabel("acf")
plt.tight_layout()
plt.show()

print(f"Mean error: {errors.mean():.6f}")
print(f"Std error:  {errors.std():.6f}")
```

The time-series panel helps spot pricing-error spikes, while the histogram and ACF panel reveal distribution shape and persistence.

If low-lag bars cross the red bounds, we interpret that as remaining predictable structure that our instrument set has not absorbed.

The printed mean and standard deviation summarize Euler-error magnitude and dispersion in the fitted sample.

To make identification geometry visible, let's plot the GMM criterion over a $(\gamma,\beta)$ grid.

```{code-cell} ipython3
γ_grid, β_grid, objective = gmm_objective_surface(emp_data, n_lags=2)
log_obj = np.log10(objective + 1e-12)

fig, ax = plt.subplots()
contours = ax.contourf(γ_grid, β_grid, log_obj, levels=30, cmap="viridis")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$\beta$")
ax.plot(gmm_hac_results[2].params[0], gmm_hac_results[2].params[1], "r*", ms=12, lw=2)
plt.colorbar(contours, ax=ax)
plt.tight_layout()
plt.show()
```

The contour figure maps the $(\gamma,\beta)$ criterion surface, and the red star marks the estimated optimum.

An elongated valley signals weak separate identification of γ and β even if a combination is pinned down.

## Likelihood estimation

Let's repeat the simulation-then-estimation pattern to verify that MLE recovers known parameters from the triangular representation.

```{code-cell} ipython3
true_params = np.array([
    -0.40,
    0.993,
    0.005,
    0.040,
    0.0001,
    0.002,
    0.20,
    0.10,
])

sim_mle_data = simulate_triangular_var(
    params=true_params,
    n_obs=500,
    n_lags=1,
    burn_in=200,
    seed=7,
)

fit_sim = estimate_mle(sim_mle_data, n_lags=1, verbose=False)
print(f"Converged: {fit_sim['converged']}")
print(r"hat_α: " + f"{fit_sim['params'][0]:.4f}")
print(r"hat_β: " + f"{fit_sim['params'][1]:.4f}")
print(f"loglike:   {fit_sim['loglike']:.2f}")
```

The printed convergence flag and parameter estimates confirm that the optimizer recovers $\hat\alpha$ and $\hat\beta$ near their true values.

Now let's apply MLE to the historical sample so we can compare it directly with the GMM results.

```{code-cell} ipython3
emp_log_data = to_mle_array(emp_data)
mle_table, mle_fits = run_mle_by_lag(emp_log_data, lags=(2, 4, 6), verbose=False)
mle_pretty = mle_table.rename(columns={
    "α_hat": r"\hat{\alpha}", "se_α": r"\mathrm{se}(\hat{\alpha})",
    "β_hat": r"\hat{\beta}", "se_β": r"\mathrm{se}(\hat{\beta})",
    "loglike": "logL", "aic": "AIC", "bic": "BIC",
})
display_table(mle_pretty, title="Likelihood Estimates by Lag Length", fmt={
    r"\hat{\alpha}": "{:.4f}", r"\mathrm{se}(\hat{\alpha})": "{:.4f}",
    r"\hat{\beta}": "{:.4f}", r"\mathrm{se}(\hat{\beta})": "{:.4f}",
    "logL": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}",
})
```

Compared with GMM, these MLE preference estimates come from stronger distributional assumptions.

AIC and BIC capture the fit-versus-complexity tradeoff across lag choices.

Let's compute explicit likelihood-ratio comparisons between nested lag specifications.

```{code-cell} ipython3
lr_2_to_4 = likelihood_ratio_test(
    fit_restricted=mle_fits[2],
    fit_unrestricted=mle_fits[4],
    df_diff=(6 + 2 * 4) - (6 + 2 * 2),
)
lr_4_to_6 = likelihood_ratio_test(
    fit_restricted=mle_fits[4],
    fit_unrestricted=mle_fits[6],
    df_diff=(6 + 2 * 6) - (6 + 2 * 4),
)

print(f"2 lags vs 4 lags: LR={lr_2_to_4['lr_stat']:.3f}, p={lr_2_to_4['p_value']:.3f}")
print(f"4 lags vs 6 lags: LR={lr_4_to_6['lr_stat']:.3f}, p={lr_4_to_6['p_value']:.3f}")
```

These LR outputs tell us whether adding lags delivers statistically meaningful fit improvements.

When a reported $p$-value is small, we reject the simpler lag specification in favor of the richer one.

Finally, let's inspect residual paths, histograms, and diagnostic statistics for the chosen lag length.

```{code-cell} ipython3
diag_lag = 2
diag_fit = mle_fits[diag_lag]
resid = diag_fit["residuals"]

if diag_fit["converged"] and resid is not None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(resid[:, 0], lw=2)
    axes[0, 0].axhline(0.0, color="black", lw=2)
    axes[0, 0].set_ylabel("consumption residual")
    axes[0, 0].set_xlabel("observation")

    axes[0, 1].plot(resid[:, 1], lw=2)
    axes[0, 1].axhline(0.0, color="black", lw=2)
    axes[0, 1].set_ylabel("return residual")
    axes[0, 1].set_xlabel("observation")

    axes[1, 0].hist(resid[:, 0], bins=30, edgecolor="white")
    axes[1, 0].set_xlabel("consumption residual")
    axes[1, 0].set_ylabel("count")

    axes[1, 1].hist(resid[:, 1], bins=30, edgecolor="white")
    axes[1, 1].set_xlabel("return residual")
    axes[1, 1].set_ylabel("count")
    plt.tight_layout()
    plt.show()

    diag = residual_diagnostics(resid)
    diag_df = pd.DataFrame({
        "JB stat": [diag["consumption_jb_stat"], diag["return_jb_stat"]],
        "JB p-val": [diag["consumption_jb_pval"], diag["return_jb_pval"]],
        "DW": [diag["consumption_dw"], diag["return_dw"]],
    }, index=pd.Index(["consumption", "return"], name="series"))
    display_table(diag_df, title="Residual Diagnostics", fmt={
        "JB stat": "{:.2f}", "JB p-val": "{:.4f}", "DW": "{:.3f}",
    })
```

The residual plots help assess drift, skewness, and tail thickness relative to the Gaussian benchmark.

A small Jarque-Bera p-value rejects normality, while a Durbin-Watson statistic far from 2 flags first-order serial correlation.

Rejection of normality here is unsurprising given the well-known fat tails in financial-return data.

## Summary

The GMM estimator of {cite}`hansen1982generalized` serves as a robust benchmark because it relies on orthogonality conditions and HAC covariance without a fully parametric state process.

The MLE approach of {cite}`hansen1983stochastic` offers higher efficiency but depends on stronger lognormal and triangular assumptions.

Together, these methods give us a coherent workflow in which we benchmark with GMM and then sharpen inference with likelihood methods when assumptions seem defensible.
