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

(hansen_singleton_1983)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Risk Aversion, Asset Returns, and the Equity Premium Puzzle

```{index} single: Asset Pricing; MLE Estimation
```

```{contents} Contents
:depth: 2
```

## Overview

This lecture implements the maximum likelihood estimator of {cite:t}`hansen1983stochastic` and connects its empirical findings to the equity premium puzzle of {cite:t}`MehraPrescott1985`.

{cite:t}`hansen1983stochastic` study a consumption-based asset pricing model in which a representative consumer with CRRA preferences chooses how to allocate wealth across traded assets.

The first-order conditions deliver stochastic Euler equations relating consumption growth, asset returns, and preference parameters.

{cite:t}`hansen1983stochastic` assume that consumption growth and asset returns are *jointly lognormal*.

Under these assumptions, the Euler equation implies a set of restrictions on a linear time-series representation of the logarithms of consumption growth and returns.

Specifically, predictable movements in log returns must be proportional to predictable movements in log consumption growth, with proportionality factor $-\alpha$.

In the notation of {cite:t}`hansen1983stochastic`, utility is $U(c_t)=c_t^\gamma/\gamma$ with $\gamma<1$, so $\alpha=\gamma-1$ and the coefficient of relative risk aversion is $\rho = 1-\gamma = -\alpha$.

This restricted representation takes the form of a triangular VAR that can be estimated by maximum likelihood.

The empirical findings of {cite:t}`hansen1983stochastic` foreshadow what {cite:t}`MehraPrescott1985` would formalize as the **equity premium puzzle**:

1. Point estimates imply relatively low risk aversion ($\rho=-\alpha$ is typically between 0 and 2), too low to explain the large observed gap between stock returns and the risk-free rate.
2. The predictable component of stock returns ($R_R^2$ of 0.02 to 0.06) is tiny relative to total return variation, even when consumption growth itself has forecastable variation.
3. The model fits aggregate value-weighted stock returns reasonably well, but is strongly rejected for Treasury bills (where the restrictions on the risk-free rate cannot be reconciled with consumption data) and for individual stocks.
4. The low estimates of $\rho=-\alpha$ combined with the high observed equity premium imply that CRRA preferences cannot simultaneously match the level of the risk-free rate and the equity premium.

These are precisely the facts that constitute the equity premium puzzle.

The following companion lecture {doc}`hansen_singleton_1982` develops a robust alternative based on GMM that does not require the lognormality assumption.

We cover:

- the lognormal restriction and its implications for the conditional mean of returns
- the triangular-system likelihood and its maximum likelihood estimation
- likelihood ratio tests comparing the restricted model to an unrestricted VAR
- predictability metrics and the $R^2$ restriction
- return-difference tests that bypass consumption measurement
- the connection to the equity premium puzzle and subsequent literature

Relative to the full empirical scope of {cite:t}`hansen1983stochastic`, this notebook makes deliberate simplifications:

- We estimate bivariate systems with one return at a time (value-weighted market proxy or T-bill), not the full multi-asset systems behind Tables 2, 3, and 5.
- We focus on monthly nondurable consumption (`ND`) and do not re-estimate the paper's `NDS` or quarterly specifications.
- We include the return-difference logic and a simulation check, but we do not replicate the paper's multi-stock return-difference rejection table.
- We do not implement the `NLAG = 0` just-identified comparison emphasized in the paper's joint-system discussion.

In addition to what comes with Anaconda, this lecture requires `pandas-datareader`

```{code-cell} ipython3
:tags: [hide-output]

!pip install pandas-datareader
```

```{code-cell} ipython3
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Math
from numba import njit
from pandas_datareader import data as web
from scipy import stats
from scipy.linalg import LinAlgError, cholesky, solve_triangular
from scipy.optimize import minimize
from statsmodels.stats.stattools import durbin_watson
```

```{code-cell} ipython3
:tags: [hide-cell]

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

## Data

Both this lecture and the companion lecture {doc}`hansen_singleton_1982` use the same data construction.

Both {cite:t}`hansen1982generalized` and {cite:t}`hansen1983stochastic` use monthly data on real per capita consumption (nondurables) and stock returns from CRSP for the period 1959:2 through 1978:12.

To align with the paper, we set the default sample to 1959:2--1978:12.

You can pass different `start` and `end` dates to study later periods.

This lecture pulls stock-market and one-month bill returns from the Ken French data library (`F-F_Research_Data_Factors`) and constructs gross nominal returns as `1 + (Mkt-RF + RF)/100` for the market and `1 + RF/100` for bills.

Hansen-Singleton use CRSP value-weighted NYSE returns.

Exact CRSP NYSE replication is not open-data feasible, so this lecture uses Ken French's CRSP value-weighted market factor as the closest public proxy.

To keep the core message clear, we use one consumption construction throughout: nondurables (`ND`) with the nondurables deflator (the specification most directly aligned with Table 1 discussion in the paper).

To compute in-sample growth rates at the requested start month, we fetch one extra pre-sample month internally and then trim back to the exact user-specified sample window.

```{code-cell} ipython3
:tags: [hide-cell]

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


def load_hs_monthly_data(
    start="1959-02-01",
    end="1978-12-01",
):
    """
    Build monthly gross real return and gross consumption-growth series.
    """
    start_period = pd.Timestamp(start).to_period("M")
    end_period = pd.Timestamp(end).to_period("M")

    # Pull one extra month to build the first in-sample growth rate.
    fetch_start = (start_period - 1).to_timestamp(how="start")
    fetch_end = end_period.to_timestamp("M")
    sample_start = start_period.to_timestamp("M")
    sample_end = end_period.to_timestamp("M")

    fred = web.DataReader(list(FRED_CODES.values()), "fred", fetch_start, fetch_end)
    fred = fred.rename(columns={v: k for k, v in FRED_CODES.items()})
    fred.index = to_month_end(fred.index)
    fred["cons_real_level"] = fred["cons_nd_real_index"]
    fred["cons_price_index"] = fred["cons_nd_price_index"]
    fred["consumption_per_capita"] = fred["cons_real_level"] / fred["population_16plus"]
    fred["gross_cons_growth"] = (
        fred["consumption_per_capita"] / fred["consumption_per_capita"].shift(1)
    )
    fred["gross_inflation_cons"] = (
        fred["cons_price_index"] / fred["cons_price_index"].shift(1)
    )

    ff = web.DataReader("F-F_Research_Data_Factors", "famafrench", fetch_start, fetch_end)[0].copy()
    ff.columns = [str(col).strip() for col in ff.columns]
    if ("Mkt-RF" not in ff.columns) or ("RF" not in ff.columns):
        raise KeyError("Fama-French data missing required columns: 'Mkt-RF' and 'RF'.")

    # Mkt-RF and RF are reported in percent per month.
    ff["gross_nom_return"] = 1.0 + (ff["Mkt-RF"] + ff["RF"]) / 100.0
    ff["gross_nom_tbill"] = 1.0 + ff["RF"] / 100.0
    ff.index = ff.index.to_timestamp(how="end")
    ff.index = to_month_end(ff.index)
    market = ff[["gross_nom_return", "gross_nom_tbill"]]

    out = fred.join(market, how="inner")
    out["gross_real_return"] = out["gross_nom_return"] / out["gross_inflation_cons"]
    out["gross_real_tbill"] = out["gross_nom_tbill"] / out["gross_inflation_cons"]
    out = out.loc[sample_start:sample_end].dropna()

    required_cols = [
        "gross_real_return",
        "gross_cons_growth",
        "gross_inflation_cons",
        "consumption_per_capita",
        "gross_real_tbill",
    ]
    return out[required_cols].copy()


def get_estimation_data(
    start="1959-02-01",
    end="1978-12-01",
):
    """
    Return (dataframe, array, source_label) using observed data.
    """
    frame = load_hs_monthly_data(start=start, end=end)
    data = frame[["gross_real_return", "gross_cons_growth"]].to_numpy()
    source = (
        "Ken French CRSP market proxy (Mkt-RF + RF) + FRED ND consumption "
        "(closest open-data analog to HS 1983 Table 1 inputs)"
    )
    return frame, data, source


def get_tbill_estimation_data(
    start="1959-02-01",
    end="1978-12-01",
):
    """
    Return (dataframe, array, source_label) using Treasury bill data.
    """
    frame = load_hs_monthly_data(start=start, end=end)
    data = frame[["gross_real_tbill", "gross_cons_growth"]].to_numpy()
    source = (
        "Ken French RF (1m bill), deflated by FRED ND consumption deflator "
        "(closest open-data analog to HS Treasury-bill setup)"
    )
    return frame, data, source
```

## The Euler equation under lognormality

Using the paper's parameterization, the Euler equation (derived in detail in {doc}`hansen_singleton_1982`) is

$$
E_t\!\left[\beta \left(\frac{C_{t+1}}{C_t}\right)^{\alpha} R_{t+1}^i\right] = 1,
$$

where $\alpha=\gamma-1$ in the utility index $U(c_t)=c_t^\gamma/\gamma$, $\beta$ is the subjective discount factor, and $R_{t+1}^i$ is the gross real return on asset $i$.

The corresponding coefficient of relative risk aversion is $\rho = -\alpha$.

Let $x_t = c_t / c_{t-1}$ denote the consumption ratio, and define $u_{it} = \beta x_t^\alpha r_{it}$ where $r_{it}$ is the gross real return on asset $i$.

The Euler equation states $E_{t-1}[u_{it}] = 1$.

This is the same Euler restriction as above, written with a one-period reindexing.

{cite:t}`hansen1983stochastic` define $u_{it} = x_t^\alpha r_{it}$ and write $E_{t-1}[u_{it}] = 1/\beta$.

Our normalization is equivalent and absorbs $\beta$ directly into $u_{it}$.

Define log variables $X_t = \log x_t$, $R_{it} = \log r_{it}$, and $U_{it} = \log u_{it}$, so that

```{math}
:label: hs83-u-def

U_{i,t}= \alpha X_{t}+R_{i,t}+\log\beta.
```

{cite:t}`hansen1983stochastic` now make their key distributional assumption.

The vector process $\{Y_t\} = \{(X_t, R_{1t}, \ldots, R_{nt})^\top\}$ is jointly stationary and Gaussian.

Under this assumption, $U_{it}$ conditional on the information set $\psi_{t-1}$ is normal with constant variance $\sigma_i^2$ and a conditional mean $\mu_{i,t-1}$ that is a linear function of past $Y$'s.

Because $u_{it} = \exp(U_{it})$ is conditionally lognormal, we can evaluate $E_{t-1}[u_{it}]$ in closed form:

```{math}
:label: hs83-lognormal-identity

E_{t-1}[u_{it}] = \exp\left(\mu_{i,t-1} + \tfrac{1}{2}\sigma_i^2\right).
```

Setting $E_{t-1}[u_{it}] = 1$ and taking logs gives $\mu_{i,t-1} + \sigma_i^2/2 = 0$.

Now define the innovation

```{math}
:label: hs83-v-it

V_{i,t}=\alpha X_{t}+R_{i,t}+\log\beta+\frac{\sigma_i^2}{2},
\quad E_{t-1}[V_{i,t}]=0,
```

where $\sigma_i^2=\operatorname{Var}_{t-1}(\alpha X_t + R_{it})$ is constant under the stationarity and Gaussian assumptions.

Equation {eq}`hs83-v-it` implies the key conditional mean restriction

```{math}
:label: hs83-cond-mean

E_{t-1}[R_{i,t}] = -\alpha\, E_{t-1}[X_{t}] - \log\beta - \frac{\sigma_i^2}{2}.
```

Equation {eq}`hs83-cond-mean` is the central result of {cite:t}`hansen1983stochastic`.

It says that the predictable component of each asset's log return is proportional to the predictable component of log consumption growth, with proportionality factor $-\alpha$.

The intercept absorbs the discount factor $\beta$ and a Jensen's inequality correction $\sigma_i^2 / 2$.

This restriction has three important special cases that illuminate the connection to the equity premium puzzle:

- Risk neutrality ($\alpha = 0$, so $\rho=0$): Returns are unpredictable and have constant expected log return $-\log\beta - \sigma_i^2/2$. All assets have the same expected return up to a Jensen's inequality correction.
- Log utility ($\alpha = -1$, so $\rho=1$): The difference $R_{it} - X_t$ is unpredictable, so returns and consumption growth share the same predictable component.
- Risk aversion (concave CRRA utility, $\alpha<0$ so $\rho>0$): Assets whose returns covary more with consumption growth must offer higher expected returns to compensate risk-averse investors.

The last point is the consumption CAPM pricing kernel.

When $\rho=-\alpha$ is large, the expected return spread between risky assets and the risk-free rate is large; when $\rho$ is small, it is small.

The equity premium puzzle arises because the observed spread is large but estimated $|\alpha|$ is small.

We implement the log transformation with the helper below.

```{code-cell} ipython3
def to_mle_array(data):
    valid = (data[:, 0] > 0.0) & (data[:, 1] > 0.0)
    return np.column_stack(
        [np.log(data[valid, 1]), np.log(data[valid, 0])])
```

## The triangular system and its likelihood

To build a likelihood, we need to parameterize the conditional expectation $E_{t-1}[X_t]$.

In the single-return case, write $Y_t = (X_t, R_t)^\top$ and assume that the predictable component of $X_t$ is a finite-order linear function of past observations:

```{math}
:label: hs83-x-forecast

E(X_t\mid\psi_{t-1})=a(L)^\top Y_{t-1}+\mu_x,
```

where $a(L)$ is a vector of lag polynomial coefficients in past $(X, R)$ and $\mu_x$ is a constant.

The consumption-growth equation is unrestricted, so $X_t$ depends freely on its own lags and on lagged returns.

The return equation, however, is restricted by the Euler equation.

Combining {eq}`hs83-cond-mean` with {eq}`hs83-x-forecast` forces the predictable part of $R_t$ to be $-\alpha$ times the predictable part of $X_t$, plus a constant.

This gives the triangular system

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
\mu=\begin{bmatrix}\mu_x\\-\log\beta-\sigma_U^2/2\end{bmatrix},
```

where $\sigma_U^2 \equiv \operatorname{Var}_{t-1}(\alpha X_t + R_t) = \alpha^2 \sigma_{XX} + \sigma_{RR} + 2\alpha \sigma_{XR}$ under conditional homoskedasticity.

The sign in the second element of $\mu$ follows directly from {eq}`hs83-cond-mean`.

The system is "triangular" because $A_0$ is unit lower triangular.

The first equation determines consumption growth, and the second equation pins down the return conditional on consumption growth.

Because $\det(A_0) = 1$, the Jacobian of the transformation from innovations $V_t$ to observables $Y_t$ is unity.

This makes the Gaussian log-likelihood straightforward.

Given $T$ observations and conditional on initial values, the log-likelihood is (see equation (17) of {cite:t}`hansen1983stochastic`)

```{math}
:label: hs83-loglik

L(\theta) = -\frac{T}{2}\log|\Sigma| - \frac{1}{2}\sum_{t=1}^{T}(A_0 Y_t - A_1(L)Y_{t-1} - \mu)^\top\Sigma^{-1}(A_0 Y_t - A_1(L)Y_{t-1} - \mu),
```

where $\Sigma$ is the covariance matrix of the innovation $V_t$ and we have dropped the constant $-T\log(2\pi)$.

The restrictions imposed by the Euler equation enter through the structure of $A_0$, $A_1(L)$, and $\mu$.

The return equation has no free lag coefficients and its intercept is determined by $\alpha$, $\beta$, and $\Sigma$.

An unrestricted VAR would estimate both equations freely, with $2(1 + 2p) + 3$ free parameters (where $p$ is the lag length).

The restricted triangular system has only $6 + 2p$ free parameters because the Euler equation ties the return equation's dynamics and intercept to the consumption equation.

The difference in degrees of freedom is the basis for the likelihood ratio tests reported by {cite:t}`hansen1983stochastic`.

## Likelihood implementation

We now implement the likelihood {eq}`hs83-loglik`.

The building blocks are a function to construct lagged data matrices $(Y_t, Y_{t-1}, \ldots, Y_{t-p})$, a function to map the parameter vector into the matrices $A_0$, $A_1$, $\mu$, $\Sigma$, a function to compute the triangular-system residuals $V_t = A_0 Y_t - A_1(L) Y_{t-1} - \mu$, and finally the Gaussian log-likelihood itself.

```{code-cell} ipython3
def build_lagged_data(data, n_lags, base_lags=None):
    """
    Build Y_t and lag stacks [Y_{t-1}, ..., Y_{t-p}] for bivariate data.
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must be T x 2.")
    if base_lags is None:
        base_lags = n_lags
    if base_lags < n_lags:
        raise ValueError("base_lags must be at least n_lags.")
    if data.shape[0] <= base_lags:
        raise ValueError("Sample size must exceed base_lags.")

    t_obs = data.shape[0]
    n_obs = t_obs - base_lags
    y_t = data[base_lags:, :]
    y_lag = np.empty((n_obs, 2 * n_lags))

    for lag in range(1, n_lags + 1):
        y_lag[:, 2 * (lag - 1) : 2 * lag] = data[base_lags - lag : t_obs - lag, :]

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
    if not np.isfinite(α):
        return None
    if not (tol < β):
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
    sigma_x = float(parsed["sigma_x"])
    sigma_r = float(parsed["sigma_r"])
    cov_xr = float(parsed["cov_xr"])
    mu_x = float(parsed["mu_x"])
    a_lags = np.asarray(parsed["a_lags"])

    a0 = np.array([[1.0, 0.0], [α, 1.0]])
    a1 = np.zeros((2, 2 * n_lags))
    a1[0, :] = a_lags
    sigma_u2 = α ** 2 * sigma_x ** 2 + sigma_r ** 2 + 2.0 * α * cov_xr
    mu = np.array([mu_x, -np.log(β) - 0.5 * sigma_u2])

    resid = y_t @ a0.T - y_lag @ a1.T - mu[None, :]
    if np.any(np.abs(resid) > 1e10):
        return None
    return resid
```

The triangular structure also gives us a direct simulation recursion: given parameters, we draw innovations $V_t \sim N(0, \Sigma)$ and solve $Y_t = A_0^{-1}(A_1(L) Y_{t-1} + \mu + V_t)$ forward in time.

This allows us to generate data from the model and verify that MLE recovers the known parameters.

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

    sigma_e = np.array(
        [
            [sigma_x ** 2, cov_xr],
            [cov_xr, sigma_r ** 2],
        ]
    )

    a0 = np.array([[1.0, 0.0], [α, 1.0]])
    sigma_v = a0 @ sigma_e @ a0.T

    eigvals = np.linalg.eigvals(sigma_v)
    if np.min(eigvals) <= 0.0:
        sigma_v += np.eye(2) * 1e-6

    a1 = np.zeros((2, 2 * n_lags))
    a1[0, :] = a_lags
    sigma_u2 = α ** 2 * sigma_x ** 2 + sigma_r ** 2 + 2.0 * α * cov_xr
    mu = np.array([mu_x, -np.log(β) - 0.5 * sigma_u2])

    total_n = n_obs + burn_in
    y = np.zeros((total_n, 2))

    for t in range(n_lags, total_n):
        lag_stack = []
        for lag in range(1, n_lags + 1):
            lag_stack.append(y[t - lag, :])
        lag_vec = np.concatenate(lag_stack)
        shock = np.random.multivariate_normal(np.zeros(2), sigma_v)
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

    α = float(parsed["α"])
    sigma_x = float(parsed["sigma_x"])
    sigma_r = float(parsed["sigma_r"])
    cov_xr = float(parsed["cov_xr"])

    sigma_e = np.array(
        [
            [sigma_x ** 2, cov_xr],
            [cov_xr, sigma_r ** 2],
        ]
    )

    a0 = np.array([[1.0, 0.0], [α, 1.0]])
    sigma_v = a0 @ sigma_e @ a0.T

    try:
        chol = cholesky(sigma_v, lower=True)
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

We keep $\beta$ positive but do not force $\beta < 1$ in estimation, matching common empirical practice in this literature and avoiding boundary-driven LR distortions.

```{code-cell} ipython3
def parameter_bounds(n_lags):
    """
    Bounds for optimization.
    """
    bounds = [
        (-10.0, 10.0),
        (1e-8, 2.0),
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
def starting_values(y_t, y_lag, n_lags, n_starts=24):
    """
    Generate multiple data-driven starting values.

    Includes an unrestricted-VAR-informed seed to reduce local-optimum risk.
    """
    rng = np.random.default_rng(123)
    starts = []
    n_params = 6 + 2 * n_lags

    base = np.zeros(n_params)
    base[0] = -0.5
    base[1] = 0.999
    base[2] = max(float(np.std(y_t[:, 0])), 1e-3)
    base[3] = max(float(np.std(y_t[:, 1])), 1e-3)
    base[4] = float(np.cov(y_t.T)[0, 1])
    base[5] = float(np.mean(y_t[:, 0]))
    base[6:] = 0.1
    starts.append(base.copy())

    # OLS seed from unrestricted VAR
    n_obs = y_t.shape[0]
    x = np.column_stack([np.ones(n_obs), y_lag])
    coef = np.linalg.lstsq(x, y_t, rcond=None)[0]
    resid = y_t - x @ coef
    sigma_e = resid.T @ resid / max(1, n_obs)

    a_lags_ols = coef[1:, 0]
    r_lags_ols = coef[1:, 1]
    denom = float(a_lags_ols @ a_lags_ols)
    if denom > 1e-10:
        α_ols = -float((a_lags_ols @ r_lags_ols) / denom)
    else:
        α_ols = -0.5

    mu_x_ols = float(coef[0, 0])
    mu_r_ols = float(coef[0, 1])
    sigma_x_ols = float(np.sqrt(max(sigma_e[0, 0], 1e-8)))
    sigma_r_ols = float(np.sqrt(max(sigma_e[1, 1], 1e-8)))
    cov_xr_ols = float(sigma_e[0, 1])
    sigma_u2_ols = (
        α_ols ** 2 * sigma_x_ols ** 2
        + sigma_r_ols ** 2
        + 2.0 * α_ols * cov_xr_ols
    )
    β_ols = float(np.exp(-(mu_r_ols + α_ols * mu_x_ols + 0.5 * sigma_u2_ols)))
    β_ols = float(np.clip(β_ols, 1e-6, 2.0))

    ols_seed = np.zeros(n_params)
    ols_seed[0] = α_ols
    ols_seed[1] = β_ols
    ols_seed[2] = sigma_x_ols
    ols_seed[3] = sigma_r_ols
    ols_seed[4] = cov_xr_ols
    ols_seed[5] = mu_x_ols
    ols_seed[6:] = a_lags_ols
    starts.append(ols_seed.copy())

    seeds = [base, ols_seed]
    while len(starts) < n_starts:
        seed = seeds[len(starts) % len(seeds)]
        trial = seed.copy()
        trial[:2] += rng.normal(0.0, 0.2, 2)
        trial[2:6] *= 1.0 + rng.normal(0.0, 0.15, 4)
        trial[6:] += rng.normal(0.0, 0.08, 2 * n_lags)
        trial[1] = max(trial[1], 1e-6)
        trial[2] = max(trial[2], 1e-6)
        trial[3] = max(trial[3], 1e-6)
        starts.append(trial)

    return starts
```

Standard errors come from an outer-product-of-gradients (OPG) approximation to the
information matrix, computed by finite differences of per-observation
log-likelihood contributions.

This tends to be more numerically stable than finite-difference Hessians in this
application.

```{code-cell} ipython3
def log_likelihood_contributions(
    params,
    y_t,
    y_lag,
    n_lags,
    include_const=False,
):
    """
    Vector of per-observation Gaussian log-likelihood contributions.

    Returns an array of length T, or None if the parameter vector is infeasible.
    """
    parsed = unpack_parameters(params, n_lags)
    if parsed is None:
        return None

    resid = triangular_residuals(params, y_t, y_lag, n_lags)
    if resid is None:
        return None

    α = float(parsed["α"])
    sigma_x = float(parsed["sigma_x"])
    sigma_r = float(parsed["sigma_r"])
    cov_xr = float(parsed["cov_xr"])

    sigma_e = np.array(
        [
            [sigma_x ** 2, cov_xr],
            [cov_xr, sigma_r ** 2],
        ]
    )
    a0 = np.array([[1.0, 0.0], [α, 1.0]])
    sigma_v = a0 @ sigma_e @ a0.T

    try:
        chol = cholesky(sigma_v, lower=True)
        log_det = 2.0 * np.sum(np.log(np.diag(chol) + 1e-16))
        std_resid = solve_triangular(chol, resid.T, lower=True).T
    except (LinAlgError, ValueError):
        return None

    quad = np.sum(std_resid ** 2, axis=1)
    ll_t = -0.5 * log_det - 0.5 * quad
    if include_const:
        ll_t -= np.log(2.0 * np.pi)
    if not np.all(np.isfinite(ll_t)):
        return None
    return ll_t


def opg_standard_errors(
    params,
    y_t,
    y_lag,
    n_lags,
    step=1e-6,
    max_step_shrink=12,
    eig_floor=1e-12,
):
    """
    Standard errors via OPG approximation to the information matrix.
    """
    n = len(params)
    ll0 = log_likelihood_contributions(params, y_t, y_lag, n_lags, include_const=False)
    if ll0 is None:
        return np.full(n, np.nan)

    n_obs = int(ll0.shape[0])
    scores = np.empty((n_obs, n))

    for i in range(n):
        base_step = step * (abs(params[i]) + 1.0)
        hi = base_step
        ll_plus = None
        ll_minus = None

        for _ in range(max_step_shrink + 1):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += hi
            p_minus[i] -= hi
            ll_plus = log_likelihood_contributions(
                p_plus, y_t, y_lag, n_lags, include_const=False
            )
            ll_minus = log_likelihood_contributions(
                p_minus, y_t, y_lag, n_lags, include_const=False
            )
            if ll_plus is not None and ll_minus is not None:
                break
            hi *= 0.5

        if ll_plus is None or ll_minus is None:
            return np.full(n, np.nan)

        scores[:, i] = (ll_plus - ll_minus) / (2.0 * hi)

    if not np.all(np.isfinite(scores)):
        return np.full(n, np.nan)

    # Center scores to mitigate numerical drift away from the first-order condition.
    scores = scores - scores.mean(axis=0, keepdims=True)

    opg = scores.T @ scores
    if not np.all(np.isfinite(opg)):
        return np.full(n, np.nan)
    opg = 0.5 * (opg + opg.T)

    try:
        eigvals, eigvecs = np.linalg.eigh(opg)
    except (LinAlgError, ValueError):
        return np.full(n, np.nan)

    floor = float(eig_floor) * max(1.0, float(np.max(eigvals)))
    eigvals = np.clip(eigvals, floor, None)
    cov = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    se[~np.isfinite(se)] = np.nan
    return se


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

    if not np.all(np.isfinite(hess)):
        return np.full(n, np.nan)

    hess = 0.5 * (hess + hess.T)

    try:
        eigvals, eigvecs = np.linalg.eigh(hess)
    except (LinAlgError, ValueError):
        return np.full(n, np.nan)

    # Numerical Hessians are often nearly singular in this application.
    eig_floor = 1e-8
    eigvals_clipped = np.clip(eigvals, eig_floor, None)
    cov = eigvecs @ np.diag(1.0 / eigvals_clipped) @ eigvecs.T
    diagonal = np.diag(cov)
    se = np.sqrt(np.maximum(diagonal, 0.0))
    se[~np.isfinite(se)] = np.nan
    return se
```

When OPG standard errors fail (e.g., due to infeasible perturbations) we fall back
to a finite-difference Hessian and then to the optimizer's inverse-Hessian
approximation.

```{code-cell} ipython3
def lbfgs_standard_errors(opt_result, eig_floor=1e-12):
    """
    Fallback standard errors from L-BFGS-B inverse-Hessian approximation.
    """
    try:
        h_inv = opt_result.hess_inv
        cov = np.asarray(h_inv.todense() if hasattr(h_inv, "todense") else h_inv)
        cov = 0.5 * (cov + cov.T)
        if not np.all(np.isfinite(cov)):
            return None

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, eig_floor, None)
        cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        se = np.sqrt(np.maximum(np.diag(cov_psd), 0.0))
        se[~np.isfinite(se)] = np.nan
        return se
    except (LinAlgError, ValueError, AttributeError, TypeError):
        return None
```

Let's combine the pieces in a multi-start MLE estimator that returns parameters, fit criteria, and residuals.

```{code-cell} ipython3
def estimate_mle(data, n_lags, base_lags=None, verbose=False):
    """
    Estimate the restricted triangular model by multi-start local optimization.
    """
    y_t, y_lag = build_lagged_data(data, n_lags, base_lags=base_lags)
    bounds = parameter_bounds(n_lags)
    starts = starting_values(y_t, y_lag, n_lags)

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

        if np.isfinite(result.fun):
            ll_val = log_likelihood_mle(result.x, y_t, y_lag, n_lags)
            if np.isfinite(ll_val) and ll_val > best_ll:
                best_ll = ll_val
                best_result = result
                if verbose:
                    print(
                        f"start={i}, success={result.success}, "
                        f"status={result.status}, loglike={ll_val:.2f}"
                    )

    n_params = 6 + 2 * n_lags

    if best_result is None:
        return {
            "params": np.full(n_params, np.nan),
            "se": np.full(n_params, np.nan),
            "loglike": -np.inf,
            "aic": np.inf,
            "bic": np.inf,
            "converged": False,
            "optimizer_success": False,
            "residuals": None,
            "n_obs": int(y_t.shape[0]),
        }

    params = best_result.x
    se = opg_standard_errors(params, y_t, y_lag, n_lags)
    if np.any(~np.isfinite(se)):
        se_hess = numerical_standard_errors(params, y_t, y_lag, n_lags)
        if se_hess is not None and se_hess.shape == se.shape:
            se = np.where(np.isfinite(se), se, se_hess)

    if np.any(~np.isfinite(se)):
        se_lbfgs = lbfgs_standard_errors(best_result)
        if se_lbfgs is not None and se_lbfgs.shape == se.shape:
            se = np.where(np.isfinite(se), se, se_lbfgs)
    resid = triangular_residuals(params, y_t, y_lag, n_lags)
    ll_val = log_likelihood_mle(params, y_t, y_lag, n_lags)

    return {
        "params": params,
        "se": se,
        "loglike": ll_val,
        "aic": -2.0 * ll_val + 2.0 * n_params,
        "bic": -2.0 * ll_val + n_params * np.log(y_t.shape[0]),
        "converged": bool(np.isfinite(ll_val)),
        "optimizer_success": bool(best_result.success),
        "residuals": resid,
        "n_obs": int(y_t.shape[0]),
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
    common_sample=False,
    verbose=False,
):
    """
    Estimate restricted MLE models by lag length.
    """
    rows = []
    fits = {}
    base_lags = max(lags) if common_sample else None

    for lag in lags:
        fit = estimate_mle(data, n_lags=lag, base_lags=base_lags, verbose=verbose)
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
                "n_obs": fit["n_obs"],
            }
        )

    table = pd.DataFrame(rows).set_index("n_lags")
    return table, fits
```

## Identification and likelihood ratio tests

The restricted triangular system identifies preference parameters through cross-equation restrictions.

The parameter $\alpha$ links predictable variation in returns to predictable variation in consumption growth.

Under the model, the return equation's dependence on lagged variables is entirely determined by $\alpha$ times the consumption equation's lag coefficients.

The parameter $\beta$ shifts the return-equation intercept through $-\log\beta - \sigma_U^2/2$.

{cite:t}`hansen1983stochastic` test these restrictions by comparing the restricted triangular system to an unrestricted bivariate VAR estimated on the same sample.

If the Euler-equation restrictions are correct, the restricted model should fit nearly as well as the unrestricted model.

The standard test is the likelihood ratio statistic

```{math}
:label: hs83-lr-test

LR = 2(\ell_u - \ell_r) \Rightarrow \chi^2_d,
```

where $\ell_u$ and $\ell_r$ are the maximized log-likelihoods of the unrestricted and restricted models, and $d$ is the difference in the number of free parameters.

Both likelihoods must be evaluated on the same effective sample for the LR distribution to be valid.

{cite:t}`hansen1983stochastic` report that for the value-weighted aggregate stock return, the $\chi^2$ test statistics have probability values around 0.52 to 0.83 across lag lengths (Table 1), providing little evidence against the model.

In the tables below we report both `chi2.cdf(LR, df)` and the usual right-tail `p(LR) = 1 - chi2.cdf`.

However, for individual Dow Jones stocks and for Treasury bills, the restrictions are strongly rejected.

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
        return {"lr_stat": np.nan, "p_value": np.nan, "chi2_cdf": np.nan}
    if fit_restricted.get("n_obs") != fit_unrestricted.get("n_obs"):
        return {"lr_stat": np.nan, "p_value": np.nan, "chi2_cdf": np.nan}

    lr_stat = 2.0 * (fit_unrestricted["loglike"] - fit_restricted["loglike"])
    chi2_cdf = stats.chi2.cdf(lr_stat, df=df_diff)
    p_value = 1.0 - chi2_cdf
    return {
        "lr_stat": float(lr_stat),
        "p_value": float(p_value),
        "chi2_cdf": float(chi2_cdf),
    }
```

The unrestricted benchmark is a Gaussian VAR for $Y_t = (X_t, R_t)^\top$ with free coefficients on all lags in both equations.

```{code-cell} ipython3
def estimate_unrestricted_var(data, n_lags, base_lags=None):
    """
    Estimate an unrestricted Gaussian VAR for Y_t = [X_t, R_t].
    """
    y_t, y_lag = build_lagged_data(data, n_lags, base_lags=base_lags)
    n_obs = y_t.shape[0]
    x = np.column_stack([np.ones(n_obs), y_lag])
    coef = np.linalg.lstsq(x, y_t, rcond=None)[0]
    resid = y_t - x @ coef
    sigma = resid.T @ resid / n_obs

    try:
        chol = cholesky(sigma, lower=True)
        log_det = 2.0 * np.sum(np.log(np.diag(chol) + 1e-16))
        std_resid = solve_triangular(chol, resid.T, lower=True).T
        quad_form = np.sum(std_resid ** 2)
    except (LinAlgError, ValueError):
        return {
            "coef": coef,
            "sigma": np.full((2, 2), np.nan),
            "residuals": resid,
            "loglike": -np.inf,
            "aic": np.inf,
            "bic": np.inf,
            "converged": False,
            "n_obs": int(n_obs),
        }

    d = y_t.shape[1]
    loglike = float(-0.5 * n_obs * d * np.log(2.0 * np.pi) - 0.5 * n_obs * log_det - 0.5 * quad_form)
    n_params = 2 * (1 + 2 * n_lags) + 3
    aic = float(-2.0 * loglike + 2.0 * n_params)
    bic = float(-2.0 * loglike + n_params * np.log(n_obs))

    return {
        "coef": coef,
        "sigma": sigma,
        "residuals": resid,
        "loglike": loglike,
        "aic": aic,
        "bic": bic,
        "converged": True,
        "n_obs": int(n_obs),
    }
```

For paper-style replication, lag-specific samples are often used (`common_sample=False`), so each lag uses all observations available at that lag.

If you want direct AIC/BIC comparability across lag lengths, set `common_sample=True`.

```{code-cell} ipython3
def run_unrestricted_var_by_lag(data, lags=(2, 4, 6), common_sample=False):
    """
    Estimate unrestricted VAR models by lag length.
    """
    rows = []
    fits = {}
    base_lags = max(lags) if common_sample else None

    for lag in lags:
        fit = estimate_unrestricted_var(data, n_lags=lag, base_lags=base_lags)
        fits[lag] = fit
        rows.append(
            {
                "n_lags": lag,
                "loglike": fit["loglike"],
                "aic": fit["aic"],
                "bic": fit["bic"],
                "n_obs": fit["n_obs"],
            }
        )

    table = pd.DataFrame(rows).set_index("n_lags")
    return table, fits
```

The following function computes the LR statistic at each lag length, replicating the testing strategy of Table 1 in {cite:t}`hansen1983stochastic`.

```{code-cell} ipython3
def restricted_vs_unrestricted_lr(mle_fits, unrestricted_fits, lags=(2, 4, 6)):
    """
    Compute LR tests of restricted triangular model versus unrestricted VAR.
    """
    rows = []

    for lag in lags:
        fit_r = mle_fits[lag]
        fit_u = unrestricted_fits[lag]
        df_diff = (2 * (1 + 2 * lag) + 3) - (6 + 2 * lag)
        lr = likelihood_ratio_test(fit_restricted=fit_r, fit_unrestricted=fit_u, df_diff=df_diff)
        rows.append(
            {
                "n_lags": lag,
                "lr_stat": lr["lr_stat"],
                "p_value": lr["p_value"],
                "chi2_cdf": lr["chi2_cdf"],
                "df": df_diff,
                "T": fit_r.get("n_obs", np.nan),
            }
        )

    return pd.DataFrame(rows).set_index("n_lags")
```

## Predictability and the $R^2$ restriction

A key empirical implication of the restricted system, emphasized in Section II of {cite:t}`hansen1983stochastic`, concerns the predictability of asset returns.

From {eq}`hs83-cond-mean` and {eq}`hs83-x-forecast`, the predictable component of the log return is

```{math}
:label: hs83-predictable-return

E(R_t \mid \psi_{t-1}) = -\alpha\, E(X_t \mid \psi_{t-1}) - \log\beta - \frac{\sigma_U^2}{2}.
```

Since the predictable return is a linear function of the predictable consumption growth, the variance of the predictable return component is exactly $\alpha^2$ times the variance of the predictable consumption-growth component:

```{math}
:label: hs83-var-pred

\operatorname{Var}[E(R_t \mid \psi_{t-1})] = \alpha^2 \operatorname{Var}[E(X_t \mid \psi_{t-1})].
```

{cite:t}`hansen1983stochastic` derive the implied $R^2$ of the return projection onto $\psi_{t-1}$:

```{math}
:label: hs83-r2

R_R^2 = \frac{\alpha^2 \operatorname{Var}[E(X_t \mid \psi_{t-1})]}{\operatorname{Var}(R_t \mid \psi_{t-1}) + \alpha^2 \operatorname{Var}[E(X_t \mid \psi_{t-1})]}.
```

This expression has important implications.

If $\alpha = 0$ (risk neutrality), then $R_R^2 = 0$ and asset returns are unpredictable.

If $\alpha = -1$ (log utility), then $R_t - X_t$ is unpredictable, so returns and consumption growth share the same predictable component.

More generally, the $R_R^2$ for returns will be small whenever the variance of the unpredictable return component $\operatorname{Var}(R_t \mid \psi_{t-1})$ is large relative to the predictable variance, which is the case for stock returns.

{cite:t}`hansen1983stochastic` report $R_R^2$ values of 0.02 to 0.06 for the value-weighted stock return, meaning that although the model implies return predictability when agents are risk averse, the predictable component is swamped by the unpredictable component.

The function below reports:
- restriction-side predictable-variance terms implied by the Euler equation, and
- $R_X^2$ and $R_R^2$ from the unrestricted VAR (matching the way the paper labels these $R^2$ statistics).

```{code-cell} ipython3
def predictability_metrics(data, restricted_fit, unrestricted_fit, n_lags, base_lags=None):
    """
    Compute predictable-component metrics and unrestricted VAR R^2 values.
    """
    y_t, y_lag = build_lagged_data(data, n_lags, base_lags=base_lags)
    parsed = unpack_parameters(restricted_fit["params"], n_lags)
    α = float(parsed["α"])
    β = float(parsed["β"])
    sigma_x = float(parsed["sigma_x"])
    sigma_r = float(parsed["sigma_r"])
    cov_xr = float(parsed["cov_xr"])
    mu_x = float(parsed["mu_x"])
    a_lags = np.asarray(parsed["a_lags"])

    pred_x = y_lag @ a_lags + mu_x
    sigma_u2 = α ** 2 * sigma_x ** 2 + sigma_r ** 2 + 2.0 * α * cov_xr
    pred_r = -α * pred_x - np.log(β) - 0.5 * sigma_u2

    x = y_t[:, 0]
    r = y_t[:, 1]
    resid_x = x - pred_x
    resid_r = r - pred_r

    r2_x = 1.0 - np.var(resid_x) / np.var(x)
    r2_r = 1.0 - np.var(resid_r) / np.var(r)

    if unrestricted_fit["converged"] and unrestricted_fit.get("coef") is not None:
        n_obs = y_t.shape[0]
        x_u = np.column_stack([np.ones(n_obs), y_lag])
        pred_u = x_u @ unrestricted_fit["coef"]
        resid_u = y_t - pred_u
        var_x = np.var(y_t[:, 0])
        var_r = np.var(y_t[:, 1])
        r2_x_unres = np.nan if var_x <= 0.0 else 1.0 - np.var(resid_u[:, 0]) / var_x
        r2_r_unres = np.nan if var_r <= 0.0 else 1.0 - np.var(resid_u[:, 1]) / var_r
    else:
        r2_x_unres = np.nan
        r2_r_unres = np.nan

    return {
        "alpha_hat": α,
        "var_pred_x": float(np.var(pred_x)),
        "var_pred_r": float(np.var(pred_r)),
        "alpha2_var_pred_x": float(α ** 2 * np.var(pred_x)),
        "r2_x_restricted": float(r2_x),
        "r2_r_restricted": float(r2_r),
        "r2_x_unrestricted": float(r2_x_unres),
        "r2_r_unrestricted": float(r2_r_unres),
        "T": int(y_t.shape[0]),
    }
```

## Return-difference tests

{cite:t}`hansen1983stochastic` also propose tests based on differences in log returns across assets.

From {eq}`hs83-cond-mean`, the difference $R_{it} - R_{jt}$ has conditional mean $-(\sigma_i^2 - \sigma_j^2)/2 = (\sigma_j^2 - \sigma_i^2)/2$, which is constant.

This means that return differences should be unpredictable if the model is correct, regardless of the values of $\alpha$ and $\beta$.

These tests avoid the need to measure consumption or to align consumption timing with return timing, at the cost of losing the ability to identify $\alpha$ and $\beta$.

If we generalize the CRRA utility to include a multiplicative preference shock $\lambda_t$, so that $U(c_t, \lambda_t) = c_t^\gamma \lambda_t / \gamma$ (with $\alpha=\gamma-1$), the Euler equation becomes $E_t[\beta (c_{t+1}/c_t)^\alpha (\lambda_{t+1}/\lambda_t) r_{it+1}] = 1$.

Under lognormality, the difference in log returns between assets $i$ and $j$ satisfies

$$
R_{it} - R_{jt} = \frac{\tilde\sigma_j^2 - \tilde\sigma_i^2}{2} + \tilde V_{it} - \tilde V_{jt},
$$

where the consumption, preference shock, and discount factor terms cancel.

This sign convention is the one implied by {eq}`hs83-cond-mean`; some printed versions reverse this constant term.

This means return differences should be unpredictable if the model holds, regardless of $\alpha$, $\beta$, or the preference shock.

{cite:t}`hansen1983stochastic` report that the return-difference restrictions are strongly rejected for models with multiple stock returns, providing substantial evidence against the CRRA-lognormal specification even when consumption measurement problems are eliminated.

The code below is an illustration of this logic on simulated data; reproducing the paper's empirical return-difference tables requires estimating multi-asset systems that are outside this notebook's scope.

```{code-cell} ipython3
def simulate_multi_asset_nominal_returns(
    n_obs,
    n_assets=3,
    α_true=-0.4,
    β_true=0.993,
    seed=123,
):
    """
    Simulate log nominal returns satisfying E_t[beta * exp(alpha X) * r_i] = 1.
    """
    if n_assets < 2:
        raise ValueError("n_assets must be at least 2.")

    rng = np.random.default_rng(seed)
    x = np.empty(n_obs)
    x[0] = 0.001
    for t in range(1, n_obs):
        x[t] = 0.001 + 0.4 * (x[t - 1] - 0.001) + 0.006 * rng.standard_normal()

    sigmas = np.linspace(0.03, 0.06, n_assets)
    eps = rng.standard_normal((n_obs, n_assets)) * sigmas
    log_returns = -np.log(β_true) - α_true * x[:, None] + eps - 0.5 * sigmas[None, :] ** 2
    return x, log_returns


def return_difference_test(log_returns, n_lags=2):
    """
    Test predictability of pairwise log-return differences using lagged returns as instruments.
    """
    if log_returns.ndim != 2 or log_returns.shape[1] < 2:
        raise ValueError("log_returns must be T x m with m >= 2.")
    if log_returns.shape[0] <= n_lags + 1:
        raise ValueError("Sample size must exceed n_lags + 1.")

    t_obs, n_assets = log_returns.shape
    pairs = list(combinations(range(n_assets), 2))
    n_obs = t_obs - n_lags - 1
    z = np.empty((n_obs, 1 + n_assets * n_lags))
    z[:, 0] = 1.0

    for j in range(n_lags):
        z[:, 1 + j * n_assets : 1 + (j + 1) * n_assets] = log_returns[n_lags - j : t_obs - 1 - j, :]

    rows = []
    for i, j in pairs:
        y = log_returns[n_lags + 1 :, i] - log_returns[n_lags + 1 :, j]
        coef = np.linalg.lstsq(z, y, rcond=None)[0]
        resid = y - z @ coef
        sigma2 = float((resid @ resid) / max(1, n_obs - z.shape[1]))
        cov = sigma2 * np.linalg.pinv(z.T @ z)
        slopes = coef[1:]
        cov_slopes = cov[1:, 1:]
        stat = float(slopes @ np.linalg.pinv(cov_slopes) @ slopes)
        p_value = float(1.0 - stats.chi2.cdf(stat, df=slopes.shape[0]))
        rows.append(
            {
                "pair": f"{i+1}-{j+1}",
                "wald_chi2": stat,
                "p_value": p_value,
                "mean_spread": float(np.mean(y)),
            }
        )

    return pd.DataFrame(rows).set_index("pair")
```

## Simulation exercises

Before applying MLE to real data, we verify that it recovers known parameters from simulated triangular-system data.

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

The optimizer recovers $\hat\alpha$ and $\hat\beta$ near their true values, confirming that the likelihood implementation is correct.

We also verify with simulated return differences.

```{code-cell} ipython3
_, sim_log_returns = simulate_multi_asset_nominal_returns(
    n_obs=900,
    n_assets=3,
    α_true=-0.4,
    β_true=0.993,
    seed=99,
)

spread_test = return_difference_test(sim_log_returns, n_lags=2)
spread_pretty = spread_test.rename(columns={
    "wald_chi2": r"\chi^2", "p_value": "p", "mean_spread": r"\overline{\Delta R}",
})
display_table(spread_pretty, title="Return-Difference Predictability Tests (Simulated)", fmt={
    r"\chi^2": "{:.3f}",
    "p": "{:.3f}",
    r"\overline{\Delta R}": "{:.5f}",
})
```

Large $p$-values confirm that return differences are unpredictable in this simulation, as the model predicts.

## Empirical MLE estimation

We now apply the maximum likelihood estimator of {cite:t}`hansen1983stochastic` to the historical sample, following the format of Table 1 in {cite:t}`hansen1983stochastic`.

By exploiting lognormality, the MLE should yield precise estimates when the distributional assumption is correct.

```{code-cell} ipython3
LAGS = (2, 4, 6)
USE_COMMON_SAMPLE = False  # False matches lag-specific sample handling in HS tables
BASE_LAGS = max(LAGS) if USE_COMMON_SAMPLE else None

emp_frame, emp_data, source = get_estimation_data()

print(f"Data source: {source}")
print(f"Sample window: {emp_frame.index.min().date()} to {emp_frame.index.max().date()}")
print(f"Sample size: {len(emp_data)}")
print(f"Mean net real return: {(emp_data[:, 0].mean() - 1.0) * 100:.3f}%")
print(f"Std net real return: {emp_data[:, 0].std() * 100:.3f}%")
print(f"Mean net consumption growth: {(emp_data[:, 1].mean() - 1.0) * 100:.3f}%")
print(f"Std net consumption growth: {emp_data[:, 1].std() * 100:.3f}%")
print(f"Correlation: {np.corrcoef(emp_data[:, 0], emp_data[:, 1])[0, 1]:.4f}")
```

```{code-cell} ipython3
emp_log_data = to_mle_array(emp_data)
mle_table, mle_fits = run_mle_by_lag(
    emp_log_data, lags=LAGS, common_sample=USE_COMMON_SAMPLE, verbose=False
)
mle_pretty = mle_table.rename(columns={
    "α_hat": r"\hat{\alpha}", "se_α": r"\mathrm{se}(\hat{\alpha})",
    "β_hat": r"\hat{\beta}", "se_β": r"\mathrm{se}(\hat{\beta})",
    "loglike": "logL", "aic": "AIC", "bic": "BIC", "n_obs": "T",
})
display_table(mle_pretty, title="Likelihood Estimates by Lag Length", fmt={
    r"\hat{\alpha}": "{:.4f}", r"\mathrm{se}(\hat{\alpha})": "{:.4f}",
    r"\hat{\beta}": "{:.4f}", r"\mathrm{se}(\hat{\beta})": "{:.4f}",
    "logL": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}", "T": "{:.0f}",
})
```

The table reports $\hat\alpha$ and $\hat\beta$ by lag length for the sample used in the code cell above.

For comparison, {cite:t}`hansen1983stochastic` report $\hat\alpha$ values of $-0.33$ to $-1.25$ (standard errors 0.65 to 0.83) for the value-weighted return with nondurables consumption (Table 1 of their paper).

In risk-aversion units, this corresponds to $\hat\rho=-\hat\alpha$ between $0.33$ and $1.25$.

With `USE_COMMON_SAMPLE=False` above, each lag uses its own maximum available sample (closer to paper-style tables).

Set it to `True` for strictly common-sample AIC/BIC comparison across rows.

We now compute the predictability summaries that are central to {cite:t}`hansen1983stochastic`.

```{code-cell} ipython3
unres_table, unres_fits = run_unrestricted_var_by_lag(
    emp_log_data,
    lags=LAGS,
    common_sample=USE_COMMON_SAMPLE,
)

unres_pretty = unres_table.rename(columns={"loglike": "logL", "aic": "AIC", "bic": "BIC", "n_obs": "T"})
display_table(unres_pretty, title="Unrestricted VAR Benchmarks", fmt={
    "logL": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}", "T": "{:.0f}",
})

pred_rows = []
for lag in LAGS:
    metrics = predictability_metrics(
        emp_log_data,
        restricted_fit=mle_fits[lag],
        unrestricted_fit=unres_fits[lag],
        n_lags=lag,
        base_lags=BASE_LAGS,
    )
    pred_rows.append({"n_lags": lag, **metrics})

pred_df = pd.DataFrame(pred_rows).set_index("n_lags")
pred_pretty = pred_df[
    [
        "alpha_hat",
        "var_pred_x",
        "var_pred_r",
        "alpha2_var_pred_x",
        "r2_x_unrestricted",
        "r2_r_unrestricted",
        "T",
    ]
].rename(columns={
    "alpha_hat": r"\hat{\alpha}",
    "var_pred_x": r"\mathrm{Var}(\hat E[X_t\mid\psi_{t-1}])",
    "var_pred_r": r"\mathrm{Var}(\hat E[R_t\mid\psi_{t-1}])",
    "alpha2_var_pred_x": r"\hat{\alpha}^2\,\mathrm{Var}(\hat E[X_t\mid\psi_{t-1}])",
    "r2_x_unrestricted": r"$R_X^2$ (unres VAR)",
    "r2_r_unrestricted": r"$R_R^2$ (unres VAR)",
    "T": "T",
})
display_table(pred_pretty, title="Predictability Metrics (Hansen-Singleton 1983)", fmt={
    r"\hat{\alpha}": "{:.4f}",
    r"\mathrm{Var}(\hat E[X_t\mid\psi_{t-1}])": "{:.6f}",
    r"\mathrm{Var}(\hat E[R_t\mid\psi_{t-1}])": "{:.6f}",
    r"\hat{\alpha}^2\,\mathrm{Var}(\hat E[X_t\mid\psi_{t-1}])": "{:.6f}",
    r"$R_X^2$ (unres VAR)": "{:.4f}",
    r"$R_R^2$ (unres VAR)": "{:.4f}",
    "T": "{:.0f}",
})
```

The column $\hat\alpha^2 \operatorname{Var}(\hat E[X_t \mid \psi_{t-1}])$ should equal $\operatorname{Var}(\hat E[R_t \mid \psi_{t-1}])$ if the restriction {eq}`hs83-var-pred` holds.

The columns labeled "(unres VAR)" match the paper convention of reporting $R_X^2$ and $R_R^2$ from the unrestricted VAR projections.

In {cite:t}`hansen1983stochastic`, $R_R^2$ is small (0.02 to 0.06) even when $R_X^2$ is nontrivial, highlighting that most stock-return variation is unpredictable.

We now test the Euler-equation restrictions formally by comparing the restricted triangular system to an unrestricted VAR.

```{code-cell} ipython3
lr_hs83 = restricted_vs_unrestricted_lr(mle_fits, unres_fits, lags=LAGS)
lr_pretty = lr_hs83.rename(columns={
    "lr_stat": "LR",
    "chi2_cdf": "chi2.cdf(LR,df)",
    "p_value": "p(LR)",
    "df": "df",
    "T": "T",
})
display_table(lr_pretty, title="Restricted vs Unrestricted LR Tests", fmt={
    "LR": "{:.3f}",
    "chi2.cdf(LR,df)": "{:.3f}",
    "p(LR)": "{:.3f}",
    "df": "{:.0f}",
    "T": "{:.0f}",
})

sig_level = 0.05
rejected_lags = [int(lag) for lag in lr_hs83.index if lr_hs83.loc[lag, "p_value"] < sig_level]
not_rejected_lags = [int(lag) for lag in lr_hs83.index if lr_hs83.loc[lag, "p_value"] >= sig_level]
print(f"Rejected at 5%: {rejected_lags if rejected_lags else 'none'}")
print(f"Not rejected at 5%: {not_rejected_lags if not_rejected_lags else 'none'}")

if not rejected_lags:
    print("This run is consistent with Hansen-Singleton Table 1 non-rejection for aggregate stock returns.")
else:
    print("This run differs from Hansen-Singleton Table 1, which reports non-rejection for aggregate stock returns.")
    print("Likely drivers are data vintage/construction differences (modern FRED revisions versus the original historical data files).")
```

Use the printed comparison message above to reconcile your run with Table 1 of {cite:t}`hansen1983stochastic`.

### Treasury bill estimation

We now repeat the estimation using the 1-month Treasury bill return in place of the stock return.

{cite:t}`hansen1983stochastic` find that the model is strongly rejected for Treasury bills (Table 4 of their paper).

The nearly predictable, low-volatility behavior of T-bill returns is difficult to reconcile with the volatile, largely unpredictable behavior of consumption growth.

```{code-cell} ipython3
tbill_frame, tbill_data, tbill_source = get_tbill_estimation_data()

print(f"Data source: {tbill_source}")
print(f"Sample window: {tbill_frame.index.min().date()} to {tbill_frame.index.max().date()}")
print(f"Sample size: {len(tbill_data)}")
print(f"Mean net real T-bill return: {(tbill_data[:, 0].mean() - 1.0) * 100:.3f}%")
print(f"Std net real T-bill return: {tbill_data[:, 0].std() * 100:.3f}%")
print(f"Mean net consumption growth: {(tbill_data[:, 1].mean() - 1.0) * 100:.3f}%")
print(f"Correlation: {np.corrcoef(tbill_data[:, 0], tbill_data[:, 1])[0, 1]:.4f}")
```

```{code-cell} ipython3
tbill_log_data = to_mle_array(tbill_data)
tbill_mle_table, tbill_mle_fits = run_mle_by_lag(
    tbill_log_data, lags=LAGS, common_sample=USE_COMMON_SAMPLE, verbose=False
)

tbill_mle_pretty = tbill_mle_table.rename(columns={
    "α_hat": r"\hat{\alpha}", "se_α": r"\mathrm{se}(\hat{\alpha})",
    "β_hat": r"\hat{\beta}", "se_β": r"\mathrm{se}(\hat{\beta})",
    "loglike": "logL", "aic": "AIC", "bic": "BIC", "n_obs": "T",
})
display_table(tbill_mle_pretty, title="Treasury Bill: Likelihood Estimates by Lag Length", fmt={
    r"\hat{\alpha}": "{:.4f}", r"\mathrm{se}(\hat{\alpha})": "{:.4f}",
    r"\hat{\beta}": "{:.4f}", r"\mathrm{se}(\hat{\beta})": "{:.4f}",
    "logL": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}", "T": "{:.0f}",
})
```

If some T-bill standard errors are very large (or `nan`), that is typically a numerical-information-matrix issue (common with finite differences in near risk-free data) rather than an economic conclusion.

The LR test below remains the main specification check in the paper.

```{code-cell} ipython3
tbill_unres_table, tbill_unres_fits = run_unrestricted_var_by_lag(
    tbill_log_data,
    lags=LAGS,
    common_sample=USE_COMMON_SAMPLE,
)

tbill_unres_pretty = tbill_unres_table.rename(columns={"loglike": "logL", "aic": "AIC", "bic": "BIC", "n_obs": "T"})
display_table(tbill_unres_pretty, title="Treasury Bill: Unrestricted VAR Benchmarks", fmt={
    "logL": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}", "T": "{:.0f}",
})
```

```{code-cell} ipython3
tbill_pred_rows = []
for lag in LAGS:
    metrics = predictability_metrics(
        tbill_log_data,
        restricted_fit=tbill_mle_fits[lag],
        unrestricted_fit=tbill_unres_fits[lag],
        n_lags=lag,
        base_lags=BASE_LAGS,
    )
    tbill_pred_rows.append({"n_lags": lag, **metrics})

tbill_pred_df = pd.DataFrame(tbill_pred_rows).set_index("n_lags")
tbill_pred_pretty = tbill_pred_df[
    [
        "alpha_hat",
        "var_pred_x",
        "var_pred_r",
        "alpha2_var_pred_x",
        "r2_x_unrestricted",
        "r2_r_unrestricted",
        "T",
    ]
].rename(columns={
    "alpha_hat": r"\hat{\alpha}",
    "var_pred_x": r"\mathrm{Var}(\hat E[X_t\mid\psi_{t-1}])",
    "var_pred_r": r"\mathrm{Var}(\hat E[R_t\mid\psi_{t-1}])",
    "alpha2_var_pred_x": r"\hat{\alpha}^2\,\mathrm{Var}(\hat E[X_t\mid\psi_{t-1}])",
    "r2_x_unrestricted": r"$R_X^2$ (unres VAR)",
    "r2_r_unrestricted": r"$R_R^2$ (unres VAR)",
    "T": "T",
})
display_table(tbill_pred_pretty, title="Treasury Bill: Predictability Metrics (Hansen-Singleton 1983)", fmt={
    r"\hat{\alpha}": "{:.4f}",
    r"\mathrm{Var}(\hat E[X_t\mid\psi_{t-1}])": "{:.6f}",
    r"\mathrm{Var}(\hat E[R_t\mid\psi_{t-1}])": "{:.6f}",
    r"\hat{\alpha}^2\,\mathrm{Var}(\hat E[X_t\mid\psi_{t-1}])": "{:.6f}",
    r"$R_X^2$ (unres VAR)": "{:.4f}",
    r"$R_R^2$ (unres VAR)": "{:.4f}",
    "T": "{:.0f}",
})
```

```{code-cell} ipython3
tbill_lr = restricted_vs_unrestricted_lr(tbill_mle_fits, tbill_unres_fits, lags=LAGS)
tbill_lr_pretty = tbill_lr.rename(columns={
    "lr_stat": "LR",
    "chi2_cdf": "chi2.cdf(LR,df)",
    "p_value": "p(LR)",
    "df": "df",
    "T": "T",
})
display_table(tbill_lr_pretty, title="Treasury Bill: Restricted vs Unrestricted LR Tests", fmt={
    "LR": "{:.3f}",
    "chi2.cdf(LR,df)": "{:.3f}",
    "p(LR)": "{:.3f}",
    "df": "{:.0f}",
    "T": "{:.0f}",
})

tbill_rejected_lags = [int(lag) for lag in tbill_lr.index if tbill_lr.loc[lag, "p_value"] < 0.05]
print(f"T-bill model rejected at 5% for lags: {tbill_rejected_lags if tbill_rejected_lags else 'none'}")
```

{cite:t}`hansen1983stochastic` find the same qualitative pattern in their 1959--1978 sample (Table 4): the Treasury bill model is rejected much more strongly than the value-weighted stock-return model.

The model cannot reconcile the smooth, nearly predictable behavior of Treasury bill returns with consumption growth.

This is a precursor to the *risk-free rate puzzle* of {cite:t}`Weil_1989`.

We inspect the residual paths, histograms, and diagnostic statistics from the restricted model to assess whether the maintained assumptions of normality and serial independence are plausible.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Restricted-model residual diagnostics
    name: fig-hs83-residual-diagnostics
---
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

The residual time-series plots reveal periods of unusually large innovations (volatility clustering), while the histograms show departures from the Gaussian bell curve.

The Jarque-Bera statistics are large for both series, decisively rejecting the normality assumption that underlies the likelihood.

The Durbin-Watson statistics are close to 2 for returns (no serial correlation) but somewhat above 2 for consumption (mild negative autocorrelation in the residuals).

Rejection of normality is expected given the well-known fat tails in financial returns.

This is precisely the situation where the robustness of GMM (see {doc}`hansen_singleton_1982`) becomes valuable.

Even if the lognormality assumption fails and the MLE is misspecified, the GMM estimates based on orthogonality conditions alone remain consistent.

## Connection to the equity premium puzzle

The empirical findings of {cite:t}`hansen1983stochastic` contain the key ingredients of what {cite:t}`MehraPrescott1985` would formalize as the **equity premium puzzle**.

Let us assemble the evidence.

**Low estimated risk aversion.** The estimated $\hat\alpha$ values (and thus $\hat\rho=-\hat\alpha$) from the table above can be compared with {cite:t}`hansen1983stochastic`, who report $\hat\alpha$ between $-0.33$ and $-1.25$ (Table 1).

These imply modest risk aversion by equity-premium standards.

**Tiny return predictability.** Compare the unrestricted-VAR $R_R^2$ values with the 0.02 to 0.06 range in {cite:t}`hansen1983stochastic`.

In both cases, most return variation is unpredictable.

**Strong rejection for Treasury bills.** As the LR tests above confirm, the Euler-equation restrictions are decisively rejected when the model is estimated with Treasury bill returns.

{cite:t}`hansen1983stochastic` find the same result at essentially zero significance levels (Table 4 of their paper).

The model cannot reconcile the smooth, nearly predictable behavior of Treasury bill returns with the volatile, unpredictable behavior of consumption growth.

This is a precursor to the *risk-free rate puzzle* of {cite:t}`Weil_1989`.

Matching the low level and smooth behavior of the risk-free rate requires a discount factor $\beta$ very close to or above 1, which conflicts with the positive time preference needed to match average consumption growth.

**Rejection for individual stocks.** When the model is estimated using returns on individual Dow Jones Industrials (American Brands, Exxon, IBM), the $\chi^2$ tests reject the restrictions at extremely low significance levels (Table 3 of {cite:t}`hansen1983stochastic`).

The CRRA-lognormal model cannot explain the cross-section of expected returns.

**The puzzle formalized.** {cite:t}`MehraPrescott1985` crystallized these findings into a sharp quantitative statement.

In a calibrated version of the consumption-based model with CRRA utility, they showed that for relative risk aversion $\gamma_{\text{MP}}$ in the range of 0 to 10 (the range considered "reasonable" by most economists), the model could not simultaneously match:

1. the average annual equity premium of about 6\%,
2. the average annual risk-free rate of about 1\%.

Matching the equity premium requires very high $\gamma_{\text{MP}}$ (above 30 in many calibrations), while matching the risk-free rate requires $\gamma_{\text{MP}}$ near 1.

This is exactly the tension visible in the {cite:t}`hansen1983stochastic` estimates.

The implied risk aversion level $\hat\rho=-\hat\alpha$ is too low to generate a large equity premium.

**Why doesn't the LR test reject for aggregate stock returns?** At first glance, the equity premium puzzle might seem to predict that the LR test should reject.

If the CRRA model is fundamentally wrong, shouldn't the data tell us so?

The answer involves a subtle but important distinction between *economic* rejection and *statistical* rejection.

The equity premium puzzle is about the *level* of expected returns: $\gamma_{\text{MP}} \approx 1$ cannot generate a 6\% annual equity premium.

But the LR test of {cite:t}`hansen1983stochastic` does not test the level of expected returns.

It tests the *proportionality restriction* that predictable movements in log returns equal $-\alpha$ times predictable movements in log consumption growth.

When unrestricted-VAR $R_R^2$ is only 0.02 to 0.06, the predictable component of returns accounts for a tiny fraction of total return variation.

The unrestricted VAR, which is free to fit this component without the proportionality restriction, barely improves the fit.

The LR statistic is therefore small and the test does not reject.

Non-rejection here reflects the *low power* of the test: there is so little predictable variation that the data cannot distinguish the restricted model from the unrestricted one.

The model fails on economic grounds even where it passes on statistical grounds.

The estimated parameters imply only modest risk aversion, which implies an equity premium far smaller than the observed value.

This contradiction is invisible to the LR test because the test examines time-series restrictions, not whether the implied risk premium matches its observed level.

For Treasury bills the situation is different.

T-bill returns are smooth and highly predictable, so there is ample predictable variation for the test to work with, and the proportionality restriction is strongly violated.

The aggregate stock-return LR outcome is sensitive to sample choice and data construction.

This is why reproducing the paper's sample window is important when comparing conclusions.

**Subsequent literature.** The equity premium puzzle spurred a large literature seeking to resolve it by modifying preferences, beliefs, or market structure:

- **Habit formation** ({cite:t}`abel1990asset`, {cite:t}`campbell1999force`): utility depends on consumption relative to a habit level, amplifying effective risk aversion without raising $\gamma_{\text{MP}}$.
- **Long-run risks** ({cite:t}`bansal2004risks`): small persistent components in consumption growth, amplified by Epstein-Zin preferences, can generate large risk premia.
- **Rare disasters** ({cite:t}`barro2006rare`): low-probability catastrophic events justify a large equity premium even with moderate $\gamma_{\text{MP}}$.
- **Robustness and model uncertainty** (see {doc}`doubts_or_variability`): agents who distrust their model of consumption dynamics demand higher compensation for bearing uncertainty.

Each of these approaches effectively finds a way to make the stochastic discount factor more volatile than CRRA preferences with low $\gamma_{\text{MP}}$ would allow, without requiring implausible levels of constant relative risk aversion.

## Robustness versus efficiency

The interplay between {cite:t}`hansen1982generalized` and {cite:t}`hansen1983stochastic` provides a clean illustration of the fundamental econometric tradeoff between *robustness* and *efficiency*.

The GMM estimator of {doc}`hansen_singleton_1982` is robust to distributional misspecification.

It remains consistent whether or not consumption and returns are jointly lognormal, but at the cost of larger standard errors.

The MLE estimator of this lecture is efficient under lognormality.

It exploits the full distributional structure to produce the smallest possible standard errors.

But when lognormality fails, as the Jarque-Bera tests on return residuals invariably suggest, the MLE is not just imprecise but potentially inconsistent.

{cite:t}`hansen1982generalized` (Section 4 and Table II) compare the two approaches directly.

They note that the MLE standard errors are smaller than the GMM standard errors, but caution that this may reflect distributional misspecification rather than genuine efficiency gains.

The residual diagnostics from our empirical section confirm that lognormality is rejected for stock returns, validating the concern that motivated the GMM approach in the first place.

## Summary

This lecture has implemented the maximum likelihood estimator of {cite:t}`hansen1983stochastic` for the consumption-based Euler equation under joint lognormality.

The key findings are low estimated risk aversion, tiny return predictability, and strong rejection for Treasury bills and individual stocks.

These are precisely the empirical facts that {cite:t}`MehraPrescott1985` formalized as the equity premium puzzle.

In the original sample of {cite:t}`hansen1983stochastic`, the CRRA-lognormal model is not rejected for aggregate stock returns, but it fails for individual stocks, Treasury bills, and the joint stock-bond system.

Across samples, the model struggles to explain the level of the risk-free rate and the magnitude of the equity premium.

These failures motivated a rich subsequent literature that has explored habit formation, long-run risks, rare disasters, and robustness as potential resolutions.

The comparison between the GMM approach of {doc}`hansen_singleton_1982` and the MLE approach of this lecture illustrates the robustness-efficiency tradeoff that pervades empirical work in asset pricing and macroeconomics.
