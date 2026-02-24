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

This lecture studies the maximum likelihood estimator of {cite:t}`hansen1983stochastic` and connects its empirical findings to the equity premium puzzle of {cite:t}`MehraPrescott1985`.

{cite:t}`hansen1983stochastic` study a consumption-based asset pricing model in which a representative consumer with CRRA preferences chooses how to allocate wealth across traded assets.

The first-order conditions generate stochastic Euler equations relating consumption growth, asset returns, and preference parameters.

{cite:t}`hansen1983stochastic` assume that consumption growth and asset returns are *jointly lognormal*.

Under these assumptions, the Euler equation implies a set of restrictions on a linear time-series representation of the logarithms of consumption growth and returns.

Specifically, predictable movements in log returns must be proportional to predictable movements in log consumption growth, with proportionality factor $-\alpha$.

In the notation of {cite:t}`hansen1983stochastic`, utility is $U(c_t)=c_t^\gamma/\gamma$ with $\gamma<1$, so $\alpha=\gamma-1$ and the coefficient of relative risk aversion is $-\alpha$.

This restricted representation takes the form of a triangular VAR that can be estimated by maximum likelihood.

The empirical findings of {cite:t}`hansen1983stochastic` foreshadow what {cite:t}`MehraPrescott1985` would formalize as the **equity premium puzzle**:

1. Point estimates imply relatively low risk aversion ($-\alpha$ is typically between 0 and 2), too low to explain the large observed gap between stock returns and the risk-free rate.
2. The predictable component of stock returns ($R_R^2$ of 0.02 to 0.06) is tiny relative to total return variation, even when consumption growth itself has forecastable variation.
3. The model fits aggregate value-weighted stock returns reasonably well, but is strongly rejected for Treasury bills (where the restrictions on the risk-free rate cannot be reconciled with consumption data) and for individual stocks.
4. The low estimates of $-\alpha$ combined with the high observed equity premium imply that CRRA preferences cannot simultaneously match the level of the risk-free rate and the equity premium.

The following companion lecture {doc}`hansen_singleton_1982` develops a robust alternative based on GMM that does not require the lognormality assumption.

Relative to {cite:t}`hansen1983stochastic`, we simplify by estimating one return at a time (market proxy or T-bill) rather than full multi-asset systems, using only monthly nondurable consumption (`ND`), and omitting the multi-stock return-difference rejection table and the just-identified (`NLAG = 0`) comparison.

In addition to what comes with Anaconda, this lecture requires `pandas-datareader`

```{code-cell} ipython3
:tags: [hide-output]

!pip install pandas-datareader
```

```{code-cell} ipython3
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Math
from pandas_datareader import data as web
from scipy import stats
from scipy.linalg import LinAlgError, cholesky, solve_triangular
from scipy.optimize import minimize
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings(
    "ignore", message=".*date_parser.*", category=FutureWarning
)
```

We also define a helper to display DataFrames as LaTeX arrays in the hidden cell below

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

While Hansen-Singleton use CRSP value-weighted NYSE returns, we choose to use the Ken French market factor as the closest open-data proxy for the CRSP value-weighted market return.

To keep the core message clear, we use one consumption construction throughout: nondurables (`ND`) with the nondurables deflator.

The hidden cell below pulls the relevant FRED series, constructs per capita real consumption, and joins with the Ken French returns

```{code-cell} ipython3
:tags: [hide-cell]

fred_codes = {
    "population_16plus": "CNP16OV",
    "cons_nd_real_index": "DNDGRA3M086SBEA",
    "cons_nd_price_index": "DNDGRG3M086SBEA",
}

def to_month_end(index):
    """
    Convert a date index to month-end timestamps.
    """
    return pd.PeriodIndex(pd.DatetimeIndex(index), freq="M").to_timestamp("M")


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

    fred = web.DataReader(
        list(fred_codes.values()), "fred", fetch_start, fetch_end)
    fred = fred.rename(columns={v: k for k, v in fred_codes.items()})
    fred.index = to_month_end(fred.index)
    fred["cons_real_level"] = fred["cons_nd_real_index"]
    fred["cons_price_index"] = fred["cons_nd_price_index"]
    fred["consumption_per_capita"] = fred["cons_real_level"] \
        / fred["population_16plus"]
    fred["gross_cons_growth"] = (
        fred["consumption_per_capita"] / fred["consumption_per_capita"].shift(1)
    )
    fred["gross_inflation_cons"] = (
        fred["cons_price_index"] / fred["cons_price_index"].shift(1)
    )

    ff = web.DataReader(
        "F-F_Research_Data_Factors", "famafrench", 
        fetch_start, fetch_end)[0].copy()
    ff.columns = [str(col).strip() for col in ff.columns]
    if ("Mkt-RF" not in ff.columns) or ("RF" not in ff.columns):
        raise KeyError(
            "Fama-French data missing required columns: 'Mkt-RF' and 'RF'.")

    # Mkt-RF and RF are reported in percent per month.
    ff["gross_nom_return"] = 1.0 + (ff["Mkt-RF"] + ff["RF"]) / 100.0
    ff["gross_nom_tbill"] = 1.0 + ff["RF"] / 100.0
    ff.index = ff.index.to_timestamp(how="end")
    ff.index = to_month_end(ff.index)
    market = ff[["gross_nom_return", "gross_nom_tbill"]]

    out = fred.join(market, how="inner")
    out["gross_real_return"] = out["gross_nom_return"] \
        / out["gross_inflation_cons"]
    out["gross_real_tbill"] = out["gross_nom_tbill"] \
        / out["gross_inflation_cons"]
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

## Euler equation

Consider a single-good economy of identical consumers whose utility functions are of the CRRA type:

```{math}
:label: hs83-crra

U(c_t) = c_t^{\gamma}/\gamma, \quad \gamma < 1,
```

where $c_t$ is aggregate real per capita consumption and $U(\cdot)$ is the period utility function.

The representative consumer chooses a stochastic consumption plan to maximize the expected value of a time-additive utility function,

```{math}
:label: hs83-objective

E_0 \sum_{t=0}^{\infty} \beta^t U(c_t), \quad 0 < \beta < 1.
```

Consumers substitute present for future consumption by trading the ownership rights of $N$ financial and capital assets.

Let $\mathbf{w}_t$ denote the holdings of the $N$ assets at date $t$, $\mathbf{q}_t$ the vector of asset prices, $\mathbf{d}_t$ the vector of dividends, and $y_t$ real labor income.

A feasible consumption and investment plan $\{c_t, \mathbf{w}_t\}$ must satisfy the sequence of budget constraints

```{math}
:label: hs83-budget

c_t + \mathbf{q}_t \cdot \mathbf{w}_{t+1} \leq (\mathbf{q}_t + \mathbf{d}_t) \cdot \mathbf{w}_t + y_t,
```

where $(\mathbf{q}_t + \mathbf{d}_t) \cdot \mathbf{w}_t$ is the cum-dividend value of the portfolio carried into period $t$.

The first-order necessary conditions for the maximization of {eq}`hs83-objective` subject to {eq}`hs83-budget` that involve the equilibrium prices of the $n$ assets are ({cite:t}`Lucas1978`, {cite:t}`Brock1982`):

```{math}
:label: hs83-foc

U'(c_t) = \beta E_t\!\left[U'(c_{t+1})\, r_{it+1}\right], \quad i = 1, \ldots, N,
```

where $r_{it+1}$ is the gross real return on asset $i$.

Substituting the CRRA marginal utility $U'(c_t) = c_t^{\gamma-1} = c_t^{\alpha}$ with $\alpha \equiv \gamma - 1$ into {eq}`hs83-foc` and rearranging gives

```{math}
:label: hs83-euler

E_t\!\left[\beta \left(\frac{c_{t+1}}{c_t}\right)^{\alpha} r_{it+1}\right] = 1, \quad i = 1, \ldots, N.
```

The coefficient of relative risk aversion is $-\alpha$.

## The Euler equation under lognormality

Using the Euler equation {eq}`hs83-euler` derived above (see also {doc}`hansen_singleton_1982` for generalizations), we now impose the distributional assumptions of {cite:t}`hansen1983stochastic`.

Let $x_t = c_t / c_{t-1}$ denote the consumption ratio, and define $u_{it} = x_t^\alpha r_{it}$ where $r_{it}$ is the gross real return on asset $i$.

The Euler equation states $E_{t-1}[u_{it}] = 1/\beta$.

This is the same Euler restriction as above but reindexed to time $t-1$.

Define log variables $X_t = \log x_t$, $R_{it} = \log r_{it}$, and $U_{it} = \log u_{it}$, so that

```{math}
:label: hs83-u-def

U_{i,t}= \alpha X_{t}+R_{i,t}.
```

{cite:t}`hansen1983stochastic` now make their key distributional assumption.

The vector process $\{Y_t\} = \{(X_t, R_{1t}, \ldots, R_{nt})^\top\}$ is jointly stationary and Gaussian.

Under this assumption, $U_{it}$ conditional on the information set $\psi_{t-1}$ is normal with constant variance $\sigma_i^2$ and a conditional mean $\mu_{i,t-1}$ that is a linear function of past $Y$'s.

Because $u_{it} = \exp(U_{it})$ is conditionally lognormal, we can evaluate $E_{t-1}[u_{it}]$ in closed form:

```{math}
:label: hs83-lognormal-identity

E_{t-1}[u_{it}] = \exp\left(\mu_{i,t-1} + \tfrac{1}{2}\sigma_i^2\right).
```

Setting $E_{t-1}[u_{it}] = 1/\beta$ and taking logs gives $\mu_{i,t-1} + \sigma_i^2/2 = -\log\beta$.

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

- Risk neutrality ($\alpha = 0$): Each asset's log return equals a constant plus a serially uncorrelated error, so returns are serially uncorrelated.
- Log utility ($\alpha = -1$): The difference $R_{it} - X_t$ is unpredictable, so returns and consumption growth share the same predictable component.
- Risk aversion ($\alpha < 0$): The predictable component of $R_{it}$ is $-\alpha$ times the predictable component of $X_t$. 
    - A larger $|\alpha|$ amplifies the link between forecastable consumption growth and forecastable returns.

When $-\alpha$ is large, the expected return spread between risky assets and the risk-free rate is large; when $-\alpha$ is small, it is small.

The equity premium puzzle arises because the observed spread is large but estimated $|\alpha|$ is small.

## The triangular system and its likelihood

To build a likelihood, we need to parameterize the conditional expectation $E_{t-1}[X_t]$.

In the single-return case, write $\mathbf{Y}_t = (X_t, R_t)^\top$ and assume that the predictable component of $X_t$ is a finite-order linear function of past observations:

```{math}
:label: hs83-x-forecast

E(X_t\mid\psi_{t-1})=\mathbf{a}(L)^\top \mathbf{Y}_{t-1}+\mu_x,
```

where $\mathbf{a}(L)$ is a vector of lag polynomial coefficients in past $(X, R)$ and $\mu_x$ is a constant.

The consumption-growth equation is unrestricted, so $X_t$ depends freely on its own lags and on lagged returns.

The return equation, however, is restricted by the Euler equation.

Combining {eq}`hs83-cond-mean` with {eq}`hs83-x-forecast` forces the predictable part of $R_t$ to be $-\alpha$ times the predictable part of $X_t$, plus a constant.

This gives the triangular system

```{math}
:label: hs83-triangular

\mathbf{A}_0\mathbf{Y}_t=\mathbf{A}_1(L)\mathbf{Y}_{t-1}+\boldsymbol{\mu}+\mathbf{V}_t,
```

with

```{math}
:label: hs83-a0a1

\mathbf{A}_0=\begin{bmatrix}1&0\\\alpha&1\end{bmatrix},
\quad
\mathbf{A}_1(L)=\begin{bmatrix}\mathbf{a}(L)^\top\\0\end{bmatrix},
\quad
\boldsymbol{\mu}=\begin{bmatrix}\mu_x\\-\log\beta-\sigma_U^2/2\end{bmatrix},
```

where $\sigma_U^2 \equiv \operatorname{Var}_{t-1}(\alpha X_t + R_t) = \alpha^2 \sigma_{XX} + \sigma_{RR} + 2\alpha \sigma_{XR}$ under conditional homoskedasticity.

The sign in the second element of $\boldsymbol{\mu}$ follows directly from {eq}`hs83-cond-mean`.

The system {eq}`hs83-triangular` is "triangular" because $\mathbf{A}_0$ in {eq}`hs83-a0a1` is unit lower triangular.

The first row of {eq}`hs83-triangular` determines consumption growth, and the second row pins down the return conditional on consumption growth.

Because $\det(\mathbf{A}_0) = 1$, the Jacobian of the transformation from innovations $\mathbf{V}_t$ to observables $\mathbf{Y}_t$ is unity.

This makes the Gaussian log-likelihood straightforward.

Given $T$ observations and conditional on initial values, the log-likelihood is

```{math}
:label: hs83-loglik

L(\theta) = -\frac{T}{2}\log|\boldsymbol{\Sigma}| - \frac{1}{2}\sum_{t=1}^{T}(\mathbf{A}_0 \mathbf{Y}_t - \mathbf{A}_1(L)\mathbf{Y}_{t-1} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{A}_0 \mathbf{Y}_t - \mathbf{A}_1(L)\mathbf{Y}_{t-1} - \boldsymbol{\mu}),
```

where $\boldsymbol{\Sigma}$ is the covariance matrix of the innovation $\mathbf{V}_t$, $\theta$ collects all free parameters — $\alpha$, $\beta$, the covariance parameters, the first-row intercept $\mu_x$, and the first-row lag coefficients.

Moreover, we have dropped the constant $-T\log(2\pi)$.

The restrictions imposed by {eq}`hs83-euler` enter {eq}`hs83-loglik` through the structure of $\mathbf{A}_0$, $\mathbf{A}_1(L)$, and $\boldsymbol{\mu}$ in {eq}`hs83-a0a1`.

The second row of {eq}`hs83-triangular` has no free lag coefficients and its intercept is determined by $\alpha$, $\beta$, and $\boldsymbol{\Sigma}$.

An unrestricted bivariate VAR($p$) would estimate the first row and second row of {eq}`hs83-triangular` freely.

Each row has 1 intercept plus $2p$ lag coefficients ($p$ lags $\times$ 2 variables), giving $2(1 + 2p)$ mean parameters, plus 3 free covariance parameters ($\sigma_{XX}, \sigma_{RR}, \sigma_{XR}$), for a total of $5 + 4p$.

The restricted system {eq}`hs83-triangular` has only $6 + 2p$ free parameters: the first row contributes $1 + 2p$ (its intercept $\mu_x$ and $2p$ lag coefficients), plus $\alpha$, $\beta$, and the 3 covariance parameters. 

The second row adds nothing because its lag structure and intercept are pinned down by $\alpha$, $\beta$, and $\boldsymbol{\Sigma}$ via {eq}`hs83-cond-mean`.

The difference $(\smash{5 + 4p}) - (\smash{6 + 2p}) = 2p - 1$ gives the degrees of freedom for the likelihood ratio tests reported by {cite:t}`hansen1983stochastic`.

## Likelihood implementation

We now implement the likelihood {eq}`hs83-loglik`.


Since we are working with log-transformed data, we define a helper for the transformation below

```{code-cell} ipython3
def to_mle_array(data):
    valid = (data[:, 0] > 0.0) & (data[:, 1] > 0.0)
    return np.column_stack(
        [np.log(data[valid, 1]), np.log(data[valid, 0])])
```


The building blocks are a function to construct lagged data matrices $(\mathbf{Y}_t, \mathbf{Y}_{t-1}, \ldots, \mathbf{Y}_{t-p})$, a function to map the parameter vector into the matrices $\mathbf{A}_0$, $\mathbf{A}_1$, $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$, a function to compute the triangular-system residuals $\mathbf{V}_t = \mathbf{A}_0 \mathbf{Y}_t - \mathbf{A}_1(L) \mathbf{Y}_{t-1} - \boldsymbol{\mu}$, and finally the Gaussian log-likelihood itself.

First we build the lagged data matrices, which are the inputs to the likelihood function

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

Next, we validate and unpack the parameter vector while enforcing feasibility conditions

```{code-cell} ipython3
def unpack_parameters(params, n_lags):
    """
    Validate and unpack parameter vector.
    """
    if len(params) != 6 + 2 * n_lags:
        return None

    α, β, σ_x, σ_r, cov_xr, μ_x = params[:6]
    a_lags = params[6:]

    tol = 1e-8
    if not np.isfinite(α):
        return None
    if not (tol < β):
        return None
    if not (σ_x > tol and σ_r > tol):
        return None

    Σ = np.array(
        [
            [σ_x ** 2, cov_xr],
            [cov_xr, σ_r ** 2],
        ]
    )

    try:
        cholesky(Σ, lower=True)
    except (LinAlgError, ValueError):
        return None

    return {
        "α": np.array(α),
        "β": np.array(β),
        "σ_x": np.array(σ_x),
        "σ_r": np.array(σ_r),
        "cov_xr": np.array(cov_xr),
        "μ_x": np.array(μ_x),
        "a_lags": a_lags,
    }
```

The next step maps parameters and lagged data into triangular-system residuals

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
    σ_x = float(parsed["σ_x"])
    σ_r = float(parsed["σ_r"])
    cov_xr = float(parsed["cov_xr"])
    μ_x = float(parsed["μ_x"])
    a_lags = np.asarray(parsed["a_lags"])

    A0 = np.array([[1.0, 0.0], [α, 1.0]])
    A1 = np.zeros((2, 2 * n_lags))
    A1[0, :] = a_lags
    σ_u2 = α ** 2 * σ_x ** 2 + σ_r ** 2 + 2.0 * α * cov_xr
    μ = np.array([μ_x, -np.log(β) - 0.5 * σ_u2])

    resid = y_t @ A0.T - y_lag @ A1.T - μ[None, :]
    if np.any(np.abs(resid) > 1e10):
        return None
    return resid
```

The triangular structure also gives us a way to generate simulations given parameters. 

We draw innovations $\mathbf{V}_t \sim N(\mathbf{0}, \boldsymbol{\Sigma})$ and solve $\mathbf{Y}_t = \mathbf{A}_0^{-1}(\mathbf{A}_1(L) \mathbf{Y}_{t-1} + \boldsymbol{\mu} + \mathbf{V}_t)$ forward in time.

This allows us to generate data from the model and verify that MLE recovers the known parameters in a Monte Carlo exercise

```{code-cell} ipython3
def simulate_triangular_var(
    params,
    n_obs,
    n_lags,
    burn_in=200,
    seed=0,
):
    """
    Simulate [log consumption growth, log return] from the triangular model.
    """
    if seed is not None:
        np.random.seed(seed)

    if len(params) != 6 + 2 * n_lags:
        raise ValueError("Parameter vector length must be 6 + 2 * n_lags.")

    α, β, σ_x, σ_r, cov_xr, μ_x = params[:6]
    a_lags = params[6:]

    Σ_e = np.array(
        [
            [σ_x ** 2, cov_xr],
            [cov_xr, σ_r ** 2],
        ]
    )

    A0 = np.array([[1.0, 0.0], [α, 1.0]])
    Σ_v = A0 @ Σ_e @ A0.T

    eigvals = np.linalg.eigvals(Σ_v)
    if np.min(eigvals) <= 0.0:
        Σ_v += np.eye(2) * 1e-6

    A1 = np.zeros((2, 2 * n_lags))
    A1[0, :] = a_lags
    σ_u2 = α ** 2 * σ_x ** 2 + σ_r ** 2 + 2.0 * α * cov_xr
    μ = np.array([μ_x, -np.log(β) - 0.5 * σ_u2])

    total_n = n_obs + burn_in
    y = np.zeros((total_n, 2))

    for t in range(n_lags, total_n):
        lag_stack = []
        for lag in range(1, n_lags + 1):
            lag_stack.append(y[t - lag, :])
        lag_vec = np.concatenate(lag_stack)
        shock = np.random.multivariate_normal(np.zeros(2), Σ_v)
        y[t, :] = np.linalg.solve(A0, A1 @ lag_vec + μ + shock)

    return y[burn_in:, :]
```

Next, we encode the Gaussian log-likelihood implied by the residual covariance matrix

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
    σ_x = float(parsed["σ_x"])
    σ_r = float(parsed["σ_r"])
    cov_xr = float(parsed["cov_xr"])

    Σ_e = np.array(
        [
            [σ_x ** 2, cov_xr],
            [cov_xr, σ_r ** 2],
        ]
    )

    A0 = np.array([[1.0, 0.0], [α, 1.0]])
    Σ_v = A0 @ Σ_e @ A0.T

    try:
        chol = cholesky(Σ_v, lower=True)
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

To optimize numerically, let's wrap the log-likelihood as a minimization objective

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

We keep $\beta$ positive but do not force $\beta < 1$ in estimation

```{code-cell} ipython3
def parameter_bounds(n_lags):
    """
    Bounds for optimization.
    """
    bounds = [
        (-200.0, 200.0),   # α (risk aversion)
        (1e-8, 2.0),     # β (discount factor)
        (1e-8, None),    # σ_x (std dev of consumption innovation)
        (1e-8, None),    # σ_r (std dev of return innovation)
        (None, None),    # cov_xr (covariance)
        (None, None),    # μ_x (consumption growth intercept)
    ]
    bounds += [(-0.99, 0.99)] * (2 * n_lags)  # VAR lag coefficients
    return bounds
```

Several perturbed starting vectors help local solvers escape poor initializations.

```{code-cell} ipython3
def starting_values(y_t, y_lag, n_lags, n_starts=10):
    """
    Generate multiple data-driven starting values.
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
    Σ_e = resid.T @ resid / max(1, n_obs)

    a_lags_ols = coef[1:, 0]
    r_lags_ols = coef[1:, 1]
    denom = float(a_lags_ols @ a_lags_ols)
    if denom > 1e-10:
        α_ols = -float((a_lags_ols @ r_lags_ols) / denom)
    else:
        α_ols = -0.5

    μ_x_ols = float(coef[0, 0])
    μ_r_ols = float(coef[0, 1])
    σ_x_ols = float(np.sqrt(max(Σ_e[0, 0], 1e-8)))
    σ_r_ols = float(np.sqrt(max(Σ_e[1, 1], 1e-8)))
    cov_xr_ols = float(Σ_e[0, 1])
    σ_u2_ols = (
        α_ols ** 2 * σ_x_ols ** 2
        + σ_r_ols ** 2
        + 2.0 * α_ols * cov_xr_ols
    )
    β_ols = float(np.exp(-(μ_r_ols + α_ols * μ_x_ols + 0.5 * σ_u2_ols)))
    β_ols = float(np.clip(β_ols, 1e-6, 2.0))

    ols_seed = np.zeros(n_params)
    ols_seed[0] = α_ols
    ols_seed[1] = β_ols
    ols_seed[2] = σ_x_ols
    ols_seed[3] = σ_r_ols
    ols_seed[4] = cov_xr_ols
    ols_seed[5] = μ_x_ols
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
    σ_x = float(parsed["σ_x"])
    σ_r = float(parsed["σ_r"])
    cov_xr = float(parsed["cov_xr"])

    Σ_e = np.array(
        [
            [σ_x ** 2, cov_xr],
            [cov_xr, σ_r ** 2],
        ]
    )
    A0 = np.array([[1.0, 0.0], [α, 1.0]])
    Σ_v = A0 @ Σ_e @ A0.T

    try:
        chol = cholesky(Σ_v, lower=True)
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

    # Center scores to mitigate numerical drift away
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
```

Let's combine the pieces in a multi-start MLE estimator that returns parameters, fit criteria, and residuals.

```{code-cell} ipython3
def estimate_mle(data, n_lags, verbose=False):
    """
    Estimate the restricted triangular model by multi-start local optimization.
    """
    y_t, y_lag = build_lagged_data(data, n_lags)
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
            "converged": False,
            "optimizer_success": False,
            "residuals": None,
            "n_obs": int(y_t.shape[0]),
        }

    params = best_result.x
    se = opg_standard_errors(params, y_t, y_lag, n_lags)
    resid = triangular_residuals(params, y_t, y_lag, n_lags)
    ll_val = log_likelihood_mle(params, y_t, y_lag, n_lags)

    return {
        "params": params,
        "se": se,
        "loglike": ll_val,
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
                "n_obs": fit["n_obs"],
            }
        )

    table = pd.DataFrame(rows).set_index("n_lags")
    return table, fits
```

## Simulation recovery check

Before applying the likelihood to empirical data, we verify that it recovers known parameters from simulated triangular-system data.

To avoid confusion about the different "a" objects in the simulation code:

- `a_lags` are the lag-polynomial coefficients in the consumption-growth forecasting regression, i.e., the vector $\mathbf{a}(L)$ in {eq}`hs83-x-forecast`.
- `A0` and `A1` are just the matrices $\mathbf{A}_0$ and $\mathbf{A}_1(L)$ in {eq}`hs83-triangular`, built from $\alpha$ and `a_lags` (they are not additional free parameters).

For `n_lags = 1`, the parameter vector is

```{math}
\theta = (\alpha,\ \beta,\ \sigma_x,\ \sigma_r,\ \mathrm{cov}_{xr},\ \mu_x,\ a_{x,1},\ a_{r,1}),
```

where the last two entries are the coefficients on $X_{t-1}$ and $R_{t-1}$ in the first-row regression for $X_t$.

More generally, for `n_lags = p` we pack `a_lags` in the order
$[a_{x,1}, a_{r,1}, \ldots, a_{x,p}, a_{r,p}]$, so the full parameter vector has length $6 + 2p$.

The covariance parameters $(\sigma_x, \sigma_r, \mathrm{cov}_{xr})$ describe the reduced-form shocks to $(X_t, R_t)$: if $\boldsymbol{\varepsilon}_t = (\varepsilon_{x,t}, \varepsilon_{r,t})^\top \sim N(0, \boldsymbol{\Sigma}_\varepsilon)$, then
$\boldsymbol{\Sigma}_\varepsilon = \begin{bmatrix}\sigma_x^2 & \mathrm{cov}_{xr}\\ \mathrm{cov}_{xr} & \sigma_r^2\end{bmatrix}$.

The triangular-system innovation is $\mathbf{V}_t = \mathbf{A}_0 \boldsymbol{\varepsilon}_t$, so its covariance is $\boldsymbol{\Sigma}_V = \mathbf{A}_0 \boldsymbol{\Sigma}_\varepsilon \mathbf{A}_0^\top$, which is what enters the likelihood.

In the simulation recursion we draw $\mathbf{V}_t \sim N(0, \boldsymbol{\Sigma}_V)$ and solve $\mathbf{Y}_t = \mathbf{A}_0^{-1}(\mathbf{A}_1(L)\mathbf{Y}_{t-1} + \boldsymbol{\mu} + \mathbf{V}_t)$ forward.

The following table compares the true parameters to the MLE estimates from a large simulated sample

```{code-cell} ipython3
α_true = -1.00
β_true = 0.993
σ_x_true = 0.015
σ_r_true = 0.020
cov_xr_true = 0.0001
μ_x_true = 0.002
a_x1_true = 0.40
a_r1_true = 0.10

true_params = np.array(
    [
        α_true,
        β_true,
        σ_x_true,
        σ_r_true,
        cov_xr_true,
        μ_x_true,
        a_x1_true,
        a_r1_true,
    ]
)

sim_mle_data = simulate_triangular_var(
    params=true_params,
    n_obs=50000,
    n_lags=1,
    burn_in=5000,
    seed=0,
)

fit_sim = estimate_mle(sim_mle_data, n_lags=1, verbose=False)
```

```{code-cell} ipython3
:tags: [hide-input]

sim_results = pd.DataFrame({
    "true": true_params[:2],
    "estimate": fit_sim["params"][:2],
    "se": fit_sim["se"][:2],
}, index=[r"α", r"β"])
sim_results[r"t\ (H_0{:}\ \text{true})"] = (
    (sim_results["estimate"] - sim_results["true"]) / sim_results["se"]
)
display_table(sim_results, fmt={
    "true": "{:.4f}", "estimate": "{:.4f}", "se": "{:.6f}", r"t\ (H_0{:}\ \text{true})": "{:.2f}",
})
```

Both estimates fall within one or two standard errors of the true values, confirming that the likelihood implementation is correct.
With large simulated samples, the standard error for $\beta$ can be extremely small, so even a tiny absolute deviation can generate a $t$ statistic around 1--2.

The deviation of $\hat\alpha$ from the true value reflects normal sampling variation: with finite data, the cross-equation restriction that identifies $\alpha$ is estimated with noise because the predictable component of returns is small relative to total return variation.

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

The unrestricted benchmark is a Gaussian VAR for $\mathbf{Y}_t = (X_t, R_t)^\top$ with free coefficients on all lags in both equations.

```{code-cell} ipython3
def estimate_unrestricted_var(data, n_lags):
    """
    Estimate an unrestricted Gaussian VAR for Y_t = [X_t, R_t].
    """
    y_t, y_lag = build_lagged_data(data, n_lags)
    n_obs = y_t.shape[0]
    x = np.column_stack([np.ones(n_obs), y_lag])
    coef = np.linalg.lstsq(x, y_t, rcond=None)[0]
    resid = y_t - x @ coef
    Σ = resid.T @ resid / n_obs

    try:
        chol = cholesky(Σ, lower=True)
        log_det = 2.0 * np.sum(np.log(np.diag(chol) + 1e-16))
        std_resid = solve_triangular(chol, resid.T, lower=True).T
        quad_form = np.sum(std_resid ** 2)
    except (LinAlgError, ValueError):
        return {
            "coef": coef,
            "Σ": np.full((2, 2), np.nan),
            "residuals": resid,
            "loglike": -np.inf,
            "converged": False,
            "n_obs": int(n_obs),
        }

    d = y_t.shape[1]
    loglike = float(-0.5 * n_obs * d * np.log(2.0 * np.pi) - 0.5 * n_obs * log_det - 0.5 * quad_form)

    return {
        "coef": coef,
        "Σ": Σ,
        "residuals": resid,
        "loglike": loglike,
        "converged": True,
        "n_obs": int(n_obs),
    }
```

```{code-cell} ipython3
def run_unrestricted_var_by_lag(data, lags=(2, 4, 6)):
    """
    Estimate unrestricted VAR models by lag length.
    """
    rows = []
    fits = {}

    for lag in lags:
        fit = estimate_unrestricted_var(data, n_lags=lag)
        fits[lag] = fit
        rows.append(
            {
                "n_lags": lag,
                "loglike": fit["loglike"],
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
def predictability_metrics(data, restricted_fit, unrestricted_fit, n_lags):
    """
    Compute predictable-component metrics and unrestricted VAR R^2 values.
    """
    y_t, y_lag = build_lagged_data(data, n_lags)
    parsed = unpack_parameters(restricted_fit["params"], n_lags)
    α = float(parsed["α"])
    β = float(parsed["β"])
    σ_x = float(parsed["σ_x"])
    σ_r = float(parsed["σ_r"])
    cov_xr = float(parsed["cov_xr"])
    μ_x = float(parsed["μ_x"])
    a_lags = np.asarray(parsed["a_lags"])

    pred_x = y_lag @ a_lags + μ_x
    σ_u2 = α ** 2 * σ_x ** 2 + σ_r ** 2 + 2.0 * α * cov_xr
    pred_r = -α * pred_x - np.log(β) - 0.5 * σ_u2

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

From {eq}`hs83-cond-mean`, the conditional mean of asset $i$'s log return is $E_{t-1}[R_{it}] = -\alpha\, E_{t-1}[X_t] - \log\beta - \sigma_i^2/2$.

The term $-\alpha\, E_{t-1}[X_t] - \log\beta$ is common to all assets, so it cancels in the difference:

$$
E_{t-1}[R_{it} - R_{jt}] = \frac{\sigma_j^2 - \sigma_i^2}{2},
$$

which is a constant that does not depend on time-$(t-1)$ information.

Return differences should therefore be unpredictable if the model is correct, regardless of the values of $\alpha$ and $\beta$.

These tests avoid the need to measure consumption, at the cost of losing the ability to identify $\alpha$ and $\beta$.

{cite:t}`hansen1983stochastic` report that the return-difference restrictions are strongly rejected for models with multiple stock returns, providing substantial evidence against the CRRA-lognormal specification even when consumption measurement problems are eliminated.

The code below is an illustration of this logic on simulated data; reproducing the paper's empirical return-difference tables requires estimating multi-asset systems that are outside this lecture's scope.

```{code-cell} ipython3
def simulate_multi_asset_nominal_returns(
    n_obs,
    n_assets=3,
    α_true=-1.0,
    β_true=0.993,
    seed=0,
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
    log_returns = -np.log(β_true) - α_true * x[:, None] + eps \
            - 0.5 * sigmas[None, :] ** 2
    return x, log_returns


def return_difference_test(log_returns, n_lags=2):
    """
    Test predictability of pairwise log-return differences.
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
        z[:, 1 + j * n_assets : 1 + (j + 1) * n_assets] = log_returns[
            n_lags - j : t_obs - 1 - j, :
        ]

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

We run `return_difference_test` on simulated data with $m = 3$ assets, giving $\binom{3}{2} = 3$ pairs.

For each pair, the function regresses the return spread on a constant and `n_lags` lags of all asset returns, then tests whether the slope coefficients are jointly zero using a Wald $\chi^2$ statistic

```{code-cell} ipython3
_, sim_log_returns = simulate_multi_asset_nominal_returns(
    n_obs=1500,
    n_assets=3,
    α_true=-1.0,
    β_true=0.993,
    seed=0,
)

spread_test = return_difference_test(sim_log_returns, n_lags=2)
spread_pretty = spread_test.rename(columns={
    "wald_chi2": r"\chi^2", "p_value": "p", "mean_spread": r"\overline{\Delta R}",
})
display_table(spread_pretty, fmt={
    r"\chi^2": "{:.3f}",
    "p": "{:.3f}",
    r"\overline{\Delta R}": "{:.5f}",
})
```

Large $p$-values confirm that return differences are unpredictable in this simulation, exactly as the model predicts.

## Empirical MLE estimation

We now apply the maximum likelihood estimator of {cite:t}`hansen1983stochastic` to the historical sample, following the format of Table 1 in {cite:t}`hansen1983stochastic`.

By exploiting lognormality, the MLE should yield precise estimates when the assumption is correct.

```{code-cell} ipython3
lags = (2, 4, 6)

emp_frame, emp_data, source = get_estimation_data()
```

The following table reports the restricted MLE estimates of $\hat\alpha$ and $\hat\beta$ by lag length

```{code-cell} ipython3
emp_log_data = to_mle_array(emp_data)
mle_table, mle_fits = run_mle_by_lag(
    emp_log_data, lags=lags, verbose=False
)
mle_pretty = mle_table.rename(columns={
    "α_hat": r"\hat{\alpha}", "se_α": r"\mathrm{se}(\hat{\alpha})",
    "β_hat": r"\hat{\beta}", "se_β": r"\mathrm{se}(\hat{\beta})",
    "loglike": "logL", "n_obs": "T",
})
display_table(mle_pretty, fmt={
    r"\hat{\alpha}": "{:.4f}", r"\mathrm{se}(\hat{\alpha})": "{:.4f}",
    r"\hat{\beta}": "{:.4f}", r"\mathrm{se}(\hat{\beta})": "{:.4f}",
    "logL": "{:.1f}", "T": "{:.0f}",
})
```

The table reports $\hat\alpha$ and $\hat\beta$ by lag length for the sample used in the code cell above.

For comparison, {cite:t}`hansen1983stochastic` report $\hat\alpha$ values of $-0.32$ to $-1.25$ (standard errors $0.65$ to $0.83$) for the value-weighted return with nondurables consumption.

Our numbers are very close to the paper's.

In risk-aversion units, this corresponds to $-\hat\alpha$ between $0.32$ and $1.25$.

We now compute the predictability summaries that are central to {cite:t}`hansen1983stochastic`

```{code-cell} ipython3
unres_table, unres_fits = run_unrestricted_var_by_lag(
    emp_log_data,
    lags=lags,
)
```

```{code-cell} ipython3
:tags: [hide-input]

unres_pretty = unres_table.rename(columns={"loglike": "logL", "n_obs": "T"})
display_table(unres_pretty, fmt={
    "logL": "{:.1f}", "T": "{:.0f}",
})

pred_rows = []
for lag in lags:
    metrics = predictability_metrics(
        emp_log_data,
        restricted_fit=mle_fits[lag],
        unrestricted_fit=unres_fits[lag],
        n_lags=lag,
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
    "r2_x_unrestricted": r"$R_X^2$",
    "r2_r_unrestricted": r"$R_R^2$",
    "T": "T",
})
display_table(pred_pretty, fmt={
    r"\hat{\alpha}": "{:.4f}",
    r"\mathrm{Var}(\hat E[X_t\mid\psi_{t-1}])": "{:.6f}",
    r"\mathrm{Var}(\hat E[R_t\mid\psi_{t-1}])": "{:.6f}",
    r"\hat{\alpha}^2\,\mathrm{Var}(\hat E[X_t\mid\psi_{t-1}])": "{:.6f}",
    r"$R_X^2$": "{:.4f}",
    r"$R_R^2$": "{:.4f}",
    "T": "{:.0f}",
})
```

The column $\hat\alpha^2 \operatorname{Var}(\hat E[X_t \mid \psi_{t-1}])$ should equal $\operatorname{Var}(\hat E[R_t \mid \psi_{t-1}])$ if the restriction {eq}`hs83-var-pred` holds.

The columns  $R_X^2$ and $R_R^2$ match the paper convention of reporting the values from the unrestricted VAR projections.

In {cite:t}`hansen1983stochastic`, $R_R^2$ is small ($0.02$ to $0.06$) even when $R_X^2$ is nontrivial, highlighting that most stock-return variation is unpredictable.

This is also consistent with our run.

We now test the Euler-equation restrictions formally by comparing the restricted triangular system to an unrestricted VAR.

```{code-cell} ipython3
lr_hs83 = restricted_vs_unrestricted_lr(mle_fits, unres_fits, lags=lags)
lr_pretty = lr_hs83.rename(columns={
    "lr_stat": "LR",
    "chi2_cdf": "chi2.cdf(LR,df)",
    "p_value": "p(LR)",
    "df": "df",
    "T": "T",
})
display_table(lr_pretty, fmt={
    "LR": "{:.3f}",
    "chi2.cdf(LR,df)": "{:.3f}",
    "p(LR)": "{:.3f}",
    "df": "{:.0f}",
    "T": "{:.0f}",
})

sig_level = 0.05
rejected_lags = [int(lag) for lag in lr_hs83.index if lr_hs83.loc[lag, "p_value"] < sig_level]
not_rejected_lags = [int(lag) for lag in lr_hs83.index if lr_hs83.loc[lag, "p_value"] >= sig_level]
print(f"Not rejected at 5%: {not_rejected_lags if not_rejected_lags else 'none'}")
```

We find that the LR test does not reject the model for the value-weighted return, consistent with {cite:t}`hansen1983stochastic`.

This means that the Euler-equation restrictions are not strongly contradicted by the data.

### Treasury bill estimation

We now repeat the estimation using the 1-month Treasury bill return in place of the stock return.

{cite:t}`hansen1983stochastic` find that the model is strongly rejected for Treasury bills (Table 4 of their paper).

The nearly predictable, low-volatility behavior of T-bill returns is difficult to reconcile with the volatile, largely unpredictable behavior of consumption growth.

```{code-cell} ipython3
tbill_frame, tbill_data, tbill_source = get_tbill_estimation_data()
```

The following table reports the restricted MLE estimates for Treasury bill returns by lag length.

```{code-cell} ipython3
tbill_log_data = to_mle_array(tbill_data)
tbill_mle_table, tbill_mle_fits = run_mle_by_lag(
    tbill_log_data, lags=lags, verbose=False
)
```

```{code-cell} ipython3
:tags: [hide-input]

tbill_mle_pretty = tbill_mle_table.rename(columns={
    "α_hat": r"\hat{\alpha}", "se_α": r"\mathrm{se}(\hat{\alpha})",
    "β_hat": r"\hat{\beta}", "se_β": r"\mathrm{se}(\hat{\beta})",
    "loglike": "logL", "n_obs": "T",
})
display_table(tbill_mle_pretty, fmt={
    r"\hat{\alpha}": "{:.4f}", r"\mathrm{se}(\hat{\alpha})": "{:.4f}",
    r"\hat{\beta}": "{:.4f}", r"\mathrm{se}(\hat{\beta})": "{:.4f}",
    "logL": "{:.1f}", "T": "{:.0f}",
})
```

The LR test below remains the main specification check in the paper

```{code-cell} ipython3
tbill_unres_table, tbill_unres_fits = run_unrestricted_var_by_lag(
    tbill_log_data,
    lags=lags,
)

tbill_unres_pretty = tbill_unres_table.rename(columns={"loglike": "logL", "n_obs": "T"})
display_table(tbill_unres_pretty, fmt={
    "logL": "{:.1f}", "T": "{:.0f}",
})
```

```{code-cell} ipython3
:tags: [hide-input]

tbill_pred_rows = []
for lag in lags:
    metrics = predictability_metrics(
        tbill_log_data,
        restricted_fit=tbill_mle_fits[lag],
        unrestricted_fit=tbill_unres_fits[lag],
        n_lags=lag,
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
    "r2_x_unrestricted": r"$R_X^2$",
    "r2_r_unrestricted": r"$R_R^2$",
    "T": "T",
})
display_table(tbill_pred_pretty, fmt={
    r"\hat{\alpha}": "{:.4f}",
    r"\mathrm{Var}(\hat E[X_t\mid\psi_{t-1}])": "{:.6f}",
    r"\mathrm{Var}(\hat E[R_t\mid\psi_{t-1}])": "{:.6f}",
    r"\hat{\alpha}^2\,\mathrm{Var}(\hat E[X_t\mid\psi_{t-1}])": "{:.6f}",
    r"$R_X^2$": "{:.4f}",
    r"$R_R^2$": "{:.4f}",
    "T": "{:.0f}",
})
```

The following table reports the likelihood ratio tests for the Treasury bill model.

```{code-cell} ipython3
tbill_lr = restricted_vs_unrestricted_lr(tbill_mle_fits, tbill_unres_fits, lags=lags)
```

```{code-cell} ipython3
:tags: [hide-input]
tbill_lr_pretty = tbill_lr.rename(columns={
    "lr_stat": "LR",
    "chi2_cdf": "chi2.cdf(LR,df)",
    "p_value": "p(LR)",
    "df": "df",
    "T": "T",
})
display_table(tbill_lr_pretty, fmt={
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

## Residual diagnostics

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
```

The following table reports Jarque-Bera normality tests and Durbin-Watson serial correlation statistics for the restricted-model residuals.

```{code-cell} ipython3
if diag_fit["converged"] and resid is not None:
    diag = residual_diagnostics(resid)
    diag_df = pd.DataFrame({
        "JB stat": [diag["consumption_jb_stat"], diag["return_jb_stat"]],
        "JB p-val": [diag["consumption_jb_pval"], diag["return_jb_pval"]],
        "DW": [diag["consumption_dw"], diag["return_dw"]],
    }, index=pd.Index(["consumption", "return"], name="series"))
    display_table(diag_df, fmt={
        "JB stat": "{:.2f}", "JB p-val": "{:.4f}", "DW": "{:.3f}",
    })
```

The residual time-series plots reveal periods of unusually large innovations (volatility clustering), while the histograms show departures from the Gaussian bell curve.

The Jarque-Bera statistics are large for both series, decisively rejecting the normality assumption that underlies the likelihood.

The Durbin-Watson statistics are close to 2 for consumption and return residuals, suggesting that serial correlation is not a major concern.

Rejection of normality is expected given the well-known fat tails in financial returns.

This is precisely the situation where the robustness of GMM (see {doc}`hansen_singleton_1982`) becomes valuable.

We will say more about this in that lecture.

## Connection to the equity premium puzzle

The empirical findings of {cite:t}`hansen1983stochastic` contain the key ingredients of what {cite:t}`MehraPrescott1985` would formalize as the **equity premium puzzle**.

Let us assemble the evidence:

- *Low estimated risk aversion:* The estimated $\hat\alpha$ values (and thus risk aversion $-\hat\alpha$) from the table above is similar to the numbers in {cite:t}`hansen1983stochastic`, who report $\hat\alpha$ between $-0.32$ and $-1.25$.

- *Tiny return predictability:* The unrestricted-VAR $R_R^2$ values have a similar range with the 0.02 to 0.06 range in {cite:t}`hansen1983stochastic`, confirming that the predictable component of stock returns is small relative to the unpredictable component.

- *Strong rejection for Treasury bills.* The Euler-equation restrictions are decisively rejected for Treasury bill returns, just as in Table 4 of {cite:t}`hansen1983stochastic`;
the model cannot reconcile the smooth, nearly predictable behavior of Treasury bill returns with the volatile behavior of consumption growth — a precursor to the *risk-free rate puzzle* of {cite:t}`Weil_1989`.

The CRRA-lognormal model cannot explain the cross-section of expected returns.

{cite:t}`MehraPrescott1985` crystallized these findings into a sharp quantitative statement.

In a calibrated version of the consumption-based model with CRRA utility, they showed that for relative risk aversion $\gamma_{\text{MP}}$ in the range of 0 to 10 (the range considered "reasonable" by most economists), the model could not simultaneously match:

1. the average annual equity premium of about 6\%,
2. the average annual risk-free rate of about 1\%.

Matching the equity premium requires very high $\gamma_{\text{MP}}$ (above 30 in many calibrations), while matching the risk-free rate requires $\gamma_{\text{MP}}$ near 1.

This is exactly the tension visible in the {cite:t}`hansen1983stochastic` estimates: the implied risk aversion $-\hat\alpha$ is too low to generate a large equity premium.

One question remains for a careful reader:
*If the LR test does not reject for aggregate stock returns, doesn't that mean the model works — and therefore the equity premium puzzle does not exist?*

The answer is no: non-rejection reflects **low test power**, not model adequacy.

The LR test examines the cross-equation restrictions on the *predictable* variation in returns and consumption growth.
When aggregate stock returns are nearly unpredictable ($R_R^2 \approx 0.02$–$0.06$ in Table 1 of {cite:t}`hansen1983stochastic`), there is almost no predictable variation to constrain.
The proportionality restriction is nearly vacuous, and the test simply cannot detect violations.

{cite:t}`hansen1983stochastic` note a related pattern (p. 258): as NLAG increases, the probability values of the $\chi^2$ statistics tend to *decline*.
They explain this by examining the unrestricted autoregression estimates (their Table 2): values of the autoregressive coefficients of consumption growth and returns beyond the second lag are "not very useful in forecasting consumption."
Consequently, the additional cross-equation restrictions imposed at higher lag orders fall on coefficients that are near zero, and "there are relatively smaller increases in the $\chi^2$ statistics" relative to the increase in degrees of freedom.

The equity premium puzzle, by contrast, is about the **unconditional mean level** of excess stock returns — roughly 6\% per year — being too high for the low observed variability of consumption growth.
This is a statement about *levels*, not about time-series dynamics.
Since the LR test only examines whether the *predictable variation* in returns obeys proportionality, it is the wrong tool for detecting the equity premium puzzle.

The $\hat\alpha$ estimates tell the same story from a different angle: they center around $-0.3$ to $-1.5$ (risk aversion of 0.3 to 1.5), but with standard errors so large that the data cannot pin down risk aversion.
The model "passes" not because $\alpha \approx -1$ is right, but because the data cannot distinguish it from $\alpha \approx -10$.
{cite:t}`MehraPrescott1985` later showed that $\alpha$ near $-10$ or beyond is needed to match the equity premium — a value within Hansen and Singleton's confidence intervals but far from their point estimates.

For Treasury bills the situation is strikingly different.
{cite:t}`hansen1983stochastic` call the $\chi^2$ statistics "the most dramatic differences" between the stock return and Treasury bill results (p. 261): "For the Treasury bill models, the marginal significance levels are essentially zero, providing strong evidence against the restrictions."
Since $R_R^2$ for Treasury bills is much larger ($\approx 0.13$–$0.20$ in their Table 4), the proportionality restriction actually binds, the test has power, and it rejects decisively.