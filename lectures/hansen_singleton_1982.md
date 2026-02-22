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

(hansen_singleton_1982)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Estimating Euler Equations by GMM

```{index} single: Asset Pricing; GMM Estimation
```

```{contents} Contents
:depth: 2
```

## Overview

This lecture implements the Generalized Method of Moments (GMM) estimator introduced by {cite:t}`hansen1982generalized` to estimate the consumption-based Euler equation for asset pricing.

The economic model features a representative consumer with CRRA preferences choosing how much to consume and how to allocate wealth across traded assets.

The first-order conditions deliver stochastic Euler equations that relate consumption growth, asset returns, and preference parameters.

{cite:t}`hansen1982generalized` propose an estimation strategy that works directly with the Euler equation's orthogonality conditions.

A key attraction of GMM is what it does *not* require:

- no distributional assumptions on the joint process of consumption and returns,
- no closed-form solution for equilibrium prices,
- no complete specification of the production technology or other "forcing variables."

The econometrician needs only the Euler equation itself and a set of predetermined instruments.

This makes GMM applicable to a broad class of nonlinear rational expectations models.

The cost is that, by ignoring potential information about the distribution of the data, the estimator may be less precise than methods that exploit that information.

The preceding lecture {doc}`hansen_singleton_1983` shows what can be gained by imposing additional distributional structure, and connects the empirical findings to the equity premium puzzle.

The findings there, including low estimated risk aversion, strong rejection for Treasury bills, and departures from lognormality, motivate the robust GMM approach developed here.

We cover:

- the consumption CRRA Euler equation and its stochastic discount factor representation
- GMM estimation using lagged instruments and HAC covariance, with Hansen's $J$ test for overidentifying restrictions
- the multi-period case with overlapping horizons and finite-order MA covariance
- simulation exercises verifying that GMM recovers known parameters
- empirical estimation on monthly FRED consumption and the Ken French CRSP market proxy

In addition to what comes with Anaconda, this lecture requires `pandas-datareader`.

```{code-cell} ipython3
:tags: [hide-output]

!pip install pandas-datareader
```

Let's start by importing packages and defining a helper for displaying tables in LaTeX format

```{code-cell} ipython3
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Math
from numba import njit
from pandas_datareader import data as web
from scipy import stats
from scipy.optimize import minimize
from statsmodels.sandbox.regression import gmm
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


## The economic model

We consider a single-good economy of identical consumers whose preferences are of the CRRA type, following {cite:t}`hansen1982generalized` and {cite:t}`hansen1983stochastic`.

The representative consumer chooses stochastic consumption and investment plans to maximize

```{math}
:label: hs82-problem

\max E_0 \sum_{t=0}^{\infty} \beta^t u(C_t)
```

where $C_t$ is consumption in period $t$, $\beta \in (0,1)$ is a subjective discount factor, and $u(\cdot)$ is a strictly concave period utility function.

The consumer trades $N$ assets with potentially different maturities.

Let $Q_{jt}$ denote the quantity of asset $j$ held at the end of date $t$, $P_{jt}$ its price at date $t$, and $R_{jt}$ the date-$t$ payoff from holding one unit of an $M_j$-period asset purchased at date $t - M_j$.

Feasible consumption and investment plans must satisfy the sequence of budget constraints

```{math}
:label: hs82-budget

C_t + \sum_{j=1}^{N} P_{jt} Q_{jt} \leq \sum_{j=1}^{N} R_{jt} Q_{jt-M_j} + W_t,
```

where $W_t$ is real labor income at date $t$.

We specialize to CRRA preferences

```{math}
:label: hs82-crra

u(C_t) = \frac{C_t^{1-\gamma}}{1-\gamma}, \quad \gamma > 0,
```

where $\gamma$ is the coefficient of relative risk aversion.

The maximization of {eq}`hs82-problem` subject to {eq}`hs82-budget` gives the first-order necessary conditions (see {cite:t}`Lucas1978`, {cite:t}`Brock1982`, {cite:t}`PrescottMehra1980`)

```{math}
:label: hs82-general-euler

P_{jt} u'(C_t) = \beta^{M_j} E_t\!\left[R_{jt+M_j} u'(C_{t+M_j})\right], \quad j = 1, \ldots, N.
```

When asset $j$ is a one-period stock ($M_j = 1$) with $R_{jt+1} = P_{jt+1} + D_{jt+1}$ where $D_{jt}$ is the dividend, substituting the CRRA marginal utility into {eq}`hs82-general-euler` and dividing both sides by $P_{jt} u'(C_t)$ yields the Euler equation

```{math}
:label: hs82-euler

E_t\!\left[\beta \left(\frac{C_{t+1}}{C_t}\right)^{-\gamma} R_{t+1}^i\right] = 1,
```

where $R_{t+1}^i$ is the gross real return on asset $i$.

We define the **stochastic discount factor** $M_{t+1}(\theta) = \beta (C_{t+1}/C_t)^{-\gamma}$ with parameter vector $\theta = (\gamma, \beta)$.

In this notation the Euler equation becomes $E_t[M_{t+1}(\theta) R_{t+1}^i - 1] = 0$.

Equation {eq}`hs82-euler` is the central object of both {cite:t}`hansen1982generalized` and {cite:t}`hansen1983stochastic`.

It holds for every traded asset and at every date, and it depends on observable quantities (consumption growth and returns) and unknown preference parameters ($\gamma$ and $\beta$).

The challenge that {cite:t}`hansen1982generalized` address is how to estimate $\theta$ from {eq}`hs82-euler` without specifying the rest of the economic environment.

## From conditional to unconditional moments

A natural approach to estimating $\theta$ from {eq}`hs82-euler` would be to specify the entire economic environment, solve for equilibrium, and apply maximum likelihood.

{cite:t}`hansen1982generalized` argue that this is impractical for most nonlinear models because it requires strong assumptions about the stochastic properties of "forcing variables" (technology shocks, labor income) and the production technology.

Their alternative is to work directly with the Euler equation's implications for observable moments.

The Euler equation {eq}`hs82-euler` states that $E_t[M_{t+1}(\theta_0) R_{t+1}^i - 1] = 0$.

Let $z_t$ denote any $q$-dimensional vector of variables that are in the agent's time-$t$ information set and observed by the econometrician.

Because $z_t$ is known at time $t$, the law of iterated expectations implies

```{math}
:label: hs82-uncond

E\!\left[\left(M_{t+1}(\theta_0)R_{t+1}^i - 1\right) \otimes z_t\right] = 0.
```

Equation {eq}`hs82-uncond` converts a conditional rational-expectations restriction into a set of *unconditional* moment conditions that can be estimated from sample averages.

Each element of $z_t$ generates one orthogonality condition, so with $q$ instruments and $m$ Euler equations we obtain $mq$ moment restrictions for estimating the parameter vector $\theta$.

For one return and $p$ lags of instruments, we use

```{math}
:label: hs82-instruments

z_t = \left[1, R_t, g_t, R_{t-1}, g_{t-1}, \ldots, R_{t-p+1}, g_{t-p+1}\right]^\top,
```

where $g_t = C_t / C_{t-1}$ is gross consumption growth.

An important practical point from {cite:t}`hansen1982generalized` is that instruments need only be "predetermined" relative to the time-$t$ information set: they need not be exogenous in the regression sense.

Current and lagged values of consumption growth and returns are valid instruments because they are known to agents when portfolio decisions are made.

More generally, {cite:t}`hansen1982generalized` write the first-order condition as $E_t[h(x_{t+n}, b_0)] = 0$ with arbitrary lead $n$, where $x_{t+n}$ is a vector of observables dated $t+n$ and $b_0$ is the true parameter vector.

The disturbance $u_{t+n} = h(x_{t+n}, b_0)$ is serially uncorrelated when $n=1$ (the one-period stock case) but has moving-average structure of order $n-1$ when $n>1$ (the multi-period bond case).

Stacking moment conditions via the Kronecker product gives $f(x_{t+n}, z_t, b) = h(x_{t+n}, b) \otimes z_t$, a vector of dimension $mq$.

The resulting unconditional restriction $E[f(x_{t+n}, z_t, b_0)] = 0$ nests both the single-return one-period Euler equation and the multi-maturity asset pricing restrictions in {cite:t}`hansen1982generalized`.

The code below encodes the orthogonality condition and lagged instrument vector exactly as in equations {eq}`hs82-uncond` and {eq}`hs82-instruments`.

```{code-cell} ipython3
def euler_error_horizon(params, exog, horizon=1):
    """
    Compute Euler-equation pricing errors for (γ, β) at a given horizon.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one.")
    γ, β = params
    gross_return = exog[:, 0]
    gross_cons_growth = exog[:, 1]
    return (β ** horizon) * gross_cons_growth ** (-γ) * gross_return - 1.0


def euler_error(params, exog):
    """
    One-period Euler-equation pricing errors for (γ, β).
    """
    return euler_error_horizon(params, exog, horizon=1)
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

When the asset has maturity $n > 1$, the Euler equation involves $n$-period compounded returns and consumption growth.

For CRRA preferences, the $n$-period Euler restriction is

```{math}
:label: hs82-euler-n

E_t\!\left[\beta^n \left(\frac{C_{t+n}}{C_t}\right)^{-\gamma} R_{t,t+n}^i\right] = 1.
```

If one instead uses $\beta$ (not $\beta^n$) in {eq}`hs82-euler-n`, the estimated discount parameter is interpreted as $\tilde\beta = \beta^n$ rather than the one-period $\beta$.

The following helper builds these overlapping horizon aggregates, as needed for the multi-period case in {cite:t}`hansen1982generalized`.

```{code-cell} ipython3
def build_overlapping_horizon_data(data, horizon):
    """
    Build overlapping horizon aggregates for [gross return, gross consumption growth].
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one.")
    if data.shape[0] < horizon:
        raise ValueError("Sample size must be at least horizon.")

    t_obs = data.shape[0]
    n_obs = t_obs - horizon + 1
    out = np.empty((n_obs, 2))

    for t in range(n_obs):
        window = data[t : t + horizon, :]
        out[t, 0] = np.prod(window[:, 0])
        out[t, 1] = np.prod(window[:, 1])

    return out
```

When $n > 1$, overlapping horizons induce serial dependence in the Euler disturbance $u_{t+n} = h(x_{t+n}, b_0)$.

The disturbance has MA($n-1$) structure because the $n$-period return windows overlap.

{cite:t}`hansen1982generalized` show that the optimal weighting matrix in this case involves a finite autocovariance sum truncated at order $n-1$, rather than an infinite HAC kernel.

We implement this directly as a finite-order covariance estimator.

```{code-cell} ipython3
def finite_ma_covariance(moment_series, ma_order):
    """
    Estimate S = Gamma_0 + sum_{j=1}^{ma_order}(Gamma_j + Gamma_j.T) for moment vectors.
    """
    if ma_order < 0:
        raise ValueError("ma_order must be nonnegative.")
    if moment_series.ndim != 2:
        raise ValueError("moment_series must be 2D.")

    t_obs, n_mom = moment_series.shape
    if t_obs <= ma_order:
        raise ValueError("Need more observations than ma_order.")

    centered = moment_series - moment_series.mean(axis=0, keepdims=True)
    s_hat = centered.T @ centered / t_obs

    for j in range(1, ma_order + 1):
        gamma_j = centered[j:, :].T @ centered[:-j, :] / t_obs
        s_hat += gamma_j + gamma_j.T

    ridge = 1e-8 * np.eye(n_mom)
    return s_hat + ridge
```

The estimation procedure in {cite:t}`hansen1982generalized` is a two-step generalized instrumental variables procedure.

In the first step, we minimize the GMM criterion with a suboptimal weighting matrix (the identity) to obtain a consistent preliminary estimate $b_T$.

In the second step, we use $b_T$ to construct the estimated optimal weighting matrix $\hat W_T^* = \hat S_T^{-1}$ and re-minimize the criterion.

The following function implements this logic.

```{code-cell} ipython3
def two_step_gmm(data, n_lags, ma_order=0, horizon=1, start_params=None):
    """
    Two-step GMM with finite-order covariance.

    The Euler error uses β**horizon, consistent with the n-period
    restriction in Hansen and Singleton (1982).
    """
    if start_params is None:
        start_params = np.array([1.0, 0.99])

    _, exog, instruments = build_gmm_arrays(data, n_lags)
    n_obs = exog.shape[0]

    def sample_moments(params):
        err = euler_error_horizon(params, exog, horizon=horizon)
        return err[:, None] * instruments

    def objective(params, weight_matrix):
        g_bar = sample_moments(params).mean(axis=0)
        return float(g_bar @ weight_matrix @ g_bar)

    q = instruments.shape[1]
    w_identity = np.eye(q)
    step1 = minimize(objective, x0=start_params, args=(w_identity,), method="BFGS")
    params1 = step1.x

    m1 = sample_moments(params1)
    s_hat = finite_ma_covariance(m1, ma_order=ma_order)
    w_opt = np.linalg.pinv(s_hat)

    step2 = minimize(objective, x0=params1, args=(w_opt,), method="BFGS")
    params2 = step2.x
    g2 = sample_moments(params2).mean(axis=0)
    j_stat = float(n_obs * (g2 @ w_opt @ g2))
    df = instruments.shape[1] - len(params2)
    j_prob = float(stats.chi2.cdf(j_stat, df=df)) if df > 0 else np.nan
    p_value = float(1.0 - j_prob) if df > 0 else np.nan

    return {
        "params_step1": params1,
        "params_step2": params2,
        "weight_opt": w_opt,
        "j_stat": j_stat,
        "j_df": int(df),
        "j_prob": j_prob,
        "j_pval": p_value,
        "n_obs": int(n_obs),
        "success": bool(step2.success),
    }
```

This gives a transparent reference algorithm for the finite-order weighting in {cite:t}`hansen1982generalized`.

Our baseline estimator for the one-period case uses statsmodels' nonlinear IV machinery, which implements the same two-step logic with a Newey-West HAC kernel.

## Data

Both {cite:t}`hansen1982generalized` and {cite:t}`hansen1983stochastic` use monthly data on real per capita consumption (nondurables) and stock returns from CRSP for the period 1959:2 through 1978:12.

To stay close to their empirical setup with open data, this lecture uses:

- FRED nondurables consumption and deflator series,
- Ken French's CRSP value-weighted market proxy (`Mkt-RF + RF`) for stock returns,
- sample window 1959:2 through 1978:12.

Exact CRSP NYSE replication is not open-data feasible, so coefficient-by-coefficient equality with the published tables is not expected.

We also build a simulator that generates synthetic return-growth pairs satisfying the Euler equation by construction, so that we can verify our estimators recover known parameters before applying them to actual data.

```{code-cell} ipython3
FRED_CODES = {
    # HS82 divide by a Census population series; we offer both a total-population
    # proxy (POP) and the commonly used civilian noninstitutional 16+ series.
    "population_total": "POP",
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

The simulation block below produces synthetic return-growth pairs that satisfy the Euler equation by construction.

We generate log consumption growth from a stationary AR(1), compute the stochastic discount factor at known true parameters, and construct gross returns as $R_{t+1} = (1 + \eta_{t+1}) / M_{t+1}(\theta_0)$ where $\eta_{t+1}$ is an iid shock.

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

The empirical data loader below merges FRED consumption series with the Ken French CRSP market proxy.

{cite:t}`hansen1982generalized` construct *real per capita* consumption by dividing by a Census population series.

Our default uses `POP` as an open-data proxy for that population series; we also support `CNP16OV` (civilian noninstitutional population 16+) as an alternative.

Nominal returns are deflated by the nondurables price index to obtain real gross returns.

```{code-cell} ipython3
def load_hs_monthly_data(
    start="1959-02-01",
    end="1978-12-01",
    population_key="population_total",
    per_capita=True,
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
    if population_key not in fred.columns:
        raise KeyError(f"Missing population series '{population_key}' in FRED pull.")
    if per_capita:
        fred["consumption_per_capita"] = (
            fred["cons_nd_real_index"] / fred[population_key]
        )
    else:
        fred["consumption_per_capita"] = fred["cons_nd_real_index"]
    fred["gross_cons_growth"] = (
        fred["consumption_per_capita"] / fred["consumption_per_capita"].shift(1)
    )
    fred["gross_inflation_nd"] = (
        fred["cons_nd_price_index"] / fred["cons_nd_price_index"].shift(1)
    )

    ff = web.DataReader("F-F_Research_Data_Factors", "famafrench", fetch_start, fetch_end)[0].copy()
    ff.columns = [str(col).strip() for col in ff.columns]
    if ("Mkt-RF" not in ff.columns) or ("RF" not in ff.columns):
        raise KeyError("Fama-French data missing required columns: 'Mkt-RF' and 'RF'.")

    # Mkt-RF and RF are reported in percent per month.
    ff["gross_nom_return"] = 1.0 + (ff["Mkt-RF"] + ff["RF"]) / 100.0
    ff.index = ff.index.to_timestamp(how="end")
    ff.index = to_month_end(ff.index)

    out = fred.join(ff[["gross_nom_return"]], how="inner")
    out["gross_real_return"] = out["gross_nom_return"] / out["gross_inflation_nd"]
    out = out.loc[sample_start:sample_end].dropna()

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
    start="1959-02-01",
    end="1978-12-01",
    population_key="population_total",
    per_capita=True,
):
    """
    Return (dataframe, array, source_label) using observed data.
    """
    frame = load_hs_monthly_data(
        start=start,
        end=end,
        population_key=population_key,
        per_capita=per_capita,
    )
    data = frame[["gross_real_return", "gross_cons_growth"]].to_numpy()
    pop_label = "per-capita" if per_capita else "aggregate"
    source = (
        "Ken French CRSP market proxy (Mkt-RF + RF) + "
        f"FRED nondurables consumption ({pop_label}, {population_key})"
    )
    return frame, data, source
```

## GMM criterion and asymptotic theory

We now formalize the estimation procedure from Section 3 of {cite:t}`hansen1982generalized`.

Let $m_t(\theta) = (M_{t+1}(\theta) R_{t+1}^i - 1) \otimes z_t$ denote the vector of moment conditions at date $t$, and define the sample mean

```{math}
:label: hs82-sample-moments

g_T(\theta) = \frac{1}{T} \sum_{t=1}^T m_t(\theta).
```

If the model is correctly specified, $g_T(\theta_0)$ should be close to zero for large $T$.

We estimate $\theta$ by choosing the parameter vector that makes $g_T$ as close to zero as possible, measured by a quadratic form:

```{math}
:label: hs82-criterion

\hat\theta = \arg\min_\theta g_T(\theta)^\top W_T g_T(\theta)
```

where $W_T$ is a symmetric positive-definite weighting matrix.

Under regularity conditions given in {cite:t}`Hansen1982`, the GMM estimator is consistent, asymptotically normal, and has the sandwich covariance matrix

```{math}
:label: hs82-asymptotic

\sqrt{T}(\hat\theta-\theta_0) \Rightarrow N\!\left(0, (D^\top W D)^{-1} D^\top W S W D (D^\top W D)^{-1}\right),
```

where $D = E[\partial m_t(\theta_0)/\partial\theta^\top]$ is the Jacobian of the moment conditions, $S$ is the long-run covariance matrix of $m_t(\theta_0)$, and $W$ is the probability limit of $W_T$.

{cite:t}`hansen1982generalized` show that the optimal weighting matrix is $W^* = S^{-1}$, which yields the smallest asymptotic covariance matrix among all choices of $W$.

Under $W = S^{-1}$ the sandwich simplifies to $(D^\top S^{-1} D)^{-1}$.

When the number of moment conditions $q$ exceeds the number of parameters $k$, the model is overidentified and we can test whether the data are consistent with the maintained restrictions.

{cite:t}`hansen1982generalized` propose the $J$ test:

```{math}
:label: hs82-jtest

J_T = T\, g_T(\hat\theta)^\top \hat S^{-1} g_T(\hat\theta) \Rightarrow \chi^2_{q-k},
```

where $\hat S$ is a consistent estimator of $S$.

A large $J_T$ relative to $\chi^2_{q-k}$ critical values leads to rejection of the model's overidentifying restrictions.

For the multi-period case ($n > 1$), the optimal $S$ involves a finite autocovariance sum because the MA($n-1$) structure of $u_{t+n}$ means that $E[m_t m_{t-j}^\top] = 0$ for $|j| \geq n$:

```{math}
:label: hs82-finite-so

S_0 = \sum_{j=-n+1}^{n-1} E\!\left[f(x_{t+n}, z_t, b_0)\, f(x_{t+n-j}, z_{t-j}, b_0)^\top\right].
```

This finite-order structure is important because it means the covariance estimator does not require a bandwidth or kernel choice when the horizon $n$ is known.

## Covariance estimation and the choice of instruments

For the one-period Euler equation ($n = 1$), the disturbance $u_{t+1} = M_{t+1}(\theta_0) R_{t+1} - 1$ is a martingale difference sequence.

In this case the moment vector $m_t(\theta_0) = u_{t+1} \otimes z_t$ is serially uncorrelated, and the long-run covariance $S$ equals the contemporaneous variance $E[m_t m_t^\top]$.

The covariance estimator therefore requires no kernel or bandwidth choice.

We simply use the sample analogue $\hat S = T^{-1} \sum_t m_t m_t^\top$.

In our implementation below, we use a HAC (heteroskedasticity and autocorrelation consistent) estimator as a robustness device against possible mild serial dependence from time aggregation or measurement timing.

This is a modern precaution, not part of the original {cite:t}`hansen1982generalized` procedure, which exploits the known MA order directly as in {eq}`hs82-finite-so`.

The number of instrument lags $p$ determines how many orthogonality conditions we use and hence the power of the $J$ test.

{cite:t}`hansen1982generalized` report results for NLAG $= 1, 2, 4, 6$ and note that the number of overidentifying restrictions and the finite-sample behavior of the $J$ test both change with $p$.

```{code-cell} ipython3
def estimate_gmm(
    data,
    n_lags,
    start_params=None,
    use_hac=True,
    hac_maxlag=None,
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
        if hac_maxlag is None:
            hac_maxlag = max(1, int(np.floor(4.0 * (endog.shape[0] / 100.0) ** (2.0 / 9.0))))
        result = model.fit(
            start_params=start_params,
            maxiter=maxiter,
            optim_method="bfgs",
            optim_args={"disp": False},
            weights_method="hac",
            wargs={"maxlag": hac_maxlag},
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

{cite:t}`hansen1982generalized` emphasize that increasing NLAG adds more orthogonality conditions to the estimation, which can improve efficiency but also increases the number of overidentifying restrictions being tested.

We report estimates across several lag lengths to examine this tradeoff.

```{code-cell} ipython3
def run_gmm_by_lag(
    data,
    lags=(1, 2, 4, 6),
    use_hac=True,
    hac_maxlag=None,
):
    """
    Estimate GMM models by lag length and return a summary table.
    """
    rows = []
    results = {}

    for lag in lags:
        res = estimate_gmm(data, n_lags=lag, use_hac=use_hac, hac_maxlag=hac_maxlag)
        results[lag] = res
        j_stat, j_pval, j_df = res.jtest()
        j_prob = float(stats.chi2.cdf(j_stat, df=j_df)) if j_df > 0 else np.nan
        rows.append(
            {
                "n_lags": lag,
                "γ_hat": res.params[0],
                "se_γ": res.bse[0],
                "β_hat": res.params[1],
                "se_β": res.bse[1],
                "j_stat": j_stat,
                "j_prob": j_prob,
                "j_pval": j_pval,
                "j_df": int(j_df),
                "n_obs": int(res.nobs),
            }
        )

    table = pd.DataFrame(rows).set_index("n_lags")
    return table, results
```

A difficulty noted in both papers is that the preference parameters $\gamma$ and $\beta$ can be weakly identified.

The criterion surface may have elongated valleys where many parameter combinations fit the moments nearly equally well.

We compute the objective over a parameter grid to visualize the identification geometry.

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

## Simulation exercises

Before applying the estimator to real data, we verify that GMM recovers known parameters from simulated data.

This is essential because the estimator involves nonlinear optimization, and we want to confirm that the code correctly implements the econometric theory.

We set $\gamma = 2$ and $\beta = 0.995$ as the true parameters and generate 700 monthly observations from the Euler-consistent DGP.

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

We now estimate GMM across lag lengths, following the format of Table I in {cite:t}`hansen1982generalized`.

```{code-cell} ipython3
sim_table, sim_results = run_gmm_by_lag(sim_data, lags=(1, 2, 4, 6), use_hac=True)
sim_pretty = sim_table.rename(columns={
    "γ_hat": r"\hat{\gamma}", "se_γ": r"\mathrm{se}(\hat{\gamma})",
    "β_hat": r"\hat{\beta}", "se_β": r"\mathrm{se}(\hat{\beta})",
    "j_stat": "J", "j_prob": "Prob(J)", "j_pval": "p(J)", "j_df": "df", "n_obs": "T",
})
display_table(sim_pretty, title="GMM Simulation Results", fmt={
    r"\hat{\gamma}": "{:.4f}", r"\mathrm{se}(\hat{\gamma})": "{:.4f}",
    r"\hat{\beta}": "{:.4f}", r"\mathrm{se}(\hat{\beta})": "{:.4f}",
    "J": "{:.3f}", "Prob(J)": "{:.3f}", "p(J)": "{:.3f}", "df": "{:.0f}", "T": "{:.0f}",
})
```

GMM recovers the true $\gamma$ and $\beta$ across lag specifications, and the $J$ statistics are small with small `Prob(J)` values (equivalently large right-tail `p(J)`), confirming that the simulated moment conditions are not rejected.

To illustrate the multi-period case from Section 2 of {cite:t}`hansen1982generalized`, we aggregate to overlapping three-period returns and estimate with the finite-order covariance targeting appropriate for MA(2) disturbances.

```{code-cell} ipython3
horizon_n = 3
sim_data_n = build_overlapping_horizon_data(sim_data, horizon=horizon_n)
two_step = two_step_gmm(
    sim_data_n,
    n_lags=2,
    ma_order=horizon_n - 1,
    horizon=horizon_n,
)

print(f"Horizon n: {horizon_n}")
print(f"Step-2 converged: {two_step['success']}")
print(f"Step-2 gamma: {two_step['params_step2'][0]:.4f}")
print(f"Step-2 beta (one-period): {two_step['params_step2'][1]:.4f}")
print(f"Step-2 beta^{horizon_n}: {two_step['params_step2'][1] ** horizon_n:.4f}")
print(
    f"J({two_step['j_df']}): {two_step['j_stat']:.3f}, "
    f"Prob={two_step['j_prob']:.3f}, p={two_step['j_pval']:.3f}"
)

_, exog_n, _ = build_gmm_arrays(sim_data_n, n_lags=2)
acf_n = acf(
    euler_error_horizon(two_step["params_step2"], exog_n, horizon=horizon_n),
    nlags=6,
    fft=True,
)
print("Euler-error ACF lags 1-3:", ", ".join([f"{v:.3f}" for v in acf_n[1:4]]))
```

The ACF values at low lags confirm the MA(2) dependence in the overlapping-horizon disturbance, exactly as predicted by the theory in {cite:t}`hansen1982generalized`.

We now run a Monte Carlo exercise with 500 replications to visualize the finite-sample distribution of $\hat\gamma$, $\hat\beta$, and the $J$ statistic and verify that the asymptotic theory from Section 3 of {cite:t}`hansen1982generalized` provides a reasonable approximation.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Monte Carlo GMM sampling distributions
    name: fig-hs82-monte-carlo
---
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

## Empirical GMM estimation

We now apply GMM to observed data, following the empirical strategy of Section 5 in {cite:t}`hansen1982generalized`.

{cite:t}`hansen1982generalized` use monthly per capita consumption of nondurables (ND) and nondurables plus services (NDS) paired with the equally-weighted (EWR) and value-weighted (VWR) aggregate stock returns from CRSP, for 1959:2 through 1978:12.

We focus on the closest open-data analogue to their ND+VWR specification: FRED nondurables consumption and the Ken French CRSP value-weighted market proxy, on the same 1959:2--1978:12 window.

Even with matching dates, exact replication of the original CRSP NYSE series and historical data vintages is not open-data feasible, so some numerical differences remain expected.

We first examine the raw data moments.

```{code-cell} ipython3
LAGS = (1, 2, 4, 6)

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

The key feature of these data is the large gap between the volatility of returns and the volatility of consumption growth.

If you want to see how sensitive the $\hat\gamma$ estimates are to the population adjustment, rerun the block above with
`get_estimation_data(population_key="population_16plus")` (or with `per_capita=False` to drop the population adjustment entirely).

This is the empirical fact underlying the equity premium puzzle of {cite:t}`MehraPrescott1985`: matching the observed equity premium with CRRA preferences requires implausibly high risk aversion.

The weak consumption-return comovement compounds the difficulty.

We now estimate GMM with HAC and HC0 covariance matrices, reporting results for NLAG $= 1, 2, 4, 6$ as in Table I of {cite:t}`hansen1982generalized`.

```{code-cell} ipython3
gmm_hac_table, gmm_hac_results = run_gmm_by_lag(emp_data, lags=LAGS, use_hac=True)
gmm_hc0_table, _ = run_gmm_by_lag(emp_data, lags=LAGS, use_hac=False)

gmm_hac_pretty = gmm_hac_table.rename(columns={
    "γ_hat": r"\hat{\gamma}", "se_γ": r"\mathrm{se}(\hat{\gamma})",
    "β_hat": r"\hat{\beta}", "se_β": r"\mathrm{se}(\hat{\beta})",
    "j_stat": "J", "j_prob": "Prob(J)", "j_pval": "p(J)", "j_df": "df", "n_obs": "T",
})
gmm_hc0_pretty = gmm_hc0_table[["se_γ", "se_β", "j_stat", "j_prob", "j_pval", "j_df"]].rename(columns={
    "se_γ": r"\mathrm{se}(\hat{\gamma})", "se_β": r"\mathrm{se}(\hat{\beta})",
    "j_stat": "J", "j_prob": "Prob(J)", "j_pval": "p(J)", "j_df": "df",
})

gmm_fmt = {
    r"\hat{\gamma}": "{:.4f}", r"\mathrm{se}(\hat{\gamma})": "{:.4f}",
    r"\hat{\beta}": "{:.4f}", r"\mathrm{se}(\hat{\beta})": "{:.4f}",
    "J": "{:.3f}", "Prob(J)": "{:.3f}", "p(J)": "{:.3f}", "df": "{:.0f}", "T": "{:.0f}",
}
display_table(gmm_hac_pretty, title="GMM Estimates (HAC Covariance)", fmt=gmm_fmt)
display_table(gmm_hc0_pretty, title="GMM Estimates (HC0 Covariance)", fmt={
    r"\mathrm{se}(\hat{\gamma})": "{:.4f}", r"\mathrm{se}(\hat{\beta})": "{:.4f}",
    "J": "{:.3f}", "Prob(J)": "{:.3f}", "p(J)": "{:.3f}", "df": "{:.0f}",
})
```

For comparison, Table I of {cite:t}`hansen1982generalized` reports for ND+VWR:

- $\hat\alpha$ from about $-0.90$ to $-0.82$ (so $\hat\gamma=-\hat\alpha$ is about $0.82$ to $0.90$),
- $\hat\beta$ near $0.997$,
- $\mathrm{se}(\hat\alpha)$ from about $0.106$ down to $0.063$ (so $\mathrm{se}(\hat\gamma)$ is the same up to sign),
- $\mathrm{se}(\hat\beta)$ around $0.0024$ to $0.0025$,
- `Prob` values around 0.50 to 0.88.

Use the table above to compare your run against those benchmarks.

In this lecture, `Prob(J)` is the paper-style `chi2.cdf(J, df)`, while `p(J)` is the right-tail value `1 - Prob(J)`.

Comparing the HAC and HC0 columns shows that the serial-correlation correction has only a modest effect on inference, consistent with the near-martingale-difference property of one-period Euler errors.

We inspect pricing errors and their autocorrelation structure to diagnose fit beyond summary statistics.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Euler error diagnostics and autocorrelation
    name: fig-hs82-euler-diagnostics
---
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

The time-series panel reveals pricing-error spikes (often associated with market stress episodes), while the histogram shows the distribution shape and the ACF panel displays persistence.

For the one-period martingale-difference case, pricing errors should be serially uncorrelated under correct specification. 

Bars crossing the red significance bounds indicate remaining predictable structure that the instrument set has not absorbed.

## Scope relative to the 1982 paper

This lecture reproduces the paper's Euler-equation GMM logic and Table I style reporting, but it does not reproduce all components of {cite:t}`hansen1982generalized`.

Two parts worth keeping in mind are:

- Section 4 and Table II, which compare GIV/GMM with maximum-likelihood estimates under additional distributional assumptions.
- Table III, which estimates the model jointly across multiple returns and finds substantially stronger rejections of the CRRA restrictions.

These omissions matter for interpretation: the paper's full empirical argument is not only about single-return GMM estimates, but also about how conclusions change under ML assumptions and multi-return systems.

To visualize the identification geometry, we plot the GMM criterion over a $(\gamma, \beta)$ grid.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GMM objective contour surface
    name: fig-hs82-objective-contour
---
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

The contour figure maps the criterion surface over $(\gamma, \beta)$ space, with the red star marking the estimated optimum.

An elongated valley in the contour plot signals weak separate identification of $\gamma$ and $\beta$: many parameter combinations produce similar moment conditions, even though a particular linear combination may be well identified.

## Summary

This lecture has implemented the GMM estimation strategy of {cite:t}`hansen1982generalized` for the consumption-based Euler equation.

The GMM estimator requires only the orthogonality conditions implied by the Euler equation and a set of predetermined instruments.

It does not require assumptions about the joint distribution of consumption and returns, the production technology, or any other part of the economic environment beyond the representative agent's first-order conditions.

This robustness comes at the cost of efficiency. 

GMM does not exploit information about the distribution of the data that could sharpen inference.

Relative to the original table, the open-data proxy reproduces the same exercise and sample window, but not the exact CRSP NYSE series or historical vintages.

The comparison is therefore about closeness of estimates and test outcomes, not exact equality row by row.
