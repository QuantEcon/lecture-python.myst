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

(doubts_or_variability)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Doubts or Variability?

```{contents} Contents
:depth: 2
```

## Overview

Robert Lucas Jr. opened a 2003 essay with a challenge:

> *No one has found risk aversion parameters of 50 or 100 in the diversification of
> individual portfolios, in the level of insurance deductibles, in the wage premiums
> associated with occupations with high earnings risk, or in the revenues raised by
> state-operated lotteries.*

Tallarini {cite}`Tallarini_2000` had shown that a recursive preference specification could match the equity premium and the risk-free rate puzzle simultaneously.
But matching required setting the risk-aversion coefficient $\gamma$ to around 50 for a random-walk consumption model and around 75 for a trend-stationary model --- exactly the range that provoked Lucas's skepticism.

{cite}`BHS_2009` ask whether those large $\gamma$ values really measure aversion to atemporal risk.
Their answer is no.
The same recursion that defines Tallarini's risk-sensitive agent is observationally equivalent to a
second recursion in which the agent has unit risk aversion but fears that the probability model governing consumption growth may be wrong.
Under this reading, the parameter that looked like extreme risk aversion instead measures
the agent's concern about **model misspecification**.

The question then becomes: how much misspecification is plausible?
Rather than calibrating $\gamma$ through Pratt-style thought experiments about known gambles,
{cite}`BHS_2009` calibrate through a **detection-error probability** --- the chance of confusing the agent's baseline (approximating) model with the pessimistic (worst-case) model after seeing a finite sample.
When detection-error probabilities are moderate, the implied $\gamma$ values are large enough to reach the Hansen--Jagannathan volatility bound.

This reinterpretation changes the welfare question that asset prices answer.
Large measured risk premia no longer imply large gains from smoothing known aggregate risk.
Instead, they imply large gains from resolving model uncertainty --- a very different policy object.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

np.set_printoptions(precision=4, suppress=True)


def set_seed(seed=1234):
    np.random.seed(seed)


set_seed()
```

## The economic idea

A representative consumer has a baseline probabilistic description of consumption growth.
Call it the **approximating model**.
The consumer does not fully trust this model.
To formalize that distrust, she surrounds the approximating model with a set of nearby alternatives that are difficult to distinguish statistically in finite samples.

Among those alternatives, a minimizing player inside the consumer's head selects a **worst-case model**.
The resulting max-min problem generates a likelihood-ratio distortion $\hat g_{t+1}$ that tilts one-step-ahead probabilities toward adverse states.
That distortion enters the stochastic discount factor alongside the usual intertemporal marginal rate of substitution, and its standard deviation is the **market price of model uncertainty** (MPU).

The discipline on how much distortion is allowed comes not from introspection about
willingness to pay for small known gambles, but from a statistical detection problem:
given $T$ observations, how likely is a Bayesian to confuse the approximating model with the worst-case model?
The answer is a **detection-error probability** $p(\theta^{-1})$.
High $p$ means the two models are nearly indistinguishable and the consumer's fear of misspecification is hard to dismiss.

## Four agent types and one key equivalence

The analysis compares four preference specifications that are useful for different purposes.

* **Type I** (Kreps--Porteus--Epstein--Zin--Tallarini): risk-sensitive recursive utility with risk-aversion parameter $\gamma$ and IES fixed at 1.
* **Type II** (multiplier preferences): unit risk aversion but a penalty parameter $\theta$ on the relative entropy of probability distortions.
* **Type III** (constraint preferences): unit risk aversion with a hard bound $\eta$ on discounted relative entropy.
* **Type IV** (pessimistic ex post Bayesian): log utility under a single pessimistic probability model $\hat\Pi_\infty$.

The pivotal result is that **types I and II are observationally equivalent** over consumption plans in this environment.
The mapping is $\theta = [(1-\beta)(\gamma - 1)]^{-1}$.
So when Tallarini sets $\gamma = 50$ to reach the Hansen--Jagannathan bound, one can equally say
the consumer has unit risk aversion and a model-uncertainty penalty $\theta$ that corresponds to a
moderate detection-error probability.
The quantitative fit is unchanged; only the economic interpretation shifts.

## Setup

The calibration uses quarterly U.S. data from 1948:2--2006:4 for consumption **growth rates** (a sample length of $T = 235$ quarters).
When we plot **levels** of log consumption (as in Fig. 6), we align the time index to 1948:1--2006:4, which yields $T+1 = 236$ quarterly observations.
Parameter estimates for two consumption-growth specifications (random walk and trend stationary)
come from Table 2 of {cite}`BHS_2009`, and asset-return moments come from their Table 1.
Following footnote 8 in {cite}`BHS_2009`, consumption is measured as real personal consumption expenditures on nondurable goods and services, deflated by its implicit chain price deflator, and expressed in per-capita terms using the civilian noninstitutional population aged 16+.

### Data

Most numerical inputs in this lecture are taken directly from {cite}`BHS_2009`.
Table 2 provides the maximum-likelihood estimates of $(\mu, \sigma_\varepsilon, \rho, \zeta)$ for the two consumption-growth specifications, and Table 1 provides the asset-return moments used in the Hansen--Jagannathan bound calculation.

Following footnote 8 of {cite}`BHS_2009`, consumption is measured as real personal consumption expenditures on nondurable goods and services, deflated by its implicit chain price deflator, and expressed in per-capita terms using the civilian noninstitutional population aged 16+.
We construct this measure from three [FRED](https://fred.stlouisfed.org) series:

| FRED series | Description |
| --- | --- |
| `PCNDGC96` | Real PCE: nondurable goods (billions of chained 2017 \$, SAAR) |
| `PCESVC96` | Real PCE: services (billions of chained 2017 \$, SAAR) |
| `CNP16OV` | Civilian noninstitutional population, 16+ (thousands, monthly) |

The BEA deflates each PCE component by its own chain-type price index internally.
Summing the chained-dollar components introduces a small Fisher-index non-additivity error, but this is negligible for our purposes and avoids the larger error of deflating the ND+SV nominal aggregate by the *overall* PCE deflator (which includes durables with secularly declining prices).

The processing pipeline is:

1. Add real nondurables and services: $C_t^{real} = C_t^{nd} + C_t^{sv}$.
2. Convert to per-capita (millions of dollars per person): divide by the quarterly average of the monthly population series and by $10^6$.
   This normalization matches the units in {cite}`BHS_2009`, where the trend-stationary intercept is $\zeta = -4.48$.
3. Compute log consumption: $c_t = \log C_t^{real,pc}$.

```{code-cell} ipython3
start_date = "1947-01-01"
end_date = "2007-01-01"


def _fetch_fred_series(series_id, start_date, end_date):
    # Keyless pull from FRED CSV endpoint.
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    if df.empty:
        raise ValueError(f"FRED returned an empty table for '{series_id}'")

    # Be robust to header variations (e.g., DATE vs date, BOM, whitespace).
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    date_col = df.columns[0]
    value_col = series_id if series_id in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
    if value_col is None:
        raise ValueError(
            f"Unexpected FRED CSV format for '{series_id}'. Columns: {list(df.columns)}"
        )

    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    out = pd.Series(values.to_numpy(), index=dates, name=series_id).dropna().sort_index()
    out = out.loc[start_date:end_date].dropna()
    if out.empty:
        raise ValueError(f"FRED series '{series_id}' returned no data in sample window")
    return out


# Fetch real PCE components and population from FRED (no API key required)
real_nd = _fetch_fred_series("PCNDGC96", start_date, end_date)
real_sv = _fetch_fred_series("PCESVC96", start_date, end_date)
pop_m = _fetch_fred_series("CNP16OV", start_date, end_date)

# Step 1: aggregate real nondurables + services
real_total = real_nd + real_sv

# Step 2: align to quarterly frequency first, then convert to per-capita
# real_total is in billions ($1e9), pop is in thousands ($1e3)
# per-capita in millions: real_total * 1e9 / (pop * 1e3) / 1e6 = real_total / pop
real_total_q = real_total.resample("QS").mean()
pop_q = pop_m.resample("QS").mean()
real_pc = (real_total_q / pop_q).dropna()

# Restrict to sample period 1948Q1–2006Q4
real_pc = real_pc.loc["1948-01-01":"2006-12-31"].dropna()

# FRED-only fallback: use BEA per-capita quarterly components directly.
# This avoids index-alignment failures in some pandas/FRED combinations.
if real_pc.empty:
    nd_pc = _fetch_fred_series("A796RX0Q048SBEA", start_date, end_date)
    sv_pc = _fetch_fred_series("A797RX0Q048SBEA", start_date, end_date)
    real_pc = ((nd_pc + sv_pc) / 1e6).loc["1948-01-01":"2006-12-31"].dropna()

if real_pc.empty:
    raise RuntimeError("FRED returned no usable observations after alignment/filtering")

# Step 3: log consumption
log_c_data = np.log(real_pc.to_numpy(dtype=float).reshape(-1))
years_data = (real_pc.index.year + (real_pc.index.month - 1) / 12.0).to_numpy(dtype=float)

print(f"Fetched {len(log_c_data)} quarterly observations from FRED")
print(f"Sample: {years_data[0]:.1f} – {years_data[-1] + 0.25:.1f}")
print(f"Observations: {len(log_c_data)}")
print(f"c_0 = {log_c_data[0]:.3f} (paper Fig 6: ≈ −4.6)")
```

### Consumption plans and the state-space representation

{cite}`BHS_2009` cast the analysis in terms of a general class of consumption plans.
Let $x_t$ be an $n \times 1$ state vector and $\varepsilon_{t+1}$ an $m \times 1$ shock.
A consumption plan belongs to the set $\mathcal{C}(A, B, H; x_0)$ if it admits the recursive representation

```{math}
:label: bhs_state_space
x_{t+1} = A x_t + B \varepsilon_{t+1},
\qquad
c_t = H x_t,
```

where the eigenvalues of $A$ are bounded in modulus by $1/\sqrt{\beta}$.
The time-$t$ element of a consumption plan can therefore be written as

```{math}
c_t = H\!\left(B\varepsilon_t + AB\varepsilon_{t-1} + \cdots + A^{t-1}B\varepsilon_1\right) + HA^t x_0.
```

The equivalence theorems and Bellman equations in the paper are stated for arbitrary plans in $\mathcal{C}(A,B,H;x_0)$.
The random-walk and trend-stationary models below are two special cases.

### Consumption dynamics

Let $c_t = \log C_t$ be log consumption.

The random-walk specification is

```{math}
c_{t+1} = c_t + \mu + \sigma_\varepsilon \varepsilon_{t+1}, \qquad \varepsilon_{t+1} \sim \mathcal{N}(0, 1).
```

The trend-stationary specification can be written as a deterministic trend plus a stationary AR(1) component {cite}`BHS_2009`:

```{math}
c_t = \zeta + \mu t + z_t,
\qquad
z_{t+1} = \rho z_t + \sigma_\varepsilon \varepsilon_{t+1},
\qquad
\varepsilon_{t+1} \sim \mathcal{N}(0, 1).
```

Equivalently, defining the detrended series $\tilde c_t := c_t - \mu t$,

```{math}
\tilde c_{t+1} - \zeta = \rho(\tilde c_t - \zeta) + \sigma_\varepsilon \varepsilon_{t+1}.
```

Table 2 in {cite}`BHS_2009` reports $(\mu, \sigma_\varepsilon)$ for the random walk and $(\mu, \sigma_\varepsilon, \rho, \zeta)$ for the trend-stationary case.

```{code-cell} ipython3
# Preferences and sample length
β = 0.995
T = 235  # quarterly sample length used in the paper

# Table 2 parameters
rw = dict(μ=0.00495, σ_ε=0.0050)
ts = dict(μ=0.00418, σ_ε=0.0050, ρ=0.980, ζ=-4.48)

# Table 1 moments, converted from percent to decimals
r_e_mean, r_e_std = 0.0227, 0.0768
r_f_mean, r_f_std = 0.0032, 0.0061
r_excess_std = 0.0767

R_mean = np.array([1.0 + r_e_mean, 1.0 + r_f_mean])  # gross returns
cov_erf = (r_e_std**2 + r_f_std**2 - r_excess_std**2) / 2.0
Σ_R = np.array(
    [
        [r_e_std**2, cov_erf],
        [cov_erf, r_f_std**2],
    ]
)
Σ_R_inv = np.linalg.inv(Σ_R)

print("Table 2 parameters")
print(f"random walk: μ={rw['μ']:.5f}, σ_ε={rw['σ_ε']:.5f}")
print(
    f"trend stationary: μ={ts['μ']:.5f}, σ_ε={ts['σ_ε']:.5f}, "
    f"ρ={ts['ρ']:.3f}, ζ={ts['ζ']:.2f}"
)
print()
print("Table 1 moments")
print(f"E[r_e]={r_e_mean:.4f}, std[r_e]={r_e_std:.4f}")
print(f"E[r_f]={r_f_mean:.4f}, std[r_f]={r_f_std:.4f}")
print(f"std[r_e-r_f]={r_excess_std:.4f}")
```

We can verify Table 2 by computing sample moments of log consumption growth from our FRED data:

```{code-cell} ipython3
# Growth rates: 1948Q2 to 2006Q4 (T = 235 quarters)
Δc = np.diff(log_c_data)

μ_hat = Δc.mean()
σ_hat = Δc.std(ddof=1)

print("Sample estimates from FRED data vs Table 2:")
print(f"  μ̂   = {μ_hat:.5f}   (Table 2 RW: {rw['μ']:.5f})")
print(f"  σ̂_ε = {σ_hat:.4f}    (Table 2: {rw['σ_ε']:.4f})")
print(f"  T   = {len(Δc)} quarters")
```

## Preferences, distortions, and detection

The type I recursion is

```{math}
:label: bhs_type1_recursion
\log V_t
=
(1-\beta)c_t
+
\frac{\beta}{1-\gamma}
\log E_t\left[(V_{t+1})^{1-\gamma}\right].
```

### The transformed continuation value

A key intermediate step in {cite}`BHS_2009` is to define the transformed continuation value

```{math}
:label: bhs_Ut_def
U_t \equiv \frac{\log V_t}{1-\beta}
```

and the robustness parameter

```{math}
:label: bhs_theta_def
\theta = \frac{-1}{(1-\beta)(1-\gamma)}.
```

Substituting into {eq}`bhs_type1_recursion` yields the **risk-sensitive recursion**

```{math}
:label: bhs_risk_sensitive
U_t = c_t - \beta\theta \log E_t\!\left[\exp\!\left(\frac{-U_{t+1}}{\theta}\right)\right].
```

When $\gamma = 1$ (equivalently $\theta = +\infty$), the $\log E \exp$ term reduces to $E_t U_{t+1}$
and the recursion becomes standard discounted expected log utility: $U_t = c_t + \beta E_t U_{t+1}$.

For consumption plans in $\mathcal{C}(A, B, H; x_0)$, the recursion {eq}`bhs_risk_sensitive` implies the Bellman equation

```{math}
:label: bhs_bellman_type1
U(x) = c - \beta\theta \log \int \exp\!\left[\frac{-U(Ax + B\varepsilon)}{\theta}\right] \pi(\varepsilon)\,d\varepsilon.
```

The stochastic discount factor can then be written as

```{math}
:label: bhs_sdf_Ut
m_{t+1,t}
=
\beta \frac{C_t}{C_{t+1}}
\cdot
\frac{\exp(-U_{t+1}/\theta)}{E_t[\exp(-U_{t+1}/\theta)]}.
```

The second factor is the likelihood-ratio distortion $\hat g_{t+1}$: an exponential tilt of the continuation value that shifts probability toward states with low $U_{t+1}$.

### Martingale likelihood ratios

To formalize model distortions, {cite}`BHS_2009` use a nonnegative martingale $G_t$ with $E(G_t \mid x_0) = 1$ as a Radon--Nikodym derivative.
Its one-step increments

```{math}
g_{t+1} = \frac{G_{t+1}}{G_t},
\qquad
E_t[g_{t+1}] = 1,
\quad
g_{t+1} \ge 0,
\qquad
G_0 = 1,
```

define distorted conditional expectations: $\tilde E_t[b_{t+1}] = E_t[g_{t+1}\,b_{t+1}]$.
The conditional relative entropy of the distortion is $E_t[g_{t+1}\log g_{t+1}]$, and the discounted entropy over the entire path is $\beta E\bigl[\sum_{t=0}^{\infty} \beta^t G_t\,E_t(g_{t+1}\log g_{t+1})\,\big|\,x_0\bigr]$.

### Type II: multiplier preferences

A type II agent's **multiplier** preference ordering over consumption plans $C^\infty \in \mathcal{C}(A,B,H;x_0)$ is defined by

```{math}
:label: bhs_type2_objective
\min_{\{g_{t+1}\}}
\sum_{t=0}^{\infty} E\!\left\{\beta^t G_t
\left[c_t + \beta\theta\,E_t\!\left(g_{t+1}\log g_{t+1}\right)\right]
\,\Big|\, x_0\right\},
```

where $G_{t+1} = g_{t+1}G_t$, $E_t[g_{t+1}] = 1$, $g_{t+1} \ge 0$, and $G_0 = 1$.
The parameter $\theta > 0$ penalizes the relative entropy of probability distortions.

The value function satisfies the Bellman equation

```{math}
:label: bhs_bellman_type2
W(x)
=
c + \min_{g(\varepsilon) \ge 0}\;
\beta \int \bigl[g(\varepsilon)\,W(Ax + B\varepsilon)
+ \theta\,g(\varepsilon)\log g(\varepsilon)\bigr]\,\pi(\varepsilon)\,d\varepsilon
```

subject to $\int g(\varepsilon)\,\pi(\varepsilon)\,d\varepsilon = 1$.
Note that $g(\varepsilon)$ multiplies both the continuation value $W$ and the entropy penalty --- this is the key structural feature that makes $\hat g$ a likelihood ratio.

The minimizer is

```{math}
:label: bhs_ghat
\hat g_{t+1}
=
\frac{\exp\!\bigl(-W(Ax_t + B\varepsilon_{t+1})/\theta\bigr)}{E_t\!\left[\exp\!\bigl(-W(Ax_t + B\varepsilon_{t+1})/\theta\bigr)\right]}.
```

Substituting {eq}`bhs_ghat` back into {eq}`bhs_bellman_type2` gives

$$W(x) = c - \beta\theta \log \int \exp\!\left[\frac{-W(Ax + B\varepsilon)}{\theta}\right]\pi(\varepsilon)\,d\varepsilon,$$

which is identical to {eq}`bhs_bellman_type1`.
Therefore $W(x) \equiv U(x)$, establishing that **types I and II are observationally equivalent** over elements of $\mathcal{C}(A,B,H;x_0)$.
The mapping between parameters is

```{math}
\theta = \left[(1-\beta)(\gamma - 1)\right]^{-1}.
```

```{code-cell} ipython3
def θ_from_γ(γ, β=β):
    if γ <= 1:
        return np.inf
    return 1.0 / ((1.0 - β) * (γ - 1.0))


def γ_from_θ(θ, β=β):
    if np.isinf(θ):
        return 1.0
    return 1.0 + 1.0 / ((1.0 - β) * θ)
```

### Type III: constraint preferences

Type III (constraint) preferences replace the entropy penalty with a hard bound.
The agent minimizes expected discounted log consumption under the worst-case model,
subject to a cap $\eta$ on discounted relative entropy:

```{math}
J(x_0)
=
\min_{\{g_{t+1}\}}
\sum_{t=0}^{\infty} E\!\left[\beta^t G_t\,c_t \,\Big|\, x_0\right]
```

subject to $G_{t+1} = g_{t+1}G_t$, $E_t[g_{t+1}] = 1$, $g_{t+1} \ge 0$, $G_0 = 1$, and

```{math}
\beta E\!\left[\sum_{t=0}^{\infty} \beta^t G_t\,E_t\!\left(g_{t+1}\log g_{t+1}\right)\,\Big|\,x_0\right] \le \eta.
```

The Lagrange multiplier on the entropy constraint is $\theta$, which connects type III to type II:
for the particular $A, B, H$ and $\theta$ used to derive the worst-case joint distribution $\hat\Pi_\infty$,
the shadow prices of uncertain claims for a type III agent match those of a type II agent.

### Type IV: ex post Bayesian

Type IV is an ordinary expected-utility agent with log preferences evaluated under a single pessimistic probability model $\hat\Pi_\infty$:

```{math}
\hat E_0 \sum_{t=0}^{\infty} \beta^t c_t.
```

The joint distribution $\hat\Pi_\infty(\cdot \mid x_0, \theta)$ is the one associated with the type II agent's worst-case distortion.
For the particular $A, B, H$ and $\theta$ used to construct $\hat\Pi_\infty$, the type IV value function equals $J(x)$ from type III.

### Stochastic discount factor

Across all four types, the stochastic discount factor can be written compactly as

```{math}
:label: bhs_sdf
m_{t+1,t}
=
\beta \frac{C_t}{C_{t+1}} \hat g_{t+1}.
```

The distortion $\hat g_{t+1}$ is a likelihood ratio between the approximating and worst-case one-step models.

With log utility, $C_t/C_{t+1} = \exp(-(c_{t+1}-c_t))$ is the usual intertemporal marginal rate of substitution.
Robustness multiplies that term by $\hat g_{t+1}$, so uncertainty aversion enters pricing only through the distortion.

### Gaussian mean-shift distortions

Under the random-walk model, the shock is $\varepsilon_{t+1} \sim \mathcal{N}(0, 1)$.
The worst-case model shifts its mean to $-w$, which implies the likelihood ratio

```{math}
\hat g_{t+1}
=
\exp\left(-w \varepsilon_{t+1} - \frac{1}{2}w^2\right),
\qquad
E_t[\hat g_{t+1}] = 1.
```

Hence $\log \hat g_{t+1}$ is normal with mean $-w^2/2$ and variance $w^2$, and

```{math}
\operatorname{std}(\hat g_{t+1}) = \sqrt{e^{w^2}-1}.
```

For our Gaussian calibrations, the worst-case mean shift is summarized by

```{math}
:label: bhs_w_formulas
w_{rw}(\theta) = -\frac{\sigma_\varepsilon}{(1-\beta)\theta},
\qquad
w_{ts}(\theta) = -\frac{\sigma_\varepsilon}{(1-\rho\beta)\theta}.
```

```{code-cell} ipython3
def w_from_θ(θ, model):
    if np.isinf(θ):
        return 0.0
    if model == "rw":
        return -rw["σ_ε"] / ((1.0 - β) * θ)
    if model == "ts":
        return -ts["σ_ε"] / ((1.0 - β * ts["ρ"]) * θ)
    raise ValueError("model must be 'rw' or 'ts'")
```

The **market price of model uncertainty** (MPU) is the conditional standard deviation of the distortion:

```{math}
:label: bhs_mpu_formula
\text{MPU}
=
\operatorname{std}(\hat g_{t+1})
=
\sqrt{e^{w(\theta)^2}-1}
\approx |w(\theta)|.
```

The detection error probability is

```{math}
:label: bhs_detection_formula
p(\theta^{-1})
=
\frac{1}{2}\left(p_A + p_B\right),
```

and in our Gaussian mean-shift case reduces to

```{math}
:label: bhs_detection_closed
p(\theta^{-1}) = \Phi\!\left(-\frac{|w(\theta)|\sqrt{T}}{2}\right).
```

```{code-cell} ipython3
def detection_probability(θ, model):
    w = abs(w_from_θ(θ, model))
    return norm.cdf(-0.5 * w * np.sqrt(T))


def θ_from_detection_probability(p, model):
    if p >= 0.5:
        return np.inf
    w_abs = -2.0 * norm.ppf(p) / np.sqrt(T)
    if model == "rw":
        return rw["σ_ε"] / ((1.0 - β) * w_abs)
    if model == "ts":
        return ts["σ_ε"] / ((1.0 - β * ts["ρ"]) * w_abs)
    raise ValueError("model must be 'rw' or 'ts'")
```

### Likelihood-ratio testing and detection errors

Let $L_T$ be the log likelihood ratio between the worst-case and approximating models based on a sample of length $T$.
Define

```{math}
p_A = \Pr_A(L_T < 0),
\qquad
p_B = \Pr_B(L_T > 0),
```

where $\Pr_A$ and $\Pr_B$ denote probabilities under the approximating and worst-case models.
Then $p(\theta^{-1}) = \frac{1}{2}(p_A + p_B)$ is the average probability of choosing the wrong model.

In the Gaussian mean-shift setting, $L_T$ is normal with mean $\pm \tfrac{1}{2}w^2T$ and variance $w^2T$, which yields the closed-form expression above.

### Interpreting the calibration objects

The parameter $\theta$ indexes how expensive it is for the minimizing player to distort the approximating model.

A small $\theta$ means a cheap distortion and therefore stronger robustness concerns.

The associated $\gamma = 1 + \left[(1-\beta)\theta\right]^{-1}$ can be large even when we do not want to interpret behavior as extreme atemporal risk aversion.

The distortion magnitude $|w(\theta)|$ is a direct measure of how pessimistically the agent tilts one-step probabilities.

Detection error probability $p(\theta^{-1})$ translates that tilt into a statistical statement about finite-sample distinguishability.

High $p(\theta^{-1})$ means the two models are hard to distinguish.

Low $p(\theta^{-1})$ means they are easier to distinguish.

This translation is the bridge between econometric identification and preference calibration.

Finally, the relative-entropy distance associated with the worst-case distortion is

```{math}
E_t[\hat g_{t+1}\log \hat g_{t+1}] = \frac{1}{2}w(\theta)^2,
```

so the discounted entropy used in type III preferences is

```{math}
\eta
=
\frac{\beta}{1-\beta}\cdot \frac{w(\theta)^2}{2},
```

```{code-cell} ipython3
def η_from_θ(θ, model):
    w = w_from_θ(θ, model)
    return β * w**2 / (2.0 * (1.0 - β))
```

This is the mapping behind the right panel of the detection-probability figure below.

## Tallarini's success and its cost

Hansen and Jagannathan {cite}`Hansen_Jagannathan_1991` showed that any valid stochastic discount factor $m_{t+1,t}$ must satisfy a volatility bound: $\sigma(m)/E(m)$ must be at least as large as the maximum Sharpe ratio attainable in the market.
Using postwar U.S. returns on the value-weighted NYSE and Treasury bills, this bound sets a
high bar that time-separable CRRA preferences struggle to clear without also distorting the
risk-free rate.

In terms of the vector of gross returns $R_{t+1}$ with mean $E(R)$ and covariance matrix $\Sigma_R$,
the bound can be written as

```{math}
\frac{\sigma(m)}{E(m)}
\;\ge\;
\sqrt{b^\top \Sigma_R^{-1} b},
\qquad
b = \mathbf{1} - E(m) E(R).
```

```{code-cell} ipython3
def hj_std_bound(E_m):
    b = np.ones(2) - E_m * R_mean
    var_lb = b @ Σ_R_inv @ b
    return np.sqrt(np.maximum(var_lb, 0.0))
```

Tallarini {cite}`Tallarini_2000` showed that recursive preferences with IES $= 1$ can clear this bar.
By separating risk aversion $\gamma$ from the IES, the recursion pushes $\sigma(m)/E(m)$ upward
while leaving $E(m)$ roughly consistent with the observed risk-free rate.

For the two consumption specifications, {cite}`BHS_2009` derive closed-form expressions for the unconditional SDF moments.

**Random walk** (eqs 15--16 of the paper):

```{math}
:label: bhs_Em_rw
E[m] = \beta \exp\!\left[-\mu + \frac{\sigma_\varepsilon^2}{2}(2\gamma - 1)\right],
```

```{math}
:label: bhs_sigma_rw
\frac{\sigma(m)}{E[m]} = \sqrt{\exp\!\left(\sigma_\varepsilon^2 \gamma^2\right) - 1}.
```

**Trend stationary** (eqs 17--18):

```{math}
:label: bhs_Em_ts
E[m] = \beta \exp\!\left[-\mu + \frac{\sigma_\varepsilon^2}{2}\!\left(1 - \frac{2(1-\beta)(1-\gamma)}{1-\beta\rho} + \frac{1-\rho}{1+\rho}\right)\right],
```

```{math}
:label: bhs_sigma_ts
\frac{\sigma(m)}{E[m]} = \sqrt{\exp\!\left[\sigma_\varepsilon^2\!\left(\!\left(\frac{(1-\beta)(1-\gamma)}{1-\beta\rho} - 1\right)^{\!2} + \frac{1-\rho}{1+\rho}\right)\right] - 1}.
```

These are what the code below implements.

The figure below makes this visible.
For each value of $\gamma \in \{1, 5, 10, \ldots, 50\}$, we plot the implied $(E(m),\;\sigma(m)/E(m))$ pair
for three specifications: time-separable CRRA (crosses), type I recursive preferences with random-walk consumption (circles), and type I recursive preferences with trend-stationary consumption (pluses).

```{code-cell} ipython3
def moments_type1_rw(γ):
    θ = θ_from_γ(γ)
    w = w_from_θ(θ, "rw")
    var_log_m = (w - rw["σ_ε"]) ** 2
    mean_log_m = np.log(β) - rw["μ"] - 0.5 * w**2
    E_m = np.exp(mean_log_m + 0.5 * var_log_m)
    mpr = np.sqrt(np.exp(var_log_m) - 1.0)
    return E_m, mpr


def moments_type1_ts(γ):
    θ = θ_from_γ(γ)
    w = w_from_θ(θ, "ts")
    var_z = ts["σ_ε"] ** 2 / (1.0 - ts["ρ"] ** 2)
    var_log_m = (1.0 - ts["ρ"]) ** 2 * var_z + (w - ts["σ_ε"]) ** 2
    mean_log_m = np.log(β) - ts["μ"] - 0.5 * w**2
    E_m = np.exp(mean_log_m + 0.5 * var_log_m)
    mpr = np.sqrt(np.exp(var_log_m) - 1.0)
    return E_m, mpr


def moments_crra_rw(γ):
    var_log_m = (γ * rw["σ_ε"]) ** 2
    mean_log_m = np.log(β) - γ * rw["μ"]
    E_m = np.exp(mean_log_m + 0.5 * var_log_m)
    mpr = np.sqrt(np.exp(var_log_m) - 1.0)
    return E_m, mpr
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: stochastic discount factor moments and the Hansen-Jagannathan volatility
      bound
    name: fig-bhs-1
---
γ_grid = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=float)

Em_rw = np.array([moments_type1_rw(γ)[0] for γ in γ_grid])
MPR_rw = np.array([moments_type1_rw(γ)[1] for γ in γ_grid])

Em_ts = np.array([moments_type1_ts(γ)[0] for γ in γ_grid])
MPR_ts = np.array([moments_type1_ts(γ)[1] for γ in γ_grid])

Em_crra = np.array([moments_crra_rw(γ)[0] for γ in γ_grid])
MPR_crra = np.array([moments_crra_rw(γ)[1] for γ in γ_grid])

Em_grid = np.linspace(0.8, 1.01, 1000)
HJ_std = np.array([hj_std_bound(x) for x in Em_grid])

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Em_grid, HJ_std, lw=2, color="black", label="Hansen-Jagannathan bound")
ax.plot(Em_rw, MPR_rw, "o", lw=2, label="type I, random walk")
ax.plot(Em_ts, MPR_ts, "+", lw=2, label="type I, trend stationary")
ax.plot(Em_crra, MPR_crra, "x", lw=2, label="time-separable CRRA")

ax.set_xlabel(r"$E(m)$")
ax.set_ylabel(r"$\sigma(m)/E(m)$")
ax.legend(frameon=False)
ax.set_xlim(0.8, 1.01)
ax.set_ylim(0.0, 0.42)

plt.tight_layout()
plt.show()
```

The crosses trace the familiar CRRA failure: as $\gamma$ rises, $\sigma(m)/E(m)$ grows but $E(m)$ falls well below the range consistent with the observed risk-free rate.
This is the risk-free-rate puzzle of Weil {cite}`Weil_1989`.

The circles and pluses show Tallarini's solution.
Recursive utility with IES $= 1$ pushes volatility upward while keeping $E(m)$ roughly constant near $1/(1+r^f)$.
For the random-walk model, the bound is reached around $\gamma = 50$; for the trend-stationary model, around $\gamma = 75$.

The quantitative achievement is real.
But Lucas's challenge still stands: what microeconomic evidence supports $\gamma = 50$?
That tension is the starting point for the reinterpretation that follows.

## A new calibration language: detection-error probabilities

If $\gamma$ should not be calibrated by introspection about atemporal gambles, what replaces it?

The answer is a statistical test.
Fix a sample size $T$ (here 235 quarters, matching the postwar U.S. data).
For a given $\theta$, compute the worst-case model and ask:
if a Bayesian ran a likelihood-ratio test to distinguish the approximating model from the worst-case model, what fraction of the time would she make an error?
That fraction is the detection-error probability $p(\theta^{-1})$.

A high $p$ (near 0.5) means the two models are nearly indistinguishable --- the consumer's fear is hard to rule out.
A low $p$ means the worst case is easy to reject and the robustness concern is less compelling.

The left panel below plots $p(\theta^{-1})$ against $\theta^{-1}$ for the two consumption specifications.
Notice that the same numerical $\theta$ corresponds to very different detection probabilities across models, because baseline dynamics differ.
The right panel resolves this by plotting detection probabilities against discounted relative entropy $\eta$, which normalizes the statistical distance.
Indexed by $\eta$, the two curves coincide.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: detection probabilities under random-walk and trend-stationary approximating
      models
    name: fig-bhs-2
---
θ_inv_grid = np.linspace(0.0, 1.8, 400)
θ_grid = np.full_like(θ_inv_grid, np.inf)
mask_θ = θ_inv_grid > 0.0
θ_grid[mask_θ] = 1.0 / θ_inv_grid[mask_θ]

p_rw = np.array([detection_probability(θ, "rw") for θ in θ_grid])
p_ts = np.array([detection_probability(θ, "ts") for θ in θ_grid])

η_rw = np.array([η_from_θ(θ, "rw") for θ in θ_grid])
η_ts = np.array([η_from_θ(θ, "ts") for θ in θ_grid])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(θ_inv_grid, 100.0 * p_rw, lw=2, label="random walk")
axes[0].plot(θ_inv_grid, 100.0 * p_ts, lw=2, label="trend stationary")
axes[0].set_xlabel(r"$\theta^{-1}$")
axes[0].set_ylabel("detection error probability (percent)")
axes[0].legend(frameon=False)

axes[1].plot(η_rw, 100.0 * p_rw, lw=2, label="random walk")
axes[1].plot(η_ts, 100.0 * p_ts, lw=2, ls="--", label="trend stationary")
axes[1].set_xlabel(r"discounted entropy $\eta$")
axes[1].set_ylabel("detection error probability (percent)")
axes[1].set_xlim(0.0, 10)
axes[1].legend(frameon=False)

plt.tight_layout()
plt.show()
```

This is why detection-error probabilities (or equivalently, discounted entropy) are the right cross-model yardstick.
Holding $\theta$ fixed when switching from a random walk to a trend-stationary specification
implicitly changes how much misspecification the consumer fears.
Holding $\eta$ or $p$ fixed keeps the statistical difficulty of detecting misspecification constant.

The explicit mapping that equates discounted entropy across models is (eq 41 of the paper):

```{math}
:label: bhs_theta_cross_model
\theta_{\text{TS}}
=
\left(\frac{\sigma_\varepsilon^{\text{TS}}}{\sigma_\varepsilon^{\text{RW}}}\right)
\frac{1-\beta}{1-\rho\beta}\;\theta_{\text{RW}}.
```

At our calibration $\sigma_\varepsilon^{\text{TS}} = \sigma_\varepsilon^{\text{RW}}$, this simplifies to
$\theta_{\text{TS}} = \frac{1-\beta}{1-\rho\beta}\,\theta_{\text{RW}}$.
Because $\rho = 0.98$ and $\beta = 0.995$, the ratio $(1-\beta)/(1-\rho\beta)$ is much less than one,
so holding entropy fixed requires a substantially smaller $\theta$ (stronger robustness) for the trend-stationary model than for the random walk.

## The punchline: detection probabilities unify the two models

We can now redraw Tallarini's figure using the new language.
For each detection-error probability $p(\theta^{-1}) = 0.50, 0.45, \ldots, 0.01$,
invert to find the model-specific $\theta$, convert to $\gamma$, and plot the implied $(E(m),\;\sigma(m)/E(m))$ pair.

```{code-cell} ipython3
p_points = np.array([0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.01])

θ_rw_points = np.array([θ_from_detection_probability(p, "rw") for p in p_points])
θ_ts_points = np.array([θ_from_detection_probability(p, "ts") for p in p_points])

γ_rw_points = np.array([γ_from_θ(θ) for θ in θ_rw_points])
γ_ts_points = np.array([γ_from_θ(θ) for θ in θ_ts_points])

Em_rw_p = np.array([moments_type1_rw(γ)[0] for γ in γ_rw_points])
MPR_rw_p = np.array([moments_type1_rw(γ)[1] for γ in γ_rw_points])
Em_ts_p = np.array([moments_type1_ts(γ)[0] for γ in γ_ts_points])
MPR_ts_p = np.array([moments_type1_ts(γ)[1] for γ in γ_ts_points])

print("p      γ_rw      γ_ts")
for p, g1, g2 in zip(p_points, γ_rw_points, γ_ts_points):
    print(f"{p:>4.2f} {g1:>9.2f} {g2:>9.2f}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: pricing loci obtained from common detection probabilities
    name: fig-bhs-3
---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Em_rw_p, MPR_rw_p, "o-", lw=2, label="random walk")
ax.plot(Em_ts_p, MPR_ts_p, "+-", lw=2, label="trend stationary")
ax.plot(Em_grid, HJ_std, lw=2, color="black", label="Hansen-Jagannathan bound")

ax.set_xlabel(r"$E(m)$")
ax.set_ylabel(r"$\sigma(m)/E(m)$")
ax.legend(frameon=False, loc="upper right")
ax.set_xlim(0.96, 1.05)
ax.set_ylim(0.0, 0.34)

plt.tight_layout()
plt.show()
```

The striking result: the random-walk and trend-stationary loci nearly coincide.

Recall that under Tallarini's $\gamma$-calibration, reaching the Hansen--Jagannathan bound required $\gamma \approx 50$ for the random walk but $\gamma \approx 75$ for the trend-stationary model --- very different numbers for the "same" preference parameter.
Under detection-error calibration, both models reach the bound at the same detectability level (around $p = 0.05$).

The model dependence was an artifact of using $\gamma$ as a cross-model yardstick.
Once we measure robustness concerns in units of statistical detectability, the two consumption specifications tell the same story:
a representative consumer with moderate, difficult-to-dismiss fears about model misspecification
behaves as if she had very high risk aversion.

## What do risk premia measure? Two mental experiments

Lucas {cite}`Lucas_2003` asked how much consumption a representative consumer would sacrifice to eliminate
aggregate fluctuations.
His answer --- very little --- rested on the assumption that the consumer knows the data-generating process.

The robust reinterpretation introduces a second, distinct mental experiment.
Instead of eliminating all randomness, suppose we keep randomness but remove the consumer's
fear of model misspecification (set $\theta = \infty$).
How much would she pay for that relief alone?

Formally, define $\Delta c_0$ as a permanent proportional reduction in initial consumption that leaves the agent indifferent between
the original environment and a counterfactual in which either (i) risk alone is removed or (ii) model uncertainty is removed.
Because utility is log and the consumption process is Gaussian, these compensations are available in closed form.

For type II preferences in the random-walk model, the decomposition is

```{math}
:label: bhs_type2_rw_decomp
\Delta c_0^{risk}
=
\frac{\beta \sigma_\varepsilon^2}{2(1-\beta)},
\qquad
\Delta c_0^{uncertainty}
=
\frac{\beta \sigma_\varepsilon^2}{2(1-\beta)^2\theta}.
```

For type III preferences in the random-walk model, the uncertainty term is twice as large:

```{math}
:label: bhs_type3_rw_decomp
\Delta c_0^{uncertainty, III}
=
\frac{\beta \sigma_\varepsilon^2}{(1-\beta)^2\theta}.
```

For the trend-stationary model, denominators replace $(1-\beta)$ with $(1-\beta \rho)$ or $(1-\beta \rho^2)$ as detailed in Table 3 of {cite}`BHS_2009`, but the qualitative message is the same.

The risk-only term $\Delta c_0^{risk}$ is tiny at postwar consumption volatility --- this is Lucas's well-known result.
The model-uncertainty term $\Delta c_0^{uncertainty}$ can be first order whenever the detection-error probability is moderate, because $\theta$ appears in the denominator.

## Visualizing the welfare decomposition

We set $\beta = 0.995$ and calibrate $\theta$ so that $p(\theta^{-1}) = 0.10$, a conservative detection-error level.

```{code-cell} ipython3
p_star = 0.10
θ_star = θ_from_detection_probability(p_star, "rw")
γ_star = γ_from_θ(θ_star)
w_star = w_from_θ(θ_star, "rw")

# Type II compensations, random walk model
comp_risk_only = β * rw["σ_ε"] ** 2 / (2.0 * (1.0 - β))
comp_risk_unc = comp_risk_only + β * rw["σ_ε"] ** 2 / (2.0 * (1.0 - β) ** 2 * θ_star)

# Two useful decompositions in levels
risk_only_pct = 100.0 * (np.exp(comp_risk_only) - 1.0)
risk_unc_pct = 100.0 * (np.exp(comp_risk_unc) - 1.0)
uncertainty_only_pct = 100.0 * (np.exp(comp_risk_unc - comp_risk_only) - 1.0)

print(f"p*={p_star:.2f}, θ*={θ_star:.4f}, γ*={γ_star:.2f}, w*={w_star:.4f}")
print(f"risk only compensation (log units): {comp_risk_only:.6f}")
print(f"risk + uncertainty compensation (log units): {comp_risk_unc:.6f}")
print(f"risk only compensation (percent): {risk_only_pct:.3f}%")
print(f"risk + uncertainty compensation (percent): {risk_unc_pct:.3f}%")
print(f"uncertainty component alone (percent): {uncertainty_only_pct:.3f}%")

h = 250
t = np.arange(h + 1)

# Baseline approximating model fan
mean_base = rw["μ"] * t
std_base = rw["σ_ε"] * np.sqrt(t)

# Certainty equivalent line from Eq. (47), shifted by compensating variations
certainty_slope = rw["μ"] + 0.5 * rw["σ_ε"] ** 2
ce_risk = -comp_risk_only + certainty_slope * t
ce_risk_unc = -comp_risk_unc + certainty_slope * t

# Alternative models from the ambiguity set in panel B
mean_low = (rw["μ"] + rw["σ_ε"] * w_star) * t
mean_high = (rw["μ"] - rw["σ_ε"] * w_star) * t
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: certainty-equivalent paths and the set of nearby models under robustness
    name: fig-bhs-4
---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Panel A
ax = axes[0]
ax.fill_between(t, mean_base - std_base, mean_base + std_base, alpha=0.25, color="tab:blue")
ax.plot(t, ce_risk_unc, lw=2, ls="--", color="black", label="certainty equivalent: risk + uncertainty")
ax.plot(t, ce_risk, lw=2, color="tab:orange", label="certainty equivalent: risk only")
ax.plot(t, mean_base, lw=2, color="tab:blue", label="approximating-model mean")
ax.set_xlabel("quarters")
ax.set_ylabel("log consumption")
ax.legend(frameon=False, fontsize=8, loc="upper left")

# Panel B
ax = axes[1]
ax.fill_between(t, mean_base - std_base, mean_base + std_base, alpha=0.20, color="tab:blue")
ax.fill_between(t, mean_low - std_base, mean_low + std_base, alpha=0.20, color="tab:red")
ax.fill_between(t, mean_high - std_base, mean_high + std_base, alpha=0.20, color="tab:green")
ax.plot(t, ce_risk_unc, lw=2, ls="--", color="black", label="certainty equivalent: risk + uncertainty")
ax.plot(t, mean_base, lw=2, color="tab:blue", label="approximating-model mean")
ax.plot(t, mean_low, lw=2, color="tab:red", label="worst-case-leaning mean")
ax.plot(t, mean_high, lw=2, color="tab:green", label="best-case-leaning mean")
ax.set_xlabel("quarters")
ax.set_ylabel("log consumption")
ax.legend(frameon=False, fontsize=8, loc="upper left")

plt.tight_layout()
plt.show()
```

**Left panel.**
The small gap between the baseline mean path and the "risk only" certainty equivalent is Lucas's result:
at postwar consumption volatility, the welfare gain from eliminating well-understood aggregate risk is tiny.

The much larger gap between the baseline and the "risk + uncertainty" certainty equivalent
is the new object.
Most of that gap is compensation for model uncertainty, not risk.

**Right panel.**
The cloud of nearby models shows what the robust consumer guards against.
The red-shaded and green-shaded fans correspond to pessimistic and optimistic mean-shift distortions
whose detection-error probability is $p = 0.10$.
These models are statistically close to the baseline (blue) but imply very different long-run consumption levels.
The consumer's caution against such alternatives is what drives the large certainty-equivalent gap in the left panel.

## How large are the welfare gains from resolving model uncertainty?

A type III (constraint-preference) agent evaluates the worst model inside an entropy ball of radius $\eta$.
As $\eta$ grows, the set of plausible misspecifications expands and the welfare cost of confronting model uncertainty rises.
Because $\eta$ is abstract, {cite}`BHS_2009` instead index these costs by the associated detection error probability $p(\eta)$.
The figure below reproduces their display: compensation for removing model uncertainty, measured as a proportion of consumption, plotted against $p(\eta)$.

```{code-cell} ipython3
η_grid = np.linspace(0.0, 5.0, 300)

# Use w and η relation, then convert to θ model by model
w_abs_grid = np.sqrt(2.0 * (1.0 - β) * η_grid / β)

θ_rw_from_η = np.full_like(w_abs_grid, np.inf)
θ_ts_from_η = np.full_like(w_abs_grid, np.inf)
mask_w = w_abs_grid > 0.0
θ_rw_from_η[mask_w] = rw["σ_ε"] / ((1.0 - β) * w_abs_grid[mask_w])
θ_ts_from_η[mask_w] = ts["σ_ε"] / ((1.0 - β * ts["ρ"]) * w_abs_grid[mask_w])

# Type III uncertainty terms from Table 3
gain_rw = np.where(
    np.isinf(θ_rw_from_η),
    0.0,
    β * rw["σ_ε"] ** 2 / ((1.0 - β) ** 2 * θ_rw_from_η),
)
gain_ts = np.where(
    np.isinf(θ_ts_from_η),
    0.0,
    β * ts["σ_ε"] ** 2 / ((1.0 - β * ts["ρ"]) ** 2 * θ_ts_from_η),
)

# Convert log compensation to percent of initial consumption in levels
gain_rw_pct = 100.0 * (np.exp(gain_rw) - 1.0)
gain_ts_pct = 100.0 * (np.exp(gain_ts) - 1.0)

# Detection error probabilities implied by η (common across RW/TS for the Gaussian mean-shift case)
p_eta_pct = 100.0 * norm.cdf(-0.5 * w_abs_grid * np.sqrt(T))
order = np.argsort(p_eta_pct)
p_plot = p_eta_pct[order]
gain_rw_plot = gain_rw_pct[order]
gain_ts_plot = gain_ts_pct[order]
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: type III compensation for model uncertainty across detection-error probabilities
    name: fig-bhs-5
---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(p_plot, gain_rw_plot, lw=2, color="black", label="RW type III")
ax.plot(p_plot, gain_ts_plot, lw=2, ls="--", color="gray", label="TS type III")
ax.set_xlabel(r"detection error probability $p(\eta)$ (percent)")
ax.set_ylabel("proportion of consumption (percent)")
ax.set_xlim(0.0, 50.0)
ax.set_ylim(0.0, 30.0)
ax.legend(frameon=False, loc="upper right")

plt.tight_layout()
plt.show()
```

The random-walk model delivers somewhat larger costs than the trend-stationary model at the same detection-error probability, but both curves dwarf the classic Lucas cost of business cycles.

To put the magnitudes in perspective: Lucas estimated that eliminating all aggregate consumption risk
is worth roughly 0.05% of consumption.
At detection-error probabilities of 10--20%, the model-uncertainty
compensation alone runs to several percent of consumption.

This is the welfare counterpart to the pricing result.
The large risk premia that Tallarini matched with high $\gamma$ are, under the robust reading,
compensations for bearing model uncertainty --- and the implied welfare gains from resolving that uncertainty are correspondingly large.

## Why doesn't learning eliminate these fears?

A natural objection: if the consumer has 235 quarters of data, why can't she learn the true drift
well enough to dismiss the worst-case model?

The answer is that drift is a low-frequency feature.
Estimating the mean of a random walk to the precision needed to reject small but economically meaningful
shifts requires far more data than estimating volatility.
The figure below makes this concrete.

```{code-cell} ipython3
p_fig6 = 0.20

# Figure 6 overlays deterministic lines on the loaded consumption data.
# Use sample-estimated RW moments to avoid data-vintage drift mismatches.
rw_fig6 = dict(μ=μ_hat, σ_ε=σ_hat)
w_fig6 = 2.0 * norm.ppf(p_fig6) / np.sqrt(T)

# Use FRED data loaded earlier in the lecture
c = log_c_data
years = years_data

t6 = np.arange(T + 1)
c0 = c[0]
line_approx = c0 + rw_fig6["μ"] * t6
line_worst = c0 + (rw_fig6["μ"] + rw_fig6["σ_ε"] * w_fig6) * t6

p_right = np.linspace(0.01, 0.50, 500)
w_right = 2.0 * norm.ppf(p_right) / np.sqrt(T)
μ_worst_right = rw_fig6["μ"] + rw_fig6["σ_ε"] * w_right

μ_se = rw_fig6["σ_ε"] / np.sqrt(T)
upper_band = rw_fig6["μ"] + 2.0 * μ_se
lower_band = rw_fig6["μ"] - 2.0 * μ_se
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: robustly distorted growth rates and finite-sample uncertainty about drift
    name: fig-bhs-6
---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(years, c, lw=2, color="tab:blue", label="log consumption")
ax.plot(years, line_approx, lw=2, ls="--", color="black", label="approximating model")
ax.plot(
    years,
    line_worst,
    lw=2,
    ls=":",
    color="black",
    label=rf"wc model $p(\theta^{{-1}})={p_fig6:.1f}$",
)
ax.set_xlabel("year")
ax.set_ylabel("log consumption")
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax = axes[1]
ax.plot(
    100.0 * p_right,
    1_000.0 * μ_worst_right,
    lw=2,
    color="tab:red",
    label=r"$\mu + \sigma_\varepsilon w(\theta)$",
)
ax.axhline(1_000.0 * rw_fig6["μ"], lw=2, color="black", label=r"$\hat\mu$")
ax.axhline(1_000.0 * upper_band, lw=2, ls="--", color="gray", label=r"$\hat\mu \pm 2\hat s.e.$")
ax.axhline(1_000.0 * lower_band, lw=2, ls="--", color="gray")
ax.set_xlabel("detection error probability (percent)")
ax.set_ylabel(r"mean consumption growth ($\times 10^{-3}$)")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("2 standard deviation band", fontsize=10)
ax.set_xlim(0.0, 50.0)
ax.set_ylim(3.0, 6.0)

plt.tight_layout()
plt.show()
```

**Left panel.**
Postwar U.S. log consumption is shown alongside two deterministic trend lines:
the approximating-model drift $\mu$ and the worst-case drift $\mu + \sigma_\varepsilon w(\theta)$ for $p(\theta^{-1}) = 0.20$.
The plotted consumption series is constructed from FRED data following the processing pipeline described in the Data section above.
The two trends are close enough that, even with decades of data, it is hard to distinguish them by eye.

**Right panel.**
As the detection-error probability rises (models become harder to tell apart), the worst-case mean growth rate moves back toward $\hat\mu$.
The dashed gray lines mark a two-standard-error band around the maximum-likelihood estimate of $\mu$.
Even at detection probabilities in the 5--20% range, the worst-case drift remains inside (or very near) this confidence band.

The upshot: drift distortions that are economically large --- large enough to generate substantial model-uncertainty premia --- are statistically small relative to sampling uncertainty in $\hat\mu$.
A dogmatic Bayesian who conditions on a single approximating model and updates using Bayes' law
will not learn her way out of this problem in samples of the length available.
Robustness concerns survive long histories precisely because the low-frequency features that matter most for pricing are the hardest to pin down.

## Concluding remarks

The title asks a question: are large risk premia prices of **variability** (atemporal risk aversion)
or prices of **doubts** (model uncertainty)?

The analysis above shows that the answer cannot be settled by asset-pricing data alone,
because the two interpretations are observationally equivalent.
But the choice matters enormously for what we conclude.

Under the risk-aversion reading, high Sharpe ratios imply that consumers would pay a great deal to smooth
known aggregate consumption fluctuations.
Under the robustness reading, those same Sharpe ratios tell us consumers would pay a great deal
to resolve uncertainty about which probability model governs consumption growth --- a fundamentally different policy object.

Three features of the analysis support the robustness reading:

1. Detection-error probabilities provide a more stable calibration language than $\gamma$: the two consumption models that required very different $\gamma$ values to match the data yield nearly identical pricing implications when indexed by detectability.
2. The welfare gains implied by asset prices decompose overwhelmingly into a model-uncertainty component, with the pure risk component remaining small --- consistent with Lucas's original finding.
3. The drift distortions that drive pricing are small enough to hide inside standard-error bands, so finite-sample learning cannot eliminate the consumer's fears.

Whether one ultimately prefers the risk or the uncertainty interpretation, the framework
clarifies that the question is not about the size of risk premia but about the economic object those premia identify.
