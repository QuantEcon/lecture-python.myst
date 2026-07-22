---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(phillips_drifts_volatilities)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img
                    style="width:250px;display:inline;"
                    width="250px"
                    src="https://assets.quantecon.org/img/qe-menubar-logo.svg"
                    alt="QuantEcon"
                >
        </a>
</div>
```

# Drifts and Volatilities

```{contents} Contents
:depth: 2
```

## Overview

The lectures in this section have told a story about how a government's *model*
of the Phillips curve, and the *policy* it induces, can drift over time.

In {doc}`phillips_learning` and {doc}`phillips_escaping_nash` a government that
fits and refits an approximating Phillips curve is repeatedly pushed away from a
bad {doc}`self-confirming equilibrium <phillips_self_confirming>` along an
*escape route*, while {doc}`phillips_priors` and {doc}`phillips_lost_conquest`
use drifting beliefs to interpret the rise and fall of American inflation.

Those lectures were mostly about *theory*.

This lecture turns to the *data*.

It studies {cite:t}`CogleySargent2005`, "Drifts and Volatilities: Monetary
Policies and Outcomes in the Post WWII US", which asks a deceptively simple
question:

> When we look at postwar U.S. time series on inflation, unemployment, and
> interest rates, do we see evidence that the dynamics have *drifted*?

Tim Cogley and Thomas Sargent began this work as an empirical companion to the
*Conquest* book {cite}`Sargent1999` and the escape-route papers
{cite}`ChoWilliamsSargent2002`.

It is also a response to searching comments by {cite:t}`Sims2001comment` and
{cite:t}`Stock2001comment` on an earlier paper {cite}`CogleySargent2001`, and it
grew into a friendly debate with {cite:t}`SimsZha2006` and
{cite:t}`BernankeMihov1998` about a question that organizes this whole section:

*Was the Great Inflation of the 1970s and its conquest in the 1980s a story of
bad policy, or of bad luck?*

To let the data speak to that question we need a statistical model flexible
enough to accommodate *both* answers.

That model is a *Bayesian vector autoregression whose coefficients drift as
random walks and whose shock variances evolve as stochastic volatilities*.

Fitting it requires a Markov chain Monte Carlo algorithm that combines the
{doc}`Kalman filter <kalman>`, the forward-filter/backward-sample smoother of
{cite:t}`CarterKohn1994`, and the stochastic-volatility sampler of
{cite:t}`Jacquier1994`.

We work through the data transformation, prior, sampler, and main empirical
results in the order in which they arise.

All model and post-processing routines are defined immediately below the objects
that they implement.

Let's start with some imports and the path to the frozen data.

```{code-cell} ipython3
from pathlib import Path
import hashlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.special import expit
from scipy.stats import invwishart


def locate_data_assets():
    """Find assets from either a MyST build or the repository root."""
    relative = Path('_static/lecture_specific/phillips_drifts_volatilities')
    candidates = (relative, Path('lectures') / relative)
    for candidate in candidates:
        if (candidate / 'NEWQDATA.csv').is_file():
            return candidate
    searched = ', '.join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(f'NEWQDATA.csv was not found; searched {searched}')


asset_path = locate_data_assets()
data_path = asset_path / 'NEWQDATA.csv'

data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
assert data_sha256 == (
    '98fb884bf59fbf211a4095c558b210ff8df471f863357803cc3ad88d0066bcaf'
)
```

## Bad policy or bad luck?

Two respectable views compete to explain the American Great Inflation.

The **bad policy** view is the one dramatized throughout this section and in the
*Conquest* book {cite}`Sargent1999`.

Something about Arthur Burns's *model* of the economy, his *patience*, or his
inability to *commit* to a better rule led the Federal Reserve to administer
monetary policy in a way that produced the greatest peacetime inflation in U.S.
history, while an improved model, more patience, or greater discipline led Paul
Volcker to conquer it {cite}`DeLong1997,Taylor1997comment`.

On this view, what changed between the 1970s and the 1980s was the *systematic
part* of policy, namely the way the Fed's interest-rate setting responded to
inflation and unemployment.

The **bad luck** view says something quite different.

What distinguished the Burns and Volcker eras was not their models or policies,
but the *shocks* that hit the economy.

On this view the *coefficients* of a reduced-form description of the economy
were essentially constant, and what changed was the *size* of the disturbances,
namely the *volatility*.

{cite:t}`BernankeMihov1998` and {cite:t}`SimsZha2006` marshaled evidence for
this second view, in part by applying classical tests that *failed to reject*
the hypothesis that VAR coefficients were time invariant.

How can we discriminate?

A model with constant coefficients and constant volatility cannot represent the
bad-luck story, while a model with drifting coefficients but constant volatility
risks attributing to drift what is really changing volatility.

So Cogley and Sargent build a model that has room for *both* channels at once,
and they let a Bayesian posterior sort out how much of each the data call for.

## A VAR with drifting coefficients and stochastic volatility

Let the variables be ordered as nominal interest, transformed unemployment, and
inflation,

$$
y_t = \begin{bmatrix} i_t & u_t & \pi_t \end{bmatrix}'.
$$

The measurement equation is a VAR with two lags and date-specific coefficients,

```{math}
:label: csdv_measurement
y_t = X_t'\theta_t + \varepsilon_t,
\qquad
X_t' = I_3 \otimes \begin{bmatrix} 1 & y_{t-1}' & y_{t-2}' \end{bmatrix}.
```

Each equation has an intercept and six lag coefficients, so $\theta_t$ contains
$3(1+2\times 3)=21$ elements.

The following function implements this equation-major coefficient stacking and
its companion representation.

```{code-cell} ipython3
n_variables = 3
n_lags = 2
n_regressors = 1 + n_variables * n_lags
n_coefficients = n_variables * n_regressors


def companion_matrix(theta):
    """Return the intercept and companion matrix for one coefficient vector."""
    blocks = np.asarray(theta, dtype=float).reshape(n_variables, n_regressors)
    intercept = np.r_[blocks[:, 0], np.zeros(n_variables)]
    companion = np.zeros((n_variables * n_lags, n_variables * n_lags))
    companion[:n_variables] = blocks[:, 1:]
    companion[n_variables:, :n_variables] = np.eye(n_variables)
    return intercept, companion


def design_matrix(regressors):
    """Return the observation matrix X_t prime for one date."""
    return np.kron(np.eye(n_variables), np.asarray(regressors, dtype=float))
```

The coefficient vector follows a driftless random walk,

```{math}
:label: csdv_transition
\theta_t = \theta_{t-1} + v_t,
\qquad
v_t \sim N(0,Q).
```

Cogley and Sargent rule out explosive paths by retaining a path only when the
companion matrix is stable at every date and using the truncated prior

```{math}
:label: csdv_stability
p(\theta^T,Q) \propto I(\theta^T) f(\theta^T \mid Q) f(Q),
```

where $I(\theta^T)=1$ denotes a stable path.

This restriction encodes the belief that the economy did not in fact follow an
explosive path.

This restriction also tilts the marginal prior for $Q$ toward values that are
less likely to generate explosive coefficient paths.

The code below applies the stability restriction to an entire trajectory rather
than clipping individual roots.

```{code-cell} ipython3
def companion_roots(theta_path):
    """Return all companion roots along a path with shape (21, T)."""
    theta_path = np.asarray(theta_path, dtype=float)
    if theta_path.ndim == 1:
        theta_path = theta_path[:, None]
    companions = np.stack(
        [
            companion_matrix(theta_path[:, t])[1]
            for t in range(theta_path.shape[1])
        ]
    )
    return np.linalg.eigvals(companions)


def is_stable(theta_path):
    """Test whether every companion root is strictly inside the unit circle."""
    return bool(np.max(np.abs(companion_roots(theta_path))) < 1)
```

The reduced-form innovation covariance changes over time according to

```{math}
:label: csdv_covariance
\varepsilon_t = R_t^{1/2}\xi_t,
\qquad
\xi_t \sim N(0,I_3),
\qquad
R_t = B^{-1} H_t B^{-1\prime},
```

where

$$
B =
\begin{bmatrix}
1 & 0 & 0 \\
\beta_{21} & 1 & 0 \\
\beta_{31} & \beta_{32} & 1
\end{bmatrix},
\qquad
H_t = \operatorname{diag}(h_{1t},h_{2t},h_{3t}).
$$

The matrix $B$ orthogonalizes the reduced-form innovations but is not
interpreted as a structural identification scheme.

The diagonal elements $h_{it}$ let the size of each orthogonalized shock wax and
wane over time.

The next two functions construct the triangular factor and the reduced-form
innovation covariance.

```{code-cell} ipython3
def b_matrix(beta):
    """Construct B from beta_21, beta_31, and beta_32."""
    matrix = np.eye(n_variables)
    matrix[1, 0], matrix[2, 0], matrix[2, 1] = np.asarray(beta, dtype=float)
    return matrix


def innovation_covariance(h, beta):
    """Construct R_t from one vector of orthogonalized variances."""
    inverse = np.linalg.inv(b_matrix(beta))
    return inverse @ np.diag(h) @ inverse.T
```

Each diagonal volatility is a geometric random walk,

```{math}
:label: csdv_volatility
\log h_{it} = \log h_{i,t-1} + \sigma_i \eta_{it},
\qquad
\eta_{it} \sim N(0,1).
```

The standardized measurement innovations, coefficient innovations, and
volatility innovations are mutually independent.

Setting $Q=0$ produces constant coefficients with drifting volatility, while
holding $H_t$ fixed produces drifting coefficients with constant volatility.

The posterior contains the full paths $\theta^T$ and $H^T$ together with $Q$,
$\beta$, and $(\sigma_1,\sigma_2,\sigma_3)$.

This posterior has thousands of dimensions, which is why we simulate it one
conditional block at a time.

## The data

We begin with a frozen quarterly U.S. dataset ending in 2000Q4.

Inflation is the log difference of the seasonally adjusted CPI for all urban
consumers, point sampled in the third month of each quarter.

Unemployment is the quarterly average of the seasonally adjusted civilian
unemployment rate and enters the VAR as $0.01\log[u/(1-u)]$, a logit
transformation that maps a bounded rate into an unconstrained variable.

The nominal interest rate is the log of one plus the three-month Treasury-bill
rate, averaged over daily observations in the first month of each quarter and
expressed as a quarterly fraction.

The frozen file starts in 1948Q2 because its first inflation observation is
already differenced, so two VAR lags make 1948Q4 the first usable regression
date.

Its SHA-256 checksum is asserted above so that every run uses the same
historical input.

The following cell performs every transformation and constructs the VAR(2) data
directly from the frozen series.

```{code-cell} ipython3
def prepare_data(source, ordering=('i', 'u', 'pi')):
    """Transform a quarterly table and construct the VAR data."""
    if isinstance(source, (str, Path)):
        table = pd.read_csv(source)
    else:
        table = source.copy()
    variables = {
        'i': table['y3'].to_numpy(dtype=float),
        'u': 0.01 * np.log(
            table['ur'].to_numpy(dtype=float)
            / (1 - table['ur'].to_numpy(dtype=float))
        ),
        'pi': table['dp'].to_numpy(dtype=float),
    }
    if sorted(ordering) != ['i', 'pi', 'u']:
        raise ValueError("ordering must be a permutation of ('i', 'u', 'pi')")
    raw_y = np.column_stack([variables[name] for name in ordering])
    raw_dates = table['date'].to_numpy(dtype=float)
    regressors = np.ones((len(table) - n_lags, n_regressors))
    for lag in range(1, n_lags + 1):
        left = 1 + n_variables * (lag - 1)
        regressors[:, left:left + n_variables] = raw_y[n_lags-lag:-lag]
    targets = raw_y[n_lags:]
    dates = raw_dates[n_lags:]
    n_training = 4 * 11 - n_lags - 1
    return {
        'raw_dates': raw_dates,
        'raw_y': raw_y,
        'prior_dates': dates[:n_training],
        'prior_y': targets[:n_training],
        'prior_x': regressors[:n_training],
        'dates': dates[n_training:],
        'y': targets[n_training:],
        'x': regressors[n_training:],
    }


data = prepare_data(data_path)

data_summary = pd.Series(
    {
        'ordering': 'interest, unemployment, inflation',
        'prior sample': '1948Q4--1958Q4',
        'prior observations': len(data['prior_dates']),
        'posterior sample': '1959Q1--2000Q4',
        'posterior observations': len(data['dates']),
        'VAR lags': n_lags,
        'coefficient dimension': n_coefficients,
    },
    name='value',
)

data_summary.to_frame()
```

The first 41 usable observations calibrate the prior, while the remaining 168
observations form the posterior sample.

Let us view the data in familiar economic units.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: U.S. macroeconomic series
    name: fig-csdv-historical-data
---
dates_raw = data['raw_dates']
interest = 400 * np.expm1(data['raw_y'][:, 0])
unemployment = 100 * expit(100 * data['raw_y'][:, 1])
inflation = 400 * data['raw_y'][:, 2]

fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
axes[0].plot(dates_raw, inflation, lw=2)
axes[0].set_ylabel('inflation (annual %)')
axes[1].plot(dates_raw, unemployment, lw=2)
axes[1].set_ylabel('unemployment (%)')
axes[2].plot(dates_raw, interest, lw=2)
axes[2].set_ylabel('interest (annual %)')
axes[2].set_xlabel('year')
plt.tight_layout()
fig.show()
```

The rise in inflation and nominal interest rates during the 1970s, the Volcker
disinflation, and the subsequent moderation are all visible before a model is
fitted.

The question is what a flexible statistical model makes of them.

## Priors

The prior blocks are independent and deliberately weak so that, in the authors'
phrase, "the data are free to speak."

They are calibrated from a time-invariant VAR fitted to the short 1948--1958
training sample.

The initial coefficient prior is a stable, truncated Gaussian,

$$
p(\theta_0) \propto I(\theta_0)N(\bar\theta,\bar P),
$$

where $\bar\theta$ and $\bar P$ come from a constant-coefficient seemingly
unrelated regression fitted to the 1948--1958 training sample.

Because all three equations have identical regressors, the SUR coefficient
estimates equal equation-by-equation OLS estimates, which lets us implement the
calibration compactly.

```{code-cell} ipython3
def sur_prior(y, x):
    """Calibrate the Gaussian coefficient prior from a constant VAR."""
    xx_inverse = np.linalg.inv(x.T @ x)
    coefficients = xx_inverse @ x.T @ y
    residuals = y - x @ coefficients
    residual_covariance = np.cov(residuals, rowvar=False, ddof=1)
    theta = coefficients.T.reshape(-1)
    covariance = np.kron(residual_covariance, xx_inverse)
    return theta, covariance, residual_covariance


theta_bar, p_bar, r_bar = sur_prior(data['prior_y'], data['prior_x'])

assert is_stable(theta_bar)
```

The coefficient-drift covariance has the inverse-Wishart prior

```{math}
:label: csdv_q_prior
Q \sim IW_{21}\left(T_0,T_0\bar Q\right),
\qquad
T_0 = 22,
\qquad
\bar Q = \gamma^2 \bar P,
\qquad
\gamma^2 = 3.5\times 10^{-4}.
```

The convention in {eq}`csdv_q_prior` lists degrees of freedom first and the
inverse-Wishart scale matrix second.

Because $T_0$ is only one greater than the dimension of $\theta_t$, the prior is
proper but has no finite mean.

The matrix $\bar Q$ is therefore a conservative scale calibration rather than
the expectation of $Q$.

This calibration favors slow coefficient drift before the posterior sees the
main sample.

The remaining priors are

$$
\begin{aligned}
\log h_{i0} &\sim N(\log \bar R_{ii},10), \\
\beta &\sim N(0,10000I_3), \\
\sigma_i^2 &\sim IG\left(\frac{1}{2},\frac{0.01^2}{2}\right),
\end{aligned}
$$

where $\bar R$ is the residual covariance from the training-sample regression.

The next function gathers these hyperparameters so that the same sampler can be
applied to every variable ordering.

```{code-cell} ipython3
def calibrate_prior(model_data, gamma_squared=3.5e-4):
    """Return every prior block calibrated to one variable ordering."""
    theta_mean, theta_covariance, residual_covariance = sur_prior(
        model_data['prior_y'], model_data['prior_x']
    )
    degrees_freedom = n_coefficients + 1
    q_center = gamma_squared * theta_covariance
    return {
        'theta_mean': theta_mean,
        'theta_covariance': theta_covariance,
        'q_center': q_center,
        'q_scale': degrees_freedom * q_center,
        'q_degrees_freedom': degrees_freedom,
        'log_h_mean': np.log(np.diag(residual_covariance)),
        'log_h_variance': 10.0,
        'beta_mean': np.zeros(3),
        'beta_variance': 10000.0,
        'sigma_degrees_freedom': 1.0,
        'sigma_scale': 0.01**2,
        'gamma_squared': gamma_squared,
    }


prior = calibrate_prior(data)
q_bar = prior['q_center']

prior_summary = pd.Series(
    {
        'dim(theta)': n_coefficients,
        'T0': prior['q_degrees_freedom'],
        'gamma squared': prior['gamma_squared'],
        'trace(Q bar)': np.trace(q_bar),
        'log-h prior variance': prior['log_h_variance'],
        'beta prior variance': prior['beta_variance'],
        'sigma-squared IG shape': prior['sigma_degrees_freedom'] / 2,
        'sigma-squared IG scale': prior['sigma_scale'] / 2,
    },
    name='value',
)

prior_summary.to_frame()
```

## A Metropolis-within-Gibbs sampler

We simulate the posterior by cycling through the five conditional blocks in
Appendix B of {cite:t}`CogleySargent2005`.

For the stability-truncated coefficient block, we replace repeated whole-path
rejection with the exact elliptical-slice transition described below; the
target posterior is unchanged.

1. An auxiliary Gaussian coefficient path is drawn by a Carter--Kohn
   forward-filter/backward-sample step and used to update the stable path by
   elliptical slice sampling.

2. The drift covariance $Q$ is drawn from an inverse-Wishart distribution
   conditional on the coefficient innovations.

3. The volatility innovation variances $\sigma_i^2$ are drawn from inverse-gamma
   distributions conditional on the volatility increments.

4. The covariance parameters $\beta$ are drawn from two transformed Gaussian
   regressions among the VAR residuals.

5. The volatility paths $H^T$ are drawn one date at a time by a
   Jacquier--Polson--Rossi Metropolis step.

This ordering matters: every block in a sweep is paired with the values on which
it was actually conditioned.

### Coefficient path

Conditional on $R^T$ and $Q$, a forward Kalman filter followed by the
Carter--Kohn backward simulator draws the entire coefficient path
{cite}`CarterKohn1994`.

The forward pass is a Kalman filter, while the backward pass samples
$\theta_T,\theta_{T-1},\ldots,\theta_0$ in reverse with each state conditioned
on the draw that follows it.

Sampling $\theta_0$ is essential.

It supplies all $T$ random-walk increments to the conjugate update for $Q$;
integrating it out while retaining an inverse-Wishart $Q$ update would be
inconsistent.

```{code-cell} ipython3
def covariance_root(matrix):
    """Return a numerically stable lower covariance factor."""
    matrix = 0.5 * (matrix + matrix.T)
    scale = max(1.0, np.max(np.abs(np.diag(matrix))))
    return np.linalg.cholesky(matrix + 1e-12 * scale * np.eye(len(matrix)))


def draw_coefficient_path(
    y, x, q, h, beta, prior, rng, return_mean=False
):
    """Draw theta_0,...,theta_T and optionally return its smoothing mean."""
    periods = len(y)
    filtered_mean = np.empty((periods + 1, n_coefficients))
    filtered_covariance = np.empty(
        (periods + 1, n_coefficients, n_coefficients)
    )
    predicted_covariance = np.empty_like(filtered_covariance)
    filtered_mean[0] = prior['theta_mean']
    filtered_covariance[0] = prior['theta_covariance']
    predicted_covariance[0] = prior['theta_covariance']
    for t in range(1, periods + 1):
        observation = design_matrix(x[t - 1])
        prediction_covariance = filtered_covariance[t - 1] + q
        r_t = innovation_covariance(h[t], beta)
        forecast_covariance = (
            observation @ prediction_covariance @ observation.T + r_t
        )
        gain = linalg.solve(
            forecast_covariance,
            (prediction_covariance @ observation.T).T,
            assume_a='pos',
        ).T
        mean = filtered_mean[t - 1]
        mean = mean + gain @ (y[t - 1] - observation @ mean)
        covariance = (
            prediction_covariance
            - gain @ observation @ prediction_covariance
        )
        covariance = 0.5 * (covariance + covariance.T)
        filtered_mean[t] = mean
        filtered_covariance[t] = covariance
        predicted_covariance[t] = prediction_covariance
    path = np.empty((n_coefficients, periods + 1))
    if not return_mean:
        path[:, -1] = (
            filtered_mean[-1]
            + covariance_root(filtered_covariance[-1])
            @ rng.standard_normal(n_coefficients)
        )
        for t in range(periods - 1, -1, -1):
            smoother = linalg.solve(
                predicted_covariance[t + 1],
                filtered_covariance[t].T,
                assume_a='pos',
            ).T
            mean = filtered_mean[t] + smoother @ (
                path[:, t + 1] - filtered_mean[t]
            )
            covariance = (
                filtered_covariance[t]
                - smoother @ predicted_covariance[t + 1] @ smoother.T
            )
            path[:, t] = mean + covariance_root(covariance) @ (
                rng.standard_normal(n_coefficients)
            )
        return path

    smoothed_mean = np.empty_like(path)
    centered_draw = np.empty_like(path)
    smoothed_mean[:, -1] = filtered_mean[-1]
    centered_draw[:, -1] = covariance_root(filtered_covariance[-1]) @ (
        rng.standard_normal(n_coefficients)
    )
    for t in range(periods - 1, -1, -1):
        smoother = linalg.solve(
            predicted_covariance[t + 1],
            filtered_covariance[t].T,
            assume_a='pos',
        ).T
        smoothed_mean[:, t] = filtered_mean[t] + smoother @ (
            smoothed_mean[:, t + 1] - filtered_mean[t]
        )
        covariance = (
            filtered_covariance[t]
            - smoother @ predicted_covariance[t + 1] @ smoother.T
        )
        centered_draw[:, t] = (
            smoother @ centered_draw[:, t + 1]
            + covariance_root(covariance) @ rng.standard_normal(n_coefficients)
        )
    return smoothed_mean + centered_draw, smoothed_mean


def draw_stable_coefficient_path(
    current, y, x, q, h, beta, prior, rng, max_contractions=100
):
    """Elliptical-slice update of the stability-truncated Gaussian path."""
    if not is_stable(current):
        raise ValueError('the elliptical-slice update needs a stable path')
    gaussian_draw, mean = draw_coefficient_path(
        y, x, q, h, beta, prior, rng, return_mean=True
    )
    current_centered = current - mean
    innovation = gaussian_draw - mean
    angle = rng.uniform(0, 2 * np.pi)
    lower = angle - 2 * np.pi
    upper = angle
    for contractions in range(max_contractions + 1):
        proposal = (
            mean
            + current_centered * np.cos(angle)
            + innovation * np.sin(angle)
        )
        if is_stable(proposal):
            return proposal, contractions
        if angle < 0:
            lower = angle
        else:
            upper = angle
        angle = rng.uniform(lower, upper)
    raise RuntimeError('elliptical-slice stability bracket did not contract')
```

### Drift covariance

Conditional on the coefficient increments, $Q$ has an inverse-Wishart full
conditional.

The retained simulation draws $Q$ from a scale matrix formed by its prior scale
and all $T$ squared increments from $\theta_0$ through $\theta_T$.

```{code-cell} ipython3
def draw_q(theta_path, prior, rng):
    """Draw Q conditional on the sampled coefficient path."""
    increments = np.diff(theta_path, axis=1)
    scale = prior['q_scale'] + increments @ increments.T
    degrees_freedom = prior['q_degrees_freedom'] + increments.shape[1]
    return invwishart.rvs(df=degrees_freedom, scale=scale, random_state=rng)
```

### Volatility parameters and paths

Conditional on the volatility increments, each $\sigma_i^2$ has an inverse-gamma
full conditional.

Conditional on the VAR residuals and $H^T$, the free elements of $B$ are drawn
from two Gaussian regressions.

Conditional on the orthogonalized residuals, each volatility state is updated
with the single-site Metropolis step of {cite:t}`Jacquier1994`.

The random-walk neighbors determine the Gaussian proposal for a log volatility,
while the corresponding orthogonalized residual determines whether that proposal
is accepted.

The following code implements the conditional updates, including the different
endpoint proposals.

```{code-cell} ipython3
def var_residuals(y, x, theta_path):
    """Return residuals with shape (T, 3)."""
    if theta_path.shape[1] == len(y) + 1:
        theta_path = theta_path[:, 1:]
    if theta_path.shape[1] != len(y):
        raise ValueError('theta_path must contain T or T + 1 states')
    coefficients = theta_path.T.reshape(len(y), n_variables, n_regressors)
    fitted = np.einsum('tk,tnk->tn', x, coefficients)
    return y - fitted


def draw_sigma(h, prior, rng):
    """Draw the three log-volatility innovation standard deviations."""
    increments = np.diff(np.log(h), axis=0)
    shape = (prior['sigma_degrees_freedom'] + increments.shape[0]) / 2
    scales = (prior['sigma_scale'] + np.sum(increments**2, axis=0)) / 2
    sigma_squared = scales / rng.gamma(shape, 1.0, size=n_variables)
    return np.sqrt(sigma_squared)


def draw_beta(residuals, h, prior, rng):
    """Draw the free elements of B from transformed Gaussian regressions."""
    beta = np.empty(3)
    offset = 0
    for equation in range(1, n_variables):
        standardized = residuals / np.sqrt(h[1:, equation])[:, None]
        dependent = standardized[:, equation]
        regressors = -standardized[:, :equation]
        prior_precision = np.eye(equation) / prior['beta_variance']
        covariance = np.linalg.inv(prior_precision + regressors.T @ regressors)
        prior_slice = prior['beta_mean'][offset:offset + equation]
        mean = covariance @ (
            prior_precision @ prior_slice + regressors.T @ dependent
        )
        beta[offset:offset + equation] = (
            mean + covariance_root(covariance) @ rng.standard_normal(equation)
        )
        offset += equation
    return beta


def accept_volatility(proposal, current, residual, rng):
    """Apply the Jacquier--Polson--Rossi likelihood acceptance step."""
    log_ratio = (
        -0.5 * np.log(proposal)
        - residual**2 / (2 * proposal)
        + 0.5 * np.log(current)
        + residual**2 / (2 * current)
    )
    return proposal if np.log(rng.random()) <= min(0.0, log_ratio) else current


def draw_volatility_path(h, residuals, beta, sigma, prior, rng):
    """Update all stochastic-volatility states one date at a time."""
    periods = len(residuals)
    orthogonalized = (b_matrix(beta) @ residuals.T).T
    updated = np.empty_like(h)
    for equation in range(n_variables):
        variance = sigma[equation]**2
        initial_variance = (
            prior['log_h_variance'] * variance
            / (variance + prior['log_h_variance'])
        )
        initial_mean = initial_variance * (
            prior['log_h_mean'][equation] / prior['log_h_variance']
            + np.log(h[1, equation]) / variance
        )
        updated[0, equation] = np.exp(
            initial_mean + np.sqrt(initial_variance) * rng.standard_normal()
        )
        for t in range(1, periods):
            mean = 0.5 * (
                np.log(updated[t - 1, equation]) + np.log(h[t + 1, equation])
            )
            proposal = np.exp(
                mean + np.sqrt(variance / 2) * rng.standard_normal()
            )
            updated[t, equation] = accept_volatility(
                proposal,
                h[t, equation],
                orthogonalized[t - 1, equation],
                rng,
            )
        proposal = np.exp(
            np.log(updated[-2, equation])
            + sigma[equation] * rng.standard_normal()
        )
        updated[-1, equation] = accept_volatility(
            proposal,
            h[-1, equation],
            orthogonalized[-1, equation],
            rng,
        )
    return updated
```

### Complete sampler

The restricted posterior assigns zero density to coefficient paths that are
explosive at any date, including $\theta_0$.

Repeated independent draws from the Gaussian coefficient-path conditional are
wasteful because requiring all 169 states to be stable can lead to many
full-path rejections.

We instead use an elliptical-slice transition: one Carter--Kohn innovation
defines an ellipse through the current stable path, and a cheap one-dimensional
angle bracket contracts until the proposed path is stable.

This transition leaves the same stability-truncated Gaussian conditional
invariant and computes the Kalman filter only once per sweep.

The next function composes the five blocks and includes a stochastic-volatility
warm-up.

```{code-cell} ipython3
def initial_volatilities(y, prior):
    """Construct the sampler's initial volatility path."""
    changes = np.diff(y, axis=0)
    centered = changes - changes.mean(axis=0)
    log_h = np.empty((len(y) + 1, n_variables))
    log_h[:2] = prior['log_h_mean']
    log_h[2:] = np.log(np.maximum(centered**2, np.finfo(float).tiny))
    return np.exp(log_h)


def run_sampler(
    y,
    x,
    prior,
    n_sweeps=1_000,
    burn=500,
    thin=1,
    seed=42,
    warmup=200,
    max_contractions=100,
    stable=True,
    retain=('S0D', 'SD', 'QD', 'HD', 'CD', 'VD', 'stable_draw'),
    progress_every=0,
):
    """Run a coherent Gibbs sampler for the unrestricted or stable posterior.

    For the stable posterior, an elliptical-slice transition updates the FFBS
    path inside its stability-truncated Gaussian full conditional.
    """
    if not (0 <= burn < n_sweeps and thin >= 1):
        raise ValueError('require 0 <= burn < n_sweeps and thin >= 1')
    if (n_sweeps - burn) % thin:
        raise ValueError('(n_sweeps - burn) must be divisible by thin')
    valid_retain = {'S0D', 'SD', 'QD', 'HD', 'CD', 'VD', 'stable_draw'}
    unknown = set(retain) - valid_retain
    if unknown:
        raise ValueError(f'unknown retained arrays: {sorted(unknown)}')

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    h = initial_volatilities(y, prior)
    beta = prior['beta_mean'].copy()
    warm_theta = np.repeat(prior['theta_mean'][:, None], len(y), axis=1)
    warm_residuals = var_residuals(y, x, warm_theta)
    for _ in range(warmup):
        sigma = draw_sigma(h, prior, rng)
        beta = draw_beta(warm_residuals, h, prior, rng)
        h = draw_volatility_path(
            h, warm_residuals, beta, sigma, prior, rng
        )

    q = prior['q_center'].copy()
    theta = np.repeat(
        prior['theta_mean'][:, None], len(y) + 1, axis=1
    )
    if stable and not is_stable(theta):
        raise ValueError('the prior mean does not provide a stable start')
    slice_contractions = 0
    maximum_slice_contractions = 0

    retained = {name: [] for name in retain}
    saved_stability = []
    for sweep in range(1, n_sweeps + 1):
        if stable:
            theta, contractions = draw_stable_coefficient_path(
                theta,
                y,
                x,
                q,
                h,
                beta,
                prior,
                rng,
                max_contractions=max_contractions,
            )
        else:
            theta = draw_coefficient_path(y, x, q, h, beta, prior, rng)
            contractions = 0
        slice_contractions += contractions
        maximum_slice_contractions = max(
            maximum_slice_contractions, contractions
        )
        q = draw_q(theta, prior, rng)
        residuals = var_residuals(y, x, theta)
        sigma = draw_sigma(h, prior, rng)
        beta = draw_beta(residuals, h, prior, rng)
        h = draw_volatility_path(
            h, residuals, beta, sigma, prior, rng
        )

        if sweep > burn and (sweep - burn) % thin == 0:
            path_is_stable = is_stable(theta)
            saved_stability.append(path_is_stable)
            values = {
                'S0D': theta[:, 0],
                'SD': theta[:, 1:],
                'QD': q,
                'HD': h,
                'CD': beta,
                'VD': sigma,
                'stable_draw': path_is_stable,
            }
            for name in retained:
                retained[name].append(np.asarray(values[name]).copy())

        if progress_every and sweep % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f'{sweep:,}/{n_sweeps:,} sweeps; '
                f'{slice_contractions:,} slice contractions; '
                f'{elapsed / 60:.1f} minutes',
                flush=True,
            )

    stack_axis = {
        'S0D': 1,
        'SD': 2,
        'QD': 2,
        'HD': 2,
        'CD': 1,
        'VD': 1,
        'stable_draw': 0,
    }
    result = {
        name: np.stack(values, axis=stack_axis[name])
        for name, values in retained.items()
    }
    result['diagnostics'] = {
        'sampler_version': 3,
        'seed': int(seed),
        'stable_restriction': bool(stable),
        'n_sweeps': int(n_sweeps),
        'burn': int(burn),
        'thin': int(thin),
        'warmup': int(warmup),
        'retained_draws': int((n_sweeps - burn) // thin),
        'slice_contractions': int(slice_contractions),
        'mean_slice_contractions': float(slice_contractions / n_sweeps),
        'maximum_slice_contractions': int(maximum_slice_contractions),
        'retained_stability_rate': float(np.mean(saved_stability)),
        'elapsed_seconds': float(time.perf_counter() - started),
    }
    return result
```

The executable version below uses 1,000 sweeps, discards the first 500, and
retains the remaining 500.

It uses the complete historical sample and the stable ordering $(i,u,\pi)$, and
it does not load posterior draws or precomputed results.

This short teaching run is exploratory rather than publication-precision
inference, so we report effective sample sizes and Monte Carlo standard errors
alongside its estimates.

```{code-cell} ipython3
posterior = run_sampler(
    data['y'],
    data['x'],
    prior,
    n_sweeps=1_000,
    burn=500,
    thin=1,
    seed=42,
    warmup=200,
    stable=True,
    progress_every=0,
)

def validate_posterior(result, periods):
    """Check posterior shapes, finiteness, positivity, and stability."""
    draws = result['diagnostics']['retained_draws']
    expected = {
        'S0D': (n_coefficients, draws),
        'SD': (n_coefficients, periods, draws),
        'QD': (n_coefficients, n_coefficients, draws),
        'HD': (periods + 1, n_variables, draws),
        'CD': (3, draws),
        'VD': (3, draws),
        'stable_draw': (draws,),
    }
    assert {name: result[name].shape for name in expected} == expected
    assert all(np.all(np.isfinite(result[name])) for name in expected)
    assert np.all(result['HD'] > 0)
    assert np.all(result['VD'] > 0)
    assert np.all(result['stable_draw'])
    return expected


expected_shapes = validate_posterior(posterior, len(data['dates']))

pd.Series(posterior['diagnostics'], name='value').to_frame()
```

## What the data say

We summarize the posterior by its mean coefficient path $E(\theta_t\mid T)$ and
mean covariance path $E(R_t\mid T)$, and then read the economically interesting
objects from them.

The retained in-memory draws are also used for a few probability statements and
Monte Carlo diagnostics.

### The rate and structure of drift

The trace of $Q$ measures the total rate of coefficient drift, with
$\operatorname{tr}(Q)=0$ corresponding to constant coefficients.

The histogram shows the retained $Q$ draws and the prior scale.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Posterior coefficient-drift rate
    name: fig-csdv-drift-rate
---
trace_q = np.trace(posterior['QD'], axis1=0, axis2=1)
fig, ax = plt.subplots()
ax.hist(trace_q, bins=30, histtype='step', lw=2)
ax.axvline(np.trace(q_bar), color='C1', lw=2,
           label=r'prior $\mathrm{tr}(\bar Q)$')
ax.set_xlabel(r'$\mathrm{tr}(Q)$')
ax.set_ylabel('frequency')
ax.legend()
fig.show()
```

The posterior mean paths show visible drift concentrated in a subset of
coefficients.

We do not simulate the stability-truncated prior paths here because doing so by
whole-path rejection recreates the runtime problem that the elliptical-slice
update avoids.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Posterior mean coefficient paths
    name: fig-csdv-coefficient-paths
---
theta_mean = posterior['SD'].mean(axis=2)
coefficient_labels = (
    'constant',
    r'$i_{t-1}$',
    r'$u_{t-1}$',
    r'$\pi_{t-1}$',
    r'$i_{t-2}$',
    r'$u_{t-2}$',
    r'$\pi_{t-2}$',
)
equation_labels = (
    'interest equation',
    'unemployment equation',
    'inflation equation',
)


def plot_coefficient_blocks(axes, dates, theta_path):
    """Plot seven labeled coefficients for each VAR equation."""
    first_lines = None
    for equation, (ax, label) in enumerate(zip(axes, equation_labels)):
        start = equation * len(coefficient_labels)
        stop = start + len(coefficient_labels)
        lines = ax.plot(dates, theta_path[start:stop].T, lw=2)
        for line, coefficient_label in zip(lines, coefficient_labels):
            line.set_label(coefficient_label)
        if first_lines is None:
            first_lines = lines
        ax.axhline(0, color='0.65', lw=1)
        ax.set_xlabel('year')
        ax.set_ylabel(f'{label} coefficient')
    return first_lines


fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=True)
coefficient_lines = plot_coefficient_blocks(
    axes,
    data['dates'],
    theta_mean,
)
fig.legend(
    coefficient_lines,
    coefficient_labels,
    loc='lower center',
    ncol=4,
)
plt.tight_layout(rect=(0, 0.17, 1, 1))
fig.show()
```

We summarize drift for the stable ordering $(i,u,\pi)$.

```{code-cell} ipython3
def scalar_mcmc_ess(values):
    """Estimate scalar ESS with Geyer's initial-positive-sequence rule."""
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    n = len(centered)
    n_fft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=n_fft)
    autocovariance = np.fft.irfft(
        spectrum * spectrum.conj(), n=n_fft
    )[:n]
    autocovariance /= np.arange(n, 0, -1)
    if autocovariance[0] <= np.finfo(float).tiny:
        return float(n)
    rho = autocovariance / autocovariance[0]
    paired = rho[1:1 + 2 * ((n - 1) // 2)].reshape(-1, 2).sum(axis=1)
    nonpositive = np.flatnonzero(paired <= 0)
    if len(nonpositive):
        paired = paired[:nonpositive[0]]
    tau = max(1.0, 1.0 + 2.0 * paired.sum())
    return min(float(n), float(n / tau))


trace_q = np.trace(posterior['QD'], axis1=0, axis2=1)
q_mean = posterior['QD'].mean(axis=2)
trace_ess = scalar_mcmc_ess(trace_q)
trace_mcse = trace_q.std(ddof=1) / np.sqrt(trace_ess)
drift_summary = pd.Series(
    {
        r'posterior mean $\operatorname{tr}(Q)$': np.trace(q_mean),
        'posterior mean largest eigenvalue': np.linalg.eigvalsh(q_mean)[-1],
        r'prior $\operatorname{tr}(\bar Q)$': np.trace(q_bar),
    },
    name='estimate',
)
drift_summary.to_frame().round(4)
```

```{code-cell} ipython3
pd.Series(
    {
        'trace ESS': trace_ess,
        'trace MCSE': trace_mcse,
        'first-half trace mean': trace_q[:len(trace_q) // 2].mean(),
        'second-half trace mean': trace_q[len(trace_q) // 2:].mean(),
        'mean slice contractions per sweep': posterior['diagnostics'][
            'mean_slice_contractions'
        ],
        'maximum slice contractions': posterior['diagnostics'][
            'maximum_slice_contractions'
        ],
        'elapsed seconds': posterior['diagnostics']['elapsed_seconds'],
    },
    name='MCMC diagnostic',
).to_frame().round(4)
```

The estimate of $\operatorname{tr}(Q)$ is well above the conservative prior
scale, which points to economically meaningful coefficient drift.

Its precision should be judged from the ESS, MCSE, and split-chain means rather
than from the displayed decimal places.

The same Monte Carlo caveat applies to the eigenvalue shares and nonlinear
feature paths below.

The analysis that follows adopts the $(i,u,\pi)$ ordering, which places the
nominal interest rate first and inflation last.

Diagonalizing the posterior mean of $Q$ reveals that the drift is low
dimensional.

The following eigendecomposition summarizes the posterior mean of $Q$.

```{code-cell} ipython3
q_mean = posterior['QD'].mean(axis=2)
q_eigenvalues = np.linalg.eigvalsh(q_mean)[::-1]
q_cumulative = np.cumsum(q_eigenvalues) / q_eigenvalues.sum()

drift_structure = pd.DataFrame(
    {
        'eigenvalue': q_eigenvalues[:3],
        'cumulative share': q_cumulative[:3],
    },
    index=pd.Index(range(1, 4), name='principal component'),
)
drift_structure.round(4)
```

The first three principal components account for 96.3 percent of total
coefficient drift even though the VAR contains 21 coefficients.

### The evolution of volatility

We first ask how the *size* of the shocks changed.

The posterior mean of $R_t$ shows large and systematic movements in both
innovation standard deviations and correlations.

Equation {eq}`csdv_covariance` can be averaged over draws without constructing a
four-dimensional covariance array.

```{code-cell} ipython3
def mean_innovation_covariance(h_draws, beta_draws):
    """Compute E(R_t | T) with working memory proportional to T times D."""
    n_draws = h_draws.shape[2]
    matrices = np.broadcast_to(np.eye(3), (n_draws, 3, 3)).copy()
    matrices[:, 1, 0] = beta_draws[0]
    matrices[:, 2, 0] = beta_draws[1]
    matrices[:, 2, 1] = beta_draws[2]
    inverses = np.linalg.solve(
        matrices,
        np.broadcast_to(np.eye(3), matrices.shape),
    )
    h = h_draws[1:]
    mean = np.empty((h.shape[0], 3, 3))
    for row in range(3):
        for column in range(row + 1):
            value = np.zeros(h.shape[0])
            for shock in range(3):
                weights = inverses[:, row, shock] * inverses[:, column, shock]
                value += h[:, shock, :] @ weights
            mean[:, row, column] = value / n_draws
            mean[:, column, row] = mean[:, row, column]
    return mean


r_mean = mean_innovation_covariance(posterior['HD'], posterior['CD'])
```

The next plot shows the innovation standard deviations and correlations.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Innovation volatility and correlation
    name: fig-csdv-volatility-correlation
---
variances = ((0, 'Nominal interest'), (2, 'Inflation'), (1, 'Unemployment'))
correlations = (
    (0, 1, 'Interest--unemployment'),
    (0, 2, 'Interest--inflation'),
    (2, 1, 'Inflation--unemployment'),
)
fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharex=True)
for row, (index, label) in enumerate(variances):
    axes[row, 0].plot(
        data['dates'], 10000 * np.sqrt(r_mean[:, index, index]), lw=2
    )
    axes[row, 0].text(0.03, 0.88, label, transform=axes[row, 0].transAxes)
for row, (left, right, label) in enumerate(correlations):
    scale = np.sqrt(r_mean[:, left, left] * r_mean[:, right, right])
    axes[row, 1].plot(data['dates'], r_mean[:, left, right] / scale, lw=2)
    axes[row, 1].text(0.03, 0.88, label, transform=axes[row, 1].transAxes)
axes[1, 0].set_ylabel(
    r'innovation standard deviation $\times 10^4$'
)
axes[1, 1].set_ylabel('correlation')
axes[-1, 0].set_xlabel('year')
axes[-1, 1].set_xlabel('year')
plt.tight_layout()
fig.show()
```

The standard deviation of the interest-rate innovation spikes dramatically
around 1979--1982, the years of the Volcker experiment with nonborrowed-reserves
targeting.

The following calculation reports both the raw endpoint change in the
unemployment innovation standard deviation and a less noisy comparison of the
first and last 16-quarter averages.

```{code-cell} ipython3
unemployment_sd = np.sqrt(r_mean[:, 1, 1])
unemployment_sd_endpoint_decline = 1 - unemployment_sd[-1] / unemployment_sd[0]
unemployment_sd_smoothed_decline = (
    1 - unemployment_sd[-16:].mean() / unemployment_sd[:16].mean()
)

pd.Series(
    {
        'endpoint decline': unemployment_sd_endpoint_decline,
        'first/last 16-quarter mean decline': (
            unemployment_sd_smoothed_decline
        ),
    },
    name='fractional decline',
).to_frame().round(2)
```

The interest-rate and inflation innovation variances spike between 1979 and 1981
before falling sharply.

Interest--unemployment innovations are negatively correlated, and
interest--inflation innovations are positively correlated.

The inflation--unemployment correlation is mostly negative and most pronounced
around the Volcker disinflation, but is near zero and briefly slightly positive
early in the estimated path.

The signs of these correlations caution against interpreting the
nominal-interest innovation as a structural monetary-policy shock.

The log determinant of the posterior mean covariance matrix summarizes the
generalized one-step innovation variance {cite}`Whittle1953`.

The following transformation summarizes generalized innovation variance.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Generalized innovation variance
    name: fig-csdv-total-variance
---
sign, logdet_r = np.linalg.slogdet(r_mean)
assert np.all(sign > 0)
fig, ax = plt.subplots()
ax.plot(data['dates'], logdet_r, lw=2)
ax.set_xlabel('year')
ax.set_ylabel(r'$\log |E(R_t\mid T)|$')
fig.show()
```

Generalized innovation variance rises in two steps before 1981 and then falls
during the Volcker era.

The pattern documents substantial movement in volatility and gives the bad-luck
account an important role.

The decline after 1981 looks more like a return to an earlier stability than an
unprecedented new regime, which connects it to the Great Moderation documented
by {cite:t}`KimNelson1999` and {cite:t}`McConnellPerezQuiros2000`.

### Core inflation and the natural rate

To study the systematic dynamics, write the VAR at date $t$ in companion form as

$$
z_t = \mu_{t\mid T} + A_{t\mid T}z_{t-1} + e_t.
$$

Local mean inflation and unemployment are defined by freezing the posterior mean
coefficients at date $t$,

```{math}
:label: csdv_local_means
\bar\pi_t = s_\pi(I-A_{t\mid T})^{-1}\mu_{t\mid T},
\qquad
\bar u_t = s_u(I-A_{t\mid T})^{-1}\mu_{t\mid T}.
```

These are local linear approximations rather than unconditional means of the
globally drifting process.

Core inflation is therefore the long-horizon inflation forecast implied by
freezing the date-$t$ coefficients, while the natural rate is the corresponding
long-horizon unemployment forecast.

The following implementation annualizes core inflation and reverses the
archive's unemployment transformation.

```{code-cell} ipython3
def local_means(theta_path):
    """Compute local core inflation and the natural unemployment rate."""
    theta_path = np.asarray(theta_path, dtype=float)
    if theta_path.ndim == 1:
        theta_path = theta_path[:, None]
    core = np.empty(theta_path.shape[1])
    natural = np.empty(theta_path.shape[1])
    for t in range(theta_path.shape[1]):
        intercept, companion = companion_matrix(theta_path[:, t])
        mean = np.linalg.solve(np.eye(6) - companion, intercept)
        core[t] = 4 * mean[2]
        natural[t] = expit(100 * mean[1])
    return core, natural


core_inflation, natural_rate = local_means(theta_mean)
```

We plot the fourth-quarter observation from each year.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Core inflation and natural rate
    name: fig-csdv-local-means
---
def annual_indices(dates, start=1960):
    """Return the final observation in each year from a start date."""
    year_values = np.floor(dates + 1e-8).astype(int)
    return np.array(
        [
            np.flatnonzero(year_values == year)[-1]
            for year in np.unique(year_values)
            if year >= start
        ]
    )


years = np.floor(data['dates'] + 1e-8).astype(int)
annual = annual_indices(data['dates'])

fig, ax = plt.subplots()
ax.plot(data['dates'][annual], 100 * core_inflation[annual], 'o-', lw=2,
        markersize=3, label='core inflation')
ax.plot(data['dates'][annual], 100 * natural_rate[annual], '+-', lw=2,
        markersize=5, label='natural rate')
ax.set_xlabel('year')
ax.set_ylabel('percent')
ax.legend()
fig.show()
```

```{code-cell} ipython3
core_summary = pd.Series(
    {
        'early-1960s mean core inflation (%)': (
            100 * core_inflation[(years >= 1960) & (years <= 1964)].mean()
        ),
        'peak core inflation (%)': 100 * core_inflation[annual].max(),
        '1985--2000 mean core inflation (%)': (
            100 * core_inflation[(years >= 1985) & (years <= 2000)].mean()
        ),
    },
    name='estimate',
)
core_summary.to_frame().round(2)
```

Core inflation rises into the 1970s and then falls after the Volcker
disinflation.

The natural rate follows a similar lower-frequency pattern, although both
quantities become sensitive when a local companion root approaches one.

The following calculation summarizes their comovement over the full posterior
sample.

```{code-cell} ipython3
core_natural_correlation = np.corrcoef(core_inflation, natural_rate)[0, 1]
```

The quarterly correlation between posterior-mean core inflation and the natural
rate is 0.838.

This strong positive association says that the model's low-frequency movements
in inflation and unemployment tend to occur together.

Because $(I-A_t)^{-1}$ amplifies small coefficient changes when the largest root
is close to one, long-run means are intrinsically more sensitive than
short-horizon forecasts.

### Inflation persistence

We now ask whether the *systematic* dynamics drifted on top of the moving
volatilities.

The main summary is inflation persistence, measured by the normalized spectrum
of inflation at frequency zero.

The date-local spectral density of inflation is

```{math}
:label: csdv_spectrum
f_{\pi\pi}(\omega,t)
=
\frac{1}{2\pi}
s_\pi
(I-A_{t\mid T}e^{-i\omega})^{-1}
\mathcal R_t
(I-A_{t\mid T}'e^{i\omega})^{-1}
s_\pi',
```

where $\mathcal R_t$ embeds $E(R_t\mid T)$ in the companion system.

Low-frequency power depends on both the autoregressive coefficients and the
innovation covariance.

The next function implements equation (32) at any frequency measured in cycles
per quarter and also returns the date-local inflation variance.

```{code-cell} ipython3
def inflation_spectrum(theta, covariance, frequencies):
    """Compute inflation power and its variance-normalized counterpart."""
    _, companion = companion_matrix(theta)
    innovation = np.zeros((6, 6))
    innovation[:3, :3] = covariance
    selector = np.zeros(6)
    selector[2] = 1
    stationary = linalg.solve_discrete_lyapunov(companion, innovation)
    variance = float(selector @ stationary @ selector)
    power = np.empty(len(frequencies))
    for index, frequency in enumerate(frequencies):
        phase = np.exp(-2j * np.pi * frequency)
        transfer = np.linalg.solve(np.eye(6) - companion * phase, np.eye(6))
        power[index] = np.real(
            selector @ transfer @ innovation @ transfer.conj().T @ selector
        ) / (2 * np.pi)
    return power, power / variance

```

The normalized spectrum divides by the date-local inflation variance,

```{math}
:label: csdv_normalized_spectrum
g_{\pi\pi}(\omega,t)
=
\frac{f_{\pi\pi}(\omega,t)}
{\int_{-\pi}^{\pi}f_{\pi\pi}(\omega,t)d\omega},
```

so $g_{\pi\pi}(0,t)$ measures persistence after adjusting for changes in
innovation variance.

The first figure isolates frequency zero as a one-dimensional persistence
summary.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Inflation persistence
    name: fig-csdv-inflation-persistence
---
zero_frequency = np.array([0.0])
inflation_persistence = np.array([
    inflation_spectrum(theta_mean[:, t], r_mean[t], zero_frequency)[1][0]
    for t in range(len(data['dates']))
])

fig, ax = plt.subplots()
ax.plot(
    data['dates'][annual],
    inflation_persistence[annual],
    'o-',
    lw=2,
    markersize=3,
)
ax.set_xlabel('year')
ax.set_ylabel(r'$g_{\pi\pi}(0,t)$')
fig.show()
```

```{code-cell} ipython3
persistence_summary = pd.Series(
    {
        '1960--64 mean': inflation_persistence[
            (years >= 1960) & (years <= 1964)
        ].mean(),
        '1970--79 mean': inflation_persistence[
            (years >= 1970) & (years <= 1979)
        ].mean(),
        '1985--2000 mean': inflation_persistence[
            (years >= 1985) & (years <= 2000)
        ].mean(),
        'peak': inflation_persistence[annual].max(),
        'peak year': years[annual][np.argmax(inflation_persistence[annual])],
    },
    name='estimate',
)
persistence_summary.to_frame().round(3)
```

The normalized spectrum rises gradually during the late 1960s, remains high in
the 1970s, and falls sharply after 1980.

For comparison, an $AR(1)$ with coefficient $\rho$ has normalized zero-frequency
power $(1+\rho)/[2\pi(1-\rho)]$.

Values between 2 and 10 correspond to $\rho$ between approximately $0.85$ and
$0.97$.

The zero-frequency path omits the rest of the frequency distribution.

The following heatmaps show how raw and variance-normalized inflation power move
over both time and frequency.

```{code-cell} ipython3
def inflation_spectrum_surface(theta_path, covariance_path, frequencies):
    """Evaluate the date-local inflation spectrum on a frequency grid."""
    raw = np.empty((len(frequencies), theta_path.shape[1]))
    normalized = np.empty_like(raw)
    for date in range(theta_path.shape[1]):
        raw[:, date], normalized[:, date] = inflation_spectrum(
            theta_path[:, date],
            covariance_path[date],
            frequencies,
        )
    return raw, normalized


spectrum_frequencies = np.linspace(0, 0.5, 41)
raw_spectrum, normalized_spectrum = inflation_spectrum_surface(
    theta_mean,
    r_mean,
    spectrum_frequencies,
)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Inflation spectra over time
    name: fig-csdv-inflation-spectra
---
spectrum_start = data['dates'] >= 1960
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
surfaces = (
    (raw_spectrum, 'raw spectrum', 'log10 power'),
    (normalized_spectrum, 'normalized spectrum', 'log10 normalized power'),
)
for ax, (surface, title, color_label) in zip(axes, surfaces):
    image = ax.pcolormesh(
        data['dates'][spectrum_start],
        spectrum_frequencies,
        np.log10(surface[:, spectrum_start]),
        shading='auto',
    )
    ax.set_xlabel('year')
    ax.set_ylabel(f'{title}\ncycles per quarter')
    fig.colorbar(image, ax=ax, label=color_label)
plt.tight_layout()
fig.show()
```

Raw low-frequency power rises most sharply when persistence and innovation
variance are both elevated.

Normalization removes the changing local variance and leaves a broad
low-frequency ridge through the 1970s that recedes after 1980.

Point estimates do not reveal how strongly the data locate these paths.

We therefore compute the same annual features from every retained draw.

```{code-cell} ipython3
def posterior_feature_draws(result, indices):
    """Compute selected local means and persistence for retained draws."""
    n_draws = result['SD'].shape[2]
    shape = (len(indices), n_draws)
    core = np.empty(shape)
    natural = np.empty(shape)
    persistence = np.empty(shape)
    for draw in range(n_draws):
        core[:, draw], natural[:, draw] = local_means(
            result['SD'][:, indices, draw]
        )
        for row, date in enumerate(indices):
            covariance = innovation_covariance(
                result['HD'][date + 1, :, draw],
                result['CD'][:, draw],
            )
            persistence[row, draw] = inflation_spectrum(
                result['SD'][:, date, draw],
                covariance,
                zero_frequency,
            )[1][0]
    return {
        'core': 100 * core,
        'natural': 100 * natural,
        'persistence': persistence,
    }


historical_feature_draws = posterior_feature_draws(posterior, annual)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Uncertainty in inflation dynamics
    name: fig-csdv-feature-uncertainty
---
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
feature_specs = (
    ('core', 'core inflation (%)'),
    ('natural', 'natural rate (%)'),
    ('persistence', r'$g_{\pi\pi}(0,t)$'),
)
for ax, (key, ylabel) in zip(axes, feature_specs):
    lower, median, upper = np.quantile(
        historical_feature_draws[key],
        (0.05, 0.5, 0.95),
        axis=1,
    )
    line, = ax.plot(data['dates'][annual], median, lw=2)
    ax.fill_between(
        data['dates'][annual],
        lower,
        upper,
        color=line.get_color(),
        alpha=0.2,
    )
    ax.set_ylabel(ylabel)
axes[-1].set_xlabel('year')
plt.tight_layout()
fig.show()
```

The shaded regions are pointwise 90 percent posterior intervals rather than a
simultaneous band for each whole path.

The solid line is the posterior median of the nonlinear feature at each date.

Uncertainty expands when local roots approach one, especially for long-horizon
means and zero-frequency persistence.

Core inflation and inflation persistence move closely together over the sample.

The following plot shows the two series together.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Core inflation and persistence
    name: fig-csdv-core-persistence
---
core_persistence_correlation = np.corrcoef(
    core_inflation, inflation_persistence
)[0, 1]

fig, ax = plt.subplots()
ax.plot(data['dates'][annual], 100 * core_inflation[annual], 'o-',
        lw=2, markersize=3, label='core inflation (%)')
ax.plot(data['dates'][annual], inflation_persistence[annual], 'x-',
        lw=2, markersize=4, label='normalized spectrum at zero')
ax.set_xlabel('year')
ax.legend()
fig.show()
```

The quarterly correlation between posterior-mean core inflation and persistence
is 0.909.

The rise through the late 1960s and 1970s followed by the sharp fall after 1980
is a central finding because it shows that the systematic dynamics moved along
with the volatilities.

The fall in persistence during the Volcker disinflation conflicts with
escape-route models in which persistence grows along a transition from high to
low inflation {cite}`Sargent1999,ChoWilliamsSargent2002`.

That tension helped motivate later learning models in which policymakers became
reluctant to disinflate during the 1970s and then changed course.

### Monetary policy activism

Cogley and Sargent summarize systematic policy with a forward-looking Taylor
rule,

```{math}
:label: csdv_policy_rule
i_t = \beta_0
+ \beta_1 E_t\bar\pi_{t,t+h_\pi}
+ \beta_2 E_t\bar u_{t,t+h_u}
+ \beta_3 i_{t-1}
+ \nu_t.
```

They define the activism coefficient as $\mathcal A_t=\beta_1/(1-\beta_3)$ and
call policy active when $\mathcal A_t\geq 1$.

At each date, population two-stage least squares projections implied by the
local VAR produce the policy-rule coefficients.

The benchmark horizons are $h_\pi=4$ quarters and $h_u=2$ quarters, reflecting
conventional views about monetary-policy lags.

The following function implements equation (34) and footnote 18 from the
stationary second moments of each local VAR.

```{code-cell} ipython3
def policy_activism(
    theta,
    covariance,
    h_pi=4,
    h_u=2,
    return_denominator=False,
):
    """Compute the date-local population-2SLS activism coefficient."""
    _, companion = companion_matrix(theta)
    innovation = np.zeros((6, 6))
    innovation[:3, :3] = covariance
    stationary = linalg.solve_discrete_lyapunov(companion, innovation)
    selectors = np.eye(6)
    power = np.eye(6)
    inflation_loading = np.zeros(6)
    unemployment_loading = np.zeros(6)
    for horizon in range(1, max(h_pi, h_u) + 1):
        power = power @ companion
        if horizon <= h_pi:
            inflation_loading += selectors[2] @ power
        if horizon <= h_u:
            unemployment_loading += selectors[1] @ power
    inflation_loading /= h_pi
    unemployment_loading /= h_u
    loadings = np.vstack(
        (inflation_loading, unemployment_loading, selectors[0])
    )
    regressor_covariance = loadings @ stationary @ loadings.T
    dependent_covariance = loadings @ stationary @ companion.T @ selectors[0]
    coefficients = np.linalg.solve(regressor_covariance, dependent_covariance)
    denominator = 1 - coefficients[2]
    activism = coefficients[0] / denominator
    if return_denominator:
        return activism, denominator
    return activism


def break_ratio_crossings(values, denominator):
    """Break a ratio path on both sides of denominator sign changes."""
    broken = np.asarray(values, dtype=float).copy()
    denominator = np.asarray(denominator, dtype=float)
    crossings = np.flatnonzero(
        denominator[1:] * denominator[:-1] <= 0
    ) + 1
    broken[crossings] = np.nan
    broken[np.maximum(crossings - 1, 0)] = np.nan
    return broken


activism_pairs = np.array(
    [
        policy_activism(
            theta_mean[:, t],
            r_mean[t],
            return_denominator=True,
        )
        for t in range(len(data['dates']))
    ]
)
activism = activism_pairs[:, 0]
activism_denominator = activism_pairs[:, 1]
activism_path = break_ratio_crossings(
    activism,
    activism_denominator,
)
```

The plug-in path evaluates activism at posterior-mean coefficients and
covariances.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Plug-in monetary policy activism
    name: fig-csdv-policy-activism
---
fig, ax = plt.subplots()
ax.plot(data['dates'], activism_path, lw=2)
ax.axhline(1, color='0.45', lw=1)
ax.set_xlabel('year')
ax.set_ylabel('activism coefficient')
fig.show()
```

The plug-in activism coefficient is below one in much of the 1970s and above one
after the early 1980s.

Policy activism is negatively related to both core inflation and inflation
persistence.

The following scatter plots use fourth-quarter observations.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Activism and inflation dynamics
    name: fig-csdv-activism-correlations
---
activism_core_correlation = np.corrcoef(
    activism[annual], core_inflation[annual]
)[0, 1]
activism_persistence_correlation = np.corrcoef(
    activism[annual], inflation_persistence[annual]
)[0, 1]

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
pairs = (
    (100 * core_inflation, 'core inflation (%)'),
    (inflation_persistence, 'normalized spectrum at zero'),
)
for ax, (feature, label) in zip(axes, pairs):
    ax.scatter(activism[annual], feature[annual], s=18)
    ax.axvline(1, color='0.45', lw=1)
    ax.set_xlabel('policy activism')
    ax.set_ylabel(label)
plt.tight_layout()
fig.show()
```

The fourth-quarter correlations of activism with core inflation and persistence
are -0.78 and -0.74, respectively.

Thus, periods with a stronger estimated policy response tend to have lower core
inflation and less low-frequency inflation persistence.

The activism transformation is weakly identified at some dates, so a path based
on posterior-mean inputs understates uncertainty.

We therefore calculate activism from every retained draw in 1975, 1985, and
1995.

```{code-cell} ipython3
selected_years = (1975, 1985, 1995)
selected_dates = [np.flatnonzero(years == year)[-1] for year in selected_years]

activity_draws = {
    year: np.empty(posterior['SD'].shape[2]) for year in selected_years
}
for draw in range(posterior['SD'].shape[2]):
    for year, date in zip(selected_years, selected_dates):
        covariance = innovation_covariance(
            posterior['HD'][date + 1, :, draw],
            posterior['CD'][:, draw],
        )
        activity_draws[year][draw] = policy_activism(
            posterior['SD'][:, date, draw], covariance
        )
```

These draws give posterior probabilities of active policy at each date and of a
rise in activism after 1975.

```{code-cell} ipython3
activity_events = (
    activity_draws[1975] > 1,
    activity_draws[1985] > 1,
    activity_draws[1995] > 1,
    activity_draws[1985] > activity_draws[1975],
    activity_draws[1995] > activity_draws[1975],
)
activity_probability_values = np.array(
    [event.mean() for event in activity_events]
)
activity_probability_ess = np.array(
    [scalar_mcmc_ess(event) for event in activity_events]
)

activity_probability_index = (
    'P(A_1975 > 1)',
    'P(A_1985 > 1)',
    'P(A_1995 > 1)',
    'P(A_1985 > A_1975)',
    'P(A_1995 > A_1975)',
)
activity_probabilities = pd.DataFrame(
    {
        'estimate': activity_probability_values,
        'indicator ESS': activity_probability_ess,
    },
    index=activity_probability_index,
)
activity_probabilities['MCSE'] = np.sqrt(
    activity_probabilities['estimate']
    * (1 - activity_probabilities['estimate'])
    / activity_probability_ess
)

activity_probabilities.round(3)
```

The probabilities should be read with their indicator ESS and MCSE.

They support a move from passive policy in the mid-1970s toward active policy
after 1980, but overlapping posterior tails keep that conclusion probabilistic
rather than dispositive.

The central draw distributions expose the overlap and skewness behind those
probabilities.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Central posterior policy activism
    name: fig-csdv-activism-distributions
---
pooled_activity = np.concatenate(tuple(activity_draws.values()))
activity_limits = np.quantile(pooled_activity, (0.05, 0.95))
activity_bins = np.linspace(*activity_limits, 31)

fig, ax = plt.subplots()
for year in selected_years:
    central_draws = activity_draws[year]
    central_draws = central_draws[
        (central_draws >= activity_limits[0])
        & (central_draws <= activity_limits[1])
    ]
    ax.hist(
        central_draws,
        bins=activity_bins,
        histtype='step',
        lw=2,
        label=str(year),
    )
ax.axvline(1, color='0.45', lw=1)
ax.set_xlabel('activism coefficient')
ax.set_ylabel('retained draws')
ax.legend()
fig.show()
```

The figure plots unmodified draws inside the pooled 5th and 95th percentiles,
while all probability calculations above use every draw.

(csdv-updated-evidence)=
## Another quarter-century of evidence

The frozen sample ends in 2000Q4, so it misses the financial crisis, the
zero-interest-rate period, the pandemic, and the 2021--2022 inflation surge.

The question is whether these episodes alter the earlier evidence about drift,
volatility, inflation persistence, and systematic policy.

### The new observations

To examine those observations, we append current data after 2000Q4 while leaving
the frozen history unchanged.

This splice prevents revisions to pre-2001 CPI and unemployment data from being
mistaken for information in the additional quarter-century.

We download seasonally adjusted [CPI][fred-cpi], seasonally adjusted
[unemployment][fred-unemployment], and the [three-month Treasury
yield][fred-interest] from FRED.

[fred-cpi]: https://fred.stlouisfed.org/series/CPIAUCSL
[fred-unemployment]: https://fred.stlouisfed.org/series/UNRATE
[fred-interest]: https://fred.stlouisfed.org/series/GS3M

The transformations and within-quarter timing remain unchanged: CPI comes from
the third month, unemployment is a three-month average, and the interest rate
comes from the first month.

```{code-cell} ipython3
---
tags: [hide-input]
---
fred_url = (
    'https://fred.stlouisfed.org/graph/fredgraph.csv?'
    'id=CPIAUCSL%2CUNRATE%2CGS3M'
)
fred_monthly = pd.read_csv(
    fred_url,
    parse_dates=['observation_date'],
).set_index('observation_date')

```

[BLS reports](https://www.bls.gov/web/empsit/cpsee_e12.pdf) that reliable
2025Q4 unemployment estimates could not be produced because the October 2025
observation was not collected during the federal shutdown.

Consequently, 2025Q4 has no authoritative three-month unemployment average, and
the updated estimation sample ends in 2025Q3.

This endpoint keeps the extension fully observed and quarterly contiguous.

```{code-cell} ipython3
---
tags: [hide-input]
---
def fred_quarterly_table(unemployment_monthly):
    """Construct transformed quarterly observations after 2000Q4."""
    interest = fred_monthly.loc[
        fred_monthly.index.month.isin((1, 4, 7, 10)), 'GS3M'
    ].copy()
    interest.index = interest.index.to_period('Q').start_time

    cpi = fred_monthly.loc[
        fred_monthly.index.month.isin((3, 6, 9, 12)), 'CPIAUCSL'
    ].copy()
    cpi.index = cpi.index.to_period('Q').start_time

    unemployment = unemployment_monthly.resample('QS').mean()
    quarterly = pd.concat(
        {
            'interest': interest,
            'unemployment': unemployment,
            'cpi': cpi,
        },
        axis=1,
    )
    quarterly['y3'] = np.log1p(quarterly['interest'] / 400)
    quarterly['ur'] = quarterly['unemployment'] / 100
    quarterly['dp'] = np.log(quarterly['cpi']).diff()
    quarterly['date'] = (
        quarterly.index.year + (quarterly.index.quarter - 1) / 4
    )
    columns = ['date', 'y3', 'ur', 'dp']
    return quarterly.loc['2001-01-01':, columns].dropna()


unemployment_monthly = fred_monthly['UNRATE']
internal_unemployment = unemployment_monthly.loc[
    unemployment_monthly.first_valid_index():
    unemployment_monthly.last_valid_index()
]
missing_unemployment = internal_unemployment.index[
    internal_unemployment.isna()
]
assert missing_unemployment.equals(
    pd.DatetimeIndex([pd.Timestamp('2025-10-01')])
)

unemployment_counts = unemployment_monthly.resample('QS').count()
quarterly_unfilled = fred_quarterly_table(unemployment_monthly)
incomplete_quarters = unemployment_counts.loc[
    quarterly_unfilled.index[0]:quarterly_unfilled.index[-1]
]
incomplete_quarters = incomplete_quarters[incomplete_quarters < 3]
first_incomplete_quarter = incomplete_quarters.index[0]
complete_extension = quarterly_unfilled.loc[
    quarterly_unfilled.index < first_incomplete_quarter
]

frozen_table = pd.read_csv(data_path)
extended_observations = pd.concat(
    (frozen_table, complete_extension.reset_index(drop=True)),
    ignore_index=True,
)


def quarter_label(timestamp):
    """Format a timestamp as year and quarter."""
    return str(timestamp.to_period('Q'))


update_summary = pd.Series(
    {
        'updated sample end': quarter_label(
            complete_extension.index[-1]
        ),
        'retrieval date': pd.Timestamp.now(
            tz='America/New_York'
        ).date().isoformat(),
    },
    name='value',
)
update_summary.to_frame()
```

No missing value is filled or otherwise treated as observed.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Extended data through 2025Q3
    name: fig-csdv-updated-data
---
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
extended_plot = extended_observations[
    extended_observations['date'] >= 1959
]
updated_sample_end = complete_extension['date'].iloc[-1]
axes[0].plot(
    extended_plot['date'],
    400 * extended_plot['dp'],
    lw=2,
)
axes[0].set_ylabel('inflation (annual %)')
axes[1].plot(
    extended_plot['date'],
    100 * extended_plot['ur'],
    lw=2,
)
axes[1].set_ylabel('unemployment (%)')
axes[2].plot(
    extended_plot['date'],
    400 * np.expm1(extended_plot['y3']),
    lw=2,
)
axes[2].set_ylabel('interest (annual %)')
axes[2].set_xlabel('year')
for index, ax in enumerate(axes):
    frozen_label = 'frozen sample end' if index == 0 else None
    sample_end_label = 'updated sample end' if index == 0 else None
    ax.axvline(
        2000.75,
        color='0.65',
        ls='--',
        lw=1,
        label=frozen_label,
    )
    ax.axvline(
        updated_sample_end,
        color='0.45',
        ls=':',
        lw=1,
        label=sample_end_label,
    )
axes[0].legend()
plt.tight_layout()
fig.show()
```

```{code-cell} ipython3
endpoint_observation = complete_extension.iloc[-1]
pd.Series(
    {
        'annualized quarterly inflation (%)': (
            400 * endpoint_observation['dp']
        ),
        'unemployment (%)': 100 * endpoint_observation['ur'],
        'three-month interest rate (%)': (
            400 * np.expm1(endpoint_observation['y3'])
        ),
    },
    name=quarter_label(complete_extension.index[-1]),
).to_frame().round(3)
```

The pandemic produces an exceptionally sharp unemployment movement, while the
inflation surge is large but much shorter than the 1970s episode.

The short interest rate also spends years near zero, which makes it a less
complete summary of the stance of policy after 2008.

We fit the same stable model through the authoritative 2025Q3 endpoint.

```{code-cell} ipython3
---
tags: [hide-input, hide-output]
---
def append_extension(extension):
    """Append a transformed FRED extension to the frozen history."""
    extension = extension.reset_index(drop=True)
    table = pd.concat((frozen_table, extension), ignore_index=True)
    assert table['date'].is_unique
    assert np.allclose(np.diff(table['date']), 0.25)
    return table


def fit_updated_model(table):
    """Calibrate and fit the stable drifting VAR to one table."""
    model_data = prepare_data(table)
    model_prior = calibrate_prior(model_data)
    result = run_sampler(
        model_data['y'],
        model_data['x'],
        model_prior,
        n_sweeps=5_000,
        burn=2_500,
        thin=5,
        seed=42,
        warmup=500,
        stable=True,
        progress_every=500,
    )
    validate_posterior(result, len(model_data['dates']))
    trace = np.trace(result['QD'], axis1=0, axis2=1)
    return {
        'data': model_data,
        'prior': model_prior,
        'posterior': result,
        'trace': trace,
        'trace_ess': scalar_mcmc_ess(trace),
    }


updated_fit = fit_updated_model(append_extension(complete_extension))
```

### Did coefficient drift continue?

The updated posterior still puts substantial mass on economically meaningful
coefficient drift.

```{code-cell} ipython3
def updated_drift_diagnostics(fit):
    """Summarize coefficient drift and its Monte Carlo precision."""
    trace = fit['trace']
    q_mean = fit['posterior']['QD'].mean(axis=2)
    eigenvalues = np.linalg.eigvalsh(q_mean)[::-1]
    return pd.Series(
        {
            r'$E[\operatorname{tr}(Q)]$': trace.mean(),
            'trace ESS': fit['trace_ess'],
            'trace MCSE': trace.std(ddof=1) / np.sqrt(fit['trace_ess']),
            'first-three drift share': (
                eigenvalues[:3].sum() / eigenvalues.sum()
            ),
        }
    )


updated_label = quarter_label(complete_extension.index[-1])
updated_diagnostics = updated_drift_diagnostics(updated_fit).to_frame(
    name=updated_label
).T
updated_diagnostics.round(3)
```

The drift-rate distributions and coefficient paths provide the same views used
for the frozen sample.

The dashed vertical line in updated time-series figures marks the frozen
sample's 2000Q4 endpoint.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Updated coefficient-drift rate
    name: fig-csdv-updated-drift-rate
---
fig, ax = plt.subplots()
ax.hist(updated_fit['trace'], bins=30, histtype='step', lw=2)
ax.axvline(
    np.trace(updated_fit['prior']['q_center']),
    color='C1',
    lw=2,
    label=r'prior $\mathrm{tr}(\bar Q)$',
)
ax.set_xlabel(r'$\mathrm{tr}(Q)$')
ax.set_ylabel('frequency')
ax.legend()
fig.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Updated coefficient paths
    name: fig-csdv-updated-coefficient-paths
---
fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=True)
updated_dates = updated_fit['data']['dates']
updated_coefficients = updated_fit['posterior']['SD'].mean(axis=2)
coefficient_lines = plot_coefficient_blocks(
    axes,
    updated_dates,
    updated_coefficients,
)
for ax in axes:
    ax.axvline(2000.75, color='0.65', ls='--', lw=1)
fig.legend(
    coefficient_lines,
    coefficient_labels,
    loc='lower center',
    ncol=4,
)
plt.tight_layout(rect=(0, 0.17, 1, 1))
fig.show()
```

The posterior mean trace is about $0.040$, but an ESS near 20 limits fine
inferences about its magnitude.

The first three eigen-directions account for about 96 percent of the posterior
mean drift variance, so the estimated movement remains low dimensional.

The stability restriction now applies to a longer path, which makes the
posterior drift rate a property of the full 1959--2025 sample.

Several coefficient paths continue moving after 2000, especially in the
inflation equation.

### Volatility after the Great Moderation

The retained posterior draws let us separate the sizes and correlations of new
innovations from changes in the VAR dynamics.

```{code-cell} ipython3
---
tags: [hide-input]
---
def model_features(fit):
    """Compute plug-in features from one fitted posterior."""
    result = fit['posterior']
    theta = result['SD'].mean(axis=2)
    stable_mean_path = np.array(
        [is_stable(theta[:, t]) for t in range(theta.shape[1])]
    )
    if not np.all(stable_mean_path):
        raise ValueError('posterior-mean coefficient path is unstable')
    covariance = mean_innovation_covariance(result['HD'], result['CD'])
    core, natural = local_means(theta)
    persistence = np.array([
        inflation_spectrum(
            theta[:, t], covariance[t], zero_frequency
        )[1][0]
        for t in range(len(fit['data']['dates']))
    ])
    sign, logdet = np.linalg.slogdet(covariance)
    assert np.all(sign > 0)
    activism_pairs = np.array(
        [
            policy_activism(
                theta[:, t],
                covariance[t],
                return_denominator=True,
            )
            for t in range(len(fit['data']['dates']))
        ]
    )
    return {
        'theta': theta,
        'covariance': covariance,
        'core': core,
        'natural': natural,
        'persistence': persistence,
        'logdet': logdet,
        'activism': activism_pairs[:, 0],
        'activism_denominator': activism_pairs[:, 1],
    }


updated_features = model_features(updated_fit)
```

The innovation covariance paths reproduce the earlier decomposition through
2025Q3.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Updated innovation volatility and correlation
    name: fig-csdv-updated-volatility-correlation
---
fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharex=True)
dates = updated_fit['data']['dates']
covariance = updated_features['covariance']
for row, (index, _) in enumerate(variances):
    axes[row, 0].plot(
        dates,
        10000 * np.sqrt(covariance[:, index, index]),
        lw=2,
    )
for row, (left, right, _) in enumerate(correlations):
    scale = np.sqrt(
        covariance[:, left, left] * covariance[:, right, right]
    )
    axes[row, 1].plot(
        dates,
        covariance[:, left, right] / scale,
        lw=2,
    )
for row, (_, label) in enumerate(variances):
    axes[row, 0].text(0.03, 0.88, label, transform=axes[row, 0].transAxes)
for row, (_, _, label) in enumerate(correlations):
    axes[row, 1].text(0.03, 0.88, label, transform=axes[row, 1].transAxes)
for ax in axes.flat:
    ax.axvline(2000.75, color='0.65', ls='--', lw=1)
axes[1, 0].set_ylabel(
    r'innovation standard deviation $\times 10^4$'
)
axes[1, 1].set_ylabel('correlation')
axes[-1, 0].set_xlabel('year')
axes[-1, 1].set_xlabel('year')
plt.tight_layout()
fig.show()
```

The interest-rate innovation remains largest around the Volcker transition,
while the financial crisis is the largest inflation-volatility episode.

The pandemic instead dominates unemployment volatility and aggregate
one-step uncertainty.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Updated generalized innovation variance
    name: fig-csdv-updated-total-variance
---
fig, ax = plt.subplots()
ax.plot(
    updated_fit['data']['dates'],
    updated_features['logdet'],
    lw=2,
)
ax.axvline(2000.75, color='0.65', ls='--', lw=1)
ax.set_xlabel('year')
ax.set_ylabel(r'$\log |E(R_t\mid T)|$')
fig.show()
```

```{code-cell} ipython3
def decimal_quarter_label(value):
    """Format a decimal quarterly date."""
    year = int(np.floor(value + 1e-8))
    quarter = int(round(4 * (value - year))) + 1
    return f'{year}Q{quarter}'


def volatility_summary(fit, features):
    """Summarize innovation-volatility peaks and endpoints."""
    dates = fit['data']['dates']
    standard_deviation = 10000 * np.sqrt(
        np.diagonal(features['covariance'], axis1=1, axis2=2)
    )
    names = ('interest', 'unemployment', 'inflation')
    return pd.DataFrame(
        {
            'peak quarter': [
                decimal_quarter_label(
                    dates[np.argmax(standard_deviation[:, index])]
                )
                for index in range(n_variables)
            ],
            r'endpoint standard deviation $\times 10^4$': (
                standard_deviation[-1]
            ),
        },
        index=names,
    )


updated_volatility = volatility_summary(updated_fit, updated_features)
updated_volatility
```

All three innovation standard deviations are well below their peaks at the
2025Q3 endpoint.

This pattern supports a transient-shock interpretation inside this model, but
it does not establish that the pandemic disturbances were structurally
exogenous or harmless.

### Core inflation and the natural rate after 2000

The same local-mean calculation distinguishes temporary inflation from a shift
in the model's long-horizon forecast.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Updated local means
    name: fig-csdv-updated-local-means
---
updated_dates = updated_fit['data']['dates']
updated_annual = annual_indices(updated_dates)
fig, ax = plt.subplots()
ax.plot(
    updated_dates[updated_annual],
    100 * updated_features['core'][updated_annual],
    'o-',
    lw=2,
    markersize=3,
    label='core inflation',
)
ax.plot(
    updated_dates[updated_annual],
    100 * updated_features['natural'][updated_annual],
    '+-',
    lw=2,
    markersize=5,
    label='natural rate',
)
ax.axvline(2000.75, color='0.65', ls='--', lw=1)
ax.set_xlabel('year')
ax.set_ylabel('percent')
ax.legend()
fig.show()
```

The final annual point represents 2025Q3 because the sample ends before the
fourth quarter.

```{code-cell} ipython3
def endpoint_features(features):
    """Return economically scaled endpoint features."""
    return pd.Series(
        {
            'core inflation (%)': 100 * features['core'][-1],
            'natural rate (%)': 100 * features['natural'][-1],
            'normalized persistence': features['persistence'][-1],
            'log generalized innovation variance': features['logdet'][-1],
        }
    )


updated_endpoints = endpoint_features(updated_features).to_frame(
    name=updated_label
).T
updated_endpoints.round(3)
```

The draw-wise annual paths show how uncertainty evolves inside the updated fit.

```{code-cell} ipython3
---
tags: [hide-input]
---
updated_feature_draws = posterior_feature_draws(
    updated_fit['posterior'],
    updated_annual,
)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Updated feature uncertainty
    name: fig-csdv-updated-feature-uncertainty
---
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for ax, (key, ylabel) in zip(axes, feature_specs):
    lower, median, upper = np.quantile(
        updated_feature_draws[key],
        (0.05, 0.5, 0.95),
        axis=1,
    )
    line, = ax.plot(updated_dates[updated_annual], median, lw=2)
    ax.fill_between(
        updated_dates[updated_annual],
        lower,
        upper,
        color=line.get_color(),
        alpha=0.2,
    )
    ax.axvline(2000.75, color='0.65', ls='--', lw=1)
    ax.set_ylabel(ylabel)
axes[-1].set_xlabel('year')
plt.tight_layout()
fig.show()
```

The solid lines are posterior medians and the shaded regions are pointwise 90
percent intervals rather than simultaneous whole-path bands.

The endpoint rows summarize uncertainty in the three nonlinear features.

```{code-cell} ipython3
def endpoint_feature_intervals(draws):
    """Summarize draw-wise uncertainty in endpoint model features."""
    labels = {
        'core': 'core inflation (%)',
        'natural': 'natural rate (%)',
        'persistence': 'normalized persistence',
    }
    rows = {}
    for key, label in labels.items():
        values = draws[key][-1]
        rows[label] = {
            'median': np.median(values),
            '5th percentile': np.quantile(values, 0.05),
            '95th percentile': np.quantile(values, 0.95),
        }
    return pd.DataFrame.from_dict(rows, orient='index')


updated_endpoint_intervals = endpoint_feature_intervals(
    updated_feature_draws
)
updated_endpoint_intervals.round(3)
```

The raw inflation path in {numref}`fig-csdv-updated-data` rises much more than
the estimated core rate because core is a long-horizon forecast.

The 2025Q3 estimates place core inflation near 2.7 percent and the natural rate
near 5.2 percent.

The natural-rate interval remains comparatively wide, consistent with the
amplification that occurs when a companion root approaches one.

### Did inflation become persistent again?

The normalized spectrum asks whether the recent inflation surge changed the
systematic dynamics rather than only the size of shocks.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Updated inflation persistence
    name: fig-csdv-updated-persistence
---
fig, ax = plt.subplots()
ax.plot(
    updated_dates[updated_annual],
    updated_features['persistence'][updated_annual],
    'o-',
    lw=2,
    markersize=3,
)
ax.axvline(2000.75, color='0.65', ls='--', lw=1)
ax.set_xlabel('year')
ax.set_ylabel(r'$g_{\pi\pi}(0,t)$')
fig.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Updated core and persistence
    name: fig-csdv-updated-core-persistence
---
fig, axes = plt.subplots(
    2,
    1,
    figsize=(9, 6),
    sharex=True,
)
axes[0].plot(
    updated_dates[updated_annual],
    100 * updated_features['core'][updated_annual],
    'o-',
    lw=2,
    markersize=3,
)
axes[1].plot(
    updated_dates[updated_annual],
    updated_features['persistence'][updated_annual],
    'x-',
    lw=2,
    markersize=4,
)
for ax in axes:
    ax.axvline(2000.75, color='0.65', ls='--', lw=1)
axes[0].set_ylabel('core inflation (%)')
axes[1].set_ylabel('normalized spectrum at zero')
axes[1].set_xlabel('year')
plt.tight_layout()
fig.show()
```

The post-2020 rise in core inflation is visible, but the posterior-mean
persistence path stays below its 1970s values.

The 2025Q3 persistence median is $0.38$, while its 90 percent interval from
$0.15$ to $4.39$ leaves meaningful upper-tail uncertainty.

This result says that the fitted local VAR views the recent surge as less
persistent, not that all alternative models must classify it as transitory.

The full spectrum shows where the difference comes from.

```{code-cell} ipython3
---
tags: [hide-input]
---
updated_raw_spectrum, updated_normalized_spectrum = (
    inflation_spectrum_surface(
        updated_features['theta'],
        updated_features['covariance'],
        spectrum_frequencies,
    )
)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Updated inflation spectra
    name: fig-csdv-updated-spectra
---
fig, axes = plt.subplots(
    1,
    2,
    figsize=(9, 4),
    sharey=True,
    constrained_layout=True,
)
updated_surfaces = (
    (updated_raw_spectrum, 'raw spectrum', 'log10 power'),
    (
        updated_normalized_spectrum,
        'normalized spectrum',
        'log10 normalized power',
    ),
)
for ax, (surface, title, color_label) in zip(axes, updated_surfaces):
    image = ax.pcolormesh(
        updated_dates,
        spectrum_frequencies,
        np.log10(surface),
        shading='auto',
    )
    ax.axvline(2000.75, color='0.85', ls='--', lw=1)
    ax.set_xlabel('year')
    ax.set_ylabel(f'{title}\ncycles per quarter')
    fig.colorbar(image, ax=ax, label=color_label)
fig.show()
```

The pandemic and recent inflation surge add raw power across frequencies, while
the normalized low-frequency ridge remains most prominent in the 1970s.

```{code-cell} ipython3
def episode_summary(fit, features):
    """Average selected features over economically distinct episodes."""
    years = np.floor(fit['data']['dates'] + 1e-8).astype(int)
    periods = {
        '1970--1979': (years >= 1970) & (years <= 1979),
        '1985--2000': (years >= 1985) & (years <= 2000),
        '2001--2019': (years >= 2001) & (years <= 2019),
        '2020--2022': (years >= 2020) & (years <= 2022),
        '2023--2025Q3': years >= 2023,
    }
    rows = {}
    for label, mask in periods.items():
        rows[label] = {
            'core inflation (%)': 100 * features['core'][mask].mean(),
            'normalized persistence': features['persistence'][mask].mean(),
            'log generalized innovation variance': (
                features['logdet'][mask].mean()
            ),
        }
    return pd.DataFrame.from_dict(rows, orient='index')


updated_episodes = episode_summary(updated_fit, updated_features)
updated_episodes.round(3)
```

The episode averages separate the high volatility of 2020--2022 from the lower
average persistence of the post-2000 decades.

### Can recent policy activism be measured?

The same projection-based activism path can be computed after 2000, but its
ratio becomes unstable whenever $1-\beta_3$ approaches zero.

In the following figures, the gray line marks the active-policy threshold
$\mathcal A=1$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Updated plug-in policy activism
    name: fig-csdv-updated-activism
---
fig, ax = plt.subplots()
updated_activism_path = break_ratio_crossings(
    updated_features['activism'],
    updated_features['activism_denominator'],
)
ax.plot(updated_dates, updated_activism_path, lw=2)
ax.axhline(1, color='0.45', lw=1)
ax.axvline(2000.75, color='0.65', ls='--', lw=1)
ax.set_yscale('symlog', linthresh=1)
ax.set_xlabel('year')
ax.set_ylabel('activism coefficient')
fig.show()
```

The symmetric-log scale preserves the sign and exposes ratio explosions rather
than hiding them behind a truncated axis.

The association with inflation features is therefore descriptive and not a
stable structural-policy estimate.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Post-2000 plug-in activism relationships
    name: fig-csdv-updated-activism-correlations
---
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
recent_annual = annual_indices(updated_dates, start=2001)
axes[0].scatter(
    updated_features['activism'][recent_annual],
    100 * updated_features['core'][recent_annual],
    s=18,
)
axes[1].scatter(
    updated_features['activism'][recent_annual],
    updated_features['persistence'][recent_annual],
    s=18,
)
for ax in axes:
    ax.set_xscale('symlog', linthresh=1)
    ax.axvline(1, color='0.45', lw=1)
    ax.set_xlabel('policy activism')
axes[0].set_ylabel('core inflation (%)')
axes[1].set_ylabel('normalized spectrum at zero')
plt.tight_layout()
fig.show()
```

We also examine the central draw distribution at the 2025Q3 endpoint.

```{code-cell} ipython3
def endpoint_activism_draws(fit):
    """Compute the endpoint activism ratio for every retained draw."""
    result = fit['posterior']
    draws = np.empty(result['SD'].shape[2])
    for draw in range(len(draws)):
        covariance = innovation_covariance(
            result['HD'][-1, :, draw],
            result['CD'][:, draw],
        )
        draws[draw] = policy_activism(
            result['SD'][:, -1, draw], covariance
        )
    return draws


def activism_uncertainty(draws):
    """Summarize endpoint activism without relying on its mean."""
    active = draws > 1
    ess = scalar_mcmc_ess(active)
    probability = active.mean()
    return pd.Series(
        {
            'median': np.median(draws),
            '5th percentile': np.quantile(draws, 0.05),
            '95th percentile': np.quantile(draws, 0.95),
            'P(active)': probability,
            'indicator ESS': ess,
            'probability MCSE': np.sqrt(
                probability * (1 - probability) / ess
            ),
        }
    )


updated_activity_draws = endpoint_activism_draws(updated_fit)
updated_activism = activism_uncertainty(
    updated_activity_draws
).to_frame(name=updated_label).T
updated_activism.round(3)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Central 90-percent activism draws
    name: fig-csdv-updated-activism-distributions
---
updated_activity_limits = np.quantile(
    updated_activity_draws,
    (0.05, 0.95),
)
updated_activity_bins = np.linspace(*updated_activity_limits, 31)

fig, ax = plt.subplots()
central_draws = updated_activity_draws[
    (updated_activity_draws >= updated_activity_limits[0])
    & (updated_activity_draws <= updated_activity_limits[1])
]
ax.hist(
    central_draws,
    bins=updated_activity_bins,
    histtype='step',
    lw=2,
)
ax.axvline(1, color='0.45', lw=1)
ax.set_xlabel('activism coefficient')
ax.set_ylabel('draw count')
fig.show()
```

The 2025Q3 activism median is $2.42$, but its 90 percent interval from $-16.99$
to $17.57$ makes the active-versus-passive classification weak.

The active-policy probability is $0.69$ with an MCSE of $0.04$, so the evidence
leans active without becoming decisive.

The figure plots unmodified draws between the 5th and 95th percentiles, while
the table uses every draw.

The zero lower bound and unconventional monetary policy further weaken the
economic interpretation of a short-rate projection after 2008.

### What the additional observations change

The extra quarter-century adds a dramatic episode consistent with volatility
moving independently of persistent inflation dynamics.

The pandemic is the dominant aggregate uncertainty episode, but the recent
inflation surge does not reproduce the 1970s low-frequency persistence ridge.

The data still support coefficient drift, although the trace ESS limits the
precision of the drift-rate estimate.

The 2025Q3 natural-rate and activism estimates remain weakly pinned down, and
both are nonlinear functions that become fragile near singular cases.

## Bad policy or bad luck? A verdict

The Bayesian VAR delivers a nuanced answer to the question that opened this
lecture.

- *Volatilities drifted:* the size of the shocks changed enormously, with a
  Volcker-era spike and a subsequent Great Moderation, so the bad-luck story
  captures something real.

- *Coefficients drifted too:* inflation persistence and core inflation rose
  through the 1970s and fell in the 1980s, so the systematic dynamics were not
  time invariant.

- *The new observations do not overturn that distinction:* the pandemic
  produces an extreme volatility episode, while the recent inflation surge does
  not recreate the persistence of the 1970s.

The reduced-form VAR cannot by itself prove that changes in Federal Reserve
beliefs caused the coefficient drift, because private behavior and other omitted
mechanisms can also change reduced-form dynamics.

There is one more twist, and it loops us back to the theory of this section.

The escape-route models of {doc}`phillips_learning` and
{doc}`phillips_escaping_nash` predict that inflation persistence should *grow*
along a disinflation as a learning government becomes reluctant to abandon a
high-inflation self-confirming equilibrium.

The data show the opposite because persistence fell as inflation came down after
1980.

It helped motivate later learning models, including
{cite:t}`CogleySargentConquest2005` and {cite:t}`Primiceri2006`, in which
policymakers' reluctance to disinflate in the 1970s and their eventual
conversion generate persistence that first rises and then falls.

The friendly debate with Sims, Zha, Bernanke, and Mihov thus did more than
adjudicate a historical question.

It sharpened the theoretical models of learning and drift that run through this
section, from the {doc}`self-confirming equilibria <phillips_self_confirming>`
of the *Conquest* book to the
{doc}`drifting Fed beliefs <phillips_lost_conquest>` used to interpret later
inflation.

## Exercise

```{exercise}
:label: csdv_ex1

For an $AR(1)$ process, normalized zero-frequency power is

$$
g(0)=\frac{1+\rho}{2\pi(1-\rho)}.
$$

Compute $g(0)$ for $\rho=0$, $0.85$, and $0.97$, and use the persistence
path above to interpret how inflation dynamics changed around 1980.
```

```{solution-start} csdv_ex1
:class: dropdown
```

```{code-cell} ipython3
rho = np.array([0.0, 0.85, 0.97])
g0 = (1 + rho) / (2 * np.pi * (1 - rho))
pd.Series(g0, index=rho, name='normalized power at zero').to_frame()
```

White noise has $g(0)=1/(2\pi)$, while values between roughly 2 and 10
correspond to highly persistent autoregressions with coefficients between about
$0.85$ and $0.97$.

Thus, the rise in zero-frequency power during the Great Inflation and its
decline after 1980 represent a large change in persistence.

```{solution-end}
```
