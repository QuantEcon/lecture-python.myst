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

(phillips_drifts_volatilities_2)=
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

# Drifts and Volatilities after 2000

```{index} single: Phillips Curve; Drifts and Volatilities after 2000
```

```{contents} Contents
:depth: 2
```

## Overview

{doc}`phillips_drifts_volatilities` fit the Bayesian vector autoregression of
{cite:t}`CogleySargent2005`, with drifting coefficients and stochastic
volatilities, to quarterly U.S. data on interest, unemployment, and inflation
for 1959--2000.

It found that both the volatilities and the coefficients drifted: shocks were
large around 1980, and core inflation and inflation persistence rose through the
1970s and fell together after 1980.

Cogley and Sargent's sample ends in 2000, so it misses the financial crisis, the
long stretch at the zero lower bound, the pandemic, and the 2021--2022 inflation
surge.

This lecture re-estimates the same model, with the same priors and the same
1948--1958 training sample, on data extended through the latest complete
quarter, and asks two questions.

1. Do the new observations change what the model says about 1959--2000?
2. Did the recent inflation surge bring back 1970s-style persistence?

The code repeats that of {doc}`phillips_drifts_volatilities`, where each piece is
explained; it is collapsed below so that this lecture runs on its own.

We begin with imports and the addresses of the data.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.special import expit
from threadpoolctl import threadpool_limits

# The matrices in this lecture are small, so extra BLAS threads only add overhead.
threadpool_limits(1)

archive_url = 'https://github.com/QuantEcon/data-lectures/raw/main/lectures/NEWQDATA.csv'
fred_url = ('https://fred.stlouisfed.org/graph/fredgraph.csv?'
            'id=CPIAUCSL%2CUNRATE%2CTB3MS')
```

## The model and the sampler

The model is the one in {ref}`csdv-model`: a VAR(2) in the nominal interest rate,
the logit of unemployment, and inflation, $y_t = X_t^\top\theta_t+\varepsilon_t$,
whose coefficients follow a random walk $\theta_t=\theta_{t-1}+v_t$ with
$v_t\sim N(0,Q)$, restricted to stable paths, and whose innovation covariance
$R_t=B^{-1}H_tB^{-1\prime}$ has random-walk log volatilities.

The first cell builds the model, the data, and the priors.

```{code-cell} ipython3
:tags: [hide-input]

n_variables, n_lags = 3, 2
n_regressors = 1 + n_variables * n_lags
n_coefficients = n_variables * n_regressors


def companion_matrix(θ):
    """Return the intercept and companion matrix for one coefficient vector."""
    rows = np.asarray(θ, dtype=float).reshape(n_variables, n_regressors)
    intercept = np.r_[rows[:, 0], np.zeros(n_variables)]
    companion = np.zeros((n_variables * n_lags, n_variables * n_lags))
    companion[:n_variables] = rows[:, 1:]
    companion[n_variables:, :n_variables] = np.eye(n_variables)
    return intercept, companion


def is_stable(θ_path):
    """Test whether every companion root lies strictly inside the unit circle.

    Faddeev-LeVerrier gives each characteristic polynomial, and the Schur-Cohn
    recursion checks its roots without computing them.
    """
    rows = np.asarray(θ_path, dtype=float).reshape(n_variables, n_regressors, -1)
    A = np.zeros((rows.shape[2], 6, 6))
    A[:, :n_variables] = rows[:, 1:].transpose(2, 0, 1)
    A[:, n_variables:, :n_variables] = np.eye(n_variables)
    p = np.ones((len(A), 7))            # det(λI - A) = λ^6 + p_1 λ^5 + ... + p_6
    M = np.zeros_like(A)
    for k in range(1, 7):
        AM = A @ (M + p[:, k - 1, None, None] * np.eye(6))
        p[:, k] = -np.einsum('tii->t', AM) / k
        M = AM
    for m in range(6, 0, -1):
        reflection = p[:, m] / p[:, 0]
        if np.any(np.abs(reflection) >= 1):
            return False
        p = p[:, :m] - reflection[:, None] * p[:, m:0:-1]
    return True


def b_matrix(β):
    """Construct B from β_21, β_31, and β_32."""
    matrix = np.eye(n_variables)
    matrix[1, 0], matrix[2, 0], matrix[2, 1] = β
    return matrix


def innovation_covariance(h, β):
    """Construct R_t = B^{-1} H_t B^{-1}' from one vector of variances."""
    inverse = np.linalg.inv(b_matrix(β))
    return inverse @ np.diag(h) @ inverse.T


def prepare_data(table):
    """Transform a quarterly table and construct the VAR(2) data."""
    ur = table['ur'].to_numpy(dtype=float)
    raw_y = np.column_stack([table['y3'].to_numpy(dtype=float),
                             0.01 * np.log(ur / (1 - ur)),
                             table['dp'].to_numpy(dtype=float)])
    regressors = np.ones((len(raw_y) - n_lags, n_regressors))
    for lag in range(1, n_lags + 1):
        left = 1 + n_variables * (lag - 1)
        regressors[:, left:left + n_variables] = raw_y[n_lags - lag:-lag]
    targets = raw_y[n_lags:]
    dates = table['date'].to_numpy(dtype=float)[n_lags:]
    n_training = 4 * 11 - n_lags - 1          # 1948Q4--1958Q4
    return {'prior_y': targets[:n_training], 'prior_x': regressors[:n_training],
            'dates': dates[n_training:], 'y': targets[n_training:],
            'x': regressors[n_training:]}


def calibrate_prior(model_data, γ_squared=3.5e-4):
    """Calibrate every prior from a constant-coefficient training-sample VAR."""
    y, x = model_data['prior_y'], model_data['prior_x']
    xx_inverse = np.linalg.inv(x.T @ x)
    coefficients = xx_inverse @ x.T @ y
    r_bar = np.cov(y - x @ coefficients, rowvar=False, ddof=1)
    p_bar = np.kron(r_bar, xx_inverse)
    T0 = n_coefficients + 1
    return {'θ_mean': coefficients.T.reshape(-1), 'θ_covariance': p_bar,
            'q_center': γ_squared * p_bar, 'q_scale': T0 * γ_squared * p_bar,
            'q_degrees_freedom': T0, 'log_h_mean': np.log(np.diag(r_bar)),
            'log_h_variance': 10.0, 'β_variance': 10_000.0,
            'σ_degrees_freedom': 1.0, 'σ_scale': 0.01**2}
```

The second cell is the sampler of {ref}`csdv-sampler`: a Carter--Kohn draw of
the coefficient path, redrawn until it is stable, followed by the updates of
$Q$, the volatility scales $\sigma$, the covariance parameters $\beta$, and the
volatility paths.

```{code-cell} ipython3
:tags: [hide-input]

def stacked_cholesky(matrices):
    """Cholesky factors of a stack of covariance matrices, with tiny jitter."""
    matrices = 0.5 * (matrices + np.swapaxes(matrices, -1, -2))
    scale = np.abs(np.diagonal(matrices, axis1=-2, axis2=-1)).max(axis=-1)
    jitter = 1e-12 * np.maximum(1.0, scale)[..., None, None]
    return np.linalg.cholesky(matrices + jitter * np.eye(matrices.shape[-1]))


def design_matrices(x):
    """Stack the observation matrices X_t' = I_3 kron x_t'."""
    X = np.einsum('ij,tk->tijk', np.eye(n_variables), x)
    return X.reshape(len(x), n_variables, n_coefficients)


def kalman_forward(y, x, q, r, prior):
    """Kalman filter for θ_t given Q and R_1, ..., R_T."""
    T, k = len(y), n_coefficients
    X = design_matrices(x)
    means = np.empty((T + 1, k))
    covariances = np.empty((T + 1, k, k))
    predicted = np.empty((T + 1, k, k))
    means[0], covariances[0] = prior['θ_mean'], prior['θ_covariance']
    for t in range(T):
        p = covariances[t] + q
        px = p @ X[t].T
        gain = np.linalg.solve(X[t] @ px + r[t], px.T).T
        means[t + 1] = means[t] + gain @ (y[t] - X[t] @ means[t])
        c = p - gain @ px.T
        covariances[t + 1] = 0.5 * (c + c.T)
        predicted[t + 1] = p
    return means, covariances, predicted


def draw_coefficient_path(current, y, x, q, h, β, prior, rng,
                          max_passes=50):
    """Redraw the Carter-Kohn path until it is stable (else keep current)."""
    b_inverse = np.linalg.inv(b_matrix(β))
    r = np.einsum('ij,tj,kj->tik', b_inverse, h, b_inverse)
    means, covariances, predicted = kalman_forward(y, x, q, r, prior)
    gains = np.swapaxes(np.linalg.solve(predicted[1:], covariances[:-1]), 1, 2)
    roots = stacked_cholesky(np.concatenate((
        covariances[:-1] - covariances[:-1] @ np.swapaxes(gains, 1, 2),
        covariances[-1:])))
    T = len(y)
    for passes in range(1, max_passes + 1):
        noise = np.einsum('tij,tj->ti', roots, rng.standard_normal(means.shape))
        path = np.empty_like(means)
        path[T] = means[T] + noise[T]
        for t in range(T - 1, -1, -1):
            path[t] = means[t] + gains[t] @ (path[t + 1] - means[t]) + noise[t]
        if is_stable(path.T):
            return path.T, passes
    return current, max_passes + 1


def draw_q(θ_path, prior, rng):
    """Inverse-Wishart draw of Q given the increments (Bartlett decomposition)."""
    increments = np.diff(θ_path, axis=1)
    scale = prior['q_scale'] + increments @ increments.T
    degrees_freedom = prior['q_degrees_freedom'] + increments.shape[1]
    k = len(scale)
    A = np.tril(rng.standard_normal((k, k)), -1)
    A[np.diag_indices(k)] = np.sqrt(rng.chisquare(degrees_freedom - np.arange(k)))
    M = np.linalg.solve(A, np.linalg.cholesky(scale).T).T
    return M @ M.T


def var_residuals(y, x, θ_path):
    """VAR residuals at dates 1, ..., T for a path θ_0, ..., θ_T."""
    coefficients = θ_path[:, 1:].T.reshape(len(y), n_variables, n_regressors)
    return y - np.einsum('tk,tnk->tn', x, coefficients)


def draw_σ(h, prior, rng):
    """Inverse-gamma draws of the log-volatility innovation variances."""
    increments = np.diff(np.log(h), axis=0)
    shape = (prior['σ_degrees_freedom'] + len(increments)) / 2
    scale = (prior['σ_scale'] + np.sum(increments**2, axis=0)) / 2
    return np.sqrt(scale / rng.gamma(shape, 1.0, size=n_variables))


def draw_β(residuals, h, prior, rng):
    """Gaussian draws of the free elements of B."""
    β = np.empty(3)
    for equation, slots in ((1, slice(0, 1)), (2, slice(1, 3))):
        standardized = residuals / np.sqrt(h[1:, equation])[:, None]
        regressors = -standardized[:, :equation]
        covariance = np.linalg.inv(np.eye(equation) / prior['β_variance']
                                   + regressors.T @ regressors)
        mean = covariance @ regressors.T @ standardized[:, equation]
        β[slots] = mean + np.linalg.cholesky(covariance) @ (
            rng.standard_normal(equation))
    return β


def draw_volatility_path(h, residuals, β, σ, prior, rng):
    """Jacquier-Polson-Rossi single-site updates of log h_0, ..., log h_T."""
    T = len(residuals)
    squared = (residuals @ b_matrix(β).T)**2
    old, new = np.log(h), np.empty_like(h)
    variance = σ**2
    v0 = (prior['log_h_variance'] * variance
          / (variance + prior['log_h_variance']))
    new[0] = v0 * (prior['log_h_mean'] / prior['log_h_variance']
                   + old[1] / variance) + np.sqrt(v0) * rng.standard_normal(3)
    z = rng.standard_normal((T, n_variables))
    log_u = np.log(rng.random((T, n_variables)))
    for t in range(1, T + 1):
        if t < T:
            proposal = (0.5 * (new[t - 1] + old[t + 1])
                        + np.sqrt(variance / 2) * z[t - 1])
        else:
            proposal = new[t - 1] + σ * z[t - 1]
        log_ratio = 0.5 * (old[t] - proposal) + 0.5 * squared[t - 1] * (
            np.exp(-old[t]) - np.exp(-proposal))
        new[t] = np.where(log_u[t - 1] <= log_ratio, proposal, old[t])
    return np.exp(new)


def run_sampler(y, x, prior, n_sweeps, burn, thin, seed, warmup=200):
    """Simulate the stability-restricted posterior; keep every thin-th sweep."""
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    T = len(y)
    θ = np.repeat(prior['θ_mean'][:, None], T + 1, axis=1)
    residuals = var_residuals(y, x, θ)
    changes = np.diff(y, axis=0)
    h = np.exp(np.vstack((
        np.tile(prior['log_h_mean'], (2, 1)),
        np.log(np.maximum((changes - changes.mean(axis=0))**2, 1e-300)))))
    β = np.zeros(3)
    for _ in range(warmup):
        σ = draw_σ(h, prior, rng)
        β = draw_β(residuals, h, prior, rng)
        h = draw_volatility_path(h, residuals, β, σ, prior, rng)
    q = prior['q_center']
    n_keep = len(range(burn, n_sweeps, thin))
    draws = {'θ': np.empty((n_keep, n_coefficients, T)),
             'h': np.empty((n_keep, T, n_variables)),
             'β': np.empty((n_keep, 3)),
             'Q': np.empty((n_keep, n_coefficients, n_coefficients))}
    trace_q = np.empty(n_sweeps)
    passes = np.empty(n_sweeps, dtype=int)
    for sweep in range(n_sweeps):
        θ, passes[sweep] = draw_coefficient_path(θ, y, x, q, h, β, prior, rng)
        q = draw_q(θ, prior, rng)
        residuals = var_residuals(y, x, θ)
        σ = draw_σ(h, prior, rng)
        β = draw_β(residuals, h, prior, rng)
        h = draw_volatility_path(h, residuals, β, σ, prior, rng)
        trace_q[sweep] = np.trace(q)
        if sweep >= burn and (sweep - burn) % thin == 0:
            j = (sweep - burn) // thin
            draws['θ'][j], draws['h'][j] = θ[:, 1:], h[1:]
            draws['β'][j], draws['Q'][j] = β, q
    draws.update(trace_q=trace_q, passes=passes,
                 minutes=(time.perf_counter() - started) / 60)
    return draws
```

The third cell computes the summaries of {ref}`csdv-results`: the posterior mean
of $R_t$, core inflation and the natural rate, and the normalized inflation
spectrum $g_{\pi\pi}(\omega,t)$.

```{code-cell} ipython3
:tags: [hide-input]

def mean_innovation_covariance(h_draws, β_draws):
    """E(R_t | T): the average of B^{-1} H_t B^{-1}' over retained draws."""
    b_inverse = np.linalg.inv(np.stack([b_matrix(β) for β in β_draws]))
    total = np.einsum('dij,dtj,dkj->tik', b_inverse, h_draws, b_inverse,
                      optimize=True)
    return total / len(β_draws)


def local_means(θ_path):
    """Core inflation (annual rate) and the natural unemployment rate."""
    θ_path = np.asarray(θ_path, dtype=float).reshape(n_coefficients, -1)
    core, natural = np.empty(θ_path.shape[1]), np.empty(θ_path.shape[1])
    for t, θ in enumerate(θ_path.T):
        intercept, companion = companion_matrix(θ)
        m = np.linalg.solve(np.eye(6) - companion, intercept)
        core[t], natural[t] = 4 * m[2], expit(100 * m[1])
    return core, natural


def inflation_spectrum(θ, covariance, frequencies):
    """Inflation power f(ω) and normalized power g(ω) at one date."""
    _, companion = companion_matrix(θ)
    innovation = np.zeros((6, 6))
    innovation[:3, :3] = covariance
    variance = linalg.solve_discrete_lyapunov(companion, innovation)[2, 2]
    power = np.empty(len(frequencies))
    for j, frequency in enumerate(frequencies):
        transfer = np.linalg.inv(
            np.eye(6) - companion * np.exp(-2j * np.pi * frequency))
        power[j] = np.real(transfer @ innovation @ transfer.conj().T)[2, 2]
    return power / (2 * np.pi), power / (2 * np.pi * variance)


def posterior_feature_draws(draws, indices):
    """Core inflation, the natural rate, and g(0) at selected dates, by draw."""
    n_draws = len(draws['β'])
    core = np.empty((len(indices), n_draws))
    natural, persistence = np.empty_like(core), np.empty_like(core)
    for d in range(n_draws):
        θ_d = draws['θ'][d][:, indices]
        core[:, d], natural[:, d] = local_means(θ_d)
        for row, t in enumerate(indices):
            covariance = innovation_covariance(draws['h'][d][t], draws['β'][d])
            persistence[row, d] = inflation_spectrum(
                θ_d[:, row], covariance, [0.0])[1][0]
    return {'core': 100 * core, 'natural': 100 * natural,
            'persistence': persistence}


def integrated_autocorrelation_time(x, c=5):
    """Sokal's adaptive-window estimate of the integrated autocorrelation time."""
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    spectrum = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    τ = 2 * np.cumsum(acf / acf[0]) - 1
    return τ[np.flatnonzero(np.arange(n) >= c * τ)[0]]
```

## The extended sample

We keep Cogley and Sargent's archived data through 2000Q4 and append newer
observations from FRED: seasonally adjusted [CPI][fred-cpi], seasonally
adjusted [unemployment][fred-unemployment], and the [three-month Treasury-bill
rate][fred-interest].

[fred-cpi]: https://fred.stlouisfed.org/series/CPIAUCSL
[fred-unemployment]: https://fred.stlouisfed.org/series/UNRATE
[fred-interest]: https://fred.stlouisfed.org/series/TB3MS

The transformations and within-quarter timing are unchanged: CPI comes from the
third month of each quarter, unemployment is a three-month average, and the
interest rate comes from the first month.

Keeping the archive for 1948--2000 means that revisions to old data are not
mistaken for information in the new quarters; the first appended inflation rate
uses revised CPI levels for both 2000Q4 and 2001Q1.

As the [BLS notes](https://www.bls.gov/web/empsit/cpsee_e12.pdf), the October
2025 unemployment rate was never collected, because of the federal government
shutdown, so 2025Q4 lacks one of its three months.

Rather than end the sample at 2025Q3 forever, we fill isolated missing months
by linear interpolation between their neighbors, and we print the months filled.

The sample then ends at the last quarter for which all three series are
complete, found automatically.

```{code-cell} ipython3
archive = pd.read_csv(archive_url)
fred = pd.read_csv(fred_url, parse_dates=['observation_date'],
                   index_col='observation_date')

unrate = fred['UNRATE'].dropna()
unrate = unrate.reindex(pd.date_range(unrate.index[0], unrate.index[-1],
                                      freq='MS'))
filled_months = list(unrate.index[unrate.isna()].strftime('%Y-%m'))
unrate = unrate.interpolate(limit=1, limit_area='inside')


def by_quarter(series, months):
    """Observations from the given months of each quarter, indexed by quarter."""
    picked = series[series.index.month.isin(months)].dropna()
    picked.index = picked.index.to_period('Q')
    return picked


month_counts = unrate.groupby(unrate.index.to_period('Q')).count()
table = pd.concat({
    'interest': by_quarter(fred['TB3MS'], (1, 4, 7, 10)),
    'cpi': by_quarter(fred['CPIAUCSL'], (3, 6, 9, 12)),
    'unemployment': unrate.groupby(unrate.index.to_period('Q')).mean()[
        month_counts == 3],
}, axis=1)
table['y3'] = np.log1p(table['interest'] / 400)
table['ur'] = table['unemployment'] / 100
table['dp'] = np.log(table['cpi']).diff()
table['date'] = table.index.year + (table.index.quarter - 1) / 4

extension = table.loc['2001Q1':, ['date', 'y3', 'ur', 'dp']]
incomplete = extension.index[extension.isna().any(axis=1)]
if len(incomplete):
    extension = extension.loc[:incomplete[0] - 1]
spliced = pd.concat((archive, extension.reset_index(drop=True)),
                    ignore_index=True)
assert np.allclose(np.diff(spliced['date']), 0.25)
sample_end = extension.index[-1]

print('months filled by interpolation:', filled_months)
print('last quarter of the sample:', sample_end)
```

The next figure shows the data, with a dashed line at the end of Cogley and
Sargent's sample.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Observed $\pi_t$, $u_t$, and $i_t$ in the extended sample
    name: fig-csdv2-data
---
shown = spliced[spliced['date'] >= 1959]
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
axes[0].plot(shown['date'], 400 * shown['dp'], lw=2)
axes[0].set_ylabel('inflation (annual %)')
axes[1].plot(shown['date'], 100 * shown['ur'], lw=2)
axes[1].set_ylabel('unemployment (%)')
axes[2].plot(shown['date'], 400 * np.expm1(shown['y3']), lw=2)
axes[2].set_ylabel('interest (annual %)')
axes[2].set_xlabel('year')
for ax in axes:
    ax.axvline(2000.75, color='0.6', ls='--', lw=1)
plt.tight_layout()
plt.show()
```

The extension adds the financial-crisis recession, a pandemic spike in quarterly
unemployment, a short inflation surge in 2021--2022, and two long stretches of
near-zero interest rates.

Because inflation is annualized from quarterly changes, isolated movements look
large, but the recent surge is much shorter than the sustained rise of the
1970s.

## Fitting the extended sample

The sample is now much longer, and stable coefficient paths are much harder to
draw over it, so a sweep costs several times more than in
{doc}`phillips_drifts_volatilities`.

To bound the cost, a sweep tries at most 50 backward passes before keeping the
current path, which still leaves the posterior invariant (the table reports how
often that happens), and we run 10,000 sweeps, discard the first 1,000, and keep
every 5th of the rest.

```{code-cell} ipython3
data = prepare_data(spliced)
prior = calibrate_prior(data)
burn = 1_000
posterior = run_sampler(data['y'], data['x'], prior,
                        n_sweeps=10_000, burn=burn, thin=5, seed=42)

dates = data['dates']
years = np.floor(dates + 1e-8).astype(int)
annual = np.array([np.flatnonzero(years == year)[-1]
                   for year in np.unique(years)])
trace_q = posterior['trace_q'][burn:]
τ = integrated_autocorrelation_time(trace_q)
passes = posterior['passes'][burn:]

pd.Series({
    'run time (minutes)': posterior['minutes'],
    'quarters in the estimation sample': len(dates),
    'retained draws': len(posterior['β']),
    'mean backward passes per sweep': passes.mean(),
    'sweeps that kept the current path': np.sum(passes > 50),
    'autocorrelation time of tr(Q)': τ,
    'effective sample size of tr(Q)': len(trace_q) / τ,
}, name='value').to_frame().round(3)
```

As in {doc}`phillips_drifts_volatilities`, the chain moves slowly in the
direction of $Q$, and with fewer sweeps the effective sample size of
$\operatorname{tr}(Q)$ is even smaller, so magnitudes that depend on $Q$ carry
large Monte Carlo error; the text relies only on features that held in runs
with other seeds.

## Did coefficient drift continue?

The histogram shows $\operatorname{tr}(Q)$ at every sweep after the burn-in.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Posterior $\operatorname{tr}(Q)$ in the extended sample and prior $\operatorname{tr}(\bar Q)$
    name: fig-csdv2-drift-rate
---
fig, ax = plt.subplots()
ax.hist(trace_q, bins=50, histtype='step', lw=2)
ax.axvline(np.trace(prior['q_center']), color='C1', lw=2,
           label=r'prior $\mathrm{tr}(\bar Q)$')
ax.set_xlabel(r'$\mathrm{tr}(Q)$')
ax.set_ylabel('sweeps')
ax.legend()
plt.show()
```

The table below gives the numbers behind the figure.

```{code-cell} ipython3
q_eigenvalues = np.linalg.eigvalsh(posterior['Q'].mean(axis=0))[::-1]
pd.Series({
    'posterior mean of tr(Q)': trace_q.mean(),
    'smallest tr(Q) / tr(Q bar)': trace_q.min() / np.trace(prior['q_center']),
    'share of the first principal component': q_eigenvalues[0] / q_eigenvalues.sum(),
    'share of the first three': q_eigenvalues[:3].sum() / q_eigenvalues.sum(),
}, name='value').to_frame().round(3)
```

The drift rate is several times larger than in the 1959--2000 fit of
{doc}`phillips_drifts_volatilities`, and the drift is concentrated in even fewer
directions.

The chain explores this posterior very slowly: in runs with other seeds the
posterior mean of $\operatorname{tr}(Q)$ differed by a factor of more than two,
although it was always several times the 1959--2000 value.

One likely contributor is the pandemic: in 2020Q2 the logit of unemployment
jumped by dozens of innovation standard deviations, far more than a random-walk
volatility can absorb within a quarter, so part of the jump is attributed to
the coefficients.

Because $Q$ is a single drift covariance for the whole sample, this histogram
cannot by itself say whether drift slowed or sped up after 2000; for that we
look at what the coefficients imply.

## Volatility after the Great Moderation

The innovation standard deviations and correlations come from the posterior mean
$\mathbb{E}(R_t\mid T)$, as in {doc}`phillips_drifts_volatilities`.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Standard deviations and correlations implied by $E(R_t\mid T)$, extended sample
    name: fig-csdv2-volatility-correlation
---
r_mean = mean_innovation_covariance(posterior['h'], posterior['β'])
sd = 1e4 * np.sqrt(np.diagonal(r_mean, axis1=1, axis2=2))


def correlation(left, right):
    return r_mean[:, left, right] / np.sqrt(r_mean[:, left, left]
                                            * r_mean[:, right, right])


fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharex=True)
for row, (index, label) in enumerate(((0, 'Nominal interest'),
                                      (2, 'Inflation'), (1, 'Unemployment'))):
    axes[row, 0].plot(dates, sd[:, index], lw=2)
    axes[row, 0].set_title(label)
for row, (left, right, label) in enumerate(((0, 1, 'Interest--unemployment'),
                                            (0, 2, 'Interest--inflation'),
                                            (2, 1, 'Inflation--unemployment'))):
    axes[row, 1].plot(dates, correlation(left, right), lw=2)
    axes[row, 1].set_title(label)
for ax in axes.flat:
    ax.axvline(2000.75, color='0.6', ls='--', lw=1)
axes[1, 0].set_ylabel(r'standard deviation $\times 10^4$')
axes[1, 1].set_ylabel('correlation')
axes[-1, 0].set_xlabel('year')
axes[-1, 1].set_xlabel('year')
plt.tight_layout()
plt.show()
```

The table below gives the numbers behind the figure.

```{code-cell} ipython3
pd.DataFrame({
    'peak date': [dates[np.argmax(sd[:, i])] for i in range(3)],
    'standard deviation at the end': sd[-1],
}, index=['interest', 'unemployment', 'inflation']).round(2)
```

The Volcker years still dominate interest-rate innovations, the financial crisis
produces the largest inflation innovations, and the pandemic produces by far the
largest unemployment innovations, together with a sharp swing in the
inflation--unemployment correlation.

Recent episodes thus changed the size and composition of reduced-form shocks
dramatically, which is direct evidence for a bad-luck component, although
reduced-form innovations are not structural shocks.

The generalized innovation variance $\log|\mathbb{E}(R_t\mid T)|$ summarizes the
total one-step uncertainty.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Generalized innovation variance $\log |E(R_t\mid T)|$, extended sample
    name: fig-csdv2-total-variance
---
logdet_r = np.linalg.slogdet(r_mean)[1]
fig, ax = plt.subplots()
ax.plot(dates, logdet_r, lw=2)
ax.axvline(2000.75, color='0.6', ls='--', lw=1)
ax.set_xlabel('year')
ax.set_ylabel(r'$\log |E(R_t\mid T)|$')
plt.show()
```

The table below gives the numbers behind the figure.

```{code-cell} ipython3
def period_mean(values, first, last):
    """Average a quarterly series over the years first, ..., last."""
    return values[(years >= first) & (years <= last)].mean()


display_periods = {'1980--1982': (1980, 1982), '1990--2000': (1990, 2000),
                   '2010--2015': (2010, 2015), '2020--2021': (2020, 2021)}


def extreme(values, first, last, statistic):
    return statistic(values[(years >= first) & (years <= last)])


pd.DataFrame({
    'interest innovation sd (basis points)': [
        period_mean(sd[:, 0], *p) for p in display_periods.values()],
    'interest-inflation correlation': [
        period_mean(correlation(0, 2), *p) for p in display_periods.values()],
    'log |E(R_t | T)|, max or min': [
        extreme(logdet_r, 1980, 1982, np.max), extreme(logdet_r, 1990, 2000, np.min),
        extreme(logdet_r, 2010, 2015, np.min), extreme(logdet_r, 2020, 2021, np.max)],
}, index=display_periods.keys()).round(3)
```

Joint innovation variance rises in the financial crisis, falls to a trough in
the early 2010s deeper than its 1990s low, and jumps in 2020 above even its
1981 peak before falling back.

The trough is not a second Great Moderation.

At the zero lower bound the Treasury-bill rate barely moved, so its innovations
nearly vanished and lost their correlation with inflation innovations, as the
table shows, and the log determinant collapses with them.

For the same reason a short-rate measure of policy activism carries little
information after 2008, and we do not compute one here.

(csdv2_persistence)=
## Core inflation, the natural rate, and persistence

We compute core inflation, the natural rate, and $g_{\pi\pi}(0,t)$ from every
retained draw, as in {ref}`csdv_persistence`.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Posterior medians and pointwise 90 percent intervals for $\bar\pi_t$, $\bar u_t$, and $g_{\pi\pi}(0,t)$, extended sample
    name: fig-csdv2-feature-uncertainty
---
feature_draws = posterior_feature_draws(posterior, annual)
medians = {key: np.median(value, axis=1) for key, value in feature_draws.items()}

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for ax, (key, ylabel) in zip(axes, (('core', 'core inflation (%)'),
                                    ('natural', 'natural rate (%)'),
                                    ('persistence', r'$g_{\pi\pi}(0,t)$'))):
    lower, upper = np.quantile(feature_draws[key], (0.05, 0.95), axis=1)
    line, = ax.plot(dates[annual], medians[key], lw=2)
    ax.fill_between(dates[annual], lower, upper, color=line.get_color(),
                    alpha=0.2)
    ax.axvline(2000.75, color='0.6', ls='--', lw=1)
    ax.set_ylabel(ylabel)
axes[2].axhline(1 / (2 * np.pi), color='0.45', ls=':', lw=1)
axes[-1].set_xlabel('year')
plt.tight_layout()
plt.show()
```

The dotted line in the bottom panel is the white-noise value $1/(2\pi)$.

The table summarizes the posterior medians by period.

```{code-cell} ipython3
annual_years = years[annual]


def median_summary(key, first, last, statistic=np.mean):
    """Apply a statistic to the annual posterior medians over first, ..., last."""
    mask = (annual_years >= first) & (annual_years <= last)
    return statistic(medians[key][mask])


core_peak = np.argmax(medians['core'] * (annual_years <= 2000))
pd.Series({
    'core inflation, 1970s peak (%)': medians['core'][core_peak],
    'year of the core peak': annual_years[core_peak],
    'natural rate, 1970--1979 range (%)': (
        f"{median_summary('natural', 1970, 1979, np.min):.1f}--"
        f"{median_summary('natural', 1970, 1979, np.max):.1f}"),
    'g(0), 1970--1979 mean': median_summary('persistence', 1970, 1979),
    'last year before 2001 with g(0) > 1': annual_years[
        (annual_years <= 2000) & (medians['persistence'] > 1)].max(),
    'core inflation, 2001--2019 range (%)': (
        f"{median_summary('core', 2001, 2019, np.min):.1f}--"
        f"{median_summary('core', 2001, 2019, np.max):.1f}"),
    'core inflation, 2020 onward mean (%)': median_summary('core', 2020, 2100),
    'g(0), 2001--2019 mean': median_summary('persistence', 2001, 2019),
    'g(0), 2020 onward mean': median_summary('persistence', 2020, 2100),
}, name='posterior median').to_frame()
```

Adding a quarter-century of data changes the estimated history of 1959--2000 a
great deal.

With the larger drift rate, many draws put a root of the late-1970s VAR close to
the unit circle, so the medians of core inflation and of the natural rate around
1979 are higher than in {doc}`phillips_drifts_volatilities`, in some runs far
higher, their bands are enormous, and persistence stays above one until 1980 or
later.

How large these numbers are depends on how far the chain has wandered into the
tail of $Q$, and they differed a lot across runs with other seeds; what survives
in all runs is the shape: core inflation and persistence rise through the 1970s
and collapse in the early 1980s.

After 2000 the picture is calm and robust.

Core inflation stays below about 4 percent, and persistence stays close to the
white-noise value, apart from a brief rise around 2020--2021 that is far
smaller than in the 1970s and soon reverses.

A burst of large inflation innovations kept measured inflation high for several
quarters in 2021--2022 without the strong propagation that the model finds in
the 1970s.

The normalized spectrum over all frequencies shows the same thing.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Normalized inflation spectrum $g_{\pi\pi}(\omega,t)$, extended sample
    name: fig-csdv2-spectrum
---
θ_mean = posterior['θ'].mean(axis=0)
frequencies = np.linspace(0, 0.5, 41)
normalized = np.column_stack([
    inflation_spectrum(θ_mean[:, t], r_mean[t], frequencies)[1]
    for t in range(len(dates))
])
fig, ax = plt.subplots(figsize=(8, 4))
image = ax.pcolormesh(dates, frequencies, np.log10(normalized), shading='auto')
ax.axvline(2000.75, color='0.9', ls='--', lw=1)
ax.set_xlabel('year')
ax.set_ylabel('cycles per quarter')
fig.colorbar(image, ax=ax, label=r'$\log_{10} g_{\pi\pi}(\omega,t)$')
plt.show()
```

The normalized spectrum has a low-frequency ridge from the late 1960s to the
early 1980s and, after 2000, only a brief one around 2020: the recent inflation
surge shows up mainly as large shocks, not as lasting stronger propagation.

### Two measures of persistence

{doc}`phillips_lost_conquest` gives the Fed a learning model in which perceived
persistence is the coefficient on lagged inflation in a regression of
*year-over-year* inflation on its own lag, sampled quarterly.

Year-over-year rates overlap: consecutive values share three of their four
quarters.

Even if quarterly inflation were white noise, year-over-year inflation sampled
quarterly would have a first-order autocorrelation of $3/4$, so a persistence
measure based on it is pushed toward one and falls later and less than a
measure based on quarterly rates.

The next cell compares least-squares $AR(1)$ coefficients of the two inflation
rates in our data, period by period; {ref}`lc_ex5` reruns that lecture's
learning Fed on quarterly inflation.

```{code-cell} ipython3
cpi_inflation = pd.Series(400 * spliced['dp'].to_numpy(), index=spliced['date'])
year_over_year = cpi_inflation.rolling(4).mean()


def ar1_coefficient(series, first, last):
    """Least-squares AR(1) slope over the years first, ..., last."""
    z = series[(series.index >= first) & (series.index < last + 1)].dropna()
    X = np.column_stack((np.ones(len(z) - 1), z.to_numpy()[:-1]))
    return np.linalg.lstsq(X, z.to_numpy()[1:], rcond=None)[0][1]


periods = ((1960, 1969), (1970, 1979), (1985, 1999), (2000, 2019), (2020, 2100))
pd.DataFrame({
    'quarterly inflation': [ar1_coefficient(cpi_inflation, *p) for p in periods],
    'year-over-year inflation': [ar1_coefficient(year_over_year, *p)
                                 for p in periods],
}, index=['1960s', '1970s', '1985--1999', '2000--2019', '2020 onward']).round(2)
```

The $AR(1)$ coefficient of year-over-year inflation is near one in the 1960s and
1970s, and it stays high after 1985 and even in 2000--2019, when the coefficient
for quarterly inflation is about zero.

The overlap both inflates the year-over-year measure and blunts its decline,
which is why the perceived persistence in {doc}`phillips_lost_conquest` falls
later and less than the quarterly measures here and in
{doc}`phillips_drifts_volatilities`.

## What the additional observations change

The extra quarter-century strengthens the bad-luck evidence for the recent period.

- It adds large volatility episodes, the financial crisis in inflation and the
  pandemic in unemployment, and a zero-lower-bound period in which the short rate
  says little about policy.
- The recent inflation surge did not bring back 1970s-style propagation:
  persistence after 2000 stays near the white-noise value, and core inflation
  stays low.
- The drift rate, and with it the estimated history of the 1970s, becomes much
  harder to pin down: the posterior puts weight on far faster drift than the
  1959--2000 fit, and a chain short enough for a lecture cannot explore that
  tail reliably, so magnitudes that depend on $Q$ should not be taken at face
  value.

The shape of the earlier drift documented in {doc}`phillips_drifts_volatilities`
survives: persistence and core inflation still rise in the 1970s and fall in
the early 1980s.

The next and last lecture, {doc}`phillips_lost_conquest`, returns to a learning
Fed: it asks whether drifting beliefs about persistence and the slope of the
Phillips curve, of the kind that {doc}`phillips_learning` and
{doc}`phillips_priors` model, help explain the Fed's slow response to the
inflation of 2021--2022, and so whether the conquest of American inflation was
lost.

## Exercises

```{exercise}
:label: csdv2_ex1

{cite:t}`Sims2001comment` pointed out that *filtered* estimates of drifting
coefficients wander even when the true coefficients are constant, which is why
{cite:t}`CogleySargent2005` report *smoothed* estimates.

Simulate $y_t=\rho_t y_{t-1}+w_t$, $w_t\sim N(0,1)$, for $T=168$ quarters in two
cases: (a) $\rho_t=0.5$ for all $t$, and (b) $\rho_t$ a random walk that starts
at $0.5$ with innovation variance $4\times 10^{-4}$.

In each case run the scalar Kalman filter of {doc}`kalman` that treats $\rho_t$
as a random walk with innovation variance $q=4\times 10^{-4}$, starting from mean
0 and variance 1, and then the smoother of {ref}`csdv-appendix`.

Compare the standard deviations over time of the true, filtered, and smoothed
paths, ignoring the first 20 quarters.

What does the smoothed path look like if the filter assumes $q=0$?

At the last date of a sample the smoothed and filtered estimates coincide.
What does that imply for the estimates in this lecture that bear on the 2020s?
```

```{solution-start} csdv2_ex1
:class: dropdown
```

Here is one solution.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Filtered and smoothed estimates of a constant and a drifting coefficient
    name: fig-csdv2-filter-smooth
---
def tvp_ar1(T, ρ0, q, rng):
    """Simulate y_t = ρ_t y_{t-1} + w_t with ρ_t a random walk."""
    ρ = ρ0 + np.r_[0, np.cumsum(np.sqrt(q) * rng.standard_normal(T))]
    y = np.zeros(T + 1)
    for t in range(1, T + 1):
        y[t] = ρ[t] * y[t - 1] + rng.standard_normal()
    return y, ρ


def filter_and_smooth(y, q, mean0=0.0, var0=1.0):
    """Scalar Kalman filter and Rauch-Tung-Striebel smoother for ρ_t."""
    T = len(y) - 1
    m, P, P_pred = np.empty(T + 1), np.empty(T + 1), np.empty(T + 1)
    m[0], P[0] = mean0, var0
    for t in range(1, T + 1):
        P_pred[t] = P[t - 1] + q
        K = P_pred[t] * y[t - 1] / (y[t - 1]**2 * P_pred[t] + 1)
        m[t] = m[t - 1] + K * (y[t] - y[t - 1] * m[t - 1])
        P[t] = P_pred[t] - K * y[t - 1] * P_pred[t]
    s = m.copy()
    for t in range(T - 1, -1, -1):
        s[t] = m[t] + P[t] / P_pred[t + 1] * (s[t + 1] - m[t])
    return m, s


rng = np.random.default_rng(3)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
rows = {}
for ax, (q_true, title) in zip(axes, ((0.0, r'constant $\rho$'),
                                      (4e-4, r'drifting $\rho$'))):
    y, ρ = tvp_ar1(168, 0.5, q_true, rng)
    filtered, smoothed = filter_and_smooth(y, q=4e-4)
    _, flat = filter_and_smooth(y, q=0.0)
    for path, label in ((ρ, 'true'), (filtered, 'filtered'),
                        (smoothed, 'smoothed')):
        ax.plot(path[1:], lw=2 if label == 'true' else 1.5, label=label)
    ax.set_title(title)
    ax.set_xlabel('quarter')
    rows[title] = {'true': ρ[20:].std(), 'filtered': filtered[20:].std(),
                   'smoothed': smoothed[20:].std(),
                   'smoothed, q = 0': flat[20:].std()}
axes[0].set_ylabel(r'$\rho_t$')
axes[0].legend()
plt.tight_layout()
plt.show()
```

The table below gives the numbers behind the figure.

```{code-cell} ipython3
pd.DataFrame(rows).T.round(3)
```

When $\rho$ is constant, the filtered estimate wanders, with a standard
deviation over time several times that of the smoothed estimate, which is
nearly flat; that is Sims's point.

When $\rho$ drifts, the filtered path adds transient noise to the drift, while
the smoothed path is smoother than the truth.

With $q=0$ the smoother returns an exactly flat path, as in Cogley and Sargent's
footnote 16: the backward pass has $J_t=1$ and no noise, so every date receives
the end-of-sample estimate.

Near the end of a sample the smoother has few later observations to draw on,
so its estimates behave more like filtered ones, and at the last date they are
the filtered estimate.

The estimates for the 2020s in this lecture therefore carry some of the
transient wandering that Sims warned about, and they will be revised as new
quarters arrive, which is one more reason to read the endpoint bands in
{ref}`csdv2_persistence` with care.

```{solution-end}
```

```{exercise}
:label: csdv2_ex2

A Kalman filter on coefficients that drift as random walks behaves like
constant-gain least squares, and this exercise asks how large the implied gain
is.

1. For the local-level model $y_t=\mu_t+w_t$, $\mu_t=\mu_{t-1}+v_t$, with
   $\operatorname{var}(w_t)=1$ and $\operatorname{var}(v_t)=q$, show that the
   steady-state predicted variance is $p=(q+\sqrt{q^2+4q})/2$ and the
   steady-state Kalman gain is $K=p/(1+p)\approx\sqrt q$ for small $q$.
   Check this numerically by iterating the Riccati equation for
   $q\in\{10^{-4},10^{-3},10^{-2}\}$.
2. Run the Kalman filter of {ref}`csdv-appendix` for the 21 coefficients with
   $Q$ and $R_t$ fixed at their posterior means.
   At each date compute $w_t$, the weight that the updated fitted value of
   inflation puts on the newest inflation forecast error, which is the
   $(3,3)$ element of $X_t^\top K_t$.
3. Under constant-gain least squares, that weight is the gain times
   $x_t^\top R^{-1}x_t$, where $R$ is the regressors' second-moment matrix, and
   $x_t^\top R^{-1}x_t$ averages $k$, the number of regressors.
   Convert $w_t$ into an equivalent gain $w_t/7$ and compare its average by
   decade with the gains used in this section: 0.025 in {doc}`phillips_learning`,
   0.02 in {ref}`olg_phillips`, and 0.03 in {doc}`phillips_lost_conquest`.
```

```{solution-start} csdv2_ex2
:class: dropdown
```

In steady state the predicted variance satisfies
$p = p - p^2/(p+1) + q$, so $p^2 - qp - q = 0$ and
$p=(q+\sqrt{q^2+4q})/2\approx\sqrt q$, which gives $K=p/(1+p)\approx\sqrt q$.

```{code-cell} ipython3
for q in (1e-4, 1e-3, 1e-2):
    p = 1.0
    for _ in range(10_000):
        p = p - p**2 / (p + 1) + q
    formula = (q + np.sqrt(q**2 + 4 * q)) / 2
    print(f'q = {q:g}: iterated gain {p / (1 + p):.4f}, '
          f'formula {formula / (1 + formula):.4f}, sqrt(q) {np.sqrt(q):.4f}')
```

For the VAR we run the forward recursion with the posterior means of $Q$ and
$R_t$ and record the weight on the newest inflation surprise.

```{code-cell} ipython3
q_bar = posterior['Q'].mean(axis=0)
X = design_matrices(data['x'])
P = prior['θ_covariance'].copy()
weight = np.empty(len(dates))
for t in range(len(dates)):
    predicted = P + q_bar
    px = predicted @ X[t].T
    gain = np.linalg.solve(X[t] @ px + r_mean[t], px.T).T
    weight[t] = (X[t] @ gain)[2, 2]
    P = predicted - gain @ px.T

decades = {f'{d}s': (d, d + 9) for d in (1960, 1970, 1980, 1990, 2000, 2010)}
decades['2020 onward'] = (2020, 2100)
pd.DataFrame({
    'weight on newest surprise': [period_mean(weight, *p) for p in decades.values()],
    'equivalent gain per regressor': [period_mean(weight, *p) / 7
                                      for p in decades.values()],
}, index=decades.keys()).round(3)
```

The steady-state gain is close to $\sqrt q$, as the formula says.

For the VAR, the weight on the newest inflation surprise is largest before 2000,
when the equivalent gain per regressor is roughly two to five times the 0.025
of {doc}`phillips_learning`; after 2000 it falls to between about one and two
times that value, close to the gains of 0.02 to 0.03 used in this section.

These gains inherit the uncertainty about $Q$ noted above, because the gain
grows with the drift covariance.

```{solution-end}
```

```{exercise}
:label: csdv2_ex3

{doc}`sargent_surico` interprets a long-run slope as a ratio of spectral
densities at frequency zero, $h(0)=S_{yz}(0)/S_z(0)$.

Compute the analogous low-frequency slope of inflation on unemployment implied
by the local VAR at each date,

$$
h_{\pi u}(0,t)=\frac{\operatorname{Re} f_{\pi u}(0,t)}{f_{uu}(0,t)},
$$

draw by draw, and convert it into percentage points of annual inflation per
percentage point of the unemployment rate, using the natural rate $\bar u_t$
to undo the logit transformation.

Plot the posterior median and a 90 percent band.

When was the low-frequency slope negative, as a long-run Phillips tradeoff would
require?
```

```{solution-start} csdv2_ex3
:class: dropdown
```

At frequency zero the companion system gives
$2\pi f(0,t)=(I-A_t)^{-1}\mathcal R_t(I-A_t)^{-\top}$.

Inflation is a quarterly log change, so it is multiplied by 400; the logit
satisfies $du_t = 10^{-4}\,dU/[\bar u_t(1-\bar u_t)]$ when $dU$ is measured in
percentage points.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Low-frequency slope $h_{\pi u}(0,t)$ of inflation on unemployment
    name: fig-csdv2-low-frequency-slope
---
def low_frequency_slope(θ, covariance):
    """Low-frequency slope of annual inflation on the unemployment rate (pp/pp)."""
    intercept, companion = companion_matrix(θ)
    inverse = np.linalg.inv(np.eye(6) - companion)
    innovation = np.zeros((6, 6))
    innovation[:3, :3] = covariance
    S = inverse @ innovation @ inverse.T
    logit_u = 100 * (inverse @ intercept)[1]
    density = expit(logit_u) * expit(-logit_u)      # u(1 - u) at the natural rate
    if density == 0:                                # natural rate at 0 or 100%
        return np.nan
    return 400 * S[2, 1] / S[1, 1] * 1e-4 / density


slopes = np.array([
    [low_frequency_slope(posterior['θ'][d][:, t],
                         innovation_covariance(posterior['h'][d][t],
                                               posterior['β'][d]))
     for t in annual]
    for d in range(len(posterior['β']))
])
lower, median, upper = np.nanquantile(slopes, (0.05, 0.5, 0.95), axis=0)

fig, ax = plt.subplots()
ax.plot(dates[annual], median, lw=2, label='posterior median')
ax.fill_between(dates[annual], lower, upper, alpha=0.2, label='90 percent band')
ax.axhline(0, color='0.45', lw=1)
ax.axvline(2000.75, color='0.6', ls='--', lw=1)
ax.set_xlabel('year')
ax.set_ylabel(r'$h_{\pi u}(0,t)$, percent per percentage point')
ax.set_ylim(-3, 4)     # a few draws near the unit-root boundary explode
ax.legend()
plt.show()
```

The table below gives the numbers behind the figure.

```{code-cell} ipython3
pd.DataFrame({'5%': lower, 'median': median, '95%': upper},
             index=annual_years).loc[[1962, 1970, 1975, 1980, 1985, 1990, 2000,
                                      2010, 2020, annual_years[-1]]].round(2)
```

The low-frequency slope is positive in the 1970s, with bands that exclude zero
in the second half of the decade: at low frequencies, inflation and unemployment
rose together, the opposite of an exploitable long-run tradeoff.

It is negative in the early 1960s, and close to zero and imprecise after 2000.

The vertical axis is cut off because in a few years some draws lie near the
unit-root boundary, where the slope explodes.

These are reduced-form statistics: as {cite:t}`Sargent1971` and
{cite:t}`Lucas1972` argued, a low-frequency slope is not by itself a structural
tradeoff that policy could exploit.

```{solution-end}
```
