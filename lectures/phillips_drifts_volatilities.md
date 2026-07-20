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
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Drifts and Volatilities

```{contents} Contents
:depth: 2
```

In addition to what's in Anaconda, this lecture will use the following library to download data from FRED:

```{code-cell} ipython3
:tags: [hide-output]

!pip install pandas_datareader
```

## Overview

The lectures in this section have told a story about how a government's *model* of the Phillips curve, and the *policy* it induces, can drift over time.

In {doc}`phillips_learning` and {doc}`phillips_escaping_nash` a government that fits and refits an approximating Phillips curve is repeatedly pushed away from a bad {doc}`self-confirming equilibrium <phillips_self_confirming>` along an *escape route*, and in {doc}`phillips_priors` and {doc}`phillips_lost_conquest` we watched drifting beliefs rationalize the rise and fall — and the recent return — of American inflation.

Those lectures were mostly about *theory*.

This lecture turns to the *data*.

It studies {cite}`CogleySargent2005`, "Drifts and Volatilities: Monetary Policies and Outcomes in the Post WWII US", which asks a deceptively simple question:

> When we look at postwar U.S. time series on inflation, unemployment, and interest rates, do we see evidence that the dynamics have *drifted*?

Tim Cogley and Thomas Sargent began this work as an empirical companion to the *Conquest* book {cite}`Sargent1999` and the escape-route papers {cite}`ChoWilliamsSargent2002`.

It is also a response to searching comments by Christopher Sims {cite}`Sims2001comment` and James Stock {cite}`Stock2001comment` on an earlier paper {cite}`CogleySargent2001`, and it grew into a friendly debate with Sims and Tao Zha {cite}`SimsZha2006` and with Ben Bernanke and Ilian Mihov {cite}`BernankeMihov1998` about a question that organizes this whole section:

**Was the Great Inflation of the 1970s and its conquest in the 1980s a story of bad *policy*, or of bad *luck*?**

To let the data speak to that question we need a statistical model flexible enough to accommodate *both* answers.

That model is a **Bayesian vector autoregression (VAR) whose coefficients drift as random walks and whose shock variances themselves evolve as stochastic volatilities.**

Fitting it requires a Markov chain Monte Carlo (MCMC) algorithm that is a beautiful application of the {doc}`Kalman filter <kalman>` — specifically the Carter–Kohn {cite}`CarterKohn1994` forward-filter/backward-sample smoother — combined with the stochastic-volatility sampler of Jacquier, Polson, and Rossi {cite}`Jacquier1994`.

Let's start with some imports.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from numpy.linalg import inv, cholesky, eigvals, solve
from scipy.stats import invwishart
```

## Bad policy or bad luck?

Two respectable views compete to explain the American Great Inflation.

The **bad policy** view is the one dramatized throughout this section and in the *Conquest* book {cite}`Sargent1999`.

Something about Arthur Burns's *model* of the economy, his *patience*, or his inability to *commit* to a better rule led the Federal Reserve to administer monetary policy in a way that produced the greatest peacetime inflation in U.S. history; and an improved model, more patience, or greater discipline led Paul Volcker to conquer it (see {cite}`DeLong1997` and {cite}`Taylor1997comment`).

On this view, what changed between the 1970s and the 1980s was the *systematic part* of policy — the way the Fed's interest-rate setting responded to inflation and unemployment.

The **bad luck** view says something quite different.

What distinguished the Burns and Volcker eras was not their models or policies, but the *shocks* that hit the economy.

On this view the *coefficients* of a reduced-form description of the economy were essentially constant, and what changed was the *size* of the disturbances — the *volatility*.

Bernanke and Mihov {cite}`BernankeMihov1998` and Sims and Zha {cite}`SimsZha2006` marshaled evidence for this second view, in part by applying classical tests that *failed to reject* the hypothesis that VAR coefficients were time-invariant.

How can we discriminate?

A model with constant coefficients and constant volatility cannot represent the bad-luck story; a model with drifting coefficients but constant volatility (as in {cite}`CogleySargent2001`) risks *attributing to drift what is really changing volatility* — exactly the criticism Sims and Stock leveled.

So Cogley and Sargent build a model that has room for **both** channels at once, and they let a Bayesian posterior sort out how much of each the data call for.

## A VAR with drifting coefficients and stochastic volatility

Collect the three variables of interest in a vector

$$
y_t = \begin{bmatrix} i_t \\ u_t \\ \pi_t\end{bmatrix},
$$

the nominal interest rate, (a transform of) the unemployment rate, and inflation.

The **measurement equation** is a VAR with two lags whose coefficients carry a time subscript,

```{math}
:label: csdv_meas
y_t = X_t' \theta_t + \varepsilon_t ,
\qquad X_t' = I_3 \otimes [\,1,\ y_{t-1}',\ y_{t-2}'\,],
```

so that $\theta_t$ stacks the $21 = 3\times 7$ intercepts and autoregressive coefficients that prevail at date $t$.

The coefficients follow a **driftless random walk**,

```{math}
:label: csdv_trans
\theta_t = \theta_{t-1} + v_t, \qquad v_t \sim N(0, Q),
```

subject to a *stability* (reflecting-barrier) restriction: paths of $\theta_t$ that would make the implied VAR explosive are ruled out a priori.

Writing $I(\theta^T)$ for the indicator that the *entire* trajectory $\theta^T = \{\theta_t\}_{t=1}^T$ stays in the nonexplosive region, the prior is

$$
p(\theta^T \mid Q) \propto I(\theta^T)\, f(\theta^T \mid Q),
$$

the random-walk prior *truncated* to stable paths. This restriction encodes our belief that the economy did not in fact explode.

The **innovations** are conditionally Gaussian with a time-varying covariance matrix,

```{math}
:label: csdv_R
\varepsilon_t = R_t^{1/2}\,\xi_t, \qquad \xi_t \sim N(0, I_3),
\qquad R_t = B^{-1} H_t B^{-1\prime}.
```

Here $B$ is a fixed lower-triangular matrix with ones on the diagonal,

$$
B = \begin{bmatrix} 1 & 0 & 0 \\ \beta_{21} & 1 & 0 \\ \beta_{31} & \beta_{32} & 1\end{bmatrix},
$$

and $H_t = \operatorname{diag}(h_{1t}, h_{2t}, h_{3t})$ collects three **stochastic volatilities**, each of which evolves as a geometric random walk,

```{math}
:label: csdv_vol
\ln h_{it} = \ln h_{i,t-1} + \sigma_i\, \eta_{it}, \qquad \eta_{it}\sim N(0,1).
```

The matrix $B$ orthogonalizes the innovations; the diagonal elements $h_{it}$ let the size of each orthogonalized shock wax and wane over time.

The model nests both stories.

Set $Q = 0$ and the coefficients are constant — the *bad luck* model (only volatilities move).

Freeze $H_t$ and the volatilities are constant — a pure *drifting-coefficients* model.

The unknowns to be inferred are the whole history of coefficients $\theta^T$, the whole history of volatilities $H^T$, and the hyperparameters $Q$, $\sigma = (\sigma_1,\sigma_2,\sigma_3)$, and $\beta = (\beta_{21},\beta_{31},\beta_{32})$.

The posterior $p(\theta^T, H^T, Q, \sigma, \beta \mid y^T)$ has thousands of dimensions, which is why we need MCMC.

## The data

Following the paper, we use quarterly U.S. data.

* inflation $\pi_t$ is the log-difference of the consumer price index, point-sampled in the third month of each quarter;
* unemployment enters as the *logit* of the civilian unemployment rate (a transform that keeps a rate bounded in $(0,1)$ from wandering outside it), averaged over the quarter;
* the interest rate $i_t$ is the three-month Treasury bill rate in the first month of the quarter, expressed at a quarterly rate.

We download the raw series from [FRED](https://fred.stlouisfed.org/) and build the three transformed series.

The paper's sample runs through 2000; we extend it through 2019, adding the Great Moderation, the 2008 financial crisis, and the recovery that followed.

We stop before the COVID-19 pandemic, whose enormous 2020 swings in measured unemployment break a linear-Gaussian VAR (we return to the pandemic inflation in {doc}`phillips_lost_conquest`).

```{code-cell} ipython3
start, end = '1947-01-01', '2019-12-31'
cpi = web.DataReader('CPIAUCSL', 'fred', start, end)['CPIAUCSL']
ur  = web.DataReader('UNRATE',   'fred', start, end)['UNRATE']
tb  = web.DataReader('TB3MS',    'fred', start, end)['TB3MS']

# inflation: log-change of CPI point-sampled in the third month of each quarter
cpi_q = cpi[cpi.index.month.isin([3, 6, 9, 12])]
infl = np.log(cpi_q).diff().dropna()
infl.index = infl.index.to_period('Q')

# unemployment: quarterly average, as a fraction, then logit
u_q = (ur / 100).resample('QE').mean()
u_q.index = u_q.index.to_period('Q')
logit_u = np.log(u_q / (1 - u_q))

# interest: 3-month T-bill, first month of quarter, at a quarterly rate
i_q = tb[tb.index.month.isin([1, 4, 7, 10])] / 100 / 4
i_q.index = i_q.index.to_period('Q')

data = pd.concat({'i': i_q, 'u': logit_u, 'pi': infl}, axis=1).dropna().loc['1948Q1':]
data.tail()
```

Let us look at the three series, with inflation and the interest rate shown at annualized percentage rates and unemployment untransformed back to a percentage.

```{code-cell} ipython3
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
tt = data.index.to_timestamp()
axes[0].plot(tt, data['pi'] * 400)
axes[0].set_ylabel('inflation (ann. %)')
axes[1].plot(tt, 100 / (1 + np.exp(-data['u'])))
axes[1].set_ylabel('unemployment (%)')
axes[2].plot(tt, data['i'] * 400)
axes[2].set_ylabel('interest (ann. %)')
axes[2].set_xlabel('year')
plt.tight_layout()
plt.show()
```

The rise of inflation and interest rates through the 1970s, the Volcker disinflation of the early 1980s, and the long calm that followed are all plainly visible.

The question is what a flexible statistical model makes of them.

## Setting up the state space

We now build the pieces the sampler needs.

Because the three series live on very different numerical scales (the logit of unemployment is around $-3$, while quarterly inflation is around $0.01$), we standardize each series by its sample standard deviation before estimation.

This is a standard preconditioning in time-varying VAR work {cite}`Primiceri2005`: it puts the random-walk innovations $v_t$ on comparable scales and keeps the regressor cross-product matrix well conditioned. We undo the scaling when we report results in economic units.

```{code-cell} ipython3
n, p = 3, 2               # variables, lags
k = 1 + n * p             # 7 regressors per equation
d = n * k                 # 21 drifting VAR coefficients

scale = data.values.std(0)
Yz = data.values / scale  # standardized data

def companion(theta):
    "Companion matrix of the VAR(2) implied by a coefficient vector theta."
    C = theta.reshape(n, k)
    F = np.zeros((n * p, n * p))
    F[:n, :n] = C[:, 1:1+n]
    F[:n, n:2*n] = C[:, 1+n:1+2*n]
    F[n:, :n] = np.eye(n)
    return F

# stack regressors x_t = [1, y_{t-1}, y_{t-2}] and outcomes y_t
rows_x, rows_y = [], []
for t in range(p, len(Yz)):
    rows_x.append(np.concatenate([[1.0], Yz[t-1], Yz[t-2]]))
    rows_y.append(Yz[t])
X_all, Y_all = np.array(rows_x), np.array(rows_y)
dates_all = data.index[p:]

# training sample (through 1958) calibrates the priors; estimate on 1959-2019
train = dates_all <= pd.Period('1958Q4', 'Q')
est = (dates_all >= pd.Period('1959Q1', 'Q')) & (dates_all <= pd.Period('2019Q4', 'Q'))
Xtr, Ytr = X_all[train], Y_all[train]
X, Y = X_all[est], Y_all[est]
dates = dates_all[est]
T = len(X)
print(f'{T} quarters, {dates[0]} to {dates[-1]}')
```

## Priors

The paper's priors are deliberately weak, chosen so that "the data are free to speak."

They are calibrated from a time-invariant VAR fit to the short 1948–1958 training sample.

Let $\hat\theta$ and $\hat P$ be the OLS point estimate and its covariance from that regression, and let $\Sigma_0$ be the OLS residual covariance.

* The initial coefficient vector has a Gaussian prior $\theta_0 \sim N(\hat\theta, \hat P)$, truncated to stable values.
* The drift covariance has an inverse-Wishart prior $Q \sim IW(\bar Q, \tau)$. We center it on a small multiple of $\hat P$ so that a *priori* the coefficients drift only slowly — the "business as usual" prior of {cite}`Primiceri2005`. The scalar $k_Q^2$ controls how much drift the prior permits.
* The volatility innovation standard deviations have an inverse-gamma prior $\sigma_i^2 \sim IG(0.5, 5\times 10^{-5})$, and the covariance parameters $\beta$ a diffuse Gaussian prior.

```{code-cell} ipython3
XtX = Xtr.T @ Xtr
B_ols = solve(XtX, Xtr.T @ Ytr)
Sigma0 = (Ytr - Xtr @ B_ols).T @ (Ytr - Xtr @ B_ols) / len(Xtr)

theta0 = B_ols.flatten(order='F')       # prior mean for the coefficients
P0 = np.kron(Sigma0, inv(XtX))          # prior covariance

tau = len(Xtr)
kQ = 0.01                               # controls the prior amount of drift
Q_scale = kQ**2 * tau * P0              # inverse-Wishart scale matrix
Q_df = tau                              # inverse-Wishart degrees of freedom

# initial volatilities from the training-sample orthogonalized residuals
B_init = inv(cholesky(Sigma0))
B_init /= np.diag(B_init)[:, None]
lnh0 = np.log(np.diag(B_init @ Sigma0 @ B_init.T))
lnh0_var = 10.0

# precompute the measurement matrices Z_t = I_3 ⊗ x_t'
Z = [np.kron(np.eye(n), X[t][None, :]) for t in range(T)]
```

## A Metropolis-within-Gibbs sampler

We simulate the posterior by cycling through five blocks, drawing each group of unknowns conditional on the others.

This is the algorithm in Appendix B of {cite}`CogleySargent2005`.

1. **Coefficients $\theta^T$**, given the volatilities and hyperparameters.
Conditional on the sequence of covariance matrices $R_t$, {eq}`csdv_meas`–{eq}`csdv_trans` is a linear Gaussian state-space model, and the whole coefficient path can be drawn in one sweep by the **Carter–Kohn** {cite}`CarterKohn1994` forward-filter/backward-sample algorithm — a stochastic version of the {doc}`Kalman smoother <kalman>`. We *reject* any proposed path that violates the stability restriction, keeping the previous draw instead.
2. **Drift covariance $Q$**, an inverse-Wishart draw given the coefficient innovations $v_t = \theta_t - \theta_{t-1}$.
3. **Covariance parameters $\beta$**, drawn from a pair of transformed regressions.
4. **Volatilities $H^T$**, drawn one date at a time by the **Jacquier–Polson–Rossi** {cite}`Jacquier1994` Metropolis step for stochastic-volatility models.
5. **Volatility innovation variances $\sigma_i^2$**, an inverse-gamma draw given the volatility path.

We first write a helper to draw from a multivariate normal via its Cholesky factor.

```{code-cell} ipython3
def draw_mvn(mean, cov):
    cov = 0.5 * (cov + cov.T) + 1e-12 * np.eye(len(cov))
    return mean + cholesky(cov) @ rng.standard_normal(len(mean))
```

### Block 1: the coefficient path

The forward pass is the Kalman filter run through the sample; the backward pass samples $\theta_T, \theta_{T-1}, \dots, \theta_0$ in reverse, each conditioned on the draw that follows it.

If the sampled path is stable everywhere we accept it; otherwise we keep the previous path — this is how the reflecting-barrier prior is imposed.

```{code-cell} ipython3
def draw_theta(Rseq, Q, theta_prev):
    # forward pass: Kalman filter with time-varying measurement covariance R_t
    tf = np.zeros((T + 1, d))
    Pf = np.zeros((T + 1, d, d))
    tf[0], Pf[0] = theta0, P0
    for t in range(1, T + 1):
        Ppred = Pf[t-1] + Q
        S = Z[t-1] @ Ppred @ Z[t-1].T + Rseq[t-1]
        K = Ppred @ Z[t-1].T @ inv(S)
        tf[t] = tf[t-1] + K @ (Y[t-1] - Z[t-1] @ tf[t-1])
        Pf[t] = Ppred - K @ Z[t-1] @ Ppred
    # backward pass: sample the path from T back to 0
    Th = np.zeros((T + 1, d))
    Th[T] = draw_mvn(tf[T], Pf[T])
    for t in range(T - 1, -1, -1):
        J = Pf[t] @ inv(Pf[t] + Q)
        Th[t] = draw_mvn(tf[t] + J @ (Th[t+1] - tf[t]), Pf[t] - J @ Pf[t])
    # impose the stability (reflecting-barrier) prior
    stable = all(np.max(np.abs(eigvals(companion(Th[t])))) < 1 for t in range(1, T + 1))
    return (Th, True) if stable else (theta_prev, False)
```

### Blocks 3 and 4: covariance parameters and volatilities

Given a coefficient path we can form the VAR residuals $\varepsilon_t$.

The parameters $\beta$ that build the orthogonalizing matrix $B$ come from two "seemingly unrelated" regressions among the residuals, and the volatilities are then updated one date at a time.

```{code-cell} ipython3
def Bmatrix(beta):
    B = np.eye(n)
    B[1, 0], B[2, 0], B[2, 1] = beta
    return B

def draw_beta(eps, h):
    prior = 1 / 10000.0
    # equation 2: regress h2^{-1/2} eps2 on -h2^{-1/2} eps1
    z2 = eps[:, 1] / np.sqrt(h[:, 1])
    x2 = (-eps[:, 0] / np.sqrt(h[:, 1]))[:, None]
    V = 1 / (prior + (x2.T @ x2)[0, 0])
    b21 = V * (x2.T @ z2)[0] + np.sqrt(V) * rng.standard_normal()
    # equation 3: regress h3^{-1/2} eps3 on -h3^{-1/2}(eps1, eps2)
    z3 = eps[:, 2] / np.sqrt(h[:, 2])
    x3 = np.column_stack([-eps[:, 0] / np.sqrt(h[:, 2]),
                          -eps[:, 1] / np.sqrt(h[:, 2])])
    V3 = inv(prior * np.eye(2) + x3.T @ x3)
    b3 = draw_mvn(V3 @ (x3.T @ z3), V3)
    return np.array([b21, b3[0], b3[1]])
```

The Jacquier–Polson–Rossi step draws each $\ln h_{it}$ from a proposal centered on its neighbors in the random walk and accepts or rejects it by comparing how well it fits the orthogonalized residual $u_{it}$.

```{code-cell} ipython3
def draw_vol(lnh, u, sig2):
    for i in range(n):
        s2 = sig2[i]
        # initial log-volatility from a Gaussian full conditional
        v0 = 1 / (1 / lnh0_var + 1 / s2)
        lnh[0, i] = v0 * (lnh0[i] / lnh0_var + lnh[1, i] / s2) \
            + np.sqrt(v0) * rng.standard_normal()
        for t in range(1, T + 1):
            if t < T:
                mu, sc2 = 0.5 * (lnh[t+1, i] + lnh[t-1, i]), 0.5 * s2
            else:
                mu, sc2 = lnh[t-1, i], s2
            old = lnh[t, i]
            new = mu + np.sqrt(sc2) * rng.standard_normal()
            log_acc = -0.5 * (new - old) \
                - 0.5 * u[t-1, i]**2 * (np.exp(-new) - np.exp(-old))
            if np.log(rng.random()) < log_acc:
                lnh[t, i] = new
    return lnh
```

### The main loop

The loop ties the blocks together.

Each sweep produces one draw from the posterior; after a burn-in we keep every few draws to reduce autocorrelation and storage.

```{code-cell} ipython3
def run(ndraw, nburn, thin, seed=2024):
    global rng
    rng = np.random.default_rng(seed)
    Th = np.tile(theta0, (T + 1, 1)).astype(float)
    Q = Q_scale / (Q_df - d - 1)
    beta = np.zeros(3)
    sig2 = np.full(3, 0.05**2)
    lnh = np.tile(lnh0, (T + 1, 1)).astype(float)
    keep_th, keep_R = [], []
    accept = 0
    for it in range(ndraw):
        # build the covariance sequence R_t = B^{-1} H_t B^{-1'}
        Binv = inv(Bmatrix(beta))
        h = np.exp(lnh[1:])
        Rseq = np.array([Binv @ np.diag(h[t]) @ Binv.T for t in range(T)])
        # 1) coefficient path
        Th, ok = draw_theta(Rseq, Q, Th)
        accept += ok
        # 2) drift covariance Q
        v = np.diff(Th, axis=0)
        Q = invwishart.rvs(df=Q_df + T, scale=Q_scale + v.T @ v, random_state=rng)
        # residuals, then 3) beta and 4) volatilities
        eps = np.array([Y[t] - Z[t] @ Th[t+1] for t in range(T)])
        beta = draw_beta(eps, h)
        u = (Bmatrix(beta) @ eps.T).T
        lnh = draw_vol(lnh, u, sig2)
        # 5) volatility innovation variances
        for i in range(n):
            sig2[i] = (1e-4 + np.sum(np.diff(lnh[:, i])**2)) / 2 \
                / rng.gamma((1 + T) / 2, 1)
        # store
        if it >= nburn and (it - nburn) % thin == 0:
            keep_th.append(Th[1:].copy())
            Binv = inv(Bmatrix(beta))
            h = np.exp(lnh[1:])
            keep_R.append(np.array([Binv @ np.diag(h[t]) @ Binv.T
                                    for t in range(T)]))
    print(f'acceptance rate for coefficient paths: {accept / ndraw:.2f}')
    return np.array(keep_th), np.array(keep_R)
```

```{note}
The original paper ran 100,000 sweeps and kept every tenth after a 50,000-draw burn-in.
We use a much shorter run so the lecture builds in a few minutes.
The posterior *means* we report below stabilize quickly; the price of the short run is noisier tails, and — because inflation is so persistent that drifting paths frequently graze the stability barrier — a coefficient-path acceptance rate well below one.
```

Now we run the sampler.

```{code-cell} ipython3
theta_draws, R_draws = run(ndraw=3000, nburn=1000, thin=4)
theta_draws.shape
```

## What the data say

We summarize the posterior by its mean coefficient path $\bar\theta_t$ and mean covariance path $\bar R_t$, and read the economically interesting objects off them — exactly as in the paper, which evaluates its measures at the posterior mean.

```{code-cell} ipython3
thb = theta_draws.mean(0)
Rb = R_draws.mean(0)
S = np.diag(scale)
tt = dates.to_timestamp()
```

### The evolution of volatility

First we ask how the *size* of the shocks changed.

For each quarter we translate the orthogonalized volatilities back into the standard deviations of the reduced-form innovations to inflation and the interest rate (in original units), and we compute the log determinant of $R_t$ — a scalar measure of the total one-step-ahead forecast uncertainty entering the system, following {cite}`Whittle1953`.

```{code-cell} ipython3
sig_i = np.zeros(T)
sig_pi = np.zeros(T)
logdetR = np.zeros(T)
for t in range(T):
    Ro = S @ Rb[t] @ S                       # covariance in original units
    sig_i[t] = np.sqrt(Ro[0, 0]) * 400       # annualized %
    sig_pi[t] = np.sqrt(Ro[2, 2]) * 400
    logdetR[t] = np.log(np.linalg.det(Ro))

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
axes[0].plot(tt, sig_i)
axes[0].set_title('interest innovation std (ann. %)')
axes[1].plot(tt, sig_pi)
axes[1].set_title('inflation innovation std (ann. %)')
axes[2].plot(tt, logdetR)
axes[2].set_title('total prediction variance $\\log|R_t|$')
for ax in axes:
    ax.set_xlabel('year')
plt.tight_layout()
plt.show()
```

The stochastic volatilities are far from constant.

The standard deviation of the interest-rate innovation *spikes* dramatically around 1979–1982, the years of the Volcker experiment with nonborrowed-reserves targeting — the single most visible feature of the whole sample.

The inflation innovation swells through the 1970s and peaks near 1980, and again during the 2008 financial crisis.

The total-prediction-variance measure $\log|R_t|$ rises in two steps into the early 1980s and then falls substantially — the "Great Moderation" documented by {cite}`KimNelson1999` and {cite}`McConnellPerezQuiros2000`.

**There is emphatically evidence for the bad-luck story: the volatilities move a lot.**

But that is not the end of the matter.

### Inflation persistence

The question that most interests Cogley and Sargent is whether, *on top of* the moving volatilities, the *systematic* dynamics also drifted.

Their favorite summary of the systematic dynamics is the **persistence** of inflation, measured by the normalized spectrum of inflation at frequency zero.

Write the drifting VAR in companion form with autoregressive matrix $A_t = A_{1t}+A_{2t}$ (the sum of the two lag matrices) and innovation covariance $R_t$.

The spectral density of inflation at frequency zero is the $(\pi,\pi)$ element of

$$
f(0) = \frac{1}{2\pi}(I - A_t)^{-1} R_t (I - A_t)^{-1\prime},
$$

and dividing it by the unconditional variance of inflation gives a scale-free number $g_{\pi\pi}(0,t)$ that behaves like the persistence of a univariate autoregression.

(For an $AR(1)$ with coefficient $\rho$, $g_{\pi\pi}(0) = (1+\rho)/[2\pi(1-\rho)]$; values of 2 to 10 correspond to $\rho$ between 0.85 and 0.97.)

```{code-cell} ipython3
def unconditional_var(F, R):
    "Solve the discrete Lyapunov equation for the companion-form variance."
    m = F.shape[0]
    Sc = np.zeros((m, m))
    Sc[:n, :n] = R
    return (inv(np.eye(m * m) - np.kron(F, F)) @ Sc.flatten()).reshape(m, m)

persist = np.zeros(T)
for t in range(T):
    C = thb[t].reshape(n, k)
    Asum = C[:, 1:1+n] + C[:, 1+n:1+2*n]
    M = inv(np.eye(n) - Asum)
    f0 = (M @ Rb[t] @ M.T) / (2 * np.pi)
    G0 = unconditional_var(companion(thb[t]), Rb[t])
    persist[t] = f0[2, 2] / G0[2, 2]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(tt, persist)
ax.set_title('inflation persistence: normalized spectrum at zero $g_{\\pi\\pi}(0,t)$')
ax.set_xlabel('year')
plt.tight_layout()
plt.show()
```

This is the paper's central finding.

Inflation persistence was low and roughly flat in the early 1960s, **rose sharply through the late 1960s and 1970s** to a peak around 1980, and then **fell just as sharply during and after the Volcker disinflation**, settling at a low level for the Great Moderation decades.

The evidence says the *systematic dynamics drifted too* — not just the volatilities.

So the answer to "bad policy or bad luck" is: the data want **both**.

### Core inflation

A closely related object is *core inflation* — the long-horizon forecast of inflation implied by the drifting VAR, $\bar\pi_t = s_\pi (I - A_t)^{-1} \mu_t$, where $\mu_t$ collects the drifting intercepts.

```{code-cell} ipython3
core = np.zeros(T)
for t in range(T):
    C = thb[t].reshape(n, k)
    mu = C[:, 0]
    Asum = C[:, 1:1+n] + C[:, 1+n:1+2*n]
    core[t] = (scale * (inv(np.eye(n) - Asum) @ mu))[2] * 400

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(tt, core)
ax.set_title('core inflation (annualized %)')
ax.set_xlabel('year')
plt.tight_layout()
plt.show()
```

Core inflation traces out the familiar hump — rising into the early 1970s and declining through the 1980s and 1990s — and it moves *together* with the persistence measure, just as in the paper.

```{admonition} A caveat about long-run means
:class: warning
The *level* of core inflation is a fragile object.
It is the long-run mean of a system whose largest root sits very close to one, so $(I - A_t)^{-1}$ is an enormous amplifier and small movements across equations can nearly cancel.
Our short posterior run recovers the *shape* of core inflation reliably but compresses its *amplitude* relative to the paper's long run.
The persistence and volatility measures above are far more robust — a useful reminder that not every function of a fitted model is estimated equally well.
```

## How powerful are tests of coefficient stability?

We have found drift.

Yet Bernanke and Mihov {cite}`BernankeMihov1998`, applying classical tests of parameter constancy due to Andrews {cite}`Andrews1993` and Nyblom {cite}`Nyblom1989`, *could not reject* the hypothesis that VAR coefficients were time-invariant, and read that failure as evidence for the bad-luck view.

Cogley and Sargent's rejoinder is a *power* calculation.

A failure to reject is evidence against an alternative only if the test has decent power against it.

So they ask: if the data were *actually generated* by a drifting-coefficient VAR like the one we just fit, how often would a standard stability test detect the drift?

We reproduce that experiment for a sup-Wald test of a single break at an unknown date (a version of {cite}`Andrews1993`), applied to the inflation equation.

We

1. fit a *constant*-coefficient VAR to the data to serve as the null data-generating process, and calibrate the test's 5% critical value by simulating from it;
2. simulate many samples from our *drifting* VAR and count how often the test rejects constancy.

```{code-cell} ipython3
# initial conditions: the two data lags at the start of the estimation sample
j0 = list(data.index[p:]).index(pd.Period('1959Q1', 'Q'))
y_init = [Yz[j0 - 2], Yz[j0 - 1]]

def simulate_drift():
    y = list(y_init)
    out = []
    for t in range(T):
        C = thb[t].reshape(n, k)
        L = cholesky(Rb[t] + 1e-12 * np.eye(n))
        yt = C[:, 0] + C[:, 1:1+n] @ y[-1] + C[:, 1+n:1+2*n] @ y[-2] \
            + L @ rng.standard_normal(n)
        out.append(yt)
        y.append(yt)
    return np.array(out)

# constant-coefficient null, fit to the estimation sample
Xc = np.column_stack([np.ones(T - 2), Y[1:-1], Y[:-2]])
Bc = solve(Xc.T @ Xc, Xc.T @ Y[2:])
Sc = cholesky((Y[2:] - Xc @ Bc).T @ (Y[2:] - Xc @ Bc) / (T - 2) + 1e-12 * np.eye(n))

def simulate_const():
    y = list(y_init)
    out = []
    for t in range(T):
        yt = Bc[0] + Bc[1:1+n].T @ y[-1] + Bc[1+n:1+2*n].T @ y[-2] \
            + Sc @ rng.standard_normal(n)
        out.append(yt)
        y.append(yt)
    return np.array(out)

def sup_wald(Yd, eq=2):
    "sup-Wald statistic for a break in the coefficients of one equation."
    Xr = np.column_stack([np.ones(T - 2), Yd[1:-1], Yd[:-2]])
    yv = Yd[2:, eq]
    N = len(yv)
    stats = []
    for kk in range(int(0.15 * N), int(0.85 * N)):
        X1, X2 = Xr[:kk], Xr[kk:]
        y1, y2 = yv[:kk], yv[kk:]
        b1 = solve(X1.T @ X1, X1.T @ y1)
        b2 = solve(X2.T @ X2, X2.T @ y2)
        s2 = ((y1 - X1 @ b1) @ (y1 - X1 @ b1)
              + (y2 - X2 @ b2) @ (y2 - X2 @ b2)) / (N - 2 * k)
        db = b1 - b2
        stats.append(db @ inv(s2 * (inv(X1.T @ X1) + inv(X2.T @ X2))) @ db)
    return max(stats)
```

```{code-cell} ipython3
rng = np.random.default_rng(0)
n_mc = 300
null = np.array([sup_wald(simulate_const()) for _ in range(n_mc)])
crit = np.percentile(null, 95)
power = np.mean([sup_wald(simulate_drift()) > crit for _ in range(n_mc)])
print(f'5% critical value: {crit:.1f}')
print(f'power against the drifting VAR: {power:.2f}')
```

The power is low — the test detects the drift only a fraction of the time, even though the drift is really there.

This reproduces the paper's striking conclusion (its Tables 4–8): the classical stability tests that the bad-luck camp relied on have **little power against smoothly drifting alternatives** of exactly the kind our Bayesian model finds.

A failure to reject constancy is therefore *not* evidence that the coefficients are constant.

As Cogley and Sargent put it, a model with economically meaningful drift often falls in the "indeterminate range" where such tests simply cannot see it.

## Bad policy or bad luck? A verdict

The Bayesian VAR delivers a nuanced answer to the question that opened this lecture.

* **Volatilities drifted.** The size of the shocks changed enormously, with a Volcker-era spike and a subsequent Great Moderation. The bad-luck story captures something real.
* **Coefficients drifted too.** Inflation persistence and core inflation rose through the 1970s and fell in the 1980s. The systematic dynamics were *not* time-invariant, so the bad-policy story also captures something real.
* **The classical tests could not have told us.** Their low power against drifting alternatives means that "we cannot reject constancy" was never good evidence for pure bad luck.

There is one more twist, and it loops us back to the theory of this section.

The escape-route models of {doc}`phillips_learning` and {doc}`phillips_escaping_nash` predict that inflation persistence should **grow** along a disinflation, as a learning government becomes reluctant to abandon a high-inflation self-confirming equilibrium.

The data show the **opposite**: persistence *fell* as inflation came down after 1980.

Cogley and Sargent flag this tension honestly.

It is precisely what motivated the next round of learning models — Cogley and Sargent's own {cite}`CogleySargentConquest2005` and Primiceri's {cite}`Primiceri2006` — in which policymakers' *reluctance to disinflate in the 1970s*, and their eventual conversion, generate a persistence that rises and then falls, matching the drift we have measured here.

The friendly debate with Sims, Zha, Bernanke, and Mihov thus did more than adjudicate a historical question.

It sharpened the theoretical models of learning and drift that run through this entire section — from the {doc}`self-confirming equilibria <phillips_self_confirming>` of the *Conquest* book to the {doc}`drifting Fed beliefs <phillips_lost_conquest>` of the 2020s.

## Exercises

```{exercise}
:label: csdv_ex1

The model nests the pure "bad luck" hypothesis as the special case $Q = 0$: constant coefficients, drifting volatilities only.

Modify the sampler to impose $Q = 0$ (so `draw_theta` is called with a zero drift covariance and the coefficients never move), refit, and compare the *volatility* paths and the *persistence* series to those from the full model.

What happens to measured inflation persistence when the coefficients are forced to be constant? Explain why.
```

```{solution-start} csdv_ex1
:class: dropdown
```

With $Q=0$ the coefficient path collapses to a single constant vector (the smoothed estimate), so `companion(thb[t])` no longer depends on $t$ and the *only* source of time variation in

$$
f(0)_t = \tfrac{1}{2\pi}(I-A)^{-1} R_t (I-A)^{-1\prime}
$$

is the drifting covariance $R_t$.

But the *normalized* spectrum $g_{\pi\pi}(0,t) = f(0)_t / \operatorname{Var}(\pi_t)$ divides out the scale of $R_t$, so with constant $A$ it is very nearly *flat*: forcing constant coefficients removes essentially all of the rise-and-fall in persistence.

That is exactly why one needs drifting *coefficients*, not just drifting volatilities, to reproduce the persistence dynamics — the substantive point of the paper.

The estimated volatility paths, by contrast, look much like those from the full model, because they are identified mostly by the size of the residuals.

A sketch of the change to the sampler:

```{code-cell} ipython3
:tags: [skip-execution]

# inside run(), replace the Q block with a frozen Q of zeros
Q = np.zeros((d, d))
# ... and skip the invwishart.rvs draw so Q stays zero throughout
```

```{solution-end}
```

```{exercise}
:label: csdv_ex2

The prior scalar `kQ` controls how much the coefficients are allowed to drift.

Re-run the sampler with `kQ = 0.005` and `kQ = 0.02` and plot the three resulting persistence series together.

How sensitive is the rise-and-fall of inflation persistence to this prior? What tradeoff do you face as you raise `kQ`?
```

```{solution-start} csdv_ex2
:class: dropdown
```

Raising `kQ` loosens the prior and permits faster drift.

The qualitative rise-and-fall of persistence — low in the early 1960s, high through the 1970s, low again after the Volcker disinflation — is present for all three values, which is reassuring: it is a feature of the data, not of the prior.

But the *amplitude* and *smoothness* change.

A larger `kQ` lets the coefficients chase the data more aggressively, which can *lower* the estimated peak persistence (some of what was drift in the systematic part gets reallocated) and makes the path more jagged; it also lowers the acceptance rate for coefficient paths, because looser drift more often proposes an explosive path that the stability prior rejects.

A smaller `kQ` gives a smoother, more strongly shrunk path.

The lesson is the one in the caveat about core inflation: the broad drift pattern is robust, while finer quantitative features depend on the prior and on the length of the run.

```{solution-end}
```
