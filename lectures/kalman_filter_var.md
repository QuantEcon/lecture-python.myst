---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# The Kalman Filter and Vector Autoregressions

```{index} single: Kalman Filter
```

```{index} single: Vector Autoregression; and Kalman filter
```

## Overview

This lecture derives the **Kalman filter** for a linear Gaussian state space system
and then uses it to construct **vector autoregressions (VARs)**.

Our approach rests on repeated applications of the **population linear least squares**
projection formula, the insight that computing a conditional expectation of a jointly
normal random vector is the same as running a population OLS regression.

The lecture covers:

- deriving the Kalman filter recursions from first principles
- the **matrix Riccati difference equation** governing conditional covariance matrices
- the **innovations representation** and the **Gram-Schmidt** whitening property
- the structure of a **hidden Markov model**
- the **likelihood function** for a state space system and its role in maximum likelihood
  and Bayesian estimation
- how the time-invariant Kalman filter generates a **vector autoregression**
- why the Kalman filter is an essential tool for **interpreting VARs** estimated from
  economic data

## The state space system

The Kalman filter applies to the **state space system** for $t \geq 0$:

$$
\begin{aligned}
x_{t+1} &= A x_t + C w_{t+1} \\
y_t      &= G x_t + v_t
\end{aligned}
$$ (eq:statespace)

where

- $x_t$ is an $n \times 1$ **state vector** (hidden, unobserved)
- $y_t$ is an $m \times 1$ vector of **signals** on the hidden state (observed)
- $w_{t+1}$ is a $p \times 1$ IID sequence of normal random variables with mean
  $0$ and identity covariance matrix
- $v_t$ is an IID sequence of normal random variables with mean zero and
  covariance matrix $R$
- $w_{t+1}$ and $v_s$ are orthogonal for all $t+1$ and $s \geq 0$

The initial state satisfies

$$
x_0 \sim N(\hat{x}_0, \Sigma_0)
$$ (eq:kalf3)

We observe $y_t, \ldots, y_0$ but **not** $x_t, \ldots, x_0$ at time $t$,
and we know all first and second moments implied by
{eq}`eq:statespace` and {eq}`eq:kalf3`.

## The Kalman filter

### Starting distribution

Working forward in time starting at $t = 0$, before observing $y_0$,
the specification {eq}`eq:statespace`-{eq}`eq:kalf3` implies that the
marginal distribution of $y_0$ is

$$
y_0 \sim N(G \hat{x}_0,\; G \Sigma_0 G^\top + R)
$$ (eq:kalf4)

For $t \geq 0$ let $y^t = [y_t, y_{t-1}, \ldots, y_0]$.

We want a convenient recursive representation of the conditional distribution of
$y_t$ given history $y^{t-1}$.

The Kalman filter attains this by constructing recursive formulas for
$\hat{x}_t$ and $\Sigma_t$ such that the distribution of $y_t$ conditional on
$y^{t-1}$ generalises {eq}`eq:kalf4` to

$$
y_t \sim N(G \hat{x}_t,\; G \Sigma_t G^\top + R)
$$ (eq:kalf400)

for $t \geq 1$, where the distribution of $x_t$ conditional on $y^{t-1}$ is
$N(\hat{x}_t, \Sigma_t)$.

The objects $\hat{x}_t$ and $\Sigma_t$ characterise the **population regression**

$$
\hat{x}_t = \mathbb{E}[x_t \mid y_{t-1}, \ldots, y_0]
$$

and the **conditional covariance matrix**

$$
\Sigma_t = \mathbb{E}\!\left[(x_t - \hat{x}_t)(x_t - \hat{x}_t)^\top \mid y_{t-1}, \ldots, y_0\right]
$$

### Derivation

At each date, our approach is to **regress what we do not know on what we know**.

```{note}
Because our assumptions imply that $\{x_t, y_t\}_{t=0}^\infty$ is a jointly normal
stochastic process, linear least squares regressions equal conditional mathematical
expectations.

Each step below is an application of Bayes' law.

Under the weaker assumption that all means and covariances exist without joint normality,
the same calculations yield "wide-sense conditional expectations" that coincide with true
conditional expectations only when those conditional expectations are linear.
```

**Arrive at $t = 0$** knowing $\hat{x}_0$ and $\Sigma_0$.

**Innovation:** The information about $x_0$ in $y_0$ that is new relative to
$(\hat{x}_0, \Sigma_0)$ is the **innovation**

$$
a_0 \equiv y_0 - G \hat{x}_0
$$

**Updating beliefs about $x_0$:** The conditional mean
$\mathbb{E}[x_0 \mid y_0] = \hat{x}_0 + L_0(y_0 - G\hat{x}_0)$ satisfies the
population regression formula

$$
x_0 - \hat{x}_0 = L_0(y_0 - G\hat{x}_0) + \eta
$$ (eq:kalf5)

where $\eta$ is the least squares residual.

Orthogonality of $\eta$ to
$(y_0 - G\hat{x}_0)$ pins down $L_0$ via the normal equations

$$
\mathbb{E}(x_0 - \hat{x}_0)(y_0 - G\hat{x}_0)^\top
= L_0\, \mathbb{E}(y_0 - G\hat{x}_0)(y_0 - G\hat{x}_0)^\top
$$

Evaluating the moment matrices and solving for $L_0$ gives

$$
L_0 = \Sigma_0 G^\top(G \Sigma_0 G^\top + R)^{-1}
$$ (eq:kalf6)

**Forecasting $x_1$:** Note that

$$
x_1 = A\hat{x}_0 + A(x_0 - \hat{x}_0) + C w_1
$$ (eq:kalf6a)

Applying {eq}`eq:kalf5` gives $\mathbb{E}[x_1 \mid y_0] = A\hat{x}_0 + AL_0(y_0 - G\hat{x}_0)$,
which we write as

$$
\hat{x}_1 = A\hat{x}_0 + K_0(y_0 - G\hat{x}_0)
$$ (eq:kalf7)

where the **Kalman gain** at time 0 is

$$
K_0 = A \Sigma_0 G^\top(G \Sigma_0 G^\top + R)^{-1}
$$ (eq:kalf7a)

**Updating the covariance:** Subtracting {eq}`eq:kalf7` from {eq}`eq:kalf6a` yields

$$
x_1 - \hat{x}_1 = A(x_0 - \hat{x}_0) + C w_1 - K_0(y_0 - G\hat{x}_0)
$$ (eq:kalf8)

Using {eq}`eq:kalf8` and $y_0 = G x_0 + v_0$ to evaluate
$\Sigma_1 \equiv \mathbb{E}[(x_1 - \hat{x}_1)(x_1 - \hat{x}_1)^\top \mid y_0]$ gives

$$
\Sigma_1 = (A - K_0 G)\Sigma_0(A - K_0 G)^\top + CC^\top + K_0 R K_0^\top
$$ (eq:kalf9)

Thus $f(x_1 \mid y_0) \sim N(\hat{x}_1, \Sigma_1)$.

Collecting the time-$0$ equations:

$$
\begin{aligned}
a_0       &= y_0 - G\hat{x}_0 \\
K_0       &= A\Sigma_0 G^\top(G\Sigma_0 G^\top + R)^{-1} \\
\hat{x}_1 &= A\hat{x}_0 + K_0 a_0 \\
\Sigma_1  &= CC^\top + K_0 R K_0^\top + (A - K_0 G)\Sigma_0(A - K_0 G)^\top
\end{aligned}
$$ (eq:kalf1000)

System {eq}`eq:kalf1000` maps a mean-covariance pair $(\hat{x}_0, \Sigma_0)$ into a new
pair $(\hat{x}_1, \Sigma_1)$, with auxiliary outputs $(a_0, K_0)$.

Recognising that "we are in the same situation at the start of period 1 as at
the start of period 0" activates a recursion, the **Kalman filter**.

### The Kalman filter recursions

Iterating system {eq}`eq:kalf1000` yields the **Kalman filter** for $t \geq 0$:

$$
\begin{aligned}
a_t           &= y_t - G\hat{x}_t \\
K_t           &= A\Sigma_t G^\top(G\Sigma_t G^\top + R)^{-1} \\
\hat{x}_{t+1} &= A\hat{x}_t + K_t a_t \\
\Sigma_{t+1}  &= CC^\top + K_t R K_t^\top + (A - K_t G)\Sigma_t(A - K_t G)^\top
\end{aligned}
$$ (eq:kalf10)

Here $K_t$ is the **Kalman gain** at time $t$.

### The matrix Riccati equation

Substituting the expression for $K_t$ from the second line of {eq}`eq:kalf10`
into the fourth line gives an equivalent update formula:

$$
\Sigma_{t+1} = A\Sigma_t A^\top + CC^\top
  - A\Sigma_t G^\top(G\Sigma_t G^\top + R)^{-1} G\Sigma_t A^\top
$$ (eq:riccati)

Equation {eq}`eq:riccati` is the **matrix Riccati difference equation**.

It governs the sequence of conditional covariance matrices $\{\Sigma_t\}_{t=0}^\infty$
without reference to the observations $\{y_t\}$.

```{index} single: Riccati equation; matrix difference
```

## The Gram-Schmidt process

The random vector

$$
a_t = y_t - \mathbb{E}[y_t \mid y_{t-1}, \ldots, y_0]
$$

is the **innovation** of $y_t$ with respect to $y^{t-1}$, the part of $y_t$
that cannot be predicted from past observations.

Note that $\mathbb{E} a_t a_t^\top = G\Sigma_t G^\top + R$, the matrix whose inverse appears in
the Kalman gain formula {eq}`eq:kalf10`.

A direct calculation using $a_t = G(x_t - \hat{x}_t) + v_t$ shows that
$\mathbb{E} a_t a_{t-1}^\top = 0$ and, more generally, $\mathbb{E}[a_t \mid a_{t-1}, \ldots, a_0] = 0$.

```{note}
An alternative argument from first principles: let $H(y^t)$ denote the closed linear
span of $y^t$.

Since $a_{t+1} = y_{t+1} - \mathbb{E}[y_{t+1} \mid y^t]$ is a least-squares
error, $a_{t+1} \perp H(y^t)$, and in particular $a_{t+1} \perp a_t$.

Thus $\{a_t\}$ is a white-noise process of innovations to $\{y_t\}$.
```

Sometimes {eq}`eq:kalf10` is called a **whitening filter**: it takes the signal process
$\{y_t\}$ as input and produces the white-noise innovation process $\{a_t\}$ as output.

The linear space $H(a^t)$ is an orthogonal basis for the linear space $H(y^t)$.

Rather than computing $\mathbb{E}[x_t \mid y_{t-1}, \ldots, y_0]$ via one large regression,
the Kalman filter performs a sequence of small regressions on successive orthogonal
components of the basis $[a_{t-1}, \ldots, a_0]$, an instance of the
**Gram-Schmidt procedure**.

```{index} single: Gram-Schmidt process
```

## Hidden Markov model

System {eq}`eq:statespace`-{eq}`eq:kalf3` is an example of a **hidden Markov model**.

```{index} single: hidden Markov model
```

The observed process $\{y_t\}_{t=0}^\infty$ is *not* Markov, but the hidden
process $\{x_t\}_{t=0}^\infty$ *is* Markov.

So is the process of means and
covariances $\{(\hat{x}_t, \Sigma_t)\}$, which are sufficient statistics for
the distribution of $x_t$ conditional on $[y_{t-1}, \ldots, y_0]$.

## Estimation

### The innovations representation

The **innovations representation** emerging from the Kalman filter is

$$
\begin{aligned}
\hat{x}_{t+1} &= A\hat{x}_t + K_t a_t \\
y_t           &= G\hat{x}_t + a_t
\end{aligned}
$$ (eq:innovrep)

where $\hat{x}_t = \mathbb{E}[x_t \mid y^{t-1}]$ for $t \geq 1$ and
$\mathbb{E}[a_t a_t^\top \mid y^{t-1}] = G\Sigma_t G^\top + R \equiv \Omega_t$.

For $t \geq 1$, $\mathbb{E}[y_t \mid y^{t-1}] = G\hat{x}_t$ and the conditional
distribution of $y_t$ given $y^{t-1}$ is $N(G\hat{x}_t, \Omega_t)$.

The objects $(G\hat{x}_t, \Omega_t)$ emerging from the Kalman filter recursions
therefore completely characterise this conditional distribution.

### The likelihood function

We can factor the likelihood of a sample $(y_T, y_{T-1}, \ldots, y_0)$ as

$$
f(y_T, \ldots, y_0)
  = f(y_T \mid y^{T-1})\, f(y_{T-1} \mid y^{T-2}) \cdots f(y_1 \mid y_0)\, f(y_0)
$$ (eq:diff100)

The log conditional density of the $m \times 1$ vector $y_t$ is

$$
\log f(y_t \mid y^{t-1})
  = -\frac{m}{2}\log(2\pi)
    - \frac{1}{2}\log\det(\Omega_t)
    - \frac{1}{2}\, a_t^\top \Omega_t^{-1} a_t
$$ (eq:gauss100)

Using {eq}`eq:gauss100` and {eq}`eq:kalf10` together, we can evaluate
the likelihood {eq}`eq:diff100` recursively for any parameter vector $\theta$
that underlies the matrices $A, G, C, R$.

Such calculations are at the heart of efficient strategies for computing
**maximum likelihood estimators** of free parameters.

### Bayesian inference

The likelihood function is also central to **Bayesian inference**.

Where $\theta$ is the parameter vector, $y_0^T$ the data, and
$\tilde{p}(\theta)$ a prior density over $\theta$ before seeing $y_0^T$,
Bayes' law gives the **posterior**

$$
\tilde{p}(\theta \mid y_0^T)
  = \frac{f(y_0^T \mid \theta)\,\tilde{p}(\theta)}
         {\int f(y_0^T \mid \theta)\,\tilde{p}(\theta)\, d\theta}
$$

The denominator is the marginal joint density $f(y_0^T)$.

## Vector autoregressions and the Kalman filter

### Convergence to a steady state

Under conditions discussed by Anderson, Hansen, McGrattan, and Sargent (1996),
iterations on the Riccati equation {eq}`eq:riccati` converge to a
**time-invariant** matrix $\Sigma$ from any positive semi-definite starting value
$\Sigma_0$.

A time-invariant fixed point $\Sigma_t = \Sigma$ of {eq}`eq:riccati` is the
covariance matrix of $x_t$ around

$$
\mathbb{E}\!\left[x_t \mid \{y_s\}_{s \leq t-1}\right]
$$

where the conditioning extends over the **semi-infinite** past $s \leq t-1$.

### A time-invariant VAR

If the fixed point $\Sigma$ exists and we initialise the filter at $\Sigma_0 = \Sigma$,
the innovations representation {eq}`eq:innovrep` becomes time-invariant:

$$
\begin{aligned}
\hat{x}_{t+1} &= A\hat{x}_t + K a_t \\
y_t           &= G\hat{x}_t + a_t
\end{aligned}
$$ (eq:innovti)

where $\mathbb{E} a_t a_t^\top = G\Sigma G^\top + R$ and the **steady-state Kalman gain** is
$K = A\Sigma G^\top(G\Sigma G^\top + R)^{-1}$.

From {eq}`eq:innovti` we obtain $\hat{x}_{t+1} = (A - KG)\hat{x}_t + K y_t$.

If the eigenvalues of $A - KG$ are bounded in modulus strictly below unity, we can
solve this equation forward to get

$$
\hat{x}_{t+1} = \sum_{j=0}^\infty (A - KG)^j K\, y_{t-j}
$$ (eq:xhatform)

Substituting {eq}`eq:xhatform` into the observation equation of {eq}`eq:innovti`
gives the **vector autoregression**

$$
y_t = G \sum_{j=0}^\infty (A - KG)^j K\, y_{t-j-1} + a_t
$$ (eq:var1)

where by construction

$$
\mathbb{E}\!\left[a_t\, y_{t-j-1}^\top\right] = 0 \quad \forall\, j \geq 0
$$ (eq:varorth)

The orthogonality conditions {eq}`eq:varorth` identify {eq}`eq:var1` as a
**vector autoregression**.

Defining the lag operator $L$ by $L x_{t+1} \equiv x_t$, the
**moving average representation** deduced from {eq}`eq:innovti` is

$$
y_t = \left[I + G(I - AL)^{-1} KL\right] a_t
    = \left[I + G\sum_{j=0}^\infty A^j K L^{j+1}\right] a_t
$$

```{index} single: vector autoregression
```

### Interpreting VARs

Equilibria of economic models (or their linear or log-linear approximations)
typically take the form of state space system {eq}`eq:statespace`.

This hidden Markov model disturbs the state $x_t$ by the $p \times 1$ shock
vector $w_{t+1}$ and perturbs the $m \times 1$ vector of observables $y_t$ by the
$m \times 1$ measurement error $v_t$.

An economic theory typically makes $w_{t+1}$ and $v_t$ directly interpretable as
shocks to preferences, technologies, endowments, or information sets.

The state space system {eq}`eq:statespace` represents $\{y_t\}$ in terms of these
**interpretable shocks**.

However, in the typical situation these shocks
**cannot** be recovered directly from the $y_t$'s, even when $A, G, C, R$ are known.

The innovations representation {eq}`eq:innovti` represents the *same* stochastic
process $\{y_t\}$ in terms of the $m \times 1$ vector $a_t$ of innovations that
would be recovered by running an infinite-order population vector autoregression.

Its role in mapping the original representation {eq}`eq:statespace` to the VAR
{eq}`eq:var1` makes the Kalman filter an indispensable tool for
**interpreting vector autoregressions**.

```{index} single: Kalman Filter; and vector autoregressions
```

## Spectral factorization identity

```{index} single: spectral factorization identity
```

Because the original state space system {eq}`eq:statespace` and the innovations
representation {eq}`eq:innovti` describe the **same** stochastic process $\{y_t\}$,
they imply two distinct formulas for the **spectral density matrix** of $\{y_t\}$.

Equating those formulas yields the *spectral factorization identity*.

### Two representations of the spectral density

**From the original state space system.**  Writing the first line of
{eq}`eq:statespace` as $x_t = (zI - A)^{-1} C w_{t+1}$ (using the $z$-transform
convention $z^{-1} x_t = x_{t-1}$), the covariance generating function of
$\{x_t\}$ is

$$
S_x(z) = (zI - A)^{-1} CC^\top (z^{-1}I - A^\top)^{-1}.
$$

Since $v_t$ is orthogonal to $x_t$, the spectral density of $\{y_t\}$ is

$$
S_y(z) = G(zI - A)^{-1} CC^\top (z^{-1}I - A^\top)^{-1} G^\top + R.
$$ (eq:sf_original)

**From the innovations representation.**  The time-invariant innovations
representation {eq}`eq:innovti` gives $y_t = [G(zI - A)^{-1}K + I]\, a_t$.
Since $a_t$ is white noise with covariance matrix $G\Sigma G^\top + R$, the
spectral density is also

$$
S_y(z) = \bigl[G(zI-A)^{-1}K + I\bigr]
          \bigl(G\Sigma G^\top + R\bigr)
          \bigl[K^\top(z^{-1}I - A^\top)^{-1}G^\top + I\bigr].
$$ (eq:sf_innov)

### The spectral factorization identity

Equating {eq}`eq:sf_original` and {eq}`eq:sf_innov` gives the
**spectral factorization identity**:

$$
G(zI - A)^{-1} CC^\top (z^{-1}I - A^\top)^{-1} G^\top + R =
\bigl[G(zI-A)^{-1}K + I\bigr]
\bigl(G\Sigma G^\top + R\bigr)
\bigl[K^\top(z^{-1}I - A^\top)^{-1}G^\top + I\bigr].
$$ (eq:sf_identity)

The left side expresses $S_y(z)$ in terms of the **structural shocks**
$(w_{t+1}, v_t)$ and the matrices $(A, C, G, R)$.

The right side expresses the same object as a spectral factor built from the
**innovations** $a_t$ and the steady-state Kalman gain $K$.

### Connection to the Wold and autoregressive representations

The factorization {eq}`eq:sf_identity` underpins the passage from the innovations
representation to the Wold moving average and to the VAR.

**Wold representation.**  From {eq}`eq:innovti`, solving for $\hat{x}_t$ over the
semi-infinite past gives $\hat{x}_{t+1} = (I - AL)^{-1} K a_t$, so

$$
y_t = \bigl[G(I - AL)^{-1} KL + I\bigr]\, a_t.
$$ (eq:sf_wold)

**Autoregressive (VAR) representation.**  Applying the inverse of the
moving-average operator in {eq}`eq:sf_wold`, using the identity

$$
\bigl[G(I-AL)^{-1}KL + I\bigr]^{-1} = I - G\bigl[I-(A-KG)L\bigr]^{-1}KL,
$$

gives

$$
y_t = G\bigl[I-(A-KG)L\bigr]^{-1}K\, y_{t-1} + a_t
    = \sum_{j=1}^\infty G(A-KG)^{j-1}K\, y_{t-j} + a_t,
$$ (eq:sf_var)

which is the **vector autoregression** already stated in {eq}`eq:var1`.

The key analytical fact behind both representations is that, under mild stability
conditions, the zeros of $\det[G(zI-A)^{-1}K + I]$ all lie **inside** the unit
circle.

This ensures that the moving-average operator in {eq}`eq:sf_wold` has a
causal (one-sided) inverse, so the innovation $a_t$ lies in the closed linear span
of current and past observations $y^t$, confirming that $a_t$ is the true
population forecast error from the VAR.

## Python implementation

We now illustrate the theory using the `quantecon` library, which provides
`LinearStateSpace` and `Kalman` classes that implement everything derived above.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
```

### A scalar hidden AR(1) model

Consider a scalar hidden AR(1) state observed with measurement noise:

$$
\begin{aligned}
x_{t+1} &= \rho\, x_t + \sigma_w\, w_{t+1} \\
y_t      &= x_t + \sigma_v\, v_t
\end{aligned}
$$

with $w_t, v_t \sim N(0, 1)$ IID.

```{code-cell} ipython3
# Model parameters
ρ = 0.9
σ_w = 0.5
σ_v = 1.0

# State-space matrices
A = np.array([[ρ]])
C = np.array([[σ_w]])
G = np.array([[1.0]])
R = np.array([[σ_v**2]])

# Build a LinearStateSpace and a Kalman filter object
H = np.array([[σ_v]])   # measurement noise factor: R = H @ H.T
lss = qe.LinearStateSpace(
  A, C, G, H, mu_0=np.zeros(1), Sigma_0=np.eye(1) * 10.0)
kf = qe.Kalman(lss)
kf.set_state(np.zeros(1), np.eye(1) * 10.0)  # diffuse prior
```

**Simulate a sample path of the true hidden state and noisy observations.**

```{code-cell} ipython3
T = 200
x_path, y_path = lss.simulate(ts_length=T, random_state=42)

# Shapes: x_path is (n, T+1), y_path is (m, T)
x_true = x_path[0, :T]
y_obs = y_path[0, :]
```

**Run the Kalman filter manually step by step to collect filtered estimates.**

```{code-cell} ipython3
x_hats = np.zeros(T)
Sigmas = np.zeros(T)

for t in range(T):
    kf.update(y_obs[t:t+1])          # one full filter cycle
    x_hats[t] = kf.x_hat.item()
    Sigmas[t] = kf.Sigma.item()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Scalar Kalman filtering
    name: fig-kfvar-scalar
---
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

t_range = np.arange(T)

axes[0].plot(t_range, x_true, lw=2, 
        label='true state $x_t$')
axes[0].plot(t_range, x_hats, lw=2, 
        linestyle='--', label=r'$\hat{x}_t$ (Kalman)')
axes[0].plot(t_range, y_obs, alpha=0.35, lw=2, label='observation $y_t$')
axes[0].set_title('state and observation')
axes[0].legend(fontsize=9)

axes[1].plot(t_range, Sigmas, color='C1', lw=2,
             label=r'conditional variance $\Sigma_t$')
axes[1].axhline(kf.Sigma_infinity[0, 0], ls='--', color='k',
                label=r'steady-state $\Sigma_\infty$')
axes[1].set_title('conditional variance')
axes[1].legend(fontsize=9)

axes[2].plot(t_range, y_obs - x_hats, color='C2', lw=2, alpha=0.7,
             label=r'innovation $a_t = y_t - G\hat{x}_t$')
axes[2].set_title('innovation')
axes[2].set_xlabel('time $t$')
axes[2].legend(fontsize=9)
fig.tight_layout()
plt.show()
```

### Convergence of the Riccati equation

The `Kalman` class computes the steady-state covariance $\Sigma_\infty$ by
solving the discrete algebraic Riccati equation directly.

```{code-cell} ipython3
Sigma_inf, K_inf = kf.stationary_values()

print(f"Steady-state covariance  Σ_inf = {Sigma_inf[0, 0]:.6f}")
print(f"Kalman filter converged to Σ_t = {Sigmas[-1]:.6f}")
print(f"Steady-state Kalman gain K  = {K_inf[0, 0]:.6f}")

A_minus_KG = A - K_inf @ G
eigval = np.linalg.eigvals(A_minus_KG)[0]
print(f"\nEigenvalue of (A - KG)      = {eigval:.6f}")
print(f"Stable VAR: {np.abs(eigval) < 1}")
```

### The VAR representation

Using {eq}`eq:var1`, the coefficients in the infinite-order VAR representation are
$G(A - KG)^j K$ for $j = 0, 1, 2, \ldots$

We retrieve them via `stationary_coefficients`:

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: VAR coefficients from the innovations representation
    name: fig-kfvar-varcoef
---
J = 30
var_coeffs = kf.stationary_coefficients(J, coeff_type='var')

# Lag j+1 coefficient matrices
lags = np.arange(1, J + 1)
coeff_values = np.array([var_coeffs[j][0, 0] for j in range(J)])

fig, ax = plt.subplots()
ax.stem(lags, coeff_values, basefmt=' ')
ax.set_xlabel('lag $j$')
ax.set_ylabel(r'VAR coefficient $G(A{-}KG)^{j-1}K$')
fig.tight_layout()
plt.show()
```

The coefficients decay geometrically, confirming that the infinite-order VAR
{eq}`eq:var1` is well approximated by a finite-lag truncation.

### Likelihood evaluation

We use {eq}`eq:gauss100` to evaluate the log-likelihood of the simulated sample.

```{code-cell} ipython3
def log_likelihood(A, C, G, R, y_data, x_hat_0, Sigma_0):
    """Evaluate the log-likelihood using the Kalman filter recursions."""
    H_ = np.linalg.cholesky(R)   # R = H_ @ H_.T
    lss_ = qe.LinearStateSpace(A, C, G, H_, mu_0=x_hat_0, Sigma_0=Sigma_0)
    kf_ = qe.Kalman(lss_)
    kf_.set_state(x_hat_0, Sigma_0)

    T_, m_ = y_data.shape
    loglik = 0.0

    for t in range(T_):
        x_h = kf_.x_hat
        Sig = kf_.Sigma
        Omega = G @ Sig @ G.T + R        # innovation covariance
        a_t = y_data[t] - (G @ x_h).flatten()

        sign, logdet = np.linalg.slogdet(Omega)
        loglik += -0.5 * (m_ * np.log(2 * np.pi) + logdet
                          + float(a_t @ np.linalg.solve(Omega, a_t)))
        kf_.update(y_data[t])

    return loglik


y_data_col = y_obs.reshape(-1, 1)
ll = log_likelihood(A, C, G, R,
                    y_data_col,
                    np.zeros(1), np.eye(1) * 10.0)
print(f"Log-likelihood of sample: {ll:.4f}")
```

## An example

We now work through a structured example that shows how a bivariate VAR(2) fits naturally into the state space framework and how the Kalman filter delivers a
Wold (innovations) representation.

### Linear state-space system

The state and observation equations are

$$
x_{t+1} = A x_t + C w_{t+1}
$$ (eq:ex_state)

$$
y_t = G x_t + v_t
$$ (eq:ex_obs)

with initial condition and shock distributions

$$
x_0 \sim N(M_0, \Omega_0), \quad
w_{t+1} \sim N(0, I), \quad
v_t \sim N(0, R).
$$

### Steady-state Riccati equation

The steady-state error covariance matrix $\Sigma$ satisfies

$$
\Sigma = A \Sigma A^\top + CC^\top
         - A \Sigma G^\top \bigl(G \Sigma G^\top + R\bigr)^{-1} G \Sigma A^\top
$$ (eq:ex_riccati)

### Kalman gain

$$
K = A \Sigma G^\top \bigl(G \Sigma G^\top + R\bigr)^{-1}
$$ (eq:ex_gain)

### Kalman filter recursion

Starting from an initial estimate $\hat{x}_0$, the Kalman filter updates the state estimate via

$$
\hat{x}_{t+1} = A \hat{x}_t + K a_t
$$ (eq:ex_kf_update)

where the **innovation** (prediction error) is

$$
a_t = y_t - G \hat{x}_t
$$ (eq:ex_innovation)

Substituting {eq}`eq:ex_innovation` into {eq}`eq:ex_kf_update` and expanding:

$$
\hat{x}_{t+1} = A \hat{x}_t + K(y_t - G\hat{x}_t)
              = (A - KG)\hat{x}_t + K y_t
              = (A - KG)\hat{x}_t + K G x_t + K v_t
$$ (eq:ex_kf_expanded)

### Impulse responses of $y_t$ to the innovations $a_t$

It is useful to compute the **ordinary impulse response functions** of the
observable vector $y_t$ to its own innovations $a_t$, the moving-average (Wold)
representation that is the mirror image of the VAR {eq}`eq:var1`.

From the time-invariant innovations representation {eq}`eq:innovti`

$$
\hat{x}_{t+1} = A\hat{x}_t + K a_t, \qquad y_t = G\hat{x}_t + a_t,
$$

the moving-average representation {eq}`eq:sf_wold` is

$$
y_t = \bigl[I + G(I - AL)^{-1} K L\bigr]\, a_t
    = a_t + \sum_{h=1}^{\infty} G A^{h-1} K\, a_{t-h}.
$$

Hence the impulse response of $y_t$ to a unit innovation $a_t$ is

$$
\Psi_0 = I, \qquad \Psi_h = G A^{h-1} K \quad (h \ge 1).
$$ (eq:ex_y_to_a)

These coefficients decay at the rate governed by the eigenvalues of $A$, in
contrast to the innovation-to-structural-shock responses studied later in the
numerical example, which decay at the rate governed by the eigenvalues of
$A - KG$.

We can read the coefficients {eq}`eq:ex_y_to_a` directly off a `quantecon`
`LinearStateSpace` object.

We build a state-space system whose state is the
filtered estimate $\hat{x}_t$, whose single "shock" is the innovation $a_t$
loaded through $C = K$, and whose observation matrix is $G$.

The
`impulse_response` method of that object returns the sequence $G A^{j} K$ for
$j = 0, 1, 2, \ldots$, which are exactly the $\Psi_h$ for $h \ge 1$; we prepend
$\Psi_0 = I$ to capture the contemporaneous feed-through $y_t = G\hat{x}_t + a_t$.

```{code-cell} ipython3
def y_to_a_irf(A, K, G, T=40):
    """
    Ordinary impulse responses of the observable y_t to its own
    innovations a_t in the innovations (Wold) representation

        x_hat_{t+1} = A x_hat_t + K a_t
        y_t      = G x_hat_t + a_t.

    The moving-average coefficients are
        Ψ_0 = I,   Ψ_h = G A^{h-1} K   (h >= 1).

    The h >= 1 terms come from quantecon's LinearStateSpace: loading the
    innovation a_t through C = K makes its impulse_response method return
    G A^j K for j = 0, 1, 2, ....  We prepend Ψ_0 = I for the
    contemporaneous feed-through.

    Returns an array of shape (T, m, m); entry [h, i, j] is the response
    of observable i at horizon h to a unit innovation in component j.
    """
    n, m = A.shape[0], G.shape[0]
    lss = qe.LinearStateSpace(A, K, G, np.zeros((m, m)), mu_0=np.zeros(n))
    _, ycoef = lss.impulse_response(j=T - 2)      # [GK, GAK, GA^2K, ...]
    Psi = np.empty((T, m, m))
    Psi[0] = np.eye(m)                          # contemporaneous response
    for h in range(1, T):
        Psi[h] = ycoef[h - 1]
    return Psi
```

### Augmented state-space representation

We want to express the innovations $a_t$ as a function of the state shocks
$w_{t+1}$ and the measurement error $v_t$.

To accomplish this, we start by stacking the true state $x_t$ and the filter estimate $\hat{x}_t$ into an augmented state vector that
obeys

$$
\begin{pmatrix} x_{t+1} \\ \hat{x}_{t+1} \end{pmatrix}
=
\begin{pmatrix} A & 0 \\ KG & A - KG \end{pmatrix}
\begin{pmatrix} x_t \\ \hat{x}_t \end{pmatrix}
+
\begin{pmatrix} C & 0 \\ 0 & K \end{pmatrix}
\begin{pmatrix} w_{t+1} \\ v_t \end{pmatrix}
$$ (eq:ex_augmented)


### Bivariate VAR(2) in state-space form

Consider two observable series $r_t$ and $z_t$.

Stack them into the state vector $x_t = (r_t,\; r_{t-1},\; z_t,\; z_{t-1})^\top$.

We posit the VAR(2) state-transition equation:

$$
\begin{pmatrix} r_{t+1} \\ r_t \\ z_{t+1} \\ z_t \end{pmatrix}
=
\begin{pmatrix}
  d_1      & d_2      & d_3      & d_4      \\
  1        & 0        & 0        & 0        \\
  \delta_1 & \delta_2 & \delta_3 & \delta_4 \\
  0        & 0        & 1        & 0
\end{pmatrix}
\begin{pmatrix} r_t \\ r_{t-1} \\ z_t \\ z_{t-1} \end{pmatrix}
+
\begin{pmatrix}
  c_{11} & c_{12} \\
  0      & 0      \\
  c_{21} & c_{22} \\
  0      & 0
\end{pmatrix}
\begin{pmatrix} w_{1,t+1} \\ w_{2,t+1} \end{pmatrix}
$$ (eq:ex_var2_state)

We consider two possible  observation equations.

The first is a bivariate observation of $r_t$ and $z_t$:

$$
\begin{pmatrix} r_t \\ z_t \end{pmatrix}
=
\begin{pmatrix}
  1 & 0 & 0 & 0 \\
  0 & 0 & 1 & 0
\end{pmatrix}
\begin{pmatrix} r_t \\ r_{t-1} \\ z_t \\ z_{t-1} \end{pmatrix}
+
\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
\begin{pmatrix} v_{1t} \\ v_{2t} \end{pmatrix}
$$ (eq:ex_var2_obs)

The second is a univariate observation of $r_t$:

$$
y_t = \begin{pmatrix} 1 & 0 & 0 & 0 \end{pmatrix}
\begin{pmatrix} r_t \\ r_{t-1} \\ z_t \\ z_{t-1} \end{pmatrix}
+ v_{1t}
$$ (eq:ex_scalar_obs)

For the bivariate observation case,
the Wold representation is

$$
\begin{pmatrix} r_t \\ z_t \end{pmatrix}
=
\underbrace{G \bigl(I - (A - KG)L\bigr)^{-1} K}_{\text{transfer matrix}}
\, a_t
$$ (eq:ex_wold_bivariate)

where $\mathbb{E} a_t a_t^\top = G \Sigma G^\top + R$ and $L$ is the lag operator.

The innovation expressed in terms of the augmented state is

$$
a_t
= \begin{bmatrix} G & -G \end{bmatrix}
  \begin{pmatrix} x_t \\ \hat{x}_t \end{pmatrix}
+ \begin{bmatrix} 0 & I \end{bmatrix}
  \begin{pmatrix} w_{t+1} \\ v_t \end{pmatrix}
= G(x_t - \hat{x}_t) + v_t
$$ (eq:ex_innovation_aug)



For the scalar ($1 \times 1$) case with a single observable $r_t$, the observation matrix $G_1$ is the first row of $G$ and the scalar gain $K_1$ is the outcome of the Kalman filter for that case.  The univariate Wold representation is

$$
r_t = G_1 \bigl(I - (A - K_1 G_1) L\bigr)^{-1} K_1 \, u_t
$$ (eq:ex_wold_scalar)

where the univariate innovation $u_t$ is given by

$$
u_t
= \begin{bmatrix} G_1 & -G_1 \end{bmatrix}
  \begin{pmatrix} x_t \\ \hat{x}_t \end{pmatrix}
+ \begin{bmatrix} 0 & 1 \end{bmatrix}
  \begin{pmatrix} w_{t+1} \\ v_t \end{pmatrix}
= G_1(x_t - \hat{x}_t) + v_{1,t}
$$ (eq:ex_innovation_aug2)

and $\mathbb{E} u_t^2 = G_1 \check \Sigma G_1^\top + R_{11}$, and $\check \Sigma$ is the steady-state covariance of the augmented state vector associated with
$G_1, R_{11}$.

### Numerical example: impulse responses of innovations

We now set specific parameter values and compute the **impulse response functions
of the Kalman innovations** ($a_t$ in System 1 and $u_t$ in System 2) to the
two structural shocks $w_{1,t+1}$ and $w_{2,t+1}$.

The key object is **not** the response of the observable $y_t$ to the shocks, but
rather the response of the innovation that the Kalman filter produces.

In System 1, $a_t$ is the $2 \times 1$ forecast error of $(r_t, z_t)^\top$ given the
filter's estimate; in System 2, $u_t$ is the scalar forecast error of $r_t$ given
the filter's estimate based on the $r_t$ history alone.

The parameter values are:

$$
\begin{aligned}
d_1 &= 0.80,\quad d_2 = 0.05,\quad d_3 = 0.75,\quad d_4 = -0.72 \\
\delta_1 &= 0.00,\quad \delta_2 = 0.00,\quad \delta_3 = 0.75,\quad \delta_4 = 0.20 \\
c_{11} &= 1.0,\quad c_{12} = 0.0,\quad c_{21} = 0.0,\quad c_{22} = 1.0 \\
R &= 0.0001 \times I_2 \quad \text{(bivariate case)}, \qquad
R = 0.0001 \quad \text{(univariate case)}.
\end{aligned}
$$

These give the $4 \times 4$ transition matrix and $4 \times 2$ shock-loading matrix

$$
A = \begin{pmatrix}
0.80 & 0.05 & 0.75 & -0.72 \\
1    & 0    & 0    & 0    \\
0    & 0    & 0.75 & 0.20 \\
0    & 0    & 1    & 0
\end{pmatrix}, \qquad
C = \begin{pmatrix}
1   & 0   \\
0   & 0   \\
0   & 1   \\
0   & 0
\end{pmatrix}.
$$

**System 1** uses the bivariate observation equation {eq}`eq:ex_var2_obs`, so
$G$ selects $(r_t, z_t)^\top$ from the state and the innovation $a_t$ is $2 \times 1$.

**System 2** uses the univariate observation equation {eq}`eq:ex_scalar_obs`, so
$G_1$ selects only $r_t$ and the innovation $u_t$ is scalar.

Because the innovation equals $a_t = G(x_t - \hat{x}_t) + v_t$, a unit shock $w_j$
propagates through the **augmented** state $(x_t^\top, \hat{x}_t^\top)^\top$.

A straightforward
calculation using the augmented recursion {eq}`eq:ex_augmented` shows that the
**impulse response of the innovation** at horizon $h \geq 0$ is

$$
\text{IRF}_{a}(h) = G(A - KG)^h C e_j, \qquad j = 1, 2,
$$

where $e_j$ is the $j$-th column of $I_2$ and $K$ is the appropriate steady-state
Kalman gain (System 1 uses $K$ from the bivariate filter; System 2 uses $K_1$ from
the univariate filter).

At $h=0$ the shock hits and the innovation equals $GCe_j$;
for $h \geq 1$ the response decays at the rate governed by the eigenvalues of $A - KG$.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe

# Parameters
d1, d2, d3, d4 = 0.80, 0.05, 0.75, -.72
δ1, δ2, δ3, δ4 = 0.00, 0.00, 0.75, 0.20
c11, c12, c21, c22 = 1.0,  0.0,  0.0,  1.0
σ_v = 0.01  # sqrt(0.0001)

# Shared matrices
A_var = np.array([[d1,     d2,     d3,     d4    ],
                  [1.0,    0.0,    0.0,    0.0   ],
                  [δ1, δ2, δ3, δ4],
                  [0.0,    0.0,    1.0,    0.0   ]])

C_var = np.array([[c11, c12],
                  [0.0, 0.0],
                  [c21, c22],
                  [0.0, 0.0]])

# System 1: bivariate observation
G_biv = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]])
H_biv = σ_v * np.eye(2)          # H @ H.T = 0.0001 * I_2

lss_biv = qe.LinearStateSpace(A_var, C_var, G_biv, H_biv,
                               mu_0=np.zeros(4), Sigma_0=np.eye(4))
kf_biv = qe.Kalman(lss_biv)
Sigma_biv, K_biv = kf_biv.stationary_values()

print("System 1 - steady-state Kalman gain K (4x2):")
print(np.round(K_biv, 5))

# System 2: univariate observation
G_uni = np.array([[1.0, 0.0, 0.0, 0.0]])
H_uni = np.array([[σ_v]])         # H @ H.T = 0.0001

lss_uni = qe.LinearStateSpace(A_var, C_var, G_uni, H_uni,
                               mu_0=np.zeros(4), Sigma_0=np.eye(4))
kf_uni = qe.Kalman(lss_uni)
Sigma_uni, K_uni = kf_uni.stationary_values()

print("\nSystem 2 - steady-state Kalman gain K (4x1):")
print(np.round(K_uni, 5))
```

```{code-cell} ipython3
# Covariance comparisons

# State-noise and innovation covariances
CC_prime = C_var @ C_var.T
R_biv = (σ_v**2) * np.eye(2)
innov_cov_biv = G_biv @ Sigma_biv @ G_biv.T + R_biv

print("State-noise covariance  C @ C.T  (4x4):")
print(np.round(CC_prime, 6))

print("\nSystem 1 - innovation covariance  G @ Σ @ G.T + R  (2x2):")
print(np.round(innov_cov_biv, 6))

# Steady-state covariance comparison
print("\nSystem 1 - steady-state state covariance  Σ  (4x4):")
print(np.round(Sigma_biv, 6))

print("\nSystem 2 - steady-state state covariance  Σ_check  (4x4):")
print(np.round(Sigma_uni, 6))

print("\nDifference  Σ_check - Σ  (System 2 minus System 1):")
print(np.round(Sigma_uni - Sigma_biv, 6))
```

#### Impulse responses of $y_t$ to the innovations $a_t$

We now apply the helper `y_to_a_irf` defined above to compute the ordinary
impulse responses {eq}`eq:ex_y_to_a` of the observable $y_t$ to its own
innovations $a_t$, for both System 1 (bivariate, so $a_t$ is $2 \times 1$) and
System 2 (univariate, so $u_t$ is scalar).

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: System 1 responses to own innovations
    name: fig-kfvar-sys1-ya
---
T_irf = 40
horizons = np.arange(T_irf)

Psi_biv = y_to_a_irf(A_var, K_biv, G_biv, T_irf)   # System 1: (T, 2, 2)
Psi_uni = y_to_a_irf(A_var, K_uni, G_uni, T_irf)   # System 2: (T, 1, 1)

obs_labels = [r'$r_t$', r'$z_t$']
innov_labels = [r'$a_{1,t}$', r'$a_{2,t}$']

# System 1 responses
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
for i, obs in enumerate(obs_labels):
    for j, inn in enumerate(innov_labels):
        ax = axes[i, j]
        ax.plot(horizons, Psi_biv[:, i, j], lw=2)
        ax.axhline(0, color='k', lw=0.6, ls='--')
        ax.set_title(fr'{obs} to {inn}', fontsize=9)
        if i == 1:
            ax.set_xlabel('horizon $h$')
        if j == 0:
            ax.set_ylabel('response')
fig.tight_layout()
plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: System 2 response to own innovation
    name: fig-kfvar-sys2-ya
---
fig, ax = plt.subplots()
ax.plot(horizons, Psi_uni[:, 0, 0], lw=2)
ax.axhline(0, color='k', lw=0.6, ls='--')
ax.set_xlabel('horizon $h$')
ax.set_ylabel('response')
fig.tight_layout()
plt.show()
```

At horizon $h = 0$ each innovation passes one-for-one into its own component of
$y_t$, the contemporaneous response equals the identity matrix $\Psi_0 = I$, so
the diagonal panels start at $1$ and the off-diagonal panels start at $0$.

For $h \ge 1$ the responses propagate through the state matrix $A$ and decay
geometrically, tracing out the Wold moving-average representation of the
bivariate (System 1) and univariate (System 2) processes.

#### Impulse responses of the innovations to the structural shocks

```{code-cell} ipython3
def innovation_irf(A, K, G, C, T=40):
    """
    Impulse responses of the Kalman innovation to each structural shock.

    The innovation at horizon h to a unit shock e_j is
        G (A - KG)^h C e_j,   h = 0, 1, ..., T-1.

    Returns irf of shape (T, n_a, n_w).
    """
    AKG = A - K @ G
    n_a = G.shape[0]
    n_w = C.shape[1]
    irf = np.zeros((T, n_a, n_w))
    x = C.copy()  # (A-KG)^0 @ C at h = 0
    for h in range(T):
        irf[h] = G @ x
        x = AKG @ x
    return irf

T_irf = 40
horizons = np.arange(T_irf)

irf_biv = innovation_irf(A_var, K_biv, G_biv, C_var, T_irf)   # (T, 2, 2)
irf_uni = innovation_irf(A_var, K_uni, G_uni, C_var, T_irf)   # (T, 1, 2)

shock_labels = [r'shock $w_{1,t+1}$', r'shock $w_{2,t+1}$']
innov_biv_lbl = [r'$a_{1,t}$', r'$a_{2,t}$']
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: System 1 innovation IRFs
    name: fig-kfvar-sys1-innovation-irfs
---
# System 1 innovation IRFs
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
for j, shock in enumerate(shock_labels):
    for i, albl in enumerate(innov_biv_lbl):
        ax = axes[i, j]
        ax.plot(horizons, irf_biv[:, i, j], lw=2)
        ax.axhline(0, color='k', lw=0.6, ls='--')
        ax.set_title(fr'{albl} to {shock}', fontsize=9)
        ax.set_xlabel('horizon $h$')
        if j == 0:
            ax.set_ylabel('innovation response')
fig.tight_layout()
plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: System 2 innovation IRFs
    name: fig-kfvar-sys2-innovation-irfs
---
# System 2 innovation IRFs
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for j, shock in enumerate(shock_labels):
    axes[j].plot(horizons, irf_uni[:, 0, j], lw=2)
    axes[j].axhline(0, color='k', lw=0.6, ls='--')
    axes[j].set_title(fr'$u_t$ to {shock}', fontsize=9)
    axes[j].set_xlabel('horizon $h$')
axes[0].set_ylabel('innovation response')
fig.tight_layout()
plt.show()
```

The two sets of innovation impulse responses illustrate an important point.

In **System 1** the filter uses both $r_t$ and $z_t$, so $a_t$ is $2 \times 1$ and
its response to each shock is a pair of numbers at every horizon.

In **System 2** only $r_t$ is observed, so $u_t$ is scalar and carries less
information about the two-dimensional shock vector.

Because $K$ differs across the two systems, the decay matrix $A - KG$ differs
as well, so the innovation IRFs $G(A-KG)^h C e_j$ differ across the two systems
even though the structural DGP is identical.

The covariance comparisons printed above reveal two further insights.

**State-noise $CC^\top$ vs. innovation covariance $G\Sigma G^\top + R$ (System 1).**

The matrix $CC^\top$ is the unconditional covariance contributed by the structural
shocks to the state transition in one period.

The innovation covariance $G\Sigma G^\top + R$ is the covariance of the one-step
forecast error in the *observable* vector after the Kalman filter has processed
all available information.

In this example $G C C^\top G^\top$ has diagonal entries equal to $1$.

The printed innovation covariance has diagonal entries slightly above $1$ because it also
contains residual uncertainty from estimating the hidden state and the small
measurement-error variance $R = 0.0001 I_2$.

Thus $G\Sigma G^\top + R$ should not be compared directly with $G C C^\top G^\top$ as a
strict variance reduction.

The variance reduction created by filtering is measured relative to the broader
forecast-error covariance before conditioning on the available observations.

**System 1 covariance $\Sigma$ vs. System 2 covariance $\check\Sigma$.**

Restricting observations to $r_t$ alone (System 2) yields less information about
the hidden state, so the filter is less precise: $\check\Sigma \succeq \Sigma$,
i.e., the difference $\check\Sigma - \Sigma$ is positive semi-definite.

The printed difference matrix confirms this ordering for our parameter values.

## Exercises

```{exercise-start}
:label: kf_ex1
```

Consider the scalar AR(1) state space system used above with $\rho = 0.9$,
$\sigma_w = 0.5$, $\sigma_v = 1.0$.

Derive an algebraic expression for the **steady-state** conditional variance
$\Sigma_\infty$ by solving the scalar Riccati equation {eq}`eq:riccati` at its
fixed point $\Sigma_{t+1} = \Sigma_t = \Sigma$.

Show that $\Sigma$ satisfies a quadratic equation, find its positive root, and
verify numerically that your formula matches `kf.Sigma_infinity`.

```{exercise-end}
```

```{solution-start} kf_ex1
:class: dropdown
```

Setting $\Sigma_{t+1} = \Sigma_t = \Sigma$ in the scalar version of
{eq}`eq:riccati` with $A = \rho$, $CC^\top = \sigma_w^2$, $GG^\top = 1$,
$R = \sigma_v^2$:

$$
\Sigma = \rho^2 \Sigma + \sigma_w^2 - \frac{\rho^2 \Sigma^2}{\Sigma + \sigma_v^2}
$$

Multiplying through by $\Sigma + \sigma_v^2$ and rearranging:

$$
\Sigma^2 + \left[\sigma_v^2(1-\rho^2) - \sigma_w^2\right]\Sigma
  - \sigma_v^2 \sigma_w^2 = 0
$$

Taking the positive root of this quadratic:

$$
\Sigma_\infty
  = \frac{\sigma_w^2 - \sigma_v^2(1-\rho^2)
          + \sqrt{\left[\sigma_v^2(1-\rho^2) - \sigma_w^2\right]^2
          + 4 \sigma_v^2 \sigma_w^2}}{2}
$$

```{code-cell} ipython3
ρ_, σ_w_, σ_v_ = 0.9, 0.5, 1.0

b = σ_v_**2 * (1 - ρ_**2) - σ_w_**2
discriminant = b**2 + 4 * σ_v_**2 * σ_w_**2
Sigma_formula = (-b + np.sqrt(discriminant)) / 2

A_ = np.array([[ρ_]])
C_ = np.array([[σ_w_]])
G_ = np.array([[1.0]])
R_ = np.array([[σ_v_**2]])
H_ = np.array([[σ_v_]])   # R_ = H_ @ H_.T
lss_ = qe.LinearStateSpace(A_, C_, G_, H_, mu_0=np.zeros(1), Sigma_0=np.eye(1))
kf_ = qe.Kalman(lss_)

print(f"Analytical Σ_inf   = {Sigma_formula:.8f}")
print(f"Numerical  Σ_inf   = {kf_.Sigma_infinity[0, 0]:.8f}")
```

```{solution-end}
```

```{exercise-start}
:label: kf_ex2
```

**Bivariate example.**

Consider a two-dimensional state with a one-dimensional observation:

$$
A = \begin{pmatrix} 0.9 & 0.1 \\ 0 & 0.8 \end{pmatrix}, \quad
C = \begin{pmatrix} 0.4 \\ 0.1 \end{pmatrix}, \quad
G = \begin{pmatrix} 1 & 0 \end{pmatrix}, \quad
R = [0.5]
$$

1. Simulate $T = 500$ observations from this system starting from a diffuse prior.

2. Run the Kalman filter and plot both components of $\hat{x}_t$ against the
   true hidden state path.

3. Compute and report the steady-state covariance $\Sigma_\infty$ and
   Kalman gain $K_\infty$.

4. Check that the eigenvalues of $A - K_\infty G$ lie strictly inside the
   unit circle, confirming that the VAR representation {eq}`eq:var1` is stable.

```{exercise-end}
```

```{solution-start} kf_ex2
:class: dropdown
```

```{code-cell} ipython3
A2 = np.array([[0.9, 0.1],
               [0.0, 0.8]])
C2 = np.array([[0.4],
               [0.1]])
G2 = np.array([[1.0, 0.0]])
R2 = np.array([[0.5]])
H2 = np.array([[np.sqrt(0.5)]])   # R2 = H2 @ H2.T

lss2 = qe.LinearStateSpace(A2, C2, G2, H2,
                             mu_0=np.zeros(2),
                             Sigma_0=np.eye(2) * 5.0)
kf2 = qe.Kalman(lss2)
kf2.set_state(np.zeros(2), np.eye(2) * 5.0)

T2 = 500
x2_path, y2_path = lss2.simulate(ts_length=T2, random_state=0)

x_hats2 = np.zeros((T2, 2))
for t in range(T2):
    kf2.update(y2_path[:, t])
    x_hats2[t] = kf2.x_hat.ravel()

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(x2_path[i, :T2], lw=2, label=f'true $x_{{{i+1},t}}$')
    ax.plot(x_hats2[:, i], lw=2, ls='--', label=rf'$\hat{{x}}_{{{i+1},t}}$')
    ax.set_title(f'component {i+1}')
    ax.legend(fontsize=9)
    ax.set_ylabel(f'component {i+1}')
axes[1].set_xlabel('time $t$')
fig.suptitle('Kalman filter: bivariate hidden state')
fig.tight_layout()
plt.show()

# Steady-state values
Sigma2_inf, K2_inf = kf2.stationary_values()
print("Steady-state covariance Σ_inf:")
print(np.round(Sigma2_inf, 5))
print("\nSteady-state Kalman gain K_inf:")
print(np.round(K2_inf, 5))

# Eigenvalues of A - K_inf G
AKG2 = A2 - K2_inf @ G2
eigvals2 = np.linalg.eigvals(AKG2)
print(f"\nEigenvalues of A - K_inf G: {np.round(eigvals2, 5)}")
print(f"Stable VAR: {np.all(np.abs(eigvals2) < 1)}")
```

```{solution-end}
```

```{exercise-start}
:label: kf_ex3
```

**Likelihood and parameter estimation.**

Using the scalar model from the main text with true parameters
$(\rho, \sigma_w, \sigma_v) = (0.9, 0.5, 1.0)$:

1. Simulate $T = 300$ observations.

2. Write a function that evaluates the **log-likelihood** as a function of
   $\rho \in (0, 1)$, holding $\sigma_w = 0.5$ and $\sigma_v = 1.0$ fixed,
   and plot the log-likelihood against $\rho$ for a grid of values.

3. Locate the maximum numerically and check that it is close to the true value
   $\rho = 0.9$.

```{exercise-end}
```

```{solution-start} kf_ex3
:class: dropdown
```

```{code-cell} ipython3
# True parameters
ρ_true, sw_true, sv_true = 0.9, 0.5, 1.0

A_t = np.array([[ρ_true]])
C_t = np.array([[sw_true]])
G_t = np.array([[1.0]])
R_t = np.array([[sv_true**2]])
H_t = np.array([[sv_true]])   # R_t = H_t @ H_t.T

lss_t = qe.LinearStateSpace(A_t, C_t, G_t, H_t,
                             mu_0=np.zeros(1), Sigma_0=np.eye(1))
_, y_sim = lss_t.simulate(ts_length=300, random_state=7)
y_sim = y_sim.T          # shape (300, 1)

def ll_rho(ρ_val):
    A_ = np.array([[ρ_val]])
    C_ = np.array([[sw_true]])
    G_ = np.array([[1.0]])
    R_ = np.array([[sv_true**2]])
    return log_likelihood(A_, C_, G_, R_, y_sim,
                          np.zeros(1), np.eye(1) * 10.0)

ρ_grid = np.linspace(0.5, 0.99, 60)
ll_vals = np.array([ll_rho(r) for r in ρ_grid])

ρ_mle = ρ_grid[np.argmax(ll_vals)]

fig, ax = plt.subplots()
ax.plot(ρ_grid, ll_vals, lw=2)
ax.axvline(ρ_true, color='k',   ls='--', label=f'true ρ = {ρ_true}')
ax.axvline(ρ_mle, color='C1', ls=':', label=f'MLE  ρ_hat = {ρ_mle:.3f}')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel('log-likelihood')
ax.set_title('Profile log-likelihood as a function of $\\rho$')
ax.legend()
fig.tight_layout()
plt.show()

print(f"True ρ = {ρ_true},  MLE ρ_hat = {ρ_mle:.4f}")
```

```{solution-end}
```
