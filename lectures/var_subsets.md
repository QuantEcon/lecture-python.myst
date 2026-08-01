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

(var_subsets)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Vector Autoregressions for Subsets of Variables

```{index} single: Vector Autoregression; subsystems
```

```{index} single: Kalman Filter; and vector autoregressions
```

```{contents} Contents
:depth: 2
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

## Overview

An economic model delivers a vector autoregression for a list of variables $Y_t$.

An econometrician often observes only *some* of them.

This lecture answers three questions about that situation.

Given an $m$th order VAR for an $n \times 1$ vector $Y_t$ and a selector matrix
$S_y$ that extracts an $n_y \times 1$ subvector $y_t = S_y Y_t$:

1. What vector autoregression does $y_t$ obey?
2. What is its moving average representation?
3. How are the innovations in the small system related to the innovations in the
   large one?

The answers all come from the {doc}`Kalman filter <kalman_filter_var>`.

The state is the history of $Y_t$ that a finite-order VAR requires, and the
observation is the subvector $y_t$.

Because $y_t$ is a subvector of the state and not a noisy signal about it, this
is a state space system with *no* measurement error.

The main results are

- $y_t$ obeys an **infinite-order** VAR whose coefficients we compute exactly,
- the innovation $a_t$ of the small system is a **one-sided distributed lag** of
  the innovations $\varepsilon_t$ of the large system,
- that distributed lag has a wide coefficient matrix at every lag, so
  $\varepsilon_t$ *cannot* be recovered from the history of $y_t$,
- the forecast error variance of the small system exceeds that of the large one
  by a quantity we compute.

Two special cases where the small VAR stays finite-order are identified and verified.

This lecture generalizes the worked example that used to close
{doc}`kalman_filter_var`.

Let's start with imports.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe

plt.rcParams['figure.figsize'] = (10, 5)
np.set_printoptions(precision=4, suppress=True)
```

## The large system

Let $Y_t$ be $n \times 1$ and suppose it obeys the $m$th order vector autoregression

$$
Y_{t+1} = A_1 Y_t + A_2 Y_{t-1} + \cdots + A_m Y_{t-m+1} + \varepsilon_{t+1},
\qquad
\mathbb{E}\, \varepsilon_t \varepsilon_t^\top = V ,
$$ (eq:vs_bigvar)

where $\{\varepsilon_t\}$ is a serially uncorrelated sequence with
$\mathbb{E}[\varepsilon_{t+1} \mid Y_t, Y_{t-1}, \ldots] = 0$.

Stack $m$ lags into the $nm \times 1$ state vector

$$
X_t = \begin{pmatrix} Y_t \\ Y_{t-1} \\ \vdots \\ Y_{t-m+1}\end{pmatrix} .
$$

Then {eq}`eq:vs_bigvar` becomes the first-order **companion form**

$$
X_{t+1} = A X_t + C \varepsilon_{t+1},
\qquad
A = \begin{pmatrix}
A_1 & A_2 & \cdots & A_{m-1} & A_m \\
I   & 0   & \cdots & 0       & 0 \\
0   & I   & \cdots & 0       & 0 \\
\vdots & & \ddots & & \vdots \\
0   & 0   & \cdots & I       & 0
\end{pmatrix},
\qquad
C = \begin{pmatrix} I_n \\ 0 \\ \vdots \\ 0 \end{pmatrix} .
$$ (eq:vs_companion)

We write $A$ without a subscript for the companion matrix and $A_1, \ldots, A_m$
with subscripts for the VAR coefficient matrices.

Let $J = \begin{pmatrix} I_n & 0 & \cdots & 0\end{pmatrix}$ be the $n \times nm$
matrix that reads $Y_t$ off the state, so that $Y_t = J X_t$ and $C = J^\top$.

We assume all eigenvalues of $A$ are strictly inside the unit circle, so
$\{Y_t\}$ is covariance stationary.

## The small system

The econometrician observes

$$
y_t = S_y Y_t = G X_t,
\qquad
G = S_y J ,
$$ (eq:vs_obs)

where $S_y$ is $n_y \times n$ with $n_y < n$.

Usually $S_y$ picks out $n_y$ coordinates of $Y_t$, but nothing below requires
that, so linear combinations are allowed too.

Equations {eq}`eq:vs_companion` and {eq}`eq:vs_obs` are a state space system of the
form studied in {doc}`kalman_filter_var`, with shock loading $C$, observation
matrix $G$, and measurement error covariance

$$
R = 0 .
$$

The observation is *exact*, but it is a strict subvector of the state, so the
econometrician still faces a filtering problem.

Let $\Sigma$ solve the steady-state Riccati equation

$$
\Sigma = A \Sigma A^\top + C V C^\top
         - A \Sigma G^\top \bigl(G \Sigma G^\top\bigr)^{-1} G \Sigma A^\top
$$ (eq:vs_riccati)

with associated Kalman gain

$$
K = A \Sigma G^\top \bigl(G \Sigma G^\top\bigr)^{-1} .
$$ (eq:vs_gain)

Here $\Sigma$ is the covariance matrix of $X_t - \mathbb{E}[X_t \mid y^{t-1}]$.

```{note}
Because $X_t$ contains lags of $Y_t$ whose $S_y$ components the econometrician
has already seen exactly, $\Sigma$ is singular.

That is harmless.

What the Kalman gain {eq}`eq:vs_gain` requires is that the *innovation*
covariance $G \Sigma G^\top$ be nonsingular, which holds whenever no linear
combination of $y_t$ is perfectly predictable from $y^{t-1}$.
```

The innovation in the small system is

$$
a_t = y_t - \mathbb{E}[y_t \mid y^{t-1}] = G\bigl(X_t - \hat X_t\bigr),
\qquad
\Omega \equiv \mathbb{E}\, a_t a_t^\top = G \Sigma G^\top .
$$ (eq:vs_innov)

## Four representations

### The Wold representation

The steady-state innovations representation derived in {doc}`kalman_filter_var` is

$$
\hat X_{t+1} = A \hat X_t + K a_t,
\qquad
y_t = G \hat X_t + a_t .
$$ (eq:vs_innovrep)

Solving {eq}`eq:vs_innovrep` forward gives the moving average representation of
$y_t$ in terms of its own innovations,

$$
y_t = \sum_{h=0}^{\infty} \Psi_h\, a_{t-h},
\qquad
\Psi_0 = I_{n_y},
\qquad
\Psi_h = G A^{h-1} K \quad (h \geq 1) .
$$ (eq:vs_wold)

### The vector autoregression

Solving {eq}`eq:vs_innovrep` backward instead gives

$$
y_t = \sum_{j=1}^{\infty} B_j\, y_{t-j} + a_t,
\qquad
B_j = G (A - KG)^{j-1} K .
$$ (eq:vs_var)

This is an infinite-order VAR, convergent because the eigenvalues of $A - KG$
lie inside the unit circle.

### Innovations of the small system in terms of the large one

This is the question that motivates the lecture.

Let $e_t = X_t - \hat X_t$ be the filtering error.

Subtracting the Kalman recursion $\hat X_{t+1} = A \hat X_t + K a_t$ from the
state equation {eq}`eq:vs_companion` and using $a_t = G e_t$ gives

$$
e_{t+1} = (A - KG) e_t + C \varepsilon_{t+1} .
$$ (eq:vs_error)

Solving {eq}`eq:vs_error` backward and premultiplying by $G$ yields the answer.

```{prf:proposition}
:label: vs_prop_innov

The innovations of the small system are the one-sided distributed lag

$$
a_t = \sum_{j=0}^{\infty} \Gamma_j\, \varepsilon_{t-j},
\qquad
\Gamma_j = G (A - KG)^j C ,
$$ (eq:vs_gamma)

of the innovations of the large system, with leading coefficient

$$
\Gamma_0 = G C = S_y J J^\top = S_y .
$$
```

Since each $\Gamma_j$ is $n_y \times n$ with $n_y < n$, the map from
$\{\varepsilon_t\}$ to $\{a_t\}$ has no inverse.

Knowing the entire history of $y_t$ is not enough to recover $\varepsilon_t$.

### The structural moving average

For comparison, iterating {eq}`eq:vs_companion` gives $y_t$ directly in terms of
the large system's innovations,

$$
y_t = \sum_{j=0}^{\infty} \Phi_j\, \varepsilon_{t-j},
\qquad
\Phi_j = G A^j C .
$$ (eq:vs_phi)

### A forecast error that contains past shocks

Separating the $j = 0$ term in {eq}`eq:vs_gamma` and using $\Gamma_0 = S_y$ gives

$$
a_t = \underbrace{S_y \varepsilon_t}_{\text{full-information forecast error}}
    \; + \;
      \underbrace{\sum_{j=1}^{\infty} \Gamma_j\, \varepsilon_{t-j}}_{\text{shocks realized before } t} .
$$ (eq:vs_split)

The first term is what it appears to be.

Because $\mathbb{E}[\varepsilon_t \mid Y^{t-1}] = 0$, we have
$y_t - \mathbb{E}[y_t \mid Y^{t-1}] = S_y \varepsilon_t$, so $S_y \varepsilon_t$
is the error made in forecasting $y_t$ by someone who observes the *entire*
history of $Y$.

The second term deserves a pause, because at first sight it looks impossible.

By construction $a_t$ is a forecast error, orthogonal to everything known at
$t-1$.

Yet {eq}`eq:vs_split` says that $a_t$ loads on $\varepsilon_{t-1},
\varepsilon_{t-2}, \ldots$, shocks that had already been realized by then.

The two facts are consistent, and both are true:

$$
\mathbb{E}\, a_t\, \varepsilon_{t-j}^\top = \Gamma_j V \neq 0
\quad (j \geq 1),
\qquad \text{while} \qquad
\mathbb{E}\, a_t\, y_{t-k}^\top = 0
\quad (k \geq 1) .
$$ (eq:vs_orth)

The resolution is that past shocks are known to the *large* system's
econometrician, not to the small one.

The small econometrician's information set is $H(y^{t-1})$, the closed linear
span of $y_{t-1}, y_{t-2}, \ldots$, and $\varepsilon_{t-j}$ does not lie in it.

Let $P_{t-1}$ denote projection onto $H(y^{t-1})$.

Applying $P_{t-1}$ to {eq}`eq:vs_split`, using $P_{t-1} a_t = 0$ and
$P_{t-1}\varepsilon_t = 0$, delivers the identity

$$
\sum_{j=1}^{\infty} \Gamma_j\, P_{t-1}\varepsilon_{t-j} = 0 .
$$ (eq:vs_pred)

So the distributed lag in {eq}`eq:vs_split` loads only on the parts of past
shocks that the small econometrician has *not yet* learned,

$$
a_t = S_y \varepsilon_t
    + \sum_{j=1}^{\infty} \Gamma_j
      \bigl(\varepsilon_{t-j} - P_{t-1}\varepsilon_{t-j}\bigr) .
$$ (eq:vs_unlearned)

A shock that happened three quarters ago can still be news today, if the only
series you watch has not finished revealing it.

That is the whole content of the filtering problem, and it is why $\Omega$
exceeds $S_y V S_y^\top$.

### What the coefficients $\Gamma_j$ are

Substituting the moving average {eq}`eq:vs_phi` into the autoregression
{eq}`eq:vs_var` and matching the coefficient on $\varepsilon_{t-k}$ gives a
second formula for the same objects,

$$
\Gamma_k = \Phi_k - \sum_{j=1}^{k} B_j\, \Phi_{k-j} .
$$ (eq:vs_gamma_alt)

So $\Gamma_k$ measures the failure of the small system's own autoregression to
reproduce the large system's $k$-lag response.

The case $k = 1$ is worth writing out.

Since $\Phi_0 = S_y$ and $\Phi_1 = G A C = S_y A_1$,

$$
\Gamma_1 = S_y A_1 - B_1 S_y .
$$ (eq:vs_gamma1)

Suppose $S_y$ selects coordinates, and partition
$Y_t = (y_t^\top, \tilde y_t^\top)^\top$ as before.

Then $B_1 S_y$ has zeros in the columns belonging to the dropped variables, so
{eq}`eq:vs_gamma1` reads

$$
\Gamma_1 = \begin{pmatrix} A_1^{yy} - B_1 & A_1^{y \tilde y}\end{pmatrix} .
$$ (eq:vs_gamma1_block)

The loading of $a_t$ on last period's *omitted* shocks is exactly
$A_1^{y\tilde y}$, the block through which the omitted variables enter the
retained equations.

Equation {eq}`eq:vs_gamma1_block` also previews
{prf:ref}`vs_prop_blockexog`: block exogeneity sets $A_1^{y\tilde y} = 0$, which
forces $B_1 = A_1^{yy}$ and hence $\Gamma_1 = 0$.

### A factorization identity

Representations {eq}`eq:vs_wold`, {eq}`eq:vs_gamma`, and {eq}`eq:vs_phi` describe
the same process, so with $\Psi(z) = \sum_h \Psi_h z^h$ and similarly for
$\Gamma$ and $\Phi$,

$$
\Phi(z) = \Psi(z)\, \Gamma(z) .
$$ (eq:vs_factor)

Equation {eq}`eq:vs_factor` says the structural moving average operator factors
into the Wold operator of the small system times the innovation map.

It is a sharp numerical check on everything above, and we use it as one.

Matching variances at each lag also gives

$$
\Omega = \sum_{j=0}^{\infty} \Gamma_j V \Gamma_j^\top
       = S_y V S_y^\top + \sum_{j=1}^{\infty} \Gamma_j V \Gamma_j^\top .
$$ (eq:vs_varloss)

Every term in the second sum is positive semidefinite, so

$$
\Omega \succeq S_y V S_y^\top .
$$

The small system's one-step forecast error variance is never smaller than the
corresponding block of the large system's, and {eq}`eq:vs_varloss` says exactly
how much is lost.

## Two cases where nothing is lost

```{prf:proposition}
:label: vs_prop_full

If $S_y = I_n$, then $\Gamma_0 = I_n$ and $\Gamma_j = 0$ for $j \geq 1$, so
$a_t = \varepsilon_t$, and {eq}`eq:vs_var` collapses to the original $m$th order
VAR {eq}`eq:vs_bigvar`.
```

Nothing is hidden, so the Wold innovations *are* the structural innovations.

The second case is more interesting.

Partition $Y_t = (y_t^\top, \tilde y_t^\top)^\top$ and correspondingly

$$
A_k = \begin{pmatrix} A_k^{yy} & A_k^{y\tilde y} \\
                      A_k^{\tilde y y} & A_k^{\tilde y \tilde y}\end{pmatrix},
\qquad k = 1, \ldots, m .
$$

```{prf:proposition}
:label: vs_prop_blockexog

Suppose $y_t$ is **block exogenous**, meaning $A_k^{y\tilde y} = 0$ for all $k$,
so that no lag of the omitted variables appears in the equations for $y$.

Then $\Gamma_j = 0$ for $j \geq 1$, $a_t = S_y \varepsilon_t$,
$\Omega = S_y V S_y^\top$, and $y_t$ obeys the finite $m$th order VAR

$$
y_t = \sum_{k=1}^{m} A_k^{yy}\, y_{t-k} + a_t .
$$
```

```{prf:proof}
Block exogeneity makes the $y$ rows of {eq}`eq:vs_bigvar` read
$y_{t+1} = \sum_k A_k^{yy} y_{t+1-k} + S_y \varepsilon_{t+1}$.

The right side involves only lags of $y$, and $S_y \varepsilon_{t+1}$ is
orthogonal to the whole history $Y^t$ and hence to $y^t$.

So this *is* the projection of $y_{t+1}$ on $y^t$, which identifies it as the
Wold representation.
```

Note what {prf:ref}`vs_prop_blockexog` does *not* require: $V$ need not be block
diagonal.

Contemporaneous correlation between $\varepsilon^y$ and $\varepsilon^{\tilde y}$
is fine.

What matters for whether an omitted variable damages a VAR is Granger causality,
not contemporaneous correlation.

## Code

The class below packages everything.

```{code-cell} ipython3
class VARSubsystem:
    """
    A VAR for Y and the implied representations for a subvector y = S_y Y.

        Y[t+1] = A_1 Y[t] + ... + A_m Y[t-m+1] + eps[t+1],  E eps eps' = V
        y[t]   = S_y Y[t]

    Parameters
    ----------
    A_list : list of (n, n) arrays, the VAR coefficient matrices A_1, ..., A_m
    V      : (n, n) array, covariance matrix of eps
    S_y    : (n_y, n) selector matrix
    """

    def __init__(self, A_list, V, S_y):
        self.A_list = [np.atleast_2d(np.asarray(a, dtype=float)) for a in A_list]
        self.V = np.atleast_2d(np.asarray(V, dtype=float))
        self.S_y = np.atleast_2d(np.asarray(S_y, dtype=float))
        n, m = self.A_list[0].shape[0], len(self.A_list)
        self.n, self.m, self.n_y = n, m, self.S_y.shape[0]

        # companion form
        self.A = np.zeros((n * m, n * m))
        self.A[:n] = np.hstack(self.A_list)
        if m > 1:
            self.A[n:, :n * (m - 1)] = np.eye(n * (m - 1))
        self.J = np.zeros((n, n * m))
        self.J[:, :n] = np.eye(n)
        self.C = self.J.T
        self.G = self.S_y @ self.J
        self.Q = self.C @ self.V @ self.C.T
        self._Sigma = self._K = None

    def companion_eigenvalues(self):
        return np.linalg.eigvals(self.A)

    def stationary_filter(self):
        """Steady-state (Sigma, K) from the Riccati equation with R = 0."""
        if self._Sigma is None:
            A, G = self.A, self.G
            R = np.zeros((self.n_y, self.n_y))
            Sigma = qe.solve_discrete_riccati(A.T, G.T, self.Q, R)
            Omega = G @ Sigma @ G.T
            self._Sigma = Sigma
            self._K = A @ Sigma @ G.T @ np.linalg.inv(Omega)
        return self._Sigma, self._K

    def innovation_cov(self):
        """Omega = E a a', the one-step forecast error covariance of y."""
        Sigma, _ = self.stationary_filter()
        return self.G @ Sigma @ self.G.T

    def wold(self, h_max=20):
        """Psi[h] in y[t] = sum_h Psi[h] a[t-h]; Psi[0] = I."""
        _, K = self.stationary_filter()
        Psi = np.empty((h_max + 1, self.n_y, self.n_y))
        Psi[0], P = np.eye(self.n_y), np.eye(self.A.shape[0])
        for h in range(1, h_max + 1):
            Psi[h] = self.G @ P @ K
            P = P @ self.A
        return Psi

    def var_coefficients(self, h_max=20):
        """B[j-1] in y[t] = sum_j B[j] y[t-j] + a[t], for j = 1, ..., h_max."""
        _, K = self.stationary_filter()
        M = self.A - K @ self.G
        B, P = np.empty((h_max, self.n_y, self.n_y)), np.eye(self.A.shape[0])
        for j in range(h_max):
            B[j] = self.G @ P @ K
            P = P @ M
        return B

    def innovation_map(self, h_max=20):
        """Gamma[j] in a[t] = sum_j Gamma[j] eps[t-j]."""
        _, K = self.stationary_filter()
        M = self.A - K @ self.G
        Gamma, P = np.empty((h_max + 1, self.n_y, self.n)), np.eye(self.A.shape[0])
        for j in range(h_max + 1):
            Gamma[j] = self.G @ P @ self.C
            P = P @ M
        return Gamma

    def structural_ma(self, h_max=20):
        """Phi[j] in y[t] = sum_j Phi[j] eps[t-j]."""
        Phi, P = np.empty((h_max + 1, self.n_y, self.n)), np.eye(self.A.shape[0])
        for j in range(h_max + 1):
            Phi[j] = self.G @ P @ self.C
            P = P @ self.A
        return Phi

    def simulate(self, T, seed=0, burn=200):
        """Simulate Y and the innovations eps that generated it."""
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(self.V)
        eps = rng.standard_normal((T + burn, self.n)) @ L.T
        X, Y = np.zeros(self.n * self.m), np.zeros((T + burn, self.n))
        for t in range(T + burn):
            X = self.A @ X + self.C @ eps[t]
            Y[t] = self.J @ X
        return Y[burn:], eps[burn:]

    def filter_innovations(self, y_path):
        """Recover a[t] from observed y by running the steady-state filter."""
        _, K = self.stationary_filter()
        x_hat = np.zeros(self.A.shape[0])
        a = np.empty((len(y_path), self.n_y))
        for t in range(len(y_path)):
            a[t] = y_path[t] - self.G @ x_hat
            x_hat = self.A @ x_hat + K @ a[t]
        return a


def convolve(Psi, Gamma, h_max):
    """(Psi * Gamma)[h] = sum_{k=0}^{h} Psi[k] Gamma[h-k]."""
    out = np.zeros((h_max + 1, Psi.shape[1], Gamma.shape[2]))
    for h in range(h_max + 1):
        for k in range(h + 1):
            out[h] += Psi[k] @ Gamma[h - k]
    return out
```

A single routine collects the diagnostics we want to see for every example.

```{code-cell} ipython3
def report(model, h_max=30, label=''):
    """Print the identities that every subsystem must satisfy."""
    Sigma, K = model.stationary_filter()
    A, G, V = model.A, model.G, model.V
    resid = Sigma - (A @ Sigma @ A.T + model.Q
                     - A @ Sigma @ G.T @ np.linalg.inv(G @ Sigma @ G.T)
                       @ G @ Sigma @ A.T)
    Psi = model.wold(h_max)
    Gamma = model.innovation_map(h_max)
    Phi = model.structural_ma(h_max)
    Omega, Vy = model.innovation_cov(), model.S_y @ V @ model.S_y.T

    print(f'--- {label} (n = {model.n}, m = {model.m}, n_y = {model.n_y})')
    print(f'  max |eig| of companion A      {np.max(abs(model.companion_eigenvalues())):.6f}')
    print(f'  max |eig| of A - KG           {np.max(abs(np.linalg.eigvals(A - K @ G))):.6f}')
    print(f'  Riccati residual              {np.abs(resid).max():.2e}')
    print(f'  |Gamma[0] - S_y|              {np.abs(Gamma[0] - model.S_y).max():.2e}')
    print(f'  |Phi - Psi * Gamma|           '
          f'{np.abs(convolve(Psi, Gamma, h_max) - Phi).max():.2e}')
    Gamma_long = model.innovation_map(300)      # the sum in (SS) is infinite
    print(f'  |Omega - sum Gamma V Gamma\'|  '
          f'{np.abs(Omega - sum(Gamma_long[j] @ V @ Gamma_long[j].T for j in range(301))).max():.2e}')
    print(f'  max |Gamma[j]|, j >= 1        {np.abs(Gamma[1:]).max():.3e}')
    print(f'  det Omega / det S_y V S_y\'    '
          f'{np.linalg.det(Omega) / np.linalg.det(Vy):.4f}')
    return Psi, Gamma, Phi
```

## Example 1: a bivariate VAR(2)

Two observable series $r_t$ and $z_t$ obey the VAR(2)

$$
\begin{pmatrix} r_{t+1} \\ z_{t+1}\end{pmatrix}
= A_1 \begin{pmatrix} r_t \\ z_t \end{pmatrix}
+ A_2 \begin{pmatrix} r_{t-1} \\ z_{t-1}\end{pmatrix}
+ \varepsilon_{t+1},
$$

with

$$
A_1 = \begin{pmatrix} 0.80 & 0.75 \\ 0 & 0.75 \end{pmatrix},
\qquad
A_2 = \begin{pmatrix} 0.05 & -0.72 \\ 0 & 0.20 \end{pmatrix},
\qquad
V = I_2 .
$$

Note that $z$ is block exogenous: its equation contains no lags of $r$.

But $r$ is *not*, since $z$ enters the $r$ equation with both lags.

So dropping $z$ should matter, while dropping $r$ should not.

```{code-cell} ipython3
A1 = np.array([[0.80,  0.75],
               [0.00,  0.75]])
A2 = np.array([[0.05, -0.72],
               [0.00,  0.20]])
V2 = np.eye(2)

S_both = np.eye(2)                    # observe (r, z)
S_r    = np.array([[1.0, 0.0]])       # observe r only
S_z    = np.array([[0.0, 1.0]])       # observe z only

mod_both = VARSubsystem([A1, A2], V2, S_both)
mod_r    = VARSubsystem([A1, A2], V2, S_r)
mod_z    = VARSubsystem([A1, A2], V2, S_z)

Psi_both, Gam_both, _ = report(mod_both, label='observe r and z')
print()
Psi_r, Gam_r, _ = report(mod_r, label='observe r only')
print()
Psi_z, Gam_z, _ = report(mod_z, label='observe z only')
```

Every identity holds to machine precision.

The three cases differ exactly as the propositions predict.

Observing both variables gives $\Gamma_j = 0$ for $j \geq 1$, so
$a_t = \varepsilon_t$, as {prf:ref}`vs_prop_full` requires.

Observing only $z$, which is block exogenous, also gives $\Gamma_j = 0$ for
$j \geq 1$, so $a_t = \varepsilon_{z,t}$, as {prf:ref}`vs_prop_blockexog`
requires.

Observing only $r$ is different.

Here $\Gamma_j \neq 0$ for $j \geq 1$, and the ratio of forecast error variances
reports how much the $r$-only econometrician loses.

```{code-cell} ipython3
print('observe r only:')
print(f'  Omega           = {mod_r.innovation_cov()[0, 0]:.4f}')
print(f'  S_y V S_y\'      = {(S_r @ V2 @ S_r.T)[0, 0]:.4f}')
print('\n  Gamma[j] for j = 0, ..., 5 (rows: response of a to eps_r, eps_z)')
print(Gam_r[:6, 0, :])
```

The forecast error variance is over 50 percent larger than $V_{11}$.

The extra variance is entirely attributable to past $\varepsilon_z$ shocks that
the econometrician sees only through their effect on $r$.

### The VAR for the subsystem

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: VAR coefficients for the subsystem
    name: fig-vs-varcoef
---
B_r = mod_r.var_coefficients(12)

fig, ax = plt.subplots()
ax.stem(np.arange(1, 13), B_r[:, 0, 0], basefmt=' ')
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel('lag $j$')
ax.set_ylabel('$B_j$')
ax.set_title(r'Population VAR coefficients for $r_t$ when only $r$ is observed')
fig.tight_layout()
plt.show()

print('B_1, ..., B_6:', np.round(B_r[:6, 0, 0], 5))
print('\nfor comparison, A_1[0,0] and A_2[0,0]:', A1[0, 0], A2[0, 0])
```

The infinite-order VAR for $r$ alone is dominated by two lags, but neither
coefficient equals the corresponding coefficient in the bivariate system.

Dropping $z$ does not simply delete the $z$ columns of the VAR; it changes what
is left.

### Wold impulse responses

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Wold responses when both variables are observed
    name: fig-vs-wold-both
---
H = 25
h = np.arange(H + 1)
Psi_both = mod_both.wold(H)

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
names, shocks = [r'$r_t$', r'$z_t$'], [r'$a_{r,t}$', r'$a_{z,t}$']
for i in range(2):
    for j in range(2):
        axes[i, j].plot(h, Psi_both[:, i, j], lw=2)
        axes[i, j].axhline(0, color='k', lw=0.6, ls='--')
        axes[i, j].set_title(f'{names[i]} to {shocks[j]}', fontsize=10)
        if i == 1:
            axes[i, j].set_xlabel('horizon $h$')
fig.suptitle('Wold responses, both variables observed')
fig.tight_layout()
plt.show()
```

Because $a_t = \varepsilon_t$ here, these Wold responses coincide with the
structural responses of the bivariate VAR.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Wold response when only r is observed
    name: fig-vs-wold-r
---
Psi_r = mod_r.wold(H)
Phi_r = mod_r.structural_ma(H)

fig, ax = plt.subplots()
ax.plot(h, Psi_r[:, 0, 0], lw=2, label=r'$\Psi_h$: $r_t$ to its own innovation $a_t$')
ax.plot(h, Phi_r[:, 0, 0], lw=2, ls='--',
        label=r'$\Phi_h$: $r_t$ to $\varepsilon_{r,t}$')
ax.plot(h, Phi_r[:, 0, 1], lw=2, ls=':',
        label=r'$\Phi_h$: $r_t$ to $\varepsilon_{z,t}$')
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel('horizon $h$')
ax.set_ylabel('response')
ax.set_title(r'Wold versus structural responses of $r_t$')
ax.legend()
fig.tight_layout()
plt.show()
```

The Wold response to $a_t$ is not the response to either structural shock.

It is a blend, and {eq}`eq:vs_factor` says exactly how the blending works.

### Checking the innovation map by simulation

Proposition {prf:ref}`vs_prop_innov` is a statement about population objects.

We check it on a long simulated sample by comparing the innovations that the
Kalman filter extracts from the observed $r$ series with the distributed lag
$\sum_j \Gamma_j \varepsilon_{t-j}$ built from the shocks that generated it.

```{code-cell} ipython3
T = 100_000
Y_sim, eps_sim = mod_r.simulate(T, seed=1)
y_sim = Y_sim @ S_r.T

a_filtered = mod_r.filter_innovations(y_sim)

J_lag = 400          # the distributed lag is infinite, so truncate generously
Gam_long = mod_r.innovation_map(J_lag)
a_theory = np.zeros_like(a_filtered)
for j in range(J_lag + 1):
    a_theory[j:] += eps_sim[:T - j] @ Gam_long[j].T

burn = J_lag + 1
print(f'correlation of the two series   '
      f'{np.corrcoef(a_filtered[burn:, 0], a_theory[burn:, 0])[0, 1]:.8f}')
print(f'max absolute difference         '
      f'{np.abs(a_filtered[burn:] - a_theory[burn:]).max():.2e}')
print(f'sample variance of a            {a_filtered[burn:, 0].var():.4f}')
print(f'population Omega                {mod_r.innovation_cov()[0, 0]:.4f}')
print(f'sample corr(a_t, a_(t-1))       '
      f'{np.corrcoef(a_filtered[burn + 1:, 0], a_filtered[burn:-1, 0])[0, 1]:.4f}')
```

The two constructions of $a_t$ agree to the accuracy of the truncated
distributed lag, the sample variance of $a_t$ matches $\Omega$, and $a_t$ is
serially uncorrelated.

### The two orthogonality facts

Now we check {eq}`eq:vs_orth` directly, by running two regressions on the
simulated data.

The first regresses $a_t$ on current and lagged *structural* shocks, which the
small econometrician cannot see.

The second regresses $a_t$ on lagged *observations*, which are all that the small
econometrician can see.

```{code-cell} ipython3
P_lags = 4
Z_eps = np.column_stack([eps_sim[P_lags - l:T - l] for l in range(P_lags + 1)])
target = a_filtered[P_lags:, 0]
b_eps = np.linalg.lstsq(Z_eps, target, rcond=None)[0].reshape(P_lags + 1, 2)
fit = Z_eps @ b_eps.ravel()
r2_eps = 1 - ((target - fit) ** 2).sum() / target.var() / len(target)

print('OLS of a_t on eps_t, ..., eps_{t-4}:')
print(b_eps)
print('population Gamma_0, ..., Gamma_4:')
print(Gam_long[:P_lags + 1, 0, :])
print(f'max discrepancy {np.abs(b_eps - Gam_long[:P_lags + 1, 0, :]).max():.2e}')
print(f'R^2 = {r2_eps:.6f}')
```

```{code-cell} ipython3
Q_lags = 8
Z_y = np.column_stack([y_sim[Q_lags - l - 1:T - l - 1, 0] for l in range(Q_lags)])
tgt = a_filtered[Q_lags:, 0]
b_y = np.linalg.lstsq(Z_y, tgt, rcond=None)[0]
resid = tgt - Z_y @ b_y
r2_y = 1 - (resid ** 2).sum() / tgt.var() / len(tgt)

print('OLS of a_t on y_{t-1}, ..., y_{t-8}:')
print(np.round(b_y, 5))
print(f'R^2 = {r2_y:.6f}')
```

The first regression recovers the $\Gamma_j$ and fits almost perfectly, the
small shortfall coming only from truncating the distributed lag at four lags.

So $a_t$ really is built out of shocks stretching back before $t$.

The second explains essentially nothing, confirming that $a_t$ is nonetheless
orthogonal to the small econometrician's information set.

We can see why by asking how much of each past shock the small econometrician has
managed to learn.

```{code-cell} ipython3
print('R^2 from projecting a structural shock on y_{t-1}, ..., y_{t-8}')
for lag in [0, 1, 2, 3]:
    r2s = []
    for k in range(2):
        shock = eps_sim[Q_lags - lag:T - lag, k]
        c = np.linalg.lstsq(Z_y, shock, rcond=None)[0]
        e = shock - Z_y @ c
        r2s.append(1 - (e ** 2).sum() / shock.var() / len(shock))
    print(f'   eps_(t-{lag}):   eps_r {r2s[0]:6.4f}    eps_z {r2s[1]:6.4f}')
```

The current shock $\varepsilon_t$ is entirely unpredictable from $y^{t-1}$, as it
must be.

Shocks from two or more periods back are substantially learned.

The interesting row is $\varepsilon_{t-1}$: its $r$ component is largely known,
while its $z$ component is *completely* unknown, because $z_{t-1}$ reaches $r$
only with a one-period lag and so has not yet shown up anywhere in $y^{t-1}$.

That is why {eq}`eq:vs_gamma1_block` gives $a_t$ a loading on
$\varepsilon_{z,t-1}$ equal to the full structural coefficient
$A_1^{y\tilde y} = 0.75$, while its loading on $\varepsilon_{r,t-1}$ is only the
much smaller residual $A_1^{yy} - B_1$.

```{code-cell} ipython3
B1 = mod_r.var_coefficients(1)[0]
print(f'Gamma_1                  = {Gam_long[1, 0, :]}')
print(f'[A1[0,0] - B_1,  A1[0,1]] = '
      f'[{A1[0, 0] - B1[0, 0]:.4f}, {A1[0, 1]:.4f}]')

Phi_r = mod_r.structural_ma(6)
B_r6 = mod_r.var_coefficients(6)
recursion = np.array([Phi_r[k] - sum(B_r6[j - 1] @ Phi_r[k - j]
                                     for j in range(1, k + 1))
                      for k in range(7)])
print(f'\nmax |Gamma_k - (Phi_k - sum_j B_j Phi_(k-j))| = '
      f'{np.abs(recursion - Gam_long[:7]).max():.2e}')
```

## Example 2: an omitted interest rate

Now a trivariate VAR(1) in output growth $g_t$, inflation $\pi_t$, and an
interest rate $i_t$, from which the econometrician drops $i_t$.

We contrast two coefficient matrices that differ *only* in whether the interest
rate feeds back onto $g$ and $\pi$.

$$
A_1^{\text{exog}} =
\begin{pmatrix}
0.60 & 0.10 & 0.00 \\
0.15 & 0.55 & 0.00 \\
0.30 & 0.40 & 0.70
\end{pmatrix},
\qquad
A_1^{\text{fb}} =
\begin{pmatrix}
0.60 & 0.10 & -0.35 \\
0.15 & 0.55 &  0.25 \\
0.30 & 0.40 &  0.70
\end{pmatrix} .
$$

The shock covariance matrix $V$ is the same in both, and it is *not* diagonal, so
the interest rate innovation is contemporaneously correlated with the other two.

```{code-cell} ipython3
V3 = np.array([[0.36, 0.05, 0.02],
               [0.05, 0.25, 0.06],
               [0.02, 0.06, 0.16]])
S_gpi = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])

A_exog = np.array([[0.60, 0.10, 0.00],
                   [0.15, 0.55, 0.00],
                   [0.30, 0.40, 0.70]])
A_fb = np.array([[0.60, 0.10, -0.35],
                 [0.15, 0.55,  0.25],
                 [0.30, 0.40,  0.70]])

mod_exog = VARSubsystem([A_exog], V3, S_gpi)
mod_fb = VARSubsystem([A_fb], V3, S_gpi)

_, Gam_exog, _ = report(mod_exog, label='i is block exogenous')
print()
_, Gam_fb, _ = report(mod_fb, label='i feeds back')
```

The block exogenous case behaves exactly as {prf:ref}`vs_prop_blockexog` says it
must, despite the correlated shocks.

The feedback case does not.

```{code-cell} ipython3
print('block exogenous: B_1 versus the (g, pi) block of A_1')
print(mod_exog.var_coefficients(3)[0])
print(A_exog[:2, :2])
print(f'  max |B_j| for j >= 2: {np.abs(mod_exog.var_coefficients(12)[1:]).max():.2e}')

print('\nfeedback: B_1 versus the (g, pi) block of A_1')
print(mod_fb.var_coefficients(3)[0])
print(A_fb[:2, :2])
print(f'  max |B_j| for j >= 2: {np.abs(mod_fb.var_coefficients(12)[1:]).max():.4f}')
```

With block exogeneity the subsystem VAR is *exactly* the corresponding block of
the large VAR, and it stops at one lag.

With feedback the one-lag coefficients are distorted and higher-order terms
appear.

The effect on the coefficient of lagged inflation in the output growth equation
is worth noticing: a positive number in the large system becomes a negative one
in the subsystem.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Innovation map coefficients with and without feedback
    name: fig-vs-gamma
---
J_max = 10
labels = [r'$\varepsilon_g$', r'$\varepsilon_\pi$', r'$\varepsilon_i$']
fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
for col, (Gam, ttl) in enumerate([(Gam_exog, 'block exogenous'),
                                  (Gam_fb, 'feedback')]):
    for row, obs in enumerate([r'$a_g$', r'$a_\pi$']):
        ax = axes[row, col]
        for k in range(3):
            ax.plot(np.arange(J_max + 1), Gam[:J_max + 1, row, k],
                    marker='o', ms=3, lw=1.5, label=labels[k])
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(f'{obs}, {ttl}', fontsize=10)
        if row == 1:
            ax.set_xlabel('lag $j$')
        if row == 0 and col == 0:
            ax.legend(fontsize=8)
axes[0, 0].set_ylabel(r'$\Gamma_j$')
axes[1, 0].set_ylabel(r'$\Gamma_j$')
fig.suptitle(r'Coefficients $\Gamma_j$ in $a_t = \sum_j \Gamma_j \varepsilon_{t-j}$')
fig.tight_layout()
plt.show()
```

In the left column only the $j = 0$ coefficients are nonzero, and they equal the
rows of $S_y$.

In the right column the omitted interest rate shock $\varepsilon_i$ leaks into
the observed innovations at every lag.

## Summary

A finite-order VAR for $Y_t$ implies, for any subvector $y_t = S_y Y_t$, an
infinite-order VAR whose coefficients $B_j = G(A - KG)^{j-1}K$ come from the
steady-state Kalman filter for the companion system.

The innovations of the small system are a one-sided distributed lag
$a_t = \sum_j \Gamma_j \varepsilon_{t-j}$ of the innovations of the large system,
with $\Gamma_j = G(A - KG)^j C$ and $\Gamma_0 = S_y$.

Because $\Gamma_j$ is wide, that map cannot be inverted, which is a precise
statement of the informational deficiency of a subsystem VAR.

The price is measured by $\Omega - S_y V S_y^\top = \sum_{j \geq 1} \Gamma_j V \Gamma_j^\top$.

The price is zero when everything is observed, and also when the retained block
is block exogenous, in which case the subsystem VAR is exactly the corresponding
block of the original one.

## Exercises

```{exercise-start}
:label: vs_ex1
```

Take the trivariate system of Example 2 and put the feedback of the interest rate
onto $(g, \pi)$ under your control by writing

$$
A_1(\theta) = A_1^{\text{exog}} + \theta \begin{pmatrix} 0 & 0 & -0.35 \\
0 & 0 & 0.25 \\ 0 & 0 & 0 \end{pmatrix} .
$$

For $\theta$ on a grid from $0$ to $1.5$, plot

1. $\det \Omega / \det(S_y V S_y^\top)$, the information lost by dropping $i_t$,
2. $\max_{j \geq 1} |\Gamma_j|$, the size of the leakage of past shocks into the
   observed innovations.

Explain the shape you find at $\theta = 0$.

```{exercise-end}
```

```{solution-start} vs_ex1
:class: dropdown
```

Here is one solution:

```{code-cell} ipython3
E = np.zeros((3, 3))
E[0, 2], E[1, 2] = -0.35, 0.25

thetas = np.linspace(0, 1.5, 31)
det_ratio, leak = [], []
for th in thetas:
    mod = VARSubsystem([A_exog + th * E], V3, S_gpi)
    Om = mod.innovation_cov()
    det_ratio.append(np.linalg.det(Om)
                     / np.linalg.det(S_gpi @ V3 @ S_gpi.T))
    leak.append(np.abs(mod.innovation_map(40)[1:]).max())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(thetas, det_ratio, lw=2)
axes[0].set(xlabel=r'$\theta$', ylabel='determinant ratio',
            title='information lost by dropping $i_t$')
axes[1].plot(thetas, leak, lw=2, color='C1')
axes[1].set(xlabel=r'$\theta$', ylabel=r'$\max_{j \geq 1} |\Gamma_j|$',
            title='leakage of past shocks into $a_t$')
for ax in axes:
    ax.axhline(ax.get_ylim()[0], color='k', lw=0.6)
fig.tight_layout()
plt.show()

print(f'at theta = 0:   ratio = {det_ratio[0]:.6f}, leakage = {leak[0]:.2e}')
print(f'at theta = 1:   ratio = {det_ratio[20]:.4f}, leakage = {leak[20]:.4f}')
```

Both curves start at their minima, exactly zero leakage and a determinant ratio
of exactly one.

That is {prf:ref}`vs_prop_blockexog`: at $\theta = 0$ the interest rate does not
Granger cause $(g, \pi)$, so dropping it costs nothing even though its
innovation is contemporaneously correlated with the others.

Both measures rise as the feedback strengthens.

```{solution-end}
```

```{exercise-start}
:label: vs_ex2
```

Return to the bivariate system, but replace $A_1$ and $A_2$ by the single matrix

$$
A_1 = \begin{pmatrix} 0.5 & 0.6 \\ 0 & \rho_z \end{pmatrix},
\qquad V = I_2 ,
$$

so that $\rho_z$ controls the persistence of the omitted variable $z$.

Observing $r$ only, report for $\rho_z \in \{0.2, 0.5, 0.75, 0.9, 0.95, 0.99\}$

1. the forecast error variance $\Omega$,
2. the number of lags needed before $\sum_{j > p} |B_j| < 10^{-3}$.

What does a persistent omitted variable do to the VAR that an econometrician
should fit?

```{exercise-end}
```

```{solution-start} vs_ex2
:class: dropdown
```

Here is one solution:

```{code-cell} ipython3
print(' rho_z    Omega    lags needed')
for rho_z in [0.2, 0.5, 0.75, 0.9, 0.95, 0.99]:
    mod = VARSubsystem([np.array([[0.5, 0.6], [0.0, rho_z]])],
                       np.eye(2), np.array([[1.0, 0.0]]))
    B = mod.var_coefficients(400)
    tails = [np.abs(B[p:]).sum() for p in range(400)]
    p_need = next(p for p in range(400) if tails[p] < 1e-3)
    print(f' {rho_z:5.2f}   {mod.innovation_cov()[0, 0]:6.4f}      {p_need:3d}')
```

Both columns rise with $\rho_z$.

A persistent omitted variable both inflates the forecast error variance and
lengthens the autoregression, because the econometrician must reach further back
to extract the same information about $z$ from the history of $r$.

An empirical VAR with too few lags will therefore be worst exactly where the
missing variable is most persistent.

```{solution-end}
```

```{exercise-start}
:label: vs_ex3
```

The coefficients $B_j$ are population objects.

Simulate $T = 200{,}000$ observations of the bivariate system of Example 1,
retain only $r_t$, and fit finite-order autoregressions of orders
$p = 1, 2, 4, 8$ by ordinary least squares.

Compare the estimates with $B_1, \ldots, B_p$ and the residual variance with
$\Omega$.

Which order is too short, and what does fitting too short a lag length do to the
first coefficient?

```{exercise-end}
```

```{solution-start} vs_ex3
:class: dropdown
```

Here is one solution:

```{code-cell} ipython3
Y_big, _ = mod_r.simulate(200_000, seed=3)
r_series = Y_big[:, 0]
B_pop = mod_r.var_coefficients(8)[:, 0, 0]

for p in [1, 2, 4, 8]:
    X = np.column_stack([r_series[p - 1 - l:len(r_series) - 1 - l]
                         for l in range(p)])
    zz = r_series[p:]
    b_hat = np.linalg.lstsq(X, zz, rcond=None)[0]
    resid = zz - X @ b_hat
    print(f'p = {p}')
    print(f'   OLS        {np.round(b_hat, 4)}')
    print(f'   population {np.round(B_pop[:p], 4)}')
    print(f'   residual variance {resid.var():.4f}   Omega {mod_r.innovation_cov()[0, 0]:.4f}')
```

An AR(1) is too short.

Its single coefficient is pulled well above $B_1$, because it has to stand in for
the omitted second lag, and its residual variance exceeds $\Omega$.

From $p = 2$ onward the estimates track the population coefficients and the
residual variance settles on $\Omega$, which matches the finding above that
$B_j$ is negligible beyond the second lag in this example.

```{solution-end}
```
