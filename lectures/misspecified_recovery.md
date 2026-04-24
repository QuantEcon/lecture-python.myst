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

(misspecified_recovery)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Misspecified Recovery

```{contents} Contents
:depth: 2
```

## Overview

Asset prices are forward-looking: they encode investors' expectations about future economic
states and their valuations of different risks.  

A long-standing question in finance is
whether one can *recover* the probability distribution used by investors — their subjective
beliefs — from observed asset prices alone.

{cite}`BorovickaHansenScheinkman2016` study the challenge of separating investors'
beliefs from their risk preferences using **Perron–Frobenius theory**. 

Their key finding
is that Perron–Frobenius theory applied to Arrow prices recovers a **long-term risk-neutral
measure** that absorbs all long-horizon risk adjustments.  

This recovered measure coincides
with investors' subjective beliefs only under a stringent — and often empirically
implausible — restriction on the stochastic discount factor.

After completing this lecture you will be able to:

- Explain why Arrow prices alone cannot identify both transition probabilities and stochastic
  discount factors without additional restrictions.
- Construct **risk-neutral** and **long-term risk-neutral** transition matrices from Arrow
  prices using the Perron–Frobenius eigenvalue–eigenvector decomposition.
- Decompose any stochastic discount factor process into a trend component, a state-dependent
  component, and a **martingale component**, and explain what the martingale encodes.
- Identify the exact condition under which {cite}`Ross2015`'s Recovery Theorem succeeds,
  and show that this condition fails in empirically relevant models with recursive utility
  or permanent consumption shocks.
- Simulate the {cite}`Bansal_Yaron_2004` long-run risk model and compare the stationary
  distributions under the physical and recovered probability measures.

### Related lectures

- {doc}`affine_risk_prices`: affine models of the stochastic discount factor and term structure.
- {doc}`markov_asset`: Markov asset pricing and stationary equilibria.
- {doc}`harrison_kreps`: risk-neutral pricing and the change-of-measure approach.

## Setup

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import linalg
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
    'figure.dpi': 110,
})
```

## Arrow prices and the identification challenge

### Arrow prices and stochastic discount factors

Consider a discrete-time economy with an $n$-state Markov chain $\{X_t\}$ governed
by transition matrix $\mathbf{P} = [p_{ij}]$.  

An **Arrow price** $q_{ij}$ is the
date-$t$ price of a claim that pays $\$1$ tomorrow in state $j$ given that the current
state is $i$.  

Collect these prices in a matrix $\mathbf{Q} = [q_{ij}]$.

A **stochastic discount factor** (SDF) $s_{ij}$ prices risk by discounting the payoff
in state $j$ tomorrow when today's state is $i$.  

Arrow prices and the SDF are linked by

$$
q_{ij} = s_{ij} \, p_{ij}.
$$

Given $\mathbf{Q}$, any pair $(\mathbf{S}, \mathbf{P})$ satisfying $q_{ij} = s_{ij} p_{ij}$
for all $(i,j)$ is consistent with the observed prices.  

The fundamental identification
problem is that $\mathbf{Q}$ has $n^2$ entries, $\mathbf{P}$ has $n(n-1)$ free entries
(rows sum to one), and $\mathbf{S}$ has $n^2$ free entries — so there are far more
unknowns than equations.

To make progress, we can impose restrictions on the SDF.

Two classical restrictions are
studied in the sections that follow.

### A three-state illustration

To build intuition, we work with a three-state Markov chain representing
**recession**, **normal**, and **expansion** phases of the business cycle.

The physical transition matrix and consumption levels are:

```{code-cell} ipython3
# Physical transition matrix (recession, normal, expansion)
P_phys = np.array([
    [0.70, 0.25, 0.05],   # from recession
    [0.15, 0.65, 0.20],   # from normal
    [0.05, 0.30, 0.65],   # from expansion
])

# Consumption levels in each state (arbitrary units)
c_levels = np.array([0.85, 1.00, 1.15])
state_names = ['recession', 'normal', 'expansion']

# Preference parameters
δ = 0.99    # monthly discount factor
γ = 5.0     # coefficient of relative risk aversion

# Arrow price matrix under power utility with rational expectations:
#   q_ij = δ * (c_j / c_i)^{-γ} * p_ij
n = len(c_levels)
Q_mat = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        Q_mat[i, j] = δ * (c_levels[j] / c_levels[i])**(-γ) * P_phys[i, j]

print("Arrow price matrix Q:")
print(np.round(Q_mat, 5))
print(f"\nSum of each row (= price of risk-free bond): {Q_mat.sum(axis=1).round(5)}")
```

## Risk-neutral probabilities

The **risk-neutral restriction** sets

$$
\bar{s}_{i,j} = \bar{q}_i
$$

where $\bar{q}_i = \sum_j q_{ij}$ is the price of a one-period discount bond in state $i$.

Under this restriction all future states are discounted equally from state $i$, so risk
adjustments depend only on the current state.  

The resulting risk-neutral probabilities are

$$
\bar{p}_{ij} = \frac{q_{ij}}{\bar{q}_i}.
$$

```{code-cell} ipython3
def risk_neutral_probs(Q):
    """Compute risk-neutral transition matrix from Arrow price matrix."""
    q_bonds = Q.sum(axis=1)            # one-period bond prices
    P_bar = Q / q_bonds[:, np.newaxis]
    return P_bar, q_bonds


P_bar, q_bonds = risk_neutral_probs(Q_mat)

print("One-period bond prices (risk-free discount factors):")
for i, (s, qb) in enumerate(zip(state_names, q_bonds)):
    print(f"  {s:12s}: {qb:.5f}  (annualized yield ~ {-np.log(qb)*12:.2%})")

print("\nRisk-neutral transition matrix P_bar:")
print(np.round(P_bar, 4))
print(f"\nRow sums: {P_bar.sum(axis=1)}")
```

```{note}
Risk-neutral probabilities absorb **one-period** (short-run) risk adjustments.

They are
widely used in financial engineering but are generally *not* equal to investors' beliefs.

When short-term interest rates vary across states, risk-neutral probabilities are
also horizon-dependent: the $t$-period forward measure differs from $\bar{\mathbf{P}}^t$.
```

## Long-term risk-neutral probabilities: Perron–Frobenius theory

### The eigenvalue problem

The long-term behavior of discount factors is governed by a different restriction.
**Long-term risk pricing** sets

$$
\hat{s}_{ij} = \exp(\hat{\eta}) \frac{\hat{e}_i}{\hat{e}_j}
$$

for a scalar $\hat{\eta}$ and a vector of positive numbers $\{\hat{e}_i\}$.
Substituting into $q_{ij} = s_{ij} p_{ij}$ gives:

$$
\hat{p}_{ij} = \exp(-\hat{\eta}) \, q_{ij} \, \frac{\hat{e}_j}{\hat{e}_i}.
$$

For $\hat{\mathbf{P}}$ to be a valid transition matrix (rows summing to one), we need
$\sum_j \hat{p}_{ij} = 1$, which requires

$$
\sum_j q_{ij} \hat{e}_j = \exp(\hat{\eta}) \hat{e}_i, \quad \text{i.e.,} \quad \mathbf{Q} \hat{\mathbf{e}} = \exp(\hat{\eta}) \hat{\mathbf{e}}.
$$

This is an **eigenvalue–eigenvector problem** for the Arrow price matrix $\mathbf{Q}$.

By the **Perron–Frobenius theorem**, if $\mathbf{Q}$ has strictly positive entries, the
dominant eigenvalue is unique, real, and positive, and its eigenvector has strictly
positive entries.  

This gives a unique construction:

1. Solve $\mathbf{Q} \hat{\mathbf{e}} = \exp(\hat{\eta}) \hat{\mathbf{e}}$ for the
   dominant eigenvalue–eigenvector pair.
2. Set $\hat{p}_{ij} = \exp(-\hat{\eta}) \, q_{ij} \, \hat{e}_j / \hat{e}_i$.

{cite}`BorovickaHansenScheinkman2016` call the resulting $\hat{\mathbf{P}}$ the
**long-term risk-neutral measure** because, under $\hat{\mathbf{P}}$, the long-horizon
risk premia on stochastically growing cash flows are identically zero.

### Python implementation

```{code-cell} ipython3
def perron_frobenius(Q):
    """
    Compute the Perron-Frobenius decomposition of an Arrow price matrix.

    Parameters
    ----------
    Q : ndarray, shape (n, n)
        Arrow price matrix.

    Returns
    -------
    η_hat : float — log of the dominant eigenvalue
    exp_η : float — dominant eigenvalue exp(η_hat)
    e_hat : ndarray — dominant eigenvector (positive, normalized to sum=1)
    P_hat : ndarray — long-term risk-neutral transition matrix
    """
    eigenvalues, eigenvectors = linalg.eig(Q)

    # Dominant eigenvalue: largest real part (real & positive by Perron–Frobenius)
    idx = np.argmax(eigenvalues.real)
    exp_η = eigenvalues[idx].real
    e_hat = eigenvectors[:, idx].real

    # Ensure positive entries (PF guarantees existence; numpy may flip sign)
    if e_hat.mean() < 0:
        e_hat = -e_hat
    e_hat = np.abs(e_hat) / np.abs(e_hat).sum()   # normalize to sum = 1

    η_hat = np.log(exp_η)

    # Long-term risk-neutral transition matrix
    # P_hat[i,j] = exp(-η_hat) * Q[i,j] * e_hat[j] / e_hat[i]
    P_hat = (1.0 / exp_η) * Q * e_hat[np.newaxis, :] / e_hat[:, np.newaxis]

    return η_hat, exp_η, e_hat, P_hat


η_hat, exp_η, e_hat, P_hat = perron_frobenius(Q_mat)

print(f"Dominant eigenvalue  exp(η_hat) = {exp_η:.6f}")
print(f"Log eigenvalue       η_hat      = {η_hat:.5f}  "
      f"(annualized ~ {η_hat*12:.4f})")
print(f"\nEigenvector e_hat = {e_hat.round(5)}")
print(f"\nLong-term risk-neutral P_hat:")
print(np.round(P_hat, 4))
print(f"\nRow sums: {P_hat.sum(axis=1)}")
```

### Comparing the three probability measures

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

matrices = [
    (P_phys, r'physical  $\mathbf{P}$', 'Blues'),
    (P_bar, r'risk-neutral  $\bar{\mathbf{P}}$', 'Oranges'),
    (P_hat, r'long-term risk-neutral $\hat{\mathbf{P}}$', 'Greens'),
]

for ax, (mat, title, cmap) in zip(axes, matrices):
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=0.85, aspect='auto')
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks(range(n));  ax.set_yticks(range(n))
    ax.set_xticklabels(state_names, rotation=20, fontsize=9)
    ax.set_yticklabels(state_names, fontsize=9)
    ax.set_xlabel('next state', fontsize=9)
    ax.set_ylabel('current state', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(n):
        for j in range(n):
            clr = 'white' if mat[i, j] > 0.45 else 'black'
            ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center',
                    fontsize=9, color=clr)

plt.suptitle('transition matrices under alternative probability measures',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# Stationary distributions under each measure
def stationary_dist(P):
    """Compute stationary distribution of an ergodic transition matrix P."""
    n = P.shape[0]
    A = (P.T - np.eye(n))
    A[-1] = 1.0
    b = np.zeros(n);  b[-1] = 1.0
    return linalg.solve(A, b)

π_phys = stationary_dist(P_phys)
π_bar = stationary_dist(P_bar)
π_hat = stationary_dist(P_hat)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(n)
w = 0.25
labels = [r'physical $P$', r'risk-neutral $\bar{P}$',
          r'long-term risk-neutral $\hat{P}$']
colors = ['steelblue', 'darkorange', 'forestgreen']
for k, (π, lbl, col) in enumerate(zip([π_phys, π_bar, π_hat], labels, colors)):
    bars = ax.bar(x + k*w, π, width=w, label=lbl, color=col, alpha=0.85,
                  edgecolor='white')
    for b_, v in zip(bars, π):
        ax.text(b_.get_x() + w/2, v + 0.008, f'{v:.3f}',
                ha='center', va='bottom', fontsize=9)

ax.set_xticks(x + w);  ax.set_xticklabels(state_names)
ax.set_ylabel('stationary probability')
ax.set_title('stationary distributions under three probability measures')
ax.legend(fontsize=9)
plt.tight_layout();  plt.show()

print("Stationary distributions:")
for lbl, π in zip(labels, [π_phys, π_bar, π_hat]):
    print(f"  {lbl:45s}: {np.round(π,4)}")
```

The long-term risk-neutral measure $\hat{\mathbf{P}}$ assigns **higher weight to bad
states** (recession) and **lower weight to good states** (expansion) than the physical
measure $\mathbf{P}$.  

This is the risk adjustment for long-run growth uncertainty: a
risk-averse investor's long-run discount rates embed a premium for permanent income risk.

## The martingale decomposition

### Decomposing the SDF process

Let $\hat{\mathbf{e}}$ and $\hat{\eta}$ solve the Perron–Frobenius problem.

Define the
process

$$
\frac{\hat{H}_{t+1}}{\hat{H}_t} = (X_t)' \hat{\mathbf{H}} X_{t+1},
\quad \text{where} \quad
\hat{h}_{ij} = \frac{\hat{p}_{ij}}{p_{ij}}.
$$

Because $\sum_j \hat{h}_{ij} p_{ij} = \sum_j \hat{p}_{ij} = 1$, the process $\hat{H}$
is a martingale under the physical measure $\mathbf{P}$.  

The accumulated SDF then admits
the **multiplicative decomposition**:

$$
S_t = \exp(\hat{\eta} t) \left(\frac{\hat{e}(X_0)}{\hat{e}(X_t)}\right)
      \left(\frac{\hat{H}_t}{\hat{H}_0}\right).
$$

The three components are:

| Component | Interpretation |
|---|---|
| $\exp(\hat{\eta} t)$ | Deterministic exponential discounting; $-\hat{\eta}$ is the long-run yield |
| $\hat{e}(X_0)/\hat{e}(X_t)$ | State-dependent trend; mean-stationary under $\hat{\mathbf{P}}$ |
| $\hat{H}_t/\hat{H}_0$ | Martingale; encodes long-run risk adjustments |

```{code-cell} ipython3
# SDF matrix: s_ij = q_ij / p_ij
S_mat = np.where(P_phys > 0, Q_mat / P_phys, 0.0)

# Trend SDF: s_hat_ij = exp(η_hat) * e_hat_i / e_hat_j
S_hat = exp_η * e_hat[:, np.newaxis] / e_hat[np.newaxis, :]

# Martingale increment: h_hat_ij = P_hat_ij / P_ij  (also = S_ij / S_hat_ij)
H_incr = np.where(P_phys > 0, P_hat / P_phys, 0.0)

print("SDF matrix S = Q/P:")
print(np.round(S_mat, 4))
print("\nTrend SDF S_hat = exp(η_hat) * e_hat_i / e_hat_j:")
print(np.round(S_hat, 4))
print("\nMartingale increment h_hat = S_hat * H_tilde_incr (= P_hat/P):")
print(np.round(H_incr, 4))

# Verify martingale property: E[h_hat_{ij} | X_t=i] = sum_j h_hat_ij * p_ij = 1
mart_check = (H_incr * P_phys).sum(axis=1)
print(f"\nMartingale property check — E[h_hat | X_t=i] = {mart_check}")
```

Higher risk aversion amplifies the pessimistic distortion: as $\gamma$ increases, the
recovered measure assigns growing probability to the recession state.


(Gigures illustrating this will appear below, after we define the Epstein–Zin utility
function that is needed to compute them.)

## When does recovery succeed?

### The Ross recovery condition

{cite}`Ross2015` proposes to identify investors' subjective beliefs by imposing

$$
\widetilde{S}_t = \exp(-\delta t) \frac{m(X_t)}{m(X_0)}
$$

for some positive function $m$ and discount rate $\delta$ (Condition 4 in
{cite}`BorovickaHansenScheinkman2016`).  

Under this restriction, the SDF has **no
martingale component**: $\hat{H}_t \equiv 1$.

Equivalently, recovery succeeds if and only if the physical stochastic discount factor
takes the "long-term risk pricing" form

$$
s_{ij} = \exp(\hat{\eta}) \frac{\hat{e}_i}{\hat{e}_j}
$$

with $\hat{h}_{ij} \equiv 1$.

In this case $\hat{\mathbf{P}} = \mathbf{P}$ and the
Perron–Frobenius procedure recovers the true probabilities.

The critical question is: when is the martingale component degenerate?

### Power utility with trend-stationary consumption

Consider a power-utility investor with risk aversion $\gamma$ and *trend-stationary*
consumption $C_t = \exp(g_c t)(c \cdot X_t)$ where $c$ is a positive vector.

The
one-period SDF is

$$
s_{ij} = \exp(-\delta - \gamma g_c) \left(\frac{c_j}{c_i}\right)^{-\gamma}.
$$

This has the exact long-term risk pricing form with $\hat{e}_j = c_j^\gamma$ and
$\hat{\eta} = -(\delta + \gamma g_c)$.

Therefore $\hat{h}_{ij} \equiv 1$ and **Ross
recovery succeeds exactly** when consumption fluctuations around a deterministic trend
are the only source of risk.

```{code-cell} ipython3
# Verify: for trend-stationary power utility, h_hat_ij = 1 identically
gc = 0.002   # monthly trend growth

# Trend-stationary: consumption growth ratio depends only on state, not history
# s_ij = exp(-δ - γ*gc) * (c_j/c_i)^(-γ)
S_trend = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        S_trend[i, j] = np.exp(-δ - γ*gc) * (c_levels[j]/c_levels[i])**(-γ)

Q_trend = S_trend * P_phys

_, exp_η_t, e_hat_t, P_hat_t = perron_frobenius(Q_trend)

H_incr_trend = np.where(P_phys > 0, P_hat_t / P_phys, 0.0)

print("Martingale increment h_hat_ij for trend-stationary power utility:")
print(np.round(H_incr_trend, 6))
print(f"\nMax deviation from 1: {np.abs(H_incr_trend[P_phys>0] - 1).max():.2e}")
print("-> Martingale is trivial: Recovery succeeds.")
```

### Recursive (Epstein–Zin) utility

When the investor has **Epstein–Zin recursive preferences** with risk aversion
$\gamma \neq 1$, continuation values $V_t$ satisfy the recursion

$$
V_t = \bigl[1-\exp(-\delta)\bigr] \log C_t
      + \frac{\exp(-\delta)}{1-\gamma}
        \log \mathbf{E}_t\bigl[\exp\bigl((1-\gamma)V_{t+1}\bigr)\bigr].
$$

The SDF takes the form (see {cite}`BorovickaHansenScheinkman2016`, Example 2)

$$
s_{ij} = \exp(-\delta - g_c)\frac{c_i}{c_j}
         \left(\frac{v^*_j}{\mathbf{P}_i v^*}\right),
$$

where $v^*_i = \exp\!\bigl[(1-\gamma)v_i\bigr]$ and $\mathbf{P}_i$ is the $i$-th row of
$\mathbf{P}$.  

The additional factor $v^*_j/(\mathbf{P}_i v^*)$ introduces a **nontrivial
martingale component** whenever continuation values vary across states.

```{code-cell} ipython3
def solve_ez_finite(P, c, δ, γ, gc, tol=1e-12, max_iter=5000):
    """
    Solve for Epstein-Zin continuation values in finite Markov chain.

    Solves the fixed-point v_i = (1-β)log(c_i) + β/(1-γ) log(P_i @ exp((1-γ)v))
    where β = exp(-δ - gc).  The special case γ = 1 (log utility) is handled
    separately to avoid the 0/0 indeterminate form: the recursion reduces to
    v = (I - β P)^{-1} (1-β) log(c) and the SDF simplifies to
    s_ij = exp(-δ - g_c) c_i / c_j.

    Returns
    -------
    v : ndarray — continuation values (net of time trend)
    vstar : ndarray — exp((1-γ)v)
    s : ndarray — one-period SDF matrix
    """
    β = np.exp(-δ - gc)
    log_c = np.log(c)
    n = len(c)

    if abs(γ - 1.0) < 1e-10:
        # Log utility: (I - β P) v = (1-β) log c
        v = linalg.solve(np.eye(n) - β * P, (1 - β) * log_c)
        vstar = np.ones(n)     # exp((1-1)*v) = 1
        Pv = np.ones(n)        # P @ ones = ones
    else:
        # General recursive utility: fixed-point iteration
        v = log_c.copy()
        for _ in range(max_iter):
            vstar = np.exp((1 - γ) * v)
            Pv = P @ vstar
            v_new = ((1 - β) * log_c
                     + β / (1 - γ) * np.log(Pv))
            if np.max(np.abs(v_new - v)) < tol:
                v = v_new
                break
            v = v_new
        vstar = np.exp((1 - γ) * v)
        Pv = P @ vstar

    # SDF matrix
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s[i, j] = np.exp(-δ - gc) * (c[i] / c[j]) * (vstar[j] / Pv[i])

    return v, vstar, s


# Compare: γ = 1 (log utility, degenerate martingale) vs γ = 5
gc_ex = 0.001   # monthly consumption trend growth

for γ_val, label in [(1.0, 'γ = 1  (log utility)'), (5.0, 'γ = 5  (risk aversion)')]:
    v_ez, vstar_ez, S_ez = solve_ez_finite(P_phys, c_levels,
                                            δ, γ_val, gc_ex)
    Q_ez = S_ez * P_phys
    _, _, _, P_hat_ez = perron_frobenius(Q_ez)
    H_ez = np.where(P_phys > 0, P_hat_ez / P_phys, 0.0)

    π_hat_ez = stationary_dist(P_hat_ez)
    print(f"\n{label}")
    print(f"  Continuation values v = {v_ez.round(4)}")
    print(f"  Max |h_hat_ij - 1|        = {np.abs(H_ez[P_phys>0] - 1).max():.4f}")
    print(f"  Stationary P_hat         = {π_hat_ez.round(4)}")
    print(f"  Stationary P          = {π_phys.round(4)}")
```

```{code-cell} ipython3
# Show how the martingale depends on γ for recursive utility
# Start at 1.0: the γ=1 special case in solve_ez_finite is handled explicitly.
γs_ez = np.linspace(1.0, 10.0, 50)
mart_errors = []
π_rec_hat = []

for γ_val in γs_ez:
    v_g, _, S_g = solve_ez_finite(P_phys, c_levels, δ, γ_val, gc_ex)
    Q_g = S_g * P_phys
    _, _, _, Ph = perron_frobenius(Q_g)
    H_g = np.where(P_phys > 0, Ph / P_phys, 0.0)
    mart_errors.append(np.abs(H_g[P_phys > 0] - 1).max())
    π_rec_hat.append(stationary_dist(Ph)[0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(γs_ez, mart_errors, color='firebrick', lw=2.5)
ax1.set_xlabel('risk aversion  γ')
ax1.set_ylabel(r'$\max_{i,j} |\hat{h}_{ij} - 1|$')
ax1.set_title('martingale non-degeneracy vs risk aversion\n(Epstein–Zin utility)')

ax2.plot(γs_ez, π_rec_hat, color='steelblue', lw=2.5,
         label=r'recession weight under $\hat{P}$')
ax2.axhline(π_phys[0], ls='--', color='grey', lw=1.5,
            label=f'recession weight under $P$  ({π_phys[0]:.3f})')
ax2.set_xlabel('risk aversion  γ')
ax2.set_ylabel('stationary probability')
ax2.set_title('recovered recession probability vs risk aversion')
ax2.legend(fontsize=9)

plt.tight_layout();  plt.show()
```

```{code-cell} ipython3
# Visualize the martingale increment using Epstein-Zin utility (γ=5).
# Trend-stationary power utility always yields h_hat_ij = 1 by construction (see Exercise 4),
# so we use recursive utility here to reveal a genuinely non-trivial martingale.
γ_ez_demo = 5.0
_, _, S_ez_demo = solve_ez_finite(P_phys, c_levels, δ, γ_ez_demo, gc_ex)
Q_ez_demo = S_ez_demo * P_phys
_, _, _, P_hat_ez_demo = perron_frobenius(Q_ez_demo)
H_incr_ez = np.where(P_phys > 0, P_hat_ez_demo / P_phys, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

vmax_h = max(1.5, H_incr_ez.max() * 1.05)
vmin_h = min(0.5, H_incr_ez.min() * 0.95)
im0 = axes[0].imshow(H_incr_ez, cmap='RdYlGn', vmin=vmin_h, vmax=vmax_h, aspect='auto')
axes[0].set_title(
    r'martingale increment $\hat{h}_{ij} = \hat{p}_{ij}/p_{ij}$' '\n'
    r'(Epstein–Zin utility, $\gamma=5$)',
    fontsize=11)
for i in range(n):
    for j in range(n):
        axes[0].text(j, i, f'{H_incr_ez[i,j]:.3f}',
                     ha='center', va='center', fontsize=10)
axes[0].set_xticks(range(n));  axes[0].set_yticks(range(n))
axes[0].set_xticklabels(state_names, rotation=20, fontsize=9)
axes[0].set_yticklabels(state_names, fontsize=9)
axes[0].set_xlabel('next state');  axes[0].set_ylabel('current state')
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# How risk aversion γ shifts the recovered measure under Epstein-Zin utility.
γs_shift = np.linspace(1.0, 12, 60)
rec_wts_ez = []
for g in γs_shift:
    _, _, S_g = solve_ez_finite(P_phys, c_levels, δ, g, gc_ex)
    Q_g = S_g * P_phys
    _, _, _, Ph = perron_frobenius(Q_g)
    rec_wts_ez.append(stationary_dist(Ph)[0])

axes[1].plot(γs_shift, rec_wts_ez, color='steelblue', lw=2.5)
axes[1].axhline(π_phys[0], color='grey', ls='--', lw=1.5,
                label=fr'physical recession prob = {π_phys[0]:.3f}')
axes[1].set_xlabel('risk aversion  γ')
axes[1].set_ylabel(r'recession weight under $\hat{P}$')
axes[1].set_title(r'how $\gamma$ shifts the long-term risk-neutral measure'
                  '\n(Epstein–Zin utility)')
axes[1].legend(fontsize=9)
plt.tight_layout();  plt.show()
```

At $\gamma = 1$ (log utility), the continuation value is constant across states and the
martingale is trivial, so recovery succeeds.

For $\gamma > 1$, continuation values vary
with the state, generating a non-degenerate martingale that grows with risk aversion.

## The long-run risk model

We now illustrate the results quantitatively using the Bansal–Yaron
{cite}`Bansal_Yaron_2004` long-run risk model, calibrated to {cite}`BorovickaHansenScheinkman2016`
(Figure 1).

### Model setup

The state vector $X_t = (X_{1t}, X_{2t})'$ follows the continuous-time diffusion

$$
\begin{aligned}
dX_{1t} &= \bar{\mu}_{11}(X_{1t} - \iota_1)\,dt + \sqrt{X_{2t}}\,\bar{\sigma}_1 dW_t \\
dX_{2t} &= \bar{\mu}_{22}(X_{2t} - \iota_2)\,dt + \sqrt{X_{2t}}\,\bar{\sigma}_2 dW_t,
\end{aligned}
$$

where $W_t$ is a three-dimensional Brownian motion.

Here $X_{1t}$ is the
**predictable component of consumption growth** and $X_{2t}$ is **stochastic volatility**.

The representative agent has Epstein–Zin preferences with unit elasticity of substitution.

The stochastic discount factor satisfies

$$
d\log S_t = -\delta\,dt - d\log C_t + d\log H^*_t,
$$

where $H^*$ is a martingale determined by the continuation value of the recursive utility.

```{code-cell} ipython3
# Model parameters from Borovicka-Hansen-Scheinkman (2016), Figure 1
# Monthly frequency
lrr_params = dict(
    δ = 0.002,         # subjective discount rate
    γ = 10.0,          # risk aversion
    μ11 = -0.021,      # mean reversion of X1
    μ12 = 0.0,         # (under P; becomes non-zero under P_hat)
    μ22 = -0.013,      # mean reversion of X2
    ι1 = 0.0,          # long-run mean of X1
    ι2 = 1.0,          # long-run mean of X2 (normalized)
    σ1 = np.array([0.0, 0.00034, 0.0]),   # diffusion of X1 (1*3)
    σ2 = np.array([0.0, 0.0, -0.038]),    # diffusion of X2 (1*3)
    β_c0 = 0.0015,     # consumption drift constant
    β_c1 = 1.0,        # loading on X1
    β_c2 = 0.0,        # loading on X2
    α_c = np.array([0.0078, 0.0, 0.0]),   # consumption diffusion (1*3)
)
```

### Solving the value function

The log continuation value $v(X_t)$ is affine in the state: $v(x) = \bar{v}_0 + \bar{v}_1 x_1 + \bar{v}_2 x_2$.

The coefficients satisfy the algebraic system in Appendix D of {cite}`BorovickaHansenScheinkman2016`.

```{code-cell} ipython3
def solve_value_function(p):
    """
    Solve for Epstein-Zin value function coefficients in the LRR model.

    The continuation value satisfies:
        log V_t = log C_t + v_bar0 + v_bar1*X1_t + v_bar2*X2_t

    Returns v_bar1, v_bar2.
    """
    δ, γ = p['δ'], p['γ']
    μ11, μ12, μ22 = p['μ11'], p['μ12'], p['μ22']
    σ1, σ2 = p['σ1'], p['σ2']
    β_c1, β_c2 = p['β_c1'], p['β_c2']
    α_c = p['α_c']

    # Linear equation for v_bar1
    # δ v_bar1 = β_c1 + μ_bar11 v_bar1  =>  v_bar1 = β_c1 / (δ - μ_bar11)
    v1 = β_c1 / (δ - μ11)

    # Quadratic equation for v_bar2
    # 0 = (μ_bar22 - δ)v_bar2 + β_c2 + μ_bar12 v_bar1 + (1/2)(1-γ)|A + B v_bar2|^2
    # where A = α_c_bar + σ_bar1 v_bar1,  B = σ_bar2
    A_vec = α_c + σ1 * v1
    B_vec = σ2

    a = 0.5 * (1 - γ) * np.dot(B_vec, B_vec)
    b = (μ22 - δ) + (1 - γ) * np.dot(A_vec, B_vec)
    c = β_c2 + μ12 * v1 + 0.5 * (1 - γ) * np.dot(A_vec, A_vec)

    disc = b**2 - 4*a*c
    if disc < 0:
        raise ValueError("Value function does not exist for these parameters.")

    # "Minus" solution (generates ergodic dynamics under P_hat)
    v2 = (-b - np.sqrt(disc)) / (2 * a)
    return v1, v2, A_vec, B_vec


v1, v2, A_vec, B_vec = solve_value_function(lrr_params)
print(f"Value-function slope on X1:  v_bar1 = {v1:.4f}")
print(f"Value-function slope on X2:  v_bar2 = {v2:.4f}")
print(f"\nInterpretation:")
print(f"  Higher X1 (better expected growth) raises continuation value (v_bar1 > 0)")
print(f"  Higher X2 (more volatility) lowers continuation value (v_bar2 < 0)")
```

### Perron–Frobenius and recovered dynamics

```{code-cell} ipython3
def solve_pf_lrr(p, v1, v2, A_vec):
    """
    Solve the Perron-Frobenius problem for the long-run risk model.

    Eigenfunction guess: e_hat(x) = exp(e_bar1 x1 + e_bar2 x2).

    Returns e_bar1, e_bar2, η_hat, and the SDF diffusion vector α_s.
    """
    δ, γ = p['δ'], p['γ']
    μ11, μ12, μ22 = p['μ11'], p['μ12'], p['μ22']
    ι1, ι2 = p['ι1'], p['ι2']
    σ1, σ2 = p['σ1'], p['σ2']
    α_c = p['α_c']
    β_c0 = p['β_c0']
    β_c1, β_c2 = p['β_c1'], p['β_c2']

    # SDF diffusion:  α_s = -γ α_c + (1-γ)(σ_bar1 v_bar1 + σ_bar2 v_bar2)
    α_s = (-γ * α_c
           + (1 - γ) * (σ1 * v1 + σ2 * v2))

    # SDF drift parameters in  β_s(x) = β_s0 + β_s11(x1-ι1) + β_s12(x2-ι2)
    β_s11 = -β_c1
    β_s12 = -β_c2 - 0.5 * np.dot(α_s, α_s)
    β_s0 = (-δ - β_c0
            - 0.5 * ι2 * np.dot(α_s, α_s))

    # Equation 0 = β_s11 + μ_bar11 e_bar1  =>  e_bar1 = -β_s11 / μ_bar11
    e1 = -β_s11 / μ11

    # Quadratic for e_bar2
    # 0 = (β_s12 + (1/2)|α_s|^2)  +  e_bar1(μ_bar12 + σ_bar1*α_s) + (1/2)e_bar1^2|σ_bar1|^2
    #     + e_bar2(μ_bar22 + σ_bar2*α_s + e_bar1 σ_bar1*σ_bar2') + (1/2)e_bar2^2|σ_bar2|^2
    const_pf = (β_s12 + 0.5*np.dot(α_s, α_s)    # = 0 by construction
                + e1*(μ12 + np.dot(σ1, α_s))
                + 0.5*e1**2*np.dot(σ1, σ1))
    lin_pf = μ22 + np.dot(σ2, α_s) + e1*np.dot(σ1, σ2)
    quad_pf = 0.5 * np.dot(σ2, σ2)

    disc = lin_pf**2 - 4*quad_pf*const_pf
    e2_m = (-lin_pf - np.sqrt(disc)) / (2*quad_pf)
    e2_p = (-lin_pf + np.sqrt(disc)) / (2*quad_pf)

    # η_hat = β_s0 - β_s12*ι2 - e_bar2*μ_bar22*ι2  (ι1 = 0)
    η_m = β_s0 - β_s12*ι2 - e2_m*μ22*ι2
    η_p = β_s0 - β_s12*ι2 - e2_p*μ22*ι2

    # Choose solution with smaller |η_hat| (ergodicity requirement)
    if abs(η_m) <= abs(η_p):
        e2, η_hat = e2_m, η_m
    else:
        e2, η_hat = e2_p, η_p

    return e1, e2, η_hat, α_s


e1, e2, η_hat_lrr, α_s = solve_pf_lrr(lrr_params, v1, v2, A_vec)

print(f"PF eigenfunction coefficients:  e_bar1 = {e1:.4f},  e_bar2 = {e2:.4f}")
print(f"Log eigenvalue:                 η_hat  = {η_hat_lrr:.6f}  "
      f"(annualized = {η_hat_lrr*12:.4f})")
print(f"\nInterpretation:")
print(f"  e_bar1 = {e1:.2f}: e_hat down-weights high-X1 (good growth) states")
print(f"  e_bar2 = {e2:.2f}: e_hat up-weights high-X2 (high volatility) states")
```

### Computing the P_hat dynamics

```{code-cell} ipython3
def compute_phat_dynamics(p, e1, e2, α_s):
    """
    Compute the drift parameters of X under the recovered measure P_hat.

    Under P_hat, the Brownian motion is
        dW_hat_t = -sqrt(X2_t) * α_hat_h dt + dW_t
    where α_hat_h = α_s + σ_bar1 e_bar1 + σ_bar2 e_bar2.
    """
    μ11, μ12, μ22 = p['μ11'], p['μ12'], p['μ22']
    ι1, ι2 = p['ι1'], p['ι2']
    σ1, σ2 = p['σ1'], p['σ2']

    # Martingale drift correction
    α_h = α_s + σ1 * e1 + σ2 * e2

    # New drift parameters under P_hat
    μ_hat_11 = μ11
    μ_hat_12 = μ12 + np.dot(σ1, α_h)
    μ_hat_22 = μ22 + np.dot(σ2, α_h)

    # New long-run means
    ι_hat_2 = (μ22 / μ_hat_22) * ι2
    ι_hat_1 = (ι1
               + (1.0/μ11) * (μ12*ι2 - μ_hat_12*ι_hat_2))

    return dict(
        μ_hat_11 = μ_hat_11,
        μ_hat_12 = μ_hat_12,
        μ_hat_22 = μ_hat_22,
        ι_hat_1 = ι_hat_1,
        ι_hat_2 = ι_hat_2,
        α_h = α_h,
        σ1 = σ1,
        σ2 = σ2,
    )


phat_dyn = compute_phat_dynamics(lrr_params, e1, e2, α_s)

print("Dynamics of X under P_hat  (vs physical P):")
print(f"  μ_hat_11 = {phat_dyn['μ_hat_11']:.4f}  "
      f"(same as physical μ_bar_11 = {lrr_params['μ11']:.4f})")
print(f"  μ_hat_12 = {phat_dyn['μ_hat_12']:.6f}  "
      f"(physical = 0 — new coupling created by risk adjustment)")
print(f"  μ_hat_22 = {phat_dyn['μ_hat_22']:.5f}  "
      f"(physical = {lrr_params['μ22']:.4f})")
print(f"  ι_hat_1  = {phat_dyn['ι_hat_1']:.5f}  "
      f"(physical ι1 = {lrr_params['ι1']:.4f}  — lower mean growth under P_hat)")
print(f"  ι_hat_2  = {phat_dyn['ι_hat_2']:.5f}  "
      f"(physical ι2 = {lrr_params['ι2']:.4f}  — higher mean volatility under P_hat)")
```

### Simulating and comparing stationary distributions

```{code-cell} ipython3
def simulate_lrr(dyn, T=600_000, seed=42):
    """
    Simulate the LRR state vector using Euler-Maruyama (monthly steps).

    Parameters
    ----------
    dyn  : dict with μ11, μ12, μ22, ι1, ι2, σ1, σ2
    T    : number of monthly steps
    seed : random seed

    Returns
    -------
    X1, X2 : ndarray — stationary sample paths (burn-in discarded)
    """
    rng = np.random.default_rng(seed)
    μ11 = dyn.get('μ11', dyn.get('μ_hat_11'))
    μ12 = dyn.get('μ12', dyn.get('μ_hat_12', 0.0))
    μ22 = dyn.get('μ22', dyn.get('μ_hat_22'))
    ι1 = dyn.get('ι1', dyn.get('ι_hat_1'))
    ι2 = dyn.get('ι2', dyn.get('ι_hat_2'))
    σ1 = dyn['σ1']
    σ2 = dyn['σ2']

    X1 = np.zeros(T)
    X2 = np.full(T, ι2)

    for t in range(1, T):
        X2t = max(X2[t-1], 1e-9)
        sq_X2 = np.sqrt(X2t)
        dW = rng.standard_normal(3)          # monthly Δt = 1

        X1[t] = X1[t-1] + (μ11*(X1[t-1]-ι1) + μ12*(X2t-ι2)) + sq_X2*np.dot(σ1, dW)
        X2[t] = max(X2[t-1] + μ22*(X2t-ι2) + sq_X2*np.dot(σ2, dW),  1e-9)

    burn = T // 5
    return X1[burn:], X2[burn:]


# Simulation under physical P
print("Simulating under physical measure P ...")
X1_P, X2_P = simulate_lrr(
    dict(μ11=lrr_params['μ11'], μ12=lrr_params['μ12'],
         μ22=lrr_params['μ22'], ι1=lrr_params['ι1'],
         ι2=lrr_params['ι2'],
         σ1=lrr_params['σ1'], σ2=lrr_params['σ2']),
    T=600_000
)

# Simulation under recovered measure P_hat
print("Simulating under recovered measure P_hat ...")
X1_Ph, X2_Ph = simulate_lrr(
    dict(μ_hat_11=phat_dyn['μ_hat_11'],
         μ_hat_12=phat_dyn['μ_hat_12'],
         μ_hat_22=phat_dyn['μ_hat_22'],
         ι_hat_1=phat_dyn['ι_hat_1'],
         ι_hat_2=phat_dyn['ι_hat_2'],
         σ1=lrr_params['σ1'],
         σ2=lrr_params['σ2']),
    T=600_000
)
print("Done.")
```

```{code-cell} ipython3
# Reproduce Figure 1 of Borovicka-Hansen-Scheinkman (2016)
def kde2d_contour(ax, X1, X2, levels=8, color='k', alpha=1.0, lw=1.5,
                  bandwidth=None):
    """Plot contour lines of a 2D kernel density estimate."""
    xy = np.vstack([X2, X1])
    kde = gaussian_kde(xy, bw_method=bandwidth)
    x2g = np.linspace(X2.min()*0.9, X2.max()*1.1, 120)
    x1g = np.linspace(X1.min()*0.9, X1.max()*1.1, 120)
    X2g, X1g = np.meshgrid(x2g, x1g)
    Z = kde(np.vstack([X2g.ravel(), X1g.ravel()])).reshape(X2g.shape)
    ax.contour(X2g, X1g, Z, levels=levels, colors=color, alpha=alpha,
               linewidths=lw)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# Left panel: distribution under P
kde2d_contour(ax1, X1_P, X2_P, color='navy', levels=7)
ax1.set_xlabel('conditional volatility  $X_2$', fontsize=11)
ax1.set_ylabel('mean growth rate  $X_1$', fontsize=11)
ax1.set_title(r'physical measure  $P$', fontsize=12)

# Right panel: distribution under P_hat, plus outermost contour of P_bar (risk-neutral)
kde2d_contour(ax2, X1_Ph, X2_Ph, color='navy', levels=7)
ax2.set_xlabel('conditional volatility  $X_2$', fontsize=11)
ax2.set_title(r'long-term risk-neutral  $\hat{P}$', fontsize=12)

# Annotate distributional shifts
for ax in (ax1, ax2):
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.axvline(lrr_params['ι2'], color='grey', lw=0.8, ls='--')

ax1.annotate(f"mean X1 ~ {X1_P.mean():.4f}", xy=(0.05, 0.92),
             xycoords='axes fraction', fontsize=9, color='navy')
ax1.annotate(f"mean X2 ~ {X2_P.mean():.4f}", xy=(0.05, 0.85),
             xycoords='axes fraction', fontsize=9, color='navy')
ax2.annotate(f"mean X1 ~ {X1_Ph.mean():.4f}", xy=(0.05, 0.92),
             xycoords='axes fraction', fontsize=9, color='navy')
ax2.annotate(f"mean X2 ~ {X2_Ph.mean():.4f}", xy=(0.05, 0.85),
             xycoords='axes fraction', fontsize=9, color='navy')

plt.suptitle('stationary distributions of $(X_1, X_2)$ under $P$ and $\\hat{P}$\n'
             '(reproducing Figure 1 of Borovička, Hansen & Scheinkman 2016)',
             fontsize=12, y=1.02)
plt.tight_layout();  plt.show()
```

The recovered measure $\hat{P}$ concentrates around **lower mean growth** (more negative
$X_1$) and **higher conditional volatility** (larger $X_2$).

Forecasts made using
$\hat{P}$ are systematically pessimistic compared to forecasts based on the true
distribution $P$.

## Measuring the martingale component

### Entropy bounds

Even without observing the full array of Arrow prices, we can obtain **lower bounds**
on the size of the martingale component.

For a convex function
$\phi_\theta(r) = [(r)^{1+\theta} - 1] / [\theta(1+\theta)]$, the discrepancy
between $\hat{P}$ and $P$ satisfies

$$
\lambda_\theta = E\!\left[\phi_\theta\!\left(\frac{\hat{H}_{t+1}}{\hat{H}_t}\right)\right]
\geq 0,
$$

with equality if and only if the martingale is trivial.

Two special cases are:

- **$\theta = -1$**: $\phi_{-1}(r) = -\log r$, so $\lambda_{-1} = -E[\log(\hat{H}_{t+1}/\hat{H}_t)]$ is the **expected log-likelihood** (entropy).
- **$\theta = 1$**: $\lambda_1 = \tfrac{1}{2}\mathrm{Var}[\hat{H}_{t+1}/\hat{H}_t]$.

```{code-cell} ipython3
def φ_θ(r, θ):
    """Discrepancy function φ_θ(r) = [(r)^{1+θ} - 1] / [θ(1+θ)]."""
    if abs(θ) < 1e-10:      # θ -> 0: relative entropy r log r
        return r * np.log(r)
    if abs(θ + 1) < 1e-10:  # θ -> -1: -log r
        return -np.log(r)
    return (r**(1 + θ) - 1) / (θ * (1 + θ))


def martingale_entropy(Q, P, θ=-1):
    """
    Compute the stationary-average discrepancy E[φ_θ(h_hat)] for the finite-state chain.
    """
    _, exp_η, e_hat, P_hat = perron_frobenius(Q)
    H_incr = np.where(P > 0, P_hat / P, 1.0)  # h_hat_ij
    π_hat = stationary_dist(P_hat)

    # Stationary-average: sum_i sum_j π_hat_i h_hat_ij p_ij  φ_θ(h_hat_ij)
    disc = 0.0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] > 0:
                disc += π_hat[i] * P[i, j] * φ_θ(H_incr[i, j], θ)
    return disc


# Compute entropy for different γ values
γs_ent = np.linspace(1.0, 10.0, 50)  # γ=1 handled by solve_ez_finite
entropies = {'θ=-1 (neg. log)': [], 'θ=0 (rel. entropy)': [], 'θ=1 (variance/2)': []}

for γ_val in γs_ent:
    v_g, _, S_g = solve_ez_finite(P_phys, c_levels, δ, γ_val, gc_ex)
    Q_g = S_g * P_phys
    for θ, key in [(-1, 'θ=-1 (neg. log)'), (0, 'θ=0 (rel. entropy)'),
                   (1, 'θ=1 (variance/2)')]:
        entropies[key].append(martingale_entropy(Q_g, P_phys, θ=θ))

fig, ax = plt.subplots(figsize=(8, 4.5))
colors_ent = ['firebrick', 'darkorange', 'steelblue']
for (label, vals), col in zip(entropies.items(), colors_ent):
    ax.plot(γs_ent, vals, label=label, color=col, lw=2)

ax.set_xlabel('risk aversion  γ')
ax.set_ylabel(r'$E[\phi_\theta(\hat{H}_{t+1}/\hat{H}_t)]$')
ax.set_title('discrepancy measures for the martingale component\n'
             '(larger values <-> larger deviation from Ross recovery)')
ax.legend(fontsize=9)
plt.tight_layout();  plt.show()
```

All three discrepancy measures increase with risk aversion, confirming that a higher
$\gamma$ implies a larger — and more economically significant — martingale component.

{cite}`AlvarezJermann2005` and {cite}`BakshiChabiYo2012` use analogous bounds with
long-maturity bond returns to find empirically large martingale components in U.S. data.

## Exercises

```{exercise}
:label: ex_risk_neutral

**Verify risk-neutral probabilities.** Consider a two-state Markov chain with physical
transition matrix

$$
\mathbf{P} = \begin{pmatrix} 0.8 & 0.2 \\ 0.4 & 0.6 \end{pmatrix}
$$

and Arrow price matrix

$$
\mathbf{Q} = \begin{pmatrix} 0.72 & 0.15 \\ 0.36 & 0.42 \end{pmatrix}.
$$

1. Compute the risk-neutral transition matrix $\bar{\mathbf{P}}$ and verify it is a
   valid probability matrix.
2. Compute the one-period discount bond prices and the implied risk-free rates in each
   state.
3. Show that the SDF $\bar{s}_{ij} = \bar{q}_i$ is independent of the next state $j$.
```

```{solution-start} ex_risk_neutral
:class: dropdown
```

```{code-cell} ipython3
# Exercise 1 solution
P2 = np.array([[0.8, 0.2],
               [0.4, 0.6]])
Q2 = np.array([[0.72, 0.15],
               [0.36, 0.42]])

P_bar2, q_bonds2 = risk_neutral_probs(Q2)

print("Risk-neutral transition matrix P_bar:")
print(np.round(P_bar2, 4))
print(f"\nRow sums: {P_bar2.sum(axis=1)}")
print(f"\nOne-period bond prices q_bar_i: {q_bonds2}")
print(f"Annualized risk-free rates: {(-np.log(q_bonds2)*12).round(4)}")

# Verify SDF independence from j
S2 = Q2 / P2
print(f"\nSDF matrix S = Q/P:")
print(np.round(S2, 4))
print("Row 0: all entries should equal q_bar0 =", round(q_bonds2[0], 4))
print("Row 1: all entries should equal q_bar1 =", round(q_bonds2[1], 4))
```

```{solution-end}
```

```{exercise}
:label: ex_gamma_sensitivity

**Risk aversion and recovery distortion.** Using the three-state example from the
lecture (with $\delta = 0.99$ and trend-stationary consumption levels
$c = [0.85, 1.00, 1.15]$), investigate how the recovered probability
vector $\hat{\boldsymbol{\pi}}$ depends on the risk aversion parameter $\gamma$.

1. For each $\gamma \in \{1, 2, 5, 10, 15\}$, compute the long-term risk-neutral
   stationary distribution $\hat{\boldsymbol{\pi}}$.
2. Plot all five distributions as grouped bar charts alongside the physical
   distribution $\boldsymbol{\pi}$.
3. At what value of $\gamma$ does the recession probability under $\hat{\mathbf{P}}$
   exceed $50\%$?
```

```{solution-start} ex_gamma_sensitivity
:class: dropdown
```

```{code-cell} ipython3
# Exercise 2 solution
γs_ex2 = [1, 2, 5, 10, 15]
all_π = []

for γ_val in γs_ex2:
    Q_g = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Q_g[i, j] = δ * (c_levels[j]/c_levels[i])**(-γ_val) * P_phys[i, j]
    _, _, _, Ph_g = perron_frobenius(Q_g)
    all_π.append(stationary_dist(Ph_g))

fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(3)
w = 0.13
colors_g = plt.cm.Blues(np.linspace(0.3, 0.9, len(γs_ex2)))

# Physical distribution
bars = ax.bar(x - 3*w, π_phys, width=w, color='grey', alpha=0.7, label='physical P')
for b_, v in zip(bars, π_phys):
    ax.text(b_.get_x()+w/2, v+0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=7)

for k, (γ_val, π_g, col) in enumerate(zip(γs_ex2, all_π, colors_g)):
    bars = ax.bar(x + (k-1.5)*w, π_g, width=w, color=col,
                  label=f'γ={γ_val}')
    for b_, v in zip(bars, π_g):
        ax.text(b_.get_x()+w/2, v+0.005, f'{v:.3f}',
                ha='center', va='bottom', fontsize=7)

ax.set_xticks(x);  ax.set_xticklabels(state_names)
ax.set_ylabel('stationary probability')
ax.set_title(r'stationary distribution of $\hat{P}$ for varying risk aversion $\gamma$')
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout();  plt.show()

# Part 3: find γ where recession probability under P_hat exceeds 50%
γs_fine = np.linspace(1, 30, 200)
rec_probs = []
for γ_val in γs_fine:
    Q_g = np.array([[δ*(c_levels[j]/c_levels[i])**(-γ_val)*P_phys[i,j]
                     for j in range(3)] for i in range(3)])
    _, _, _, Ph_g = perron_frobenius(Q_g)
    rec_probs.append(stationary_dist(Ph_g)[0])

# Interpolate crossing point
idx50 = np.where(np.array(rec_probs) > 0.5)[0]
if len(idx50) > 0:
    print(f"\nRecession prob under P_hat exceeds 50% at approximately γ ~ {γs_fine[idx50[0]]:.1f}")
else:
    print(f"\nRecession prob under P_hat does not exceed 50% for γ <= 30")
    print(f"  Maximum recession prob = {max(rec_probs):.4f} at γ = 30")
```

```{solution-end}
```

```{exercise}
:label: ex_lrr_gamma

**Effect of risk aversion in the long-run risk model.** Repeat the long-run risk
simulation from the lecture for $\gamma \in \{5, 10, 15\}$ (keeping all other
parameters fixed at their calibrated values).

1. For each $\gamma$, compute $(\bar{e}_1, \bar{e}_2)$ and $\hat{\eta}$.
2. Plot $\hat{\iota}_1$ (long-run mean of $X_1$ under $\hat{P}$) as a function of $\gamma$.
   Interpret the result in terms of long-run expected consumption growth.
3. Plot $\hat{\iota}_2$ (long-run mean of $X_2$ under $\hat{P}$) as a function of $\gamma$.
   Interpret in terms of long-run volatility.
```

```{solution-start} ex_lrr_gamma
:class: dropdown
```

```{code-cell} ipython3
# Exercise 3 solution
γs_lrr = np.linspace(2.0, 18.0, 40)
ι_hat_1_vals = []
ι_hat_2_vals = []
η_hat_vals = []

p_copy = dict(lrr_params)  # copy to modify γ

for γ_val in γs_lrr:
    p_copy['γ'] = γ_val
    try:
        v1g, v2g, A_g, _ = solve_value_function(p_copy)
        e1g, e2g, η_g, α_sg = solve_pf_lrr(p_copy, v1g, v2g, A_g)
        dyn_g = compute_phat_dynamics(p_copy, e1g, e2g, α_sg)
        ι_hat_1_vals.append(dyn_g['ι_hat_1'])
        ι_hat_2_vals.append(dyn_g['ι_hat_2'])
        η_hat_vals.append(η_g)
    except Exception:
        ι_hat_1_vals.append(np.nan)
        ι_hat_2_vals.append(np.nan)
        η_hat_vals.append(np.nan)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(γs_lrr, ι_hat_1_vals, color='steelblue', lw=2.5)
axes[0].axhline(lrr_params['ι1'], ls='--', color='grey', lw=1.5,
                label=f"physical ι1 = {lrr_params['ι1']}")
axes[0].set_xlabel('risk aversion  γ');  axes[0].set_ylabel(r'$\hat{\iota}_1$')
axes[0].set_title('long-run mean of $X_1$ under $\\hat{P}$\n(down = lower expected growth)')
axes[0].legend(fontsize=9)

axes[1].plot(γs_lrr, ι_hat_2_vals, color='firebrick', lw=2.5)
axes[1].axhline(lrr_params['ι2'], ls='--', color='grey', lw=1.5,
                label=f"physical ι2 = {lrr_params['ι2']}")
axes[1].set_xlabel('risk aversion  γ');  axes[1].set_ylabel(r'$\hat{\iota}_2$')
axes[1].set_title('long-run mean of $X_2$ under $\\hat{P}$\n(up = higher expected volatility)')
axes[1].legend(fontsize=9)

axes[2].plot(γs_lrr, np.array(η_hat_vals)*12, color='purple', lw=2.5)
axes[2].set_xlabel('risk aversion  γ');  axes[2].set_ylabel(r'annualized $\hat{\eta}$')
axes[2].set_title('long-run discount rate $\\hat{\\eta}$\n(more negative = higher long-run yield)')

plt.tight_layout();  plt.show()

print("Higher γ -> more negative ι_hat1 (P_hat expects lower growth than P)")
print("Higher γ -> higher ι_hat2 (P_hat expects higher volatility than P)")
```

```{solution-end}
```

```{exercise}
:label: ex_recovery_test

**Testing the Ross recovery condition.** Show algebraically and numerically that, for
any $n$-state power-utility model with trend-stationary consumption (as in Example 1 of
{cite}`BorovickaHansenScheinkman2016`), the martingale increment satisfies
$\hat{h}_{ij} \equiv 1$.

1. Write the SDF as $s_{ij} = A \cdot (c_j/c_i)^{-\gamma}$ for some constant $A$.
   Show that the Perron-Frobenius eigenvector is $\hat{e}_j = c_j^\gamma$ (up to scale)
   and find $\hat{\eta}$.
2. Compute $\hat{p}_{ij} = \exp(-\hat{\eta}) q_{ij} \hat{e}_j / \hat{e}_i$ and verify
   it equals $p_{ij}$.
3. Confirm numerically for the three-state example with $\gamma = 5$ and
   $c = [0.85, 1.00, 1.15]$.
```

```{solution-start} ex_recovery_test
:class: dropdown
```

**Analytical derivation:**

With $s_{ij} = A \cdot (c_j/c_i)^{-\gamma}$ we have $q_{ij} = A(c_j/c_i)^{-\gamma} p_{ij}$.

Guess $\hat{e}_j = c_j^\gamma$.

Then

$$
[\mathbf{Q} \hat{\mathbf{e}}]_i
= \sum_j q_{ij} \hat{e}_j
= A \sum_j \frac{c_j^{-\gamma}}{c_i^{-\gamma}} p_{ij} \cdot c_j^\gamma
= A \sum_j p_{ij}
= A.
$$

So $\mathbf{Q}\hat{\mathbf{e}} = A \hat{\mathbf{e}}$, confirming $\hat{\mathbf{e}} = \{c_j^\gamma\}$
and $\exp(\hat{\eta}) = A$.  Therefore

$$
\hat{p}_{ij}
= \frac{1}{A} q_{ij} \frac{\hat{e}_j}{\hat{e}_i}
= \frac{1}{A} \cdot A \frac{c_j^{-\gamma}}{c_i^{-\gamma}} p_{ij}
  \cdot \frac{c_j^\gamma}{c_i^\gamma}
= p_{ij}.
$$

Hence $\hat{h}_{ij} = \hat{p}_{ij}/p_{ij} = 1$ for all $(i,j)$.

```{code-cell} ipython3
# Exercise 4 numerical verification
# Use trend-stationary power utility (Section "When Does Recovery Succeed?")
gc_ex4 = 0.002
S_ts = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        S_ts[i, j] = np.exp(-δ - γ*gc_ex4) * (c_levels[j]/c_levels[i])**(-γ)

Q_ts = S_ts * P_phys

# Perron-Frobenius
_, exp_η_ts, e_hat_ts, P_hat_ts = perron_frobenius(Q_ts)

# Check eigenvector is proportional to c^γ
e_theory = c_levels**γ
e_theory /= e_theory.sum()

print("Computed eigenvector e_hat:", np.round(e_hat_ts, 6))
print("Theoretical c^γ / norm: ", np.round(e_theory, 6))
print(f"Max discrepancy: {np.abs(e_hat_ts - e_theory).max():.2e}")

H_ts = np.where(P_phys > 0, P_hat_ts / P_phys, 0.0)
print(f"\nMartingale increment matrix h_hat:")
print(np.round(H_ts, 6))
print(f"Max |h_hat_ij - 1|: {np.abs(H_ts[P_phys>0] - 1).max():.2e}")
print("-> Recovery is exact for trend-stationary power utility.")
```

```{solution-end}
```

