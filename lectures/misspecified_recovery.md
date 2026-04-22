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
states and their valuations of different risks.  A long-standing question in finance is
whether we can *recover* the probability distribution used by investors — their subjective
beliefs — from observed asset prices alone.

{cite}`BorovickaHansenScheinkman2016` study the challenge of separating investors'
beliefs from their risk preferences using **Perron–Frobenius theory**.  The key finding
is that Perron–Frobenius theory applied to Arrow prices recovers a **long-term risk-neutral
measure** that absorbs all long-horizon risk adjustments.  This recovered measure coincides
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

## Arrow Prices and the Identification Challenge

### Arrow prices and stochastic discount factors

Consider a discrete-time economy with an $n$-state Markov chain $\{X_t\}$ governed
by transition matrix $\mathbf{P} = [p_{ij}]$.  An **Arrow price** $q_{ij}$ is the
date-$t$ price of a claim that pays $\$1$ tomorrow in state $j$ given that the current
state is $i$.  We collect these prices in a matrix $\mathbf{Q} = [q_{ij}]$.

A **stochastic discount factor** (SDF) $s_{ij}$ prices risk by discounting the payoff
in state $j$ tomorrow when today's state is $i$.  Arrow prices and the SDF are linked by

$$
q_{ij} = s_{ij} \, p_{ij}.
$$

Given $\mathbf{Q}$, any pair $(\mathbf{S}, \mathbf{P})$ satisfying $q_{ij} = s_{ij} p_{ij}$
for all $(i,j)$ is consistent with the observed prices.  The fundamental identification
problem is that $\mathbf{Q}$ has $n^2$ entries, $\mathbf{P}$ has $n(n-1)$ free entries
(rows sum to one), and $\mathbf{S}$ has $n^2$ free entries — so there are far more
unknowns than equations.

To make progress, we can impose restrictions on the SDF.  Two classical restrictions are
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
state_names = ['Recession', 'Normal', 'Expansion']

# Preference parameters
delta  = 0.99    # monthly discount factor
gamma  = 5.0     # coefficient of relative risk aversion

# Arrow price matrix under power utility with rational expectations:
#   q_ij = delta * (c_j / c_i)^{-gamma} * p_ij
n = len(c_levels)
Q_mat = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        Q_mat[i, j] = delta * (c_levels[j] / c_levels[i])**(-gamma) * P_phys[i, j]

print("Arrow price matrix Q:")
print(np.round(Q_mat, 5))
print(f"\nSum of each row (= price of risk-free bond): {Q_mat.sum(axis=1).round(5)}")
```

## Risk-Neutral Probabilities

The **risk-neutral restriction** sets

$$
\bar{s}_{i,j} = \bar{q}_i
$$

where $\bar{q}_i = \sum_j q_{ij}$ is the price of a one-period discount bond in state $i$.
Under this restriction all future states are discounted equally from state $i$, so risk
adjustments depend only on the current state.  The resulting risk-neutral probabilities are

$$
\bar{p}_{ij} = \frac{q_{ij}}{\bar{q}_i}.
$$

```{code-cell} ipython3
def risk_neutral_probs(Q):
    """Compute risk-neutral transition matrix from Arrow price matrix."""
    q_bonds = Q.sum(axis=1)            # one-period bond prices
    P_bar   = Q / q_bonds[:, np.newaxis]
    return P_bar, q_bonds


P_bar, q_bonds = risk_neutral_probs(Q_mat)

print("One-period bond prices (risk-free discount factors):")
for i, (s, qb) in enumerate(zip(state_names, q_bonds)):
    print(f"  {s:12s}: {qb:.5f}  (annualized yield ≈ {-np.log(qb)*12:.2%})")

print("\nRisk-neutral transition matrix P̄:")
print(np.round(P_bar, 4))
print(f"\nRow sums: {P_bar.sum(axis=1)}")
```

```{note}
Risk-neutral probabilities absorb **one-period** (short-run) risk adjustments.  They are
widely used in financial engineering but are generally *not* equal to investors' beliefs.
When short-term interest rates vary across states, risk-neutral probabilities are
also horizon-dependent: the $t$-period forward measure differs from $\bar{\mathbf{P}}^t$.
```

## Long-Term Risk-Neutral Probabilities: Perron–Frobenius Theory

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
positive entries.  This gives a unique construction:

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
    eta_hat   : float   — log of the dominant eigenvalue
    exp_eta   : float   — dominant eigenvalue exp(η̂)
    e_hat     : ndarray — dominant eigenvector (positive, normalized to sum=1)
    P_hat     : ndarray — long-term risk-neutral transition matrix
    """
    eigenvalues, eigenvectors = linalg.eig(Q)

    # Dominant eigenvalue: largest real part (real & positive by Perron–Frobenius)
    idx     = np.argmax(eigenvalues.real)
    exp_eta = eigenvalues[idx].real
    e_hat   = eigenvectors[:, idx].real

    # Ensure positive entries (PF guarantees existence; numpy may flip sign)
    if e_hat.mean() < 0:
        e_hat = -e_hat
    e_hat = np.abs(e_hat) / np.abs(e_hat).sum()   # normalize to sum = 1

    eta_hat = np.log(exp_eta)

    # Long-term risk-neutral transition matrix
    # P_hat[i,j] = exp(-η̂) * Q[i,j] * e_hat[j] / e_hat[i]
    P_hat = (1.0 / exp_eta) * Q * e_hat[np.newaxis, :] / e_hat[:, np.newaxis]

    return eta_hat, exp_eta, e_hat, P_hat


eta_hat, exp_eta, e_hat, P_hat = perron_frobenius(Q_mat)

print(f"Dominant eigenvalue  exp(η̂) = {exp_eta:.6f}")
print(f"Log eigenvalue       η̂      = {eta_hat:.5f}  "
      f"(annualized ≈ {eta_hat*12:.4f})")
print(f"\nEigenvector ê = {e_hat.round(5)}")
print(f"\nLong-term risk-neutral P̂:")
print(np.round(P_hat, 4))
print(f"\nRow sums: {P_hat.sum(axis=1)}")
```

### Comparing the three probability measures

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

matrices = [
    (P_phys, r'Physical  $\mathbf{P}$',            'Blues'),
    (P_bar,  r'Risk-neutral  $\bar{\mathbf{P}}$',  'Oranges'),
    (P_hat,  r'Long-term risk-neutral $\hat{\mathbf{P}}$', 'Greens'),
]

for ax, (mat, title, cmap) in zip(axes, matrices):
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=0.85, aspect='auto')
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks(range(n));  ax.set_yticks(range(n))
    ax.set_xticklabels(state_names, rotation=20, fontsize=9)
    ax.set_yticklabels(state_names, fontsize=9)
    ax.set_xlabel('Next state', fontsize=9)
    ax.set_ylabel('Current state', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(n):
        for j in range(n):
            clr = 'white' if mat[i, j] > 0.45 else 'black'
            ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center',
                    fontsize=9, color=clr)

plt.suptitle('Transition Matrices Under Alternative Probability Measures',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# Stationary distributions under each measure
def stationary_dist(P):
    """Compute stationary distribution of an ergodic transition matrix P."""
    n = P.shape[0]
    A      = (P.T - np.eye(n))
    A[-1]  = 1.0
    b      = np.zeros(n);  b[-1] = 1.0
    return linalg.solve(A, b)

pi_phys = stationary_dist(P_phys)
pi_bar  = stationary_dist(P_bar)
pi_hat  = stationary_dist(P_hat)

fig, ax = plt.subplots(figsize=(8, 4))
x  = np.arange(n)
w  = 0.25
labels = [r'Physical $P$', r'Risk-neutral $\bar{P}$',
          r'Long-term risk-neutral $\hat{P}$']
colors = ['steelblue', 'darkorange', 'forestgreen']
for k, (pi, lbl, col) in enumerate(zip([pi_phys, pi_bar, pi_hat], labels, colors)):
    bars = ax.bar(x + k*w, pi, width=w, label=lbl, color=col, alpha=0.85,
                  edgecolor='white')
    for b_, v in zip(bars, pi):
        ax.text(b_.get_x() + w/2, v + 0.008, f'{v:.3f}',
                ha='center', va='bottom', fontsize=9)

ax.set_xticks(x + w);  ax.set_xticklabels(state_names)
ax.set_ylabel('Stationary probability')
ax.set_title('Stationary Distributions Under Three Probability Measures')
ax.legend(fontsize=9)
plt.tight_layout();  plt.show()

print("Stationary distributions:")
for lbl, pi in zip(labels, [pi_phys, pi_bar, pi_hat]):
    print(f"  {lbl:45s}: {np.round(pi,4)}")
```

The long-term risk-neutral measure $\hat{\mathbf{P}}$ assigns **higher weight to bad
states** (recession) and **lower weight to good states** (expansion) than the physical
measure $\mathbf{P}$.  This is the risk adjustment for long-run growth uncertainty: a
risk-averse investor's long-run discount rates embed a premium for permanent income risk.

## The Martingale Decomposition

### Decomposing the SDF process

Let $\hat{\mathbf{e}}$ and $\hat{\eta}$ solve the Perron–Frobenius problem.  Define the
process

$$
\frac{\hat{H}_{t+1}}{\hat{H}_t} = (X_t)' \hat{\mathbf{H}} X_{t+1},
\quad \text{where} \quad
\hat{h}_{ij} = \frac{\hat{p}_{ij}}{p_{ij}}.
$$

Because $\sum_j \hat{h}_{ij} p_{ij} = \sum_j \hat{p}_{ij} = 1$, the process $\hat{H}$
is a martingale under the physical measure $\mathbf{P}$.  The accumulated SDF then admits
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

# Trend SDF: ŝ_ij = exp(η̂) * e_hat_i / e_hat_j
S_hat = exp_eta * e_hat[:, np.newaxis] / e_hat[np.newaxis, :]

# Martingale increment: ĥ_ij = P̂_ij / P_ij  (also = S_ij / Ŝ_ij)
H_incr = np.where(P_phys > 0, P_hat / P_phys, 0.0)

print("SDF matrix S = Q/P:")
print(np.round(S_mat, 4))
print("\nTrend SDF Ŝ = exp(η̂) × ê_i / ê_j:")
print(np.round(S_hat, 4))
print("\nMartingale increment ĥ = Ŝ × H̃_incr (= P̂/P):")
print(np.round(H_incr, 4))

# Verify martingale property: E[ĥ_{ij} | X_t=i] = sum_j ĥ_ij * p_ij = 1
mart_check = (H_incr * P_phys).sum(axis=1)
print(f"\nMartingale property check — E[ĥ | X_t=i] = {mart_check}")
```

Higher risk aversion amplifies the pessimistic distortion: as $\gamma$ increases, the
recovered measure assigns growing probability to the recession state.
(The figures illustrating this appear below, after we define the Epstein–Zin utility
function that is needed to compute them.)

## When Does Recovery Succeed?

### The Ross recovery condition

{cite}`Ross2015` proposes to identify investors' subjective beliefs by imposing

$$
\widetilde{S}_t = \exp(-\delta t) \frac{m(X_t)}{m(X_0)}
$$

for some positive function $m$ and discount rate $\delta$ (Condition 4 in
{cite}`BorovickaHansenScheinkman2016`).  Under this restriction, the SDF has **no
martingale component**: $\hat{H}_t \equiv 1$.

Equivalently, recovery succeeds if and only if the physical stochastic discount factor
takes the "long-term risk pricing" form

$$
s_{ij} = \exp(\hat{\eta}) \frac{\hat{e}_i}{\hat{e}_j}
$$

with $\hat{h}_{ij} \equiv 1$.  In this case $\hat{\mathbf{P}} = \mathbf{P}$ and the
Perron–Frobenius procedure recovers the true probabilities.

The critical question is: when is the martingale component degenerate?

### Power utility with trend-stationary consumption

Consider a power-utility investor with risk aversion $\gamma$ and *trend-stationary*
consumption $C_t = \exp(g_c t)(c \cdot X_t)$ where $c$ is a positive vector.  The
one-period SDF is

$$
s_{ij} = \exp(-\delta - \gamma g_c) \left(\frac{c_j}{c_i}\right)^{-\gamma}.
$$

This has the exact long-term risk pricing form with $\hat{e}_j = c_j^\gamma$ and
$\hat{\eta} = -(\delta + \gamma g_c)$.  Therefore $\hat{h}_{ij} \equiv 1$ and **Ross
recovery succeeds exactly** when consumption fluctuations around a deterministic trend
are the only source of risk.

```{code-cell} ipython3
# Verify: for trend-stationary power utility, ĥ_ij = 1 identically
gc = 0.002   # monthly trend growth

# Trend-stationary: consumption growth ratio depends only on state, not history
# s_ij = exp(-delta - gamma*gc) * (c_j/c_i)^(-gamma)
S_trend = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        S_trend[i, j] = np.exp(-delta - gamma*gc) * (c_levels[j]/c_levels[i])**(-gamma)

Q_trend = S_trend * P_phys

_, exp_eta_t, e_hat_t, P_hat_t = perron_frobenius(Q_trend)

H_incr_trend = np.where(P_phys > 0, P_hat_t / P_phys, 0.0)

print("Martingale increment ĥ_ij for trend-stationary power utility:")
print(np.round(H_incr_trend, 6))
print(f"\nMax deviation from 1: {np.abs(H_incr_trend[P_phys>0] - 1).max():.2e}")
print("→ Martingale is trivial: Recovery succeeds.")
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
$\mathbf{P}$.  The additional factor $v^*_j/(\mathbf{P}_i v^*)$ introduces a **nontrivial
martingale component** whenever continuation values vary across states.

```{code-cell} ipython3
def solve_ez_finite(P, c, delta, gamma, gc, tol=1e-12, max_iter=5000):
    """
    Solve for Epstein-Zin continuation values in finite Markov chain.

    Solves the fixed-point v_i = (1-β)log(c_i) + β/(1-γ) log(P_i @ exp((1-γ)v))
    where β = exp(-delta - gc).  The special case γ = 1 (log utility) is handled
    separately to avoid the 0/0 indeterminate form: the recursion reduces to
    v = (I - β P)^{-1} (1-β) log(c) and the SDF simplifies to
    s_ij = exp(-δ - g_c) c_i / c_j.

    Returns
    -------
    v     : ndarray — continuation values (net of time trend)
    vstar : ndarray — exp((1-γ)v)
    s     : ndarray — one-period SDF matrix
    """
    beta  = np.exp(-delta - gc)
    log_c = np.log(c)
    n     = len(c)

    if abs(gamma - 1.0) < 1e-10:
        # Log utility: (I - β P) v = (1-β) log c
        v     = linalg.solve(np.eye(n) - beta * P, (1 - beta) * log_c)
        vstar = np.ones(n)     # exp((1-1)*v) = 1
        Pv    = np.ones(n)     # P @ ones = ones
    else:
        # General recursive utility: fixed-point iteration
        v = log_c.copy()
        for _ in range(max_iter):
            vstar = np.exp((1 - gamma) * v)
            Pv    = P @ vstar
            v_new = ((1 - beta) * log_c
                     + beta / (1 - gamma) * np.log(Pv))
            if np.max(np.abs(v_new - v)) < tol:
                v = v_new
                break
            v = v_new
        vstar = np.exp((1 - gamma) * v)
        Pv    = P @ vstar

    # SDF matrix
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s[i, j] = np.exp(-delta - gc) * (c[i] / c[j]) * (vstar[j] / Pv[i])

    return v, vstar, s


# Compare: γ = 1 (log utility, degenerate martingale) vs γ = 5
gc_ex = 0.001   # monthly consumption trend growth

for gam, label in [(1.0, 'γ = 1  (log utility)'), (5.0, 'γ = 5  (risk aversion)')]:
    v_ez, vstar_ez, S_ez = solve_ez_finite(P_phys, c_levels,
                                            delta, gam, gc_ex)
    Q_ez    = S_ez * P_phys
    _, _, _, P_hat_ez = perron_frobenius(Q_ez)
    H_ez    = np.where(P_phys > 0, P_hat_ez / P_phys, 0.0)

    pi_hat_ez  = stationary_dist(P_hat_ez)
    print(f"\n{label}")
    print(f"  Continuation values v = {v_ez.round(4)}")
    print(f"  Max |ĥ_ij - 1|        = {np.abs(H_ez[P_phys>0] - 1).max():.4f}")
    print(f"  Stationary P̂         = {pi_hat_ez.round(4)}")
    print(f"  Stationary P          = {pi_phys.round(4)}")
```

```{code-cell} ipython3
# Show how the martingale depends on γ for recursive utility
# Start at 1.0: the gamma=1 special case in solve_ez_finite is handled explicitly.
gammas_ez = np.linspace(1.0, 10.0, 50)
mart_errors = []
pi_rec_hat  = []

for gam in gammas_ez:
    v_g, _, S_g = solve_ez_finite(P_phys, c_levels, delta, gam, gc_ex)
    Q_g          = S_g * P_phys
    _, _, _, Ph  = perron_frobenius(Q_g)
    H_g          = np.where(P_phys > 0, Ph / P_phys, 0.0)
    mart_errors.append(np.abs(H_g[P_phys > 0] - 1).max())
    pi_rec_hat.append(stationary_dist(Ph)[0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(gammas_ez, mart_errors, color='firebrick', lw=2.5)
ax1.set_xlabel('Risk aversion  γ')
ax1.set_ylabel(r'$\max_{i,j} |\hat{h}_{ij} - 1|$')
ax1.set_title('Martingale non-degeneracy vs risk aversion\n(Epstein–Zin utility)')

ax2.plot(gammas_ez, pi_rec_hat, color='steelblue', lw=2.5,
         label=r'Recession weight under $\hat{P}$')
ax2.axhline(pi_phys[0], ls='--', color='grey', lw=1.5,
            label=f'Recession weight under $P$  ({pi_phys[0]:.3f})')
ax2.set_xlabel('Risk aversion  γ')
ax2.set_ylabel('Stationary probability')
ax2.set_title('Recovered recession probability vs risk aversion')
ax2.legend(fontsize=9)

plt.tight_layout();  plt.show()
```

```{code-cell} ipython3
# Visualize the martingale increment using Epstein-Zin utility (γ=5).
# Trend-stationary power utility always yields ĥ_ij = 1 by construction (see Exercise 4),
# so we use recursive utility here to reveal a genuinely non-trivial martingale.
gamma_ez_demo = 5.0
_, _, S_ez_demo = solve_ez_finite(P_phys, c_levels, delta, gamma_ez_demo, gc_ex)
Q_ez_demo      = S_ez_demo * P_phys
_, _, _, P_hat_ez_demo = perron_frobenius(Q_ez_demo)
H_incr_ez      = np.where(P_phys > 0, P_hat_ez_demo / P_phys, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

vmax_h = max(1.5, H_incr_ez.max() * 1.05)
vmin_h = min(0.5, H_incr_ez.min() * 0.95)
im0 = axes[0].imshow(H_incr_ez, cmap='RdYlGn', vmin=vmin_h, vmax=vmax_h, aspect='auto')
axes[0].set_title(
    r'Martingale increment $\hat{h}_{ij} = \hat{p}_{ij}/p_{ij}$' '\n'
    r'(Epstein–Zin utility, $\gamma=5$)',
    fontsize=11)
for i in range(n):
    for j in range(n):
        axes[0].text(j, i, f'{H_incr_ez[i,j]:.3f}',
                     ha='center', va='center', fontsize=10)
axes[0].set_xticks(range(n));  axes[0].set_yticks(range(n))
axes[0].set_xticklabels(state_names, rotation=20, fontsize=9)
axes[0].set_yticklabels(state_names, fontsize=9)
axes[0].set_xlabel('Next state');  axes[0].set_ylabel('Current state')
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# How risk aversion γ shifts the recovered measure under Epstein-Zin utility.
gammas_shift = np.linspace(1.0, 12, 60)
rec_wts_ez   = []
for g in gammas_shift:
    _, _, S_g = solve_ez_finite(P_phys, c_levels, delta, g, gc_ex)
    Q_g       = S_g * P_phys
    _, _, _, Ph = perron_frobenius(Q_g)
    rec_wts_ez.append(stationary_dist(Ph)[0])

axes[1].plot(gammas_shift, rec_wts_ez, color='steelblue', lw=2.5)
axes[1].axhline(pi_phys[0], color='grey', ls='--', lw=1.5,
                label=fr'Physical recession prob = {pi_phys[0]:.3f}')
axes[1].set_xlabel('Risk aversion  γ')
axes[1].set_ylabel(r'Recession weight under $\hat{P}$')
axes[1].set_title(r'How $\gamma$ shifts the long-term risk-neutral measure'
                  '\n(Epstein–Zin utility)')
axes[1].legend(fontsize=9)
plt.tight_layout();  plt.show()
```

At $\gamma = 1$ (log utility), the continuation value is constant across states and the
martingale is trivial, so recovery succeeds.  For $\gamma > 1$, continuation values vary
with the state, generating a non-degenerate martingale that grows with risk aversion.

## The Long-Run Risk Model

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

where $W_t$ is a three-dimensional Brownian motion.  Here $X_{1t}$ is the
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
    delta    = 0.002,          # subjective discount rate
    gamma    = 10.0,           # risk aversion
    mu11     = -0.021,         # mean reversion of X1
    mu12     = 0.0,            # (under P; becomes non-zero under P̂)
    mu22     = -0.013,         # mean reversion of X2
    iota1    = 0.0,            # long-run mean of X1
    iota2    = 1.0,            # long-run mean of X2 (normalized)
    sigma1   = np.array([0.0,  0.00034, 0.0  ]),   # diffusion of X1 (1×3)
    sigma2   = np.array([0.0,  0.0,    -0.038]),    # diffusion of X2 (1×3)
    beta_c0  = 0.0015,         # consumption drift constant
    beta_c1  = 1.0,            # loading on X1
    beta_c2  = 0.0,            # loading on X2
    alpha_c  = np.array([0.0078, 0.0, 0.0]),        # consumption diffusion (1×3)
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
    delta, gamma     = p['delta'], p['gamma']
    mu11, mu12, mu22 = p['mu11'],  p['mu12'], p['mu22']
    sigma1, sigma2   = p['sigma1'], p['sigma2']
    beta_c1, beta_c2 = p['beta_c1'], p['beta_c2']
    mu12_p, alpha_c  = p['mu12'], p['alpha_c']

    # Linear equation for v̄₁
    # δ v̄₁ = β̄c,1 + μ̄₁₁ v̄₁  ⟹  v̄₁ = β̄c,1 / (δ - μ̄₁₁)
    v1 = beta_c1 / (delta - mu11)

    # Quadratic equation for v̄₂
    # 0 = (μ̄₂₂ - δ)v̄₂ + β̄c,2 + μ̄₁₂v̄₁ + ½(1-γ)|A + B v̄₂|²
    # where A = ᾱc + σ̄₁ v̄₁,  B = σ̄₂
    A_vec = alpha_c + sigma1 * v1
    B_vec = sigma2

    a = 0.5 * (1 - gamma) * np.dot(B_vec, B_vec)
    b = (mu22 - delta) + (1 - gamma) * np.dot(A_vec, B_vec)
    c = beta_c2 + mu12 * v1 + 0.5 * (1 - gamma) * np.dot(A_vec, A_vec)

    disc = b**2 - 4*a*c
    if disc < 0:
        raise ValueError("Value function does not exist for these parameters.")

    # "Minus" solution (generates ergodic dynamics under P̂)
    v2 = (-b - np.sqrt(disc)) / (2 * a)
    return v1, v2, A_vec, B_vec


v1, v2, A_vec, B_vec = solve_value_function(lrr_params)
print(f"Value-function slope on X1:  v̄₁ = {v1:.4f}")
print(f"Value-function slope on X2:  v̄₂ = {v2:.4f}")
print(f"\nInterpretation:")
print(f"  Higher X1 (better expected growth) raises continuation value (v̄₁ > 0)")
print(f"  Higher X2 (more volatility) lowers continuation value (v̄₂ < 0)")
```

### Perron–Frobenius and recovered dynamics

```{code-cell} ipython3
def solve_pf_lrr(p, v1, v2, A_vec):
    """
    Solve the Perron-Frobenius problem for the long-run risk model.

    Eigenfunction guess: ê(x) = exp(ē₁ x₁ + ē₂ x₂).

    Returns ē₁, ē₂, η̂, and the SDF diffusion vector ᾱs.
    """
    delta, gamma     = p['delta'], p['gamma']
    mu11, mu12, mu22 = p['mu11'],  p['mu12'], p['mu22']
    iota1, iota2     = p['iota1'], p['iota2']
    sigma1, sigma2   = p['sigma1'], p['sigma2']
    alpha_c          = p['alpha_c']
    beta_c0          = p['beta_c0']
    beta_c1, beta_c2 = p['beta_c1'], p['beta_c2']

    # SDF diffusion:  ᾱs = −γ ᾱc + (1−γ)(σ̄₁v̄₁ + σ̄₂v̄₂)
    alpha_s = (-gamma * alpha_c
               + (1 - gamma) * (sigma1 * v1 + sigma2 * v2))

    # SDF drift parameters in  βs(x) = β̄s0 + β̄s11(x1−ι1) + β̄s12(x2−ι2)
    beta_s11 = -beta_c1
    beta_s12 = -beta_c2 - 0.5 * np.dot(alpha_s, alpha_s)
    beta_s0  = (-delta - beta_c0
                - 0.5 * iota2 * np.dot(alpha_s, alpha_s))

    # Equation 0 = β̄s11 + μ̄₁₁ ē₁  ⟹  ē₁ = −β̄s11 / μ̄₁₁
    e1 = -beta_s11 / mu11

    # Quadratic for ē₂
    # 0 = (β̄s12 + ½|ᾱs|²)  +  ē₁(μ̄₁₂ + σ̄₁·ᾱs) + ½ē₁²|σ̄₁|²
    #     + ē₂(μ̄₂₂ + σ̄₂·ᾱs + ē₁ σ̄₁·σ̄₂') + ½ē₂²|σ̄₂|²
    const_pf = (beta_s12 + 0.5*np.dot(alpha_s, alpha_s)    # = 0 by construction
                + e1*(mu12 + np.dot(sigma1, alpha_s))
                + 0.5*e1**2*np.dot(sigma1, sigma1))
    lin_pf   = mu22 + np.dot(sigma2, alpha_s) + e1*np.dot(sigma1, sigma2)
    quad_pf  = 0.5 * np.dot(sigma2, sigma2)

    disc = lin_pf**2 - 4*quad_pf*const_pf
    e2_m = (-lin_pf - np.sqrt(disc)) / (2*quad_pf)
    e2_p = (-lin_pf + np.sqrt(disc)) / (2*quad_pf)

    # η̂ = β̄s0 - β̄s12·ι₂ - ē₂·μ̄₂₂·ι₂  (ι₁ = 0)
    eta_m = beta_s0 - beta_s12*iota2 - e2_m*mu22*iota2
    eta_p = beta_s0 - beta_s12*iota2 - e2_p*mu22*iota2

    # Choose solution with smaller |η̂| (ergodicity requirement)
    if abs(eta_m) <= abs(eta_p):
        e2, eta_hat = e2_m, eta_m
    else:
        e2, eta_hat = e2_p, eta_p

    return e1, e2, eta_hat, alpha_s


e1, e2, eta_hat_lrr, alpha_s = solve_pf_lrr(lrr_params, v1, v2, A_vec)

print(f"PF eigenfunction coefficients:  ē₁ = {e1:.4f},  ē₂ = {e2:.4f}")
print(f"Log eigenvalue:                 η̂  = {eta_hat_lrr:.6f}  "
      f"(annualized = {eta_hat_lrr*12:.4f})")
print(f"\nInterpretation:")
print(f"  ē₁ = {e1:.2f}: ê down-weights high-X1 (good growth) states")
print(f"  ē₂ = {e2:.2f}: ê up-weights high-X2 (high volatility) states")
```

### Computing the P̂ dynamics

```{code-cell} ipython3
def compute_phat_dynamics(p, e1, e2, alpha_s):
    """
    Compute the drift parameters of X under the recovered measure P̂.

    Under P̂, the Brownian motion is
        dŴ_t = −√X₂_t α̂_h dt + dW_t
    where α̂_h = ᾱs + σ̄₁ ē₁ + σ̄₂ ē₂.
    """
    mu11, mu12, mu22 = p['mu11'],  p['mu12'], p['mu22']
    iota1, iota2     = p['iota1'], p['iota2']
    sigma1, sigma2   = p['sigma1'], p['sigma2']

    # Martingale drift correction
    alpha_h = alpha_s + sigma1 * e1 + sigma2 * e2

    # New drift parameters under P̂
    mu_hat_11 = mu11
    mu_hat_12 = mu12 + np.dot(sigma1, alpha_h)
    mu_hat_22 = mu22 + np.dot(sigma2, alpha_h)

    # New long-run means
    iota_hat_2 = (mu22 / mu_hat_22) * iota2
    iota_hat_1 = (iota1
                  + (1.0/mu11) * (mu12*iota2 - mu_hat_12*iota_hat_2))

    return dict(
        mu_hat_11  = mu_hat_11,
        mu_hat_12  = mu_hat_12,
        mu_hat_22  = mu_hat_22,
        iota_hat_1 = iota_hat_1,
        iota_hat_2 = iota_hat_2,
        alpha_h    = alpha_h,
        sigma1     = sigma1,
        sigma2     = sigma2,
    )


phat_dyn = compute_phat_dynamics(lrr_params, e1, e2, alpha_s)

print("Dynamics of X under P̂  (vs physical P):")
print(f"  μ̂₁₁ = {phat_dyn['mu_hat_11']:.4f}  "
      f"(same as physical μ̄₁₁ = {lrr_params['mu11']:.4f})")
print(f"  μ̂₁₂ = {phat_dyn['mu_hat_12']:.6f}  "
      f"(physical = 0 — new coupling created by risk adjustment)")
print(f"  μ̂₂₂ = {phat_dyn['mu_hat_22']:.5f}  "
      f"(physical = {lrr_params['mu22']:.4f})")
print(f"  ι̂₁  = {phat_dyn['iota_hat_1']:.5f}  "
      f"(physical ι₁ = {lrr_params['iota1']:.4f}  — lower mean growth under P̂)")
print(f"  ι̂₂  = {phat_dyn['iota_hat_2']:.5f}  "
      f"(physical ι₂ = {lrr_params['iota2']:.4f}  — higher mean volatility under P̂)")
```

### Simulating and comparing stationary distributions

```{code-cell} ipython3
def simulate_lrr(dyn, T=600_000, seed=42):
    """
    Simulate the LRR state vector using Euler-Maruyama (monthly steps).

    Parameters
    ----------
    dyn  : dict with mu11, mu12, mu22, iota1, iota2, sigma1, sigma2
    T    : number of monthly steps
    seed : random seed

    Returns
    -------
    X1, X2 : ndarray — stationary sample paths (burn-in discarded)
    """
    rng     = np.random.default_rng(seed)
    mu11    = dyn.get('mu11',     dyn.get('mu_hat_11'))
    mu12    = dyn.get('mu12',     dyn.get('mu_hat_12', 0.0))
    mu22    = dyn.get('mu22',     dyn.get('mu_hat_22'))
    iota1   = dyn.get('iota1',    dyn.get('iota_hat_1'))
    iota2   = dyn.get('iota2',    dyn.get('iota_hat_2'))
    sigma1  = dyn['sigma1']
    sigma2  = dyn['sigma2']

    X1 = np.zeros(T)
    X2 = np.full(T, iota2)

    for t in range(1, T):
        X2t      = max(X2[t-1], 1e-9)
        sq_X2    = np.sqrt(X2t)
        dW       = rng.standard_normal(3)          # monthly Δt = 1

        X1[t] = X1[t-1] + (mu11*(X1[t-1]-iota1) + mu12*(X2t-iota2)) + sq_X2*np.dot(sigma1, dW)
        X2[t] = max(X2[t-1] + mu22*(X2t-iota2)   + sq_X2*np.dot(sigma2, dW),  1e-9)

    burn = T // 5
    return X1[burn:], X2[burn:]


# Simulation under physical P
print("Simulating under physical measure P ...")
X1_P, X2_P = simulate_lrr(
    dict(mu11=lrr_params['mu11'],  mu12=lrr_params['mu12'],
         mu22=lrr_params['mu22'],  iota1=lrr_params['iota1'],
         iota2=lrr_params['iota2'],
         sigma1=lrr_params['sigma1'], sigma2=lrr_params['sigma2']),
    T=600_000
)

# Simulation under recovered measure P̂
print("Simulating under recovered measure P̂ ...")
X1_Ph, X2_Ph = simulate_lrr(
    dict(mu11    = phat_dyn['mu_hat_11'],
         mu12    = phat_dyn['mu_hat_12'],
         mu22    = phat_dyn['mu_hat_22'],
         iota1   = phat_dyn['iota_hat_1'],
         iota2   = phat_dyn['iota_hat_2'],
         sigma1  = lrr_params['sigma1'],
         sigma2  = lrr_params['sigma2']),
    T=600_000
)
print("Done.")
```

```{code-cell} ipython3
# Reproduce Figure 1 of Borovicka-Hansen-Scheinkman (2016)
def kde2d_contour(ax, X1, X2, levels=8, color='k', alpha=1.0, lw=1.5,
                  bandwidth=None):
    """Plot contour lines of a 2D kernel density estimate."""
    xy     = np.vstack([X2, X1])
    kde    = gaussian_kde(xy, bw_method=bandwidth)
    x2g    = np.linspace(X2.min()*0.9, X2.max()*1.1, 120)
    x1g    = np.linspace(X1.min()*0.9, X1.max()*1.1, 120)
    X2g, X1g = np.meshgrid(x2g, x1g)
    Z      = kde(np.vstack([X2g.ravel(), X1g.ravel()])).reshape(X2g.shape)
    ax.contour(X2g, X1g, Z, levels=levels, colors=color, alpha=alpha,
               linewidths=lw)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# Left panel: distribution under P
kde2d_contour(ax1, X1_P, X2_P, color='navy', levels=7)
ax1.set_xlabel('Conditional volatility  $X_2$', fontsize=11)
ax1.set_ylabel('Mean growth rate  $X_1$',       fontsize=11)
ax1.set_title(r'Physical measure  $P$', fontsize=12)

# Right panel: distribution under P̂, plus outermost contour of P̄ (risk-neutral)
kde2d_contour(ax2, X1_Ph, X2_Ph, color='navy', levels=7)
ax2.set_xlabel('Conditional volatility  $X_2$', fontsize=11)
ax2.set_title(r'Long-term risk-neutral  $\hat{P}$', fontsize=12)

# Annotate distributional shifts
for ax in (ax1, ax2):
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.axvline(lrr_params['iota2'], color='grey', lw=0.8, ls='--')

ax1.annotate(f"Mean X₁ ≈ {X1_P.mean():.4f}", xy=(0.05, 0.92),
             xycoords='axes fraction', fontsize=9, color='navy')
ax1.annotate(f"Mean X₂ ≈ {X2_P.mean():.4f}", xy=(0.05, 0.85),
             xycoords='axes fraction', fontsize=9, color='navy')
ax2.annotate(f"Mean X₁ ≈ {X1_Ph.mean():.4f}", xy=(0.05, 0.92),
             xycoords='axes fraction', fontsize=9, color='navy')
ax2.annotate(f"Mean X₂ ≈ {X2_Ph.mean():.4f}", xy=(0.05, 0.85),
             xycoords='axes fraction', fontsize=9, color='navy')

plt.suptitle('Stationary Distributions of $(X_1, X_2)$ Under $P$ and $\\hat{P}$\n'
             '(reproducing Figure 1 of Borovička, Hansen & Scheinkman 2016)',
             fontsize=12, y=1.02)
plt.tight_layout();  plt.show()
```

The recovered measure $\hat{P}$ concentrates around **lower mean growth** (more negative
$X_1$) and **higher conditional volatility** (larger $X_2$).  Forecasts made using
$\hat{P}$ are systematically pessimistic compared to forecasts based on the true
distribution $P$.

## Measuring the Martingale Component

### Entropy bounds

Even without observing the full array of Arrow prices, we can obtain **lower bounds**
on the size of the martingale component.  For a convex function
$\phi_\theta(r) = [(r)^{1+\theta} - 1] / [\theta(1+\theta)]$, the discrepancy
between $\hat{P}$ and $P$ satisfies

$$
\lambda_\theta = E\!\left[\phi_\theta\!\left(\frac{\hat{H}_{t+1}}{\hat{H}_t}\right)\right]
\geq 0,
$$

with equality if and only if the martingale is trivial.  Two special cases are:

- **$\theta = -1$**: $\phi_{-1}(r) = -\log r$, so $\lambda_{-1} = -E[\log(\hat{H}_{t+1}/\hat{H}_t)]$ is the **expected log-likelihood** (entropy).
- **$\theta = 1$**: $\lambda_1 = \tfrac{1}{2}\mathrm{Var}[\hat{H}_{t+1}/\hat{H}_t]$.

```{code-cell} ipython3
def phi_theta(r, theta):
    """Discrepancy function φ_θ(r) = [(r)^{1+θ} - 1] / [θ(1+θ)]."""
    if abs(theta) < 1e-10:      # θ → 0: relative entropy r log r
        return r * np.log(r)
    if abs(theta + 1) < 1e-10:  # θ → -1: -log r
        return -np.log(r)
    return (r**(1 + theta) - 1) / (theta * (1 + theta))


def martingale_entropy(Q, P, theta=-1):
    """
    Compute the stationary-average discrepancy E[φ_θ(ĥ)] for the finite-state chain.
    """
    _, exp_eta, e_hat, P_hat = perron_frobenius(Q)
    H_incr   = np.where(P > 0, P_hat / P, 1.0)  # ĥ_ij
    pi_hat   = stationary_dist(P_hat)

    # Stationary-average: Σ_i Σ_j π̂_i ĥ_ij p_ij  φ_θ(ĥ_ij)
    disc = 0.0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] > 0:
                disc += pi_hat[i] * P[i, j] * phi_theta(H_incr[i, j], theta)
    return disc


# Compute entropy for different γ values
gammas_ent = np.linspace(1.0, 10.0, 50)  # gamma=1 handled by solve_ez_finite
entropies  = {'θ=-1 (neg. log)': [], 'θ=0 (rel. entropy)': [], 'θ=1 (variance/2)': []}

for gam in gammas_ent:
    v_g, _, S_g = solve_ez_finite(P_phys, c_levels, delta, gam, gc_ex)
    Q_g         = S_g * P_phys
    for theta, key in [(-1, 'θ=-1 (neg. log)'), (0, 'θ=0 (rel. entropy)'),
                        (1, 'θ=1 (variance/2)')]:
        entropies[key].append(martingale_entropy(Q_g, P_phys, theta=theta))

fig, ax = plt.subplots(figsize=(8, 4.5))
colors_ent = ['firebrick', 'darkorange', 'steelblue']
for (label, vals), col in zip(entropies.items(), colors_ent):
    ax.plot(gammas_ent, vals, label=label, color=col, lw=2)

ax.set_xlabel('Risk aversion  γ')
ax.set_ylabel(r'$E[\phi_\theta(\hat{H}_{t+1}/\hat{H}_t)]$')
ax.set_title('Discrepancy Measures for the Martingale Component\n'
             '(larger values ↔ larger deviation from Ross recovery)')
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

print("Risk-neutral transition matrix P̄:")
print(np.round(P_bar2, 4))
print(f"\nRow sums: {P_bar2.sum(axis=1)}")
print(f"\nOne-period bond prices q̄ᵢ: {q_bonds2}")
print(f"Annualized risk-free rates: {(-np.log(q_bonds2)*12).round(4)}")

# Verify SDF independence from j
S2 = Q2 / P2
print(f"\nSDF matrix S = Q/P:")
print(np.round(S2, 4))
print("Row 0: all entries should equal q̄₀ =", round(q_bonds2[0], 4))
print("Row 1: all entries should equal q̄₁ =", round(q_bonds2[1], 4))
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
gammas_ex2 = [1, 2, 5, 10, 15]
all_pi = []

for gam in gammas_ex2:
    Q_g = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Q_g[i, j] = delta * (c_levels[j]/c_levels[i])**(-gam) * P_phys[i, j]
    _, _, _, Ph_g = perron_frobenius(Q_g)
    all_pi.append(stationary_dist(Ph_g))

fig, ax = plt.subplots(figsize=(10, 4.5))
x   = np.arange(3)
w   = 0.13
colors_g = plt.cm.Blues(np.linspace(0.3, 0.9, len(gammas_ex2)))

# Physical distribution
bars = ax.bar(x - 3*w, pi_phys, width=w, color='grey', alpha=0.7, label='Physical P')
for b_, v in zip(bars, pi_phys):
    ax.text(b_.get_x()+w/2, v+0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=7)

for k, (gam, pi_g, col) in enumerate(zip(gammas_ex2, all_pi, colors_g)):
    bars = ax.bar(x + (k-1.5)*w, pi_g, width=w, color=col,
                  label=f'γ={gam}')
    for b_, v in zip(bars, pi_g):
        ax.text(b_.get_x()+w/2, v+0.005, f'{v:.3f}',
                ha='center', va='bottom', fontsize=7)

ax.set_xticks(x);  ax.set_xticklabels(state_names)
ax.set_ylabel('Stationary probability')
ax.set_title(r'Stationary distribution of $\hat{P}$ for varying risk aversion $\gamma$')
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout();  plt.show()

# Part 3: find γ where recession probability under P̂ exceeds 50%
gammas_fine = np.linspace(1, 30, 200)
rec_probs = []
for gam in gammas_fine:
    Q_g = np.array([[delta*(c_levels[j]/c_levels[i])**(-gam)*P_phys[i,j]
                     for j in range(3)] for i in range(3)])
    _, _, _, Ph_g = perron_frobenius(Q_g)
    rec_probs.append(stationary_dist(Ph_g)[0])

# Interpolate crossing point
idx50 = np.where(np.array(rec_probs) > 0.5)[0]
if len(idx50) > 0:
    print(f"\nRecession prob under P̂ exceeds 50% at approximately γ ≈ {gammas_fine[idx50[0]]:.1f}")
else:
    print(f"\nRecession prob under P̂ does not exceed 50% for γ ≤ 30")
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
gammas_lrr = np.linspace(2.0, 18.0, 40)
iota_hat_1_vals = []
iota_hat_2_vals = []
eta_hat_vals    = []

p_copy = dict(lrr_params)  # copy to modify gamma

for gam in gammas_lrr:
    p_copy['gamma'] = gam
    try:
        v1g, v2g, A_g, _ = solve_value_function(p_copy)
        e1g, e2g, eta_g, alpha_sg = solve_pf_lrr(p_copy, v1g, v2g, A_g)
        dyn_g = compute_phat_dynamics(p_copy, e1g, e2g, alpha_sg)
        iota_hat_1_vals.append(dyn_g['iota_hat_1'])
        iota_hat_2_vals.append(dyn_g['iota_hat_2'])
        eta_hat_vals.append(eta_g)
    except Exception:
        iota_hat_1_vals.append(np.nan)
        iota_hat_2_vals.append(np.nan)
        eta_hat_vals.append(np.nan)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(gammas_lrr, iota_hat_1_vals, color='steelblue', lw=2.5)
axes[0].axhline(lrr_params['iota1'], ls='--', color='grey', lw=1.5,
                label=f"Physical ι₁ = {lrr_params['iota1']}")
axes[0].set_xlabel('Risk aversion  γ');  axes[0].set_ylabel(r'$\hat{\iota}_1$')
axes[0].set_title('Long-run mean of $X_1$ under $\\hat{P}$\n(↓ = lower expected growth)')
axes[0].legend(fontsize=9)

axes[1].plot(gammas_lrr, iota_hat_2_vals, color='firebrick', lw=2.5)
axes[1].axhline(lrr_params['iota2'], ls='--', color='grey', lw=1.5,
                label=f"Physical ι₂ = {lrr_params['iota2']}")
axes[1].set_xlabel('Risk aversion  γ');  axes[1].set_ylabel(r'$\hat{\iota}_2$')
axes[1].set_title('Long-run mean of $X_2$ under $\\hat{P}$\n(↑ = higher expected volatility)')
axes[1].legend(fontsize=9)

axes[2].plot(gammas_lrr, np.array(eta_hat_vals)*12, color='purple', lw=2.5)
axes[2].set_xlabel('Risk aversion  γ');  axes[2].set_ylabel(r'Annualized $\hat{\eta}$')
axes[2].set_title('Long-run discount rate $\\hat{\\eta}$\n(more negative = higher long-run yield)')

plt.tight_layout();  plt.show()

print("Higher γ → more negative ι̂₁ (P̂ expects lower growth than P)")
print("Higher γ → higher ι̂₂ (P̂ expects higher volatility than P)")
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
Guess $\hat{e}_j = c_j^\gamma$.  Then

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
S_ts   = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        S_ts[i, j] = np.exp(-delta - gamma*gc_ex4) * (c_levels[j]/c_levels[i])**(-gamma)

Q_ts = S_ts * P_phys

# Perron-Frobenius
_, exp_eta_ts, e_hat_ts, P_hat_ts = perron_frobenius(Q_ts)

# Check eigenvector is proportional to c^gamma
e_theory = c_levels**gamma
e_theory /= e_theory.sum()

print("Computed eigenvector ê:", np.round(e_hat_ts, 6))
print("Theoretical cʸ / norm: ", np.round(e_theory, 6))
print(f"Max discrepancy: {np.abs(e_hat_ts - e_theory).max():.2e}")

H_ts = np.where(P_phys > 0, P_hat_ts / P_phys, 0.0)
print(f"\nMartingale increment matrix ĥ:")
print(np.round(H_ts, 6))
print(f"Max |ĥ_ij - 1|: {np.abs(H_ts[P_phys>0] - 1).max():.2e}")
print("→ Recovery is exact for trend-stationary power utility.")
```

```{solution-end}
```

## References

```{bibliography}
:filter: docname in docnames
```
