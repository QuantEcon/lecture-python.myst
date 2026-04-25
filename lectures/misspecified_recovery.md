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

Asset prices are forward-looking: they encode investors' expectations about future
economic states and their valuations of different risks.

A long-standing question in finance is whether one can *recover* the probability
distribution used by investors -- their subjective beliefs -- from observed asset prices
alone.

{cite:t}`BorovickaHansenScheinkman2016` study the challenge of separating investors'
beliefs from their risk preferences using **Perron–Frobenius theory**.

Their key finding is that Perron–Frobenius theory applied to Arrow prices recovers a
**long-term risk-neutral measure** that absorbs all long-horizon risk adjustments.

This recovered measure coincides with investors' subjective beliefs only under a
stringent -- and often empirically implausible -- restriction on the stochastic discount
factor.

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

Consider a discrete-time economy with an $n$-state Markov chain $\{X_t\}$ governed by
transition matrix $\mathbf{P} = [p_{ij}]$.

An **Arrow price** $q_{ij}$ is the date-$t$ price of a claim that pays $\$1$ tomorrow in
state $j$ given that the current state is $i$.

Collect these prices in a matrix $\mathbf{Q} = [q_{ij}]$.

A **stochastic discount factor** (SDF) $s_{ij}$ prices risk by discounting the payoff in
state $j$ tomorrow when today's state is $i$.

Arrow prices and the SDF are linked by

$$
q_{ij} = s_{ij} \, p_{ij}.
$$

Given $\mathbf{Q}$, any pair $(\mathbf{S}, \mathbf{P})$ satisfying
$q_{ij} = s_{ij} p_{ij}$ for all $(i,j)$ is consistent with the observed prices.

The fundamental identification problem is that $\mathbf{Q}$ has $n^2$ entries,
$\mathbf{P}$ has $n(n-1)$ free entries (rows sum to one), and $\mathbf{S}$ has $n^2$
free entries -- so there are far more unknowns than equations.

To make progress, we can impose restrictions on the SDF.

Two classical restrictions are studied in the sections that follow.

### A three-state illustration

To build intuition, we work with a three-state Markov chain representing **recession**,
**normal**, and **expansion** phases of the business cycle.

The physical transition matrix and consumption levels are:

```{code-cell} ipython3
P_phys = np.array([
    [0.70, 0.25, 0.05],   # from recession
    [0.15, 0.65, 0.20],   # from normal
    [0.05, 0.30, 0.65],   # from expansion
])

c_levels = np.array([0.85, 1.00, 1.15])
state_names = ['recession', 'normal', 'expansion']

δ = -np.log(0.99)  # monthly subjective discount rate, so exp(-δ) = 0.99
γ = 5.0     # coefficient of relative risk aversion

n = len(c_levels)
Q_mat = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        Q_mat[i, j] = np.exp(-δ) * (c_levels[j] / c_levels[i])**(-γ) * P_phys[i, j]

print("Arrow price matrix Q")
print(np.round(Q_mat, 5))
print("Risk-free discount factors:", Q_mat.sum(axis=1).round(5))
```

## Risk-neutral probabilities

The **risk-neutral restriction** sets

$$
\bar{s}_{i,j} = \bar{q}_i
$$

where $\bar{q}_i = \sum_j q_{ij}$ is the price of a one-period discount bond in state
$i$.

Under this restriction all future states are discounted equally from state $i$, so risk
adjustments depend only on the current state.

The resulting risk-neutral probabilities are

$$
\bar{p}_{ij} = \frac{q_{ij}}{\bar{q}_i}.
$$

```{code-cell} ipython3
def risk_neutral_probs(Q):
    """Normalize Arrow prices by one-period bond prices."""
    q_bonds = Q.sum(axis=1)
    P_bar = Q / q_bonds[:, np.newaxis]
    return P_bar, q_bonds


P_bar, q_bonds = risk_neutral_probs(Q_mat)

print("One-period bond prices:")
for i, (s, qb) in enumerate(zip(state_names, q_bonds)):
    print(f"  {s:12s}: {qb:.5f}  (annualized yield ~ {-np.log(qb)*12:.2%})")

print("\nRisk-neutral P_bar:")
print(np.round(P_bar, 4))
print("Row sums:", P_bar.sum(axis=1))
```

```{note}
Risk-neutral probabilities absorb **one-period** (short-run) risk adjustments.

They are widely used in financial engineering but are generally *not* equal to
investors' beliefs.

When short-term interest rates vary across states, risk-neutral probabilities are also
horizon-dependent: the $t$-period forward measure differs from $\bar{\mathbf{P}}^t$.
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

The next theorem is the mathematical reason this construction is well defined.

It is not yet a theorem about recovering investors' true beliefs.

Instead, it proves that a positive pricing operator has one distinguished positive
eigenvalue-eigenvector pair.

The proof idea, stated informally, is that a positive matrix maps the positive cone back
into itself.

Repeatedly applying the matrix and renormalizing pushes all positive vectors toward the
same ray; the expansion rate along that ray is the Perron root.

In this lecture we use that ray to define the state-dependent component
$\hat{\mathbf e}$ and use the expansion rate to define the long-run discount rate
$\hat{\eta}$.

```{prf:theorem} Perron--Frobenius
:label: thm-pf-mis

If $A$ is a matrix with strictly positive entries, then

1. $A$ has a unique largest positive real eigenvalue $r$ (the Perron root).
2. There exists a strictly positive eigenvector $e \gg 0$ with $Ae = re$, unique up to scaling.
```

By {prf:ref}`thm-pf-mis`, the eigenvalue problem for $\mathbf{Q}$ has a unique solution.

What has been proved at this stage is uniqueness of the long-term risk-neutral
construction, not equality between $\hat{\mathbf P}$ and the physical transition matrix
$\mathbf P$.

This gives a unique construction:

1. Solve $\mathbf{Q} \hat{\mathbf{e}} = \exp(\hat{\eta}) \hat{\mathbf{e}}$ for the
   dominant eigenvalue–eigenvector pair.
2. Set $\hat{p}_{ij} = \exp(-\hat{\eta}) \, q_{ij} \, \hat{e}_j / \hat{e}_i$.

{cite:t}`BorovickaHansenScheinkman2016` call the resulting $\hat{\mathbf{P}}$ the
**long-term risk-neutral measure** because, under $\hat{\mathbf{P}}$, the long-horizon
risk premia on stochastically growing cash flows are identically zero.

### Python implementation

```{code-cell} ipython3
def perron_frobenius(Q):
    """Return the Perron root, eigenvector, and long-term risk-neutral matrix."""
    eigenvalues, eigenvectors = linalg.eig(Q)

    # Use the positive Perron eigenpair and discard numerical complex roots.
    real_mask = np.isreal(eigenvalues)
    real_eigenvalues = eigenvalues[real_mask].real
    real_eigenvectors = eigenvectors[:, real_mask].real

    idx = np.argmax(real_eigenvalues)
    exp_η = real_eigenvalues[idx]
    e_hat = real_eigenvectors[:, idx]

    if e_hat.sum() < 0:
        e_hat = -e_hat
    if np.any(e_hat <= 0):
        raise ValueError("Dominant eigenvector is not strictly positive.")
    e_hat = e_hat / e_hat.sum()

    η_hat = np.log(exp_η)

    # Change measure using the Perron eigenfunction.
    P_hat = (1.0 / exp_η) * Q * e_hat[np.newaxis, :] / e_hat[:, np.newaxis]

    return η_hat, exp_η, e_hat, P_hat


η_hat, exp_η, e_hat, P_hat = perron_frobenius(Q_mat)

print(f"exp(η_hat) = {exp_η:.6f}")
print(f"η_hat      = {η_hat:.5f}  (annualized ~ {η_hat*12:.4f})")
print(f"e_hat      = {e_hat.round(5)}")
print("\nLong-term risk-neutral P_hat:")
print(np.round(P_hat, 4))
print("Row sums:", P_hat.sum(axis=1))
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
def stationary_dist(P):
    """Stationary distribution of an ergodic transition matrix."""
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

print("Stationary distributions")
for lbl, π in zip(labels, [π_phys, π_bar, π_hat]):
    print(f"  {lbl:45s}: {np.round(π,4)}")
```

In this first trend-stationary power-utility example, the long-term risk-neutral measure
$\hat{\mathbf{P}}$ coincides with the physical measure $\mathbf{P}$.

This is the special success case in {cite}`BorovickaHansenScheinkman2016`: the SDF has
only the Perron--Frobenius trend component and no martingale component.

The one-period risk-neutral measure $\bar{\mathbf P}$, by contrast, still absorbs
short-run risk adjustments and therefore differs from $\mathbf P$.

## The martingale decomposition

### Decomposing the SDF process

The decomposition in this section answers a diagnostic question: after we remove the
long-run discount rate and the state-dependent Perron--Frobenius trend from the SDF, is
anything left?

If the answer is yes, the leftover term is a martingale that changes probabilities
between $\mathbf P$ and $\hat{\mathbf P}$.

The proof is obtained by writing the one-period pricing identity in Perron--Frobenius
form and multiplying those one-period identities over time.

Let $\hat{\mathbf{e}}$ and $\hat{\eta}$ solve the Perron–Frobenius problem.

Define the process

$$
\frac{\hat{H}_{t+1}}{\hat{H}_t} = (X_t)' \hat{\mathbf{H}} X_{t+1},
\quad \text{where} \quad
\hat{h}_{ij} = \frac{\hat{p}_{ij}}{p_{ij}}.
$$

Because $\sum_j \hat{h}_{ij} p_{ij} = \sum_j \hat{p}_{ij} = 1$, the process $\hat{H}$ is
a martingale under the physical measure $\mathbf{P}$.

The accumulated SDF then admits the **multiplicative decomposition**:

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
# Physical SDF implied by Arrow prices and physical probabilities.
S_mat = np.where(P_phys > 0, Q_mat / P_phys, 0.0)

# Perron-Frobenius trend component of the SDF.
S_hat = exp_η * e_hat[:, np.newaxis] / e_hat[np.newaxis, :]

# Martingale likelihood-ratio increment between P_hat and P.
H_incr = np.where(P_phys > 0, P_hat / P_phys, 0.0)

print("SDF matrix S = Q/P:")
print(np.round(S_mat, 4))
print("\nTrend SDF S_hat = exp(η_hat) * e_hat_i / e_hat_j:")
print(np.round(S_hat, 4))
print("\nMartingale increment h_hat = P_hat/P:")
print(np.round(H_incr, 4))

mart_check = (H_incr * P_phys).sum(axis=1)
print(f"\nE[h_hat | X_t=i] = {mart_check}")
```

Here $\hat h_{ij}=1$ for every transition, so there is no recovery distortion.

The pessimistic distortion appears below once recursive utility introduces a nontrivial
continuation-value martingale.

## When does recovery succeed?

### The Ross recovery condition

{cite:t}`Ross2015` proposes to identify investors' subjective beliefs by imposing

$$
\widetilde{S}_t = \exp(-\delta t) \frac{m(X_t)}{m(X_0)}
$$

for some positive function $m$ and discount rate $\delta$ (Condition 4 in
{cite}`BorovickaHansenScheinkman2016`).

Under this restriction, the SDF has **no martingale component**: $\hat{H}_t \equiv 1$.

The proposition below states the exact object being tested.

It asks whether the Perron--Frobenius transition matrix $\hat{\mathbf P}$ is the same as
the physical transition matrix $\mathbf P$.

The proof is just an accounting exercise: divide the recovered probabilities by the
physical probabilities and see whether the resulting likelihood-ratio increment is
identically one.

```{prf:proposition} Ross Recovery Condition
:label: prop-ross-recovery-condition

({cite}`BorovickaHansenScheinkman2016`) Recovery succeeds -- i.e.,
$\hat{\mathbf{P}} = \mathbf{P}$ -- if and only if the physical stochastic discount
factor takes the long-term risk pricing form

$$
s_{ij} = \exp(\hat{\eta}) \frac{\hat{e}_i}{\hat{e}_j}
$$

with $\hat{h}_{ij} \equiv 1$, so that the SDF has no martingale component.
```

```{prf:proof}
Using $q_{ij}=s_{ij}p_{ij}$ and the Perron--Frobenius construction,

$$
\hat{p}_{ij}
= \exp(-\hat{\eta})q_{ij}\frac{\hat{e}_j}{\hat{e}_i}
= \exp(-\hat{\eta})s_{ij}p_{ij}\frac{\hat{e}_j}{\hat{e}_i}.
$$

Hence the likelihood-ratio increment between the recovered and physical measures is

$$
\hat{h}_{ij}
= \frac{\hat{p}_{ij}}{p_{ij}}
= \exp(-\hat{\eta})s_{ij}\frac{\hat{e}_j}{\hat{e}_i}.
$$

Thus $\hat{\mathbf P}=\mathbf P$ if and only if $\hat{h}_{ij}=1$ for every feasible
transition $(i,j)$, which is equivalent to

$$
s_{ij} = \exp(\hat{\eta})\frac{\hat{e}_i}{\hat{e}_j}.
$$

This is precisely the case in which the martingale term in the multiplicative
decomposition is degenerate.
```

The critical question is: when is the martingale component degenerate?

### Power utility with trend-stationary consumption

Consider a power-utility investor with risk aversion $\gamma$ and *trend-stationary*
consumption $C_t = \exp(g_c t)(c \cdot X_t)$ where $c$ is a positive vector.

The one-period SDF is

$$
s_{ij} = \exp(-\delta - \gamma g_c) \left(\frac{c_j}{c_i}\right)^{-\gamma}.
$$

The corollary shows one important case where the recovery condition is satisfied.

What is being proved is that trend-stationary consumption risk can be absorbed entirely
into the state-dependent ratio $\hat e_i/\hat e_j$.

The proof works by guessing the Perron--Frobenius eigenvector from marginal utility,
then checking that the recovered transition probabilities reduce to the original
physical probabilities.

```{prf:corollary} Recovery under Power Utility
:label: cor-recovery-power-utility

For a power-utility investor with trend-stationary consumption, the SDF takes the exact
long-term risk pricing form with $\hat{e}_j = c_j^\gamma$ and
$\hat{\eta} = -(\delta + \gamma g_c)$.

Therefore $\hat{h}_{ij} \equiv 1$ and Ross recovery succeeds exactly when consumption
fluctuations around a deterministic trend are the only source of risk.
```

```{prf:proof}
Let

$$
A = \exp(-\delta-\gamma g_c)
$$

so that

$$
q_{ij} = A\left(\frac{c_j}{c_i}\right)^{-\gamma}p_{ij}.
$$

Guess $\hat e_i=c_i^\gamma$.

Then

$$
[\mathbf Q\hat{\mathbf e}]_i
= \sum_j A\left(\frac{c_j}{c_i}\right)^{-\gamma}p_{ij}c_j^\gamma
= A c_i^\gamma \sum_j p_{ij}
= A\hat e_i.
$$

Thus $\exp(\hat\eta)=A$ and $\hat{\mathbf e}$ is the Perron--Frobenius eigenvector.

Substituting into the recovered transition probabilities gives

$$
\hat p_{ij}
= \frac{1}{A}q_{ij}\frac{\hat e_j}{\hat e_i}
= \frac{1}{A}
   A\left(\frac{c_j}{c_i}\right)^{-\gamma}p_{ij}
   \frac{c_j^\gamma}{c_i^\gamma}
= p_{ij}.
$$

Hence $\hat h_{ij}=\hat p_{ij}/p_{ij}=1$ for all feasible transitions.
```

```{code-cell} ipython3
gc = 0.002   # monthly trend growth

S_trend = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        S_trend[i, j] = np.exp(-δ - γ*gc) * (c_levels[j]/c_levels[i])**(-γ)

Q_trend = S_trend * P_phys

_, exp_η_t, e_hat_t, P_hat_t = perron_frobenius(Q_trend)

H_incr_trend = np.where(P_phys > 0, P_hat_t / P_phys, 0.0)

print("Trend-stationary h_hat:")
print(np.round(H_incr_trend, 6))
print(f"Max deviation from 1: {np.abs(H_incr_trend[P_phys>0] - 1).max():.2e}")
```

### Recursive (Epstein–Zin) utility

The previous corollary is a success case for recovery.

The next calculation is a failure case: it shows exactly where the power-utility proof
breaks once continuation values enter the SDF.

The key step is to identify an extra term that cannot, in general, be written only as a
ratio of the current and next states.

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

The additional factor $v^*_j/(\mathbf{P}_i v^*)$ introduces a **nontrivial martingale
component** whenever $v^*$ is not constant across states.

```{code-cell} ipython3
def solve_ez_finite(P, c, δ, γ, gc, tol=1e-12, max_iter=5000):
    """Solve finite-state Epstein-Zin continuation values and SDF."""
    β = np.exp(-δ)
    log_c = np.log(c)
    n = len(c)
    flow = (1 - β) * log_c + β * gc

    if abs(γ - 1.0) < 1e-10:
        # Log utility avoids the (1-gamma) denominator in the recursion.
        v = linalg.solve(np.eye(n) - β * P, flow)
        vstar = np.ones(n)
        Pv = np.ones(n)
    else:
        # Fixed-point iteration for the transformed continuation value term.
        v = log_c.copy()
        for _ in range(max_iter):
            vstar = np.exp((1 - γ) * v)
            Pv = P @ vstar
            v_new = flow + β / (1 - γ) * np.log(Pv)
            if np.max(np.abs(v_new - v)) < tol:
                v = v_new
                break
            v = v_new
        vstar = np.exp((1 - γ) * v)
        Pv = P @ vstar

    # The SDF includes the continuation-value likelihood-ratio term.
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s[i, j] = np.exp(-δ - gc) * (c[i] / c[j]) * (vstar[j] / Pv[i])

    return v, vstar, s


gc_ex = 0.001   # monthly consumption trend growth

for γ_val, label in [(1.0, 'γ = 1  (log utility)'), (5.0, 'γ = 5  (risk aversion)')]:
    v_ez, vstar_ez, S_ez = solve_ez_finite(P_phys, c_levels,
                                            δ, γ_val, gc_ex)
    Q_ez = S_ez * P_phys
    _, _, _, P_hat_ez = perron_frobenius(Q_ez)
    H_ez = np.where(P_phys > 0, P_hat_ez / P_phys, 0.0)

    π_hat_ez = stationary_dist(P_hat_ez)
    print(f"\n{label}")
    print(f"  Max |h_hat_ij - 1|        = {np.abs(H_ez[P_phys>0] - 1).max():.4f}")
    print(f"  Stationary P_hat         = {π_hat_ez.round(4)}")
    print(f"  Stationary P             = {π_phys.round(4)}")
```

```{code-cell} ipython3
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

At $\gamma = 1$ (log utility), $v^*=\exp((1-\gamma)v)$ is constant across states, so the
continuation-value martingale is trivial and recovery succeeds.

For $\gamma > 1$, the transformed continuation value $v^*$ varies with the state,
generating a non-degenerate martingale that grows with risk aversion.

## The long-run risk model

We now illustrate the results quantitatively using the Bansal–Yaron
{cite}`Bansal_Yaron_2004` long-run risk model, calibrated to
{cite}`BorovickaHansenScheinkman2016` (Figure 1).

### Model setup

The state vector $X_t = (X_{1t}, X_{2t})'$ follows the continuous-time diffusion

$$
\begin{aligned}
dX_{1t} &= \bar{\mu}_{11}(X_{1t} - \iota_1)\,dt + \sqrt{X_{2t}}\,\bar{\sigma}_1 dW_t \\
dX_{2t} &= \bar{\mu}_{22}(X_{2t} - \iota_2)\,dt + \sqrt{X_{2t}}\,\bar{\sigma}_2 dW_t,
\end{aligned}
$$

where $W_t$ is a three-dimensional Brownian motion.

Here $X_{1t}$ is the **predictable component of consumption growth** and $X_{2t}$ is
**stochastic volatility**.

The representative agent has Epstein–Zin preferences with unit elasticity of
substitution.

The stochastic discount factor satisfies

$$
d\log S_t = -\delta\,dt - d\log C_t + d\log H^*_t,
$$

where $H^*$ is a martingale determined by the continuation value of the recursive
utility.

```{code-cell} ipython3
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

The log continuation value $v(X_t)$ is affine in the state:
$v(x) = \bar{v}_0 + \bar{v}_1 x_1 + \bar{v}_2 x_2$.

The coefficients satisfy the algebraic system in Appendix D of
{cite}`BorovickaHansenScheinkman2016`.

```{code-cell} ipython3
def solve_value_function(p):
    """Solve the affine Epstein-Zin value-function coefficients."""
    δ, γ = p['δ'], p['γ']
    μ11, μ12, μ22 = p['μ11'], p['μ12'], p['μ22']
    σ1, σ2 = p['σ1'], p['σ2']
    β_c1, β_c2 = p['β_c1'], p['β_c2']
    α_c = p['α_c']

    # The X1 coefficient solves a scalar linear equation.
    v1 = β_c1 / (δ - μ11)

    # The X2 coefficient solves the quadratic equation from the affine recursion.
    A_vec = α_c + σ1 * v1
    B_vec = σ2

    a = 0.5 * (1 - γ) * np.dot(B_vec, B_vec)
    b = (μ22 - δ) + (1 - γ) * np.dot(A_vec, B_vec)
    c = β_c2 + μ12 * v1 + 0.5 * (1 - γ) * np.dot(A_vec, A_vec)

    disc = b**2 - 4*a*c
    if disc < 0:
        raise ValueError("Value function does not exist for these parameters.")

    v2 = (-b - np.sqrt(disc)) / (2 * a)
    return v1, v2, A_vec, B_vec


v1, v2, A_vec, B_vec = solve_value_function(lrr_params)
print(f"Value-function slope on X1:  v_bar1 = {v1:.4f}")
print(f"Value-function slope on X2:  v_bar2 = {v2:.4f}")
```

### Perron–Frobenius and recovered dynamics

```{code-cell} ipython3
def solve_pf_lrr(p, v1, v2, A_vec):
    """Solve the LRR Perron-Frobenius coefficients."""
    δ, γ = p['δ'], p['γ']
    μ11, μ12, μ22 = p['μ11'], p['μ12'], p['μ22']
    ι1, ι2 = p['ι1'], p['ι2']
    σ1, σ2 = p['σ1'], p['σ2']
    α_c = p['α_c']
    β_c0 = p['β_c0']
    β_c1, β_c2 = p['β_c1'], p['β_c2']

    # H* is the continuation-value martingale in the recursive utility SDF.
    α_h_star = (1 - γ) * (α_c + σ1 * v1 + σ2 * v2)

    # α_s is the diffusion loading of d log S_t.
    α_s = -α_c + α_h_star

    # The Ito correction uses d log H*, not the total log-SDF diffusion.
    β_s11 = -β_c1
    β_s12 = -β_c2 - 0.5 * np.dot(α_h_star, α_h_star)
    β_s0 = (-δ - β_c0
            - 0.5 * ι2 * np.dot(α_h_star, α_h_star))

    # The first Perron coefficient solves a scalar linear equation.
    e1 = -β_s11 / μ11

    # The second Perron coefficient solves a quadratic equation.
    const_pf = (β_s12 + 0.5*np.dot(α_s, α_s)
                + e1*(μ12 + np.dot(σ1, α_s))
                + 0.5*e1**2*np.dot(σ1, σ1))
    lin_pf = μ22 + np.dot(σ2, α_s) + e1*np.dot(σ1, σ2)
    quad_pf = 0.5 * np.dot(σ2, σ2)

    disc = lin_pf**2 - 4*quad_pf*const_pf
    e2_m = (-lin_pf - np.sqrt(disc)) / (2*quad_pf)
    e2_p = (-lin_pf + np.sqrt(disc)) / (2*quad_pf)

    η_m = β_s0 - β_s12*ι2 - e2_m*μ22*ι2
    η_p = β_s0 - β_s12*ι2 - e2_p*μ22*ι2

    # Select the lower eigenvalue root that generates stationary recovered dynamics.
    if η_m <= η_p:
        e2, η_hat = e2_m, η_m
    else:
        e2, η_hat = e2_p, η_p

    return e1, e2, η_hat, α_s


e1, e2, η_hat_lrr, α_s = solve_pf_lrr(lrr_params, v1, v2, A_vec)

print(f"PF eigenfunction coefficients:  e_bar1 = {e1:.4f},  e_bar2 = {e2:.4f}")
print(f"Log eigenvalue:                 η_hat  = {η_hat_lrr:.6f}  "
      f"(annualized = {η_hat_lrr*12:.4f})")
```

### Computing the P_hat dynamics

```{code-cell} ipython3
def compute_phat_dynamics(p, e1, e2, α_s):
    """Drift parameters under the recovered measure P_hat."""
    μ11, μ12, μ22 = p['μ11'], p['μ12'], p['μ22']
    ι1, ι2 = p['ι1'], p['ι2']
    σ1, σ2 = p['σ1'], p['σ2']

    # α_h is the likelihood-ratio loading for the recovered measure.
    α_h = α_s + σ1 * e1 + σ2 * e2

    μ_hat_11 = μ11
    μ_hat_12 = μ12 + np.dot(σ1, α_h)
    μ_hat_22 = μ22 + np.dot(σ2, α_h)

    # Rewrite the drift in mean-reversion form under P_hat.
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

print("P_hat dynamics:")
print(f"  μ_hat_11 = {phat_dyn['μ_hat_11']:.4f}  "
      f"(physical {lrr_params['μ11']:.4f})")
print(f"  μ_hat_12 = {phat_dyn['μ_hat_12']:.6f}  "
      f"(physical 0)")
print(f"  μ_hat_22 = {phat_dyn['μ_hat_22']:.5f}  "
      f"(physical {lrr_params['μ22']:.4f})")
print(f"  ι_hat_1  = {phat_dyn['ι_hat_1']:.5f}  "
      f"(physical {lrr_params['ι1']:.4f})")
print(f"  ι_hat_2  = {phat_dyn['ι_hat_2']:.5f}  "
      f"(physical {lrr_params['ι2']:.4f})")
```

For comparison with the paper's Figure 1, we also compute the instantaneous risk-neutral
dynamics.

This change of measure uses the martingale component of the normalized SDF, whose
diffusion vector is $\alpha_s$.

```{code-cell} ipython3
def compute_rn_dynamics(p, α_s):
    """Drift parameters under the one-period risk-neutral measure."""
    μ11, μ12, μ22 = p['μ11'], p['μ12'], p['μ22']
    ι1, ι2 = p['ι1'], p['ι2']
    σ1, σ2 = p['σ1'], p['σ2']

    # Risk-neutral dynamics use the normalized SDF loading.
    μ_rn_11 = μ11
    μ_rn_12 = μ12 + np.dot(σ1, α_s)
    μ_rn_22 = μ22 + np.dot(σ2, α_s)

    # Rewrite the drift in mean-reversion form under P_bar.
    ι_rn_2 = (μ22 / μ_rn_22) * ι2
    ι_rn_1 = (ι1
              + (1.0/μ11) * (μ12*ι2 - μ_rn_12*ι_rn_2))

    return dict(
        μ_rn_11 = μ_rn_11,
        μ_rn_12 = μ_rn_12,
        μ_rn_22 = μ_rn_22,
        ι_rn_1 = ι_rn_1,
        ι_rn_2 = ι_rn_2,
    )


rn_dyn = compute_rn_dynamics(lrr_params, α_s)

print("Dynamics of X under P_bar (risk-neutral):")
print(f"  μ_rn_12 = {rn_dyn['μ_rn_12']:.6f}")
print(f"  μ_rn_22 = {rn_dyn['μ_rn_22']:.5f}")
print(f"  ι_rn_1  = {rn_dyn['ι_rn_1']:.5f}")
print(f"  ι_rn_2  = {rn_dyn['ι_rn_2']:.5f}")
```

### Simulating and comparing stationary distributions

```{code-cell} ipython3
def simulate_lrr(dyn, T=600_000, seed=42):
    """Simulate stationary LRR paths by Euler-Maruyama."""
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

        # Monthly Euler step with dt = 1.
        dW = rng.standard_normal(3)

        X1[t] = X1[t-1] + (μ11*(X1[t-1]-ι1) + μ12*(X2t-ι2)) + sq_X2*np.dot(σ1, dW)
        X2[t] = max(X2[t-1] + μ22*(X2t-ι2) + sq_X2*np.dot(σ2, dW),  1e-9)

    burn = T // 5
    return X1[burn:], X2[burn:]


X1_P, X2_P = simulate_lrr(
    dict(μ11=lrr_params['μ11'], μ12=lrr_params['μ12'],
         μ22=lrr_params['μ22'], ι1=lrr_params['ι1'],
         ι2=lrr_params['ι2'],
         σ1=lrr_params['σ1'], σ2=lrr_params['σ2']),
    T=600_000
)

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

X1_RN, X2_RN = simulate_lrr(
    dict(μ11=rn_dyn['μ_rn_11'],
         μ12=rn_dyn['μ_rn_12'],
         μ22=rn_dyn['μ_rn_22'],
         ι1=rn_dyn['ι_rn_1'],
         ι2=rn_dyn['ι_rn_2'],
         σ1=lrr_params['σ1'],
         σ2=lrr_params['σ2']),
    T=600_000
)
```

```{code-cell} ipython3
def kde2d_contour(ax, X1, X2, levels=8, color='k', alpha=1.0, lw=1.5,
                  bandwidth=None, linestyle='solid'):
    """Add 2D KDE contours to an axis."""
    xy = np.vstack([X2, X1])
    kde = gaussian_kde(xy, bw_method=bandwidth)
    x2g = np.linspace(X2.min()*0.9, X2.max()*1.1, 120)
    x1g = np.linspace(X1.min()*0.9, X1.max()*1.1, 120)
    X2g, X1g = np.meshgrid(x2g, x1g)
    Z = kde(np.vstack([X2g.ravel(), X1g.ravel()])).reshape(X2g.shape)
    ax.contour(X2g, X1g, Z, levels=levels, colors=color, alpha=alpha,
               linewidths=lw, linestyles=linestyle)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

kde2d_contour(ax1, X1_P, X2_P, color='navy', levels=7)
ax1.set_xlabel('conditional volatility  $X_2$', fontsize=11)
ax1.set_ylabel('mean growth rate  $X_1$', fontsize=11)
ax1.set_title(r'physical measure  $P$', fontsize=12)

kde2d_contour(ax2, X1_Ph, X2_Ph, color='navy', levels=7)
kde2d_contour(ax2, X1_RN, X2_RN, color='black', levels=3,
              alpha=0.65, lw=1.2, linestyle='--')
ax2.set_xlabel('conditional volatility  $X_2$', fontsize=11)
ax2.set_title(r'long-term risk-neutral  $\hat{P}$', fontsize=12)

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
ax2.plot([], [], color='navy', lw=1.5, label=r'$\hat{P}$')
ax2.plot([], [], color='black', lw=1.2, ls='--', label=r'risk-neutral $\bar{P}$')
ax2.legend(fontsize=9, loc='lower right')

plt.suptitle('stationary distributions of $(X_1, X_2)$ under $P$ and $\\hat{P}$\n'
             '(based on Figure 1 of Borovička, Hansen & Scheinkman 2016)',
             fontsize=12, y=1.02)
plt.tight_layout();  plt.show()
```

The recovered measure $\hat{P}$ concentrates around **lower mean growth** (more negative
$X_1$) and **higher conditional volatility** (larger $X_2$).

Forecasts made using $\hat{P}$ are systematically pessimistic compared to forecasts
based on the true distribution $P$.

## Measuring the martingale component

### Entropy bounds

Even without observing the full array of Arrow prices, we can obtain **lower bounds** on
the size of the martingale component.

For a convex function $\phi_\theta(r) = [(r)^{1+\theta} - 1] / [\theta(1+\theta)]$, the
discrepancy between $\hat{P}$ and $P$ satisfies

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
    """Discrepancy function."""
    if abs(θ) < 1e-10:      # θ -> 0: relative entropy r log r
        return r * np.log(r)
    if abs(θ + 1) < 1e-10:  # θ -> -1: -log r
        return -np.log(r)
    return (r**(1 + θ) - 1) / (θ * (1 + θ))


def martingale_entropy(Q, P, θ=-1):
    """Stationary-average discrepancy E[φ_θ(h_hat)]."""
    _, exp_η, e_hat, P_hat = perron_frobenius(Q)
    H_incr = np.where(P > 0, P_hat / P, 1.0)
    π = stationary_dist(P)

    disc = 0.0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] > 0:
                disc += π[i] * P[i, j] * φ_θ(H_incr[i, j], θ)
    return disc


γs_ent = np.linspace(1.0, 10.0, 50)
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
$\gamma$ implies a larger -- and more economically significant -- martingale component.

{cite:t}`AlvarezJermann2005` and {cite:t}`BakshiChabiYo2012` use analogous bounds with
long-maturity bond returns to find empirically large martingale components in U.S. data.

## Exercises

```{exercise}
:label: ex_risk_neutral

**Verify risk-neutral probabilities.**

Consider a two-state Markov chain with physical transition matrix

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
P2 = np.array([[0.8, 0.2],
               [0.4, 0.6]])
Q2 = np.array([[0.72, 0.15],
               [0.36, 0.42]])

P_bar2, q_bonds2 = risk_neutral_probs(Q2)

print("Risk-neutral P_bar:")
print(np.round(P_bar2, 4))
print(f"\nRow sums: {P_bar2.sum(axis=1)}")
print(f"\nBond prices q_bar_i: {q_bonds2}")
print(f"Annualized risk-free rates: {(-np.log(q_bonds2)*12).round(4)}")

S_bar2 = np.repeat(q_bonds2[:, np.newaxis], P_bar2.shape[1], axis=1)
print(f"\nRisk-neutral SDF matrix S_bar:")
print(np.round(S_bar2, 4))
print("Check Q = S_bar * P_bar:", np.allclose(Q2, S_bar2 * P_bar2))

S2 = Q2 / P2
print(f"\nPhysical SDF matrix S = Q/P:")
print(np.round(S2, 4))
```

```{solution-end}
```

```{exercise}
:label: ex_gamma_sensitivity

**Risk aversion and recovery distortion under recursive utility.**

Using the three-state Epstein--Zin example from the lecture (with $\exp(-\delta)=0.99$,
$g_c=0.001$, and consumption levels $c = [0.85, 1.00, 1.15]$), investigate how the
recovered probability vector $\hat{\boldsymbol{\pi}}$ depends on the risk aversion
parameter $\gamma$.

1. For each $\gamma \in \{1, 2, 5, 10, 15\}$, compute the long-term risk-neutral
   stationary distribution $\hat{\boldsymbol{\pi}}$ using the recursive-utility SDF.
2. Plot all five distributions as grouped bar charts alongside the physical
   distribution $\boldsymbol{\pi}$.
3. Does the recession probability under $\hat{\mathbf{P}}$ exceed $50\%$ for
   $\gamma \leq 30$?

If not, report the maximum value on that range.
```

```{solution-start} ex_gamma_sensitivity
:class: dropdown
```

```{code-cell} ipython3
γs_ex2 = [1, 2, 5, 10, 15]
all_π = []

for γ_val in γs_ex2:
    _, _, S_g = solve_ez_finite(P_phys, c_levels, δ, γ_val, gc_ex)
    Q_g = S_g * P_phys
    _, _, _, Ph_g = perron_frobenius(Q_g)
    all_π.append(stationary_dist(Ph_g))

fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(3)
w = 0.13
colors_g = plt.cm.Blues(np.linspace(0.3, 0.9, len(γs_ex2)))

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

γs_fine = np.linspace(1, 30, 200)
rec_probs = []
for γ_val in γs_fine:
    _, _, S_g = solve_ez_finite(P_phys, c_levels, δ, γ_val, gc_ex)
    Q_g = S_g * P_phys
    _, _, _, Ph_g = perron_frobenius(Q_g)
    rec_probs.append(stationary_dist(Ph_g)[0])

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

**Effect of risk aversion in the long-run risk model.**

Repeat the long-run risk simulation from the lecture for $\gamma \in \{5, 10, 15\}$
(keeping all other parameters fixed at their calibrated values).

1. For each $\gamma$, compute $(\bar{e}_1, \bar{e}_2)$ and $\hat{\eta}$.
2. Plot $\hat{\iota}_1$ (long-run mean of $X_1$ under $\hat{P}$) as a function of
   $\gamma$ and interpret the result in terms of long-run expected consumption growth.
3. Plot $\hat{\iota}_2$ (long-run mean of $X_2$ under $\hat{P}$) as a function of
   $\gamma$ and interpret it in terms of long-run volatility.
```

```{solution-start} ex_lrr_gamma
:class: dropdown
```

```{code-cell} ipython3
γs_lrr = np.linspace(2.0, 18.0, 40)
ι_hat_1_vals = []
ι_hat_2_vals = []
η_hat_vals = []

p_copy = dict(lrr_params)

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

```

```{solution-end}
```

```{exercise}
:label: ex_recovery_test

**Testing the Ross recovery condition.**

Show algebraically and numerically that, for any $n$-state power-utility model with
trend-stationary consumption (as in Example 1 of
{cite}`BorovickaHansenScheinkman2016`), the martingale increment satisfies
$\hat{h}_{ij} \equiv 1$.

1. Write the SDF as $s_{ij} = A \cdot (c_j/c_i)^{-\gamma}$ for some constant $A$,
   show that the Perron-Frobenius eigenvector is $\hat{e}_j = c_j^\gamma$ (up to
   scale), and find $\hat{\eta}$.
2. Compute $\hat{p}_{ij} = \exp(-\hat{\eta}) q_{ij} \hat{e}_j / \hat{e}_i$ and verify
   it equals $p_{ij}$.
3. Confirm numerically for the three-state example with $\gamma = 5$ and
   $c = [0.85, 1.00, 1.15]$.
```

```{solution-start} ex_recovery_test
:class: dropdown
```

**Analytical derivation:**

With $s_{ij} = A \cdot (c_j/c_i)^{-\gamma}$ we have
$q_{ij} = A(c_j/c_i)^{-\gamma} p_{ij}$.

Guess $\hat{e}_j = c_j^\gamma$.

Then

$$
[\mathbf{Q} \hat{\mathbf{e}}]_i
= \sum_j q_{ij} \hat{e}_j
= A \sum_j \frac{c_j^{-\gamma}}{c_i^{-\gamma}} p_{ij} \cdot c_j^\gamma
= A c_i^\gamma \sum_j p_{ij}
= A \hat e_i.
$$

So $\mathbf{Q}\hat{\mathbf{e}} = A \hat{\mathbf{e}}$, confirming
$\hat{\mathbf{e}} = \{c_j^\gamma\}$ and $\exp(\hat{\eta}) = A$.

Therefore

$$
\hat{p}_{ij}
= \frac{1}{A} q_{ij} \frac{\hat{e}_j}{\hat{e}_i}
= \frac{1}{A} \cdot A \frac{c_j^{-\gamma}}{c_i^{-\gamma}} p_{ij}
  \cdot \frac{c_j^\gamma}{c_i^\gamma}
= p_{ij}.
$$

Hence $\hat{h}_{ij} = \hat{p}_{ij}/p_{ij} = 1$ for all $(i,j)$.

```{code-cell} ipython3
gc_ex4 = 0.002
S_ts = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        S_ts[i, j] = np.exp(-δ - γ*gc_ex4) * (c_levels[j]/c_levels[i])**(-γ)

Q_ts = S_ts * P_phys

_, exp_η_ts, e_hat_ts, P_hat_ts = perron_frobenius(Q_ts)

e_theory = c_levels**γ
e_theory /= e_theory.sum()

print("e_hat:", np.round(e_hat_ts, 6))
print("c^γ normalized:", np.round(e_theory, 6))
print(f"Max discrepancy: {np.abs(e_hat_ts - e_theory).max():.2e}")

H_ts = np.where(P_phys > 0, P_hat_ts / P_phys, 0.0)
print("\nh_hat:")
print(np.round(H_ts, 6))
print(f"Max |h_hat_ij - 1|: {np.abs(H_ts[P_phys>0] - 1).max():.2e}")
```

```{solution-end}
```
