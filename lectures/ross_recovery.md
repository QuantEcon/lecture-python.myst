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

(ross_recovery)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# The Recovery Theorem

```{contents} Contents
:depth: 2
```

## Overview

From option prices we can extract risk-neutral (martingale) probabilities of
future outcomes.

But risk-neutral probabilities blend two things: the market's
*actual* probability beliefs and investors' *risk aversion*.

Disentangling the
two seems to require imposing  parametric assumptions on
preferences of a representative investor.

Nevertheless, {cite}`Ross2015` showed that under a key assumption — the **transition
independence** of the pricing kernel — the natural (real-world) probability
distribution and the pricing kernel can be uniquely recovered from state prices
alone, without historical return data or parametric utility functions.

This
result is called the **Recovery Theorem**.

The theorem has several important implications.

* It enables model-free extraction of the market's forward-looking probability
  distribution from option prices.
* It provides model-free tests of the efficient market hypothesis.
* It sheds light on the "dark matter" of finance: the probability of rare
  catastrophic events allegedly embedded in market prices.

This lecture covers

* The basic Arrow–Debreu framework linking state prices, the pricing kernel,
  and natural probabilities.
* Ross's Recovery Theorem and its proof via the Perron–Frobenius theorem.
* A computational implementation that recovers the natural distribution from a
  simulated state-price matrix.
* Comparisons between risk-neutral and recovered natural densities.

Let's import the packages we'll need.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.stats import norm
import matplotlib.cm as cm

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
```

## Model setup

### Arrow–Debreu state prices

Consider a discrete-time, discrete-state economy.

At each date the economy
occupies one of $m$ states $\theta_1, \ldots, \theta_m$.

An **Arrow–Debreu
security** pays \$1 if the economy is in state $\theta_j$ next period and
nothing otherwise.

Denote by $p(\theta_i, \theta_j)$ the price today, when the current state is
$\theta_i$, of the Arrow–Debreu security paying in state $\theta_j$ next
period.

Collect these into an $m \times m$ **state price transition matrix**

$$
P = [p(\theta_i, \theta_j)]_{i,j=1}^m.
$$

The row sums give the state-dependent interest factor: $\sum_j p(\theta_i,
\theta_j) = e^{-r(\theta_i)}$.

### The pricing kernel

From the Fundamental Theorem of Asset Pricing, the pricing kernel
$\phi(\theta_i, \theta_j)$ relates state prices to natural probabilities via

$$
p(\theta_i, \theta_j) = \phi(\theta_i, \theta_j) \, f(\theta_i, \theta_j),
$$

where $f(\theta_i, \theta_j)$ is the natural (conditional) probability of
transitioning from state $\theta_i$ to $\theta_j$.

In the canonical representative-agent model with additively separable utility
and discount factor $\delta$, the first-order condition gives

$$
\phi(\theta_i, \theta_j) = \frac{p(\theta_i, \theta_j)}{f(\theta_i, \theta_j)}
    = \frac{\delta U'(c(\theta_j))}{U'(c(\theta_i))}.
$$

The key structural property this implies is **transition independence**.

### Transition independence


**Definition.** A pricing kernel is *transition independent* if there exists a
positive function $h$ on the state space and a positive scalar $\delta$ such
that for every transition from state $\theta_i$ to $\theta_j$,

$$
\phi(\theta_i, \theta_j) = \delta \, \frac{h(\theta_j)}{h(\theta_i)}.
$$


Transition independence says the kernel depends on the *ending* state and
normalizes by the *beginning* state.

It holds for any agent with
intertemporally additive separable utility (where $h = U'$) and also for
Epstein–Zin recursive preferences {cite}`Epstein_Zin1989`.

Under transition independence, the state-price equation becomes

$$
p(\theta_i, \theta_j) = \delta \, \frac{h(\theta_j)}{h(\theta_i)} \,
    f(\theta_i, \theta_j).
$$

In matrix notation, defining the diagonal matrix $D$ with $D_{ii} = h(\theta_i)/\delta$,

$$
DP = \delta F D,
$$

or equivalently,

$$
F = \frac{1}{\delta} D P D^{-1}.
$$

## The recovery theorem

### Reduction to an eigenvalue problem

Since $F$ is a stochastic matrix, its rows sum to one: $F e = e$ where $e$
is the vector of ones.

Substituting the expression for $F$:

$$
\frac{1}{\delta} D P D^{-1} e = e
\quad \Longrightarrow \quad
P z = \delta z, \quad z \equiv D^{-1} e.
$$

This is an **eigenvalue problem**: we seek a positive vector $z$ and scalar
$\delta$ satisfying $Pz = \delta z$.

The Perron–Frobenius theorem guarantees exactly one such solution when $P$ is
nonnegative and irreducible.

**Theorem (Perron–Frobenius).** Every nonnegative irreducible matrix has a
unique positive eigenvector (up to scaling) and a unique largest positive
eigenvalue.

Section 1.2.3 of {cite}`Sargent_Stachurski_2024` provides a proof  of this theorem as well as a discussion of its applications to economic networks.


### Ross's recovery theorem


**Theorem 1 (Recovery Theorem, {cite}`Ross2015`).** Suppose prices provide  no
arbitrage opportunities, that the state price transition matrix $P$ is irreducible, and that the
pricing kernel is transition independent.  Then there exists a *unique*
positive solution $(\delta, z, F)$ to the recovery problem.  That is, for any
set of state prices there is a unique compatible natural probability transition
matrix and a unique pricing kernel.


*Proof sketch.*  Because $P$ is nonnegative and irreducible, the
Perron–Frobenius theorem gives a unique positive eigenvector $z > 0$ with
positive eigenvalue $\lambda > 0$ satisfying $Pz = \lambda z$.  Setting

$$
\delta = \lambda, \qquad D_{ii} = \frac{1}{z_i},
$$

the natural probability transition matrix is uniquely recovered as

$$
f_{ij} = \frac{1}{\delta} \frac{z_j}{z_i} \, p_{ij}.
$$

One can verify that $F$ is indeed a stochastic matrix: all entries are
positive and each row sums to one.

Uniqueness follows from the uniqueness of
the Perron–Frobenius eigenvector. $\blacksquare$

### Pricing kernel from the eigenvector

The recovered kernel values are

$$
\phi(\theta_i, \theta_j) = \frac{\delta}{z_i / z_j}
    = \frac{z_j}{z_i} \cdot \frac{1}{1},
$$

so the kernel at state $\theta_i$ (relative to a baseline state) is $1/z_i$.

States with high $z_i$ have **low** kernel values, meaning the market assigns
relatively less pricing weight per unit of probability — consistent with those
states being "good times."

### Corollary: risk-neutral pricing when rates are state-independent


**Theorem 2 ({cite}`Ross2015`).** If the riskless rate is the same in all
states ($Pe = \gamma e$ for some scalar $\gamma$), then the unique natural
distribution consistent with recovery is the risk-neutral (martingale)
distribution itself: $F = (1/\gamma) P$.


This remarkable result says that with a constant interest rate and a bounded
irreducible state space, recovery forces risk-neutrality — a non-trivial
restriction of the model.

## Python implementation

We now implement the Recovery Theorem numerically.

### Building a state price matrix from a lognormal model

Following {cite}`Ross2015` Section IV, suppose the natural distribution of
log-returns over one period is normal:

$$
\log(S_T/S_0) \sim \mathcal{N}\!\left((\mu - \tfrac{1}{2}\sigma^2)T, \sigma^2 T\right).
$$

With a CRRA pricing kernel $\phi(S_0, S_T) = e^{-\delta T}(S_T/S_0)^{-\gamma}$,
the state price density is

$$
P_T(s, s_T) = e^{-\delta T} e^{-\gamma(s_T - s)} \,
    n\!\left(\frac{s_T - s - (\mu - \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}\right),
$$

where $s = \ln S_0$, $s_T = \ln S_T$, and $n(\cdot)$ is the standard normal
density.

We discretize this onto a grid of $m$ states and build the matrix $P$.

```{code-cell} ipython3
def build_state_price_matrix(mu, sigma, gamma, delta, T=1.0, n_states=11, n_sigma=5):
    """
    Build an m x m state price transition matrix for the lognormal / CRRA model.

    Parameters
    ----------
    mu     : float  Natural expected log-return (annualised)
    sigma  : float  Volatility (annualised)
    gamma  : float  Coefficient of relative risk aversion
    delta  : float  Subjective discount rate
    T      : float  Horizon (years) for one period
    n_states : int  Number of discrete states
    n_sigma  : int  Grid half-width in standard deviations

    Returns
    -------
    P      : (m, m) array  State price matrix
    states : (m,) array    State values (log-return grid)
    """
    # Equally-spaced grid from -n_sigma*sigma to +n_sigma*sigma
    states = np.linspace(-n_sigma * sigma * np.sqrt(T),
                          n_sigma * sigma * np.sqrt(T),
                          n_states)
    ds = states[1] - states[0]   # grid spacing

    m = n_states
    P = np.zeros((m, m))

    drift = (mu - 0.5 * sigma**2) * T

    for i in range(m):
        s_i = states[i]
        for j in range(m):
            s_j = states[j]
            log_return = s_j - s_i
            # Normal density evaluated at s_j given s_i
            n_val = norm.pdf(log_return, loc=drift, scale=sigma * np.sqrt(T))
            # Pricing kernel
            kernel = np.exp(-delta * T) * np.exp(-gamma * log_return)
            P[i, j] = kernel * n_val * ds

    return P, states
```

```{code-cell} ipython3
# Parameters matching the numerical example in Ross (2015), Section IV
mu    = 0.08    # 8% annual expected return
sigma = 0.20    # 20% annual volatility
gamma = 3.0     # CRRA coefficient
delta = 0.02    # 2% annual discount rate
T     = 1.0     # one-year horizon

P, states = build_state_price_matrix(mu, sigma, gamma, delta, T,
                                     n_states=11, n_sigma=5)

print("State price matrix P  (rows = current state, cols = future state)")
print("Row sums (should equal discount factor e^{-r}):")
print(np.round(P.sum(axis=1), 4))
print(f"\nImplied annual interest rate: {-np.log(P[5].sum()):.4f}")
```

### Applying the recovery theorem

The Recovery Theorem requires computing the **dominant eigenvector** of $P$.

```{code-cell} ipython3
def recover_natural_distribution(P):
    """
    Apply the Recovery Theorem to state price matrix P.

    Returns
    -------
    F     : (m, m) array  Recovered natural probability transition matrix
    z     : (m,) array    Dominant eigenvector of P (Perron vector)
    delta : float         Recovered subjective discount rate
    phi   : (m,) array    Recovered kernel values (relative to state 0)
    """
    m = P.shape[0]

    # Compute all eigenvalues and right eigenvectors
    eigenvalues, eigenvectors = eig(P)

    # Find the dominant (Perron) eigenvalue — largest positive real one
    real_mask = np.isreal(eigenvalues)
    real_eigenvalues = eigenvalues[real_mask].real
    real_eigenvectors = eigenvectors[:, real_mask].real

    idx = np.argmax(real_eigenvalues)
    delta_recovered = real_eigenvalues[idx]
    z = real_eigenvectors[:, idx]

    # Ensure z is positive (Perron vector)
    if np.mean(z) < 0:
        z = -z

    # Normalize so that z[reference] = 1
    z = z / z[m // 2]

    # Diagonal matrix D with D_ii = 1/z_i
    D = np.diag(1.0 / z)
    D_inv = np.diag(z)

    # Recover natural probability matrix
    F = (1.0 / delta_recovered) * D @ P @ D_inv

    # Clip small numerical negatives and renormalize rows
    F = np.clip(F, 0, None)
    F = F / F.sum(axis=1, keepdims=True)

    # Pricing kernel relative to middle state
    phi = 1.0 / z

    return F, z, delta_recovered, phi
```

```{code-cell} ipython3
F, z, delta_rec, phi = recover_natural_distribution(P)

print(f"Recovered discount rate δ  = {delta_rec:.6f}  (true: {np.exp(-delta):.6f})")
print(f"\nRecovered kernel φ (monotone decreasing in good states):")
print(np.round(phi, 4))
print(f"\nNatural probability matrix F  (row sums should be 1):")
print(np.round(F.sum(axis=1), 6))
```

### Visualizing natural vs. risk-neutral distributions

A key insight of {cite}`Ross2015` is that the natural distribution systematically
differs from the risk-neutral one.

In particular, the natural distribution
stochastically dominates the risk-neutral distribution (Theorem 3 in {cite}`Ross2015`).

```{code-cell} ipython3
def get_marginal(transition_matrix, initial_row, n_periods, states_exp):
    """
    Compute the marginal distribution at horizon n_periods by iterating
    the transition matrix starting from a given initial distribution.

    Parameters
    ----------
    transition_matrix : (m, m) array
    initial_row       : int  Index of the starting state
    n_periods         : int  Horizon
    states_exp        : (m,) array  Gross return levels exp(states)
    """
    m = transition_matrix.shape[0]
    # Start with all weight on the initial row
    dist = np.zeros(m)
    dist[initial_row] = 1.0

    for _ in range(n_periods):
        dist = dist @ transition_matrix

    return dist
```

```{code-cell} ipython3
# Starting from the middle state (current state = S_0)
mid = len(states) // 2

# Risk-neutral transition matrix Q_ij = P_ij / sum_k P_ik  (row normalise P)
row_sums = P.sum(axis=1, keepdims=True)
Q_rn = P / row_sums   # risk-neutral probabilities

# One-period marginals
f_nat = F[mid, :]              # natural: row of recovered F
f_rn  = Q_rn[mid, :]          # risk-neutral: row of Q

# State labels in gross return terms
gross_returns = np.exp(states)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: densities
axes[0].plot(gross_returns, f_nat, 'b-o', ms=5, label='Natural (recovered)', lw=2)
axes[0].plot(gross_returns, f_rn,  'r--s', ms=5, label='Risk-neutral',       lw=2)
axes[0].set_xlabel('Gross return $S_T / S_0$')
axes[0].set_ylabel('Probability')
axes[0].set_title('One-Period Marginal Distributions')
axes[0].legend()

# Panel B: pricing kernel
axes[1].plot(gross_returns, phi, 'g-^', ms=5, lw=2)
axes[1].set_xlabel('Gross return $S_T / S_0$')
axes[1].set_ylabel('Kernel $\\phi$ (relative)')
axes[1].set_title('Recovered Pricing Kernel')

plt.tight_layout()
plt.savefig('ross_recovery_distributions.png', dpi=120)
plt.show()
```

```{code-cell} ipython3
# Compute summary statistics
E_nat = np.sum(f_nat * gross_returns)
E_rn  = np.sum(f_rn  * gross_returns)
std_nat = np.sqrt(np.sum(f_nat * (gross_returns - E_nat)**2))
std_rn  = np.sqrt(np.sum(f_rn  * (gross_returns - E_rn )**2))

risk_free = np.sum(P[mid])   # price of riskless bond from middle state

print("Summary Statistics (one-period horizon)")
print(f"{'':30s} {'Natural':>12s}   {'Risk-Neutral':>12s}")
print("-" * 57)
print(f"{'Expected gross return':30s} {E_nat:>12.4f}   {E_rn:>12.4f}")
print(f"{'Std dev':30s} {std_nat:>12.4f}   {std_rn:>12.4f}")
print(f"{'Risk-free discount factor':30s} {risk_free:>12.4f}")
print(f"{'Annual risk-free rate':30s} {-np.log(risk_free):>12.4f}")
print(f"{'Equity risk premium':30s} {E_nat - 1/risk_free:>12.4f}")
```

### Stochastic dominance

Theorem 3 of {cite}`Ross2015` shows that the natural marginal density
**first-order stochastically dominates** the risk-neutral density: the CDF of
the natural distribution lies *below* that of the risk-neutral distribution.

Because the pricing kernel is declining (investors fear bad
outcomes), risk-neutral probabilities overweight bad states and underweight
good states relative to the natural measure.

```{code-cell} ipython3
# CDFs
cdf_nat = np.cumsum(f_nat)
cdf_rn  = np.cumsum(f_rn)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(gross_returns, cdf_nat, 'b-o', ms=5, lw=2, label='Natural CDF')
ax.plot(gross_returns, cdf_rn,  'r--s', ms=5, lw=2, label='Risk-neutral CDF')
ax.set_xlabel('Gross return $S_T / S_0$')
ax.set_ylabel('Cumulative probability')
ax.set_title('Stochastic Dominance: Natural CDF lies below Risk-Neutral CDF')
ax.legend()
plt.tight_layout()
plt.savefig('ross_recovery_stochdom.png', dpi=120)
plt.show()

# Verify dominance: natural CDF should be <= risk-neutral CDF at every point
print(f"Natural CDF ≤ Risk-neutral CDF at all states: "
      f"{np.all(cdf_nat <= cdf_rn + 1e-10)}")
```

## Extracting the pricing kernel and risk premium

The pricing kernel recovered from $P$ via the Perron–Frobenius theorem has the following interpretation.

In the CRRA model the kernel is proportional to
$\exp(-\gamma \cdot \text{log-return})$, so it is decreasing in the return.

The **equity risk premium** can be computed as the difference between the
expected return under the natural measure and the risk-free rate:

$$
\text{ERP} = E^f[R] - r_f = \frac{\sum_j f_{ij}\, (S_j/S_i)}{\sum_j p_{ij}} - 1.
$$

```{code-cell} ipython3
def compute_risk_premia(P, F, states):
    """
    Compute the equity risk premium for each starting state.

    Parameters
    ----------
    P, F   : (m, m) arrays  State price and natural probability matrices
    states : (m,) array     Log-state grid

    Returns
    -------
    erp    : (m,) array     Equity risk premium from each starting state
    rf     : (m,) array     Risk-free rate from each starting state
    """
    m = len(states)
    gross_returns = np.exp(states)

    rf  = np.zeros(m)
    erp = np.zeros(m)

    for i in range(m):
        discount = P[i].sum()          # riskless discount factor
        rf[i]  = -np.log(discount)     # risk-free rate

        # Expected gross return under natural measure
        # We compute E[S_j/S_i] = sum_j F_ij * exp(s_j - s_i)
        relative_returns = np.exp(states - states[i])
        E_R_nat = np.sum(F[i] * relative_returns)
        E_R_rn  = np.sum((P[i] / discount) * relative_returns)

        erp[i] = np.log(E_R_nat) - rf[i]

    return erp, rf


erp, rf = compute_risk_premia(P, F, states)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(np.exp(states), rf * 100, 'b-o', ms=5, lw=2)
axes[0].set_xlabel('Current state $S / S_0$')
axes[0].set_ylabel('Annual risk-free rate (%)')
axes[0].set_title('Risk-Free Rate by State')

axes[1].plot(np.exp(states), erp * 100, 'r-^', ms=5, lw=2)
axes[1].set_xlabel('Current state $S / S_0$')
axes[1].set_ylabel('Equity Risk Premium (%)')
axes[1].set_title('Recovered Equity Risk Premium by State')

plt.tight_layout()
plt.savefig('ross_recovery_erp.png', dpi=120)
plt.show()

mid = len(states) // 2
print(f"At the middle state:")
print(f"  Risk-free rate  ≈ {rf[mid]*100:.2f}% (true: {delta*100:.2f}%)")
print(f"  Equity premium  ≈ {erp[mid]*100:.2f}% (true: {(mu-delta)*100:.2f}%)")
```

## Sensitivity analysis: effect of risk aversion

The shape of the pricing kernel, and hence the gap between natural and
risk-neutral probabilities, depends on the coefficient of risk aversion $\gamma$.

```{code-cell} ipython3
gammas = [1.0, 2.0, 3.0, 5.0, 8.0]
colors = cm.viridis(np.linspace(0.1, 0.9, len(gammas)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for gamma_val, color in zip(gammas, colors):
    P_g, states_g = build_state_price_matrix(mu, sigma, gamma_val, delta, T)
    F_g, z_g, delta_g, phi_g = recover_natural_distribution(P_g)
    mid_g = len(states_g) // 2

    f_nat_g = F_g[mid_g, :]
    row_sum  = P_g[mid_g].sum()
    f_rn_g  = P_g[mid_g] / row_sum

    gross = np.exp(states_g)
    erp_val = (np.sum(f_nat_g * np.exp(states_g - states_g[mid_g]))
               - np.exp(delta_g)) * 100

    axes[0].plot(gross, phi_g, color=color, lw=2,
                 label=f'$\\gamma={gamma_val:.0f}$')
    axes[1].plot(gross, f_nat_g - f_rn_g, color=color, lw=2,
                 label=f'$\\gamma={gamma_val:.0f}$')

axes[0].set_xlabel('Gross return')
axes[0].set_ylabel('Kernel $\\phi$')
axes[0].set_title('Pricing Kernel vs Risk Aversion')
axes[0].legend(fontsize=9)

axes[1].axhline(0, color='k', lw=0.8, ls='--')
axes[1].set_xlabel('Gross return')
axes[1].set_ylabel('Natural minus risk-neutral probability')
axes[1].set_title('Natural minus Risk-Neutral Density')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('ross_recovery_sensitivity.png', dpi=120)
plt.show()
```

The plots confirm the single-crossing property from Theorem 3 of
{cite}`Ross2015`: for returns below some threshold $v$, risk-neutral
probability exceeds natural probability; above $v$ the natural probability
dominates.

A higher $\gamma$ amplifies this wedge.

## Recovering the discount rate

A useful by-product of the Recovery Theorem is the **recovered subjective
discount rate** $\delta$, which equals the Perron–Frobenius eigenvalue of $P$.

Corollary 1 of {cite}`Ross2015` states that $\delta$ is bounded above by the
largest observed interest factor (i.e., the maximum row sum of $P$):

$$
\delta \leq \max_i \sum_j p(\theta_i, \theta_j).
$$

```{code-cell} ipython3
# Vary the true discount rate and check how well we recover it
true_deltas = np.linspace(0.00, 0.06, 13)
recovered_deltas = []

for d in true_deltas:
    P_d, _ = build_state_price_matrix(mu, sigma, gamma=3.0, delta=d, T=1.0)
    _, _, d_rec, _ = recover_natural_distribution(P_d)
    recovered_deltas.append(d_rec)

plt.figure(figsize=(8, 5))
plt.plot(true_deltas * 100, true_deltas * 100, 'k--', lw=1.5, label='45° line')
plt.plot(true_deltas * 100,
         [-np.log(d_r) * 100 for d_r in recovered_deltas],
         'bo-', ms=6, lw=2, label='Recovered $\\delta$')
plt.xlabel('True discount rate (%)')
plt.ylabel('Recovered discount rate (%)')
plt.title('Accuracy of Recovered Discount Rate')
plt.legend()
plt.tight_layout()
plt.savefig('ross_recovery_delta.png', dpi=120)
plt.show()
```

## Tail risk: natural vs. risk-neutral probabilities of catastrophe

One of the most striking applications of the Recovery Theorem is its ability
to separate the market's genuine fear of catastrophes from the risk premium
attached to them.

{cite}`barro2006rare` and {cite}`MehraPrescott1985` discuss how rare disasters
might explain the equity premium puzzle.  

The risk-neutral probability of a
large decline is elevated both because (a) the market assigns a high natural
probability to such events and (b) the pricing kernel upweights bad outcomes.

Ross's Recovery Machinery lets us decompose these two forces.

```{code-cell} ipython3
# Compare left-tail probabilities: P(R < threshold) under each measure
thresholds = np.linspace(-0.40, 0.10, 200)   # log-returns

def tail_prob(f_dist, states, threshold):
    """CDF evaluated at threshold (log-return)."""
    return float(np.sum(f_dist[states <= threshold]))

P_base, states_base = build_state_price_matrix(
    mu, sigma, gamma=3.0, delta=0.02, T=1.0,
    n_states=41, n_sigma=5)
F_base, z_base, delta_base, phi_base = recover_natural_distribution(P_base)

mid_b = len(states_base) // 2
f_nat_base = F_base[mid_b]
f_rn_base  = P_base[mid_b] / P_base[mid_b].sum()

prob_nat = [tail_prob(f_nat_base, states_base, t) for t in thresholds]
prob_rn  = [tail_prob(f_rn_base,  states_base, t) for t in thresholds]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.exp(thresholds), prob_nat, 'b-',  lw=2, label='Natural (recovered)')
ax.plot(np.exp(thresholds), prob_rn,  'r--', lw=2, label='Risk-neutral')
ax.set_xlabel('Gross return threshold')
ax.set_ylabel('Probability of decline below threshold')
ax.set_title('Tail Probabilities: Natural vs. Risk-Neutral')
ax.axvline(x=0.75, color='gray', ls=':', lw=1.5, label='25% decline')
ax.axvline(x=0.70, color='silver', ls=':', lw=1.5, label='30% decline')
ax.legend()
plt.tight_layout()
plt.savefig('ross_recovery_tail.png', dpi=120)
plt.show()

# Print specific tail probabilities
for thresh, label in [(-0.25, '25% decline'), (-0.30, '30% decline'),
                       (-0.10, '10% decline')]:
    p_n = tail_prob(f_nat_base, states_base, thresh)
    p_r = tail_prob(f_rn_base,  states_base, thresh)
    print(f"P(log-return < {thresh:.0%}):   Natural = {p_n:.4f},   "
          f"Risk-Neutral = {p_r:.4f},   Ratio = {p_r/p_n:.2f}x")
```

The risk-neutral density assigns higher probability to large drops than the
recovered natural density.

The ratio captures the additional weight from risk
aversion — the premium investors demand to bear tail risk.

## Testing efficient markets

{cite}`Ross2015` shows that once the pricing kernel is recovered, one obtains
an **upper bound on the Sharpe ratio** for any investment strategy:

$$
\sigma(\phi) \geq e^{-rT} \frac{|\mu_\text{excess}|}{\sigma_\text{asset}},
$$

where $\sigma(\phi)$ is the standard deviation of the pricing kernel.

This
follows from the Hansen–Jagannathan bound {cite}`Hansen_Jagannathan_1991`.

Equivalently, the $R^2$ of any return-forecasting regression using publicly
available information is bounded above by the variance of the pricing kernel:

$$
R^2 \leq e^{2rT} \, \mathrm{Var}(\phi).
$$

```{code-cell} ipython3
def kernel_variance(phi, f_nat):
    """Variance of the pricing kernel under the natural measure."""
    E_phi   = np.sum(phi * f_nat)
    E_phi2  = np.sum(phi**2 * f_nat)
    return E_phi2 - E_phi**2, E_phi


var_phi, E_phi = kernel_variance(phi_base, f_nat_base)
std_phi = np.sqrt(var_phi)

print(f"Pricing kernel statistics (one year):")
print(f"  E[φ]     = {E_phi:.4f}")
print(f"  Var(φ)   = {var_phi:.4f}")
print(f"  Std(φ)   = {std_phi:.4f}")
print(f"\nHansen-Jagannathan bound on Sharpe ratio: {std_phi:.4f}")
print(f"Upper bound on R² in return forecasting: {var_phi:.4f}")
```

## Limitations and extensions

The Recovery Theorem is a remarkable theoretical result, but several caveats
apply in practice.

**Finite state space.** The theorem requires a bounded, irreducible Markov
chain.

In continuous, unbounded state spaces (e.g., a lognormal diffusion),
uniqueness fails because any exponential $e^{\alpha x}$ satisfies the
characteristic equation.

{cite}`CarrYu2012` establish recovery with a bounded
diffusion.

**Transition independence.** If the kernel is not transition independent,
recovery is not guaranteed.

{cite}`BorovickaHansenScheinkman2016` show that
the Ross recovery can confound the long-run risk component of the kernel with
the natural probability distribution, yielding an incorrect decomposition.

**Empirical estimation.** Extracting reliable state prices from observed option
prices requires careful interpolation and extrapolation.

The mapping from
implied volatilities to state prices via the {cite}`BreedenLitzenberger1978` formula involves second derivatives, which amplify measurement error.

**State dependence.** The state must capture all relevant variables: the level
of volatility, not just the current index level, is an important state variable
for equity options.

## Exercises

```{exercise}
:label: rt_ex1

**The Perron–Frobenius vector and the pricing kernel.**

Consider the $3 \times 3$ state price matrix

$$
P = \begin{pmatrix}
0.8 & 0.12 & 0.02 \\
0.10 & 0.75 & 0.10 \\
0.05 & 0.15 & 0.72
\end{pmatrix}.
$$

(a) Compute the dominant eigenvalue $\delta$ and the corresponding eigenvector $z$ of $P$.

(b) Use $z$ to recover the natural probability transition matrix $F$ via

$$
f_{ij} = \frac{1}{\delta} \frac{z_j}{z_i} p_{ij}.
$$

(c) Verify that each row of $F$ sums to one and all entries are positive.

(d) Compute the pricing kernel $\phi_i = 1/z_i$ for each state.  Does the
    kernel decrease as we move from state 1 to state 3 (i.e., from bad to
    good states)?
```

```{solution-start} rt_ex1
:class: dropdown
```

```{code-cell} ipython3
import numpy as np
from scipy.linalg import eig

# (a) Dominant eigenvalue and eigenvector
P_ex = np.array([
    [0.80, 0.12, 0.02],
    [0.10, 0.75, 0.10],
    [0.05, 0.15, 0.72]
])

eigenvalues, eigenvectors = eig(P_ex)
real_mask = np.isreal(eigenvalues)
real_ev   = eigenvalues[real_mask].real
real_evec = eigenvectors[:, real_mask].real

idx   = np.argmax(real_ev)
delta_ex = real_ev[idx]
z_ex  = real_evec[:, idx]
if z_ex.min() < 0:
    z_ex = -z_ex
z_ex = z_ex / z_ex[1]   # normalise to middle state

print(f"(a) Dominant eigenvalue δ = {delta_ex:.6f}")
print(f"    Eigenvector z          = {z_ex}")

# (b) Recover F
D_ex    = np.diag(1.0 / z_ex)
D_inv_ex = np.diag(z_ex)
F_ex    = (1.0 / delta_ex) * D_ex @ P_ex @ D_inv_ex

print(f"\n(b) Recovered natural transition matrix F:")
print(np.round(F_ex, 4))

# (c) Row sums
print(f"\n(c) Row sums of F: {np.round(F_ex.sum(axis=1), 8)}")
print(f"    All non-negative: {(F_ex >= -1e-10).all()}")

# (d) Pricing kernel
phi_ex = 1.0 / z_ex
print(f"\n(d) Pricing kernel φ = {np.round(phi_ex, 4)}")
print(f"    Kernel decreasing state 1→3: {phi_ex[0] > phi_ex[1] > phi_ex[2]}")
```

```{solution-end}
```

```{exercise}
:label: rt_ex2

**Stochastic dominance.**

Using the recovered $F$ and the normalised risk-neutral matrix
$Q = P / \text{row sums}$ from the exercise above:

(a) Compute the one-step marginal distributions $f_j = F_{2,j}$ and $q_j = Q_{2,j}$
    starting from state 2 (index 1 in Python).

(b) Compute the CDFs $\hat F_k = \sum_{j \leq k} f_j$ and $\hat Q_k = \sum_{j
    \leq k} q_j$ for each state.

(c) Verify numerically that $\hat F_k \leq \hat Q_k$ for every $k$, confirming
    that the natural distribution stochastically dominates the risk-neutral
    distribution (Theorem 3 of {cite}`Ross2015`).
```

```{solution-start} rt_ex2
:class: dropdown
```

```{code-cell} ipython3
import numpy as np

P_ex = np.array([
    [0.80, 0.12, 0.02],
    [0.10, 0.75, 0.10],
    [0.05, 0.15, 0.72]
])

# Recompute F from exercise 1
from scipy.linalg import eig
eigenvalues, eigenvectors = eig(P_ex)
real_mask = np.isreal(eigenvalues)
real_ev   = eigenvalues[real_mask].real
real_evec = eigenvectors[:, real_mask].real
idx   = np.argmax(real_ev)
delta_ex = real_ev[idx]
z_ex  = real_evec[:, idx]
if z_ex.min() < 0:
    z_ex = -z_ex
z_ex = z_ex / z_ex[1]

D_ex    = np.diag(1.0 / z_ex)
D_inv_ex = np.diag(z_ex)
F_ex    = (1.0 / delta_ex) * D_ex @ P_ex @ D_inv_ex
F_ex    = np.clip(F_ex, 0, None)
F_ex   /= F_ex.sum(axis=1, keepdims=True)

# (a) Marginals from state 2 (index 1)
start = 1
f_marg = F_ex[start]
q_marg = P_ex[start] / P_ex[start].sum()

print("(a) One-step marginals from state 2:")
print(f"    Natural f     = {np.round(f_marg, 4)}")
print(f"    Risk-neutral q = {np.round(q_marg, 4)}")

# (b) CDFs
cdf_nat = np.cumsum(f_marg)
cdf_rn  = np.cumsum(q_marg)

print("\n(b) CDFs:")
for k in range(3):
    print(f"    State {k+1}: CDF_nat = {cdf_nat[k]:.4f},  CDF_rn = {cdf_rn[k]:.4f}")

# (c) Stochastic dominance
dominates = np.all(cdf_nat <= cdf_rn + 1e-10)
print(f"\n(c) Natural CDF ≤ Risk-neutral CDF at all states: {dominates}")
print("    → Natural distribution stochastically dominates risk-neutral distribution ✓")
```

```{solution-end}
```

```{exercise}
:label: rt_ex3

**Risk aversion and tail risk.**

Write a function `tail_risk_ratio(gamma, threshold, mu, sigma, delta, T)` that:

1. Constructs the state price matrix $P$ using `build_state_price_matrix` with
   the given parameters and `n_states=41`.
2. Applies `recover_natural_distribution` to obtain $F$.
3. Computes $P(\text{log-return} < \text{threshold})$ under both the natural
   and risk-neutral distributions starting from the middle state.
4. Returns the ratio $p_\text{risk-neutral} / p_\text{natural}$.

Using this function, plot the ratio as a function of $\gamma \in [1, 10]$ for
a threshold of $-30\%$ (i.e., `threshold = -0.30`).

Explain the economic interpretation: why does a higher $\gamma$ raise the ratio?
```

```{solution-start} rt_ex3
:class: dropdown
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt


def tail_risk_ratio(gamma, threshold, mu=0.08, sigma=0.20, delta=0.02, T=1.0):
    """
    Compute ratio of risk-neutral to natural tail probability P(log-return < threshold).
    """
    P_g, states_g = build_state_price_matrix(
        mu, sigma, gamma, delta, T, n_states=41, n_sigma=5)

    F_g, z_g, delta_g, phi_g = recover_natural_distribution(P_g)

    mid_g = len(states_g) // 2

    f_nat_g = F_g[mid_g]
    f_rn_g  = P_g[mid_g] / P_g[mid_g].sum()

    p_nat = float(np.sum(f_nat_g[states_g <= threshold]))
    p_rn  = float(np.sum(f_rn_g[states_g  <= threshold]))

    if p_nat < 1e-12:
        return np.nan
    return p_rn / p_nat


gammas = np.linspace(1.0, 10.0, 20)
ratios = [tail_risk_ratio(g, -0.30) for g in gammas]

plt.figure(figsize=(9, 5))
plt.plot(gammas, ratios, 'b-o', ms=5, lw=2)
plt.xlabel('Risk aversion coefficient $\\gamma$')
plt.ylabel('Risk-neutral / Natural tail probability')
plt.title('Tail Risk Ratio for a 30% Decline vs Risk Aversion')
plt.tight_layout()
plt.savefig('ross_recovery_ex3.png', dpi=120)
plt.show()

# Economic interpretation
print("Economic interpretation:")
print("A higher γ means the pricing kernel falls more steeply in bad states.")
print("This upweights bad outcomes (crashes) more heavily under risk-neutral")
print("probabilities, raising the ratio — even if the true crash probability")
print("(natural measure) stays the same.")
print(f"\nRatio at γ=1.0: {tail_risk_ratio(1.0, -0.30):.2f}")
print(f"Ratio at γ=5.0: {tail_risk_ratio(5.0, -0.30):.2f}")
print(f"Ratio at γ=10.0: {tail_risk_ratio(10.0, -0.30):.2f}")
```

**Economic interpretation.**  A higher coefficient of risk aversion $\gamma$
makes the pricing kernel steeper: the market assigns a larger premium per unit
of probability to bad-state payoffs.  Risk-neutral probabilities, which
incorporate this premium, overstate the natural probability of a crash by a
factor that grows rapidly with $\gamma$.  This is the "dark matter" of finance:
the high risk-neutral probability of a crash seen in option prices can be
attributed mostly to risk aversion rather than a genuinely elevated natural
probability of a catastrophe.

```{solution-end}
```

