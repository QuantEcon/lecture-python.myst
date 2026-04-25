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

Option prices reveal **risk-neutral probabilities**, the probabilities implied by asset
prices once risk adjustments have been folded in.

These are not the **natural probabilities** that investors actually assign to future
states of the world.

The two differ because risk-neutral probabilities blend together two distinct objects:
the market's true beliefs about the future, and investors' aversion to risk.

The link between them is the **pricing kernel**, which reweights natural probabilities
to deliver state prices.

For example, using the Arrow-security language from {doc}`ge_arrow`, suppose tomorrow
has two states, recession and boom, with natural probabilities $(0.5, 0.5)$ and pricing
kernels $(1.2, 0.7)$.

The Arrow prices are then $(0.6, 0.35)$, so the riskless discount factor is the row sum
$0.95$.

Normalizing the Arrow prices by this row sum gives risk-neutral probabilities
$(0.6/0.95, 0.35/0.95) \approx (0.63, 0.37)$, which overweight the recession state even
though the natural probability of recession is only $0.5$.

Separating beliefs from risk aversion has traditionally required parametric assumptions
about the preferences of a representative investor.

{cite:t}`Ross2015` showed otherwise.

Under a structural restriction on the pricing kernel called **transition independence**,
the natural probability distribution and the pricing kernel can be uniquely recovered
from state prices alone with no historical return data and no assumed utility
function.

This is the **Recovery Theorem**.

It has several important implications:

* It enables model-free extraction of the market's forward-looking probability
  distribution from option prices.
* It provides model-free tests of the efficient market hypothesis.
* It sheds light on the "dark matter" of finance: the probability of rare
  catastrophic events embedded in market prices.

This lecture covers

* the Arrow–Debreu framework linking state prices, the pricing kernel, and natural
  probabilities,
* Ross's Recovery Theorem and its proof via the Perron–Frobenius theorem,
* an implementation that recovers the natural distribution from a
  simulated state-price matrix, and
* comparisons between risk-neutral and recovered natural densities.

Let's import the packages we'll need.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.stats import norm
import matplotlib.cm as cm
```

## Model setup

### Arrow–Debreu state prices

Consider a discrete-time, discrete-state economy.

At each date the economy occupies one of $m$ states $\theta_1, \ldots, \theta_m$.

An **Arrow–Debreu security** pays \$1 if the economy is in state $\theta_j$ next period
and nothing otherwise.

Denote by $p(\theta_i, \theta_j)$ the price today, when the current state is $\theta_i$,
of the Arrow–Debreu security paying in state $\theta_j$ next period.

Collect these into an $m \times m$ **state price transition matrix**

$$
P = [p(\theta_i, \theta_j)]_{i,j=1}^m.
$$

As in {doc}`ge_arrow`, the row sums give the state-dependent riskless discount factor:
$\sum_j p(\theta_i, \theta_j) = e^{-r(\theta_i)}$.

### The pricing kernel

Using the stochastic-discount-factor notation studied in {doc}`markov_asset` and the
Arrow-security notation used in {doc}`ge_arrow`, the pricing kernel $\phi(\theta_i,
\theta_j)$ relates state prices to natural probabilities via

$$
p(\theta_i, \theta_j) = \phi(\theta_i, \theta_j) \, f(\theta_i, \theta_j),
$$

where $f(\theta_i, \theta_j)$ is the natural (conditional) probability of transitioning
from state $\theta_i$ to $\theta_j$.

As in the representative-agent equilibrium calculation in {doc}`ge_arrow`, the
canonical additively separable model with discount factor $\delta$ gives

$$
\phi(\theta_i, \theta_j) = \frac{p(\theta_i, \theta_j)}{f(\theta_i, \theta_j)}
    = \frac{\delta U'(c(\theta_j))}{U'(c(\theta_i))}.
$$ (eq:canon_ge)

The key structural property this implies is **transition independence**.

### Transition independence


```{prf:definition} Transition Independence
:label: def-transition-independence

A pricing kernel is **transition independent** if there exists a positive function $h$ on
the state space and a positive scalar $\delta$ such that for every transition from state
$\theta_i$ to $\theta_j$,

$$
\phi(\theta_i, \theta_j) = \delta \, \frac{h(\theta_j)}{h(\theta_i)}.
$$
```


Transition independence says the kernel depends on the *ending* state and normalizes by
the *beginning* state.

It holds for any agent with intertemporally additive separable utility (where $h = U'$).

In particular, this holds for {eq}`eq:canon_ge`.

Ross also notes that some Epstein--Zin specifications can produce transition-independent
kernels {cite}`Epstein_Zin1989`, although {doc}`misspecified_recovery` shows that
recursive utility with nontrivial continuation-value martingales need not satisfy the
Ross restriction.

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

Since $F$ is a stochastic matrix, its rows sum to one: $F e = e$ where $e$ is the vector
of ones.

Substituting the expression for $F$:

$$
\frac{1}{\delta} D P D^{-1} e = e
\quad \Longrightarrow \quad
P z = \delta z, \quad z \equiv D^{-1} e.
$$

This is an **eigenvalue problem**: we seek a positive vector $z$ and scalar $\delta$
satisfying $Pz = \delta z$.

In principle every eigenvalue-eigenvector pair of $P$ is a formal solution, but only the
one with a strictly positive eigenvector is economically valid: $D_{ii} = 1/z_i$ must be
positive (so $z_i > 0$), and $F$ must have nonnegative entries.

The theorem below guarantees that exactly one such pair exists.

```{prf:theorem} Perron--Frobenius
:label: thm-perron-frobenius

If $A$ is a nonnegative irreducible matrix, then

1. $A$ has a unique largest positive real eigenvalue $r$ (the Perron root).
2. There exists a strictly positive eigenvector $z \gg 0$ with $Az = rz$,
   unique up to scaling.
```

The proof uses the invariance of the positive cone and irreducibility to isolate the
unique positive ray associated with the Perron root.

See Section 1.2.3 of {cite}`Sargent_Stachurski_2024` for details.

Applied to the recovery problem: the Perron root is $\delta$ (the subjective discount
factor) and the Perron vector $z$ determines $D$ via $D_{ii} = 1/z_i$, closing the
system uniquely.


### Ross's recovery theorem

The three assumptions in the theorem each carry a specific role.

No-arbitrage guarantees that $P$ has nonnegative entries and that the state prices
encode a well-defined pricing measure.

Irreducibility ensures the economy is not divided into disconnected sub-economies --
without it, the Perron–Frobenius theorem gives multiple candidate eigenvectors and
recovery breaks down.

Transition independence is the key economic restriction: it says the pricing kernel
factors as $\delta h(\theta_j)/h(\theta_i)$, so the entire kernel is pinned down by a
single vector $h$ (or equivalently $z$).


```{prf:theorem} Recovery Theorem
:label: thm-ross-recovery

Suppose prices provide no arbitrage opportunities, that the state
price transition matrix $P$ is irreducible, and that the pricing kernel is transition
independent.

Then there exists a *unique* positive solution $(\delta, z, F)$ to the recovery problem.

That is, for any set of state prices there is a unique compatible natural probability
transition matrix and a unique pricing kernel.
```

```{prf:proof}
Because $P$ is nonnegative and irreducible, the Perron–Frobenius theorem gives a unique
positive eigenvector $z \gg 0$ with positive eigenvalue $\lambda > 0$ satisfying
$Pz = \lambda z$.

Setting

$$
\delta = \lambda, \qquad D_{ii} = \frac{1}{z_i},
$$

the natural probability transition matrix is uniquely recovered as

$$
f_{ij} = \frac{1}{\delta} \frac{z_j}{z_i} \, p_{ij}.
$$

To confirm $F$ is stochastic, note that all entries are nonnegative (since
$p_{ij} \geq 0$ and $z_i, z_j > 0$) and

$$
\sum_j f_{ij}
= \frac{1}{\delta z_i} \sum_j z_j \, p_{ij}
= \frac{[Pz]_i}{\delta z_i}
= \frac{\delta z_i}{\delta z_i} = 1.
$$

Uniqueness follows from the uniqueness of the Perron--Frobenius eigenvector.
```

### Pricing kernel from the eigenvector

The recovered kernel values are

$$
\phi(\theta_i, \theta_j) = \delta \frac{z_i}{z_j},
\qquad h(\theta_i) = \frac{\delta}{z_i},
$$

where $h(\theta_i) = \delta/z_i$ follows from $D_{ii} = h(\theta_i)/\delta = 1/z_i$.

Destination states with high $z_j$ have **low** kernel values: for a fixed origin $i$,
the kernel $\delta z_i/z_j$ is decreasing in $z_j$.

This means the market assigns relatively less pricing weight per unit of probability to
high-$z_j$ outcomes -- consistent with those states being "good times" that require less
insurance.

### Corollary: risk-neutral pricing when rates are state-independent


```{prf:corollary}
:label: cor-risk-neutral-recovery

If the riskless rate is the same in all states ($Pe = \gamma e$ for
some scalar $\gamma$), then the unique natural distribution consistent with recovery is
the risk-neutral (martingale) distribution itself: $F = (1/\gamma) P$.
```

```{prf:proof}
When $Pe = \gamma e$, the vector of ones $e$ is the Perron eigenvector with eigenvalue
$\gamma$.

By the uniqueness part of the Perron--Frobenius theorem, $z = e$ (up to scaling) and
$\delta = \gamma$.

Setting $z = e$ gives $D = I$ (the identity matrix), so

$$
F = \frac{1}{\delta} D P D^{-1} = \frac{1}{\gamma} P. \qquad \square
$$
```

This remarkable result says that with a constant interest rate and a bounded irreducible
state space, recovery forces risk-neutrality -- a non-trivial restriction of the model.

## Python implementation

We now implement the Recovery Theorem numerically.

### Building a state price matrix from a lognormal model

Following {cite}`Ross2015` Section IV, suppose the natural distribution of log-returns
over one period is normal:

$$
\log(S_T/S_0) \sim \mathcal{N}\!\left((\mu - \tfrac{1}{2}\sigma^2)T, \sigma^2 T\right).
$$

With a CRRA pricing kernel $\phi(S_0, S_T) = e^{-\delta T}(S_T/S_0)^{-\gamma}$, the
state price density is

$$
f_{ij} \propto
    n\!\left(\frac{s_j - s_i - (\mu - \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}\right)
    \Delta s,
\qquad
p_{ij} = e^{-\delta T} e^{-\gamma(s_j - s_i)} f_{ij},
$$

where $s_i = \ln S_i$, $s_j = \ln S_j$, $n(\cdot)$ is the standard normal density, and
the discretized probabilities $f_{ij}$ are normalized row by row.

We discretize this onto a grid of $m$ states and build the matrix $P$.

```{code-cell} ipython3
def build_state_price_matrix(μ, σ, γ, δ, T=1.0, n_states=11, n_σ=5):
    """Build a discretized lognormal/CRRA state-price matrix."""
    states = np.linspace(-n_σ * σ * np.sqrt(T),
                          n_σ * σ * np.sqrt(T),
                          n_states)
    ds = states[1] - states[0]

    m = n_states
    P = np.zeros((m, m))
    F = np.zeros((m, m))

    drift = (μ - 0.5 * σ**2) * T

    # First build a row-stochastic natural transition matrix on the bounded grid.
    for i in range(m):
        s_i = states[i]
        for j in range(m):
            s_j = states[j]
            log_return = s_j - s_i
            F[i, j] = norm.pdf(log_return, loc=drift,
                               scale=σ * np.sqrt(T)) * ds

        F[i] = F[i] / F[i].sum()

        # Price each Arrow claim as natural probability times the CRRA kernel.
        for j in range(m):
            log_return = states[j] - s_i
            kernel = np.exp(-δ * T) * np.exp(-γ * log_return)
            P[i, j] = kernel * F[i, j]

    return P, states
```

```{code-cell} ipython3
μ = 0.08    # 8% annual expected return
σ = 0.20    # 20% annual volatility
γ = 3.0     # CRRA coefficient
δ = 0.02    # 2% annual discount rate
T = 1.0     # one-year horizon

P, states = build_state_price_matrix(μ, σ, γ, δ, T,
                                     n_states=11, n_σ=5)

print("State-price row sums:")
print(np.round(P.sum(axis=1), 4))
print(f"Middle-state risk-free rate: {-np.log(P[5].sum()):.4f}")
```

### Applying the recovery theorem

The Recovery Theorem requires computing the **dominant eigenvector** of $P$.

```{code-cell} ipython3
def recover_natural_distribution(P):
    """Recover natural probabilities and the pricing kernel from state prices."""
    m = P.shape[0]

    eigenvalues, eigenvectors = eig(P)

    # Ross recovery uses the Perron root and its strictly positive eigenvector.
    real_mask = np.isreal(eigenvalues)
    real_eigenvalues = eigenvalues[real_mask].real
    real_eigenvectors = eigenvectors[:, real_mask].real

    idx = np.argmax(real_eigenvalues)
    δ_recovered = real_eigenvalues[idx]
    z = real_eigenvectors[:, idx]

    if np.mean(z) < 0:
        z = -z

    z = z / z[m // 2]

    D = np.diag(1.0 / z)
    D_inv = np.diag(z)

    # The diagonal similarity transform converts state prices into probabilities.
    F = (1.0 / δ_recovered) * D @ P @ D_inv

    F = np.clip(F, 0, None)
    F = F / F.sum(axis=1, keepdims=True)

    # The kernel is reported relative to the middle state normalization.
    φ = 1.0 / z

    return F, z, δ_recovered, φ
```

```{code-cell} ipython3
F, z, δ_rec, φ = recover_natural_distribution(P)

print(f"Recovered discount factor δ = {δ_rec:.6f}  (true: {np.exp(-δ):.6f})")
print(f"Recovered discount rate     = {-np.log(δ_rec):.6f}  (true: {δ:.6f})")
print("Recovered kernel φ:")
print(np.round(φ, 4))
print("Row sums of recovered F:")
print(np.round(F.sum(axis=1), 6))
```

### Visualizing natural vs. risk-neutral distributions

A key insight of {cite}`Ross2015` is that the natural distribution systematically
differs from the risk-neutral one.

In particular, the natural distribution stochastically dominates the risk-neutral
distribution (Theorem 3 in {cite}`Ross2015`).

```{code-cell} ipython3
mid = len(states) // 2

row_sums = P.sum(axis=1, keepdims=True)

# Normalize Arrow prices by the one-period riskless bond price in each state.
Q_rn = P / row_sums

f_nat = F[mid, :]
f_rn = Q_rn[mid, :]

gross_returns = np.exp(states)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(gross_returns, f_nat, 'b-o', ms=5, label='natural (recovered)', lw=2)
axes[0].plot(gross_returns, f_rn, 'r--s', ms=5, label='risk-neutral', lw=2)
axes[0].set_xlabel('gross return $S_T / S_0$')
axes[0].set_ylabel('probability')
axes[0].set_title('one-period marginal distributions')
axes[0].legend()

axes[1].plot(gross_returns, φ, 'g-^', ms=5, lw=2)
axes[1].set_xlabel('gross return $S_T / S_0$')
axes[1].set_ylabel('kernel $\\phi$ (relative)')
axes[1].set_title('recovered pricing kernel')
plt.show()
```

```{code-cell} ipython3
E_nat = np.sum(f_nat * gross_returns)
E_rn = np.sum(f_rn * gross_returns)
std_nat = np.sqrt(np.sum(f_nat * (gross_returns - E_nat)**2))
std_rn = np.sqrt(np.sum(f_rn * (gross_returns - E_rn)**2))

# The row sum is the price of a sure one-dollar payoff next period.
risk_free = np.sum(P[mid])

print("One-period summary")
print(f"{'':30s} {'Natural':>12s}   {'Risk-Neutral':>12s}")
print("-" * 57)
print(f"{'Expected gross return':30s} {E_nat:>12.4f}   {E_rn:>12.4f}")
print(f"{'Std dev':30s} {std_nat:>12.4f}   {std_rn:>12.4f}")
print(f"{'Risk-free discount factor':30s} {risk_free:>12.4f}")
print(f"{'Annual risk-free rate':30s} {-np.log(risk_free):>12.4f}")
print(f"{'Arithmetic equity premium':30s} {E_nat - 1/risk_free:>12.4f}")
```

### Stochastic dominance

Theorem 3 of {cite}`Ross2015` shows that the natural marginal density **first-order
stochastically dominates** the risk-neutral density: the CDF of the natural distribution
lies *below* that of the risk-neutral distribution.

Because the pricing kernel is declining (investors fear bad outcomes), risk-neutral
probabilities overweight bad states and underweight good states relative to the natural
measure.

```{code-cell} ipython3
cdf_nat = np.cumsum(f_nat)
cdf_rn = np.cumsum(f_rn)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(gross_returns, cdf_nat, 'b-o', ms=5, lw=2, label='natural cdf')
ax.plot(gross_returns, cdf_rn, 'r--s', ms=5, lw=2, label='risk-neutral cdf')
ax.set_xlabel('gross return $S_T / S_0$')
ax.set_ylabel('cumulative probability')
ax.set_title('stochastic dominance: natural cdf lies below risk-neutral cdf')
ax.legend()
plt.show()

print(f"Natural CDF <= Risk-neutral CDF at all states: "
      f"{np.all(cdf_nat <= cdf_rn + 1e-10)}")
```

## Extracting the pricing kernel and risk premium

The pricing kernel recovered from $P$ via the Perron–Frobenius theorem has the following
interpretation.

In the CRRA model the kernel is proportional to $\exp(-\gamma \cdot \text{log-return})$,
so it is decreasing in the return.

The **log equity risk premium** computed below is the log expected gross return under
the natural measure minus the continuously compounded risk-free rate:

$$
\text{log ERP}_i
= \log\left(\sum_j f_{ij}\frac{S_j}{S_i}\right) - r_{f,i},
\qquad
r_{f,i} = -\log\left(\sum_j p_{ij}\right).
$$

```{code-cell} ipython3
def compute_risk_premia(P, F, states):
    """Compute log equity premia and risk-free rates by starting state."""
    m = len(states)
    gross_returns = np.exp(states)

    rf = np.zeros(m)
    erp = np.zeros(m)

    for i in range(m):
        discount = P[i].sum()
        rf[i] = -np.log(discount)

        # Gross return from current state i to future state j.
        relative_returns = np.exp(states - states[i])
        E_R_nat = np.sum(F[i] * relative_returns)

        erp[i] = np.log(E_R_nat) - rf[i]

    return erp, rf


erp, rf = compute_risk_premia(P, F, states)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(np.exp(states), rf * 100, 'b-o', ms=5, lw=2)
axes[0].set_xlabel('current state $S / S_0$')
axes[0].set_ylabel('annual risk-free rate (%)')
axes[0].set_title('risk-free rate by state')

axes[1].plot(np.exp(states), erp * 100, 'r-^', ms=5, lw=2)
axes[1].set_xlabel('current state $S / S_0$')
axes[1].set_ylabel('log equity risk premium (%)')
axes[1].set_title('recovered log equity risk premium by state')

plt.show()

mid = len(states) // 2
print("Middle state:")
print(f"  Risk-free rate       approx {rf[mid]*100:.2f}% "
      f"(calibration: {δ*100:.2f}%)")
print(f"  Log equity premium  approx {erp[mid]*100:.2f}% "
      f"(calibration: {(μ-δ)*100:.2f}%)")
```

## Sensitivity analysis: effect of risk aversion

The shape of the pricing kernel, and hence the gap between natural and risk-neutral
probabilities, depends on the coefficient of risk aversion $\gamma$.

```{code-cell} ipython3
γs = [1.0, 2.0, 3.0, 5.0, 8.0]
colors = cm.viridis(np.linspace(0.1, 0.9, len(γs)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for γ_val, color in zip(γs, colors):
    P_g, states_g = build_state_price_matrix(μ, σ, γ_val, δ, T)
    F_g, z_g, δ_g, φ_g = recover_natural_distribution(P_g)
    mid_g = len(states_g) // 2

    f_nat_g = F_g[mid_g, :]
    row_sum = P_g[mid_g].sum()
    f_rn_g = P_g[mid_g] / row_sum

    gross = np.exp(states_g)

    axes[0].plot(gross, φ_g, color=color, lw=2,
                 label=f'$\\gamma={γ_val:.0f}$')
    axes[1].plot(gross, f_nat_g - f_rn_g, color=color, lw=2,
                 label=f'$\\gamma={γ_val:.0f}$')

axes[0].set_xlabel('gross return')
axes[0].set_ylabel('kernel $\\phi$')
axes[0].set_title('pricing kernel vs risk aversion')
axes[0].legend(fontsize=9)

axes[1].axhline(0, color='k', lw=0.8, ls='--')
axes[1].set_xlabel('gross return')
axes[1].set_ylabel('natural minus risk-neutral probability')
axes[1].set_title('natural minus risk-neutral density')
axes[1].legend(fontsize=9)

plt.show()
```

The plots confirm the single-crossing property from Theorem 3 of {cite}`Ross2015`: for
returns below some threshold $v$, risk-neutral probability exceeds natural probability;
above $v$ the natural probability dominates.

A higher $\gamma$ amplifies this wedge.

## Recovering the discount rate

A useful by-product of the Recovery Theorem is the **recovered subjective discount
factor** $\delta$, which equals the Perron–Frobenius eigenvalue of $P$.

The corresponding continuously compounded discount rate is $-\log \delta$.

Corollary 1 of {cite}`Ross2015` states that $\delta$ is bounded above by the largest
observed interest factor (i.e., the maximum row sum of $P$):

$$
\delta \leq \max_i \sum_j p(\theta_i, \theta_j).
$$

```{code-cell} ipython3
true_δs = np.linspace(0.00, 0.06, 13)
recovered_δs = []

for d in true_δs:
    P_d, _ = build_state_price_matrix(μ, σ, γ=3.0, δ=d, T=1.0)
    _, _, d_rec, _ = recover_natural_distribution(P_d)
    recovered_δs.append(d_rec)

plt.figure(figsize=(8, 5))
plt.plot(true_δs * 100, true_δs * 100, 'k--', lw=1.5, label='45 deg line')
plt.plot(true_δs * 100,
         [-np.log(d_r) * 100 for d_r in recovered_δs],
         'bo-', ms=6, lw=2, label='recovered $\\delta$')
plt.xlabel('true discount rate (%)')
plt.ylabel('recovered discount rate (%)')
plt.title('accuracy of recovered discount rate')
plt.legend()
plt.show()
```

## Tail risk: natural vs. risk-neutral probabilities of catastrophe

One of the most striking applications of the Recovery Theorem is its ability to separate
the market's genuine fear of catastrophes from the risk premium attached to them.

{cite:t}`barro2006rare` and {cite:t}`MehraPrescott1985` discuss how rare disasters might
explain the equity premium puzzle.

The risk-neutral probability of a large decline is elevated both because (a) the market
assigns a high natural probability to such events and (b) the pricing kernel upweights
bad outcomes.

Ross's Recovery Machinery lets us decompose these two forces.

```{code-cell} ipython3
thresholds = np.linspace(-0.40, 0.10, 200)

def tail_prob(f_dist, states, threshold):
    """Left-tail probability for log returns."""
    return float(np.sum(f_dist[states <= threshold]))

P_base, states_base = build_state_price_matrix(
    μ, σ, γ=3.0, δ=0.02, T=1.0,
    n_states=41, n_σ=5)
F_base, z_base, δ_base, φ_base = recover_natural_distribution(P_base)

mid_b = len(states_base) // 2
f_nat_base = F_base[mid_b]
f_rn_base = P_base[mid_b] / P_base[mid_b].sum()

prob_nat = [tail_prob(f_nat_base, states_base, t) for t in thresholds]
prob_rn = [tail_prob(f_rn_base, states_base, t) for t in thresholds]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.exp(thresholds), prob_nat, 'b-', lw=2, label='natural (recovered)')
ax.plot(np.exp(thresholds), prob_rn, 'r--', lw=2, label='risk-neutral')
ax.set_xlabel('gross return threshold')
ax.set_ylabel('probability of decline below threshold')
ax.set_title('tail probabilities: natural vs. risk-neutral')
ax.axvline(x=0.75, color='gray', ls=':', lw=1.5, label='25% decline')
ax.axvline(x=0.70, color='silver', ls=':', lw=1.5, label='30% decline')
ax.legend()
plt.show()

for thresh, label in [(-0.25, '25% decline'), (-0.30, '30% decline'),
                       (-0.10, '10% decline')]:
    p_n = tail_prob(f_nat_base, states_base, thresh)
    p_r = tail_prob(f_rn_base, states_base, thresh)
    print(f"P(log-return < {thresh:.0%}):   Natural = {p_n:.4f},   "
          f"Risk-Neutral = {p_r:.4f},   Ratio = {p_r/p_n:.2f}x")
```

The risk-neutral density assigns higher probability to large drops than the recovered
natural density.

The ratio captures the additional weight from risk aversion -- the premium investors
demand to bear tail risk.

## Testing efficient markets

{cite:t}`Ross2015` shows that once the pricing kernel is recovered, one obtains an **upper
bound on the Sharpe ratio** for any investment strategy:

$$
\sigma(\phi) \geq e^{-rT} \frac{|\mu_\text{excess}|}{\sigma_\text{asset}},
$$

where $\sigma(\phi)$ is the standard deviation of the pricing kernel.

This follows from the Hansen–Jagannathan bound {cite}`Hansen_Jagannathan_1991`.

Equivalently, the $R^2$ of any return-forecasting regression using publicly available
information is bounded above by the variance of the pricing kernel:

$$
R^2 \leq e^{2rT} \, \mathrm{Var}(\phi).
$$

```{code-cell} ipython3
def kernel_variance(φ, f_nat):
    """Return Var(φ) and E[φ]."""
    E_φ = np.sum(φ * f_nat)
    E_φ2 = np.sum(φ**2 * f_nat)
    return E_φ2 - E_φ**2, E_φ


var_φ, E_φ = kernel_variance(φ_base, f_nat_base)
std_φ = np.sqrt(var_φ)

print("Pricing kernel statistics:")
print(f"  E[φ]     = {E_φ:.4f}")
print(f"  Var(φ)   = {var_φ:.4f}")
print(f"  Std(φ)   = {std_φ:.4f}")
print(f"\nHansen-Jagannathan bound on Sharpe ratio: {std_φ:.4f}")
print(f"Upper bound on R^2 in return forecasting: {var_φ:.4f}")
```

## Limitations and extensions

The Recovery Theorem is a remarkable theoretical result, but several caveats apply in
practice.

**Finite state space.**

The theorem requires a bounded, irreducible Markov chain.

In continuous, unbounded state spaces (e.g., a lognormal diffusion), uniqueness fails
because any exponential $e^{\alpha x}$ satisfies the characteristic equation.

{cite:t}`CarrYu2012` establish recovery with a bounded diffusion.

**Transition independence.**

If the kernel is not transition independent, recovery is not guaranteed.

{cite:t}`BorovickaHansenScheinkman2016` show that the Ross recovery can confound the
long-run risk component of the kernel with the natural probability distribution,
yielding an incorrect decomposition.

**Empirical estimation.**

Extracting reliable state prices from observed option prices requires careful
interpolation and extrapolation.

The mapping from implied volatilities to state prices via the
{cite}`BreedenLitzenberger1978` formula involves second derivatives, which amplify
measurement error.

**State dependence.**

The state must capture all relevant variables: the level of volatility, not just the
current index level, is an important state variable for equity options.

## Exercises

```{exercise}
:label: rt_ex1

**The Perron–Frobenius vector and the pricing kernel.**

Consider the $3 \times 3$ state price matrix

$$
P = \begin{pmatrix}
0.5950 & 0.1700 & 0.0272 \\
0.1594 & 0.5525 & 0.1360 \\
0.0664 & 0.3188 & 0.5525
\end{pmatrix}.
$$

(a) Compute the dominant eigenvalue $\delta$ and the corresponding eigenvector $z$ of
$P$.

(b) Use $z$ to recover the natural probability transition matrix $F$ via

$$
f_{ij} = \frac{1}{\delta} \frac{z_j}{z_i} p_{ij}.
$$

(c) Verify that each row of $F$ sums to one and all entries are positive.

(d) Compute the pricing kernel $\phi_i = 1/z_i$ for each state.

Does the kernel decrease as we move from state 1 to state 3 (i.e., from bad to good
states)?
```

```{solution-start} rt_ex1
:class: dropdown
```

```{code-cell} ipython3
import numpy as np
from scipy.linalg import eig

P_ex = np.array([
    [0.5950, 0.1700, 0.0272],
    [0.159375, 0.5525, 0.1360],
    [0.06640625, 0.31875, 0.5525]
])

eigenvalues, eigenvectors = eig(P_ex)
real_mask = np.isreal(eigenvalues)
real_ev = eigenvalues[real_mask].real
real_evec = eigenvectors[:, real_mask].real

idx = np.argmax(real_ev)
δ_ex = real_ev[idx]
z_ex = real_evec[:, idx]
if z_ex.min() < 0:
    z_ex = -z_ex
z_ex = z_ex / z_ex[1]

print(f"δ = {δ_ex:.6f}")
print(f"z = {z_ex}")

D_ex = np.diag(1.0 / z_ex)
D_inv_ex = np.diag(z_ex)
F_ex = (1.0 / δ_ex) * D_ex @ P_ex @ D_inv_ex

print("\nRecovered F:")
print(np.round(F_ex, 4))

print(f"\nRow sums: {np.round(F_ex.sum(axis=1), 8)}")
print(f"Nonnegative: {(F_ex >= -1e-10).all()}")

φ_ex = 1.0 / z_ex
print(f"\nφ = {np.round(φ_ex, 4)}")
print(f"Decreasing: {φ_ex[0] > φ_ex[1] > φ_ex[2]}")
```

```{solution-end}
```

```{exercise}
:label: rt_ex2

**Stochastic dominance.**

Using the recovered $F$ and the normalised risk-neutral matrix $Q = P / \text{row sums}$
from the exercise above:

(a) Compute the one-step marginal distributions $f_j = F_{2,j}$ and $q_j = Q_{2,j}$
starting from state 2 (index 1 in Python).

(b) Compute the CDFs $\hat F_k = \sum_{j \leq k} f_j$ and
$\hat Q_k = \sum_{j \leq k} q_j$ for each state.

(c) Verify numerically that $\hat F_k \leq \hat Q_k$ for every $k$, confirming that the
natural distribution stochastically dominates the risk-neutral distribution (Theorem 3
of {cite}`Ross2015`).
```

```{solution-start} rt_ex2
:class: dropdown
```

```{code-cell} ipython3
import numpy as np

P_ex = np.array([
    [0.5950, 0.1700, 0.0272],
    [0.159375, 0.5525, 0.1360],
    [0.06640625, 0.31875, 0.5525]
])

from scipy.linalg import eig
eigenvalues, eigenvectors = eig(P_ex)
real_mask = np.isreal(eigenvalues)
real_ev = eigenvalues[real_mask].real
real_evec = eigenvectors[:, real_mask].real
idx = np.argmax(real_ev)
δ_ex = real_ev[idx]
z_ex = real_evec[:, idx]
if z_ex.min() < 0:
    z_ex = -z_ex
z_ex = z_ex / z_ex[1]

D_ex = np.diag(1.0 / z_ex)
D_inv_ex = np.diag(z_ex)
F_ex = (1.0 / δ_ex) * D_ex @ P_ex @ D_inv_ex
F_ex = np.clip(F_ex, 0, None)
F_ex /= F_ex.sum(axis=1, keepdims=True)

start = 1
f_marg = F_ex[start]
q_marg = P_ex[start] / P_ex[start].sum()

print("One-step marginals from state 2:")
print(f"natural     = {np.round(f_marg, 4)}")
print(f"risk-neutral = {np.round(q_marg, 4)}")

cdf_nat = np.cumsum(f_marg)
cdf_rn = np.cumsum(q_marg)

print("\nCDFs:")
for k in range(3):
    print(f"state {k+1}: natural = {cdf_nat[k]:.4f}, risk-neutral = {cdf_rn[k]:.4f}")

dominates = np.all(cdf_nat <= cdf_rn + 1e-10)
print(f"\nNatural CDF <= risk-neutral CDF: {dominates}")
```

```{solution-end}
```

```{exercise}
:label: rt_ex3

**Risk aversion and tail risk.**

Write a function `tail_risk_ratio(γ, threshold, μ, σ, δ, T)` that:

1. Constructs the state price matrix $P$ using `build_state_price_matrix` with
   the given parameters and `n_states=41`.
2. Applies `recover_natural_distribution` to obtain $F$.
3. Computes $P(\text{log-return} < \text{threshold})$ under both the natural
   and risk-neutral distributions starting from the middle state.
4. Returns the ratio $p_\text{risk-neutral} / p_\text{natural}$.

Using this function, plot the ratio as a function of $\gamma \in [1, 10]$ for a
threshold of $-30\%$ (i.e., `threshold = -0.30`).

Explain the economic interpretation: why does a higher $\gamma$ raise the ratio?
```

```{solution-start} rt_ex3
:class: dropdown
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt


def tail_risk_ratio(γ, threshold, μ=0.08, σ=0.20, δ=0.02, T=1.0):
    """Risk-neutral / natural left-tail probability."""
    P_g, states_g = build_state_price_matrix(
        μ, σ, γ, δ, T, n_states=41, n_σ=5)

    F_g, _, _, _ = recover_natural_distribution(P_g)

    mid_g = len(states_g) // 2

    f_nat_g = F_g[mid_g]
    f_rn_g  = P_g[mid_g] / P_g[mid_g].sum()

    p_nat = float(np.sum(f_nat_g[states_g <= threshold]))
    p_rn = float(np.sum(f_rn_g[states_g  <= threshold]))

    if p_nat < 1e-12:
        return np.nan
    return p_rn / p_nat


γs = np.linspace(1.0, 10.0, 20)
ratios = [tail_risk_ratio(g, -0.30) for g in γs]

plt.figure(figsize=(9, 5))
plt.plot(γs, ratios, 'b-o', ms=5, lw=2)
plt.xlabel('risk aversion coefficient $\\gamma$')
plt.ylabel('risk-neutral / natural tail probability')
plt.title('tail risk ratio for a 30% decline vs risk aversion')
plt.show()

print(f"Ratio at γ=1.0: {tail_risk_ratio(1.0, -0.30):.2f}")
print(f"Ratio at γ=5.0: {tail_risk_ratio(5.0, -0.30):.2f}")
print(f"Ratio at γ=10.0: {tail_risk_ratio(10.0, -0.30):.2f}")
```

**Economic interpretation.**

A higher coefficient of risk aversion $\gamma$ makes the pricing kernel steeper: the
market assigns a larger premium per unit of probability to bad-state payoffs.

Risk-neutral probabilities, which incorporate this premium, overstate the natural
probability of a crash by a factor that grows rapidly with $\gamma$.

This is the "dark matter" of finance: the high risk-neutral probability of a crash seen
in option prices can be attributed mostly to risk aversion rather than a genuinely
elevated natural probability of a catastrophe.

```{solution-end}
```
