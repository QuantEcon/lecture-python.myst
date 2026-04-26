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

# Misspecified recovery

```{contents} Contents
:depth: 2
```

## Overview

The lecture {doc}`ross_recovery` studies conditions under which recovery is valid.

There, **transition independence** lets us use Arrow prices to separate investors'
beliefs from the pricing kernel.

This lecture asks what the same Perron--Frobenius calculation delivers when transition
independence fails.

{cite:t}`BorovickaHansenScheinkman2016` show that the stochastic discount factor can be
decomposed into three pieces: a deterministic long-run discount component, a
state-dependent eigenfunction ratio, and a martingale likelihood ratio.

The first two pieces are exactly what the Perron--Frobenius eigenpair can absorb.

The martingale piece is different: it changes the probability measure.

In their words, it produces a probability measure that "absorbs long-term risk
adjustments" {cite}`BorovickaHansenScheinkman2016`.

Thus the probabilities recovered from Arrow prices need not be the correctly specified
transition probabilities for the state process.

Instead, they can already include compensation for long-run risk.

The likelihood ratio between the recovered probabilities and the correctly specified
probabilities is the martingale component.

If that martingale is constant, Ross recovery returns the correctly specified
transition probabilities.

If it is not constant, the recovered measure embeds long-horizon risk adjustments.

In the examples below, this typically shifts probability toward adverse long-run-risk
states, so the recovered measure looks more pessimistic than the correctly specified
probability law.

We will:

- use results from {doc}`ross_recovery` without re-proving it,
- diagnose misspecification through the likelihood-ratio martingale,
- show why recursive utility and permanent shocks break recovery,
- measure the difference in a long-run risk model.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
```

The next cell contains code inherited from the previous lecture: row-normalizing Arrow
prices, finding a positive Perron pair, and computing stationary distributions.

```{code-cell} ipython3
:tags: [hide-input]

def risk_neutral_probs(Q):
    """Normalize Arrow prices by one-period bond prices."""
    q_bonds = Q.sum(axis=1)
    P_bar = Q / q_bonds[:, None]
    return P_bar, q_bonds


def perron_frobenius(Q):
    """Positive Perron pair and induced long-term risk-neutral transition matrix."""
    eigenvalues, eigenvectors = linalg.eig(Q)
    eigenvalues = np.real_if_close(eigenvalues, tol=1000)
    eigenvectors = np.real_if_close(eigenvectors, tol=1000)

    real_mask = np.isreal(eigenvalues)
    vals = np.asarray(eigenvalues[real_mask].real, dtype=float)
    vecs = np.asarray(eigenvectors[:, real_mask].real, dtype=float)

    for idx in np.argsort(vals)[::-1]:
        exp_eta = vals[idx]
        e = vecs[:, idx]
        if e.sum() < 0:
            e = -e
        if exp_eta > 0 and np.all(e > 0):
            break
    else:
        raise ValueError("No strictly positive Perron eigenvector found")

    e = e / e.sum()
    eta = np.log(exp_eta)
    P_hat = (1 / exp_eta) * Q * e[None, :] / e[:, None]

    if np.max(np.abs(P_hat.sum(axis=1) - 1)) > 1e-8:
        raise ValueError("Recovered transition matrix is not stochastic")
    if P_hat.min() < -1e-10:
        raise ValueError("Recovered transition matrix has negative entries")

    return eta, exp_eta, e, P_hat


def stationary_dist(P):
    """Stationary distribution of an ergodic transition matrix."""
    n = P.shape[0]
    A = P.T - np.eye(n)
    A[-1] = 1
    b = np.zeros(n)
    b[-1] = 1
    return linalg.solve(A, b)


def martingale_increment(Q, P):
    """Likelihood-ratio increment from actual to recovered probabilities."""
    eta, exp_eta, e, P_hat = perron_frobenius(Q)
    H = np.ones_like(P)
    mask = P > 0
    H[mask] = P_hat[mask] / P[mask]
    return H, eta, e, P_hat
```

## One-period and long-term risk-neutral matrices

Let $\mathbf{P}=[p_{ij}]$ denote the correctly specified transition matrix and
$\mathbf{Q}=[q_{ij}]$ the Arrow price matrix.

Here "correctly specified" means the transition law that actually governs the Markov
state in the model.

The one-period stochastic discount factor (SDF) satisfies

$$
q_{ij} = s_{ij} p_{ij}.
$$

We will compare $\mathbf{P}$ with two probability matrices constructed from the same
Arrow price matrix $\mathbf{Q}$.

First, the **one-period risk-neutral matrix** divides each row of $\mathbf{Q}$ by the
price of a one-period discount bond in the current state:

$$
\bar p_{ij}
= \frac{q_{ij}}{\sum_k q_{ik}}.
$$

This matrix absorbs one-period risk adjustments into transition probabilities.

Second, the **long-term risk-neutral matrix** uses the positive Perron eigenpair of
$\mathbf{Q}$.

Let $(\exp(\hat \eta), \hat e)$ solve

$$
\mathbf{Q}\hat e = \exp(\hat \eta)\hat e.
$$

Then define

$$
\hat p_{ij}
= \exp(-\hat \eta) q_{ij} \frac{\hat e_j}{\hat e_i}.
$$

This construction removes the state-dependent Perron eigenfunction from Arrow prices
and returns a stochastic matrix $\hat{\mathbf{P}}$.

In {doc}`ross_recovery`, transition independence pins down the split between $s_{ij}$
and $p_{ij}$.

Here we drop transition independence.

The question is whether $\hat{\mathbf{P}}$ still
equals the correctly specified transition matrix $\mathbf{P}$.

### Where recovery works

We start with a three-state economy: recession, normal, and expansion.

The correctly specified transition matrix is deliberately simple.

For trend-stationary consumption and power utility, the SDF is

$$
s_{ij}=A\left(\frac{c_j}{c_i}\right)^{-\gamma}.
$$

This is a case where Ross recovery should return the correctly specified transition
matrix.

```{code-cell} ipython3
P_true = np.array([
    [0.70, 0.25, 0.05],
    [0.15, 0.65, 0.20],
    [0.05, 0.30, 0.65],
])

c_levels = np.array([0.997, 1.000, 1.003])
state_names = ['recession', 'normal', 'expansion']

δ = -np.log(0.99)   # monthly subjective discount rate
γ_power = 5.0       # risk aversion
g_c = 0.002         # monthly trend growth

# Price Arrow claims as actual probabilities times the power-utility SDF
S_power = (
    np.exp(-δ - γ_power * g_c)
    * (c_levels[None, :] / c_levels[:, None])**(-γ_power)
)
Q_power = S_power * P_true
```

We now compute both risk-neutral matrices from the same Arrow price matrix.

```{code-cell} ipython3
P_bar, q_bonds = risk_neutral_probs(Q_power)
η_hat, exp_η, e_hat, P_hat = perron_frobenius(Q_power)
π_true = stationary_dist(P_true)
π_bar = stationary_dist(P_bar)
π_hat = stationary_dist(P_hat)
```

The one-period risk-neutral matrix differs from the correctly specified matrix because
it includes one-period risk adjustments.

The long-term risk-neutral transition matrix coincides with the correctly specified
transition matrix because, after the Perron eigenfunction is removed, no
likelihood-ratio term remains.

Here is the likelihood-ratio term explicitly.

Define

$$
\hat h_{ij}
= \frac{\hat p_{ij}}{p_{ij}}
= \exp(-\hat\eta)s_{ij}\frac{\hat e_j}{\hat e_i}.
$$

For a path of states, the product

$$
\hat H_t
= \prod_{\tau=1}^t \hat h_{X_{\tau-1}, X_\tau}
$$

is the likelihood-ratio martingale that changes probabilities from the correctly
specified measure to the recovered measure.

In the power-utility example, write

$$
A = \exp(-\delta-\gamma g_c),
\qquad
s_{ij}=A\left(\frac{c_j}{c_i}\right)^{-\gamma}.
$$

Taking $\hat e_i=c_i^\gamma$, up to scale, gives

$$
[\mathbf{Q}\hat e]_i
= \sum_j A\left(\frac{c_j}{c_i}\right)^{-\gamma}p_{ij}c_j^\gamma
= A c_i^\gamma
= A\hat e_i,
$$

so $\exp(\hat\eta)=A$.

Consequently,

$$
\hat h_{ij}
= A^{-1}A\left(\frac{c_j}{c_i}\right)^{-\gamma}
  \frac{c_j^\gamma}{c_i^\gamma}
=1.
$$

```{code-cell} ipython3
matrices = [
    ("correctly specified P", P_true),
    ("one-period risk-neutral P_bar", P_bar),
    ("long-term risk-neutral P_hat", P_hat),
]

for label, mat in matrices:
    print(label)
    print(np.round(mat, 3))
    print()
```

```{code-cell} ipython3
H_power = np.divide(P_hat, P_true, out=np.ones_like(P_true), where=P_true > 0)
e_theory = c_levels**γ_power

print("Perron eigenfunction: numerical vs c^gamma")
for name, e_num, e_th in zip(state_names, e_hat / e_hat[1],
                             e_theory / e_theory[1]):
    print(f"{name:9s}: {e_num:.6f}  {e_th:.6f}")

print("\nlikelihood-ratio increment h_hat = P_hat / P")
print(np.round(H_power, 6))

print("\nconditional means under P")
print(np.round((P_true * H_power).sum(axis=1), 6))

print(f"\nmax |h_hat - 1| = "
      f"{np.max(np.abs(H_power[P_true > 0] - 1)):.2e}")
```

The output illustrates the difference between short-horizon and long-horizon risk
adjustments.

The one-period risk-neutral matrix $\bar{\mathbf{P}}$ is close to, but not the same as,
the correctly specified matrix $\mathbf{P}$.

It changes the transition probabilities because one-period Arrow prices include
one-period risk adjustments.

By contrast, the long-term risk-neutral matrix $\hat{\mathbf{P}}$ is exactly the same
as $\mathbf{P}$ in this example.

The diagnostic confirms why: the likelihood-ratio increment $\hat h_{ij}$ is one for
every transition, so the martingale $\hat H_t$ is identically one.

This is the condition under which Ross recovery returns the correctly specified
transition matrix: after the Perron eigenfunction removes the state-dependent part of
the SDF, no likelihood-ratio martingale remains.

## The martingale diagnostic

Let $(\hat \eta, \hat e)$ be the positive Perron pair of $\mathbf{Q}$:

$$
\mathbf{Q} \hat e = \exp(\hat\eta) \hat e.
$$

The associated long-term risk-neutral transition matrix is

$$
\hat p_{ij}
= \exp(-\hat\eta) q_{ij} \frac{\hat e_j}{\hat e_i}.
$$

Compare $\hat{\mathbf{P}}$ with the correctly specified transition matrix
$\mathbf{P}$ by defining

$$
\hat h_{ij} = \frac{\hat p_{ij}}{p_{ij}}.
$$

For a fixed current state $i$, the numbers $\hat h_{ij}$ average to one under the
correctly specified transition probabilities:

$$
\sum_j \hat h_{ij} p_{ij}=1.
$$

Thus $\hat h_{ij}$ is a one-period likelihood-ratio increment.

Multiplying these
increments over time gives a martingale.

The one-period SDF can be written as

$$
s_{ij}
= \exp(\hat\eta) \frac{\hat e_i}{\hat e_j} \hat h_{ij}.
$$

The Perron calculation therefore separates the SDF into:

| Part | Role |
|---|---|
| $\exp(\hat\eta)$ | deterministic long-run discounting |
| $\hat e_i / \hat e_j$ | state-dependent long-run term |
| $\hat h_{ij}$ | likelihood ratio that changes probabilities |

If $\hat h_{ij}=1$ for every feasible transition, then the recovered transition matrix
and the correctly specified transition matrix are the same.

This is the condition under which Ross recovery returns the correctly specified
transition matrix.

```{prf:proposition} Recovery diagnostic
:label: prop-misspecified-recovery-diagnostic

For a finite-state Markov model with correctly specified transition matrix $\mathbf{P}$ and Arrow
matrix $\mathbf{Q}$, Perron--Frobenius recovery returns the correctly specified transition matrix
if and only if $\hat h_{ij}=1$ for every transition with $p_{ij}>0$.

Equivalently, recovery returns the correctly specified transition matrix if and only if
the SDF has no nonconstant likelihood-ratio martingale:

$$
s_{ij}=\exp(\hat\eta)\frac{\hat e_i}{\hat e_j}.
$$
```

```{prf:proof}
Using $q_{ij}=s_{ij}p_{ij}$,

$$
\hat h_{ij}
=\frac{\hat p_{ij}}{p_{ij}}
=\exp(-\hat\eta)s_{ij}\frac{\hat e_j}{\hat e_i}.
$$

Thus $\hat{\mathbf{P}}=\mathbf{P}$ if and only if $\hat h_{ij}=1$ on every feasible
transition.

This condition is the same as saying that the SDF can be written in the displayed form
with no extra likelihood-ratio term.
```

The power-utility calculation above illustrates the proposition: the likelihood-ratio increment $\hat h_{ij}$ is a constant one.

## Recursive utility

The previous example worked because all risk adjustment in the SDF could be written as
a ratio of a function of today's state to a function of tomorrow's state.

The Perron eigenfunction removes exactly that kind of term.

Recursive utility usually adds something else: a continuation-value term that behaves
like a likelihood ratio.

For the unit-EIS Epstein--Zin case in {cite:t}`BorovickaHansenScheinkman2016`, with
$C_t=\exp(g_c t)c(X_t)$, write the translated continuation value as $V_t=g_c t+v(X_t)$,
and define

$$
v_i^*=\exp((1-\gamma)v_i).
$$

The SDF is

$$
s_{ij}
= \exp(-\delta-g_c) \frac{c_i}{c_j}
  \frac{v_j^*}{\sum_k p_{ik}v_k^*}.
$$

The denominator is the conditional expectation of $v_j^*$ given current state $i$, so
the last fraction has conditional mean one under $\mathbf{P}$.

It is therefore a likelihood-ratio increment.

When $v^*$ is not constant, that likelihood ratio varies across next-period states.

That variation is why recovery no longer returns the correct transition matrix.

The next cell solves the finite-state continuation-value equation and builds the SDF.

```{code-cell} ipython3
def solve_ez_unit_eis(P, c, δ, γ, g_c, tol=1e-12, max_iter=10_000):
    """Finite-state unit-EIS Epstein-Zin continuation values and SDF."""
    β = np.exp(-δ)
    log_c = np.log(c)
    n = len(c)
    flow = (1 - β) * log_c + β * g_c

    if abs(γ - 1) < 1e-10:
        v = linalg.solve(np.eye(n) - β * P, flow)
        v_star = np.ones(n)
        Pv_star = np.ones(n)
    else:
        v = log_c.copy()
        for _ in range(max_iter):
            v_star = np.exp((1 - γ) * v)
            Pv_star = P @ v_star
            v_new = flow + β / (1 - γ) * np.log(Pv_star)
            if np.max(np.abs(v_new - v)) < tol:
                v = v_new
                break
            v = v_new
        else:
            raise ValueError("Epstein-Zin fixed point did not converge.")

        v_star = np.exp((1 - γ) * v)
        Pv_star = P @ v_star

    S = (
        np.exp(-δ - g_c)
        * (c[:, None] / c[None, :])
        * (v_star[None, :] / Pv_star[:, None])
    )

    return v, v_star, S
```

At log utility, $v^*$ is constant and the martingale disappears.

As risk aversion rises, continuation values matter more.

The recovered probability measure then moves farther away from the correctly specified
probability measure.

To make the mechanism visible in a small three-state example, the figure below uses
the more dispersed consumption vector

$$
c=(0.85, 1.00, 1.15).
$$

The heatmap reports percentage deviations of the likelihood-ratio increment from one:
$100(\hat h_{ij}-1)$.

Positive entries are transitions that receive more probability under the recovered
measure than under the correctly specified probability measure.

The right panel reports the increase in the recovered recession probability, measured
in percentage points.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Recursive utility generates a nonconstant likelihood-ratio increment that distorts recovery.
    name: fig-mr-recursive-martingale
---
c_recursive = np.array([0.85, 1.00, 1.15])
γ_demo = 10.0
_, _, S_demo = solve_ez_unit_eis(P_true, c_recursive, δ, γ_demo, g_c)
Q_demo = S_demo * P_true
H_demo, _, _, P_hat_demo = martingale_increment(Q_demo, P_true)
H_dev = 100 * (H_demo - 1)

γ_grid = np.linspace(1, 15, 80)
rec_prob = []
for γ in γ_grid:
    _, _, S_g = solve_ez_unit_eis(P_true, c_recursive, δ, γ, g_c)
    Q_g = S_g * P_true
    _, _, _, P_hat_g = martingale_increment(Q_g, P_true)
    rec_prob.append(stationary_dist(P_hat_g)[0])
rec_prob = np.array(rec_prob)
rec_prob_gain = 100 * (rec_prob - π_true[0])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

bound = np.max(np.abs(H_dev))
im = axes[0].imshow(H_dev, cmap='coolwarm', vmin=-bound, vmax=bound, aspect='auto')
axes[0].set_xticks(range(3))
axes[0].set_yticks(range(3))
axes[0].set_xticklabels(state_names, rotation=20)
axes[0].set_yticklabels(state_names)
axes[0].set_xlabel('next state')
axes[0].set_ylabel(r'current state')
axes[0].set_title(r'likelihood-ratio distortion, $\gamma=10$')

for i in range(3):
    for j in range(3):
        axes[0].text(j, i, f"{H_dev[i, j]:.1f}",
                     ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04,
             label=r'$100(\hat h_{ij}-1)$')

axes[1].plot(γ_grid, rec_prob_gain, lw=2.5)
axes[1].axhline(0, ls='--', lw=1.5, color='0.5')
axes[1].set_xlabel(r"risk aversion $\gamma$")
axes[1].set_ylabel('increase in recession probability\n(percentage points)')
axes[1].set_title('recession probability distortion')
axes[1].set_ylim(0, rec_prob_gain.max() * 1.08)

plt.tight_layout()
plt.show()
```

## Permanent shocks

Recursive utility is one way to generate a nonconstant likelihood ratio.

Permanent shocks provide another.

Suppose consumption has a permanent multiplicative shock,

$$
\log C_{t+1}-\log C_t
= g + x(X_{t+1})-x(X_t) + \sigma \varepsilon_{t+1},
$$

where $\varepsilon_{t+1}$ is independent over time.

With power utility, the SDF contains

$$
\exp(-\delta-\gamma g)
\exp\{-\gamma[x(X_{t+1})-x(X_t)]\}
\exp(-\gamma\sigma\varepsilon_{t+1}).
$$

The middle term depends only on the current and next Markov states.

It is a ratio of
state functions, so the Perron eigenfunction can absorb it.

The permanent shock term depends on the new shock $\varepsilon_{t+1}$.

Because that
shock is not summarized by the finite Markov state in this calculation, it cannot be
removed by a state eigenfunction.

After dividing by its conditional mean, the shock term becomes a likelihood-ratio
increment:

$$
\frac{\exp(-\gamma\sigma\varepsilon_{t+1})}
     {E[\exp(-\gamma\sigma\varepsilon_{t+1})]}.
$$

Thus permanent consumption shocks can break belief recovery, even under ordinary power
utility.

## Long-run risk

We now use the Bansal--Yaron long-run risk model, in the calibration reported by
{cite:t}`BorovickaHansenScheinkman2016`.

The point is to see how different the recovered measure can look in a standard
macro-finance model.

The state vector $X_t=(X_{1t},X_{2t})'$ follows

$$
\begin{aligned}
dX_{1t}
&= [\mu_{11}(X_{1t}-\iota_1)+\mu_{12}(X_{2t}-\iota_2)]dt
   + \sqrt{X_{2t}}\sigma_1 dW_t, \\
dX_{2t}
&= \mu_{22}(X_{2t}-\iota_2)dt
   + \sqrt{X_{2t}}\sigma_2 dW_t .
\end{aligned}
$$

Here $X_1$ is predictable consumption growth and $X_2$ is stochastic volatility.

The representative agent has Epstein--Zin utility with unit elasticity of intertemporal
substitution.

The continuation value introduces a martingale $H^*$ into the SDF:

$$
d\log S_t = -\delta dt - d\log C_t + d\log H_t^*.
$$

The next cell sets the calibration.

```{code-cell} ipython3
lrr_params = dict(
    δ=0.002,
    γ=10.0,
    μ11=-0.021,
    μ12=0.0,
    μ22=-0.013,
    ι1=0.0,
    ι2=1.0,
    σ1=np.array([0.0, 0.00034, 0.0]),
    σ2=np.array([0.0, 0.0, -0.038]),
    β_c0=0.0015,
    β_c1=1.0,
    β_c2=0.0,
    α_c=np.array([0.0078, 0.0, 0.0]),
)
```

In this affine model, the continuation value and the Perron eigenfunction have a simple
exponential-affine form.

Think of the translated continuation value as having slopes $(v_1, v_2)$ with respect
to predictable growth and volatility.

The Perron eigenfunction has analogous slopes $(e_1, e_2)$.

Once those slopes are known, changing probabilities is a drift adjustment: the
instantaneous risk-neutral measure uses the SDF shock exposure, while the long-term
risk-neutral measure also includes the Perron eigenfunction shock exposure.

The next functions compute these pieces in that order.

This is why the code below is organized as value-function coefficients, Perron
coefficients, and then the drift of $X$ under the recovered and risk-neutral probability
measures.

```{code-cell} ipython3
def solve_value_function(p):
    """Slopes of the affine continuation value."""
    δ, γ = p["δ"], p["γ"]
    μ11, μ12, μ22 = p["μ11"], p["μ12"], p["μ22"]
    σ1, σ2 = p["σ1"], p["σ2"]
    β_c1, β_c2 = p["β_c1"], p["β_c2"]
    α_c = p["α_c"]

    v1 = β_c1 / (δ - μ11)

    # The volatility slope solves a scalar quadratic.
    A_vec = α_c + σ1 * v1
    B_vec = σ2

    a = 0.5 * (1 - γ) * np.dot(B_vec, B_vec)
    b = (μ22 - δ) + (1 - γ) * np.dot(A_vec, B_vec)
    c = β_c2 + μ12 * v1 + 0.5 * (1 - γ) * np.dot(A_vec, A_vec)

    disc = b**2 - 4 * a * c
    if disc < 0:
        raise ValueError("Value function does not exist for these parameters.")

    v2 = (-b - np.sqrt(disc)) / (2 * a)
    return v1, v2


def solve_pf_lrr(p, v1, v2):
    """Perron eigenfunction slopes and the SDF diffusion loading."""
    δ, γ = p["δ"], p["γ"]
    μ11, μ12, μ22 = p["μ11"], p["μ12"], p["μ22"]
    ι1, ι2 = p["ι1"], p["ι2"]
    σ1, σ2 = p["σ1"], p["σ2"]
    α_c = p["α_c"]
    β_c0, β_c1, β_c2 = p["β_c0"], p["β_c1"], p["β_c2"]

    α_h_star = (1 - γ) * (α_c + σ1 * v1 + σ2 * v2)
    α_s = -α_c + α_h_star

    β_s11 = -β_c1
    β_s12 = -β_c2 - 0.5 * np.dot(α_h_star, α_h_star)
    β_s0 = -δ - β_c0 - 0.5 * ι2 * np.dot(α_h_star, α_h_star)

    e1 = -β_s11 / μ11

    const = (β_s12 + 0.5 * np.dot(α_s, α_s)
             + e1 * (μ12 + np.dot(σ1, α_s))
             + 0.5 * e1**2 * np.dot(σ1, σ1))
    lin = μ22 + np.dot(σ2, α_s) + e1 * np.dot(σ1, σ2)
    quad = 0.5 * np.dot(σ2, σ2)

    disc = lin**2 - 4 * quad * const
    roots = [(-lin - np.sqrt(disc)) / (2 * quad),
             (-lin + np.sqrt(disc)) / (2 * quad)]

    candidates = []
    for e2 in roots:
        eta = (β_s0 - β_s11 * ι1 - β_s12 * ι2
               - e1 * (μ11 * ι1 + μ12 * ι2) - e2 * μ22 * ι2)
        candidates.append((eta, e2))

    # Choose the stable Perron root used for the long-term factorization.
    eta, e2 = min(candidates)
    return e1, e2, eta, α_s


def recovered_lrr_dynamics(p, e1, e2, α_s):
    """State dynamics under the long-term risk-neutral measure."""
    μ11, μ12, μ22 = p["μ11"], p["μ12"], p["μ22"]
    ι1, ι2 = p["ι1"], p["ι2"]
    σ1, σ2 = p["σ1"], p["σ2"]

    α_h = α_s + σ1 * e1 + σ2 * e2

    μ_hat_11 = μ11
    μ_hat_12 = μ12 + np.dot(σ1, α_h)
    μ_hat_22 = μ22 + np.dot(σ2, α_h)

    ι_hat_2 = (μ22 / μ_hat_22) * ι2
    ι_hat_1 = ι1 + (μ12 * ι2 - μ_hat_12 * ι_hat_2) / μ11

    return dict(
        μ11=μ_hat_11,
        μ12=μ_hat_12,
        μ22=μ_hat_22,
        ι1=ι_hat_1,
        ι2=ι_hat_2,
        σ1=σ1,
        σ2=σ2,
        α_h=α_h,
    )


def risk_neutral_lrr_dynamics(p, α_s):
    """State dynamics under the instantaneous risk-neutral measure."""
    μ11, μ12, μ22 = p["μ11"], p["μ12"], p["μ22"]
    ι1, ι2 = p["ι1"], p["ι2"]
    σ1, σ2 = p["σ1"], p["σ2"]

    μ_bar_11 = μ11
    μ_bar_12 = μ12 + np.dot(σ1, α_s)
    μ_bar_22 = μ22 + np.dot(σ2, α_s)

    ι_bar_2 = (μ22 / μ_bar_22) * ι2
    ι_bar_1 = ι1 + (μ12 * ι2 - μ_bar_12 * ι_bar_2) / μ11

    return dict(
        μ11=μ_bar_11,
        μ12=μ_bar_12,
        μ22=μ_bar_22,
        ι1=ι_bar_1,
        ι2=ι_bar_2,
        σ1=σ1,
        σ2=σ2,
    )
```

For the calibration used here, the recovered measure changes the long-run state
distribution.

It lowers the mean of expected growth and raises the mean of volatility.

```{code-cell} ipython3
v1, v2 = solve_value_function(lrr_params)
e1, e2, η_lrr, α_s = solve_pf_lrr(lrr_params, v1, v2)
dyn_hat = recovered_lrr_dynamics(lrr_params, e1, e2, α_s)
dyn_bar = risk_neutral_lrr_dynamics(lrr_params, α_s)

print(f"value slopes:       v1 = {v1:.4f}, v2 = {v2:.4f}")
print(f"Perron coefficients: e1 = {e1:.4f}, e2 = {e2:.4f}")
print(f"log eigenvalue:     eta = {η_lrr:.6f}  "
      f"(annualized {12 * η_lrr:.4f})")
print()
print("Long-run means under three measures")
print("measure        iota_1     iota_2     mu_12      mu_22")
print("---------   --------   --------   --------   --------")
print(f"actual      {lrr_params['ι1']:8.5f}   {lrr_params['ι2']:8.5f}"
      f"   {lrr_params['μ12']:8.5f}   {lrr_params['μ22']:8.5f}")
print(f"risk-neut.  {dyn_bar['ι1']:8.5f}   {dyn_bar['ι2']:8.5f}"
      f"   {dyn_bar['μ12']:8.5f}   {dyn_bar['μ22']:8.5f}")
print(f"long-term   {dyn_hat['ι1']:8.5f}   {dyn_hat['ι2']:8.5f}"
      f"   {dyn_hat['μ12']:8.5f}   {dyn_hat['μ22']:8.5f}")
```

These numbers show the mechanism clearly.

The positive value slope $v_1$ says that the continuation value is very sensitive to
predictable consumption growth.

The volatility slope $v_2$ is negative in this calibration, so higher volatility lowers
continuation value.

The Perron coefficient $e_1$ has the opposite sign: the long-term change of measure
loads negatively on predictable growth.

Thus the recovered measure tilts probability toward histories with lower expected
growth.

The positive $e_2$ works in the other direction for volatility, tilting probability
toward higher-volatility states.

The table translates those coefficients into state dynamics.

Relative to the correctly specified law, both risk-neutral measures lower the long-run
mean of predictable growth and raise the long-run mean of volatility.

The long-term risk-neutral measure moves further in that direction than the
instantaneous risk-neutral measure: $\iota_1$ falls from $0$ to about $-0.0027$, while
$\iota_2$ rises from $1$ to about $1.13$.

The small negative log eigenvalue means that the Perron discount factor is slightly
below one; with the usual yield sign convention, $-\eta$ is the corresponding long-run
discount rate.

### State probabilities

Figure 1 in {cite:t}`BorovickaHansenScheinkman2016` is about forecasting after
treating the recovered measure as beliefs.

It is the same message as the coefficient table above, but shown as a distribution
rather than as long-run means.

The table said that the recovered measure lowers the long-run mean of predictable
growth $X_1$ and raises the long-run mean of volatility $X_2$.

The figure shows the same distortion geometrically: probability mass moves down and to
the right.

The left panel uses the correctly specified probability measure $\mathbf{P}$.

The right panel uses the probability measure recovered from the Perron--Frobenius
calculation, $\hat{\mathbf{P}}$.

The main message is not just that the two densities differ.

The recovered measure puts more probability on bad long-run-risk states.

These are states with lower predictable growth $X_1$ and higher volatility $X_2$.

It also makes low growth and high volatility occur together more often.

The dashed contour adds the instantaneous risk-neutral distribution. In this calibration,
the risk-neutral and recovered stationary distributions are close to each other and both
are far from the correctly specified distribution.

This means that the martingale likelihood ratio is responsible for much of the risk
adjustment.

The plot below is drawn in three steps.

First, we simulate the state process under each set of drift parameters: the correctly
specified dynamics, the recovered long-term risk-neutral dynamics, and the instantaneous
risk-neutral dynamics.

Second, after discarding an initial burn-in, we estimate the stationary joint density of
$(X_2, X_1)$ with a two-dimensional kernel density estimator.

Third, we draw density contours on the same axes.

The horizontal line marks $X_1=0$ and the vertical line marks the correctly specified
mean of volatility, $X_2=\iota_2$.

The code uses the paper's calibration but keeps the simulation and KDE choices simple.

```{code-cell} ipython3
def simulate_lrr(dyn, T=180_000, seed=123):
    """Euler simulation of the LRR state process under one probability measure."""
    rng = np.random.default_rng(seed)
    X1 = np.zeros(T)
    X2 = np.full(T, dyn["ι2"])

    # Euler step with monthly time increment
    for t in range(1, T):
        X2_prev = max(X2[t-1], 1e-9)
        dW = rng.standard_normal(3)
        sqrt_X2 = np.sqrt(X2_prev)

        X1[t] = (
            X1[t-1]
            + dyn["μ11"] * (X1[t-1] - dyn["ι1"])
            + dyn["μ12"] * (X2_prev - dyn["ι2"])
            + sqrt_X2 * np.dot(dyn["σ1"], dW)
        )
        X2[t] = max(
            X2_prev
            + dyn["μ22"] * (X2_prev - dyn["ι2"])
            + sqrt_X2 * np.dot(dyn["σ2"], dW),
            1e-9,
        )

    burn = T // 5
    return X1[burn:], X2[burn:]


def kde2d_contour(ax, X1, X2, label, levels=7, fill=True,
                  linestyle='solid', outer_only=False):
    """Estimate the stationary density and draw its contours."""
    m = min(25_000, len(X1))
    idx = np.linspace(0, len(X1) - 1, m, dtype=int)
    x1 = X1[idx]
    x2 = X2[idx]

    kde = gaussian_kde(np.vstack([x2, x1]))
    x2_grid = np.linspace(0.6, 1.6, 140)
    x1_grid = np.linspace(-0.006, 0.006, 140)
    X2g, X1g = np.meshgrid(x2_grid, x1_grid)
    Z = kde(np.vstack([X2g.ravel(), X1g.ravel()])).reshape(X2g.shape)

    contour_levels = np.linspace(0.12 * Z.max(), 0.9 * Z.max(), levels)
    if outer_only:
        contour_levels = contour_levels[:1]

    if fill:
        fill_levels = np.r_[contour_levels, Z.max()]
        ax.contourf(X2g, X1g, Z, levels=fill_levels, cmap='Greys',
                    alpha=0.85)
        ax.contour(X2g, X1g, Z, levels=contour_levels, colors='0.55',
                   linewidths=0.4)
        ax.plot([], [], color='0.25', lw=1.5, label=label)
    else:
        ax.contour(X2g, X1g, Z, levels=contour_levels, colors='black',
                   linewidths=1.5, linestyles=linestyle)
        ax.plot([], [], color='black', lw=1.5, ls=linestyle, label=label)


dyn_true = dict(
    μ11=lrr_params["μ11"],
    μ12=lrr_params["μ12"],
    μ22=lrr_params["μ22"],
    ι1=lrr_params["ι1"],
    ι2=lrr_params["ι2"],
    σ1=lrr_params["σ1"],
    σ2=lrr_params["σ2"],
)

X1_P, X2_P = simulate_lrr(dyn_true, seed=1)
X1_H, X2_H = simulate_lrr(dyn_hat, seed=2)
X1_B, X2_B = simulate_lrr(dyn_bar, seed=3)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
kde2d_contour(axes[0], X1_P, X2_P, label=r'correctly specified $\mathbf{P}$')
kde2d_contour(axes[1], X1_H, X2_H,
              label=r'long-term risk-neutral $\hat{\mathbf{P}}$')
kde2d_contour(axes[1], X1_B, X2_B,
              label=r'risk-neutral $\bar{\mathbf{P}}$',
              fill=False, linestyle='--', outer_only=True)

for ax in axes:
    ax.axhline(0, lw=0.8, ls='--')
    ax.axvline(lrr_params["ι2"], lw=0.8, ls='--')
    ax.set_xlim(0.6, 1.6)
    ax.set_ylim(-0.006, 0.006)
    ax.set_xlabel(r"conditional volatility $X_2$")
    ax.legend(fontsize=9)

axes[0].set_ylabel(r"mean growth rate $X_1$")
plt.tight_layout()
plt.show()
```

The movement below the horizontal line means lower expected growth, while movement to
the right of the vertical line means higher volatility.

### Yield implications

Figure 2 in {cite:t}`BorovickaHansenScheinkman2016` asks how the probability
difference affects yields.

For a cash flow $G_t$, the yield compares a forecast of the payoff with its asset price:

$$
y_t[G](x)
= \frac{1}{t}\log E[G_t \mid X_0=x]
  - \frac{1}{t}\log E[S_tG_t \mid X_0=x].
$$

The first term is a forecast of the cash flow.

The second term is its price, written using the stochastic discount factor.

If an analyst treats $\hat{\mathbf{P}}$ as investors' beliefs, the forecast term changes
while the observed price is held fixed.

The left panel applies this comparison to a payoff equal to aggregate consumption at
maturity.

The recovered measure treats adverse long-run-risk states as more likely.

As a result, it removes much of the long-run risk compensation from aggregate
consumption cash flows.

The resulting consumption yields are lower than the yields computed with the correctly
specified probability measure.

The bond panel gives the comparison case: for a zero-coupon payoff, changing the payoff
forecast does not change the numerator. It isolates the maturity-matched discounting
against which the aggregate-consumption cash flow is compared.

The calculation below uses the affine formulas implied by the long-run risk model.

If a multiplicative functional $M$ has log drift affine in $X$ and diffusion proportional
to $\sqrt{X_2}$, then

$$
E[M_t \mid X_0=x]
= \exp\{\theta_0(t)+\theta_1(t)x_1+\theta_2(t)x_2\},
$$

where the coefficients solve Riccati equations.

The code below computes these affine expectations under the correctly specified
measure, recomputes only the consumption forecast under the recovered measure, and keeps
asset prices fixed.

It then plots median and interquartile yield bands across the same simulated initial
states.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: >-
      Yield implications of using recovered probabilities as beliefs. Dashed
      consumption-yield bands use recovered payoff forecasts with prices fixed; bond
      yields are unchanged because the zero-coupon payoff has no forecast term.
    name: fig-mr-lrr-figure-2
---
def affine_expectation_coeffs(dyn, β0, β1, β2, α, horizons):
    """Riccati coefficients for log E[M_t | X_0=x]."""
    μ11, μ12, μ22 = dyn["μ11"], dyn["μ12"], dyn["μ22"]
    ι1, ι2 = dyn["ι1"], dyn["ι2"]
    σ1, σ2 = dyn["σ1"], dyn["σ2"]

    def ode(_, θ):
        θ0, θ1, θ2 = θ
        θ0_dot = (β0 - β1 * ι1 - β2 * ι2
                  - θ1 * (μ11 * ι1 + μ12 * ι2)
                  - θ2 * μ22 * ι2)
        θ1_dot = β1 + μ11 * θ1
        θ2_dot = (β2 + μ12 * θ1 + μ22 * θ2
                  + 0.5 * np.dot(α, α)
                  + θ1 * np.dot(σ1, α)
                  + θ2 * np.dot(σ2, α)
                  + 0.5 * θ1**2 * np.dot(σ1, σ1)
                  + θ1 * θ2 * np.dot(σ1, σ2)
                  + 0.5 * θ2**2 * np.dot(σ2, σ2))
        return [θ0_dot, θ1_dot, θ2_dot]

    sol = solve_ivp(ode, (0, horizons[-1]), np.zeros(3),
                    t_eval=horizons, rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise ValueError("Riccati equation failed to solve")
    return sol.y.T


def log_expectation(θ, X1, X2):
    """Evaluate log E[M_t | X_0=x] on simulated states."""
    return θ[:, 0, None] + θ[:, 1, None] * X1[None, :] + θ[:, 2, None] * X2[None, :]


def yield_quantiles(log_num, log_den, horizons):
    """Quartiles of annualized yields across initial states."""
    yields = 12 * (log_num - log_den) / horizons[:, None]
    return np.quantile(yields, [0.25, 0.5, 0.75], axis=1)


def transform_functional(β0, β1, β2, α, dyn_old, dyn_new, α_h):
    """Rewrite a multiplicative functional after changing probabilities."""
    # The drift changes because the recovered likelihood ratio changes the
    # Brownian shock exposure used to forecast the cash flow.
    β_level = β0 - β1 * dyn_old["ι1"] - β2 * dyn_old["ι2"]
    β2_new = β2 + np.dot(α, α_h)
    β0_new = β_level + β1 * dyn_new["ι1"] + β2_new * dyn_new["ι2"]
    return β0_new, β1, β2_new, α


def sdf_coefficients(p, v1, v2):
    """SDF coefficients used in the affine expectation calculation."""
    δ, γ = p["δ"], p["γ"]
    α_c, σ1, σ2 = p["α_c"], p["σ1"], p["σ2"]

    α_h_star = (1 - γ) * (α_c + σ1 * v1 + σ2 * v2)
    α_s = -α_c + α_h_star

    β_s1 = -p["β_c1"]
    β_s2 = -p["β_c2"] - 0.5 * np.dot(α_h_star, α_h_star)
    β_s0 = -δ - p["β_c0"] - 0.5 * p["ι2"] * np.dot(α_h_star, α_h_star)

    return β_s0, β_s1, β_s2, α_s


quarters = np.arange(1, 101)
horizons = 3 * quarters

β_c0, β_c1, β_c2 = (lrr_params["β_c0"],
                    lrr_params["β_c1"],
                    lrr_params["β_c2"])
α_c = lrr_params["α_c"]

β_s0, β_s1, β_s2, α_s = sdf_coefficients(lrr_params, v1, v2)

# Numerators and denominators for yields under the correctly specified measure
θ_C_P = affine_expectation_coeffs(dyn_true, β_c0, β_c1, β_c2, α_c, horizons)
θ_S_P = affine_expectation_coeffs(dyn_true, β_s0, β_s1, β_s2, α_s, horizons)
θ_SC_P = affine_expectation_coeffs(
    dyn_true, β_s0 + β_c0, β_s1 + β_c1, β_s2 + β_c2,
    α_s + α_c, horizons
)

# Recovered-belief numerator for the aggregate-consumption payoff
β_Ch0, β_Ch1, β_Ch2, α_Ch = transform_functional(
    β_c0, β_c1, β_c2, α_c, dyn_true, dyn_hat, dyn_hat["α_h"]
)
θ_C_H = affine_expectation_coeffs(dyn_hat, β_Ch0, β_Ch1, β_Ch2,
                                  α_Ch, horizons)

log_C_P = log_expectation(θ_C_P, X1_P, X2_P)
log_C_H = log_expectation(θ_C_H, X1_P, X2_P)
log_S_P = log_expectation(θ_S_P, X1_P, X2_P)
log_SC_P = log_expectation(θ_SC_P, X1_P, X2_P)

qC_P = yield_quantiles(log_C_P, log_SC_P, horizons)
qC_H = yield_quantiles(log_C_H, log_SC_P, horizons)
qB_P = yield_quantiles(np.zeros_like(log_S_P), log_S_P, horizons)
# A zero-coupon payoff has the same numerator, log E[1] = 0, under either belief.
qB_H = qB_P.copy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

def plot_yield_band(ax, x, q, color, label, linestyle='solid',
                    alpha=0.35):
    """Plot quartile band and quartile lines."""
    ax.fill_between(x, q[0], q[2], color=color, alpha=alpha, linewidth=0)
    ax.plot(x, q[1], color=color, lw=2.4, ls=linestyle, label=label)
    ax.plot(x, q[0], color=color, lw=1.3, ls=linestyle)
    ax.plot(x, q[2], color=color, lw=1.3, ls=linestyle)


plot_yield_band(axes[0], quarters, qC_P, color='0.2',
                label='correctly specified measure', alpha=0.45)
plot_yield_band(axes[0], quarters, qC_H, color='0.65',
                label='recovered measure', linestyle='--', alpha=0.35)
plot_yield_band(axes[1], quarters, qB_P, color='0.2',
                label='correctly specified measure', alpha=0.45)
plot_yield_band(axes[1], quarters, qB_H, color='0.65',
                label='recovered measure', linestyle='--', alpha=0.25)

axes[0].set_xlabel('maturity (quarters)')
axes[0].set_ylabel('consumption yield to maturity')
axes[1].set_xlabel('maturity (quarters)')
axes[1].set_ylabel('bond yield to maturity')

axes[0].legend(fontsize=9)

plt.tight_layout()
plt.show()
```

The left panel is the key one: recovered beliefs put more mass on low-growth,
high-volatility states, so they forecast lower consumption and imply lower consumption
yields when prices are held fixed.

The bond panel is a check.

Since $\log E[1]=0$ under any measure, the solid and dashed
bond-yield bands coincide.

## Additional state vector

{cite:t}`BorovickaHansenScheinkman2016` then asks whether the recovery
problem can be fixed by enlarging the state vector.

So far, the Perron eigenfunction has depended only on the Markov state $X_t$.

But many models also contain a growing component $Y_t$, such as log consumption, with
increments driven by the same shocks:

$$
X_{t+1}=\phi_x(X_t,W_{t+1}),
\qquad
Y_{t+1}-Y_t=\phi_y(X_t,W_{t+1}).
$$

If we allow the eigenfunction to depend on both $(X_t,Y_t)$, then a natural candidate is

$$
\varepsilon(x,y)=\exp(\zeta \cdot y)e_\zeta(x).
$$

This form is natural because $Y$ enters through increments.

Along a path,

$$
\exp(\zeta \cdot Y_{t+1})
= \exp(\zeta \cdot Y_t)
  \exp\{\zeta \cdot (Y_{t+1}-Y_t)\}.
$$

Since $Y_{t+1}-Y_t$ is a function of $(X_t,W_{t+1})$, the ratio
$\exp(\zeta \cdot Y_{t+1})/\exp(\zeta \cdot Y_t)$ is a one-period
multiplicative shock.

Thus multiplying the old eigenfunction by $\exp(\zeta \cdot y)$ does not destroy the
Perron structure; it simply changes the one-period pricing operator by the extra factor
$\exp\{\zeta \cdot (Y_{t+1}-Y_t)\}$.

For each choice of $\zeta$, the remaining $x$-dependent part solves a different Perron
problem:

$$
E\left[
    \frac{S_{t+1}}{S_t}
    \exp\{\zeta \cdot (Y_{t+1}-Y_t)\}
    e_\zeta(X_{t+1})
    \mid X_t=x
\right]
=\exp(\eta_\zeta)e_\zeta(x).
$$

Changing $\zeta$ changes how much long-run growth risk is loaded into the eigenfunction.

Thus adding $Y_t$ can make the subjective probability law one possible solution, but it
also creates a family of possible solutions.

The extra state variable therefore does not remove the identification problem; it
usually makes the selection problem more explicit.

The paper also points out a related practical issue.

Highly persistent stationary processes can be hard to distinguish from processes with
stationary increments.

A stationary approximation may have a unique Perron solution for each finite persistence
level, but as persistence becomes extreme, the limiting problem can have many
near-solutions.

Numerically, this means recovery can become fragile exactly in the cases where a
stationary model is being used to approximate stochastic growth.

There is, however, a structured way forward.

If the analyst supplies a reference multiplicative functional $Y^r$ that is known to
contain the martingale component of the SDF, then one can restrict the enlarged
eigenfunction to the form

$$
(Y^r)^{-1}e(x).
$$

This restriction chooses which long-run martingale component is allowed into the
eigenfunction.

With this extra structure, Arrow prices can again reveal subjective probabilities.

But the key input is external: the long-run martingale component has been supplied by
the analyst, not recovered from Arrow prices alone.

## Lessons

The Perron--Frobenius calculation remains useful under misspecification, but it no
longer solves the belief-recovery problem by itself.

It delivers a probability measure that may include long-horizon risk premia.

That measure equals investors' beliefs only when the likelihood-ratio martingale is
constant.

Recursive utility, permanent shocks, and long-run risk models give this martingale an
economically important role, so it should not be overlooked when assessing the
implications of transition independence for belief recovery.

## Exercises

```{exercise}
:label: ex_misspecified_recovery_diagnostic

**A two-state diagnostic.**

Let

$$
\mathbf{P} =
\begin{pmatrix}
0.8 & 0.2 \\
0.4 & 0.6
\end{pmatrix},
\qquad
\mathbf{Q} =
\begin{pmatrix}
0.72 & 0.15 \\
0.36 & 0.42
\end{pmatrix}.
$$

1. Compute the one-period risk-neutral transition matrix $\bar{\mathbf{P}}$.
2. Compute the recovered transition matrix $\hat{\mathbf{P}}$.
3. Compute $\hat h_{ij}=\hat p_{ij}/p_{ij}$ and decide whether recovery returns the
   correctly specified transition matrix.
```

```{solution-start} ex_misspecified_recovery_diagnostic
:class: dropdown
```

Here is one solution:

```{code-cell} ipython3
P2 = np.array([[0.8, 0.2],
               [0.4, 0.6]])
Q2 = np.array([[0.72, 0.15],
               [0.36, 0.42]])

Pbar2, qb2 = risk_neutral_probs(Q2)
H2, eta2, e2, Phat2 = martingale_increment(Q2, P2)

print("One-period risk-neutral transition matrix P_bar")
print(np.round(Pbar2, 4))
print("\nRecovered transition matrix P_hat")
print(np.round(Phat2, 4))
print("\nMartingale increment h_hat")
print(np.round(H2, 4))
print("\nRecovery returns P:", np.allclose(H2[P2 > 0], 1))
```

```{solution-end}
```

```{exercise}
:label: ex_power_utility_success

**Power utility benchmark.**

For trend-stationary consumption and power utility,

$$
s_{ij}=A\left(\frac{c_j}{c_i}\right)^{-\gamma}.
$$

Show that $\hat e_i=c_i^\gamma$ is the Perron eigenvector and that
$\hat{\mathbf{P}}=\mathbf{P}$.

Then verify the result numerically using the three-state baseline in the lecture.
```

```{solution-start} ex_power_utility_success
:class: dropdown
```

The analytical check is:

$$
[\mathbf{Q}\hat e]_i
=\sum_j A\left(\frac{c_j}{c_i}\right)^{-\gamma}p_{ij}c_j^\gamma
=A c_i^\gamma
=A\hat e_i.
$$

Thus $\exp(\hat\eta)=A$ and

$$
\hat p_{ij}
=\frac{1}{A}q_{ij}\frac{\hat e_j}{\hat e_i}
=p_{ij}.
$$

Below is the numerical check.

```{code-cell} ipython3
H_power, _, e_power, P_hat_power = martingale_increment(Q_power, P_true)
e_theory = c_levels**γ_power
e_theory = e_theory / e_theory.sum()

print("Perron eigenvector")
print(np.round(e_power, 6))
print("\nNormalized c^gamma")
print(np.round(e_theory, 6))
print("\nmax |P_hat - P|:",
      np.max(np.abs(P_hat_power - P_true)))
print("max |h_hat - 1|:",
      np.max(np.abs(H_power[P_true > 0] - 1)))
```

```{solution-end}
```

```{exercise}
:label: ex_recursive_utility_distortion

**Recursive utility and risk aversion.**

Using the finite-state Epstein--Zin example with
$c=(0.85, 1.00, 1.15)$, compute the stationary distribution of
$\hat{\mathbf{P}}$ for $\gamma \in \{1, 5, 10, 15\}$.

Which state receives the largest increase in stationary probability as $\gamma$ rises?
```

```{solution-start} ex_recursive_utility_distortion
:class: dropdown
```

Here is one solution:

```{code-cell} ipython3
for γ in [1, 5, 10, 15]:
    _, _, S_g = solve_ez_unit_eis(P_true, c_recursive, δ, γ, g_c)
    Q_g = S_g * P_true
    _, _, _, P_hat_g = martingale_increment(Q_g, P_true)
    π_g = stationary_dist(P_hat_g)
    print(f"gamma={γ:2.0f}: {np.round(π_g, 4)}")

print("\nCorrectly specified:", np.round(π_true, 4))
```

The recession state receives the largest increase.

```{solution-end}
```
