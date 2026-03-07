---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(affine_risk_prices)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Affine Models of Asset Prices

## Overview

This lecture describes a class of **affine** or **exponential quadratic** models of the
stochastic discount factor that have become widely used in empirical finance.

These models are presented in chapter 15 of {cite:t}`Ljungqvist2012`.

The models discussed here take a different approach from the time-separable CRRA
stochastic discount factor of {cite:t}`hansen1983stochastic`, studied in our
{doc}`companion lecture <hansen_singleton_1983>`.

The CRRA stochastic discount factor is

$$
m_{t+1} = \exp\left(-r_t - \frac{1}{2}\sigma_c^2 \gamma^2 - \gamma\sigma_c\varepsilon_{t+1}\right)
$$

where $r_t = \rho + \gamma\mu - \frac{1}{2}\sigma_c^2\gamma^2$.

This model asserts that exposure to the random part of aggregate consumption growth,
$\sigma_c\varepsilon_{t+1}$, is the *only* priced risk — the sole source of discrepancies
among expected returns across assets.

Empirical difficulties with this specification (the equity premium puzzle, the
risk-free rate puzzle, and the Hansen-Jagannathan bounds discussed in
{doc}`Doubts or Variability? <doubts_or_variability>`) motivate the alternative approach
described in this lecture.

The **affine model** maintains $\mathbb{E}(m_{t+1}R_{j,t+1}) = 1$ but *divorces* the
stochastic discount factor from consumption risk.

Instead, it

* specifies an analytically tractable stochastic process for $m_{t+1}$, and
* uses overidentifying restrictions from $\mathbb{E}(m_{t+1}R_{j,t+1}) = 1$ applied to $N$
  assets to let the data reveal risks and their prices.

Key applications we study include:

1. **Pricing risky assets** — how risk prices and exposures determine excess returns.
1. **Affine term structure models** — bond yields as affine functions of a state vector
   ({cite:t}`AngPiazzesi2003`).
1. **Risk-neutral probabilities** — a change-of-measure representation of the pricing equation.
1. **Distorted beliefs** — reinterpreting risk price estimates when agents hold systematically
   biased forecasts ({cite:t}`piazzesi2015trend`); see also {doc}`Risk Aversion or Mistaken Beliefs? <risk_aversion_or_mistaken_beliefs>`.

We start with some standard imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from numpy.linalg import eigvals
```

## The model

### State dynamics and short rate

The model has two components.

**Component 1** is a vector autoregression that describes the state of the economy
and the evolution of the short rate:

```{math}
:label: eq_var

z_{t+1} = \mu + \phi z_t + C\varepsilon_{t+1}
```

```{math}
:label: eq_shortrate

r_t = \delta_0 + \delta_1^\top z_t
```

Here

* $\phi$ is a stable $m \times m$ matrix,
* $C$ is an $m \times m$ matrix,
* $\varepsilon_{t+1} \sim \mathcal{N}(0, I)$ is an i.i.d. $m \times 1$ random vector,
* $z_t$ is an $m \times 1$ state vector.

Equation {eq}`eq_shortrate` says that the **short rate** $r_t$ — the net yield on a
one-period risk-free claim — is an affine function of the state $z_t$.

**Component 2** is a vector of **risk prices** $\lambda_t$ and an associated stochastic
discount factor $m_{t+1}$:

```{math}
:label: eq_riskprices

\lambda_t = \lambda_0 + \lambda_z z_t
```

```{math}
:label: eq_sdf

\log(m_{t+1}) = -r_t - \frac{1}{2}\lambda_t^\top\lambda_t - \lambda_t^\top\varepsilon_{t+1}
```

Here $\lambda_0$ is $m \times 1$ and $\lambda_z$ is $m \times m$.

The entries of $\lambda_t$ that multiply corresponding entries of the risks
$\varepsilon_{t+1}$ are called **risk prices** because they determine how exposures
to each risk component affect expected returns (as we show below).

Because $\lambda_t$ is affine in $z_t$, the stochastic discount factor $m_{t+1}$ is
**exponential quadratic** in the state $z_t$.

### Properties of the SDF

Since $\lambda_t^\top\varepsilon_{t+1}$ is conditionally normal, it follows that

$$
\mathbb{E}_t(m_{t+1}) = \exp(-r_t)
$$

and   

$$
\text{std}_t(m_{t+1}) \approx |\lambda_t|.
$$

The first equation confirms that $r_t$ is the net yield on a risk-free one-period bond.

That is why $r_t$ is called **the short rate** in the exponential quadratic literature.

The second equation says that the conditional standard deviation of the SDF
is approximately the magnitude of the vector of risk prices — a measure of overall
**market price of risk**.

## Pricing risky assets

### Lognormal returns

Consider a risky asset $j$ whose gross return has a lognormal conditional distribution:

```{math}
:label: eq_return

R_{j,t+1} = \exp\left(\nu_t(j) - \frac{1}{2}\alpha_t(j)^\top\alpha_t(j) + \alpha_t(j)^\top\varepsilon_{t+1}\right)
```

where the **exposure vector**

```{math}
:label: eq_exposure

\alpha_t(j) = \alpha_0(j) + \alpha_z(j)\, z_t
```

Here $\alpha_0(j)$ is $m \times 1$ and $\alpha_z(j)$ is $m \times m$.

The components of $\alpha_t(j)$ express the **exposures** of $\log R_{j,t+1}$ to
corresponding components of the risk vector $\varepsilon_{t+1}$.

The specification {eq}`eq_return` implies $\mathbb{E}_t R_{j,t+1} = \exp(\nu_t(j))$,
so $\nu_t(j)$ is the expected net log return.

### Expected excess returns

Applying the pricing equation $\mathbb{E}_t(m_{t+1}R_{j,t+1}) = 1$ together with the
formula for the mean of a lognormal random variable gives

```{math}
:label: eq_excess

\nu_t(j) = r_t + \alpha_t(j)^\top\lambda_t
```

This is a central result.

It says:

> The expected net return on asset $j$ equals the short rate plus the inner product
> of the asset's exposure vector $\alpha_t(j)$ with the risk price vector $\lambda_t$.

Each component of $\lambda_t$ prices the corresponding component of $\varepsilon_{t+1}$.

An asset that loads heavily on a risk component with a large risk price earns a
correspondingly high expected return.

## Affine term structure of yields

One of the most important applications is the **affine term structure model** studied
by {cite:t}`AngPiazzesi2003`.

### Bond prices

Let $p_t(n)$ be the price at time $t$ of a risk-free pure discount bond maturing at
$t + n$ (paying one unit of consumption).

The one-period gross return on holding an $(n+1)$-period bond from $t$ to $t+1$ is

$$
R_{t+1} = \frac{p_{t+1}(n)}{p_t(n+1)}
$$

The pricing equation $\mathbb{E}_t(m_{t+1}R_{t+1}) = 1$ implies

```{math}
:label: eq_bondrecur

p_t(n+1) = \mathbb{E}_t\bigl(m_{t+1}\,p_{t+1}(n)\bigr)
```

with the initial condition

$$
p_t(1) = \mathbb{E}_t(m_{t+1}) = \exp(-r_t) = \exp(-\delta_0 - \delta_1^\top z_t).
$$

### Exponential affine prices

The recursion {eq}`eq_bondrecur` has an **exponential affine** solution:

```{math}
:label: eq_bondprice

p_t(n) = \exp\!\bigl(\bar A_n + \bar B_n^\top z_t\bigr)
```

where the scalar $\bar A_n$ and the $m \times 1$ vector $\bar B_n$ satisfy the
**Riccati difference equations**

```{math}
:label: eq_riccati_A

\bar A_{n+1} = \bar A_n + \bar B_n^\top(\mu - C\lambda_0) + \frac{1}{2}\bar B_n^\top CC^\top\bar B_n - \delta_0
```

```{math}
:label: eq_riccati_B

\bar B_{n+1}^\top = \bar B_n^\top(\phi - C\lambda_z) - \delta_1^\top
```

with initial conditions $\bar A_1 = -\delta_0$ and $\bar B_1 = -\delta_1$.

### Yields

The **yield to maturity** on an $n$-period bond is

$$
y_t(n) = -\frac{\log p_t(n)}{n}
$$

Substituting {eq}`eq_bondprice` gives

```{math}
:label: eq_yield

y_t(n) = A_n + B_n^\top z_t
```

where $A_n = -\bar A_n / n$ and $B_n = -\bar B_n / n$.

*Yields are affine functions of the state vector $z_t$.*

This is the defining property of affine term structure models.

## Python implementation

We now implement the affine term structure model and compute bond prices, yields,
and risk premiums numerically.

```{code-cell} ipython3
AffineModel = namedtuple('AffineModel',
    ('μ', 'φ', 'C', 'δ_0', 'δ_1', 'λ_0', 'λ_z', 'm', 'φ_rn', 'μ_rn'))

def create_affine_model(μ, φ, C, δ_0, δ_1, λ_0, λ_z):
    """Create an affine term structure model."""
    μ = np.asarray(μ, float)
    φ = np.asarray(φ, float)
    C = np.asarray(C, float)
    δ_1 = np.asarray(δ_1, float)
    λ_0, λ_z = np.asarray(λ_0, float), np.asarray(λ_z, float)
    return AffineModel(μ=μ, φ=φ, C=C, δ_0=float(δ_0), δ_1=δ_1,
                       λ_0=λ_0, λ_z=λ_z, m=len(μ),
                       φ_rn=φ - C @ λ_z, μ_rn=μ - C @ λ_0)

def bond_coefficients(model, n_max):
    """Compute (A_bar_n, B_bar_n) for n = 1, ..., n_max."""
    A_bar = np.zeros(n_max + 1)
    B_bar = np.zeros((n_max + 1, model.m))
    A_bar[1], B_bar[1] = -model.δ_0, -model.δ_1
    CC = model.C @ model.C.T
    for n in range(1, n_max):
        Bn = B_bar[n]
        A_bar[n + 1] = (A_bar[n] + Bn @ model.μ_rn
                       + 0.5 * Bn @ CC @ Bn
                       - model.δ_0)
        B_bar[n + 1] = model.φ_rn.T @ Bn - model.δ_1
    return A_bar, B_bar

def compute_yields(model, z, n_max):
    """Compute yield curve y_t(n) for n = 1, ..., n_max."""
    A_bar, B_bar = bond_coefficients(model, n_max)
    ns = np.arange(1, n_max + 1)
    return np.array([(-A_bar[n] - B_bar[n] @ z) / n for n in ns])

def bond_prices(model, z, n_max):
    """Compute bond prices p_t(n) for n = 1, ..., n_max."""
    A_bar, B_bar = bond_coefficients(model, n_max)
    return np.array([np.exp(A_bar[n] + B_bar[n] @ z)
                     for n in range(1, n_max + 1)])

def simulate(model, z0, T, rng=None):
    """Simulate the state process for T periods."""
    if rng is None:
        rng = np.random.default_rng(42)
    Z = np.zeros((T + 1, model.m))
    Z[0] = z0
    for t in range(T):
        ε = rng.standard_normal(model.m)
        Z[t + 1] = model.μ + model.φ @ Z[t] + model.C @ ε
    return Z

def short_rate(model, z):
    """Compute r_t = δ_0 + δ_1^⊤ z_t."""
    return model.δ_0 + model.δ_1 @ z

def risk_prices(model, z):
    """Compute λ_t = λ_0 + λ_z z_t."""
    return model.λ_0 + model.λ_z @ z
```

### A one-factor Gaussian example

To build intuition, we start with a single-factor ($m=1$) Gaussian model.

```{code-cell} ipython3
# One-factor Gaussian model (quarterly)
μ      = np.array([0.0])
φ      = np.array([[0.95]])
C      = np.array([[1.0]])
δ_0    = 0.01                  # 1%/quarter ≈ 4% p.a.
δ_1    = np.array([0.001])
λ_0    = np.array([0.05])
λ_z    = np.array([[-0.01]])   # countercyclical

model_1f = create_affine_model(μ, φ, C, δ_0, δ_1, λ_0, λ_z)

φ_Q = model_1f.φ_rn[0, 0]
half_life = np.log(2) / (-np.log(φ[0, 0]))
σ_z = 1.0 / np.sqrt(1 - φ[0, 0]**2)
print(f"Physical AR(1):      φ   = {φ[0,0]:.3f}"
      f"  (half-life {half_life:.1f} quarters)")
print(f"Risk-neutral AR(1):  φ^Q = {φ_Q:.3f}  "
      f"({'stable' if abs(φ_Q) < 1 else 'UNSTABLE'})")
print(f"Unconditional std of z:  σ_z = {σ_z:.2f}")
r_mean = short_rate(model_1f, np.array([0.0])) * 4 * 100
print(f"Mean short rate = {r_mean:.1f}% p.a.")
print(f"Short rate range (±2σ): [{(δ_0-δ_1[0]*2*σ_z)*4*100:.1f}%, "
      f"{(δ_0+δ_1[0]*2*σ_z)*4*100:.1f}%] p.a.")
```

### Yield curve shapes

We compute yield curves across a range of short-rate states $z_t$.

```{code-cell} ipython3
n_max_1f = 60
maturities_1f = np.arange(1, n_max_1f + 1)

z_low  = np.array([-5.0])
z_mid  = np.array([0.0])
z_high = np.array([5.0])

fig, ax = plt.subplots(figsize=(9, 5.5))

r_low = short_rate(model_1f, z_low) * 4 * 100
r_mid = short_rate(model_1f, z_mid) * 4 * 100
r_high = short_rate(model_1f, z_high) * 4 * 100

for z, label, color in [
    (z_low,  f"Low state  (r₁ = {r_low:.1f}%)",
     "steelblue"),
    (z_mid,  f"Median state (r₁ = {r_mid:.1f}%)",
     "seagreen"),
    (z_high, f"High state (r₁ = {r_high:.1f}%)",
     "firebrick"),
]:
    y = compute_yields(model_1f, z, n_max_1f) * 4 * 100
    ax.plot(maturities_1f, y, color=color, lw=2.2, label=label)
    ax.plot(1, y[0], 'o', color=color, ms=7, zorder=5)

r_bar = short_rate(model_1f, np.array([0.0])) * 4 * 100
ax.axhline(r_bar, color='grey', ls=':', lw=1.2, alpha=0.7,
           label=f"Mean short rate ({r_bar:.1f}%)")

ax.set_xlabel("Maturity (quarters)")
ax.set_ylabel("Yield (% per annum)")
ax.set_title("Yield Curves — One-Factor Affine Model")
ax.legend(fontsize=10, loc='best')
ax.set_xlim(1, n_max_1f)

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
year_ticks = [4, 8, 12, 20, 28, 40, 60]
ax2.set_xticks(year_ticks)
ax2.set_xticklabels([f"{t/4:.0f}y" for t in year_ticks])
ax2.set_xlabel("Maturity (years)")

plt.tight_layout()
plt.show()
```

The model generates upward-sloping, flat, and inverted yield curves as the short
rate moves across states — a key qualitative feature of observed bond markets.

### Short rate dynamics

```{code-cell} ipython3
T = 200
Z = simulate(model_1f, np.array([0.0]), T)
short_rates = np.array([short_rate(model_1f, Z[t]) * 4 * 100
                        for t in range(T + 1)])
r_bar_pct = short_rate(model_1f, np.array([0.0])) * 4 * 100

fig, ax = plt.subplots(figsize=(10, 4))
quarters = np.arange(T + 1)
ax.plot(quarters, short_rates, color="steelblue", lw=1.3)
ax.axhline(r_bar_pct, color="red", ls="--", lw=1.3,
           label=f"Unconditional mean ({r_bar_pct:.1f}%)")
ax.fill_between(quarters, short_rates, r_bar_pct,
                alpha=0.08, color="steelblue")
ax.set_xlabel("Quarter")
ax.set_ylabel("Short rate (% p.a.)")
ax.set_title("Simulated Short Rate — One-Factor Model (50 years)")
ax.set_xlim(0, T)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()
```

### A two-factor model

To match richer yield-curve dynamics, practitioners routinely use $m \geq 2$ factors.

We now introduce a two-factor specification in which the factors
can be interpreted as a **level** component and a **slope** component.

```{code-cell} ipython3
# Two-factor model: z = [level, slope]
μ_2  = np.array([0.0,  0.0])
φ_2  = np.array([[0.97, -0.03],
                  [0.00,  0.90]])
C_2  = np.eye(2)
δ_0_2 = 0.01
δ_1_2 = np.array([0.002, 0.001])
λ_0_2 = np.array([0.01,  0.005])
λ_z_2 = np.array([[-0.005, 0.0],
                   [ 0.0, -0.003]])

model_2f = create_affine_model(μ_2, φ_2, C_2, δ_0_2, δ_1_2, λ_0_2, λ_z_2)

print("Physical measure VAR:")
print(f"  φ =\n{φ_2}")
print(f"  eigenvalues of φ: {eigvals(φ_2).real.round(4)}")
print()
print("Risk-neutral measure VAR:")
print(f"  φ^Q = φ - Cλ_z =\n{model_2f.φ_rn.round(4)}")
eigs_Q = eigvals(model_2f.φ_rn).real
stable = all(abs(e) < 1 for e in eigs_Q)
status = "stable" if stable else "UNSTABLE"
print(f"  eigenvalues of φ^Q: {eigs_Q.round(4)}"
      f"  ({status})")
print()
print("Risk prices make Q dynamics more persistent than P dynamics.")
```

```{code-cell} ipython3
n_max_2f = 60
maturities_2f = np.arange(1, n_max_2f + 1)

states = {
    "Normal":              np.array([0.0,   0.0]),
    "Low short rate":      np.array([-4.0,  3.0]),
    "High short rate":     np.array([4.0,  -3.0]),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

colors_2f = ["seagreen", "steelblue", "firebrick"]
for (label, z), color in zip(states.items(), colors_2f):
    r_now = short_rate(model_2f, z) * 4 * 100
    y = compute_yields(model_2f, z, n_max_2f) * 4 * 100
    ax1.plot(maturities_2f, y, lw=2.2, color=color,
             label=f"{label} (r₁ = {r_now:.1f}%)")
    ax1.plot(1, y[0], 'o', color=color, ms=7, zorder=5)

ax1.annotate("Curves converge as\nmean reversion dominates",
             xy=(50, 3.8), fontsize=9, color="gray", ha='center',
             style='italic')
ax1.set_xlabel("Maturity (quarters)")
ax1.set_ylabel("Yield (% p.a.)")
ax1.set_title("Yield Curves — Two-Factor Model")
ax1.legend(fontsize=10)
ax1.set_xlim(1, n_max_2f)

A_bar, B_bar = bond_coefficients(model_2f, n_max_2f)
ns = np.arange(1, n_max_2f + 1)
B_n = np.array([-B_bar[n] / n for n in ns])

ax2.plot(ns, B_n[:, 0], lw=2.2, color="purple",
         label=r"Level loading $B_{n,1}$")
ax2.plot(ns, B_n[:, 1], lw=2.2, color="orange",
         label=r"Slope loading $B_{n,2}$")
ax2.axhline(0, color='black', lw=0.6)
ax2.set_xlabel("Maturity (quarters)")
ax2.set_ylabel(r"Yield loading $B_{n,k}$")
ax2.set_title("Factor Loadings on Yields")
ax2.legend(fontsize=11)
ax2.set_xlim(1, n_max_2f)
ax2.annotate("Level factor stays\nimportant at long maturities",
             xy=(45, B_n[44, 0]), fontsize=9, color="purple",
             ha='center', va='bottom')

for ax in (ax1, ax2):
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    year_ticks = [4, 12, 20, 40, 60]
    ax_top.set_xticks(year_ticks)
    ax_top.set_xticklabels([f"{t/4:.0f}y" for t in year_ticks])

plt.tight_layout()
plt.show()
```

## Risk premiums

A key object in the affine term structure model is the **term premium** — the extra
expected return on a long-term bond relative to rolling over short-term bonds.

For an $(n+1)$-period bond held for one period, the excess log return is
approximately

$$
\mathbb{E}_t\left[\log R_{t+1}^{(n+1)}\right] - r_t \;=\; -\bar B_n^\top C \lambda_t
$$

That is, the term premium equals (minus) the product of the bond's exposure to
the shocks $(-\bar B_n^\top C)$ with the risk prices $\lambda_t$.

```{code-cell} ipython3
def term_premiums(model, z, n_max):
    """Approximate term premiums for maturities 1 to n_max."""
    A_bar, B_bar = bond_coefficients(model, n_max + 1)
    λ_t = risk_prices(model, z)
    return np.array([-B_bar[n] @ model.C @ λ_t
                     for n in range(1, n_max + 1)])

n_max_tp = 60
maturities_tp = np.arange(1, n_max_tp + 1)

z_states_tp = {
    "Low rate (z₁ < 0)":  np.array([-3.0, 2.0]),
    "High rate (z₁ > 0)": np.array([3.0, -2.0]),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

tp_colors = ["steelblue", "firebrick"]
for (label, z), color in zip(z_states_tp.items(), tp_colors):
    tp = term_premiums(model_2f, z, n_max_tp) * 4 * 100
    r_now = short_rate(model_2f, z) * 4 * 100
    lam = risk_prices(model_2f, z)
    ax1.plot(maturities_tp, tp, color=color, lw=2.2,
             label=(f"{label}\n  r={r_now:.1f}%,"
                    f" λ=[{lam[0]:.3f}, {lam[1]:.3f}]"))

ax1.axhline(0, color="black", lw=0.8, ls="--")
ax1.set_xlabel("Maturity (quarters)")
ax1.set_ylabel("Term premium (% p.a.)")
ax1.set_title("Term Premiums — Two Regimes\n"
              r"($\lambda_z < 0$: higher premiums when rates are low)")
ax1.legend(fontsize=9)
ax1.set_xlim(1, n_max_tp)

z_decomp = np.array([0.0, 0.0])
A_bar_d, B_bar_d = bond_coefficients(model_2f, n_max_tp + 1)
λ_t = risk_prices(model_2f, z_decomp)
C_lam = model_2f.C @ λ_t

tp_level = np.array([-B_bar_d[n, 0] * C_lam[0]
                      for n in range(1, n_max_tp + 1)]) * 4 * 100
tp_slope = np.array([-B_bar_d[n, 1] * C_lam[1]
                      for n in range(1, n_max_tp + 1)]) * 4 * 100
tp_total = tp_level + tp_slope

ax2.plot(maturities_tp, tp_total, 'k-', lw=2.2, label="Total")
ax2.plot(maturities_tp, tp_level, color="purple", lw=1.8, ls="--",
         label="Level factor")
ax2.plot(maturities_tp, tp_slope, color="orange", lw=1.8, ls="--",
         label="Slope factor")
ax2.axhline(0, color="black", lw=0.6, ls=":")
ax2.set_xlabel("Maturity (quarters)")
ax2.set_ylabel("Term premium (% p.a.)")
ax2.set_title("Factor Decomposition at z = [0, 0]")
ax2.legend(fontsize=10)
ax2.set_xlim(1, n_max_tp)

for ax in (ax1, ax2):
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    year_ticks = [4, 12, 20, 40, 60]
    ax_top.set_xticks(year_ticks)
    ax_top.set_xticklabels([f"{t/4:.0f}y" for t in year_ticks])

plt.tight_layout()
plt.show()
```

## Risk-neutral probabilities

The stochastic discount factor {eq}`eq_sdf` defines a **change of measure** from the
physical measure $P$ to the **risk-neutral measure** $Q$.

Define the likelihood ratio

```{math}
:label: eq_RN_ratio

\frac{\xi^Q_{t+1}}{\xi^Q_t} = \exp\!\left(-\frac{1}{2}\lambda_t^\top\lambda_t - \lambda_t^\top\varepsilon_{t+1}\right)
```

Then

$$
m_{t+1} = \frac{\xi^Q_{t+1}}{\xi^Q_t}\exp(-r_t)
$$

and the pricing equation $\mathbb{E}^P_t(m_{t+1}R_{j,t+1}) = 1$ becomes

```{math}
:label: eq_Qpricing

\mathbb{E}^Q_t R_{j,t+1} = \exp(r_t)
```

*Under the risk-neutral measure, expected returns on all assets equal the risk-free return.*

### The risk-neutral VAR

Multiplying the physical conditional distribution of $z_{t+1}$ by the likelihood
ratio {eq}`eq_RN_ratio` gives the **risk-neutral conditional distribution**

$$
z_{t+1} \mid z_t \;\overset{Q}{\sim}\; \mathcal{N}\!\bigl(\mu - C\lambda_0 + (\phi - C\lambda_z)z_t,\; CC^\top\bigr)
$$

In other words, under $Q$ the state vector follows

$$
z_{t+1} = (\mu - C\lambda_0) + (\phi - C\lambda_z)\,z_t + C\varepsilon^Q_{t+1}
$$

where $\varepsilon^Q_{t+1} \sim \mathcal{N}(0, I)$ under $Q$.

The risk-neutral drift adjustments $-C\lambda_0$ (constant) and $-C\lambda_z$ (state-dependent)
encode exactly how the asset pricing formula $\mathbb{E}^P_t m_{t+1}R_{j,t+1}=1$ adjusts
expected returns for exposure to the risks $\varepsilon_{t+1}$.

### Verification via risk-neutral pricing

Bond prices can be computed by discounting at $r_t$ under $Q$:

$$
p_t(n) = \mathbb{E}^Q_t\! \left[\exp\!\left(-\sum_{s=0}^{n-1}r_{t+s}\right)\right]
$$

We can verify that this agrees with {eq}`eq_bondprice` by iterating the affine
recursion under the risk-neutral VAR.

Below we confirm this numerically.

```{code-cell} ipython3
def bond_price_mc_Q(model, z0, n, n_sims=50_000, rng=None):
    """Estimate p_t(n) by Monte Carlo under Q."""
    if rng is None:
        rng = np.random.default_rng(2024)
    m = len(z0)
    Z = np.tile(z0, (n_sims, 1))
    disc = np.zeros(n_sims)
    for _ in range(n):
        disc += model.δ_0 + Z @ model.δ_1
        ε = rng.standard_normal((n_sims, m))
        Z = model.μ_rn + Z @ model.φ_rn.T + ε @ model.C.T
    return np.mean(np.exp(-disc))

z_test = np.array([0.01, 0.005])
p_analytic = bond_prices(model_2f, z_test, 40)

rng = np.random.default_rng(2024)
maturities_check = [4, 12, 24, 40]
mc_prices = [bond_price_mc_Q(model_2f, z_test, n, n_sims=80_000, rng=rng)
             for n in maturities_check]

header = (f"{'Maturity':>10}  {'Analytic':>12}"
          f"  {'Monte Carlo':>12}  {'Error (bps)':>12}")
print(header)
print("-" * 52)
for n, mc in zip(maturities_check, mc_prices):
    analytic = p_analytic[n - 1]
    error_bp = abs(analytic - mc) / analytic * 10_000
    print(f"{n:>10}  {analytic:>12.6f}  {mc:>12.6f}  {error_bp:>12.2f}")
```

The analytical and Monte Carlo bond prices agree closely, validating the
Riccati recursion {eq}`eq_riccati_A`–{eq}`eq_riccati_B`.

## Distorted beliefs

{cite:t}`piazzesi2015trend` assemble survey
evidence suggesting that economic experts' forecasts are *systematically biased*
relative to the physical measure.

### The subjective measure

Let $\hat z_{t+1}$ be one-period-ahead expert forecasts.

Regressing these on $z_t$:

$$
\hat z_{t+1} = \hat\mu + \hat\phi\, z_t + e_{t+1}
$$

yields estimates $\hat\mu, \hat\phi$ that differ from the physical parameters $\mu, \phi$.

To formalise the distortion, let $\kappa_t = \kappa_0 + \kappa_z z_t$ and define
the likelihood ratio

```{math}
:label: eq_Srat

\frac{\xi^S_{t+1}}{\xi^S_t}
= \exp\!\left(-\frac{1}{2}\kappa_t^\top\kappa_t - \kappa_t^\top\varepsilon_{t+1}\right)
```

Multiplying the physical conditional distribution of $z_{t+1}$ by this likelihood
ratio gives the **subjective (S) conditional distribution**

$$
z_{t+1} \mid z_t \;\overset{S}{\sim}\;
\mathcal{N}\!\bigl(\mu - C\kappa_0 + (\phi - C\kappa_z)\,z_t,\; CC^\top\bigr)
$$

Comparing with the regression implies

$$
\hat\mu = \mu - C\kappa_0, \qquad \hat\phi = \phi - C\kappa_z
$$

Piazzesi et al. find that experts behave as if the level and slope of the yield
curve are *more persistent* than under the physical measure: $\hat\phi$ has
larger eigenvalues than $\phi$.

### Pricing under distorted beliefs

A representative agent with subjective beliefs $S$ and risk prices $\lambda^\star_t$
satisfies

$$
\mathbb{E}^S_t\bigl(m^\star_{t+1} R_{j,t+1}\bigr) = 1
$$

Expanding this in terms of the physical measure $P$, one finds that the
**rational-expectations econometrician** who imposes $P$ will estimate risk prices

$$
\hat\lambda_t = \lambda^\star_t + \kappa_t
$$

That is, the econometrician's estimate conflates true risk prices $\lambda^\star_t$
and belief distortions $\kappa_t$.

Part of what looks like a high price of risk is actually a systematic forecast bias.

### Numerical illustration

```{code-cell} ipython3
φ_P = φ_2.copy()
μ_P = μ_2.copy()

# Subjective parameters: experts believe factors are MORE persistent
φ_S = np.array([[0.985, -0.025], [0.00, 0.94]])
μ_S = np.array([-0.005, 0.0])

κ_z = np.linalg.solve(C_2, φ_P - φ_S)
κ_0 = np.linalg.solve(C_2, μ_P - μ_S)

print("Distortion parameters"
      " (κ quantifies how experts' beliefs"
      " differ from P):")
print(f"  κ_0 = {κ_0.round(4)}")
print(f"  κ_z =\n{κ_z.round(4)}")
print()
print("Eigenvalue comparison:")
eig_P = sorted(eigvals(φ_P).real, reverse=True)
eig_S = sorted(eigvals(φ_S).real, reverse=True)
print(f"  Physical φ eigenvalues:   {[round(e, 4) for e in eig_P]}")
print(f"  Subjective φ̂ eigenvalues: {[round(e, 4) for e in eig_S]}")
print("  Experts believe both factors are more persistent.")
print()

λ_star_0 = np.array([0.03, 0.015])
λ_star_z = np.array([[-0.006, 0.0], [0.0, -0.004]])

λ_hat_0 = λ_star_0 + κ_0
λ_hat_z = λ_star_z + κ_z

print("True risk prices:         λ*_0 =", λ_star_0.round(4))
print("Econometrician estimates: λ̂_0  =", λ_hat_0.round(4))
print(f"  Belief distortion inflates λ̂_0 by κ_0 = {κ_0.round(4)}.")
```

```{code-cell} ipython3
model_true = create_affine_model(
    μ_2, φ_2, C_2, δ_0_2, δ_1_2, λ_star_0, λ_star_z)
model_econ = create_affine_model(
    μ_2, φ_2, C_2, δ_0_2, δ_1_2, λ_hat_0, λ_hat_z)

for name, mdl in [("True", model_true), ("Econometrician", model_econ)]:
    eigs = eigvals(mdl.φ_rn).real
    status = "stable" if all(abs(e) < 1 for e in eigs) else "UNSTABLE"
    print(f"{name} model: φ^Q eigenvalues = {eigs.round(4)} ({status})")

z_ref = np.array([0.0, 0.0])
n_max_db = 60
maturities_db = np.arange(1, n_max_db + 1)

tp_true = term_premiums(model_true, z_ref, n_max_db) * 4 * 100
tp_econ = term_premiums(model_econ, z_ref, n_max_db) * 4 * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.plot(maturities_db, tp_true, lw=2.2, color="steelblue",
         label=r"True risk prices $\lambda^\star_t$")
ax1.plot(maturities_db, tp_econ, lw=2.2, color="firebrick", ls="--",
         label=(r"RE econometrician"
                r" $\hat\lambda_t = \lambda^\star_t + \kappa_t$"))
ax1.fill_between(maturities_db, tp_true, tp_econ,
                 alpha=0.15, color="firebrick",
                 label="Belief distortion component")
ax1.axhline(0, color="black", lw=0.8, ls=":")
ax1.set_xlabel("Maturity (quarters)")
ax1.set_ylabel("Term premium (% p.a.)")
ax1.set_title("True vs. Distorted-Belief Term Premiums")
ax1.legend(fontsize=9.5)
ax1.set_xlim(1, n_max_db)

mask = np.abs(tp_true) > 1e-8
ratio = np.full_like(tp_true, np.nan)
ratio[mask] = tp_econ[mask] / tp_true[mask]

ax2.plot(maturities_db[mask], ratio[mask], lw=2.2, color="darkred")
ax2.axhline(1, color="black", lw=0.8, ls="--",
            label="No distortion (ratio = 1)")
ax2.set_xlabel("Maturity (quarters)")
ax2.set_ylabel(r"$\hat{tp}\, /\, tp^\star$")
ax2.set_title("Overstatement Ratio from Ignoring Belief Bias")
ax2.legend(fontsize=11)
ax2.set_xlim(1, n_max_db)

for ax in (ax1, ax2):
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    year_ticks = [4, 12, 20, 40, 60]
    ax_top.set_xticks(year_ticks)
    ax_top.set_xticklabels([f"{t/4:.0f}y" for t in year_ticks])

plt.tight_layout()
plt.show()
```

When expert beliefs are overly persistent ($\hat\phi$ has larger eigenvalues than
$\phi$), the rational-expectations econometrician attributes too much of the
observed risk premium to risk aversion.

Disentangling belief distortions from genuine risk prices requires additional
data — for example, the survey forecasts used by Piazzesi, Salomao, and Schneider.

Our {doc}`Risk Aversion or Mistaken Beliefs? <risk_aversion_or_mistaken_beliefs>` lecture
explores this confounding in greater depth.

## The bond price recursion

We verify the exponential affine form {eq}`eq_bondprice` by induction.

**Claim:** If $p_{t+1}(n) = \exp(\bar A_n + \bar B_n^\top z_{t+1})$, then
$p_t(n+1) = \exp(\bar A_{n+1} + \bar B_{n+1}^\top z_t)$ with $\bar A_{n+1}$ and
$\bar B_{n+1}$ given by {eq}`eq_riccati_A`–{eq}`eq_riccati_B`.

**Proof sketch.**  Using the SDF {eq}`eq_sdf` and the VAR {eq}`eq_var`:

$$
\log m_{t+1} + \log p_{t+1}(n)
= -r_t - \tfrac{1}{2}\lambda_t^\top\lambda_t
  + (\bar A_n + \bar B_n^\top\mu + \bar B_n^\top\phi z_t)
  + (-\lambda_t + C^\top\bar B_n)^\top\varepsilon_{t+1}
$$

Taking the conditional expectation (and using $\varepsilon_{t+1}\sim\mathcal{N}(0,I)$):

$$
\log p_t(n+1) = -r_t - \tfrac{1}{2}\lambda_t^\top\lambda_t
  + \bar A_n + \bar B_n^\top(\mu + \phi z_t)
  + \tfrac{1}{2}(\lambda_t - C^\top\bar B_n)^\top(\lambda_t - C^\top\bar B_n)
$$

Substituting $r_t = \delta_0 + \delta_1^\top z_t$ and $\lambda_t = \lambda_0 + \lambda_z z_t$,
collecting constant and linear-in-$z_t$ terms, and equating coefficients gives
exactly {eq}`eq_riccati_A`–{eq}`eq_riccati_B`. $\blacksquare$

## Concluding remarks

The affine model of the stochastic discount factor provides a flexible and tractable
framework for studying asset prices.

Key features are:

1. **Analytical tractability** — Bond prices are exponential affine in $z_t$;
   expected returns decompose cleanly into a short rate plus a risk-price×exposure inner product.
2. **Empirical flexibility** — The free parameters $(\mu, \phi, C, \delta_0, \delta_1, \lambda_0, \lambda_z)$
   can be estimated by maximum likelihood (the {doc}`Kalman filter <kalman>` chapter describes
   the relevant methods) without imposing restrictions from a full general equilibrium model.
3. **Multiple risks** — The vector structure accommodates many sources of risk (monetary
   policy, real activity, volatility, etc.).
4. **Belief distortions** — The framework naturally accommodates non-rational beliefs via
   likelihood-ratio twists of the physical measure, as in
   {cite:t}`piazzesi2015trend`.

The model also connects directly to the Hansen–Jagannathan bounds studied in
{doc}`Doubts or Variability? <doubts_or_variability>` and to robust
control interpretations of the stochastic discount factor described in other
chapters of {cite:t}`Ljungqvist2012`.

For finite-state approaches to asset pricing that complement the continuous-state
framework here, see {doc}`Asset Pricing: Finite State Models <markov_asset>`.
