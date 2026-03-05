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

These models are presented in chapter 15 of {cite}`Ljungqvist2012`.

The models discussed here take a different approach from the time-separable CRRA
stochastic discount factor of Hansen and Singleton {cite}`hansen1983stochastic`.

The CRRA stochastic discount factor is

$$
m_{t+1} = \exp\left(-r_t - \frac{1}{2}\sigma_c^2 \gamma^2 - \gamma\sigma_c\varepsilon_{t+1}\right)
$$

where $r_t = \rho + \gamma\mu - \frac{1}{2}\sigma_c^2\gamma^2$.

This model asserts that exposure to the random part of aggregate consumption growth,
$\sigma_c\varepsilon_{t+1}$, is the **only** priced risk — the sole source of discrepancies
among expected returns across assets.

Empirical difficulties with this specification (the equity premium puzzle, the
risk-free rate puzzle, and the Hansen-Jagannathan bounds) motivate the alternative approach
described in this lecture.

The **affine model** maintains $\mathbb{E}(m_{t+1}R_{j,t+1}) = 1$ but **divorces** the
stochastic discount factor from consumption risk.  Instead, it

* specifies an analytically tractable stochastic process for $m_{t+1}$, and
* uses overidentifying restrictions from $\mathbb{E}(m_{t+1}R_{j,t+1}) = 1$ applied to $N$
  assets to let the data reveal risks and their prices.

Key applications we study include:

1. **Pricing risky assets** — how risk prices and exposures determine excess returns.
1. **Affine term structure models** — bond yields as affine functions of a state vector
   (Ang and Piazzesi {cite}`AngPiazzesi2003`).
1. **Risk-neutral probabilities** — a change-of-measure representation of the pricing equation.
1. **Distorted beliefs** — reinterpreting risk price estimates when agents hold systematically
   biased forecasts (Piazzesi, Salomao, and Schneider {cite}`piazzesi2015trend`).

We start with some standard imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power, eigvals
```

## The Model

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

r_t = \delta_0 + \delta_1' z_t
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

\log(m_{t+1}) = -r_t - \frac{1}{2}\lambda_t'\lambda_t - \lambda_t'\varepsilon_{t+1}
```

Here $\lambda_0$ is $m \times 1$ and $\lambda_z$ is $m \times m$.

The entries of $\lambda_t$ that multiply corresponding entries of the risks
$\varepsilon_{t+1}$ are called **risk prices** because they determine how exposures
to each risk component affect expected returns (as we show below).

Because $\lambda_t$ is affine in $z_t$, the stochastic discount factor $m_{t+1}$ is
**exponential quadratic** in the state $z_t$.

### Properties of the SDF

Since $\lambda_t'\varepsilon_{t+1}$ is conditionally normal, it follows that

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

## Pricing Risky Assets

### Lognormal returns

Consider a risky asset $j$ whose gross return has a lognormal conditional distribution:

```{math}
:label: eq_return

R_{j,t+1} = \exp\left(\nu_t(j) - \frac{1}{2}\alpha_t(j)'\alpha_t(j) + \alpha_t(j)'\varepsilon_{t+1}\right)
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

\nu_t(j) = r_t + \alpha_t(j)'\lambda_t
```

This is a central result.  It says:

> The expected net return on asset $j$ equals the short rate plus the inner product
> of the asset's exposure vector $\alpha_t(j)$ with the risk price vector $\lambda_t$.

Each component of $\lambda_t$ prices the corresponding component of $\varepsilon_{t+1}$.
An asset that loads heavily on a risk component with a large risk price earns a
correspondingly high expected return.

## Affine Term Structure of Yields

One of the most important applications is the **affine term structure model** studied
by Ang and Piazzesi {cite}`AngPiazzesi2003`.

### Bond prices

Let $p_t(n)$ be the price at time $t$ of a risk-free pure discount bond maturing at
$t + n$ (paying one unit of consumption).  The one-period gross return on holding
an $(n+1)$-period bond from $t$ to $t+1$ is

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
p_t(1) = \mathbb{E}_t(m_{t+1}) = \exp(-r_t) = \exp(-\delta_0 - \delta_1'z_t).
$$

### Exponential affine prices

The recursion {eq}`eq_bondrecur` has an **exponential affine** solution:

```{math}
:label: eq_bondprice

p_t(n) = \exp\!\bigl(\bar A_n + \bar B_n' z_t\bigr)
```

where the scalar $\bar A_n$ and the $m \times 1$ vector $\bar B_n$ satisfy the
**Riccati difference equations**

```{math}
:label: eq_riccati_A

\bar A_{n+1} = \bar A_n + \bar B_n'(\mu - C\lambda_0) + \frac{1}{2}\bar B_n' CC'\bar B_n - \delta_0
```

```{math}
:label: eq_riccati_B

\bar B_{n+1}' = \bar B_n'(\phi - C\lambda_z) - \delta_1'
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

y_t(n) = A_n + B_n' z_t
```

where $A_n = -\bar A_n / n$ and $B_n = -\bar B_n / n$.

**Yields are affine functions of the state vector $z_t$.**  This is the defining
property of affine term structure models.

## Python Implementation

We now implement the affine term structure model and compute bond prices, yields,
and risk premiums numerically.

```{code-cell} ipython3
class AffineTermStructure:
    """
    Implements the affine term structure model of
    Ang and Piazzesi (2003).

    State dynamics:
        z_{t+1} = μ + φ z_t + C ε_{t+1},   ε_{t+1} ~ N(0, I)
    Short rate:
        r_t = δ_0 + δ_1' z_t
    Risk prices:
        λ_t = λ_0 + λ_z z_t
    Log SDF:
        log(m_{t+1}) = -r_t - (1/2) λ_t' λ_t - λ_t' ε_{t+1}

    Parameters
    ----------
    mu : array (m,)
    phi : array (m, m)
    C : array (m, m)
    delta0 : float
    delta1 : array (m,)
    lambda0 : array (m,)
    lambdaz : array (m, m)
    """

    def __init__(self, mu, phi, C, delta0, delta1, lambda0, lambdaz):
        self.mu = np.asarray(mu, dtype=float)
        self.phi = np.asarray(phi, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.delta0 = float(delta0)
        self.delta1 = np.asarray(delta1, dtype=float)
        self.lambda0 = np.asarray(lambda0, dtype=float)
        self.lambdaz = np.asarray(lambdaz, dtype=float)
        self.m = len(self.mu)
        # Risk-adjusted drift: φ - C λ_z  and  μ - C λ_0
        self.phi_rn = self.phi - self.C @ self.lambdaz
        self.mu_rn = self.mu - self.C @ self.lambda0

    def bond_coefficients(self, n_max):
        """
        Compute (Ā_n, B̄_n) for n = 1, ..., n_max via forward recursion.

        Returns
        -------
        A_bar : array (n_max + 1,)   A_bar[n] = Ā_n
        B_bar : array (n_max + 1, m) B_bar[n] = B̄_n
        """
        A_bar = np.zeros(n_max + 1)
        B_bar = np.zeros((n_max + 1, self.m))

        # Initial conditions: Ā_1 = -δ_0, B̄_1 = -δ_1
        A_bar[1] = -self.delta0
        B_bar[1] = -self.delta1

        CC = self.C @ self.C.T

        for n in range(1, n_max):
            Bn = B_bar[n]
            A_bar[n + 1] = (A_bar[n]
                            + Bn @ (self.mu - self.C @ self.lambda0)
                            + 0.5 * Bn @ CC @ Bn
                            - self.delta0)
            B_bar[n + 1] = self.phi_rn.T @ Bn - self.delta1

        return A_bar, B_bar

    def yields(self, z, n_max):
        """
        Compute the yield curve y_t(n) = A_n + B_n' z_t for n=1,...,n_max.

        Parameters
        ----------
        z : array (m,)  current state vector
        n_max : int     maximum maturity

        Returns
        -------
        y : array (n_max,)  yields y(1), ..., y(n_max)
        """
        A_bar, B_bar = self.bond_coefficients(n_max)
        ns = np.arange(1, n_max + 1)
        y = np.array([(-A_bar[n] - B_bar[n] @ z) / n for n in ns])
        return y

    def bond_prices(self, z, n_max):
        """
        Compute bond prices p_t(n) = exp(Ā_n + B̄_n' z_t) for n=1,...,n_max.

        Parameters
        ----------
        z : array (m,)
        n_max : int

        Returns
        -------
        p : array (n_max,)
        """
        A_bar, B_bar = self.bond_coefficients(n_max)
        p = np.array([np.exp(A_bar[n] + B_bar[n] @ z) for n in range(1, n_max + 1)])
        return p

    def simulate(self, z0, T, rng=None):
        """
        Simulate the state process for T periods.

        Parameters
        ----------
        z0 : array (m,)  initial state
        T : int          number of periods
        rng : np.random.Generator (optional)

        Returns
        -------
        Z : array (T+1, m)  simulated states including z0
        """
        if rng is None:
            rng = np.random.default_rng(42)
        Z = np.zeros((T + 1, self.m))
        Z[0] = z0
        for t in range(T):
            eps = rng.standard_normal(self.m)
            Z[t + 1] = self.mu + self.phi @ Z[t] + self.C @ eps
        return Z

    def short_rate(self, z):
        """Short rate r_t = δ_0 + δ_1' z_t."""
        return self.delta0 + self.delta1 @ z

    def risk_prices(self, z):
        """Risk price vector λ_t = λ_0 + λ_z z_t."""
        return self.lambda0 + self.lambdaz @ z

    def expected_excess_return(self, z, alpha0, alphaz):
        """
        Expected excess return ν_t(j) - r_t = α_t(j)' λ_t.

        Parameters
        ----------
        z : array (m,)
        alpha0 : array (m,)
        alphaz : array (m, m)
        """
        alpha_t = alpha0 + alphaz @ z
        lambda_t = self.risk_prices(z)
        return alpha_t @ lambda_t
```

### A one-factor Gaussian example

To build intuition, we start with a single-factor ($m=1$) Gaussian model.

```{code-cell} ipython3
# ── One-factor Gaussian model (quarterly, standardized state) ───────────
# z_t has unit innovation variance; C = I absorbs the volatility.
# δ_1 translates z into short-rate units (small: ~1bp per unit of z).

mu      = np.array([0.0])          # VAR intercept → E[z]=0
phi     = np.array([[0.95]])       # persistence
C       = np.array([[1.0]])        # identity (standardized shocks)
delta0  = 0.01                     # base short rate: 1%/quarter = 4% p.a.
delta1  = np.array([0.001])        # ≈ 40bp p.a. per std-dev of z
lambda0 = np.array([0.05])         # constant risk price
lambdaz = np.array([[-0.01]])      # countercyclical (λ falls when z rises)

model_1f = AffineTermStructure(mu, phi, C, delta0, delta1, lambda0, lambdaz)

# Check stability
phi_Q = model_1f.phi_rn[0, 0]
half_life = np.log(2) / (-np.log(phi[0, 0]))
uncond_std_z = 1.0 / np.sqrt(1 - phi[0,0]**2)   # std dev of stationary z
print(f"Physical AR(1):      φ   = {phi[0,0]:.3f}  (half-life {half_life:.1f} quarters)")
print(f"Risk-neutral AR(1):  φ^Q = {phi_Q:.3f}  "
      f"{'✓ stable' if abs(phi_Q) < 1 else '✗ UNSTABLE!'}")
print(f"Unconditional std of z:  σ_z = {uncond_std_z:.2f}")
print(f"Mean short rate = {model_1f.short_rate(np.array([0.0]))*4*100:.1f}% p.a.")
print(f"Short rate range (±2σ): [{(delta0-delta1[0]*2*uncond_std_z)*4*100:.1f}%, "
      f"{(delta0+delta1[0]*2*uncond_std_z)*4*100:.1f}%] p.a.")
```

### Yield curve shapes

We compute yield curves across a range of short-rate states $z_t$.

```{code-cell} ipython3
# ── Yield curves: one-factor model ──────────────────────────────────────
n_max_1f = 60   # 60 quarters = 15 years
maturities_1f = np.arange(1, n_max_1f + 1)

# Three states spanning the short-rate range
# z < 0 → low short rate, z > 0 → high short rate
z_low  = np.array([-5.0])     # short rate well below mean → expect rise → upward slope
z_mid  = np.array([0.0])      # at the mean
z_high = np.array([5.0])      # short rate well above mean → expect decline → inverted

fig, ax = plt.subplots(figsize=(9, 5.5))

for z, label, color in [
    (z_low,  f"Low state  (r₁ = {model_1f.short_rate(z_low)*4*100:.1f}%)", "steelblue"),
    (z_mid,  f"Median state (r₁ = {model_1f.short_rate(z_mid)*4*100:.1f}%)", "seagreen"),
    (z_high, f"High state (r₁ = {model_1f.short_rate(z_high)*4*100:.1f}%)", "firebrick"),
]:
    y = model_1f.yields(z, n_max_1f) * 4 * 100   # annualised %
    ax.plot(maturities_1f, y, color=color, lw=2.2, label=label)
    ax.plot(1, y[0], 'o', color=color, ms=7, zorder=5)

# Mark unconditional mean short rate
r_bar = model_1f.short_rate(np.array([0.0])) * 4 * 100
ax.axhline(r_bar, color='grey', ls=':', lw=1.2, alpha=0.7,
           label=f"Mean short rate ({r_bar:.1f}%)")

ax.set_xlabel("Maturity (quarters)")
ax.set_ylabel("Yield (% per annum)")
ax.set_title("Yield Curves — One-Factor Affine Model")
ax.legend(fontsize=10, loc='best')
ax.set_xlim(1, n_max_1f)

# Secondary x-axis in years
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
# ── Simulated short rate ────────────────────────────────────────────────
T = 200
Z = model_1f.simulate(np.array([0.0]), T)
short_rates = np.array([model_1f.short_rate(Z[t]) * 4 * 100 for t in range(T + 1)])
r_bar_pct = model_1f.short_rate(np.array([0.0])) * 4 * 100

fig, ax = plt.subplots(figsize=(10, 4))
quarters = np.arange(T + 1)
ax.plot(quarters, short_rates, color="steelblue", lw=1.3)
ax.axhline(r_bar_pct, color="red", ls="--", lw=1.3,
           label=f"Unconditional mean ({r_bar_pct:.1f}%)")
ax.fill_between(quarters, short_rates, r_bar_pct, alpha=0.08, color="steelblue")
ax.set_xlabel("Quarter")
ax.set_ylabel("Short rate (% p.a.)")
ax.set_title("Simulated Short Rate — One-Factor Model (50 years)")
ax.set_xlim(0, T)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()
```

### A two-factor model

To match richer yield-curve dynamics, practitioners routinely use $m \geq 2$
factors.  We now introduce a two-factor specification in which the factors
can be interpreted as a **level** component and a **slope** component.

```{code-cell} ipython3
# ── Two-factor model (quarterly, standardized state) ────────────────────
# z = [level, slope]'
mu2      = np.array([0.0,  0.0])
phi2     = np.array([[0.97, -0.03],    # level very persistent, slope feeds back
                     [0.00,  0.90]])   # slope moderately persistent
C2       = np.array([[1.0, 0.0],       # identity (standardized shocks)
                     [0.0, 1.0]])
delta0_2 = 0.01                         # base short rate = 4% p.a.
delta1_2 = np.array([0.002, 0.001])     # level matters more for r
lambda0_2 = np.array([0.01,  0.005])    # small constant risk prices
lambdaz_2 = np.array([[-0.005, 0.0],    # level risk price falls when level rises
                      [ 0.0, -0.003]])  # slope risk price falls when slope rises

model_2f = AffineTermStructure(mu2, phi2, C2,
                                delta0_2, delta1_2,
                                lambda0_2, lambdaz_2)

print("Physical measure VAR:")
print(f"  φ =\n{phi2}")
print(f"  eigenvalues of φ: {eigvals(phi2).real.round(4)}")
print()
print("Risk-neutral measure VAR:")
print(f"  φ^Q = φ - Cλ_z =\n{model_2f.phi_rn.round(4)}")
eigs_Q = eigvals(model_2f.phi_rn).real
print(f"  eigenvalues of φ^Q: {eigs_Q.round(4)}")
stable = all(abs(e) < 1 for e in eigs_Q)
print(f"  {'✓ All eigenvalues inside unit circle' if stable else '✗ UNSTABLE!'}")
print()
print("→ Risk prices make Q dynamics MORE persistent than P dynamics")
```

```{code-cell} ipython3
# ── Yield curves + factor loadings: two-factor model ────────────────────
n_max_2f = 60
maturities_2f = np.arange(1, n_max_2f + 1)

states = {
    "Normal":              np.array([0.0,   0.0]),
    "Low short rate":      np.array([-4.0,  3.0]),
    "High short rate":     np.array([4.0,  -3.0]),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Left: Yield curves ──
colors_2f = ["seagreen", "steelblue", "firebrick"]
for (label, z), color in zip(states.items(), colors_2f):
    r_now = model_2f.short_rate(z) * 4 * 100
    y = model_2f.yields(z, n_max_2f) * 4 * 100
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

# ── Right: Factor loadings B_n ──
A_bar, B_bar = model_2f.bond_coefficients(n_max_2f)
ns = np.arange(1, n_max_2f + 1)
B_n = np.array([-B_bar[n] / n for n in ns])   # B_n = -B̄_n / n

ax2.plot(ns, B_n[:, 0], lw=2.2, color="purple", label=r"Level loading $B_{n,1}$")
ax2.plot(ns, B_n[:, 1], lw=2.2, color="orange", label=r"Slope loading $B_{n,2}$")
ax2.axhline(0, color='black', lw=0.6)
ax2.set_xlabel("Maturity (quarters)")
ax2.set_ylabel(r"Yield loading $B_{n,k}$")
ax2.set_title("Factor Loadings on Yields")
ax2.legend(fontsize=11)
ax2.set_xlim(1, n_max_2f)
ax2.annotate("Level factor stays\nimportant at long maturities",
             xy=(45, B_n[44, 0]), fontsize=9, color="purple",
             ha='center', va='bottom')

# Year labels on top
for ax in (ax1, ax2):
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    year_ticks = [4, 12, 20, 40, 60]
    ax_top.set_xticks(year_ticks)
    ax_top.set_xticklabels([f"{t/4:.0f}y" for t in year_ticks])

plt.tight_layout()
plt.show()
```

## Risk Premiums

A key object in the affine term structure model is the **term premium** — the extra
expected return on a long-term bond relative to rolling over short-term bonds.

For an $(n+1)$-period bond held for one period, the excess log return is
approximately

$$
\mathbb{E}_t\left[\log R_{t+1}^{(n+1)}\right] - r_t \;=\; -\bar B_n' C \lambda_t
$$

That is, the term premium equals (minus) the product of the bond's exposure to
the shocks $(-\bar B_n'C)$ with the risk prices $\lambda_t$.

```{code-cell} ipython3
def term_premiums(model, z, n_max):
    """
    Compute approximate term premiums for maturities 1 to n_max.

    Term premium for holding an (n+1)-period bond ≈ -B̄_n' C λ_t.
    """
    A_bar, B_bar = model.bond_coefficients(n_max + 1)
    lambda_t = model.risk_prices(z)
    tp = np.array([-B_bar[n] @ model.C @ lambda_t
                   for n in range(1, n_max + 1)])
    return tp

n_max_tp = 60
maturities_tp = np.arange(1, n_max_tp + 1)

z_states_tp = {
    "Low rate (z₁ < 0)":  np.array([-3.0, 2.0]),
    "High rate (z₁ > 0)": np.array([3.0, -2.0]),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Left: Total term premiums at two states ──
tp_colors = ["steelblue", "firebrick"]
for (label, z), color in zip(z_states_tp.items(), tp_colors):
    tp = term_premiums(model_2f, z, n_max_tp) * 4 * 100   # annualised %
    r_now = model_2f.short_rate(z) * 4 * 100
    lam = model_2f.risk_prices(z)
    ax1.plot(maturities_tp, tp, color=color, lw=2.2,
             label=f"{label}\n  r={r_now:.1f}%, λ=[{lam[0]:.3f}, {lam[1]:.3f}]")

ax1.axhline(0, color="black", lw=0.8, ls="--")
ax1.set_xlabel("Maturity (quarters)")
ax1.set_ylabel("Term premium (% p.a.)")
ax1.set_title("Term Premiums — Two Regimes\n"
              r"($\lambda_z < 0$: higher premiums when rates are low)")
ax1.legend(fontsize=9)
ax1.set_xlim(1, n_max_tp)

# ── Right: Decomposition by factor at z = [0,0] ──
z_decomp = np.array([0.0, 0.0])
A_bar_d, B_bar_d = model_2f.bond_coefficients(n_max_tp + 1)
lam_t = model_2f.risk_prices(z_decomp)
C_lam = model_2f.C @ lam_t

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

# Year labels
for ax in (ax1, ax2):
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    year_ticks = [4, 12, 20, 40, 60]
    ax_top.set_xticks(year_ticks)
    ax_top.set_xticklabels([f"{t/4:.0f}y" for t in year_ticks])

plt.tight_layout()
plt.show()
```

## Risk-Neutral Probabilities

The stochastic discount factor {eq}`eq_sdf` defines a **change of measure** from the
physical measure $P$ to the **risk-neutral measure** $Q$.

Define the likelihood ratio

```{math}
:label: eq_RN_ratio

\frac{\xi^Q_{t+1}}{\xi^Q_t} = \exp\!\left(-\frac{1}{2}\lambda_t'\lambda_t - \lambda_t'\varepsilon_{t+1}\right)
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

**Under the risk-neutral measure, expected returns on all assets equal the risk-free return.**

### The risk-neutral VAR

Multiplying the physical conditional distribution of $z_{t+1}$ by the likelihood
ratio {eq}`eq_RN_ratio` gives the **risk-neutral conditional distribution**

$$
z_{t+1} \mid z_t \;\overset{Q}{\sim}\; \mathcal{N}\!\bigl(\mu - C\lambda_0 + (\phi - C\lambda_z)z_t,\; CC'\bigr)
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
recursion under the risk-neutral VAR.  Below we confirm this numerically.

```{code-cell} ipython3
# ── Risk-neutral Monte Carlo verification ────────────────────────────
def bond_price_monte_carlo_Q(model, z0, n, n_sims=50_000, rng=None):
    """Estimate p_t(n) by Monte Carlo under Q."""
    if rng is None:
        rng = np.random.default_rng(2024)
    m_dim = len(z0)
    Z = np.tile(z0, (n_sims, 1))
    disc = np.zeros(n_sims)
    phi_Q = model.phi_rn
    mu_Q  = model.mu_rn
    C_mat = model.C
    for _ in range(n):
        r_t = model.delta0 + Z @ model.delta1
        disc += r_t
        eps = rng.standard_normal((n_sims, m_dim))
        Z = mu_Q + Z @ phi_Q.T + eps @ C_mat.T
    return np.mean(np.exp(-disc))

# Compare at selected maturities
z_test = np.array([0.01, 0.005])
n_max_test = 40
p_analytic = model_2f.bond_prices(z_test, n_max_test)

rng = np.random.default_rng(2024)
maturities_check = [4, 12, 24, 40]
mc_prices = [bond_price_monte_carlo_Q(model_2f, z_test, n, n_sims=80_000, rng=rng)
             for n in maturities_check]

print(f"{'Maturity':>10}  {'Analytic':>12}  {'Monte Carlo':>12}  {'Error (bps)':>12}")
print("-" * 52)
for n, mc in zip(maturities_check, mc_prices):
    analytic = p_analytic[n - 1]
    error_bp = abs(analytic - mc) / analytic * 10_000
    print(f"{n:>10}  {analytic:>12.6f}  {mc:>12.6f}  {error_bp:>12.2f}")
```

The analytical and Monte Carlo bond prices agree closely, validating the
Riccati recursion {eq}`eq_riccati_A`–{eq}`eq_riccati_B`.

## Distorted Beliefs

Piazzesi, Salomao, and Schneider {cite}`piazzesi2015trend` assemble survey
evidence suggesting that economic experts' forecasts are **systematically biased**
relative to the physical measure.

### The subjective measure

Let $\hat z_{t+1}$ be one-period-ahead expert forecasts.  Regressing these on $z_t$:

$$
\hat z_{t+1} = \hat\mu + \hat\phi\, z_t + e_{t+1}
$$

yields estimates $\hat\mu, \hat\phi$ that differ from the physical parameters $\mu, \phi$.

To formalise the distortion, let $\kappa_t = \kappa_0 + \kappa_z z_t$ and define
the likelihood ratio

```{math}
:label: eq_Srat

\frac{\xi^S_{t+1}}{\xi^S_t}
= \exp\!\left(-\frac{1}{2}\kappa_t'\kappa_t - \kappa_t'\varepsilon_{t+1}\right)
```

Multiplying the physical conditional distribution of $z_{t+1}$ by this likelihood
ratio gives the **subjective (S) conditional distribution**

$$
z_{t+1} \mid z_t \;\overset{S}{\sim}\;
\mathcal{N}\!\bigl(\mu - C\kappa_0 + (\phi - C\kappa_z)\,z_t,\; CC'\bigr)
$$

Comparing with the regression implies

$$
\hat\mu = \mu - C\kappa_0, \qquad \hat\phi = \phi - C\kappa_z
$$

Piazzesi et al. find that experts behave as if the level and slope of the yield
curve are **more persistent** than under the physical measure: $\hat\phi$ has
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
and belief distortions $\kappa_t$.  Part of what looks like a high price of risk
is actually a systematic forecast bias.

### Numerical illustration

```{code-cell} ipython3
# ── Distorted beliefs: recover κ_0, κ_z ────────────────────────────────
phi_P = phi2.copy()
mu_P  = mu2.copy()

# Subjective parameters: experts believe factors are MORE persistent
phi_S = np.array([[0.985, -0.025], [0.00, 0.94]])
mu_S  = np.array([-0.005, 0.0])

# With C = I, distortion parameters are simply:
# κ_z = C⁻¹(φ_P - φ_S) = φ_P - φ_S
# κ_0 = C⁻¹(μ_P - μ_S) = μ_P - μ_S
C2_mat = model_2f.C
kappa_z = np.linalg.solve(C2_mat, phi_P - phi_S)
kappa_0 = np.linalg.solve(C2_mat, mu_P - mu_S)

print("Distortion parameters (κ quantifies how experts' beliefs differ from P):")
print(f"  κ_0 = {kappa_0.round(4)}")
print(f"  κ_z =\n{kappa_z.round(4)}")
print()
print("Eigenvalue comparison:")
eig_P = sorted(eigvals(phi_P).real, reverse=True)
eig_S = sorted(eigvals(phi_S).real, reverse=True)
print(f"  Physical φ eigenvalues:   {[round(e, 4) for e in eig_P]}")
print(f"  Subjective φ̂ eigenvalues: {[round(e, 4) for e in eig_S]}")
print("  → Experts believe both factors are more persistent")
print()

# True risk prices (what a correctly-specified agent would use)
lambda_star_0 = np.array([0.03, 0.015])
lambda_star_z = np.array([[-0.006, 0.0], [0.0, -0.004]])

# Econometrician who ignores belief distortion attributes:
lambda_hat_0 = lambda_star_0 + kappa_0
lambda_hat_z = lambda_star_z + kappa_z

print("True risk prices:         λ*_0 =", lambda_star_0.round(4))
print("Econometrician estimates: λ̂_0  =", lambda_hat_0.round(4))
print(f"  → Belief distortion inflates λ̂_0 by κ_0 = {kappa_0.round(4)}")
```

```{code-cell} ipython3
# ── Term premiums: true vs. distorted ──────────────────────────────────
model_true = AffineTermStructure(mu2, phi2, C2,
                                  delta0_2, delta1_2,
                                  lambda_star_0, lambda_star_z)

model_econ = AffineTermStructure(mu2, phi2, C2,
                                  delta0_2, delta1_2,
                                  lambda_hat_0, lambda_hat_z)

# Verify both models have stable Q dynamics
for name, mdl in [("True", model_true), ("Econometrician", model_econ)]:
    eigs = eigvals(mdl.phi_rn).real
    print(f"{name} model: φ^Q eigenvalues = {eigs.round(4)}, "
          f"{'✓' if all(abs(e) < 1 for e in eigs) else '✗ UNSTABLE'}")

z_ref = np.array([0.0, 0.0])
n_max_db = 60
maturities_db = np.arange(1, n_max_db + 1)

tp_true = term_premiums(model_true, z_ref, n_max_db) * 4 * 100
tp_econ = term_premiums(model_econ, z_ref, n_max_db) * 4 * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Left: Level comparison with shading ──
ax1.plot(maturities_db, tp_true, lw=2.2, color="steelblue",
         label=r"True risk prices $\lambda^\star_t$")
ax1.plot(maturities_db, tp_econ, lw=2.2, color="firebrick", ls="--",
         label=r"RE econometrician $\hat\lambda_t = \lambda^\star_t + \kappa_t$")
ax1.fill_between(maturities_db, tp_true, tp_econ, alpha=0.15, color="firebrick",
                 label="Belief distortion component")
ax1.axhline(0, color="black", lw=0.8, ls=":")
ax1.set_xlabel("Maturity (quarters)")
ax1.set_ylabel("Term premium (% p.a.)")
ax1.set_title("True vs. Distorted-Belief Term Premiums")
ax1.legend(fontsize=9.5)
ax1.set_xlim(1, n_max_db)

# ── Right: Ratio ──
mask = np.abs(tp_true) > 1e-8
ratio = np.full_like(tp_true, np.nan)
ratio[mask] = tp_econ[mask] / tp_true[mask]

ax2.plot(maturities_db[mask], ratio[mask], lw=2.2, color="darkred")
ax2.axhline(1, color="black", lw=0.8, ls="--", label="No distortion (ratio = 1)")
ax2.set_xlabel("Maturity (quarters)")
ax2.set_ylabel(r"$\hat{tp}\, /\, tp^\star$")
ax2.set_title("Overstatement Ratio from Ignoring Belief Bias")
ax2.legend(fontsize=11)
ax2.set_xlim(1, n_max_db)

# Year labels
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
observed risk premium to risk aversion.  Disentangling belief distortions from
genuine risk prices requires additional data — for example, the survey forecasts
used by Piazzesi, Salomao, and Schneider.

## The Bond Price Recursion

We verify the exponential affine form {eq}`eq_bondprice` by induction.

**Claim:** If $p_{t+1}(n) = \exp(\bar A_n + \bar B_n' z_{t+1})$, then
$p_t(n+1) = \exp(\bar A_{n+1} + \bar B_{n+1}' z_t)$ with $\bar A_{n+1}$ and
$\bar B_{n+1}$ given by {eq}`eq_riccati_A`–{eq}`eq_riccati_B`.

**Proof sketch.**  Using the SDF {eq}`eq_sdf` and the VAR {eq}`eq_var`:

$$
\log m_{t+1} + \log p_{t+1}(n)
= -r_t - \tfrac{1}{2}\lambda_t'\lambda_t
  + (\bar A_n + \bar B_n'\mu + \bar B_n'\phi z_t)
  + (-\lambda_t + C'\bar B_n)'\varepsilon_{t+1}
$$

Taking the conditional expectation (and using $\varepsilon_{t+1}\sim\mathcal{N}(0,I)$):

$$
\log p_t(n+1) = -r_t - \tfrac{1}{2}\lambda_t'\lambda_t
  + \bar A_n + \bar B_n'(\mu + \phi z_t)
  + \tfrac{1}{2}(\lambda_t - C'\bar B_n)'(\lambda_t - C'\bar B_n)
$$

Substituting $r_t = \delta_0 + \delta_1'z_t$ and $\lambda_t = \lambda_0 + \lambda_z z_t$,
collecting constant and linear-in-$z_t$ terms, and equating coefficients gives
exactly {eq}`eq_riccati_A`–{eq}`eq_riccati_B`. $\blacksquare$

## Concluding Remarks

The affine model of the stochastic discount factor provides a flexible and tractable
framework for studying asset prices.  Key features are:

1. **Analytical tractability** — Bond prices are exponential affine in $z_t$;
   expected returns decompose cleanly into a short rate plus a risk-price×exposure inner product.
2. **Empirical flexibility** — The free parameters $(\mu, \phi, C, \delta_0, \delta_1, \lambda_0, \lambda_z)$
   can be estimated by maximum likelihood (the {doc}`Kalman filter <kalman>` chapter describes
   the relevant methods) without imposing restrictions from a full general equilibrium model.
3. **Multiple risks** — The vector structure accommodates many sources of risk (monetary
   policy, real activity, volatility, etc.).
4. **Belief distortions** — The framework naturally accommodates non-rational beliefs via
   likelihood-ratio twists of the physical measure, as in
   Piazzesi, Salomao, and Schneider {cite}`piazzesi2015trend`.

The model also connects directly to the Hansen–Jagannathan bounds and to robust
control interpretations of the stochastic discount factor described in other
chapters of {cite}`Ljungqvist2012`.
