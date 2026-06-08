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

(subjective_beliefs_bc)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Survey Data and Subjective Beliefs in Business Cycles

```{index} single: Subjective Beliefs; Business Cycles
```

```{contents} Contents
:depth: 2
```

## Overview

This lecture presents key ideas from {cite}`bhandari2025survey`, who study
whether household survey data on macroeconomic expectations can shed light on
business cycle dynamics.

Their central finding is that household forecasts of unemployment and inflation
exhibit **systematic upward biases** relative to professional forecasters and
model-consistent rational expectations.  These biases — which the authors call
*belief wedges* — are:

* **Persistent and countercyclical**: they are larger during recessions.
* **Positively correlated**: optimism/pessimism about unemployment and inflation
  move together.
* **One-factor in structure**: a single latent state accounts for most
  variation across wedges.

The paper interprets this evidence through the lens of
**robust preferences** ({cite}`HansenSargent2001`; {cite}`HansenSargent2008`).

A household that fears model misspecification behaves as if it tilts
probabilities toward bad outcomes.

When calibrated to the Michigan Survey of
Consumers (1982Q1–2019Q4), this mechanism yields a time-varying *belief shock*
that substantially reduces the well-known **unemployment volatility puzzle** —
the fact that standard New Keynesian models with only technology and monetary
policy shocks generate far too little unemployment volatility.

By the end of this lecture you will understand:

* How to define and measure belief wedges from household survey data.
* How robust preferences generate time-varying subjective beliefs.
* How belief distortions propagate through a linearised DSGE model.
* Why a calibrated belief shock helps resolve the unemployment volatility
  puzzle.

## Setup

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.linalg import solve_discrete_lyapunov

np.random.seed(42)

plt.rcParams.update({
    'figure.figsize': (10, 5),
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
```

## Measuring Belief Wedges

### Definition

Let $E_t[\cdot]$ denote expectations under the **data-generating** (objective)
probability measure and $\tilde{E}_t[\cdot]$ denote **subjective** (survey)
expectations.

For any scalar variable $z_{t+1}$, the **one-period belief
wedge** is

$$

\Delta_t^{(1)}(z) \;=\; \tilde{E}_t[z_{t+1}] - E_t[z_{t+1}].

$$

A positive wedge means households are more pessimistic than the model predicts:
they expect $z_{t+1}$ to be higher than the model-consistent forecast.

For
unemployment and inflation this sign convention implies an upward bias.

In practice, {cite}`BhandariBorovickaHo2024` measure
$\tilde{E}_t[\cdot]$ from the Michigan Survey of Consumers, and
$E_t[\cdot]$ from a benchmark DSGE model estimated on the same data.

The
discrepancy is the wedge.

### Empirical Facts

Using data from 1982Q1 to 2019Q4, the authors document:

| Statistic | Unemployment wedge | Inflation wedge |
|---|---|---|
| Mean | 0.52 pp | 1.22 pp |
| Standard deviation | 0.67 pp | 1.03 pp |
| Correlation with output gap | −0.49 | −0.30 |

Both wedges are **positive on average** (households are pessimistic) and
**countercyclical** (pessimism rises in recessions).

Moreover, the first
principal component of the joint wedge series explains **78.6%** of its
variation — a striking one-factor structure.

The following code simulates artificial wedge series that match these
moments, so we can visualise the key stylised facts before turning to theory.

```{code-cell} ipython3
# ---------------------------------------------------------------------------
# Simulate stylised belief-wedge time series calibrated to match the
# empirical moments in Bhandari, Borovicka, Ho (2025).
# ---------------------------------------------------------------------------

# Calibrated parameters (Table 1 of the paper)
mu_theta    = 5.64   # mean of belief-shock parameter θ
rho_theta   = 0.714  # AR(1) persistence of θ
sigma_theta = 4.3    # innovation volatility of θ (units of θ)

# Wedge loadings: Δᵤ = cᵤ θ,  Δπ = cπ θ  (c chosen to match the means)
c_u  = 0.52 / mu_theta   # ≈ 0.0922 pp per unit of θ
c_pi = 1.22 / mu_theta   # ≈ 0.2163 pp per unit of θ

T      = 152   # 38 years × 4 quarters
dt     = 0.25  # quarterly

# Simulate the belief-shock AR(1)
rng   = np.random.default_rng(42)
theta = np.zeros(T)
theta[0] = mu_theta
for t in range(1, T):
    theta[t] = ((1 - rho_theta) * mu_theta
                + rho_theta * theta[t-1]
                + sigma_theta * rng.standard_normal())

# Belief wedges (in percentage points)
wedge_u  = c_u  * theta
wedge_pi = c_pi * theta

# Generate quarters 1982Q1 – 2019Q4
import datetime
quarters = [datetime.date(1982 + (q // 4), 3 * (q % 4) + 1, 1)
            for q in range(T)]
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

axes[0].plot(quarters, wedge_u, color='steelblue', linewidth=1.5,
             label='Unemployment belief wedge')
axes[0].axhline(np.mean(wedge_u), color='steelblue', linestyle='--',
                linewidth=0.8, alpha=0.7)
axes[0].set_ylabel('Percentage points')
axes[0].set_title('Unemployment belief wedge  $\\Delta_t^{(1)}(u)$')
axes[0].legend(loc='upper left')

axes[1].plot(quarters, wedge_pi, color='darkorange', linewidth=1.5,
             label='Inflation belief wedge')
axes[1].axhline(np.mean(wedge_pi), color='darkorange', linestyle='--',
                linewidth=0.8, alpha=0.7)
axes[1].set_ylabel('Percentage points')
axes[1].set_title('Inflation belief wedge  $\\Delta_t^{(1)}(\\pi)$')
axes[1].legend(loc='upper left')

for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# Show the one-factor structure: scatter of unemployment vs inflation wedge
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(wedge_u, wedge_pi, c=range(T), cmap='RdYlGn_r',
                alpha=0.7, s=20)
plt.colorbar(sc, ax=ax, label='Quarter (dark = recent)')
ax.set_xlabel('Unemployment wedge (pp)')
ax.set_ylabel('Inflation wedge (pp)')
ax.set_title('Positive comovement: one-factor structure')
corr = np.corrcoef(wedge_u, wedge_pi)[0, 1]
ax.text(0.05, 0.93, f'Correlation = {corr:.2f}',
        transform=ax.transAxes, fontsize=11)
plt.tight_layout()
plt.show()
```

The scatter plot reveals the strong positive correlation between the two
wedges.

Both series are high when the belief shock $\theta_t$ is high, and low otherwise.

This is the one-factor structure that motivates the
theoretical framework.

## A Model of Pessimism

### Robust Preferences

Why would households have systematically biased beliefs?  One disciplined
answer comes from **robust control** or **multiplier preferences**
({cite}`HansenSargent2001`, {cite}`HansenSargent2008`).

An agent who fears that her reference model may be misspecified solves

$$

V_t \;=\; \min_{\substack{m_{t+1} > 0 \\ E_t[m_{t+1}] = 1}}
\Bigl\{
  u(x_t)
  + \beta E_t\!\left[m_{t+1} V_{t+1}\right]
  + \frac{\beta}{\theta_t}\, E_t\!\left[m_{t+1} \log m_{t+1}\right]
\Bigr\}.

$$

Here $m_{t+1}$ is a **likelihood ratio** (Radon–Nikodym derivative) that
distorts the reference measure, and the last term is an entropy penalty that
keeps the distortion from being too extreme.

The scalar $\theta_t > 0$
controls the *degree* of concern for misspecification: larger $\theta_t$ means
more pessimism.

The inner minimisation has a closed-form solution:

$$

m_{t+1}^* \;=\;
\frac{\exp(-\theta_t V_{t+1})}{E_t[\exp(-\theta_t V_{t+1})]}.

$$

Since $m_{t+1}^*$ assigns higher weight to states where $V_{t+1}$ is low (bad
outcomes), pessimistic agents effectively over-weight recessions in their
probability assessments.

### Connection to the Belief Wedge

The belief wedge is the expected deviation between subjective and objective
forecasts.

Using $\tilde{E}_t[\cdot] = E_t[m_{t+1}^* \cdot]$:

$$

\Delta_t^{(1)}(z)
= \tilde{E}_t[z_{t+1}] - E_t[z_{t+1}]
= E_t\!\left[m_{t+1}^* z_{t+1}\right] - E_t[z_{t+1}]
= \operatorname{Cov}_t(m_{t+1}^*, z_{t+1}).

$$

So the belief wedge equals the covariance between the distorted likelihood
ratio and the variable of interest.

When $V_{t+1}$ is high in states where
$z_{t+1}$ is also high, $m_{t+1}^*$ will be low in those states, making the
covariance negative — i.e.\ the agent *underestimates* good-state variables.

For unemployment (which varies inversely with good economic outcomes), the
wedge is positive: pessimists expect higher unemployment than the model predicts.

### Illustration: Optimal Belief Distortion

To see the mechanism concretely, consider an **endowment economy** with a
scalar log-consumption state $x_t$ and dynamics

$$

x_{t+1} = \rho_x x_t + \sigma_x w_{t+1}, \qquad w_{t+1} \sim N(0,1).

$$

With log utility, the continuation value is linear: $V_t = V_x x_t + V_q$.


Under the objective measure, $x_{t+1}$ is normal with mean $\rho_x x_t$ and
standard deviation $\sigma_x$.

The distorted measure $m_{t+1}^*$ shifts the mean of $w_{t+1}$ to

$$

\nu_t \;=\; -\theta_t (V_x \sigma_x).

$$

Hence, under the subjective measure, the innovation distribution becomes

$$

w_{t+1} \;\sim\; N\!\left(\nu_t,\; 1\right).

$$

The belief wedge for the state variable $x$ is

$$

\Delta_t^{(1)}(x) \;=\; \sigma_x \nu_t \;=\; -\theta_t V_x \sigma_x^2.

$$

When $V_x > 0$ (good consumption state is good) and $\theta_t > 0$
(pessimism), the wedge is negative — the agent *underestimates*
consumption growth.

For unemployment (enter with a negative sign in the
value function), the same pessimism generates a **positive** wedge.

```{code-cell} ipython3
# ---------------------------------------------------------------------------
# Illustrate the optimal belief distortion in the simple endowment economy.
# ---------------------------------------------------------------------------

class BeliefDistortionModel:
    """
    Simple scalar AR(1) endowment economy illustrating the robust-preference
    mechanism from Bhandari, Borovicka, Ho (2024).

    State dynamics:   x_{t+1} = rho_x * x_t + sigma_x * w_{t+1}
    Period utility:   u(x_t)  = (1 - beta) * x_t  [log utility]
    Continuation value (linearised):  V_t = Vx * x_t + Vq

    Under the distorted measure the shock innovation has mean
        nu_t = -theta_t * Vx * sigma_x
    which produces the belief wedge
        Delta_t^(1)(x) = sigma_x * nu_t = -theta_t * Vx * sigma_x^2.
    """

    def __init__(self, beta=0.994, rho_x=0.85, sigma_x=0.005,
                 mu_theta=5.64, rho_theta=0.714, sigma_theta=4.3):
        self.beta       = beta
        self.rho_x      = rho_x
        self.sigma_x    = sigma_x
        self.mu_theta   = mu_theta
        self.rho_theta  = rho_theta
        self.sigma_theta = sigma_theta
        self.Vx         = self._solve_Vx()

    def _solve_Vx(self):
        """
        Solve the Riccati equation:
            Vx = u_x - (beta/2) * Vx^2 * sigma_x^2 * mu_theta + beta * Vx * rho_x.
        This gives a quadratic in Vx; take the root closest to the RE solution.
        """
        u_x = 1.0 - self.beta        # marginal utility of log consumption

        a = (self.beta / 2.0) * self.sigma_x**2 * self.mu_theta
        b = -(1.0 - self.beta * self.rho_x)
        c = u_x

        # Rational-expectations (theta=0) solution
        Vx_re = u_x / (1.0 - self.beta * self.rho_x)

        if abs(a) < 1e-14:          # essentially no pessimism
            return Vx_re

        disc = b**2 - 4.0 * a * c
        if disc < 0:
            return Vx_re            # fall back to RE if no real root

        r1 = (-b + np.sqrt(disc)) / (2.0 * a)
        r2 = (-b - np.sqrt(disc)) / (2.0 * a)
        return r1 if abs(r1 - Vx_re) < abs(r2 - Vx_re) else r2

    def belief_drift(self, theta):
        """Mean shift of innovations under subjective belief: nu_t = -theta * Vx * sigma_x."""
        return -theta * self.Vx * self.sigma_x

    def belief_wedge(self, theta):
        """One-period belief wedge for the state: Delta = sigma_x * nu_t."""
        return self.sigma_x * self.belief_drift(theta)

    def simulate_theta(self, T=200, seed=42):
        """Simulate the AR(1) belief-shock process."""
        rng   = np.random.default_rng(seed)
        theta = np.zeros(T)
        theta[0] = self.mu_theta
        for t in range(1, T):
            theta[t] = ((1 - self.rho_theta) * self.mu_theta
                        + self.rho_theta * theta[t - 1]
                        + self.sigma_theta * rng.standard_normal())
        return theta

    def simulate(self, T=200, seed=42):
        """Simulate belief wedge time series."""
        theta = self.simulate_theta(T, seed)
        return theta, self.belief_wedge(theta)


model = BeliefDistortionModel()
print(f"RE value of Vx:       {(1-model.beta)/(1-model.beta*model.rho_x):.4f}")
print(f"Robust value of Vx:   {model.Vx:.4f}")
print(f"Belief drift at θ̄:   {model.belief_drift(model.mu_theta)*100:.4f} pp")
print(f"Belief wedge at θ̄:   {model.belief_wedge(model.mu_theta)*100:.4f} pp")
```

```{code-cell} ipython3
# Compare the objective (N(0,1)) and subjective shock distributions.
# The actual model drift nu = -theta * Vx * sigma_x is of order 1e-3 —
# negligible on a unit-shock axis.  We therefore plot the *standardised*
# subjective distribution, rescaling nu by sigma_x so the axis reflects
# the state-change x_{t+1} - rho_x * x_t (in units of sigma_x).

theta_vals  = [0, model.mu_theta, 2 * model.mu_theta]
labels      = ['θ = 0  (rational)',
               f'θ = θ̄ = {model.mu_theta:.1f}  (mean)',
               f'θ = 2θ̄  (pessimistic)']
colors      = ['black', 'steelblue', 'firebrick']

# nu_tilde = nu / sigma_x = -theta * Vx  is the mean shift in units of sigma_x
nu_tilde = [-theta * model.Vx for theta in theta_vals]

x_grid = np.linspace(-4, 4, 500)

fig, ax = plt.subplots(figsize=(9, 4))
for mu, label, color in zip(nu_tilde, labels, colors):
    pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x_grid - mu)**2)
    ax.plot(x_grid, pdf, label=label, color=color, linewidth=2)

ax.axvline(0, color='grey', linestyle=':', linewidth=0.8)
ax.set_xlabel('Standardised innovation  $(w_{t+1} - \\nu_t)$ with $\\nu_t = -\\theta_t V_x \\sigma_x$')
ax.set_ylabel('Density')
ax.set_title('Objective vs. subjective shock distributions\n'
             '(higher θ → leftward shift → pessimism about consumption growth)')
ax.legend()
plt.tight_layout()
plt.show()

print("Mean shifts (in units of σ_x):")
for theta_val, mu, label in zip(theta_vals, nu_tilde, labels):
    print(f"  {label:35s}  ν̃ = {mu:.4f}")
```

The figure shows how pessimism (higher $\theta_t$) shifts the perceived
distribution of future shocks to the left.

An agent with $\theta_t > 0$
believes bad shocks are more likely than they actually are.

## Linear Approximation with Belief Distortions

### The Perturbation Method

For quantitative analysis, {cite}`bhandari2025survey` extend the standard
first-order perturbation method to accommodate time-varying belief distortions.

Let the state vector be $x_t \in \mathbb{R}^n$ with **objective** law of
motion

$$

x_{t+1} = \psi_q + \psi_x x_t + \psi_w w_{t+1}, \qquad
w_{t+1} \sim N(0, I_k).

$$

Under the optimal belief distortion the shocks are re-centred:

$$

w_{t+1} \;\sim\; N\!\left(- \theta_t (\bar{x} + x_{1t})
      (V_x \psi_w)',\; I_k\right),

$$

where $V_x$ is the row vector of first derivatives of the continuation value
and $\bar{x}$ is the non-stochastic steady state.

The perturbation is exact
to first order.

The resulting **belief wedge** for any variable $z$ with model-consistent
expected value $\bar{z}' x$ is

$$

\Delta_t^{(1)}(z)
\;=\; -\theta_t (\bar{x} + x_{1t})\, \bar{z}' (\psi_w \psi_w') V_x'.

$$

### Riccati Equation for $V_x$

The key object is $V_x$, which solves

$$

V_x
\;=\; u_x
  - \frac{\beta}{2}\, V_x \psi_w \psi_w' V_x' \bar\theta
  + \beta\, V_x \psi_x.

$$

This is a modified Riccati equation: the middle term arises from the entropy
penalty on beliefs and vanishes under rational expectations ($\bar\theta = 0$).

### One-Factor Structure

An important consequence of the formula for $\Delta_t^{(1)}(z)$ is that the
*time variation* in all belief wedges is driven by the **single scalar**
$\theta_t$.  The cross-sectional loadings $\bar{z}'(\psi_w\psi_w')V_x'$ are
fixed by the model's structural parameters.

This theoretical prediction
matches the empirical finding that one principal component explains 78.6%
of the joint variation in household forecast errors.

```{code-cell} ipython3
# ---------------------------------------------------------------------------
# Demonstrate the one-factor structure by computing wedges for two
# different variables as theta varies, holding the structural parameters fixed.
# ---------------------------------------------------------------------------

theta_grid = np.linspace(0, 20, 200)

# Loading vector (structural: proportional to bar_z' * psi_w * psi_w' * Vx')
# Calibrated so that at mean theta the steady-state wedges match the data
loading_u  = c_u   # 0.52 / 5.64 pp per unit of θ (unemployment)
loading_pi = c_pi  # 1.22 / 5.64 pp per unit of θ (inflation)

wedge_u_grid  = loading_u  * theta_grid
wedge_pi_grid = loading_pi * theta_grid

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(theta_grid, wedge_u_grid, color='steelblue', linewidth=2,
             label='$\\Delta^{(1)}(u)$')
axes[0].plot(theta_grid, wedge_pi_grid, color='darkorange', linewidth=2,
             label='$\\Delta^{(1)}(\\pi)$')
axes[0].axvline(mu_theta, color='grey', linestyle='--', linewidth=0.9,
                label=f'$\\bar{{\\theta}} = {mu_theta}$')
axes[0].set_xlabel('Belief-shock level $\\theta$')
axes[0].set_ylabel('Belief wedge (pp)')
axes[0].set_title('Wedges are linear in $\\theta_t$\n(one-factor structure)')
axes[0].legend()

# Scatter of (wedge_u, wedge_pi) with theta as the driver
theta_sim = model.simulate_theta(T=400, seed=7)
wu_sim  = loading_u  * theta_sim
wpi_sim = loading_pi * theta_sim
axes[1].scatter(wu_sim, wpi_sim, c=theta_sim, cmap='Blues', alpha=0.6, s=12)
axes[1].set_xlabel('Unemployment wedge (pp)')
axes[1].set_ylabel('Inflation wedge (pp)')
axes[1].set_title('Both wedges driven by the same factor $\\theta_t$')

plt.tight_layout()
plt.show()
```

## A New Keynesian Model with Belief Distortions

### Model Description

{cite}`bhandari2025survey` embed the belief-distortion mechanism in a
New Keynesian model with a **search-and-matching** labour market
({cite}`Shimer2005`; {cite}`ChristianoEichenbaumTrabandt2016`).

The key
components are:

**Households** — Have log utility in consumption and disutility of hours.
They apply robust preferences (indexed by $\theta_t$) when forming
subjective forecasts.

**Firms** — Post vacancies and match with workers.  Calvo-style price
stickiness (parameter $\chi_p$) and wage stickiness ($\chi_w$) generate
standard New Keynesian Phillips curves.

**Monetary policy** — A Taylor rule that reacts to inflation and the output gap.

**Exogenous shocks** — Three shocks drive the model:

1. **Belief shock** $\theta_t$: an AR(1) capturing time-varying pessimism.
2. **TFP shock** $a_t$: standard technology shock.
3. **Monetary policy shock** $r_t$: i.i.d.\ deviation from the Taylor rule.

### Calibration

The model is calibrated to quarterly U.S. data, 1982Q1–2019Q4.

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Discount factor | $\beta$ | 0.994 | Quarterly |
| Elast. of substitution | $\varepsilon$ | 6 | Price markup |
| Price stickiness | $\chi_p$ | 0.75 | Calvo parameter |
| Wage stickiness | $\chi_w$ | 0.925 | Calvo parameter |
| Mean pessimism | $\mu_\theta$ | 5.64 | |
| Persistence of $\theta$ | $\rho_\theta$ | 0.714 | |
| Volatility of $\theta$ shock | $\sigma_\theta$ | 4.3 | |
| TFP persistence | $\rho_a$ | 0.840 | |
| TFP volatility | $100\sigma_a$ | 0.568% | |
| MP volatility | $100\sigma_r$ | 0.078% | |
| Matching elasticity | $\eta$ | 0.72 | Hosios condition |
| Worker bargaining | $\mu$ | 0.67 | |
| Job-separation rate | $\rho$ | 0.89 | Quarterly |

### Simplified Reduced-Form Representation

We capture the model's linearised solution through a reduced-form
vector autoregression

$$

s_{t+1} = A\, s_t + B\, \epsilon_{t+1},

$$

where $s_t = (u_t, \pi_t, y_t, \theta_t, a_t)'$ collects unemployment,
inflation, output, the belief shock, and TFP, and
$\epsilon_{t+1} \sim N(0, I_3)$ contains the three structural shocks.

The coefficient matrices $A$ and $B$ are calibrated so that the
impulse-response functions reproduce the key moments reported in Table 2 and
Figure 7 of {cite}`bhandari2025survey`.

```{code-cell} ipython3
class ReducedFormNKModel:
    """
    Reduced-form linear model calibrated to Bhandari, Borovicka, Ho (2024).

    State vector s_t = [u_t, pi_t, y_t, theta_t, a_t]
    Shocks: epsilon = [w_theta, w_a, w_r]

    Belief wedges:
        Delta_u  = c_u  * theta_t
        Delta_pi = c_pi * theta_t
    """

    # Index map for the state vector
    I_U, I_PI, I_Y, I_THETA, I_A = 0, 1, 2, 3, 4

    def __init__(self):
        # ---- exogenous-process parameters (Table 1) ----
        self.rho_theta   = 0.714
        self.sigma_theta = 4.3
        self.rho_a       = 0.840
        self.sigma_a     = 0.00568
        self.sigma_r     = 0.00078

        # ---- wedge loadings on theta ----
        self.c_u  = 0.52 / 5.64
        self.c_pi = 1.22 / 5.64

        # ---- calibrated impact effects ----
        # State variables are stored in FRACTIONS (e.g. u=0.06 for 6%).
        # Display code converts: *100 for u,y (→ pp/%); *400 for pi (→ ann. pp).
        #
        # Belief shock targets (Figure 7, 1-std-dev = sigma_theta = 4.3 units):
        #   u: +0.90 pp → phi_u_th × 4.3 × 100 = 0.90  → phi_u_th = 0.009/4.3
        #   pi: +0.35 pp ann → quarterly frac 0.000875  → phi_pi_th = 0.000875/4.3
        #   y: −0.90%       → phi_y_th = −0.009/4.3
        phi_u_th  =  0.009    / self.sigma_theta
        phi_pi_th =  0.000875 / self.sigma_theta
        phi_y_th  = -0.009    / self.sigma_theta

        # TFP shock (sigma_a = 0.00568 fraction):
        #   u: ~ −0.23 pp peak → phi_u_a × sigma_a × 100 ≈ −0.23
        #   pi: ~ −0.23 pp ann peak → phi_pi_a × sigma_a × 400 ≈ −0.23
        #   y: ~ +0.68% peak → phi_y_a × sigma_a × 100 ≈ +0.68
        phi_u_a  = -0.40
        phi_pi_a = -0.10
        phi_y_a  =  1.20

        # Persistence of endogenous variables (quarterly, reduced-form)
        rho_u  = 0.35
        rho_pi = 0.50
        rho_y  = 0.35

        # ---- state transition matrix ----
        self.A = np.array([
            [rho_u,  0,      0,     phi_u_th,  phi_u_a ],   # unemployment
            [0,      rho_pi, 0,     phi_pi_th, phi_pi_a],   # inflation
            [0,      0,      rho_y, phi_y_th,  phi_y_a ],   # output
            [0,      0,      0,     self.rho_theta, 0   ],   # belief shock
            [0,      0,      0,     0,         self.rho_a],  # TFP
        ])

        # ---- shock loading matrix ----
        # Columns: [w_theta, w_a, w_r].  All entries in fraction units.
        self.B = np.array([
            [0,                0,             0.5e-3 ],   # MP → u fraction
            [0,                0,            -0.1e-3 ],   # MP → pi fraction
            [0,                0,            -0.5e-3 ],   # MP → y fraction
            [self.sigma_theta, 0,             0      ],   # theta innovation
            [0,                self.sigma_a,  0      ],   # TFP innovation
        ])

    def irf(self, shock_idx, T=25):
        """
        Impulse-response function for a one-std-dev shock.

        Parameters
        ----------
        shock_idx : int
            0 = belief shock, 1 = TFP shock, 2 = monetary policy shock
        T : int
            Number of periods

        Returns
        -------
        resp : ndarray (5, T)   responses of state vector
        wu   : ndarray (T,)     unemployment wedge response
        wpi  : ndarray (T,)     inflation wedge response
        """
        n = self.A.shape[0]
        resp = np.zeros((n, T))
        s = self.B[:, shock_idx].copy()   # impact response

        for t in range(T):
            resp[:, t] = s
            s = self.A @ s

        wu  = self.c_u  * resp[self.I_THETA, :]
        wpi = self.c_pi * resp[self.I_THETA, :]
        return resp, wu, wpi

    def simulate(self, T=200, seed=42):
        """Simulate the model for T periods."""
        rng     = np.random.default_rng(seed)
        n, k    = self.A.shape[0], self.B.shape[1]
        s       = np.zeros((n, T))
        for t in range(1, T):
            s[:, t] = self.A @ s[:, t-1] + self.B @ rng.standard_normal(k)
        return s

    def unconditional_stds(self, include_theta_shock=True):
        """
        Unconditional standard deviations computed from the Lyapunov equation.
        """
        B_use = self.B.copy()
        if not include_theta_shock:
            B_use[:, 0] = 0.0          # zero out the belief shock
        Sigma = solve_discrete_lyapunov(self.A, B_use @ B_use.T)
        return np.sqrt(np.diag(Sigma))


nk = ReducedFormNKModel()
```

## Quantitative Results

### Impulse Responses to the Belief Shock

A positive innovation to $\theta_t$ makes households more pessimistic.

The
mechanism works this way:

1. Pessimistic households expect worse future outcomes and reduce consumption
   demand.
2. Lower demand raises unemployment and reduces output.
3. Upward wage pressure from labour-market tightness feeds into inflation.
4. The belief wedges jump on impact, then decay with the persistence
   $\rho_\theta = 0.714$.

```{code-cell} ipython3
T_irf = 25
periods = np.arange(T_irf)

resp_theta, wu_theta, wpi_theta = nk.irf(shock_idx=0, T=T_irf)

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()

titles = ['Unemployment (pp)', 'Inflation (pp, ann.)', 'Output (%)',
          'Belief shock θ', 'Unemp. wedge Δ(u) (pp)',
          'Inflation wedge Δ(π) (pp)']
series = [resp_theta[0] * 100,   # unemployment in pp  (fraction × 100)
          resp_theta[1] * 400,   # inflation ann. pp  (quarterly frac × 400)
          resp_theta[2] * 100,   # output in %        (fraction × 100)
          resp_theta[3],         # belief shock θ (theta units)
          wu_theta,              # unemp. wedge (pp): c_u × θ, already in pp
          wpi_theta]             # infl. wedge  (pp): c_pi × θ, already in pp
colors = ['steelblue'] * 3 + ['purple', 'steelblue', 'darkorange']

for ax, title, y, color in zip(axes, titles, series, colors):
    ax.plot(periods, y, color=color, linewidth=2)
    ax.axhline(0, color='grey', linewidth=0.7, linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Quarters')

fig.suptitle("Impulse responses to a one-std-dev belief shock",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
```

The impulse responses show that a belief shock:

* Raises unemployment persistently (peak effect around 1 pp).
* Raises inflation on impact, as higher pessimism tightens labour markets
  in the model.
* Generates belief wedges for both unemployment and inflation that closely
  mirror the dynamics of $\theta_t$ itself — consistent with the one-factor
  structure.

### The Unemployment Volatility Puzzle

A long-standing challenge for New Keynesian models is that standard TFP and
monetary policy shocks generate far too little unemployment volatility
({cite}`Shimer2005`).

With only TFP and monetary policy shocks, the model
produces unemployment volatility of roughly 0.55%, compared to about 1.70%
in the data.

Adding the belief shock substantially closes the gap:

```{code-cell} ipython3
std_full    = nk.unconditional_stds(include_theta_shock=True)
std_no_theta = nk.unconditional_stds(include_theta_shock=False)

labels_vol = ['Unemployment', 'Inflation', 'Output']
idx        = [nk.I_U, nk.I_PI, nk.I_Y]
scale      = [100, 400, 100]    # convert to pp (unemployment, annualised inflation, %)

std_full_scaled    = [std_full[i]     * scale[j] for j, i in enumerate(idx)]
std_no_theta_scaled = [std_no_theta[i] * scale[j] for j, i in enumerate(idx)]

# Reference values from Table 2 of the paper
data_std = [1.70, 1.07, 2.23]    # data standard deviations

x = np.arange(len(labels_vol))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - width, std_no_theta_scaled, width, label='Model (no belief shock)',
            color='steelblue', alpha=0.7)
b2 = ax.bar(x,          std_full_scaled,    width, label='Model (with belief shock)',
            color='firebrick', alpha=0.7)
b3 = ax.bar(x + width,  data_std,           width, label='Data',
            color='grey', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(labels_vol)
ax.set_ylabel('Standard deviation (%  or  pp, ann.)')
ax.set_title('Belief shocks substantially raise unemployment volatility')
ax.legend()
plt.tight_layout()
plt.show()

print("Unconditional standard deviations:")
print(f"{'Variable':<18} {'No belief shock':>16} {'With belief shock':>18} {'Data':>10}")
print('-' * 65)
for label, std_n, std_f, std_d in zip(labels_vol, std_no_theta_scaled,
                                       std_full_scaled, data_std):
    print(f"{label:<18} {std_n:>16.2f} {std_f:>18.2f} {std_d:>10.2f}")
```

The table confirms the key quantitative message: without the belief shock,
unemployment volatility is far below its empirical counterpart, but adding
the calibrated belief shock nearly doubles it, bringing the model much closer
to the data.

### Impulse Responses to TFP and Monetary Policy Shocks

For completeness, we also show responses to the other two shocks.

```{code-cell} ipython3
resp_a,  _, _ = nk.irf(shock_idx=1, T=T_irf)   # TFP shock
resp_r,  _, _ = nk.irf(shock_idx=2, T=T_irf)   # Monetary policy shock

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()

for col, (resp, shock_name) in enumerate(
        [(resp_a, 'TFP'), (resp_r, 'Monetary policy')]):
    # We show three variables per shock
    pass   # handled below by row/col indexing

series_a = [resp_a[0]*100, resp_a[1]*400, resp_a[2]*100]
series_r = [resp_r[0]*100, resp_r[1]*400, resp_r[2]*100]
var_names = ['Unemployment (pp)', 'Inflation (pp, ann.)', 'Output (%)']

for j, (vname, ya, yr) in enumerate(zip(var_names, series_a, series_r)):
    # TFP
    axes[j].plot(periods, ya, color='steelblue', linewidth=2,
                 label='TFP shock')
    axes[j].axhline(0, color='grey', linewidth=0.7, linestyle='--')
    axes[j].set_title(f'{vname} — TFP shock')
    axes[j].set_xlabel('Quarters')

    # Monetary policy
    axes[j+3].plot(periods, yr, color='darkorange', linewidth=2,
                   label='MP shock')
    axes[j+3].axhline(0, color='grey', linewidth=0.7, linestyle='--')
    axes[j+3].set_title(f'{vname} — MP shock')
    axes[j+3].set_xlabel('Quarters')

plt.tight_layout()
plt.show()
```

### Role of Firms' Beliefs

{cite}`bhandari2025survey` also study a variant in which **firms** hold
subjective beliefs.

The key channel is through the price-setting equation:
when firms fear that future demand will be weaker than the model predicts,
they raise prices today to protect margins, generating **higher inflation** on
impact.

This mechanism strengthens the comovement between the unemployment
wedge and the inflation wedge, which is needed to match the data.

The sign of the inflation response to a belief shock is therefore a
diagnostic: positive responses to pessimistic shocks require firms (not just
households) to hold subjective beliefs.

### Countercyclicality of Wedges

A final important prediction is that belief wedges are countercyclical.

Recessions are periods of high $\theta_t$, which raises both the unemployment
wedge and the inflation wedge simultaneously.

The code below simulates a
long run of the model and shows this property:

```{code-cell} ipython3
sim = nk.simulate(T=400, seed=99)
u_sim     = sim[nk.I_U]   * 100
theta_sim = sim[nk.I_THETA]
y_sim     = sim[nk.I_Y]   * 100

# c_u and c_pi are in pp per unit θ, so the wedge is already in pp
wu_sim_series  = nk.c_u  * theta_sim
wpi_sim_series = nk.c_pi * theta_sim

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

axes[0].plot(y_sim, color='steelblue', linewidth=1.5, label='Output gap (%)')
axes[0].axhline(0, color='grey', linestyle='--', linewidth=0.7)
axes[0].set_ylabel('%')
axes[0].legend(loc='upper right')

axes[1].plot(wu_sim_series, color='firebrick', linewidth=1.5,
             label='Unemployment belief wedge (pp)')
axes[1].set_ylabel('pp')
axes[1].legend(loc='upper right')

axes[2].plot(wpi_sim_series, color='darkorange', linewidth=1.5,
             label='Inflation belief wedge (pp)')
axes[2].set_ylabel('pp')
axes[2].legend(loc='upper right')
axes[2].set_xlabel('Quarter')

fig.suptitle('Simulated model: wedges are high when output is low',
             fontsize=13)
plt.tight_layout()
plt.show()

# Confirm countercyclicality numerically
corr_u  = np.corrcoef(y_sim, wu_sim_series)[0, 1]
corr_pi = np.corrcoef(y_sim, wpi_sim_series)[0, 1]
print(f"Corr(output gap, unemployment wedge) = {corr_u:.3f}  "
      f"(data: −0.49)")
print(f"Corr(output gap, inflation wedge)    = {corr_pi:.3f}  "
      f"(data: −0.30)")
```

The simulated correlations are negative, confirming the countercyclicality
predicted by the model and documented in the survey data.

## Extensions

The paper explores several important extensions:

**Heterogeneous beliefs** — A natural question is whether households and
firms should hold the same subjective beliefs.  The paper shows that
allowing firms to be *rational* while households are pessimistic changes
the inflation dynamics substantially.  This separation is identified from
the relative sizes of the unemployment and inflation wedges.

**Higher-order perturbation** — The first-order approximation provides
clean analytical formulas for belief wedges, but second-order effects
(precautionary savings, volatility feedback) matter for welfare analysis.
The paper develops second-order expansions and shows they affect the wedge
levels but not the one-factor structure.

**Idiosyncratic risk** — In the full model households face idiosyncratic
labour-market risk.  The interaction between aggregate pessimism and
uninsurable idiosyncratic shocks amplifies the effect of belief distortions
on precautionary savings, strengthening the unemployment channel.

## Appendix: The Series Expansion Method

This appendix follows the Online Appendix of {cite}`BhandariBorovickaHo2024`
and fills in the computational and theoretical details underlying the
linearisation presented in the main lecture.

### Multi-Period Belief Wedges

The main text focused on the one-period belief wedge
$\Delta_t^{(1)}(z)$.
The paper also uses $\tau$-period-ahead wedges
$\Delta_t^{(\tau)}(z) = \tilde E_t[z_{t+\tau}] - E_t[z_{t+\tau}]$,
which are needed to match survey respondents' longer-horizon forecasts.

Under linear dynamics

$$

x_{t+1} = \psi_q + \psi_x x_t + \psi_w w_{t+1},
\qquad w_{t+1} \sim N(0, I_k),

$$

the $\tau$-period-ahead expectation under the data-generating measure
satisfies the recursion

$$

G_x^{(\tau)} = \psi_x G_x^{(\tau-1)} + \psi_x,
\qquad
G_x^{(0)} = 0,

$$

so that $E_t[x_{t+\tau} - x_t] = G_x^{(\tau)} x_{1t} + G_0^{(\tau)}$.

Under the **subjective** measure, the mean of $w_{t+1}$ is shifted to
$\nu_t = H + HF x_{1t}$ (equation OA.1 of the appendix).  For the
stationary model the relevant identifications are

$$

F = \bar\theta,
\qquad
H = -(V_x \psi_w)',
\qquad
\bar H = -\bar\theta\,\bar x\,(V_x \psi_w)'.

$$

The subjective expectation then obeys a modified recursion

$$

\tilde G_x^{(\tau)} = \psi_x \tilde G_x^{(\tau-1)} + \psi_x
  + \bigl(\psi_w + \tilde G_x^{(\tau-1)}\psi_w\bigr) HF,

$$

and the $\tau$-period belief wedge is

$$

\Delta_t^{(\tau)} = \bigl(\tilde G_x^{(\tau)} - G_x^{(\tau)}\bigr) x_{1t}
  + \tilde G_0^{(\tau)} - G_0^{(\tau)}.

$$

The code below implements these recursions and shows how belief wedges grow
with the forecast horizon.

```{code-cell} ipython3
def compute_tau_wedge_loadings(psi_x, psi_w, H, H_bar, F, tau_max=20):
    """
    Compute tau-period belief wedge loadings using the recursions from
    Online Appendix OA.1 of Bhandari, Borovicka, Ho (2024).

    For simplicity we work with the scalar stationary case (all quantities
    are scalars or 1-d arrays).

    Returns
    -------
    wedge_const : array (tau_max,)   constant term of wedge  (G0_tilde - G0)
    wedge_slope : array (tau_max,)   x1t loading of wedge    (Gx_tilde - Gx)
    """
    n = psi_x.shape[0]
    Gx      = np.zeros((n, n))
    Gx_tild = np.zeros((n, n))
    G0      = np.zeros(n)
    G0_tild = np.zeros(n)

    wedge_const = np.zeros(tau_max)
    wedge_slope = np.zeros((tau_max, n))

    for tau in range(1, tau_max + 1):
        # data-generating measure
        new_Gx = (Gx + np.eye(n)) @ psi_x
        new_G0 = G0 + (Gx + np.eye(n)) @ psi_w @ np.zeros(psi_w.shape[1])
        # (constant-shock term is zero under objective measure)

        # subjective measure
        new_Gx_tild = ((Gx_tild + np.eye(n)) @ psi_x
                       + (Gx_tild + np.eye(n)) @ psi_w @ (H @ F))
        new_G0_tild = (G0_tild
                       + (Gx_tild + np.eye(n)) @ psi_w @ H_bar)

        Gx, G0         = new_Gx, new_G0
        Gx_tild, G0_tild = new_Gx_tild, new_G0_tild

        wedge_slope[tau - 1] = (Gx_tild - Gx)[0]
        wedge_const[tau - 1] = (G0_tild - G0)[0]

    return wedge_const, wedge_slope
```

```{code-cell} ipython3
# -----------------------------------------------------------------
# Illustrate the wedge horizon profile in the simple endowment economy
# using the solved BeliefDistortionModel.
# -----------------------------------------------------------------

# Scalar model: psi_x = [[rho_x]], psi_w = [[sigma_x]]
psi_x_sc = np.array([[model.rho_x]])
psi_w_sc = np.array([[model.sigma_x]])
F_sc     = np.array([[model.mu_theta]])           # theta-bar
H_sc     = np.array([[-model.Vx * model.sigma_x]])  # -(Vx psi_w)'
H_bar_sc = model.mu_theta * model.rho_x * np.array([[-model.Vx * model.sigma_x]])

tau_max = 20
wc, ws = compute_tau_wedge_loadings(psi_x_sc, psi_w_sc, H_sc, H_bar_sc, F_sc, tau_max)

# Steady-state wedge at mean theta (x1t = 0 at the mean)
wedge_at_mean = wc + ws[:, 0] * 0    # x1t = 0 => only constant term
# For illustration, evaluate at x1t = +1 std dev of theta
theta_std = model.sigma_theta / np.sqrt(1 - model.rho_theta**2)

fig, ax = plt.subplots(figsize=(9, 4))
taus = np.arange(1, tau_max + 1)
ax.plot(taus, wc * 100,
        color='steelblue', linewidth=2, label='Wedge at mean ($x_{1t}=0$)')
ax.plot(taus, (wc + ws[:, 0] * theta_std) * 100,
        color='firebrick', linewidth=2, linestyle='--',
        label='Wedge at $+1\\,\\sigma_\\theta$ deviation')
ax.axhline(0, color='grey', linewidth=0.7, linestyle=':')
ax.set_xlabel('Forecast horizon $\\tau$ (quarters)')
ax.set_ylabel('Belief wedge (pp)')
ax.set_title('$\\tau$-period belief wedge in the endowment economy\n'
             '(higher horizon → persistent pessimism)')
ax.legend()
plt.tight_layout()
plt.show()
```

### The Series Expansion

{cite}`BhandariBorovickaHo2024` solve the full general-equilibrium model
using a **series expansion** (perturbation) method
({cite}`BorovickaHansen2014`).  The key innovation is a **joint
perturbation** of the shock volatility $q$ and the penalty parameter
$\theta_t$.

#### Law of Motion

Index the model by a scalar perturbation parameter $\mathsf{q}$ that
scales shock volatility:

$$

x_{t+1}(\mathsf{q}) = \psi\!\left(x_t(\mathsf{q}),\,
  \mathsf{q}\, w_{t+1},\, \mathsf{q}\right).

$$

Expanding around $\mathsf{q} = 0$ gives

$$

x_t(\mathsf{q}) \approx \bar x + \mathsf{q}\, x_{1t}
  + \tfrac{\mathsf{q}^2}{2}\, x_{2t} + \cdots

$$

The first-order dynamics are

$$

x_{1,t+1} = \psi_q + \psi_x x_{1t} + \psi_w w_{t+1}.

$$

#### Continuation Value and the Riccati Equation

To preserve a nontrivial role for beliefs at first order, the penalty
parameter is **jointly scaled** with $\mathsf{q}$: the effective
penalisation in the perturbed recursion (OA.8) is
$\mathsf{q}/[\bar\theta(\bar x + x_{1t})]$,
which shrinks together with shock volatility.  This ensures that the
deterministic steady state does not collapse to the rational-expectations
solution.

Guessing $V_{1t} = V_x x_{1t} + V_q$ and matching coefficients yields
the **Riccati equation for $V_x$** (equation OA.20 of the appendix):

$$

V_x = u_x - \frac{\beta}{2}\, V_x \psi_w \psi_w' V_x' \bar\theta
  + \beta\, V_x \psi_x,

$$

and the constant

$$

V_q = u_q - \frac{\beta}{2}\,\bar\theta\, \bar x\,
  V_x \psi_w \psi_w' V_x' + \beta\, V_x \psi_q + \beta V_q.

$$

The Riccati equation is quadratic in $V_x$.  For the stationary scalar case it
reduces to

$$

a\, V_x^2 + b\, V_x + c = 0,
\qquad
a = \frac{\beta}{2}\sigma_x^2 \bar\theta,\quad
b = -(1 - \beta\rho_x),\quad
c = u_x.

$$

#### Shock Distribution under Subjective Beliefs

Substituting the first-order expansion into the distortion formula
(OA.10) shows that the leading term $m_{0,t+1}$ is a lognormal change of
measure.  With Gaussian shocks, this is equivalent to shifting the
innovation mean (equation OA.12):

$$

w_{t+1} \;\sim\;
N\!\left(-\bar\theta(\bar x + x_{1t})(V_x \psi_w)',\; I_k\right).

$$

Belief wedges for the state vector follow immediately:

$$

\Delta_t^{(1)} = \tilde E_t[x_{t+1}] - E_t[x_{t+1}]
= \psi_w\, \tilde E_t[w_{t+1}]
= -\bar\theta(\bar x + x_{1t})(\psi_w \psi_w') V_x'.

$$

#### Equilibrium Conditions with Subjective Beliefs

The full model's equilibrium conditions take the form

$$

0 = E_t\!\left[\mathbb{M}_{t+1}\, g(x_{t+1}, x_t, x_{t-1}, w_{t+1}, w_t)\right],

$$

where $\mathbb{M}_{t+1} = \mathrm{diag}(m_{t+1}^{\sigma_1}, \ldots,
m_{t+1}^{\sigma_n})$ selects which equations involve subjective
expectations ($\sigma_i = 1$) versus objective ones ($\sigma_i = 0$).

First-order expansion of these conditions gives a system in the unknown
policy matrices $\psi_x, \psi_w, \psi_q$:

$$

0 = (g_{x^+}\psi_x + g_x - \mathbb{E})\,\psi_x + g_{x^-}

$$

$$

0 = (g_{x^+}\psi_x + g_x - \mathbb{E})\,\psi_w + g_w

$$

$$

0 = (g_{x^+}\psi_x + g_{x^+} + g_x)\,\psi_q + g_q
  - \mathbb{E}(\bar x + \psi_q),

$$

where the **belief distortion matrix** $\mathbb{E}$ collects the impact
of subjective expectations on each equation:

$$

\mathbb{E} = \operatorname{stack}\Bigl\{
  \sigma_i\, [g_{x^+}\psi_w + g_{w^+}]^i\,
  (V_x \psi_w)'\, \bar\theta
\Bigr\}.

$$

These equations (OA.17–OA.21) are solved jointly with the Riccati
equation for $V_x$.  Compared with the standard Blanchard–Kahn solution,
the only modification is the additive term $-\mathbb{E}$ that shifts the
characteristic matrix; when $\bar\theta = 0$ we recover the standard
rational-expectations solution.

#### The AR(1) Belief Shock as a Special Case

In the paper's application $\theta_t$ is itself an exogenous AR(1)
process (equation OA.22):

$$

f_{t+1} = (1 - \rho_f)\bar f + \rho_f f_t + \sigma_f w_{t+1}^f.

$$

Appending $f_t$ to the state vector, the first-order dynamics become

$$

\begin{pmatrix} x_{1,t+1} \\ f_{1,t+1} \end{pmatrix}
= \begin{pmatrix} \psi_q \\ 0 \end{pmatrix}
+ \begin{pmatrix} \psi_x & \rho_f \psi_{xf} \\ 0 & \rho_f \end{pmatrix}
\begin{pmatrix} x_{1t} \\ f_{1t} \end{pmatrix}
+ \begin{pmatrix} \psi_w & \sigma_f \psi_{xf} \\ 0 & \sigma_f \end{pmatrix}
\begin{pmatrix} w_{t+1} \\ w_{t+1}^f \end{pmatrix}.

$$

The new coefficient $\psi_{xf}$ measures how a unit change in the belief
shock $f_{1t}$ feeds into the endogenous state variables.  It is determined
by the backward-induction algorithm (equations OA.31–OA.34), which iterates
from a distant terminal date $T$ (where belief distortions vanish) back to
the present.

The continuation value in the $f$-direction satisfies a separate recursion
for $V_f$ (equation OA.29), and the belief distortion matrix becomes

$$

\mathbb{E} = \operatorname{stack}\Bigl\{
  \sigma_i\bigl[
    g_{x^+}\psi_{xf}\sigma_f^2(V_f + V_x\psi_{xf})
    + (g_{x^+}\psi_w + g_{w^+})\psi_w' V_x'
  \bigr]^i
\Bigr\}\bar\theta_f.

$$

The algorithm therefore decomposes cleanly into two stages:

1. **Stage 1 (rational-expectations block)**: solve (OA.24) and (OA.26) for
   $\psi_x$, $\psi_w$ using the standard Blanchard–Kahn method — these are
   *unaffected* by the belief shock.

2. **Stage 2 (belief distortion block)**: given $\psi_x, \psi_w, V_x$,
   iterate (OA.31–OA.34) backward to convergence to find $\psi_{xf}$,
   $V_f$, and $\mathbb{E}$.

This separation is a major practical advantage: existing rational-expectations
solvers can be used for Stage 1 with only a wrapper for Stage 2.

```{code-cell} ipython3
# -----------------------------------------------------------------
# Demonstrate Stage 2 (backward induction for psi_xf and Vf) in a
# stylised scalar economy.
#
# Setup:
#   - Endogenous state x, belief shock f (= theta_t)
#   - psi_x (1x1), psi_w (1x1) known from Stage 1
#   - Vx known from the Riccati equation
#   - Solve for Vf and psi_xf by iterating backward from T -> 0
# -----------------------------------------------------------------

beta      = model.beta
rho_x     = model.rho_x
sigma_x   = model.sigma_x
rho_f     = model.rho_theta
sigma_f   = model.sigma_theta
Vx        = model.Vx

# Stage 1 objects (RE solution)
psi_x_s1 = rho_x
psi_w_s1 = sigma_x

# gx+ = beta * (1 - beta) in the simple log-utility endowment economy
# (partial derivative of marginal utility w.r.t. x_{t+1})
gx_plus  = beta * (1 - beta)

# Backward induction from T = large
T_bwd   = 500
Vf      = 0.0
psi_xf  = 0.0

theta_f = 1.0   # theta_f = 1 since f directly IS theta (partitioned as (0, 1))

for _ in range(T_bwd):
    # Equation (OA.33)
    E_next = (gx_plus * psi_xf * sigma_f
              * (Vf * sigma_f + Vx * psi_xf * sigma_f)
              + (gx_plus * psi_w_s1) * psi_w_s1 * Vx) * theta_f

    # Equation (OA.32): update Vf
    Vf_new = (- (beta * theta_f / 2.0)
              * ((Vx * psi_w_s1)**2 + (Vx * psi_xf * sigma_f + Vf * sigma_f)**2)
              + beta * rho_f * (Vf + Vx * psi_xf))

    # Equation (OA.34): update psi_xf
    # (gx+ * psi_x + gx)^{-1} * (E_next - gx+ * psi_xf_prev * rho_f)
    # In the scalar endowment economy gx = -(1-beta) and gx+ * psi_x = gx_plus * psi_x_s1
    # The policy-function derivative is obtained from the linearised Euler equation
    psi_xf_new = (E_next - gx_plus * psi_xf * rho_f) / (gx_plus * psi_x_s1 - (1 - beta))

    Vf     = Vf_new
    psi_xf = psi_xf_new

print("Converged Stage 2 solution:")
print(f"  Vf      = {Vf:.6f}")
print(f"  psi_xf  = {psi_xf:.6f}  (impact of belief shock on endogenous state)")
print()
print("Interpretation: a one-unit rise in f_t changes x by psi_xf =",
      f"{psi_xf:.4f} per period.")
print("The steady-state wedge for x: Delta = psi_w * nu_bar =",
      f"{sigma_x * (-model.mu_theta * (Vx * sigma_x)):.4f}")
```

### Sequence Problem and Dynamic Consistency

The recursive formulation used throughout the lecture emerges from the
following sequence problem (Online Appendix OA.3).  Define the discounted
entropy functional

$$

\mathcal{E}_t \;=\; E_t \sum_{j=0}^{\infty} \beta^j
  \left[ M_{t,t+j} \frac{\beta}{\theta_{t+j}}
    E_{t+j}[m_{t+j+1} \log m_{t+j+1}]
  \right],

$$

where $M_{t,t+j} = \prod_{k=1}^j m_{t+k}$.  The agent's problem is

$$

V_t^* = \max_{\{y_{t+j}\}} \min_{\substack{m_{t+j}>0 \\ E_{t+j-1}[m_{t+j}]=1}}
  \sum_{j=0}^{\infty} \beta^j E_t[M_{t,t+j} u_{t+j}]
  + \mathcal{E}_t.

$$

The penalty functional $\mathcal{E}_t$ **discounts future entropies
weighted by future penalty parameters $\theta_{t+j}$**, which makes the
agent's choices dynamically consistent: she anticipates how her
pessimism will evolve.

This differs from the infinite-horizon discounted entropy used in
{cite}`HansenSargent2001`, which is not generally dynamically consistent
when $\theta_t$ is time-varying.  The recursive form is:

$$

\mathcal{E}_t = \frac{\beta}{\theta_t} E_t[m_{t+1} \log m_{t+1}]
  + \beta E_t[m_{t+1} \mathcal{E}_{t+1}].

$$

Under this penalty, the minimax inequality is an equality, and the value
function satisfies the recursive form stated in the main lecture:

$$

V_t^* = \max_{y_t} \min_{\substack{m_{t+1}>0 \\ E_t[m_{t+1}]=1}}
  u_t + \frac{\beta}{\theta_t} E_t[m_{t+1} \log m_{t+1}]
  + E_t[m_{t+1} V_{t+1}^*].

$$

```{code-cell} ipython3
# -----------------------------------------------------------------
# Illustrate the role of dynamic consistency by comparing two penalty
# specifications:
#   (a) The paper's specification: E_t = (beta/theta_t) * H_t + beta * E[m*E_{t+1}]
#       where H_t = E_t[m_{t+1} log m_{t+1}]
#   (b) A myopic version that uses only the one-period entropy:
#       E_t^{myopic} = (beta/theta_t) * H_t
#
# We compare the implied belief wedge as theta varies.
# -----------------------------------------------------------------

theta_path = np.array([3.0, 5.64, 8.0, 12.0])   # rising pessimism scenario

def one_period_entropy(theta, Vx, sigma_x):
    """
    Entropy E_t[m_{t+1} log m_{t+1}] under the optimal distortion
    for Gaussian shocks: = (1/2) * (theta * Vx * sigma_x)^2.
    """
    return 0.5 * (theta * Vx * sigma_x) ** 2

print("Effect of time-varying θ on entropy and belief wedge:")
print(f"{'θ_t':>8}  {'H_t (entropy)':>16}  {'Δ(x) = σ_x ν_t (pp)':>22}")
print('-' * 52)
for th in theta_path:
    H  = one_period_entropy(th, model.Vx, model.sigma_x)
    bw = model.belief_wedge(th) * 100
    print(f"{th:>8.2f}  {H:>16.6f}  {bw:>22.4f}")

print()
print("The entropy penalty grows quadratically in θ,")
print("constraining the agent from distorting beliefs too heavily.")
```

## Exercises

```{exercise}
:label: sbbc_ex1

**Belief wedge sign**

In the simple endowment economy of the `BeliefDistortionModel`, suppose the state
variable is log consumption $x_t$ with $\rho_x = 0.90$, $\sigma_x = 0.01$,
$\beta = 0.99$.

(a) Compute $V_x$ under rational expectations and under pessimism
    $\mu_\theta = 4$.
(b) What is the sign of the belief wedge for consumption growth?
(c) If instead the agent forecasts unemployment (which enters the value
    function with a negative sign, so $u_x < 0$), what is the sign of the
    unemployment belief wedge?
```

```{solution-start} sbbc_ex1
:label: sbbc_ex1_sol
:class: dropdown
```

**Part (a)** — Under rational expectations ($\theta = 0$):

$$

V_x^{RE} = \frac{u_x}{1 - \beta \rho_x}
         = \frac{1 - \beta}{1 - \beta \rho_x}.

$$

```{code-cell} ipython3
beta_ex  = 0.99
rho_x_ex = 0.90
sigma_x_ex = 0.01
mu_th_ex = 4.0

Vx_re_ex = (1 - beta_ex) / (1 - beta_ex * rho_x_ex)
print(f"V_x (rational expectations): {Vx_re_ex:.4f}")

m_ex = BeliefDistortionModel(beta=beta_ex, rho_x=rho_x_ex,
                              sigma_x=sigma_x_ex, mu_theta=mu_th_ex)
print(f"V_x (with pessimism θ̄={mu_th_ex}):   {m_ex.Vx:.4f}")
```

**Part (b)** — The belief wedge for consumption growth is

$$

\Delta_t^{(1)}(x)
= -\theta_t V_x \sigma_x^2.

$$

Since $V_x > 0$ and $\theta_t > 0$, the wedge is **negative**: pessimistic
agents underestimate consumption growth relative to the model.

**Part (c)** — For unemployment, $u_x < 0$, so $V_x^u < 0$.  The belief
wedge becomes

$$

\Delta_t^{(1)}(u)
= -\theta_t V_x^u \sigma_x^2 > 0

$$

(positive, because pessimism makes agents over-estimate unemployment).
This matches the empirical finding of a positive mean unemployment wedge.

```{solution-end}
```

```{exercise}
:label: sbbc_ex2

**Persistence and wedge volatility**

Using the `BeliefDistortionModel` class, vary $\rho_\theta$ from 0.3 to
0.95 (holding $\sigma_\theta = 4.3$ fixed) and plot how the standard
deviation of the belief wedge changes.  Explain the economic intuition.
```

```{solution-start} sbbc_ex2
:label: sbbc_ex2_sol
:class: dropdown
```

```{code-cell} ipython3
rho_vals    = np.linspace(0.3, 0.95, 30)
wedge_stds  = []

for rho in rho_vals:
    m_temp = BeliefDistortionModel(rho_theta=rho)
    theta_sim_temp = m_temp.simulate_theta(T=5000, seed=0)
    wedge_sim_temp = m_temp.belief_wedge(theta_sim_temp)
    wedge_stds.append(np.std(wedge_sim_temp))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(rho_vals, np.array(wedge_stds) * 100, color='steelblue', linewidth=2)
ax.set_xlabel('Persistence $\\rho_\\theta$')
ax.set_ylabel('Std of belief wedge (pp)')
ax.set_title('More persistent belief shocks → higher wedge volatility')
plt.tight_layout()
plt.show()
```

Higher persistence $\rho_\theta$ means that a given innovation to $\theta_t$
has more prolonged effects: the unconditional variance of an AR(1) with
volatility $\sigma$ is $\sigma^2 / (1 - \rho^2)$, which increases in $\rho$.
Since the wedge is proportional to $\theta_t$, its standard deviation
inherits this relationship and rises with $\rho_\theta$.

```{solution-end}
```

```{exercise}
:label: sbbc_ex3

**Unemployment volatility decomposition**

Using the `ReducedFormNKModel` class:

(a) Compute the fraction of unemployment variance explained by each of the
    three shocks.
(b) Show that the belief shock is the dominant driver of unemployment
    fluctuations, while TFP is the dominant driver of output fluctuations.
```

```{solution-start} sbbc_ex3
:label: sbbc_ex3_sol
:class: dropdown
```

```{code-cell} ipython3
# Variance decomposition via the Lyapunov equation, shock-by-shock

shock_names = ['Belief shock (θ)', 'TFP shock', 'MP shock']
var_labels  = ['Unemployment', 'Inflation', 'Output']

nk2 = ReducedFormNKModel()

# Compute the variance of each variable attributable to each shock
n_states = nk2.A.shape[0]
var_by_shock = np.zeros((n_states, 3))

for j in range(3):
    B_j   = np.outer(nk2.B[:, j], nk2.B[:, j])
    Sigma_j = solve_discrete_lyapunov(nk2.A, B_j)
    var_by_shock[:, j] = np.diag(Sigma_j)

# Total variance
var_total = var_by_shock.sum(axis=1)

# Print share of variance for key variables
print(f"{'Variable':<16}", *[f"{s:>20}" for s in shock_names])
print('-' * 77)
for i, label in zip([nk2.I_U, nk2.I_PI, nk2.I_Y], var_labels):
    shares = var_by_shock[i] / var_total[i] * 100
    print(f"{label:<16}", *[f"{s:>19.1f}%" for s in shares])
```

The belief shock accounts for the majority of unemployment variance, as
reported in the paper.  Technology shocks drive most of the output variance
(through their high persistence and direct effect on productivity).  Monetary
policy shocks play a smaller role for both variables.

```{solution-end}
```

```{exercise}
:label: sbbc_ex4

**Changing the degree of pessimism**

Solve the Riccati equation in the `BeliefDistortionModel` for a grid of
$\mu_\theta$ values from 0 (rational expectations) to 15.  For each value,
compute the steady-state (unconditional mean) belief wedge and the ratio of
robust to rational $V_x$.  Discuss how the robust value function differs from
the rational-expectations value function.
```

```{solution-start} sbbc_ex4
:label: sbbc_ex4_sol
:class: dropdown
```

```{code-cell} ipython3
mu_grid  = np.linspace(0, 15, 100)
Vx_vals  = []
wedge_ss = []

Vx_re = (1 - 0.994) / (1 - 0.994 * 0.85)

for mu in mu_grid:
    m_temp = BeliefDistortionModel(mu_theta=mu)
    Vx_vals.append(m_temp.Vx)
    wedge_ss.append(m_temp.belief_wedge(mu) * 100)   # in pp

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(mu_grid, Vx_vals, color='steelblue', linewidth=2)
axes[0].axhline(Vx_re, color='grey', linestyle='--',
                label=f'RE value $V_x^{{RE}}={Vx_re:.3f}$')
axes[0].set_xlabel('Mean pessimism $\\mu_\\theta$')
axes[0].set_ylabel('$V_x$')
axes[0].set_title('Robust continuation-value sensitivity')
axes[0].legend()

axes[1].plot(mu_grid, np.array(wedge_ss), color='firebrick', linewidth=2)
axes[1].set_xlabel('Mean pessimism $\\mu_\\theta$')
axes[1].set_ylabel('Steady-state wedge (pp)')
axes[1].set_title('Steady-state belief wedge increases with $\\mu_\\theta$')

plt.tight_layout()
plt.show()
```

As $\mu_\theta$ rises, the Riccati equation introduces an additional
curvature term that lowers $V_x$ (less marginal value to the current state)
because the agent effectively prices in the possibility of bad future
outcomes.  The steady-state wedge grows approximately linearly in
$\mu_\theta$, since $\Delta^{(1)} \propto \mu_\theta V_x \sigma_x^2$ and
$V_x$ is approximately constant for small $\mu_\theta$.

```{solution-end}
```


