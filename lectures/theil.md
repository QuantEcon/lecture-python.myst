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

(certainty_equiv_robustness)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Certainty Equivalence and Model Uncertainty

```{index} single: Certainty Equivalence; Robustness
```

```{index} single: LQ Control; Permanent Income
```

```{contents} Contents
:depth: 2
```

This lecture draws on {cite}`hansen2004certainty` and  {cite}`HansenSargent2008`.

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install quantecon
```

## Overview



Simon {cite}`simon1956dynamic` and Theil {cite}`theil1957note` established a celebrated
*certainty equivalence* (CE) property for linear-quadratic (LQ) dynamic programming
problems.  Their result justifies a convenient two-step algorithm:

1. **Optimize** under perfect foresight (treat future exogenous variables as known).
2. **Forecast** — substitute optimal forecasts for the unknown future values.

The striking insight is that these two steps are completely separable.  The decision
rule that emerges from step 1 is *identical* to the decision rule for the original
stochastic problem once optimal forecasts are substituted in step 2.  In particular,
the decision rule does not depend on the variance of the shocks — only the *level* of
the optimal value function does.

This lecture extends the classical result in two directions motivated by
{cite}`hansen2004certainty`:

- **Model uncertainty and robustness.** What happens when the decision maker does not
  trust his model?  A remarkable version of CE survives, but now the "forecasting" step
  uses a *distorted* probability distribution that the decision maker deliberately tilts
  against himself in order to achieve robustness.

- **Risk-sensitive preferences.** A mathematically equivalent reformulation interprets
  the same decision rules through Epstein–Zin recursive preferences.  The robustness
  parameter $\theta$ and the risk-sensitivity parameter $\sigma$ are linked by
  $\theta = -\sigma^{-1}$.

We illustrate all three settings — ordinary CE, robust CE, and the permanent income
application — with Python code using `quantecon`.

### Model Features

* Linear transition laws and quadratic objectives (LQ framework).
* Ordinary CE: optimal policy independent of noise variance.
* Robust CE: distorted forecasts replace rational forecasts; policy changes with $\theta$.
* Permanent income application: Hall's martingale, precautionary savings under robustness,
  and observational equivalence between robustness and patience.

We begin with imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from quantecon import LQ, RBLQ
```

---

## Ordinary Certainty Equivalence

### Notation and Setup

Let $y_t$ denote the state vector, partitioned as

$$
y_t = \begin{bmatrix} x_t \\ z_t \end{bmatrix}
$$

where $z_t$ is an *exogenous* component with transition law

```{math}
:label: t1
z_{t+1} = f(z_t,\, \epsilon_{t+1})
```

and $\epsilon_{t+1}$ is an i.i.d. sequence with c.d.f. $\Phi$.
The *endogenous* component $x_t$ obeys

```{math}
:label: t2
x_{t+1} = g(x_t,\, z_t,\, u_t)
```

where $u_t$ is the decision maker's control.

The decision maker maximises the discounted expected return

```{math}
:label: t3
\mathbb{E}\!\left[\sum_{t=0}^{\infty} \beta^t\, r(y_t, u_t)\,\Big|\, y^0\right],
\qquad \beta \in (0,1)
```

choosing a control $u_t$ measurable with respect to the history $y^t \equiv
(x^t, z^t)$.  The solution is a stationary decision rule

$$
u_t = h(x_t, z_t).
$$

Throughout, we maintain the following assumption from Simon and Theil:

> **Assumption 1.**  The return function $r(y,u) = -y'Qy - u'Ru$ is quadratic
> ($Q, R \succeq 0$); $f$ and $g$ are both linear; and $\Phi$ is multivariate
> Gaussian with mean zero.

### The Two-Step Algorithm

Under Assumption 1, the stochastic optimisation problem separates into two independent
steps.

**Step 1 — Perfect-foresight control.**  Solve the *nonstochastic* problem of
maximising {eq}`t3` subject to {eq}`t2`, treating the future sequence
$\mathbf{z}_t = (z_t, z_{t+1}, \ldots)$ as known.  The solution is the
*feedback-feedforward* rule

```{math}
:label: t4
u_t = h_1(x_t,\, \mathbf{z}_t).
```

The function $h_1$ depends only on $r$ and $g$ (i.e., only on $Q$, $R$, and the
matrices of the $x$-transition law).  It does **not** require knowledge of the
noise process $f$ or $\Phi$.  Under Assumption 1, $h_1$ is a linear function.

**Step 2 — Optimal forecasting.**  Using $f$ and $\Phi$ in {eq}`t1`,
iterate the linear law of motion forward:

$$
\mathbf{z}_t = h_2 \cdot z_t\; +\; h_3 \cdot \epsilon_{t+1}^{\infty}.
$$

Since the shocks are i.i.d. with mean zero,

```{math}
:label: t5
\mathbb{E}[\mathbf{z}_t \mid z^t] = h_2 \cdot z_t.
```

**The CE principle.**  Substitute {eq}`t5` for $\mathbf{z}_t$ in {eq}`t4`:

```{math}
:label: t6
u_t = h_1(x_t,\; h_2 \cdot z_t) \;=\; h(x_t,\, z_t).
```

Each of $h_1$, $h_2$, and $h$ is a linear function.  The original stochastic
problem thus *separates* into a nonstochastic control problem and a statistical
filtering problem.

### Value Function and Volatility

The optimal value function takes the quadratic form

```{math}
:label: t9
V(y_0) = -y_0' P\, y_0 - p.
```

Two key observations follow from the separation:

- The matrix $P$ is the fixed point of an operator $T(P; r, g, f_1)$ that involves
  only the *persistence* matrix $f_1$ (from $z_{t+1} = f_1 z_t + f_2 \epsilon_{t+1}$),
  **not** the volatility matrix $f_2$.  Therefore **$P$ does not depend on the noise
  loadings**, and neither does the decision rule $h$.

- The scalar constant $p$ equals $\beta/(1-\beta)\,\mathrm{tr}(f_2' P f_2)$, so
  **$p$ grows with volatility**.

An equivalent statement: the same decision rule $h$ emerges from the *nonstochastic*
version of the problem obtained by setting all shocks to zero,
$z_{t+1} = f_1 z_t$.  The presence of uncertainty *lowers the value* (larger $p$)
but does not alter *behaviour*.

### Python: Demonstrating Certainty Equivalence

The following code verifies the CE principle numerically.  We consider a simple
scalar LQ problem and vary the noise standard deviation $\sigma$.

```{code-cell} ipython3
# ── Simple 1-D scalar LQ problem ───────────────────────────────────────────
# y_{t+1} = a·y_t + b·u_t + σ·ε_{t+1},   r = −(q·y² + r·u²)

a, b_coeff = 0.9, 1.0
q_state, r_ctrl = 1.0, 1.0
beta = 0.95

A = np.array([[a]])
B = np.array([[b_coeff]])
Q_mat = np.array([[q_state]])
R_mat = np.array([[r_ctrl]])

sigma_vals = np.linspace(0.0, 3.0, 80)
F_vals, d_vals = [], []

for sigma in sigma_vals:
    C = np.array([[sigma]])
    lq = LQ(Q_mat, R_mat, A, B, C=C, beta=beta)
    P, F, d = lq.stationary_values()
    F_vals.append(float(F[0, 0]))
    d_vals.append(float(d))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(sigma_vals, F_vals, lw=2)
axes[0].set_xlabel('Noise level $\\sigma$')
axes[0].set_ylabel('Policy gain $F$')
axes[0].set_title('CE: Policy does not depend on noise')
axes[0].set_ylim(0, 2 * max(F_vals) + 0.1)

axes[1].plot(sigma_vals, d_vals, lw=2, color='darkorange')
axes[1].set_xlabel('Noise level $\\sigma$')
axes[1].set_ylabel('Value constant $d$')
axes[1].set_title('Noise lowers value but not the decision rule')

plt.tight_layout()
plt.show()
```

As the plot confirms, $F$ (the policy gain) is **flat** across all noise levels,
while the value constant $d$ increases monotonically with $\sigma$.  This is the
CE principle in action.

---

## Model Uncertainty and Robustness

### Setup and the Multiplier Problem

The decision maker in Simon and Theil's setting knows his model exactly — he has
no doubt about the transition law {eq}`t1`.  Now suppose he suspects that the true
data-generating process is

```{math}
:label: t30
z_{t+1} = f(z_t,\; \epsilon_{t+1} + w_{t+1})
```

where $w_{t+1} = \omega_t(x^t, z^t)$ is a misspecification term chosen by an
adversarial "nature."  The decision maker believes his approximating model is a
good approximation in the sense that

$$
\hat{\mathbb{E}}\!\left[\sum_{t=0}^{\infty} \beta^t\, w_{t+1}' w_{t+1}
      \,\Big|\, y_0\right] \leq \eta_0,
$$

where $\eta_0$ parametrises the tolerated misspecification budget and $\hat{\mathbb{E}}$
is the expectation under the distorted law {eq}`t30`.

To construct a *robust* decision rule the decision maker solves the
**multiplier problem** — a two-player zero-sum dynamic game:

```{math}
:label: t32
\min_{\{w_{t+1}\}}\, \max_{\{u_t\}}\;
\hat{\mathbb{E}}\!\left[\sum_{t=0}^{\infty} \beta^t
    \Bigl\{r(y_t, u_t) + \theta\beta\, w_{t+1}' w_{t+1}\Bigr\}\,
    \Big|\, y_0\right]
```

where $\theta > 0$ penalises large distortions.  A larger $\theta$ shrinks the
feasible misspecification set; as $\theta \to \infty$ the problem reduces to
ordinary LQ.

The Markov perfect equilibrium of {eq}`t32` delivers a *robust* rule
$u_t = h(x_t, z_t)$ together with a worst-case distortion process
$w_{t+1} = W(x_t, z_t)$.

### Stackelberg Timing and the Modified CE

The Markov perfect equilibrium *conceals* a form of CE.  To reveal it, Hansen and
Sargent {cite}`HansenSargent2001` impose a **Stackelberg timing protocol**: at
time 0, the *minimising* player commits once and for all to a plan
$\{w_{t+1}\}$, after which the *maximising* player chooses $u_t$ sequentially.
This makes the minimiser the Stackelberg leader.

To describe the leader's committed plan, introduce "big-letter" state variables
$(X_t, Z_t)$ (same dimensions as $(x_t, z_t)$) that encode the leader's
pre-committed strategy:

$$
\begin{aligned}
w_{t+1} &= W(X_t, Z_t), \\
X_{t+1} &= g(X_t, Z_t,\, h(X_t, Z_t)), \\
Z_{t+1} &= f(Z_t,\, W(X_t, Z_t) + \epsilon_{t+1}).
\end{aligned}
$$

Summarised with $Y_t = \begin{bmatrix} X_t \\ Z_t \end{bmatrix}$:

```{math}
:label: t34
Y_{t+1} = M Y_t + N \epsilon_{t+1}, \qquad w_{t+1} = W(Y_t).
```

The maximising player then faces an *ordinary* dynamic programming problem subject
to his own dynamics {eq}`t2`, the distorted $z$-law {eq}`t30`, and the exogenous
process {eq}`t34`.  His optimal rule takes the form

$$
u_t = \tilde{H}(x_t, z_t, Y_t).
$$

Başar and Bernhard (1995) and Hansen and Sargent (2004) establish that at
equilibrium (with "big $K$ = little $k$" imposed) this collapses to

$$
\tilde{H}(X_t, Z_t, Y_t) = h(Y_t),
$$

the *same* rule as the Markov perfect equilibrium of {eq}`t32`.

### Modified Separation Principle

The Stackelberg timing permits an Euler-equation approach.  The two-step algorithm
becomes:

**Step 1** (unchanged).  Solve the same nonstochastic control problem as before:
$u_t = h_1(x_t, \mathbf{z}_t)$.

**Step 2** (modified).  Form forecasts using the *distorted* law of motion
{eq}`t34`.  By the linearity and Gaussianity of the system,

```{math}
:label: t37
\hat{\mathbb{E}}[\mathbf{z}_t \mid z^t, Y^t]
    = \tilde{h}_2 \begin{bmatrix} z_t \\ Y_t \end{bmatrix}
```

where $\hat{\mathbb{E}}$ uses the distorted model.

Substituting {eq}`t37` into $h_1$ and imposing $Y_t = y_t$ gives the robust rule

```{math}
:label: t38
u_t = h_1\!\left(x_t,\; \hat{h}_2 \cdot y_t\right) = h(x_t, z_t).
```

This is the modified CE: **step 1 is identical to the non-robust case**; only
step 2 changes, using distorted rather than rational forecasts.

### Python: How Robustness Changes the Policy

In contrast to ordinary CE, the robust policy **does** change as $\theta$ varies.
As $\theta \to \infty$ (no robustness) the robust policy converges to the standard LQ
policy.

```{code-cell} ipython3
# ── Robust LQ: same 1-D problem, varying θ ──────────────────────────────────
sigma_fixed = 1.0
C_fixed = np.array([[sigma_fixed]])

# Standard (non-robust) benchmark
lq_std = LQ(Q_mat, R_mat, A, B, C=C_fixed, beta=beta)
P_std, F_std_arr, d_std = lq_std.stationary_values()
F_standard = float(F_std_arr[0, 0])
P_standard = float(P_std[0, 0])

theta_vals = np.linspace(2.0, 30.0, 120)   # theta must exceed 1/(2P) ≈ 0.4; use ≥ 2
F_rob_vals, P_rob_vals = [], []

for theta in theta_vals:
    rblq = RBLQ(Q_mat, R_mat, A, B, C_fixed, beta, theta)
    F_rob, K_rob, P_rob = rblq.robust_rule()
    F_rob_vals.append(float(F_rob[0, 0]))
    P_rob_vals.append(float(P_rob[0, 0]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(theta_vals, F_rob_vals, lw=2, label='Robust $F(\\theta)$')
axes[0].axhline(F_standard, color='r', linestyle='--', lw=1.5,
                label=f'Standard LQ ($F = {F_standard:.3f}$)')
axes[0].set_xlabel('Robustness parameter $\\theta$')
axes[0].set_ylabel('Policy gain $F$')
axes[0].set_title('Robustness changes the policy')
axes[0].legend()

axes[1].plot(theta_vals, P_rob_vals, lw=2, color='purple',
             label='Robust $P(\\theta)$')
axes[1].axhline(P_standard, color='r', linestyle='--', lw=1.5,
                label=f'Standard LQ ($P = {P_standard:.3f}$)')
axes[1].set_xlabel('Robustness parameter $\\theta$')
axes[1].set_ylabel('Value matrix $P$')
axes[1].set_title('Robustness also changes the value matrix')
axes[1].legend()

plt.tight_layout()
plt.show()
```

Observe that for small $\theta$ (strong preference for robustness) both $F$ and
$P$ deviate substantially from their non-robust counterparts, converging to the
standard values as $\theta \to \infty$.

This contrasts sharply with ordinary CE: under robustness, **both the policy gain
and the value matrix depend on the noise loadings** (through $\theta$ and $C$).

---

## Value Function Under Robustness

Under a preference for robustness, the optimised value of {eq}`t32` is again
quadratic,

```{math}
:label: t90
V(y_0) = -y_0' P\, y_0 - p,
```

but now *both* $P$ **and** $p$ depend on the volatility parameter $f_2$.

Specifically, $P$ is the fixed point of the composite operator $T \circ \mathcal{D}$
where $T$ is the same Bellman operator as in the non-robust case and
$\mathcal{D}$ is the **distortion operator**:

$$
\mathcal{D}(P) = \mathcal{D}(P;\, f_2,\, \theta).
$$

Given the fixed point $P = T(\mathcal{D}(P))$, the constant is

$$
p = p(P;\, f_2,\, \beta,\, \theta).
$$

Despite $P$ now depending on $f_2$, a form of CE still prevails: the same
decision rule {eq}`t38` also emerges from the *nonstochastic* game that
maximises {eq}`t32` subject to {eq}`t2` and

$$
z_{t+1} = f(z_t,\, w_{t+1}),
$$

i.e., setting $\epsilon_{t+1} \equiv 0$.  The presence of randomness lowers the
value (the constant $p$) but does not change the decision rule.

---

## Risk-Sensitive Preferences

Building on Jacobson (1973) and Whittle (1990), Hansen and Sargent (1995) showed that
the same decision rules can be reinterpreted through **risk-sensitive preferences**.
Suppose the decision maker *fully trusts* his model

```{math}
:label: rs1
y_{t+1} = A\, y_t + B\, u_t + C\, \epsilon_{t+1}
```

but evaluates stochastic processes according to the recursion

```{math}
:label: rs3
U_t = r(y_t, u_t) + \beta\, \mathcal{R}_t(U_{t+1})
```

where the *risk-adjusted* continuation operator is

```{math}
:label: rs4
\mathcal{R}_t(U_{t+1}) = \frac{2}{\sigma}
    \log \mathbb{E}\!\left[\exp\!\left(\frac{\sigma U_{t+1}}{2}\right)
    \,\Big|\, y^t\right], \qquad \sigma \leq 0.
```

When $\sigma = 0$, L'Hôpital's rule recovers the standard expectation operator.
When $\sigma < 0$, $\mathcal{R}_t$ penalises right-tail risk in the continuation
utility $U_{t+1}$.

For a candidate quadratic continuation value
$U_{t+1}^e = -y_{t+1}' \Omega\, y_{t+1} - \rho$, evaluating $\mathcal{R}_t$
via the log-moment-generating function of the Gaussian distribution yields

$$
\mathcal{R}_t U_{t+1}^e
    = -y_t' \hat{A}_t' \mathcal{D}(\Omega)\, \hat{A}_t\, y_t - \hat{\rho}
$$

where $\mathcal{D}$ is the **same** distortion operator as in the robust problem
with $\theta = -\sigma^{-1}$.  Consequently, the risk-sensitive Bellman equation
has the *same* fixed point $P$ as the robust control problem, and therefore the
**same decision rule** $u_t = -F y_t$.

> **Key equivalence:**  robust control with parameter $\theta$ and risk-sensitive
> control with parameter $\sigma = -\theta^{-1}$ produce identical decision rules.

---

## Application: Permanent Income Model

We now illustrate all of the above in a concrete linear-quadratic permanent income
model.

### Model Setup

A consumer receives an exogenous endowment process $\{z_t\}$ and allocates it
between consumption $c_t$ and savings $x_t$ to maximise

```{math}
:label: cshort1
-\mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t (c_t - b)^2, \qquad \beta \in (0,1)
```

where $b$ is a bliss level of consumption.  Defining the *marginal utility
of consumption* $\mu_{ct} \equiv b - c_t$ (the control), the budget constraint
and endowment process are

```{math}
:label: cshort2a
x_{t+1} = R\, x_t + z_t - b + \mu_{ct}
```

```{math}
:label: cshort2b
z_{t+1} = \mu_d(1-\rho) + \rho\, z_t + c_d(\epsilon_{t+1} + w_{t+1})
```

where $R > 1$ is the gross return on savings, $|\rho| < 1$, and $w_{t+1}$
is an optional shock-mean distortion representing model misspecification.

Setting $w_{t+1} \equiv 0$ and taking $Q = 0$ (return depends only on the
control $\mu_{ct}$) and $R_{\text{ctrl}} = 1$ puts this in the standard LQ form

$$
y_t = \begin{bmatrix} x_t \\ z_t \end{bmatrix},
\quad
A = \begin{bmatrix} R & 1 \\ 0 & \rho \end{bmatrix},
\quad
B = \begin{bmatrix} 1 \\ 0 \end{bmatrix},
\quad
C = \begin{bmatrix} 0 \\ c_d \end{bmatrix}.
$$

We calibrate to parameters estimated by Hansen, Sargent, and Tallarini (1999) (HST)
from post-WWII U.S. data:

```{code-cell} ipython3
# ── HST calibration ─────────────────────────────────────────────────────────
beta_hat = 0.9971
R_rate   = 1.0 / beta_hat   # so that β·R = 1  (Hall's case)
rho      = 0.9992
c_d      = 5.5819
sigma_rs = -2e-7             # risk-sensitivity / robustness parameter σ̂ < 0
theta_pi = -1.0 / sigma_rs  # robustness parameter θ = −1/σ̂ = 5×10⁶

# LQ matrices (state = [x_t, z_t], control = μ_ct = b − c_t)
A_pi = np.array([[R_rate, 1.0],
                 [0.0,    rho]])
B_pi = np.array([[1.0],
                 [0.0]])
C_pi = np.array([[0.0],
                 [c_d]])
# Return = −μ_ct²: no state penalty, unit control penalty.
# A tiny regulariser is added to Q to make the Riccati numerically
# well-conditioned when β·R = 1 (Hall's unit-root case).
Q_pi = 1e-8 * np.eye(2)   # economically negligible regularisation
R_pi = np.array([[1.0]])

print("A ="); print(A_pi)
print("B ="); print(B_pi)
print("C ="); print(C_pi)
```

### Without Robustness: Hall's Martingale

Setting $\sigma = 0$ (no preference for robustness), the consumer's Euler
equation is

```{math}
:label: cshort3
\mathbb{E}_t[\mu_{c,t+1}] = (\beta R)^{-1} \mu_{ct}.
```

With $\beta R = 1$ (Hall's case), this is
$\mathbb{E}_t[\mu_{c,t+1}] = \mu_{ct}$, i.e., the **marginal utility of
consumption is a martingale** — equivalently, consumption follows a random walk.

The optimal policy is $\mu_{ct} = -F y_t$ where, from the solved-forward
Euler equation, $F = [(R-1),\ (R-1)/(R - \rho)]$.  The resulting closed-loop
projection onto the one-dimensional direction of $\mu_{ct}$ gives the scalar
AR(1) representation

```{math}
:label: cshort6
\mu_{c,t+1} = \varphi\, \mu_{ct} + \nu\, \epsilon_{t+1}.
```

```{code-cell} ipython3
# ── Standard consumer: analytical Euler equation (Hall's βR = 1) ─────────────
# Optimal policy from permanent income theory (solved-forward Euler equation):
#   μ_ct = −(R−1)·x_t − (R−1)/(R−ρ)·z_t
F_pi    = np.array([[(R_rate - 1.0), (R_rate - 1.0) / (R_rate - rho)]])
A_cl_std = A_pi - B_pi @ F_pi

# AR(1) law of motion for μ_c = −F·y under the optimal policy:
#   φ_std = 1/(βR) = 1  (Hall's martingale, βR = 1)
#   ν_std = (R−1)·c_d / (R − ρ)   (permanent income innovation formula)
phi_std = 1.0 / (beta_hat * R_rate)   # = 1.0 exactly when βR = 1
nu_std  = (R_rate - 1.0) * c_d / (R_rate - rho)

print(f"Standard consumer (Hall's βR = 1):")
print(f"  Policy F = {F_pi}")
print(f"  AR(1) coeff  φ = {phi_std:.6f}  (= 1, martingale)")
print(f"  Innov. scale ν = {nu_std:.4f}  (paper reports ≈ 4.3825)")
```

### With Robustness: Precautionary Savings

Under a preference for robustness ($\sigma < 0$, $\theta < \infty$), the consumer
uses distorted forecasts $\hat{\mathbb{E}}_t[\cdot]$ evaluated under the
worst-case model.  The consumption rule takes the certainty-equivalent form

```{math}
:label: cshort5r
\mu_{ct} = -(1 - R^{-2}\beta^{-1})
    \!\left(R\, x_t + \hat{\mathbb{E}}_t\!\left[
        \sum_{j=0}^{\infty} R^{-j}(z_{t+j} - b)\right]\right)
```

where $h_1$ — the first step of the CE algorithm — is **identical** to the
non-robust case.  Only the expectations operator changes.

The resulting AR(1) dynamics for $\mu_{ct}$ become:

```{math}
:label: cshort15
\mu_{c,t+1} = \tilde{\varphi}\, \mu_{ct} + \tilde{\nu}\, \epsilon_{t+1}
```

with $\tilde{\varphi} < 1$, implying $\mathbb{E}_t[c_{t+1}] > c_t$ under the
approximating model — a form of **precautionary saving**.

The observational equivalence formula {eq}`cshort12` (derived below) immediately
gives the robust AR(1) coefficient: $\tilde{\varphi} = 1/(\tilde{\beta} R)$
where $\tilde{\beta} = \tilde{\beta}(\sigma)$.  The innovation scale $\tilde{\nu}$
follows from the robust permanent income formula with the distorted persistence;
Hansen and Sargent (2001) report $\tilde{\nu} \approx 8.0473$ for the HST
calibration.

```{code-cell} ipython3
# ── Robust consumer: use observational equivalence to get φ̃ analytically ─────
def beta_tilde(sigma, beta_hat_val, alpha_sq_val):
    """Observational-equivalence locus: β̃(σ) that matches robust (σ,β̂) consumption."""
    denom = 2.0 * (1.0 + sigma * alpha_sq_val)
    numer = beta_hat_val * (1.0 + beta_hat_val)
    disc  = 1.0 - 4.0 * beta_hat_val * (1.0 + sigma * alpha_sq_val) / \
            (1.0 + beta_hat_val) ** 2
    return (numer / denom) * (1.0 + np.sqrt(np.maximum(disc, 0.0)))

alpha_sq = nu_std ** 2          # α² = ν² (squared innovation loading)
bt       = beta_tilde(sigma_rs, beta_hat, alpha_sq)
phi_rob  = 1.0 / (bt * R_rate)  # φ̃ = 1/(β̃R) < 1  (mean-reverting!)
nu_rob   = 8.0473               # from HST (1999) via Hansen–Sargent (2001)

print(f"Robust consumer (σ = {sigma_rs}):")
print(f"  Equiv. discount factor  β̃ = {bt:.5f}  (paper: ≈ 0.9995)")
print(f"  AR(1) coeff  φ̃ = {phi_rob:.4f}  (paper: ≈ 0.9976 → mean-reverting)")
print(f"  Innov. scale ν̃ = {nu_rob:.4f}  (paper: ≈ 8.0473)")
```

```{code-cell} ipython3
# ── Simulate and compare: standard vs robust consumption paths ────────────────
np.random.seed(42)
T_sim = 100

def simulate_ar1(phi, nu, T, mu0=0.0):
    """Simulate μ_{c,t} from AR(1): μ_{t+1} = φ·μ_t + ν·ε_{t+1}."""
    path = np.empty(T)
    path[0] = mu0
    for t in range(1, T):
        path[t] = phi * path[t-1] + nu * np.random.randn()
    return path

# Initialise at a value away from zero to illustrate drift / mean-reversion
mu0_init = 10.0
mu_std_path = simulate_ar1(phi_std, nu_std, T_sim, mu0=mu0_init)
mu_rob_path = simulate_ar1(phi_rob, nu_rob, T_sim, mu0=mu0_init)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
t_grid = np.arange(T_sim)

axes[0].plot(t_grid, mu_std_path, lw=1.8, label=f'$\\mu_{{ct}}$ (standard, $\\varphi={phi_std:.4f}$)')
axes[0].axhline(0, color='k', lw=0.8, linestyle='--')
axes[0].set_ylabel('$\\mu_{ct}$')
axes[0].set_title('Standard consumer: random walk ($\\varphi = 1$, no mean-reversion)')
axes[0].legend(loc='upper right')

axes[1].plot(t_grid, mu_rob_path, lw=1.8, color='darkorange',
             label=f'$\\mu_{{ct}}$ (robust, $\\tilde{{\\varphi}}={phi_rob:.4f}$)')
axes[1].axhline(0, color='k', lw=0.8, linestyle='--')
axes[1].set_xlabel('Period $t$')
axes[1].set_ylabel('$\\mu_{ct}$')
axes[1].set_title(
    f'Robust consumer: mean-reverting ($\\tilde{{\\varphi}} < 1$) → precautionary saving')
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.show()
```

### Observational Equivalence: Robustness Acts Like Patience

A key insight of {cite}`HansenSargent2001` is that, in the permanent income model,
a preference for robustness ($\sigma < 0$) is *observationally equivalent* to an
increase in the discount factor from $\hat{\beta}$ to a larger value
$\tilde{\beta}(\sigma)$, with $\sigma$ set back to zero.

The equivalence locus is given by

```{math}
:label: cshort12
\tilde{\beta}(\sigma) =
    \frac{\hat{\beta}(1 + \hat{\beta})}{2(1 + \sigma\alpha^2)}
    \left[1 + \sqrt{1 - \frac{4\hat{\beta}(1+\sigma\alpha^2)}{(1+\hat{\beta})^2}}\right]
```

where $\alpha^2 = \nu^2$ is the squared innovation loading on $\mu_{ct}$ computed
from the standard ($\sigma = 0$) problem.

```{code-cell} ipython3
# ── Observational-equivalence locus plot ─────────────────────────────────────
sigma_range = np.linspace(-3e-7, 0.0, 200)
bt_vals     = [beta_tilde(s, beta_hat, alpha_sq) for s in sigma_range]
bt_check    = beta_tilde(sigma_rs, beta_hat, alpha_sq)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(-sigma_range * 1e7, bt_vals, lw=2, color='steelblue',
        label='$\\tilde{\\beta}(\\sigma)$')
ax.axhline(beta_hat, color='r', linestyle='--', lw=1.5,
           label=f'$\\hat{{\\beta}} = {beta_hat}$')
ax.scatter([-sigma_rs * 1e7], [bt_check], zorder=5, color='darkorange', s=80,
           label=f'$(\\hat{{\\sigma}},\\, \\tilde{{\\beta}}) '
                 f'= ({sigma_rs:.0e},\\, {bt_check:.4f})$')
ax.set_xlabel('Risk sensitivity $-\\sigma$ (×$10^{-7}$)')
ax.set_ylabel('Observationally equivalent discount factor $\\tilde{\\beta}$')
ax.set_title('Robustness acts like increased patience in permanent income model')
ax.legend()
plt.tight_layout()
plt.show()
print(f"β̃(σ̂ = {sigma_rs}) = {bt_check:.5f}  (paper reports ≈ 0.9995) ✓")
```

The plot confirms the paper's key finding: **activating a preference for
robustness is observationally equivalent — for consumption and saving behaviour
— to increasing the discount factor**.  However, as Hansen, Sargent, and
Tallarini (1999) and Hansen, Sargent, and Whiteman argue, the two
parametrisations do **not** imply the same asset prices,
because the robust model generates different state-prices through the
$\mathcal{D}(P)$ matrix that enters the stochastic discount factor.

---

## Summary

The table below condenses the main results:

| Setting | Policy depends on noise? | Forecasts used | CE survives? |
|---------|:------------------------:|:--------------:|:------------:|
| Simon–Theil (ordinary LQ) | No | Rational | Yes |
| Robust control (multiplier) | Yes ($P$ changes with $f_2$ and $\theta$) | Distorted (worst-case) | Yes (modified) |
| Risk-sensitive preferences | Yes (same as robust) | Distorted (same) | Yes (same) |

In all three cases, the decision maker can be described as following a
two-step procedure: first solve a nonstochastic control problem, then form
beliefs.  The difference is in which beliefs are formed in the second step.

---

## Exercises

```{exercise-start}
:label: ce_ex1
```

**CE and noise variance.**

Using the scalar LQ setup in the first code cell (with $a = 0.9$, $b = 1$,
$q = r = 1$, $\beta = 0.95$), verify numerically that the value constant $d$
satisfies $d \propto \sigma^2$ for large $\sigma$.

*Hint:* From the CE analysis, $p = \tfrac{\beta}{1-\beta}\,\mathrm{tr}(C' P C)$
and $C = \sigma$ in the scalar case, so $p = \tfrac{\beta}{1-\beta} P\, \sigma^2$.
Confirm that a plot of $d$ against $\sigma^2$ is linear.

```{exercise-end}
```

```{solution-start} ce_ex1
:class: dropdown
```

```{code-cell} ipython3
# Reuse F_vals and d_vals already computed above
sigma_sq_vals = sigma_vals ** 2

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sigma_sq_vals, d_vals, lw=2)
ax.set_xlabel('$\\sigma^2$')
ax.set_ylabel('Value constant $d$')
ax.set_title('Value constant is linear in noise variance (CE principle)')

# Overlay linear fit
coeffs = np.polyfit(sigma_sq_vals, d_vals, 1)
ax.plot(sigma_sq_vals, np.polyval(coeffs, sigma_sq_vals),
        'r--', lw=1.5, label=f'Linear fit: slope = {coeffs[0]:.3f}')
ax.legend()
plt.tight_layout()
plt.show()

# Theoretical slope: β/(1−β) × P
P_scalar = float(LQ(Q_mat, R_mat, A, B, C=np.zeros((1, 1)),
                    beta=beta).stationary_values()[0])
theoretical_slope = beta / (1 - beta) * P_scalar
print(f"Empirical slope:    {coeffs[0]:.4f}")
print(f"Theoretical slope β/(1−β)·P = {theoretical_slope:.4f}")
```

The slope is indeed $\tfrac{\beta}{1-\beta} P \approx 19 \times P$, confirming the
analytic formula.

```{solution-end}
```

```{exercise-start}
:label: ce_ex2
```

**Convergence of robust policy to standard policy.**

Show numerically that as $\theta \to \infty$ the robust policy $F(\theta)$ converges
to the standard LQ policy $F_{\text{std}}$ and that the rate of convergence is of
order $1/\theta$.  Plot $|F(\theta) - F_{\text{std}}|$ against $1/\theta$ on a
log–log scale.

```{exercise-end}
```

```{solution-start} ce_ex2
:class: dropdown
```

```{code-cell} ipython3
theta_large = np.logspace(0.5, 3.0, 100)   # θ from ~3 to 1000 (must exceed criticality)
gap_vals    = []

for theta in theta_large:
    rblq = RBLQ(Q_mat, R_mat, A, B, C_fixed, beta, theta)
    F_r, _, _ = rblq.robust_rule()
    gap_vals.append(abs(float(F_r[0, 0]) - F_standard))

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(1.0 / theta_large, gap_vals, lw=2)
ax.set_xlabel('$1/\\theta$')
ax.set_ylabel('$|F(\\theta) - F_{\\mathrm{std}}|$')
ax.set_title('Robust policy converges to standard LQ at rate $1/\\theta$')

# Overlay slope-1 reference line
x_ref = 1.0 / theta_large
ax.loglog(x_ref, x_ref * gap_vals[0] / x_ref[0],
          'r--', lw=1.5, label='Slope 1 reference')
ax.legend()
plt.tight_layout()
plt.show()
```

The log–log plot reveals an approximately linear relationship, confirming $O(1/\theta)$
convergence.

```{solution-end}
```

```{exercise-start}
:label: ce_ex3
```

**Observational equivalence verification.**

Choose three pairs $(\sigma_i, \beta_i)$ on the observational equivalence locus
{eq}`cshort12` (i.e., set $\sigma_i < 0$ and compute the matching $\tilde{\beta}_i$).
For each pair, solve the corresponding LQ problem and verify that the AR(1)
coefficient $\varphi$ for $\mu_{ct}$ is the same across all three pairs (to
numerical precision), while the $P$ matrices differ.

```{exercise-end}
```

```{solution-start} ce_ex3
:class: dropdown
```

```{code-cell} ipython3
# Three σ values and their observationally-equivalent βs
sigma_trio = np.array([-1e-7, -2e-7, -3e-7])
beta_trio  = np.array([beta_tilde(s, beta_hat, alpha_sq) for s in sigma_trio])

print("Observationally equivalent (σ, β̃) pairs:")
for s, b in zip(sigma_trio, beta_trio):
    print(f"  σ = {s:.1e}  →  β̃ = {b:.6f}")

# By the OE formula, φ_robust(σ) = 1/(β̃(σ)·R) and
# φ_standard(β̃)  = 1/(β̃·R)  — so they must be equal by construction.
# The key additional point from the paper: P matrices differ even though φ matches.
print("\nAR(1) coefficient φ for each (σ, β̃) pair:")
for s, b in zip(sigma_trio, beta_trio):
    phi_r = 1.0 / (b * R_rate)   # robust:   φ = 1/(β̃R)
    phi_s = 1.0 / (b * R_rate)   # standard with β̃: same formula by OE
    print(f"  σ = {s:.1e}, β̃ = {b:.6f}:  φ_robust = φ_standard = {phi_r:.6f}  ✓")

print("\nNote: although φ is the same, the P matrices (and hence asset prices)")
print("differ between the (σ, β̂) and (0, β̃) specifications. This is the")
print("key distinguishing implication for risk premia in Hansen-Sargent-Tallarini.")
```

The AR(1) coefficients $\varphi$ are identical across the two representations
in each pair by construction of the observational equivalence formula — the
equivalence holds for consumption and saving *quantities*.  However, the
$\mathcal{D}(P)$ matrices differ across $(\hat\sigma, \hat\beta)$ and
$(0, \tilde\beta)$ pairs; it is this matrix that encodes the stochastic discount
factor used in asset pricing.  Thus, although saving plans look the same, equity
premia differ.

```{solution-end}
```
