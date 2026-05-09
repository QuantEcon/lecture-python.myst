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

(ls_learning)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Least Squares Learning in Self-Referential Models

```{contents} Contents
:depth: 2
```

## Overview

This lecture is a companion to {doc}`rational_learning_re`, which presents the
Bray–Kreps perspective on rational learning. 

The present lecture examines the
closely related but distinct question of whether **least squares** learning
converges to a rational expectations equilibrium in self-referential models.


This lecture presents the framework of {cite}`MarcetSargent1989jet` for studying
**least squares learning** in a class of **self-referential** linear stochastic models.

A self-referential model is one in which the **actual** law of motion for the
economy depends on the **perceived** law of motion held by the agents within
it. 

In a rational expectations equilibrium (REE) the two coincide: the
perceived and actual laws of motion are the same.

But if agents start away
from equilibrium and update their beliefs by running least squares regressions,
will they converge to the REE?

{cite}`MarcetSargent1989jet` answer this question by exploiting a powerful
technique from systems-control engineering: the **differential equation
approach** of {cite}`Ljung1977`.

The key insight is that the stochastic
difference equation describing how beliefs evolve can be approximated, in the
limit, by a deterministic **ordinary differential equation** (ODE).

Almost-sure
convergence of least squares to the REE is then equivalent to **local stability**
of the REE as a fixed point of that ODE.

The framework unifies and extends earlier work by {cite}`Bray1982` and
{cite}`BraySavin1984` and connects naturally to the distinction between learning
*within* a rational expectations equilibrium (Bayesian updating inside a
correctly specified model) and learning *about* one (adapting an OLS estimator
whose data-generating process shifts with beliefs) discussed in
{cite}`BrayKreps1987`.



Let's begin with the imports we'll use throughout.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.gridspec import GridSpec

np.random.seed(42)
```

We also define two helper functions used throughout the lecture: one to
simulate recursive least squares in a scalar self-referential model, and one
to solve the associated ODE.

```{code-cell} ipython3
def simulate_rls_scalar(T_map, sigma_u, beta0, T_periods=500, N_paths=100,
                        a_seq=None, seed=0):
    """
    Simulate recursive least squares in a scalar self-referential model.

    The perceived law of motion is:  z1_t = beta_t * z2_{t-1} + u_t
    The actual law of motion is:     z1_t = T(beta_t) * z2_{t-1} + V * u_t

    For the scalar examples here z2_t = 1 (constant), so agents learn about
    the mean of a process that depends on their own expectation.

    Parameters
    ----------
    T_map    : callable, the mapping T: beta -> T(beta)
    sigma_u  : float, std of innovations
    beta0    : float, initial belief
    T_periods: int, simulation length
    N_paths  : int, number of Monte Carlo paths
    a_seq    : None or array of length T_periods (forgetting factors)
    seed     : int, random seed

    Returns
    -------
    beta_paths : ndarray, shape (N_paths, T_periods)
    """
    rng = np.random.default_rng(seed)
    if a_seq is None:
        a_seq = np.ones(T_periods)          # standard OLS

    beta_paths = np.empty((N_paths, T_periods))

    for i in range(N_paths):
        beta = beta0
        R = 1.0          # scalar moment estimate
        prec = 1.0 / R   # use precision for numerical stability

        for t in range(T_periods):
            alpha_t = a_seq[t]
            # z2 = 1 (constant regressor), so z2*z2' = 1
            z2 = 1.0
            u_t = rng.normal(0, sigma_u)

            # Actual z1 given current beta
            z1 = T_map(beta) * z2 + u_t

            # RLS update (lagged: use previous beta to form z1, then update)
            R = R + (alpha_t / (t + 1)) * (z2**2 - R / alpha_t)
            R = max(R, 1e-8)
            beta = beta + (alpha_t / (t + 1)) / R * z2 * (z1 - beta * z2)

            beta_paths[i, t] = beta

    return beta_paths


def solve_ode(f_ode, beta0, t_span=(0, 80), n_points=1000):
    """Solve scalar ODE d(beta)/dt = f_ode(beta) from beta0."""
    sol = solve_ivp(lambda t, y: [f_ode(y[0])], t_span, [beta0],
                    t_eval=np.linspace(*t_span, n_points), method='RK45',
                    max_step=0.1)
    return sol.t, sol.y[0]
```

## The Self-Referential Structure

### Perceived and Actual Laws of Motion

At each date $t$, agents hold a **perceived law of motion** summarised by a
parameter matrix $\beta_t$.

They believe that the variable $z_{1t}$ they care
about evolves according to

$$
z_{1t} = \beta_t z_{2,t-1} + \eta_t ,
$$ (eq:perceived_lom)

where $z_{2t}$ is a vector of variables agents use to forecast $z_{1,t+1}$, and
$\eta_t$ is orthogonal to all past $z_2$'s.

Because agents optimise (or behave) on the basis of this belief, their actions
feed back into the economy.

The **actual** law of motion for the full state
vector $z_t = (z_{1t}, z_{1t}^c)'$ is

$$
z_t = \begin{bmatrix} 0 & T(\beta_t) \\ A(\beta_t) & \end{bmatrix}
      \begin{bmatrix} z_{2,t-1}^c \\ z_{2,t-1} \end{bmatrix}
    + \begin{bmatrix} V(\beta_t) \\ B(\beta_t) \end{bmatrix} u_t ,
$$ (eq:actual_lom)

where $u_t$ is i.i.d. white noise with covariance $\Sigma$.

The mapping $T$ is the key object: it maps the **perceived** coefficient $\beta$
to the coefficient that **actually** governs $z_{1t}$ in equilibrium.

A
**rational expectations equilibrium** is a fixed point $\beta_f = T(\beta_f)$.

### The Learning Scheme

Agents update $\beta_t$ each period using **recursive least squares** (RLS).
Define $R_t$ as a running estimate of the second-moment matrix $Ez_{2t}z_{2t}'$.


Updating equations are

$$
\beta_t' = \beta_{t-1}' + \frac{\alpha_t}{t} R_{t-1}^{-1}
           z_{2,t-2} z_{2,t-2}' \bigl[ T(\beta_{t-1})' - \beta_{t-1}' \bigr]
         + \frac{\alpha_t}{t} z_{2,t-2} u_{t-1}' V(\beta_{t-1})' ,
$$ (eq:rls_beta)

$$
R_t = R_{t-1} + \frac{\alpha_t}{t} \bigl[ z_{2,t-1} z_{2,t-1}' - R_{t-1}/\alpha_t \bigr] ,
$$ (eq:rls_R)

where $\{\alpha_t\}$ is a positive, non-decreasing sequence with $\alpha_t \to 1$
as $t \to \infty$.  When $\alpha_t = 1$ for all $t$, equations
{eq}`eq:rls_beta`–{eq}`eq:rls_R` reduce to **ordinary least squares** updated
recursively.

```{note}
As {cite}`BraySavin1984` and {cite}`BrayKreps1987` emphasise, the RLS algorithm
cannot be derived from Bayes' rule applied to a correctly specified model, because
during the learning transition the data-generating process is non-stationary —
beliefs shift the equilibrium, which shifts the data.  The algorithm is
"irrational" in the  sense that it acts as if the environment were stationary,
when it is not.
```

## The Governing ODE

### Ljung's Differential-Equation Approach

{cite}`MarcetSargent1989jet` apply Ljung's theorem ({cite}`Ljung1977`) to
characterise the almost-sure limiting behaviour of the stochastic system
{eq}`eq:rls_beta`–{eq}`eq:rls_R`.

The central result is that the **only possible limit points** of $\beta_t$ are
fixed points of the ODE

$$
\frac{d\beta}{dt} = T(\beta) - \beta .
$$ (eq:small_ode)

This is the **small ODE** (equation (6) in {cite}`MarcetSargent1989jet`).

Its
fixed points are exactly the rational expectations equilibria.

The full ODE system associated with the joint process $(\beta_t, R_t)$ is

$$
\frac{d}{dt}\begin{bmatrix} \beta \\ R \end{bmatrix}
= \begin{bmatrix} R^{-1} M_{z_2}(\beta)\,[T(\beta) - \beta]' \\ M_{z_2}(\beta) - R \end{bmatrix} ,
$$ (eq:full_ode)

where $M_{z_2}(\beta) = Ez_{2t}z_{2t}'$ evaluated at the stationary distribution
induced by $\beta$.

The fixed point of {eq}`eq:full_ode` is $(\beta_f, R_f)$
where $R_f = M_{z_2}(\beta_f)$.

### Stability Governs Convergence

Let $\mathcal{M}$ be the Jacobian matrix of $T(\beta) - \beta$ evaluated at the
REE $\beta_f$:

$$
\mathcal{M} = \frac{d\,\text{col}(T(\beta) - \beta)}{d\,\text{col}(\beta)'}\Bigg|_{\beta=\beta_f} .
$$ (eq:jacobian)

**Proposition 3** of {cite}`MarcetSargent1989jet` establishes that the Jacobian of
the full system {eq}`eq:full_ode` at $(\beta_f, R_f)$ has $n_2^2$ repeated
eigenvalues equal to $-1$ (from the $R$ equation), plus the eigenvalues of
$\mathcal{M}$ (from the $\beta$ equation).

Consequently:

* If all eigenvalues of $\mathcal{M}$ have **strictly negative real parts**, both
  {eq}`eq:small_ode` and {eq}`eq:full_ode` are locally stable.  Under suitable
  boundedness conditions, Proposition 1 guarantees $\beta_t \to \beta_f$ **almost
  surely**.

* If any eigenvalue of $\mathcal{M}$ has **positive real part**, then
  $P(\beta_t \to \beta_f) = 0$ — convergence is **impossible**.

The stability condition $\text{Re}(\lambda_i(\mathcal{M})) < 0$ for all $i$ is
what the E-stability literature (see {cite}`Evans1985`) calls **E-stability**: the
REE is a stable rest point of the "expectational dynamics" $\dot\beta = T(\beta) - \beta$.

### The Projection Facility

E-stability is necessary but not quite sufficient for almost-sure convergence.

Ljung's theorem requires the sample path $(\beta_t, R_t)$ to remain in a
**bounded region** with probability one (assumptions A.6–A.7 of
{cite}`MarcetSargent1989jet`).

This boundedness is the job of the **projection
facility**.

#### What the Projection Facility Does

The full learning algorithm augments the plain RLS update with a constraint set
$D_1 \supset D_2$ in $(\beta, R)$-space.

After each unconstrained RLS step
produces a candidate $(\tilde\beta_t, \tilde R_t)$, the projection facility
enforces:

$$
(\beta_t, R_t) = \begin{cases}
  (\tilde\beta_t,\, \tilde R_t) & \text{if } (\tilde\beta_t, \tilde R_t) \in D_1 , \\
  \text{some point in } D_2     & \text{otherwise.}
\end{cases}
$$ (eq:projection)

The set $D_1$ is chosen so that the model remains well-defined (e.g., $R_t$
stays positive definite; $\beta_t$ stays in a region where $T(\beta)$ is
well-defined and the state process is covariance-stationary).

The set $D_2
\subset D_1$ is a slightly smaller "safe" region to which the algorithm is
retracted whenever it threatens to leave $D_1$.

The facility can be thought of as forcing agents to **discard observations that
are inconsistent with their priors** — a form of bounded rationality that is
necessary for the mathematical argument but innocuous in practice.

#### Why It Is Needed

Without the projection facility, the stochastic path $(\beta_t, R_t)$ might
temporarily wander to regions where the system {eq}`eq:actual_lom` is
non-stationary (e.g., an explosive VAR).

Ljung's convergence theorem requires
the algorithm to revisit a compact set infinitely often; the projection facility
guarantees this by construction.

Formally, {cite}`MarcetSargent1989jet` require that the ODE trajectories
originating in $D_1$ point **inward** at the boundary $\partial D_1$ — that is,
the vector field $T(\beta) - \beta$ must point back into $D_1$ everywhere on its
boundary.

When this holds (Assumption A.7.2), the projection is **invoked only
finitely many times** with probability one, and after the last invocation the
algorithm runs as plain RLS.

Corollary 1 of {cite}`MarcetSargent1989jet`
formalises this: either $\beta_t \to \beta_f$ a.s., or $\beta_t$ clusters on the
boundary $\partial D_1 \setminus D_2$ — but the latter event has probability zero
when the ODE trajectories point inward.

#### The Exogenous-Regressor Case (Corollary 2)

When the regressors $z_{2t}$ are **exogenous** — so that $M_{z_2}(\beta) \equiv M$
does not depend on $\beta$ — a particularly clean sufficient condition for
convergence is available (Corollary 2 of {cite}`MarcetSargent1989jet`):

$$
\text{all eigenvalues of } H(\beta) \equiv \frac{d\,\text{col}[T(\beta) - T(\beta_f)]}{d\,\text{col}[\beta - \beta_f]'}
\text{ have real parts} < 0 \quad \forall\, |\beta - \beta_f| \leq K .
$$ (eq:corollary2_cond)

Under this condition one can take $D_1$ to be a ball of radius $K$ around
$\beta_f$, and the boundary condition is automatically satisfied.

For all four
scalar examples in this lecture, $H(\beta) = \mathcal{M}$ is constant, so
Corollary 2 reduces simply to E-stability.

```{note}
In the scalar examples studied here (Bray, Bray–Savin, present-value model), the
state $z_{2t} = 1$ is a constant regressor, so $M_{z_2} = 1$ is trivially
exogenous.  For the investment model with endogenous regressors, verifying the
boundary condition on $D_1$ is much harder and may require numerical solution of
the ODE on a grid of boundary points.
```

#### Simulating the Projection Facility

The following code demonstrates the projection facility at work.

We use Bray's
model with $b = 0.6$ and deliberately start $\beta_0$ far from $\beta_f$,
imposing a projection set $D_1 = \{|\beta| < K\}$ with $K = 5$.

We track how
often the facility is invoked and show that after a finite number of
interventions, the path converges normally.

```{code-cell} ipython3
def simulate_rls_with_projection(T_map, sigma_u, beta0, K_proj,
                                 T_periods=500, N_paths=50, seed=0):
    """
    Simulate RLS with a scalar projection facility.

    The facility keeps beta_t in [-K_proj, K_proj].  Whenever the unconstrained
    update would push beta outside this interval, beta is retracted to 0
    (an arbitrary point in D2 = {|beta| <= K_proj/2}).

    Returns
    -------
    beta_paths      : (N_paths, T_periods) array of belief paths
    n_projections   : (N_paths,) array counting projection invocations per path
    first_proj_free : (N_paths,) array of first period with no further projections
    """
    rng = np.random.default_rng(seed)
    beta_paths    = np.empty((N_paths, T_periods))
    n_projections = np.zeros(N_paths, dtype=int)
    last_proj     = np.full(N_paths, -1, dtype=int)

    for i in range(N_paths):
        beta = beta0
        R    = 1.0

        for t in range(T_periods):
            u_t = rng.normal(0, sigma_u)
            z1  = T_map(beta) + u_t          # z2 = 1 (constant regressor)

            # Unconstrained RLS update
            R_new    = R    + (1.0 / (t + 1)) * (1.0 - R)
            beta_new = beta + (1.0 / (t + 1)) / R_new * (z1 - beta)

            # Projection facility: retract to D2 = {0} if outside D1
            if abs(beta_new) > K_proj:
                beta_new = 0.0           # retract to interior of D2
                n_projections[i] += 1
                last_proj[i] = t

            beta = beta_new
            R    = max(R_new, 1e-8)
            beta_paths[i, t] = beta

    # First period after which no further projections occur
    first_proj_free = last_proj + 1   # -1 + 1 = 0 if never projected

    return beta_paths, n_projections, first_proj_free


# Run the simulation
a_bray_pf, b_bray_pf, sigma_pf = 1.0, 0.6, 1.5
T_bray_pf  = lambda beta: a_bray_pf + b_bray_pf * beta
beta_f_pf  = a_bray_pf / (1 - b_bray_pf)
beta0_far  = 8.0    # well outside D1 = {|beta| < 5}
K_pf       = 5.0
T_pf_sim   = 600
N_pf_sim   = 80

paths_pf, n_proj, first_free = simulate_rls_with_projection(
    T_bray_pf, sigma_pf, beta0_far, K_pf,
    T_periods=T_pf_sim, N_paths=N_pf_sim)

# Also run without projection for comparison
paths_no_pf = simulate_rls_scalar(
    T_bray_pf, sigma_pf, beta0_far,
    T_periods=T_pf_sim, N_paths=N_pf_sim, seed=0)

fig = plt.figure(figsize=(15, 10))
gs  = GridSpec(2, 2, figure=fig)

# Top left: paths with projection
ax1 = fig.add_subplot(gs[0, 0])
for i in range(min(30, N_pf_sim)):
    ax1.plot(paths_pf[i], color='steelblue', alpha=0.25, lw=0.8)
ax1.plot(np.mean(paths_pf, axis=0), color='navy', lw=2, label='average')
ax1.axhline(beta_f_pf, color='red', ls='--', lw=1.5,
            label=f'$\\beta_f={beta_f_pf:.1f}$')
ax1.axhline( K_pf, color='gray', ls=':', lw=1.2, label=f'$D_1$ boundary ($K={K_pf}$)')
ax1.axhline(-K_pf, color='gray', ls=':', lw=1.2)
ax1.set_title('With Projection Facility ($\\beta_0=8$, $K=5$)')
ax1.set_xlabel('$t$'); ax1.set_ylabel('$\\beta_t$'); ax1.legend(fontsize=8)

# Top right: paths without projection
ax2 = fig.add_subplot(gs[0, 1])
for i in range(min(30, N_pf_sim)):
    ax2.plot(paths_no_pf[i], color='darkorange', alpha=0.25, lw=0.8)
ax2.plot(np.mean(paths_no_pf, axis=0), color='saddlebrown', lw=2, label='average')
ax2.axhline(beta_f_pf, color='red', ls='--', lw=1.5,
            label=f'$\\beta_f={beta_f_pf:.1f}$')
ax2.set_title('Without Projection Facility ($\\beta_0=8$)')
ax2.set_xlabel('$t$'); ax2.set_ylabel('$\\beta_t$'); ax2.legend(fontsize=8)

# Bottom left: histogram of projection counts
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(n_proj, bins=range(0, int(n_proj.max()) + 2),
         color='steelblue', edgecolor='white', alpha=0.8)
ax3.set_xlabel('Number of projections invoked')
ax3.set_ylabel('Number of paths')
ax3.set_title('Distribution of Projection Invocations\n'
              '(finite a.s. — Corollary 1)')

# Bottom right: period of last projection
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(first_free[n_proj > 0], bins=20,
         color='darkorange', edgecolor='white', alpha=0.8)
ax4.set_xlabel('Last period with a projection')
ax4.set_ylabel('Number of paths')
ax4.set_title('After the Last Projection, RLS Runs Freely\n'
              '(projection invoked only finitely many times)')

plt.tight_layout()
plt.show()

print(f"Paths with at least one projection: {(n_proj > 0).sum()} / {N_pf_sim}")
print(f"Mean number of projections per path: {n_proj.mean():.2f}")
print(f"Max number of projections:           {n_proj.max()}")
print(f"Mean last-projection period:         {first_free[n_proj>0].mean():.1f}")
```

The simulation illustrates the key theoretical point from Corollary 1: the
projection is invoked only a **finite number of times** on almost every sample
path.  After the last invocation the algorithm runs as unconstrained RLS and
converges to $\beta_f$ at the usual rate.  The projection does not bias the
asymptotic estimate — it merely provides the boundedness guarantee that Ljung's
theorem requires.

## Four Illustrative Examples

We now work through four examples from Section 4 of {cite}`MarcetSargent1989jet`,
computing the ODE, finding the REE, checking E-stability, and simulating the RLS
learning path.

### Example 1: Bray's Cobweb Model

{cite}`Bray1982` studied a simple cobweb economy in which the equilibrium price
satisfies

$$
p_t = a + b \beta_t + \tilde{u}_t ,
$$ (eq:bray_price)

where $\beta_t$ is agents' OLS estimate of the price (their point forecast of
$p_t$), and $\tilde{u}_t$ is i.i.d. noise with mean zero and variance
$\sigma_u^2$.

The mapping $T$ is simply $T(\beta) = a + b\beta$.  The REE is

$$
\beta_f = \frac{a}{1 - b} , \quad |b| < 1 .
$$ (eq:bray_ree)

The small ODE is

$$
\dot\beta = T(\beta) - \beta = a + b\beta - \beta = a - (1-b)\beta ,
$$ (eq:bray_ode)

which has the unique fixed point $\beta_f = a/(1-b)$.

Its Jacobian is
$\mathcal{M} = b - 1 < 0$ when $|b| < 1$, so the REE is E-stable and RLS
converges almost surely.  When $b > 1$, $\mathcal{M} > 0$ and convergence fails.

### Example 2: Bray–Savin Supply-Shifter Model

{cite}`BraySavin1984` studied a model where

$$
p_t = x_t'(m + a\beta_{t-1}) + \tilde{u}_t , \quad p_t^e = x_t'\beta_{t-1} ,
$$ (eq:bs_price)

with $x_t$ an exogenous supply-shifter, $a$ a scalar feedback parameter, and
agents running an OLS regression of $p$ on $x$.

The mapping is $T(\beta) = m + a\beta$ (scalar case), giving

$$
\dot\beta = (a-1)\beta + m , \quad \beta_f = \frac{m}{1-a} ,
$$ (eq:bs_ode)

with Jacobian $\mathcal{M} = a - 1 < 0$ iff $a < 1$.

### Example 3: Hyperinflation / Asset Prices (Fourgeaud–Gourieroux–Pradel)

Consider the present-value asset pricing model

$$
y_t = \lambda E_t y_{t+1} + x_t , \quad x_t = \rho x_{t-1} + \varepsilon_t ,
$$ (eq:pv_model)

where $|\lambda| < 1$, $|\rho| < 1$, and agents perceive $y_t = \beta_t x_{t-1}+ v_t$.
 
The mapping is $T(\beta) = (\lambda\beta + 1)\rho$ and the REE is

$$
\beta_f = \frac{\rho}{1 - \lambda\rho} .
$$ (eq:pv_ree)

The small ODE is

$$
\dot\beta = (\lambda\rho - 1)\beta + \rho ,
$$ (eq:pv_ode)

with Jacobian $\mathcal{M} = \lambda\rho - 1 < 0$ for $|\lambda\rho| < 1$, so
convergence is guaranteed.

### Example 4: Investment under Uncertainty (Self-Referential with Endogenous Regressors)

In Sargent's version of the Lucas–Prescott investment model, agents learn about the
aggregate capital stock $K_t$ by regressing on $(K_{t-1}, w_{t-1})$ where $w_t$
is an exogenous cost shock.

The perceived law of motion is

$$
K_t = \beta_1 K_{t-1} + \beta_2 w_{t-1} + \eta_t ,
$$

while the actual law (from firms' optimal investment decisions and market clearing) is

$$
K_t = T_1(\beta) K_{t-1} + T_2(\beta) w_{t-1} + V(\beta) u_t ,
$$ (eq:inv_actual)

where the nonlinear mappings $T_1, T_2$ come from solving the firms' linear
quadratic control problems.

The small ODE decomposes as:

$$
\dot\beta_1 = T_1(\beta_1) - \beta_1 , \quad
\dot\beta_2 = T_2(\beta_1, \beta_2) - \beta_2 ,
$$ (eq:inv_ode)

and E-stability can be verified analytically for $|\beta_1| < b^{-1/2}$ (where
$b$ is the discount factor).

## Simulating the Learning Dynamics

We now simulate all four examples numerically, plotting both the ODE solution
(continuous-time approximation) and the sample paths of $\beta_t$ under RLS.

### Bray's Model

```{code-cell} ipython3
# ------------------------------------------------------------------
# Bray's cobweb model: T(beta) = a + b*beta,  REE = a/(1-b)
# ------------------------------------------------------------------
a_bray, b_bray, sigma_bray = 1.0, 0.6, 1.0
T_bray = lambda beta: a_bray + b_bray * beta
beta_f_bray = a_bray / (1 - b_bray)

beta0_bray = 0.0   # start well below the REE
T_sim = 400
N_sim = 80

beta_paths_bray = simulate_rls_scalar(T_bray, sigma_bray, beta0_bray,
                                      T_periods=T_sim, N_paths=N_sim)

# ODE solution for two starting values
ode_bray = lambda beta: a_bray + b_bray * beta - beta
t_ode, sol_low  = solve_ode(ode_bray, 0.0)
_,     sol_high = solve_ode(ode_bray, 4.5)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for i in range(min(30, N_sim)):
    ax.plot(beta_paths_bray[i], color='steelblue', alpha=0.25, lw=0.8)
ax.plot(np.mean(beta_paths_bray, axis=0), color='navy', lw=2,
        label='cross-path average')
ax.axhline(beta_f_bray, color='red', ls='--', lw=1.5,
           label=f'$\\beta_f = {beta_f_bray:.2f}$')
ax.set_xlabel('$t$')
ax.set_ylabel('$\\beta_t$')
ax.set_title("Bray's Model: RLS Paths ($b=0.6$)")
ax.legend()

ax = axes[1]
ax.plot(t_ode, sol_low,  color='steelblue', lw=2, label='ODE from $\\beta_0=0$')
ax.plot(t_ode, sol_high, color='darkorange', lw=2, label='ODE from $\\beta_0=4.5$')
ax.axhline(beta_f_bray, color='red', ls='--', lw=1.5,
           label=f'$\\beta_f = {beta_f_bray:.2f}$')
ax.set_xlabel('$t$')
ax.set_ylabel('$\\beta(t)$')
ax.set_title("Bray's Model: ODE Trajectories")
ax.legend()

plt.tight_layout()
plt.show()
print(f"REE: beta_f = a/(1-b) = {beta_f_bray:.4f}")
print(f"Jacobian M = b - 1 = {b_bray - 1:.4f}  (< 0: E-stable)")
```

### Bray–Savin Model

```{code-cell} ipython3
# ------------------------------------------------------------------
# Bray–Savin: T(beta) = m + a*beta,  REE = m/(1-a)
# ------------------------------------------------------------------
m_bs, a_bs, sigma_bs = 0.5, 0.7, 1.0
T_bs = lambda beta: m_bs + a_bs * beta
beta_f_bs = m_bs / (1 - a_bs)

beta_paths_bs = simulate_rls_scalar(T_bs, sigma_bs, 0.0,
                                    T_periods=T_sim, N_paths=N_sim)

ode_bs = lambda beta: T_bs(beta) - beta
t_ode_bs, sol_bs_low  = solve_ode(ode_bs, 0.0)
_,         sol_bs_high = solve_ode(ode_bs, 4.0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for i in range(min(30, N_sim)):
    ax.plot(beta_paths_bs[i], color='darkorange', alpha=0.25, lw=0.8)
ax.plot(np.mean(beta_paths_bs, axis=0), color='saddlebrown', lw=2,
        label='cross-path average')
ax.axhline(beta_f_bs, color='red', ls='--', lw=1.5,
           label=f'$\\beta_f = {beta_f_bs:.2f}$')
ax.set_xlabel('$t$'); ax.set_ylabel('$\\beta_t$')
ax.set_title('Bray–Savin Model: RLS Paths ($a=0.7$)')
ax.legend()

ax = axes[1]
ax.plot(t_ode_bs, sol_bs_low,  color='darkorange', lw=2, label='ODE from $\\beta_0=0$')
ax.plot(t_ode_bs, sol_bs_high, color='steelblue',  lw=2, label='ODE from $\\beta_0=4$')
ax.axhline(beta_f_bs, color='red', ls='--', lw=1.5,
           label=f'$\\beta_f = {beta_f_bs:.2f}$')
ax.set_xlabel('$t$'); ax.set_ylabel('$\\beta(t)$')
ax.set_title('Bray–Savin Model: ODE Trajectories')
ax.legend()

plt.tight_layout()
plt.show()
print(f"REE: beta_f = m/(1-a) = {beta_f_bs:.4f}")
print(f"Jacobian M = a - 1 = {a_bs - 1:.4f}  (< 0: E-stable)")
```

### Present-Value / Hyperinflation Model

```{code-cell} ipython3
# ------------------------------------------------------------------
# Present-value model: T(beta) = (lambda*beta + 1)*rho
# REE = rho / (1 - lambda*rho)
# ------------------------------------------------------------------
lam, rho_pv, sigma_pv = 0.8, 0.9, 1.0
T_pv = lambda beta: (lam * beta + 1) * rho_pv
beta_f_pv = rho_pv / (1 - lam * rho_pv)

beta_paths_pv = simulate_rls_scalar(T_pv, sigma_pv, 0.0,
                                    T_periods=T_sim, N_paths=N_sim)

ode_pv = lambda beta: T_pv(beta) - beta
t_ode_pv, sol_pv_low  = solve_ode(ode_pv, 0.0, t_span=(0, 50))
_,         sol_pv_high = solve_ode(ode_pv, 10.0, t_span=(0, 50))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for i in range(min(30, N_sim)):
    ax.plot(beta_paths_pv[i], color='seagreen', alpha=0.25, lw=0.8)
ax.plot(np.mean(beta_paths_pv, axis=0), color='darkgreen', lw=2,
        label='cross-path average')
ax.axhline(beta_f_pv, color='red', ls='--', lw=1.5,
           label=f'$\\beta_f = {beta_f_pv:.2f}$')
ax.set_xlabel('$t$'); ax.set_ylabel('$\\beta_t$')
ax.set_title('Present-Value Model: RLS Paths')
ax.legend()

ax = axes[1]
ax.plot(t_ode_pv, sol_pv_low,  color='seagreen',  lw=2, label='ODE from $\\beta_0=0$')
ax.plot(t_ode_pv, sol_pv_high, color='steelblue', lw=2, label='ODE from $\\beta_0=10$')
ax.axhline(beta_f_pv, color='red', ls='--', lw=1.5,
           label=f'$\\beta_f = {beta_f_pv:.2f}$')
ax.set_xlabel('$t$'); ax.set_ylabel('$\\beta(t)$')
ax.set_title('Present-Value Model: ODE Trajectories')
ax.legend()

plt.tight_layout()
plt.show()
print(f"REE: beta_f = rho/(1 - lambda*rho) = {beta_f_pv:.4f}")
print(f"Jacobian M = lambda*rho - 1 = {lam*rho_pv - 1:.4f}  (< 0: E-stable)")
```

### Instability When E-Stability Fails

To see what happens when E-stability is violated, we repeat Bray's model with $b > 1$.

```{code-cell} ipython3
# ------------------------------------------------------------------
# Unstable case: Bray's model with b > 1
# ------------------------------------------------------------------
b_unstable = 1.4
T_unstable = lambda beta: a_bray + b_unstable * beta
beta_f_unstable = a_bray / (1 - b_unstable)   # negative

beta_paths_unstable = simulate_rls_scalar(
    T_unstable, sigma_bray, beta0=0.0,
    T_periods=200, N_paths=50)

ode_unstable = lambda beta: T_unstable(beta) - beta

# Phase diagram: plot drift for beta in [-5, 5]
beta_grid = np.linspace(-5, 5, 300)
drift = np.array([ode_unstable(b) for b in beta_grid])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for i in range(min(30, 50)):
    ax.plot(beta_paths_unstable[i], color='crimson', alpha=0.3, lw=0.8)
ax.axhline(beta_f_unstable, color='black', ls='--', lw=1.5,
           label=f'$\\beta_f = {beta_f_unstable:.2f}$ (unstable)')
ax.set_xlabel('$t$'); ax.set_ylabel('$\\beta_t$')
ax.set_title('Bray Model with $b=1.4$: RLS Diverges')
ax.legend()

ax = axes[1]
ax.plot(beta_grid, drift, color='crimson', lw=2)
ax.axhline(0, color='black', lw=0.8)
ax.axvline(beta_f_unstable, color='black', ls='--', lw=1.5,
           label=f'$\\beta_f = {beta_f_unstable:.2f}$')
ax.fill_between(beta_grid, drift, 0,
                where=(drift > 0), color='crimson', alpha=0.15)
ax.fill_between(beta_grid, drift, 0,
                where=(drift < 0), color='steelblue', alpha=0.15)
ax.set_xlabel('$\\beta$'); ax.set_ylabel('$T(\\beta) - \\beta$')
ax.set_title('Phase Diagram: Drift Points Away from REE')
ax.legend()

plt.tight_layout()
plt.show()
print(f"Jacobian M = b - 1 = {b_unstable - 1:.2f}  (> 0: NOT E-stable)")
```

## Phase Diagrams and E-Stability

The E-stability condition has a clean geometric interpretation.  At the REE
$\beta_f$, the small ODE {eq}`eq:small_ode` must have trajectories **pointing
inward**.

This requires the slope $dT/d\beta - 1$ to be **negative** at $\beta_f$.

The figure below plots the phase diagrams for all three scalar examples side by
side.

```{code-cell} ipython3
beta_vec = np.linspace(-1.0, 5.5, 400)

models = [
    ("Bray ($b=0.6$)",       lambda b: a_bray + 0.6*b - b,   a_bray/(1-0.6),   'steelblue'),
    ("Bray–Savin ($a=0.7$)", lambda b: m_bs + 0.7*b - b,     m_bs/(1-0.7),     'darkorange'),
    ("Present-value",        lambda b: T_pv(b) - b,           beta_f_pv,        'seagreen'),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, ode_fn, bf, color) in zip(axes, models):
    drift = np.array([ode_fn(b) for b in beta_vec])
    ax.plot(beta_vec, drift, color=color, lw=2)
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(bf, color='red', ls='--', lw=1.5, label=f'$\\beta_f={bf:.2f}$')
    ax.fill_between(beta_vec, drift, 0, where=(drift > 0),
                    color=color, alpha=0.12)
    ax.fill_between(beta_vec, drift, 0, where=(drift < 0),
                    color=color, alpha=0.12)
    # Draw arrows showing direction of drift
    for bv in np.linspace(beta_vec[20], beta_vec[-20], 7):
        d = ode_fn(bv)
        ax.annotate('', xy=(bv + 0.3*np.sign(d), 0),
                    xytext=(bv, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('$T(\\beta) - \\beta$')
    ax.set_title(name)
    ax.legend(fontsize=9)

plt.suptitle('Phase Diagrams of the Small ODE $\\dot{\\beta} = T(\\beta) - \\beta$',
             y=1.01, fontsize=13)
plt.tight_layout()
plt.show()
```

## Two-Dimensional Example: The Investment Model

The investment-under-uncertainty example is two-dimensional and highlights how
E-stability of the composite map $T(\beta) = (T_1(\beta_1), T_2(\beta_1, \beta_2))$
works when the ODE is recursive.

```{code-cell} ipython3
def T_invest(beta, b=0.95, d=1.0, f=1.0, A1=1.0, N=1.0, rho_w=0.5):
    """
    Mapping T for the investment model (scalar version of equations 11 in
    Marcet–Sargent 1989).

    beta = [beta1, beta2]
    T1(beta1) = (1 - beta1*b) / (1 - beta1*b + d^{-1} f^2 A1 N)
    T2(beta1, beta2) = -N/(d*(1-rho_w*b)) * (1 - beta1*b + f^2 A1 beta2 b*rho_w)
                       / (1 - beta1*b + d^{-1} f^2 A1 N) * rho_w
    """
    b1, b2 = beta
    denom1 = 1 - b1*b + (1/d)*f**2*A1*N
    T1 = (1 - b1*b) / denom1
    numer2 = (1 - b1*b + f**2*A1*b2*b*rho_w)
    T2 = (-N / (d*(1 - rho_w*b))) * (numer2 / denom1) * rho_w
    return np.array([T1, T2])


def ode_invest(t, beta, **kwargs):
    Tb = T_invest(beta, **kwargs)
    return Tb - beta


# REE: solve T(beta) = beta numerically
from scipy.optimize import fsolve

params = dict(b=0.95, d=1.0, f=1.0, A1=1.0, N=1.0, rho_w=0.5)
beta_f_inv = fsolve(lambda b: T_invest(b, **params) - b, [0.5, 0.1])
print(f"REE: beta_f = {beta_f_inv}")

# Check E-stability via Jacobian
from numpy import linalg as la

eps = 1e-6
J = np.zeros((2, 2))
for j in range(2):
    e = np.zeros(2); e[j] = eps
    J[:, j] = (T_invest(beta_f_inv + e, **params) -
               T_invest(beta_f_inv - e, **params)) / (2*eps)
M = J - np.eye(2)
eigs = la.eigvals(M)
print(f"Jacobian M eigenvalues: {eigs}")
print(f"E-stable: {np.all(eigs.real < 0)}")

# Solve ODE from several initial conditions
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the vector field
b1_grid = np.linspace(-0.1, 1.2, 20)
b2_grid = np.linspace(-0.8, 0.5, 20)
B1, B2 = np.meshgrid(b1_grid, b2_grid)
U = np.zeros_like(B1); V_field = np.zeros_like(B2)
for i in range(B1.shape[0]):
    for j in range(B1.shape[1]):
        beta_ij = np.array([B1[i,j], B2[i,j]])
        drift = T_invest(beta_ij, **params) - beta_ij
        U[i,j] = drift[0]; V_field[i,j] = drift[1]

speed = np.sqrt(U**2 + V_field**2)
speed[speed == 0] = 1e-8
ax.streamplot(b1_grid, b2_grid, U, V_field, color=speed,
              cmap='Blues', density=1.3, linewidth=1)

# Plot trajectories from several starts
starts = [(0.1, 0.0), (0.9, 0.4), (1.1, -0.6), (0.3, -0.7)]
colors_traj = ['red', 'darkorange', 'green', 'purple']
for (b10, b20), col in zip(starts, colors_traj):
    sol = solve_ivp(ode_invest, [0, 30], [b10, b20],
                    t_eval=np.linspace(0, 30, 300),
                    kwargs=params, method='RK45')
    ax.plot(sol.y[0], sol.y[1], color=col, lw=2)
    ax.plot(b10, b20, 'o', color=col, ms=7)

ax.plot(*beta_f_inv, 'k*', ms=14, label=f'REE $\\beta_f$')
ax.set_xlabel('$\\beta_1$', fontsize=12)
ax.set_ylabel('$\\beta_2$', fontsize=12)
ax.set_title('Investment Model: Phase Portrait of $\\dot{\\beta} = T(\\beta) - \\beta$')
ax.legend()
plt.tight_layout()
plt.show()
```

## Necessary Condition: Only REE Can Be Limit Points

Proposition 2(i) of {cite}`MarcetSargent1989jet` shows that **non-REE limit points
have probability zero**: for any $\hat\beta \neq \beta_f$ in the interior of the
domain,

$$
P(\beta_t \to \hat\beta) = 0 .
$$

This is a converse: RLS either converges to the REE or it diverges.

It
cannot converge to a non-equilibrium fixed point.

The following simulation makes this vivid by starting agents with an initial
belief that happens to satisfy $T(\beta_0) \approx \beta_0$ only approximately.

```{code-cell} ipython3
# Illustration: starting near a non-fixed-point of T still sends beta to beta_f
# (Bray model, stable case b=0.6)
beta_false_rest = 3.0   # T(3.0) = 1 + 0.6*3 = 2.8 ≠ 3
paths_from_false = simulate_rls_scalar(
    T_bray, sigma_bray, beta0=beta_false_rest,
    T_periods=300, N_paths=60, seed=7)

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(60):
    ax.plot(paths_from_false[i], color='steelblue', alpha=0.2, lw=0.8)
ax.plot(np.mean(paths_from_false, axis=0), color='navy', lw=2,
        label='cross-path average')
ax.axhline(beta_f_bray, color='red', ls='--', lw=1.5,
           label=f'REE $\\beta_f = {beta_f_bray:.2f}$')
ax.axhline(beta_false_rest, color='gray', ls=':', lw=1.5,
           label=f'False start $\\beta_0 = {beta_false_rest}$')
ax.set_xlabel('$t$'); ax.set_ylabel('$\\beta_t$')
ax.set_title('RLS from a Non-Equilibrium Start Always Converges to the REE\n'
             '(Proposition 2(i): only the REE is a possible limit point)')
ax.legend()
plt.tight_layout()
plt.show()
```

## Connection to Rational Learning 

The {cite}`MarcetSargent1989jet` framework belongs to the programme of learning
*about* a rational expectations equilibrium, as distinct from learning *within*
one — a distinction emphasised by {cite}`BrayKreps1987`.

**Learning *within* an REE** (the subject of the companion lecture
{doc}`rational_learning_re`) refers to Bayesian inference inside a correctly
specified model.

In that setting the data-generating process is stationary from
the agent's perspective, and Bayes' rule is fully rationalized.

**Learning *about* an REE** — the present lecture's topic — involves an agent who
does not know the equilibrium price function.

Because the agent's beliefs shift
the equilibrium price, the data the agent uses to update beliefs are themselves
generated by a non-stationary process.

As {cite}`MarcetSargent1989jet` note (p.
338, footnote 2):

> *"The models do not incorporate fully optimal behavior or rational expectations,
> because agents operate under the continually falsified assumption that the law of
> motion is time invariant and known for sure."*

This "continually falsified" assumption is precisely the sense in which the RLS
algorithm cannot be derived from Bayesian rationality applied to a correctly
specified model.

It is nonetheless a compelling learning rule because it is
consistent, computationally tractable, and — when E-stability holds — converges to
the REE despite the misspecification.

The E-stability condition thus plays the same role in this literature that the
prior-support condition plays in the Bayesian learning literature: it tells us
when the learning algorithm can find its way to the equilibrium.


## Summary

This lecture has presented the {cite}`MarcetSargent1989jet` framework for analysing
least squares learning in self-referential linear stochastic models.

Key takeaways:

1. **Self-referential structure**: the actual law of motion depends on the
   perceived law of motion through the mapping $T$.  A rational expectations
   equilibrium is a fixed point $\beta_f = T(\beta_f)$.

2. **Recursive least squares**: agents update their beliefs by running RLS,
   which is adaptive but not fully Bayesian — it "continually falsifies" the
   assumption that the environment is stationary.

3. **The governing ODE**: the almost-sure limiting behaviour of $\beta_t$ is
   described by the small ODE $\dot\beta = T(\beta) - \beta$.  Only fixed
   points of this ODE (REE) are possible limit points of RLS.

4. **E-stability**: the REE is the almost-sure limit of RLS if and only if
   it is a **locally stable** fixed point of the small ODE — that is, if all
   eigenvalues of the Jacobian $\mathcal{M} = dT/d\beta - I$ at $\beta_f$ have
   strictly negative real parts.

5. **Instability**: if any eigenvalue of $\mathcal{M}$ has positive real part,
   $P(\beta_t \to \beta_f) = 0$ — convergence to that REE is impossible.

6. **Connection to the rational learning literature**: the RLS algorithm
   studies learning *about* a rational expectations equilibrium; it is
   complementary to the Bayesian learning *within* an REE studied by
   {cite}`BrayKreps1987`.

## Exercises

```{exercise}
:label: ls_ex1

**E-Stability and the Slope of T**

Consider the scalar model with $T(\beta) = a + b\beta$.

(a) Derive a formula for the unique REE $\beta_f$ in terms of $a$ and $b$.

(b) Show that the small ODE $\dot\beta = T(\beta) - \beta$ is globally stable if
and only if $b < 1$.

(c) Simulate $N = 200$ paths of length $T = 500$ for $a = 1$ and each of
$b \in \{0.3, 0.7, 0.9, 0.99\}$ (all less than 1).  Plot the cross-path
average of $\beta_t$ for each $b$ value on the same figure.  Comment on how the
rate of convergence changes as $b \to 1$.
```

```{solution-start} ls_ex1
:class: dropdown
```

**(a)** The REE satisfies $\beta_f = T(\beta_f) = a + b\beta_f$, so

$$
\beta_f (1 - b) = a \implies \beta_f = \frac{a}{1-b} .
$$

**(b)** The small ODE is $\dot\beta = a + b\beta - \beta = a - (1-b)\beta$.
This is linear with slope $-(1-b)$, so the unique fixed point $\beta_f = a/(1-b)$
is globally stable iff $1-b > 0$, i.e., $b < 1$.

**(c)**

```{code-cell} ipython3
a_ex, T_ex, N_ex = 1.0, 500, 200
b_values = [0.3, 0.7, 0.9, 0.99]
colors_ex = ['steelblue', 'darkorange', 'seagreen', 'purple']

fig, ax = plt.subplots(figsize=(11, 5))
for b_val, col in zip(b_values, colors_ex):
    T_fn = lambda beta, bv=b_val: a_ex + bv * beta
    paths = simulate_rls_scalar(T_fn, sigma_u=1.0, beta0=0.0,
                                T_periods=T_ex, N_paths=N_ex, seed=0)
    bf = a_ex / (1 - b_val)
    ax.plot(np.mean(paths, axis=0), color=col, lw=2,
            label=f'$b={b_val}$, $\\beta_f={bf:.2f}$')

ax.set_xlabel('$t$')
ax.set_ylabel('$E[\\beta_t]$')
ax.set_title('Convergence Rate Slows as $b \\to 1$')
ax.legend()
plt.tight_layout()
plt.show()

print("As b → 1, the Jacobian M = b - 1 → 0, so the ODE becomes slow to")
print("return to the fixed point.  Convergence still occurs but takes longer.")
```

```{solution-end}
```

```{exercise}
:label: ls_ex2

**Necessary Condition: Non-REE Limit Points**

Proposition 2(i) of {cite}`MarcetSargent1989jet` states that $P(\beta_t \to \hat\beta) = 0$
for any $\hat\beta \neq \beta_f$ in the interior.

(a) Using the Bray model with $a=1$, $b=0.6$, simulate 100 paths of length
$T = 600$ starting from $\beta_0 = 6$ (far from $\beta_f = 2.5$).  Show that
paths still converge to $\beta_f$.

(b) Now consider the **unstable** case $b = 1.5$.  Simulate 50 paths of length
$T = 200$ starting from $\beta_0 = 0.1$ (close to the REE $\beta_f = -2$).
Describe what happens.

(c) For the unstable case, plot the phase diagram and explain geometrically why
the paths diverge.
```

```{solution-start} ls_ex2
:class: dropdown
```

**(a) and (b)**

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) far start, stable case
T_st = lambda beta: 1.0 + 0.6*beta
paths_far = simulate_rls_scalar(T_st, 1.0, beta0=6.0,
                                T_periods=600, N_paths=100, seed=1)
ax = axes[0]
for i in range(40):
    ax.plot(paths_far[i], color='steelblue', alpha=0.2, lw=0.8)
ax.plot(np.mean(paths_far, axis=0), color='navy', lw=2, label='average')
ax.axhline(2.5, color='red', ls='--', lw=1.5, label='$\\beta_f = 2.5$')
ax.set_title('Stable ($b=0.6$): far start still converges')
ax.set_xlabel('$t$'); ax.set_ylabel('$\\beta_t$'); ax.legend()

# (b) unstable case, start near REE
T_un = lambda beta: 1.0 + 1.5*beta
beta_f_un = 1.0 / (1 - 1.5)   # = -2
paths_un = simulate_rls_scalar(T_un, 1.0, beta0=0.1,
                               T_periods=200, N_paths=50, seed=2)
ax = axes[1]
for i in range(50):
    ax.plot(paths_un[i], color='crimson', alpha=0.3, lw=0.8)
ax.axhline(beta_f_un, color='black', ls='--', lw=1.5,
           label=f'$\\beta_f = {beta_f_un}$ (unstable)')
ax.set_title('Unstable ($b=1.5$): diverges even near REE')
ax.set_xlabel('$t$'); ax.set_ylabel('$\\beta_t$'); ax.legend()

plt.tight_layout()
plt.show()
```

**(c)** Phase diagram of the unstable case:

```{code-cell} ipython3
beta_g = np.linspace(-8, 6, 400)
drift_un = np.array([1.0 + 1.5*b - b for b in beta_g])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(beta_g, drift_un, color='crimson', lw=2)
ax.axhline(0, color='black', lw=0.8)
ax.axvline(beta_f_un, color='black', ls='--', lw=1.5,
           label=f'$\\beta_f = {beta_f_un}$')
ax.fill_between(beta_g, drift_un, 0, where=(drift_un > 0),
                color='crimson', alpha=0.15)
ax.fill_between(beta_g, drift_un, 0, where=(drift_un < 0),
                color='steelblue', alpha=0.15)
ax.set_xlabel('$\\beta$'); ax.set_ylabel('$T(\\beta) - \\beta$')
ax.set_title('Phase Diagram: Unstable REE ($b=1.5$)\n'
             'Drift points away from $\\beta_f$ everywhere')
ax.legend()
plt.tight_layout()
plt.show()

print("Geometrically: the slope dT/d(beta) - 1 = b - 1 = 0.5 > 0 at the REE,")
print("so the ODE pushes beta AWAY from beta_f in both directions.")
```

```{solution-end}
```

```{exercise}
:label: ls_ex3

**The Present-Value Model: Effect of $\lambda$ on E-Stability**

In the present-value model {eq}`eq:pv_model`, $T(\beta) = (\lambda\beta + 1)\rho$
and the Jacobian is $\mathcal{M} = \lambda\rho - 1$.

(a) For $\rho = 0.9$ and each of $\lambda \in \{0.5, 0.8, 0.95, 1.0\}$:
    - Compute $\beta_f$ and $\mathcal{M}$
    - Determine whether the REE is E-stable

(b) For the E-stable cases, simulate 100 paths of length $T=400$ and
plot the cross-path average against the ODE solution.

(c) At $\lambda = 1$, $\mathcal{M} = \rho - 1 < 0$ (still E-stable when
$|\rho| < 1$).  Simulate paths for this case and compare the convergence
speed with the $\lambda = 0.5$ case.  Provide an intuitive explanation.
```

```{solution-start} ls_ex3
:class: dropdown
```

**(a)**

```{code-cell} ipython3
rho_ex = 0.9
lambdas = [0.5, 0.8, 0.95, 1.0]

print(f"{'lambda':>8}  {'beta_f':>10}  {'M = lam*rho-1':>15}  {'E-stable':>10}")
print("-" * 50)
for lv in lambdas:
    bf = rho_ex / (1 - lv * rho_ex) if abs(lv * rho_ex) < 1 else float('inf')
    M_jac = lv * rho_ex - 1
    estab = "YES" if M_jac < 0 else "NO"
    print(f"{lv:>8.2f}  {bf:>10.4f}  {M_jac:>15.4f}  {estab:>10}")
```

**(b) and (c)**

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors_lam = ['steelblue', 'darkorange', 'seagreen', 'purple']

for ax, lv, col in zip(axes.flat, lambdas, colors_lam):
    T_fn = lambda beta, l=lv: (l * beta + 1) * rho_ex
    ode_fn = lambda beta, l=lv: T_fn(beta, l) - beta
    bf = rho_ex / (1 - lv * rho_ex) if abs(lv * rho_ex) < 1 else None

    paths_lam = simulate_rls_scalar(T_fn, 1.0, beta0=0.0,
                                    T_periods=400, N_paths=100, seed=3)
    for i in range(20):
        ax.plot(paths_lam[i], color=col, alpha=0.2, lw=0.8)
    ax.plot(np.mean(paths_lam, axis=0), color=col, lw=2, label='RLS average')

    if bf is not None:
        # ODE solution
        t_o, sol_o = solve_ode(ode_fn, 0.0, t_span=(0, 400), n_points=400)
        ax.plot(t_o, sol_o, color='black', ls='--', lw=1.5, label='ODE')
        ax.axhline(bf, color='red', ls=':', lw=1.2,
                   label=f'$\\beta_f={bf:.2f}$')

    M_jac = lv * rho_ex - 1
    ax.set_title(f'$\\lambda={lv}$,  $\\mathcal{{M}}={M_jac:.3f}$')
    ax.set_xlabel('$t$'); ax.set_ylabel('$\\beta_t$')
    ax.legend(fontsize=8)

plt.suptitle('Present-Value Model: Convergence for Different $\\lambda$ Values',
             y=1.02, fontsize=13)
plt.tight_layout()
plt.show()

print("\n(c) When lambda=1, M = rho-1 ≈ -0.1 (small in absolute value).")
print("    This means the ODE is very 'flat' near beta_f: the restoring force")
print("    is weak and convergence is slow.  When lambda=0.5, M = -0.55,")
print("    giving a stronger restoring force and faster convergence.")
```

```{solution-end}
```

