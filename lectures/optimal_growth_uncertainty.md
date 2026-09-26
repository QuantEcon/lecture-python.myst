---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(optimal_growth_uncertainty)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Optimal Growth Under Uncertainty and Tobin's q

```{contents} Contents
:depth: 2
```

## Overview

This lecture studies two papers.

The first is {cite}`BrockMirman1972`, a classic analysis of optimal one-sector growth when the production function is hit by random shocks.

The second is {cite}`Sargent1980q`, which adds one small ingredient -- **irreversible investment** -- and uses the result to think about James Tobin's $q$ theory of investment.

The lecture {doc}`lucas_prescott_investment` studied a competitive industry whose equilibrium is the solution of a planning problem, and whose equilibrium is a Markov process with an invariant distribution.

Brock and Mirman ask the same questions about a one-sector growth model:

* does the planning problem have a well-behaved solution, with continuous and monotone policy functions?
* the optimal policy makes capital a Markov process; does that process have an **invariant distribution**?
* does the distribution of capital converge to it from *any* initial condition?
* do **time averages** computed from a single realization converge to population moments?

The last two questions are the reason both papers matter for econometrics.

A model that answers them affirmatively delivers a stationary, ergodic stochastic process for observable time series.

Sample moments computed from one long realization then estimate population moments, which is exactly what the rational expectations econometrics of {cite}`HansenSargent1980` requires.

{cite}`Sargent1980q` puts the Brock-Mirman apparatus to work on a question in macroeconomics: when, and in what sense, is there an investment demand schedule relating investment to Tobin's $q$?

Along the way, this lecture pauses over a technical issue that turns out to be central:

* the derivative of the planner's value function with respect to capital **is** the competitive price of used capital, i.e. it is essentially Tobin's $q$
* so the analysis stands or falls on whether that derivative exists
* irreversible investment creates a **corner**, and at a corner the standard argument for differentiability breaks down

We'll describe the difficulty, note a gap in the published argument, and give a corrected proof.

Let's start with some imports:

```{code-cell} ipython
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
```

## The Brock-Mirman planning problem

A planner chooses consumption to maximize

```{math}
:label: bm_objective
E_0 \sum_{t=0}^\infty \beta^t u(c_t), \qquad 0 < \beta < 1,
```

subject to

$$
c_t + i_t \leq f(k_t, r_t), \qquad c_t \geq 0, \quad i_t \geq 0 ,
$$

where $k_t$ is capital per worker, $i_t$ is investment, and $\{r_t\}$ is a sequence of independent and identically distributed shocks to production.

Brock and Mirman assume that for each realization of the shock, $f(\cdot, r)$ is increasing, strictly concave, and satisfies the Inada conditions

$$
f(0, r) = 0, \qquad f'(0, r) = +\infty, \qquad f'(\infty, r) = 0 ,
$$

and that $f$ is increasing in $r$, so that a higher shock means more output.

The period utility function $u$ is increasing, strictly concave, and differentiable.

In their formulation capital depreciates fully in one period, so that $k_{t+1} = i_t$ and current resources are $s_t = f(k_{t-1}, r_{t-1})$.

The Bellman equation is

```{math}
:label: bm_bellman
v(s) = \max_{0 \leq c \leq s} \left\{ u(c) + \beta \int v\bigl(f(s - c, r)\bigr) \, \nu(dr) \right\} ,
```

where $\nu$ is the distribution of the shock.

Standard arguments -- a contraction mapping, plus preservation of monotonicity and concavity -- give a unique bounded solution $v$, attained by a unique consumption policy $c = g(s)$ and investment policy $x = h(s) = s - g(s)$.

```{prf:proposition}
:label: bm_policies

The optimal policies $g$ and $h$ are increasing and continuous, with $g(0) = h(0) = 0$.
```

Brock and Mirman prove monotonicity from the first-order condition

```{math}
:label: bm_euler
u'(g(s)) = \beta \int u'\bigl(g(f(s - g(s), r))\bigr) \, f'(s - g(s), r) \, \nu(dr) ,
```

which is the stochastic analogue of the Euler equation.

If $g$ did not increase with $s$, the left side of {eq}`bm_euler` would stay constant while the right side would fall, because $f$ increases and $u'$ and $f'$ decrease.

## Long-run behavior

The optimal policy makes capital a Markov process,

```{math}
:label: bm_markov
k_{t+1} = h\bigl(f(k_t, r_t)\bigr) ,
```

and Brock and Mirman devote most of their paper to the long-run behavior of this process.

The strategy resembles the one used in {doc}`lucas_prescott_investment`.

Consider the deterministic difference equations obtained by fixing the shock at its lowest possible value $\alpha$ and at its highest possible value $\beta_r$:

$$
k_{t+1} = h(f(k_t, \alpha)), \qquad k_{t+1} = h(f(k_t, \beta_r)) .
$$

Let $k_a$ be a positive fixed point of the first and $k_b$ a positive fixed point of the second.

Because $f$ is increasing in the shock and $h$ is increasing, $k_a \leq k_b$.

Brock and Mirman say that the model has a **stable configuration of fixed points** when $k_a$ and $k_b$ are positive and $k_a < k_b$.

They then show that

* the sets of capital stocks below $k_a$ and above $k_b$ are **transient**: the process leaves them and does not return
* the interval $[k_a, k_b]$ is where the process eventually lives

```{prf:theorem}
:label: bm_theorem

There is a distribution function $F$ such that the distribution $F_t$ of $k_t$ converges to $F$ uniformly, and $F$ does not depend on the initial capital stock.
```

{prf:ref}`bm_theorem` is the counterpart of the invariant-distribution theorem of {doc}`lucas_prescott_investment`.

An important part of the proof is an argument about the **inverse optimal process**, which runs the Markov process backwards in time; Brock and Mirman use it to show that a stationary distribution exists and is unique without appealing to heavier machinery from probability theory.

Convergence of distributions is not by itself enough for econometrics.

What an econometrician needs in addition is a **mean-ergodic theorem**: a guarantee that averages computed along a single realization converge to moments of $F$,

```{math}
:label: bm_ergodic
\frac{1}{T}\sum_{t=1}^T \phi(k_t) \to \int \phi \, dF \qquad \text{with probability one} .
```

For Markov processes with a unique invariant distribution and a stable configuration of fixed points, such laws of large numbers are available; {cite}`Sargent1980q` invokes the version in Doob's book to justify computing population moments of his model and comparing them with sample moments.

This is the bridge from Brock and Mirman's theory to the rational expectations econometrics developed in the late 1970s and early 1980s.

Without {prf:ref}`bm_theorem` and {eq}`bm_ergodic`, a likelihood function computed from a single time series would have no firm justification.

```{note}
{cite}`BrockMirman1972` study the case in which the shocks are independent and identically distributed.

The lecture {doc}`lucas_prescott_investment` describes a parallel set of results for a model in which the shock is serially correlated.

In both cases the economics is the same: the state must eventually wander inside a compact set on which the process is well behaved, and the boundaries of that set are the stationary points associated with the most and least favorable shocks.
```

## Irreversible investment and Tobin's q

{cite}`Sargent1980q` modifies the model in one respect: capital, once installed, cannot be eaten.

Output can be consumed or added to the capital stock, but the conversion does not run backwards,

```{math}
:label: q_irreversible
K_{t+1} = (1 - \delta) K_t + I_t, \qquad I_t \geq 0 ,
```

where $0 < \delta < 1$ is the depreciation rate.

Production is $y_t = f(K_t)\theta_t$, and the one-period utility function $u(c_t, e_t)$ is hit by a preference shock $e_t$.

The shocks $(\theta_t, e_t)$ are independent and identically distributed over time.

Why does this small change matter?

In the model with reversible investment, the price of a unit of installed capital always equals the price of a unit of newly produced output, so Tobin's $q$ is identically one and firms have no investment demand schedule at all.

The irreversibility constraint is a **friction** that lets the price of installed capital fall below the price of new capital.

Sargent describes a competitive economy in which households own capital, rent it to firms, and trade claims to installed capital at a relative price $p_{Kt}$, which is Tobin's $q$.

Following {cite}`Lucas_Prescott_1971` -- exactly the strategy of {doc}`lucas_prescott_investment` -- he studies the competitive equilibrium indirectly, through the planning problem that generates it.

The planner solves

```{math}
:label: q_bellman
v(K, \theta, e) = \max_{K' \geq (1-\delta)K}
\left\{ u\bigl(f(K)\theta + (1-\delta)K - K', e\bigr)
+ \beta E\bigl[v(K', \theta', e')\bigr] \right\} .
```

Let $\lambda \geq 0$ be the multiplier on the irreversibility constraint.

The first-order condition is

```{math}
:label: q_foc
u_c(c, e) = \beta E\bigl[v_K(K', \theta', e')\bigr] + \lambda ,
\qquad \lambda \geq 0, \quad \lambda I = 0 ,
```

and the price of installed capital is

```{math}
:label: q_definition
q = \frac{\beta E\bigl[v_K(K', \theta', e')\bigr]}{u_c(c,e)} = 1 - \frac{\lambda}{u_c(c,e)} .
```

So this model implies

* $q \leq 1$ always
* $q = 1$ exactly when investment is positive
* $q < 1$ exactly when the irreversibility constraint binds, which is when investment is zero

Notice what {eq}`q_definition` says: **Tobin's $q$ is the derivative of the planner's value function**, normalized by marginal utility.

Everything therefore depends on whether $v_K$ exists.

## Differentiability of the value function

Here is the difficulty.

The standard tool for differentiability of a value function is the theorem of {cite}`BenvenisteScheinkman1979`.

```{prf:theorem} Benveniste-Scheinkman
:label: bs_theorem

Let $v$ be concave on an open set $D$ and let $K_0 \in D$.

Suppose $W$ is concave, differentiable at $K_0$, and satisfies $W(K) \leq v(K)$ on a neighborhood of $K_0$, with $W(K_0) = v(K_0)$.

Then $v$ is differentiable at $K_0$ and $v'(K_0) = W'(K_0)$.
```

The usual way to build such a $W$ is to take the optimal plan at $K_0$, **freeze** next period's capital at its optimal value $K_0'$, and let consumption absorb the change in $K$:

```{math}
:label: q_W1
W_1(K) = u\bigl(f(K)\theta + (1-\delta)K - K_0', e\bigr) + \beta E\bigl[v(K_0', \theta', e')\bigr] .
```

This is a feasible plan, so $W_1 \leq v$, with equality at $K_0$, and it is differentiable.

But feasibility requires $K_0' \geq (1-\delta)K$, that is $K \leq K_0'/(1-\delta)$.

When investment is positive, $K_0' > (1-\delta)K_0$ and the plan is feasible on a full neighborhood of $K_0$.

When investment is **zero**, $K_0' = (1-\delta)K_0$ exactly, and the plan is infeasible for every $K > K_0$.

At a corner, the standard support function is available only from the left.

This is precisely where the published argument becomes delicate, and in our reading it has gaps.

```{note}
Two specific difficulties.

First, {cite}`Sargent1980q` states the derivative of the value function in the form

$$
v_K(K,\theta,e) = u_c(c,e)\bigl[f'(K)\theta + (1-\delta)\bigr] ,
$$

which is obtained by substituting the first-order condition {eq}`q_foc` **with equality** into the envelope condition.

That substitution is legitimate only where investment is positive.

Where investment is zero, $\lambda > 0$ and the formula overstates $v_K$; the paper's own subsequent inequality records this, so the two statements are not consistent with each other.

Second, the argument establishes differentiability of the value function by induction along the iterates $v^j = T^j v^0$, and then passes to the limit.

That route requires knowing how the set of capital stocks at which the constraint just binds is structured.

The published proof handles three configurations, drawn in its Figures 3-5, and appeals to an appendix that assumes the iterates are twice differentiable almost everywhere.

The appendix in turn justifies this by differentiating the first-order condition and observing that "since the right-hand side exists almost everywhere, so does the left," which does not follow as stated, and it asserts without proof that the set of binding points has Lebesgue measure zero.
```

Fortunately the conclusion is true, and there is a route to it that avoids the induction on iterates entirely.

The key observation is that at a corner a **different** feasible plan supplies the needed support function: investing nothing is feasible at *every* capital stock, so it works on both sides.

````{prf:proposition}
:label: q_differentiability

Assume $u$ and $f$ are increasing, strictly concave and continuously differentiable, with $u_c(0,e) = \infty$ and $f'(0) = \infty$.

Then for each $(\theta, e)$ the value function $v(\cdot,\theta,e)$ defined by {eq}`q_bellman` is continuously differentiable on $(0,\infty)$, and

```{math}
:label: q_envelope
v_K(K,\theta,e) = u_c(c,e) f'(K)\theta + \beta(1-\delta) E\bigl[v_K(K',\theta',e')\bigr] ,
```

where $c$ and $K'$ are optimal at $(K,\theta,e)$.

Where investment is positive, {eq}`q_envelope` reduces to $v_K = u_c(c,e)[f'(K)\theta + (1-\delta)]$.

Where investment is zero, $v_K < u_c(c,e)[f'(K)\theta + (1-\delta)]$.
````

````{prf:proof}
**Step 0.** The operator associated with {eq}`q_bellman` maps bounded continuous concave functions into bounded continuous strictly concave functions, so $v(\cdot,\theta,e)$ is concave and continuous, and the optimal policy is single valued and continuous.

**Step 1 (investment positive).** Suppose $I(K_0,\theta,e) > 0$, and let $K_0'$ be the optimal choice.

Then $K_0' > (1-\delta)K_0$, so by continuity the plan that freezes next period's capital at $K_0'$ is feasible for all $K$ in a neighborhood of $K_0$.

The function $W_1$ of {eq}`q_W1` is therefore concave, continuously differentiable, dominated by $v$, and equal to $v$ at $K_0$.

{prf:ref}`bs_theorem` gives differentiability at $K_0$ with

$$
v_K(K_0,\theta,e) = u_c(c,e)\bigl[f'(K_0)\theta + (1-\delta)\bigr] .
$$

Because the first-order condition holds with equality here, $\beta E[v_K(K_0',\cdot)] = u_c(c,e)$, and substituting gives {eq}`q_envelope`.

**Step 2 (investment zero, conditional on a smaller capital stock).** Suppose $I(K_0,\theta,e) = 0$, so that $K_0' = (1-\delta)K_0$ and investing nothing is optimal at $K_0$.

Consider instead the plan that invests nothing this period and behaves optimally thereafter:

$$
W_2(K) = u\bigl(f(K)\theta, e\bigr) + \beta E\bigl[v\bigl((1-\delta)K, \theta', e'\bigr)\bigr] .
$$

Investing nothing is feasible at **every** capital stock, so $W_2 \leq v$ everywhere, with $W_2(K_0) = v(K_0)$, and $W_2$ is concave.

Suppose that $v(\cdot,\theta',e')$ is differentiable at $(1-\delta)K_0$ for every $(\theta',e')$.

Then $W_2$ is differentiable at $K_0$; differentiation under the expectation is legitimate because the functions $v(\cdot,\theta',e')$ are concave, hence locally Lipschitz with a common bound on a compact neighborhood.

{prf:ref}`bs_theorem` then gives differentiability of $v$ at $K_0$, with

$$
v_K(K_0,\theta,e) = u_c(c,e) f'(K_0)\theta + \beta(1-\delta) E\bigl[v_K((1-\delta)K_0, \theta', e')\bigr] ,
$$

which is {eq}`q_envelope`.

**Step 3 (a region where the corner cannot bind).** Because $f'(0) = \infty$, there is an $\eta > 0$ such that $I(K,\theta,e) > 0$ for every $K \leq \eta$ and every $(\theta,e)$.

The intuition is that the marginal product of capital becomes arbitrarily large as capital approaches zero, so that giving up a little consumption today buys a great deal of consumption tomorrow.

Making this precise requires comparing the rates at which the two sides of {eq}`q_foc` diverge as $K \downarrow 0$, since with, say, logarithmic utility both sides become unbounded.

{cite}`Sargent1980q` proves the statement in his Appendix A (his Proposition A2), under an explicit restriction on $u$ and $f$ that he imposes for this purpose.

That result is the one ingredient of his appendix that the argument below borrows, and it is independent of the differentiability question at issue here.

By Step 1, $v(\cdot,\theta,e)$ is therefore differentiable on $(0,\eta]$ for every $(\theta,e)$.

**Step 4 (bootstrap).** Let

$$
D = \{K > 0 : v(\cdot,\theta,e) \text{ is differentiable at } K \text{ for every } (\theta,e)\} .
$$

We claim that if $(0, A) \subseteq D$ then $(0, A/(1-\delta)) \subseteq D$.

Take $K < A/(1-\delta)$ and any $(\theta,e)$.

If $I(K,\theta,e) > 0$, Step 1 applies directly.

If $I(K,\theta,e) = 0$, then $(1-\delta)K < A$, so $v(\cdot,\theta',e')$ is differentiable at $(1-\delta)K$ for every $(\theta',e')$, and Step 2 applies.

By Step 3 we may start the induction at $A = \eta$.

Since $1/(1-\delta) > 1$, iterating the claim gives $(0, \eta (1-\delta)^{-n}) \subseteq D$ for every $n$, and therefore $D = (0,\infty)$.

**Step 5 (continuity).** A concave function that is differentiable on an open interval has a continuous derivative there, so $v(\cdot,\theta,e)$ is continuously differentiable.

Finally, where investment is zero the first-order condition holds with $\lambda > 0$, so $\beta E[v_K(K',\cdot)] < u_c$, and {eq}`q_envelope` gives $v_K < u_c[f'(K)\theta + (1-\delta)]$.
````

The corrected formula has a natural reading.

Iterating {eq}`q_envelope` forward gives

```{math}
:label: q_envelope_sum
v_K(K_t,\theta_t,e_t) = E_t \sum_{j=0}^\infty \beta^j (1-\delta)^j \,
u_c(c_{t+j}, e_{t+j}) \, f'(K_{t+j})\theta_{t+j} .
```

A marginal unit of capital installed today yields marginal products for as long as it survives, and each is valued at the marginal utility of consumption on the date it arrives.

When investment is positive, the market prices that whole stream at the cost of a unit of new output, and {eq}`q_envelope_sum` collapses to $u_c[f'(K)\theta + (1-\delta)]$.

When investment is zero, it does not, and the difference is exactly what pushes $q$ below one.

## Computing the model

We now solve the model of {eq}`q_bellman` with

$$
u(c,e) = e \ln c, \qquad f(K) = K^a .
$$

The shocks $\theta$ and $e$ each take two values with equal probability and are independent of each other and over time.

One numerical detail matters a great deal.

The irreversibility constraint says $K' \geq (1-\delta)K$, so we want $(1-\delta)K$ to be a point of the capital grid whenever $K$ is.

Following {cite}`Sargent1980q`, we use a **geometric** grid with ratio $(1-\delta)^{1/z}$ for an integer $z$, so that $(1-\delta)K_i = K_{i-z}$ exactly.

```{code-cell} ipython
Model = namedtuple("Model", "β δ a z K θ e W nθ ne")

def create_model(β=0.95, δ=0.05, a=0.25, z=10, n_K=500, K_hi=20.0,
                 θ_vals=(0.85, 1.15), e_vals=(0.6, 1.4)):
    "Geometric capital grid so that (1-δ)K is itself a grid point."
    ratio = (1 - δ)**(1/z)
    K = K_hi * ratio**np.arange(n_K - 1, -1, -1)
    θ, e = np.array(θ_vals), np.array(e_vals)
    W = np.outer(np.ones(len(θ))/len(θ), np.ones(len(e))/len(e))
    return Model(β, δ, a, z, K, θ, e, W, len(θ), len(e))

f  = lambda m, K: K**m.a
f_prime = lambda m, K: m.a * K**(m.a - 1)
```

The planner chooses next period's capital from the grid, subject to positive consumption and to irreversibility.

```{code-cell} ipython
def solve_model(m, irreversible=True, tol=1e-10, maxit=5000, howard=50):
    "Value function iteration with Howard policy improvement steps."
    nK = len(m.K)
    C = f(m, m.K)[:, None] * m.θ[None, :]                       # output (K, θ)
    C = C[:, None, :] + (1 - m.δ)*m.K[:, None, None] - m.K[None, :, None]
    ok = C > 1e-12                                              # (K, K', θ)

    if irreversible:                                            # K' ≥ (1-δ)K
        irr = np.zeros((nK, nK), bool)
        for i in range(nK):
            irr[i, max(i - m.z, 0):] = True
        ok = ok & irr[:, :, None]

    U = np.where(ok[:, :, :, None],
                 m.e[None, None, None, :] * np.log(np.where(ok, C, 1.0))[:, :, :, None],
                 -1e12)

    v = np.zeros((nK, m.nθ, m.ne))
    for it in range(maxit):
        EV = np.tensordot(v, m.W, axes=([1, 2], [0, 1]))        # E v(K') over (θ', e')
        obj = U + m.β * EV[None, :, None, None]
        idx = obj.argmax(axis=1)
        v_new = np.take_along_axis(obj, idx[:, None, :, :], axis=1)[:, 0, :, :]

        for _ in range(howard):
            EV = np.tensordot(v_new, m.W, axes=([1, 2], [0, 1]))
            v_new = (np.take_along_axis(U, idx[:, None, :, :], axis=1)[:, 0, :, :]
                     + m.β * EV[idx])

        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new

    K_next = m.K[idx]
    I = K_next - (1 - m.δ) * m.K[:, None, None]
    C_pol = (f(m, m.K)[:, None, None] * m.θ[None, :, None]
             + (1 - m.δ) * m.K[:, None, None] - K_next)
    return v, idx, K_next, I, C_pol
```

To compute $q$ we need $v_K$, and {prf:ref}`q_differentiability` tells us how to get it: equation {eq}`q_envelope` is a linear fixed point problem in $v_K$ given the optimal policy, and it is a contraction with modulus $\beta(1-\delta)$.

```{code-cell} ipython
def marginal_value(m, idx, C_pol, tol=1e-13, maxit=50_000):
    "Solve the envelope equation v_K = u_c f'(K)θ + β(1-δ) E v_K(K')."
    vK = np.zeros_like(C_pol)
    direct = (m.e[None, None, :]/C_pol) * f_prime(m, m.K)[:, None, None] * m.θ[None, :, None]
    for _ in range(maxit):
        EvK = np.tensordot(vK, m.W, axes=([1, 2], [0, 1]))
        new = direct + m.β * (1 - m.δ) * EvK[idx]
        if np.max(np.abs(new - vK)) < tol:
            return new
        vK = new
    return vK

m = create_model()
v, idx, K_next, I, C_pol = solve_model(m)
vK = marginal_value(m, idx, C_pol)

u_c = m.e[None, None, :] / C_pol
E_vK = np.tensordot(vK, m.W, axes=([1, 2], [0, 1]))
q = m.β * E_vK[idx] / u_c
corner = I <= 1e-12

print(f"share of states with zero investment: {corner.mean():.3f}")
```

Let's check {prf:ref}`q_differentiability` numerically, by comparing the envelope formula with a finite-difference derivative of the computed value function.

```{code-cell} ipython
vK_fd = np.gradient(v, m.K, axis=0)
naive = u_c * (f_prime(m, m.K)[:, None, None] * m.θ[None, :, None] + (1 - m.δ))
mid = ((m.K > 2.2) & (m.K < 7.0))[:, None, None] & np.ones_like(corner)

rel = lambda x: np.median((np.abs(x - vK_fd)/np.abs(vK_fd))[mid & sel])

sel = ~corner
print(f"investment positive: envelope error {rel(vK):.4f}, "
      f"naive formula error {rel(naive):.4f}")
sel = corner
print(f"investment zero:     envelope error {rel(vK):.4f}, "
      f"naive formula error {rel(naive):.4f}")
```

Where investment is positive, both formulas agree with the numerical derivative.

Where investment is zero, the envelope formula {eq}`q_envelope` remains accurate while the formula $u_c[f'(K)\theta + (1-\delta)]$ is off by tens of percent.

This is the practical content of the correction: at corners the two expressions are genuinely different objects, and it is the envelope formula that gives the price of installed capital.

We can also verify Step 3 of the proof, which asserts a region of small capital stocks where the corner never binds.

```{code-cell} ipython
m_wide = create_model(n_K=700)        # a grid reaching lower capital stocks
_, idx_w, _, I_w, _ = solve_model(m_wide)
frac_corner = (I_w <= 1e-12).reshape(len(m_wide.K), -1).mean(axis=1)
first = np.argmax(frac_corner > 0)
print(f"grid runs from K = {m_wide.K.min():.3f}")
print(f"investment is positive for every shock when K < {m_wide.K[first]:.3f}")
```

### Policies and the price of installed capital

```{code-cell} ipython
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
show = (m.K > 1.5) & (m.K < 9)

for j in range(m.nθ):
    for l in range(m.ne):
        lab = rf'$\theta={m.θ[j]}, e={m.e[l]}$'
        axes[0].plot(m.K[show], K_next[show, j, l], label=lab)
        axes[1].plot(m.K[show], q[show, j, l], label=lab)

axes[0].plot(m.K[show], m.K[show], 'k--', lw=1, label='45 degree line')
axes[0].plot(m.K[show], (1-m.δ)*m.K[show], 'k:', lw=1, label=r'$(1-\delta)K$')
axes[0].set_xlabel('$K$'); axes[0].set_ylabel("$K'$"); axes[0].set_title('law of motion')
axes[1].axhline(1.0, color='k', ls='--', lw=1)
axes[1].set_xlabel('$K$'); axes[1].set_ylabel('$q$'); axes[1].set_title("Tobin's $q$")
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

The law of motion runs along the lower dotted line $(1-\delta)K$ whenever the irreversibility constraint binds.

Exactly on that region, $q$ falls below one.

### The invariant distribution

As in {doc}`lucas_prescott_investment`, we check that the long-run distribution does not depend on where the economy starts.

```{code-cell} ipython
def simulate(m, idx, K0, T=50_000, seed=0, burn=500):
    "Simulate the equilibrium Markov process."
    rng = np.random.default_rng(seed)
    j_idx = rng.integers(0, m.nθ, T)
    l_idx = rng.integers(0, m.ne, T)
    k = np.abs(m.K - K0).argmin()
    path = np.empty(T, int)
    for t in range(T):
        path[t] = k
        k = idx[k, j_idx[t], l_idx[t]]
    return path[burn:], j_idx[burn:], l_idx[burn:]

p_lo, j_lo, l_lo = simulate(m, idx, K0=2.0, seed=7)
p_hi, j_hi, l_hi = simulate(m, idx, K0=15.0, seed=7)

print(f"mean capital starting from K0 = 2:  {m.K[p_lo].mean():.4f}")
print(f"mean capital starting from K0 = 15: {m.K[p_hi].mean():.4f}")
print(f"capital visited: [{m.K[p_lo].min():.2f}, {m.K[p_lo].max():.2f}]")
```

```{code-cell} ipython
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].hist(m.K[p_lo], bins=60, density=True, alpha=0.5, label='from $K_0 = 2$')
axes[0].hist(m.K[p_hi], bins=60, density=True, alpha=0.5, label='from $K_0 = 15$')
axes[0].set_xlabel('$K$'); axes[0].set_ylabel('density')
axes[0].set_title('invariant distribution of capital')
axes[0].legend()

q_sim, I_sim = q[p_lo, j_lo, l_lo], I[p_lo, j_lo, l_lo]
axes[1].scatter(q_sim, I_sim, s=4, alpha=0.2)
axes[1].set_xlabel('$q$'); axes[1].set_ylabel('$I$')
axes[1].set_title("investment and Tobin's $q$")

plt.tight_layout()
plt.show()
```

The two histograms coincide, as {prf:ref}`bm_theorem` leads us to expect.

The scatter plot on the right reproduces the central picture of {cite}`Sargent1980q`.

Investment is positive only when $q$ equals one; when $q$ is below one, investment is exactly zero.

```{code-cell} ipython
print(f"q when I > 0: [{q_sim[I_sim > 1e-12].min():.4f}, {q_sim[I_sim > 1e-12].max():.4f}]")
print(f"q when I = 0: [{q_sim[I_sim <= 1e-12].min():.4f}, {q_sim[I_sim <= 1e-12].max():.4f}]")
```

The small departures from exactly one reflect the discreteness of the capital grid.

Sargent's point is now visible.

There *is* a positive relationship between investment and $q$ in these data, and an econometrician could certainly run that regression.

But the relationship is not an investment demand schedule.

It is a **mongrel relation** that mixes together preferences, technology and the distribution of the shocks, and it will shift whenever any of them changes.

{ref}`ogu_ex3` asks you to demonstrate this.

## Exercises

```{exercise}
:label: ogu_ex1

{cite}`BrockMirman1972` is famous partly for a special case that can be solved by hand.

Let $u(c) = \ln c$, let $f(k, r) = r k^\alpha$ with $0 < \alpha < 1$, and let capital depreciate fully each period, so that $k_{t+1} = f(k_t, r_t) - c_t$.

1. Verify that the optimal policy is $k_{t+1} = \alpha\beta\, r_t k_t^\alpha$.
1. Show that $\ln k_t$ follows an AR(1) process, and derive the mean and variance of its invariant distribution when $\ln r_t \sim N(\mu, \sigma^2)$.
1. Simulate the model and check both the invariant distribution and the mean-ergodic property {eq}`bm_ergodic`.
```

```{solution-start} ogu_ex1
:class: dropdown
```

Guess that a constant fraction of resources is saved, $k_{t+1} = s\, r_t k_t^\alpha$.

The Euler equation for this problem is

$$
\frac{1}{c_t} = \beta E_t \left[ \frac{1}{c_{t+1}} \alpha r_{t+1} k_{t+1}^{\alpha - 1} \right] .
$$

With $c_t = (1-s) r_t k_t^\alpha$ and $k_{t+1} = s r_t k_t^\alpha$, the right side becomes

$$
\beta E_t \left[\frac{\alpha r_{t+1}k_{t+1}^{\alpha-1}}{(1-s) r_{t+1} k_{t+1}^{\alpha}}\right]
= \frac{\beta \alpha}{(1-s) k_{t+1}}
= \frac{\beta\alpha}{(1-s) s r_t k_t^{\alpha}} ,
$$

while the left side is $1/[(1-s) r_t k_t^\alpha]$.

Equating the two gives $s = \alpha\beta$.

Taking logs of the policy,

$$
\ln k_{t+1} = \ln(\alpha\beta) + \alpha \ln k_t + \ln r_t ,
$$

an AR(1) with coefficient $\alpha$.

With $\ln r_t \sim N(\mu, \sigma^2)$, the invariant distribution of $\ln k$ is normal with

$$
\text{mean} = \frac{\ln(\alpha\beta) + \mu}{1 - \alpha},
\qquad
\text{variance} = \frac{\sigma^2}{1 - \alpha^2} .
$$

```{code-cell} ipython
α, β_d, μ, σ = 0.4, 0.95, 0.0, 0.1
T = 200_000
rng = np.random.default_rng(0)
ln_r = μ + σ * rng.normal(size=T)

k = np.empty(T + 1)
k[0] = 1.0
for t in range(T):
    k[t+1] = α * β_d * np.exp(ln_r[t]) * k[t]**α

ln_k = np.log(k[1000:])
print(f"mean of ln k: simulated {ln_k.mean():.4f}, "
      f"theory {(np.log(α*β_d) + μ)/(1-α):.4f}")
print(f"var  of ln k: simulated {ln_k.var():.4f}, "
      f"theory {σ**2/(1-α**2):.4f}")
```

The time averages match the population moments of the invariant distribution, which is the mean-ergodic property {eq}`bm_ergodic` in action.

```{solution-end}
```

```{exercise}
:label: ogu_ex2

This exercise examines the difference between the two candidate formulas for $v_K$ discussed above.

For the baseline model, compute at every state

* the envelope formula {eq}`q_envelope`
* the formula $u_c(c,e)[f'(K)\theta + (1-\delta)]$
* a finite-difference derivative of the value function

Then

1. Confirm that all three agree where investment is positive.
1. Confirm that the second disagrees with the other two where investment is zero, and report by how much.
1. Explain, using {eq}`q_foc`, why the second formula is an upper bound for $v_K$.
1. What would go wrong with the computed $q$ if you used the second formula everywhere?
```

```{solution-start} ogu_ex2
:class: dropdown
```

The second formula is what you get by substituting the first-order condition {eq}`q_foc` into the envelope condition **as though** $\lambda = 0$.

Since $\lambda \geq 0$, dropping it can only raise the expression, so

$$
v_K = u_c f'(K)\theta + (1-\delta)\bigl[u_c - \lambda\bigr]
\leq u_c\bigl[f'(K)\theta + (1-\delta)\bigr] ,
$$

with equality exactly when $\lambda = 0$, that is, when investment is positive.

```{code-cell} ipython
gap = np.abs(naive - vK_fd)/np.abs(vK_fd)
env = np.abs(vK - vK_fd)/np.abs(vK_fd)

for name, sel in (("investment positive", ~corner), ("investment zero", corner)):
    s = mid & sel
    print(f"{name:20}: envelope {np.median(env[s]):.4f}   "
          f"naive {np.median(gap[s]):.4f}   max naive {gap[s].max():.4f}")

q_naive = m.β * np.tensordot(naive, m.W, axes=([1, 2], [0, 1]))[idx] / u_c
print(f"\nusing the naive formula, max q = {q_naive.max():.3f} "
      f"(theory says q ≤ 1)")
print(f"fraction of states with q > 1: {(q_naive > 1 + 1e-8).mean():.3f}")
```

Using the second formula everywhere produces values of $q$ that exceed one, which the theory rules out.

The reason is economic, not numerical: at a corner the market does not price installed capital at the cost of new capital, precisely because the household would like to sell capital and cannot.

```{solution-end}
```

```{exercise}
:label: ogu_ex3

Show that the regression of investment on $q$ is not structural.

Solve and simulate the model for several economies that differ in

* the amount of preference risk (the spread of $e$)
* the amount of technology risk (the spread of $\theta$)
* the depreciation rate $\delta$

For each economy, report the slope and $R^2$ of a regression of $I$ on $q$, together with the frequency with which investment is zero.

Comment on what this implies for an econometrician who estimates an "investment demand schedule" relating investment to $q$.
```

```{solution-start} ogu_ex3
:class: dropdown
```

Here is one solution.

```{code-cell} ipython
def q_regression(**kwargs):
    "Solve an economy, simulate it, and regress I on q."
    mm = create_model(**kwargs)
    vv, ii, _, II, CC = solve_model(mm)
    vvK = marginal_value(mm, ii, CC)
    uu_c = mm.e[None, None, :]/CC
    qq = mm.β * np.tensordot(vvK, mm.W, axes=([1, 2], [0, 1]))[ii] / uu_c
    p, j, l = simulate(mm, ii, K0=3.5)
    q_s, I_s = qq[p, j, l], II[p, j, l]
    slope = np.polyfit(q_s, I_s, 1)[0]
    corr = np.corrcoef(I_s, q_s)[0, 1]
    return slope, corr**2, (I_s <= 1e-12).mean()

cases = [("baseline",              {}),
         ("less preference risk",  dict(e_vals=(0.8, 1.2))),
         ("more preference risk",  dict(e_vals=(0.4, 1.6))),
         ("less technology risk",  dict(θ_vals=(0.95, 1.05))),
         ("more technology risk",  dict(θ_vals=(0.7, 1.3))),
         ("faster depreciation",   dict(δ=0.10))]

print(f"{'economy':24}{'slope':>9}{'R^2':>8}{'zero I':>9}")
for name, kw in cases:
    slope, r2, zero = q_regression(**kw)
    print(f"{name:24}{slope:9.3f}{r2:8.3f}{zero:9.3f}")
```

The slope and the fit both move around substantially across economies that share the same preferences and technology parameters but differ in the distribution of shocks or in the depreciation rate.

Nothing in the regression is invariant, so it cannot be used to predict what investment would do under a policy that changed any of these features.

This is the message of {cite}`Sargent1980q`, and it is a concrete instance of the Lucas critique: the very friction that makes $q$ interesting -- the occasionally binding irreversibility constraint -- also makes the relationship between $I$ and $q$ a reduced-form artifact rather than a decision rule.

```{solution-end}
```
