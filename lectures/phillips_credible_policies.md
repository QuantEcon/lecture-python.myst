---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(phillips_credible_policies)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Credible Government Policies

```{index} single: Phillips Curve; Credible Policies
```

```{contents} Contents
:depth: 2
```

## Overview

{doc}`phillips_credibility` left the government in a trap.

Without a technology for committing itself, a government that re-optimizes every period ends
at the Nash outcome, inflating at rate $\theta U^*$ and getting nothing for it.

That was a one-period story, and it invited an obvious objection: a government that expects to
face the same public again tomorrow has a *reputation* to protect.

This lecture, following chapter 4 of {cite:t}`Sargent1999`, takes the objection seriously.

We repeat the Kydland–Prescott economy forever, let the government's current action depend on
the whole history of outcomes, and ask which outcome paths can be supported as **subgame
perfect equilibria**.

The answer is not the one the objection anticipated.

Reputation does not single out the Ramsey outcome, and it does not confirm the Nash outcome either.

It delivers a *continuum* of equilibrium values, some better than Nash and some considerably
worse, with no principle inside the model to select among them.

That is the conclusion this lecture exists to establish, and Sargent states it in one sentence:

> My conclusion is that the multiplicity of credible plans replaces pessimism with agnosticism.

It matters for the argument of the whole suite.

{doc}`phillips_two_stories` rests part of its case against the *triumph of natural-rate theory*
on exactly this weakness: a theory with so many equilibria makes few predictions, so a story
that depends on policy makers having learned the right one is doing work the theory cannot do
for it.

### What we build

The machinery is the recursive method of Abreu, Pearce, and Stacchetti {cite}`APS1990`, which
computes not an optimal *value* but the whole *set* of equilibrium values.

Ordinary dynamic programming iterates an operator that maps continuation *values* into values.

APS iterate an operator that maps *sets* of continuation values into sets of values, and the
set of subgame perfect equilibrium values is its largest fixed point.

```{note}
Making a promised value into a state variable, and then doing dynamic programming in that
state, is the technique sometimes called **dynamic programming squared**.

It is the organizing idea of the QuantEcon lecture on
{doc}`Stackelberg plans <advanced:dyn_stack>` and of the recursive treatments of Ramsey problems
in {cite:t}`Ljungqvist2012`.

This lecture is a compact instance: the state is a promised value, and the object we compute is
a set.

A *scalar* promised value suffices here because the private sector's problem is static: the
public's forecast $x_t$ concerns only this period's inflation, so it is pinned down by the
government's current prescription.

In the monetary model of {cite:t}`chang1998credible`, households' money demand looks forward,
so the government must also promise them a continuation marginal value of money, and the state
becomes a pair; {doc}`advanced:chang_ramsey` and {doc}`advanced:chang_credible` compute the
resulting two-dimensional sets with the same set-iteration idea.

{doc}`advanced:calvo_abreu` builds Abreu's stick-and-carrot plans, which appear below, in a
linear-quadratic version of Calvo's model.
```

We then use the machinery three ways.

We construct particular equilibria by guess-and-verify — infinite repetition of Nash, infinite
repetition of something better, and Abreu's stick-and-carrot, which can be *worse* than Nash.

We compute the best and worst equilibrium values from two small programming problems, and check
them against a direct iteration of the APS operator.

And we exhibit three quite different equilibria that all attain the same worst value, which is
the sharpest illustration of how little the equilibrium concept pins down.

```{note}
Sargent's own advice to the reader is worth passing on: this chapter is difficult if you have
not seen the theory before, and a reader willing to accept its verdict — that the theory of
credible policy yields agnosticism rather than a prediction — can go straight to
{doc}`phillips_adaptive` without losing the thread of the argument.
```

Let's start with our imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import NamedTuple
```

## The repeated economy

The one-period economy is the one from {doc}`phillips_credibility`.

Let $(U, y, x)$ be unemployment, inflation, and the public's expectation of inflation.

Unemployment obeys the expectations-augmented Phillips curve $U = U^* - \theta(y - x)$, and the
government's one-period payoff, written as a function of $(x, y)$, is

```{math}
:label: cp_r

r(x, y) = -\frac{1}{2}\left[\bigl(U^* - \theta(y - x)\bigr)^2 + y^2\right].
```

The government's one-period best response to an expectation $x$ is

```{math}
:label: cp_B

B(x) = \frac{\theta\,(U^* + \theta x)}{\theta^2 + 1},
```

the Nash outcome is $y^N = \theta U^*$, and the Ramsey outcome is $y^R = 0$.

Two things are new.

First, the economy repeats for $t = 1, 2, \ldots$, and the government ranks outcome paths
$(x, y) = \{x_t, y_t\}_{t=1}^\infty$ by

```{math}
:label: cp_value

V^g(x, y) = (1 - \delta)\sum_{t=1}^{\infty}\delta^{t-1} r(x_t, y_t),
\qquad \delta \in (0, 1).
```

The factor $(1-\delta)$ puts $V^g$ in the same units as a one-period payoff, so values and
one-period returns can be compared directly.

Second, inflation is confined to a bounded interval $Y = [0, y^\#]$.

The lower bound is the Ramsey outcome; the upper bound $y^\#$ is what makes the government's
problem non-trivial, as we shall see when we look for the *worst* equilibrium.

We use the parameter values of the numerical example in chapter 4 of {cite:t}`Sargent1999`:
$\theta = 1.25$, $U^* = 5.5$, $y^\# = 10$, and $\delta = 0.95$.

They differ from the $\theta = 1$, $U^* = 5$ of {doc}`phillips_credibility`, so that our numbers
can be checked against the book's.

```{code-cell} ipython3
class Model(NamedTuple):
    θ: float = 1.25          # slope of the Phillips curve
    U_star: float = 5.5      # natural rate of unemployment
    y_max: float = 10.0      # y^#, the highest admissible inflation rate
    δ: float = 0.95          # discount factor


def r(m, x, y):
    "One-period government payoff when the public expects x and inflation is y."
    U = m.U_star - m.θ * (y - x)
    return -0.5 * (U**2 + y**2)


def B(m, x):
    "The government's one-period best response to expected inflation x."
    return m.θ * (m.U_star + m.θ * x) / (m.θ**2 + 1)


def y_nash(m):
    return m.θ * m.U_star
```

Two payoff schedules do all the work below, both read as functions of a single inflation rate
$y$ that the public has come to expect.

The first is the **rational-expectations payoff** $r(y, y)$: what the government gets if it
delivers the inflation the public expects.

The second is the **deviation payoff** $r(y, B(y))$: what it gets if the public expects $y$ but
the government yields to temptation and best-responds.

Both have closed forms worth recording, because they explain everything that follows:

```{math}
:label: cp_schedules

r(y, y) = -\frac{1}{2}\left(U^{*2} + y^2\right),
\qquad
r\bigl(y, B(y)\bigr) = -\frac{\left(U^* + \theta y\right)^2}{2\left(1 + \theta^2\right)} .
```

```{code-cell} ipython3
def r_keep(m, y):
    "Payoff from delivering the expected inflation rate: r(y, y)."
    return -0.5 * (m.U_star**2 + y**2)


def r_cheat(m, y):
    "Payoff from best-responding to an expectation of y: r(y, B(y))."
    return -(m.U_star + m.θ * y)**2 / (2 * (1 + m.θ**2))


m = Model()
ys = np.linspace(0, m.y_max, 7)
print("closed forms agree with the definitions:",
      np.allclose(r_keep(m, ys), r(m, ys, ys)),
      np.allclose(r_cheat(m, ys), r(m, ys, B(m, ys))))
```

The gap between them is a perfect square,

```{math}
:label: cp_gap

r\bigl(y, B(y)\bigr) - r(y, y) = \frac{\left(y - y^N\right)^2}{2\left(1 + \theta^2\right)} \;\geq\; 0 .
```

So the deviation schedule lies above the other everywhere except at the Nash rate, where the two
curves are *tangent*: there the temptation to deviate vanishes because $B(y^N) = y^N$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "The two payoff schedules: delivering expected inflation against deviating from it"
    name: fig-cpol-schedules
---
grid = np.linspace(0, m.y_max, 400)
yN = y_nash(m)
v_R, v_N = r_keep(m, 0.0), r_keep(m, yN)
v_cand = r_cheat(m, m.y_max)                 # candidate worst value

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(grid, r_keep(m, grid), 'C0', lw=1.6, label='$r(y, y)$: deliver what is expected')
ax.plot(grid, r_cheat(m, grid), 'C1', lw=1.6, label='$r(y, B(y))$: deviate')
ax.plot(0.0, v_R, 'ko', ms=6)
ax.annotate('$v^R$', (0.0, v_R), textcoords='offset points', xytext=(8, 4))
ax.plot(yN, v_N, 'ko', ms=6)
ax.annotate('$v^N$', (yN, v_N), textcoords='offset points', xytext=(8, 4))
ax.plot(m.y_max, v_cand, 'ko', ms=6)
ax.annotate(r'$r(y^\#, B(y^\#))$', (m.y_max, v_cand), textcoords='offset points',
            xytext=(-70, 6))
ax.axvline(yN, color='k', lw=0.6, ls=':')
ax.set_xlabel('inflation rate $y$ expected by the public')
ax.set_ylabel('one-period payoff')
ax.legend(loc='lower left')
plt.show()
```

The vertical gap between the curves is the government's one-period gain from deviating.

Below $y^N$, which includes the Ramsey outcome, $B(y) > y$, so the temptation is to inflate
*more* than expected.

Above $y^N$, $B(y) < y$, and the temptation is to inflate *less* than expected.

By {eq}`cp_gap` the gap is symmetric about $y^N$, so the gap alone does not favor the top of $Y$.

What will put the *worst* equilibrium at the top of $Y$ is the level of the deviation schedule.

$r(y, B(y))$ falls monotonically in $y$, so the least a government can secure for itself by
deviating is smallest when the public expects $y^\#$.

The point $r(y^\#, B(y^\#))$ marked in the figure is our first candidate for the worst
equilibrium value.

## Recursive strategies and promised values

A government strategy must be able to condition today's action on the entire past.

Carrying whole histories around is unmanageable, so we follow {cite:t}`Sargent1999` in restricting
attention to strategies with a recursive representation — a restriction that costs nothing,
since it excludes no equilibrium *values*.

```{prf:definition} Recursive government strategy
:label: cp_strategy

A recursive government strategy is a pair of functions $\sigma = (\sigma_1, \sigma_2)$ together
with an initial condition $v_1$, with the structure

$$
v_1 \in \mathbb{R} \text{ given}, \qquad
y_t = \sigma_1(v_t), \qquad
v_{t+1} = \sigma_2(v_t, x_t, y_t),
$$

where $v_t$ is a state variable that summarizes the history of outcomes before $t$.
```

Each member of the private sector knows $v_t$ and knows $\sigma$, and so forecasts

```{math}
:label: cp_expect

x_t = \sigma_1(v_t) .
```

Equation {eq}`cp_expect` builds rational expectations into the private sector: the public is
never surprised along the equilibrium path.

A strategy $(\sigma, v_1)$ generates an entire outcome path, and hence a value
$V^g(\sigma, v_1)$ through {eq}`cp_value`.

The theory of credible policy now does something that at first looks like a trick.

It ties the past to the future by requiring the state variable to *be* the value it generates,

```{math}
:label: cp_fixedpt

v = V^g(\sigma, v).
```

So $v$ works two shifts at once.

In $v_{t+1} = \sigma_2(v_t, x_t, y_t)$ it is a bookkeeping device that records what has
happened.

In {eq}`cp_fixedpt` it is a **promised value** — the discounted future the government is owed
on entering the period.

Reading it the second way is what makes the machinery go: the strategy hands the government
current and future outcomes that make it *want* to do what is expected of it.

```{note}
Two readings of $\sigma$ are available and, within an equilibrium, impossible to tell apart.

It can be a decision rule that the government chooses, or a description of a system of public
expectations to which the government conforms.

We return to this ambiguity in the interpretation section, because it is where the theory's
agnosticism comes from.
```

### Historical antecedents

Formalizing credibility was an achievement of the 1980s, but a sophisticated grasp of the idea
is much older.

In 1784 the finance minister Jacques Necker explained to Louis XVI why an absolute monarch, free
to renege on debt payments, found it hard to borrow {cite:p}`SargentVelde1995`:

> Therefore one can rekindle or sustain public trust only by giving reassurances on the
> sovereign's intentions, and by proving that no motive can incite him to fail his obligations.

Every clause of that sentence is in the modern definition.

A king proves he will never fail his obligations by never *wanting* to fail them — by
conforming to a system of expectations that carries its own incentive constraints, so that at
every date and in every contingency his current payoff plus continuation value is higher if he
confirms expectations than if he disappoints them.

```{prf:definition} Subgame perfect equilibrium
:label: cp_spe

A recursive strategy with promised value $v$ is a **subgame perfect equilibrium** (SPE) if and
only if

(a) $\sigma_2(v, \sigma_1(v), \eta)$ is itself a value attainable by a subgame perfect
equilibrium, for every $\eta \in Y$; and

(b) writing $y = \sigma_1(v)$,

$$
v = (1-\delta)\, r(y, y) + \delta\, \sigma_2(v, y, y)
  \;\geq\; (1-\delta)\, r(y, \eta) + \delta\, \sigma_2(v, y, \eta),
  \qquad \forall\, \eta \in Y .
$$
```

The definition attaches four objects to an equilibrium: a promised value $v$; a first-period
outcome $(y, y)$; a continuation value $v'$ if the prescribed outcome is observed; and another
continuation value $\tilde v$ if it is not.

In those terms condition (b) reads

$$
v = (1-\delta) r(y, y) + \delta v' \;\geq\; (1-\delta) r(y, \eta) + \delta \tilde v,
\qquad \forall \eta \in Y,
$$

which simply says the government does better adhering than departing.

Condition (a) says the continuation values must themselves be equilibrium values.

The definition is circular — equilibrium values appear on both sides — and that circularity is
exactly what recursivity buys, and what APS learned to exploit.

(cpol_aps)=
## The Abreu–Pearce–Stacchetti method

Dynamic programming computes an optimal value function by iterating the Bellman equation, a map
that turns tomorrow's value function into today's.

APS adapted the idea to equilibrium *sets*.

Start with a candidate set $W \subset \mathbb{R}$ of continuation values.

Pick a first-period rational-expectations outcome $(y, y)$ and two continuation values from
$W$: a $w_1$ to reward adherence and a $w_2$ to punish deviation.

If $w_1$ is high enough and $w_2$ low enough, the pair supports $y$, and delivers the value

$$
w = (1-\delta) r(y, y) + \delta w_1 \;\geq\; (1-\delta) r(y, \eta) + \delta w_2,
\qquad \forall \eta \in Y .
$$

```{prf:definition} Admissibility
:label: cp_admissible

A pair $(y, w)$ is **admissible** with respect to a set of continuation values $W$ if there
exist $w_1, w_2 \in W$ with
$w = (1-\delta) r(y,y) + \delta w_1 \geq (1-\delta) r(y,\eta) + \delta w_2$ for all
$\eta \in Y$.
```

Let $\mathcal{T}(W)$ collect the $w$ components of all admissible pairs.

APS and {cite:t}`Sargent1999` call this operator $B$; we write $\mathcal{T}$ to keep $B$ for the
government's best response.

This construction builds in condition (b) of {prf:ref}`cp_spe` but ignores condition (a),
because the continuation values were drawn from an arbitrary set.

```{prf:definition} Self-generation
:label: cp_selfgen

A set $W$ of prospective continuation values is **self-generating** if
$W \subseteq \mathcal{T}(W)$.
```

Every value in a self-generating set is supported by continuation values drawn from that same
set — which is condition (a).

APS proved four things:

1. the set $V$ of SPE values is the largest self-generating set;
1. $\mathcal{T}$ maps compact sets into compact sets;
1. $\mathcal{T}$ is monotone: $W_2 \subseteq W_1$ implies
   $\mathcal{T}(W_2) \subseteq \mathcal{T}(W_1)$;
1. starting from any compact $W_0$ with $V \subseteq \mathcal{T}(W_0) \subseteq W_0$, the
   iteration $W_j = \mathcal{T}(W_{j-1})$ converges monotonically to $V = \mathcal{T}(V)$
   ({cite:t}`APS1990`, Theorem 5).

The requirement $V \subseteq \mathcal{T}(W_0)$ in the last item cannot be dropped, although the
book's statement of the theorem omits it.

The single point $W_0 = \{v^N\}$ satisfies $\mathcal{T}(W_0) \subseteq W_0$, yet it is a fixed
point of $\mathcal{T}$ far smaller than $V$ (exercise {ref}`cpol_ex6`).

A safe choice is the set of all feasible values, $W_0 = [r(y^\#, y^\#),\, 0]$.

It contains $V$, and $\mathcal{T}$ maps it into itself, so monotonicity gives
$V = \mathcal{T}(V) \subseteq \mathcal{T}(W_0) \subseteq W_0$.

### Computing the operator

Two simplifications make $\mathcal{T}$ easy to evaluate here.

Because the government's temptation is worst when it best-responds, the constraint in
{prf:ref}`cp_admissible` binds hardest at $\eta = B(y)$, so we may replace "for all $\eta$" by
that single deviation.

And because a lower punishment relaxes the constraint, we may always set $w_2$ to the smallest
element of $W$.

Writing $W = [\underline w, \overline w]$, the outcome $y$ is then admissible provided some
$w_1 \in W$ satisfies

```{math}
:label: cp_bound

w_1 \;\geq\; \frac{(1-\delta)\left[r(y, B(y)) - r(y, y)\right]}{\delta} + \underline w
\;\equiv\; \ell(y) ,
```

and the values it generates sweep out the interval

$$
\left[(1-\delta) r(y,y) + \delta \max(\underline w, \ell(y)),\;
(1-\delta) r(y,y) + \delta \overline w\right] .
$$

```{code-cell} ipython3
def T_operator(m, W, n_grid=4001):
    """
    One application of the APS operator 𝒯 to an interval W = [w_lo, w_hi].

    Returns the new interval, or None if no first-period outcome is admissible.
    """
    w_lo, w_hi = W
    y = np.linspace(0.0, m.y_max, n_grid)
    ℓ = (1 - m.δ) * (r_cheat(m, y) - r_keep(m, y)) / m.δ + w_lo   # the bound ℓ(y) of cp_bound
    ok = ℓ <= w_hi                                                 # admissible outcomes
    if not ok.any():
        return None
    keep = (1 - m.δ) * r_keep(m, y)
    lows = keep[ok] + m.δ * np.maximum(w_lo, ℓ[ok])
    highs = keep[ok] + m.δ * w_hi
    return lows.min(), highs.max()


def solve_aps(m, W0=None, tol=1e-12, max_iter=10_000):
    """
    Iterate the APS operator from W0 until it settles.

    The default W0 is the set of all feasible values, [r(y^#, y^#), 0].
    """
    W = (r_keep(m, m.y_max), 0.0) if W0 is None else W0
    for it in range(max_iter):
        W_new = T_operator(m, W)
        if W_new is None:
            return None, it
        if max(abs(W_new[0] - W[0]), abs(W_new[1] - W[1])) < tol:
            return W_new, it
        W = W_new
    return W, max_iter
```

We start from $W_0 = [r(y^\#, y^\#),\, 0]$ and iterate.

```{code-cell} ipython3
W, iters = solve_aps(m)
print(f"converged in {iters} iterations")
print(f"set of SPE values V = [{W[0]:.4f}, {W[1]:.4f}]")
```

## The best and the worst equilibrium

APS's iteration is general but tells us little about *why* the set ends where it does.

For this economy both endpoints can be found by hand, and the arguments are instructive.

### The worst

The worst equilibrium value solves

$$
\underline v = \min_{y \in Y,\; v_1 \in V}\ \left[(1-\delta) r(y,y) + \delta v_1\right]
\quad\text{subject to}\quad
(1-\delta) r(y,y) + \delta v_1 \geq (1-\delta) r(y, B(y)) + \delta \underline v ,
$$

where the *worst* value is used as the continuation in the event of a deviation — the harshest
punishment available.

For a given $y$, the objective is smallest when $v_1$ is as low as the constraint allows, so at
the minimum the constraint binds, and the two sides collapse to $\underline v = r(y, B(y))$ for
the minimizing $y$.

If every $y \in Y$ could be used, finding the worst equilibrium would reduce to a
one-dimensional problem:

```{math}
:label: cp_worst

\underline v = \min_{y \in Y}\ r\bigl(y, B(y)\bigr)
= -\frac{\left(U^* + \theta y^\#\right)^2}{2\left(1 + \theta^2\right)} ,
```

where the closed form follows from {eq}`cp_schedules`, and the minimizing action is $y^\#$
because the deviation payoff falls monotonically in $y$.

But {eq}`cp_worst` is only a *candidate*.

To deliver it with a first period at $y^\#$, the strategy must reward adherence with the
continuation value $v_1$ that solves $\underline v = (1-\delta) r(y^\#, y^\#) + \delta v_1$,

```{math}
:label: cp_reward

v_1 = \frac{r\bigl(y^\#, B(y^\#)\bigr) - (1-\delta)\, r(y^\#, y^\#)}{\delta} ,
```

and that reward must itself be an equilibrium value: $v_1 \leq \overline v$.

That check is the last step of the book's construction: compute the candidate, compute the best
value $\overline v$ that it implies (below), and verify $v_1 \leq \overline v$.

The check can fail.

The reward {eq}`cp_reward` grows like $1/\delta$: an impatient government cannot be promised
enough future reward to accept a period at $y^\#$.

The worst equilibrium then starts at a lower inflation rate, the two programming problems must be
solved jointly, and we let the APS iteration do that.

Now the role of the upper bound $y^\#$ is clear.

The coefficient on $y^{\#2}$ in {eq}`cp_reward` is $\left[(1-\delta) - \theta^2/(1+\theta^2)\right] / (2\delta)$.

So when $\delta > 1/(1+\theta^2)$, the reward needed falls without limit as $y^\#$ rises, the
check keeps passing, and $y^\#$ is what keeps the worst punishment finite.

For a less patient government the check eventually fails, and beyond some point raising $y^\#$
changes nothing.

```{prf:proposition} The worst SPE is self-enforcing
:label: cp_prop_worst

At the worst equilibrium the continuation value following a deviation equals the initial
promised value, so a deviation simply restarts the equilibrium.
```

### The best

Given $\underline v$ as the threat, the best value solves

```{math}
:label: cp_best

\overline v = \max_{y \in Y}\ r(y, y)
\quad\text{subject to}\quad
r(y, y) \geq (1-\delta) r\bigl(y, B(y)\bigr) + \delta \underline v ,
```

where we have used the fact that the best value must reward adherence with itself, so that
$\overline v = (1-\delta) r(y,y) + \delta \overline v = r(y,y)$.

```{prf:proposition} The best SPE is self-rewarding
:label: cp_prop_best

At the best equilibrium the continuation value following adherence equals the promised value.
```

The code below implements both programming problems, with the check.

```{code-cell} ipython3
def next_value(m, v):
    "Solve v = (1-δ) r(y^#, y^#) + δ v' for the continuation value v'."
    return (v - (1 - m.δ) * r_keep(m, m.y_max)) / m.δ


def best_value(m, v_lo, n_grid=200_001):
    "The best SPE value, given the worst value as the punishment threat."
    y = np.linspace(0.0, m.y_max, n_grid)
    feasible = r_keep(m, y) >= (1 - m.δ) * r_cheat(m, y) + m.δ * v_lo
    if not feasible.any():
        return None, None
    vals = np.where(feasible, r_keep(m, y), -np.inf)
    k = vals.argmax()
    return vals[k], y[k]


def worst_value(m):
    """
    The worst SPE value and the action that attains it.

    The candidate r(y^#, B(y^#)) of cp_worst is accepted if the reward v_1 of
    cp_reward is no larger than the best value the candidate implies.
    Otherwise the worst value comes from the APS iteration, and it is attained
    at the highest action whose required reward ℓ(y) of cp_bound equals the
    best value (solved for y using cp_gap).
    """
    v_cand = r_cheat(m, m.y_max)
    v_best, _ = best_value(m, v_cand)
    if next_value(m, v_cand) <= v_best:
        return v_cand, m.y_max
    (w_lo, w_hi), _ = solve_aps(m)
    y_hat = y_nash(m) + np.sqrt(2 * (1 + m.θ**2) * m.δ * (w_hi - w_lo) / (1 - m.δ))
    return w_lo, y_hat


v_lo, y_sharp = worst_value(m)
v_hi, y_best = best_value(m, v_lo)

print(f"worst value  v_lo = {v_lo:.4f}   attained with y = {y_sharp:.2f}")
print(f"reward required by the candidate  v_1 = {next_value(m, v_lo):.4f}"
      f"  <= v_hi? {next_value(m, v_lo) <= v_hi}")
print(f"best  value  v_hi = {v_hi:.4f}   attained with y = {y_best:.2f}")
print(f"APS iteration gave [{W[0]:.4f}, {W[1]:.4f}]")
print(f"the two agree: {np.allclose(W, (v_lo, v_hi), atol=1e-6)}")
```

The programming problems and the set iteration agree, and at this discount factor the best
equilibrium is Ramsey itself.

The check passes easily at $\delta = 0.95$, but not for every discount factor.

```{code-cell} ipython3
print("    δ    candidate   reward v_1   best given candidate   check   worst value  at y")
for δ in (0.02, 0.04, 0.05, 0.2, 0.95):
    md = m._replace(δ=δ)
    v_cand = r_cheat(md, md.y_max)
    v_1, v_best = next_value(md, v_cand), best_value(md, v_cand)[0]
    v_worst, y_worst = worst_value(md)
    print(f"{δ:6.2f} {v_cand:11.4f} {v_1:12.4f} {v_best:16.4f} {str(v_1 <= v_best):>11}"
          f" {v_worst:12.4f} {y_worst:6.2f}")
```

For the two smallest discount factors the reward the candidate needs exceeds the best equilibrium
value.

The worst equilibrium then starts below $y^\#$, and its value lies above
$r(y^\#, B(y^\#))$; exercise {ref}`cpol_ex2` finds the discount factor at which this begins.

The next cell varies the upper bound $y^\#$ for a patient and an impatient government.

```{code-cell} ipython3
print(f"1/(1 + θ²) = {1 / (1 + m.θ**2):.3f}")
for δ in (0.2, 0.95):
    for y_max in (10.0, 30.0, 100.0):
        v_worst, y_worst = worst_value(m._replace(δ=δ, y_max=y_max))
        print(f"δ = {δ:4.2f}, y^# = {y_max:5.1f}:  worst value {v_worst:10.2f}"
              f"  attained with y = {y_worst:6.2f}")
```

For the patient government the worst value keeps falling as $y^\#$ rises, and it is always
attained at $y^\#$.

For the impatient government it stops falling: once $y^\#$ is large enough, the worst
equilibrium starts at the same inflation rate below $y^\#$, whatever the bound (the small
differences in the output come from the grid in `T_operator`).

Without a bound on inflation, almost anything could be supported, but only for a patient
government.

## Examples of recursive equilibria

We now construct particular equilibria by guess-and-verify, in the order in which the literature
found them.

### Infinite repetition of Nash

The easiest equilibrium repeats the one-period Nash outcome forever.

Take $v_1 = v^N = r(y^N, y^N)$, $\sigma_1(v) = y^N$ for every $v$, and
$\sigma_2(v, x, y) = v^N$ for every $(v, x, y)$.

Condition (a) holds by construction, and condition (b) collapses to
$r(y^N, y^N) \geq r(y^N, B(y^N))$, which holds *with equality* because $y^N$ is a fixed point of
the best response map.

Nothing deters the government here, and nothing needs to: it is already doing what it most
wants to do.

This is the Markov perfect equilibrium of the repeated game.

The economy has no payoff-relevant state variable, so a Markov strategy cannot condition on
anything, and the government simply plays its one-period best response every period
({doc}`markov_perf` studies Markov perfect equilibria of dynamic games that do have states).

### Infinite repetition of something better

Let $v^b = r(y^b, y^b) > v^N$ be a better-than-Nash value, and suppose

```{math}
:label: cp_bg

r\bigl(y^b, B(y^b)\bigr) - r(y^b, y^b)
\;\leq\; \frac{\delta}{1-\delta}\left(v^b - v^N\right).
```

The left side is the one-period gain from cheating; the right side is the discounted loss from
reverting to Nash forever.

When {eq}`cp_bg` holds, the strategy that prescribes $y^b$ while the promised value is $v^b$ and
reverts to Nash after any deviation is an SPE.

{cite:t}`BarroGordon1983` and {cite:t}`stokey1989reputation` studied the case $y^b = y^R$:
anticipated reversion to Nash supports Ramsey forever.

Whether it does depends on patience, and for this economy the threshold has a remarkably clean
form.

Setting $y^b = 0$ in {eq}`cp_bg` and using the closed forms {eq}`cp_schedules`, every $U^*$ and
every scale factor cancels, leaving

```{math}
:label: cp_cutoff

\delta \;\geq\; \delta^\star = \frac{1}{2 + \theta^2} .
```

```{code-cell} ipython3
def supports_ramsey_by_nash(m):
    "Does reversion to Nash sustain Ramsey forever?"
    gain = r_cheat(m, 0.0) - r_keep(m, 0.0)
    loss = m.δ / (1 - m.δ) * (r_keep(m, 0.0) - r_keep(m, y_nash(m)))
    return gain <= loss


δ_star = 1 / (2 + m.θ**2)
print(f"closed-form cutoff  δ* = 1/(2 + θ²) = {δ_star:.6f}\n")
for δ in (0.20, 0.27, δ_star - 1e-6, δ_star + 1e-6, 0.50, 0.95):
    md = m._replace(δ=δ)
    print(f"  δ = {δ:.8f}: Nash reversion supports Ramsey? "
          f"{supports_ramsey_by_nash(md)}")
```

Above $\delta^\star \approx 0.28$ the threat of reverting to Nash is enough to hold the
government at zero inflation; below it, the threat is too weak.

(cpol_abreu)=
### Something worse: Abreu's stick and carrot

When reversion to Nash is not a strong enough threat, a *worse* equilibrium might serve as the
punishment.

{cite:t}`Abreu1986`, studying oligopoly, and {cite:t}`Stokey1991`, studying credible public
policy, showed how to construct one; {cite:t}`Abreu` developed the general theory of such optimal
punishments.

The device is a stick-and-carrot strategy: the government is required to inflate at the maximum
rate $y^\#$ for one period — the stick — after which it is rewarded with the Ramsey outcome
forever.

The value is

```{math}
:label: cp_abreu

\tilde v = (1-\delta)\, r(y^\#, y^\#) + \delta\, v^R ,
```

and a deviation at any date, in either phase, *restarts* the whole strategy, so the continuation
value after a deviation is always $\tilde v$ itself.

The strategy is an SPE only if neither phase tempts the government.

*The stick.* In the first period the incentive constraint is

$$
\tilde v = (1-\delta)\, r(y^\#, y^\#) + \delta\, v^R
\;\geq\; (1-\delta)\, r\bigl(y^\#, B(y^\#)\bigr) + \delta\, \tilde v .
$$

Moving $\delta \tilde v$ to the left and dividing by $1-\delta$ leaves
$\tilde v \geq r\bigl(y^\#, B(y^\#)\bigr)$: the stick must be worth at least the candidate worst
value {eq}`cp_worst`.

*The carrot.* Once at Ramsey, the government must prefer staying there to grabbing the
one-period gain from $B(0)$ and restarting the stick:

```{math}
:label: cp_abreu_carrot

v^R \;\geq\; (1-\delta)\, r\bigl(0, B(0)\bigr) + \delta\, \tilde v .
```

Substituting {eq}`cp_abreu`, and using {eq}`cp_gap` and $v^R - r(y^\#, y^\#) = y^{\#2}/2$, the two
constraints become

$$
\delta \;\geq\; \frac{\left(y^\# - y^N\right)^2}{(1+\theta^2)\, y^{\#2}}
\quad \text{(stick)},
\qquad
\delta \;\geq\; \delta_A = \frac{\theta^2 U^{*2}}{(1+\theta^2)\, y^{\#2}}
\quad \text{(carrot)} .
$$

Here $y^N > y^\# - y^N$, so the carrot is the constraint that binds.

```{code-cell} ipython3
def abreu_value(m):
    "Value of the stick-and-carrot strategy: one period at y^#, then Ramsey."
    return (1 - m.δ) * r_keep(m, m.y_max) + m.δ * r_keep(m, 0.0)


def supports_ramsey_with_threat(m, threat):
    "Does a permanent threat value sustain Ramsey forever?"
    return r_keep(m, 0.0) >= (1 - m.δ) * r_cheat(m, 0.0) + m.δ * threat


def is_abreu_spe(m):
    """
    Check both incentive constraints of the stick-and-carrot strategy, in which
    any deviation restarts the strategy.

    Returns (stick constraint holds, carrot constraint holds, both hold).
    """
    v_tilde = abreu_value(m)
    stick = v_tilde >= (1 - m.δ) * r_cheat(m, m.y_max) + m.δ * v_tilde
    carrot = supports_ramsey_with_threat(m, v_tilde)        # cp_abreu_carrot
    return stick, carrot, stick and carrot


δ_stick = (m.y_max - y_nash(m))**2 / ((1 + m.θ**2) * m.y_max**2)
δ_A = m.θ**2 * m.U_star**2 / ((1 + m.θ**2) * m.y_max**2)
print(f"stick holds for δ >= {δ_stick:.4f};  carrot holds for δ >= δ_A = {δ_A:.4f}\n")
for δ in (0.10, 0.20, 0.95):
    md = m._replace(δ=δ)
    va, vn = abreu_value(md), r_keep(md, y_nash(md))
    stick, carrot, spe = is_abreu_spe(md)
    print(f"δ = {δ:4.2f}:  v_abreu = {va:8.4f}   v^N = {vn:8.4f}   "
          f"worse than Nash? {str(va < vn):5}   stick? {str(stick):5}   "
          f"carrot? {str(carrot):5}   SPE? {spe}")
```

At $\delta = 0.95$ the stick-and-carrot value is *better* than Nash, because a single bad period
is heavily outweighed by an eternity of Ramsey.

At $\delta = 0.2$ it is far *worse* than Nash — and that is the point.

At $\delta = 0.1$ the government would still accept the stick, but it would not stay at Ramsey: the
one-period gain from inflating outweighs the discounted cost of restarting the stick.

So at $\delta = 0.1$ the strategy is not an equilibrium, even though its first-period constraint
holds.

The carrot constraint {eq}`cp_abreu_carrot` asks whether the threat $\tilde v$ sustains Ramsey
forever.

A punishment worse than Nash is a stronger deterrent than reversion to Nash, so it can support
outcomes that Nash reversion cannot.

```{code-cell} ipython3
m2 = m._replace(δ=0.20)

print(f"at δ = 0.2:")
print(f"  threat = Nash    ({r_keep(m2, y_nash(m2)):8.4f}): "
      f"supports Ramsey? {supports_ramsey_with_threat(m2, r_keep(m2, y_nash(m2)))}")
print(f"  threat = Abreu   ({abreu_value(m2):8.4f}): "
      f"supports Ramsey? {supports_ramsey_with_threat(m2, abreu_value(m2))}")
```

At $\delta = 0.2$, an impatient government cannot be held at Ramsey by the prospect of Nash, but
*can* be held there by the prospect of Abreu's stick.

The worst equilibrium value $\underline v$ is a harsher punishment still, and it holds Ramsey
in place down to an even lower discount factor (exercise {ref}`cpol_ex4`).

Two definitions name the structures we have just met.

```{prf:definition} Self-enforcing and self-rewarding
:label: cp_selfenf

A recursive SPE is **self-enforcing** if the continuation value following a deviation equals the
initial promised value, and **self-rewarding** if the continuation value following adherence
equals the promised value.
```

Abreu's stick-and-carrot is self-enforcing; infinite repetition of a better-than-Nash outcome is
self-rewarding.

{prf:ref}`cp_prop_worst` and {prf:ref}`cp_prop_best` say that these two structures are not
curiosities: they are exactly what the extreme equilibria look like.

## Multiplicity

Two layers of multiplicity inhabit this theory.

There is a continuum of equilibrium *values*, and — the sharper point — many different outcome
*paths* attain the same value.

To see the second, we construct three equilibria that all deliver the worst value
$\underline v$.

All three begin the same way: the first-period promised value is $\underline v$, the prescribed
action is $y^\#$, and the continuation value $v_2$ solves
$\underline v = (1-\delta) r(y^\#, y^\#) + \delta v_2$.

They differ in what they do afterwards.

The code reuses `next_value` from the worst-value section.

```{code-cell} ipython3
def action_delivering(m, v):
    "The inflation rate ỹ with r(ỹ, ỹ) = v."
    return np.sqrt(max(-2.0 * v - m.U_star**2, 0.0))


def method_1(m, v_lo, v_hi, max_steps=500):
    "Climb at y^# until one more step would overshoot v_hi, then settle at v_hi."
    v, y = [v_lo], []
    for _ in range(max_steps):
        nxt = next_value(m, v[-1])
        if nxt > v_hi:
            break
        y.append(m.y_max)
        v.append(nxt)
    target = (v[-1] - m.δ * v_hi) / (1 - m.δ)     # r(ỹ, ỹ) for the switching period
    y.append(action_delivering(m, target))
    v.append(v_hi)
    y.append(action_delivering(m, v_hi))          # stay at v_hi forever after
    return np.array(v), np.array(y)


def method_2(m, v_lo, max_steps=500):
    "Climb at y^# only until the promised value first exceeds v^N, then freeze."
    v_N = r_keep(m, y_nash(m))
    v, y = [v_lo], []
    for _ in range(max_steps):
        nxt = next_value(m, v[-1])
        y.append(m.y_max)
        v.append(nxt)
        if nxt > v_N:
            break
    y.append(action_delivering(m, v[-1]))          # freeze at v** forever
    return np.array(v), np.array(y)


def method_3(m, v_lo):
    "One period at y^#, then freeze immediately at v_2."
    v_2 = next_value(m, v_lo)
    return (np.array([v_lo, v_2, v_2]),
            np.array([m.y_max, action_delivering(m, v_2),
                      action_delivering(m, v_2)]))
```

The figure plots the promised values and inflation rates along the three paths.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Three subgame perfect equilibria that attain the same worst value (the book's Figures 4.2-4.4)
    name: fig-cpol-three-methods
---
paths = {'method 1': method_1(m, v_lo, v_hi),
         'method 2': method_2(m, v_lo),
         'method 3': method_3(m, v_lo)}

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex='col')
for k, (name, (v, y)) in enumerate(paths.items()):
    axes[0, k].plot(range(1, len(v) + 1), v, 'C0o-', ms=3.5, lw=1)
    axes[0, k].axhline(v_lo, color='k', ls='--', lw=0.8)
    axes[0, k].axhline(v_hi, color='C2', ls=':', lw=0.8)
    axes[0, k].set_title(f'{name}: continuation values', fontsize=10)
    axes[1, k].plot(range(1, len(y) + 1), y, 'C1o-', ms=3.5, lw=1)
    axes[1, k].set_ylim(-0.4, m.y_max + 0.4)
    axes[1, k].set_xlabel('$t$')
    axes[1, k].set_title(f'{name}: inflation', fontsize=10)
axes[0, 0].set_ylabel('promised value $v_t$')
axes[1, 0].set_ylabel('inflation $y_t$')
plt.tight_layout()
plt.show()
```

Here are their lengths and terminal values.

```{code-cell} ipython3
print(f"Nash value {r_keep(m, y_nash(m)):.4f} at inflation {y_nash(m):.4f};  "
      f"worst value {v_lo:.4f}\n")
for name, (v, y) in paths.items():
    print(f"{name:9s}: {len(y):3d} periods before settling,  "
          f"first-period value {v[0]:.4f},  "
          f"terminal value {v[-1]:8.4f} at inflation {y[-1]:.4f}")
```

The three paths could hardly be less alike.

The first climbs at maximum inflation for about sixty periods before easing down to Ramsey.

The second climbs for a shorter spell and then freezes forever at a value just *better* than
Nash, sustained by an inflation rate just below the Nash rate.

The third inflates at the maximum for a single period and then settles immediately, at a value
barely above the worst one.

All three are subgame perfect, and all three deliver exactly the same value to the government.

The equilibrium concept has nothing to say about which of them describes the world.

## Numerical example

Here is the numerical example reported in chapter 4 of {cite:t}`Sargent1999`, computed from
scratch.

```{code-cell} ipython3
rows = [
    ("$\\theta$", m.θ, ""),
    ("$U^*$", m.U_star, ""),
    ("$y^\\#$", m.y_max, ""),
    ("$\\delta$", m.δ, ""),
    ("$y^N$ (Nash inflation)", y_nash(m), "6.8750"),
    ("$y^R$ (Ramsey inflation)", 0.0, "0"),
    ("$v^R$", r_keep(m, 0.0), "-15.1250"),
    ("$v^N$", r_keep(m, y_nash(m)), "-38.7578"),
    ("$\\underline v$", v_lo, "-63.2195"),
    ("$v_{\\rm abreu}$", abreu_value(m), "-17.6250"),
]
pd.DataFrame(rows, columns=["object", "computed", "reported in the book"]).set_index("object")
```

The book also reports the Nash-reversion cutoff and the value of the stick and carrot at $\delta = 0.2$.

```{code-cell} ipython3
print(f"cutoff discount factor δ*        : {1 / (2 + m.θ**2):.4f}  (book reports 0.2807)")
print(f"Abreu stick-and-carrot at δ = 0.2: {abreu_value(m._replace(δ=0.2)):.4f}"
      f"  (book reports -55.125)")
```

Every entry reproduces the book.

## Interpretations

The literature on credible plans bears mixed news for the *triumph of natural-rate theory*
story of {doc}`phillips_two_stories`.

It certainly rescues the government from the pessimism of {doc}`phillips_credibility`: better
outcomes than Nash are available, and at plausible discount factors Ramsey itself is an
equilibrium.

But it rescues too much.

Values worse than Nash are equilibria too, and between the extremes lies a continuum.

The multitude of outcomes mutes the model empirically, and it undoes the very thing early
rational-expectations researchers wanted from the hypothesis — the elimination of free
parameters describing expectations.

### Whose expectations are they?

In his 1979 review of an OECD report edited by Paul McCracken, Lucas protested the report's
recommendation that "governments should try to promote good expectations," as though
expectations were an extra set of policy instruments.

In 1979 that protest was well aimed: rational-expectations models then took government policy as
exogenous and made expectations a *function* of it, linked by cross-equation restrictions.

The theory of credible policy changes the picture in a way that vindicates neither side
cleanly.

It turns systems of expectations into free parameters that influence outcomes — but into
parameters that nobody inside the model gets to choose.

The government complies with equilibrium expectations about its own behavior.

Within an equilibrium, the government's strategy is simultaneously a decision rule and a
description of the public's expectations, and the two cannot be disentangled.

As Sargent puts it: the authors of the McCracken report believed in multiplicity and
manipulation, while Lucas doubted both; the literature on credible plans supports multiplicity
but not manipulation.

(cpol_remedies)=
### Remedies

Reputation alone is a weak foundation for anti-inflation policy, and that weakness has inspired
proposals to change the game rather than to hope for a good equilibrium within it.

{cite:t}`Rogoff1985` proposed delegating monetary policy to someone who cares less about
unemployment than society does.

Assigning it to someone unaware even of a *temporary* tradeoff would be equally effective, and
Alan Blinder later suggested a related device: an authority who knows the natural rate and never
wants unemployment to differ from it.

Exercise {ref}`pc_ex3` works out both remedies in the one-period economy of
{doc}`phillips_credibility`, including the price Rogoff's remedy pays when there are shocks to
stabilize.

Maintaining a pool of potential central bankers with different inflation–unemployment
preferences, some like Rogoff's conservative and others like the authority in Kydland and
Prescott's model, can also improve outcomes {cite:p}`BarroGordon1983,Ball1995`.

When the public does not know which kind of banker is in office, even a government of the
inflationary kind can gain by acting like the conservative kind and building a reputation;
{cite:t}`BackusDriffill1985` and {cite:t}`Barro1986` model this as a game of incomplete
information.

### Where this leaves us

Every remedy just listed changes the *institutions*, not the theory.

Within the theory, the multiplicity is irreducible, and Sargent locates its source precisely: it
stems from the rationality imputed to *everyone* in the system.

Perfection is what generates the continuum, because a system of expectations sophisticated
enough to support one equilibrium is sophisticated enough to support many.

So the book retreats from perfection.

The remaining lectures replace fully rational participants with ones whose understanding of the
economy is limited.

First the public forecasts adaptively, in {doc}`phillips_adaptive`, which comes next.

Then agents fit misspecified models to the data their own beliefs generate, in
{doc}`phillips_misspecified` and {doc}`phillips_self_confirming`; in the self-confirming
equilibria of the latter, the misspecified agent is the government.

Finally, from {doc}`phillips_learning` on, the government re-estimates its model in real time.

Those models are closer to what Lucas had in mind when he criticized the McCracken report, and,
unlike the theory of this lecture, they make predictions.

## Exercises

```{exercise-start}
:label: cpol_ex1
```

The cutoff {eq}`cp_cutoff` claims that reversion to Nash sustains Ramsey forever if and only if
$\delta \geq 1/(2 + \theta^2)$ — a threshold that depends on $\theta$ alone, and not on the
natural rate $U^*$ or the upper bound $y^\#$.

Verify both claims numerically.

For a grid of $\theta$ values, find the smallest $\delta$ on a fine grid at which reversion to
Nash supports Ramsey, and compare it with $1/(2+\theta^2)$.

Then check that changing $U^*$ leaves the answer untouched.

```{exercise-end}
```

```{solution-start} cpol_ex1
:class: dropdown
```

```{code-cell} ipython3
δ_grid = np.linspace(0.01, 0.99, 9801)

rows = []
for θ in (0.5, 1.0, 1.25, 2.0, 3.0):
    for U_star in (5.5, 11.0):
        md = Model(θ=θ, U_star=U_star)
        ok = [δ for δ in δ_grid
              if supports_ramsey_by_nash(md._replace(δ=δ))]
        rows.append((θ, U_star, min(ok), 1 / (2 + θ**2)))

pd.DataFrame(rows, columns=["$\\theta$", "$U^*$", "smallest $\\delta$ found",
                            "$1/(2+\\theta^2)$"]).round(4)
```

The grid search matches the closed form to within the grid spacing, and doubling $U^*$ changes
nothing.

The reason both $U^*$ and $y^\#$ drop out is visible in {eq}`cp_schedules`: the one-period
temptation at $y = 0$ and the Nash–Ramsey value gap are both proportional to $U^{*2}$, so the
scale cancels from the inequality, and $y^\#$ never enters because neither side involves the
upper bound.

A larger unemployment response $\theta$ to surprise inflation makes the Nash outcome worse
relative to Ramsey, which strengthens the threat and lets a *less* patient government be held at
zero inflation.

(In the $(U, y)$ plane of {doc}`phillips_credibility` the Phillips curve has slope $-1/\theta$,
so a larger $\theta$ means a *flatter* curve.)

```{solution-end}
```

```{exercise-start}
:label: cpol_ex2
```

The set of equilibrium values shrinks as the government becomes impatient.

Use `solve_aps` to compute $V = [\underline v, \overline v]$ for discount factors between
$0.001$ and $0.98$, and plot both endpoints against $\delta$ on a logarithmic axis, marking the
Nash and Ramsey values.

1. Below what discount factor $\delta_R$ does the best equilibrium value stop being Ramsey?
   Find $\delta_R$ exactly, by asking when the punishment $r(y^\#, B(y^\#))$ makes the
   government prefer Ramsey to the one-period gain from deviating, and confirm it with
   `solve_aps`.
1. At small $\delta$ the lower edge of $V$ bends away from $r(y^\#, B(y^\#))$.
   Find where, and explain why, using the check of the worst-value section.
1. What happens to $V$ as $\delta \to 0$, and why is that the answer you should expect?

```{exercise-end}
```

```{solution-start} cpol_ex2
:class: dropdown
```

Let $G = r(0, B(0)) - v^R$ be the one-period gain from deviating at Ramsey.

With $\underline v$ as a permanent punishment, Ramsey is sustained when
$v^R \geq (1-\delta)(v^R + G) + \delta \underline v$, that is, when
$\delta \geq \delta_R = G / (G + v^R - \underline v)$.

For the bend, we find by bisection the discount factor $\delta_w$ at which the reward
{eq}`cp_reward` that the candidate $r(y^\#, B(y^\#))$ requires just equals the best value it
implies.

```{code-cell} ipython3
v_R, v_N = r_keep(m, 0.0), r_keep(m, y_nash(m))
v_cand = r_cheat(m, m.y_max)
G = r_cheat(m, 0.0) - v_R                  # one-period gain from deviating at Ramsey
δ_R = G / (G + v_R - v_cand)


def check_gap(δ):
    "Reward required by the candidate worst value minus the best value it implies."
    md = m._replace(δ=δ)
    return next_value(md, v_cand) - best_value(md, v_cand)[0]


lo, hi = 0.01, 0.1                     # the check fails at lo and passes at hi
for _ in range(50):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if check_gap(mid) > 0 else (lo, mid)
δ_w = hi

print(f"δ_R = {δ_R:.4f};  δ_w = {δ_w:.4f};  δ* = {1 / (2 + m.θ**2):.4f}")
for δ in (δ_R - 0.001, δ_R + 0.001):
    Wδ, _ = solve_aps(m._replace(δ=δ))
    print(f"  δ = {δ:.4f}: best value {Wδ[1]:.4f}, equals Ramsey? "
          f"{np.isclose(Wδ[1], v_R, atol=1e-6)}")
for δ in (0.001, 0.01, 0.02):
    Wδ, _ = solve_aps(m._replace(δ=δ))
    print(f"  δ = {δ:.3f}: V = [{Wδ[0]:.3f}, {Wδ[1]:.3f}], "
          f"width / δ = {(Wδ[1] - Wδ[0]) / δ:.0f}")
```

The figure marks the three discount factors.

```{code-cell} ipython3
δs = np.geomspace(0.001, 0.98, 120)
lows, highs = [], []
for δ in δs:
    Wδ, _ = solve_aps(m._replace(δ=δ))
    lows.append(Wδ[0])
    highs.append(Wδ[1])

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.fill_between(δs, lows, highs, alpha=0.2, color='C0', label='set of SPE values $V$')
ax.plot(δs, highs, 'C0', lw=1.4)
ax.plot(δs, lows, 'C0', lw=1.4)
ax.axhline(v_R, color='C2', ls=':', lw=1.2, label='Ramsey $v^R$')
ax.axhline(v_N, color='k', ls='--', lw=1, label='Nash $v^N$')
for δ_mark, name in [(δ_w, r'$\delta_w$'), (δ_R, r'$\delta_R$'),
                     (1 / (2 + m.θ**2), r'$\delta^\star$')]:
    ax.axvline(δ_mark, color='gray', lw=0.8)
    ax.annotate(name, (δ_mark, -60), textcoords='offset points', xytext=(3, 0))
ax.set_xscale('log')
ax.set_xlabel(r'discount factor $\delta$')
ax.set_ylabel('value')
ax.legend(loc='center right')
plt.show()
```

The best equilibrium value equals Ramsey for $\delta \geq \delta_R \approx 0.161$, well *below* the
$\delta^\star \approx 0.28$ of {eq}`cp_cutoff`.

The cutoff $\delta^\star$ uses reversion to *Nash* as the threat, while the APS set uses the
worst equilibrium, a much harsher threat, so Ramsey survives at lower discount factors than
Barro and Gordon's argument alone would suggest.

Below $\delta_w \approx 0.044$ the check of the worst-value section fails.

The reward {eq}`cp_reward` needed to make one period at $y^\#$ acceptable grows like $1/\delta$,
and below $\delta_w$ it exceeds the best equilibrium value.

The worst equilibrium must then start at a lower inflation rate, and the lower edge of $V$ rises
above $r(y^\#, B(y^\#))$ toward $v^N$.

As $\delta \to 0$ the set collapses toward the single point $v^N$, with a width roughly
proportional to $\delta$.

That is what it must do: an entirely impatient government cares only about the current period,
promises about the future carry no weight, and the only outcome that can be sustained is the one
the government would choose myopically — the Nash outcome.

The sets are nested, growing with $\delta$; {cite:t}`APS1990` prove that, in their setting, the
set of average equilibrium values is monotone in the discount factor.

```{solution-end}
```

```{exercise-start}
:label: cpol_ex3
```

Method 3 in the multiplicity section is the boldest of the three: it inflates at $y^\#$ for a
single period and then freezes forever at a constant inflation rate.

Because it settles immediately, its incentive constraint is the tightest of the three, so it is
worth checking rather than assuming.

Verify directly that method 3 is a subgame perfect equilibrium: confirm that its first-period
promise is honoured, that its frozen continuation value lies inside $V$, and that the government
prefers adherence to deviation in *both* phases, using $\underline v$ as the punishment.

```{exercise-end}
```

```{solution-start} cpol_ex3
:class: dropdown
```

```{code-cell} ipython3
v_2 = next_value(m, v_lo)
y_tilde = action_delivering(m, v_2)

# phase 1: promised v_lo, prescribed action y^#, continuation v_2 on adherence
lhs_1 = (1 - m.δ) * r_keep(m, m.y_max) + m.δ * v_2
rhs_1 = (1 - m.δ) * r_cheat(m, m.y_max) + m.δ * v_lo

# phase 2: promised v_2, prescribed action ỹ, continuation v_2 on adherence
lhs_2 = (1 - m.δ) * r_keep(m, y_tilde) + m.δ * v_2
rhs_2 = (1 - m.δ) * r_cheat(m, y_tilde) + m.δ * v_lo

print(f"v_2 = {v_2:.4f},  ỹ = {y_tilde:.4f}")
print(f"v_2 lies inside V = [{v_lo:.4f}, {v_hi:.4f}]: "
      f"{v_lo <= v_2 <= v_hi}")
print(f"phase 1 delivers the promise: {np.isclose(lhs_1, v_lo)}")
print(f"phase 1 incentive: {lhs_1:.4f} vs {rhs_1:.4f}  ->  "
      f"holds with equality? {np.isclose(lhs_1, rhs_1)}")
print(f"phase 2 delivers the promise: {np.isclose(lhs_2, v_2)}")
print(f"phase 2 incentive: {lhs_2:.4f} >= {rhs_2:.4f}  ->  {lhs_2 >= rhs_2}")
print(f"slack in phase 2: {lhs_2 - rhs_2:.6f}")
```

All the conditions hold, and the first-period promise is delivered exactly.

The phase-1 constraint holds with equality, and that is just the definition of $\underline v$:
at $y^\#$ the government can always deviate to $B(y^\#)$ and accept the punishment
$\underline v$, which is worth $(1-\delta) r(y^\#, B(y^\#)) + \delta \underline v$, and
$\underline v = r(y^\#, B(y^\#))$ promises exactly that much.

The margin in the second phase is very thin, and that is not an accident either.

The frozen value $v_2$ is only barely above $\underline v$, so the punishment for deviating is
only barely worse than the equilibrium itself — which is precisely what it means to be near the
bottom of the set of equilibrium values.

```{solution-end}
```

```{exercise-start}
:label: cpol_ex4
```

This exercise compares three ways of holding the government at Ramsey forever.

1. Suppose any deviation from Ramsey is followed by a continuation value $v_p$ that does not
   depend on $\delta$.
   Show that Ramsey is sustained if and only if $\delta \geq G/(G + v^R - v_p)$, where
   $G = r(0, B(0)) - v^R$.
1. Evaluate this threshold for Nash reversion ($v_p = v^N$) and for the worst equilibrium
   ($v_p = \underline v = r(y^\#, B(y^\#))$, the worst value for every $\delta$ above the
   $\delta_w$ of exercise {ref}`cpol_ex2`).
1. Abreu's stick and carrot punishes with $\tilde v$, which depends on $\delta$; its threshold is
   the $\delta_A$ of {ref}`cpol_abreu`.
   Compare $\tilde v$ at $\delta_A$ with $\underline v$, and explain the difference.
1. Order the three thresholds, and check $\delta_A$ with `is_abreu_spe`.

```{exercise-end}
```

```{solution-start} cpol_ex4
:class: dropdown
```

**1.** Deviating from Ramsey earns $r(0, B(0)) = v^R + G$ for one period and $v_p$ thereafter,
so the constraint is $v^R \geq (1-\delta)(v^R + G) + \delta v_p$, which rearranges to
$\delta (G + v^R - v_p) \geq G$.

**2.** By {eq}`cp_gap`, $G = \theta^2 U^{*2} / \bigl(2(1+\theta^2)\bigr)$, and
$v^R - v^N = \theta^2 U^{*2}/2$, so Nash reversion gives $1/(2+\theta^2)$, the cutoff
{eq}`cp_cutoff`.

The worst equilibrium gives the $\delta_R$ of exercise {ref}`cpol_ex2`.

```{code-cell} ipython3
def ramsey_threshold(m, v_p):
    "Smallest δ at which the permanent punishment v_p sustains Ramsey."
    G = r_cheat(m, 0.0) - r_keep(m, 0.0)
    return G / (G + r_keep(m, 0.0) - v_p)


δ_nash = ramsey_threshold(m, r_keep(m, y_nash(m)))
δ_opt = ramsey_threshold(m, r_cheat(m, m.y_max))
δ_A = m.θ**2 * m.U_star**2 / ((1 + m.θ**2) * m.y_max**2)
print(f"Nash reversion      δ* = {δ_nash:.4f}")
print(f"Abreu stick/carrot  δ_A = {δ_A:.4f}")
print(f"worst equilibrium   δ_R = {δ_opt:.4f}\n")

m_A = m._replace(δ=δ_A)
print(f"at δ_A: Abreu's punishment {abreu_value(m_A):.4f},  worst value "
      f"{r_cheat(m, m.y_max):.4f},  reward after the stick in the worst equilibrium "
      f"{next_value(m_A, r_cheat(m, m.y_max)):.4f}\n")
for δ in (δ_A - 1e-4, δ_A + 1e-4):
    print(f"δ = {δ:.5f}: stick and carrot is an SPE? {is_abreu_spe(m._replace(δ=δ))[2]}")
```

**3.** Both punishments begin with a period at $y^\#$.

Abreu's then pays the full carrot, Ramsey forever.

The worst equilibrium pays a smaller reward after the stick, the $v_1$ of {eq}`cp_reward`, which
is itself worse than Nash, so its punishment is harsher.

**4.** The ordering is $\delta^\star > \delta_A > \delta_R$.

The harsher the punishment, the lower the discount factor at which it deters the one-period
gain $G$.

Nash reversion is the mildest of the three punishments, the worst equilibrium the harshest, and
Abreu's simple stick and carrot lies between them.

```{solution-end}
```

```{exercise-start}
:label: cpol_ex5
```

For $\delta < \delta^\star$, reversion to Nash cannot sustain Ramsey.

The best value it can sustain is equation (25) of chapter 4 of {cite:t}`Sargent1999`:

$$
v^{nr} = r(y^{nr}, y^{nr}) = (1-\delta)\, r\bigl(y^{nr}, B(y^{nr})\bigr) + \delta\, v^N ,
$$

where $y^{nr}$ is the lowest inflation rate at which adhering is at least as good as deviating
and reverting to Nash forever.

1. Using {eq}`cp_gap`, show that for $\delta \leq \delta^\star$

   $$
   y^{nr} = \theta U^* \, \frac{1 - \delta(2 + \theta^2)}{1 + \delta \theta^2} .
   $$

1. Plot $v^{nr}$ against $\delta \in (0, \delta^\star]$, together with the best SPE value
   $\overline v$ from `solve_aps`, and compare the two at $\delta = 0.2$.

```{exercise-end}
```

```{solution-start} cpol_ex5
:class: dropdown
```

**1.** The constraint $r(y, y) \geq (1-\delta) r(y, B(y)) + \delta v^N$ is equivalent to
$\delta \left[r(y, y) - v^N\right] \geq (1-\delta)\left[r(y, B(y)) - r(y, y)\right]$.

Write $z = y^N - y \geq 0$ for the distance below Nash.

Then $r(y, y) - v^N = z (2 y^N - z)/2$, and by {eq}`cp_gap` the right side is
$(1-\delta) z^2 / \bigl(2(1+\theta^2)\bigr)$.

Dividing by $z > 0$, the constraint becomes $z \leq 2 y^N \delta (1+\theta^2)/(1 + \delta\theta^2)$.

The lowest admissible inflation rate is $y^N$ minus that bound, which gives the formula.

It reaches zero exactly at $\delta = 1/(2+\theta^2) = \delta^\star$.

Every value between $v^N$ and $v^{nr}$ can also be sustained by Nash reversion, but no value above
$v^{nr}$.

```{code-cell} ipython3
def nash_reversion_best(m):
    "The best value sustainable by reversion to Nash (book eq. (25)) and its action."
    y_nr = max(y_nash(m) * (1 - m.δ * (2 + m.θ**2)) / (1 + m.δ * m.θ**2), 0.0)
    return r_keep(m, y_nr), y_nr


δ_star = 1 / (2 + m.θ**2)
δs_nr = np.linspace(0.005, δ_star, 60)
v_nr = [nash_reversion_best(m._replace(δ=δ))[0] for δ in δs_nr]
v_best = [solve_aps(m._replace(δ=δ))[0][1] for δ in δs_nr]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(δs_nr, v_best, 'C0', lw=1.6, label=r'best SPE value $\overline{v}$')
ax.plot(δs_nr, v_nr, 'C1', lw=1.6, label=r'best value with Nash reversion $v^{nr}$')
ax.axhline(r_keep(m, 0.0), color='C2', ls=':', lw=1.2, label='Ramsey $v^R$')
ax.axhline(r_keep(m, y_nash(m)), color='k', ls='--', lw=1, label='Nash $v^N$')
ax.set_xlabel(r'discount factor $\delta$')
ax.set_ylabel('value')
ax.legend()
plt.show()

m_02 = m._replace(δ=0.2)
v_nr_02, y_nr_02 = nash_reversion_best(m_02)
print(f"δ = 0.2: Nash reversion sustains at best {v_nr_02:.4f}, with y = {y_nr_02:.4f};"
      f"  best SPE value {solve_aps(m_02)[0][1]:.4f}")
```

At $\delta = 0.2$, Nash reversion can at best hold inflation down to about 1.5 percent, while
the optimal punishment sustains Ramsey itself.

The two curves meet at $\delta^\star$, where Nash reversion first sustains Ramsey.

As $\delta \to 0$ both curves fall back to $v^N$, because the worst equilibrium value itself
approaches $v^N$ (exercise {ref}`cpol_ex2`).

The gap closes only in levels: measured from $v^N$, the optimal punishment still sustains about
twice the improvement that Nash reversion does.

```{code-cell} ipython3
v_N = r_keep(m, y_nash(m))
for δ in [0.02, 0.005]:
    m_δ = m._replace(δ=δ)
    ratio = (solve_aps(m_δ)[0][1] - v_N) / (nash_reversion_best(m_δ)[0] - v_N)
    print(f"δ = {δ}: (v̄ - v^N) / (v^nr - v^N) = {ratio:.2f}")
```

```{solution-end}
```

```{exercise-start}
:label: cpol_ex6
```

The algorithm theorem of {ref}`cpol_aps` requires the starting set to satisfy
$V \subseteq \mathcal{T}(W_0) \subseteq W_0$.

1. Show that $W_0 = \{v^N\}$ satisfies $\mathcal{T}(W_0) = W_0$.
   Confirm it with `T_operator` and `solve_aps`, and say which hypothesis of the theorem fails.
1. Start instead from $W_0 = \{v^R\}$.
   Where does the iteration go, and why?
1. What do the two experiments say about what a set of continuation values needs in order to
   support anything other than Nash?

```{exercise-end}
```

```{solution-start} cpol_ex6
:class: dropdown
```

**1.** With a single continuation value $w$, the reward and the punishment coincide,
$w_1 = w_2 = w$, so admissibility requires $(1-\delta) r(y, y) \geq (1-\delta) r(y, B(y))$.

By {eq}`cp_gap` only $y = y^N$ qualifies, and it generates the single value
$(1-\delta) v^N + \delta w$.

So $\mathcal{T}(\{w\}) = \{(1-\delta) v^N + \delta w\}$, and $\mathcal{T}(\{v^N\}) = \{v^N\}$.

The hypothesis that fails is $V \subseteq \mathcal{T}(W_0)$.

The Nash rate $y^N = 6.875$ is a point of the grid in `T_operator`, so the numerical operator
finds it.

```{code-cell} ipython3
v_N, v_R = r_keep(m, y_nash(m)), r_keep(m, 0.0)
print(f"v^N = {v_N:.4f};  T({{v^N}}) = {T_operator(m, (v_N, v_N))}")
W_N, it_N = solve_aps(m, W0=(v_N, v_N))
print(f"iterating from {{v^N}}: stops after {it_N} iterations at {W_N}")
W_R, it_R = solve_aps(m, W0=(v_R, v_R))
print(f"iterating from {{v^R}}: stops after {it_R} iterations at "
      f"[{W_R[0]:.4f}, {W_R[1]:.4f}]")
```

**2.** Starting from $\{v^R\}$, the iterates are the single points
$(1-\delta^j)\, v^N + \delta^j v^R$, which converge to $\{v^N\}$, not to $V$.

This start fails the other hypothesis, since $\mathcal{T}(\{v^R\}) \not\subseteq \{v^R\}$.

**3.** Incentives come from the *spread* between the reward for adhering and the punishment for
deviating.

A single continuation value offers no spread, so the only outcome it can support is the one the
government chooses without incentives, the static best response.

A starting set must be wide enough to contain every reward and every punishment that equilibria
use, which is what $V \subseteq \mathcal{T}(W_0)$ guarantees, and what the set of all feasible
values $[r(y^\#, y^\#), 0]$ provides.

```{solution-end}
```
