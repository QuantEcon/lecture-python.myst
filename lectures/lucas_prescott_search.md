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

(lucas_prescott_search)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Equilibrium Search and Unemployment

```{contents} Contents
:depth: 2
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install quantecon
```

## Overview

This lecture studies {cite:t}`LucasPrescott1974`, the labor-market companion to the paper on investment that we studied in {doc}`lucas_prescott_investment`.

The two papers were written by the same authors at the same place within a few years of each other, and they use the same tools: a Bellman equation solved by successive approximations, Blackwell's theorem, and a Markov process whose invariant distribution describes the long run.

What changes is the question.

The question is one that the search literature of the 1960s had raised but could not answer.

{doc}`mccall_model` describes that literature's central problem: a worker draws wage offers from a known distribution and decides when to stop searching.

It is a model of an individual, and it takes the wage distribution as given.

But where does the distribution come from?

Lucas and Prescott set out to answer that, and discovered that the answer requires a general equilibrium model:

> *optimal labor supply behavior and the wage distribution on which it is based must be simultaneously determined within a model of market (as opposed to individual) behavior.*

They build an economy of many spatially separate markets -- "islands" -- each with its own randomly shifting demand for labor.

Workers in a market can work at the wage that clears it, or leave.

Leaving means earning nothing this period and arriving somewhere else next period, so **unemployment is the act of moving between markets**.

Everything is competitive, everyone optimizes, and expectations are rational in the sense of {cite:t}`muth1961`: the distribution that workers use to evaluate their prospects is the one that their own decisions generate.

The paper's central object is a single number, $\lambda$, the expected present value of search.

An individual market takes $\lambda$ as given, in the way a competitive firm takes a price as given.

Economy-wide, $\lambda$ adjusts until the workforce that markets want to hold equals the workforce the economy has.

In equilibrium some markets always have more workers than they want to employ, so

```{note}
**Unemployment is positive in every period.**

Lucas and Prescott argue that it is also efficient, and the argument is worth stating in their terms, although the paper proves no welfare theorem and this lecture will not either.

There is no congestion externality in the model and no wage rigidity.

The searching worker who crowds his fellows does so in the same way that any seller of a good crowds other sellers, and $\lambda$ is the price that signals the social return to an additional searcher.
```

Let's start with some imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from collections import namedtuple
```

## The economy

There are many markets, indexed by the state of demand in them.

Demand shifts are **independent across markets**, so that aggregate demand is constant, and **persistent within** a market, so that today's demand tells a worker something about tomorrow's.

In each market a competitive industry produces a single good with labor as the only input.

If $p = D(s, Q)$ is the inverse demand curve in demand state $s$ and $q(n)$ is the production function of each of the $m$ identical firms, then profit maximization sets the wage equal to the marginal value product of labor, and the function

```{math}
:label: lps_R
R(s, n) = D\big(s, m\,q(n)\big)\, q'(n)
```

summarizes everything about the demand side of this labor market.

Lucas and Prescott discard $D$ and $q$ and place assumptions directly on $R$: it is positive, bounded and differentiable, with

$$
R_s(s,n) > 0, \qquad R_n(s,n) < 0 .
$$

The demand state $s$ follows a Markov process with transition density $f(s', s)$ possessing a unique stationary distribution.

### Timing

A market begins a period with a **workforce** $y$, which is the most it can employ.

Then, within the period:

1. demand $s$ is realized and the labor market clears, employing $n \le y$ workers at the wage $w = R(s, n)$
1. the $y - n$ workers who are not employed earn nothing and join the pool of searchers
1. employed workers remain in the market into next period; searchers are allocated across markets and arrive next period

So a worker is **unemployed for exactly one period** each time he moves, and the unemployed are workers in transit.

### The value of search

Because demand shifts are independent across markets and there are many of them, the economy-wide distribution of workers across markets is constant even though each market's workforce fluctuates.

A searcher's prospects therefore do not depend on the market he is leaving, and the expected present value of search is a single number $\lambda$.

Searchers are allocated to markets in a way that **equalizes** the expected value of arriving, so no market offers an arriving worker more than $\lambda$.

We take $\lambda$ as given for now and determine it in {ref}`lps_economy_wide`.

## A worker's problem

Let $v(s, y, \lambda)$ be the expected present value of the earnings of a worker who finds himself in a market with demand $s$ and workforce $y$ at the start of a period, and let $\beta \in (0,1)$ be the discount factor.

Lucas and Prescott arrive at a functional equation for $v$ by considering three cases.

**Case A: some workers leave.** Departing workers get $\lambda$. The workers who stay can have no more, since the leavers could have stayed, and no less, since they could have left. So $v = \lambda$.

**Case B1: all workers stay and no new workers arrive.** Current employment is $y$, the wage is $R(s,y)$, and next period's workforce is still $y$:

$$
v(s,y,\lambda) = R(s,y) + \beta \int v(s', y, \lambda) f(s', s)\,ds' .
$$

**Case B2: all workers stay and new workers arrive.** Arriving workers are searchers, so the continuation value of being in this market next period must be exactly $\lambda$:

$$
v(s,y,\lambda) = R(s,y) + \lambda .
$$

Cases B1 and B2 differ according to whether the continuation value at the unchanged workforce exceeds $\lambda$, and in case A the worker takes $\lambda$.

All three collapse into one equation:

```{math}
:label: lps_bellman
v(s, y, \lambda) = \max\left\{\lambda,\;
R(s,y) + \min\left[\lambda,\;
\beta \int v(s', y, \lambda) f(s', s)\, ds' \right]\right\} .
```

Equation {eq}`lps_bellman` repays a careful look, because its two operators do different jobs.

The **max** is the worker's own option: he can always leave and collect $\lambda$.

The **min** is not the worker's choice at all.

It is a restriction that competition among searchers imposes: if the continuation value in this market exceeded $\lambda$, searchers would flow in until it did not.

```{prf:proposition}
:label: lps_prop_v

For each $\lambda$, equation {eq}`lps_bellman` has a unique bounded continuous solution $v$, and successive approximations converge to it from any starting point.

The solution is nondecreasing in $s$ and in $\lambda$ and nonincreasing in $y$.
```

The proof is the one used throughout {doc}`lucas_prescott_investment`: the operator defined by the right side of {eq}`lps_bellman` is monotone and satisfies a discounting condition, so Blackwell's theorem {cite}`Blackwell1965` applies.

That $v$ is nonincreasing in $y$ is worth a word: a worker is worse off the more crowded his market, because a larger workforce means a lower marginal product and a lower wage.

## Employment, wages, and two thresholds

With $v$ in hand, the rest of a market's behavior follows from two threshold rules.

### How many work

Let $\bar n(s, \lambda)$ be the employment at which the marginal worker is just indifferent between staying and searching:

```{math}
:label: lps_nbar
R(s, \bar n) + \beta \int v(s', \bar n, \lambda) f(s', s)\, ds' = \lambda .
```

The left side is what a worker gets by staying when employment and next period's workforce are both $\bar n$: this period's wage plus the discounted continuation.

It is decreasing in $\bar n$, so {eq}`lps_nbar` has a unique solution.

Employment is then

```{math}
:label: lps_employment
n(s, y, \lambda) = \min\{\bar n(s,\lambda),\; y\},
\qquad w = R(s, n) ,
```

and unemployment in this market is $y - n$, which is positive exactly when the workforce exceeds $\bar n$.

### How many arrive

A market attracts searchers only when it employs its whole workforce and the continuation value still exceeds $\lambda$.

Let $\bar y(s,\lambda)$ be the workforce at which that stops being true:

```{math}
:label: lps_ybar
\beta \int v(s', \bar y, \lambda) f(s', s)\, ds' = \lambda .
```

Arrivals fill a market up to $\bar y(s, \lambda)$ and no further.

Note that $\bar y < \bar n$: since $R > 0$, a market that is just worth entering for a searcher is strictly inside the range where everyone works.

### Three regions, one transition law

Together, {eq}`lps_employment` and {eq}`lps_ybar` partition the $(s,y)$ plane into the three regions of the paper's Figure 3, and they give the law of motion for the workforce a very simple form:

```{math}
:label: lps_transition
y' = \min\big\{\max\{y,\ \bar y(s,\lambda)\},\ \bar n(s,\lambda)\big\} .
```

A market that is too crowded sheds workers down to $\bar n(s,\lambda)$; a market that is too empty is filled up to $\bar y(s,\lambda)$; in between, nobody moves.

```{note}
The paper lets arrivals be random: a market short of workers receives $a(s,y,\lambda)\,x$ of them, where $x$ is a positive random variable with mean one, so that search "eliminates rents on average" rather than exactly.

Like the numerical examples in the paper's appendix, we set $x \equiv 1$, which gives {eq}`lps_transition`.
```

## A computable version

We follow the appendix of {cite:t}`LucasPrescott1974`: a finite number of demand states, an integer workforce, and value function iteration on {eq}`lps_bellman`.

For the demand process we use the example in the paper's own footnote: $\log s$ follows an AR(1), discretized with the method of Tauchen.

For $R$ we take a marginal revenue schedule that falls linearly in employment and reaches zero at a capacity $N$,

$$
R(s, n) = s\,\big(1 - n/N\big)^{+} ,
$$

which is the shape drawn in the paper's own Figure 5.

The capacity matters for more than convenience.

Lucas and Prescott require that employment stay bounded as demand grows without bound, and a schedule that never reaches zero violates that: markets would absorb unlimited numbers of workers in good states, and the answers would depend on where we truncated the grid rather than on the economics.

We will see below that this assumption is not a technicality.

```{code-cell} ipython3
Model = namedtuple('Model', ['P', 's', 'y', 'R', 'β'])

def create_model(n_s=7, ρ=0.9, sd=0.26, β=0.9, N=200, y_max=260):
    """An island economy.

    ρ and sd are the persistence and the stationary standard deviation of
    log demand, so that changing ρ alone does not change how variable
    demand is.  N is a market's capacity: marginal revenue reaches zero there.
    """
    mc = qe.markov.tauchen(n_s, ρ, sd*np.sqrt(1 - ρ**2))
    s = np.exp(mc.state_values)
    y = np.arange(y_max + 1)
    R = np.maximum(s[:, None]*(1 - y[None, :]/N), 0.0)
    R[:, 0] = 0.0
    return Model(P=mc.P, s=s, y=y, R=R, β=β)

model = create_model()
print("demand states:", model.s.round(3))
```

The Bellman operator is {eq}`lps_bellman` with the integral replaced by a sum.

```{code-cell} ipython3
def worker_value(model, λ, tol=1e-11, max_iter=10_000):
    "Solve the worker's functional equation by successive approximations."
    P, R, β = model.P, model.R, model.β
    v = np.full(R.shape, λ)
    for i in range(max_iter):
        v_new = np.maximum(λ, R + np.minimum(λ, β*(P @ v)))
        v_new[:, 0] = λ                     # an empty market holds no workers
        if np.max(np.abs(v_new - v)) < tol:
            return v_new, i
        v = v_new
    return v, max_iter

v, iters = worker_value(model, 7.1)
print(f"converged in {iters} iterations")
```

The two thresholds are the largest employment, and the largest workforce, at which the relevant value still covers $\lambda$.

```{code-cell} ipython3
def thresholds(model, v, λ):
    "The employment threshold n̄(s) of {eq}`lps_nbar` and the arrival threshold ȳ(s)."
    P, R, β = model.P, model.R, model.β
    EV = β*(P @ v)              # the continuation value at an unchanged workforce
    H = R + EV                  # the value of staying on as the marginal worker
    last = lambda A: np.array([np.max(np.where(A[i] >= λ)[0]) if (A[i, 1:] >= λ).any()
                               else 0 for i in range(len(model.s))])
    n_bar = last(H)
    y_bar = np.minimum(last(EV), n_bar)
    return n_bar, y_bar

n_bar, y_bar = thresholds(model, v, 7.1)
print("n̄(s) =", n_bar)
print("ȳ(s) =", y_bar)
```

Employment is capped by $\bar n$, arrivals fill up to $\bar y$, and {eq}`lps_transition` moves the workforce.

```{code-cell} ipython3
def stationary_distribution(model, n_bar, y_bar, tol=1e-14, max_iter=100_000):
    "The invariant distribution of the Markov chain on (s, y)."
    P, y = model.P, model.y
    n_s, n_y = len(model.s), len(y)
    ψ = np.ones((n_s, n_y))/(n_s*n_y)
    y_next = np.clip(y[None, :], y_bar[:, None], n_bar[:, None])
    for _ in range(max_iter):
        moved = np.array([np.bincount(y_next[i], weights=ψ[i], minlength=n_y)
                          for i in range(n_s)])
        ψ_new = P.T @ moved
        if np.max(np.abs(ψ_new - ψ)) < tol:
            return ψ_new
        ψ = ψ_new
    return ψ

def market_outcomes(model, λ):
    "Everything about a market at a given value of search."
    v, _ = worker_value(model, λ)
    n_bar, y_bar = thresholds(model, v, λ)
    ψ = stationary_distribution(model, n_bar, y_bar)
    y = model.y
    employment = np.minimum(y[None, :], n_bar[:, None])
    unemployment = y[None, :] - employment
    wage = np.take_along_axis(model.R, np.maximum(employment, 1), axis=1)
    return dict(v=v, n_bar=n_bar, y_bar=y_bar, ψ=ψ,
                workforce=(ψ*y).sum(),
                employment=(ψ*employment).sum(),
                unemployment=(ψ*unemployment).sum(),
                wage=(ψ*wage).sum())

out = market_outcomes(model, 7.1)
print(f"workforce {out['workforce']:.2f}, employment {out['employment']:.2f}, "
      f"unemployment {out['unemployment']:.2f}")
```

### What a market looks like

The thresholds describe the whole of a market's behavior, so it is worth seeing them.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The two thresholds and the wage schedule
    name: fig-lps-thresholds
---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(model.s, out['n_bar'], 'o-', lw=2, label=r'$\bar n(s)$: employment')
axes[0].plot(model.s, out['y_bar'], 's--', lw=2, label=r'$\bar y(s)$: arrivals fill to')
axes[0].set_xlabel('demand state $s$')
axes[0].set_ylabel('workers')
axes[0].legend()

wage_at_nbar = np.array([model.R[i, max(out['n_bar'][i], 1)] for i in range(len(model.s))])
axes[1].plot(out['n_bar'], wage_at_nbar, 'o-', lw=2)
axes[1].set_xlabel(r'employment $\bar n(s)$')
axes[1].set_ylabel('wage')
axes[1].set_title('the locus traced out as demand shifts')
plt.tight_layout()
plt.show()
```

The right panel is the curve that Lucas and Prescott call $SS$, the pairs of employment and wages that a market would trace out as demand shifted if its workforce never constrained it.

It is **not flat**.

A flat $SS$ would mean that workers had a single reservation wage, accepting work above it and searching below it, which is what the partial-equilibrium search models of {doc}`mccall_model` assume.

Here they do not, and the reason is instructive: a change in demand moves the current wage, but it also carries information about future wages, and it changes the workforce that the market will carry into the future.

Those effects pull in opposite directions, so the locus can slope either way.

### The workforce distribution and a sample path

Because arrivals are deterministic, the workforce moves between a small number of values, each one a threshold of some demand state.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Where the workforce settles, and how a market moves
    name: fig-lps-distribution
---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

marginal = out['ψ'].sum(axis=0)
support = marginal > 1e-10
axes[0].stem(model.y[support], marginal[support], basefmt=' ')
axes[0].set_xlabel('workforce $y$')
axes[0].set_ylabel('probability')
axes[0].set_title('invariant distribution of the workforce')

mc = qe.MarkovChain(model.P)
idx = mc.simulate_indices(60, init=3, random_state=0)
path_y = np.empty(60, dtype=int)
path_y[0] = out['n_bar'][idx[0]]
for t in range(59):
    path_y[t+1] = np.clip(path_y[t], out['y_bar'][idx[t]], out['n_bar'][idx[t]])
path_n = np.minimum(path_y, out['n_bar'][idx])

axes[1].plot(path_y, lw=2, label='workforce')
axes[1].plot(path_n, lw=2, ls='--', label='employment')
axes[1].fill_between(range(60), path_n, path_y, alpha=0.3, label='unemployed')
axes[1].set_xlabel('period')
axes[1].legend()
axes[1].set_title('one market over sixty periods')
plt.tight_layout()
plt.show()
```

The right panel shows the mechanism of the model in one picture.

A market is at rest whenever demand does not move.

When demand falls, the gap between the two lines opens for exactly one period: that period's leavers are the unemployed, and by the next period they are somewhere else.

(lps_economy_wide)=
## Economy-wide equilibrium

So far $\lambda$ has been a parameter.

Now close the model.

The economy has a fixed workforce, $\mu$ per market.

For any $\lambda$, the calculation above delivers the workforce that a market holds on average, and summing across markets gives the economy's **demand for workforce**,

```{math}
:label: lps_demand
E(y; \lambda) = \int y \, \Psi(ds, dy; \lambda) .
```

A higher $\lambda$ makes every market shed workers sooner and fill up less, so $E(y;\lambda)$ is decreasing.

Equilibrium is the $\lambda^*$ at which markets want to hold exactly the workers the economy has:

```{math}
:label: lps_equilibrium
E(y; \lambda^*) = \mu .
```

Because the schedule is decreasing, bisection finds $\lambda^*$.

```{code-cell} ipython3
def workforce_demand(model, λ):
    "The average workforce per market at the value of search λ."
    return market_outcomes(model, λ)['workforce']

def equilibrium_λ(model, μ, lo=0.3, hi=40.0, tol=1e-4):
    "Bisect on the decreasing demand schedule E(y; λ) = μ."
    while hi - lo > tol:
        mid = 0.5*(lo + hi)
        if workforce_demand(model, mid) > μ:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo + hi)

μ = 50                        # workers per market, against a capacity of 200
λ_star = equilibrium_λ(model, μ)
eq = market_outcomes(model, λ_star)

print(f"equilibrium value of search  λ* = {λ_star:.3f}")
print(f"workforce per market            = {eq['workforce']:.2f}   (target {μ})")
print(f"employment per market           = {eq['employment']:.2f}")
print(f"unemployment per market         = {eq['unemployment']:.2f}")
print(f"unemployment rate               = {100*eq['unemployment']/eq['workforce']:.2f}%")
print(f"separation rate                 = "
      f"{100*eq['unemployment']/eq['employment']:.2f}% of the employed per period")
print(f"mean wage                       = {eq['wage']:.3f}")
```

Lucas and Prescott draw this as a demand curve meeting a vertical supply curve, which is exactly what it is.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The market for workers, and unemployment along the demand curve
    name: fig-lps-equilibrium
---
grid = np.linspace(0.6, 15.0, 45)
demand = np.array([market_outcomes(model, λ) for λ in grid])
E_y = np.array([d['workforce'] for d in demand])
E_u = np.array([d['unemployment'] for d in demand])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(E_y, grid, lw=2, label=r'demand $E(y;\lambda)$')
axes[0].axvline(μ, color='C1', lw=2, ls='--', label=r'supply $\mu$')
axes[0].plot(eq['workforce'], λ_star, 'ko', ms=8)
axes[0].set_xlabel('workforce per market')
axes[0].set_ylabel(r'value of search $\lambda$')
axes[0].legend()

axes[1].plot(E_y, E_u, lw=2)
axes[1].plot(eq['workforce'], eq['unemployment'], 'ko', ms=8)
axes[1].set_xlabel('workforce per market')
axes[1].set_ylabel('unemployment per market')
plt.tight_layout()
plt.show()
```

The left panel is the paper's Figure 4 and the right panel is its Figure 5.

The second is the more interesting of the two, because unemployment is **not monotone** in the size of the workforce, a feature that Lucas and Prescott report from their own examples.

Where workers are scarce, $\lambda$ is high, every market employs all it has, and almost nobody is ever shed.

Where workers are abundant, $\lambda$ is low, every market is pressed against its capacity, and the two thresholds are squeezed together there, so even a large shift in demand moves few workers.

Unemployment is largest in between, and this is where the bounded capacity earns its keep: with a marginal revenue schedule that never reached zero, the thresholds would keep rising as $\lambda$ fell, unemployment would rise without bound, and the right panel would have no peak at all.

### The wage distribution

We can now answer the question the Overview began with.

The search literature hands its worker a distribution of wage offers and asks what he should do with it.

Here that distribution is an output: every market pays the marginal product of labor in it, and what a market pays depends on the demand it drew and on how many workers chose to be there.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Wages across identical workers, at the equilibrium
    name: fig-lps-wages
---
employed = np.minimum(model.y[None, :], eq['n_bar'][:, None])
wage = np.take_along_axis(model.R, np.maximum(employed, 1), axis=1)
weight = eq['ψ']*employed                  # weight a market by the workers in it
weight = weight/weight.sum()

w, p = wage.ravel(), weight.ravel()
order = np.argsort(w)
w, p = w[order], p[order]
cdf = np.cumsum(p)
quantile = lambda a: w[np.searchsorted(cdf, a)]
mean = (w*p).sum()

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.hist(w, bins=30, weights=p, color='C0', alpha=0.85)
ax.axvline(mean, color='k', lw=1.5, label=f'mean {mean:.3f}')
ax.axvline(λ_star*(1 - model.β), color='C1', lw=1.5, ls='--',
           label=f'flow value of search {λ_star*(1-model.β):.3f}')
ax.set_xlabel('wage')
ax.set_ylabel('share of workers')
ax.set_xlim(0.5, 1.45)            # a thin tail runs past this; see the print below
ax.legend()
plt.tight_layout()
plt.show()

print(f"mean {mean:.3f}, "
      f"p10 {quantile(0.1):.3f}, median {quantile(0.5):.3f}, p90 {quantile(0.9):.3f}")
print(f"p90/p10 = {quantile(0.9)/quantile(0.1):.2f},  "
      f"highest wage paid {w[p > 1e-12].max():.3f}")
```

Workers in this economy are identical.

They have the same skills, the same preferences and the same information, and they are free to leave at any time.

They are nevertheless paid differently, by half as much again at the ninetieth percentile as at the tenth, and the dispersion is not a friction or a mistake: it is what equalizes the *value* of being in one market or another when demand has moved markets apart and it takes a period to cross between them.

The dashed line is the flow value of search, $(1-\beta)\lambda^*$, which is what a searcher's position is worth per period.

It sits just below the mean wage, as it must: a worker gives up a period of earnings to move, so the position he moves to has to be worth more than the one he left.

```{note}
Notice what this model does **not** contain.

There is no matching function, no bargaining, and no wage rigidity.

Unemployment is here because moving takes a period, and in equilibrium somebody is always moving.

The lecture {doc}`lake_model` studies the flows between employment and unemployment that this model generates, taking the rates as given; here the rates are what the equilibrium determines.
```

## Relation to other lectures

It is worth being explicit about what this model supplies that the search lectures elsewhere in this series take as given.

{doc}`mccall_model` and its sequels study a worker who draws from a known wage offer distribution and chooses a reservation wage.

That worker's problem is a piece of this one: here too a worker compares what a market offers him with the value of looking elsewhere.

What is added is everything on the other side of the offer.

The wage a worker is offered is the marginal product of labor in a market whose workforce was determined by where other workers decided to go, and the value of looking elsewhere, $\lambda$, is the number that clears the economy-wide market for workers.

Two consequences are worth noting.

The first is that there is no reservation wage.

At the equilibrium computed above, workers stay on in markets paying as little as $0.456$, while elsewhere workers walk out of markets paying $0.674$.

No number between the two separates the markets a worker accepts from those he leaves, because what he compares is not wages but values: the market paying $0.674$ is overcrowded today, so its wage is about to fall and its workforce will take years to clear, while the market paying $0.456$ is depressed today and expects to recover.

The second is about the flow rates between employment and unemployment, and it is worth setting out as a ladder.

{doc}`lake_model` takes both rates as parameters: a job finding rate of $0.283$ and a separation rate of $0.013$ go in, and the unemployment rate comes out.

{doc}`endogenous_lake` climbs one rung, making the job finding rate the outcome of a McCall problem, though still against a wage distribution that is given.

This lecture is the next rung: both rates are equilibrium objects.

The separation rate printed above, around $2.7$ percent of the employed per period, is nothing but the frequency with which demand shifts carry markets below their employment thresholds.

```{warning}
The job finding rate in this model is **one**.

A worker who leaves is placed in some market at the start of the next period, always, so every spell of unemployment lasts exactly one period and the model has nothing to say about unemployment *duration*.

That is a direct consequence of directed search, and it is the first thing {cite:t}`AlvarezVeracierto1999` give up: when searchers arrive uniformly, a worker can land in a market that is itself shedding workers and have to move again.

Duration becomes an equilibrium object there, and averages $2.4$ periods in their calibration.
```

The connection to {doc}`lucas_prescott_investment` is methodological, and close.

In the 1971 paper a competitive industry's capital stock is a Markov process whose invariant distribution describes the long run, and the equilibrium is computed by solving a fixed point problem in a function.

Here a market's workforce is a Markov process whose invariant distribution describes the long run, and the equilibrium is computed by solving a fixed point problem in a function and then one more, in the single number $\lambda$.

The same authors wrote both papers within three years, and the second is the first one's method turned on a question about labor.

### What the island model became

{cite:t}`AlvarezVeracierto1999` took this model and made it a tool for quantitative policy analysis, in their words by "modifying Lucas and Prescott's (1974) island model to incorporate undirected search and out-of-the-labor-force participation".

Four things change.

Search becomes **undirected**: searchers "arrive uniformly across all islands", so an island's workforce next period is its own employment plus the same inflow every other island gets, whether it wants the workers or not.

The nonemployed gain a second option, home production, so that labor force participation is a margin and the model has three states rather than two.

Islands produce with a decreasing returns technology subject to persistent productivity shocks rather than facing a shifting demand curve.

And agents trade a complete set of state-contingent securities, which the authors justify by their own earlier finding that the insurance role of the policies they study is quantitatively small.

With that machinery they evaluate minimum wages, two models of unions, firing taxes and unemployment insurance, at a model period of six weeks.

```{note}
The first of those changes is worth dwelling on, because it tells us what directed search is doing here.

In this lecture $\lambda$ is a **price**: raise it and every market sheds workers sooner and fills up less, so the workforce that markets want to hold falls, and {eq}`lps_equilibrium` pins $\lambda$ down exactly as a market-clearing condition pins down any other price.

Under undirected search that mechanism is gone.

The transition $y' = n(s,y) + U$ conserves the average workforce once $U$ is the average outflow, so the aggregate is a conserved quantity rather than something a price determines, and the invariant distribution is one of a family.

{ref}`lps_ex4` asks you to verify this.

It is also why the participation margin is not an optional embellishment in {cite:t}`AlvarezVeracierto1999`: with home production the value of not searching pins down a level that undirected search by itself leaves free.
```

## Exercises

```{exercise}
:label: lps_ex1

{cite:t}`LucasPrescott1974` report two comparative statics from their own numerical examples:

> *Overall, we found for low and high persistence in demand that unemployment rates were low. In the former case, there was little gained by reallocating workers, while in the latter reallocation occurred infrequently. As expected, the greater the variability of demand, holding the degree of persistence fixed, the greater the level of unemployment.*

Check both in this calibration.

1. Compute the equilibrium unemployment rate for persistence $\rho \in \{0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99\}$, holding the **stationary** standard deviation of $\log s$ fixed at $0.26$ and $\mu$ at $50$. Is the relationship monotone?
1. Compute it for stationary standard deviations $\{0.10, 0.18, 0.26, 0.34, 0.45\}$, holding $\rho = 0.9$.

Explain the shape you find in part 1.

The two parameters you are varying are the same two that {doc}`mccall_persist_trans` varies, there as the persistence and volatility of a worker's wage offers rather than of his market's demand.

The difference is what the experiment delivers: a reservation wage there, an unemployment rate here.
```

```{solution-start} lps_ex1
:class: dropdown
```

Holding the stationary standard deviation fixed as $\rho$ varies is what the argument `sd` of `create_model` already does, so each experiment is one call.

```{code-cell} ipython3
print(f"{'ρ':>8}{'λ*':>9}{'workforce':>12}{'u rate':>10}")
u_by_ρ = []
for ρ in (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99):
    m_ρ = create_model(ρ=ρ)
    λ_ρ = equilibrium_λ(m_ρ, μ)
    r = market_outcomes(m_ρ, λ_ρ)
    u_by_ρ.append(100*r['unemployment']/r['workforce'])
    print(f"{ρ:>8}{λ_ρ:>9.3f}{r['workforce']:>12.1f}{u_by_ρ[-1]:>9.2f}%")
```

The second experiment varies how far demand moves, holding how long it stays.

```{code-cell} ipython3
print(f"{'sd':>8}{'λ*':>9}{'workforce':>12}{'u rate':>10}")
for sd in (0.10, 0.18, 0.26, 0.34, 0.45):
    m_sd = create_model(sd=sd)
    λ_sd = equilibrium_λ(m_sd, μ)
    r = market_outcomes(m_sd, λ_sd)
    print(f"{sd:>8}{λ_sd:>9.3f}{r['workforce']:>12.1f}"
          f"{100*r['unemployment']/r['workforce']:>9.2f}%")
```

Variability does what one expects: demand that moves further moves more workers, and the unemployment rate rises steadily with it.

Persistence does not.

The unemployment rate is near zero at both ends and reaches its maximum in between, which is the shape Lucas and Prescott describe.

The two ends are near zero for opposite reasons.

When demand is nearly independent across periods, a market that is unlucky today expects to be average tomorrow, so its workers see little point in moving: the thresholds $\bar n$ and $\bar y$ are close together and almost nobody is ever shed.

When demand is nearly permanent, the thresholds are far apart, but a market almost never crosses from one to the other, so the reallocation that would make workers move hardly ever happens.

Unemployment needs demand shifts that are *both* worth responding to and frequent, which is a combination that only intermediate persistence delivers.

```{solution-end}
```

```{exercise}
:label: lps_ex2

In their concluding section, Lucas and Prescott consider a lump-sum cost $c$ of leaving a market to search, so that the right side of {eq}`lps_nbar` becomes $\lambda - c$ rather than $\lambda$: a worker who leaves collects the value of search net of what it costs him to get there.

The arrival condition {eq}`lps_ybar` is unchanged, since it prices the market for workers who are already searching.

They report that the cost lowers unemployment, and remark that this

> *shows that lower average unemployment is not, in general, associated with higher welfare for workers.*

Modify the code to allow a cost $c$, and for $c \in \{0, 0.2, 0.5, 1, 2\}$ report the equilibrium $\lambda^*$, the unemployment rate, the expected value of a worker under the invariant distribution, $\sum_{s,y}\psi(s,y)v(s,y)$, and output per market,

$$
\sum_{s,y} \psi(s,y) \int_0^{n(s,y)} R(s,m)\, dm ,
$$

which is the area under the marginal revenue schedule up to employment.

Does lower unemployment mean better off?
```

```{solution-start} lps_ex2
:class: dropdown
```

Only two lines change: the outside option inside the max of {eq}`lps_bellman`, and the threshold that defines $\bar n$.

```{code-cell} ipython3
def worker_value_cost(model, λ, c, tol=1e-11, max_iter=10_000):
    "The worker's problem when leaving costs c."
    P, R, β = model.P, model.R, model.β
    v = np.full(R.shape, λ - c)
    for _ in range(max_iter):
        v_new = np.maximum(λ - c, R + np.minimum(λ, β*(P @ v)))
        v_new[:, 0] = λ - c
        if np.max(np.abs(v_new - v)) < tol:
            return v_new
        v = v_new
    return v

def outcomes_cost(model, λ, c):
    v = worker_value_cost(model, λ, c)
    P, R, β, y = model.P, model.R, model.β, model.y
    EV = β*(P @ v)
    H = R + EV
    last = lambda A, bar: np.array([np.max(np.where(A[i] >= bar)[0])
                                    if (A[i, 1:] >= bar).any() else 0
                                    for i in range(len(model.s))])
    n_bar = last(H, λ - c)                 # staying is worth λ - c to the leaver
    y_bar = np.minimum(last(EV, λ), n_bar)  # arrivals are priced at λ
    ψ = stationary_distribution(model, n_bar, y_bar)
    employment = np.minimum(y[None, :], n_bar[:, None])
    output = np.take_along_axis(np.cumsum(R, axis=1), employment, axis=1)
    return dict(workforce=(ψ*y).sum(),
                unemployment=(ψ*(y[None, :] - employment)).sum(),
                welfare=(ψ*v).sum(), output=(ψ*output).sum())

def equilibrium_λ_cost(model, μ, c, lo=0.3, hi=40.0, tol=1e-4):
    while hi - lo > tol:
        mid = 0.5*(lo + hi)
        if outcomes_cost(model, mid, c)['workforce'] > μ:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo + hi)
```

Now sweep the cost.

```{code-cell} ipython3
print(f"{'c':>6}{'λ*':>9}{'λ*-c':>9}{'u rate':>9}{'E[v]':>9}{'output':>9}")
for c in (0.0, 0.2, 0.5, 1.0, 2.0):
    λ_c = equilibrium_λ_cost(model, μ, c)
    r = outcomes_cost(model, λ_c, c)
    print(f"{c:>6}{λ_c:>9.3f}{λ_c - c:>9.3f}"
          f"{100*r['unemployment']/r['workforce']:>8.2f}%{r['welfare']:>9.3f}"
          f"{r['output']:>9.3f}")
```

A cost of two cuts measured unemployment by more than four fifths, from $2.7$ percent to $0.5$ percent.

It also cuts output, by about three percent.

That is the paper's point, and it is worth stating plainly: the unemployment that the cost eliminated was **productive**.

Those workers were not idle in any useful sense; they were in transit from markets where their marginal product had fallen to markets where it was high, and the cost keeps them where they are.

The rest of the table shows why "welfare" needs saying carefully.

The value of the option a worker holds, $\lambda^* - c$, falls at every step: the outside option is worth less, which is what makes workers stay.

Yet the expected present value of a worker's own earnings **rises**.

A worker's income is the wage when he is employed and nothing when he is searching, and the cost buys him fewer periods of nothing.

What he gains that way, the economy loses twice over in output, because the match between workers and markets is worse.

So unemployment is not a welfare index, which is the lesson Lucas and Prescott draw, and in this calibration even the two welfare measures move in opposite directions.

It would be fair to suspect that so stark a result is an artifact of a model this stripped down.

It is not.

{cite:t}`AlvarezVeracierto1999` run the same experiment as a firing tax inside a calibrated version of this economy, and find that a tax worth between three and twelve months of average wages takes the unemployment rate **down** from $5.3$ to $3.7$ percent, pushes the average duration of unemployment **up** from $2.4$ to $5.1$ periods, and cuts employment to $86$ percent of its laissez-faire level.

```{solution-end}
```

```{exercise}
:label: lps_ex3

The numerical examples in the appendix of {cite:t}`LucasPrescott1974` use just two demand states, with

$$
P = \begin{bmatrix} 0.9 & 0.1 \\ 0.1 & 0.9\end{bmatrix} ,
\qquad s \in \{1, 2\},
\qquad \beta = 0.9 .
$$

Build that economy, keeping $R(s,n) = s(1+n/n_0)^{-1/\eta}$ from the lecture.

1. For $\lambda$ on a grid, plot the workforce demand and the unemployment against each other, as in the paper's Figure 5.
1. Show that the invariant distribution puts all of its mass on **two** values of the workforce, and say which two.
1. Use that fact to derive a formula for average unemployment per market in terms of $\bar n$, $\bar y$ and the persistence $\rho$, and check it against the computed value.
```

```{solution-start} lps_ex3
:class: dropdown
```

The two-state chain needs its own constructor, but everything else is reused.

```{code-cell} ipython3
def two_state_model(ρ=0.9, s_lo=1.0, s_hi=2.0, β=0.9, N=200, y_max=260):
    P = np.array([[ρ, 1-ρ], [1-ρ, ρ]])
    s = np.array([s_lo, s_hi])
    y = np.arange(y_max + 1)
    R = np.maximum(s[:, None]*(1 - y[None, :]/N), 0.0)
    R[:, 0] = 0.0
    return Model(P=P, s=s, y=y, R=R, β=β)

two = two_state_model()
grid2 = np.linspace(0.4, 14.0, 45)
res2 = [market_outcomes(two, λ) for λ in grid2]
Ey2 = np.array([r['workforce'] for r in res2])
Eu2 = np.array([r['unemployment'] for r in res2])
```

Plotting one against the other gives the paper's Figure 5.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(Ey2, Eu2, lw=2)
ax.set_xlabel('workforce demand per market')
ax.set_ylabel('average unemployment per market')
plt.tight_layout()
plt.show()
```

The hump is the one in the paper's Figure 5.

For part 2, look at the support of the invariant distribution.

```{code-cell} ipython3
r2 = market_outcomes(two, 6.0)
support = np.where(r2['ψ'].sum(axis=0) > 1e-12)[0]
print("workforce values with positive probability:", support)
print("n̄(s) =", r2['n_bar'], "   ȳ(s) =", r2['y_bar'])
```

The two values are $\bar n(s_{\text{low}})$ and $\bar y(s_{\text{high}})$.

To see why, follow {eq}`lps_transition`.

In the low state the workforce is clipped into $[\bar y(s_{\text{low}}), \bar n(s_{\text{low}})]$ and in the high state into $[\bar y(s_{\text{high}}), \bar n(s_{\text{high}})]$, and with these parameters the second interval lies entirely above the first.

So one period in the low state carries any workforce to $\bar n(s_{\text{low}})$, one period in the high state carries it to $\bar y(s_{\text{high}})$, and the market shuttles between the two.

For part 3, unemployment is positive only in the first period of a downturn: the market enters with $\bar y(s_{\text{high}})$ workers, demand has just fallen, and employment is $\bar n(s_{\text{low}})$.

That event is "the state was high last period and is low now", which has stationary probability $\tfrac12 (1-\rho)$, so

$$
E[u] = \tfrac12 (1-\rho)\,
\big[\bar y(s_{\text{high}}) - \bar n(s_{\text{low}})\big] .
$$

```{code-cell} ipython3
predicted = 0.5*(1 - 0.9)*(r2['y_bar'][1] - r2['n_bar'][0])
print(f"formula:  {predicted:.4f}")
print(f"computed: {r2['unemployment']:.4f}")
```

The formula also explains the hump.

Average unemployment is the gap between two thresholds, multiplied by the chance of crossing it.

As $\lambda$ falls, both thresholds rise, but not in step, and the gap between them widens and then narrows again.

```{solution-end}
```

```{exercise}
:label: lps_ex4

This exercise is analytical; it asks what the model's directed search is doing.

{cite:t}`AlvarezVeracierto1999` replace it with **undirected** search: a searcher cannot choose where to go, and every market receives the same inflow $U$, which is the economy's unemployment per market.

1. Show that under undirected search the $\min$ disappears from {eq}`lps_bellman`, and say which economic condition it was enforcing.
1. Write down the transition law for a market's workforce, and show that the average workforce across markets is unchanged from one period to the next, so that it is a conserved quantity rather than one determined by $\lambda$.
1. The equilibrium condition {eq}`lps_equilibrium` is therefore no longer available to pin down the economy. What do Alvarez and Veracierto add in its place?
```

```{solution-start} lps_ex4
:class: dropdown
```

For part 1, the $\min$ in {eq}`lps_bellman` is not a choice that anyone makes.

It is the restriction that the continuation value in a market cannot exceed $\lambda$, which competition among searchers imposes: if some market promised more, searchers would go there until it no longer did.

Take away their ability to choose where to go and nothing enforces it.

A worker who stays simply collects the wage and the continuation value his own market happens to deliver,

$$
v(s, y) = \max\left\{\lambda,\; R(s, y)
 + \beta \int v\big(s', y'(s,y)\big) f(s', s)\, ds'\right\} ,
$$

and the continuation can be worth more or less than $\lambda$, because nobody can act on the difference.

For part 2, employment is still $n(s,y) = \min\{\bar n(s), y\}$, those not employed leave, and the inflow is the same everywhere, so

$$
y' = n(s, y) + U .
$$

Average across markets, and use the fact that $U$ is by definition the average outflow, $U = E[\,y - n(s,y)\,]$:

$$
E[y'] = E[n(s,y)] + U = \big(E[y] - U\big) + U = E[y] .
$$

The average workforce is therefore the same in every period: whatever the economy starts with, it keeps.

For part 3, under directed search $\lambda$ is a price.

Raising it makes every market shed workers sooner and fill up less, so $E(y;\lambda)$ falls, and {eq}`lps_equilibrium` determines $\lambda$ the way market clearing determines any price.

Under undirected search the level of the workforce is conserved rather than cleared, so that equation has nothing left to do, and the invariant distribution is one of a whole family, one for each level.

What Alvarez and Veracierto add is the **participation margin**: the nonemployed choose between searching and home production.

The value of home production is then the price that pins down how many people are in the labor force at all, and with it the level that undirected search on its own leaves free.

This is why their two modifications of Lucas and Prescott come as a pair.

```{solution-end}
```
