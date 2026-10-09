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

(ge_arrow)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Competitive Equilibria with Arrow Securities

```{index} single: Arrow Securities; competitive equilibrium
```

```{contents} Contents
:depth: 2
```

## Overview

This lecture presents Python code for experimenting with competitive equilibria of an infinite-horizon pure exchange economy with

* heterogeneous agents,
* endowments of a single consumption good that are person-specific functions of a common Markov state,
* complete markets in one-period Arrow state-contingent securities,
* discounted expected utility preferences of a kind often used in macroeconomics and finance,
* identical preferences across agents, with a common discount factor and a constant relative risk aversion (CRRA) one-period utility function, and
* common beliefs among agents.

Differences in their endowments make individuals want to reallocate consumption goods across time and Markov states.

Identical CRRA preferences imply that equilibrium consumption shares are constant, so we can compute equilibrium prices from the aggregate endowment *before* we compute the equilibrium distribution of wealth.

We impose restrictions that allow us to **Bellmanize** competitive equilibrium prices and quantities.

We use Bellman equations to describe

* asset prices,
* continuation wealth levels for each person, and
* state-by-state natural debt limits for each person.

In the course of presenting the model we shall encounter these important ideas:

* a **resolvent operator** widely used in this class of models,
* the absence of **borrowing limits** in finite-horizon economies,
* state-by-state **borrowing limits** required in infinite-horizon economies,
* a counterpart of the **law of iterated expectations** known as a **law of iterated values**, and
* a **state variable degeneracy** that prevails within a competitive equilibrium and that opens the way to various appearances of resolvent operators.

The lecture implements a Python version of the model presented in section 9.3.3 of {cite}`Ljungqvist2012`.

The material is older than it might appear.

Both the sequential trading arrangement studied here and the time $0$ arrangement with a complete set of history-contingent claims, as well as the proof that the two support the same allocation, appear in a paper that Kenneth Arrow read in Paris in May 1952 and published in French in 1953 {cite}`arrow1964`.

Arrow's sequential arrangement deserves a second name as well.

Because a household that trades securities today must act on a forecast of the prices at which goods will trade tomorrow, and because the equivalence of the two arrangements holds only when markets confirm that forecast, Arrow's sequential equilibrium is a **rational expectations** equilibrium in the sense that {cite:t}`muth1961` gave the term almost a decade later.

We develop this reading, which we learned from {cite:t}`kihlstrom2019`, in the section on {ref}`sec-rational-expectations` below.

Readers will find it helpful to know the finite-state Markov asset pricing formulas of {doc}`markov_asset` and the Markov chain concepts of {doc}`finite_markov`.

Let's start with some imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

np.set_printoptions(suppress=True)
```

## The setting

### Preferences and endowments

In each period $t \geq 0$, a stochastic event $s_t \in \mathbf{S}$ is realized.

Let the history of events up until time $t$ be denoted $s^t = [s_0, s_1, \ldots, s_{t-1}, s_t]$.

The unconditional probability of observing a particular sequence of events $s^t$ is given by a probability measure $\pi_t(s^t)$.

For $t > \tau$, we write the probability of observing $s^t$ conditional on the realization of $s^\tau$ as $\pi_t(s^t \mid s^\tau)$.

We assume that trading occurs after observing $s_0$, which we capture by setting $\pi_0(s_0) = 1$ for the initially given value of $s_0$.

In this lecture we follow much of macroeconomics and econometrics and assume that $\pi_t(s^t)$ is induced by a Markov process.

There are $K$ consumers named $k = 1, \ldots, K$.

Consumer $k$ owns a stochastic endowment of one good $y_t^k(s^t)$ that depends on the history $s^t$.

The history $s^t$ is publicly observable.

Consumer $k$ purchases a history-dependent consumption plan $c^k = \{c_t^k(s^t)\}_{t=0}^\infty$.

All consumers order consumption plans by

$$
U(c^k) = \sum_{t=0}^\infty \sum_{s^t} \beta^t u[c_t^k(s^t)] \pi_t(s^t),
$$

where $0 < \beta < 1$.

The right side equals $E_0 \sum_{t=0}^\infty \beta^t u(c_t^k)$, where $E_0$ is the mathematical expectation operator conditioned on $s_0$.

Here $u(c)$ is an increasing, twice continuously differentiable, strictly concave function of consumption $c \geq 0$ of one good.

The utility function satisfies the Inada condition

$$
\lim_{c \downarrow 0} u'(c) = +\infty .
$$

This condition implies that each agent chooses strictly positive consumption for every date-history pair $(t, s^t)$ whenever the present value of its endowment is positive.

Those interior solutions allow us to confine our analysis to Euler equations that hold with equality, and they guarantee that **natural debt limits** don't bind in economies like ours with sequential trading of Arrow securities.

We adopt the assumption, routinely employed in much of macroeconomics, that consumers share probabilities $\pi_t(s^t)$ for all $t$ and $s^t$.

A **feasible allocation** satisfies

$$
\sum_{k=1}^K c_t^k(s^t) \leq \sum_{k=1}^K y_t^k(s^t)
$$

for all $t$ and for all $s^t$.

Until we reach the section on computing an equilibrium, $u$ need only satisfy the properties listed above.

From then on we specialize to CRRA utility.

## Markov asset prices

Before setting up the equilibrium, we summarize formulas for computing asset prices in a Markov setting.

These formulas are developed at greater length in {doc}`markov_asset`.

The setup assumes the following infrastructure:

* Markov states $s \in \mathbf{S} = \{\bar{s}_1, \ldots, \bar{s}_n\}$ governed by an $n$-state Markov chain with transition probability

$$
P_{ij} = \Pr \left\{s_{t+1} = \bar{s}_j \mid s_t = \bar{s}_i \right\} ;
$$

* a collection $h = 1, \ldots, H$ of assets, where asset $h$ pays $d^h(s)$ in state $s$, so that $d^h$ is an $n \times 1$ vector; and
* an $n \times n$ pricing kernel $Q$ for one-period Arrow securities, where $Q_{ij}$ is the price at time $t$ in state $s_t = \bar s_i$ of one unit of consumption delivered at time $t+1$ if $s_{t+1} = \bar s_j$.

The price in state $\bar s_i$ of a one-period risk-free bond that pays one unit of consumption in every state is $\sum_j Q_{ij}$.

The gross rate of return on that bond is therefore

$$
R_i = \Bigl(\sum_j Q_{ij}\Bigr)^{-1} .
$$

### An exogenous pricing kernel

For now we take the pricing kernel $Q$ as exogenous, that is, determined outside the model.

Two examples are

* $Q = \beta P$, where $\beta \in (0, 1)$, and
* $Q_{ij} = m_{ij} P_{ij}$, where $m_{ij} > 0$ is the value of a **stochastic discount factor** when the Markov state moves from $\bar s_i$ to $\bar s_j$.

The second example multiplies $P$ element by element, not as a matrix product.

We now describe the prices of two types of assets.

The first is a **cum-dividend** stock that entitles its owner to the time $t$ dividend and to the option to sell the asset at time $t+1$.

Its price satisfies $p^h(\bar s_i) = d^h(\bar s_i) + \sum_j Q_{ij} p^h(\bar s_j)$, so the vector $p^h$ satisfies $p^h = d^h + Q p^h$.

Provided every eigenvalue of $Q$ has modulus less than one, this implies

$$
p^h = (I - Q)^{-1} d^h .
$$

The second is an **ex-dividend** stock purchased at the end of time $t$, which entitles its owner to the time $t+1$ dividend and to the option to sell the stock at time $t+1$.

Its price is

$$
p^h = (I - Q)^{-1} Q d^h .
$$

```{note}
The matrix geometric sum $(I - Q)^{-1} = I + Q + Q^2 + \cdots$ is an example of a **resolvent operator**.

It converges when the spectral radius of $Q$ is less than one.
```

Below we describe an equilibrium model with trading of one-period Arrow securities in which the pricing kernel is endogenous.

In constructing that model, we'll repeatedly encounter formulas that remind us of these asset pricing formulas.

### Multi-step transition probabilities and pricing kernels

The $(i,j)$ component of the $j$-step-ahead transition matrix $P^j$ is

$$
\Pr(s_{t+j} = \bar s_{j'} \mid s_t = \bar s_i) = (P^j)_{i j'} .
$$

To keep the notation light below, we write $P_j(s_{t+j} \mid s_t)$ for these $j$-step transition probabilities, so that $P_j$ is represented by the matrix $P^j$.

In the same way, the price at time $t$ in state $s_t$ of one unit of consumption delivered at time $t+j$ in state $s_{t+j}$ is $Q_j(s_{t+j} \mid s_t)$, represented by the matrix $Q^j$.

We'll use these objects to state a useful property of asset pricing theory.

### Laws of iterated expectations and iterated values

A **law of iterated values** has a mathematical structure that parallels a **law of iterated expectations**.

We can describe its structure readily in the Markov setting of this lecture.

Recall the following recursion satisfied by $j$-step-ahead transition probabilities for our finite-state Markov chain:

$$
P_j(s_{t+j} \mid s_t) = \sum_{s_{t+1}} P_{j-1}(s_{t+j} \mid s_{t+1}) P(s_{t+1} \mid s_t) .
$$

We can use this recursion to verify the law of iterated expectations applied to the conditional expectation of a random variable $d(s_{t+j})$ conditioned on $s_t$:

$$
\begin{aligned}
E \bigl[ E [ d(s_{t+j}) \mid s_{t+1} ] \bigm| s_t \bigr]
    & = \sum_{s_{t+1}} \left[ \sum_{s_{t+j}} d(s_{t+j}) P_{j-1}(s_{t+j} \mid s_{t+1}) \right] P(s_{t+1} \mid s_t) \\
    & = \sum_{s_{t+j}} d(s_{t+j}) \left[ \sum_{s_{t+1}} P_{j-1}(s_{t+j} \mid s_{t+1}) P(s_{t+1} \mid s_t) \right] \\
    & = \sum_{s_{t+j}} d(s_{t+j}) P_j(s_{t+j} \mid s_t) \\
    & = E [ d(s_{t+j}) \mid s_t ] .
\end{aligned}
$$

The pricing kernel for $j$-step-ahead Arrow securities satisfies the recursion

$$
Q_j(s_{t+j} \mid s_t) = \sum_{s_{t+1}} Q_{j-1}(s_{t+j} \mid s_{t+1}) Q(s_{t+1} \mid s_t) .
$$

The time $t$ **value** in Markov state $s_t$ of a time $t+j$ payout $d(s_{t+j})$ is

$$
W [ d(s_{t+j}) \mid s_t ] = \sum_{s_{t+j}} d(s_{t+j}) Q_j(s_{t+j} \mid s_t) .
$$

The **law of iterated values** states that

$$
W \bigl[ W [ d(s_{t+j}) \mid s_{t+1} ] \bigm| s_t \bigr] = W [ d(s_{t+j}) \mid s_t ] .
$$

We verify it with a string of equalities that are counterparts to those we used to verify the law of iterated expectations:

$$
\begin{aligned}
W \bigl[ W [ d(s_{t+j}) \mid s_{t+1} ] \bigm| s_t \bigr]
    & = \sum_{s_{t+1}} \left[ \sum_{s_{t+j}} d(s_{t+j}) Q_{j-1}(s_{t+j} \mid s_{t+1}) \right] Q(s_{t+1} \mid s_t) \\
    & = \sum_{s_{t+j}} d(s_{t+j}) \left[ \sum_{s_{t+1}} Q_{j-1}(s_{t+j} \mid s_{t+1}) Q(s_{t+1} \mid s_t) \right] \\
    & = \sum_{s_{t+j}} d(s_{t+j}) Q_j(s_{t+j} \mid s_t) \\
    & = W [ d(s_{t+j}) \mid s_t ] .
\end{aligned}
$$

## Recursive formulation

Following section 9.3.3 of {cite}`Ljungqvist2012`, we now set up a competitive equilibrium of a pure exchange economy with complete markets in one-period Arrow securities.

When endowments $y^k(s)$ are all functions of a common Markov state $s$, the pricing kernel takes the form $Q(s' \mid s)$, the price of one unit of consumption in state $s'$ at date $t+1$ when the Markov state at date $t$ is $s$.

This lets us formulate a consumer's optimization problem recursively.

Consumer $k$'s state at time $t$ is its financial wealth $a_t^k$ and the Markov state $s_t$.

Let $v^k(a, s)$ be the optimal value of consumer $k$'s problem starting from state $(a, s)$.

Thus $v^k(a, s)$ is the maximum expected discounted utility that consumer $k$ with current financial wealth $a$ can attain in Markov state $s$.

The optimal value function satisfies the Bellman equation

$$
v^k(a, s) = \max_{c, \hat a(s')} \left\{ u(c) + \beta \sum_{s'} v^k[\hat a(s'), s'] \pi(s' \mid s) \right\},
$$

where the maximization is subject to the budget constraint

$$
c + \sum_{s'} \hat a(s') Q(s' \mid s) \leq y^k(s) + a
$$

and the constraints

$$
\begin{aligned}
c & \geq 0, \\
-\hat a(s') & \leq \bar A^k(s'), \quad \forall s' \in \mathbf{S} .
\end{aligned}
$$

The second set of constraints is a collection of state-by-state debt limits.

The value function and decision rule that solve the Bellman equation depend on the pricing kernel $Q(\cdot \mid \cdot)$ because it appears in the budget constraint.

The first-order conditions for the problem on the right of the Bellman equation, together with a Benveniste–Scheinkman formula, imply

$$
Q(s_{t+1} \mid s_t) = \frac{\beta u'(c_{t+1}^k) \pi(s_{t+1} \mid s_t)}{u'(c_t^k)},
$$

where it is understood that $c_t^k = c^k(s_t)$ and $c_{t+1}^k = c^k(s_{t+1})$.

### Recursive competitive equilibrium

A **recursive competitive equilibrium** is an initial distribution of wealth $\vec a_0$, a set of borrowing limits $\{\bar A^k(s)\}_{k=1}^K$, a pricing kernel $Q(s' \mid s)$, sets of value functions $\{v^k(a, s)\}_{k=1}^K$, and decision rules $\{c^k(s), \hat a^k(s)\}_{k=1}^K$ such that

1. the state-by-state borrowing limits satisfy the recursion

$$
\bar A^k(s) = y^k(s) + \sum_{s'} Q(s' \mid s) \bar A^k(s') ;
$$

2. for all $k$, given $a_0^k$, $\bar A^k(s)$, and the pricing kernel, the value functions and decision rules solve the consumers' problems;

3. for all realizations of $\{s_t\}_{t=0}^\infty$, the consumption and asset portfolios $\{\{c_t^k, \{\hat a_{t+1}^k(s')\}_{s'}\}_k\}_t$ satisfy $\sum_k c_t^k = \sum_k y^k(s_t)$ and $\sum_k \hat a_{t+1}^k(s') = 0$ for all $t$ and $s'$; and

4. the initial financial wealth vector $\vec a_0$ satisfies $\sum_{k=1}^K a_0^k = 0$.

The third condition asserts that goods markets clear and that there are zero net aggregate claims in all Markov states.

The fourth condition asserts that the economy is closed and starts from a situation in which there are zero net aggregate claims.

(sec-rational-expectations)=
## A rational expectations equilibrium

The equilibrium that we have just defined asks each consumer to choose a portfolio $\hat a(s')$ of claims today while knowing the pricing kernel $Q(s' \mid s)$ at which those claims trade.

But the value of a portfolio to a consumer depends on what goods will cost tomorrow.

So an equilibrium of this kind embeds a forecast, and it is an equilibrium only when markets confirm that forecast.

That is what {cite:t}`radner1972` later formalized as an equilibrium of plans, prices, and price expectations.

### What a single good conceals

In our economy the only price that a sequential trader must forecast is the one-period pricing kernel $Q(s' \mid s)$.

Within a date and Markov state there are no relative prices left to forecast, because one good is the numeraire by default.

In Arrow's economy with $C > 1$ goods, a household choosing a portfolio must forecast the entire vector of spot prices $\bar p_s = (\bar p_{s1}, \ldots, \bar p_{sC})$ that will prevail in each state $s$, because only those prices tell it what a dollar delivered in state $s$ will be worth.

The equivalence of the two trading arrangements then holds *if and only if those forecasts are correct*, state by state.

That is the precise sense in which Arrow's sequential equilibrium is a rational expectations equilibrium, and it is a requirement that our one-good formulation satisfies vacuously.

{ref}`ge_arrow_ex4` asks you to verify the equivalence in a two-period economy with several goods, where the requirement has content.

### Securities denominated in a unit of account

That Arrow's equivalence turns *only* on the correctness of those forecasts is itself a consequence of his having denominated securities in a unit of account.

Because a unit of security $s$ pays one dollar if state $s$ occurs, the set of state-contingent wealth distributions that a portfolio can deliver is settled before any forecast is made.

{cite:t}`hart1975` asked what happens when securities instead pay *goods*.

If security $f$ delivers the bundle $a_f(s)$ in state $s$, its payoff in dollars is $\bar p_s \cdot a_f(s)$, so the forecast $\bar p$ determines the span of the asset market, the span determines consumers' budget sets, and those budget sets determine the prices being forecast.

Completeness becomes an equilibrium object rather than an assumption.

Two things can then go wrong that cannot go wrong in our one-good economy.

There can be several equilibria, all of them with correct forecasts, that differ in which markets are *effectively* open and that are strictly Pareto ranked.

And there can be no equilibrium at all, even with strictly concave utility functions, strictly positive endowments, and no transaction costs.

{ref}`ge_arrow_ex5` works through Hart's examples of both.

```{note}
{cite:t}`hart1975` also delimits when things go right.

An equilibrium of the kind defined by {cite:t}`radner1972` is Pareto optimal when the market structure is complete, and it is Pareto optimal relative to the set of such equilibria when there is one good and the market structure is complete up to the next-to-last date, which generalizes a result of {cite:t}`diamond1967`.

Outside those two cases, utility functions and endowments can generally be chosen so that an equilibrium fails even that weaker test.

Both of Hart's examples are knife edge, so small perturbations of preferences, endowments, or dividends destroy them.
```

### Beliefs about nature versus expectations of prices

Arrow drew a distinction here that is worth preserving.

His consumers act on *subjective* probabilities that are allowed to differ across people, and nothing in his theorems requires agreement about how likely the states are.

What his sequential arrangement does require is that consumers agree, and be right, about something else: the prices that will rule in each state.

Beliefs about exogenous nature may be heterogeneous; expectations about endogenous prices may not.

This lecture imposes the stronger assumption of common beliefs, which lets us write a single probability $\pi(s' \mid s)$ in each consumer's Euler equation.

{doc}`harrison_kreps` and {doc}`likelihood_ratio_process_2` relax that assumption while keeping the expectations of prices correct.

### Economizing on markets

Arrow observed that the sequential arrangement *permits economizing on markets*.

In our infinite-horizon economy, time $0$ trading requires a market in a claim for every date and every history, while sequential trading requires only $n$ one-period markets at each date.

In the two-period, $S$-state, $C$-good economy of {ref}`ge_arrow_ex4`, time $0$ trading requires $SC$ contingent claims markets, while the sequential arrangement requires $S$ security markets followed by $C$ spot markets in whichever state occurs.

No counting of markets, however, would reveal the extra assumption that the sequential arrangement needs, which is that the forecasts of spot prices are correct.

```{note}
{cite:t}`kihlstrom2019` remarks that when Arrow presented this material in 1952, the dynamic programming machinery that makes the equivalence transparent did not yet exist, because Bellman was in the midst of creating it.

The step that decomposes a household's once-and-for-all problem into a portfolio choice followed by a sequence of within-period problems is exactly the argument that we used to Bellmanize the equilibrium above.
```

## State variable degeneracy

{cite}`Ljungqvist2012` and {doc}`cass_koopmans_2` describe a different timing protocol, in which there is a complete menu of history-contingent claims on consumption at all dates and all trades occur once and for all at time $0$.

For the allocation and pricing kernel of a recursive competitive equilibrium to coincide with those of that time $0$ arrangement, we must impose $a_0^k = 0$ for $k = 1, \ldots, K$.

That initial condition ensures that at time $0$ the present value of each consumer's consumption equals the present value of its endowment stream, which is the single budget constraint of the time $0$ arrangement.

Starting the system with $a_0^k = 0$ for all $k$ has a striking implication that we call **state variable degeneracy**.

Although two state variables $a$ and $s$ appear in the value function $v^k(a, s)$, within a recursive competitive equilibrium that starts from $a_0^k = 0$ for all $k$ at initial Markov state $s_0$, two outcomes prevail:

* financial wealth $a_t^k$ is an exact function of the Markov state $s_t$, which we compute below, and
* $a_t^k = 0$ for all $k$ whenever the Markov state $s_t$ returns to $s_0$.

The first finding asserts that within a competitive equilibrium the exogenous Markov state is all we require to track an individual, because financial wealth is redundant.

The second finding asserts that each household returns to the zero financial wealth with which it began life whenever the Markov state returns to its initial value.

That happens infinitely often if $s_0$ is a recurrent state of the Markov chain, but it need not happen at all if $s_0$ is transient; see {doc}`finite_markov`.

This outcome depends critically on there being complete markets in Arrow securities.

For example, it does not prevail in the incomplete markets setting of {doc}`aiyagari`, where a household's wealth depends on its whole history of shocks.

## Computing a competitive equilibrium

Now we are ready to do some fun calculations.

We find it useful to think in terms of analytical **inputs** into and **outputs** from our general equilibrium theorizing.

### Inputs and outputs

The inputs are

* Markov states $s \in \mathbf{S} = \{\bar{s}_1, \ldots, \bar{s}_n\}$ governed by an $n$-state Markov chain with transition matrix $P$;
* $K$ vectors of individual endowments $y^k$, each of dimension $n \times 1$ with components $y^k(\bar s_i)$;
* the $n \times 1$ vector of aggregate endowments $y(s) \equiv \sum_{k=1}^K y^k(s)$; and
* preferences given by the common utility functional $E_0 \sum_{t=0}^\infty \beta^t u(c_t^k)$ with discount factor $\beta \in (0, 1)$ and CRRA one-period utility function

$$
u(c) = \frac{c^{1-\gamma}}{1-\gamma},
\qquad
u'(c) = c^{-\gamma} .
$$

Feasibility requires

$$
c(s) = \sum_{k=1}^K c^k(s) \leq y(s) .
$$

The outputs are

* an $n \times n$ pricing kernel $Q$ for one-period Arrow securities;
* the aggregate allocation, which in a pure exchange economy is $c(s) = y(s)$;
* a $K \times 1$ distribution of wealth $\alpha$ with $\alpha_k \geq 0$ and $\sum_{k=1}^K \alpha_k = 1$; and
* $K$ vectors of individual consumptions $c^k$, each of dimension $n \times 1$.

### The pricing kernel

For any agent $k \in \{1, \ldots, K\}$, at the equilibrium allocation, the one-period Arrow securities pricing kernel satisfies

$$
Q_{ij} = \beta \left(\frac{c^k(\bar{s}_j)}{c^k(\bar{s}_i)}\right)^{-\gamma} P_{ij} .
$$

This follows from agent $k$'s first-order necessary conditions.

Because all agents face the same pricing kernel, the Euler equations of any two agents $k$ and $m$ imply

$$
\left(\frac{c^k(\bar{s}_j)}{c^k(\bar{s}_i)}\right)^{-\gamma}
=
\left(\frac{c^m(\bar{s}_j)}{c^m(\bar{s}_i)}\right)^{-\gamma}
\quad \text{whenever } P_{ij} > 0 .
$$

So the ratio $c^k(s)/c^m(s)$ is the same in any two states connected by a positive transition probability, and hence in every state reachable from $s_0$.

Consumption shares are therefore constant, and feasibility gives

$$
c^k(s) = \alpha_k c(s) = \alpha_k y(s)
$$

for a **distribution of wealth** $\alpha$ that satisfies $\alpha_k \geq 0$ and $\sum_{k=1}^K \alpha_k = 1$.

```{note}
Identical CRRA preferences also satisfy the conditions for **Gorman aggregation**, since Engel curves are linear with a common slope, so a representative consumer exists.

The constancy of consumption shares, however, follows directly from the Euler equations above.
```

This means that we can compute the pricing kernel from

$$
Q_{ij} = \beta \left(\frac{y_j}{y_i}\right)^{-\gamma} P_{ij} .
$$ (eq:Qformula)

This is the pricing kernel of the Lucas tree economy studied in {doc}`markov_asset`, in which a representative consumer eats the aggregate endowment.

The pricing kernel $Q$ does not depend on the vector $\alpha$.

**Key finding:** We can compute competitive equilibrium **prices** prior to computing a **distribution of wealth**.

The wealth distribution $\alpha$ is not arbitrary.

It is pinned down by the initial condition $a_0^k = 0$, as we show below.

Formula {eq}`eq:Qformula` has a useful matrix form.

Let $D = \mathrm{diag}\bigl(u'(y_1), \ldots, u'(y_n)\bigr)$.

Then $Q = \beta D^{-1} P D$, so $Q$ is similar to $\beta P$.

Its eigenvalues are $\beta$ times those of $P$, and because $P$ is a stochastic matrix, the spectral radius of $Q$ equals $\beta < 1$.

This guarantees that the resolvent $(I - Q)^{-1}$ used below exists.

The factorization is an instance of the **transition independence** structure exploited in {doc}`ross_recovery`, a connection we pursue in {ref}`ge_arrow_ex2`.

### Natural debt limits

Having computed an equilibrium pricing kernel $Q$, we can compute several **values** that are required to pose or represent the solution of an individual household's optimization problem.

For each individual $k$, let $\bar A^k$ be the $n \times 1$ vector with components $\bar A^k(\bar s_i)$.

The recursion in the definition of equilibrium implies

$$
\bar A^k = \left[I - Q\right]^{-1} y^k .
$$ (eq:debtlimit)

In a competitive equilibrium of an **infinite-horizon** economy with sequential trading of one-period Arrow securities, $\bar A^k(s)$ is a state-by-state limit on the quantity of one-period Arrow securities paying off in state $s$ at time $t+1$ that individual $k$ can issue at time $t$.

These are often called **natural debt limits**.

They equal the maximum amount that individual $k$ can repay in state $s$ even if it consumes nothing forevermore.

```{note}
If utility satisfies an Inada condition at zero consumption, or if consumption is simply required to be nonnegative, then a **finite-horizon** economy with sequential trading of one-period Arrow securities needs no natural debt limits.

See the section on a finite-horizon economy below.
```

### Continuation wealth and optimal portfolios

Continuation wealth plays an important role in Bellmanizing a competitive equilibrium with sequential trading of a complete set of one-period Arrow securities.

For each individual $k$, let $\psi^k$ be the $n \times 1$ vector with components $\psi^k(\bar s_i)$, the financial wealth that consumer $k$ holds when the Markov state is $\bar s_i$.

Continuation wealth satisfies

$$
\psi^k = \left[I - Q\right]^{-1} \left[\alpha_k y - y^k\right] .
$$ (eq:continwealth)

To see why, note that with consumption $c^k = \alpha_k y$ and a portfolio $\hat a^k(s') = \psi^k(s')$, the budget constraint holds with equality in every state exactly when $\psi^k = \alpha_k y - y^k + Q \psi^k$.

Summing over $k$ shows that $\sum_{k=1}^K \psi^k = 0_{n \times 1}$, so Arrow security markets clear.

A nifty feature of the model is that an optimal portfolio of a type $k$ agent equals the continuation wealth that we just computed.

Thus, agent $k$'s purchases of Arrow securities that pay off next period depend only on next period's Markov state and equal

$$
\hat a^k(s) = \psi^k(s), \quad s \in \{\bar s_1, \ldots, \bar s_n\} .
$$ (eqn:optport)

### The equilibrium wealth distribution

With the initial state being a particular state $s_0 \in \{\bar{s}_1, \ldots, \bar{s}_n\}$, we must have

$$
\psi^k(s_0) = 0, \quad k = 1, \ldots, K,
$$

so that every agent starts debt-free and holding no financial assets.

This means that the equilibrium distribution of wealth satisfies

$$
\alpha_k = \frac{V_z y^k}{V_z y},
$$ (eqn:alphakform)

where $V \equiv \left[I - Q\right]^{-1}$ and $V_z$ is the row of $V$ corresponding to the initial state $s_0$.

Since $\sum_{k=1}^K V_z y^k = V_z y$, we have $\sum_{k=1}^K \alpha_k = 1$.

Comparing {eq}`eqn:alphakform` with {eq}`eq:debtlimit` gives a revealing interpretation,

$$
\alpha_k = \frac{\bar A^k(s_0)}{\sum_{m=1}^K \bar A^m(s_0)} .
$$

Each consumer's share of aggregate consumption equals its share of the value, in the initial state, of the aggregate endowment.

Because $\alpha$ depends on $s_0$ through $V_z$, the same economy started in different Markov states delivers different wealth distributions.

### Value functions

We can also compute optimal value functions in a competitive equilibrium with trades in a complete set of one-period state-contingent Arrow securities.

Call the optimal value function of consumer $k$ the $n \times 1$ vector $J^k$.

For the infinite-horizon economy now under study,

$$
J^k = (I - \beta P)^{-1} u(\alpha_k y),
$$

where $u(\alpha_k y)$ is the $n \times 1$ vector with components $u(\alpha_k y_i)$.

### Summary of the algorithm

Here is the logical flow of an algorithm to compute a competitive equilibrium:

1. compute $Q$ from the aggregate endowment using formula {eq}`eq:Qformula`;
2. compute the distribution of wealth $\alpha$ from formula {eq}`eqn:alphakform`;
3. using $\alpha$, assign each consumer $k$ the share $\alpha_k$ of the aggregate endowment in each state;
4. compute continuation wealths from the $\alpha$-dependent formula {eq}`eq:continwealth`;
5. set agent $k$'s portfolio equal to its continuation wealth state by state, as in {eq}`eqn:optport`; and
6. compute value functions $J^k$.

## Finite horizon

We now describe a finite-horizon version of the economy that operates for $T+1$ periods $t \in \mathbf{T} = \{0, 1, \ldots, T\}$.

We'll want time-dependent counterparts of the objects described above, with one important exception: we won't need **borrowing limits**.

* Borrowing limits aren't required in a finite-horizon economy in which the one-period utility function $u(c)$ satisfies an Inada condition that sends the marginal utility of consumption to infinity as consumption approaches zero.
* Nonnegativity of consumption at all $t \in \mathbf{T}$ automatically limits borrowing, because no one can end period $T$ in debt.

For each individual $k$ and date $t$, let $\psi_t^k$ be the $n \times 1$ vector of continuation wealths.

At the terminal date there is no future to finance, so $\psi_T^k = \alpha_k y - y^k$.

Working backward with the budget constraint $\psi_t^k = \alpha_k y - y^k + Q \psi_{t+1}^k$ gives

$$
\psi_t^k = \left[I + Q + Q^2 + \cdots + Q^{T-t}\right] \left[\alpha_k y - y^k\right],
\quad t = 0, 1, \ldots, T .
$$ (eq:vv)

As before, $\sum_{k=1}^K \psi_t^k = 0_{n \times 1}$ for all $t \in \mathbf{T}$.

With the initial state being a particular state $s_0$, we must have

$$
\psi_0^k(s_0) = 0, \quad k = 1, \ldots, K,
$$

which means the equilibrium distribution of wealth satisfies

$$
\alpha_k = \frac{V_z y^k}{V_z y},
$$ (eq:w)

where now

$$
V = \left[I + Q + Q^2 + \cdots + Q^T\right]
$$ (eq:ww)

and $V_z$ is the row of $V$ corresponding to the initial state $s_0$.

```{note}
In the finite-horizon economy, continuation wealth depends on calendar time as well as on the Markov state.

The initial condition sets $\psi_0^k(s_0) = 0$, but when the Markov state returns to $s_0$ at a later date $t$, fewer periods remain, the geometric sum in {eq}`eq:vv` is truncated sooner, and in general $\psi_t^k(s_0) \neq 0$.

The strong form of state variable degeneracy, in which wealth is a function of the Markov state alone, is special to the infinite horizon.

{ref}`ge_arrow_ex3` explores this.
```

To compute a competitive equilibrium with Arrow securities in the finite-horizon Markov economy,

1. compute $Q$ from the aggregate endowment using formula {eq}`eq:Qformula`;
2. compute the distribution of wealth $\alpha$ from formulas {eq}`eq:w` and {eq}`eq:ww`;
3. using $\alpha$, assign each consumer $k$ the share $\alpha_k$ of the aggregate endowment in each state;
4. compute continuation wealths from formula {eq}`eq:vv`; and
5. set agent $k$'s portfolio equal to its continuation wealth state by state.

The value function of consumer $k$ at time $t$ is

$$
J_t^k = \left[I + \beta P + \cdots + (\beta P)^{T-t}\right] u(\alpha_k y) .
$$

## Python code

We now create a Python class to compute the objects that comprise a competitive equilibrium with sequential trading of one-period Arrow securities.

The class handles both infinite-horizon economies and finite-horizon economies indexed by horizon $T$.

Every geometric sum in the lecture has the form $I + M + \cdots$, with $M = Q$ for prices and wealth and $M = \beta P$ for values, so a single helper method computes them all.

In the finite-horizon case the helper works backward with the recursion $S_t = I + M S_{t+1}$, starting from $S_T = I$.

With $M = Q$, this recursion is the law of iterated values at work: the time $t$ value of payouts from $t$ through $T$ is the payout at $t$ plus the time $t$ value of the time $t+1$ value of the remaining payouts.

With $M = \beta P$, it is the law of iterated expectations applied to discounted utility.

The class also has a method that prices an asset with dividend vector $d$ by applying these sums, which delivers the cum-dividend price $p = d + Q d + Q^2 d + \cdots$ and the ex-dividend price $p - d$.

In the finite-horizon case, arrays that depend on time are ordered from $t = 0$ to $t = T$, so that `ψ[t]` is $\psi_t$ and `J[t]` is $J_t$.

In the infinite-horizon case they have a single leading element.

```{code-cell} ipython3
class RecurCompetitive:
    """
    A competitive equilibrium with complete markets in one-period
    Arrow securities.

    Parameters
    ----------
    s : array of length n
        Markov states
    P : n x n array
        Markov transition matrix
    ys : n x K array
        endowments, with column k holding agent k's endowment
    γ : float
        coefficient of relative risk aversion
    β : float
        discount factor
    T : int or None
        time horizon, None for an infinite horizon
    """

    def __init__(self, s, P, ys, γ=0.5, β=0.98, T=None):

        self.s, self.P, self.ys = s, P, ys
        self.γ, self.β, self.T = γ, β, T
        self.n, self.K = ys.shape
        self.y = ys.sum(axis=1)                  # aggregate endowment

        self.Q = self.pricing_kernel()
        self.PRF = self.Q.sum(axis=1)            # price of a risk-free bond
        self.R = 1 / self.PRF                    # gross risk-free rate

        # V[t] = I + Q + ... + Q^(T-t), or [(I - Q)^(-1)] if T is None
        self.V = self.geometric_sums(self.Q)

        # time-0 values of endowments, the natural debt limits
        self.A = self.asset_price(ys)

    def u(self, c):
        "CRRA utility"
        return c ** (1 - self.γ) / (1 - self.γ)

    def u_prime(self, c):
        "Marginal utility"
        return c ** (-self.γ)

    def pricing_kernel(self):
        "Pricing kernel Q from equation (eq:Qformula)"
        mu = self.u_prime(self.y)
        return self.β * self.P * mu[None, :] / mu[:, None]

    def geometric_sums(self, M):
        """
        Return [(I - M)^(-1)] if T is None; otherwise return the sequence
        S[0], ..., S[T] with S[t] = I + M + ... + M^(T-t).
        """
        n, T = self.n, self.T
        if T is None:
            return np.linalg.inv(np.eye(n) - M)[None, :, :]
        S = np.empty((T+1, n, n))
        S[T] = np.eye(n)
        for t in range(T-1, -1, -1):
            S[t] = np.eye(n) + M @ S[t+1]      # law of iterated values
        return S

    def asset_price(self, d, ex_dividend=False):
        """
        Time-0 price of an asset with dividend vector d (n or n x K):
        cum-dividend p = d + Q d + Q^2 d + ..., or ex-dividend p - d.
        """
        p = self.V[0] @ d
        return p - d if ex_dividend else p

    def wealth_distribution(self, s0_idx):
        "Wealth distribution α when the initial state has index s0_idx"
        self.s0_idx = s0_idx
        V_z = self.V[0, s0_idx, :]
        self.α = V_z @ self.ys / (V_z @ self.y)
        return self.α

    def continuation_wealths(self):
        "Continuation wealths ψ, with ψ[t, i, k] = ψ_t^k(s_i)"
        excess = np.outer(self.y, self.α) - self.ys     # α_k y - y^k
        self.ψ = self.V @ excess
        return self.ψ

    def value_functions(self):
        "Value functions J, with J[t, i, k] = J_t^k(s_i)"
        flow = self.u(np.outer(self.y, self.α))         # u(α_k y)
        self.J = self.geometric_sums(self.β * self.P) @ flow
        return self.J
```

## Examples

We'll use our code to construct equilibrium objects in several example economies.

Our first several examples are infinite-horizon economies.

Our final example is a finite-horizon economy.

Unless we say otherwise, examples use the default parameter values $\gamma = 0.5$ and $\beta = 0.98$.

### Example 1: a constant aggregate endowment

Two agents have perfectly negatively correlated endowments, so the aggregate endowment is constant.

```{code-cell} ipython3
s = np.array([0, 1])
P = np.array([[.5, .5],
              [.5, .5]])

ys = np.empty((2, 2))
ys[:, 0] = 1 - s       # agent 1
ys[:, 1] = s           # agent 2

ex1 = RecurCompetitive(s, P, ys)
```

```{code-cell} ipython3
print("aggregate endowment y =", ex1.y)
print("pricing kernel Q = \n", ex1.Q)
print("risk-free rate R =", ex1.R)
print("natural debt limits A = \n", ex1.A)
```

Because the aggregate endowment is constant, marginal utility is constant, so $Q = \beta P$ and the risk-free rate is $\beta^{-1}$ in both states.

```{code-cell} ipython3
# initial state is state 1
print(f'α = {ex1.wealth_distribution(s0_idx=0)}')
print(f'ψ = \n{ex1.continuation_wealths()}')
print(f'J = \n{ex1.value_functions()}')
print(f'share of natural debt limits in s0: {ex1.A[0] / ex1.A[0].sum()}')
```

When the economy starts in state 1, agent 1 receives slightly more than half of aggregate consumption.

Its endowment arrives in the initial period, and consumption received sooner is discounted less.

As the last line confirms, each agent's consumption share equals its share of the natural debt limits in the initial state.

In state 2, where agent 1 has no endowment, agent 1 holds financial wealth of one unit, which finances its consumption, and agent 2 owes exactly that amount.

```{code-cell} ipython3
# initial state is state 2
print(f'α = {ex1.wealth_distribution(s0_idx=1)}')
print(f'ψ = \n{ex1.continuation_wealths()}')
print(f'J = \n{ex1.value_functions()}')
```

Starting in state 2 simply swaps the roles of the two agents.

### Example 2: a fluctuating aggregate endowment

Now agent 1 has a constant endowment while agent 2's endowment fluctuates, so the aggregate endowment fluctuates.

```{code-cell} ipython3
s = np.array([1, 2])
P = np.array([[.5, .5],
              [.5, .5]])

ys = np.empty((2, 2))
ys[:, 0] = 1.5         # agent 1
ys[:, 1] = s           # agent 2

ex2 = RecurCompetitive(s, P, ys)

print("aggregate endowment y =", ex2.y)
print("pricing kernel Q = \n", ex2.Q)
print("risk-free rate R =", ex2.R)
print("natural debt limits A = \n", ex2.A)
```

The pricing kernels in examples 1 and 2 differ because the aggregate endowment is constant in example 1 but differs across states in example 2.

We can check two off-diagonal entries of $Q$ directly against formula {eq}`eq:Qformula`.

```{code-cell} ipython3
print(ex2.β * ex2.u_prime(3.5) / ex2.u_prime(2.5) * ex2.P[0, 1], ex2.Q[0, 1])
print(ex2.β * ex2.u_prime(2.5) / ex2.u_prime(3.5) * ex2.P[1, 0], ex2.Q[1, 0])
```

A claim to consumption in the high-endowment state is cheap, because marginal utility is low there.

The risk-free rate is high in the low-endowment state, where consumption is expected to rise, and low in the high-endowment state, where consumption is expected to fall.

Now let's price some risky assets with the formulas $p^h = (I - Q)^{-1} d^h$ and $p^h = (I - Q)^{-1} Q d^h$ from the section on Markov asset prices.

We price a Lucas tree, which pays the aggregate endowment as its dividend, and claims to each agent's endowment stream.

```{code-cell} ipython3
p_tree = ex2.asset_price(ex2.y)
p_tree_ex = ex2.asset_price(ex2.y, ex_dividend=True)

print("cum-dividend tree price     p =", p_tree)
print("ex-dividend tree price      p =", p_tree_ex)
print("price-dividend ratio (ex)     =", p_tree_ex / ex2.y)
print("Bellman residual |p - d - Qp| =",
      np.abs(p_tree - ex2.y - ex2.Q @ p_tree).max())
print("claims to endowments = \n", ex2.asset_price(ex2.ys))
```

The cum-dividend price satisfies the one-step Bellman equation $p = d + Q p$ to machine precision.

The ex-dividend price-dividend ratio is higher in the low-endowment state, where the current dividend is low relative to expected future dividends.

The last array reproduces the natural debt limits computed above: agent $k$'s natural debt limit in state $s$ is the cum-dividend price of a claim to agent $k$'s own endowment stream.

That is why an agent can always repay a debt no larger than $\bar A^k(s)$: it could sell the claim to its endowment and consume nothing forever.

```{code-cell} ipython3
# initial state is state 1
print(f'α = {ex2.wealth_distribution(s0_idx=0)}')
print(f'ψ = \n{ex2.continuation_wealths()}')
print(f'J = \n{ex2.value_functions()}')
```

```{code-cell} ipython3
# initial state is state 2
print(f'α = {ex2.wealth_distribution(s0_idx=1)}')
print(f'ψ = \n{ex2.continuation_wealths()}')
print(f'J = \n{ex2.value_functions()}')
```

### Example 3: an absorbing state

In this example state 2 is absorbing, so state 1 is transient.

```{code-cell} ipython3
s = np.array([1, 2])

λ = 0.9
P = np.array([[1-λ, λ],
              [0, 1]])

ys = np.empty((2, 2))
ys[:, 0] = [1, 0]      # agent 1
ys[:, 1] = [0, 1]      # agent 2

ex3 = RecurCompetitive(s, P, ys)

print("pricing kernel Q = \n", ex3.Q)
print("natural debt limits A = \n", ex3.A)
```

The natural debt limit for agent 1 in state 2 is $0$.

Once the economy enters the absorbing state, agent 1 never receives another unit of endowment, so it cannot credibly promise to repay anything.

```{code-cell} ipython3
# initial state is state 1
print(f'α = {ex3.wealth_distribution(s0_idx=0)}')
print(f'ψ = \n{ex3.continuation_wealths()}')
print(f'J = \n{ex3.value_functions()}')
```

Starting in state 1, agent 1 receives only a small share of aggregate consumption, because its endowment arrives only while the economy remains in the transient state.

Because state 1 is transient, the economy eventually leaves it for good, and agents' wealths do not recurrently return to zero.

```{code-cell} ipython3
# initial state is state 2
print(f'α = {ex3.wealth_distribution(s0_idx=1)}')
print(f'ψ = \n{ex3.continuation_wealths()}')
print(f'J = \n{ex3.value_functions()}')
```

Starting in the absorbing state, agent 1 owns nothing of value, so $\alpha_1 = 0$ and agent 1 consumes nothing forever.

This corner is consistent with the discussion of the Inada condition above, which guarantees interior consumption only for agents whose endowments have positive value.

For the specification of the Markov chain in example 3, let's see how the equilibrium wealth distribution varies with the transition probability $\lambda$.

```{code-cell} ipython3
λ_seq = np.linspace(0, 0.99, 100)

# prepare containers
αs0_seq = np.empty((len(λ_seq), 2))
αs1_seq = np.empty((len(λ_seq), 2))

for i, λ in enumerate(λ_seq):
    P_λ = np.array([[1-λ, λ],
                    [0, 1]])
    ex3_λ = RecurCompetitive(s, P_λ, ys)

    # initial state s0 = 1
    αs0_seq[i, :] = ex3_λ.wealth_distribution(s0_idx=0)

    # initial state s0 = 2
    αs1_seq[i, :] = ex3_λ.wealth_distribution(s0_idx=1)
```

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

for i, αs_seq in enumerate([αs0_seq, αs1_seq]):
    for j in range(2):
        axs[i].plot(λ_seq, αs_seq[:, j], label=f'$\\alpha_{j+1}$')
    axs[i].set_xlabel(r'$\lambda$')
    axs[i].set_title(f'initial state $s_0 = {s[i]}$')
    axs[i].legend()

plt.show()
```

When the economy starts in state 1, a higher probability $\lambda$ of leaving that state for good shortens the expected duration of agent 1's endowment and lowers its wealth share.

When the economy starts in the absorbing state 2, $\lambda$ is irrelevant and agent 2 owns everything.

### Example 4: prosperity, a moderate state, and recession

Our last infinite-horizon example has three Markov states, which we interpret as prosperity, a moderate state, and recession.

```{code-cell} ipython3
s = np.array([1, 2, 3])

λ = .9
μ = .9
δ = .05

# prosperous, moderate, and recession states
P = np.array([[1-λ, λ, 0],
              [(1-μ)/2, μ, (1-μ)/2],
              [(1-δ)/2, (1-δ)/2, δ]])

ys = np.empty((3, 2))
ys[:, 0] = [.25, .75, .2]      # agent 1
ys[:, 1] = [1.25, .25, .2]     # agent 2

ex4 = RecurCompetitive(s, P, ys)

print("rows of P sum to", P.sum(axis=1))
print("aggregate endowment y =", ex4.y)
print("pricing kernel Q = \n", ex4.Q)
print("risk-free rate R =", ex4.R)
print("natural debt limits A = \n", ex4.A)
```

The moderate state is highly persistent, and the economy moves out of recession quickly.

The gross risk-free rate is below one in prosperity, where aggregate consumption is expected to fall, and well above one in recession, where it is expected to recover.

```{code-cell} ipython3
for i in range(3):
    print(f"initial state is state {i+1}")
    print(f'α = {ex4.wealth_distribution(s0_idx=i)}')
    print(f'ψ = \n{ex4.continuation_wealths()}')
    print(f'J = \n{ex4.value_functions()}\n')
```

Agent 1, whose endowment is concentrated in the persistent moderate state, receives about two thirds of aggregate consumption whichever state the economy starts in.

### A finite-horizon example

We now revisit the economy defined in example 1, but set the time horizon to $T = 10$.

```{code-cell} ipython3
s = np.array([0, 1])
P = np.array([[.5, .5],
              [.5, .5]])

ys = np.empty((2, 2))
ys[:, 0] = 1 - s       # agent 1
ys[:, 1] = s           # agent 2

ex1_finite = RecurCompetitive(s, P, ys, T=10)
```

```{code-cell} ipython3
# I + Q + Q^2 + ... + Q^T
ex1_finite.V[0]
```

In the finite-horizon case, `ψ` and `J` are returned as sequences ordered from $t = 0$ to $t = T$.

```{code-cell} ipython3
# initial state is state 1
print(f'α = {ex1_finite.wealth_distribution(s0_idx=0)}')
print(f'ψ = \n{ex1_finite.continuation_wealths()}\n')
print(f'J = \n{ex1_finite.value_functions()}')
```

```{code-cell} ipython3
# initial state is state 2
print(f'α = {ex1_finite.wealth_distribution(s0_idx=1)}')
print(f'ψ = \n{ex1_finite.continuation_wealths()}\n')
print(f'J = \n{ex1_finite.value_functions()}')
```

The wealth distribution is further from equal than in example 1, because with a short horizon the endowment received in the initial period is a larger fraction of the total value of an agent's endowment.

Let's check why this economy needs no borrowing limits.

At date $t$, the most that agent $k$ could ever repay from its own endowment is the value of its remaining endowment stream, $[I + Q + \cdots + Q^{T-t}]\, y^k$.

Unlike in the infinite-horizon economy, these bounds need not be imposed.

At date $T$ no securities are traded, so an agent cannot roll debt over, and nonnegative consumption at $T$ limits what it can owe to $y^k(s_T)$.

Working backward, nonnegative consumption at each earlier date implies the bound for that date.

The distance between an agent's bound and its debt, $\psi_t^k + [I + Q + \cdots + Q^{T-t}]\, y^k$, equals $[I + Q + \cdots + Q^{T-t}]\, \alpha_k y$, the value of the agent's remaining consumption.

That value is positive because consumption is positive, so the implied bounds never bind.

The following code computes the bounds for every date and state and confirms this.

```{code-cell} ipython3
ex1_finite.wealth_distribution(s0_idx=0)
ψ_finite = ex1_finite.continuation_wealths()
bounds = ex1_finite.V @ ex1_finite.ys      # bounds[t] = (I + ... + Q^(T-t)) y^k

print("implied bounds at t = T (rows: states, columns: agents):\n", bounds[-1])
print("smallest slack ψ_t + bound over all t, states, agents:",
      (ψ_finite + bounds).min().round(4))
```

At $t = T$, the bounds are just current endowments, and the slack is smallest there because only one period of consumption remains to be valued.

In the infinite-horizon economy there is no last date at which debts must be settled, so without explicit limits an agent could roll over ever larger debts forever, and the natural debt limits {eq}`eq:debtlimit` must be imposed.

We can check that as $T \rightarrow \infty$ the finite-horizon results converge to those of the infinite-horizon economy.

Both economies below start in state 2, and we compare time-0 objects.

```{code-cell} ipython3
ex1_large = RecurCompetitive(s, P, ys, T=10000)
ex1.wealth_distribution(s0_idx=1)
ex1_large.wealth_distribution(s0_idx=1)

print("V:", np.abs(ex1.V[0] - ex1_large.V[0]).max())
print("ψ:", np.abs(ex1.continuation_wealths()[0]
                   - ex1_large.continuation_wealths()[0]).max())
print("J:", np.abs(ex1.value_functions()[0]
                   - ex1_large.value_functions()[0]).max())
```

The maximum absolute differences are negligible.

## Concluding remarks

We began by promising a computable account of competitive equilibrium in an infinite-horizon exchange economy with heterogeneous endowments, complete markets in one-period Arrow securities, identical CRRA preferences, and common beliefs.

Here is how the lecture delivered on that promise.

### Prices before the wealth distribution

Because all agents face the same pricing kernel, their Euler equations force consumption shares to be constant, so that $c^k(s) = \alpha_k y(s)$.

The pricing kernel {eq}`eq:Qformula` therefore depends only on the aggregate endowment, and it coincides with the kernel of a representative-agent Lucas tree economy.

The wealth distribution $\alpha$ comes second, pinned down by the requirement that every agent start with zero financial wealth, which makes $\alpha_k$ equal to agent $k$'s share of the value of aggregate endowments in the initial state.

### Bellmanizing the equilibrium

Each object that the overview promised to describe with a Bellman equation satisfies a one-step recursion of the form $x = b + Q x$:

* asset prices satisfy $p^h = d^h + Q p^h$, as we verified for a Lucas tree in example 2;
* natural debt limits satisfy $\bar A^k = y^k + Q \bar A^k$, which makes them the prices of claims to agents' endowment streams; and
* continuation wealth satisfies $\psi^k = (\alpha_k y - y^k) + Q \psi^k$.

Value functions satisfy the analogous recursion $J^k = u(\alpha_k y) + \beta P J^k$, with $\beta P$ in place of $Q$.

These recursions are what let a single Python class, `RecurCompetitive`, compute an entire equilibrium from a few matrix operations.

### The five ideas

* **Resolvent operators.** Solving each recursion gives $(I - Q)^{-1}$ or $(I - \beta P)^{-1}$, and the similarity $Q = \beta D^{-1} P D$ guarantees that the spectral radius of $Q$ is $\beta < 1$, so these resolvents exist.
* **State-by-state borrowing limits in infinite horizons.** Natural debt limits $\bar A^k = (I - Q)^{-1} y^k$ are the largest debts that agent $k$ could repay from its own endowment, and example 3 showed that they can be zero in states where an agent's future endowment is worthless.
* **No borrowing limits in finite horizons.** When the economy ends at $T$, no one can roll debt over past the last date, and nonnegative consumption implies bounds $[I + Q + \cdots + Q^{T-t}]\, y^k$ that we computed and showed never bind, so no separate debt limits need to be imposed.
* **The law of iterated values.** Multi-period Arrow prices compound as $Q^j$ in the same way that multi-period transition probabilities compound as $P^j$, and the backward recursion $S_t = I + Q S_{t+1}$ used in `RecurCompetitive` values payouts one period at a time.
* **State variable degeneracy.** Starting from zero financial wealth, each agent's wealth is a function of the Markov state alone and returns to zero whenever the state returns to $s_0$, as {ref}`ge_arrow_ex1` verified along a simulated path; {ref}`ge_arrow_ex3` showed that this strong form of degeneracy fails in finite horizons, where wealth also depends on calendar time.

### What the assumptions bought

Complete markets, identical CRRA preferences, and common beliefs together are what make prices independent of the wealth distribution and make wealth a function of the current state alone.

Relaxing any of them breaks at least one of these properties, which is the subject of several of the lectures listed below.

### How old this is

Before leaving the model, it is worth recording how much of it was in place at the start.

Both trading arrangements, the proof that they support the same allocation, and the observation that the sequential one presumes correct forecasts of future spot prices all appear in the paper that Arrow read in 1952, nine years before {cite:t}`muth1961` named the hypothesis that such forecasts embody.

Arrow did not use the term, and the dynamic programming arguments that make the equivalence transparent were not yet available to him.

His argument also has a knife edge that his unit of account conceals and that {cite:t}`hart1975` exposed: when securities pay goods rather than dollars, consumers' forecasts determine which risks the asset market can carry, and correct forecasts no longer pin down a unique, or even an existing, equilibrium.

The single good per date and history that we assumed throughout is exactly the case in which the forecasting requirement has no bite.

## Related lectures

This lecture assumes that all agents share beliefs about nature, and that their expectations of prices are correct.

Arrow's distinction between those two uses of probability organizes the first two entries below.

* {doc}`harrison_kreps` studies an economy in which agents disagree about probabilities and short sales are constrained.
* {doc}`likelihood_ratio_process_2` studies complete markets when agents hold different beliefs, in which case wealth shares drift with likelihood ratios instead of staying constant at $\alpha$.
* {doc}`lq_bewley_complete_markets` shows, in a linear-quadratic setting, how complete markets in Arrow securities deliver a time-invariant cross-section distribution of consumption.
* {doc}`ross_recovery` and {doc}`long_run_risk_operator` study what the Perron–Frobenius eigenvalue and eigenvector of a pricing kernel like $Q$ reveal.
* {doc}`hansen_singleton_1983` confronts Euler equations like those used here with data.

## Exercises

```{exercise-start}
:label: ge_arrow_ex1
```

This exercise verifies that the objects computed by `RecurCompetitive` constitute a recursive competitive equilibrium.

Use example 4 and start the economy in state 1.

1. Check that every agent's Euler equation holds at the allocation $c^k(s) = \alpha_k y(s)$ and the pricing kernel $Q$.

2. Check that agent $k$'s budget constraint $c^k(s) + \sum_{s'} Q(s' \mid s)\,\psi^k(s') = y^k(s) + \psi^k(s)$ holds in every state when the agent holds the portfolio $\hat a^k(s') = \psi^k(s')$.

3. Check that Arrow security markets clear, $\sum_k \psi^k(s) = 0$, and that no natural debt limit binds.

4. Simulate 200 periods of the Markov chain and confirm that each agent's financial wealth is exactly zero at every visit to the initial state.

```{exercise-end}
```

```{solution-start} ge_arrow_ex1
:class: dropdown
```

Here is one solution.

```{code-cell} ipython3
ex = RecurCompetitive(ex4.s, ex4.P, ex4.ys)

α = ex.wealth_distribution(s0_idx=0)
ψ = ex.continuation_wealths()[0]
c = np.outer(ex.y, α)

euler = max(np.abs(ex.Q - ex.β * ex.P * ex.u_prime(c[:, k])[None, :]
                   / ex.u_prime(c[:, k])[:, None]).max()
            for k in range(ex.K))
budget = np.abs(c + ex.Q @ ψ - ex.ys - ψ).max()

print(f"largest Euler equation residual    {euler:.1e}")
print(f"largest budget constraint residual {budget:.1e}")
print(f"largest net supply of any security {np.abs(ψ.sum(axis=1)).max():.1e}")
print(f"all natural debt limits slack:     {np.all(ψ + ex.A > 0)}")
```

```{code-cell} ipython3
rng = np.random.default_rng(0)
T_sim = 200
states = np.empty(T_sim, dtype=int)
states[0] = 0
for t in range(T_sim - 1):
    states[t+1] = rng.choice(ex.n, p=ex.P[states[t]])

a = ψ[states]            # financial wealth of each agent along the path
print(f"visits to the initial state: {np.sum(states == 0)}")
print(f"largest |wealth| at those visits: {np.abs(a[states == 0]).max():.1e}")
```

All four conditions hold to machine precision.

The last check is **state variable degeneracy** in action: financial wealth is a function of the Markov state alone, so it returns to its initial value of zero whenever the state does.

```{solution-end}
```

```{exercise-start}
:label: ge_arrow_ex2
```

This exercise connects the lecture to {doc}`ross_recovery`.

An outside observer sees the equilibrium pricing kernel $Q$ and the aggregate endowment $y$, but not $\beta$, $\gamma$, or the transition matrix $P$.

1. Show that the vector with components $y(\bar s_j)^{\gamma}$ is a right eigenvector of $Q$ with eigenvalue $\beta$.

2. Using example 4, compute the Perron–Frobenius eigenvalue and eigenvector of $Q$, and use them to recover $\beta$, $\gamma$, and $P$.

```{exercise-end}
```

```{solution-start} ge_arrow_ex2
:class: dropdown
```

Here is one solution.

Let $v_j = y_j^{\gamma}$.

Using {eq}`eq:Qformula`,

$$
\sum_j Q_{ij} v_j
= \sum_j \beta \left(\frac{y_j}{y_i}\right)^{-\gamma} P_{ij}\, y_j^{\gamma}
= \beta\, y_i^{\gamma} \sum_j P_{ij}
= \beta\, v_i ,
$$

because the rows of $P$ sum to one.

Since $v$ is strictly positive, it is the Perron–Frobenius eigenvector of the nonnegative matrix $Q$, and $\beta$ is its largest eigenvalue.

Given $\beta$ and $v$, the transition matrix is $P_{ij} = Q_{ij} v_j / (\beta v_i)$, and $\gamma$ is the slope of $\log v$ on $\log y$.

```{code-cell} ipython3
eigvals, eigvecs = np.linalg.eig(ex4.Q)
i = np.argmax(eigvals.real)
β_hat = eigvals[i].real
v = np.abs(eigvecs[:, i].real)

γ_hat = np.polyfit(np.log(ex4.y), np.log(v), 1)[0]
P_hat = ex4.Q * v[None, :] / (β_hat * v[:, None])

print(f"recovered β = {β_hat:.6f}   (true {ex4.β})")
print(f"recovered γ = {γ_hat:.6f}   (true {ex4.γ})")
print(f"max |P_hat - P| = {np.abs(P_hat - ex4.P).max():.1e}")
```

Recovery is exact because the equilibrium kernel has precisely the **transition independence** structure studied in {doc}`ross_recovery`: $Q = \beta D^{-1} P D$, with $D$ built from the marginal utility of the aggregate endowment.

```{solution-end}
```

```{exercise-start}
:label: ge_arrow_ex3
```

In the infinite-horizon economy, continuation wealth depends only on the Markov state.

This exercise shows that in the finite-horizon economy it depends on calendar time as well.

1. For the finite-horizon version of example 1 with $T = 10$ and initial state 1, report $\psi_t^1(\bar s_1)$ for $t = 0, 1, \ldots, 10$.

2. Explain why agent 1's wealth does not return to zero when the Markov state returns to $\bar s_1$.

3. Compute the wealth distribution $\alpha$ for horizons $T = 1, \ldots, 300$ and show that it converges to the infinite-horizon distribution at a geometric rate close to $\beta$.

```{exercise-end}
```

```{solution-start} ge_arrow_ex3
:class: dropdown
```

Here is one solution.

```{code-cell} ipython3
s = np.array([0, 1])
P = np.array([[.5, .5],
              [.5, .5]])
ys = np.array([[1., 0.],
               [0., 1.]])

ex_T = RecurCompetitive(s, P, ys, T=10)
ex_T.wealth_distribution(s0_idx=0)
ψ_T = ex_T.continuation_wealths()

print("ψ_t^1(s_1), t = 0,...,10:", ψ_T[:, 0, 0].round(4))
```

From {eq}`eq:vv`, $\psi_t^k = \bigl[I + Q + \cdots + Q^{T-t}\bigr]\bigl[\alpha_k y - y^k\bigr]$.

The initial condition pins $\psi_0^k(\bar s_1) = 0$, but at a later date $t$ fewer periods remain, so the geometric sum is truncated sooner and $\psi_t^k(\bar s_1) \neq 0$.

In the infinite horizon the sum is never truncated, which is why wealth there depends on the state alone.

```{code-cell} ipython3
α_inf = RecurCompetitive(s, P, ys).wealth_distribution(s0_idx=0)[0]

T_grid = np.arange(1, 301)
gaps = np.array([abs(RecurCompetitive(s, P, ys, T=T).wealth_distribution(s0_idx=0)[0]
                     - α_inf)
                 for T in T_grid])

fig, ax = plt.subplots()
ax.semilogy(T_grid, gaps, lw=2, label=r'$|\alpha_1(T) - \alpha_1(\infty)|$')
ax.semilogy(T_grid, gaps[0] * ex_T.β ** (T_grid - 1), '--', lw=1.5,
            label=r'reference slope $\beta^{T}$')
ax.set_xlabel('horizon $T$')
ax.legend()
plt.show()
```

The gap shrinks geometrically, at a rate governed by the spectral radius of $Q$, which equals $\beta$.

```{solution-end}
```

```{exercise-start}
:label: ge_arrow_ex4
```

This exercise asks you to verify, in a setting with more than one good, the equivalence that {ref}`sec-rational-expectations` describes for our one-good economy.

Following {cite:t}`arrow1964`, consider a two-period pure exchange economy with $n$ consumers, $S$ states, and $C$ goods.

Consumer $i$ is endowed with the state-contingent vector $\omega_i = (\omega_{i1}, \ldots, \omega_{iS})$ and ranks state-contingent consumption plans $x_i$ by

$$
U_i(x_i) = \sum_{s=1}^S u_i(x_{is})\, \pi_s,
\qquad
u_i(x) = a_{i1}\sqrt{x_1} + a_{i2}\sqrt{x_2}
$$

In a **contingent claims equilibrium**, a price vector $p^*$ for claims to goods in states and an allocation $\{x_i^*\}$ are such that each consumer maximizes $U_i$ subject to $p^* \cdot x_i = p^* \cdot \omega_i$ and all $SC$ markets clear.

In the **sequential** arrangement, consumer $i$ enters a securities market with an endowment $M_i$ of dollars, a unit of security $s$ pays one dollar if state $s$ occurs and sells for $q_s$, and after the state is realized goods trade on spot markets at prices $\bar p_s$.

1. Show that consumer $i$'s spot problem has indirect utility $V_i(y, \bar p_s) = \sqrt{y}\, G_i(\bar p_s)$ with $G_i(\bar p_s) = \bigl(\sum_c a_{ic}^2/\bar p_{sc}\bigr)^{1/2}$, and derive the within-state demands. At which point does the definition of a **rational expectations equilibrium** require consumers to forecast $\bar p_s$ correctly?

2. Compute the contingent claims equilibrium of the two-state, two-good, two-consumer economy specified in the solution below.

3. Define

$$
M_i = p^* \cdot \omega_i,
\qquad
q_s^* = \frac{\sum_i p_s^* \cdot x_{is}^*}{\sum_\sigma \sum_i p_\sigma^* \cdot x_{i\sigma}^*},
\qquad
\bar p_s = \frac{p_s^*}{q_s^*},
\qquad
y_{is}^* = \bar p_s \cdot x_{is}^*
$$

   and verify that $\{y_i^*, x_i^*\}, q^*, \bar p$ is a rational expectations equilibrium: check that $\sum_s q_s^* y_{is}^* = M_i$, that $\sum_i y_{is}^* = \sum_i M_i$ for every $s$, that the spot demands reproduce $x_i^*$, and that the portfolio satisfies its own first-order conditions.

4. Verify that $\sum_s q_s^* = 1$, and explain why a consumer who could hold cash instead of securities would arbitrage against any price system that violated this equality.

5. Going the other way, define $p_s^* = q_s^* \bar p_s$ and verify that you recover the contingent claims equilibrium.

6. Finally, show that an incorrect forecast is not an equilibrium: perturb the forecast of relative prices in state $1$, recompute portfolios, and compare the spot prices that then clear the goods markets with the forecast.

```{exercise-end}
```

```{solution-start} ge_arrow_ex4
:class: dropdown
```

*Part 1.* With $u_i(x) = \sum_c a_{ic}\sqrt{x_c}$ and a spot budget $\bar p_s \cdot x = y$, the first-order conditions give $x_c \propto a_{ic}^2/\bar p_{sc}^2$, so

$$
x_{c} = \frac{a_{ic}^2/\bar p_{sc}^2}{\sum_{c'} a_{ic'}^2/\bar p_{sc'}} \, y,
\qquad
V_i(y, \bar p_s) = \sqrt{y}\,\Bigl(\sum_c a_{ic}^2/\bar p_{sc}\Bigr)^{1/2}
$$

The portfolio problem is $\max \sum_s \pi_s V_i(y_{is}, \bar p_s)$ subject to $\sum_s q_s y_{is} = M_i$.

The forecast $\bar p_s$ enters here, through $G_i(\bar p_s)$, before any state is realized.

The definition of equilibrium requires the forecast used in this portfolio problem to be the price vector that later clears the spot market in state $s$.

Since $V_i$ is increasing in $y$ and concave, and $G_i$ depends on $\bar p_s$, a consumer who misforecasts relative prices chooses a portfolio that is optimal for prices that never materialize.

*Parts 2 to 5.*

```{code-cell} ipython3
π = np.array([0.4, 0.6])                 # probabilities of the two states
A = np.array([[2.0, 1.0],                # taste parameters a_{ic}
              [1.0, 3.0]])
ω = np.array([[[2.0, 0.5], [0.5, 1.5]],  # consumer 1, by state
              [[1.0, 1.5], [1.0, 1.0]]]) # consumer 2, by state
Ω = ω.sum(axis=0)                        # aggregate endowment by state
S, C = Ω.shape

def spot_demand(p_s, y, a):
    "Within-state demands given spot prices p_s and spending y."
    w = a**2 / p_s**2
    return w * y / (w * p_s).sum()

def G(p_s, a):
    "Indirect utility coefficient: V = sqrt(y) G(p_s)."
    return np.sqrt((a**2 / p_s).sum())

def spending_shares(p, a):
    "Shares of time 0 wealth spent in each state."
    g = np.array([π[s] * G(p[s], a) for s in range(S)])
    return g**2 / (g**2).sum()

def cc_excess_demand(p_flat):
    "Excess demand in the SC contingent claims markets."
    p = p_flat.reshape((S, C))
    exc = np.zeros((S, C))
    for i in range(len(A)):
        M_i = (p * ω[i]).sum()
        share = spending_shares(p, A[i])
        for s in range(S):
            exc[s] += spot_demand(p[s], share[s] * M_i, A[i])
    return (exc - Ω).ravel()

sol = root(cc_excess_demand, np.ones(S * C), tol=1e-13)
p_star = sol.x.reshape((S, C))
p_star = p_star / p_star[0, 0]           # numeraire: good 1 in state 1

M = np.array([(p_star * ω[i]).sum() for i in range(len(A))])
x_star = np.array([[spot_demand(p_star[s], spending_shares(p_star, A[i])[s] * M[i],
                                A[i]) for s in range(S)] for i in range(len(A))])

print("p* =\n", p_star.round(5))
print("largest excess demand:", np.abs(cc_excess_demand(p_star.ravel())).max())
print("M =", M.round(5))
print("x1* =\n", x_star[0].round(5), "\nx2* =\n", x_star[1].round(5))
```

Now build the sequential arrangement from these objects.

```{code-cell} ipython3
q = np.array([(p_star[s] * Ω[s]).sum() for s in range(S)]) / M.sum()
p_bar = p_star / q[:, None]
y_star = np.array([[p_bar[s] @ x_star[i, s] for s in range(S)]
                   for i in range(len(A))])

print("q* =", q.round(5), " and Σ_s q*_s =", q.sum().round(10))
print("p_bar =\n", p_bar.round(5))
print("y* =\n", y_star.round(5))

print("\nΣ_s q*_s y*_is = M_i:          ", np.allclose(y_star @ q, M))
print("Σ_i y*_is = Σ_i M_i for each s:", np.allclose(y_star.sum(axis=0), M.sum()))
print("spot demands reproduce x*:     ",
      all(np.allclose(spot_demand(p_bar[s], y_star[i, s], A[i]), x_star[i, s])
          for i in range(len(A)) for s in range(S)))

# portfolio first-order condition: sqrt(y_is) q_s / (π_s G_i) is constant across s
for i in range(len(A)):
    ratio = np.sqrt(y_star[i]) * q / np.array([π[s] * G(p_bar[s], A[i])
                                               for s in range(S)])
    print(f"consumer {i+1} portfolio FOC, across states:", ratio.round(8))

print("\nq*_s p_bar_s recovers p*:", np.allclose(q[:, None] * p_bar, p_star))
```

The securities prices sum to one because a unit of each security, bought in equal numbers, pays one dollar for sure.

A portfolio of one unit of every security is therefore a riskless claim to one dollar, so it must cost one dollar.

If $\sum_s q_s < 1$, a consumer could buy that bundle and hold a sure dollar for less than a dollar; if $\sum_s q_s > 1$, selling the bundle and holding cash would do the same in reverse.

*Part 6.*

```{code-cell} ipython3
def realized_spot_prices(y):
    "Spot prices that clear each state's goods markets given dollar wealth y."
    out = np.zeros((S, C))
    for s in range(S):
        def excess(p_s):
            return sum(spot_demand(p_s, y[i, s], A[i])
                       for i in range(len(A))) - Ω[s]
        out[s] = root(excess, np.ones(C), tol=1e-13).x
    return out

def portfolios(forecast):
    "Optimal portfolios when consumers forecast the spot prices in forecast."
    y = np.zeros((len(A), S))
    for i in range(len(A)):
        g = np.array([π[s] * G(forecast[s], A[i]) for s in range(S)])
        y[i] = M[i] * (g / q)**2 / ((g / q)**2 * q).sum()
    return y

print("correct forecast reproduces itself:",
      np.allclose(realized_spot_prices(y_star), p_bar))

p_wrong = p_bar.copy()
p_wrong[0] = p_bar[0] * np.array([1.5, 1.0])     # misforecast relative prices
p_realized = realized_spot_prices(portfolios(p_wrong))

print(f"\nforecast relative price in state 1: "
      f"{p_wrong[0, 0] / p_wrong[0, 1]:.4f}")
print(f"realized relative price in state 1: "
      f"{p_realized[0, 0] / p_realized[0, 1]:.4f}")
```

With the correct forecast, the prices that clear the spot markets are the prices that consumers used when choosing portfolios, so the forecast is confirmed and the sequential allocation is the contingent claims allocation.

With the incorrect forecast, consumers carry the wrong dollar wealth into each state, and the prices that then clear the spot markets are not the prices they forecast.

The plans are feasible and the securities market clears, but the economy is not in a rational expectations equilibrium.

Note that only *relative* prices within a state need to be forecast correctly.

Scaling all of state $s$'s spot prices by a constant is absorbed by the security price $q_s$, which is why the normalization of $\bar p_s$ is harmless.

```{solution-end}
```

```{exercise-start}
:label: ge_arrow_ex5
```

In {ref}`ge_arrow_ex4` the securities pay dollars, so the set of state-contingent wealth distributions that a portfolio can deliver does not depend on what consumers forecast.

This exercise, which follows {cite:t}`hart1975`, asks what happens when securities pay *goods* instead.

There are two dates.

Securities trade at the first date; at the second date a state $s \in \{1,2\}$ is realized and $C=2$ goods trade on spot markets.

Two consumers care only about consumption at the second date and assign probability $1/2$ to each state, with

$$
u^1(x) = 2^{2.5} \sqrt{x_1} + 2 \sqrt{x_2},
\qquad
u^2(x) = 2 \sqrt{x_1} + 2^{2.5} \sqrt{x_2}
$$

and endowments

$$
\omega_{11} = \left( \tfrac{5}{2}, \tfrac{50}{21} \right), \quad
\omega_{21} = \left( \tfrac{1}{2}, \tfrac{13}{21} \right), \quad
\omega_{12} = \left( \tfrac{13}{21}, \tfrac{1}{2} \right), \quad
\omega_{22} = \left( \tfrac{50}{21}, \tfrac{5}{2} \right)
$$

where $\omega_{is}$ is consumer $i$'s endowment in state $s$.

The aggregate endowment of each good is $3$ in each state, so all risk is idiosyncratic, and the two consumers and the two states are mirror images of each other.

1. Suppose first that no securities are available. Compute the spot market equilibrium in each state.

2. Now let two securities trade at the first date. A unit of security $1$ delivers one unit of good $1$ if state $1$ occurs and two units of good $1$ if state $2$ occurs; a unit of security $2$ delivers two units of good $2$ if state $1$ occurs and one unit of good $2$ if state $2$ occurs. Show that if consumers forecast the spot prices of part 1, the two securities become perfect substitutes, so that no portfolio of them can move wealth between states. Conclude that this is a rational expectations equilibrium, and compute each consumer's expected utility.

3. Show that the forecast $\hat p_1 = \hat p_2 = (1,1)$ instead makes the two securities span, so that the equilibrium allocation is the contingent claims allocation $x_{1s} = (8/3, 1/3)$, $x_{2s} = (1/3, 8/3)$. Verify that this forecast is also confirmed, compute expected utilities, and find the portfolio that consumer $1$ uses.

4. Compare the two equilibria, and explain why no market force selects the better one.

5. Change only the dividends: let security $1$ deliver one unit of good $1$ in *both* states and security $2$ one unit of good $2$ in both states. Show that now no rational expectations equilibrium exists.

```{exercise-end}
```

```{solution-start} ge_arrow_ex5
:class: dropdown
```

*Part 1.*

```{code-cell} ipython3
a = np.array([[2**2.5, 2.0],      # consumer 1
              [2.0, 2**2.5]])     # consumer 2
ω_h = np.array([[[5 / 2, 50 / 21], [13 / 21, 1 / 2]],      # consumer 1, by state
                [[1 / 2, 13 / 21], [50 / 21, 5 / 2]]])     # consumer 2, by state
Ω_h = ω_h.sum(axis=0)
def u(x, a_i):
    "Expected utility of the bundle x for a consumer with tastes a_i."
    return a_i @ np.sqrt(x)

print("aggregate endowment by state:\n", Ω_h.round(6))

p_hat = np.array([[2.0, 1.0], [1.0, 2.0]])      # conjectured spot prices
x_auto = np.array([[spot_demand(p_hat[s], p_hat[s] @ ω_h[i, s], a[i])
                    for s in range(2)] for i in range(2)])

print("\nno securities, state by state:")
for s in range(2):
    print(f"  state {s+1}: p = {p_hat[s]}, "
          f"x_1 = {(21 * x_auto[0, s]).round(4)}/21, "
          f"x_2 = {(21 * x_auto[1, s]).round(4)}/21, "
          f"markets clear: {np.allclose(x_auto[:, s].sum(axis=0), Ω_h[s])}")
```

The spot equilibrium in state $1$ has prices proportional to $(2,1)$ and in state $2$ proportional to $(1,2)$.

Each consumer eats mostly the good that it likes more.

In each state the good favored by the richer consumer is the expensive one, because the aggregate endowment is the same in both states and only the distribution of wealth differs.

*Part 2.* The dollar payoff of a security is the forecast price of the good it delivers, times the quantity delivered.

```{code-cell} ipython3
def payoff_matrix(forecast, dividends):
    """
    Dollar payoffs of securities, given forecast spot prices.

    dividends[f, s] is the bundle that security f delivers in state s.
    """
    return np.array([[forecast[s] @ dividends[f, s] for s in range(2)]
                     for f in range(2)])

div_b = np.array([[[1.0, 0.0], [2.0, 0.0]],     # security 1: good 1, 1 then 2 units
                  [[0.0, 2.0], [0.0, 1.0]]])    # security 2: good 2, 2 then 1 unit

Z = payoff_matrix(p_hat, div_b)
print("payoff matrix with the part 1 forecast:\n", Z)
print("rank:", np.linalg.matrix_rank(Z))

EU_auto = [sum(0.5 * u(x_auto[i, s], a[i]) for s in range(2)) for i in range(2)]
print(f"\nexpected utilities with no risk sharing: {EU_auto[0]:.4f}, {EU_auto[1]:.4f}")
```

Both securities pay $(2,2)$, so they are perfect substitutes and must bear the same price.

No portfolio of them can move wealth between the two states, so no securities trade, the realized spot prices are those of part 1, and the forecast is confirmed.

This is a rational expectations equilibrium in which the asset market is effectively incomplete.

*Part 3.*

```{code-cell} ipython3
p_span = np.array([[1.0, 1.0], [1.0, 1.0]])
Z_span = payoff_matrix(p_span, div_b)
print("payoff matrix with the forecast (1,1):\n", Z_span)
print("rank:", np.linalg.matrix_rank(Z_span))

x_cm = np.array([[8 / 3, 1 / 3], [1 / 3, 8 / 3]])    # complete markets allocation
print("\nallocation clears:", np.allclose(x_cm.sum(axis=0), Ω_h[0]))
for i in range(2):
    ratio = x_cm[i, 1] / x_cm[i, 0]
    print(f"  consumer {i+1}: x_2/x_1 = {ratio:.4f}, "
          f"(a_2/a_1)^2 = {(a[i, 1] / a[i, 0])**2:.4f}")

EU_cm = [u(x_cm[i], a[i]) for i in range(2)]
print(f"\nexpected utilities with complete markets: {EU_cm[0]:.4f}, {EU_cm[1]:.4f}")

wealth = np.array([p_span[s] @ ω_h[0, s] for s in range(2)])
z1 = np.linalg.solve(Z_span.T, p_span[0] @ x_cm[0] - wealth)
print(f"\nconsumer 1 state-contingent wealth: {(42 * wealth).round(3)}/42")
print(f"consumer 1 portfolio: {(42 * z1).round(3)}/42")
```

With this forecast the payoff vectors $(1,2)$ and $(2,1)$ are linearly independent, so the securities span and the allocation must be the contingent claims allocation.

Because each consumer's demands satisfy $x_2/x_1 = (a_2/a_1)^2$ at prices proportional to $(1,1)$, those are indeed the spot prices that clear the markets, so this forecast is confirmed too.

Consumer $1$ reaches the allocation by buying $79/42$ units of security $1$ and selling $79/42$ units of security $2$, a swap that costs nothing because the two securities bear the same price, and that converts its state-contingent wealth of $(205/42, 47/42)$ into $(3,3)$.

Consumer $2$ holds the mirror-image portfolio.

*Part 4.*

```{code-cell} ipython3
print(f"{'':12}{'no risk sharing':>18}{'complete markets':>19}")
for i in range(2):
    print(f"consumer {i+1}: {EU_auto[i]:>17.4f}{EU_cm[i]:>19.4f}")
```

Both consumers strictly prefer the second equilibrium, so it Pareto dominates the first.

Yet both are rational expectations equilibria: in each, consumers forecast correctly and markets clear.

Hart's point is that no market force selects the better one, because the trading opportunities available to a consumer depend on forecasts of spot prices, and a competitive consumer takes those forecasts as given.

A consumer cannot unilaterally make the two securities span, since spanning is a property of prices that no individual chooses.

*Part 5.*

```{code-cell} ipython3
div_e = np.array([[[1.0, 0.0], [1.0, 0.0]],     # security 1: one unit of good 1
                  [[0.0, 1.0], [0.0, 1.0]]])    # security 2: one unit of good 2

# each candidate forecast implies a span, the span implies an allocation,
# and the allocation implies the spot prices that actually clear markets
for name, forecast, implied in [
    ("(2,1) and (1,2)", p_hat,  p_span),
    ("(1,1) and (1,1)", p_span, p_hat),
]:
    Z_e = payoff_matrix(forecast, div_e)
    rank = np.linalg.matrix_rank(Z_e)
    spans = "spans" if rank == 2 else "does not span"
    allocation = "complete markets" if rank == 2 else "no risk sharing"
    print(f"forecast {name}: rank {rank}, {spans}")
    print(f"  implied allocation: {allocation}")
    print(f"  spot prices it implies: {implied[0]} and {implied[1]}")
    print(f"  forecast confirmed: {np.allclose(forecast, implied)}\n")
```

With these dividends the payoff vectors are $(\hat p_{11}, \hat p_{21})$ and $(\hat p_{12}, \hat p_{22})$, so the securities span if and only if the two states' price vectors are not proportional.

Suppose a forecast makes them proportional.

Then no risk can be traded, so the realized spot prices are those of part 1, which are *not* proportional, and the forecast is wrong.

Suppose instead that a forecast makes them non-proportional.

Then the securities span, so the allocation is the complete markets allocation of part 3, whose spot prices *are* proportional, and again the forecast is wrong.

So no rational expectations equilibrium exists, even though preferences are strictly concave, endowments are strictly positive, and there are no transaction costs.

Nothing like parts 2 to 5 can happen in the sequential economy of this lecture, where there is one good per date and history and the one-period Arrow securities are denominated in that good.

There the span of the asset market is $n$-dimensional whatever consumers forecast, and the only forecast that matters is of the pricing kernel $Q$ itself.

```{solution-end}
```
