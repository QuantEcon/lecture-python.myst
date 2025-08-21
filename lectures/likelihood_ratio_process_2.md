---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(likelihood_ratio_process_2)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Heterogeneous Beliefs and Financial Markets

```{contents} Contents
:depth: 2
```

(overview)=
## Overview

A likelihood ratio process lies behind Lawrence Blume and David Easley's answer to their question
''If you're so smart, why aren't you rich?'' {cite}`blume2006if`.  

Blume and Easley constructed formal models to study how differences of opinions about probabilities governing risky income processes would influence outcomes and be reflected in prices of stocks, bonds, and insurance policies that individuals use to share and hedge risks.

```{note}
{cite}`alchian1950uncertainty` and {cite}`friedman1953essays` conjectured that, by rewarding traders with more realistic probability models, competitive markets in financial securities put wealth in the hands of better informed traders and help 
make prices of risky assets reflect realistic probability assessments. 
```


Here we'll provide an example that illustrates basic components of Blume and Easley's analysis. 

We'll focus only on their analysis of an environment with complete markets in which trades in all conceivable risky securities are possible. 

We'll study two alternative arrangements:

* perfect socialism in which individuals surrender their endowments of consumption goods each period to a central planner who then dictatorially allocates those goods
* a decentralized system of competitive markets in which selfish price-taking individuals voluntarily trade with each other in competitive markets 

The fundamental theorems of welfare economics will apply and assure us that these two arrangements end up producing exactly the same allocation of consumption goods to individuals **provided** that the social planner assigns an appropriate set of **Pareto weights**.

```{note}
You can learn about how the two welfare theorems are applied in modern macroeconomic models in {doc}`this lecture on a planning problem <cass_koopmans_1>` and {doc}`this lecture on a related competitive equilibrium <cass_koopmans_2>`.   {doc}`This quantecon lecture <ge_arrow>` presents a recursive formulation of  complete markets models with homogeneous beliefs.
```




Let's start by importing some Python tools.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from numba import vectorize, jit, prange
from math import gamma
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
import pandas as pd
from IPython.display import display, Math
import quantecon as qe
```

## Review: Likelihood Ratio Processes

We'll begin by reminding ourselves definitions and properties of likelihood ratio processes.  

A nonnegative random variable $W$ has one of two probability density functions, either
$f$ or $g$.

Before the beginning of time, nature once and for all decides whether she will draw a sequence of IID draws from either
$f$ or $g$.

We will sometimes let $q$ be the density that nature chose once and for all, so
that $q$ is either $f$ or $g$, permanently.

Nature knows which density it permanently draws from, but we the observers do not.

We know both $f$ and $g$ but we don't know which density nature
chose.

But we want to know.

To do that, we use observations.

We observe a sequence $\{w_t\}_{t=1}^T$ of $T$ IID draws that we know came from either $f$ or $g$.

We want to use these observations to infer whether nature chose $f$ or $g$.

A **likelihood ratio process** is a useful tool for this task.

To begin, we define a key component of a likelihood ratio process, namely, the time $t$ likelihood ratio as the random variable

$$
\ell (w_t)=\frac{f\left(w_t\right)}{g\left(w_t\right)},\quad t\geq1.
$$

We assume that $f$ and $g$ both put positive probabilities on the
same intervals of possible realizations of the random variable $W$.

That means that under the $g$ density, $\ell (w_t)=
\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}$
is a nonnegative random variable with mean $1$.

A **likelihood ratio process** for sequence
$\left\{ w_{t}\right\} _{t=1}^{\infty}$ is defined as

$$
L\left(w^{t}\right)=\prod_{i=1}^{t} \ell (w_i),
$$

where $w^t=\{ w_1,\dots,w_t\}$ is a history of
observations up to and including time $t$.

Sometimes for shorthand we'll write $L_t = L(w^t)$.

Notice that the likelihood process satisfies the *recursion* 

$$
L(w^t) = \ell (w_t) L (w^{t-1}) .
$$

The likelihood ratio and its logarithm are key tools for making
inferences using a classic frequentist approach due to Neyman and
Pearson {cite}`Neyman_Pearson`.

To help us appreciate how things work, the following Python code evaluates $f$ and $g$ as two different
Beta distributions, then computes and simulates an associated likelihood
ratio process by generating a sequence $w^t$ from one of the two
probability distributions, for example, a sequence of IID draws from $g$.

```{code-cell} ipython3
# Parameters in the two Beta distributions.
F_a, F_b = 1, 1
G_a, G_b = 3, 1.2

@vectorize
def p(x, a, b):
    r = gamma(a + b) / (gamma(a) * gamma(b))
    return r * x** (a-1) * (1 - x) ** (b-1)

# The two density functions.
f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))
```

```{code-cell} ipython3
@jit
def simulate(a, b, T=50, N=500):
    '''
    Generate N sets of T observations of the likelihood ratio,
    return as N x T matrix.
    '''

    l_arr = np.empty((N, T))

    for i in range(N):
        for j in range(T):
            w = np.random.beta(a, b)
            l_arr[i, j] = f(w) / g(w)

    return l_arr
```

## Blume and Easley's Setting

Let the random variable $s_t \in (0,1)$ at time $t =0, 1, 2, \ldots$ be distributed according to the same Beta distribution with parameters 
$\theta = \{\theta_1, \theta_2\}$.

We'll denote this probability density as

$$
\pi(s_t|\theta)
$$

Below, we'll often just write $\pi(s_t)$ instead of $\pi(s_t|\theta)$ to save space.

Let $s_t \equiv y_t^1$ be the endowment of a nonstorable consumption good that a person we'll call "agent 1" receives at time $t$.

Let a history $s^t = [s_t, s_{t-1}, \ldots, s_0]$ be a sequence of i.i.d. random variables with joint distribution

$$
\pi_t(s^t) = \pi(s_t) \pi(s_{t-1}) \cdots \pi(s_0)
$$ 

So in our example, the history $s^t$ is a comprehensive record of agent $1$'s endowments of the consumption good from time $0$ up to time $t$.  

If agent $1$ were to live on an island by himself, agent $1$'s consumption $c^1(s_t)$ at time $t$ is 

$$c^1(s_t) = y_t^1 = s_t. $$

But in our model, agent 1 is not alone.

## Nature and Agents' Beliefs

Nature draws i.i.d. sequences $\{s_t\}_{t=0}^\infty$ from $\pi_t(s^t)$.

* so $\pi$ without a superscript is nature's model 
* but in addition to nature, there are other entities inside our model -- artificial people that we call "agents"
* each agent has a sequence of probability distributions over $s^t$ for $t=0, \ldots$ 
* agent $i$ thinks that nature draws i.i.d. sequences $\{s_t\}_{t=0}^\infty$ from $\{\pi_t^i(s^t)\}_{t=0}^\infty$
   * agent $i$ is mistaken unless $\pi_t^i(s^t) = \pi_t(s^t)$

```{note}
A **rational expectations** model would set $\pi_t^i(s^t) = \pi_t(s^t)$ for all agents $i$.
```

There are two agents named $i=1$ and $i=2$.

At time $t$, agent $1$ receives an endowment

$$
y_t^1 = s_t 
$$

of a nonstorable consumption good, while agent $2$ receives an endowment of 

$$
y_t^2 = 1 - s_t 
$$

The aggregate endowment of the consumption good is

$$
y_t^1 + y_t^2 = 1
$$

at each date $t \geq 0$. 

At date $t$ agent $i$ consumes $c_t^i(s^t)$ of the good.

A (non wasteful) feasible allocation of the aggregate endowment of $1$ each period satisfies

$$
c_t^1 + c_t^2 = 1 .
$$

## A Socialist Risk-Sharing Arrangement

In order to share risks, a benevolent social planner  dictates a history-dependent consumption allocation that takes the form of a sequence of functions 

$$
c_t^i = c_t^i(s^t)
$$

that satisfy

$$
c_t^1(s^t) + c_t^2(s^t) = 1  
$$ (eq:feasibility)

for all $s^t$ for all $t \geq 0$. 

To design a socially optimal allocation, the social planner wants to know what agent $1$ believes about the endowment sequence and how they feel about bearing risks.

As for the endowment sequences, agent $i$ believes that nature draws i.i.d. sequences from joint densities 

$$
\pi_t^i(s^t) = \pi(s_t)^i \pi^i(s_{t-1}) \cdots \pi^i(s_0)
$$ 

As for attitudes toward bearing risks, agent $i$ has a one-period utility function

$$
u(c_t^i) = \ln (c_t^i)
$$

with marginal utility of consumption in period $i$

$$
u'(c_t^i) = \frac{1}{c_t^i}
$$

Putting its beliefs about its random endowment sequence and its attitudes toward bearing risks together, agent $i$ has intertemporal utility function 

$$
V^i = \sum_{t=0}^{\infty} \sum_{s^t} \delta^t u(c_t^i(s^t)) \pi_t^i(s^t) ,
$$ (eq:objectiveagenti)

where $\delta \in (0,1)$ is an intertemporal discount factor, and $u(\cdot)$ is a strictly increasing, concave one-period utility function.


## Social Planner's Allocation Problem

The benevolent dictator has all the information it requires to choose a consumption allocation that maximizes the social welfare criterion 

$$
W = \lambda V^1 + (1-\lambda) V^2
$$ (eq:welfareW)

where $\lambda \in [0,1]$ is a Pareto weight that tells how much the planner likes agent $1$ and $1 - \lambda$ is a Pareto weight that tells how much the social planner likes agent $2$.  

Setting $\lambda = .5$ expresses ''egalitarian'' social preferences. 

Notice how social welfare criterion {eq}`eq:welfareW` takes into account both agents' preferences as represented by formula {eq}`eq:objectiveagenti`.

This means that the social planner knows and respects

* each agent's one period utility function $u(\cdot) = \ln(\cdot)$
* each agent $i$'s probability model $\{\pi_t^i(s^t)\}_{t=0}^\infty$

Consequently, we anticipate that these objects will appear in the social planner's rule for allocating the aggregate endowment each period. 


First-order necessary conditions for maximizing welfare criterion {eq}`eq:welfareW` subject to the feasibility constraint {eq}`eq:feasibility` are 

$$\frac{\pi_t^2(s^t)}{\pi_t^1(s^t)} \frac{(1/c_t^2(s^t))}{(1/c_t^1(s^t))} = \frac{\lambda}{1-\lambda}$$

which can be rearranged to become




$$
\frac{c_t^1(s^t)}{c_t^2(s^t)} = \frac{\lambda}{1-\lambda} l_t(s^t)
$$ (eq:allocationrule0)


where

$$ l_t(s^t) = \frac{\pi_t^1(s^t)}{\pi_t^2(s^t)} $$

is the likelihood ratio of agent 1's joint density to agent 2's joint density. 

Using 

$$c_t^1(s^t) + c_t^2(s^t) = 1$$

we can rewrite allocation rule {eq}`eq:allocationrule0` as 



$$\frac{c_t^1(s^t)}{1 - c_t^1(s^t)} = \frac{\lambda}{1-\lambda} l_t(s^t)$$

or 

$$c_t^1(s^t) = \frac{\lambda}{1-\lambda} l_t(s^t)(1 - c_t^1(s^t))$$

which implies that the social planner's allocation rule is

$$
c_t^1(s^t) = \frac{\lambda l_t(s^t)}{1-\lambda + \lambda l_t(s^t)}
$$ (eq:allocationrule1)

If we define a temporary or **continuation Pareto weight** process as 

$$
\lambda_t(s^t) = \frac{\lambda l_t(s^t)}{1-\lambda + \lambda l_t(s^t)},
$$

then we can represent the social planner's allocation rule as

$$
c_t^1(s^t) = \lambda_t(s^t) .
$$




## If You're So Smart, $\ldots$ 


Let's compute some values of limiting allocations {eq}`eq:allocationrule1` for some interesting possible limiting
values of the likelihood ratio process $l_t(s^t)$:

 $$l_\infty (s^\infty)= 1; \quad c_\infty^1 = \lambda$$
 
  * In the above case, both agents are equally smart (or equally not smart) and the consumption allocation stays put at a $\lambda, 1 - \lambda$ split between the two agents. 

$$l_\infty (s^\infty) = 0; \quad c_\infty^1 = 0$$

* In the above case, agent 2 is ''smarter'' than agent 1, and agent 1's share of the aggregate endowment converges to zero.




$$l_\infty (s^\infty)= \infty; \quad c_\infty^1 = 1$$

* In the above case, agent 1 is smarter than agent 2, and agent 1's share of the aggregate endowment converges to 1. 

```{note}
These three cases are somehow telling us about how relative **wealths** of the agents evolve as time passes.
* when the two agents are equally smart and $\lambda \in (0,1)$, agent 1's wealth share stays at $\lambda$ perpetually.
* when agent 1 is smarter and $\lambda \in (0,1)$, agent 1 eventually "owns" the entire continuation endowment and agent 2 eventually "owns" nothing.
* when agent 2 is smarter and $\lambda \in (0,1)$, agent 2 eventually "owns" the entire continuation endowment and agent 1 eventually "owns" nothing.
Continuation wealths can be defined precisely after we introduce a competitive equilibrium **price** system below.
```


Soon we'll do some simulations that will shed further light on possible outcomes.

But before we do that, let's take a detour and study some  "shadow prices" for the social planning problem that can readily be
converted to "equilibrium prices" for a competitive equilibrium. 

Doing this will allow us to connect our analysis with an argument of {cite}`alchian1950uncertainty` and {cite}`friedman1953essays` that competitive market processes can make prices of risky assets better reflect realistic probability assessments. 



## Competitive Equilibrium Prices 

Two fundamental welfare theorems for general equilibrium models lead us to anticipate that there is  a connection between the allocation that solves the social planning problem we have been studying and the allocation in a  **competitive equilibrium**  with complete markets in history-contingent commodities.

```{note}
For the two welfare theorems and their history, see   <https://en.wikipedia.org/wiki/Fundamental_theorems_of_welfare_economics>.
Again, for applications to a classic  macroeconomic growth  model, see {doc}`this lecture on a planning problem <cass_koopmans_1>`  and {doc}`this lecture on a related competitive equilibrium <cass_koopmans_2>` 
```

Such a connection prevails for our model.  

We'll sketch it now.

In a competitive equilibrium, there is no social planner that dictatorially collects everybody's endowments and then reallocates them.

Instead, there is a comprehensive centralized   market that meets at one point in time.

There are **prices** at which price-taking agents can buy or sell whatever goods that they want.  

Trade is multilateral in the sense that that there is a "Walrasian auctioneer" who lives outside the model and whose job is to verify that
each agent's budget constraint is satisfied.  

That budget constraint involves the total value of the agent's endowment stream and the total value of its consumption stream.  

These values are computed at price vectors that the agents take as given -- they are "price-takers" who assume that they can buy or sell
whatever quantities that they want at those prices.  

Suppose that at time $-1$, before time $0$ starts, agent $i$ can purchase one unit $c_t(s^t)$ of consumption at time $t$ after history
$s^t$ at price $p_t(s^t)$.  

Notice that there is (very long) **vector** of prices.  

 * there is one price $p_t(s^t)$ for each history $s^t$ at every date $t = 0, 1, \ldots, $. 
 * so there are as many prices as there are histories and dates.

These prices determined at time $-1$ before the economy starts.

The market meets once at time $-1$.

At times $t =0, 1, 2, \ldots$ trades made at time $-1$ are executed.

 

* in the background, there is an "enforcement" procedure that forces agents to carry out the exchanges or "deliveries"  that they agreed to at time $-1$.



We want to study how agents'  beliefs influence equilibrium prices.  

Agent $i$ faces a **single** intertemporal budget constraint

$$
\sum_{t=0}^\infty\sum_{s^t} p_t(s^t) c_t^i (s^t) \leq \sum_{t=0}^\infty\sum_{s^t} p_t(s^t) y_t^i (s^t)
$$ (eq:budgetI)

According to budget constraint {eq}`eq:budgetI`,  trade is **multilateral** in the following  sense

* we can imagine  that  agent $i$ first sells his random endowment stream $\{y_t^i (s^t)\}$ and then uses the proceeds (i.e., his "wealth") to purchase a random consumption stream $\{c_t^i (s^t)\}$. 

Agent $i$ puts a Lagrange multiplier $\mu_i$ on {eq}`eq:budgetI` and once-and-for-all chooses a consumption plan $\{c^i_t(s^t)\}_{t=0}^\infty$
to maximize criterion {eq}`eq:objectiveagenti` subject to budget constraint {eq}`eq:budgetI`.

This means that the agent $i$  chooses many objects, namely, $c_t^i(s^t)$ for all $s^t$ for $t = 0, 1, 2, \ldots$.


For convenience, let's remind ourselves of criterion $V^i$ defined in {eq}`eq:objectiveagenti`:  

$$
V^i = \sum_{t=0}^{\infty} \sum_{s^t} \delta^t u(c_t^i(s^t)) \pi_t^i(s^t)
$$

First-order necessary conditions for maximizing objective $V^i$ defined in {eq}`eq:objectiveagenti` with respect to $c_t^i(s^t)$ are 

$$
\delta^t u'(c^i_t(s^t)) \pi_t^i(s^t) = \mu_i p_t(s^t) ,
$$ 

which we can rearrange to obtain

$$
p_t(s^t) = \frac{ \delta^t \pi_t^i(s^t)}{\mu_i c^i_t(s^t)}   
$$ (eq:priceequation1)

for $i=1,2$.  

If we divide equation {eq}`eq:priceequation1` for agent $1$ by the appropriate  version of equation {eq}`eq:priceequation1` for agent 2, use
$c^2_t(s^t) = 1 - c^1_t(s^t)$, and do some algebra, we'll obtain

$$
c_t^1(s^t) = \frac{\mu_1 l_t(s^t)}{\mu_2 + \mu_1 l_t(s^t)} .
$$ (eq:allocationce)

We now engage in an extended "guess-and-verify" exercise that involves matching objects in our competitive equilibrium with objects in 
our social planning problem.  

* we'll match consumption allocations in the planning problem with equilibrium consumption allocations in the competitive equilibrium
* we'll match "shadow" prices in the planning problem with competitive equilibrium prices. 

Notice that if we set $\mu_1 = 1-\lambda$ and $\mu_2 = \lambda$, then  formula {eq}`eq:allocationce` agrees with formula
{eq}`eq:allocationrule1`.  

  * doing this amounts to choosing a **numeraire** or normalization for the price system $\{p_t(s^t)\}_{t=0}^\infty$

```{note}
For information about how a numeraire  must be chosen to pin down the absolute price level in a model like ours that determines only
relative prices,   see <https://en.wikipedia.org/wiki/Num%C3%A9raire>.
```

If we substitute formula  {eq}`eq:allocationce` for $c_t^1(s^t)$ into formula {eq}`eq:priceequation1` and rearrange, we obtain

$$
p_t(s^t) = \frac{\delta^t}{\lambda(1-\lambda)} \pi_t^2(s^t) \bigl[1 - \lambda + \lambda l_t(s^t)\bigr]
$$ 

or

$$
p_t(s^t) = \frac{\delta^t}{\lambda(1-\lambda)}  \bigl[(1 - \lambda) \pi_t^2(s^t) + \lambda \pi_t^1(s^t)\bigr]
$$ (eq:pformulafinal)

According to formula {eq}`eq:pformulafinal`, we have the following possible limiting cases:

* when $l_\infty = 0$, $c_\infty^1 = 0 $ and tails of competitive equilibrium prices reflect agent $2$'s probability model $\pi_t^2(s^t)$ according to $p_t(s^t) \propto \delta^t \pi_t^2(s^t) $
* when $l_\infty = \infty$, $c_\infty^1 = 1 $ and tails of competitive equilibrium prices reflect agent $1$'s probability model $\pi_t^1(s^t)$ according to $p_t(s^t) \propto \delta^t \pi_t^1(s^t) $
* for small $t$'s, competitive equilibrium prices reflect both agents' probability models.  

## Simulations 

Now let's implement some simulations when agent $1$ believes marginal density 

$$\pi^1(s_t) = f(s_t) $$

and agent $2$ believes marginal density 

$$ \pi^2(s_t) = g(s_t) $$

where $f$ and $g$ are Beta distributions like ones that  we used in earlier  sections of this lecture.

Meanwhile, we'll assume that  nature believes a  marginal density

$$
\pi(s_t) = h(s_t) 
$$

where $h(s_t)$ is perhaps a  mixture of $f$ and $g$.


First, we write a function to compute the likelihood ratio process

```{code-cell} ipython3
def compute_likelihood_ratios(sequences, f, g):
    """Compute likelihood ratios and cumulative products."""
    l_ratios = f(sequences) / g(sequences)
    L_cumulative = np.cumprod(l_ratios, axis=1)
    return l_ratios, L_cumulative
```

Let's compute the Kullback–Leibler discrepancies by quadrature
integration.

```{code-cell} ipython3
def compute_KL(f, g):
    """
    Compute KL divergence KL(f, g)
    """
    integrand = lambda w: f(w) * np.log(f(w) / g(w))
    val, _ = quad(integrand, 1e-5, 1-1e-5)
    return val
```

We also create a helper function to compute KL divergence with respect to a reference distribution $h$

```{code-cell} ipython3
def compute_KL_h(h, f, g):
    """
    Compute KL divergence with reference distribution h
    """

    Kf = compute_KL(h, f)
    Kg = compute_KL(h, g)

    return Kf, Kg
```

Let's  write a Python function that computes agent 1's  consumption share

```{code-cell} ipython3
def simulate_blume_easley(sequences, f_belief=f, g_belief=g, λ=0.5):
    """Simulate Blume-Easley model consumption shares."""
    l_ratios, l_cumulative = compute_likelihood_ratios(sequences, f_belief, g_belief)
    c1_share = λ * l_cumulative / (1 - λ + λ * l_cumulative)
    return l_cumulative, c1_share
```

Now let's use this function to generate sequences in which  

* nature draws from $f$ each period, or 
* nature draws from $g$ each period, or
* nature flips a fair coin each period to decide whether to draw from $f$ or $g$

```{code-cell} ipython3
λ = 0.5
T = 100
N = 10000

# Nature follows f, g, or mixture
s_seq_f = np.random.beta(F_a, F_b, (N, T))
s_seq_g = np.random.beta(G_a, G_b, (N, T))

h = jit(lambda x: 0.5 * f(x) + 0.5 * g(x))
model_choices = np.random.rand(N, T) < 0.5
s_seq_h = np.empty((N, T))
s_seq_h[model_choices] = np.random.beta(F_a, F_b, size=model_choices.sum())
s_seq_h[~model_choices] = np.random.beta(G_a, G_b, size=(~model_choices).sum())

l_cum_f, c1_f = simulate_blume_easley(s_seq_f)
l_cum_g, c1_g = simulate_blume_easley(s_seq_g)
l_cum_h, c1_h = simulate_blume_easley(s_seq_h)
```

Before looking at the figure below, have some fun by guessing whether agent 1 or agent 2 will have a larger and larger consumption share as time passes in our three cases. 

To make better guesses,  let's visualize instances of the likelihood ratio processes in  the three cases.

```{code-cell} ipython3
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

titles = ["Nature = f", "Nature = g", "Nature = mixture"]
data_pairs = [(l_cum_f, c1_f), (l_cum_g, c1_g), (l_cum_h, c1_h)]

for i, ((l_cum, c1), title) in enumerate(zip(data_pairs, titles)):
    # Likelihood ratios
    ax = axes[0, i]
    for j in range(min(50, l_cum.shape[0])):
        ax.plot(l_cum[j, :], alpha=0.3, color='blue')
    ax.set_yscale('log')
    ax.set_xlabel('time')
    ax.set_ylabel('Likelihood ratio $l_t$')
    ax.set_title(title)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)

    # Consumption shares
    ax = axes[1, i]
    for j in range(min(50, c1.shape[0])):
        ax.plot(c1[j, :], alpha=0.3, color='green')
    ax.set_xlabel('time')
    ax.set_ylabel("Agent 1's consumption share")
    ax.set_ylim([0, 1])
    ax.axhline(y=λ, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

In the left panel, nature chooses $f$. Agent 1's consumption reaches $1$ very quickly.

In the middle panel, nature chooses $g$. Agent 1's consumption ratio tends to move towards $0$ but not as fast as in the first case.

In the right panel, nature flips coins each period. We see a very similar pattern to the processes in the left panel.

The figures in the top panel remind us of the discussion in [this section](KL_link).

We invite readers to revisit [that section](llr_h) and try to infer the relationships among $D_{KL}(f\|g)$, $D_{KL}(g\|f)$, $D_{KL}(h\|f)$, and $D_{KL}(h\|g)$.


Let's compute values of KL divergence

```{code-cell} ipython3
shares = [np.mean(c1_f[:, -1]), np.mean(c1_g[:, -1]), np.mean(c1_h[:, -1])]
Kf_g, Kg_f = compute_KL(f, g), compute_KL(g, f)
Kf_h, Kg_h = compute_KL_h(h, f, g)

print(f"Final shares: f={shares[0]:.3f}, g={shares[1]:.3f}, mix={shares[2]:.3f}")
print(f"KL divergences: \nKL(f,g)={Kf_g:.3f}, KL(g,f)={Kg_f:.3f}")
print(f"KL(h,f)={Kf_h:.3f}, KL(h,g)={Kg_h:.3f}")
```

We find that $KL(f,g) > KL(g,f)$ and $KL(h,g) > KL(h,f)$.

The first inequality tells us that the average "surprise" from having belief $g$ when nature chooses $f$ is greater than the "surprise" from having  belief $f$ when nature chooses $g$.

This explains the difference between the first two panels we noted above.

The second inequality tells us that agent 1's belief distribution $f$ is closer to nature's pick than agent 2's belief $g$.

+++

To make this idea more concrete, let's compare two cases:

- agent 1's belief distribution $f$ is close to agent 2's belief distribution $g$;
- agent 1's belief distribution $f$ is far from agent 2's belief distribution $g$.


We use the two distributions visualized below

```{code-cell} ipython3
def plot_distribution_overlap(ax, x_range, f_vals, g_vals, 
                            f_label='f', g_label='g', 
                            f_color='blue', g_color='red'):
    """Plot two distributions with their overlap region."""
    ax.plot(x_range, f_vals, color=f_color, linewidth=2, label=f_label)
    ax.plot(x_range, g_vals, color=g_color, linewidth=2, label=g_label)
    
    overlap = np.minimum(f_vals, g_vals)
    ax.fill_between(x_range, 0, overlap, alpha=0.3, color='purple', label='Overlap')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.legend()
    
# Define close and far belief distributions
f_close = jit(lambda x: p(x, 1, 1))
g_close = jit(lambda x: p(x, 1.1, 1.05))

f_far = jit(lambda x: p(x, 1, 1))
g_far = jit(lambda x: p(x, 3, 1.2))

# Visualize the belief distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

x_range = np.linspace(0.001, 0.999, 200)

# Close beliefs
f_close_vals = [f_close(x) for x in x_range]
g_close_vals = [g_close(x) for x in x_range]
plot_distribution_overlap(ax1, x_range, f_close_vals, g_close_vals,
                         f_label='f (Beta(1, 1))', g_label='g (Beta(1.1, 1.05))')
ax1.set_title(f'Close Beliefs')

# Far beliefs
f_far_vals = [f_far(x) for x in x_range]
g_far_vals = [g_far(x) for x in x_range]
plot_distribution_overlap(ax2, x_range, f_far_vals, g_far_vals,
                         f_label='f (Beta(1, 1))', g_label='g (Beta(3, 1.2))')
ax2.set_title(f'Far Beliefs')

plt.tight_layout()
plt.show()
```

Let's draw the same consumption ratio plots as above for agent 1.

We replace the simulation paths with median and percentiles to make the figure cleaner.

Staring at the figure below, can we infer the relation between $KL(f,g)$ and $KL(g,f)$?

From the right panel, can we infer the relation between $KL(h,g)$ and $KL(h,f)$?

```{code-cell} ipython3
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
nature_params = {'close': [(1, 1), (1.1, 1.05), (2, 1.5)],
                 'far':   [(1, 1), (3, 1.2),   (2, 1.5)]}
nature_labels = ["Nature = f", "Nature = g", "Nature = h"]
colors = {'close': 'blue', 'far': 'red'}

threshold = 1e-5  # "close to zero" cutoff

for row, (f_belief, g_belief, label) in enumerate([
                        (f_close, g_close, 'close'),
                        (f_far, g_far, 'far')]):
    
    for col, nature_label in enumerate(nature_labels):
        params = nature_params[label][col]
        s_seq = np.random.beta(params[0], params[1], (1000, 200))
        _, c1 = simulate_blume_easley(s_seq, f_belief, g_belief, λ)
        
        median_c1 = np.median(c1, axis=0)
        p10, p90 = np.percentile(c1, [10, 90], axis=0)
        
        ax = axes[row, col]
        color = colors[label]
        ax.plot(median_c1, color=color, linewidth=2, label='Median')
        ax.fill_between(range(len(median_c1)), p10, p90, alpha=0.3, color=color, label='10–90%')
        ax.set_xlabel('time')
        ax.set_ylabel("Agent 1's share")
        ax.set_ylim([0, 1])
        ax.set_title(nature_label)
        ax.axhline(y=λ, color='gray', linestyle='--', alpha=0.5)
        below = np.where(median_c1 < threshold)[0]
        above = np.where(median_c1 > 1-threshold)[0]
        if below.size > 0: first_zero = (below[0], True)
        elif above.size > 0: first_zero = (above[0], False)
        else: first_zero = None
        if first_zero is not None:
            ax.axvline(x=first_zero[0], color='black', linestyle='--',
                       alpha=0.7, 
                       label=fr'Median $\leq$ {threshold}' if first_zero[1]
                       else fr'Median $\geq$ 1-{threshold}')
        ax.legend()

plt.tight_layout()
plt.show()
```

Holding to our guesses, let's calculate the four values

```{code-cell} ipython3
# Close case
Kf_g, Kg_f = compute_KL(f_close, g_close), compute_KL(g_close, f_close)
Kf_h, Kg_h = compute_KL_h(h, f_close, g_close)

print(f"KL divergences (close): \nKL(f,g)={Kf_g:.3f}, KL(g,f)={Kg_f:.3f}")
print(f"KL(h,f)={Kf_h:.3f}, KL(h,g)={Kg_h:.3f}")

# Far case
Kf_g, Kg_f = compute_KL(f_far, g_far), compute_KL(g_far, f_far)
Kf_h, Kg_h = compute_KL_h(h, f_far, g_far)

print(f"KL divergences (far): \nKL(f,g)={Kf_g:.3f}, KL(g,f)={Kg_f:.3f}")
print(f"KL(h,f)={Kf_h:.3f}, KL(h,g)={Kg_h:.3f}")
```

We find that in the first case, $KL(f,g) \approx KL(g,f)$ and both are relatively small, so although either agent 1 or agent 2 will eventually consume everything, convergence displayed in the first two panels on the top is pretty slow.

In the first two panels at the bottom, we see convergence occurring faster (as indicated by the black dashed line) because the divergence gaps $KL(f, g)$ and $KL(g, f)$ are larger.

Since $KL(f,g) > KL(g,f)$, we see faster convergence in the first panel at the bottom when nature chooses $f$ than in the second panel where nature chooses $g$.

This ties in nicely with {eq}`eq:kl_likelihood_link`.



## Related Lectures

Complete markets models with homogeneous beliefs, a kind often used in macroeconomics and finance,  are studied in this quantecon lecture {doc}`ge_arrow`.

{cite}`blume2018case` discuss a paternalistic  case against complete markets.  Their analysis assumes that a social planner should disregard  individuals preferences in the sense that it should disregard the subjective belief components of their preferences.  

Likelihood processes play an important role in Bayesian learning, as described in {doc}`likelihood_bayes` and as applied in {doc}`odu`.

Likelihood ratio processes appear again in {doc}`advanced:additive_functionals`. 



{doc}`ge_arrow`




## Exercises

```{exercise}
:label: lr_ex3

Starting from {eq}`eq:priceequation1`, show that the competitive equilibrium prices can be expressed as

$$
p_t(s^t) = \frac{\delta^t}{\lambda(1-\lambda)} \pi_t^2(s^t) \bigl[1 - \lambda + \lambda l_t(s^t)\bigr]
$$

```

```{solution-start} lr_ex3
:class: dropdown
```

Starting from

$$
p_t(s^t) = \frac{\delta^t \pi_t^i(s^t)}{\mu_i c_t^i(s^t)}, \qquad i=1,2.
$$

Since both expressions equal the same price, we can equate them

$$
\frac{\pi_t^1(s^t)}{\mu_1 c_t^1(s^t)} = \frac{\pi_t^2(s^t)}{\mu_2 c_t^2(s^t)}
$$

Rearranging gives

$$
\frac{c_t^1(s^t)}{c_t^2(s^t)} = \frac{\mu_2}{\mu_1} l_t(s^t)
$$

where $l_t(s^t) \equiv \pi_t^1(s^t)/\pi_t^2(s^t)$ is the likelihood ratio process.

Using $c_t^2(s^t) = 1 - c_t^1(s^t)$:

$$
\frac{c_t^1(s^t)}{1 - c_t^1(s^t)} = \frac{\mu_2}{\mu_1} l_t(s^t)
$$

Solving for $c_t^1(s^t)$

$$
c_t^1(s^t) = \frac{\mu_2 l_t(s^t)}{\mu_1 + \mu_2 l_t(s^t)}
$$


The planner's solution gives

$$
c_t^1(s^t) = \frac{\lambda l_t(s^t)}{1 - \lambda + \lambda l_t(s^t)}
$$

To match agent 1's choice in a competitive equilibrium with the planner's choice for agent 1,  the following equality must hold

$$
\frac{\mu_2}{\mu_1} = \frac{\lambda}{1 - \lambda}
$$

Hence we have

$$
\mu_1 = 1 - \lambda, \qquad \mu_2 = \lambda
$$


With $\mu_1 = 1-\lambda$ and $c_t^1(s^t) = \frac{\lambda l_t(s^t)}{1-\lambda+\lambda l_t(s^t)}$,
we have

$$
\begin{aligned}
p_t(s^t) &= \frac{\delta^t \pi_t^1(s^t)}{(1-\lambda) c_t^1(s^t)} \\
&= \frac{\delta^t \pi_t^1(s^t)}{(1-\lambda)} \cdot \frac{1 - \lambda + \lambda l_t(s^t)}{\lambda l_t(s^t)} \\
&= \frac{\delta^t \pi_t^1(s^t)}{(1-\lambda)\lambda l_t(s^t)} \bigl[1 - \lambda + \lambda l_t(s^t)\bigr].
\end{aligned}
$$

Since $\pi_t^1(s^t) = l_t(s^t) \pi_t^2(s^t)$, we have

$$
\begin{aligned}
p_t(s^t) &= \frac{\delta^t l_t(s^t) \pi_t^2(s^t)}{(1-\lambda)\lambda l_t(s^t)} \bigl[1 - \lambda + \lambda l_t(s^t)\bigr] \\
&= \frac{\delta^t \pi_t^2(s^t)}{(1-\lambda)\lambda} \bigl[1 - \lambda + \lambda l_t(s^t)\bigr] \\
&= \frac{\delta^t}{\lambda(1-\lambda)} \pi_t^2(s^t) \bigl[1 - \lambda + \lambda l_t(s^t)\bigr].
\end{aligned}
$$

```{solution-end}
```

```{exercise}
:label: lr_ex4

In this exercise, we'll study two agents, each of whom updates its posterior probability as
data arrive.

 * each agent applies Bayes' law in the way studied  in {doc}`likelihood_bayes`.

The following  two models are on the table

$$
f(s^t) = f(s_1) f(s_2) \cdots f(s_t) 
$$

and 

$$
g(s^t) = g(s_1) g(s_2) \cdots g(s_t)  
$$

as is an associated likelihood ratio process

$$
L(s^t) = \frac{f(s^t)}{g(s^t)} .
$$

Let $\pi_0 \in (0,1)$ be a prior probability and 

$$
\pi_t = \frac{ \pi_0 L(s^t)}{ \pi_0 L(s^t) + (1-\pi_0) } .
$$

Each of our two agents deploys its own version of the mixture model

$$
m(s^t) = \pi_t f(s^t) + (1- \pi_t) g(s^t)
$$ (eq:be_mix_model)

We'll endow each type of consumer with model {eq}`eq:be_mix_model`.

 * The two agents share the same $f$ and $g$, but
 * they have different initial priors, say $\pi_0^1$ and $\pi_0^2$

Thus, consumer $i$'s probability model is 

$$
m^i(s^t) = \pi^i_t f(s^t) + (1- \pi^i_t) g(s^t)
$$ (eq:prob_model)

We now  hand probability models {eq}`eq:prob_model` for $i=1,2$ to the social planner.

We want to deduce allocation $c^i(s^t), i = 1,2$, and watch what happens when

  * nature's model is $f$
  * nature's model is $g$

We  expect that  consumers will eventually learn the "truth", but that one of them will learn faster.  

To explore things, please  set $f \sim \text{Beta}(1.5, 1)$ and $g \sim \text{Beta}(1, 1.5)$.

Please write Python code that answers the following questions.

 *  How do  consumption shares evolve? 
 *  Which agent learns faster when nature follows $f$? 
 *  Which agent learns faster when nature follows $g$?
 *  How does a difference in initial priors $\pi_0^1$ and $\pi_0^2$ affect the convergence speed?



```

```{solution-start} lr_ex4
:class: dropdown
```



First, let's write helper functions that compute model components including each agent's subjective belief function.

```{code-cell} ipython3
def bayesian_update(π_0, L_t):
    """
    Bayesian update of belief probability given likelihood ratio.
    """
    return (π_0 * L_t) / (π_0 * L_t + (1 - π_0))

def mixture_density_belief(s_seq, f_func, g_func, π_seq):
    """
    Compute the mixture density beliefs m^i(s^t) for agent i.
    """
    f_vals = f_func(s_seq)
    g_vals = g_func(s_seq)
    return π_seq * f_vals + (1 - π_seq) * g_vals
```

Now let's write code that simulates the Blume-Easley model with our two agents.

```{code-cell} ipython3
def simulate_learning_blume_easley(sequences, f_belief, g_belief, 
                                        π_0_1, π_0_2, λ=0.5):
    """
    Simulate Blume-Easley model with learning agents.
    """
    N, T = sequences.shape
    
    # Initialize arrays to store results
    π_1_seq = np.full((N, T), np.nan)
    π_2_seq = np.full((N, T), np.nan)
    c1_share = np.full((N, T), np.nan)
    l_agents_seq = np.full((N, T), np.nan)

    π_1_seq[:, 0] = π_0_1
    π_2_seq[:, 0] = π_0_2
    
    for n in range(N):
        # Initialize cumulative likelihood ratio for beliefs
        L_cumul = 1.0
        
        # Initialize likelihood ratio between agent densities
        l_agents_cumul = 1.0
        
        for t in range(1, T):
            s_t = sequences[n, t]
            
            # Compute likelihood ratio for this observation
            l_t = f_belief(s_t) / g_belief(s_t)
            
            # Update cumulative likelihood ratio
            L_cumul *= l_t
            
            # Bayesian update of beliefs
            π_1_t = bayesian_update(π_0_1, L_cumul)
            π_2_t = bayesian_update(π_0_2, L_cumul)
            
            # Store beliefs
            π_1_seq[n, t] = π_1_t
            π_2_seq[n, t] = π_2_t
            
            # Compute mixture densities for each agent
            m1_t = π_1_t * f_belief(s_t) + (1 - π_1_t) * g_belief(s_t)
            m2_t = π_2_t * f_belief(s_t) + (1 - π_2_t) * g_belief(s_t)
            
            # Update cumulative likelihood ratio between agents
            l_agents_cumul *= (m1_t / m2_t)
            l_agents_seq[n, t] = l_agents_cumul
            
            # c_t^1(s^t) = λ * l_t(s^t) / (1 - λ + λ * l_t(s^t))
            # where l_t(s^t) is the cumulative likelihood ratio between agents
            c1_share[n, t] = λ * l_agents_cumul / (1 - λ + λ * l_agents_cumul)
    
    return {
        'π_1': π_1_seq,
        'π_2': π_2_seq, 
        'c1_share': c1_share,
        'l_agents': l_agents_seq
    }
```

Let's run simulations for different scenarios.

We use $\lambda = 0.5$, $T=40$, and $N=1000$.

```{code-cell} ipython3
λ = 0.5
T = 40
N = 1000

F_a, F_b = 1.5, 1
G_a, G_b = 1, 1.5

f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))
```

We'll  start with different initial priors $\pi^i_0 \in (0, 1)$ and widen the gap between them.

```{code-cell} ipython3
# Different initial priors
π_0_scenarios = [
    (0.3, 0.7),
    (0.7, 0.3),
    (0.1, 0.9),
]
```

Now we can run simulations for different scenarios

```{code-cell} ipython3
# Nature follows f
s_seq_f = np.random.beta(F_a, F_b, (N, T))

# Nature follows g
s_seq_g = np.random.beta(G_a, G_b, (N, T)) 

results_f = {}
results_g = {}

for i, (π_0_1, π_0_2) in enumerate(π_0_scenarios):
    # When nature follows f
    results_f[i] = simulate_learning_blume_easley(
            s_seq_f, f, g, π_0_1, π_0_2, λ)
    # When nature follows g  
    results_g[i] = simulate_learning_blume_easley(
            s_seq_g, f, g, π_0_1, π_0_2, λ)
```

Let's visualize the results

```{code-cell} ipython3
def plot_learning_results(results, π_0_scenarios, nature_type, truth_value):
    """
    Plot beliefs and consumption shares for learning agents.
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    
    scenario_labels = [
        rf'$\pi_0^1 = {π_0_1}, \pi_0^2 = {π_0_2}$'
        for π_0_1, π_0_2 in π_0_scenarios
    ]
    
    for row, (scenario_idx, scenario_label) in enumerate(
                zip(range(3), scenario_labels)):

        res = results[scenario_idx]
        
        # Plot beliefs
        ax = axes[row, 0]
        π_1_med = np.median(res['π_1'], axis=0)
        π_2_med = np.median(res['π_2'], axis=0) 
        ax.plot(π_1_med, 'C0', label=r'agent 1', linewidth=2)
        ax.plot(π_2_med, 'C1', label=r'agent 2', linewidth=2)
        ax.axhline(y=truth_value, color='gray', linestyle='--', 
                   alpha=0.5, label=f'truth ({nature_type})')
        ax.set_title(f'Beliefs when nature = {nature_type}\n{scenario_label}')
        ax.set_ylabel(r'median $\pi_i^t$')
        ax.set_ylim([-0.05, 1.05])
        ax.legend()
        
        # Plot consumption shares
        ax = axes[row, 1]
        c1_med = np.median(res['c1_share'], axis=0)
        ax.plot(c1_med, 'g-', linewidth=2, label='median')
        ax.axhline(y=0.5, color='gray', linestyle='--', 
                   alpha=0.5)
        ax.set_title(f'Agent 1 consumption share (Nature = {nature_type})')
        ax.set_ylabel('consumption share')
        ax.set_ylim([0, 1])
        ax.legend()
        
        # Add x-labels
        for col in range(2):
            axes[row, col].set_xlabel('$t$')
    
    plt.tight_layout()
    return fig, axes
```

Now we'll plot outcome when nature follows f:

```{code-cell} ipython3
fig_f, axes_f = plot_learning_results(
                results_f, π_0_scenarios, 'f', 1.0)
plt.show()
```

We can see that the agent with the  more accurate belief gets higher consumption share.

Moreover, the further apart are  initial beliefs, the longer it takes for the consumption ratio to converge.

The longer  it takes for the "less accurate" agent to learn, the lower its ultimate consumption share. 

Now let's plot outcomes when nature follows g:

```{code-cell} ipython3
fig_g, axes_g = plot_learning_results(results_g, π_0_scenarios, 'g', 0.0)
plt.show()
```

We observe  symmetrical outcomes.

```{solution-end}
```

```{exercise}
:label: lr_ex5

In the previous exercise, we purposefully  set the two beta distributions to be relatively close to each other.

That made it challenging to distinguish  the  distributions.

Now let's study outcomes when  the  distributions are further apart.

Let's set $f \sim \text{Beta}(2, 5)$ and $g \sim \text{Beta}(5, 2)$.

Please use  the Python code you  have written to study outcomes. 
```

```{solution-start} lr_ex5
:class: dropdown
```

Here is one solution

```{code-cell} ipython3
λ = 0.5
T = 40
N = 1000

F_a, F_b = 2, 5
G_a, G_b = 5, 2

f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))

π_0_scenarios = [
    (0.3, 0.7),
    (0.7, 0.3),
    (0.1, 0.9),
]

s_seq_f = np.random.beta(F_a, F_b, (N, T))
s_seq_g = np.random.beta(G_a, G_b, (N, T)) 

results_f = {}
results_g = {}

for i, (π_0_1, π_0_2) in enumerate(π_0_scenarios):
    # When nature follows f
    results_f[i] = simulate_learning_blume_easley(
            s_seq_f, f, g, π_0_1, π_0_2, λ)
    # When nature follows g  
    results_g[i] = simulate_learning_blume_easley(
            s_seq_g, f, g, π_0_1, π_0_2, λ)
```

Now let's visualize the results

```{code-cell} ipython3
fig_f, axes_f = plot_learning_results(results_f, π_0_scenarios, 'f', 1.0)
plt.show()
```

```{code-cell} ipython3
fig_g, axes_g = plot_learning_results(results_g, π_0_scenarios, 'g', 0.0)
plt.show()
```

Evidently, because the two distributions are further apart, it is easier to distinguish them.

So learning occurs more quickly.


So do consumption shares.

```{solution-end}
```

```{exercise}
:label: lr_ex6

Two agents with different beliefs about three possible models.

Assume $f(x) \geq 0$, $g(x) \geq 0$, and $h(x) \geq 0$ for $x \in X$ with:
- $\int_X f(x) dx = 1$
- $\int_X g(x) dx = 1$ 
- $\int_X h(x) dx = 1$

We'll consider two agents:
* Agent 1: $\pi^g_0 = 1 - \pi^f_0$, $\pi^f_0 \in (0,1)$ (believes only in models $f$ and $g$)
* Agent 2: $\pi^g_0 = \pi^f_0 = 1/3$ (equally weights all three models)

Set $h = \pi^f_0 f + (1-\pi^f_0) g$ (a mixture of $f$ and $g$).

Simulate and visualize the evolution of consumption allocations when:
* Nature permanently draws from $f$
* Nature permanently draws from $g$

Use the existing code structure to implement this simulation and observe how the allocation evolves over time.

```

```{solution-start} lr_ex6
:class: dropdown
```

Let's implement this three-model case with two agents having different beliefs.

First, let's define $f$ and $g$ far apart, with $h$ being a mixture of $f$ and $g$

```{code-cell} ipython3
F_a, F_b = 1, 1
G_a, G_b = 3, 1.2
λ = 0.5 
π_f_0 = 0.4 

f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))
h = jit(lambda x: π_f_0 * f(x) + (1 - π_f_0) * g(x))
```

Now we can define the belief updating for the three-agent model

```{code-cell} ipython3
@jit(parallel=True)
def compute_posterior_three_models(
    s_seq, f_func, g_func, h_func, π_f_0, π_g_0):
    """
    Compute posterior probabilities for three models.
    """
    N, T = s_seq.shape
    π_h_0 = 1 - π_f_0 - π_g_0
    
    π_f = np.zeros((N, T))
    π_g = np.zeros((N, T))
    π_h = np.zeros((N, T))
    
    for n in prange(N):
        # Initialize with priors
        π_f[n, 0] = π_f_0
        π_g[n, 0] = π_g_0
        π_h[n, 0] = π_h_0
        
        # Compute cumulative likelihoods
        f_cumul = 1.0
        g_cumul = 1.0
        h_cumul = 1.0
        
        for t in range(1, T):
            s_t = s_seq[n, t]
            
            # Update cumulative likelihoods
            f_cumul *= f_func(s_t)
            g_cumul *= g_func(s_t)
            h_cumul *= h_func(s_t)
            
            # Compute posteriors using Bayes' rule
            denominator = π_f_0 * f_cumul + π_g_0 * g_cumul + π_h_0 * h_cumul
            
            π_f[n, t] = π_f_0 * f_cumul / denominator
            π_g[n, t] = π_g_0 * g_cumul / denominator
            π_h[n, t] = π_h_0 * h_cumul / denominator
    
    return π_f, π_g, π_h
```

Let's also write the simulation code following the same idea as in the previous exercises

```{code-cell} ipython3
@jit(parallel=True)
def simulate_three_model_allocation(s_seq, f_func, g_func, h_func, 
                                   π_f_1, π_g_1, π_f_2, π_g_2, λ=0.5):
    """
    Simulate two agents having beliefs over three models.
    """
    
    N, T = s_seq.shape
    
    # Compute posteriors for both agents
    π_f_1_seq, π_g_1_seq, π_h_1_seq = compute_posterior_three_models(
        s_seq, f_func, g_func, h_func, π_f_1, π_g_1)
    π_f_2_seq, π_g_2_seq, π_h_2_seq = compute_posterior_three_models(
        s_seq, f_func, g_func, h_func, π_f_2, π_g_2)
    
    # Compute consumption shares
    c1_share = np.zeros((N, T))
    
    for n in prange(N):
        l_agents_cumul = 1.0  # Initialize likelihood ratio between agents
        
        for t in range(T):
            # Agent 1's mixture density
            m1_t = (π_f_1_seq[n, t] * f_func(s_seq[n, t]) + 
                   π_g_1_seq[n, t] * g_func(s_seq[n, t]) + 
                   π_h_1_seq[n, t] * h_func(s_seq[n, t]))
            
            # Agent 2's mixture density
            m2_t = (π_f_2_seq[n, t] * f_func(s_seq[n, t]) + 
                   π_g_2_seq[n, t] * g_func(s_seq[n, t]) + 
                   π_h_2_seq[n, t] * h_func(s_seq[n, t]))
            
            # Update likelihood ratio between agents
            if t > 0:
                l_agents_cumul *= (m1_t / m2_t)
            
            # Consumption share for agent 1
            c1_share[n, t] = λ * l_agents_cumul / (1 - λ + λ * l_agents_cumul)
    
    return c1_share, π_f_1_seq, π_g_1_seq, π_h_1_seq, π_f_2_seq, π_g_2_seq, π_h_2_seq
```

The following code cell defines a plotting function to show the convergence of beliefs and consumption ratio

```{code-cell} ipython3

def plot_three_model_results(c1_data, π_data, nature_labels, λ=0.5, 
                            agent_labels=None, title_suffix="", figsize=(12, 10)):
    """
    Create plots for three-model exercises.
    """
    n_scenarios = len(nature_labels)
    fig, axes = plt.subplots(2, n_scenarios, figsize=figsize)
    if n_scenarios == 1:
        axes = axes.reshape(2, 1)
    
    colors = ['blue', 'green', 'orange']  # For different nature scenarios
    
    for i, (nature_label, c1, π_tuple) in enumerate(
                zip(nature_labels, c1_data, π_data)):
        πf1, πg1, πh1, πf2, πg2, πh2 = π_tuple

        ax = axes[i, 0]
        ax.plot(np.median(πf1, axis=0), 'C0-', linewidth=2)
        ax.plot(np.median(πg1, axis=0), 'C0--', linewidth=2)
        ax.plot(np.median(πh1, axis=0), 'C0:', linewidth=2)
        ax.plot(np.median(πf2, axis=0), 'C1-', linewidth=2)
        ax.plot(np.median(πg2, axis=0), 'C1--', linewidth=2)
        ax.plot(np.median(πh2, axis=0), 'C1:', linewidth=2)

        # Truth indicator
        truth_val = 1.0 if nature_label == 'f' else (
                        1.0 if nature_label == 'g' else 0.0)
        ax.axhline(y=truth_val, color='grey', linestyle='-.', alpha=0.7)
        
        ax.set_title(f'Beliefs when Nature = {nature_label}')
        ax.set_xlabel('$t$')
        ax.set_ylabel(r'median $\pi(\cdot)$')
        ax.set_ylim([-0.01, 1.01])

        if i == 0:
            from matplotlib.lines import Line2D
            
            # Agent colors legend
            agent_elements = [
                Line2D([0], [0], color='C0', linewidth=2, label='agent 1'),
                Line2D([0], [0], color='C1', linewidth=2, label='agent 2')
            ]
            agent_legend = ax.legend(handles=agent_elements, loc='upper left')
            
            # Line styles legend
            style_elements = [
                Line2D([0], [0], color='black', 
                        linestyle='-', label='π(f)'),
                Line2D([0], [0], color='black', 
                        linestyle='--', label='π(g)'),
                Line2D([0], [0], color='black', 
                        linestyle=':', label='π(h)'),
                Line2D([0], [0], color='grey', 
                        linestyle='-.', alpha=0.7, label='truth')
            ]
            ax.legend(handles=style_elements, loc='upper right')
            
            ax.add_artist(agent_legend)
            
        ax = axes[i, 1]
        c1_med = np.median(c1, axis=0)
        ax.plot(c1_med, color=colors[i], linewidth=2, label="median")
        ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5)
        ax.set_title(
            f'Agent 1 consumption share (Nature = {nature_label}{title_suffix})')
        ax.set_xlabel('t')
        ax.set_ylabel("median consumption share")
        ax.set_ylim([-0.01, 1.01])
        ax.legend()
    
    plt.tight_layout()
    return fig, axes
```

Now let's run the simulation. 

In our simulation, agent 1 believes only in $f$ and $g$, while agent 2 has an equal weight on all three models

```{code-cell} ipython3
T = 100
N = 1000

# Generate sequences for nature f and g
s_seq_f = np.random.beta(F_a, F_b, (N, T))
s_seq_g = np.random.beta(G_a, G_b, (N, T))

results_f = simulate_three_model_allocation(s_seq_f, 
                        f, g, h, π_f_0, 1-π_f_0, 
                        1/3, 1/3, λ)
results_g = simulate_three_model_allocation(s_seq_g, 
                        f, g, h, π_f_0, 1-π_f_0, 
                        1/3, 1/3, λ)

c1_data = [results_f[0], results_g[0]] 
π_data = [results_f[1:], results_g[1:]]
nature_labels = ['f', 'g']

fig, axes = plot_three_model_results(c1_data, π_data, nature_labels, λ)
plt.show()
```

The results show interesting dynamics:

In the top panel, Agent 1 (orange line) who initially puts weight only on $f$ (solid line) and $g$ (dashed line) eventually dominates consumption as they learn the truth faster than Agent 2 who spreads probability across all three models.

When nature draws from $g$ (lower panel), we see a similar pattern but reversed -- Agent 1's consumption share decreases as their belief converges to the truth.

For both cases, the belief on $h$ (dotted line) eventually goes to 0.

The agent with the simpler (but correct) model structure learns faster and eventually dominates consumption allocation.

```{solution-end}
```

```{exercise}
:label: lr_ex7

Two agents with extreme priors about three models.

Consider the same setup as the previous exercise, but now:
* Agent 1: $\pi^g_0 = \pi^f_0 = \frac{\epsilon}{2} > 0$, where $\epsilon$ is close to $0$ (e.g., $\epsilon = 0.01$)
* Agent 2: $\pi^g_0 = \pi^f_0 = 0$ (dogmatic belief in model $h$)

Choose $h$ to be close but not equal to either $f$ or $g$ as measured by KL divergence. For example, set $h \sim \text{Beta}(1.2, 1.1)$.

Simulate and visualize the evolution of consumption allocations when:
* Nature permanently draws from $f$
* Nature permanently draws from $g$

Observe how the presence of extreme priors affects learning and allocation dynamics.

```

```{solution-start} lr_ex7
:class: dropdown
```

Let's implement this case with extreme priors where one agent is almost dogmatic.

For this to converge, we need a longer sequence by increasing $T$ to 1000.

Let's define the parameters for distributions and verify that $h$ and $f$ are closer than $h$ and $g$

```{code-cell} ipython3
F_a, F_b = 1, 1
G_a, G_b = 3, 1.2
H_a, H_b = 1.2, 1.1

f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))
h = jit(lambda x: p(x, H_a, H_b))

Kh_f = compute_KL(h, f)
Kh_g = compute_KL(h, g)
Kf_h = compute_KL(f, h)
Kg_h = compute_KL(g, h)

print(f"KL divergences:")
print(f"KL(h,f) = {Kh_f:.4f}, KL(h,g) = {Kh_g:.4f}")
print(f"KL(f,h) = {Kf_h:.4f}, KL(g,h) = {Kg_h:.4f}")
```

Now we can set the belief models for the two agents

```{code-cell} ipython3
# Set extreme priors
ε = 0.01
λ = 0.5

# Agent 1: π_f = ε/2, π_g = ε/2, π_h = 1-ε 
# (almost dogmatic about h)
π_f_1 = ε/2
π_g_1 = ε/2

# Agent 2: π_f = 0, π_g = 0, π_h = 1 
# (fully dogmatic about h)
π_f_2 = 1e-10
π_g_2 = 1e-10
```

Now we can run the simulation

```{code-cell} ipython3
T = 1000
N = 1000

# Generate sequences for different nature scenarios
s_seq_f = np.random.beta(F_a, F_b, (N, T))
s_seq_g = np.random.beta(G_a, G_b, (N, T))

# Run simulations for both scenarios
results_f = simulate_three_model_allocation(
                                s_seq_f, 
                                f, g, h, 
                                π_f_1, π_g_1, π_f_2, π_g_2, λ)
results_g = simulate_three_model_allocation(
                                s_seq_g, 
                                f, g, h, 
                                π_f_1, π_g_1, π_f_2, π_g_2, λ)

c1_data = [results_f[0], results_g[0]] 
π_data = [results_f[1:], results_g[1:]]
nature_labels = ['f', 'g']

title_suffix = f" (Agent 1: ε={ε}, Agent 2: dogmatic)"
fig, axes = plot_three_model_results(c1_data, π_data, nature_labels, λ, 
                                    title_suffix=title_suffix)
plt.show()
```

In the top panel, observe how slowly agent 1 is adjusting to the truth -- the belief is rigid but still updating.

However, since agent 2 is dogmatic about $h$, and $f$ is very hard to distinguish from $g$ as measured by $KL(f, g)$, we can see that the belief is almost standing still.

In the bottom panel, since $g$ is further away from $h$, both agents adjust toward the truth very quickly, but agent 1 acts faster given the slightly higher weight on $f$ and $g$.

```{solution-end}
```
