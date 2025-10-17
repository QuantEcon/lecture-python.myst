---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

(morris_learn)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Speculative Behavior with Bayesian Learning

```{index} single: Models; Morris Learning
```

```{contents} Contents
:depth: 2
```

## Overview

This lecture describes a model of {cite:t}`Morris1996` that extends the Harrison–Kreps model {cite}`HarrKreps1978` of speculative asset pricing.

The model determines the price of a dividend-yielding asset that is traded by risk-neutral investors who have heterogeneous beliefs.

The Harrison-Kreps model features heterogeneous beliefs but assumes that traders have dogmatic, hard-wired beliefs about asset fundamentals.

Morris replaced dogmatic beliefs with *Bayesian learning*: traders who use Bayes' Law to update their beliefs about prospective dividends as new dividend data arrive.

Key features of Morris's model:

* All traders share the same manifold of statistical models for prospective dividends
* All observe the same dividend histories
* All use Bayes' Law to update beliefs
* But they have different initial *prior distributions* over the parameter that indexes the common statistical model

By endowing agents with different prior distributions over a parameter describing the distribution of prospective dividends, Morris builds in heterogeneous beliefs.

Along identical histories of dividends, traders have different *posterior distributions* for prospective dividends.

Those differences set the stage for possible speculation and price bubbles.

Let's start with some standard imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

Prior to reading the following, you might like to review our lectures on

* {doc}`Harrison-Kreps model <harrison_kreps>`
* {doc}`Likelihood ratio processes <likelihood_ratio_process>`
* {doc}`Bayesian versus frequentist statistics <likelihood_bayes>`


## Structure of the model

There is a fixed supply of shares of an asset.

Each share entitles its owner to a stream of *binary* i.i.d. dividends $\{d_t\}$ where

$$
d_{t+1} \in \{0,1\}
$$

The dividend equals $1$ with unknown probability $\theta \in (0,1)$ and equals $0$ with probability $1-\theta$.

Unlike Harrison-Kreps where traders have hard-wired beliefs about a Markov transition matrix, in Morris's model:

* The true dividend probability $\theta$ is unknown
* Traders have *prior beliefs* about $\theta$
* Traders observe dividend realizations and update beliefs via Bayes' Law

There is a finite set $\mathcal{I}$ of *risk-neutral* traders.

All traders have the same discount factor $\beta \in (0,1)$, which is related to the risk-free interest rate $r$ by $\beta = 1/(1+r)$.

### Trading and constraints

Traders buy and sell the risky asset in competitive markets each period $t = 0, 1, 2, \ldots$ after dividends are paid.

As in Harrison-Kreps:

* The stock is traded *ex dividend*
* An owner of a share at the end of time $t$ is entitled to the dividend at time $t+1$
* An owner also has the right to sell the share at time $t+1$

*Short sales are prohibited*.

This matters because it limits how pessimists can express their opinions:

* They *can* express themselves by selling their shares
* They *cannot* express themselves more loudly by borrowing shares and selling them

All traders have sufficient wealth to purchase the risky asset.

## Information and beliefs

All traders observe the full dividend history $(d_1, d_2, \ldots, d_t)$ and update beliefs by Bayes' rule.

However, they have *heterogeneous priors* over the unknown dividend probability $\theta$.

This heterogeneity in priors, combined with the same observed data, produces heterogeneous posterior beliefs.

### Beta prior specification

For tractability, assume trader $i$ has a Beta prior over the dividend probability 

$$
\theta \sim \text{Beta}(a_i, b_i)
$$

where $a_i, b_i > 0$ are the prior parameters.

```{note}
The definition of the Beta distribution can be found in {doc}`divergence_measures`.
```

Suppose trader $i$ observes a history of $t$ periods in which a total of $s$ dividends are paid 
(i.e., $s$ successes with dividend and $t-s$ failures without dividend). 

By Bayes' rule, the posterior density over $\theta$ is:

$$
\pi_i(\theta \mid s, t) = \frac{\theta^s (1-\theta)^{t-s} \pi_i(\theta)}{\int_0^1 \theta^s (1-\theta)^{t-s} \pi_i(\theta) d\theta}
$$

where $\pi_i(\theta)$ is trader $i$'s prior density.

```{note}
The Beta distribution is the conjugate prior for the Binomial likelihood. 

When the prior is $\text{Beta}(a_i, b_i)$ and we observe $s$ successes in $t$ trials, the posterior is $\text{Beta}(a_i+s, b_i+t-s)$.
```

The posterior mean (or expected dividend probability) is:

$$
\mu_i(s,t) = \int_0^1 \theta \pi_i(\theta \mid s, t) d\theta 
= \mathbb{E}[\text{Beta}(a_i+s, b_i+t-s)] = \frac{a_i + s}{a_i + b_i + t}
$$

Morris refers to $\mu_i(s,t)$ as trader $i$'s **fundamental valuation** of the asset after history $(s,t)$. 

This is the probability trader $i$ assigns to receiving a dividend next period, which reflects their updated belief about $\theta$.

## Market prices with learning

Fundamental valuations reflect the expected value to each trader of holding the asset *forever*. 

Equilibrium prices are determined by the most optimistic trader with the highest valuation at each history.

However, in a market where the asset can be resold, traders take into account the possibility of selling at a price higher than their fundamental valuation in some future state.

```{prf:definition} Most Optimistic Valuation
:label: most_optimistic_valuation

After history $(s,t)$, the *most optimistic fundamental valuation* is:

$$
\mu^*(s,t) = \max_{i \in \mathcal{I}} \mu_i(s,t)
$$
```

```{prf:definition} Equilibrium Asset Price
:label: equilibrium_asset_price

Write $\tilde{p}(s,t,r)$ for the competitive equilibrium price of the risky asset (in current dollars) after history $(s,t)$ when the interest rate is $r$. 

The equilibrium price satisfies:

$$
\tilde{p}(s,t,r) = \frac{1}{1+r} \Bigl[ \mu^*(s,t) \{1 + \tilde{p}(s+1,t+1,r)\} 
+ (1 - \mu^*(s,t)) \tilde{p}(s,t+1,r) \Bigr]
$$
```

The equilibrium price equals the highest expected discounted return among all traders from holding the asset to the next period.

```{prf:definition} Normalized Price
:label: normalized_price

The normalized price is defined as:

$$
p(s,t,r) = r \tilde{p}(s,t,r)
$$

Since the current dollar price of the riskless asset is $1/r$, this represents the price of the risky asset in terms of the riskless asset.
```

Substituting into the equilibrium condition gives:

$$
p(s,t,r) = \frac{r}{1+r} \mu^*(s,t) + \frac{1}{1+r} 
\Bigl[ \mu^*(s,t) p(s+1,t+1,r) + (1 - \mu^*(s,t)) p(s,t+1,r) \Bigr]
$$

or equivalently:

$$
p(s,t,r) = \mu^*(s,t) + \frac{r}{1+r} 
\Bigl[ \mu^*(s,t) p(s+1,t+1,r) + (1 - \mu^*(s,t)) p(s,t+1,r) - \mu^*(s,t) \Bigr]
$$

Following Harrison and Kreps, a price scheme satisfying the equilibrium condition can be computed recursively.

Set $p^0(s,t,r) = 0$ for all $(s,t,r)$, and define $p^{n+1}(s,t,r)$ by:

$$
p^{n+1}(s,t,r) = \frac{r}{1+r} \mu^*(s,t) + \frac{1}{1+r} 
\Bigl[ \mu^*(s,t) p^n(s+1,t+1,r) + (1 - \mu^*(s,t)) p^n(s,t+1,r) \Bigr]
$$

The sequence $\{p^n(s,t,r)\}$ converges to the equilibrium price $p(s,t,r)$.

```{prf:definition} Speculative Premium
:label: speculative_premium

When the identity of the most optimistic trader can switch with future dividend realizations, the market price exceeds *every* trader's fundamental valuation. 

In normalized units:

$$
p(s,t,r) > \mu_i(s,t) \quad \text{for all } i \in \mathcal{I}
$$

The **speculative premium** is defined as:

$$
p(s,t,r) - \mu^*(s,t) > 0
$$
```


## Two Traders

We now focus on the case with two traders having priors $(a_1,b_1)$ and $(a_2,b_2)$.

```{prf:definition} Rate Dominance (Beta Priors)
:label: rate_dominance_beta

Trader 1 **rate-dominates** trader 2 if:

$$
a_1 \geq a_2 \quad \text{and} \quad b_1 \leq b_2
$$
```

```{prf:theorem} Global Optimist (Two Traders)
:label: two_trader_optimist

For two traders with Beta priors:

1. If trader 1 rate-dominates trader 2, then trader 1 is a **global optimist**: $\mu_1(s,t) \geq \mu_2(s,t)$ for all histories $(s,t)$
2. In this case where $p(s,t,r) = \mu_1(s,t)$ for all $(s,t,r)$, there is *no speculative premium*.
```

When neither trader rate-dominates the other, the identity of the most optimistic trader can switch with dividend data.

In this perpetual switching case, the price strictly exceeds both traders' fundamental valuations before learning converges:

$$
p(s,t,r) > \max\{\mu_1(s,t), \mu_2(s,t)\}
$$

This is consistent with our discussion about the expectation of future resale opportunities 
creating a speculative premium.

### Implementation

For computational tractability, we work with a finite horizon $T$ and solve by backward induction.

We use the discount factor parameterization $\beta = 1/(1+r)$ and compute dollar prices $\tilde{p}(s,t)$ via:

$$
\tilde{p}(s,t) = \beta \max_{i\in\{1,2\}} \Bigl[ \mu_i(s,t) \{1 + \tilde{p}(s+1,t+1)\} + (1-\mu_i(s,t)) \tilde{p}(s,t+1) \Bigr]
$$

The terminal condition $\tilde{p}(s,T)$ is set equal to the perpetuity value under the most optimistic belief.

```{code-cell} ipython3
def posterior_mean(a, b, s, t):
    """
    Compute posterior mean μ_i(s,t) for Beta(a, b) prior.
    """
    return (a + s) / (a + b + t)

def perpetuity_value(a, b, s, t, β=.75):
    """
    Compute perpetuity value (β/(1-β)) * μ_i(s,t).
    """
    return (β / (1 - β)) * posterior_mean(a, b, s, t)

def price_learning_two_agents(prior1, prior2, β=.75, T=200):
    """
    Compute \tilde p(s,t) for two Beta-prior traders via backward induction.
    """
    a1, b1 = prior1
    a2, b2 = prior2
    price_array = np.zeros((T+1, T+1))

    # Terminal condition: set to perpetuity value under max belief
    for s in range(T+1):
        perp1 = perpetuity_value(a1, b1, s, T, β)
        perp2 = perpetuity_value(a2, b2, s, T, β)
        price_array[s, T] = max(perp1, perp2)

    # Backward induction
    for t in range(T-1, -1, -1):
        for s in range(t, -1, -1):
            μ1 = posterior_mean(a1, b1, s, t)
            μ2 = posterior_mean(a2, b2, s, t)
            
            # One-step continuation values under each trader's beliefs
            cont1 = μ1 * (1.0 + price_array[s+1, t+1]) \
                    + (1.0 - μ1) * price_array[s, t+1]
            cont2 = μ2 * (1.0 + price_array[s+1, t+1]) \
                    + (1.0 - μ2) * price_array[s, t+1]
            price_array[s, t] = β * max(cont1, cont2)

    def μ1_fun(s, t):
        return posterior_mean(a1, b1, s, t)
    def μ2_fun(s, t):
        return posterior_mean(a2, b2, s, t)

    return price_array, μ1_fun, μ2_fun
```

(hk_go)=
### Case A: global optimist (no premium) 

Pick priors with rate dominance, e.g., trader 1: $\text{Beta}(a_1,b_1)=(2,1)$ and trader 2: $(a_2,b_2)=(1,2)$. 

Trader 1 is the global optimist, so the normalized price equals trader 1's fundamental valuation: $p(s,t,r) = \mu_1(s,t)$.

```{code-cell} ipython3
β = 0.75
price_go, μ1_go, μ2_go = price_learning_two_agents(
        (2,1), (1,2), β=β, T=200)

perpetuity_1 = (β / (1 - β)) * μ1_go(0, 0)
perpetuity_2 = (β / (1 - β)) * μ2_go(0, 0)

print("Price at (0, 0) =", price_go[0,0])
print("Valuation of trader 1 at (0, 0) =", perpetuity_1)
print("Valuation of trader 2 at (0, 0) =", perpetuity_2)
```

The price equals trader 1's perpetuity value.

### Case B: perpetual switching (positive premium)

Now assume trader 1 has $\text{Beta}(1,1)$, trader 2 has $\text{Beta}(1/2,1/2)$. 

These produce crossing posteriors, so there is no global optimist and the price exceeds both fundamentals early on.

```{code-cell} ipython3
price_ps, μ1_ps, μ2_ps = price_learning_two_agents(
                                (1,1), (0.5,0.5), β=β, T=200)

price_00 = price_ps[0,0]
μ1_00 = μ1_ps(0,0)
μ2_00 = μ2_ps(0,0)

perpetuity_1 = (β / (1 - β)) * μ1_ps(0, 0)
perpetuity_2 = (β / (1 - β)) * μ2_ps(0, 0)

print("Price at (0, 0) =", np.round(price_00, 6))
print("Valuation of trader 1 at (0, 0) =", perpetuity_1)
print("Valuation of trader 2 at (0, 0) =", perpetuity_2)
```

The resulting premium reflects the option value of reselling to whichever trader becomes temporarily more optimistic as data arrive.

Under this setting, we reproduce the two key figures reported in {cite:t}`Morris1996`

```{code-cell} ipython3
def normalized_price_two_agents(prior1, prior2, r, T=250):
    """Return p(s,t,r) = r · \tilde p(s,t,r) for two traders."""
    β = 1.0 / (1.0 + r)
    price_array, *_ = price_learning_two_agents(prior1, prior2, β=β, T=T)
    return r * price_array

# Figure I: p*(0,0,r) as a function of r
r_grid = np.linspace(1e-3, 5.0, 200)
priors = ((1,1), (0.5,0.5))
p00 = np.array([normalized_price_two_agents(
                priors[0], priors[1], r, T=300)[0,0]
                for r in r_grid])

plt.figure(figsize=(6,4))
plt.plot(r_grid, p00)
plt.xlabel('r')
plt.ylabel(r'$p^*(0,0,r)$')
plt.axhline(0.5, color='C1', linestyle='--')
plt.title('Figure I: normalized price vs interest rate')
plt.show()
```

In the first figure, we can see:

- The resale option pushes the normalized price $p^*(0,0,r)$ above fundamentals $(0.5)$ for any finite $r$. 

- As $r$ increases ($\beta$ decreases), the option value fades and $p^*(0,0,r) \to 0.5$. 

- At $r = 0.05$ the premium is about $8–9\%$, consistent with Morris (1996, Section IV).


```{code-cell} ipython3
# Figure II: p*(t/2,t,0.05) as a function of t
r = 0.05
T = 60
p_mat = normalized_price_two_agents(priors[0], priors[1], r, T=T)
t_vals = np.arange(0, 54, 2) 
s_vals = t_vals // 2
y = np.array([p_mat[s, t] for s, t in zip(s_vals, t_vals)])

plt.figure(figsize=(6,4))
plt.plot(t_vals, y)
plt.xlabel('t')
plt.ylabel(r'$p^*(t/2,t,0.05)$')
plt.axhline(0.5, color='C1', linestyle='--')
plt.title('Figure II: normalized price vs time (r=0.05)')
plt.show()

p0 = p_mat[0,0]
mu0 = 0.5  
print("Initial normalized premium at r=0.05 (%):",
      np.round(100 * (p0 / mu0 - 1.0), 2))
```

In the second figure, we can see:

- Along the symmetric path $s = t/2$, both traders’ fundamentals equal $0.5$ at every $t$, yet the price starts above $0.5$ and declines toward $0.5$ as learning reduces disagreement and the resale option loses value. 


### General N–trader extension

The same recursion extends to any finite set of Beta priors $\{(a_i,b_i)\}_{i=1}^N$ by taking the max over $i$ each period.

```{code-cell} ipython3
def price_learning(priors, β=.75, T=200):
    """
    N-trader version with heterogeneous Beta priors.
    """
    price_array = np.zeros((T+1, T+1))

    def perp_i(i, s, t):
        a, b = priors[i]
        return perpetuity_value(a, b, s, t, β)

    # Terminal condition
    for s in range(T+1):
        price_array[s, T] = max(
            perp_i(i, s, T) for i in range(len(priors)))

    # Backward induction
    for t in range(T-1, -1, -1):
        for s in range(t, -1, -1):
            conts = []
            for (a, b) in priors:
                μ = posterior_mean(a, b, s, t)
                conts.append(μ * 
                (1.0 + price_array[s+1, t+1]) 
                     + (1.0 - μ) * price_array[s, t+1])
            price_array[s, t] = β * max(conts)

    return price_array

β = .75
priors = [(1,1), (0.5,0.5), (3,2)]
price_N = price_learning(priors, β=β, T=150)

# Compute valuations for each trader at (0,0)
mu_vals = [posterior_mean(a, b, 0, 0) for a, b in priors]
perp_vals = [(β / (1 - β)) * mu for mu in mu_vals]

print("Three-trader example at (s,t)=(0,0):")
print(f"Price at (0,0) = {np.round(price_N[0,0], 6)}")
print(f"\nTrader valuations:")
for i, (mu, perp) in enumerate(zip(mu_vals, perp_vals), 1):
    print(f"  Trader {i} = {np.round(perp, 6)}")
```

We can see that the asset price is above all traders' valuations.

Morris tells us that no rate dominance exists in this case.

Let's verify this using the code below

```{code-cell} ipython3
dominant = None
for i in range(len(priors)):
    is_dom = all(
        priors[i][0] >= priors[j][0] and priors[i][1] <= priors[j][1]
                 for j in range(len(priors)) if i != j)
    if is_dom:
        dominant = i
        break

if dominant is not None:
    print(f"\nTrader {dominant+1} is the global optimist (rate-dominant)")
else:
    print(f"\nNo global optimist and speculative premium exists")
```

Indeed, there is no global optimist and a speculative premium exists.

## Exercises

```{exercise-start}
:label: hk_ex3
```

Morris {cite}`Morris1996` provides a sharp characterization of when speculative bubbles arise.

The key condition is that there is no *global optimist*.

In this exercise, you will verify this condition for the following sets of traders with Beta priors:

1. Trader 1: $\text{Beta}(2,1)$, Trader 2: $\text{Beta}(1,2)$
2. Trader 1: $\text{Beta}(1,1)$, Trader 2: $\text{Beta}(1/2,1/2)$
3. Trader 1: $\text{Beta}(3,1)$, Trader 2: $\text{Beta}(2,1)$, Trader 3: $\text{Beta}(1,2)$
4. Trader 1: $\text{Beta}(1,1)$, Trader 2: $\text{Beta}(1/2,1/2)$, Trader 3: $\text{Beta}(3/2,3/2)$

```{exercise-end}
```

```{solution-start} hk_ex3
:class: dropdown
```

Here is one solution:

```{code-cell} ipython3
def check_rate_dominance(priors):
    """
    Check if any trader rate-dominates all others.
    """
    N = len(priors)

    for i in range(N):
        a_i, b_i = priors[i]
        is_dominant = True

        for j in range(N):
            if i == j:
                continue
            a_j, b_j = priors[j]

            # Check rate dominance condition
            if not (a_i >= a_j and b_i <= b_j):
                is_dominant = False
                break

        if is_dominant:
            return i

    return None

# Test cases
test_cases = [
    ([(2, 1), (1, 2)], "Global optimist exists"),
    ([(1, 1), (0.5, 0.5)], "Perpetual switching"),
    ([(3, 1), (2, 1), (1, 2)], "Three traders with dominant"),
    ([(1, 1), (0.5, 0.5), (1.5, 1.5)], "Three traders, no dominant")
]

for priors, description in test_cases:
    dominant = check_rate_dominance(priors)

    print(f"\n{description}")
    print(f"Priors: {priors}")
    print("=="*8)
    if dominant is not None:
        print(f"Trader {dominant+1} is the global optimist (rate-dominant)")
    else:
        print(f"No global optimist exists")
    print("=="*8 + "\n")
```

```{solution-end}
```
