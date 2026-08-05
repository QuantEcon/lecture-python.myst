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

(harrison_kreps)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Heterogeneous Beliefs and Bubbles

```{index} single: Models; Harrison Kreps
```

```{contents} Contents
:depth: 2
```

In addition to what's in Anaconda, this lecture uses following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

## Overview

This lecture describes a version of a model of Harrison and Kreps {cite}`HarrKreps1978`.

The model determines the price of a dividend-yielding asset that is traded by two types of self-interested investors.

The model features

* heterogeneous beliefs
* incomplete markets
* short sales constraints, and possibly $\ldots$
* (leverage) limits on an investor's ability to borrow in order to finance purchases of a risky asset

Let's start with some standard imports:

```{code-cell} ipython3
import numpy as np
import quantecon as qe
import scipy.linalg as la
```

### References

Prior to reading the following, you might like to review our lectures on

* {doc}`Markov chains <finite_markov>`
* {doc}`Asset pricing with finite state space <markov_asset>`

### Bubbles

Economists differ in how they define a *bubble*.

The Harrison-Kreps model illustrates the following notion of a bubble that attracts many economists:

> *A component of an asset price can be interpreted as a bubble when all investors agree that the current price of the asset exceeds what they believe the asset's underlying dividend stream justifies*.

## Structure of the Model

The model simplifies things  by ignoring alterations in the distribution of wealth
among investors who have hard-wired different beliefs about the fundamentals that determine
asset payouts.

There is a fixed number $A$ of shares of an asset.

Each share entitles its owner to a stream of dividends $\{d_t\}$ governed by a Markov chain defined on a state space $S \in \{0, 1\}$.


The dividend obeys

$$
d_t =
\begin{cases}
    0 & \text{ if } s_t = 0 \\
    1 & \text{ if } s_t = 1
\end{cases}
$$

An owner of a share at the end  of time $t$ and the beginning of time $t+1$ is entitled to the dividend paid at time $t+1$.

Thus, the stock is traded **ex dividend**.

An owner of a share at the beginning of time $t+1$ is also entitled to sell the share to another investor during time $t+1$.

Two types $h=a, b$ of investors differ only in their beliefs about a Markov transition matrix $P$ with typical element

$$
P(i,j) = \mathbb P\{s_{t+1} = j \mid s_t = i\}
$$

Investors of type $a$ believe the transition matrix

$$
P_a =
    \begin{bmatrix}
        \frac{1}{2} & \frac{1}{2} \\
        \frac{2}{3} & \frac{1}{3}
    \end{bmatrix}
$$

Investors of  type $b$ think the transition matrix is

$$
P_b =
    \begin{bmatrix}
        \frac{2}{3} & \frac{1}{3} \\
        \frac{1}{4} & \frac{3}{4}
    \end{bmatrix}
$$

Thus,  in state $0$,  a type $a$ investor is more optimistic  about next period's dividend than is investor $b$.

But in state $1$,  a type $a$ investor is more pessimistic  about next period's dividend than is investor $b$.

The stationary (i.e., invariant) distributions of these two matrices can be calculated as follows:

```{code-cell} ipython3
qa = np.array([[1/2, 1/2], [2/3, 1/3]])
qb = np.array([[2/3, 1/3], [1/4, 3/4]])
mca = qe.MarkovChain(qa)
mcb = qe.MarkovChain(qb)
mca.stationary_distributions
```

```{code-cell} ipython3
mcb.stationary_distributions
```

The stationary distribution of $P_a$ is approximately $\pi_a = \begin{bmatrix} .57 & .43 \end{bmatrix}$.

The stationary distribution of $P_b$ is approximately $\pi_b = \begin{bmatrix} .43 & .57 \end{bmatrix}$.

Thus, a type $a$ investor is more pessimistic on average.

### Ownership Rights

An owner of the asset at the end of time $t$ is entitled to the dividend at time $t+1$ and also has the right to sell the asset at time $t+1$.

Both types of investors are risk-neutral and both have the same fixed discount factor $\beta \in (0,1)$.

In our numerical example, we’ll set $\beta = .75$, just as Harrison and Kreps {cite}`HarrKreps1978` did.

We’ll eventually study the consequences of two alternative assumptions about the number of shares $A$ relative to the resources that our two types of investors can invest in the stock.

1. Both types of investors have enough resources (either wealth or the capacity to borrow) so that they can purchase the entire available stock of the asset [^f1].
1. No single type of investor has sufficient resources to purchase the entire stock.

Case 1 is the case studied in Harrison and Kreps.

In case 2, both types of investors always hold at least some of the asset.

### Short Sales Prohibited

No short sales are allowed.

This matters because it limits how  pessimists can express their opinions.

* They **can** express themselves by selling their shares.
* They **cannot** express themsevles  more loudly by artificially "manufacturing shares" -- that is, they cannot borrow shares from more optimistic investors and then immediately sell them.

### Optimism and Pessimism

The above specifications of the perceived transition matrices $P_a$ and $P_b$, taken directly from Harrison and Kreps, build in stochastically alternating temporary optimism and pessimism.

Remember that state $1$ is the high dividend state.

* In state $0$, a type $a$ agent is more optimistic about next period's dividend than a type $b$ agent.
* In state $1$, a type $b$ agent is more optimistic about next period's dividend than a type $a$ agaub is.

However, the stationary distributions $\pi_a = \begin{bmatrix} .57 & .43 \end{bmatrix}$ and $\pi_b = \begin{bmatrix} .43 & .57 \end{bmatrix}$ tell us that a type $b$ person is more optimistic about the dividend process in the long run than is a type $a$ person.

### Information

Investors know a price function mapping the state $s_t$ at $t$ into the equilibrium price $p(s_t)$ that prevails in that state.

This price function is endogenous and to be determined below.

When investors choose whether to purchase or sell the asset at $t$, they also know $s_t$.

## Solving the Model

Now let's turn to solving the model.

We'll  determine equilibrium prices under a particular specification of beliefs and constraints on trading selected from one of the specifications described above.

We shall compare equilibrium price functions under the following alternative
assumptions about beliefs:

1. There is only one type of agent, either $a$ or $b$.
1. There are two types of agents differentiated only by their beliefs. Each type of agent has sufficient resources to purchase all of the asset (Harrison and Kreps's setting).
1. There are two types of agents with different beliefs, but because of limited wealth and/or limited leverage, both types of investors hold the asset each period.

### Summary Table

The following table gives a summary of the findings obtained in the remainder of the lecture
(in an exercise you will be asked to recreate  the  table and also reinterpret parts of it).

The table reports  implications of Harrison and Kreps's specifications of $P_a, P_b, \beta$.


|    $ s_t $    |   0   |   1   |
|---------------|-------|-------|
|    $ p_a $    | 1.33  | 1.22  |
|    $ p_b $    | 1.45  | 1.91  |
|    $ p_o $    | 1.85  | 2.08  |
|    $ p_p $    |   1   |   1   |
| $ \hat{p}_a $ | 1.85  | 1.69  |
| $ \hat{p}_b $ | 1.69  | 2.08  |

Here

* $p_a$ is the equilibrium price function  under homogeneous beliefs $P_a$
* $p_b$ is the equilibrium price function under homogeneous beliefs $P_b$
* $p_o$ is the equilibrium price function under heterogeneous beliefs with optimistic marginal investors
* $p_p$ is the equilibrium price function under heterogeneous beliefs with pessimistic marginal investors
* $\hat{p}_a$ is the amount type $a$ investors are willing to pay for the asset
* $\hat{p}_b$ is the amount type $b$ investors are willing to pay for the asset

We'll explain these values and how they are calculated one row at a time.

The row corresponding to $p_o$ applies when both types of investor have enough resources to purchase the entire stock of the asset and strict short sales constraints prevail so that  temporarily optimistic investors always price the asset.

The row corresponding to $p_p$ would apply if neither type of investor has enough resources to purchase the entire stock of the asset and both types must hold the asset.

The row corresponding to $p_p$ would also  apply if both types have enough resources to buy the entire stock of the asset but  short sales are also  possible so that   temporarily pessimistic   investors price the asset.

### Single Belief Prices

We’ll start by pricing the asset under homogeneous beliefs.

(This is the case treated in {doc}`the lecture <markov_asset>` on asset pricing with finite Markov states)

Suppose that there is only one type of investor, either of type $a$ or $b$, and that this investor always "prices the asset".

Let $p_h = \begin{bmatrix} p_h(0) \cr p_h(1) \end{bmatrix}$ be the equilibrium price vector when all investors are of type $h$.

The price today equals the expected discounted value of tomorrow's dividend and tomorrow's price of the asset:

$$
p_h(s) = \beta \left( P_h(s,0) (0 + p_h(0)) + P_h(s,1) ( 1 + p_h(1)) \right), \quad s = 0, 1
$$ (eq:assetpricehomog)

These equations imply that the equilibrium price vector is

```{math}
:label: HarrKrep1

\begin{bmatrix} p_h(0) \cr p_h(1) \end{bmatrix}
= \beta [I - \beta P_h]^{-1} P_h \begin{bmatrix} 0 \cr 1 \end{bmatrix}
```

The first two rows of the table report $p_a(s)$ and $p_b(s)$.

Here's a function that can be used to compute these values

```{code-cell} ipython3
def price_single_beliefs(transition, dividend_payoff, β=.75):
    """
    Function to Solve Single Beliefs
    """
    # First compute inverse piece
    imbq_inv = la.inv(np.eye(transition.shape[0]) - β * transition)

    # Next compute prices
    prices = β * imbq_inv @ transition @ dividend_payoff

    return prices
```

#### Single Belief Prices as Benchmarks

These equilibrium prices under homogeneous beliefs are important benchmarks for the subsequent analysis.

* $p_h(s)$ tells what a type $h$ investor  thinks is the "fundamental value" of the asset.
* Here "fundamental value" means the expected discounted present value of future dividends.

We will compare these fundamental values of the asset with equilibrium values when traders have different beliefs.

### Pricing under Heterogeneous Beliefs

There are several cases to consider.

The first is when both types of agents have sufficient wealth to purchase all of the asset themselves.

In this case, the marginal investor who prices the asset is the more optimistic type so that the equilibrium price $\bar p$ satisfies Harrison and Kreps's key equation:

```{math}
:label: hakr2

\bar p(s) =
\beta
\max
\left\{
        P_a(s,0) \bar p(0) + P_a(s,1) ( 1 +  \bar p(1))
        ,\;
        P_b(s,0) \bar p(0) + P_b(s,1) ( 1 +  \bar p(1))
\right\}
```

for $s=0,1$.

In the above equation, the $max$ on the right side is over the two prospective values of next period's payout
from owning the asset.

The marginal investor who prices the asset in state $s$ is of type $a$ if

$$
P_a(s,0)  \bar p(0) + P_a(s,1) ( 1 +  \bar p(1)) >
P_b(s,0)  \bar p(0) + P_b(s,1) ( 1 +  \bar p(1))
$$

The marginal investor is of type  $b$ if

$$
P_a(s,1)  \bar p(0) + P_a(s,1) ( 1 +  \bar  p(1)) <
P_b(s,1)  \bar p(0) + P_b(s,1) ( 1 +  \bar  p(1))
$$

**Thus the marginal investor is the (temporarily) optimistic type**.

Equation {eq}`hakr2` is a functional equation that, like a Bellman equation, can be solved by

* starting with a guess for the price vector $\bar p$ and
* iterating to convergence on the operator that maps a guess $\bar p^j$ into an updated guess
  $\bar p^{j+1}$ defined by the right side of {eq}`hakr2`, namely

```{math}
:label: HarrKrep3

\bar  p^{j+1}(s)
 = \beta \max
 \left\{
        P_a(s,0) \bar p^j(0) + P_a(s,1) ( 1 + \bar p^j(1))
        ,\;
        P_b(s,0) \bar p^j(0) + P_b(s,1) ( 1 + \bar p^j(1))
\right\}
```

for $s=0,1$.

The third row of the table labeled $p_o$ reports equilibrium prices that solve the functional equation when $\beta = .75$.

Here the type that is optimistic about $s_{t+1}$ prices the asset in state $s_t$.

It is instructive to compare these prices with the equilibrium prices for the homogeneous belief economies that solve under beliefs $P_a$ and $P_b$ reported in the rows labeled $p_a$ and $p_b$, respectively.

Equilibrium prices $p_o$ in the heterogeneous beliefs economy evidently exceed what any prospective investor regards as the fundamental value of the asset in each possible state.

Nevertheless, the economy recurrently visits a state that makes each investor want to
purchase the asset for more than he believes its future dividends are
worth.

An investor is willing to pay more than what he believes is warranted by fundamental value of the prospective dividend stream because he expects to have the option later to sell the asset  to another investor who will value the asset more highly than he will then.

* Investors of type $a$ are willing to pay the following price for the asset

$$
\hat p_a(s) =
\begin{cases}
\bar p(0)  & \text{ if } s_t = 0 \\
\beta(P_a(1,0) \bar p(0) + P_a(1,1) ( 1 +  \bar p(1))) & \text{ if } s_t = 1
\end{cases}
$$

* Investors of type $b$ are willing to pay the following price for the asset

$$
\hat p_b(s) =
\begin{cases}
    \beta(P_b(0,0) \bar p(0) + P_b (0,1) ( 1 +  \bar p(1)))  & \text{ if } s_t = 0 \\
    \bar p(1)  & \text{ if } s_t =1
\end{cases}
$$

Evidently, $\hat p_a(1) < \bar p(1)$ and $\hat p_b(0) < \bar p(0)$.

Investors of type $a$ want to sell the asset in state $1$ while investors of type $b$ want to sell it in state $0$.

* The asset changes hands whenever the state changes from $0$ to $1$ or from $1$ to $0$.
* The valuations $\hat p_a(s)$ and $\hat p_b(s)$ are displayed in the fourth and fifth rows of the table.
* Even  pessimistic investors who don't buy the asset think that it is worth more than they think future dividends are worth.

Here's code to solve for $\bar p$, $\hat p_a$ and $\hat p_b$ using the iterative method described above

```{code-cell} ipython3
def price_optimistic_beliefs(transitions, dividend_payoff, β=.75,
                            max_iter=50000, tol=1e-16):
    """
    Function to Solve Optimistic Beliefs
    """
    # We will guess an initial price vector of [0, 0]
    p_new = np.array([[0], [0]])
    p_old = np.array([[10.], [10.]])

    # We know this is a contraction mapping, so we can iterate to conv
    for i in range(max_iter):
        p_old = p_new
        p_new = β * np.max([q @ p_old
                            + q @ dividend_payoff for q in transitions],
                            axis=0)

        # If we succeed in converging, break out of for loop
        if np.max(np.sqrt((p_new - p_old)**2)) < tol:
            break

    ptwiddle = β * np.min([q @ p_old
                          + q @ dividend_payoff for q in transitions],
                          axis=0)

    phat_a = np.array([p_new[0], ptwiddle[1]])
    phat_b = np.array([ptwiddle[0], p_new[1]])

    return p_new, phat_a, phat_b
```

### Insufficient Funds

Outcomes differ when the more optimistic type of investor has insufficient wealth --- or insufficient ability to borrow enough --- to hold the entire stock of the asset.

In this case, the asset price must adjust to attract pessimistic investors.

Instead of equation {eq}`hakr2`, the equilibrium price satisfies

```{math}
:label: HarrKrep4

\check p(s)
= \beta \min
\left\{
    P_a(s,0)  \check  p(0) + P_a(s,1) ( 1 +   \check  p(1)) ,\;
    P_b(s,0)  \check p(0) + P_b(s,1) ( 1 + \check p(1))
\right\}
```

and the marginal investor who prices the asset is always the one that values it *less* highly than does the other type.

Now the marginal investor is always the (temporarily) pessimistic type.

Notice from the sixth row of that the pessimistic price $p_o$ is lower than the homogeneous belief prices $p_a$ and $p_b$ in both states.

When pessimistic investors price the asset according to {eq}`HarrKrep4`, optimistic investors think that the asset is underpriced.

If they could, optimistic investors would willingly borrow at a  one-period risk-free gross interest rate $\beta^{-1}$ to purchase more of the asset.

Implicit constraints on leverage prohibit them from doing so.

When optimistic investors price the asset as in equation {eq}`hakr2`, pessimistic investors think that the asset is overpriced and would like to sell the asset short.

Constraints on short sales prevent that.

Here's code to solve for $\check p$ using iteration

```{code-cell} ipython3
def price_pessimistic_beliefs(transitions, dividend_payoff, β=.75,
                            max_iter=50000, tol=1e-16):
    """
    Function to Solve Pessimistic Beliefs
    """
    # We will guess an initial price vector of [0, 0]
    p_new = np.array([[0], [0]])
    p_old = np.array([[10.], [10.]])

    # We know this is a contraction mapping, so we can iterate to conv
    for i in range(max_iter):
        p_old = p_new
        p_new = β * np.min([q @ p_old
                            + q @ dividend_payoff for q in transitions],
                           axis=0)

        # If we succeed in converging, break out of for loop
        if np.max(np.sqrt((p_new - p_old)**2)) < tol:
            break

    return p_new
```

### Further Interpretation

Jose Scheinkman {cite}`Scheinkman2014` interprets the Harrison-Kreps model as a model of a bubble --- a situation in which an asset price exceeds what every investor thinks is merited by his or her beliefs about the value of the asset's underlying dividend stream.

Scheinkman stresses these features of the Harrison-Kreps model:

* High volume occurs when the Harrison-Kreps pricing formula {eq}`hakr2` prevails.

* Type $a$ investors sell the entire stock of the asset to type $b$ investors every time the state switches from $s_t =0$ to $s_t =1$.

* Type $b$ investors sell the asset to type $a$ investors every time the state switches from $s_t = 1$ to $s_t =0$.

Scheinkman takes this as a strength of the model because he observes high volume during *famous bubbles*.

* If the *supply* of the asset is increased sufficiently either physically (more "houses" are built) or artificially (ways are invented to short sell "houses"), bubbles end  when the asset supply has grown enough to outstrip optimistic investors’ resources for purchasing the asset.
* If optimistic investors finance their purchases by borrowing, tightening leverage constraints can extinguish a bubble.

Scheinkman extracts insights about the effects of financial regulations on bubbles.

He emphasizes how limiting short sales and limiting leverage have opposite effects.

```{exercise-start}
:label: hk_ex1
```

This exercise invites you to recreate the summary table using the functions we have built above.

|    $s_t$    |   0   |   1   |
|-------------|-------|-------|
|    $p_a$    | 1.33  | 1.22  |
|    $p_b$    | 1.45  | 1.91  |
|    $p_o$    | 1.85  | 2.08  |
|    $p_p$    |   1   |   1   |
| $\hat{p}_a$ | 1.85  | 1.69  |
| $\hat{p}_b$ | 1.69  | 2.08  |

You will want first  to define the transition matrices and dividend payoff vector.

In addition, below we'll add an interpretation of the row corresponding to $p_o$ by
inventing two additional types of agents, one of whom is **permanently optimistic**, the other who
is **permanently pessimistic**.

We construct subjective transition probability matrices for our permanently  optimistic and permanently pessimistic investors as follows.

The permanently optimistic investors(i.e., the investor with the most optimistic
beliefs in each state) believes the transition matrix

$$
P_o =
    \begin{bmatrix}
        \frac{1}{2} & \frac{1}{2} \\
        \frac{1}{4} & \frac{3}{4}
    \end{bmatrix}
$$

The permanently pessimistic investor believes the transition matrix

$$
P_p =
    \begin{bmatrix}
        \frac{2}{3} & \frac{1}{3} \\
        \frac{2}{3} & \frac{1}{3}
    \end{bmatrix}
$$

We'll use these transition matrices when we present our solution of exercise 1 below.

```{exercise-end}
```

```{solution-start} hk_ex1
:class: dropdown
```

First, we will obtain equilibrium price vectors with homogeneous beliefs, including when all
investors are optimistic or pessimistic.

```{code-cell} ipython3
qa = np.array([[1/2, 1/2], [2/3, 1/3]])    # Type a transition matrix
qb = np.array([[2/3, 1/3], [1/4, 3/4]])    # Type b transition matrix
# Optimistic investor transition matrix
qopt = np.array([[1/2, 1/2], [1/4, 3/4]])
# Pessimistic investor transition matrix
qpess = np.array([[2/3, 1/3], [2/3, 1/3]])

dividendreturn = np.array([[0], [1]])

transitions = [qa, qb, qopt, qpess]
labels = ['p_a', 'p_b', 'p_optimistic', 'p_pessimistic']

for transition, label in zip(transitions, labels):
    print(label)
    print("=" * 20)
    s0, s1 = np.round(price_single_beliefs(transition, dividendreturn), 2)
    print(f"State 0: {s0}")
    print(f"State 1: {s1}")
    print("-" * 20)
```

We will use the price_optimistic_beliefs function to find the price under
heterogeneous beliefs.

```{code-cell} ipython3
opt_beliefs = price_optimistic_beliefs([qa, qb], dividendreturn)
labels = ['p_optimistic', 'p_hat_a', 'p_hat_b']

for p, label in zip(opt_beliefs, labels):
    print(label)
    print("=" * 20)
    s0, s1 = np.round(p, 2)
    print(f"State 0: {s0}")
    print(f"State 1: {s1}")
    print("-" * 20)
```

Notice that the equilibrium price with heterogeneous beliefs is equal to the price under single beliefs
with **permanently optimistic** investors - this is due to the marginal investor in the heterogeneous beliefs equilibrium always being the type who is  temporarily optimistic.

```{solution-end}
```

[^f1]: By assuming that both types of agents always have "deep enough pockets" to purchase all of the asset, the model takes wealth dynamics off the table. The Harrison-Kreps model generates high trading volume when the state changes either from 0 to 1 or from 1 to 0.
