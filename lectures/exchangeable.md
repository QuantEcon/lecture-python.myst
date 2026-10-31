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

(exchangeable)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Exchangeability and Bayesian Updating

```{contents} Contents
:depth: 2
```

## Overview

This lecture studies  learning
via Bayes' Law.

We touch foundations of Bayesian statistical inference invented by Bruno de Finetti {cite}`definetti`.

The relevance of de Finetti's work for economists is presented forcefully  by David Kreps
in chapter 11 of {cite}`Kreps88`.

An example that we study below is a key component of {doc}`odu`.

That lecture augments the classic job search model of McCall {cite}`McCall1970`, studied in {doc}`mccall_model`, by presenting an unemployed worker with a statistical inference problem.

Here we create  graphs that illustrate the role that  a  likelihood ratio
plays in  Bayes' Law.

We'll use such graphs to provide insights into the mechanics driving outcomes in {doc}`odu`.

Among other things, this lecture discusses  connections between the statistical concepts of sequences of random variables that are

- independently and identically distributed
- exchangeable

Understanding these concepts is essential for appreciating how Bayesian updating
works.

You can read about exchangeability [here](https://en.wikipedia.org/wiki/Exchangeable_random_variables).

De Finetti's theorem states that an infinite exchangeable sequence is **conditionally independently and identically distributed**: it is IID conditional on a random variable whose value the observer does not know.[^condind]

We want to convey an answer to the question *conditional on what?*

We also tell why  an assumption of independence precludes  learning while
an assumption that a sequence is conditionally IID makes learning possible.

[^condind]: Conditional independence alone does not imply exchangeability. Let $W_t = t \theta + \varepsilon_t$, where $\theta$ is a random variable and $\{\varepsilon_t\}$ is an IID $N(0,1)$ sequence independent of $\theta$. Conditional on $\theta$, the $W_t$ are independent, but they are not identically distributed, and the sequence $\{W_t\}$ is not exchangeable.

Below, we'll often use

- $W$ to denote a random variable
- $w$ to denote a particular realization of a random variable $W$

Let’s start with some imports:

```{code-cell} ipython3
:tags: [hide-output]

from math import gamma

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from scipy.integrate import quad
from scipy.special import expit
```

## Independently and identically distributed

We begin by looking at the notion of an  **independently and identically  distributed sequence** of random variables.

An independently and identically distributed sequence is often abbreviated as IID.

Two notions are involved

- **independence**

- **identically distributed**

A sequence $W_0, W_1, \ldots$ is **independently distributed** if, for every $n \geq 0$, the joint probability density
of $W_0, W_1, \ldots, W_n$ is the **product** of the marginal densities of $W_0, W_1, \ldots, W_n$.

The sequence $W_0, W_1, \ldots$ is **independently and identically distributed** (IID) if in addition the marginal
density of $W_t$ is the same for all $t =0, 1, \ldots$.

For example,  let $p(W_0, W_1, \ldots, W_n)$ be the **joint density** of $W_0, W_1, \ldots, W_n$ and
let $p(W_t)$ be the common **marginal density** of each $W_t$.

Then the sequence $W_0, W_1, \ldots$ is IID if

$$
p(W_0, W_1, \ldots, W_n) =  p(W_0) p(W_1) \cdots p(W_n) \quad \text{for every } n \geq 0
$$

so that each joint density is the product of identical marginal densities.

### IID means past observations don't tell us anything about future observations

If a sequence of random variables is IID, past information provides no information about future realizations.

Therefore, there is **nothing to learn** from the past  about the future.

To understand these statements, let the joint distribution of a sequence of random variables $\{W_t\}_{t=0}^T$
that is not necessarily IID be

$$
p(W_T, W_{T-1}, \ldots, W_1, W_0)
$$

Using the laws of probability, we can always factor such a joint density into a product of conditional densities:

$$
\begin{aligned}
  p(W_T, W_{T-1}, \ldots, W_1, W_0)    = & p(W_T | W_{T-1}, \ldots, W_0) p(W_{T-1} | W_{T-2}, \ldots, W_0) \cdots  \cr
  & \quad \quad \cdots p(W_1 | W_0) p(W_0)
\end{aligned}
$$

In general,

$$
p(W_t | W_{t-1}, \ldots, W_0)   \neq   p(W_t)
$$

which states that the **conditional density** on the left side does not equal the **marginal density** on the right side.

But in the special IID case,

$$
p(W_t | W_{t-1}, \ldots, W_0)   =  p(W_t) ,
$$

so that the  partial history $W_{t-1}, \ldots, W_0$ contains no information about the probability of $W_t$.

So in the IID case, there is **nothing to learn** about the densities of future random variables from past random variables.

But when the sequence is not IID, there is something to learn about the future from observations of past random variables.

We turn next to an instance of the general case in which the sequence is not IID.

Please watch for what can be  learned from the past and when.

## A setting in which past observations are informative

Let $\{W_t\}_{t=0}^\infty$ be a sequence of nonnegative
scalar random variables with a joint probability distribution
constructed as follows.

There are two distinct cumulative distribution functions $F$ and $G$ that have  densities $f$ and $g$, respectively,  for a nonnegative scalar random
variable $W$.

Before the start of time, say at time $t= -1$, “nature” once and for
all selects **either** $f$ **or** $g$.

Thereafter at each time
$t \geq 0$, nature  draws a random variable $W_t$ from the selected
distribution.

So  the data are permanently generated as independently and identically distributed (IID) draws from **either** $F$ **or**
$G$.

We could say that *objectively*, meaning *after* nature has chosen either $F$ or $G$, the probability that the data are generated as draws from $F$ is either $0$
or $1$.

We now drop into this setting a partially informed decision maker who 

- knows both $F$ and $G$, but

- does not know whether  at $t = -1$ nature had drawn   $F$ or whether nature had drawn   $G$ once-and-for-all

Thus, although our decision maker knows $F$ and knows $G$, he does not know which of these two known distributions nature had selected to draw from.

The decision maker describes his ignorance with a **subjective probability**
$\tilde \pi$ and reasons as if  nature had selected $F$ with probability
$\tilde \pi \in (0,1)$ and
$G$ with probability $1 - \tilde \pi$.

Thus, we  assume that the decision maker

- **knows** both $F$ and $G$
- **doesn't know** which of these two distributions that nature has drawn
- expresses  his ignorance by **acting as if** or **thinking that** nature chose distribution $F$ with probability $\tilde \pi \in (0,1)$ and distribution
  $G$ with probability $1 - \tilde \pi$
- at date $t \geq 0$ knows  the partial history $w_t, w_{t-1}, \ldots, w_0$

To proceed, we want to know the decision maker's belief about the joint distribution of the partial history.

We'll discuss that next and in the process describe the concept of **exchangeability**.

## Relationship between IID and exchangeable

Conditional on nature selecting $F$, the joint density of
$W_0, W_1, \ldots, W_n$ is

$$
f(W_0) f(W_1) \cdots f(W_n)
$$

for every $n \geq 0$.

Conditional on nature selecting $G$, the joint density of
$W_0, W_1, \ldots, W_n$ is

$$
g(W_0) g(W_1) \cdots g(W_n)
$$

Thus,  **conditional on nature having selected** $F$, the
sequence $W_0, W_1, \ldots$ is independently and
identically distributed.

Furthermore,  **conditional on nature having
selected** $G$, the sequence $W_0, W_1, \ldots$ is also
independently and identically distributed.

But what about the **unconditional distribution** of a partial history?

The unconditional joint density of $W_0, W_1, \ldots, W_n$ is
evidently

```{math}
:label: eq_definetti

h(W_0, W_1, \ldots, W_n ) \equiv \tilde \pi [f(W_0) f(W_1) \cdots f(W_n)] + ( 1- \tilde \pi) [g(W_0) g(W_1) \cdots g(W_n)], \quad n \geq 0
```

Under the unconditional distribution {eq}`eq_definetti`, the
sequence $W_0, W_1, \ldots$ is **not** independently and
identically distributed.

To verify this claim, let $h(W_t) = \tilde \pi f(W_t) + (1-\tilde \pi) g(W_t)$ be the marginal density of $W_t$.

Then

```{math}
:label: eq_hcov

h(W_0, W_1) - h(W_0) h(W_1) = \tilde \pi (1 - \tilde \pi) \bigl(f(W_0) - g(W_0)\bigr) \bigl(f(W_1) - g(W_1)\bigr)
```

The right side is not zero when $f(W_0) \neq g(W_0)$ and $f(W_1) \neq g(W_1)$, so the joint density $h(W_0, W_1)$ is not the product of the marginal densities $h(W_0)$ and $h(W_1)$.

Thus, the conditional distribution

$$
h(W_1 | W_0) \equiv \frac{h(W_0, W_1)}{(\tilde \pi f(W_0) + (1-\tilde \pi) g(W_0))}
 \neq ( \tilde \pi f(W_1) + (1-\tilde \pi) g(W_1))
$$

This means that  random variable  $W_0$ contains information about random variable  $W_1$.

So there is something to learn from the past about the future.

Exercise {ref}`exchangeable_ex1` uses {eq}`eq_hcov` to show that $W_0$ and $W_1$ are positively correlated.


## Exchangeability

While the sequence $W_0, W_1, \ldots$ is not IID, it can be verified that it is
**exchangeable**, which means that the   joint distributions $h(W_0, W_1)$ and $h(W_1, W_0)$ of the "re-ordered" sequences
satisfy

$$
h(W_0, W_1) = h(W_1, W_0)
$$

and so on.

More generally, a sequence of random variables is said to be **exchangeable** if  the  joint probability distribution
for a sequence does not change when the positions in the sequence in which finitely many  of random variables
appear are altered.

Equation {eq}`eq_definetti` represents our instance of an exchangeable joint density over a sequence of random
variables  as a **mixture**  of  two IID joint densities over a sequence of random variables.

A Bayesian statistician interprets the mixing parameter $\tilde \pi \in (0,1)$ as a decision maker's subjective belief -- the decision maker's  **prior probability**  -- that nature had  selected probability distribution $F$.

### De Finetti's theorem

De Finetti {cite}`definetti` proved that every infinite exchangeable sequence is a mixture of IID sequences.

For binary random variables $W_t \in \{0, 1\}$, de Finetti's theorem states that the infinite sequence $W_0, W_1, \ldots$ is exchangeable if and only if there is a probability measure $\mu$ on $[0,1]$ such that

```{math}
:label: eq_definetti_bernoulli

\textrm{Prob}(W_0 = w_0, \ldots, W_{n-1} = w_{n-1}) = \int_0^1 \theta^{k} (1-\theta)^{n-k} \mu(d\theta), \quad k = \sum_{t=0}^{n-1} w_t
```

for every $n \geq 1$ and every $(w_0, \ldots, w_{n-1}) \in \{0,1\}^n$.

The mixing measure $\mu$ is unique.

It is the probability distribution of the limiting relative frequency

$$
\theta = \lim_{n \rightarrow \infty} \frac{1}{n} \sum_{t=0}^{n-1} W_t ,
$$

a limit that exists with probability one.

Conditional on $\theta$, the $W_t$ are IID Bernoulli random variables with success probability $\theta$.

This answers the question *conditional on what?*: conditional on the limiting relative frequency $\theta$.

A Bayesian statistician interprets $\mu$ as a prior probability distribution over the unknown Bernoulli parameter $\theta \in [0,1]$.

The measure $\mu$ need not have a density; it can put positive probability on finitely many points.

Hewitt and Savage {cite}`HewittSavage1955` extended de Finetti's theorem to sequences that take values in more general spaces.

In their version the mixing measure is a probability distribution over probability distributions of $W$, and the conditioning variable is the limiting empirical distribution of $W_0, W_1, \ldots$.

Equation {eq}`eq_definetti` is an instance with a two-point mixing measure that puts probability $\tilde \pi$ on $F$ and probability $1 - \tilde \pi$ on $G$.

In that instance the limiting empirical distribution of $W_0, W_1, \ldots$ is $F$ when nature has drawn $F$ and $G$ when nature has drawn $G$.

De Finetti's theorem requires an infinite sequence.

Exercise {ref}`exchangeable_ex1` describes a finite exchangeable sequence that is not a mixture of IID sequences.

The Beta–Bernoulli model in {doc}`merging_of_opinions` is the instance of {eq}`eq_definetti_bernoulli` in which $\mu$ is a Beta distribution.

Exercise {ref}`exchangeable_ex3` describes an urn scheme that generates it.

## Bayes' Law

We noted above that in our example model there is something to learn about the future from past data drawn
from our particular instance of a process that is exchangeable but not IID.

But how can we learn?

And about what?

The answer to the *about what* question is: about which distribution, $F$ or $G$, nature selected at time $-1$.

The decision maker knows his prior probability $\tilde \pi$.

What he learns is expressed in a posterior probability that moves away from $\tilde \pi$ as data arrive.

The answer to the *how* question is to use  Bayes' Law.

Another way to say *use Bayes' Law* is to say *from a (subjective) joint distribution, compute an appropriate conditional distribution*.

Let's dive into Bayes' Law in this context.

Let $q$ represent the distribution that nature actually draws $w$
 from and let

$$
\pi = \mathbb{P}\{q = f \}
$$

where we regard $\pi$ as a decision maker's **subjective probability**  (also called a **personal probability**).

Suppose that at $t \geq 0$, the decision maker has  observed a history
$w^t \equiv [w_t, w_{t-1}, \ldots, w_0]$.

We let

$$
\pi_t  = \mathbb{P}\{q = f  | w^t \}
$$

where we adopt the convention

$$
\pi_{-1}  = \tilde \pi
$$

Conditional on $q$, the draw $w_{t+1}$ is independent of the history $w^t$.

The density of $w_{t+1}$ conditional on $w^t$ is then

$$
\pi_t f + (1 - \pi_t) g .
$$

Bayes’ rule for updating $\pi_{t+1}$ is

$$
\pi_{t+1} = \frac{\pi_t f(w_{t+1})}{\pi_t f(w_{t+1}) + (1 - \pi_t) g(w_{t+1})}
$$ (eq_Bayes102)


Equation {eq}`eq_Bayes102` follows from Bayes’ rule, which
tells us that

$$
\mathbb{P}\{q = f \,|\, w^{t+1}\}
= \frac{p(w_{t+1} \,|\, q = f, w^t)\, \mathbb{P}\{q = f \,|\, w^t\}}
{p(w_{t+1} \,|\, w^t)}
$$

where $p$ denotes a density and

$$
p(w_{t+1} \,|\, w^t) = \sum_{a \in \{f, g\}} p(w_{t+1} \,|\, q = a, w^t)\, \mathbb{P}\{q = a \,|\, w^t\} .
$$

Because $w_{t+1}$ is independent of $w^t$ conditional on $q$, $p(w_{t+1} \,|\, q = f, w^t) = f(w_{t+1})$ and $p(w_{t+1} \,|\, q = g, w^t) = g(w_{t+1})$.

## More details about Bayesian updating

Let's stare at and rearrange Bayes' Law as represented in equation {eq}`eq_Bayes102` with the aim of understanding
how the **posterior** probability $\pi_{t+1}$ is influenced by the **prior** probability $\pi_t$ and the **likelihood ratio**

$$
\ell(w) = \frac{f(w)}{g(w)}
$$

It is convenient for us to rewrite the updating rule {eq}`eq_Bayes102` as

$$
\pi_{t+1}   =\frac{\pi_{t}f\left(w_{t+1}\right)}{\pi_{t}f\left(w_{t+1}\right)+\left(1-\pi_{t}\right)g\left(w_{t+1}\right)}
    =\frac{\pi_{t}\frac{f\left(w_{t+1}\right)}{g\left(w_{t+1}\right)}}{\pi_{t}\frac{f\left(w_{t+1}\right)}{g\left(w_{t+1}\right)}+\left(1-\pi_{t}\right)}
    =\frac{\pi_{t}\ell\left(w_{t+1}\right)}{\pi_{t}\ell\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}
$$

This implies that

```{math}
:label: eq_Bayes103

\frac{\pi_{t+1}}{\pi_{t}}=\frac{\ell\left(w_{t+1}\right)}{\pi_{t}\ell\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}\begin{cases} >1 &
\text{if }\ell\left(w_{t+1}\right)>1\\
\leq1 & \text{if }\ell\left(w_{t+1}\right)\leq1
\end{cases}
```

The likelihood ratio alone determines whether an observation $w_{t+1}$ leads the decision maker
to increase or decrease the subjective probability he/she attaches to distribution $F$.

When the likelihood ratio $\ell(w_{t+1})$ exceeds one, the observation $w_{t+1}$ nudges the probability
$\pi$ put on distribution $F$ upward,
and when the likelihood ratio $\ell(w_{t+1})$ is less than one, the observation $w_{t+1}$ nudges $\pi$ downward.

The prior $\pi_t$ affects the size of the move:

$$
\pi_{t+1} - \pi_t = \frac{\pi_t (1 - \pi_t) \bigl(\ell(w_{t+1}) - 1\bigr)}{\pi_t \ell(w_{t+1}) + 1 - \pi_t}
$$

The move shrinks to zero as $\pi_t$ approaches $0$ or $1$.

Representation {eq}`eq_Bayes103` is the foundation of some graphs that we'll use to display the dynamics of
$\{\pi_t\}_{t=0}^\infty$ that are  induced by
Bayes' Law.

We’ll plot $\ell\left(w\right)$ as a way to enlighten us about how
learning – i.e., Bayesian updating of the probability $\pi$ that
nature has chosen distribution $f$ – works.

We build up the picture in three steps, each of which produces one graph.

All three are built from the same ingredients: the densities $f$ and $g$, and the values of $w$ at which
the likelihood ratio $\ell(w) = f(w)/g(w)$ crosses one.

Both $f$ and $g$ are Beta densities, so we start with the general Beta density.

```{code-cell} ipython3
def p(w, a, b):
    "The Beta density with parameters a and b."
    r = gamma(a + b) / (gamma(a) * gamma(b))
    return r * w**(a - 1) * (1 - w)**(b - 1)
```

The next function assembles the ingredients for a given pair of Beta distributions.

```{code-cell} ipython3
def create_model(F_a=1, F_b=1, G_a=3, G_b=1.2):
    """
    Build the densities f and g, along with the two values of w at which
    the likelihood ratio ℓ(w) = f(w) / g(w) equals one.

    The root finder assumes that f is uniform (F_a = F_b = 1) and that
    G_a > 1 and G_b > 1.  Then g vanishes at 0 and 1 and has a single
    interior mode at which g exceeds one, so ℓ(w) = 1 / g(w) equals one
    exactly once on each side of the mode of g.
    """
    f = lambda w: p(w, F_a, F_b)
    g = lambda w: p(w, G_a, G_b)

    # The mode of g divides [0, 1] into two intervals, each holding one root
    G_mode = (G_a - 1) / (G_a + G_b - 2)
    obj = lambda w: f(w) / g(w) - 1
    roots = np.array([op.root_scalar(obj, bracket=[1e-10, G_mode]).root,
                      op.root_scalar(obj, bracket=[G_mode, 1 - 1e-10]).root])
    return f, g, roots
```

### The likelihood ratio

Our first graph plots the likelihood ratio $\ell(w)$ on the abscissa axis against $w$ on the ordinate axis.

We orient it this way so that $w$ shares an axis with the two graphs that follow.

```{code-cell} ipython3
def plot_likelihood_ratio(F_a=1, F_b=1, G_a=3, G_b=1.2):
    f, g, roots = create_model(F_a, F_b, G_a, G_b)
    w_grid = np.linspace(1e-12, 1 - 1e-12, 100)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(f(w_grid) / g(w_grid), w_grid, label=r'$\ell$', lw=2)
    ax.vlines(1, 0, 1, linestyle='--')
    ax.hlines(roots, 0, 2, linestyle='--')
    ax.set_xlim(0, 2)
    ax.legend(loc=4)
    ax.set(xlabel=r'$\ell(w) = f(w) / g(w)$', ylabel='$w$')
    plt.show()
```

We begin with $f$ uniform on $[0,1]$ --- that is, Beta with $F_a=1, F_b=1$ --- and $g$ Beta with $G_a=3, G_b=1.2$.

```{code-cell} ipython3
plot_likelihood_ratio()
```

The two horizontal dashed lines mark the values of $w$ at which $\ell(w) = 1$.

Between them the likelihood ratio is below one, so by {eq}`eq_Bayes103` a draw in that region pushes $\pi$ down.

Outside them it exceeds one, so a draw there pushes $\pi$ up.

### The densities and the probabilities of moving in each direction

Our second graph plots $f(w)$ and $g(w)$ against $w$, and shades the regions delineated by those same two values of $w$.

Shading them lets us attach a probability to each direction of movement, which we compute by integrating the relevant density over the region.

```{code-cell} ipython3
def plot_densities(F_a=1, F_b=1, G_a=3, G_b=1.2):
    f, g, roots = create_model(F_a, F_b, G_a, G_b)
    w_grid = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(f(w_grid), w_grid, label='$f$', lw=2)
    ax.plot(g(w_grid), w_grid, label='$g$', lw=2)
    ax.vlines(1, 0, 1, linestyle='--')
    ax.hlines(roots, 0, 2, linestyle='--')
    ax.legend(loc=4)
    ax.set(xlabel='$f(w), g(w)$', ylabel='$w$')

    # Probability of landing in each region, under whichever density is shaded
    area_lower = quad(f, 0, roots[0])[0]
    area_middle = quad(g, roots[0], roots[1])[0]
    area_upper = quad(f, roots[1], 1)[0]

    ax.fill_between([0, 1], 0, roots[0], color='blue', alpha=0.15)
    ax.text((f(0) + f(roots[0])) / 4, roots[0] / 2, f"{area_lower: .3g}")
    w_middle = np.linspace(roots[0], roots[1], 20)
    ax.fill_betweenx(w_middle, 0, g(w_middle), color='orange', alpha=0.15)
    ax.text(np.mean(g(roots)) / 2, np.mean(roots), f"{area_middle: .3g}")
    ax.fill_between([0, 1], roots[1], 1, color='blue', alpha=0.15)
    ax.text((f(roots[1]) + f(1)) / 4, (roots[1] + 1) / 2, f"{area_upper: .3g}")
    plt.show()
```

Let's look at the same pair of distributions as before.

```{code-cell} ipython3
plot_densities()
```

The fractions in the colored areas are probabilities that a realization of $w$ falls into the region beside them.

The blue regions are integrals of $f$ and the orange region is an integral of $g$.

For example, under true distribution $F$, $\pi$ will be updated toward $0$ if $w$ falls into the interval
$[0.524, 0.999]$, which occurs with probability $0.999 - 0.524 \approx 0.475$ under $F$.

But this would occur with probability $0.816$ if $G$ were the true distribution.

### Dynamics of the belief

Our third graph attaches an arrow to each point in the $(\pi, w)$ plane showing the change in $\pi$ that
Bayes' Law induces when the current belief is $\pi$ and the new draw is $w$.

```{code-cell} ipython3
def plot_belief_dynamics(F_a=1, F_b=1, G_a=3, G_b=1.2):
    f, g, roots = create_model(F_a, F_b, G_a, G_b)
    π_grid = np.linspace(1e-3, 1 - 1e-3, 100)

    # Change in π at each point of a coarse grid over (π, w)
    W = np.arange(0.01, 0.99, 0.08)
    Π = np.arange(0.01, 0.99, 0.08)
    lw = (f(W) / g(W))[:, None]     # likelihood ratio at each w, as a column
    ΔΠ = Π * (lw / (Π * lw + 1 - Π) - 1)
    ΔW = np.zeros_like(ΔΠ)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.quiver(Π, W, ΔΠ, ΔW, scale=2, color='r', alpha=0.8)
    ax.fill_between(π_grid, 0, roots[0], color='blue', alpha=0.15)
    ax.fill_between(π_grid, roots[0], roots[1], color='green', alpha=0.15)
    ax.fill_between(π_grid, roots[1], 1, color='blue', alpha=0.15)
    ax.hlines(roots, 0, 1, linestyle='--')
    ax.set(xlabel=r'$\pi$', ylabel='$w$')
    ax.grid()
    plt.show()
```

Again we use the same pair of distributions.

```{code-cell} ipython3
plot_belief_dynamics()
```

Arrows pointing right show when Bayes' Law makes $\pi$ increase and arrows pointing left show when it makes $\pi$ decrease.

Lengths of the arrows show magnitudes of the force from Bayes' Law impelling $\pi$ to change.

These lengths depend on both the prior probability $\pi$ on the abscissa axis and the evidence in the form
of the current draw of $w$ on the ordinate axis.

Arrows point right in the blue regions, where $\ell(w) > 1$, and left in the green region between the
dashed lines, where $\ell(w) < 1$.

For these parameters the upper blue region is a thin sliver just below $w = 1$, since $\ell(w)$ returns above
one only for draws very close to the upper limit.

### Another instance

Next we use our code to create graphs for another instance of our model.

We keep $F$ the same as in the preceding instance, namely a uniform distribution, but now assume that $G$
is a Beta distribution with parameters $G_a=2, G_b=1.6$.

```{code-cell} ipython3
plot_likelihood_ratio(G_a=2, G_b=1.6)
plot_densities(G_a=2, G_b=1.6)
plot_belief_dynamics(G_a=2, G_b=1.6)
```

Notice how the likelihood ratio, the densities, and the arrows compare with the previous instance of our example.

## Appendix

### Sample paths of $\pi_t$

Now we'll have some fun by plotting multiple realizations of sample paths of $\pi_t$ under two possible
assumptions about nature's choice of distribution, namely

- that nature permanently draws from $F$
- that nature permanently draws from $G$

Outcomes depend on a peculiar property of likelihood ratio processes  discussed in
{ref}`this section <nature_likeli>` of {doc}`likelihood_ratio_process`.

Before simulating, it pays to rewrite Bayes' Law in terms of **odds** rather than probabilities.

In terms of the odds $\pi / (1 - \pi)$, the updating rule {eq}`eq_Bayes102` becomes

```{math}
:label: eq_odds

\frac{\pi_{t+1}}{1 - \pi_{t+1}} = \ell(w_{t+1}) \frac{\pi_{t}}{1 - \pi_{t}}
```

So Bayes' Law is *multiplicative* in the odds, and iterating {eq}`eq_odds` back to the prior $\pi_{-1}$ yields

```{math}
:label: eq_odds_product

\frac{\pi_{t}}{1 - \pi_{t}} = \frac{\pi_{-1}}{1 - \pi_{-1}} \prod_{s=0}^{t} \ell(w_{s})
```

The belief path is thus a running product of likelihood ratios, which is why properties of likelihood
ratio processes govern its behavior.

Taking logarithms turns the product into a sum:

$$
\log \frac{\pi_{t}}{1 - \pi_{t}} = \log \frac{\pi_{-1}}{1 - \pi_{-1}} + \sum_{s=0}^{t} \log \ell(w_{s})
$$

We simulate with this sum.

It avoids overflow in the product and loss of precision when $\pi_t$ is close to $0$ or $1$.

It also means that we can simulate an entire ensemble of paths with a cumulative sum, with no
iteration over dates.

```{code-cell} ipython3
def simulate(rng, a, b, T=50, N=1000, π_init=0.5,
             F_a=1, F_b=1, G_a=3, G_b=1.2):
    """
    Simulate N paths of the belief π when nature draws w_0, ..., w_{T-1}
    IID from Beta(a, b).  Returns an array of shape (N, T+1) whose column 0
    holds the prior π_{-1} = π_init and whose column t+1 holds π_t.
    """
    w = rng.beta(a, b, size=(N, T))
    log_l = np.log(p(w, F_a, F_b)) - np.log(p(w, G_a, G_b))
    log_odds = np.log(π_init / (1 - π_init)) + np.cumsum(log_l, axis=1)
    return np.column_stack([np.full(N, π_init), expit(log_odds)])
```

The paths are plotted by the next function.

```{code-cell} ipython3
def plot_paths(π_paths):
    "Plot belief paths against the dates t = -1, 0, ..., T-1."
    t = np.arange(π_paths.shape[1]) - 1
    fig, ax = plt.subplots()
    ax.plot(t, π_paths.T, color='b', lw=0.8, alpha=0.5)
    ax.set(xlabel='$t$', ylabel=r'$\pi_t$')
    plt.show()
```

We begin by generating $N$ simulated $\{\pi_t\}$ paths with $T$
periods when the sequence is truly IID draws from $F$. We set an initial prior $\pi_{-1} = .5$.

```{code-cell} ipython3
rng = np.random.default_rng(42)
T = 50
```

```{code-cell} ipython3
# when nature selects F
π_paths_F = simulate(rng, a=1, b=1, T=T, N=1000)
plot_paths(π_paths_F)
```

In the above example,  for most paths $\pi_t \rightarrow 1$.

So Bayes' Law evidently eventually
discovers the truth for most of our paths.

Next, we generate paths with $T$
periods when the sequence is truly IID draws from $G$. Again, we set the initial prior $\pi_{-1} = .5$.

```{code-cell} ipython3
# when nature selects G
π_paths_G = simulate(rng, a=3, b=1.2, T=T, N=1000)
plot_paths(π_paths_G)
```

In the above graph we observe that now  most paths $\pi_t \rightarrow 0$.

### Rates of convergence

We study rates of  convergence of $\pi_t$ to $1$ when nature generates the data as IID draws from $F$
and of convergence of $\pi_t$ to $0$ when nature generates  IID draws from $G$.

We do this by averaging across simulated paths of $\{\pi_t\}_{t=-1}^{T-1}$.

Using   $N$ simulated $\pi_t$ paths, we compute
$1 - \frac{1}{N}\sum_{i=1}^{N}\pi_{i,t}$ at each $t$ when the data are generated as draws from  $F$
and compute $\frac{1}{N}\sum_{i=1}^{N}\pi_{i,t}$ when the data are generated as draws from $G$.

```{code-cell} ipython3
t_grid = np.arange(-1, T)
fig, ax = plt.subplots()
ax.plot(t_grid, 1 - np.mean(π_paths_F, axis=0), label='F generates')
ax.plot(t_grid, np.mean(π_paths_G, axis=0), label='G generates')
ax.set(xlabel='$t$', title='convergence')
ax.legend()
plt.show()
```

The two curves nearly coincide.

An exact identity explains why:

```{math}
:label: eq_conv_identity

\pi_{-1} E_f[1 - \pi_t] = (1 - \pi_{-1}) E_g[\pi_t] = E[\pi_t (1 - \pi_t)]
```

Here $E_f$ and $E_g$ denote expectations under IID draws from $F$ and from $G$, and $E$ denotes the expectation under the decision maker's subjective joint distribution {eq}`eq_definetti`.

To prove {eq}`eq_conv_identity`, let $\phi$ be a bounded function of the history $w^t$.

Because $\pi_t = \mathbb{P}\{q = f \,|\, w^t\}$, the law of iterated expectations implies

$$
E[\pi_t \phi(w^t)] = E[\mathbf{1}\{q = f\} \phi(w^t)] = \pi_{-1} E_f[\phi(w^t)] .
$$

Setting $\phi = 1 - \pi_t$ equates the first and third terms of {eq}`eq_conv_identity`.

The same argument applied to $1 - \pi_t = \mathbb{P}\{q = g \,|\, w^t\}$ with $\phi = \pi_t$ equates the second and third terms.

With $\pi_{-1} = 0.5$, the identity says that $E_f[1 - \pi_t] = E_g[\pi_t]$ for every $t$, so the two simulated curves differ only by sampling error.

With another prior, the two curves differ by the constant factor $(1 - \pi_{-1})/\pi_{-1}$.

The identity does not say that typical paths converge at the same rate under $F$ and under $G$.

By {eq}`eq_odds_product` and the law of large numbers, $\frac{1}{t+1} \log \frac{\pi_t}{1 - \pi_t}$ converges to $E_f[\log \ell(w)] = KL(f, g) = 0.759$ when $F$ generates the data and to $E_g[\log \ell(w)] = -KL(g, f) = -0.344$ when $G$ generates the data, where $KL(f, g) = \int f(w) \log \frac{f(w)}{g(w)} dw$ is the Kullback–Leibler divergence of $g$ from $f$.

So a typical path approaches $1$ under $F$ faster than it approaches $0$ under $G$.

The averages plotted above are dominated by rare paths along which $\pi_t$ leans the wrong way for a long time.

Both averages decay at the exponential rate

$$
C(f, g) = - \log \min_{s \in [0,1]} \int f(w)^{s} g(w)^{1-s} dw = 0.121 ,
$$

the Chernoff entropy of $f$ and $g$ described in {doc}`divergence_measures`.

Exercise {ref}`exchangeable_ex2` verifies these claims.

### Graph of ensemble dynamics of $\pi_t$

More insights about the dynamics of $\{\pi_t\}$ can be gleaned by computing
conditional expectations of $\frac{\pi_{t+1}}{\pi_{t}}$ as functions of $\pi_t$ via integration with respect
to the pertinent probability distribution:

$$
\begin{aligned}
E\left[\frac{\pi_{t+1}}{\pi_{t}}\biggm|q=a, \pi_{t}\right] &=E\left[\frac{\ell\left(w_{t+1}\right)}{\pi_{t}\ell\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}\biggm|q= a, \pi_{t}\right], \\
    &=\int_{0}^{1}\frac{\ell\left(w_{t+1}\right)}{\pi_{t}\ell\left(w_{t+1}\right)+\left(1-\pi_{t}\right)} a\left(w_{t+1}\right)dw_{t+1}
\end{aligned}
$$

where $a =f,g$.

The following code approximates the integral above:

```{code-cell} ipython3
def plot_expected_ratio(F_a=1, F_b=1, G_a=3, G_b=1.2):
    # Build f and g directly: here they can coincide, so ℓ(w) = 1 has no root
    f = lambda w: p(w, F_a, F_b)
    g = lambda w: p(w, G_a, G_b)
    l = lambda w: f(w) / g(w)
    π_grid = np.linspace(0.02, 0.98, 100)

    fig, ax = plt.subplots()
    for label, a in [('f', f), ('g', g)]:
        integrand = lambda w, π: a(w) * l(w) / (π * l(w) + 1 - π)
        ratios = [quad(integrand, 0, 1, args=(π,))[0] for π in π_grid]
        ax.plot(π_grid, ratios, label=f'{label} generates')

    ax.hlines(1, 0, 1, linestyle='--')
    ax.set(xlabel=r'$\pi_t$', ylabel=r'$E[\pi_{t+1} / \pi_t]$')
    ax.legend()
    plt.show()
```

First, consider the case where $F_a=F_b=1$ and
$G_a=3, G_b=1.2$.

```{code-cell} ipython3
plot_expected_ratio()
```

The above graph shows that when $F$ generates the data, $\pi_t$ on average always heads north, while
when $G$ generates the data, $\pi_t$ heads south.

Two inequalities hold for every pair $f \neq g$ and every $\pi_t = \pi \in (0,1)$.

Let $r(\ell) = \ell / (\pi \ell + 1 - \pi)$ be the ratio $\pi_{t+1}/\pi_t$ as a function of $\ell = \ell(w_{t+1})$.

The function $r$ is strictly concave and $E_g[\ell(w)] = \int f(w) dw = 1$, so Jensen's inequality implies

$$
E_g[r(\ell(w))] < r(1) = 1 .
$$

The $\pi$-weighted average of the two expectations equals one:

$$
\pi E_f[r(\ell(w))] + (1 - \pi) E_g[r(\ell(w))] = \int \frac{f(w)}{\pi f(w) + (1-\pi) g(w)} \bigl(\pi f(w) + (1-\pi) g(w)\bigr) dw = 1 .
$$

This equality is the martingale property of $\pi_t$ under the decision maker's subjective distribution, studied in {doc}`likelihood_bayes`.

It and the first inequality imply

$$
E_f[r(\ell(w))] > 1 .
$$

Thus $\pi_t$ is a submartingale when $F$ generates the data and a supermartingale when $G$ generates the data.

Next, we'll look at a degenerate case in which  $f$ and $g$ are identical beta
distributions, and $F_a=G_a=3, F_b=G_b=1.2$.

In a sense, here  there
is nothing to learn.

```{code-cell} ipython3
plot_expected_ratio(F_a=3, F_b=1.2)
```

The above graph says that $\pi_t$ is inert and  remains at its initial value.

Finally, let's look at a case in which  $f$ and $g$ are neither very
different nor identical, in particular one in which  $F_a=2, F_b=1$ and
$G_a=3, G_b=1.2$.

```{code-cell} ipython3
plot_expected_ratio(F_a=2, F_b=1, G_a=3, G_b=1.2)
```

The two inequalities still hold, but both curves now stay close to one.

On the plotted grid, $E_f[\pi_{t+1}/\pi_t]$ never exceeds $1.16$ and $E_g[\pi_{t+1}/\pi_t]$ never falls below $0.978$.

In the first case the corresponding numbers are $5.38$ and $0.811$.

The densities Beta$(2,1)$ and Beta$(3,1.2)$ are close to each other: $KL(f, g) = 0.052$ and $KL(g, f) = 0.039$, compared with $0.759$ and $0.344$ in the first case.

A draw moves $\pi_t$ only a little on average, so learning is slow.

## Sequels

We'll apply and dig deeper into some of the ideas presented in this lecture:

* {doc}`likelihood_ratio_process` describes **likelihood ratio processes**
  and their role in frequentist and Bayesian statistical theories
* {doc}`likelihood_bayes` studies how a likelihood ratio process drives the posterior probability $\pi_t$ and shows that $\pi_t$ is a martingale under the decision maker's subjective distribution {eq}`eq_definetti`
* {doc}`mix_model` studies a decision maker who uses this lecture's statistical model when nature actually mixes $F$ and $G$ anew each period
* {doc}`merging_of_opinions` studies the Beta–Bernoulli model, an instance of de Finetti's representation {eq}`eq_definetti_bernoulli`, and asks when the opinions of two Bayesians with different priors merge
* {doc}`divergence_measures` describes the Kullback–Leibler divergence and the Chernoff entropy that govern rates of learning
* {doc}`wald_friedman_2` studies a decision maker with this lecture's beliefs who decides sequentially when to stop sampling
* {doc}`navy_captain` studies a World War II US Navy Captain's hunch that a (frequentist) decision rule that the Navy had told
  him to use was  inferior to a sequential rule that Abraham
  Wald had not yet designed

## Exercises

```{exercise}
:label: exchangeable_ex1

This exercise shows that a mixture of IID sequences has nonnegative serial covariances and that a finite exchangeable sequence need not be a mixture of IID sequences.

1. Prove identity {eq}`eq_hcov`.

2. Let $\phi$ be a function for which $E_f[\phi(W)]$ and $E_g[\phi(W)]$ are finite. Use {eq}`eq_hcov` to show that under the joint density {eq}`eq_definetti`

$$
\textrm{Cov}\bigl(\phi(W_0), \phi(W_1)\bigr) = \tilde \pi (1 - \tilde \pi) \bigl(E_f[\phi(W)] - E_g[\phi(W)]\bigr)^2 \geq 0 .
$$

3. For this lecture's densities $f = $ Beta$(1, 1)$ and $g = $ Beta$(3, 1.2)$ and $\tilde \pi = 0.5$, compute $\textrm{Cov}(W_0, W_1)$ from the formula in part 2 and compare it with a simulation.

4. An urn contains one red ball and one black ball. We draw both balls without replacement. Let $W_t = 1$ if draw $t \in \{0, 1\}$ is red and $W_t = 0$ if it is black. Show that $(W_0, W_1)$ is exchangeable and that $\textrm{Cov}(W_0, W_1) < 0$. Explain why $(W_0, W_1)$ is not a mixture of IID pairs and why de Finetti's theorem requires an infinite sequence.
```

```{solution-start} exchangeable_ex1
:class: dropdown
```

Part 1.

With $h(W_t) = \tilde \pi f(W_t) + (1 - \tilde \pi) g(W_t)$,

$$
\begin{aligned}
h(W_0) h(W_1) & = \tilde \pi^2 f(W_0) f(W_1) + \tilde \pi (1 - \tilde \pi) \bigl(f(W_0) g(W_1) + g(W_0) f(W_1)\bigr) + (1 - \tilde \pi)^2 g(W_0) g(W_1) .
\end{aligned}
$$

Subtracting this from $h(W_0, W_1) = \tilde \pi f(W_0) f(W_1) + (1 - \tilde \pi) g(W_0) g(W_1)$ and using $\tilde \pi - \tilde \pi^2 = (1 - \tilde \pi) - (1 - \tilde \pi)^2 = \tilde \pi (1 - \tilde \pi)$ gives

$$
h(W_0, W_1) - h(W_0) h(W_1) = \tilde \pi (1 - \tilde \pi) \bigl(f(W_0) f(W_1) + g(W_0) g(W_1) - f(W_0) g(W_1) - g(W_0) f(W_1)\bigr) ,
$$

which factors into the right side of {eq}`eq_hcov`.

Part 2.

Multiply {eq}`eq_hcov` by $\phi(W_0) \phi(W_1)$ and integrate over $W_0$ and $W_1$:

$$
\textrm{Cov}\bigl(\phi(W_0), \phi(W_1)\bigr) = \tilde \pi (1 - \tilde \pi) \int \phi(w) \bigl(f(w) - g(w)\bigr) dw \int \phi(w) \bigl(f(w) - g(w)\bigr) dw = \tilde \pi (1 - \tilde \pi) \bigl(E_f[\phi(W)] - E_g[\phi(W)]\bigr)^2 .
$$

The same conclusion holds for a mixture of IID sequences with any mixing measure.

Let $Q$ be the distribution, drawn from the mixing measure, from which the $W_t$ are IID draws.

The law of total covariance gives

$$
\textrm{Cov}\bigl(\phi(W_0), \phi(W_1)\bigr) = E\bigl[\textrm{Cov}(\phi(W_0), \phi(W_1) \,|\, Q)\bigr] + \textrm{Cov}\bigl(E[\phi(W_0) | Q], E[\phi(W_1) | Q]\bigr) = 0 + \textrm{Var}\bigl(E[\phi(W) | Q]\bigr) \geq 0 .
$$

Part 3.

The means of Beta$(1,1)$ and Beta$(3, 1.2)$ are $1/2$ and $3/4.2$.

```{code-cell} ipython3
π_tilde = 0.5
μ_f, μ_g = 1 / (1 + 1), 3 / (3 + 1.2)
cov_theory = π_tilde * (1 - π_tilde) * (μ_f - μ_g)**2

rng = np.random.default_rng(1234)
N = 1_000_000
nature_draws_f = rng.random(N) < π_tilde
W = np.where(nature_draws_f[:, None],
             rng.beta(1, 1, size=(N, 2)),
             rng.beta(3, 1.2, size=(N, 2)))
cov_sim = np.cov(W[:, 0], W[:, 1])[0, 1]

print(f"Cov(W_0, W_1): formula {cov_theory:.5f}, simulation {cov_sim:.5f}")
```

Part 4.

The first draw is red with probability $1/2$, and then the second draw is black with probability one.

So $\textrm{Prob}(W_0 = 1, W_1 = 0) = \textrm{Prob}(W_0 = 0, W_1 = 1) = 1/2$ and $\textrm{Prob}(W_0 = W_1) = 0$.

The joint distribution is invariant to exchanging $W_0$ and $W_1$, so $(W_0, W_1)$ is exchangeable.

$E[W_0] = E[W_1] = 1/2$ and $E[W_0 W_1] = 0$, so $\textrm{Cov}(W_0, W_1) = -1/4$.

By part 2, every mixture of IID pairs has $\textrm{Cov}(W_0, W_1) \geq 0$.

So $(W_0, W_1)$ is not a mixture of IID pairs.

If $(W_0, W_1)$ were the first two members of an infinite exchangeable sequence, de Finetti's theorem would make the pair a mixture of IID pairs.

So the urn sequence cannot be extended to an infinite exchangeable sequence.

More generally, let $W_0, \ldots, W_{n-1}$ be exchangeable with common variance $\sigma^2 > 0$ and common pairwise correlation $\rho$.

Then $0 \leq \textrm{Var}(\sum_t W_t) = n \sigma^2 (1 + (n-1) \rho)$, so $\rho \geq -1/(n-1)$.

The urn attains this bound with $n = 2$ and $\rho = -1$.

The bound approaches $0$ as $n \rightarrow \infty$.

```{solution-end}
```

```{exercise}
:label: exchangeable_ex2

This exercise studies identity {eq}`eq_conv_identity` and rates of convergence of $\pi_t$.

1. Prove {eq}`eq_conv_identity` by a second route. Use $\pi_t = \pi_{-1} L_t / (\pi_{-1} L_t + 1 - \pi_{-1})$, where $L_t = \prod_{s=0}^t \ell(w_s)$, and $E_f[\phi(w^t)] = E_g[L_t \phi(w^t)]$ for every bounded function $\phi$ of $w^t$.

2. Set $\pi_{-1} = 0.3$. Plot $1 - E_f[\pi_t]$ and $E_g[\pi_t]$ against $t$, then plot $\pi_{-1}(1 - E_f[\pi_t])$ and $(1 - \pi_{-1}) E_g[\pi_t]$ against $t$.

3. Compute the slopes over $t$ of the medians across paths of $\log(\pi_t / (1 - \pi_t))$ under $F$ and under $G$. Compare them with $KL(f, g)$ and $-KL(g, f)$.

4. Compute the slope over $t$ of $\log(1 - E_f[\pi_t])$ and compare it with $-C(f, g)$, where $C(f, g)$ is the Chernoff entropy.

Hint for part 4: rare paths dominate $1 - E_f[\pi_t]$, so plain Monte Carlo estimates of it are imprecise for large $t$. Let $A(s) = \int f(w)^s g(w)^{1-s} dw$, let $s^*$ minimize $A(s)$ on $[0,1]$, and let $m(w) = f(w)^{s^*} g(w)^{1-s^*} / A(s^*)$. Then $f(w) / m(w) = A(s^*) \ell(w)^{1 - s^*}$, so

$$
E_f[\phi(w^t)] = A(s^*)^{t+1} E_m\bigl[\phi(w^t) L_t^{1-s^*}\bigr] .
$$

Draw from $m$ to estimate the expectation on the right. With $f$ uniform, $m$ is the Beta$(3 - 2 s^*, 1.2 - 0.2 s^*)$ density.
```

```{solution-start} exchangeable_ex2
:class: dropdown
```

Part 1.

Under $F$, the density of $w^t$ is $f(w_0) \cdots f(w_t) = L_t \, g(w_0) \cdots g(w_t)$, which gives $E_f[\phi(w^t)] = E_g[L_t \phi(w^t)]$.

Since $1 - \pi_t = (1 - \pi_{-1}) / (\pi_{-1} L_t + 1 - \pi_{-1})$,

$$
\pi_{-1} E_f[1 - \pi_t] = E_g\left[\frac{\pi_{-1} L_t (1 - \pi_{-1})}{\pi_{-1} L_t + 1 - \pi_{-1}}\right] = (1 - \pi_{-1}) E_g[\pi_t] .
$$

The subjective distribution {eq}`eq_definetti` has density $\pi_{-1} f(w^t) + (1 - \pi_{-1}) g(w^t) = (\pi_{-1} L_t + 1 - \pi_{-1}) g(w^t)$, so

$$
E[\pi_t (1 - \pi_t)] = E_g\bigl[(\pi_{-1} L_t + 1 - \pi_{-1}) \pi_t (1 - \pi_t)\bigr] = E_g\bigl[\pi_{-1} L_t (1 - \pi_t)\bigr] = \pi_{-1} E_f[1 - \pi_t] .
$$

Part 2.

We write a version of `simulate` that returns log odds, because $\pi_t$ rounds to exactly $1$ once its log odds exceed about $37$.

```{code-cell} ipython3
def simulate_log_odds(rng, a, b, T, N, π_init):
    "Log odds of π_t for t = -1, 0, ..., T-1 along N paths of IID Beta(a, b) draws."
    w = rng.beta(a, b, size=(N, T))
    log_l = np.log(p(w, 1, 1)) - np.log(p(w, 3, 1.2))
    log_odds_init = np.log(π_init / (1 - π_init))
    log_odds = log_odds_init + np.cumsum(log_l, axis=1)
    return np.column_stack([np.full(N, log_odds_init), log_odds])

π_init, T_ex, N_ex = 0.3, 40, 100_000
rng = np.random.default_rng(2024)
lo_F = simulate_log_odds(rng, 1, 1, T_ex, N_ex, π_init)
lo_G = simulate_log_odds(rng, 3, 1.2, T_ex, N_ex, π_init)
t_ex = np.arange(-1, T_ex)

err_F = np.mean(expit(-lo_F), axis=0)      # 1 - E_f[π_t]
err_G = np.mean(expit(lo_G), axis=0)       # E_g[π_t]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(t_ex, err_F, label=r'$1 - E_f[\pi_t]$')
axes[0].plot(t_ex, err_G, label=r'$E_g[\pi_t]$')
axes[1].plot(t_ex, π_init * err_F, label=r'$\pi_{-1}(1 - E_f[\pi_t])$')
axes[1].plot(t_ex, (1 - π_init) * err_G, '--',
             label=r'$(1 - \pi_{-1}) E_g[\pi_t]$')
for ax in axes:
    ax.set_xlabel('$t$')
    ax.legend()
plt.tight_layout()
plt.show()

for t in [0, 5, 10, 20]:
    print(f"t = {t:2d}: (1 - E_f[π_t]) / E_g[π_t] = {err_F[t+1] / err_G[t+1]:.3f}")
print(f"(1 - π_init) / π_init = {(1 - π_init) / π_init:.3f}")
```

The two curves in the left panel differ by the factor $7/3$; the two curves in the right panel coincide up to sampling error.

Part 3.

```{code-cell} ipython3
f = lambda w: p(w, 1, 1)
g = lambda w: p(w, 3, 1.2)
KL_fg = quad(lambda w: f(w) * np.log(f(w) / g(w)), 0, 1)[0]
KL_gf = quad(lambda w: g(w) * np.log(g(w) / f(w)), 0, 1)[0]

slope_F = np.polyfit(t_ex, np.median(lo_F, axis=0), 1)[0]
slope_G = np.polyfit(t_ex, np.median(lo_G, axis=0), 1)[0]
print(f"slope of median log odds under F: {slope_F:.3f},  KL(f, g) = {KL_fg:.3f}")
print(f"slope of median log odds under G: {slope_G:.3f}, -KL(g, f) = {-KL_gf:.3f}")
```

Typical paths converge faster under $F$ than under $G$.

Part 4.

We compute the Chernoff entropy and the minimizer $s^*$.

```{code-cell} ipython3
A = lambda s: quad(lambda w: f(w)**s * g(w)**(1 - s), 0, 1)[0]
res = op.minimize_scalar(A, bounds=(0, 1), method='bounded')
s_star, C = res.x, -np.log(res.fun)
print(f"s* = {s_star:.3f}, C(f, g) = {C:.4f}")
```

Next we estimate $\log(1 - E_f[\pi_t])$ by importance sampling from $m$ for $t$ up to $199$.

```{code-cell} ipython3
T_is, N_is = 200, 100_000
rng = np.random.default_rng(2025)
w = rng.beta(3 - 2 * s_star, 1.2 - 0.2 * s_star, size=(N_is, T_is))
log_L = np.cumsum(np.log(f(w)) - np.log(g(w)), axis=1)    # log L_t, t = 0, ..., 199
t_is = np.arange(T_is)

log_odds = np.log(π_init / (1 - π_init)) + log_L
weights = np.exp((1 - s_star) * log_L)                    # L_t^(1 - s*)
log_err_F = (t_is + 1) * np.log(A(s_star)) \
            + np.log(np.mean(expit(-log_odds) * weights, axis=0))

for t in [10, 20, 30]:
    print(f"t = {t}: log(1 - E_f[π_t]) importance sampling {log_err_F[t]:.3f}, "
          f"plain Monte Carlo {np.log(err_F[t+1]):.3f}")

late = t_is >= 100
slope = np.polyfit(t_is[late], log_err_F[late], 1)[0]
slope_adj = np.polyfit(t_is[late], log_err_F[late] + 0.5 * np.log(t_is[late] + 1), 1)[0]
print(f"slope of log(1 - E_f[π_t]) for t in [100, 199]: {slope:.4f}")
print(f"slope after adding log(t+1)/2:                 {slope_adj:.4f}")
print(f"-C(f, g):                                      {-C:.4f}")
```

The two estimates agree at small $t$.

Under $m$, $E_m[\log \ell(w)] = A'(s^*)/A(s^*) = 0$, so $\log L_t$ is a random walk without drift.

The expectation $E_m[(1 - \pi_t) L_t^{1-s^*}]$ therefore decays only like $t^{-1/2}$, and

$$
1 - E_f[\pi_t] \approx K \, t^{-1/2} e^{-C(f,g) t}
$$

for a constant $K$.

The fitted slope is close to $-C(f,g)$, and adding $\frac{1}{2}\log(t+1)$ to remove the factor $t^{-1/2}$ brings it closer.

By {eq}`eq_conv_identity`, $E_g[\pi_t]$ decays at the same rate.

```{solution-end}
```

```{exercise}
:label: exchangeable_ex3

A Pólya urn starts with $a$ red balls and $b$ black balls.

At each date $t = 0, 1, \ldots$ we draw a ball at random, set $W_t = 1$ if it is red and $W_t = 0$ if it is black, and return it to the urn together with one more ball of the same color.

1. Show that $\textrm{Prob}(W_0 = 1, W_1 = 0) = \textrm{Prob}(W_0 = 0, W_1 = 1)$. More generally, show that the probability of a sequence $(w_0, \ldots, w_{n-1})$ depends only on the number $k$ of red draws, so that $W_0, W_1, \ldots$ is exchangeable. Find the mixing measure $\mu$ in de Finetti's representation {eq}`eq_definetti_bernoulli`.

2. Simulate many urns with $a = 2$ and $b = 3$ for $n = 2000$ draws. Compare the distribution across urns of the fraction of red draws with the Beta$(a, b)$ density.

3. Show that the fraction of red balls in the urn after $n$ draws, $k$ of them red, equals the probability $\textrm{Prob}(W_n = 1 \,|\, w_0, \ldots, w_{n-1})$ that a Bayesian with a Beta$(a, b)$ prior over the success probability $\theta$ of IID Bernoulli draws attaches to the next draw being red.
```

```{solution-start} exchangeable_ex3
:class: dropdown
```

Part 1.

$\textrm{Prob}(W_0 = 1, W_1 = 0) = \frac{a}{a+b} \frac{b}{a+b+1}$ and $\textrm{Prob}(W_0 = 0, W_1 = 1) = \frac{b}{a+b} \frac{a}{a+b+1}$.

The two are equal.

For a sequence of $n$ draws with $k$ red draws, the denominators of the successive conditional probabilities are $a+b, a+b+1, \ldots, a+b+n-1$ in every order.

The numerators for the red draws are $a, a+1, \ldots, a+k-1$, and the numerators for the black draws are $b, b+1, \ldots, b+n-k-1$, in every order.

So

$$
\textrm{Prob}(W_0 = w_0, \ldots, W_{n-1} = w_{n-1}) = \frac{\prod_{j=0}^{k-1} (a+j) \prod_{j=0}^{n-k-1} (b+j)}{\prod_{j=0}^{n-1} (a+b+j)} = \frac{B(a+k, b+n-k)}{B(a,b)} ,
$$

where $B$ is the Beta function.

The right side equals $\int_0^1 \theta^k (1-\theta)^{n-k} \mu(d\theta)$ with $\mu$ the Beta$(a,b)$ distribution.

That is de Finetti's representation {eq}`eq_definetti_bernoulli`.

Part 2.

```{code-cell} ipython3
from scipy.stats import beta as beta_dist, kstest

def polya_urn(rng, a, b, n, N):
    "Simulate N Pólya urns for n draws; return the N x n array of draws."
    red = np.full(N, float(a))
    total = float(a + b)
    W = np.empty((N, n), dtype=np.int8)
    for t in range(n):
        W[:, t] = rng.random(N) < red / total
        red += W[:, t]
        total += 1
    return W

a, b, n, N = 2, 3, 2000, 20_000
rng = np.random.default_rng(7)
W = polya_urn(rng, a, b, n, N)
freq = W.mean(axis=1)

grid = np.linspace(0, 1, 200)
fig, ax = plt.subplots()
ax.hist(freq, bins=50, density=True, alpha=0.5,
        label=f'fraction of red draws, $n = {n}$')
ax.plot(grid, beta_dist.pdf(grid, a, b), lw=2, label=f'Beta$({a}, {b})$ density')
ax.set_xlabel(r'$\theta$')
ax.legend()
plt.show()

print(f"Prob(W_0=1, W_1=0) = {np.mean((W[:, 0] == 1) & (W[:, 1] == 0)):.4f}, "
      f"Prob(W_0=0, W_1=1) = {np.mean((W[:, 0] == 0) & (W[:, 1] == 1)):.4f}, "
      f"exact = {a * b / ((a + b) * (a + b + 1)):.4f}")
print(f"Kolmogorov-Smirnov test against Beta({a}, {b}): "
      f"p-value = {kstest(freq, beta_dist(a, b).cdf).pvalue:.3f}")
```

The fraction of red draws is distributed approximately Beta$(2, 3)$ across urns, and the Kolmogorov–Smirnov test does not reject that distribution.

Part 3.

After $n$ draws with $k$ red draws, the urn holds $a + k$ red balls out of $a + b + n$ balls, so the next draw is red with probability $(a+k)/(a+b+n)$.

A Bayesian with a Beta$(a, b)$ prior over $\theta$ has the Beta$(a + k, b + n - k)$ posterior after observing $k$ successes in $n$ IID Bernoulli trials.

His probability that the next draw is a success is the posterior mean of $\theta$, namely $(a + k)/(a + b + n)$.

The urn and the Bayesian agree.

The next cell checks this by grouping the simulated urns according to the number of red balls among the first $10$ draws.

Frequencies computed from few urns are less precise.

```{code-cell} ipython3
n0 = 10
k = W[:, :n0].sum(axis=1)
for k_val in range(0, n0 + 1, 2):
    sel = k == k_val
    print(f"k = {k_val:2d} ({sel.sum():4d} urns): "
          f"frequency of red at draw {n0} = {W[sel, n0].mean():.3f}, "
          f"(a + k) / (a + b + n) = {(a + k_val) / (a + b + n0):.3f}")
```

```{solution-end}
```
