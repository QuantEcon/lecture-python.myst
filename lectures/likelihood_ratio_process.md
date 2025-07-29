---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(likelihood_ratio_process)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Likelihood Ratio Processes

```{contents} Contents
:depth: 2
```


## Overview

This lecture describes likelihood ratio processes and some of their uses.

We'll study the same  setting that is also used in  {doc}`this lecture on exchangeability <exchangeable>`.

Among the things that we'll learn are

* How a likelihood ratio process is a key ingredient in frequentist hypothesis testing
* How a **receiver operator characteristic curve** summarizes information about a false alarm probability and power in frequentist hypothesis testing
* How a statistician can combine frequentist probabilities of type I and type II errors to form posterior probabilities of mistakes in a model selection or in an individual-classification problem
* How likelihood ratios helped Lawrence Blume and David Easley formulate an answer to  ''If you're so smart, why aren't you rich?'' {cite}`blume2006if`
* How to use a Kullback-Leibler divergence to quantify the difference between two probability distributions with the same support 
* How during World War II the United States Navy devised a decision rule for doing quality control on lots of ammunition, a topic that sets the stage for {doc}`this lecture <wald_friedman>`
* A peculiar property of likelihood ratio processes



Let's start by importing some Python tools.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from numba import vectorize, jit
from math import gamma
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import beta as beta_dist
import pandas as pd
from IPython.display import display, Math
```

## Likelihood Ratio Process

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

(nature_likeli)=
## Nature Permanently Draws from Density g

We first simulate the likelihood ratio process when nature permanently
draws from $g$.

```{code-cell} ipython3
l_arr_g = simulate(G_a, G_b)
l_seq_g = np.cumprod(l_arr_g, axis=1)
```

```{code-cell} ipython3
N, T = l_arr_g.shape

for i in range(N):

    plt.plot(range(T), l_seq_g[i, :], color='b', lw=0.8, alpha=0.5)

plt.ylim([0, 3])
plt.title("$L(w^{t})$ paths");
```

Evidently, as sample length $T$ grows, most probability mass
shifts toward zero

To see this more clearly, we plot over time the fraction of
paths $L\left(w^{t}\right)$ that fall in the interval
$\left[0, 0.01\right]$.

```{code-cell} ipython3
plt.plot(range(T), np.sum(l_seq_g <= 0.01, axis=0) / N)
plt.show()
```

Despite the evident convergence of most probability mass to a
very small interval near $0$, the unconditional mean of
$L\left(w^t\right)$ under probability density $g$ is
identically $1$ for all $t$.

To verify this assertion, first notice that as mentioned earlier the unconditional mean
$E\left[\ell \left(w_{t}\right)\bigm|q=g\right]$ is $1$ for
all $t$:

$$
\begin{aligned}
E\left[\ell \left(w_{t}\right)\bigm|q=g\right]  &=\int\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}g\left(w_{t}\right)dw_{t} \\
    &=\int f\left(w_{t}\right)dw_{t} \\
    &=1,
\end{aligned}
$$

which immediately implies

$$
\begin{aligned}
E\left[L\left(w^{1}\right)\bigm|q=g\right]  &=E\left[\ell \left(w_{1}\right)\bigm|q=g\right]\\
    &=1.\\
\end{aligned}
$$

Because $L(w^t) = \ell(w_t) L(w^{t-1})$ and
$\{w_t\}_{t=1}^t$ is an IID sequence, we have

$$
\begin{aligned}
E\left[L\left(w^{t}\right)\bigm|q=g\right]  &=E\left[L\left(w^{t-1}\right)\ell \left(w_{t}\right)\bigm|q=g\right] \\
         &=E\left[L\left(w^{t-1}\right)E\left[\ell \left(w_{t}\right)\bigm|q=g,w^{t-1}\right]\bigm|q=g\right] \\
     &=E\left[L\left(w^{t-1}\right)E\left[\ell \left(w_{t}\right)\bigm|q=g\right]\bigm|q=g\right] \\
    &=E\left[L\left(w^{t-1}\right)\bigm|q=g\right] \\
\end{aligned}
$$

for any $t \geq 1$.

Mathematical induction implies
$E\left[L\left(w^{t}\right)\bigm|q=g\right]=1$ for all
$t \geq 1$.

## Peculiar Property

How can $E\left[L\left(w^{t}\right)\bigm|q=g\right]=1$ possibly be true when most probability mass of the likelihood
ratio process is piling up near $0$ as
$t \rightarrow + \infty$?

The answer is that as $t \rightarrow + \infty$, the
distribution of $L_t$ becomes more and more fat-tailed:
enough mass shifts to larger and larger values of $L_t$ to make
the mean of $L_t$ continue to be one despite most of the probability mass piling up
near $0$.

To illustrate this peculiar property, we simulate many paths and
calculate the unconditional mean of $L\left(w^t\right)$ by
averaging across these many paths at each $t$.

```{code-cell} ipython3
l_arr_g = simulate(G_a, G_b, N=50000)
l_seq_g = np.cumprod(l_arr_g, axis=1)
```

It would be useful to use simulations to verify that unconditional means
$E\left[L\left(w^{t}\right)\right]$ equal unity by averaging across sample
paths.

But it would be too computer-time-consuming for us to do that here simply by applying a standard Monte Carlo simulation approach.

The reason is that the distribution of $L\left(w^{t}\right)$ is extremely skewed for large values of $t$.

Because the probability density in the right tail is close to $0$, it just takes too much computer time to sample enough points from the right tail.

We explain the problem in more detail in {doc}`this lecture <imp_sample>`.

There we describe an alternative way to compute the mean of a likelihood ratio by computing the mean of a _different_ random variable by sampling from a _different_ probability distribution.


## Nature Permanently Draws from Density f

Now suppose that before time $0$ nature permanently decided to draw repeatedly from density $f$.

While the mean of the likelihood ratio $\ell \left(w_{t}\right)$ under density
$g$ is $1$, its mean under the density $f$ exceeds one.

To see this, we compute

$$
\begin{aligned}
E\left[\ell \left(w_{t}\right)\bigm|q=f\right]  &=\int\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}f\left(w_{t}\right)dw_{t} \\
     &=\int\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}g\left(w_{t}\right)dw_{t} \\
     &=\int \ell \left(w_{t}\right)^{2}g\left(w_{t}\right)dw_{t} \\
     &=E\left[\ell \left(w_{t}\right)^{2}\mid q=g\right] \\
     &=E\left[\ell \left(w_{t}\right)\mid q=g\right]^{2}+Var\left(\ell \left(w_{t}\right)\mid q=g\right) \\
     &>E\left[\ell \left(w_{t}\right)\mid q=g\right]^{2} = 1 \\
       \end{aligned}
$$

This in turn implies that the unconditional mean of the likelihood ratio process $L(w^t)$
diverges toward $+ \infty$.

Simulations below confirm this conclusion.

Please note the scale of the $y$ axis.

```{code-cell} ipython3
l_arr_f = simulate(F_a, F_b, N=50000)
l_seq_f = np.cumprod(l_arr_f, axis=1)
```

```{code-cell} ipython3
N, T = l_arr_f.shape
plt.plot(range(T), np.mean(l_seq_f, axis=0))
plt.show()
```

We also plot the probability that $L\left(w^t\right)$ falls into
the interval $[10000, \infty)$ as a function of time and watch how
fast probability mass diverges to $+\infty$.

```{code-cell} ipython3
plt.plot(range(T), np.sum(l_seq_f > 10000, axis=0) / N)
plt.show()
```

## Likelihood Ratio Test

We now describe how to employ the machinery
of Neyman and Pearson {cite}`Neyman_Pearson` to test the hypothesis that history $w^t$ is generated by repeated
IID draws from density $f$.

Denote $q$ as the data generating process, so that
$q=f \text{ or } g$.

Upon observing a sample $\{W_i\}_{i=1}^t$, we want to decide
whether nature is drawing from $g$ or from $f$ by performing a (frequentist)
hypothesis test.

We specify

- Null hypothesis $H_0$: $q=f$,
- Alternative hypothesis $H_1$: $q=g$.

Neyman and Pearson proved that the best way to test this hypothesis is to use a **likelihood ratio test** that takes the
form:

- accept $H_0$ if $L(W^t) > c$,
- reject $H_0$ if $L(W^t) < c$,


where $c$ is a given discrimination threshold.

Setting $c =1$ is a common choice.

We'll discuss consequences of other choices of $c$ below.  

This test is *best* in the sense that it is **uniformly most powerful**.

To understand what this means, we have to define probabilities of two important events that
allow us to characterize a test associated with a given
threshold $c$.

The two probabilities are:


- Probability of a Type I error in which we reject $H_0$ when it is true: 
  
  $$
  \alpha \equiv  \Pr\left\{ L\left(w^{t}\right)<c\mid q=f\right\}
  $$

- Probability of a Type II error in which we accept $H_0$ when it is false:

  $$
  \beta \equiv \Pr\left\{ L\left(w^{t}\right)>c\mid q=g\right\}
  $$

These two probabilities underlie the following two concepts: 


- Probability of false alarm (= significance level = probability of
  Type I error):

  $$
  \alpha \equiv  \Pr\left\{ L\left(w^{t}\right)<c\mid q=f\right\}
  $$

- Probability of detection (= power = 1 minus probability
  of Type II error):

  $$
  1-\beta \equiv \Pr\left\{ L\left(w^{t}\right)<c\mid q=g\right\}
  $$



The [Neyman-Pearson
Lemma](https://en.wikipedia.org/wiki/Neyman–Pearson_lemma)
states that among all possible tests, a likelihood ratio test
maximizes the probability of detection for a given probability of false
alarm.

Another way to say the same thing is that among all possible tests, a likelihood ratio test
maximizes **power** for a given **significance level**.

We want a small probability of
false alarm and a large probability of detection.

With sample size $t$ fixed, we can change our two probabilities by
adjusting $c$.

A troublesome "that's life" fact is that these two probabilities move in the same direction as we vary the critical value
$c$.

Without specifying quantitative losses from making Type I and Type II errors, there is little that we can say
about how we *should* trade off probabilities of the two types of mistakes.

We do know that increasing sample size $t$ improves
statistical inference.

Below we plot some informative figures that illustrate this.

We also present a classical frequentist method for choosing a sample
size $t$.

Let's start with a case in which we fix the threshold $c$ at
$1$.

```{code-cell} ipython3
c = 1
```

Below we plot empirical distributions of logarithms of the cumulative
likelihood ratios simulated above, which are generated by either
$f$ or $g$.

Taking logarithms has no effect on calculating the probabilities because
the log is a monotonic transformation.

As $t$ increases, the probabilities of making Type I and Type II
errors both decrease, which is good.

This is because most of the probability mass of log$(L(w^t))$
moves toward $-\infty$ when $g$ is the data generating
process, while log$(L(w^t))$ goes to
$\infty$ when data are generated by $f$.

That disparate behavior of log$(L(w^t))$ under $f$ and $q$
is what makes it possible eventually to distinguish
$q=f$ from $q=g$.

```{code-cell} ipython3
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('distribution of $log(L(w^t))$ under f or under g', fontsize=15)

for i, t in enumerate([1, 7, 14, 21]):
    nr = i // 2
    nc = i % 2

    axs[nr, nc].axvline(np.log(c), color="k", ls="--")

    hist_f, x_f = np.histogram(np.log(l_seq_f[:, t]), 200, density=True)
    hist_g, x_g = np.histogram(np.log(l_seq_g[:, t]), 200, density=True)

    axs[nr, nc].plot(x_f[1:], hist_f, label="dist under f")
    axs[nr, nc].plot(x_g[1:], hist_g, label="dist under g")

    for i, (x, hist, label) in enumerate(zip([x_f, x_g], [hist_f, hist_g], ["Type I error", "Type II error"])):
        ind = x[1:] <= np.log(c) if i == 0 else x[1:] > np.log(c)
        axs[nr, nc].fill_between(x[1:][ind], hist[ind], alpha=0.5, label=label)

    axs[nr, nc].legend()
    axs[nr, nc].set_title(f"t={t}")

plt.show()
```

In the above graphs, 
  * the blue areas are related to but not equal to probabilities $\alpha $ of a type I error because 
they are integrals of $\log L_t$, not integrals of $L_t$, over rejection region $L_t < 1$  
* the orange areas are related to but not equal to probabilities $\beta $ of a type II error because 
they are integrals of $\log L_t$, not integrals of $L_t$, over acceptance region $L_t > 1$


When we hold $c$ fixed at $c=1$, the following graph shows that 
  * the probability of detection monotonically increases with increases in
$t$ 
  * the probability of a false alarm monotonically decreases with increases in $t$.

```{code-cell} ipython3
PD = np.empty(T)
PFA = np.empty(T)

for t in range(T):
    PD[t] = np.sum(l_seq_g[:, t] < c) / N
    PFA[t] = np.sum(l_seq_f[:, t] < c) / N

plt.plot(range(T), PD, label="Probability of detection")
plt.plot(range(T), PFA, label="Probability of false alarm")
plt.xlabel("t")
plt.title("$c=1$")
plt.legend()
plt.show()
```

For a given sample size $t$, the threshold $c$ uniquely pins down probabilities
of both types of error.

If for a fixed $t$ we now free up and move $c$, we will sweep out the probability
of detection as a function of the probability of false alarm.

This produces a [receiver operating characteristic
curve (ROC curve)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic).

Below, we plot receiver operating characteristic curves for different
sample sizes $t$.

```{code-cell} ipython3
PFA = np.arange(0, 100, 1)

for t in range(1, 15, 4):
    percentile = np.percentile(l_seq_f[:, t], PFA)
    PD = [np.sum(l_seq_g[:, t] < p) / N for p in percentile]

    plt.plot(PFA / 100, PD, label=f"t={t}")

plt.scatter(0, 1, label="perfect detection")
plt.plot([0, 1], [0, 1], color='k', ls='--', label="random detection")

plt.arrow(0.5, 0.5, -0.15, 0.15, head_width=0.03)
plt.text(0.35, 0.7, "better")
plt.xlabel("Probability of false alarm")
plt.ylabel("Probability of detection")
plt.legend()
plt.title("Receiver Operating Characteristic Curve")
plt.show()
```

Notice that as $t$ increases, we are assured a larger probability
of detection and a smaller probability of false alarm associated with
a given discrimination threshold $c$.

For a given sample size $t$, both $\alpha$ and $\beta$ change as we vary $c$.

As we increase $c$

* $\alpha \equiv  \Pr\left\{ L\left(w^{t}\right)<c\mid q=f\right\}$ increases  
* $\beta \equiv \Pr\left\{ L\left(w^{t}\right)>c\mid q=g\right\}$ decreases

As $t \rightarrow + \infty$, we approach the perfect detection
curve that is indicated by a right angle hinging on the blue dot.

For a given sample size $t$, the discrimination threshold $c$ determines a point on the receiver operating
characteristic curve.

It is up to the test designer to trade off probabilities of
making the two types of errors.

But we know how to choose the smallest sample size to achieve given targets for
the probabilities.

Typically, frequentists aim for a high probability of detection that
respects an upper bound on the probability of false alarm.

Below we show an example in which we fix the probability of false alarm at
$0.05$.

The required sample size for making a decision is then determined by a
target probability of detection, for example, $0.9$, as depicted in the following graph.

```{code-cell} ipython3
PFA = 0.05
PD = np.empty(T)

for t in range(T):

    c = np.percentile(l_seq_f[:, t], PFA * 100)
    PD[t] = np.sum(l_seq_g[:, t] < c) / N

plt.plot(range(T), PD)
plt.axhline(0.9, color="k", ls="--")

plt.xlabel("t")
plt.ylabel("Probability of detection")
plt.title(f"Probability of false alarm={PFA}")
plt.show()
```

The United States Navy evidently used a procedure like this to select a sample size $t$ for doing quality
control tests during World War II.

A Navy Captain who had been ordered to perform tests of this kind had doubts about it that he
presented to Milton Friedman, as we describe in {doc}`this lecture <wald_friedman>`.


(rel_entropy)=
## Kullback–Leibler Divergence

Now let's consider a case in which neither $g$ nor $f$
generates the data.

Instead, a third distribution $h$ does.

Let's study how accumulated likelihood ratios $L$ behave
when $h$ governs the data.

A key tool here is called **Kullback–Leibler divergence**.

It is also called **relative entropy**.

It measures how one probability distribution differs from another.

In our application, we want to measure how much $f$ or $g$
diverges from $h$

Two Kullback–Leibler divergences pertinent for us are $K_f$
and $K_g$ defined as

$$
\begin{aligned}
K_{f} = D_{KL}\bigl(h\|f\bigr) = KL(h, f)
          &= E_{h}\left[\log\frac{h(w)}{f(w)}\right] \\
          &= \int \log\left(\frac{h(w)}{f(w)}\right)h(w)dw .
\end{aligned}
$$

$$
\begin{aligned}
K_{g} = D_{KL}\bigl(h\|g\bigr) = KL(h,g)
          &= E_{h}\left[\log\frac{h(w)}{g(w)}\right] \\
          &= \int \log\left(\frac{h(w)}{g(w)}\right)h(w)dw .
\end{aligned}
$$

+++

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

Next we create a helper function to compute KL divergence with respect to a reference distribution $h$

```{code-cell} ipython3
def compute_KL_h(h, f, g):
    """
    Compute KL divergence with reference distribution h
    """

    Kf = compute_KL(h, f)
    Kg = compute_KL(h, g)

    return Kf, Kg
```

(KL_link)=
### A helpful formula

There is a mathematical relationship between likelihood ratios and KL divergence. 

When data is generated by distribution $h$, the expected log likelihood ratio is:

$$
\lim_{t \rightarrow + \infty} \frac{1}{t} E_{h}\!\bigl[\log L_t\bigr] = KL(h, g) - KL(h, f) = K_g - K_f
$$ (eq:kl_likelihood_link)

where $L_t=\prod_{j=1}^{t}\frac{f(w_j)}{g(w_j)}$ is the likelihood ratio process.

(For the proof, see [this note](https://nowak.ece.wisc.edu/ece830/ece830_fall11_lecture7.pdf).)

Equation {eq}`eq:kl_likelihood_link` tells us that:
- When $K_g < K_f$ (i.e., $g$ is closer to $h$ than $f$ is), the expected log likelihood ratio is negative, so $L\left(w^t\right) \rightarrow 0$.
- When $K_g > K_f$ (i.e., $f$ is closer to $h$ than $g$ is), the expected log likelihood ratio is positive, so $L\left(w^t\right) \rightarrow + \infty$.

Let's verify this using simulation.

In the simulation, we generate multiple paths using Beta distributions $f$, $g$, and $h$, and compute the paths of $\log(L(w^t))$.

First, we write a function to compute the likelihood ratio process

```{code-cell} ipython3
def compute_likelihood_ratios(sequences, f, g):
    """Compute likelihood ratios and cumulative products."""
    l_ratios = f(sequences) / g(sequences)
    L_cumulative = np.cumprod(l_ratios, axis=1)
    return l_ratios, L_cumulative
```

We consider three cases: (1) $h$ is closer to $f$, (2) $f$ and $g$ are approximately equidistant from $h$, and (3) $h$ is closer to $g$.

```{code-cell} ipython3
:tags: [hide-input]

# Define test scenarios
scenarios = [
    {
        "name": "KL(h,g) > KL(h,f)",
        "h_params": (1.2, 1.1),
        "expected": r"$L_t \to \infty$"
    },
    {
        "name": "KL(h,g) ≈ KL(h,f)",
        "h_params": (2, 1.35),
        "expected": "$L_t$ fluctuates"
    },
    {
        "name": "KL(h,g) < KL(h,f)", 
        "h_params": (3.5, 1.5),
        "expected": r"$L_t \to 0$"
    }
]

fig, axes = plt.subplots(2, 3, figsize=(15, 12))

for i, scenario in enumerate(scenarios):
    # Define h
    h = lambda x: p(x, scenario["h_params"][0], 
                    scenario["h_params"][1])
    
    # Compute KL divergences
    Kf, Kg = compute_KL_h(h, f, g)
    kl_diff = Kg - Kf
    
    # Simulate paths
    N_paths = 100
    T = 150

    # Generate data from h
    h_data = np.random.beta(scenario["h_params"][0], 
                scenario["h_params"][1], (N_paths, T))
    l_ratios, l_cumulative = compute_likelihood_ratios(h_data, f, g)
    log_l_cumulative = np.log(l_cumulative)
    
    # Plot distributions
    ax = axes[0, i]
    x_range = np.linspace(0.001, 0.999, 200)
    ax.plot(x_range, [f(x) for x in x_range], 
        'b-', label='f', linewidth=2)
    ax.plot(x_range, [g(x) for x in x_range], 
        'r-', label='g', linewidth=2)
    ax.plot(x_range, [h(x) for x in x_range], 
        'g--', label='h (data)', linewidth=2)
    ax.set_xlabel('w')
    ax.set_ylabel('density')
    ax.set_title(scenario["name"], fontsize=16)
    ax.legend()
    
    # Plot log likelihood ratio paths
    ax = axes[1, i]
    for j in range(min(20, N_paths)):
        ax.plot(log_l_cumulative[j, :], alpha=0.3, color='purple')
    
    # Plot theoretical expectation
    theory_line = kl_diff * np.arange(1, T+1)
    ax.plot(theory_line, 'k--', linewidth=2, label=r'$t \times (K_g - K_f)$')
    
    ax.set_xlabel('t')
    ax.set_ylabel('$log L_t$')
    ax.set_title(f'KL(h,f)={Kf:.3f}, KL(h,g)={Kg:.3f}\n{scenario["expected"]}', 
                 fontsize=16)
    ax.legend(fontsize=16)

plt.tight_layout()
plt.show()
```

Note that

- In the first figure, $\log L(w^t)$ diverges to $\infty$ because $K_g > K_f$.
- In the second figure, we still have $K_g > K_f$, but the difference is smaller, so $L(w^t)$ diverges to infinity at a slower pace.
- In the last figure, $\log L(w^t)$ diverges to $-\infty$ because $K_g < K_f$.
- The black dotted line, $t \left(KL(h,g) - KL(h, f)\right)$, closely fits the paths verifying {eq}`eq:kl_likelihood_link`.

These observations align with the theory.

In the [next section](hetero_agent), we will see an application of these ideas.


(hetero_agent)=
## Heterogeneous Beliefs and Financial Markets

A likelihood ratio process lies behind  Lawrence  Blume and David Easley's answer to their question
''If you're so smart, why aren't you rich?'' {cite}`blume2006if`.  

Blume and Easley constructed  formal models to study how differences of opinions about probabilities governing risky income processes would influence outcomes and be reflected in prices of stocks, bonds, and insurance policies that individuals use to share and hedge risks.

```{note}
{cite}`alchian1950uncertainty` and {cite}`friedman1953essays` can conjectured that, by rewarding traders with more realistic probability models,  competitive markets in financial securities put wealth in the hands of better informed traders and help 
make prices of risky assets  reflect realistic probability assessments. 
```


Here we'll  provide an example that illustrates  basic components of Blume and Easley's analysis. 

We'll focus only on their analysis of an environment with complete markets in which trades in all conceivable risky securities are possible. 

We'll study two alternative arrangements:

* perfect socialism in which individuals surrender their endowments of consumption goods each period to a central planner who then dictatorially allocates those goods
* a decentralized system of competitive markets in which selfish price-taking individuals voluntarily trade with each other in competitive markets 

The fundamental theorems of welfare economics will apply and assure us that these two arrangements end up producing exactly the same allocation of consumption goods to individuals **provided** that the social planner assigns  an appropriate set of **Pareto weights**.



Let the random variable $s_t \in (0,1)$ at time $t =0, 1, 2, \ldots$ be distributed according to the same  Beta distribution  with parameters 
$\theta = \{\theta_1, \theta_2\}$.

We'll denote this  probability density as

$$
\pi(s_t|\theta)
$$

Below, we'll often just write $\pi(s_t)$ instead of $\pi(s_t|\theta)$ to save space.

Let $s_t \equiv y_t^1$ be the endowment of a nonstorable consumption good  that a person we'll call "agent 1" receives at time $t$.

Let a history $s^t = [s_t, s_{t-1}, \ldots, s_0]$ be a sequence of i.i.d. random variables with joint distribution

$$
\pi_t(s^t) = \pi(s_t) \pi(s_{t-1}) \cdots \pi(s_0)
$$ 

So in our example, the history $s^t$ is a comprehensive record of agent $1$'s endowments of the consumption good from time $0$ up to time $t$.  

If agent $1$ were to live on an island by himself, agent $1$'s consumption $c^1(s_t)$ at time $t$ is 

$$c^1(s_t) = y_t^1 = s_t. $$

But in our model, agent 1 is not alone.

### Nature and agents' beliefs

Nature draws i.i.d. sequences $\{s_t\}_{t=0}^\infty$ from $\pi_t(s^t)$.

* so $\pi$ without a superscript is nature's model 
* but in addition to nature, there are other entities inside our model -- artificial people that we call "agents"
* each agent has a sequence of probability distributions over $s^t$ for $t=0, \ldots$ 
* agent $i$ thinks that nature draws i.i.d. sequences $\{s_t\}_{t=0}^\infty$ from $\pi_t^i(s^t)$
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

### A social risk-sharing arrangement

In order to share risks, a benevolent social planner will dictate a history-dependent consumption allocation in the form of a sequence of functions 

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
u(c_t^i) = u(c_t^i) = \ln (c_t^i)
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


### The social planner's allocation problem

The benevolent dictator has all the information it requires to choose a consumption allocation that maximizes the social welfare criterion 

$$
W = \lambda V^1 + (1-\lambda) V^2
$$ (eq:welfareW)

where $\lambda \in [0,1]$ is a Pareto weight tells how much the planner likes agent $1$ and $1 - \lambda$ is a Pareto weight that tells how much the social planner likes agent $2$.  

Setting $\lambda = .5$ expresses ''egalitarian'' social preferences. 

Notice how social welfare criterion {eq}`eq:welfareW` takes into account both agents' preferences as represented by formula {eq}`eq:objectiveagenti`.

This means that the social planner knows and respects

* each agent's  one period utility function $u(\cdot) = \ln(\cdot)$
* each agent $i$'s probability model $\{\pi_t^i(s^t)\}_{t=0}^\infty$

Consequently, we anticipate that these objects will appear in the social planner's rule for allocating the aggregate endowment each period. 


First-order necessary conditions for maximizing welfare criterion {eq}`eq:welfareW` subject to the feasibility constraint {eq}`eq:feasibility` are 

$$\frac{\pi_t^2(s^t)}{\pi_t^1(s^t)} \frac{(1/c_t^2(s^t))}{(1/c_t^1(s^t))} = \frac{\lambda}{1 -\lambda}$$

which can be rearranged to become




$$
\frac{c_t^1(s^t)}{c_t^2(s^t)} = \frac{\lambda}{1- \lambda} l_t(s^t)
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




### If you're so smart, $\ldots$ 


Let's compute some values   of limiting allocations {eq}`eq:allocationrule1` for some interesting possible limiting
values of the likelihood ratio process $l_t(s^t)$:

 $$l_\infty (s^\infty)= 1; \quad c_\infty^1 = \lambda$$
 
  * In the above case, both agents are equally smart (or equally not smart) and the consumption allocation stays put at a $\lambda, 1 - \lambda $ split between the two agents. 

$$l_\infty (s^\infty) = 0; \quad c_\infty^1 = 0$$

* In the above case, agent 2 is smarter than agent 1, and agent 1's share of the aggregate endowment converges to zero.  



$$l_\infty (s^\infty)= \infty; \quad c_\infty^1 = 1$$

* In the above case, agent 1 is smarter than agent 2, and agent 1's share of the aggregate endowment converges to 1. 


Soon we'll do some simulations that will shed further light on possible outcomes.

But before we do that, let's take a detour and study some  "shadow prices" for the social planning problem that can readily be
converted to "equilibrium prices" for a competitive equilibrium. 

Doing this will allow us to connect our analysis with an argument  of {cite}`alchian1950uncertainty` and {cite}`friedman1953essays` that competitive market processes can make prices of risky assets better reflect realistic probability assessments. 



### Competitive Equilibrium Prices 

The two fundamental welfare theorems for general equilibrium models lead us to anticipate that there is  a connection between the allocation that solves the social planning problem we have been studying and the allocation in a  **competitive equilibrium**  with complete markets in history-contingent commodities.

```{note}
For the two welfare theorems and their history, see   <https://en.wikipedia.org/wiki/Fundamental_theorems_of_welfare_economics>.
```

Such a connection prevails for our model.  

We'll sketch it now.

In a competitive equilibrium, there is no social planner that dictatorially collects everybody's endowments and then reallocates them.

Instead, there is a comprehensive centralized   market that meets at one point in time.

There are **prices** at which price-taking agents can buy or sell whatever goods that they want.  

Trade is multilateral in the sense that all that there is a "Walrasian auctioneer" who lives outside the model and whose job is to verify that
each agent's budget constraint is satisfied.  

That budget constraint involves the total value of the agent's endowment stream and the total value of its consumption stream.  

Suppose that at time $-1$, before time $0$ starts, agent  $i$ can purchase one unit $c_t(s^t)$ of  consumption at time $t$ after history
$s^t$ at price $p_t(s^t)$.  

Notice that there is (very long) **vector** of prices.  

We want to study how agents' diverse beliefs influence equilibrium prices.  

Agent $i$ faces a **single** intertemporal budget constraint

$$
\sum_{t=0}\sum_{s^t} p_t(s^t) c_t^i (y_t(s^t)) \leq \sum_{t=0}\sum_{s^t} p_t(s^t) y_t^i (y_t(s^t))
$$ (eq:budgetI)

Agent $i$ puts a Lagrange multiplier $\mu^i$ on {eq}`eq:budgetI` and once-and-for-all chooses a consumption plan $\{c^i_t(s^t)\}_{t=0}^\infty$
to maximize criterion {eq}`eq:objectiveagenti` subject to budget constraint {eq}`eq:budgetI`.

```{note}
For convenience, let's remind ourselves of criterion {eq}`eq:objectiveagenti`:  
$
V^i = \sum_{t=0}^{\infty} \sum_{s^t} \delta^t u_t(c_t^i(s^t)) \pi_t^i(s^t)$
```

First-order conditions for maximizing  with respect to $c_t^i(s^t)$ are 

$$
\delta^t u'(c^i(s^t)) \pi_t^i(s^t) = \mu_i p_t(s^t) ,
$$ 

which we can rearrange to obtain

$$
p_t(s^t) = \frac{ \delta^t \pi_t^i(s^t)}{\mu^i c^i(s^t)}   
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

Notice that if we set $\mu_1 = \lambda$ and $\mu_2 = 1 -\lambda$, then  formula {eq}`eq:allocationce` agrees with formula
{eq}`eq:allocationrule1`.  

  * doing this amounts to choosing a **numeraire** or normalization for the price system $\{p_t(s^t)\}_{t=0}^\infty$

```{note}
For information about how a numeraire  must be chosen to pin down the absolute price level in a model like ours that determines only
relative prices,   see <https://en.wikipedia.org/wiki/Num%C3%A9raire>.
```

If we substitute formula  {eq}`eq:allocationce` for $c_t^1(s^t)$ into formula {eq}`eq:priceequation1` and rearrange, we obtain

$$
p_t(s^t) = \frac{\delta^t \pi_t^2(s^t)}{1 - \lambda + \lambda l_t(s^t)}
$$ (eq:pformulafinal)

According to formula {eq}`eq:pformulafinal`, we have the following possible limiting cases:

* when $l_\infty = 0$, $c_\infty^2 = 0 $ and tails of competitive equilibrium prices reflect agent $2$'s probability model $\pi_t^2(s^t)$ 
* when $l_\infty = 1$, $c_\infty^1 = 0 $ and tails competitive equilibrium prices reflect agent $1$'s probability model $\pi_t^2(s^t)$ 
* for small $t$'s, competitive equilbrium prices reflect both agents' probability models.  

### Simulations 

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

Let's  write a Python function that computes agent 1's  consumption share

```{code-cell} ipython3
def simulate_blume_easley(sequences, f_belief=f, g_belief=g, λ=0.5):
    """Simulate Blume-Easley model consumption shares."""
    l_ratios, l_cumulative = compute_likelihood_ratios(sequences, f_belief, g_belief)
    c1_share = λ * l_cumulative / (1 - λ + λ * l_cumulative)
    return l_cumulative, c1_share
```

Now let's use this  function to generate sequences in which  

*  nature draws from  $f$ each period, or 
*  nature draws from  $g$ each period, or
*  or nature flips a fair coin each period  to decide whether  to draw from  $f$ or $g$

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

We invite readers to revisit [that section](rel_entropy) and try to infer the relationships among $KL(f, g)$, $KL(g, f)$, $KL(h, f)$, and $KL(h,g)$.


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

We find that in the first case, $KL(f,g) \approx KL(g,f)$ and both are relatively small, so although either agent 1 or agent  2 will eventually consume everything, convergence displaying in  first two panels on the top is pretty  slowly.

In the first two panels at the bottom, we see convergence occurring faster (as indicated by the black dashed line) because the divergence gaps $KL(f, g)$ and $KL(g, f)$ are larger.

Since $KL(f,g) > KL(g,f)$, we  see faster convergence in  the first panel at the bottom when  nature chooses $f$  than in the second panel where nature chooses $g$.

This ties in nicely with {eq}`eq:kl_likelihood_link`.

## Hypothesis Testing and Classification 

This section discusses another application of likelihood ratio processes.

We describe how a statistician can combine frequentist probabilities of type I and type II errors in order to 

* compute an anticipated frequency of  selecting a wrong model based on a sample length $T$
* compute an anticipated error  rate in a classification problem 

We consider a situation in which  nature generates data by mixing known densities $f$ and $g$ with known mixing
parameter $\pi_{-1} \in (0,1)$ so that the random variable $w$ is drawn from the density

$$
h (w) = \pi_{-1} f(w) + (1-\pi_{-1}) g(w) 
$$

We assume that the statistician knows the densities $f$ and $g$ and also the mixing parameter $\pi_{-1}$.

Below, we'll  set $\pi_{-1} = .5$, although much of the analysis would follow through with other settings of $\pi_{-1} \in (0,1)$.  

We assume that $f$ and $g$ both put positive probabilities on the same intervals of possible realizations of the random variable $W$.

  

In the simulations below, we specify that  $f$ is a $\text{Beta}(1, 1)$ distribution and that  $g$ is $\text{Beta}(3, 1.2)$ distribution.

We consider two alternative timing protocols. 

 * Timing protocol 1 is for   the model selection problem
 * Timing protocol 2 is for the individual classification problem 

**Timing Protocol 1:**  Nature flips a coin only **once** at time $t=-1$ and with probability $\pi_{-1}$  generates a sequence  $\{w_t\}_{t=1}^T$
of  IID  draws from  $f$  and with probability $1-\pi_{-1}$ generates a sequence  $\{w_t\}_{t=1}^T$
of  IID  draws from  $g$.

Let's write some Python code that implements timing protocol 1.

```{code-cell} ipython3
def protocol_1(π_minus_1, T, N=1000):
    """
    Simulate Protocol 1: 
    Nature decides once at t=-1 which model to use.
    """
    
    # On-off coin flip for the true model
    true_models_F = np.random.rand(N) < π_minus_1
    
    sequences = np.empty((N, T))
    
    n_f = np.sum(true_models_F)
    n_g = N - n_f
    if n_f > 0:
        sequences[true_models_F, :] = np.random.beta(F_a, F_b, (n_f, T))
    if n_g > 0:
        sequences[~true_models_F, :] = np.random.beta(G_a, G_b, (n_g, T))
    
    return sequences, true_models_F
```

**Timing Protocol 2.** Nature flips a coin **often**.  At each time $t \geq 0$, nature flips a coin and with probability $\pi_{-1}$ draws $w_t$ from $f$ and with probability $1-\pi_{-1}$ draws $w_t$ from $g$.

Here is  Python code that we'll use to implement timing protocol 2.

```{code-cell} ipython3
def protocol_2(π_minus_1, T, N=1000):
    """
    Simulate Protocol 2: 
    Nature decides at each time step which model to use.
    """
    
    # Coin flips for each time t upto T
    true_models_F = np.random.rand(N, T) < π_minus_1
    
    sequences = np.empty((N, T))
    
    n_f = np.sum(true_models_F)
    n_g = N * T - n_f
    if n_f > 0:
        sequences[true_models_F] = np.random.beta(F_a, F_b, n_f)
    if n_g > 0:
        sequences[~true_models_F] = np.random.beta(G_a, G_b, n_g)
    
    return sequences, true_models_F
```

**Remark:** Under timing protocol 2, the $\{w_t\}_{t=1}^T$ is a sequence of IID draws from $h(w)$. Under timing protocol 1, the the $\{w_t\}_{t=1}^T$ is 
not IID.  It is **conditionally IID** -- meaning that with probability $\pi_{-1}$ it is a sequence of IID draws from $f(w)$ and with probability $1-\pi_{-1}$ it is a sequence of IID draws from $g(w)$. For more about this, see {doc}`this lecture about exchangeability <exchangeable>`.

We  again deploy a **likelihood ratio process** with time $t$ component being the likelihood ratio  

$$
\ell (w_t)=\frac{f\left(w_t\right)}{g\left(w_t\right)},\quad t\geq1.
$$

The **likelihood ratio process** for sequence $\left\{ w_{t}\right\} _{t=1}^{\infty}$ is 

$$
L\left(w^{t}\right)=\prod_{i=1}^{t} \ell (w_i),
$$

For shorthand we'll write $L_t =  L(w^t)$.

### Model Selection Mistake Probability 

We first study  a problem that assumes  timing protocol 1.  

Consider a decision maker who wants to know whether model $f$ or model $g$ governs a data set of length $T$ observations.

The decision makers has observed a sequence $\{w_t\}_{t=1}^T$.

On the basis of that observed  sequence, a likelihood ratio test selects model $f$ when
 $L_T \geq 1 $ and model $g$ when  $L_T < 1$.  
 
When model $f$ generates the data, the probability that the likelihood ratio test selects the wrong model is 

$$ 
p_f = {\rm Prob}\left(L_T < 1\Big| f\right) = \alpha_T .
$$

When model $g$ generates the data, the probability that the likelihood ratio test selects the wrong model is 

$$ 
p_g = {\rm Prob}\left(L_T \geq 1 \Big|g \right) = \beta_T. 
$$

We can construct a probability that the likelihood ratio selects the wrong model by assigning a Bayesian prior probability of $\pi_{-1} = .5$ that nature selects model $f$ then  averaging $p_f$ and $p_g$ to form the Bayesian posterior probability of a detection error equal to

$$ 
p(\textrm{wrong decision}) = {1 \over 2} (\alpha_T + \beta_T) .
$$ (eq:detectionerrorprob)

Now let's simulate  timing protocol 1 and compute the error probabilities

```{code-cell} ipython3
# Set parameters
π_minus_1 = 0.5
T_max = 30
N_simulations = 10_000

sequences_p1, true_models_p1 = protocol_1(
                            π_minus_1, T_max, N_simulations)
l_ratios_p1, L_cumulative_p1 = compute_likelihood_ratios(sequences_p1, f, g)

# Compute error probabilities for different sample sizes
T_range = np.arange(1, T_max + 1)

# Boolean masks for true models
mask_f = true_models_p1
mask_g = ~true_models_p1

# Select cumulative likelihoods for each model
L_f = L_cumulative_p1[mask_f, :]
L_g = L_cumulative_p1[mask_g, :]

α_T = np.mean(L_f < 1, axis=0)
β_T = np.mean(L_g >= 1, axis=0)

error_prob = 0.5 * (α_T + β_T)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(T_range, α_T, 'b-', 
         label=r'$\alpha_T$', linewidth=2)
ax1.plot(T_range, β_T, 'r-', 
         label=r'$\beta_T$', linewidth=2)
ax1.set_xlabel('$T$')
ax1.set_ylabel('error probability')
ax1.legend()

ax2.plot(T_range, error_prob, 'g-', 
         label=r'$\frac{1}{2}(\alpha_T+\beta_T)$', linewidth=2)
ax2.set_xlabel('$T$')
ax2.set_ylabel('error probability')
ax2.legend()

plt.tight_layout()
plt.show()

print(f"At T={T_max}:")
print(f"α_{T_max} = {α_T[-1]:.4f}")
print(f"β_{T_max} = {β_T[-1]:.4f}")
print(f"Model selection error probability = {error_prob[-1]:.4f}")
```

Notice how the model selection  error probability approaches zero as $T$ grows.  

### Classification

We now consider a problem that assumes timing protocol 2.

A decision maker wants to classify components of an observed sequence $\{w_t\}_{t=1}^T$ as having been drawn from either $f$ or $g$.

The decision maker uses the following classification rule:

$$
\begin{aligned}
w_t  & \ {\rm is \ from \  f  \ if \ } l_t > 1 \\
w_t  & \ {\rm is \ from \  g  \ if \ } l_t \leq 1 . 
\end{aligned}
$$

Under this rule, the expected misclassification rate is

$$
p(\textrm{misclassification}) = {1 \over 2} (\tilde \alpha_t + \tilde \beta_t) 
$$ (eq:classerrorprob)

where $\tilde \alpha_t = {\rm Prob}(l_t < 1 \mid f)$ and $\tilde \beta_t = {\rm Prob}(l_t \geq 1 \mid g)$.

Since for each $t$, the decision boundary is the same, the decision boundary can be computed as

```{code-cell} ipython3
root = brentq(lambda w: f(w) / g(w) - 1, 0.001, 0.999)
```

we can plot the distributions of $f$ and $g$ and the decision boundary

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(7, 6))

w_range = np.linspace(1e-5, 1-1e-5, 1000)
f_values = [f(w) for w in w_range]
g_values = [g(w) for w in w_range]
ratio_values = [f(w)/g(w) for w in w_range]

ax.plot(w_range, f_values, 'b-', 
        label=r'$f(w) \sim Beta(1,1)$', linewidth=2)
ax.plot(w_range, g_values, 'r-', 
        label=r'$g(w) \sim Beta(3,1.2)$', linewidth=2)

type1_prob = 1 - beta_dist.cdf(root, F_a, F_b)
type2_prob = beta_dist.cdf(root, G_a, G_b)

w_type1 = w_range[w_range >= root]
f_type1 = [f(w) for w in w_type1]
ax.fill_between(w_type1, 0, f_type1, alpha=0.3, color='blue', 
                label=fr'$\tilde \alpha_t = {type1_prob:.2f}$')

w_type2 = w_range[w_range <= root]
g_type2 = [g(w) for w in w_type2]
ax.fill_between(w_type2, 0, g_type2, alpha=0.3, color='red', 
                label=fr'$\tilde \beta_t = {type2_prob:.2f}$')

ax.axvline(root, color='green', linestyle='--', alpha=0.7, 
            label=f'decision boundary: $w=${root:.3f}')

ax.set_xlabel('w')
ax.set_ylabel('probability density')
ax.legend()

plt.tight_layout()
plt.show()
```

To  the left of the  green vertical line  $g < f$,  so $l_t < 1$; therefore a  $w_t$ that falls to the left of the green line is classified as a type $g$ individual. 

 * The shaded orange area equals $\beta$ -- the probability of classifying someone as a type $g$ individual when it is really a type $f$ individual.

To  the right of the  green vertical line $g > f$, so $l_t >1 $; therefore  a  $w_t$ that falls to the right  of the green line is classified as a type $f$ individual. 

 * The shaded blue area equals $\alpha$ -- the probability of classifying someone as a type $f$ when it is really a type $g$ individual.  

This gives us clues about how to compute the theoretical classification error probability

```{code-cell} ipython3
# Compute theoretical tilde α_t and tilde β_t
def α_integrand(w):
    """Integrand for tilde α_t = P(l_t < 1 | f)"""
    return f(w) if f(w) / g(w) < 1 else 0

def β_integrand(w):
    """Integrand for tilde β_t = P(l_t >= 1 | g)"""
    return g(w) if f(w) / g(w) >= 1 else 0

# Compute the integrals
α_theory, _ = quad(α_integrand, 0, 1, limit=100)
β_theory, _ = quad(β_integrand, 0, 1, limit=100)

theory_error = 0.5 * (α_theory + β_theory)

print(f"theoretical tilde α_t = {α_theory:.4f}")
print(f"theoretical tilde β_t = {β_theory:.4f}")
print(f"theoretical classification error probability = {theory_error:.4f}")
```

Now we simulate timing protocol 2 and compute the classification error probability.

In the next cell, we also compare the theoretical classification accuracy to the empirical classification accuracy

```{code-cell} ipython3
accuracy = np.empty(T_max)

sequences_p2, true_sources_p2 = protocol_2(
                    π_minus_1, T_max, N_simulations)
l_ratios_p2, _ = compute_likelihood_ratios(sequences_p2, f, g)

for t in range(T_max):
    predictions = (l_ratios_p2[:, t] >= 1)
    actual = true_sources_p2[:, t]
    accuracy[t] = np.mean(predictions == actual)

plt.figure(figsize=(10, 6))
plt.plot(range(1, T_max + 1), accuracy, 
                'b-', linewidth=2, label='empirical accuracy')
plt.axhline(1 - theory_error, color='r', linestyle='--', 
                label=f'theoretical accuracy = {1 - theory_error:.4f}')
plt.xlabel('$t$')
plt.ylabel('accuracy')
plt.legend()
plt.ylim(0.5, 1.0)
plt.show()
```

Let's watch decisions made by  the two timing protocols as more and more observations accrue.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(T_range, error_prob, linewidth=2, 
        label='Protocol 1')
ax.plot(T_range, 1-accuracy, linestyle='--', linewidth=2, 
        label=f'Protocol 2')
ax.set_ylabel('error probability')
ax.legend()
plt.show()
```

From the figure above, we can see:

- For both timing protocols, the error probability starts at the same level, subject to a little randomness.

- For timing protocol 1, the error probability decreases as the sample size increases because we are  making just **one** decision -- i.e., selecting whether $f$ or $g$ governs  **all** individuals.  More data provides better evidence.

- For timing protocol 2, the error probability remains constant because we are making **many** decisions -- one classification decision for each observation.  

**Remark:** Think about how laws of large numbers are applied to compute error probabilities for the model selection problem and the classification problem. 

## Measuring discrepancies between distributions

A plausible guess is that  the ability of a likelihood ratio to distinguish  distributions $f$ and $g$ depends on how "different" they are.
 
But how should we measure  discrepancies between distributions?

We've already encountered one discrepancy measure -- the Kullback-Leibler (KL) divergence. 

We now briefly explore two alternative discrepancy  measures.

### Chernoff entropy

Chernoff entropy was motivated by an early application of  the [theory of large deviations](https://en.wikipedia.org/wiki/Large_deviations_theory).

```{note}
Large deviation theory provides refinements of the central limit theorem. 
```

The Chernoff entropy between probability densities $f$ and $g$ is defined as:

$$
C(f,g) = - \log \min_{\phi \in (0,1)} \int f^\phi(x) g^{1-\phi}(x) dx
$$

An upper bound on model selection error probabilty is

$$
e^{-C(f,g)T} .
$$

Thus,    Chernoff entropy is  an upper bound on  the exponential  rate at which  the selection error probability falls as sample size $T$ grows. 

Let's compute Chernoff entropy numerically with some Python code

```{code-cell} ipython3
def chernoff_integrand(ϕ, f, g):
    """
    Compute the integrand for Chernoff entropy
    """
    def integrand(w):
        return f(w)**ϕ * g(w)**(1-ϕ)

    result, _ = quad(integrand, 1e-5, 1-1e-5)
    return result

def compute_chernoff_entropy(f, g):
    """
    Compute Chernoff entropy C(f,g)
    """
    def objective(ϕ):
        return chernoff_integrand(ϕ, f, g)
    
    # Find the minimum over ϕ in (0,1)
    result = minimize_scalar(objective, 
                             # For numerical stability
                             bounds=(1e-5, 1-1e-5), 
                             method='bounded')
    min_value = result.fun
    ϕ_optimal = result.x
    
    chernoff_entropy = -np.log(min_value)
    return chernoff_entropy, ϕ_optimal

C_fg, ϕ_optimal = compute_chernoff_entropy(f, g)
print(f"Chernoff entropy C(f,g) = {C_fg:.4f}")
print(f"Optimal ϕ = {ϕ_optimal:.4f}")
```

Now let's examine how $e^{-C(f,g)T}$ behaves as a function of $T$ and compare it to the model selection error probability

```{code-cell} ipython3
T_range = np.arange(1, T_max+1)
chernoff_bound = np.exp(-C_fg * T_range)

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(T_range, chernoff_bound, 'r-', linewidth=2, 
           label=f'$e^{{-C(f,g)T}}$')
ax.semilogy(T_range, error_prob, 'b-', linewidth=2, 
           label='Model selection error probability')

ax.set_xlabel('T')
ax.set_ylabel('error probability (log scale)')
ax.legend()
plt.tight_layout()
plt.show()
```

Evidently, $e^{-C(f,g)T}$ is an upper bound on the error rate.

### Jensen-Shannon divergence

The [Jensen-Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) is another  divergence measure.  

For probability densities $f$ and $g$, the **Jensen-Shannon divergence** is defined as:

$$
D(f,g) = \frac{1}{2} KL(f, m) + \frac{1}{2} KL(g, m)
$$ (eq:compute_JS)

where $m = \frac{1}{2}(f+g)$ is a mixture of $f$ and $g$.

Below we compute Jensen-Shannon divergence numerically with some Python code

```{code-cell} ipython3
def compute_JS(f, g):
    """
    Compute Jensen-Shannon divergence
    """
    def m(w):
        return 0.5 * (f(w) + g(w))
    
    js_div = 0.5 * compute_KL(f, m) + 0.5 * compute_KL(g, m)
    return js_div
```

 
```{note}
We studied KL divergence in the [section above](rel_entropy) with respect to a reference distribution $h$.

Recall that  KL divergence $KL(f, g)$ measures expected excess surprisal from using misspecified model $g$ instead $f$ when $f$ is the true model.

Because in general $KL(f, g) \neq KL(g, f)$, KL divergence is not symmetric, but Jensen-Shannon divergence is symmetric.

(In fact, the square root of the Jensen-Shannon divergence is a metric referred to as the Jensen-Shannon distance.)

As {eq}`eq:compute_JS` shows, the Jensen-Shannon divergence computes average of the KL divergence of $f$ and $g$ with respect to a particular reference distribution $m$ defined below the equation.
```

Now let's create a comparison table showing KL divergence, Jensen-Shannon divergence, and Chernoff entropy for a set of pairs of Beta distributions.

```{code-cell} ipython3
:tags: [hide-input]

distribution_pairs = [
    # (f_params, g_params)
    ((1, 1), (0.1, 0.2)),
    ((1, 1), (0.3, 0.3)),
    ((1, 1), (0.3, 0.4)),
    ((1, 1), (0.5, 0.5)),
    ((1, 1), (0.7, 0.6)),
    ((1, 1), (0.9, 0.8)),
    ((1, 1), (1.1, 1.05)),
    ((1, 1), (1.2, 1.1)),
    ((1, 1), (1.5, 1.2)),
    ((1, 1), (2, 1.5)),
    ((1, 1), (2.5, 1.8)),
    ((1, 1), (3, 1.2)),
    ((1, 1), (4, 1)),
    ((1, 1), (5, 1))
]

# Create comparison table
results = []
for i, ((f_a, f_b), (g_a, g_b)) in enumerate(distribution_pairs):
    # Define the density functions
    f = jit(lambda x, a=f_a, b=f_b: p(x, a, b))
    g = jit(lambda x, a=g_a, b=g_b: p(x, a, b))
    
    # Compute measures
    kl_fg = compute_KL(f, g)
    kl_gf = compute_KL(g, f)
    js_div = compute_JS(f, g)
    chernoff_ent, _ = compute_chernoff_entropy(f, g)
    
    results.append({
        'Pair (f, g)': f"\\text{{Beta}}({f_a},{f_b}), \\text{{Beta}}({g_a},{g_b})",
        'KL(f, g)': f"{kl_fg:.4f}",
        'KL(g, f)': f"{kl_gf:.4f}",
        'JS': f"{js_div:.4f}",
        'C': f"{chernoff_ent:.4f}"
    })

df = pd.DataFrame(results)

# Sort by JS divergence
df['JS_numeric'] = df['JS'].astype(float)
df = df.sort_values('JS_numeric').drop('JS_numeric', axis=1)

# Generate LaTeX table manually
columns = ' & '.join([f'\\text{{{col}}}' for col in df.columns])
rows = ' \\\\\n'.join(
    [' & '.join([f'{val}' for val in row]) 
     for row in df.values])

latex_code = rf"""
\begin{{array}}{{lcccc}}
{columns} \\
\hline
{rows}
\end{{array}}
"""

display(Math(latex_code))
```

The above  table indicates how  Jensen-Shannon divergence,  and Chernoff entropy, and  KL divergence covary as we alter $f$ and $g$.

Let's also visualize how these diverge measures covary

```{code-cell} ipython3
kl_fg_values = [float(result['KL(f, g)']) for result in results]
js_values = [float(result['JS']) for result in results]
chernoff_values = [float(result['C']) for result in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# JS divergence and KL divergence
axes[0].scatter(kl_fg_values, js_values, alpha=0.7, s=60)
axes[0].set_xlabel('KL divergence KL(f, g)')
axes[0].set_ylabel('JS divergence')
axes[0].set_title('JS divergence and KL divergence')

# Chernoff Entropy and JS divergence
axes[1].scatter(js_values, chernoff_values, alpha=0.7, s=60)
axes[1].set_xlabel('JS divergence')
axes[1].set_ylabel('Chernoff entropy')
axes[1].set_title('Chernoff entropy and JS divergence')

plt.tight_layout()
plt.show()
```

To make the comparison more concrete, let's plot the distributions and the divergence measures for a few pairs of distributions.

Note that the numbers on the title changes with the area of the overlaps of two distributions

```{code-cell} ipython3
:tags: [hide-input]

def plot_dist_diff():
    """
    Plot overlap of two distributions and divergence measures
    """
    
    # Chose a subset of Beta distribution parameters
    param_grid = [
        ((1, 1), (1, 1)),   
        ((1, 1), (1.5, 1.2)),
        ((1, 1), (2, 1.5)),  
        ((1, 1), (3, 1.2)),  
        ((1, 1), (5, 1)),
        ((1, 1), (0.3, 0.3))
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    divergence_data = []
    
    for i, ((f_a, f_b), (g_a, g_b)) in enumerate(param_grid):
        row = i // 2
        col = i % 2
        
        # Create density functions
        f = jit(lambda x, a=f_a, b=f_b: p(x, a, b))
        g = jit(lambda x, a=g_a, b=g_b: p(x, a, b))
        
        # Compute divergence measures
        kl_fg = compute_KL(f, g)
        js_div = compute_JS(f, g) 
        chernoff_ent, _ = compute_chernoff_entropy(f, g)
        
        divergence_data.append({
            'f_params': (f_a, f_b),
            'g_params': (g_a, g_b),
            'kl_fg': kl_fg,
            'js_div': js_div,
            'chernoff': chernoff_ent
        })
        
        # Plot distributions
        x_range = np.linspace(0, 1, 200)
        f_vals = [f(x) for x in x_range]
        g_vals = [g(x) for x in x_range]
        
        axes[row, col].plot(x_range, f_vals, 'b-', linewidth=2, 
                           label=f'f ~ Beta({f_a},{f_b})')
        axes[row, col].plot(x_range, g_vals, 'r-', linewidth=2, 
                           label=f'g ~ Beta({g_a},{g_b})')
        
        # Fill overlap region
        overlap = np.minimum(f_vals, g_vals)
        axes[row, col].fill_between(x_range, 0, overlap, alpha=0.3, 
                                   color='purple', label='overlap')
        
        # Add divergence information
        axes[row, col].set_title(
            f'KL(f, g)={kl_fg:.3f}, JS={js_div:.3f}, C={chernoff_ent:.3f}',
            fontsize=12)
        axes[row, col].legend(fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    return divergence_data

divergence_data = plot_dist_diff()
```

### Error probability and divergence measures

Now let's return to our guess that the error probability at large sample sizes is related to the Chernoff entropy  between two distributions.

We verify this by computing the correlation between the log of the error probability at $T=50$ under Timing Protocol 1 and the divergence measures.

In the simulation below, nature draws $N / 2$ sequences from $g$ and $N/2$ sequences from $f$.

```{note}
Nature does this rather than flipping a fair coin to decide whether to draw from $g$ or $f$ once and for all before each simulation of length $T$.
```

```{code-cell} ipython3
# Parameters for simulation
T_large = 50
N_sims = 5000
N_half = N_sims // 2

# Initialize arrays
n_pairs = len(distribution_pairs)
kl_fg_vals = np.zeros(n_pairs)
kl_gf_vals = np.zeros(n_pairs) 
js_vals = np.zeros(n_pairs)
chernoff_vals = np.zeros(n_pairs)
error_probs = np.zeros(n_pairs)
pair_names = []

for i, ((f_a, f_b), (g_a, g_b)) in enumerate(distribution_pairs):
    # Create density functions
    f = jit(lambda x, a=f_a, b=f_b: p(x, a, b))
    g = jit(lambda x, a=g_a, b=g_b: p(x, a, b))

    # Compute divergence measures
    kl_fg_vals[i] = compute_KL(f, g)
    kl_gf_vals[i] = compute_KL(g, f)
    js_vals[i] = compute_JS(f, g)
    chernoff_vals[i], _ = compute_chernoff_entropy(f, g)

    # Generate samples
    sequences_f = np.random.beta(f_a, f_b, (N_half, T_large))
    sequences_g = np.random.beta(g_a, g_b, (N_half, T_large))

    # Compute likelihood ratios and cumulative products
    _, L_cumulative_f = compute_likelihood_ratios(sequences_f, f, g)
    _, L_cumulative_g = compute_likelihood_ratios(sequences_g, f, g)
    
    # Get final values
    L_cumulative_f = L_cumulative_f[:, -1]
    L_cumulative_g = L_cumulative_g[:, -1]

    # Calculate error probabilities
    error_probs[i] = 0.5 * (np.mean(L_cumulative_f < 1) + 
                            np.mean(L_cumulative_g >= 1))
    pair_names.append(f"Beta({f_a},{f_b}) and Beta({g_a},{g_b})")

cor_data =  {
    'kl_fg': kl_fg_vals,
    'kl_gf': kl_gf_vals,
    'js': js_vals, 
    'chernoff': chernoff_vals,
    'error_prob': error_probs,
    'names': pair_names,
    'T': T_large}
```

Now let's visualize the correlations

```{code-cell} ipython3
:tags: [hide-input]

def plot_error_divergence(data):
    """
    Plot correlations between error probability and divergence measures.
    """
    # Filter out near-zero error probabilities for log scale
    nonzero_mask = data['error_prob'] > 1e-6
    log_error = np.log(data['error_prob'][nonzero_mask])
    js_vals = data['js'][nonzero_mask]
    chernoff_vals = data['chernoff'][nonzero_mask]

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # function for plotting correlation
    def plot_correlation(ax, x_vals, x_label, color):
        ax.scatter(x_vals, log_error, alpha=0.7, s=60, color=color)
        ax.set_xlabel(x_label)
        ax.set_ylabel(f'log(Error probability) at T={data["T"]}')
        
        # Calculate correlation and trend line
        corr = np.corrcoef(x_vals, log_error)[0, 1]
        z = np.polyfit(x_vals, log_error, 2)
        x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_trend, np.poly1d(z)(x_trend), 
                "r--", alpha=0.8, linewidth=2)
        ax.set_title(f'Log error probability and {x_label}\n'
                     f'Correlation = {corr:.3f}')
    
    # Plot both correlations
    plot_correlation(ax1, js_vals, 'JS divergence', 'C0')
    plot_correlation(ax2, chernoff_vals, 'Chernoff entropy', 'C1')

    plt.tight_layout()
    plt.show()

plot_error_divergence(cor_data)
```

Evidently, Chernoff entropy and Jensen-Shannon entropy each covary tightly with the model selection error probability.

We'll encounter related ideas in {doc}`wald_friedman` very soon.


## Related Lectures

Likelihood processes play an important role in Bayesian learning, as described in {doc}`likelihood_bayes`
and as applied in {doc}`odu`.

Likelihood ratio processes appear again in {doc}`advanced:additive_functionals`, which contains another illustration
of the **peculiar property** of likelihood ratio processes described above.
