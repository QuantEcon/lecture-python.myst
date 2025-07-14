---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
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

We'll use a setting described in {doc}`this lecture <exchangeable>`.

Among  things that we'll learn  are

* A peculiar property of likelihood ratio processes
* How a likelihood ratio process is a key ingredient in frequentist hypothesis testing
* How a **receiver operator characteristic curve** summarizes information about a false alarm probability and power in frequentist hypothesis testing
* How a Bayesian statistician combines frequentist probabilities of type I and type II errors to form posterior probabilities of erroneous model selection or missclassification of individuals 
* How during World War II the United States Navy devised a decision rule that Captain Garret L. Schyler challenged, a topic to be studied in  {doc}`this lecture <wald_friedman>`



Let's start  by importing some Python tools.

```{code-cell} ipython3
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
import numpy as np
from numba import vectorize, jit
from math import gamma
from scipy.integrate import quad
from scipy.optimize import brentq
```

## Likelihood Ratio Process

A nonnegative random variable $W$ has one of two probability density functions, either
$f$ or $g$.

Before the beginning of time, nature once and for all decides whether she will draw a sequence of IID draws from either
$f$ or $g$.

We will sometimes let $q$ be the density that nature chose once and for all, so
that $q$ is either $f$ or $g$, permanently.

Nature knows which density it permanently draws from, but we the observers do not.

We know both $f$ and $g$ but we don’t know which density nature
chose.

But we want to know.

To do that, we use observations.

We observe a sequence $\{w_t\}_{t=1}^T$ of $T$ IID draws that we know came from either $f$ or $g$.

We want to use these observations to infer whether nature chose $f$ or $g$.

A **likelihood ratio process** is a useful tool for this task.

To begin, we define key component of a likelihood ratio process, namely, the time $t$ likelihood ratio  as the random variable

$$
\ell (w_t)=\frac{f\left(w_t\right)}{g\left(w_t\right)},\quad t\geq1.
$$

We assume that $f$ and $g$ both put positive probabilities on the
same intervals of possible realizations of the random variable $W$.

That means that under the $g$ density,  $\ell (w_t)=
\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}$
is a nonnegative  random variable with mean $1$.

A **likelihood ratio process** for sequence
$\left\{ w_{t}\right\} _{t=1}^{\infty}$ is defined as

$$
L\left(w^{t}\right)=\prod_{i=1}^{t} \ell (w_i),
$$

where $w^t=\{ w_1,\dots,w_t\}$ is a history of
observations up to and including time $t$.

Sometimes for shorthand we'll write $L_t =  L(w^t)$.

Notice that the likelihood process satisfies the *recursion* 

$$
L(w^t) = \ell (w_t) L (w^{t-1}) .
$$

The likelihood ratio and its logarithm are key tools for making
inferences using a classic frequentist approach due to Neyman and
Pearson {cite}`Neyman_Pearson`.

To help us appreciate how things work, the following Python code evaluates $f$ and $g$ as two different
beta distributions, then computes and simulates an associated likelihood
ratio process by generating a sequence $w^t$ from one of the two
probability distributions, for example, a sequence of  IID draws from $g$.

```{code-cell} ipython3
# Parameters in the two beta distributions.
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

To see it this more clearly clearly, we plot over time the fraction of
paths $L\left(w^{t}\right)$ that fall in the interval
$\left[0, 0.01\right]$.

```{code-cell} ipython3
plt.plot(range(T), np.sum(l_seq_g <= 0.01, axis=0) / N)
```

Despite the evident convergence of most probability mass to a
very small interval near $0$,  the unconditional mean of
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

How can $E\left[L\left(w^{t}\right)\bigm|q=g\right]=1$ possibly be true when most  probability mass of the likelihood
ratio process is piling up near $0$ as
$t \rightarrow + \infty$?

The answer is that as $t \rightarrow + \infty$, the
distribution of $L_t$ becomes more and more fat-tailed:
enough  mass shifts to larger and larger values of $L_t$ to make
the mean of $L_t$ continue to be one despite most of the probability mass piling up
near $0$.

To illustrate this peculiar property, we simulate many paths and
calculate the unconditional mean of $L\left(w^t\right)$ by
averaging across these many paths at each $t$.

```{code-cell} ipython3
l_arr_g = simulate(G_a, G_b, N=50000)
l_seq_g = np.cumprod(l_arr_g, axis=1)
```

It would be useful to use simulations to verify that  unconditional means
$E\left[L\left(w^{t}\right)\right]$ equal unity by averaging across sample
paths.

But it would be too computer-time-consuming for us to that  here simply by applying a standard Monte Carlo simulation approach.

The reason is that the distribution of $L\left(w^{t}\right)$ is extremely skewed for large values of  $t$.

Because the probability density in the right tail is close to $0$,  it just takes too much computer time to sample enough points from the right tail.

We explain the problem in more detail  in {doc}`this lecture <imp_sample>`.

There we describe a way to an alternative way to compute the mean of a likelihood ratio by computing the mean of a _different_ random variable by sampling from  a _different_ probability distribution.


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
```

We also plot the probability that $L\left(w^t\right)$ falls into
the interval $[10000, \infty)$ as a function of time and watch how
fast probability mass diverges  to $+\infty$.

```{code-cell} ipython3
plt.plot(range(T), np.sum(l_seq_f > 10000, axis=0) / N)
```

## Likelihood Ratio Test

We now describe how to employ the machinery
of Neyman and Pearson {cite}`Neyman_Pearson` to test the hypothesis that  history $w^t$ is generated by repeated
IID draws from density $g$.

Denote $q$ as the data generating process, so that
$q=f \text{ or } g$.

Upon observing a sample $\{W_i\}_{i=1}^t$, we want to decide
whether nature is drawing from $g$ or from $f$ by performing  a (frequentist)
hypothesis test.

We specify

- Null hypothesis $H_0$: $q=f$,
- Alternative hypothesis $H_1$: $q=g$.

Neyman and Pearson proved that the best way to test this hypothesis is to use a **likelihood ratio test** that takes the
form:

- accept $H_0$ if $L(W^t) > c$,
- reject $H_0$ if $L(W^t) < c$,


where $c$ is a given  discrimination threshold.

Setting $c =1$ is a common choice.

We'll discuss consequences of other choices of $c$ below.  

This test is *best* in the sense that it is  **uniformly most powerful**.

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

These two probabilities underly  the following two concepts: 


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

Another way to say the same thing is that  among all possible tests, a likelihood ratio test
maximizes **power** for a given **significance level**.

We want a small probability of
false alarm and a large probability of detection.

With sample size $t$ fixed, we can change our two probabilities by
adjusting $c$.

A troublesome "that's life" fact is that these two probabilities  move in the same direction as we vary the critical value
$c$.

Without specifying quantitative losses from making Type I and Type II errors, there is little that we can say
about how we *should*  trade off probabilities of the two types of mistakes.

We do know that increasing sample size $t$ improves
statistical inference.

Below we plot some informative figures that illustrate this.

We also present a classical frequentist method for choosing a sample
size $t$.

Let’s start with a case in which we fix the threshold $c$ at
$1$.

```{code-cell} ipython3
c = 1
```

Below we plot empirical distributions of logarithms of the cumulative
likelihood ratios simulated above, which are generated by either
$f$ or $g$.

Taking logarithms has no effect on calculating the probabilities because
the log  is a monotonic transformation.

As $t$ increases, the probabilities of making Type I and Type II
errors both decrease, which is good.

This is because most of the probability mass of log$(L(w^t))$
moves toward $-\infty$ when $g$ is the data generating
process, ; while log$(L(w^t))$ goes to
$\infty$ when data are generated by $f$.

That disparate  behavior of log$(L(w^t))$ under $f$ and $q$
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
they are integrals of $\log L_t$, not integrals of $L_t$, over  rejection region $L_t < 1$  
* the orange areas are related to but not equal to probabilities $\beta $ of a type II error because 
they are integrals of $\log L_t$, not integrals of $L_t$, over  acceptance region $L_t > 1$


When we hold $c$ fixed at $c=1$, the following graph shows  that 
  *  the probability of detection monotonically increases with increases in
$t$ 
  *  the probability of a false alarm monotonically decreases with increases in $t$.

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

For a given sample size $t$,  the threshold $c$ uniquely pins down  probabilities
of both types of error.

If for a fixed $t$ we now free up and move $c$, we will sweep out the probability
of detection as a function of the probability of false alarm.

This produces  a [receiver operating characteristic
curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic).

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
presented to Milton Friedman, as we describe in  {doc}`this lecture <wald_friedman>`.

(rel_entropy)=
## Kullback–Leibler Divergence

Now let’s consider a case in which neither $g$ nor $f$
generates the data.

Instead, a third distribution $h$ does.

Let’s watch how how the cumulated likelihood ratios $f/g$ behave
when $h$ governs the data.

A key tool here is called **Kullback–Leibler divergence**.

It is also called **relative entropy**.

It measures how one probability distribution differs from another.

In our application, we want to measure how $f$ or $g$
diverges from $h$

The two Kullback–Leibler divergences pertinent for us are $K_f$
and $K_g$ defined as

$$
\begin{aligned}
K_{f}   &=E_{h}\left[\log\left(\frac{f\left(w\right)}{h\left(w\right)}\right)\frac{f\left(w\right)}{h\left(w\right)}\right] \\
    &=\int\log\left(\frac{f\left(w\right)}{h\left(w\right)}\right)\frac{f\left(w\right)}{h\left(w\right)}h\left(w\right)dw \\
    &=\int\log\left(\frac{f\left(w\right)}{h\left(w\right)}\right)f\left(w\right)dw
\end{aligned}
$$

$$
\begin{aligned}
K_{g}   &=E_{h}\left[\log\left(\frac{g\left(w\right)}{h\left(w\right)}\right)\frac{g\left(w\right)}{h\left(w\right)}\right] \\
    &=\int\log\left(\frac{g\left(w\right)}{h\left(w\right)}\right)\frac{g\left(w\right)}{h\left(w\right)}h\left(w\right)dw \\
    &=\int\log\left(\frac{g\left(w\right)}{h\left(w\right)}\right)g\left(w\right)dw
\end{aligned}
$$

When $K_g < K_f$, $g$ is closer to $h$ than $f$
is.

- In that case we’ll find that $L\left(w^t\right) \rightarrow 0$.

When $K_g > K_f$, $f$ is closer to $h$ than $g$
is.

- In that case we’ll find that
  $L\left(w^t\right) \rightarrow + \infty$

We’ll now experiment with an $h$ is also a beta distribution

We’ll start by setting parameters $G_a$ and $G_b$ so that
$h$ is closer to $g$

```{code-cell} ipython3
H_a, H_b = 3.5, 1.8

h = jit(lambda x: p(x, H_a, H_b))
```

```{code-cell} ipython3
x_range = np.linspace(0, 1, 100)
plt.plot(x_range, f(x_range), label='f')
plt.plot(x_range, g(x_range), label='g')
plt.plot(x_range, h(x_range), label='h')

plt.legend()
plt.show()
```

Let’s compute the Kullback–Leibler discrepancies by quadrature
integration.

```{code-cell} ipython3
def KL_integrand(w, q, h):

    m = q(w) / h(w)

    return np.log(m) * q(w)
```

```{code-cell} ipython3
def compute_KL(h, f, g):

    Kf, _ = quad(KL_integrand, 0, 1, args=(f, h))
    Kg, _ = quad(KL_integrand, 0, 1, args=(g, h))

    return Kf, Kg
```

```{code-cell} ipython3
Kf, Kg = compute_KL(h, f, g)
Kf, Kg
```

We have $K_g < K_f$.

Next, we can verify our conjecture about $L\left(w^t\right)$ by
simulation.

```{code-cell} ipython3
l_arr_h = simulate(H_a, H_b)
l_seq_h = np.cumprod(l_arr_h, axis=1)
```

The figure below plots over time the fraction of paths
$L\left(w^t\right)$ that fall in the interval $[0,0.01]$.

Notice that, as expected,  it converges to 1  when $g$ is closer to
$h$ than $f$ is.

```{code-cell} ipython3
N, T = l_arr_h.shape
plt.plot(range(T), np.sum(l_seq_h <= 0.01, axis=0) / N)
```

We can also try an $h$ that is closer to $f$ than is
$g$ so that now $K_g$ is larger than $K_f$.

```{code-cell} ipython3
H_a, H_b = 1.2, 1.2
h = jit(lambda x: p(x, H_a, H_b))
```

```{code-cell} ipython3
Kf, Kg = compute_KL(h, f, g)
Kf, Kg
```

```{code-cell} ipython3
l_arr_h = simulate(H_a, H_b)
l_seq_h = np.cumprod(l_arr_h, axis=1)
```

Now probability mass of $L\left(w^t\right)$ falling above
$10000$ diverges to $+\infty$.

```{code-cell} ipython3
N, T = l_arr_h.shape
plt.plot(range(T), np.sum(l_seq_h > 10000, axis=0) / N)
```

## Bayesian Classification and Hypothesis Testing

We now describe how a Bayesian statistician can combine frequentist probabilities of type I and type II errors in order to 

* compute a posterior probability of selecting a wrong model
* compute an anticipated error  rate in a classification problem 

We consider a situation in which  nature generates data by mixing known densities $f$ and $g$ with known mixing
parameter $\pi_{-1} \in (0,1)$ so that the random variable $w$ is drawn from the density

$$
h (w) = \pi_{-1} f(w) + (1-\pi_{-1}) g(w) 
$$

We'll often set $\pi_{-1} = .5$.  

We assume that $f$ and $g$ both put positive probabilities on the same intervals of possible realizations of the random variable $W$.

In the simulation below, we define $f$ as the probability density function of a $\text{Beta}(1, 1)$ distribution and $g$ as the probability density function of a $\text{Beta}(3, 1.2)$ distribution as we did above.

We consider two alternative timing protocols.  

**Protocol 1:**  Nature flips a coin once at time $t=-1$ and with probability $\pi_{-1}$  generates a sequence  $\{w_t\}_{t=1}^T$
of  IID  draws from  $f$  and with probability $1-\pi_{-1}$ generates a sequence  $\{w_t\}_{t=1}^T$
of  IID  draws from  $g$

```{code-cell} ipython3
def protocol_1(π_minus_1, T, N=1000):
    """
    Simulate Protocol 1: 
    Nature decides once at t=-1 which model to use.
    """
    
    sequences = np.empty((N, T))
    true_models = np.empty(N, dtype=int)
    
    for i in range(N):
        # Nature flips coin
        if np.random.rand() < π_minus_1:
            # Generate entire sequence from f
            sequences[i, :] = np.random.beta(F_a, F_b, T)
            true_models[i] = 1
        else:
            # Generate entire sequence from g
            sequences[i, :] = np.random.beta(G_a, G_b, T)
            true_models[i] = 0
    
    return sequences, true_models
```

**Protocol 2.** At each time $t \geq 0$, nature flips a coin and with probability $\pi_{-1}$ draws $w_t$ from $f$ and with probability $1-\pi_{-1}$ draws $w_t$ from $g$.

```{code-cell} ipython3
def protocol_2(π_minus_1, T, N=1000):
    """
    Simulate Protocol 2: 
    Nature decides at each time step which model to use.
    """
    
    sequences = np.empty((N, T))
    true_models = np.empty((N, T), dtype=int)
    
    for i in range(N):
        for t in range(T):
            # Nature flips coin at each time step t
            if np.random.rand() < π_minus_1:
                sequences[i, t] = np.random.beta(F_a, F_b)
                true_models[i, t] = 1
            else:
                sequences[i, t] = np.random.beta(G_a, G_b)
                true_models[i, t] = 0
                
    return sequences, true_models
```

**Remark:** Under protocol 2, the $\{w_t\}_{t=1}^T$ is a sequence of IID draws from $h(w)$. Under protocol 1, the the $\{w_t\}_{t=1}^T$ is 
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

In the next cell, we write the likelihood ratio calculation that we have done [previously](nature_likeli) into a function

```{code-cell} ipython3
def compute_likelihood_ratios(sequences):
    """
    Compute likelihood ratios for given sequences.
    """
    
    N, T = sequences.shape
    l_ratios = np.empty((N, T))
    L_cumulative = np.empty((N, T))
    
    for i in range(N):
        for t in range(T):
            l_ratios[i, t] = f(sequences[i, t]) / g(sequences[i, t])
            
        # Compute cumulative products
        L_cumulative[i, :] = np.cumprod(l_ratios[i, :])
    
    return l_ratios, L_cumulative
```

## Bayesian Model Selection 

We first study  a problem that assumes  timing protocol 1.  

Consider a decision maker who wants to know whether model $f$ or model $g$ governs the data.

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

We can form a Bayesian prior  probability that the likelihood ratio selects the wrong model by assigning a prior probability of $\pi_{-1} = .5$ that it selects the wrong model and then  averaging $p_f$ and $p_g$ to form the Bayesian posterior probability of a detection error equal to

$$ 
p(\textrm{wrong decision}) = {1 \over 2} (\alpha_T + \beta_T) .
$$ (eq:detectionerrorprob)

Now let's simulate the protocol 1 and compute the error probabilities

```{code-cell} ipython3
# Set parameters
π_minus_1 = 0.5
T_max = 30
N_simulations = 5000

# Protocol 1: Bayesian Model Selection
print("Protocol 1: Bayesian Model Selection")

sequences_p1, true_models_p1 = protocol_1(
                            π_minus_1, T_max, N_simulations)
l_ratios_p1, L_cumulative_p1 = compute_likelihood_ratios(sequences_p1)

# Compute error probabilities for different sample sizes
T_range = range(1, T_max + 1)
α_T = np.empty(T_max)  # P(L_T < 1 | f)
β_T = np.empty(T_max)   # P(L_T >= 1 | g)
bayesian_error_prob = np.empty(T_max)

for t in T_range:
    t_idx = t - 1
    
    # Type I error: reject H_0 when it's true 
    # (model f generates data)
    f_sequences = L_cumulative_p1[true_models_p1 == 1, t_idx]
    α_T[t_idx] = np.mean(f_sequences < 1)
    
    # Type II error: accept H_0 when it's false 
    # (model g generates data)
    g_sequences = L_cumulative_p1[true_models_p1 == 0, t_idx]
    β_T[t_idx] = np.mean(g_sequences >= 1)
    
    # Bayesian error probability
    bayesian_error_prob[t_idx] = 0.5 * (α_T[t_idx] + β_T[t_idx])

# Plot results for Protocol 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(T_range, α_T, 'b-', 
         label=r'$\alpha_T$ (Type I error)', linewidth=2)
ax1.plot(T_range, β_T, 'r-', 
         label=r'$\beta_T$ (Type II error)', linewidth=2)
ax1.set_xlabel('$T$')
ax1.set_ylabel('error probability')
ax1.legend()

ax2.plot(T_range, bayesian_error_prob, 'g-', 
         label=r'$\frac{1}{2}(\alpha_T+\beta_T)$', linewidth=2)
ax2.set_xlabel('$T$')
ax2.set_ylabel('error probability')
ax2.legend()

plt.tight_layout()
plt.show()

print(f"At T={T_max}:")
print(f"α_{T_max} = {α_T[-1]:.4f}")
print(f"β_{T_max} = {β_T[-1]:.4f}")
print(f"Bayesian error probability = {bayesian_error_prob[-1]:.4f}")
```

## Classification

We now consider a problem that assumes timing protocol 2.

A decision maker wants to classify components of an observed sequence $\{w_t\}_{t=1}^T$ as having been drawn from either $f$ or $g$.

The decision maker uses the following classification rule:

$$
\begin{align*}
w_t  & \ {\rm is \ from \  f  \ if \ } l_t > 1 \\
w_t  & \ {\rm is \ from \  g  \ if \ } l_t \leq 1 . 
\end{align*}
$$

Under this rule, the expected misclassification rate is

$$
p(\textrm{misclassification}) = {1 \over 2} (\alpha_1 + \beta_1) 
$$ (eq:classerrorprob)

Now let's simulate protocol 2 and compute the error probabilities

```{code-cell} ipython3
sequences_p2, true_sources_p2 = protocol_2(
                    π_minus_1, T_max, N_simulations)
l_ratios_p2, _ = compute_likelihood_ratios(sequences_p2)

# Find decision boundary where f(w) = g(w)
root = brentq(lambda w: f(w) / g(w) - 1, 0.001, 0.999)

print(f"Decision boundary points: {root}")

# Compute theoretical α_1 and β_1 using integration
def α_integrand(w):
    """Integrand for α_1 = P(l_1 < 1 | f)"""
    return f(w) if f(w) / g(w) < 1 else 0

def β_integrand(w):
    """Integrand for β_1 = P(l_1 >= 1 | g)"""
    return g(w) if f(w) / g(w) >= 1 else 0

# Compute the integrals
α_1_theory, _ = quad(α_integrand, 0, 1, limit=100)
β_1_theory, _ = quad(β_integrand, 0, 1, limit=100)

theory_error = 0.5 * (α_1_theory + β_1_theory)

print(f"theoretical α_1 = {α_1_theory:.4f}")
print(f"theoretical β_1 = {β_1_theory:.4f}")
print(f"theoretical classification error probability = {theory_error:.4f}")

# Visualization of distributions and decision boundary
fig, ax = plt.subplots(figsize=(7, 6))

w_range = np.linspace(0.001, 0.999, 1000)
f_values = [f(w) for w in w_range]
g_values = [g(w) for w in w_range]
ratio_values = [f(w)/g(w) for w in w_range]

ax.plot(w_range, f_values, 'b-', 
        label='$f(w) \sim Beta(1,1)$', linewidth=2)
ax.plot(w_range, g_values, 'r-', 
        label='$g(w) \sim Beta(3,1.2)$', linewidth=2)

ax.axvline(root, color='green', linestyle='--', alpha=0.7, 
            label=f'decision boundary: $w=${root:.3f}')

ax.set_xlabel('w')
ax.set_ylabel('probability density')
ax.legend()

plt.tight_layout()
plt.show()
```

Let's see how it performs in the simulated data

```{code-cell} ipython3
correct_classifications = np.empty(T_max)

for t in range(T_max):
    predictions = (l_ratios_p2[:, t] > 1).astype(int)
    actual = true_sources_p2[:, t]
    correct_classifications[t] = np.mean(predictions == actual)

plt.figure(figsize=(10, 6))
plt.plot(range(1, T_max + 1), correct_classifications, 'b-', linewidth=2, label='empirical accuracy')
plt.axhline(1 - theory_error, color='r', linestyle='--', 
                label=f'theoretical accuracy = {1 - theory_error:.4f}')
plt.xlabel('time step')
plt.ylabel('accuracy')
plt.legend()
plt.ylim(0.5, 1.0)
plt.show()
```

Let's also compare the two protocols by showing how the error probabilities evolve differently.

```{code-cell} ipython3
# Comparison of error probabilities between protocols
fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(T_range, bayesian_error_prob, 'b-', linewidth=2, 
        label='Protocol 1 (Model Selection)')
ax.axhline(theory_error, color='r', linestyle='--', linewidth=2, 
        label=f'Protocol 2 (Classification) = {theory_error:.4f}')
ax.set_xlabel('T')
ax.set_ylabel('error probability')
ax.legend()
plt.show()
```

From the figure above, we can see:

- For protocol 1, the error probability decreases as we collect more data because we're trying to determine which single model generated the entire sequence. More data provides stronger evidence.

- For protocol 2, the error probability remains constant because each observation is classified independently. The accuracy depends only on the likelihood that the two models generates the single observation.

- Under Protocol 1, the Bayesian error probability converges to zero as $T \to \infty$, while under Protocol 2, it remains constant at ${1 \over 2}(\alpha_1 + \beta_1)$.

## Sequels

Likelihood processes play an important role in Bayesian learning, as described in {doc}`likelihood_bayes`
and as applied in {doc}`odu`.

Likelihood ratio processes appear again in {doc}`additive_functionals`, which contains another illustration
of the **peculiar property** of likelihood ratio processes described above.
