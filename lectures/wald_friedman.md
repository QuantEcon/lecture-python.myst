---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(wald_friedman)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`A Problem that Stumped Milton Friedman <single: A Problem that Stumped Milton Friedman>`

(and that Abraham Wald solved by inventing sequential analysis)

```{index} single: Models; Sequential analysis
```

```{contents} Contents
:depth: 2
```

## Overview

This lecture describes a statistical decision problem presented to Milton
Friedman and W. Allen Wallis during World War II when they were analysts at
the U.S. Government's  Statistical Research Group at Columbia University.

This problem led Abraham Wald {cite}`Wald47` to formulate **sequential analysis**,
an approach to statistical decision problems intimately related to dynamic programming.

In this lecture, we describe elements of Wald's formulation of the problem.

Key ideas in play will be:

- Type I and type II statistical errors
    - a type I error occurs when you reject a null hypothesis that is true
    - a type II error occures when you accept a null hypothesis that is false
- Abraham Wald's **sequential probability ratio test**
- The **power** of a statistical test
- The **critical region** of a statistical test
- A **uniformly most powerful test**

We'll begin with some imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange, float64, int64
from numba.experimental import jitclass
from math import gamma
```

This lecture uses ideas studied in {doc}`this lecture <likelihood_ratio_process>` and  {doc}`this lecture <likelihood_bayes>`.

## Origin of the Problem

On pages 137-139 of his 1998 book *Two Lucky People* with Rose Friedman {cite}`Friedman98`,
Milton Friedman described a problem presented to him and Allen Wallis
during World War II, when they worked at the US Government's
Statistical Research Group at Columbia University.

```{note}
See pages 25 and 26  of Allen Wallis's 1980 article {cite}`wallis1980statistical`  about the Statistical Research Group at Columbia University during World War II for his account of the episode and  for important contributions  that Harold Hotelling made to formulating the problem.   Also see  chapter 5 of Jennifer Burns book about
Milton Friedman {cite}`Burns_2023`.
```

Let's listen to Milton Friedman tell us what happened

> In order to understand the story, it is necessary to have an idea of a
> simple statistical problem, and of the standard procedure for dealing
> with it. The actual problem out of which sequential analysis grew will
> serve. The Navy has two alternative designs (say A and B) for a
> projectile. It wants to determine which is superior. To do so it
> undertakes a series of paired firings. On each round, it assigns the
> value 1 or 0 to A accordingly as its performance is superior or inferior
> to that of B and conversely 0 or 1 to B. The Navy asks the statistician
> how to conduct the test and how to analyze the results.

> The standard statistical answer was to specify a number of firings (say
> 1,000) and a pair of percentages (e.g., 53% and 47%) and tell the client
> that if A receives a 1 in more than 53% of the firings, it can be
> regarded as superior; if it receives a 1 in fewer than 47%, B can be
> regarded as superior; if the percentage is between 47% and 53%, neither
> can be so regarded.

> When Allen Wallis was discussing such a problem with (Navy) Captain
> Garret L. Schyler, the captain objected that such a test, to quote from
> Allen's account, may prove wasteful. If a wise and seasoned ordnance
> officer like Schyler were on the premises, he would see after the first
> few thousand or even few hundred [rounds] that the experiment need not
> be completed either because the new method is obviously inferior or
> because it is obviously superior beyond what was hoped for
> $\ldots$.

Friedman and Wallis struggled with the problem but, after realizing that
they were not able to solve it,  described the problem to  Abraham Wald.

That started Wald on the path that led him  to *Sequential Analysis* {cite}`Wald47`.



##  Neyman-Pearson Formulation

It is useful to begin by describing the theory underlying the test
that Navy Captain G. S. Schuyler had been told to use and that led him
to approach Milton Friedman and Allan Wallis to convey his conjecture
that superior practical procedures existed.

Evidently, the Navy had told Captail Schuyler to use what was then  the state-of-the-art
Neyman-Pearson test.

We'll rely on Abraham Wald's {cite}`Wald47` elegant summary of Neyman-Pearson theory.

Watch for these features of the setup:

- the assumption of a *fixed* sample size $n$
- the application of laws of large numbers, conditioned on alternative
  probability models, to interpret the probabilities $\alpha$ and
  $\beta$ defined in the Neyman-Pearson theory


In chapter 1 of **Sequential Analysis** {cite}`Wald47` Abraham Wald summarizes the
Neyman-Pearson approach to hypothesis testing.

Wald frames the problem as making a decision about a probability
distribution that is partially known.

(You have to assume that *something* is already known in order to state a well-posed
problem -- usually, *something* means *a lot*)

By limiting  what is unknown, Wald uses the following simple structure
to illustrate the main ideas:

- A decision-maker wants to decide which of two distributions
  $f_0$, $f_1$ govern an IID random variable $z$.
- The null hypothesis $H_0$ is the statement that $f_0$
  governs the data.
- The alternative hypothesis $H_1$ is the statement that
  $f_1$ governs the data.
- The problem is to devise and analyze a test of hypothesis
  $H_0$ against the alternative hypothesis $H_1$ on the
  basis of a sample of a fixed number $n$ independent
  observations $z_1, z_2, \ldots, z_n$ of the random variable
  $z$.

To quote Abraham Wald,

> A test procedure leading to the acceptance or rejection of the [null]
> hypothesis in question is simply a rule specifying, for each possible
> sample of size $n$, whether the [null] hypothesis should be accepted
> or rejected on the basis of the sample. This may also be expressed as
> follows: A test procedure is simply a subdivision of the totality of
> all possible samples of size $n$ into two mutually exclusive
> parts, say part 1 and part 2, together with the application of the
> rule that the [null] hypothesis be accepted if the observed sample is
> contained in part 2. Part 1 is also called the critical region. Since
> part 2 is the totality of all samples of size $n$ which are not
> included in part 1, part 2 is uniquely determined by part 1. Thus,
> choosing a test procedure is equivalent to determining a critical
> region.

Let's listen to Wald longer:

> As a basis for choosing among critical regions the following
> considerations have been advanced by Neyman and Pearson: In accepting
> or rejecting $H_0$ we may commit errors of two kinds. We commit
> an error of the first kind if we reject $H_0$ when it is true;
> we commit an error of the second kind if we accept $H_0$ when
> $H_1$ is true. After a particular critical region $W$ has
> been chosen, the probability of committing an error of the first
> kind, as well as the probability of committing an error of the second
> kind is uniquely determined. The probability of committing an error
> of the first kind is equal to the probability, determined by the
> assumption that $H_0$ is true, that the observed sample will be
> included in the critical region $W$. The probability of
> committing an error of the second kind is equal to the probability,
> determined on the assumption that $H_1$ is true, that the
> probability will fall outside the critical region $W$. For any
> given critical region $W$ we shall denote the probability of an
> error of the first kind by $\alpha$ and the probability of an
> error of the second kind by $\beta$.

Let's listen carefully to how Wald applies law of large numbers to
interpret $\alpha$ and $\beta$:

> The probabilities $\alpha$ and $\beta$ have the
> following important practical interpretation: Suppose that we draw a
> large number of samples of size $n$. Let $M$ be the
> number of such samples drawn. Suppose that for each of these
> $M$ samples we reject $H_0$ if the sample is included in
> $W$ and accept $H_0$ if the sample lies outside
> $W$. In this way we make $M$ statements of rejection or
> acceptance. Some of these statements will in general be wrong. If
> $H_0$ is true and if $M$ is large, the probability is
> nearly $1$ (i.e., it is practically certain) that the
> proportion of wrong statements (i.e., the number of wrong statements
> divided by $M$) will be approximately $\alpha$. If
> $H_1$ is true, the probability is nearly $1$ that the
> proportion of wrong statements will be approximately $\beta$.
> Thus, we can say that in the long run [ here Wald applies law of
> large numbers by driving $M \rightarrow \infty$ (our comment,
> not Wald's) ] the proportion of wrong statements will be
> $\alpha$ if $H_0$is true and $\beta$ if
> $H_1$ is true.

The quantity $\alpha$ is called the *size* of the critical region,
and the quantity $1-\beta$ is called the *power* of the critical
region.

Wald notes that

> one critical region $W$ is more desirable than another if it
> has smaller values of $\alpha$ and $\beta$. Although
> either $\alpha$ or $\beta$ can be made arbitrarily small
> by a proper choice of the critical region $W$, it is possible
> to make both $\alpha$ and $\beta$ arbitrarily small for a
> fixed value of $n$, i.e., a fixed sample size.

Wald summarizes Neyman and Pearson's setup as follows:

> Neyman and Pearson show that a region consisting of all samples
> $(z_1, z_2, \ldots, z_n)$ which satisfy the inequality
>
> $$
  \frac{ f_1(z_1) \cdots f_1(z_n)}{f_0(z_1) \cdots f_0(z_n)} \geq k
  $$
>
> is a most powerful critical region for testing the hypothesis
> $H_0$ against the alternative hypothesis $H_1$. The term
> $k$ on the right side is a constant chosen so that the region
> will have the required size $\alpha$.

Wald goes on to discuss Neyman and Pearson's concept of *uniformly most
powerful* test.

Here is how Wald introduces the notion of a sequential test

> A rule is given for making one of the following three decisions at any stage of
> the experiment (at the m th trial for each integral value of m ): (1) to
> accept the hypothesis H , (2) to reject the hypothesis H , (3) to
> continue the experiment by making an additional observation. Thus, such
> a test procedure is carried out sequentially. On the basis of the first
> observation, one of the aforementioned decision is made. If the first or
> second decision is made, the process is terminated. If the third
> decision is made, a second trial is performed. Again, on the basis of
> the first two observations, one of the three decision is made. If the
> third decision is made, a third trial is performed, and so on. The
> process is continued until either the first or the second decisions is
> made. The number n of observations required by such a test procedure is
> a random variable, since the value of n depends on the outcome of the
> observations.

## Wald's sequential formulation 

In contradistinction to Neyman-Pearson formulation of the problemm, in Wald's formulation


- The sample size $n$ is not fixed but rather an object to be
  chosen; technically $n$ is a random variable.
- Two  parameters $A$ and $B$ that are related to but distinct from Neyman and Pearson's  $\alpha$ and  $\beta$,  characterize cut-off   rules that Wald  uses to determine the random variable $n$.

Here is how Wald sets up the problem.

A decision-maker can observe a sequence of draws of a random variable $z$.

He (or she) wants to know which of two probability distributions $f_0$ or $f_1$ governs $z$.


To  illustrate, let's inspect some  beta distributions.

The density of a beta probability distribution with parameters $a$ and $b$ is

$$
f(z; a, b) = \frac{\Gamma(a+b) z^{a-1} (1-z)^{b-1}}{\Gamma(a) \Gamma(b)}
\quad \text{where} \quad
\Gamma(t) := \int_{0}^{\infty} x^{t-1} e^{-x} dx
$$

The next figure shows two beta distributions.

## Request for Humphrey. 

The bottom panel presents mixtures of these distributions, with various mixing probabilities $\pi_k$ --
that is inherited from the Bayesian lecture.

Please remove the bottom panel and just leave the top panel.  

This is the end of the "message"

```{code-cell} ipython3
@jit
def p(x, a, b):
    r = gamma(a + b) / (gamma(a) * gamma(b))
    return r * x**(a-1) * (1 - x)**(b-1)

f0 = lambda x: p(x, 1, 1)
f1 = lambda x: p(x, 9, 9)
grid = np.linspace(0, 1, 50)

fig, axes = plt.subplots(2, figsize=(10, 8))

axes[0].set_title("Original Distributions")
axes[0].plot(grid, f0(grid), lw=2, label="$f_0$")
axes[0].plot(grid, f1(grid), lw=2, label="$f_1$")

axes[1].set_title("Mixtures")
for π in 0.25, 0.5, 0.75:
    y = π * f0(grid) + (1 - π) * f1(grid)
    axes[1].plot(y, lw=2, label=rf"$\pi_k$ = {π}")

for ax in axes:
    ax.legend()
    ax.set(xlabel="$z$ values", ylabel="probability of $z_k$")

plt.tight_layout()
plt.show()
```

Conditional on knowing that successive observations are drawn from distribution $f_0$, the sequence of
random variables is independently and identically distributed (IID).

Conditional on knowing that successive observations are drawn from distribution $f_1$, the sequence of
random variables is also independently and identically distributed (IID).

But the observer does not know which of the two distributions generated the sequence.

For reasons explained in  [Exchangeability and Bayesian Updating](https://python.quantecon.org/exchangeable.html), this means that the sequence is not
IID.

The observer has something to learn, namely, whether the observations are drawn from  $f_0$ or from $f_1$.

The decision maker   wants  to decide which of the  two distributions is generating outcomes.





### Type I and Type II Errors

If we regard  $f=f_0$ as a null hypothesis and $f=f_1$ as an alternative hypothesis,
then 

- a type I error is an incorrect rejection of a true null hypothesis (a "false positive")
- a type II error is a failure to reject a false null hypothesis (a "false negative")

To repeat ourselves

- $\alpha$ is the probability of a type I error
- $\beta$ is the probability of a type II error

**note to Humphrey -- please leave the alpha and beta as they are in this section!

### Choices

After observing $z_k, z_{k-1}, \ldots, z_0$, the decision-maker
chooses among three distinct actions:

- He decides that $f = f_0$ and draws no more $z$'s
- He decides that $f = f_1$ and draws no more $z$'s
- He postpones deciding now and instead chooses to draw a
  $z_{k+1}$


### Message to Humphrey

We want to redraw this figure with $A$ and $B$ replacing $\alpha$ and $\beta$. I suspect that we
used the notebook to generate the figure.  



```{figure} /_static/lecture_specific/wald_friedman/wald_dec_rule.png

```





Wald proceeds as follows.

He defines

- $p_{0m} = f_0(z_0) \cdots f_0(z_k)$
- $p_{1m} = f_1(z_0) \cdots f_1(z_k)$
- $L_{m} = \frac{p_{1m}}{p_{0m}}$

Here $\{L_m\}_{m=0}^\infty$ is a **likelihood ratio process**.

One of Wald's sequential  decision rule is parameterized by two real numbers $B < A$.

For a given pair $A, B$ the decision rule is 

$$
\begin{aligned}
\textrm { accept } f=f_0 \textrm{ if } L_m \geq A \\
\textrm { accept } f=f_1 \textrm{ if } L_m \leq B \\
\textrm { draw another }  z \textrm{ if }  B < L_m < A
\end{aligned}
$$

### Links between $A,B$ and $\alpha, \beta$

In chapter 3 of **Sequential Analysis** {cite}`Wald47`  Wald establishes the inequalities

$$ 
\begin{align} 
 \frac{\alpha}{1 -\beta} & \leq \frac{1}{A} \\
 \frac{\beta}{1 - \alpha} & \leq B 
\end{align}
$$

His analysis of these inequalities leads Wald to recommend the following as rules for setting 
$A$ and $B$ that come close to attaining a decision maker's target values for probabilities $\alpha$ of
a  type I  and $\beta$ of a type II error:

$$
\begin{align}
A(\alpha,\beta) & = \frac{1-\beta}{\alpha} \\
B(\alpha,\beta)  & = \frac{\beta}{1-\alpha} 
\end{align} 
$$ (eq:Waldrule)

 For small values of $\alpha $ and $\beta$, Wald shows that {eq}`eq:Waldrule` provides  good ways to set $A$ and $B$. 


## Message to Humphrey

I want drastically to edit the following section.  We want instead to create some examples along the following lines.

- we'll set a pair of target $\alpha, \beta$ (size, power parameters)
- we'll use formulas {eq}`eq:Waldrule` for $A(\alpha,\beta), B(\alpha, \beta)$
- we'll set beta distributions for $f_1$ and $f_2$ that have lots of overlap and are more or less challenging to distinguish.
- we'll then simulate Wald's decision rule and generate distributions of 
    - stopping times $n$
    - probabilities of making type I and type II errors that we'll compare with our target $\alpha, \beta$

I welcome your suggestions for improvements and more interesting experiments.

## Simulations


The next figure shows the outcomes of 500 simulations of the decision process.

On the left is a histogram of **stopping times**, i.e.,  the number of draws of $z_k$ required to make a decision.

The average number of draws is around 6.6.

On the right is the fraction of correct decisions at the stopping time.

In this case, the decision-maker is correct 80% of the time

```{code-cell} ipython3
def simulate(wf, true_dist, h_star, π_0=0.5):

    """
    This function takes an initial condition and simulates until it
    stops (when a decision is made)
    """

    f0, f1 = wf.f0, wf.f1
    f0_rvs, f1_rvs = wf.f0_rvs, wf.f1_rvs
    π_grid = wf.π_grid
    κ = wf.κ

    if true_dist == "f0":
        f, f_rvs = wf.f0, wf.f0_rvs
    elif true_dist == "f1":
        f, f_rvs = wf.f1, wf.f1_rvs

    # Find cutoffs
    β, α = find_cutoff_rule(wf, h_star)

    # Initialize a couple of useful variables
    decision_made = False
    π = π_0
    t = 0

    while decision_made is False:
        # Maybe should specify which distribution is correct one so that
        # the draws come from the "right" distribution
        z = f_rvs()
        t = t + 1
        π = κ(z, π)
        if π < β:
            decision_made = True
            decision = 1
        elif π > α:
            decision_made = True
            decision = 0

    if true_dist == "f0":
        if decision == 0:
            correct = True
        else:
            correct = False

    elif true_dist == "f1":
        if decision == 1:
            correct = True
        else:
            correct = False

    return correct, π, t

def stopping_dist(wf, h_star, ndraws=250, true_dist="f0"):

    """
    Simulates repeatedly to get distributions of time needed to make a
    decision and how often they are correct
    """

    tdist = np.empty(ndraws, int)
    cdist = np.empty(ndraws, bool)

    for i in range(ndraws):
        correct, π, t = simulate(wf, true_dist, h_star)
        tdist[i] = t
        cdist[i] = correct

    return cdist, tdist

def simulation_plot(wf):
    h_star = solve_model(wf)
    ndraws = 500
    cdist, tdist = stopping_dist(wf, h_star, ndraws)

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    ax[0].hist(tdist, bins=np.max(tdist))
    ax[0].set_title(f"Stopping times over {ndraws} replications")
    ax[0].set(xlabel="time", ylabel="number of stops")
    ax[0].annotate(f"mean = {np.mean(tdist)}", xy=(max(tdist) / 2,
                   max(np.histogram(tdist, bins=max(tdist))[0]) / 2))

    ax[1].hist(cdist.astype(int), bins=2)
    ax[1].set_title(f"Correct decisions over {ndraws} replications")
    ax[1].annotate(f"% correct = {np.mean(cdist)}",
                   xy=(0.05, ndraws / 2))

    plt.show()

simulation_plot(wf)
```




[^f1]: The decision maker acts as if he believes that the sequence of random variables
$[z_{0}, z_{1}, \ldots]$ is *exchangeable*.  See [Exchangeability and Bayesian Updating](https://python.quantecon.org/exchangeable.html) and
{cite}`Kreps88` chapter 11, for  discussions of exchangeability.

## Related lectures

We'll dig deeper into some of the ideas used here in the following earlier and later lectures:

* {doc}`this lecture <exchangeable>` discusses the key concept of **exchangeability** that rationalizes statistical learning
* {doc}`this lecture <likelihood_ratio_process>` describes **likelihood ratio processes** and their role in frequentist and Bayesian statistical theories
* {doc}`this lecture <likelihood_bayes>` discusses the role of likelihood ratio processes in **Bayesian learning**
* {doc}`this lecture <navy_captain>` takes up the subject of this lecture and studies whether the Captain's hunch that the (frequentist) decision rule  that the Navy had ordered him to use can be expected to be better or worse than our sequential decision rule 
