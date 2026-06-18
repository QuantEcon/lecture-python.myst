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

# Two Meanings of Probability


## Overview


This lecture  illustrates two distinct interpretations of a  **probability distribution**

 * A frequentist interpretation as **relative frequencies** anticipated to occur in a large IID sample

 * A Bayesian interpretation as a **personal opinion** (about a parameter or list of parameters) after seeing a collection of observations

We recommend watching the following two videos before proceeding:

* [Hypothesis testing within the frequentist approach](https://www.youtube.com/watch?v=8JIe_cz6qGA)

* [The Bayesian approach to constructing coverage intervals](https://www.youtube.com/watch?v=Pahyv9i_X2k)

After you are familiar with the material in these videos, this lecture uses the Socratic method to help consolidate your understanding of the different questions that are answered by

 * a frequentist confidence interval

 * a Bayesian coverage interval

We do this  by inviting you to  write some  Python code.

It would be especially useful if you tried doing this after each question that we pose for you,  before
proceeding to read the rest of the lecture.

We provide our own answers as the lecture unfolds, but you'll learn more if you try writing your own code before reading and running ours.

### Code for answering questions


To answer our coding questions, we’ll start with some imports

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
import scipy.stats as st
```

Empowered with these Python tools, we'll now  explore the two meanings described above.

## Frequentist Interpretation

Consider the following classic example.

The random variable  $X $ takes on possible values $k = 0, 1, 2, \ldots, n$  with probabilities

$$
p(k \mid \theta) := \mathbb{P}\{X = k \mid \theta\} =
\binom{n}{k} \theta^k (1-\theta)^{n-k}
$$

where the fixed parameter $\theta \in (0,1)$.

This is called the [binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution).

Here

* $\theta$ is the probability that one toss of a coin will be a head, an outcome that we encode as  $Y = 1$.

* $1 -\theta$ is the probability that one toss of the coin will be a tail, an outcome that we denote $Y = 0$.

* $X$ is the total number of heads that came up after flipping the coin $n$ times.

Consider the following experiment:

Take $I$ **independent** sequences of $n$  **independent** flips of the coin

Notice the repeated use of the adjective **independent**:

* we use it once to describe that we are drawing $n$ independent times from a **Bernoulli** distribution with parameter $\theta$ to arrive at one draw from a **Binomial** distribution with parameters
$\theta,n$.

* we use it again to describe that we are then drawing $I$  sequences of $n$ coin draws.

Let $y_h^i \in \{0, 1\}$ be the realized value of $Y$ on the $h$th flip during the $i$th sequence of flips.

Let $\sum_{h=1}^n y_h^i$ denote the total number of times  heads come up during the $i$th sequence of $n$ independent coin flips.

Let $f_k$ record the fraction of samples of length $n$ for which $\sum_{h=1}^n y_h^i = k$:

$$
f_k^I = \frac{1}{I} \sum_{i=1}^I \mathbb{1}\left\{ \sum_{h=1}^n y_h^i = k \right\}
$$

The probability  $p(k \mid \theta)$ answers the following question:

* As $I$ becomes large, in what   fraction of  $I$ independent  draws of  $n$ coin flips should we anticipate  $k$ heads to occur?

As usual, a law of large numbers justifies this answer.

```{exercise}
:label: pm_ex1

1. Write Python code to compute $f_k^I$

2. Use your code to compute $f_k^I, k = 0, \ldots , n$ and compare them to
  $p(k \mid \theta)$ for various values of $\theta, n$ and $I$

3. With the Law of Large Numbers in mind, use your code to describe the relationship between $f_k^I$ and $p(k \mid \theta)$ as $I$ grows
```

```{solution-start} pm_ex1
:class: dropdown
```

Here is one solution.

We simulate the coin flips with one function and assemble the comparison table
with another.

```{code-cell} ipython3
def simulate_head_counts(θ, n, I, seed=1234):
    "Simulate I sequences of n coin flips; return the heads count of each sequence."
    rng = np.random.default_rng(seed)
    Y = (rng.random((I, n)) <= θ).astype(int)
    return Y.sum(axis=1)
```

```{code-cell} ipython3
def compare_frequencies(θ, n, I, seed=1234):
    "Tabulate theoretical binomial probabilities against simulated frequencies."
    head_counts = simulate_head_counts(θ, n, I, seed)
    rows = [
        (k, binom.pmf(k, n, θ), np.mean(head_counts == k))
        for k in range(n + 1)
    ]
    return pd.DataFrame(
        rows, columns=['k', 'Theoretical', 'Frequentist']
    ).set_index('k')
```

```{code-cell} ipython3
θ, n, k, I = 0.7, 20, 10, 1_000_000

compare_frequencies(θ, n, I)
```

From the table above, can you see the law of large numbers at work?

```{solution-end}
```

Let's do some more calculations.

### Comparison with different $\theta$

Now we fix

$$
n=20, k=10, I=1,000,000
$$

We'll vary $\theta$ from $0.01$ to $0.99$ and plot outcomes against $\theta$.

```{code-cell} ipython3
θ_low, θ_high, n_thetas = 0.01, 0.99, 50
thetas = np.linspace(θ_low, θ_high, n_thetas)
P = []
f_kI = []
for i in range(n_thetas):
    P.append(binom.pmf(k, n, thetas[i]))
    head_counts = simulate_head_counts(thetas[i], n, I, seed=i)
    f_kI.append(np.mean(head_counts == k))
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 6))
ax.grid()
ax.plot(thetas, P, '-.', label='theoretical')
ax.plot(thetas, f_kI, '--', label='fraction')
ax.set_title(r'comparison with different $\theta$',
             fontsize=16)
ax.set_xlabel(r'$\theta$', fontsize=15)
ax.set_ylabel('fraction', fontsize=15)
ax.tick_params(labelsize=13)
ax.legend()
plt.show()
```

### Comparison with different $n$

Now we fix $\theta=0.7, k=10, I=1,000,000$ and vary $n$ from $1$ to $100$.

Then we'll plot outcomes.

```{code-cell} ipython3
n_low, n_high, n_ns = 1, 100, 50
ns = np.linspace(n_low, n_high, n_ns, dtype='int')
P = []
f_kI = []
for i in range(n_ns):
    P.append(binom.pmf(k, ns[i], θ))
    head_counts = simulate_head_counts(θ, ns[i], I, seed=i)
    f_kI.append(np.mean(head_counts == k))
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 6))
ax.grid()
ax.plot(ns, P, '-.', label='theoretical')
ax.plot(ns, f_kI, '--', label='fraction')
ax.set_title(r'comparison with different $n$',
             fontsize=16)
ax.set_xlabel(r'$n$', fontsize=15)
ax.set_ylabel('fraction', fontsize=15)
ax.tick_params(labelsize=13)
ax.legend()
plt.show()
```

### Comparison with different $I$

Now we fix $\theta=0.7, n=20, k=10$ and vary $\log(I)$ from $2$ to $6$.

```{code-cell} ipython3
I_log_low, I_log_high, n_Is = 2, 6, 200
log_Is = np.linspace(I_log_low, I_log_high, n_Is)
Is = np.power(10, log_Is).astype(int)
P = []
f_kI = []
for i in range(n_Is):
    P.append(binom.pmf(k, n, θ))
    head_counts = simulate_head_counts(θ, n, Is[i], seed=i)
    f_kI.append(np.mean(head_counts == k))
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 6))
ax.grid()
ax.plot(Is, P, '-.', label='theoretical')
ax.plot(Is, f_kI, '--', label='fraction')
ax.set_title(r'comparison with different $I$',
             fontsize=16)
ax.set_xlabel(r'$I$', fontsize=15)
ax.set_ylabel('fraction', fontsize=15)
ax.tick_params(labelsize=13)
ax.legend()
plt.show()
```

From the above graphs, we can see that **$I$, the number of independent sequences,** plays an important role.

When $I$ becomes larger, the difference between theoretical probability and frequentist estimate becomes smaller.

Also, as long as $I$ is large enough, changing $\theta$ or $n$ does not substantially change the accuracy of the observed fraction
as an approximation of $p(k \mid \theta)$.

The Law of Large Numbers is at work here.

For each independent sequence $i$, define the indicator $\rho_{k,i} = \mathbb{1}\{X_i = k\}$ — that is, $\rho_{k,i}$ equals 1 if the $i$-th sequence produces exactly $k$ heads and 0 otherwise.

The $\rho_{k,i}$ are IID across $i$, each with mean $p(k \mid \theta)$ and variance

$$
p(k \mid \theta) \cdot (1-p(k \mid \theta)).
$$

So, by the LLN, the average of $\rho_{k,i}$ converges to:

$$
\mathbb{E}[\rho_{k,i}] = p(k \mid \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k}
$$

as $I$ goes to infinity.


## Bayesian Interpretation

The likelihood remains binomial, but now we treat $\theta$ as a **random variable** rather than a fixed parameter.

So $\theta$ is described by a probability distribution.

But now this probability distribution means something different than a relative frequency that we can anticipate to occur in a large IID sample.

Instead, the probability distribution of $\theta$ is now a summary of our views about  likely values of $\theta$ either

  * **before** we have seen **any** data at all, or
  * **before** we have seen **more** data, after we have seen **some** data

Thus, suppose that, before seeing any data, you have a personal prior probability distribution with density

$$
p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta -1}}{B(\alpha, \beta)}
$$

where $B(\alpha, \beta)$ is a  **beta function** , so that $p(\theta)$ is
the density of a **beta distribution** with parameters $\alpha, \beta$.

We can update this prior after observing data using Bayes' Law (see {doc}`Probability with Matrices <prob_matrix>` for an introduction).

For a sample of $n$ coin flips that yields $k$ heads, the **likelihood function** is the binomial PMF $p(k \mid \theta)$ introduced above.

Applying Bayes' Law with our beta prior, the **posterior density** is

$$
p(\theta \mid k) = \frac{p(k \mid \theta) \cdot p(\theta)}{\int_0^1 p(k \mid \theta) \cdot p(\theta) \, d\theta}
$$

Because the beta prior is conjugate to the binomial likelihood, this integral evaluates to (the kernel of) another beta density, so that

$$
\theta \mid k \sim \textrm{Beta}(\alpha + k, \, \beta + n - k)
$$ (eq:beta_posterior)

The first exercise below asks you to derive this closed form.

```{exercise}
:label: pm_ex2

**a)**  Write down the **likelihood function** for a single coin flip with outcome $Y \in \{0, 1\}$.

**b)** Write down the **posterior** distribution for $\theta$ after observing that single flip.

**c)** Derive the closed-form posterior {eq}`eq:beta_posterior` for a sample of $n$ flips that yields $k$ heads.
```


```{solution-start} pm_ex2
:class: dropdown
```

**a)** The **likelihood function** for a single coin flip with outcome $Y \in \{0, 1\}$ is

$$
p(Y \mid \theta) = \theta^Y (1-\theta)^{1-Y}
$$

**b)** By Bayes' Law, the posterior density for $\theta$ after observing a single flip $Y$ is

$$
p(\theta \mid Y) = \frac{p(Y \mid \theta) \cdot p(\theta)}{\int_{0}^{1} p(Y \mid \theta) \cdot p(\theta) \, d\theta}
$$

Substituting the likelihood from (a) and the beta prior density, this becomes

$$
p(\theta \mid Y) = \frac{\theta^Y (1-\theta)^{1-Y} \cdot \theta^{\alpha - 1} (1 - \theta)^{\beta - 1} / B(\alpha, \beta)}{\int_{0}^{1} \theta^Y (1-\theta)^{1-Y} \cdot \theta^{\alpha - 1} (1 - \theta)^{\beta - 1} / B(\alpha, \beta) \, d\theta}
$$

Collecting powers of $\theta$ and $(1-\theta)$, we recognize the kernel of a beta density:

$$
p(\theta \mid Y) = \frac{\theta^{Y+\alpha - 1} (1 - \theta)^{1-Y+\beta - 1}}{\int_{0}^{1} \theta^{Y+\alpha - 1} (1 - \theta)^{1-Y+\beta - 1} \, d\theta}
$$

which means that

$$
\theta \mid Y \sim \textrm{Beta}(\alpha + Y, \, \beta + (1-Y))
$$

**c)** The same calculation, with the binomial likelihood in place of the Bernoulli likelihood, generalizes the result to a sample of $n$ flips that yields $k$ heads.

The beta prior contributes the factor $\theta^{\alpha-1}(1-\theta)^{\beta-1}$, and the binomial likelihood contributes $\theta^{k}(1-\theta)^{n-k}$, so the posterior is proportional to

$$
\theta^{\alpha + k - 1} (1-\theta)^{\beta + n - k - 1},
$$

which is the kernel of a beta density. Hence

$$
\theta \mid k \sim \textrm{Beta}(\alpha + k, \, \beta + n - k),
$$

as stated in {eq}`eq:beta_posterior`.

```{solution-end}
```

The next exercise puts this posterior to work.

```{exercise}
:label: pm_ex3

**a)** Now pretend that the true value of $\theta = 0.4$ and that someone who doesn't know this has a beta prior distribution with parameters $\beta = \alpha = 0.5$. Write Python code to simulate this person's personal posterior distribution for $\theta$ for a _single_ sequence of $n$ draws.

**b)** Plot the posterior distribution for $\theta$ as a function of $\theta$ as $n$ grows as $1, 2, \ldots$.

**c)** For various $n$'s, describe and compute a Bayesian coverage interval for the interval $[0.45, 0.55]$.

**d)** Tell what question a Bayesian coverage interval answers.

**e)** Compute the posterior probability that $\theta \in [0.45, 0.55]$ for various values of sample size $n$.

**f)** Use your Python code to study what happens to the posterior distribution as $n \rightarrow + \infty$, again assuming that the true value of $\theta = 0.4$, though it is unknown to the person doing the updating via Bayes' Law.
```

```{solution-start} pm_ex3
:class: dropdown
```

**a)**

We use one function to simulate a sequence of coin flips and another to form the
Beta posterior from the first `n_obs` of those flips.

```{code-cell} ipython3
def simulate_flips(θ=0.4, n=1_000_000, seed=1234):
    "Simulate n coin flips, each landing heads (1) with probability θ."
    rng = np.random.default_rng(seed)
    return (rng.random(n) < θ).astype(int)
```

```{code-cell} ipython3
def form_posterior(draws, n_obs, α=0.5, β=0.5):
    "Beta posterior for θ from the first n_obs flips, given a Beta(α, β) prior."
    heads = draws[:n_obs].sum()
    return st.beta(α + heads, β + n_obs - heads)
```

**b)**

```{code-cell} ipython3
draws = simulate_flips()

n_obs_list = [1, 2, 3, 4, 5, 10, 20, 50,
              100, 1000,
              5000, 10_000, 50_000, 100_000,
              200_000, 300_000]

posterior_list = [form_posterior(draws, n_obs) for n_obs in n_obs_list]

θ_values = np.linspace(0.01, 1, 1000)

fig, ax = plt.subplots(figsize=(10, 6))

prior = st.beta(0.5, 0.5)
ax.plot(θ_values, prior.pdf(θ_values),
        label='n = 0 (prior)', linestyle='--')

for i, n_obs in enumerate(n_obs_list[:10]):
    posterior = posterior_list[i]
    ax.plot(θ_values, posterior.pdf(θ_values),
            label=f'n = {n_obs}')

ax.set_title('PDF of posterior distributions',
             fontsize=15)
ax.set_xlabel(r"$\theta$", fontsize=15)

ax.legend(fontsize=11)
plt.show()
```

**c)**

```{code-cell} ipython3
lower_bound = [post.ppf(0.05) for post in posterior_list[:10]]
upper_bound = [post.ppf(0.95) for post in posterior_list[:10]]

interval_df = pd.DataFrame()
interval_df['upper'] = upper_bound
interval_df['lower'] = lower_bound
interval_df.index = n_obs_list[:10]
interval_df = interval_df.T
interval_df
```

As $n$ increases, we can see that Bayesian coverage intervals narrow and move toward $0.4$.

**d)** The Bayesian coverage interval tells the range of $\theta$ that corresponds to the [$q_1$, $q_2$] quantiles of the cumulative distribution function (CDF) of the posterior distribution.

To construct the coverage interval we first compute a posterior distribution of the unknown parameter $\theta$.

If the CDF is $F(\theta)$, then the Bayesian coverage interval $[a,b]$ for the interval $[q_1,q_2]$ is described by

$$
F(a)=q_1,F(b)=q_2
$$

**e)**

```{code-cell} ipython3
left_value, right_value = 0.45, 0.55

posterior_prob_list = [
    post.cdf(right_value) - post.cdf(left_value)
    for post in posterior_list
]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(posterior_prob_list)
ax.set_title(
    r'posterior probability that $\theta$'
    f' ranges from {left_value:.2f}'
    f' to {right_value:.2f}',
    fontsize=13)
ax.set_xticks(np.arange(0, len(posterior_prob_list), 3))
ax.set_xticklabels(n_obs_list[::3])
ax.set_xlabel('number of observations', fontsize=11)

plt.show()
```

Notice that in the graph above the posterior probability that $\theta \in [0.45, 0.55]$ exhibits a hump shape as $n$ increases.

Two opposing forces are at work.

The first force is that the individual  adjusts his belief as he observes new outcomes, so his posterior probability distribution  becomes more and more realistic, which explains the rise of the posterior probability.

However, $[0.45, 0.55]$ actually excludes the true $\theta = 0.4$ that generates the data.

As a result, the posterior probability drops as larger and larger samples refine his  posterior probability distribution of $\theta$.

The descent seems precipitous only because of the scale of the graph  that has the number of observations increasing disproportionately.

When the number of observations becomes large enough, our Bayesian becomes so confident about $\theta$ that he considers $\theta \in [0.45, 0.55]$ very unlikely.

That is why we see a nearly horizontal line when the number of observations exceeds 1000.

**f)** Using the functions we wrote above, we can see the evolution of posterior distributions as $n$ approaches infinity.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))

for i, n_obs in enumerate(n_obs_list[10:]):
    posterior = posterior_list[i + 10]
    ax.plot(θ_values, posterior.pdf(θ_values),
            label=f'n = {n_obs:,}')

ax.set_title('PDF of posterior distributions', fontsize=15)
ax.set_xlabel(r"$\theta$", fontsize=15)
ax.set_xlim(0.3, 0.5)

ax.legend(fontsize=11)
plt.show()
```

As $n$ increases, we can see that the probability density functions _concentrate_ on $0.4$, the true value of $\theta$.

Here the  posterior mean  converges to $0.4$ while the posterior standard deviation converges to $0$ from above.

To show this, we compute the mean and standard deviation of the posterior distributions.

```{code-cell} ipython3
mean_list = [post.mean() for post in posterior_list]
std_list = [post.std() for post in posterior_list]

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(mean_list)
ax[0].set_title('mean of posterior distribution',
                fontsize=13)
ax[0].set_xticks(np.arange(0, len(mean_list), 3))
ax[0].set_xticklabels(n_obs_list[::3])
ax[0].set_xlabel('number of observations', fontsize=11)

ax[1].plot(std_list)
ax[1].set_title('std dev of posterior distribution',
                fontsize=13)
ax[1].set_xticks(np.arange(0, len(std_list), 3))
ax[1].set_xticklabels(n_obs_list[::3])
ax[1].set_xlabel('number of observations', fontsize=11)

plt.show()
```

```{solution-end}
```

How shall we interpret the patterns above?

The answer is encoded in the Bayesian updating formula derived above.

Recall that after observing $k$ heads in $n$ flips, the posterior is $\textrm{Beta}(\alpha + k, \, \beta + n - k)$.

A beta distribution with parameters $\alpha$ and $\beta$ has

* mean $\frac{\alpha}{\alpha + \beta}$

* variance $\frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$

Here $\alpha + k$ can be viewed as the number of successes (prior pseudo-count plus observed heads) and $\beta + n - k$ as the number of failures.

Since the data are generated with $\theta = 0.4$, the Law of Large Numbers tells us that, as $n$ grows, $k/n \to 0.4$ (see {ref}`pm_ex1`).

Consequently, the posterior mean converges to $0.4$ and the posterior variance shrinks to zero.

```{code-cell} ipython3
upper_bound = [post.ppf(0.95) for post in posterior_list]
lower_bound = [post.ppf(0.05) for post in posterior_list]

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(np.arange(len(upper_bound)),
           upper_bound, label='95th quantile')
ax.scatter(np.arange(len(lower_bound)),
           lower_bound, label='5th quantile')

ax.set_xticks(np.arange(0, len(upper_bound), 2))
ax.set_xticklabels(n_obs_list[::2])
ax.set_xlabel('number of observations', fontsize=12)
ax.set_title('Bayesian coverage intervals of '
             'posterior distributions', fontsize=15)

ax.legend(fontsize=11)
plt.show()
```

After observing a large number of outcomes, the  posterior distribution collapses around $0.4$.

Thus, the Bayesian statistician  comes to believe that $\theta$ is near $0.4$.

As shown in the figure above, as the number of observations grows, the Bayesian coverage intervals (BCIs) become narrower and narrower   around  $0.4$.

However, if you take a closer look, you will find that the centers of  the BCIs are not exactly $0.4$, due to the persistent influence of the prior distribution and the randomness of the simulation path.


## Role of a Conjugate Prior

We have made  assumptions that link functional forms of  our likelihood function and our prior in a way that has eased our calculations considerably.

In particular, our assumptions that the likelihood function is **binomial** and that the prior distribution is a **beta distribution** have the consequence that the posterior distribution implied by Bayes' Law is also a **beta distribution**.

So posterior and prior are both beta distributions, albeit ones with different parameters.

When a likelihood function and prior fit together like hand and glove in this way, we can  say that the  prior and posterior are **conjugate distributions**.

In this situation, we also sometimes  say that we have **conjugate prior** for the likelihood function $p(k \mid \theta)$.

Typically, the functional form of the likelihood function determines the functional form of a **conjugate prior**.

A natural question to ask is why should a person's personal prior about a parameter $\theta$ be restricted to be described by a conjugate prior?

Why not some other functional form that more sincerely describes the person's beliefs?

To be argumentative, one could ask, why should the form of the likelihood function have *anything* to say about my personal beliefs about $\theta$?

A dignified response to that question is, well, it shouldn't, but if you want to compute a posterior easily you'll just be happier if your prior is conjugate to your likelihood.

Otherwise, your posterior won't have a convenient analytical form and you'll be in the situation of wanting to apply the Markov chain Monte Carlo techniques deployed in {doc}`Non-Conjugate Priors <bayes_nonconj>`.

We also apply these powerful methods to approximating Bayesian posteriors for non-conjugate priors in
{doc}`Posterior Distributions for AR(1) Parameters <ar1_bayes>` and {doc}`Forecasting an AR(1) Process <ar1_turningpts>`.
