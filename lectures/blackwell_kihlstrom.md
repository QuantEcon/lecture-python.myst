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

(blackwell_kihlstrom)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Blackwell's Theorem on Comparing Experiments

```{contents} Contents
:depth: 2
```

## Overview

This lecture explains **Blackwell's theorem** {cite}`blackwell1951,blackwell1953` on ranking statistical
experiments, following the Bayesian exposition of {cite}`kihlstrom1984`.

Consider two random variables, $\tilde{x}_\mu$ and $\tilde{x}_\nu$, each correlated
with an unknown state $\tilde{s}$.  A decision maker wants to know which observation
conveys more information about $\tilde{s}$.

Blackwell identified a clean answer: $\tilde{x}_\mu$ is **at least as informative** as
$\tilde{x}_\nu$ if and only if any decision maker who observes $\tilde{x}_\mu$ can do
at least as well (in expected utility) as one who observes $\tilde{x}_\nu$.

Remarkably, this economic criterion is equivalent to two purely statistical ones:

- **Sufficiency** (Blackwell): $\tilde{x}_\mu$ is sufficient for $\tilde{x}_\nu$ — the
  distribution of $\tilde{x}_\nu$ can be reproduced by passing $\tilde{x}_\mu$ through
  a randomisation.
- **Uncertainty reduction** (DeGroot {cite}`degroot1962`): $\tilde{x}_\mu$ reduces
  every concave measure of uncertainty at least as much as $\tilde{x}_\nu$ does.

Kihlstrom's Bayesian restatement places the **posterior distribution** at the centre.
A more informative experiment creates a more dispersed distribution of posteriors — a
**mean-preserving spread** — which links Blackwell's ordering directly to
**second-order stochastic dominance** on the simplex of beliefs.

We proceed in the following steps:

1. Set up notation and define experiments as Markov matrices.
2. Define stochastic transformations (Markov kernels).
3. State the three equivalent criteria.
4. State and sketch the proof of the main theorem.
5. Develop the Bayesian interpretation via standard experiments and mean-preserving
   spreads.
6. Illustrate each idea with Python simulations.

Let's start by importing some tools.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.stats import dirichlet, beta as beta_dist
from scipy.optimize import minimize
from itertools import product

np.random.seed(42)
```

---

## Experiments and Markov Matrices

### The state space and experiments

Let $S = \{s_1, \ldots, s_N\}$ be a finite set of possible states of the world.

An **experiment** is described by the conditional distribution of an observed signal
$\tilde{x}$ given the state $\tilde{s}$.

When the signal space is also finite, say $X = \{x_1, \ldots, x_M\}$, an experiment
reduces to an $N \times M$ **Markov matrix**

$$
\mu = [\mu_{ij}], \qquad
\mu_{ij} = \Pr(\tilde{x}_\mu = x_j \mid \tilde{s} = s_i) \geq 0,
\quad \sum_{j=1}^{M} \mu_{ij} = 1 \;\forall\, i.
$$

Each row $i$ gives the distribution of signals when the true state is $s_i$.

```{code-cell} ipython3
# Example: two states, three signals
# mu[i, j] = Pr(signal j | state i)
mu = np.array([[0.6, 0.3, 0.1],   # state 1: signal is quite informative
               [0.1, 0.3, 0.6]])   # state 2: opposite pattern

nu = np.array([[0.5, 0.2, 0.3],   # coarser experiment
               [0.2, 0.5, 0.3]])

print("Experiment μ (rows sum to 1):")
print(mu)
print("\nExperiment ν:")
print(nu)
print("\nRow sums μ:", mu.sum(axis=1))
print("Row sums ν:", nu.sum(axis=1))
```

### Stochastic transformations (Markov kernels)

A **stochastic transformation** $Q$ maps signals of one experiment to signals of
another by a further randomisation.

In the discrete setting with $M$ input signals and $K$ output signals, $Q$ is an
$M \times K$ Markov matrix: $q_{lk} \geq 0$ and $\sum_k q_{lk} = 1$ for every row $l$.

```{admonition} Definition (Sufficiency)
:class: tip
Experiment $\mu$ is **sufficient for** $\nu$ if there exists a stochastic
transformation $Q$ (an $M \times K$ Markov matrix) such that

$$
\nu = \mu \, Q,
$$

meaning that an observer of $\tilde{x}_\mu$ can generate the distribution of
$\tilde{x}_\nu$ by passing their signal through $Q$.
```

The intuition: if you hold the more informative signal $\tilde{x}_\mu$, you can always
*throw away* information to produce a signal distributed like $\tilde{x}_\nu$;
the reverse is impossible.

```{code-cell} ipython3
def is_markov(M, tol=1e-10):
    """Check whether a matrix is a valid Markov (row-stochastic) matrix."""
    return np.all(M >= -tol) and np.allclose(M.sum(axis=1), 1.0)

def find_stochastic_transform(mu, nu, tol=1e-8):
    """
    Try to find Q such that nu ≈ mu @ Q using least-squares then project.
    Returns Q and the residual ||nu - mu @ Q||.
    This is a simple demonstration for the discrete finite case.
    """
    N, M = mu.shape
    _, K = nu.shape

    # Solve nu = mu @ Q column by column using non-negative least squares
    from scipy.optimize import lsq_linear

    Q = np.zeros((M, K))
    for k in range(K):
        b = nu[:, k]
        # Constraints: Q[:, k] >= 0, sum(Q[:, k]) = 1
        # Use lsq_linear with bounds, ignoring sum constraint for now
        result = lsq_linear(mu, b, bounds=(0, np.inf))
        Q[:, k] = result.x

    # Normalise rows so Q is row-stochastic
    row_sums = Q.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    Q = Q / row_sums
    residual = np.linalg.norm(nu - mu @ Q)
    return Q, residual

# Build a ν that is deliberately a garbling of μ
# Q maps 3 signals -> 2 signals (merge signals 2&3)
Q_true = np.array([[1.0, 0.0],
                   [0.0, 1.0],
                   [0.0, 1.0]])

nu_garbled = mu @ Q_true
print("ν = μ @ Q_true:")
print(nu_garbled)
print("ν is Markov:", is_markov(nu_garbled))

Q_found, res = find_stochastic_transform(mu, nu_garbled)
print(f"\nRecovered Q (residual = {res:.2e}):")
print(np.round(Q_found, 4))
print("Rows of Q sum to:", Q_found.sum(axis=1).round(4))
```

---

## Three Equivalent Criteria for "More Informative"

### Criterion 1 — The Economic Criterion

Let $A$ be a compact convex set of actions and $u: A \times S \to \mathbb{R}$ a
bounded utility function.

A decision maker observes $x \in X$, applies Bayes' rule to update beliefs about
$\tilde{s}$, and chooses $d(x) \in A$ to maximise expected utility.

Let $p = (p_1, \ldots, p_N)$ be the prior over states, and write

$$
P = \bigl\{(p_1, \ldots, p_N) : p_i \geq 0,\; \textstyle\sum_i p_i = 1\bigr\}
$$

for the probability simplex.

The set of **achievable expected-utility vectors** under experiment $\mu$ is

$$
B(\mu, A) = \Bigl\{v \in \mathbb{R}^N :
  v_i = \textstyle\int_X u(f(x), s_i)\,\mu_i(dx)
  \text{ for some measurable } f: X \to A \Bigr\}.
$$

```{admonition} Definition (Economic Criterion — Bonnenblust–Shapley–Sherman)
:class: tip
$\mu$ is **at least as informative as** $\nu$ in the economic sense if

$$
B(\mu, A) \supseteq B(\nu, A)
$$

for every compact convex action set $A$ and every prior $p \in P$.
```

Equivalently, every rational decision maker weakly prefers to observe $\tilde{x}_\mu$
over $\tilde{x}_\nu$.

### Criterion 2 — The Sufficiency Criterion (Blackwell)

```{admonition} Definition (Blackwell Sufficiency)
:class: tip
$\mu \geq \nu$ in Blackwell's sense if there exists a stochastic transformation
$Q: X \to Y$ such that

$$
\nu_i(E) = (Q \circ \mu_i)(E)
\quad \forall\, E \in \mathscr{G},\; i = 1, \ldots, N.
$$
```

In matrix notation for finite experiments: $\nu = \mu \, Q$.

### Criterion 3 — The Uncertainty-Function Criterion (DeGroot)

{cite}`degroot1962` calls any **concave** function $U: P \to \mathbb{R}$ an
**uncertainty function**.

The prototypical example is Shannon entropy:

$$
U(p) = -\sum_{i=1}^{N} p_i \log p_i.
$$

```{admonition} Definition (DeGroot Uncertainty Criterion)
:class: tip
$\mu$ **reduces expected uncertainty at least as much as** $\nu$ if, for every
concave $U: P \to \mathbb{R}$,

$$
\int_P U(p)\,\hat\mu^c(dp)
\;\leq\;
\int_P U(p)\,\hat\nu^c(dp),
$$

where $\hat\mu^c$ is the distribution of the posterior induced by experiment $\mu$
starting from the uniform prior $c = (1/N, \ldots, 1/N)$.
```

Jensen's inequality guarantees that observing any signal *weakly* reduces expected
uncertainty ($\int U(p^\mu)\,d\hat\mu^c \leq U(c)$).  The criterion asks whether $\mu$
always reduces it *at least as much* as $\nu$.

---

## The Main Theorem

```{admonition} Theorem (Blackwell 1951, 1953; Bonnenblust et al. 1949; DeGroot 1962)
:class: important
The following three conditions are equivalent:

(i) **Economic criterion:** $B(\mu, A) \supseteq B(\nu, A)$ for every compact
    convex $A$ and every prior $p \in P$.

(ii) **Sufficiency criterion:** There exists a stochastic transformation $Q$ from $X$
     to $Y$ such that $\nu = Q \circ \mu$.

(iii) **Uncertainty criterion:** $\int_P U(p)\,\hat\mu^c(dp) \leq \int_P U(p)\,\hat\nu^c(dp)$
      for every concave $U$ and the uniform prior $c$.
```

The proof establishes the chain
(i) $\Leftrightarrow$ (ii) $\Leftrightarrow$ (iii).

**Sketch (ii $\Rightarrow$ i):** If $\nu = \mu Q$, any decision rule $f$ for $\tilde{x}_\nu$
can be replicated by first observing $\tilde{x}_\mu$, drawing $\tilde{x}_\nu \sim Q(\tilde{x}_\mu, \cdot)$,
then applying $f$.  Hence $B(\nu, A) \subseteq B(\mu, A)$.

**Sketch (i $\Rightarrow$ ii):** This uses a separating-hyperplane argument.  Since
$B(\mu, A) \supseteq B(\nu, A)$ for every $A$, standard duality implies the existence
of a mean-preserving stochastic transformation $D$ mapping posteriors of $\nu$ to
posteriors of $\mu$, which constructs the required $Q$.

**Sketch (ii $\Leftrightarrow$ iii):** Given $Q$, Jensen's inequality applied to any
concave $U$ gives $\mathbb{E}[U(p^\mu)] \leq \mathbb{E}[U(p^\nu)]$.  The converse —
that the condition for all concave $U$ forces the existence of $Q$ — is proved in
{cite}`blackwell1953`.

---

## Kihlstrom's Bayesian Interpretation

### Posteriors and standard experiments

The central object in Kihlstrom's analysis is the **posterior belief vector**.

When prior $p$ holds and experiment $\mu$ produces signal $x$, Bayes' rule gives

$$
p_i^\mu(x) = \Pr(\tilde{s} = s_i \mid \tilde{x}_\mu = x)
= \frac{\mu_{ix} \, p_i}{\sum_j \mu_{jx}\, p_j}, \qquad i = 1, \ldots, N.
$$

The posterior $p^\mu(x) \in P$ is a *random variable* on the simplex.

```{admonition} Key property (mean preservation)
:class: note
The prior $p$ is the expectation of the posterior:

$$
\mathbb{E}[p^\mu] = \sum_x \Pr(\tilde{x}_\mu = x)\, p^\mu(x) = p.
$$

This is sometimes called the **law of iterated expectations for beliefs**.
```

The **standard experiment** ${}^c\mu^*$ records only the posterior: it maps the
prior $c$ to the random variable $p^\mu(x) \in P$.  Its distribution $\hat\mu^c$
on $P$ satisfies $\int_P p\;\hat\mu^c(dp) = c$.

Two experiments are **informationally equivalent** when they induce the same
distribution of posteriors.  The standard experiment is the minimal sufficient
statistic for $\mu$.

```{code-cell} ipython3
def compute_posteriors(mu, prior):
    """
    Compute the posterior distribution for each signal realisation.

    Parameters
    ----------
    mu    : (N, M) Markov matrix — mu[i, j] = Pr(signal j | state i)
    prior : (N,) prior probabilities over states

    Returns
    -------
    posteriors : (M, N) array — posteriors[j] = posterior given signal j
    signal_probs : (M,) marginal probability of each signal
    """
    N, M = mu.shape
    # Marginal probability of each signal: Pr(x_j) = sum_i Pr(x_j|s_i)*p_i
    signal_probs = mu.T @ prior          # shape (M,)
    # Posterior: p(s_i | x_j) = mu[i,j]*p[i] / Pr(x_j)
    posteriors = (mu.T * prior) / signal_probs[:, None]  # shape (M, N)
    return posteriors, signal_probs


def check_mean_preservation(posteriors, signal_probs, prior):
    """Verify E[posterior] == prior."""
    expected_posterior = (posteriors * signal_probs[:, None]).sum(axis=0)
    return expected_posterior, np.allclose(expected_posterior, prior)


# Two-state example: states s1, s2
N = 2
prior = np.array([0.5, 0.5])

# More informative experiment μ
mu_info = np.array([[0.8, 0.2],
                    [0.2, 0.8]])

# Less informative experiment ν
nu_info = np.array([[0.6, 0.4],
                    [0.4, 0.6]])

post_mu, probs_mu = compute_posteriors(mu_info, prior)
post_nu, probs_nu = compute_posteriors(nu_info, prior)

print("=== Experiment μ (more informative) ===")
print("Signal probabilities:", probs_mu.round(3))
print("Posteriors (row = signal, col = state):")
print(post_mu.round(3))
mean_mu, ok_mu = check_mean_preservation(post_mu, probs_mu, prior)
print(f"E[posterior] = {mean_mu.round(4)}  (equals prior: {ok_mu})")

print("\n=== Experiment ν (less informative) ===")
print("Signal probabilities:", probs_nu.round(3))
print("Posteriors:")
print(post_nu.round(3))
mean_nu, ok_nu = check_mean_preservation(post_nu, probs_nu, prior)
print(f"E[posterior] = {mean_nu.round(4)}  (equals prior: {ok_nu})")
```

### Visualising posterior distributions on the simplex

For $N = 2$ states, the simplex $P$ is the unit interval $[0, 1]$ (the probability
of state $s_1$).  We can directly plot the distribution of posteriors under
experiments $\mu$ and $\nu$.

```{code-cell} ipython3
def plot_posterior_distributions(mu_matrix, nu_matrix, prior,
                                 labels=("μ (more informative)",
                                         "ν (less informative)")):
    """
    For a two-state experiment, plot the distribution of posteriors
    (i.e., the standard experiment distribution) on [0,1].
    """
    posts_mu, probs_mu = compute_posteriors(mu_matrix, prior)
    posts_nu, probs_nu = compute_posteriors(nu_matrix, prior)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    prior_val = prior[0]

    for ax, posts, probs, label in zip(
            axes, [posts_mu, posts_nu], [probs_mu, probs_nu], labels):
        p_s1 = posts[:, 0]          # posterior prob of state s1 for each signal
        ax.vlines(p_s1, 0, probs, linewidth=6, color="steelblue", alpha=0.7)
        ax.axvline(prior_val, color="tomato", linestyle="--", linewidth=1.5,
                   label=f"prior = {prior_val:.2f}")
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"Posterior $p(s_1 \mid x)$", fontsize=12)
        ax.set_ylabel("Probability mass", fontsize=12)
        ax.set_title(label, fontsize=12)
        ax.legend()
        # Annotate mean
        mean_post = (p_s1 * probs).sum()
        ax.axvline(mean_post, color="green", linestyle=":", linewidth=1.5,
                   label=f"E[post] = {mean_post:.2f}")
        ax.legend()

    fig.suptitle("Distribution of posteriors (standard experiment)\n"
                 "More informative → more dispersed from prior",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()

plot_posterior_distributions(mu_info, nu_info, prior)
```

The more informative experiment $\mu$ pushes posteriors further from the prior in
both directions, producing a more dispersed distribution on $[0,1]$.

### Mean-preserving spreads and Blackwell's ordering

Kihlstrom's key reformulation is:

```{admonition} Theorem (Kihlstrom's Reformulation)
:class: important
$\mu \geq \nu$ in Blackwell's sense **if and only if** $\hat\mu^c$ is a
**mean-preserving spread** of $\hat\nu^c$; that is,

$$
\int_P g(p)\,\hat\mu^c(dp) \;\geq\; \int_P g(p)\,\hat\nu^c(dp)
$$

for every **convex** function $g: P \to \mathbb{R}$.
```

Equivalently, $\hat\mu^c$ **second-order stochastically dominates** $\hat\nu^c$
(in the sense of mean-preserving spreads).

The intuition: a better experiment resolves more uncertainty, spreading posteriors
further from the prior on average.  Any convex $g$ assigns higher expected value to
a more dispersed distribution (Jensen's inequality in reverse).

```{code-cell} ipython3
def check_mps_convex_functions(mu_matrix, nu_matrix, prior, n_functions=200):
    """
    Verify the mean-preserving spread condition for random convex functions.

    We test:  E[g(p^μ)] >= E[g(p^ν)] for convex g
    using a family of convex functions g(p) = (p - t)^+ = max(p-t, 0).
    """
    posts_mu, probs_mu = compute_posteriors(mu_matrix, prior)
    posts_nu, probs_nu = compute_posteriors(nu_matrix, prior)

    p_mu = posts_mu[:, 0]   # posteriors on s1
    p_nu = posts_nu[:, 0]

    thresholds = np.linspace(0, 1, n_functions)
    diffs = []
    for t in thresholds:
        Eg_mu = (np.maximum(p_mu - t, 0) * probs_mu).sum()
        Eg_nu = (np.maximum(p_nu - t, 0) * probs_nu).sum()
        diffs.append(Eg_mu - Eg_nu)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, diffs, color="steelblue", linewidth=2)
    ax.axhline(0, color="tomato", linestyle="--")
    ax.fill_between(thresholds, diffs, 0,
                    where=np.array(diffs) >= 0,
                    alpha=0.25, color="steelblue",
                    label=r"$E[g(p^\mu)] - E[g(p^\nu)] \geq 0$")
    ax.set_xlabel("Threshold $t$", fontsize=12)
    ax.set_ylabel(r"$E[\max(p-t,0)]$ difference", fontsize=12)
    ax.set_title(
        r"Mean-preserving spread check: $E[g(p^\mu)] \geq E[g(p^\nu)]$"
        "\nfor convex functions $g(p) = \max(p - t, 0)$",
        fontsize=11)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    all_non_negative = all(d >= -1e-10 for d in diffs)
    print(f"μ is a mean-preserving spread of ν: {all_non_negative}")
    return diffs

_ = check_mps_convex_functions(mu_info, nu_info, prior)
```

---

## Simulating the Blackwell Ordering with Many States

To move beyond two states we simulate richer experiments.

We take $N = 3$ states and compare a more-informative experiment $\mu$ (whose
signal is strongly correlated with the state) against a less-informative $\nu$
(a garbling of $\mu$).

```{code-cell} ipython3
# Three states, three signals
N3 = 3
prior3 = np.array([1/3, 1/3, 1/3])

# Informative experiment: strong diagonal
mu3 = np.array([[0.7, 0.2, 0.1],
                [0.1, 0.7, 0.2],
                [0.2, 0.1, 0.7]])

# Garbling matrix: merge signals 2 and 3
Q3 = np.array([[0.9, 0.05, 0.05],
               [0.05, 0.8, 0.15],
               [0.05, 0.15, 0.8]])   # row-stochastic

nu3 = mu3 @ Q3

print("μ (3×3):")
print(np.round(mu3, 2))
print("\nQ (garbling):")
print(np.round(Q3, 2))
print("\nν = μ @ Q:")
print(np.round(nu3, 3))
```

### Plotting posterior clouds on the 2-simplex

For three states, posteriors live in a 2-simplex (a triangle).  We draw many
samples from each experiment and plot where the posteriors land.

```{code-cell} ipython3
def sample_posteriors(mu_matrix, prior, n_draws=3000):
    """
    Simulate n_draws observations from the experiment and compute
    the resulting posterior beliefs.
    Returns array of shape (n_draws, N).
    """
    N, M = mu_matrix.shape
    # Draw a state
    states = np.random.choice(N, size=n_draws, p=prior)
    # Draw a signal conditioned on the state
    signals = np.array([np.random.choice(M, p=mu_matrix[s]) for s in states])
    # Compute posterior
    posteriors, signal_probs = compute_posteriors(mu_matrix, prior)
    return posteriors[signals]       # shape (n_draws, N)


def simplex_to_cart(pts):
    """Convert 3-simplex barycentric coordinates to 2-D Cartesian."""
    corners = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.5, np.sqrt(3)/2]])
    return pts @ corners


def plot_simplex_posteriors(mu_matrix, nu_matrix, prior3, n_draws=3000):
    posts_mu = sample_posteriors(mu_matrix, prior3, n_draws)
    posts_nu = sample_posteriors(nu_matrix, prior3, n_draws)

    cart_mu = simplex_to_cart(posts_mu)
    cart_nu = simplex_to_cart(posts_nu)
    prior_cart = simplex_to_cart(prior3[None, :])[0]

    corners = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.5, np.sqrt(3)/2]])
    triangle = plt.Polygon(corners, fill=False, edgecolor="black", linewidth=1.5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["μ  (more informative)", "ν  (less informative / garbled)"]
    data = [(cart_mu, "steelblue"), (cart_nu, "darkorange")]
    labels = ["$s_1$", "$s_2$", "$s_3$"]
    offsets = [(-0.07, -0.05), (1.02, -0.05), (0.48, np.sqrt(3)/2 + 0.03)]

    for ax, (cart, c), title in zip(axes, data, titles):
        tri = plt.Polygon(corners, fill=False, edgecolor="black", linewidth=1.5)
        ax.add_patch(tri)
        ax.scatter(cart[:, 0], cart[:, 1], s=4, alpha=0.25, color=c)
        ax.scatter(*prior_cart, s=120, color="red", zorder=5,
                   label="prior", marker="*")
        for i, (lbl, off) in enumerate(zip(labels, offsets)):
            ax.text(corners[i][0] + off[0], corners[i][1] + off[1],
                    lbl, fontsize=13)
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=11, loc="upper right")

    fig.suptitle("Posterior clouds on the 2-simplex\n"
                 "More informative experiment → posteriors spread further from prior",
                 fontsize=12)
    plt.tight_layout()
    plt.show()

plot_simplex_posteriors(mu3, nu3, prior3)
```

The posteriors under $\mu$ cluster near the vertices (near-certain beliefs), while
those under the garbled $\nu$ cluster closer to the centre (the prior).

---

## The DeGroot Uncertainty Function

### Concave uncertainty functions and the value of information

{cite}`degroot1962` formalises the value of information through an **uncertainty function**
$U: P \to \mathbb{R}$ that must be:

- **Concave**: by Jensen, observing any signal weakly reduces expected uncertainty.
- **Symmetric**: depends on the components of $p$, not their labelling.
- **Normalised**: maximised at $p = (1/N, \ldots, 1/N)$ and minimised at vertices.

The **value of experiment $\mu$ given prior $p$** is

$$
I(\tilde{x}_\mu;\, \tilde{s};\, U)
= U(p) - \mathbb{E}[U(p^\mu)],
$$

the expected reduction in uncertainty.  A key result is that $\mu \geq \nu$ **if and
only if** $I(\tilde{x}_\mu; \tilde{s}; U) \geq I(\tilde{x}_\nu; \tilde{s}; U)$ for
**every** concave $U$.

### Shannon entropy as a special case

The canonical uncertainty function is Shannon entropy

$$
U_H(p) = -\sum_{i=1}^{N} p_i \log p_i.
$$

Under the uniform prior $c = (1/N, \ldots, 1/N)$, DeGroot's value formula becomes

$$
I(\tilde{x}_\mu, c;\, U_H)
= \log N - H(\tilde{s} \mid \tilde{x}_\mu),
$$

where $H(\tilde{s} \mid \tilde{x}_\mu)$ is the conditional entropy of the state given
the signal — exactly the **mutual information** between $\tilde{x}_\mu$ and $\tilde{s}$.

```{note}
The Blackwell ordering implies the entropy-based inequality, but the *converse fails*:
entropy alone does not pin down the full Blackwell ordering — you need the inequality
for **every** concave $U$.
```

```{code-cell} ipython3
def entropy(p, eps=1e-12):
    """Shannon entropy of a probability vector."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))


def degroot_value(mu_matrix, prior, U_func):
    """
    Compute DeGroot's value of information I = U(prior) - E[U(posterior)].
    """
    posts, probs = compute_posteriors(mu_matrix, prior)
    prior_uncertainty = U_func(prior)
    expected_post_uncertainty = sum(
        probs[j] * U_func(posts[j]) for j in range(len(probs)))
    return prior_uncertainty - expected_post_uncertainty


# --- Several concave uncertainty functions ---
def gini_impurity(p):
    """Gini impurity: 1 - sum(p_i^2)."""
    return 1.0 - np.sum(np.asarray(p)**2)

def tsallis_entropy(p, q=2):
    """Tsallis entropy of order q (concave for q>1)."""
    p = np.clip(p, 1e-12, 1.0)
    return (1 - np.sum(p**q)) / (q - 1)

def min_entropy(p):
    """Min-entropy: -log(max(p))."""
    return -np.log(np.max(np.clip(p, 1e-12, 1.0)))

uncertainty_functions = {
    "Shannon entropy": entropy,
    "Gini impurity": gini_impurity,
    "Tsallis (q=2)": tsallis_entropy,
    "Min-entropy": min_entropy,
}

print(f"{'Uncertainty function':<22}  {'I(μ)':<10}  {'I(ν)':<10}  {'I(μ)≥I(ν)?'}")
print("-" * 58)
for name, U in uncertainty_functions.items():
    I_mu = degroot_value(mu_info, prior, U)
    I_nu = degroot_value(nu_info, prior, U)
    print(f"{name:<22}  {I_mu:<10.4f}  {I_nu:<10.4f}  {I_mu >= I_nu - 1e-10}")
```

As predicted by the theorem, $I(\mu) \geq I(\nu)$ for every concave uncertainty
function once we know $\mu \geq \nu$ in the Blackwell sense.

### Value of information as a function of experiment quality

We now parameterise a continuum of experiments that interpolate between the
completely uninformative experiment (signal is independent of the state) and the
perfectly informative one (signal perfectly reveals the state).

For $N = 2$ states, a natural family is

$$
\mu(\theta) = (1 - \theta) \cdot \tfrac{1}{2}\mathbf{1}\mathbf{1}^\top
             + \theta \cdot I_2,
\quad \theta \in [0, 1],
$$

where the first term is the completely mixed (uninformative) matrix and $I_2$ is the
identity (perfectly informative).

```{code-cell} ipython3
def make_experiment(theta, N=2):
    """
    Parameterised experiment: theta=0 is uninformative, theta=1 is perfect.
    mu(theta) = (1-theta)*(1/N)*ones + theta*I
    """
    return (1 - theta) * np.ones((N, N)) / N + theta * np.eye(N)


thetas = np.linspace(0, 1, 100)
prior2 = np.array([0.5, 0.5])

fig, ax = plt.subplots(figsize=(9, 4))
for name, U in uncertainty_functions.items():
    values = [degroot_value(make_experiment(t), prior2, U) for t in thetas]
    # Normalise to [0,1] for comparability across functions
    vmin, vmax = values[0], values[-1]
    normed = (np.array(values) - vmin) / (vmax - vmin + 1e-15)
    ax.plot(thetas, normed, label=name, linewidth=2)

ax.set_xlabel(r"Experiment quality $\theta$  (0 = uninformative, 1 = perfect)",
              fontsize=11)
ax.set_ylabel("Normalised value of information $I(\\mu(\\theta))$", fontsize=11)
ax.set_title("Value of information rises monotonically with experiment quality\n"
             "for every concave uncertainty function", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
```

Every concave uncertainty function assigns weakly higher value to a more informative
experiment — a graphical illustration of the equivalence (i) $\Leftrightarrow$ (iii).

---

## Connection to Second-Order Stochastic Dominance

The uncertainty-function representation makes the connection to **second-order
stochastic dominance (SOSD)** explicit.

Because $U$ is concave, $-U$ is convex, and the condition

$$
\mathbb{E}[U(p^\mu)] \leq \mathbb{E}[U(p^\nu)] \quad \text{for all concave } U
$$

is precisely the statement that $\hat\mu^c$ dominates $\hat\nu^c$ in the
**mean-preserving spread** sense on $P$.

The Blackwell ordering on *experiments* is therefore isomorphic to the SOSD ordering
on *distributions of posteriors*.

```{code-cell} ipython3
def lorenz_curve_1d(weights, values):
    """
    Compute the Lorenz-like CDF used for SOSD comparisons.
    Returns (sorted values, cumulative probability mass).
    """
    idx = np.argsort(values)
    sorted_vals = values[idx]
    sorted_wts  = weights[idx]
    cum_mass = np.cumsum(sorted_wts)
    return sorted_vals, cum_mass


def plot_sosd_posteriors(mu_matrix, nu_matrix, prior, title=""):
    """
    Plot the CDFs of the posterior-on-s1 distributions under mu and nu,
    and verify SOSD (mu dominates nu in the MPS sense).
    """
    posts_mu, probs_mu = compute_posteriors(mu_matrix, prior)
    posts_nu, probs_nu = compute_posteriors(nu_matrix, prior)

    p_mu = posts_mu[:, 0]
    p_nu = posts_nu[:, 0]

    sv_mu, cm_mu = lorenz_curve_1d(probs_mu, p_mu)
    sv_nu, cm_nu = lorenz_curve_1d(probs_nu, p_nu)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: CDFs
    ax = axes[0]
    for sv, cm, lbl, c in [(sv_mu, cm_mu, "μ", "steelblue"),
                            (sv_nu, cm_nu, "ν", "darkorange")]:
        xs = np.concatenate([[0], sv, [1]])
        ys = np.concatenate([[0], cm, [1]])
        ax.step(xs, ys, where="post", label=lbl, color=c, linewidth=2)
    ax.set_xlabel(r"Posterior $p(s_1 \mid x)$", fontsize=12)
    ax.set_ylabel("Cumulative probability", fontsize=12)
    ax.set_title("CDFs of posterior distributions", fontsize=11)
    ax.legend(fontsize=11)
    ax.axvline(prior[0], linestyle="--", color="gray", alpha=0.6,
               label="prior")

    # Right: integrated CDFs (SOSD criterion: F_nu integrates >= F_mu)
    ax2 = axes[1]
    grid = np.linspace(0, 1, 200)

    def integrated_cdf(sorted_vals, cum_mass, grid):
        # CDF at each grid point
        cdf = np.array([cum_mass[sorted_vals <= t].max()
                        if np.any(sorted_vals <= t) else 0.0
                        for t in grid])
        return np.cumsum(cdf) * (grid[1] - grid[0])

    int_mu = integrated_cdf(sv_mu, cm_mu, grid)
    int_nu = integrated_cdf(sv_nu, cm_nu, grid)

    ax2.plot(grid, int_mu, label="∫F_μ", color="steelblue", linewidth=2)
    ax2.plot(grid, int_nu, label="∫F_ν", color="darkorange", linewidth=2)
    ax2.fill_between(grid, int_mu, int_nu,
                     where=int_nu >= int_mu,
                     alpha=0.2, color="darkorange",
                     label="∫F_ν ≥ ∫F_μ  (μ MPS-dominates ν)")
    ax2.set_xlabel(r"$t$", fontsize=12)
    ax2.set_ylabel("Integrated CDF", fontsize=12)
    ax2.set_title("SOSD: integrated CDFs\n(μ dominates ν iff ∫F_ν ≥ ∫F_μ everywhere)",
                  fontsize=11)
    ax2.legend(fontsize=10)

    fig.suptitle(title or "Second-order stochastic dominance of posterior distributions",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()

plot_sosd_posteriors(mu_info, nu_info, prior,
                     title="μ is a mean-preserving spread of ν:\n"
                           "μ second-order stochastically dominates ν")
```

---

## The Stochastic Transformation as a Mean-Preserving Randomisation

Kihlstrom proves that (i) $\Rightarrow$ (ii) by explicit construction.

Given that $\mu$ achieves at least the value of $\nu$ for every user, he constructs
a stochastic transformation $D(p^0, \cdot)$ on $P$ that is **mean-preserving**:

$$
\int_P p\; D(p^0, dp) = p^0.
$$

Setting $Q = D$ provides the Markov kernel witnessing Blackwell sufficiency.

The mean-preservation condition says: passing $\tilde{x}_\mu$ through $Q$ to
produce a synthetic $\tilde{x}_\nu$ cannot add information — it only destroys it.

```{code-cell} ipython3
def verify_garbling_mean_preservation(mu_matrix, Q_matrix, prior):
    """
    Verify that the garbling Q is mean-preserving:
    E[posterior under ν] = E[posterior under μ].
    Both should equal the prior.
    """
    nu_matrix = mu_matrix @ Q_matrix
    posts_mu, probs_mu = compute_posteriors(mu_matrix, prior)
    posts_nu, probs_nu = compute_posteriors(nu_matrix, prior)

    mean_mu = (posts_mu * probs_mu[:, None]).sum(axis=0)
    mean_nu = (posts_nu * probs_nu[:, None]).sum(axis=0)

    print(f"Prior:               {prior.round(4)}")
    print(f"E[p^μ]:              {mean_mu.round(4)}")
    print(f"E[p^ν = p^(μQ)]:     {mean_nu.round(4)}")
    print(f"Both equal prior?    mu: {np.allclose(mean_mu, prior)}, "
          f"nu: {np.allclose(mean_nu, prior)}")


# Q_true maps 2 signals -> 2 signals (a softening garbling)
Q_soft = np.array([[0.7, 0.3],
                   [0.3, 0.7]])

verify_garbling_mean_preservation(mu_info, Q_soft, prior)
```

---

## Comparing Experiments: A Systematic Example

We now study a grid of experiments indexed by their quality parameter $\theta$
and verify that the Blackwell ordering is faithfully reflected in:

1.  The spread of posteriors (mean-preserving spread check).
2.  The value of information under every concave $U$.
3.  The SOSD ranking of posterior distributions.

```{code-cell} ipython3
thetas_grid = [0.1, 0.4, 0.7, 1.0]
prior2 = np.array([0.5, 0.5])

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flat

for ax, t in zip(axes, thetas_grid):
    mu_t = make_experiment(t)
    posts, probs = compute_posteriors(mu_t, prior2)
    p_s1 = posts[:, 0]
    ax.vlines(p_s1, 0, probs, linewidth=8, color="steelblue", alpha=0.7)
    ax.axvline(prior2[0], color="tomato", linestyle="--", linewidth=1.5,
               label=f"prior = {prior2[0]:.2f}")
    I_H   = degroot_value(mu_t, prior2, entropy)
    I_G   = degroot_value(mu_t, prior2, gini_impurity)
    ax.set_title(fr"$\theta = {t}$  |  $I_H = {I_H:.3f}$  |  $I_G = {I_G:.3f}$",
                 fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Posterior $p(s_1 \mid x)$", fontsize=11)
    ax.set_ylabel("Probability mass", fontsize=11)
    ax.legend(fontsize=10)

fig.suptitle("Distribution of posteriors for experiments of increasing quality\n"
             r"$\theta = 0$: uninformative; $\theta = 1$: perfect",
             fontsize=12)
plt.tight_layout()
plt.show()
```

As $\theta$ rises from 0 (unifomative) to 1 (perfect), posteriors migrate toward the
vertices $\{0, 1\}$, the value of information rises under every $U$, and the
distributions form a chain under the SOSD order.

---

## Application 1 — Product Quality Information (Kihlstrom 1974)

{cite}`kihlstrom1974a` applies Blackwell's theorem to consumer demand for information
about product quality.

- The unknown state $\tilde{s}$ is a product parameter $\theta$.
- A consumer can purchase $\lambda$ units of information at cost $c(\lambda)$.
- As $\lambda$ rises, the experiment becomes more informative in the Blackwell sense.

The Blackwell ordering certifies that "more information is always better" for every
expected-utility maximiser when information is free.

The consumer's demand for information equates the *marginal value of the standard
experiment* to its *marginal cost*.

```{code-cell} ipython3
def consumer_value(theta, prior2, U=entropy, cost_per_unit=0.5):
    """
    Value of purchasing experiment quality theta.
    Returns gross value I(theta) and net value I(theta) - cost.
    """
    mu_t = make_experiment(theta)
    gross = degroot_value(mu_t, prior2, U)
    net   = gross - cost_per_unit * theta
    return gross, net


thetas_fine = np.linspace(0, 1, 200)
gross_vals = []
net_vals   = []
marginal_vals = []

for t in thetas_fine:
    g, n = consumer_value(t, prior2, entropy, cost_per_unit=0.4)
    gross_vals.append(g)
    net_vals.append(n)

# Marginal value (numerical derivative)
marginal_vals = np.gradient(gross_vals, thetas_fine)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(thetas_fine, gross_vals, label="Gross value $I(\\theta)$",
        color="steelblue", linewidth=2)
ax.plot(thetas_fine, [0.4 * t for t in thetas_fine],
        label="Cost $c \\cdot \\theta$", color="tomato",
        linestyle="--", linewidth=2)
ax.plot(thetas_fine, net_vals, label="Net value", color="green", linewidth=2)
ax.set_xlabel(r"Experiment quality $\theta$", fontsize=11)
ax.set_ylabel("Value (Shannon entropy units)", fontsize=11)
ax.set_title("Gross value, cost, and net value of information", fontsize=11)
ax.legend(fontsize=10)

ax2 = axes[1]
ax2.plot(thetas_fine, marginal_vals, label="Marginal value $I'(\\theta)$",
         color="steelblue", linewidth=2)
ax2.axhline(0.4, color="tomato", linestyle="--", linewidth=2,
            label="Marginal cost $c = 0.4$")
opt_idx = np.argmin(np.abs(np.array(marginal_vals) - 0.4))
ax2.axvline(thetas_fine[opt_idx], color="green", linestyle=":",
            label=fr"Optimal $\theta^* \approx {thetas_fine[opt_idx]:.2f}$")
ax2.set_xlabel(r"Experiment quality $\theta$", fontsize=11)
ax2.set_ylabel("Marginal value / Marginal cost", fontsize=11)
ax2.set_title("Optimal demand for information:\n"
              "MV = MC at optimal $\\theta^*$", fontsize=11)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.show()
```

The optimal demand for information $\theta^*$ occurs where marginal value equals
marginal cost.  Both axes shift as the cost $c$ changes, demonstrating comparative
statics.

---

## Application 2 — Sequential Experimental Design (DeGroot 1962)

{cite}`degroot1962` applies the uncertainty-function framework to **sequential
experimental design**.

Each period a statistician observes one draw and updates their posterior.  The
question is which sequence of experiments minimises cumulative expected uncertainty.

The Blackwell theorem implies that if one experiment is more informative than another
at every stage, the optimal sequential strategy simply uses the better experiment at
every period.

We simulate sequential belief updating for experiments of different quality.

```{code-cell} ipython3
def sequential_update(mu_matrix, prior, T=20, seed=0):
    """
    Simulate T sequential belief updates under experiment mu.
    Returns the path of posterior beliefs (T+1, N).
    """
    rng = np.random.default_rng(seed)
    N, M = mu_matrix.shape
    beliefs = np.zeros((T + 1, N))
    beliefs[0] = prior.copy()

    true_state = rng.choice(N, p=prior)

    for t in range(T):
        p = beliefs[t]
        # Draw a signal from the true state
        signal = rng.choice(M, p=mu_matrix[true_state])
        # Bayes update
        unnorm = mu_matrix[:, signal] * p
        beliefs[t + 1] = unnorm / unnorm.sum()

    return beliefs, true_state


def plot_sequential_beliefs(thetas_compare, prior2, T=25):
    fig, axes = plt.subplots(1, len(thetas_compare), figsize=(14, 4), sharey=True)

    for ax, theta in zip(axes, thetas_compare):
        mu_t = make_experiment(theta, N=2)
        for seed in range(15):
            beliefs, ts = sequential_update(mu_t, prior2, T=T, seed=seed)
            c = "steelblue" if ts == 0 else "darkorange"
            ax.plot(beliefs[:, 0], alpha=0.4, color=c, linewidth=1.2)
        ax.axhline(prior2[0], linestyle="--", color="gray", linewidth=1,
                   label="prior")
        ax.axhline(1.0, linestyle=":", color="steelblue", linewidth=0.8)
        ax.axhline(0.0, linestyle=":", color="darkorange", linewidth=0.8)
        ax.set_title(fr"$\theta = {theta}$", fontsize=12)
        ax.set_xlabel("Period $t$", fontsize=11)
        if theta == thetas_compare[0]:
            ax.set_ylabel(r"Posterior $p(s_1 \mid x^t)$", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)

    fig.suptitle("Sequential belief paths under experiments of increasing quality\n"
                 "Blue = true state $s_1$; Orange = true state $s_2$",
                 fontsize=11)
    plt.tight_layout()
    plt.show()

plot_sequential_beliefs([0.2, 0.5, 0.9], prior2, T=30)
```

More informative experiments (larger $\theta$) cause beliefs to converge faster to the
truth.  Under the uniform prior and perfectly symmetric experiments, belief paths are
martingales — the law of iterated expectations for beliefs.

```{code-cell} ipython3
# Verify the martingale property: E[p_{t+1} | x^t] = p_t
def check_martingale(mu_matrix, prior, T=15, n_paths=2000, seed=0):
    """
    Simulate many belief paths and check E[p_{t+1}] ≈ E[p_t].
    Under the true prior, belief sequences are martingales.
    """
    rng = np.random.default_rng(seed)
    N, M = mu_matrix.shape
    all_paths = np.zeros((n_paths, T + 1, N))

    for k in range(n_paths):
        true_state = rng.choice(N, p=prior)
        p = prior.copy()
        all_paths[k, 0] = p
        for t in range(T):
            signal = rng.choice(M, p=mu_matrix[true_state])
            unnorm = mu_matrix[:, signal] * p
            p = unnorm / unnorm.sum()
            all_paths[k, t + 1] = p

    mean_path = all_paths[:, :, 0].mean(axis=0)   # E[p(s1)] over paths

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mean_path, color="steelblue", linewidth=2,
            label=r"$\bar p_t(s_1)$ (mean over paths)")
    ax.axhline(prior[0], linestyle="--", color="tomato", linewidth=1.5,
               label=fr"Prior $p_0 = {prior[0]:.2f}$")
    ax.set_xlabel("Period $t$", fontsize=12)
    ax.set_ylabel(r"$E[p_t(s_1)]$", fontsize=12)
    ax.set_title(r"Belief martingale: $E[p_t(s_1)]$ stays at the prior"
                 "\n(law of iterated expectations for beliefs)", fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

    print(f"Prior = {prior[0]:.4f}")
    print(f"Mean belief (averaged over {n_paths} paths and time): "
          f"{mean_path.mean():.4f}")

check_martingale(mu_info, prior, T=20, n_paths=5000)
```

The mean posterior tracks the prior throughout — reflecting the law of iterated
expectations applied to beliefs.

---

## Summary

Blackwell's theorem identifies a **partial order** on statistical experiments with
three equivalent characterisations:

| Criterion | Condition |
|-----------|-----------|
| **Economic** | Every decision maker prefers $\mu$ to $\nu$: $B(\mu,A) \supseteq B(\nu,A)$ |
| **Sufficiency** | $\nu$ is a garbling of $\mu$: $\nu = \mu Q$ for some Markov $Q$ |
| **Uncertainty** | $\mu$ reduces every concave $U$ more: $E[U(p^\mu)] \leq E[U(p^\nu)]$ |

Kihlstrom's Bayesian exposition clarifies the theorem's geometry by placing the
**posterior distribution** at the centre:

- A more informative experiment creates a **more dispersed** distribution of
  posteriors — a mean-preserving spread of the posterior distribution induced by
  the less informative experiment.
- This links the Blackwell order to **second-order stochastic dominance** on the
  probability simplex $P$.
- The uncertainty-function criterion is then transparent: because $U$ is concave,
  more dispersed posteriors (mean-preserving spread) correspond to higher expected
  $U$ — equivalently, lower expected uncertainty.

DeGroot's contribution is to extend the criterion from specific utility functions
to the *entire class* of concave uncertainty functions, confirming the full
generality of Blackwell's result.


