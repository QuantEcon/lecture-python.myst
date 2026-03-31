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



This lecture studies *Blackwell's theorem* {cite}`blackwell1951,blackwell1953` on ranking statistical experiments.

Our presentation brings in findings from a Bayesian interpretation of Blackwell's theorem by  {cite}`kihlstrom1984`.

Blackwell and Kihlstrom study questions closely related to those encountered in this QuantEcon lecture {doc}`likelihood_bayes`. 

To appreciate the connection involved, it is helpful up front to appreciate how Blackwell's notion of
an **experiment** is related to the concept of a ''probability distribution'' or ''parameterized statistical model'' appearing in  {doc}`likelihood_bayes`  

Blackwell studies a situation in which a decision maker wants to know a state $s$ living in a space $S$.

For Blackwell, an **experiment** is  a **conditional probability model** $\{\mu(\cdot \mid s) : s \in S\}$, i.e., a family of distributions indexed by the unknown state.

We are free to interpret "state" as "parameter".

In a two-state case $S = \{s_1, s_2\}$, the  two conditional densities $f(\cdot) = \mu(\cdot \mid s_1)$ and $g(\cdot) = \mu(\cdot \mid s_2)$ are the ones used repeatedly in  our studies of classical hypothesis testing and Bayesian inference in this suite of QuantEcon lectures.

Blackwell's question — *which experiment is more informative?* — is  about which conditional probability model allows a Bayesian with a prior over $\{s_1, s_2\}$ to learn more about which model governs the world.


Thus, suppose that two signals, $\tilde{x}_\mu$ and $\tilde{x}_\nu$, are both informative about an unknown state $\tilde{s}$.

Blackwell's question is which signal is more informative.

Experiment $\mu$ is **at least as informative as** experiment $\nu$ if every Bayesian decision maker can attain weakly higher expected utility with $\mu$ than with $\nu$.

This economic criterion is equivalent to two statistical criteria:

- *Sufficiency* (Blackwell): $\tilde{x}_\nu$ can be generated from $\tilde{x}_\mu$ by an additional randomization.
- *Uncertainty reduction* (DeGroot {cite}`degroot1962`): $\tilde{x}_\mu$ lowers expected uncertainty at least as much as $\tilde{x}_\nu$ for every concave uncertainty function.

Kihlstrom's reformulation focuses on the *posterior distribution*.

More informative experiments generate posterior distributions that are more dispersed in convex order.

In the two-state case, this becomes the familiar mean-preserving-spread comparison on $[0, 1]$, which can be checked with the integrated-CDF test used for second-order stochastic dominance.

The lecture proceeds as follows:

1. Set up notation and define experiments as Markov matrices.
2. Define stochastic transformations (Markov kernels).
3. State the three equivalent criteria.
4. State and sketch the proof of the main theorem.
5. Develop the Bayesian interpretation via standard experiments and mean-preserving spreads.
6. Illustrate each idea with Python simulations.

We begin with some imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)
```

## Experiments and Markov matrices

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
μ = np.array([[0.6, 0.3, 0.1],
              [0.1, 0.3, 0.6]])

ν = np.array([[0.5, 0.2, 0.3],
              [0.2, 0.5, 0.3]])

print("Experiment μ (rows sum to 1):")
print(μ)
print("\nExperiment ν:")
print(ν)
print("\nRow sums μ:", μ.sum(axis=1))
print("Row sums ν:", ν.sum(axis=1))
```

### Stochastic transformations (Markov kernels)

A **stochastic transformation** $Q$ maps signals of one experiment to signals of another by further randomization.

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

If you observe the more informative signal $\tilde{x}_\mu$, then you can always *throw away* information to reproduce a less informative signal.

The reverse is not possible: a less informative signal cannot be enriched to recover what was lost.

```{code-cell} ipython3
def is_markov(M, tol=1e-10):
    """Check whether a matrix is a valid Markov (row-stochastic) matrix."""
    return np.all(M >= -tol) and np.allclose(M.sum(axis=1), 1.0)


def find_stochastic_transform(μ, ν, tol=1e-8):
    """
    Find a row-stochastic matrix Q that minimizes ||ν - μ @ Q||.
    """
    _, M = μ.shape
    _, K = ν.shape

    def unpack(q_flat):
        return q_flat.reshape(M, K)

    def objective(q_flat):
        Q = unpack(q_flat)
        return np.linalg.norm(ν - μ @ Q)**2

    constraints = [
        {"type": "eq", "fun": lambda q_flat, 
        row=i: unpack(q_flat)[row].sum() - 1.0}
        for i in range(M)
    ]
    bounds = [(0.0, 1.0)] * (M * K)
    Q0 = np.full((M, K), 1 / K).ravel()

    result = minimize(
        objective,
        Q0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": 1_000},
    )

    Q = unpack(result.x)
    residual = np.linalg.norm(ν - μ @ Q)
    return Q, residual, result.success

Q_true = np.array([[1.0, 0.0],
                   [0.0, 1.0],
                   [0.0, 1.0]])

ν_garbled = μ @ Q_true
print("ν = μ @ Q_true:")
print(ν_garbled)
print("ν is Markov:", is_markov(ν_garbled))

Q_found, res, success = find_stochastic_transform(μ, ν_garbled)
print(f"\nRecovered Q (success = {success}, residual = {res:.2e}):")
print(np.round(Q_found, 4))
print("Rows of Q sum to:", Q_found.sum(axis=1).round(4))
```

## Three equivalent criteria

### Criterion 1: the economic criterion

Let $A$ be a compact convex set of actions and $u: A \times S \to \mathbb{R}$ a
bounded utility function.

A decision maker observes $x \in X$, updates beliefs about $\tilde{s}$ by Bayes' rule, and chooses $d(x) \in A$ to maximize expected utility.

Let $p = (p_1, \ldots, p_N)$ be the prior over states, and write

$$
P = \bigl\{(p_1, \ldots, p_N) : p_i \geq 0,\; \textstyle\sum_i p_i = 1\bigr\}
$$

for the probability simplex.

For fixed $A$ and $u$, the set of **achievable expected-utility vectors** under experiment $\mu$ is

$$
B(\mu, A, u) = \Bigl\{v \in \mathbb{R}^N :
  v_i = \textstyle\int_X u(f(x), s_i)\,\mu_i(dx)
  \text{ for some measurable } f: X \to A \Bigr\}.
$$

```{admonition} Definition (Economic criterion)
:class: tip
$\mu$ is **at least as informative as** $\nu$ in the economic sense if

$$
B(\mu, A, u) \supseteq B(\nu, A, u)
$$

for every compact convex action set $A$ and every bounded utility function $u: A \times S \to \mathbb{R}$.
```

This criterion says that experiment $\mu$ is better than experiment $\nu$ if anything a decision maker can achieve after seeing $\nu$, they can also achieve after seeing $\mu$.

The reason is that a more informative experiment lets the agent imitate a less informative one by *ignoring* or *garbling* some of the extra information.

But the reverse need not be possible.

So $B(\mu, A, u) \supseteq B(\nu, A, u)$ means that $\mu$ gives the decision maker at least as many feasible expected-utility outcomes as $\nu$.

Equivalently, every Bayesian decision maker attains weakly higher expected utility with $\tilde{x}_\mu$ than with $\tilde{x}_\nu$, for every prior $p \in P$.

### Criterion 2: the sufficiency criterion

```{admonition} Definition (Blackwell sufficiency)
:class: tip
$\mu \geq \nu$ in Blackwell's sense if there exists a stochastic transformation $Q$ from the signal space of $\mu$ to the signal space of $\nu$ such that

$$
\nu_i(E) = (Q \circ \mu_i)(E)
\quad \forall\, E \in \mathscr{G},\; i = 1, \ldots, N.
$$
```

In matrix notation for finite experiments: $\nu = \mu \, Q$.

### Criterion 3: the uncertainty criterion

{cite:t}`degroot1962` calls any concave function $U: P \to \mathbb{R}$ an **uncertainty function**.

The prototypical example is Shannon entropy:

$$
U(p) = -\sum_{i=1}^{N} p_i \log p_i.
$$

```{admonition} Definition (DeGroot uncertainty criterion)
:class: tip
$\mu$ **reduces expected uncertainty at least as much as** $\nu$ if, for every prior $p \in P$ and every concave $U: P \to \mathbb{R}$,

$$
\int_P U(q)\,\hat\mu^p(dq)
\;\leq\;
\int_P U(q)\,\hat\nu^p(dq),
$$

where $\hat\mu^p$ is the distribution of posterior beliefs induced by experiment $\mu$ under prior $p$.
```

To see this, let $Q = p^\mu(X)$ denote the random posterior induced by experiment $\mu$.

Then $Q$ has distribution $\hat\mu^p$, so

$$
\mathbb{E}[U(Q)] = \int_P U(q)\,\hat\mu^p(dq).
$$

Since $U$ is concave, Jensen's inequality gives

$$
\mathbb{E}[U(Q)] \leq U(\mathbb{E}[Q]) = U(p).
$$

Hence

$$
\int_P U(q)\,\hat\mu^p(dq) \leq U(p),
$$

so any experiment weakly lowers expected uncertainty.

Kihlstrom's standard-experiment construction will later let us compare posterior distributions under the uniform prior $c = (1 / N, \ldots, 1 / N)$.

## The main theorem

```{admonition} Theorem (Blackwell 1953; see also Blackwell 1951, Bonnenblust et al. 1949, and DeGroot 1962)
:class: important
The following three conditions are equivalent:

(i) Economic criterion: $B(\mu, A, u) \supseteq B(\nu, A, u)$ for every compact convex $A$ and every bounded utility function $u$.

(ii) Sufficiency criterion: There exists a stochastic transformation $Q$ from the signal space of $\mu$ to the signal space of $\nu$ such that $\nu = Q \circ \mu$.

(iii) Uncertainty criterion: $\int_P U(q)\,\hat\mu^p(dq) \leq \int_P U(q)\,\hat\nu^p(dq)$ for every prior $p \in P$ and every concave $U$.
```

The hard part is the equivalence between the economic and sufficiency criteria.

*Sketch (ii $\Rightarrow$ i):* If $\nu = \mu Q$, then any decision rule based on $\tilde{x}_\nu$ can be replicated by first observing $\tilde{x}_\mu$, then drawing a synthetic $\tilde{x}_\nu$ from $Q$, and then applying the same rule.

*Sketch (i $\Rightarrow$ ii):* Since $B(\mu, A, u) \supseteq B(\nu, A, u)$ for every $A$ and $u$, a separating-hyperplane (duality) argument implies the existence of a mean-preserving stochastic transformation $D$ mapping posteriors of $\nu$ to posteriors of $\mu$, which constructs the required $Q$.

*Sketch (ii $\Rightarrow$ iii):* Under a garbling, the posterior from the coarser experiment is the conditional expectation of the posterior from the finer experiment, so Jensen's inequality gives the result for every concave $U$.

*Sketch (iii $\Rightarrow$ ii):* The converse — that the inequality for all concave $U$ forces the existence of $Q$ — is proved in {cite}`blackwell1953`, and Kihlstrom's posterior-based representation makes the geometry transparent.

## Kihlstrom's Bayesian interpretation

### Posteriors and standard experiments

The key object in Kihlstrom's analysis is the *posterior belief vector*.

When prior $p$ holds and experiment $\mu$ produces signal $x$, Bayes' rule gives

$$
p_i^\mu(x) = \Pr(\tilde{s} = s_i \mid \tilde{x}_\mu = x)
= \frac{\mu_{ix} \, p_i}{\sum_j \mu_{jx}\, p_j}, \qquad i = 1, \ldots, N.
$$

The posterior $p^\mu(x) \in P$ is a random point in the simplex.

```{admonition} Key property (mean preservation)
:class: note
The prior $p$ is the expectation of the posterior:

$$
\mathbb{E}[p^\mu] = \sum_x \Pr(\tilde{x}_\mu = x)\, p^\mu(x) = p.
$$

This is sometimes called the *law of iterated expectations for beliefs*.
```

For a fixed prior $c$, Kihlstrom's **standard experiment** ${}^c\mu^*$ records only the posterior generated by $\mu$.

Its distribution $\hat\mu^c$ on $P$ satisfies $\int_P q \, \hat\mu^c(dq) = c$.

Two experiments are **informationally equivalent** when they induce the same posterior distribution.

The standard experiment strips away every detail of the signal except its posterior, so it is a *minimal sufficient statistic* for the comparison of experiments.

Any two experiments that generate the same distribution over posteriors lead to identical decisions for every Bayesian decision maker, regardless of how different their raw signal spaces may look.

```{code-cell} ipython3
def compute_posteriors(μ, prior, tol=1e-14):
    """
    Compute the posterior distribution for each signal realisation.
    """
    N, M = μ.shape
    signal_probs = μ.T @ prior
    numerators = μ.T * prior
    posteriors = np.zeros((M, N))
    np.divide(
        numerators,
        signal_probs[:, None],
        out=posteriors,
        where=signal_probs[:, None] > tol,
    )
    return posteriors, signal_probs


def check_mean_preservation(posteriors, signal_probs, prior):
    """Verify E[posterior] == prior."""
    expected_posterior = (posteriors * signal_probs[:, None]).sum(axis=0)
    return expected_posterior, np.allclose(expected_posterior, prior)


N = 2
prior = np.array([0.5, 0.5])

μ_info = np.array([[0.8, 0.2],
                   [0.2, 0.8]])

ν_info = np.array([[0.6, 0.4],
                   [0.4, 0.6]])

post_μ, probs_μ = compute_posteriors(μ_info, prior)
post_ν, probs_ν = compute_posteriors(ν_info, prior)

print("=== Experiment μ (more informative) ===")
print("Signal probabilities:", probs_μ.round(3))
print("Posteriors (row = signal, col = state):")
print(post_μ.round(3))
mean_μ, ok_μ = check_mean_preservation(post_μ, probs_μ, prior)
print(f"E[posterior] = {mean_μ.round(4)}  (equals prior: {ok_μ})")

print("\n=== Experiment ν (less informative) ===")
print("Signal probabilities:", probs_ν.round(3))
print("Posteriors:")
print(post_ν.round(3))
mean_ν, ok_ν = check_mean_preservation(post_ν, probs_ν, prior)
print(f"E[posterior] = {mean_ν.round(4)}  (equals prior: {ok_ν})")
```

### Visualizing posterior distributions on the simplex

For $N = 2$ states, the simplex $P$ is the unit interval $[0, 1]$ (the probability
of state $s_1$).  We can directly plot the distribution of posteriors under
experiments $\mu$ and $\nu$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Posterior distributions in the two-state case
    name: fig-blackwell-two-state-posteriors
---
def plot_posterior_distributions(μ_matrix, ν_matrix, prior,
                                 labels=("μ (more informative)",
                                         "ν (less informative)")):
    """
    For a two-state experiment, plot the distribution of posteriors
    (i.e., the standard experiment distribution) on [0,1].
    """
    posts_μ, probs_μ = compute_posteriors(μ_matrix, prior)
    posts_ν, probs_ν = compute_posteriors(ν_matrix, prior)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    prior_val = prior[0]

    for ax, posts, probs, label in zip(
        axes, [posts_μ, posts_ν], [probs_μ, probs_ν], labels):
        p_s1 = posts[:, 0]
        ax.vlines(p_s1, 0, probs, linewidth=6, color="steelblue", alpha=0.7)
        ax.axvline(prior_val, color="tomato", linestyle="--", linewidth=2,
                   label=f"prior = {prior_val:.2f}")
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"posterior $p(s_1 \mid x)$", fontsize=12)
        ax.set_ylabel("probability mass", fontsize=12)
        mean_post = (p_s1 * probs).sum()
        ax.axvline(mean_post, color="green", linestyle=":", linewidth=2,
                   label=f"E[post] = {mean_post:.2f}")
        ax.text(0.03, 0.94, label, transform=ax.transAxes, va="top")
        ax.legend()

    plt.tight_layout()
    plt.show()

plot_posterior_distributions(μ_info, ν_info, prior)
```

The more informative experiment $\mu$ pushes posteriors farther from the prior in both directions.

### Mean-preserving spreads and Blackwell's order

Kihlstrom's key reformulation is the following.

```{admonition} Theorem (Kihlstrom's Reformulation)
:class: important
$\mu \geq \nu$ in Blackwell's sense if and only if $\hat\mu^c$ is a
**mean-preserving spread** of $\hat\nu^c$; that is,

$$
\int_P g(p)\,\hat\mu^c(dp) \;\geq\; \int_P g(p)\,\hat\nu^c(dp)
$$

for every convex function $g: P \to \mathbb{R}$.
```

Equivalently, $\hat\mu^c$ is larger than $\hat\nu^c$ in convex order.

A better experiment spreads posterior beliefs farther from the prior while preserving their mean.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Convex-order check in the two-state case
    name: fig-blackwell-convex-order-check
---
def check_mps_convex_functions(μ_matrix, ν_matrix, prior, n_functions=200):
    """
    Verify the mean-preserving spread condition using
    convex functions g(p) = max(p - t, 0).
    """
    posts_μ, probs_μ = compute_posteriors(μ_matrix, prior)
    posts_ν, probs_ν = compute_posteriors(ν_matrix, prior)

    p_μ = posts_μ[:, 0]
    p_ν = posts_ν[:, 0]

    thresholds = np.linspace(0, 1, n_functions)
    diffs = []
    for t in thresholds:
        Eg_μ = (np.maximum(p_μ - t, 0) * probs_μ).sum()
        Eg_ν = (np.maximum(p_ν - t, 0) * probs_ν).sum()
        diffs.append(Eg_μ - Eg_ν)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, diffs, color="steelblue", linewidth=2)
    ax.axhline(0, color="tomato", linestyle="--", linewidth=2)
    ax.fill_between(thresholds, diffs, 0,
                    where=np.array(diffs) >= 0,
                    alpha=0.25, color="steelblue",
                    label="$E[g(p^μ)] - E[g(p^ν)] \\geq 0$")
    ax.set_xlabel("threshold $t$", fontsize=12)
    ax.set_ylabel(r"$E[\max(p-t,0)]$ difference", fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    all_non_negative = all(d >= -1e-10 for d in diffs)
    print(f"μ is a mean-preserving spread of ν: {all_non_negative}")
    return diffs

_ = check_mps_convex_functions(μ_info, ν_info, prior)
```

## Simulating the Blackwell order with many states

We now move to a three-state example.

Experiment $\mu$ is strongly correlated with the state, and experiment $\nu$ is a garbling of $\mu$.

```{code-cell} ipython3
N3 = 3
prior3 = np.array([1/3, 1/3, 1/3])

μ3 = np.array([[0.7, 0.2, 0.1],
               [0.1, 0.7, 0.2],
               [0.2, 0.1, 0.7]])

Q3 = np.array([[0.9, 0.05, 0.05],
               [0.05, 0.8, 0.15],
               [0.05, 0.15, 0.8]])

ν3 = μ3 @ Q3

print("μ (3×3):")
print(np.round(μ3, 2))
print("\nQ (garbling):")
print(np.round(Q3, 2))
print("\nν = μ @ Q:")
print(np.round(ν3, 3))
```


For three states, posterior beliefs live in a 2-simplex.

Let's visualize the posterior clouds under $\mu$ and $\nu$

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Posterior clouds on the 2-simplex
    name: fig-blackwell-simplex-clouds
---
def sample_posteriors(μ_matrix, prior, n_draws=3000):
    """
    Simulate n_draws observations from the experiment and compute
    the resulting posterior beliefs.
    Returns array of shape (n_draws, N).
    """
    N, M = μ_matrix.shape
    states = np.random.choice(N, size=n_draws, p=prior)
    signals = np.array([np.random.choice(M, p=μ_matrix[s]) for s in states])
    posteriors, _ = compute_posteriors(μ_matrix, prior)
    return posteriors[signals]


def simplex_to_cart(pts):
    """Convert 3-simplex barycentric coordinates to 2-D Cartesian."""
    corners = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.5, np.sqrt(3)/2]])
    return pts @ corners


def plot_simplex_posteriors(μ_matrix, ν_matrix, prior3, n_draws=3000):
    posts_μ = sample_posteriors(μ_matrix, prior3, n_draws)
    posts_ν = sample_posteriors(ν_matrix, prior3, n_draws)

    cart_μ = simplex_to_cart(posts_μ)
    cart_ν = simplex_to_cart(posts_ν)
    prior_cart = simplex_to_cart(prior3[None, :])[0]

    corners = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.5, np.sqrt(3)/2]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panel_labels = ["μ (more informative)", "ν (garbled)"]
    data = [(cart_μ, "steelblue"), (cart_ν, "darkorange")]
    labels = ["$s_1$", "$s_2$", "$s_3$"]
    offsets = [(-0.07, -0.05), (0.02, -0.05), (-0.02, 0.03)]

    for ax, (cart, c), panel_label in zip(axes, data, panel_labels):
        tri = plt.Polygon(corners, fill=False, edgecolor="black", linewidth=2)
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
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.03, 0.94, panel_label, transform=ax.transAxes, va="top")
        ax.legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.show()

plot_simplex_posteriors(μ3, ν3, prior3)
```

Under $\mu$, the posterior cloud reaches farther toward the vertices.

Under the garbled experiment $\nu$, the cloud stays closer to the center.

## The DeGroot uncertainty function

### Concave uncertainty functions and the value of information

{cite}`degroot1962` formalizes the value of information through an **uncertainty function** $U: P \to \mathbb{R}$.

In DeGroot's axiomatization, an uncertainty function is:

- *Concave*: by Jensen, observing any signal weakly reduces expected uncertainty.
- *Symmetric*: it depends on the components of $p$, not their labeling.
- *Normalized*: it is maximized at $p = (1/N, \ldots, 1/N)$ and minimized at vertices.

The **value of experiment $\mu$ given prior $p$** is

$$
I(\tilde{x}_\mu;\, \tilde{s};\, U)
= U(p) - \mathbb{E}[U(p^\mu)],
$$

This quantity is the expected reduction in uncertainty.

Blackwell's order is equivalent to the statement that $I(\tilde{x}_\mu; \tilde{s}; U) \geq I(\tilde{x}_\nu; \tilde{s}; U)$ for *every* concave $U$.

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

where $H(\tilde{s} \mid \tilde{x}_\mu)$ is the conditional entropy of the state given the signal.

To see why, write $H(\tilde{s} \mid \tilde{x}_\mu) = \sum_x \Pr(\tilde{x}_\mu = x) \, H(\tilde{s} \mid \tilde{x}_\mu = x)$, where each conditional entropy term equals $-\sum_i p_i^\mu(x) \log p_i^\mu(x) = U_H(p^\mu(x))$.

Substituting into DeGroot's formula gives $I = U_H(c) - \mathbb{E}[U_H(p^\mu)] = \log N - H(\tilde{s} \mid \tilde{x}_\mu)$, which is exactly the *mutual information* between $\tilde{x}_\mu$ and $\tilde{s}$.

```{note}
The Blackwell ordering implies the entropy-based inequality, but the *converse fails*: entropy alone does not pin down the full Blackwell ordering.

Two experiments can have the same mutual information yet differ in Blackwell rank, because a single concave function cannot detect all differences in the dispersion of posteriors.

The full Blackwell ordering requires the inequality to hold for *every* concave $U$, not just Shannon entropy.
```

```{code-cell} ipython3
def entropy(p, ε=1e-12):
    """Shannon entropy of a probability vector."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, ε, 1.0)
    return -np.sum(p * np.log(p))


def degroot_value(μ_matrix, prior, U_func):
    """
    Compute DeGroot's value of information I = U(prior) - E[U(posterior)].
    """
    posts, probs = compute_posteriors(μ_matrix, prior)
    prior_uncertainty = U_func(prior)
    expected_post_uncertainty = sum(
        probs[j] * U_func(posts[j]) for j in range(len(probs)))
    return prior_uncertainty - expected_post_uncertainty


def gini_impurity(p):
    """Gini impurity: 1 - sum(p_i^2)."""
    return 1.0 - np.sum(np.asarray(p)**2)


def tsallis_entropy(p, q=2):
    """Tsallis entropy of order q (concave for q>1)."""
    p = np.clip(p, 1e-12, 1.0)
    return (1 - np.sum(p**q)) / (q - 1)


def sqrt_index(p):
    """Concave uncertainty index based on sum(sqrt(p_i))."""
    p = np.clip(np.asarray(p), 0.0, 1.0)
    return np.sum(np.sqrt(p)) - 1.0

uncertainty_functions = {
    "Shannon entropy": entropy,
    "Gini impurity": gini_impurity,
    "Tsallis (q=2)": tsallis_entropy,
    "Square-root index": sqrt_index,
}

print(f"{'Uncertainty function':<22}  {'I(μ)':<10}  {'I(ν)':<10}  {'I(μ)>=I(ν)?'}")
print("-" * 58)
for name, U in uncertainty_functions.items():
    I_μ = degroot_value(μ_info, prior, U)
    I_ν = degroot_value(ν_info, prior, U)
    print(f"{name:<22}  {I_μ:<10.4f}  {I_ν:<10.4f}  {I_μ >= I_ν - 1e-10}")
```

As predicted by the theorem, $I(\mu) \geq I(\nu)$ for every concave uncertainty function once we know $\mu \geq \nu$ in the Blackwell sense.

### Value of information as a function of experiment quality

We now parameterize a continuum of experiments between the uninformative and perfectly informative cases.

For $N = 2$ states, a natural family is

$$
\mu(\theta) = (1 - \theta) \cdot \tfrac{1}{2}\mathbf{1}\mathbf{1}^\top
             + \theta \cdot I_2,
\quad \theta \in [0, 1],
$$

The first term is the completely mixed matrix and $I_2$ is the identity.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Value of information and experiment quality
    name: fig-blackwell-value-by-quality
---
def make_experiment(θ, N=2):
    """Parameterized experiment: θ=0 is uninformative, θ=1 is perfect."""
    return (1 - θ) * np.ones((N, N)) / N + θ * np.eye(N)


θs = np.linspace(0, 1, 100)
prior2 = np.array([0.5, 0.5])

fig, ax = plt.subplots(figsize=(9, 4))
for name, U in uncertainty_functions.items():
    values = [degroot_value(make_experiment(θ), prior2, U) for θ in θs]
    vmin, vmax = values[0], values[-1]
    normed = (np.array(values) - vmin) / (vmax - vmin + 1e-15)
    ax.plot(θs, normed, label=name, linewidth=2)

ax.set_xlabel("experiment quality θ  (0 = uninformative, 1 = perfect)",
              fontsize=11)
ax.set_ylabel("normalized value of information I(μ(θ))", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
```

Every concave uncertainty function assigns weakly higher value to a more informative experiment.

## Connection to second-order stochastic dominance

The uncertainty-function representation makes the connection to **second-order stochastic dominance (SOSD)** explicit.

Because $U$ is concave, $-U$ is convex, and the condition

$$
\mathbb{E}[U(p^\mu)] \leq \mathbb{E}[U(p^\nu)] \quad \text{for all concave } U
$$

is precisely the statement that $\hat\mu^c$ dominates $\hat\nu^c$ in the **mean-preserving spread** sense on $P$.

The Blackwell ordering on *experiments* is therefore isomorphic to the SOSD ordering on *distributions of posteriors*.

When $N = 2$, posterior beliefs are scalars in $[0, 1]$, and the SOSD comparison reduces to the classical integrated-CDF test.

Specifically, $\hat\mu^c$ is a mean-preserving spread of $\hat\nu^c$ if and only if $\int_0^t F_\nu(s)\,ds \geq \int_0^t F_\mu(s)\,ds$ for all $t \in [0,1]$, where $F_\mu$ and $F_\nu$ are the CDFs of the posterior on $s_1$ under each experiment.

We can verify this graphically for the two-state example above

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Integrated-CDF check in the two-state case
    name: fig-blackwell-integrated-cdf
---
def cdf_data_1d(weights, values):
    """Sort support points and cumulative masses for a discrete distribution."""
    idx = np.argsort(values)
    sorted_vals = values[idx]
    sorted_wts = weights[idx]
    cum_mass = np.cumsum(sorted_wts)
    return sorted_vals, cum_mass


def plot_sosd_posteriors(μ_matrix, ν_matrix, prior):
    """Plot CDFs and integrated CDFs for the posterior-on-s1 distributions."""
    posts_μ, probs_μ = compute_posteriors(μ_matrix, prior)
    posts_ν, probs_ν = compute_posteriors(ν_matrix, prior)

    p_μ = posts_μ[:, 0]
    p_ν = posts_ν[:, 0]

    sv_μ, cm_μ = cdf_data_1d(probs_μ, p_μ)
    sv_ν, cm_ν = cdf_data_1d(probs_ν, p_ν)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for sv, cm, lbl, c in [(sv_μ, cm_μ, "μ", "steelblue"),
                           (sv_ν, cm_ν, "ν", "darkorange")]:
        xs = np.concatenate([[0], sv, [1]])
        ys = np.concatenate([[0], cm, [1]])
        ax.step(xs, ys, where="post", label=lbl, color=c, linewidth=2)
    ax.axvline(prior[0], linestyle="--", color="gray", alpha=0.6, linewidth=2,
               label="prior")
    ax.set_xlabel(r"posterior $p(s_1 \mid x)$", fontsize=12)
    ax.set_ylabel("cumulative probability", fontsize=12)
    ax.text(0.03, 0.94, "CDFs", transform=ax.transAxes, va="top")
    ax.legend(fontsize=11)

    ax2 = axes[1]
    grid = np.linspace(0, 1, 200)

    def integrated_cdf(sorted_vals, cum_mass, grid):
        cdf = np.array([cum_mass[sorted_vals <= t].max()
                        if np.any(sorted_vals <= t) else 0.0
                        for t in grid])
        return np.cumsum(cdf) * (grid[1] - grid[0])

    int_μ = integrated_cdf(sv_μ, cm_μ, grid)
    int_ν = integrated_cdf(sv_ν, cm_ν, grid)

    ax2.plot(grid, int_μ, label=r"$\int F_\mu$", color="steelblue", linewidth=2)
    ax2.plot(grid, int_ν, label=r"$\int F_\nu$", color="darkorange", linewidth=2)
    ax2.fill_between(grid, int_μ, int_ν,
                     where=int_ν >= int_μ,
                     alpha=0.2, color="darkorange",
                     label=r"$\int F_\nu \geq \int F_\mu$ ($\mu$ MPS-dominates $\nu$)")
    ax2.set_xlabel(r"$t$", fontsize=12)
    ax2.set_ylabel("integrated CDF", fontsize=12)
    ax2.text(0.03, 0.94, "integrated CDFs", transform=ax2.transAxes, va="top")
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

plot_sosd_posteriors(μ_info, ν_info, prior)
```

## Mean-preserving randomization

Kihlstrom proves that (i) $\Rightarrow$ (ii) by explicit construction.

Given that $\mu$ achieves at least the value of $\nu$ for every decision maker, he constructs a stochastic transformation $D(p^0, \cdot)$ on $P$ that is **mean-preserving**:

$$
\int_P q \, D(p^0, dq) = p^0.
$$

Setting $Q = D$ provides the Markov kernel witnessing Blackwell sufficiency.

The mean-preservation condition says: passing $\tilde{x}_\mu$ through $Q$ to produce a synthetic $\tilde{x}_\nu$ cannot add information — it only destroys it.

```{code-cell} ipython3
def verify_garbling_mean_preservation(μ_matrix, Q_matrix, prior):
    """Verify that a garbling preserves the prior as the mean posterior."""
    ν_matrix = μ_matrix @ Q_matrix
    posts_μ, probs_μ = compute_posteriors(μ_matrix, prior)
    posts_ν, probs_ν = compute_posteriors(ν_matrix, prior)

    mean_μ = (posts_μ * probs_μ[:, None]).sum(axis=0)
    mean_ν = (posts_ν * probs_ν[:, None]).sum(axis=0)

    print(f"Prior:               {prior.round(4)}")
    print(f"E[p^μ]:              {mean_μ.round(4)}")
    print(f"E[p^ν = p^(μQ)]:     {mean_ν.round(4)}")
    print(f"Both equal prior?    μ: {np.allclose(mean_μ, prior)}, "
          f"ν: {np.allclose(mean_ν, prior)}")


Q_soft = np.array([[0.7, 0.3],
                   [0.3, 0.7]])

verify_garbling_mean_preservation(μ_info, Q_soft, prior)
```

## Comparing experiments systematically

We now study a grid of experiments indexed by their quality parameter $\theta$.

We will compare:

1. The spread of posterior beliefs.
2. The value of information under concave uncertainty functions.
3. The integrated-CDF ranking in the two-state case.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Posterior distributions for increasing experiment quality
    name: fig-blackwell-quality-grid
---
θ_grid = [0.1, 0.4, 0.7, 1.0]
prior2 = np.array([0.5, 0.5])

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flat

for ax, θ in zip(axes, θ_grid):
    μ_θ = make_experiment(θ)
    posts, probs = compute_posteriors(μ_θ, prior2)
    p_s1 = posts[:, 0]
    ax.vlines(p_s1, 0, probs, linewidth=8, color="steelblue", alpha=0.7)
    ax.axvline(prior2[0], color="tomato", linestyle="--", linewidth=2,
               label=f"prior = {prior2[0]:.2f}")
    I_H = degroot_value(μ_θ, prior2, entropy)
    I_G = degroot_value(μ_θ, prior2, gini_impurity)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"posterior $p(s_1 \mid x)$", fontsize=11)
    ax.set_ylabel("probability mass", fontsize=11)
    ax.text(0.03, 0.94,
            f"θ = {θ}\n" f"I_H = {I_H:.3f}\n" f"I_G = {I_G:.3f}",
            transform=ax.transAxes, va="top")
    ax.legend(fontsize=10)

plt.tight_layout()
plt.show()
```

As $\theta$ rises from 0 to 1, posterior beliefs move toward the vertices $\{0, 1\}$.

At the same time, the value of information rises under every concave uncertainty function.

## Application 1: product quality information

{cite:t}`kihlstrom1974a` applies Blackwell's theorem to consumer demand for information about product quality.

- The unknown state $\tilde{s}$ is a product parameter $\theta$.
- A consumer can purchase $\lambda$ units of information at cost $c(\lambda)$.
- As $\lambda$ rises, the experiment becomes more informative in the Blackwell sense.

The Blackwell order says that, absent costs, more information is always better for every expected-utility maximizer.

Optimal information demand equates the *marginal value of the standard experiment* to its *marginal cost*.

In the example below, we assume a linear cost $c \cdot \lambda$ and a simple family of experiments $\mu(\theta)$ as above with $c = 0.4$

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Information demand in a simple quality example
    name: fig-blackwell-information-demand
---
def consumer_value(θ, prior2, U=entropy, cost_per_unit=0.5):
    """Value of purchasing experiment quality θ."""
    μ_t = make_experiment(θ)
    gross = degroot_value(μ_t, prior2, U)
    net   = gross - cost_per_unit * θ
    return gross, net


θ_fine = np.linspace(0, 1, 200)
gross_vals = []
net_vals   = []
marginal_vals = []

for θ in θ_fine:
    g, n = consumer_value(θ, prior2, entropy, cost_per_unit=0.4)
    gross_vals.append(g)
    net_vals.append(n)

marginal_vals = np.gradient(gross_vals, θ_fine)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(θ_fine, gross_vals, label="Gross value I(θ)",
        color="steelblue", linewidth=2)
ax.plot(θ_fine, [0.4 * t for t in θ_fine],
        label="Cost c · θ", color="tomato",
        linestyle="--", linewidth=2)
ax.plot(θ_fine, net_vals, label="Net value", color="green", linewidth=2)
ax.set_xlabel("experiment quality θ", fontsize=11)
ax.set_ylabel("value (Shannon entropy units)", fontsize=11)
ax.legend(fontsize=10)

ax2 = axes[1]
ax2.plot(θ_fine, marginal_vals, label="Marginal value I'(θ)",
         color="steelblue", linewidth=2)
ax2.axhline(0.4, color="tomato", linestyle="--", linewidth=2,
            label="Marginal cost $c = 0.4$")
opt_idx = np.argmin(np.abs(np.array(marginal_vals) - 0.4))
ax2.axvline(θ_fine[opt_idx], color="green", linestyle=":",
            linewidth=2,
            label=f"Optimal θ* ≈ {θ_fine[opt_idx]:.2f}")
ax2.set_xlabel("experiment quality θ", fontsize=11)
ax2.set_ylabel("marginal value / marginal cost", fontsize=11)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.show()
```

The optimal demand for information $\theta^*$ occurs where marginal value equals marginal cost.

Comparative statics follow from shifts in either curve.

## Application 2: sequential experimental design

{cite:t}`degroot1962` applies the uncertainty-function framework to *sequential experimental design*.

Each period a statistician observes one draw and updates the posterior.

The question is which sequence of experiments minimizes cumulative expected uncertainty.

If one experiment is more informative than another at every stage, then the Blackwell order favors using the better experiment at every date.

We now simulate sequential belief updating for experiments of different quality.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Sequential posterior paths for different experiment qualities
    name: fig-blackwell-sequential-paths
---
def sequential_update(μ_matrix, prior, T=20, seed=0):
    """Simulate T sequential belief updates under experiment μ."""
    rng = np.random.default_rng(seed)
    N, M = μ_matrix.shape
    beliefs = np.zeros((T + 1, N))
    beliefs[0] = prior.copy()

    true_state = rng.choice(N, p=prior)

    for t in range(T):
        p = beliefs[t]
        signal = rng.choice(M, p=μ_matrix[true_state])
        unnorm = μ_matrix[:, signal] * p
        beliefs[t + 1] = unnorm / unnorm.sum()

    return beliefs, true_state


def plot_sequential_beliefs(θs_compare, prior2, T=25):
    fig, axes = plt.subplots(1, len(θs_compare), figsize=(14, 4), sharey=True)

    for ax, θ in zip(axes, θs_compare):
        μ_t = make_experiment(θ, N=2)
        for seed in range(15):
            beliefs, ts = sequential_update(μ_t, prior2, T=T, seed=seed)
            c = "steelblue" if ts == 0 else "darkorange"
            ax.plot(beliefs[:, 0], alpha=0.35, color=c, linewidth=2)
        ax.axhline(prior2[0], linestyle="--", color="gray", linewidth=2,
                   label="prior")
        ax.axhline(1.0, linestyle=":", color="steelblue", linewidth=2)
        ax.axhline(0.0, linestyle=":", color="darkorange", linewidth=2)
        ax.set_xlabel(r"period $t$", fontsize=11)
        if θ == θs_compare[0]:
            ax.set_ylabel(r"posterior $p(s_1 \mid x^t)$", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.text(0.03, 0.94, f"θ = {θ}", transform=ax.transAxes, va="top")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

plot_sequential_beliefs([0.2, 0.5, 0.9], prior2, T=30)
```

More informative experiments make beliefs converge faster to the truth.

Under the correct prior, the posterior process is a martingale.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Unconditional implication of the posterior martingale property
    name: fig-blackwell-martingale-mean
---
def check_martingale_mean(μ_matrix, prior, T=15, n_paths=2000, seed=0):
    """
    Simulate many belief paths and check E[p_t] = p_0.
    """
    rng = np.random.default_rng(seed)
    N, M = μ_matrix.shape
    all_paths = np.zeros((n_paths, T + 1, N))

    for k in range(n_paths):
        true_state = rng.choice(N, p=prior)
        p = prior.copy()
        all_paths[k, 0] = p
        for t in range(T):
            signal = rng.choice(M, p=μ_matrix[true_state])
            unnorm = μ_matrix[:, signal] * p
            p = unnorm / unnorm.sum()
            all_paths[k, t + 1] = p

    mean_path = all_paths[:, :, 0].mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mean_path, color="steelblue", linewidth=2,
            label=r"$\bar p_t(s_1)$ (mean over paths)")
    ax.axhline(prior[0], linestyle="--", color="tomato", linewidth=2,
               label=fr"Prior $p_0 = {prior[0]:.2f}$")
    ax.set_xlabel(r"period $t$", fontsize=12)
    ax.set_ylabel(r"$E[p_t(s_1)]$", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

    print(f"Prior = {prior[0]:.4f}")
    print(f"Average mean belief across dates: {mean_path.mean():.4f}")

check_martingale_mean(μ_info, prior, T=20, n_paths=5000)
```

The simulated cross-sectional mean stays close to the prior at every date.

This is the unconditional implication of the posterior martingale property.

## Summary

Blackwell's theorem identifies a *partial order* on statistical experiments with
three equivalent characterizations:

| Criterion | Condition |
|-----------|-----------|
| Economic | Every decision maker weakly prefers $\mu$ to $\nu$: $B(\mu, A, u) \supseteq B(\nu, A, u)$ |
| Sufficiency | $\nu$ is a garbling of $\mu$: $\nu = \mu Q$ for some Markov $Q$ |
| Uncertainty | $\mu$ reduces expected uncertainty more for every prior $p$ and every concave $U$ |

Kihlstrom's Bayesian exposition places the *posterior distribution* at the center.

A more informative experiment generates a more dispersed posterior distribution with the same mean prior.

The right probabilistic language is convex order, and the Blackwell ordering on experiments is isomorphic to the second-order stochastic dominance (SOSD) ordering on distributions of posteriors.

In the two-state case this reduces to the familiar mean-preserving-spread comparison on $[0, 1]$, which can be verified with the integrated-CDF test.

DeGroot's contribution is to extend the comparison from particular utility functions to the full class of concave uncertainty functions.

---

## Relation to Bayesian likelihood-ratio learning

The lecture {doc}`likelihood_bayes` studies Bayesian learning in a setting that is a special, dynamic instance of everything developed here.

This section transports concepts back and forth between the two lectures.

### The state space is the same

In {doc}`likelihood_bayes` the unknown "state of the world" is which density nature chose permanently: nature drew the data either from $f$ or from $g$, but not which one is known to the observer.

This is a two-element finite state space

$$
S = \{s_1, s_2\} \qquad \text{with } s_1 \leftrightarrow f,\quad s_2 \leftrightarrow g.
$$

The Bayesian prior $\pi_0 \in [0,1]$ on $s_1 = f$ plays exactly the role of the prior $p \in P$ on the probability simplex in the present lecture.

### A single draw is an experiment

A single observation $w_t$ constitutes a Blackwell experiment with signal space $X$ and Markov kernel

$$
\mu = \begin{pmatrix} f(\cdot) \\ g(\cdot) \end{pmatrix},
$$

where row $i$ is the conditional density of the signal given state $s_i$:
$\mu(\cdot \mid s_1) = f(\cdot)$ and $\mu(\cdot \mid s_2) = g(\cdot)$.

This is the continuous-signal analogue of the $N \times M$ Markov matrix studied above (with $N = 2$ states and a continuum of signals instead of $M$ discrete ones).

### $t$ IID draws form a richer experiment

Observing the history $w^t = (w_1, \ldots, w_t)$ is a strictly more informative Blackwell experiment than observing any sub-history $w^s$ for $s < t$, because the conditional joint densities for $w^t$ are

$$
\mu_t(\cdot \mid s_1) = f(w_1) f(w_2) \cdots f(w_t),
\qquad
\mu_t(\cdot \mid s_2) = g(w_1) g(w_2) \cdots g(w_t).
$$

The experiment $\mu_t$ Blackwell-dominates $\mu_s$ for any $t > s$: you can always garble $w^t$ down to $w^s$ by discarding the last $t - s$ draws, which is an explicit stochastic transformation $Q$ satisfying $\mu_s = \mu_t Q$.

The reverse is impossible — you cannot reconstruct information from fewer draws.

This is why more data is always weakly better for every expected-utility maximiser (the economic criterion of Blackwell's theorem).

### The likelihood ratio process is the sufficient statistic of the experiment

The key formula in {doc}`likelihood_bayes` is

$$
\pi_{t+1} = \frac{\pi_0 \, L(w^{t+1})}{\pi_0 \, L(w^{t+1}) + 1 - \pi_0},
\qquad
L(w^t) = \prod_{i=1}^t \frac{f(w_i)}{g(w_i)}.
$$

Because $\pi_{t+1}$ depends on $w^t$ **only through** $L(w^t)$, the likelihood ratio process is a **sufficient statistic** for the experiment $\mu_t$.

In Blackwell's language, the experiment "report $L(w^t)$" is informationally equivalent to "report $w^t$": passing $w^t$ through the deterministic map $w^t \mapsto L(w^t)$ is a (degenerate) stochastic transformation that discards nothing relevant to discriminating $f$ from $g$.

### The posterior lives on the 1-simplex and is Kihlstrom's standard experiment

With $N = 2$ states the probability simplex $P$ collapses to the unit interval $[0,1]$.
Kihlstrom's standard experiment records only the posterior

$$
\pi_t = \Pr(s = f \mid w^t),
$$

which is the sufficient statistic that the Bayesian tracks throughout.

The **distribution** of $\pi_t$ over all possible histories $w^t$ is Kihlstrom's $\hat{\mu}^c$ — the distribution of posteriors induced by the experiment $\mu_t$ starting from prior $\pi_0 = c$.

### The martingale property is mean preservation

{doc}`likelihood_bayes` proves that $\{\pi_t\}$ is a **martingale**:

$$
E[\pi_t \mid \pi_{t-1}] = \pi_{t-1},
$$

and in particular $E[\pi_t] = \pi_0$ for all $t$.

This is exactly the **mean-preservation** condition that sits at the centre of Kihlstrom's reformulation: the distribution of posteriors $\hat{\mu}^c$ must satisfy $\int_P p \, \hat{\mu}^c(dp) = c$.

Mean preservation is not a special feature of this two-state example; it is an exact consequence of Bayes' law for **any** experiment.

### Blackwell's theorem explains why more data always helps

Kihlstrom's reformulation states:

> $\mu_t \geq \mu_s$ in Blackwell's sense if and only if $\hat{\mu}_t^c$ is a **mean-preserving spread** of $\hat{\mu}_s^c$, i.e., posteriors under $\mu_t$ are more dispersed than under $\mu_s$.

In the {doc}`likelihood_bayes` setting this means the distribution of $\pi_t$ is a mean-preserving spread of the distribution of $\pi_s$ for $t > s$: more data pushes posteriors further from the prior toward either $0$ or $1$.

The almost-sure convergence $\pi_t \to 0$ or $1$ is the limit of this spreading process — perfect information resolves all uncertainty, collapsing the distribution to a degenerate point mass at a vertex of the simplex.

### DeGroot uncertainty functions and mutual information

The Shannon entropy of the two-state posterior is

$$
U_H(\pi) = -\pi \log \pi - (1-\pi)\log(1-\pi).
$$

DeGroot's value of information for the experiment that generates $t$ draws is

$$
I(\mu_t;\, U_H) = U_H(\pi_0) - E[U_H(\pi_t)],
$$

which equals the **mutual information** between the history $w^t$ and the unknown state.

Because $\mu_t$ Blackwell-dominates $\mu_s$ for $t > s$, Blackwell's theorem guarantees $I(\mu_t; U) \geq I(\mu_s; U)$ for **every** concave uncertainty function $U$ — more draws reduce expected uncertainty under every such measure, not just Shannon entropy.

### Summary table

The table below collects the complete translation between concepts in the two lectures.

| Concept in {doc}`likelihood_bayes` | Concept in this lecture |
|---|---|
| States $\{f, g\}$ | State space $S = \{s_1, s_2\}$ |
| Densities $f(\cdot)$, $g(\cdot)$ | Rows of experiment matrix $\mu$ |
| Single draw $w_t$ | Blackwell experiment with continuous signal space |
| History $w^t$ of $t$ IID draws | Richer experiment $\mu_t$ Blackwell-dominating $\mu_s$, $s < t$ |
| Likelihood ratio $L(w^t)$ | Sufficient statistic / standard experiment |
| Prior $\pi_0$ | Prior $p \in P$ on the 1-simplex $[0,1]$ |
| Posterior $\pi_t$ | Posterior random variable on $P = [0,1]$ |
| Distribution of $\pi_t$ across histories | $\hat{\mu}^c$ (Kihlstrom's posterior distribution) |
| Martingale property $E[\pi_t] = \pi_0$ | Mean preservation of $\hat{\mu}^c$ |
| $\pi_t \to 0$ or $1$ almost surely | Posteriors spread to vertices (MPS in the limit) |
| Mutual information $I(\mu_t; U_H)$ | DeGroot value of information |
| More draws $\Rightarrow$ better for all decision makers | Blackwell ordering $\mu_t \geq \mu_s$ |
| Garbling (discard last $t - s$ draws) | Stochastic transformation $Q$ with $\mu_s = \mu_t Q$ |
