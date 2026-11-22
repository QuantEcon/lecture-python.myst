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

(information_projections)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Information Projections

```{contents} Contents
:depth: 2
```

## Overview

In Euclidean geometry, the projection of a point onto a set is the point of the set that is closest to it.

When the set is a line or a plane, the projection is unique, the segment joining the point to its projection meets the set at a right angle, and the Pythagorean theorem holds.

This lecture describes analogous results for probability distributions, with the Kullback–Leibler (KL) divergence in place of squared distance.

A distribution has two kinds of projection onto a set $\mathcal{S}$ of distributions, because KL divergence is not symmetric:

* the **e-projection** of $q$ onto $\mathcal{S}$ is the $p \in \mathcal{S}$ that minimizes $D_{KL}(p \| q)$, so the distribution being projected is the second argument; and
* the **m-projection** of $p$ onto $\mathcal{S}$ is the $q \in \mathcal{S}$ that minimizes $D_{KL}(p \| q)$, so the distribution being projected is the first argument.

Two kinds of "flat" sets of distributions serve as lines and planes:

* **linear families**, such as the set of distributions that give a statistic a prescribed mean; a linear family contains every mixture $(1 - a) p_0 + a p_1$ of two of its distributions, for every real $a$ for which the mixture is a probability distribution; and
* **exponential families**, described in {doc}`exponential_families`; an exponential family contains every normalized geometric mixture of two of its models, for every real $a$ for which the geometric mixture can be normalized.

The e-projection onto a linear family and the m-projection onto an exponential family are unique, and each satisfies a Pythagorean theorem.

The two projections have simple forms:

* the e-projection of $q$ onto a linear family defined by moment conditions is an **exponential tilt** of $q$, a distribution proportional to $q(x) e^{\theta \cdot t(x)}$ for a vector $\theta$ that satisfies the moment conditions; and
* the m-projection of $p$ onto an exponential family **matches moments**: it is the model in the family that gives the sufficient statistic the same means as $p$ does.

These two facts organize several ideas that economists use:

* maximum likelihood estimation, under correct specification and under misspecification (m-projections);
* mutual information, which measures dependence between random variables (an m-projection);
* maximum entropy, Sanov's theorem, and large deviations (e-projections);
* the worst-case models of robust control (e-projections);
* the RAS method for updating input–output tables, and Sinkhorn's algorithm for entropy-regularized optimal transport (repeated e-projections); and
* the Chernoff information that governs error rates in tests based on likelihood ratios (an e-projection of two distributions at once).

The worst-case models of robust control and the solutions of the RAS and Sinkhorn problems are exponential tilts of the kind that attains the logit formula {eq}`ef_logit_base` of {doc}`exponential_families`.

The multipliers of the entropy-regularized transport problem converge to the shadow prices of the linear program in {doc}`opt_transport`, and its entropy term charges for the mutual information between origins and destinations.

This lecture builds on {doc}`exponential_families` and on {doc}`divergence_measures`.

{cite:t}`Csiszar1975` developed the theory of e-projections, which he called I-projections.

{cite:t}`CsiszarShields2004` survey the theory and its statistical applications.

{cite:t}`Nielsen2018` is a short introduction to information projections, and {cite:t}`amari_nagaoka2000` treat them as part of information geometry.

We use the following imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import brentq, linprog, minimize, minimize_scalar
from scipy.special import logsumexp, betaln, digamma
from scipy.stats import norm, poisson, chi2
import warnings
```

We write $D(p \| q)$ for the KL divergence $D_{KL}(p \| q)$ and, as in {doc}`exponential_families`, read it as "the divergence of $p$ from $q$."

We write $q_\theta$ for the models in an exponential family, because they usually appear in the second argument of $D$.

Most of our examples use distributions on a finite set.

The functions below compute KL divergence, construct the exponential tilt that raises the probability of each outcome $x$ in proportion to $e^{\theta \cdot t(x)}$, and find the tilt that gives a scalar statistic a prescribed mean.

These generalize the functions of the same names in {doc}`exponential_families`: `tilt` also accepts vector-valued statistics, and `match_mean` accepts a bracket for its root finder.

```{code-cell} ipython3
def kl(p, q):
    """KL divergence of distributions (or arrays) p and q of equal shape."""
    p, q = np.ravel(p), np.ravel(q)
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("probabilities must be nonnegative")
    mask = p > 0                       # use the convention 0 log 0 = 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def tilt(q, t, θ):
    """The distribution proportional to q(x) exp(θ · t(x)).

    t is either a vector of values t(x) or an array whose rows are t(x).
    """
    T = np.reshape(t, (len(q), -1))
    log_w = np.log(q) + T @ np.atleast_1d(θ)
    return np.exp(log_w - logsumexp(log_w))


def match_mean(q, t, c, bracket=(-50, 50)):
    """Find θ such that the tilt of q has E[t(X)] = c, for scalar t."""
    θ = brentq(lambda θ: tilt(q, t, θ) @ t - c, *bracket)
    return θ, tilt(q, t, θ)


rng = np.random.default_rng(1234)
```

## A three-point identity

Let $p$, $q$, and $r$ be three probability distributions on a finite set, with $q$ and $r$ assigning positive probability to every outcome.

Expanding the definition of KL divergence gives the identity

$$
D(p \| r) = D(p \| q) + D(q \| r) + \sum_x \left( p(x) - q(x) \right) \left( \log q(x) - \log r(x) \right) .
$$ (ip_three_point)

The Euclidean analog is

$$
\lVert p - r \rVert^2 = \lVert p - q \rVert^2 + \lVert q - r \rVert^2 + 2 (p - q) \cdot (q - r) .
$$

The Euclidean cross term is an inner product of two displacements, and it vanishes when the segments from $q$ to $p$ and from $q$ to $r$ are perpendicular.

In {eq}`ip_three_point`, the cross term is an inner product of two displacements measured in different coordinates: $p - q$ is a displacement in probabilities, and $\log q - \log r$ is a displacement in log probabilities.

In the language of {doc}`exponential_families`, the distributions with full support on a finite set $\{x_0, x_1, \ldots\}$ form an exponential family, whose expectation parameters are the probabilities $p(x)$ and whose natural parameters are the log ratios $\log(p(x)/p(x_0))$, for $x \neq x_0$.

Probabilities are expectation parameters; log probabilities are, up to a normalization, natural parameters.

When the cross term vanishes, KL divergence satisfies a **Pythagorean theorem**,

$$
D(p \| r) = D(p \| q) + D(q \| r) .
$$

We check {eq}`ip_three_point` with randomly drawn distributions.

```{code-cell} ipython3
p, q, r = rng.dirichlet(np.ones(5), size=3)
lhs = kl(p, r)
rhs = kl(p, q) + kl(q, r) + (p - q) @ (np.log(q) - np.log(r))
print(f"D(p || r) = {lhs:.10f}")
print(f"right side = {rhs:.10f}")
```

The rest of this lecture is about sets of distributions on which the cross term vanishes.

We state and prove the theorems below for distributions on finite sets.

The identity {eq}`ip_three_point`, and everything we derive from it, also holds for densities, with integrals in place of sums, whenever the divergences are finite.

## Two kinds of straight lines

Two natural paths lead from a distribution $p_0$ to a distribution $p_1$.

The **mixture path** moves along a straight line in probabilities:

$$
p_a = (1 - a) \, p_0 + a \, p_1 .
$$

The **exponential path** moves along a straight line in log probabilities and then normalizes:

$$
\log p_a = (1 - a) \log p_0 + a \log p_1 - F(a),
$$

where $F(a) = \log \sum_x p_0(x)^{1 - a} p_1(x)^a$.

Values of $a$ in $[0, 1]$ trace out a path from $p_0$ to $p_1$; other values of $a$ extend the path beyond its endpoints, the exponential path for every real $a$ and the mixture path for as long as $p_a$ stays nonnegative.

The exponential path is a one-parameter exponential family with sufficient statistic $\log(p_1 / p_0)$ and natural parameter $a$.

A **geodesic** is a geometry's analog of a straight line.

Information geometers call these paths **m-geodesics**, for mixture, and **e-geodesics**, for exponential.

In the coordinates described after {eq}`ip_three_point`, an m-geodesic is a straight line in expectation parameters, and an e-geodesic is a straight line in natural parameters.

A set of distributions is **m-flat** if, whenever it contains $p_0$ and $p_1$, it contains $p_a$ on the mixture path for every real $a$ for which $p_a$ is a probability distribution, not only for $a \in [0, 1]$; it is **e-flat** if the same holds for exponential paths.

A set that contains the mixture segments with $a \in [0, 1]$, but not necessarily the whole lines through its distributions, is merely convex in the ordinary sense.

The e-projection onto a closed convex set is still unique, but the Pythagorean identity {eq}`ip_pythagoras_e` below weakens to the inequality $D(p \| q) \geq D(p \| p^*) + D(p^* \| q)$ {cite}`Csiszar1975`.

Two kinds of sets are the leading examples.

* A **linear family** $\mathcal{L} = \{ p : E_p[t(X)] = c \}$, the set of distributions that give a statistic $t$ a prescribed mean, is m-flat, because if $p_0$ and $p_1$ satisfy $E[t(X)] = c$, so does $(1 - a) p_0 + a p_1$ for every real $a$.
* An **exponential family** $\mathcal{E} = \{ q_\theta : q_\theta(x) \propto h(x) \exp(\theta \cdot t(x)) \}$ is e-flat, because the logarithm of a normalized geometric mixture of $q_{\theta_0}$ and $q_{\theta_1}$ is $\log h + ((1 - a) \theta_0 + a \theta_1) \cdot t$ plus a constant, for every real $a$ for which the normalized geometric mixture exists.

A linear family is a straight line or plane in expectation parameters, the probabilities.

When $h > 0$ everywhere, an exponential family is a straight line or plane in natural parameters, because $\log(q_\theta(x) / q_\theta(x_0)) = \log(h(x)/h(x_0)) + \theta \cdot (t(x) - t(x_0))$ is an affine function of $\theta$.

Distributions on three outcomes live in a triangle, the probability simplex, in which we can draw both kinds of sets.

The function below maps a distribution on three outcomes to a point in an equilateral triangle whose corners represent the three outcomes.

Each corner is the distribution that puts probability one on an outcome, the center is the uniform distribution, and the probability that a point $p$ assigns to outcome $x$ is proportional to the distance from $p$ to the side opposite corner $x$.

```{code-cell} ipython3
corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def to_triangle(p):
    """Map distributions on three outcomes (rows of p) to the plane."""
    return np.atleast_2d(p) @ corners


def draw_simplex(ax, labels=("$x = 0$", "$x = 1$", "$x = 2$")):
    ax.plot(*np.vstack([corners, corners[0]]).T, color="0.6", lw=1)
    offsets = [(-0.07, -0.05), (0.02, -0.05), (-0.03, 0.03)]
    for corner, label, (dx, dy) in zip(corners, labels, offsets):
        ax.text(corner[0] + dx, corner[1] + dy, label)
    ax.set_aspect("equal")
    ax.axis("off")


def mixture_path(p0, p1, a):
    return (1 - a)[:, None] * p0 + a[:, None] * p1


def exponential_path(p0, p1, a):
    log_w = (1 - a)[:, None] * np.log(p0) + a[:, None] * np.log(p1)
    return np.exp(log_w - logsumexp(log_w, axis=1, keepdims=True))


def fisher_rao_path(p0, p1, a):
    """Shortest path for the Fisher–Rao distance: a great circle in √p."""
    φ = np.arccos(np.sum(np.sqrt(p0 * p1)))
    z = (np.sin((1 - a)[:, None] * φ) * np.sqrt(p0)
         + np.sin(a[:, None] * φ) * np.sqrt(p1)) / np.sin(φ)
    return z**2
```

The figure draws both paths between two distributions, together with a third path that we describe below.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Mixture, exponential, and Fisher–Rao paths
    name: fig-ip-paths
---
p0 = np.array([0.80, 0.15, 0.05])
p1 = np.array([0.05, 0.15, 0.80])
a = np.linspace(0, 1, 200)

fig, ax = plt.subplots(figsize=(6, 5.5))
draw_simplex(ax)
ax.plot(*to_triangle(mixture_path(p0, p1, a)).T, lw=2, label="mixture path")
ax.plot(*to_triangle(exponential_path(p0, p1, a)).T, lw=2,
        label="exponential path")
ax.plot(*to_triangle(fisher_rao_path(p0, p1, a)).T, lw=2, ls="--",
        label="Fisher–Rao path")
ax.plot(*to_triangle(np.vstack([p0, p1])).T, "ko")
ax.text(*to_triangle(p0)[0] + [0.02, 0.0], "$p_0$")
ax.text(*to_triangle(p1)[0] + [0.02, 0.0], "$p_1$")
ax.legend(loc="upper right")
plt.show()
```

We draw the simplex in probability coordinates, so the mixture path is straight.

The exponential path bends toward the middle outcome.

A geometric mean of two probabilities is small whenever either of them is small, which penalizes outcomes $0$ and $2$, each of which is improbable under one of the two distributions.

The third path is the shortest path for the Fisher–Rao distance of {doc}`exponential_families`.

The map $p \mapsto 2 \sqrt{p}$ sends the simplex to part of a sphere of radius $2$, and, as {eq}`ef_fisher_rao_all` describes, the shortest path is an arc of a great circle, with length $2 \arccos \sum_x \sqrt{p_0(x) p_1(x)}$.

The function `fisher_length` measures the length of each path with the Fisher information metric, under which a small step $dp$ from $p$ has squared length $\sum_x dp(x)^2 / p(x)$.

```{code-cell} ipython3
def fisher_length(path):
    """Fisher–Rao length of a path whose rows are distributions."""
    dp = np.diff(path, axis=0)
    mid = (path[1:] + path[:-1]) / 2
    return np.sum(np.sqrt(np.sum(dp**2 / mid, axis=1)))


a_fine = np.linspace(0, 1, 2001)
for name, path in (("mixture", mixture_path), ("exponential", exponential_path),
                   ("Fisher–Rao", fisher_rao_path)):
    points = path(p0, p1, a_fine)
    print(f"{name:12} path: length {fisher_length(points):.4f}, "
          f"midpoint {np.round(points[1000], 3)}")
print(f"2 arccos of the Bhattacharyya coefficient: "
      f"{2 * np.arccos(np.sum(np.sqrt(p0 * p1))):.4f}")
```

The mixture and exponential paths are straight lines in their own coordinates, but neither is the shortest path for the Fisher–Rao distance.

The midpoint of the Fisher–Rao path puts probability $0.194$ on the middle outcome, between the $0.15$ of the mixture path and the $0.273$ of the exponential path.

## Two projections and two Pythagorean theorems

### The e-projection onto a linear family

Let $q$ be a distribution with full support, and let $\mathcal{L} = \{ p : E_p[t(X)] = c \}$ be a linear family that contains a distribution with full support.

This holds exactly when $c$ lies in the **relative interior** of the convex hull of the points $t(x)$, that is, in its interior as a subset of the smallest affine subspace that contains it.

The same condition guarantees that the tilt in the theorem below exists.

```{prf:theorem} e-projection
:label: ip_thm_e_projection

The distribution $p^* \in \mathcal{L}$ that minimizes $D(p \| q)$ over $p \in \mathcal{L}$ is the exponential tilt

$$
p^*(x) = q(x) \exp\left( \theta \cdot t(x) - F(\theta) \right),
\qquad
F(\theta) = \log \sum_x q(x) \exp(\theta \cdot t(x)) ,
$$

with $\theta$ chosen so that $E_{p^*}[t(X)] = c$.

Moreover, for every $p \in \mathcal{L}$,

$$
D(p \| q) = D(p \| p^*) + D(p^* \| q) .
$$ (ip_pythagoras_e)
```

```{prf:proof}
Apply {eq}`ip_three_point` with $(p, p^*, q)$ in place of $(p, q, r)$.

The cross term is

$$
\sum_x \left( p(x) - p^*(x) \right) \left( \theta \cdot t(x) - F(\theta) \right) = \theta \cdot (c - c) - F(\theta) (1 - 1) = 0 ,
$$

which proves {eq}`ip_pythagoras_e`.

Since $D(p \| p^*) > 0$ unless $p = p^*$, the tilt $p^*$ is the unique minimizer.
```

When every distribution in $\mathcal{L}$ puts zero probability on some outcomes, as for a die whose mean is $6$, the e-projection still exists and is unique.

It lies outside the family of tilts, as a limit of tilts whose $\theta$ diverges.

The e-projection of $q$ lies on the exponential family generated by $q$ and the statistic $t$, at the point where that family crosses $\mathcal{L}$.

In the proof, the cross term pairs a displacement $p - p^*$ in probabilities, which lies along $\mathcal{L}$, with a displacement $\log p^* - \log q$ in log probabilities, which lies along the e-geodesic from $q$ to $p^*$.

This pairing is exactly the inner product defined by the Fisher information at $p^*$.

For tangent vectors $u$ and $v$ at $p^*$, expressed in probabilities, that inner product is $\sum_x u(x) v(x) / p^*(x)$.

Take $u = p - p^*$, the direction of $\mathcal{L}$ toward $p$, and take $v$ to be the tangent at $p^*$ of the e-geodesic $p_s \propto q^{1 - s} (p^*)^s$.

Then $v(x)/p^*(x) = \log p^*(x) - \log q(x) - D(p^* \| q)$, and the constant $D(p^* \| q)$ drops out of the inner product because $\sum_x u(x) = 0$.

The vanishing of the cross term is the precise sense in which the e-geodesic from $q$ meets $\mathcal{L}$ at a right angle.

We call $p^*$ the e-projection because $q$ reaches $\mathcal{L}$ along an e-geodesic.

When $q$ is the uniform distribution on $m$ outcomes, $D(p \| q) = \log m - H(p)$.

The e-projection of the uniform distribution is then the maximum entropy distribution of {doc}`exponential_families`, and {eq}`ip_pythagoras_e` becomes the entropy-gap formula proved there.

For a general $q$, the e-projection maximizes entropy relative to $q$: it minimizes $D(p \| q)$, the **relative entropy** of $p$ with respect to $q$.

### The m-projection onto an exponential family

Now let $\mathcal{E} = \{ q_\theta \}$ be an exponential family with sufficient statistic $t$ and base measure $h$, and let $p$ be any distribution that puts zero probability wherever $h$ does.

Without that condition, $D(p \| q_\theta)$ would be infinite for every $\theta$.

```{prf:theorem} m-projection
:label: ip_thm_m_projection

If $q_{\theta^*} \in \mathcal{E}$ matches the means of the sufficient statistic,

$$
E_{q_{\theta^*}}[t(X)] = E_p[t(X)] ,
$$

then $q_{\theta^*}$ is the unique minimizer of $D(p \| q_\theta)$ over $\theta$, and, for every $\theta$,

$$
D(p \| q_\theta) = D(p \| q_{\theta^*}) + D(q_{\theta^*} \| q_\theta) .
$$ (ip_pythagoras_m)
```

```{prf:proof}
Apply {eq}`ip_three_point` with $(p, q_{\theta^*}, q_\theta)$ in place of $(p, q, r)$.

Because $\log q_{\theta^*}(x) - \log q_\theta(x) = (\theta^* - \theta) \cdot t(x) - F(\theta^*) + F(\theta)$, the cross term is

$$
(\theta^* - \theta) \cdot \left( E_p[t(X)] - E_{q_{\theta^*}}[t(X)] \right) = 0 ,
$$

which proves {eq}`ip_pythagoras_m`.

Since $D(q_{\theta^*} \| q_\theta) > 0$ unless $\theta = \theta^*$, $q_{\theta^*}$ is the unique minimizer.
```

Conversely, every minimizer matches moments.

Writing out the definitions gives $D(p \| q_\theta) = F(\theta) - \theta \cdot E_p[t(X)]$ plus terms that do not depend on $\theta$, so the first-order condition for a minimum is $\nabla F(\theta) = E_p[t(X)]$.

An m-projection exists if and only if $E_p[t(X)]$ lies in the range of $\nabla F$.

For example, the m-projection onto the Bernoulli family of a distribution that puts all its probability on $1$ does not exist.

For the same reason, a Bernoulli sample with no failures has no maximum likelihood estimate of the natural parameter.

The m-projection of $p$ lies on the linear family of distributions that share $p$'s means of $t$, at the point where that family crosses $\mathcal{E}$.

Here the roles are reversed: the cross term pairs a displacement in probabilities along the m-geodesic from $p$ to $q_{\theta^*}$, which stays inside that linear family, with a displacement in natural parameters along $\mathcal{E}$.

The m-geodesic from $p$ meets $\mathcal{E}$ at a right angle.

We call $q_{\theta^*}$ the m-projection because $p$ reaches $\mathcal{E}$ along an m-geodesic.

In both theorems the projection is the vertex at the right angle, the point that appears in both terms on the right side.

```{note}
Conventions differ across the literature.

Check which argument of the KL divergence a source minimizes and which orthogonality condition it attaches to which Pythagorean identity.

Let $p$, $q$, and $r$ be models in an exponential family with log-partition function $F$ and Bregman divergence $B_F$, as in {doc}`exponential_families`, with natural parameters $\theta_p$, $\theta_q$, $\theta_r$ and expectation parameters $\eta_p$, $\eta_q$, $\eta_r$.

Then

$$
B_F(\theta_p, \theta_r) = B_F(\theta_p, \theta_q) + B_F(\theta_q, \theta_r)
\iff (\theta_p - \theta_q) \cdot (\eta_q - \eta_r) = 0
$$

and

$$
D(p \| r) = D(p \| q) + D(q \| r)
\iff (\eta_p - \eta_q) \cdot (\theta_q - \theta_r) = 0 .
$$

The two identities are the same statement with $p$ and $r$ interchanged: by {eq}`ef_kl_bregman` of {doc}`exponential_families`, the first says $D(r \| p) = D(r \| q) + D(q \| p)$.

Writing the letters in the same order in the two notations attaches different orthogonality conditions.

{cite:t}`Nielsen2018` (in the text on pp. 322–323 and in the formula printed in Figure 2) and {cite:t}`Nielsen2020` (in the panel labels of Figure 6) pair each identity with the other's condition; their definitions of e- and m-projections, and Theorem 6 of {cite:t}`Nielsen2020`, are correct.
```

Let's compute both projections for distributions on three outcomes with $t(x) = x$.

For the e-projection, we take $q = (0.6, 0.3, 0.1)$ and the linear family of distributions with mean $1.2$.

For the m-projection, we take $p = (0.5, 0.1, 0.4)$ and the exponential family generated by $h = (0.2, 0.5, 0.3)$ and $t$.

```{code-cell} ipython3
outcomes = np.arange(3.0)

# e-projection of q onto {p : E_p[X] = 1.2}
q_e = np.array([0.6, 0.3, 0.1])
θ_e, p_star = match_mean(q_e, outcomes, 1.2)

# distributions in the linear family: p_star plus multiples of (1, -2, 1),
# small enough to keep every probability positive
s_lo, s_hi = -min(p_star[0], p_star[2]), p_star[1] / 2
print("e-projection: D(p || q) and D(p || p*) + D(p* || q)")
for s in np.linspace(0.9 * s_lo, 0.9 * s_hi, 4):
    p = p_star + s * np.array([1.0, -2.0, 1.0])
    print(f"  {kl(p, q_e):.8f}   {kl(p, p_star) + kl(p_star, q_e):.8f}")

# m-projection of p onto the exponential family generated by h
h = np.array([0.2, 0.5, 0.3])
p_m = np.array([0.5, 0.1, 0.4])
θ_m, q_star = match_mean(h, outcomes, p_m @ outcomes)
print("\nm-projection: D(p || q_θ) and D(p || q*) + D(q* || q_θ)")
for θ in (-1.0, 0.0, 0.5, 2.0):
    q_θ = tilt(h, outcomes, θ)
    print(f"  {kl(p_m, q_θ):.8f}   {kl(p_m, q_star) + kl(q_star, q_θ):.8f}")
```

The figure shows the two projections.

In the left panel, the straight line is the linear family, and the curve through $q$ is the exponential family generated by $q$; the e-projection $p^*$ is where they cross.

The thin curves are contours of $D(p \| q)$ as a function of $p$; the linear family is tangent to the contour through $p^*$.

In the right panel, the curve is the exponential family generated by $h$, and the straight line through $p$ is the linear family of distributions that share $p$'s mean; the m-projection $q^*$ is where they cross.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The e-projection and the m-projection
    name: fig-ip-projections
---
θ_grid = np.linspace(-6, 6, 400)


def family_curve(base, t, θ_grid):
    """Models in the exponential family generated by base and t."""
    return np.vstack([tilt(base, t, θ) for θ in θ_grid])


def linear_family(c, n=400):
    """Distributions on {0, 1, 2} with mean c."""
    lo, hi = max(0, c - 1), min(1, c / 2)   # range of p(2)
    p2 = np.linspace(lo, hi, n)
    p1 = c - 2 * p2
    return np.column_stack([1 - p1 - p2, p1, p2])


fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
draw_simplex(ax)
# a barycentric grid of distributions with full support
k = 300
i, j = np.meshgrid(np.arange(1, k), np.arange(1, k))
inside = i + j < k
grid = np.column_stack([i[inside], j[inside], k - i[inside] - j[inside]]) / k
xy = to_triangle(grid)
ax.tricontour(xy[:, 0], xy[:, 1],
              np.sum(grid * np.log(grid / q_e), axis=1),
              levels=kl(p_star, q_e) * np.array([0.25, 0.5, 1.0, 1.5, 2.0]),
              colors="0.75", linewidths=0.8)
ax.plot(*to_triangle(linear_family(1.2)).T, lw=2,
        label=r"linear family $E[X] = 1.2$")
ax.plot(*to_triangle(family_curve(q_e, outcomes, θ_grid)).T, lw=2,
        label="exponential family through $q$")
for point, name in ((q_e, "$q$"), (p_star, "$p^*$")):
    ax.plot(*to_triangle(point)[0], "ko")
    ax.text(*to_triangle(point)[0] + [0.02, 0.01], name)
ax.set_title("e-projection", pad=18)
ax.legend(loc="upper left", fontsize=9, frameon=False)

ax = axes[1]
draw_simplex(ax)
ax.plot(*to_triangle(family_curve(h, outcomes, θ_grid)).T, lw=2, color="C1",
        label="exponential family through $h$")
ax.plot(*to_triangle(linear_family(p_m @ outcomes)).T, lw=2, color="C0",
        label=rf"linear family $E[X] = {p_m @ outcomes:.1f}$")
for point, name in ((p_m, "$p$"), (q_star, "$q^*$"), (h, "$h$")):
    ax.plot(*to_triangle(point)[0], "ko")
    ax.text(*to_triangle(point)[0] + [0.02, 0.01], name)
ax.set_title("m-projection", pad=18)
ax.legend(loc="upper left", fontsize=9, frameon=False)
plt.tight_layout()
plt.show()
```

## Maximum likelihood is an m-projection

### Correct specification

Let $x_1, \ldots, x_n$ be IID draws, and suppose that we fit an exponential family $\{ q_\theta \}$ by maximum likelihood.

{doc}`exponential_families` shows that the maximum likelihood estimator $\hat \theta$ matches the sample means of the sufficient statistic.

When the outcomes are discrete, let $\hat p$ be the empirical distribution of the sample.

Then the average log likelihood is $\sum_x \hat p(x) \log q_\theta(x) = -H(\hat p) - D(\hat p \| q_\theta)$.

Maximizing the likelihood is therefore minimizing $D(\hat p \| q_\theta)$, because $H(\hat p)$ does not depend on $\theta$.

By {prf:ref}`ip_thm_m_projection`, *the maximum likelihood estimate is the m-projection of the empirical distribution onto the model.*

Applying the Pythagorean theorem {eq}`ip_pythagoras_m` with $\hat p$ in the role of $p$ gives $D(\hat p \| q_\theta) - D(\hat p \| q_{\hat \theta}) = D(q_{\hat\theta} \| q_\theta)$, so that, for every $\theta$,

$$
\frac{1}{n} \left[ \ell_n(\hat \theta) - \ell_n(\theta) \right] = D(q_{\hat\theta} \| q_\theta) ,
$$ (ip_lr_kl)

where $\ell_n$ is the log likelihood.

The identity {eq}`ip_lr_kl` holds for continuous outcomes too: by the formulas of {doc}`exponential_families`, the left side equals $(\hat\theta - \theta) \cdot \bar t - F(\hat\theta) + F(\theta)$, and since $\bar t = \nabla F(\hat \theta)$, this is the Bregman divergence $B_F(\theta, \hat\theta) = D(q_{\hat\theta} \| q_\theta)$.

For an exponential family, whenever the maximum likelihood estimate exists, *the likelihood-ratio statistic $2[\ell_n(\hat\theta) - \ell_n(\theta_0)]$ equals $2 n D(q_{\hat\theta} \| q_{\theta_0})$ exactly.*

By Wilks' theorem, when $\theta_0$ is the true parameter this statistic is approximately $\chi^2$ with as many degrees of freedom as there are parameters.

We check this with Poisson data, for which $\hat \lambda = \bar x$ and $D(q_{\hat\lambda} \| q_{\lambda_0}) = \hat\lambda \log(\hat\lambda/\lambda_0) - \hat\lambda + \lambda_0$.

```{code-cell} ipython3
λ0, n, R = 3.0, 50, 20_000
x = rng.poisson(λ0, size=(R, n))
λ_hat = x.mean(axis=1)

lr = 2 * (poisson.logpmf(x, λ_hat[:, None]).sum(axis=1)
          - poisson.logpmf(x, λ0).sum(axis=1))
two_n_kl = 2 * n * (λ_hat * np.log(λ_hat / λ0) - λ_hat + λ0)
print(f"max |LR - 2n KL| = {np.max(np.abs(lr - two_n_kl)):.2e}")
print(f"P(LR > 3.84) = {np.mean(lr > chi2.ppf(0.95, 1)):.4f}  (χ²_1: 0.05)")
```

The lecture {doc}`likelihood_ratio_process` compares two fixed models; by {eq}`ip_lr_kl`, within an exponential family, the KL divergence between the best-fitting model and any other model governs the likelihood ratio between them.

### Misspecification

Suppose now that the data come from a distribution $p_0$ outside the model.

As {doc}`divergence_measures` explains, under regularity conditions the maximum likelihood estimator then converges to the **pseudo-true** parameter $\theta^*$ that minimizes $D(p_0 \| q_\theta)$ {cite}`White1982`.

{prf:ref}`ip_thm_m_projection` identifies it: for an exponential family, the pseudo-true parameter matches the population means of the sufficient statistic.

When we fit a normal distribution to data drawn from a mixture of normals, for example, the pseudo-true normal has the mean and variance of the mixture.

Let's verify this, and the Pythagorean theorem {eq}`ip_pythagoras_m`, by numerical integration.

```{code-cell} ipython3
w, (μa, σa), (μb, σb) = 0.3, (-2.0, 0.7), (1.0, 1.0)


def log_mixture_pdf(x):
    return np.logaddexp(np.log(w) + norm.logpdf(x, μa, σa),
                        np.log(1 - w) + norm.logpdf(x, μb, σb))


def mixture_pdf(x):
    return np.exp(log_mixture_pdf(x))


def kl_mixture_normal(μ, σ):
    def integrand(x):
        log_p = log_mixture_pdf(x)
        return np.exp(log_p) * (log_p - norm.logpdf(x, μ, σ))
    return integrate.quad(integrand, -np.inf, np.inf, limit=200)[0]


def kl_normal(μ1, σ1, μ2, σ2):
    return np.log(σ2 / σ1) + (σ1**2 + (μ1 - μ2)**2) / (2 * σ2**2) - 0.5


μ_star = w * μa + (1 - w) * μb
σ_star = np.sqrt(w * (σa**2 + μa**2) + (1 - w) * (σb**2 + μb**2) - μ_star**2)
print(f"moment-matched normal: μ* = {μ_star:.4f}, σ* = {σ_star:.4f}")

print("\n   μ     σ    D(p0 || N)   D(p0 || N*) + D(N* || N)")
for μ, σ in ((0.0, 1.5), (μ_star, 1.0), (1.0, 2.0), (μ_star, σ_star)):
    lhs = kl_mixture_normal(μ, σ)
    rhs = kl_mixture_normal(μ_star, σ_star) + kl_normal(μ_star, σ_star, μ, σ)
    print(f"{μ:6.2f} {σ:5.2f} {lhs:12.6f} {rhs:16.6f}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Moment-matched normal for a bimodal mixture
    name: fig-ip-mixture
---
x_vals = np.linspace(-5, 5, 400)
fig, ax = plt.subplots()
ax.plot(x_vals, mixture_pdf(x_vals), lw=2, label="mixture $p_0$")
ax.plot(x_vals, norm.pdf(x_vals, μ_star, σ_star), lw=2, ls="--",
        label=rf"m-projection $N(\mu^*, \sigma^{{*2}})$")
ax.set_xlabel("$x$")
ax.legend()
plt.show()
```

The moment-matched normal is the best normal approximation in the sense of $D(p_0 \| q)$, even though it puts substantial probability between the two modes, where the mixture puts little.

The lecture {doc}`mix_model` studies how a Bayesian learner who uses the wrong model behaves.

{doc}`Gallant and Tauchen's efficient method of moments <gallant_tauchen>` builds an estimator on the fact that the expected score of the auxiliary model vanishes at the pseudo-true value.

## Independence and mutual information

Let $p(x, y)$ be a joint distribution of two random variables that take finitely many values, with marginal distributions $p_X$ and $p_Y$.

The set of joint distributions under which $X$ and $Y$ are independent,

$$
\mathcal{E}_{\text{ind}} = \{ q_1(x) q_2(y) \} ,
$$

is an exponential family: its sufficient statistics are the indicators of the values of $x$ and of $y$ separately, and the means of these statistics are the two marginal distributions.

By {prf:ref}`ip_thm_m_projection`, the m-projection of $p$ onto $\mathcal{E}_{\text{ind}}$ is the product of the marginals, $p_X p_Y$.

The minimized divergence is the **mutual information**

$$
I(X; Y) = D(p \| p_X p_Y) = \min_{q_1, q_2} D(p \| q_1 q_2) ,
$$

the KL divergence of $p$ from the nearest joint distribution under which $X$ and $Y$ are independent.

For any marginal distributions $q_1$ and $q_2$, the Pythagorean theorem {eq}`ip_pythagoras_m` gives

$$
D(p \| q_1 q_2) = D(p \| p_X p_Y) + D(p_X p_Y \| q_1 q_2) ,
$$

and, because the logarithm of a product is a sum, the last term splits into $D(p_X \| q_1) + D(p_Y \| q_2)$.

So

$$
D(p \| q_1 q_2) = I(X; Y) + D(p_X \| q_1) + D(p_Y \| q_2) ,
$$

which splits the divergence of $p$ from an independent model into a part due to dependence and parts due to the marginals.

```{code-cell} ipython3
p_xy = np.array([[0.20, 0.05, 0.05],
                 [0.05, 0.25, 0.05],
                 [0.02, 0.08, 0.25]])
p_x, p_y = p_xy.sum(axis=1), p_xy.sum(axis=0)
mutual_info = kl(p_xy, np.outer(p_x, p_y))
print(f"I(X; Y) = {mutual_info:.6f}")

for _ in range(3):
    q1, q2 = rng.dirichlet(np.ones(3), size=2)
    lhs = kl(p_xy, np.outer(q1, q2))
    rhs = mutual_info + kl(p_x, q1) + kl(p_y, q2)
    print(f"D(p || q1 q2) = {lhs:.6f},  decomposition = {rhs:.6f}")
```

Mutual information is the measure of information flow in the theory of rational inattention initiated by {cite:t}`Sims2003`.

{ref}`ip_ex_interactions` extends the decomposition to three random variables.

## Maximum entropy and large deviations

The maximum entropy distribution $p^*$ of a die whose mean is $4.5$ is the e-projection of the uniform distribution $q$ onto the linear family $\{ p : E_p[X] = 4.5 \}$.

We recompute it because we need it below.

```{code-cell} ipython3
faces = np.arange(1.0, 7.0)
q_die = np.full(6, 1 / 6)
θ_die, p_die = match_mean(q_die, faces, 4.5)
print("e-projection of the uniform distribution:", np.round(p_die, 4))
print(f"D(p* || q) = {kl(p_die, q_die):.4f}")
```

Large deviations theory explains why the e-projection answers Jaynes's question.

**Sanov's theorem** concerns the empirical distribution $\hat p_n$ of $n$ IID draws from $q$.

It says that, for a set $\mathcal{S}$ of distributions that satisfies a mild regularity condition (for example, that $\mathcal{S}$ is the closure of its interior),

$$
\lim_{n \to \infty} -\frac{1}{n} \log P\left( \hat p_n \in \mathcal{S} \right) = \min_{p \in \mathcal{S}} D(p \| q) = D(p_{\mathcal{S}} \| q) ,
$$

where $p_{\mathcal{S}}$ is the e-projection of $q$ onto $\mathcal{S}$.

The probability that the empirical distribution lands in $\mathcal{S}$ decays exponentially, at the rate $D(p_{\mathcal{S}} \| q)$.

When we roll a fair die $n$ times, an average of at least $4.5$ is the event that $\hat p_n$ lies in the half-space $\mathcal{S} = \{ p : E_p[X] \geq 4.5 \}$.

Because $q$ has mean $3.5$, the e-projection of $q$ onto this half-space lies on its boundary, the linear family $\{ p : E_p[X] = 4.5 \}$.

Along the straight line from any $p$ with a larger mean toward $q$, $D(\cdot \| q)$ falls, because it is convex and zero at $q$; the line passes through a point with mean $4.5$.

The e-projection is therefore $p^*$, and

$$
\lim_{n \to \infty} -\frac{1}{n} \log P\left( \frac{1}{n} \sum_{i=1}^n X_i \geq 4.5 \right) = D(p^* \| q) .
$$

The same calculation says how the rare event is most likely to happen: among all the ways that $n$ fair rolls could produce an average of $4.5$, overwhelmingly the most likely is that the frequencies of the faces resemble $p^*$.

The **conditional limit theorem** makes this precise.

Suppose that we roll a fair die $n$ times and learn only that the average number of spots was exactly $4.5$, which requires $n$ to be even.

Given this information, the distribution of the number of spots on the first roll converges to the e-projection $p^*$ as $n \to \infty$ {cite}`VanCampenhoutCover1981,Csiszar1984`.

Chapter 11 of {cite:t}`CoverThomas2006` proves both theorems.

We can check them exactly.

The distribution of the sum $S_n$ of $n$ rolls is an $n$-fold convolution, and

$$
P(X_1 = k \mid S_n = s) = \frac{P(X_1 = k) \, P(S_{n - 1} = s - k)}{P(S_n = s)} ,
$$

the formula on which {cite:t}`VanCampenhoutCover1981` base their proof.

```{code-cell} ipython3
def sum_distribution(pmf, n):
    """Distribution of the sum of n IID draws from pmf on {1, ..., m}.

    Entry j of the result is the probability that the sum equals j + n.
    """
    dist = np.array([1.0])
    for _ in range(n):
        dist = np.convolve(dist, pmf)
    return dist


print("    n   P(X_1 = k | average = 4.5), k = 1, ..., 6     "
      "-log P(average ≥ 4.5)/n")
n_vals = [10, 20, 40, 80, 160, 320, 640]
rates = []
for n in n_vals:
    s = int(4.5 * n)
    dist_n1 = sum_distribution(q_die, n - 1)
    dist_n = np.convolve(dist_n1, q_die)
    cond = np.array([q_die[k - 1] * dist_n1[s - k - (n - 1)]
                     for k in range(1, 7)]) / dist_n[s - n]
    rates.append(-np.log(dist_n[s - n:].sum()) / n)
    print(f"{n:5d}   {np.round(cond, 4)}   {rates[-1]:.4f}")
print(f"  p* = {np.round(p_die, 4)}   D(p* || q) = {kl(p_die, q_die):.4f}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Decay rate converges to $D(p^* \| q)$
    name: fig-ip-sanov
---
fig, ax = plt.subplots()
ax.plot(n_vals, rates, "o-", label=r"$-\frac{1}{n}\log P(\bar X_n \geq 4.5)$")
ax.axhline(kl(p_die, q_die), color="k", ls="--", lw=1,
           label=r"$D(p^* \| q)$")
ax.set_xscale("log")
ax.set_xlabel("number of rolls $n$")
ax.legend()
plt.show()
```

The conditional distributions converge quickly to $p^*$; the decay rates converge more slowly, because the probabilities also contain factors that are polynomial in $n$.

The large deviations results in {doc}`likelihood_ratio_process` and in {doc}`the lecture on merging of opinions <merging_of_opinions>` rest on the same mathematics.

## Worst-case models in robust control

Let $\pi$ be a baseline probability distribution over states $x$, and let $u(x)$ be a decision maker's utility.

A decision maker who fears that $\pi$ is misspecified can evaluate prospects by the worst expected utility over alternative distributions $q$, penalizing alternatives by their KL divergence from $\pi$:

$$
\min_q \left\{ E_q[u(X)] + \lambda \, D(q \| \pi) \right\} ,
$$ (ip_multiplier)

where $\lambda > 0$ measures the decision maker's confidence in $\pi$.

These are the multiplier preferences of {cite:t}`HansenSargent2001`.

To solve {eq}`ip_multiplier`, define the exponential tilt

$$
q^*_\lambda(x) = \frac{\pi(x) \exp(-u(x)/\lambda)}{E_\pi \exp(-u(X)/\lambda)} .
$$

Because $\log q^*_\lambda(x) = \log \pi(x) - u(x)/\lambda - \log E_\pi \exp(-u(X)/\lambda)$, we have

$$
\lambda \, D(q \| q^*_\lambda) = \lambda \, D(q \| \pi) + E_q[u(X)] + \lambda \log E_\pi \exp\left( -u(X)/\lambda \right) ,
$$

or, for every $q$,

$$
E_q[u(X)] + \lambda \, D(q \| \pi) = \lambda \, D(q \| q^*_\lambda) - \lambda \log E_\pi \exp\left( -u(X)/\lambda \right) ,
$$

so the worst-case distribution is $q^*_\lambda$ and the value of {eq}`ip_multiplier` is $-\lambda \log E_\pi \exp(-u(X)/\lambda)$.

Each worst-case model $q^*_\lambda$ is also the e-projection of $\pi$ onto the linear family $\{ q : E_q[u(X)] = E_{q^*_\lambda}[u(X)] \}$: among all distributions that deliver that expected utility, it is closest to $\pi$.

The value $-\lambda \log E_\pi \exp(-u(X)/\lambda)$ is the risk-sensitive value function of {cite:t}`Whittle_1990` and {cite:t}`HansenSargent2008`.

It is also an instance of the logit formula {eq}`ef_logit_base` of {doc}`exponential_families`, with $v = -u$ and $\tau = \lambda$.

The letters trade places: the baseline $\pi$ here is the base distribution $q$ there, and the distribution $q$ chosen here is the vector $\pi$ there.

Minimizing $E_q[u] + \lambda D(q \| \pi)$ is maximizing $E_q[-u] - \lambda D(q \| \pi)$, so the value of {eq}`ip_multiplier` is minus the left side of {eq}`ef_logit_base`.

The worst-case models form a one-parameter exponential family that passes through $\pi$, with sufficient statistic $u$ and natural parameter $-1/\lambda$.

As $\lambda$ falls, the decision maker trusts $\pi$ less, and the worst case moves along this family, putting more probability on states with low utility.

The example below has log utility over consumption, $u(x) = \log x$, with the logarithm of consumption distributed $N(0, 0.2^2)$ under $\pi$.

Write $z = \log x$; tilting the $N(0, 0.2^2)$ density of $z$ by $\exp(-z/\lambda)$ gives the $N(-0.2^2/\lambda, 0.2^2)$ density, so each worst case is again lognormal.

Then $E_{q^*_\lambda}[u] = -0.2^2/\lambda$ and $D(q^*_\lambda \| \pi) = 0.2^2/(2 \lambda^2)$.

We discretize log consumption on a grid wide enough to contain the worst cases we consider, and compare the results with these formulas.

```{code-cell} ipython3
σ_c = 0.2
log_c = np.linspace(-2.8, 0.8, 91)
π_base = norm.pdf(log_c, 0, σ_c)
π_base /= π_base.sum()
u_c = log_c                                # u(c) = log c


def worst_case(λ):
    return tilt(π_base, u_c, -1 / λ)


def multiplier_value(λ):
    return -λ * logsumexp(-u_c / λ, b=π_base)


λ_shown = (0.2, 0.05, 0.02)
print("    λ   E_q[u]  exact   D(q || π)   exact   E_q[u] + λ D   "
      "-λ log E exp(-u/λ)")
for λ in λ_shown:
    q = worst_case(λ)
    print(f"{λ:5.2f} {q @ u_c:8.4f} {-σ_c**2 / λ:6.2f} {kl(q, π_base):10.4f} "
          f"{σ_c**2 / (2 * λ**2):7.2f} {q @ u_c + λ * kl(q, π_base):12.4f} "
          f"{multiplier_value(λ):16.4f}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Worst-case distributions under multiplier preferences
    name: fig-ip-worst-case
---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(log_c, π_base, "k", lw=2, label=r"baseline $\pi$")
for λ in λ_shown:
    ax.plot(log_c, worst_case(λ), label=rf"worst case, $\lambda$ = {λ}")
ax.set_xlabel("log consumption")
ax.set_ylabel("probability")
ax.legend()

ax = axes[1]
λ_vals = np.geomspace(0.02, 2, 100)
rel_entropies = [kl(worst_case(λ), π_base) for λ in λ_vals]
utilities = [worst_case(λ) @ u_c for λ in λ_vals]
ax.plot(rel_entropies, utilities, lw=3, alpha=0.5, label="on the grid")
D_vals = np.linspace(0, max(rel_entropies), 200)
ax.plot(D_vals, -σ_c * np.sqrt(2 * D_vals), "k--", lw=1,
        label=r"exact, $-0.2\sqrt{2D}$")
ax.set_xlabel(r"$D(q^*_\lambda \| \pi)$")
ax.set_ylabel(r"$E_{q^*_\lambda}[u(X)]$")
ax.set_title("worst-case expected utility and relative entropy")
ax.legend()
plt.tight_layout()
plt.show()
```

The right panel traces the worst-case expected utility against the relative entropy of the worst case; eliminating $\lambda$ from the two formulas above gives $E_{q^*_\lambda}[u] = -0.2 \sqrt{2 D(q^*_\lambda \| \pi)}$.

Each point on this curve also solves the problem of minimizing expected utility subject to the constraint $D(q \| \pi) \leq \kappa$, with $\kappa$ equal to the point's horizontal coordinate; {ref}`ip_ex_constraint` asks you to verify this.

These are the constraint preferences studied in the lecture {doc}`advanced:five_preferences`, which writes $\theta$ for our $\lambda$ and $\eta$ for our $\kappa$.

{doc}`Another advanced lecture <advanced:risk_aversion_or_mistaken_beliefs>` uses such worst-case models to ask whether asset prices reflect risk aversion or mistaken beliefs.

Exponential tilting also appears in {doc}`imp_sample`, where it delivers efficient importance samplers.

## Matrix balancing: RAS and Sinkhorn

### Updating an input–output table

Input–output analysts often need to update a table.

{doc}`intro:input_output` describes such tables.

A table $Q$ of flows of goods among industries is available for a base year.

For a later year, the analyst knows only the row totals $r$ (each industry's sales to other industries) and the column totals $c$ (each industry's purchases from other industries).

In this section, $r$ and $c$ denote vectors of totals.

**Matrix balancing** finds a table that has given totals and stays close to a base table.

A natural estimate of the new table is the matrix $P^*$ that has these totals and minimizes $D(P \| Q)$.

Because $P$ and $Q$ need not have the same total, we use the **generalized KL divergence** for nonnegative arrays,

$$
D(P \| Q) = \sum_{i, j} \left[ P_{ij} \log \frac{P_{ij}}{Q_{ij}} - P_{ij} + Q_{ij} \right] ,
$$

which is nonnegative and zero only when $P = Q$, and for which the three-point identity {eq}`ip_three_point` still holds.

The set of matrices with row totals $r$ is a linear family, and so is the set of matrices with column totals $c$.

The matrix $P^*$ is the e-projection of $Q$ onto their intersection.

The argument that proved {prf:ref}`ip_thm_e_projection`, with the indicators of the rows and of the columns as the statistics $t$, applies here.

It shows that if a matrix of the **biproportional** form

$$
P_{ij} = a_i \, Q_{ij} \, b_j ,
$$

with positive vectors $a$ and $b$, has the required totals, then it is $P^*$, because the cross term in {eq}`ip_three_point` becomes $\sum_i \log a_i \, (r_i - r_i) + \sum_j \log b_j \, (c_j - c_j) = 0$.

Such a matrix exists when every entry of $Q$ is positive, $r$ and $c$ are positive, and $\sum_i r_i = \sum_j c_j$ {cite}`Bacharach1970`.

The **RAS method** finds $a$ and $b$ by alternately rescaling the rows of the current table to match $r$ and its columns to match $c$; each rescaling is the e-projection of the current table onto one of the two linear families.

{cite:t}`DemingStephan1940` proposed this algorithm, which statisticians call **iterative proportional fitting**, for adjusting census tables to known marginal totals.

{cite:t}`StoneBrown1962` used it to update input–output tables.

{cite:t}`IrelandKullback1968` identified its limit as the table that minimizes the KL divergence from the base table, but their proof of convergence was incomplete.

{cite:t}`Bacharach1970` studied its properties for input–output analysis.

{cite:t}`Csiszar1975` gave a complete proof that it converges.

The Pythagorean theorem explains why the algorithm converges.

Let $P$ be any matrix with both sets of totals that is zero wherever $Q$ is zero, so that $D(P \| Q) < \infty$.

Because each rescaling is an e-projection onto a linear family that contains $P$, {eq}`ip_pythagoras_e` gives

$$
D(P \| P^{(k)}) = D(P \| P^{(k + 1)}) + D(P^{(k + 1)} \| P^{(k)}) ,
$$

where $P^{(k)}$ is the table after $k$ rescalings, counting a row rescaling and a column rescaling as two steps.

In particular, the divergence of $P^*$ from the iterates falls at every step.

Summing gives $D(P \| Q) = D(P \| P^{(k)}) + \sum_{j < k} D(P^{(j + 1)} \| P^{(j)})$, where the sum does not depend on $P$.

Taking $P = P^*$ shows that $\sum_k D(P^{(k + 1)} \| P^{(k)}) \leq D(P^* \| Q)$, so successive iterates become arbitrarily close, and every limit point $\bar P$ of the iterates has both sets of totals.

Letting $k \to \infty$ along a subsequence that converges to $\bar P$, with $P = \bar P$, gives $D(\bar P \| Q) = \sum_k D(P^{(k + 1)} \| P^{(k)})$, which by the same identity is at most $D(P \| Q)$ for every admissible $P$.

So every limit point is the minimizer $P^*$, and the iterates converge to it {cite}`Csiszar1975`.

The argument does not need the biproportional form; when $Q$ has zeros, a biproportional solution may not exist.

For example, let $Q$ have rows $(1, 1)$ and $(0, 1)$, and let all totals equal $1$.

The only admissible table is the identity matrix.

It is not of the form $a_i Q_{ij} b_j$ with positive $a$ and $b$, but RAS still converges to it.

```{code-cell} ipython3
def gen_kl(P, Q):
    """Generalized KL divergence of nonnegative arrays."""
    mask = P > 0
    return np.sum(P[mask] * np.log(P[mask] / Q[mask])) - P.sum() + Q.sum()


def ras(Q, r, c, tol=1e-12, max_iter=10_000):
    """Rescale rows and columns of Q to match totals r and c."""
    if not np.isclose(r.sum(), c.sum()):
        raise ValueError("row and column totals must have the same sum")
    P = Q.copy()
    history = [P.copy()]
    for _ in range(max_iter):
        P = P * (r / P.sum(axis=1))[:, None]
        history.append(P.copy())
        P = P * (c / P.sum(axis=0))[None, :]
        history.append(P.copy())
        if np.max(np.abs(P.sum(axis=1) - r)) < tol * r.sum():
            break
    else:
        warnings.warn("RAS did not converge")
    return P, history


Q_io = np.array([[20.0, 10.0, 5.0],
                 [8.0, 30.0, 12.0],
                 [6.0, 9.0, 25.0]])
r_new = np.array([40.0, 55.0, 45.0])
c_new = np.array([38.0, 52.0, 50.0])

P_io, history = ras(Q_io, r_new, c_new)
print("base table:\n", Q_io)
print("updated table:\n", np.round(P_io, 3))
print("row totals:", np.round(P_io.sum(axis=1), 6),
      " column totals:", np.round(P_io.sum(axis=0), 6))
```

The ratio $P^*_{ij} / Q_{ij} = a_i b_j$ factors into a row effect times a column effect.

The name RAS describes this: the updated table is $R A S$, with $A$ the base table and $R$ and $S$ diagonal matrices of row and column multipliers, our $a$ and $b$.

If the ratio $R_{ij} = P^*_{ij}/Q_{ij}$ factors as $a_i b_j$, then $R_{ij} = R_{i1} R_{1j} / R_{11}$, so dividing $R_{ij}$ by $R_{i1} R_{1j} / R_{11}$ leaves a matrix of ones.

```{code-cell} ipython3
ratio = P_io / Q_io
print("P*/Q divided by its first column and first row:")
print(np.round(ratio / np.outer(ratio[:, 0], ratio[0, :] / ratio[0, 0]), 10))
```

The divergence of $P^*$ from the iterates falls monotonically until it reaches the level of rounding error; the figure omits values below that level.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Convergence of the RAS algorithm
    name: fig-ip-ras
---
divergences = np.array([gen_kl(P_io, P) for P in history])
steps = np.arange(len(divergences))
keep = divergences > 1e-12
fig, ax = plt.subplots()
ax.semilogy(steps[keep], divergences[keep], "o-")
ax.set_xlabel("rescaling step $k$")
ax.set_ylabel(r"$D(P^* \| P^{(k)})$")
plt.show()
```

We check the Pythagorean theorem {eq}`ip_pythagoras_e` for other tables with the same totals.

```{code-cell} ipython3
# add multiples of a pattern with zero row and column sums
pattern = np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
for s in (-3.0, 2.0, 5.0):
    P_other = P_io + s * pattern
    print(f"D(P || Q) = {gen_kl(P_other, Q_io):.6f},  "
          f"D(P || P*) + D(P* || Q) = "
          f"{gen_kl(P_other, P_io) + gen_kl(P_io, Q_io):.6f}")
```

(ip_entropic_ot)=
### Entropy-regularized optimal transport

{doc}`opt_transport` studies the problem of choosing a transport plan $P$ with row totals $p$ and column totals $q$ (the notation of that lecture) to minimize the total cost $\sum_{i, j} C_{ij} P_{ij}$.

Adding $\varepsilon$ times the negative entropy of the plan gives the **entropy-regularized** problem

$$
\min_{P} \sum_{i, j} C_{ij} P_{ij} + \varepsilon \sum_{i, j} P_{ij} \left( \log P_{ij} - 1 \right) .
$$

With $K_{ij} = \exp(-C_{ij}/\varepsilon)$, the objective equals $\varepsilon \, D(P \| K) - \varepsilon \sum_{ij} K_{ij}$.

The regularized problem therefore finds the e-projection of the matrix $K$ onto the set of plans with the required totals.

The plan has the biproportional form $P_{ij} = a_i K_{ij} b_j = a_i \exp(-C_{ij}/\varepsilon) b_j$, the same exponential tilting that solves {eq}`ef_logit_base` in {doc}`exponential_families`.

The rescaling algorithm of RAS solves it; in this setting the algorithm goes by the name **Sinkhorn's algorithm** {cite}`Sinkhorn1964`.

{cite:t}`Cuturi2013` showed that Sinkhorn's algorithm makes large optimal transport problems tractable.

{cite:t}`Galichon_2016` and {cite:t}`GalichonSalanie2022` explain how the matching model of {cite:t}`ChooSiow2006`, in which agents have logit preferences over partners, leads to problems of this form.

When every agent is matched, the Choo–Siow model is exactly an entropy-regularized transport problem with fixed totals.

When $\varepsilon$ is small, the entries of $K$ are tiny, so we carry out the rescalings in logarithms.

We use the example of {doc}`opt_transport`, solve the linear program with `linprog`, and compare its solution with the solutions of the regularized problem for decreasing $\varepsilon$.

```{code-cell} ipython3
def sinkhorn(C, p, q, ε, tol=1e-10, max_iter=100_000):
    """Entropy-regularized optimal transport by rescaling in logs.

    Returns the plan and the vectors α = ε log a and β = ε log b.
    """
    log_K = -C / ε
    log_a, log_b = np.zeros(len(p)), np.zeros(len(q))
    for _ in range(max_iter):
        log_a = np.log(p) - logsumexp(log_K + log_b[None, :], axis=1)
        log_b = np.log(q) - logsumexp(log_K + log_a[:, None], axis=0)
        P = np.exp(log_a[:, None] + log_K + log_b[None, :])
        if np.max(np.abs(P.sum(axis=1) - p)) < tol * p.sum():
            break
    else:
        warnings.warn("Sinkhorn's algorithm did not converge")
    return P, ε * log_a, ε * log_b


p_ot = np.array([50.0, 100.0, 150.0])
q_ot = np.array([25.0, 115.0, 60.0, 30.0, 70.0])
C_ot = np.array([[10.0, 15.0, 20.0, 20.0, 40.0],
                 [20.0, 40.0, 15.0, 30.0, 30.0],
                 [30.0, 35.0, 40.0, 55.0, 25.0]])
m, n = C_ot.shape

# the linear program of the optimal transport lecture
A_eq = np.vstack([np.kron(np.ones((1, n)), np.identity(m)),
                  np.kron(np.identity(n), np.ones((1, m)))])
res_lp = linprog(C_ot.reshape(-1, order="F"), A_eq=A_eq[:-1],
                 b_eq=np.hstack([p_ot, q_ot])[:-1])
P_lp = res_lp.x.reshape((m, n), order="F")

print(f"linear program: cost = {res_lp.fun:.3f}")
for ε in (10.0, 3.0, 1.0, 0.1):
    P_ε, _, _ = sinkhorn(C_ot, p_ot, q_ot, ε)
    print(f"ε = {ε:5.1f}: cost = {np.sum(C_ot * P_ε):9.3f}")
```

As $\varepsilon \to 0$, the cost of the regularized solution converges to the minimal cost.

The regularized plans need not converge to the plan that `linprog` reports.

```{code-cell} ipython3
P_small, _, _ = sinkhorn(C_ot, p_ot, q_ot, 0.05)
print("linprog plan:\n", np.round(P_lp, 2))
print("limit of regularized plans:\n", np.round(P_small, 2))


def plan_entropy(P):
    """Entropy of the plan, normalized to be a probability distribution."""
    P = P[P > 1e-12] / P.sum()
    return -np.sum(P * np.log(P))


print(f"costs: {np.sum(C_ot * P_lp):.3f} and {np.sum(C_ot * P_small):.3f}")
print(f"entropies: {plan_entropy(P_lp):.4f} and {plan_entropy(P_small):.4f}")
```

This problem has many optimal plans.

`linprog` typically returns a vertex of the set of optimal plans.

The regularized plans converge to the optimal plan with the largest entropy {cite}`CominettiSanMartin1994`.

That limit is itself an e-projection.

By complementary slackness, the optimal plans are exactly the plans with the required totals that use only the cells that some optimal plan uses.

The optimal plan with the largest entropy is the result of applying the RAS method to the matrix that has ones in those cells and zeros elsewhere.

```{code-cell} ipython3
support = (P_small > 1e-6 * p_ot.sum()).astype(float)
P_maxent, _ = ras(support, p_ot, q_ot)
print("RAS applied to the optimal support:\n", np.round(P_maxent, 2))
print(f"cost = {np.sum(C_ot * P_maxent):.3f}, "
      f"entropy = {plan_entropy(P_maxent):.4f}")
```

### Shadow prices and logit choice

Write the regularized plan as

$$
P_{ij} = \exp\left( \frac{\alpha_i + \beta_j - C_{ij}}{\varepsilon} \right),
\qquad \alpha_i = \varepsilon \log a_i, \quad \beta_j = \varepsilon \log b_j .
$$ (ip_sinkhorn_plan)

The vectors $\alpha$ and $\beta$ are Lagrange multipliers on the row and column totals: setting the derivative of the Lagrangian with respect to $P_{ij}$ to zero gives $C_{ij} + \varepsilon \log P_{ij} = \alpha_i + \beta_j$, which is {eq}`ip_sinkhorn_plan`.

In {doc}`opt_transport`, the dual variables $u_i$ and $v_j$ of the linear program are shadow prices of the totals $p_i$ and $q_j$.

Adding a constant to every $u_i$ and subtracting it from every $v_j$ leaves the dual objective unchanged when total supply equals total demand, so we compare $u - u_1$ and $v + u_1$ with $\alpha - \alpha_1$ and $\beta + \alpha_1$.

In this example the dual solution is unique up to that constant.

```{code-cell} ipython3
res_dual = linprog(-np.hstack([p_ot, q_ot]), A_ub=A_eq.T,
                   b_ub=C_ot.reshape(-1, order="F"), bounds=[(None, None)] * (m + n))
u_lp, v_lp = res_dual.x[:m], res_dual.x[m:]
print(f"linear program: u - u_1 = {np.round(u_lp - u_lp[0], 3)}, "
      f"v + u_1 = {np.round(v_lp + u_lp[0], 3)}")
for ε in (1.0, 0.1, 0.03):
    _, α, β = sinkhorn(C_ot, p_ot, q_ot, ε)
    print(f"ε = {ε:4}:         α - α_1 = {np.round(α - α[0], 3)}, "
          f"β + α_1 = {np.round(β + α[0], 3)}")
```

As $\varepsilon \to 0$, the multipliers of the regularized problem converge to the shadow prices of the linear program.

The exponent in {eq}`ip_sinkhorn_plan` is minus the **reduced cost** $C_{ij} - \alpha_i - \beta_j$ divided by $\varepsilon$.

In the linear program, the reduced costs $c_{ij} - u_i - v_j$ are nonnegative, and complementary slackness makes them zero wherever an optimal plan ships a positive amount.

As $\varepsilon \to 0$, the regularized plan puts vanishing weight on every cell whose reduced cost stays positive.

```{code-cell} ipython3
print("reduced costs of the linear program:\n",
      np.round(C_ot - u_lp[:, None] - v_lp[None, :], 2))
print("cells used by the limit of the regularized plans:\n",
      (P_small > 1e-6 * p_ot.sum()).astype(int))
```

The limit of the regularized plans uses exactly the cells with zero reduced cost.

Dividing row $i$ of {eq}`ip_sinkhorn_plan` by its total $p_i$ gives the distribution of the destination of a unit shipped from origin $i$:

$$
\frac{P_{ij}}{p_i} = \frac{\exp\left( (\beta_j - C_{ij}) / \varepsilon \right)}{\sum_k \exp\left( (\beta_k - C_{ik}) / \varepsilon \right)} .
$$

This is a multinomial logit with utilities $\beta_j - C_{ij}$ and scale $\varepsilon$.

If each unit shipped from origin $i$ chose its destination by this logit rule, valuing destination $j$ at $\beta_j$ minus the cost of getting there, then the $\beta_j$ would be prices at which arrivals at each destination equal $q_j$.

In the matching model of {cite:t}`ChooSiow2006`, agents on both sides of the market choose partners by logit rules of this kind.

Sinkhorn's row update sets

$$
\alpha_i = \varepsilon \log p_i - \varepsilon \log \sum_j \exp\left( \frac{\beta_j - C_{ij}}{\varepsilon} \right) .
$$

The second term is minus the inclusive value {eq}`ef_logit` of this logit, with $\tau = \varepsilon$.

As $\varepsilon \to 0$, the inclusive value converges to the largest utility, so the update converges to $\alpha_i = \min_j (C_{ij} - \beta_j)$.

The linear program has the same relation: the dual constraints $u_i + v_j \leq c_{ij}$ hold for every $j$, and one of them binds because origin $i$ ships to some destination, so $u_i = \min_j (c_{ij} - v_j)$.

Optimal transport calls this operation the **c-transform**; with the cost $c(x, y) = -x \cdot y$, it is the Legendre transform of {doc}`exponential_families`, up to signs {cite}`KhanZhang2022`.

Sinkhorn's algorithm replaces the minimum in the c-transform by a logit inclusive value.

### An exponential family of plans

Divide a plan by its total $M = \sum_i p_i$ to turn it into a probability distribution over the cells $(i, j)$.

For a fixed cost matrix and $\varepsilon$, the plans {eq}`ip_sinkhorn_plan` for all vectors $\alpha$ and $\beta$ form an exponential family on the cells.

Its base measure is $K_{ij} = \exp(-C_{ij} / \varepsilon)$, its sufficient statistic is the pair of indicator vectors of the row $i$ and the column $j$, its natural parameters are $\alpha / \varepsilon$ and $\beta / \varepsilon$, and its expectation parameters are the row and column totals of the normalized plan {cite}`AmariKarakidaOizumi2018`.

The regularized plan is the model in this family whose expectation parameters equal the required totals $p / M$ and $q / M$, so Sinkhorn's algorithm maps expectation parameters to natural parameters.

The plan is therefore two projections at once.

It is the e-projection of $K$ onto the linear family of plans with the required totals, and, by {prf:ref}`ip_thm_m_projection`, it is the m-projection of every plan with those totals onto the exponential family.

The linear family and the exponential family cross at one point, as in the right panel of the figure of the two projections.

We check the Pythagorean theorem {eq}`ip_pythagoras_m` with two plans that have the required totals, the plan of the linear program and the independent plan $p q^\top / M$, and a model in the family with arbitrary multipliers.

```{code-cell} ipython3
M = p_ot.sum()
P_1, _, _ = sinkhorn(C_ot, p_ot, q_ot, 1.0)
rng_ot = np.random.default_rng(7)
log_w = rng_ot.normal(size=m)[:, None] + rng_ot.normal(size=n)[None, :] - C_ot
other = np.exp(log_w - logsumexp(log_w))
for name, P in (("linear program", P_lp), ("independent plan", np.outer(p_ot, q_ot) / M)):
    print(f"{name:16}: D(P || other) = {kl(P / M, other):.6f}, "
          f"D(P || P_ε) + D(P_ε || other) = {kl(P / M, P_1 / M) + kl(P_1 / M, other):.6f}")
```

### A cost of information

{cite:t}`KhanZhang2022` point out that the entropy term of the regularized problem is a KL divergence in disguise.

Let $\pi = P / M$, and let $r = p / M$ and $c = q / M$ be its marginal distributions.

For every plan with the required totals,

$$
\sum_{i, j} P_{ij} \log P_{ij} = M \left[ I(\pi) - H(r) - H(c) + \log M \right] ,
$$

where $I(\pi) = D(\pi \| r c^\top)$ is the mutual information between origin and destination, as in the section on independence and mutual information.

Only $I(\pi)$ depends on the plan, so the regularized problem is equivalent to

$$
\min_{\pi} \; E_\pi [C] + \varepsilon \, I(\pi) ,
$$ (ip_ot_information)

over joint distributions $\pi$ of origins and destinations with marginal distributions $r$ and $c$.

A dispatcher who pays $\varepsilon$ for each nat of information that the destination of a unit carries about its origin chooses the regularized plan.

{cite:t}`Sims2003` measures the cost of attention with mutual information, and {cite:t}`MatejkaMcKay2015` show that a decision maker who pays for mutual information makes logit choices.

Here the totals of the destinations are fixed, and the prices $\beta$ adjust to meet them.

As $\varepsilon$ grows, the plan approaches the independent plan $p q^\top / M$, which uses no information.

As $\varepsilon \to 0$, it approaches the optimal plan with the largest entropy, and because $H(\pi) = H(r) + H(c) - I(\pi)$, this is the optimal plan that uses the least information.

Because the regularized plan minimizes {eq}`ip_ot_information`, the pairs $(I(\pi_\varepsilon), E_{\pi_\varepsilon}[C])$ trace the lowest expected cost attainable with each amount of information, a convex curve whose slope at $\pi_\varepsilon$ is $-\varepsilon$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Cost and mutual information of regularized plans
    name: fig-ip-ot-information
---
def mutual_information(P):
    π = P / P.sum()
    return kl(π, np.outer(π.sum(axis=1), π.sum(axis=0)))


def cost_and_information(ε):
    P_ε, _, _ = sinkhorn(C_ot, p_ot, q_ot, ε)
    return mutual_information(P_ε), np.sum(C_ot * P_ε) / M


frontier = np.array([cost_and_information(ε) for ε in np.geomspace(0.05, 500, 80)])
I_5, cost_5 = cost_and_information(5.0)
I_line = np.linspace(I_5 - 0.15, I_5 + 0.15, 2)

fig, ax = plt.subplots()
ax.plot(*frontier.T, lw=2, label="regularized plans")
ax.plot(I_line, cost_5 - 5.0 * (I_line - I_5), "k--", lw=1,
        label=r"slope $-\varepsilon$ at $\varepsilon = 5$")
ax.plot(0, np.sum(C_ot * np.outer(p_ot, q_ot)) / M**2, "o", label="independent plan")
ax.plot(mutual_information(P_lp), np.sum(C_ot * P_lp) / M, "s",
        label="plan of the linear program")
ax.set_xlabel("mutual information between origin and destination (nats)")
ax.set_ylabel("expected cost per unit shipped")
ax.legend()
plt.show()

print(f"independent plan: E[C] = {np.sum(C_ot * np.outer(p_ot, q_ot)) / M**2:.3f}")
print(f"ε = 0.05: I = {frontier[0, 0]:.4f}, E[C] = {frontier[0, 1]:.3f}")
print(f"plan of the linear program: I = {mutual_information(P_lp):.4f}, "
      f"E[C] = {np.sum(C_ot * P_lp) / M:.3f}")
```

The plan that `linprog` reports attains the minimal cost with more information than it needs.

## Chernoff information

{doc}`divergence_measures` defines the Chernoff entropy, which information theorists call the Chernoff information, of densities $f$ and $g$ as

$$
C(f, g) = - \log \min_{\phi \in (0,1)} \rho(\phi), \qquad
\rho(\phi) = \int f^{\phi}(x) g^{1 - \phi}(x) \, dx ,
$$

and {doc}`likelihood_ratio_process` shows that it governs the rate at which the probability of a mistaken model selection falls as observations accumulate.

We assume that $f$ and $g$ are distinct and have the same support, so that $\log \rho$, which is zero at $\phi = 0$ and $\phi = 1$ and strictly convex, attains its minimum at an interior point $\phi^*$.

Information projections give the Chernoff information a geometric interpretation {cite}`Nielsen2013`.

The normalized geometric mixtures

$$
p_\phi(x) = \frac{f^{\phi}(x) g^{1 - \phi}(x)}{\rho(\phi)}
$$

form the exponential path between $g$ (at $\phi = 0$) and $f$ (at $\phi = 1$).

This path is a one-parameter exponential family with sufficient statistic $\log(f/g)$, natural parameter $\phi$, and log-partition function $\log \rho(\phi)$.

At the minimizer $\phi^*$ of $\log \rho$, the derivative of the log-partition function vanishes, so $E_{p_{\phi^*}}[\log(f(X)/g(X))] = 0$.

Since $D(p_\phi \| f) = (\phi - 1) E_{p_\phi}[\log(f/g)] - \log \rho(\phi)$ and $D(p_\phi \| g) = \phi E_{p_\phi}[\log(f/g)] - \log \rho(\phi)$, it follows that

$$
D(p_{\phi^*} \| f) = D(p_{\phi^*} \| g) = C(f, g) .
$$

*The Chernoff information is the common value of $D(p_{\phi^*} \| f)$ and $D(p_{\phi^*} \| g)$, where $p_{\phi^*}$ is the unique point on the exponential path between $f$ and $g$ that is equally far from both.*

Because $D(p \| f) - D(p \| g) = -E_p[\log(f(X)/g(X))]$, a distribution is equally far from $f$ and $g$ exactly when the log-likelihood ratio has mean zero under it.

The linear family $\mathcal{L}_0 = \{ p : E_p[\log(f/g)] = 0 \}$ is the analog of a perpendicular bisector.

The point $p_{\phi^*}$ is the e-projection of both $f$ and $g$ onto $\mathcal{L}_0$: the e-projection of $f$ is a tilt of $f$ by $\exp(\theta \log(f/g))$, which lies on the exponential path, and $p_{\phi^*}$ is the only point of the path in $\mathcal{L}_0$.

This explains why Chernoff information governs error rates.

A test that selects the model with the higher likelihood errs when the sample average of $\log(f/g)$ has the wrong sign.

If the data come from $f$, that is the event that the empirical distribution lies in the half-space $\{ p : E_p[\log(f/g)] \leq 0 \}$.

Since $E_f[\log(f/g)] = D(f \| g) > 0$, the e-projection of $f$ onto the half-space lies on its boundary $\mathcal{L}_0$, as in the dice example, and the probability of the event decays at rate $D(p_{\phi^*} \| f) = C(f, g)$.

For distributions on a finite set this is Sanov's theorem; for densities, Cramér's theorem applied to the sample mean of $\log(f(X)/g(X))$ gives the same rate, because its rate function at zero is $-\min_\phi \log \rho(\phi)$.

If the data come from $g$, the error event is $\{ p : E_p[\log(f/g)] \geq 0 \}$, and the rate is $D(p_{\phi^*} \| g) = C(f, g)$.

We use the two Beta distributions of {doc}`divergence_measures`, $f = $ Beta(1, 1) and $g = $ Beta(3, 1.2).

Because the Beta family is an exponential family, the exponential path between two Beta distributions stays in the Beta family: $p_\phi$ is the Beta distribution whose parameters are $\phi$ times those of $f$ plus $1 - \phi$ times those of $g$.

All the quantities we need have closed forms.

```{code-cell} ipython3
fa, fb, ga, gb = 1.0, 1.0, 3.0, 1.2


def path_params(ϕ):
    return ϕ * fa + (1 - ϕ) * ga, ϕ * fb + (1 - ϕ) * gb


def log_ρ(ϕ):
    return (betaln(*path_params(ϕ)) - ϕ * betaln(fa, fb)
            - (1 - ϕ) * betaln(ga, gb))


def kl_beta(a1, b1, a2, b2):
    """KL divergence of Beta(a1, b1) from Beta(a2, b2)."""
    return (betaln(a2, b2) - betaln(a1, b1) + (a1 - a2) * digamma(a1)
            + (b1 - b2) * digamma(b1) + (a2 - a1 + b2 - b1) * digamma(a1 + b1))


res_ϕ = minimize_scalar(log_ρ, bounds=(1e-8, 1 - 1e-8), method="bounded",
                        options={"xatol": 1e-12})
ϕ_star, C_fg = res_ϕ.x, -res_ϕ.fun
print(f"ϕ* = {ϕ_star:.6f},  Chernoff information C(f, g) = {C_fg:.6f}")
print(f"D(p_ϕ* || f) = {kl_beta(*path_params(ϕ_star), fa, fb):.6f}")
print(f"D(p_ϕ* || g) = {kl_beta(*path_params(ϕ_star), ga, gb):.6f}")
print("p_ϕ* = Beta({:.4f}, {:.4f})".format(*path_params(ϕ_star)))
```

The Chernoff information agrees with the value computed by numerical integration in {doc}`divergence_measures`.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Point equally far from both distributions
    name: fig-ip-chernoff
---
ϕ_vals = np.linspace(0.001, 0.999, 300)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(ϕ_vals, [kl_beta(*path_params(ϕ), fa, fb) for ϕ in ϕ_vals],
        label=r"$D(p_\phi \| f)$")
ax.plot(ϕ_vals, [kl_beta(*path_params(ϕ), ga, gb) for ϕ in ϕ_vals],
        label=r"$D(p_\phi \| g)$")
ax.plot(ϕ_vals, -log_ρ(ϕ_vals), "k--", lw=1, label=r"$-\log\rho(\phi)$")
ax.plot([ϕ_star], [C_fg], "ko")
ax.axvline(ϕ_star, color="0.7", lw=0.8)
ax.set_xlabel(r"$\phi$")
ax.set_ylim(0, 0.6)
ax.legend()

ax = axes[1]
z = np.linspace(0.001, 0.999, 300)
for (a_, b_), label in (((fa, fb), "$f$"), ((ga, gb), "$g$"),
                        (path_params(ϕ_star), r"$p_{\phi^*}$")):
    ax.plot(z, np.exp((a_ - 1) * np.log(z) + (b_ - 1) * np.log(1 - z)
                      - betaln(a_, b_)), lw=2, label=label)
ax.set_xlabel("$x$")
ax.legend()
plt.tight_layout()
plt.show()
```

In the left panel, the two divergences cross at $\phi^*$, where their common value equals the maximum of $-\log \rho(\phi)$, the Chernoff information.

In the right panel, $p_{\phi^*}$ lies between $f$ and $g$.

## Concluding remarks

KL divergence is not a distance.

It has a geometry with two kinds of straight lines, mixtures and normalized geometric mixtures, and two corresponding kinds of projection.

The e-projection onto a set defined by moment conditions is an exponential tilt, and the m-projection onto an exponential family matches moments; each is unique and satisfies a Pythagorean theorem.

The lecture applied m-projections to maximum likelihood estimation and mutual information, and e-projections to maximum entropy and large deviations, robust control, the updating of input–output tables, optimal transport, and Chernoff information.

The worst-case models of robust control, the biproportional tables of RAS, and the regularized transport plans are all exponential tilts of the kind that attains the logit formula {eq}`ef_logit_base` of {doc}`exponential_families`.

The regularized transport plans form an exponential family whose multipliers $\alpha$ and $\beta$, $\varepsilon$ times its natural parameters, converge up to a common constant to the shadow prices of the linear program; each plan charges for the information that destinations carry about origins.

{doc}`approximation_errors` applies m-projections to misspecified time series models and to seasonal adjustment, {doc}`specification_tests` describes specification tests as comparisons of m-projections, and {doc}`turbulence_inference` uses them to interpret a check of a calibrated search model against evidence on earnings instability.

Other applications include the EM algorithm, which {cite:t}`CsiszarTusnady1984` interpret as alternating projections, and the hierarchical decomposition of interactions among many random variables of {cite:t}`Amari2001`, which {ref}`ip_ex_interactions` takes up.

## Exercises

```{exercise}
:label: ip_ex_reweight

Survey statisticians and econometricians often reweight a sample so that it matches a population moment known from another source.

Let $x_1, \ldots, x_n$ be a sample whose average differs from a known population mean $\mu$.

Two ways to choose weights $w_1, \ldots, w_n$ that sum to one and satisfy $\sum_i w_i x_i = \mu$ are

* **exponential tilting**: choose $w$ to minimize $D(w \| q)$, where $q$ puts weight $1/n$ on each observation; and
* **empirical likelihood**: choose $w$ to minimize $D(q \| w)$, equivalently to maximize $\sum_i \log w_i$.

1. Show that exponential tilting is an e-projection and that its weights have the form $w_i \propto \exp(\theta x_i)$.
2. Show that the empirical likelihood weights have the form $w_i = 1 / \left( n (1 + \gamma (x_i - \mu)) \right)$ for a scalar $\gamma$.
3. Draw a sample of $n = 200$ pairs $(x_i, y_i)$ with $x_i$ lognormal and $y_i = 1 + 0.5 x_i + e_i$, where $e_i$ is normal noise. Compute both sets of weights so that the weighted mean of $x$ equals $1.2$ times its sample mean, compare the weights, and compare the two reweighted estimates of the mean of $y$.

{cite:t}`KitamuraStutzer1997` and {cite:t}`Owen1988` develop estimators based on these two kinds of weights.
```

```{solution-start} ip_ex_reweight
:class: dropdown
```

Here is one solution.

For part 1, the set of weight vectors with $\sum_i w_i x_i = \mu$ is a linear family, so minimizing $D(w \| q)$ over it is an e-projection; by {prf:ref}`ip_thm_e_projection`, $w_i \propto q_i \exp(\theta x_i) \propto \exp(\theta x_i)$.

Minimizing $D(q \| w)$ is instead an m-projection of $q$ onto the linear family; our theorems do not cover it.

For part 2, the Lagrangian of maximizing $\sum_i \log w_i$ subject to $\sum_i w_i = 1$ and $\sum_i w_i (x_i - \mu) = 0$ has first-order conditions $1/w_i = \alpha + \beta (x_i - \mu)$.

Multiplying by $w_i$ and summing gives $n = \alpha$, so $w_i = 1/(n(1 + \gamma (x_i - \mu)))$ with $\gamma = \beta / n$.

For part 3, we solve for $\theta$ and $\gamma$ by root finding.

```{code-cell} ipython3
rng_w = np.random.default_rng(11)
n = 200
x = rng_w.lognormal(0.0, 0.5, size=n)
y = 1 + 0.5 * x + rng_w.normal(0, 0.3, size=n)
μ = 1.2 * x.mean()
q_n = np.full(n, 1 / n)

θ_et, w_et = match_mean(q_n, x, μ, bracket=(-20, 20))

# γ must keep every weight positive
lo, hi = -1 / (x.max() - μ), -1 / (x.min() - μ)
γ_el = brentq(lambda γ: np.sum((x - μ) / (1 + γ * (x - μ))),
              lo + 1e-9, hi - 1e-9)
w_el = 1 / (n * (1 + γ_el * (x - μ)))

for name, wts in (("exponential tilting", w_et),
                  ("empirical likelihood", w_el)):
    print(f"{name:21}: sum {wts.sum():.6f}, weighted mean of x {wts @ x:.4f} "
          f"(target {μ:.4f}), weighted mean of y {wts @ y:.4f}")
print(f"unweighted mean of y: {y.mean():.4f}")

order = np.argsort(x)
fig, ax = plt.subplots()
ax.plot(x[order], n * w_et[order], "o-", ms=3, label="exponential tilting")
ax.plot(x[order], n * w_el[order], "s--", ms=3, label="empirical likelihood")
ax.set_xlabel("$x_i$")
ax.set_ylabel(r"$n \times$ weight")
ax.legend()
plt.show()
```

Both sets of weights raise the weighted mean of $y$ by roughly half the change in the mean of $x$.

Given the $x_i$, the linear relationship $y = 1 + 0.5 x + e$ makes the expected change exactly half; departures come from the noise $e_i$.

The two estimates differ because the weights differ.

Exponential tilting weights rise exponentially with $x$, while empirical likelihood weights rise more gently at first and then much more steeply for the largest observations.

```{solution-end}
```

```{exercise}
:label: ip_ex_constraint

Consider the constraint version of the robust decision problem of the lecture:

$$
\min_q E_q[u(X)] \quad \text{subject to} \quad D(q \| \pi) \leq \kappa .
$$

1. Use the identity $E_q[u] + \lambda D(q \| \pi) = \lambda D(q \| q^*_\lambda) - \lambda \log E_\pi \exp(-u/\lambda)$ to show that if $\lambda > 0$ satisfies $D(q^*_\lambda \| \pi) = \kappa$, then $q^*_\lambda$ solves the constraint problem.
2. For the consumption example of the lecture and $\kappa \in \{0.01, 0.05, 0.2\}$, find $\lambda$ and the minimized expected utility.
3. For $\kappa = 0.05$, check your answer by minimizing expected utility over all distributions on the consumption grid subject to $D(q \| \pi) \leq \kappa$ with a general-purpose constrained optimizer.
```

```{solution-start} ip_ex_constraint
:class: dropdown
```

Here is one solution.

For part 1, let $q$ satisfy $D(q \| \pi) \leq \kappa = D(q^*_\lambda \| \pi)$.

The identity gives

$$
E_q[u] = \lambda D(q \| q^*_\lambda) - \lambda D(q \| \pi) - \lambda \log E_\pi e^{-u/\lambda}
\geq - \lambda \kappa - \lambda \log E_\pi e^{-u/\lambda} = E_{q^*_\lambda}[u] ,
$$

where the last equality applies the identity at $q = q^*_\lambda$.

So no feasible $q$ does better than $q^*_\lambda$.

For part 2, $D(q^*_\lambda \| \pi)$ falls as $\lambda$ rises, so we can find $\lambda$ by root finding on $\log \lambda$.

```{code-cell} ipython3
def λ_for_entropy(κ):
    log_λ = brentq(lambda s: kl(worst_case(np.exp(s)), π_base) - κ, -6, 3)
    return np.exp(log_λ)


for κ in (0.01, 0.05, 0.2):
    λ = λ_for_entropy(κ)
    print(f"κ = {κ:4.2f}: λ = {λ:.4f} (exact {σ_c / np.sqrt(2 * κ):.4f}), "
          f"min E_q[u] = {worst_case(λ) @ u_c:.5f} "
          f"(exact {-σ_c * np.sqrt(2 * κ):.5f})")
```

The exact values come from the formulas for the lognormal example: $D(q^*_\lambda \| \pi) = 0.2^2/(2\lambda^2) = \kappa$ gives $\lambda = 0.2/\sqrt{2\kappa}$, and then $E_{q^*_\lambda}[u] = -0.2\sqrt{2\kappa}$.

For part 3, we parameterize distributions by their log probabilities $z$ and use SciPy's SLSQP method, which handles nonlinear inequality constraints.

With $q = \operatorname{softmax}(z)$, the gradients of $E_q[u]$ and of $D(q \| \pi)$ with respect to $z_k$ are $q_k (u_k - E_q[u])$ and $q_k (\log(q_k / \pi_k) - D(q \| \pi))$, which we pass to the optimizer.

```{code-cell} ipython3
κ = 0.05


def softmax(z):
    return np.exp(z - logsumexp(z))


def expected_u(z):
    q = softmax(z)
    return q @ u_c, q * (u_c - q @ u_c)


def entropy_slack(z):
    q = softmax(z)
    return κ - kl(q, π_base)


def entropy_slack_grad(z):
    q = softmax(z)
    log_ratio = np.log(q / π_base)
    return -q * (log_ratio - q @ log_ratio)


res_q = minimize(expected_u, np.log(π_base), jac=True, method="SLSQP",
                 constraints=[{"type": "ineq", "fun": entropy_slack,
                               "jac": entropy_slack_grad}],
                 options={"ftol": 1e-12, "maxiter": 1000})
q_opt = softmax(res_q.x)
q_tilt = worst_case(λ_for_entropy(κ))
print(f"constrained optimizer: E_q[u] = {res_q.fun:.6f}, "
      f"D(q || π) = {kl(q_opt, π_base):.6f}")
print(f"exponential tilt:      E_q[u] = {q_tilt @ u_c:.6f}")
print(f"max |q_opt - q_tilt| = {np.max(np.abs(q_opt - q_tilt)):.2e}")
```

The optimizer finds the exponential tilt.

```{solution-end}
```

```{exercise}
:label: ip_ex_interactions

Let $(X, Y, Z)$ be three binary random variables with a joint distribution $p$ that assigns positive probability to each of the eight outcomes.

Consider three nested exponential families of joint distributions:

* $\mathcal{E}_1$: $X$, $Y$, and $Z$ are independent;
* $\mathcal{E}_2$: $\log q(x, y, z)$ is a sum of functions of the pairs $(x, y)$, $(x, z)$, and $(y, z)$, so that there is no three-way interaction; and
* $\mathcal{E}_3$: all distributions.

1. Use {prf:ref}`ip_thm_m_projection` to show that the m-projection of $p$ onto $\mathcal{E}_1$ is the product of its one-variable marginals and that the m-projection onto $\mathcal{E}_2$ is the model in $\mathcal{E}_2$ with the same two-variable marginals as $p$.
2. Compute the m-projection onto $\mathcal{E}_2$ by iterative proportional fitting: starting from the uniform distribution, repeatedly rescale the current distribution to match each of the three two-variable marginals of $p$.
3. Verify the decomposition $D(p \| p_1) = D(p \| p_2) + D(p_2 \| p_1)$, where $p_k$ is the m-projection onto $\mathcal{E}_k$. The left side, called the multi-information, measures the total dependence among the three variables; the first term on the right measures three-way interaction, and the second measures dependence through pairs.
4. Apply the decomposition to the distribution in which $X$ and $Y$ are independent fair coin flips and $Z = (X + Y) \bmod 2$ with probability $0.9$ and $Z = (X + Y + 1) \bmod 2$ with probability $0.1$. Interpret the result.

{cite:t}`Amari2001` develops this decomposition for any number of variables.
```

```{solution-start} ip_ex_interactions
:class: dropdown
```

Here is one solution.

For part 1, $\mathcal{E}_1$ is an exponential family whose sufficient statistics are the indicators of the values of each variable; their means are the one-variable marginals.

The set $\mathcal{E}_2$ is an exponential family whose sufficient statistics are the indicators of the values of each pair; their means are the two-variable marginals.

{prf:ref}`ip_thm_m_projection` says that each m-projection matches these means.

The product of the one-variable marginals is in $\mathcal{E}_1$ and matches them, so it is the m-projection onto $\mathcal{E}_1$.

Similarly, a model in $\mathcal{E}_2$ with the same two-variable marginals as $p$ is the m-projection onto $\mathcal{E}_2$.

For part 2, the uniform starting point lies in $\mathcal{E}_2$, and each rescaling multiplies the current distribution by a function of one pair of variables, so every iterate lies in $\mathcal{E}_2$.

Each rescaling is an e-projection onto the linear family of distributions with one of $p$'s two-variable marginals.

By the argument that proved convergence of RAS, the iterates converge to a distribution in $\mathcal{E}_2$ with $p$'s two-variable marginals, which by part 1 is $p_2$.

```{code-cell} ipython3
def ipf_pairwise(p, tol=1e-13, max_iter=10_000):
    """The distribution with p's pairwise marginals and no 3-way interaction."""
    q = np.full(p.shape, 1 / p.size)
    for _ in range(max_iter):
        q_old = q.copy()
        for axis in (2, 1, 0):          # match the marginal of the other two
            q = q * (p.sum(axis=axis, keepdims=True)
                     / q.sum(axis=axis, keepdims=True))
        if np.max(np.abs(q - q_old)) < tol:
            break
    else:
        warnings.warn("iterative proportional fitting did not converge")
    return q


def independence_projection(p):
    px, py, pz = p.sum(axis=(1, 2)), p.sum(axis=(0, 2)), p.sum(axis=(0, 1))
    return px[:, None, None] * py[None, :, None] * pz[None, None, :]


def decompose(p):
    p1, p2 = independence_projection(p), ipf_pairwise(p)
    total, three_way, pairs = kl(p, p1), kl(p, p2), kl(p2, p1)
    print(f"multi-information D(p || p1) = {total:.6f}")
    print(f"three-way interaction D(p || p2) = {three_way:.6f}")
    print(f"pairwise dependence D(p2 || p1) = {pairs:.6f}")
    print(f"sum of the two parts = {three_way + pairs:.6f}")


rng_ipf = np.random.default_rng(12)
p_random = rng_ipf.dirichlet(np.ones(8)).reshape(2, 2, 2)
decompose(p_random)
```

For part 3, $p_2$ is the m-projection of $p$ onto $\mathcal{E}_2$, and $p_1$ lies in $\mathcal{E}_1 \subset \mathcal{E}_2$, so {eq}`ip_pythagoras_m` for $\mathcal{E}_2$, with $p_1$ in the role of $q_\theta$, gives $D(p \| p_1) = D(p \| p_2) + D(p_2 \| p_1)$, which the output confirms.

Because $p$ and $p_2$ have the same one-variable marginals, $p_1$ is also the m-projection of $p_2$ onto $\mathcal{E}_1$, so $D(p_2 \| p_1)$ is the multi-information of $p_2$.

For part 4, every pair of variables is independent, but the three variables are strongly dependent.

```{code-cell} ipython3
p_xor = np.zeros((2, 2, 2))
for x_ in (0, 1):
    for y_ in (0, 1):
        for z_ in (0, 1):
            agree = (x_ + y_) % 2 == z_
            p_xor[x_, y_, z_] = 0.25 * (0.9 if agree else 0.1)
decompose(p_xor)
```

All of the dependence is three-way interaction: each pair of variables is independent, so the m-projections onto $\mathcal{E}_2$ and $\mathcal{E}_1$ coincide (both are uniform), yet knowing any two of the variables nearly determines the third.

Measures of dependence that look only at pairs of variables, such as correlations, would miss it entirely.

```{solution-end}
```

```{exercise}
:label: ip_ex_inverse_ot

{cite:t}`DupuyGalichon2014` and {cite:t}`GalichonSalanie2022` estimate the parameters of matching models from observed tables of matches.

This exercise estimates the cost of a transport problem from an observed table of shipments.

Let origins sit at points $x_i$ and destinations at points $y_j$ on the unit interval, with $d_{ij} = \lvert x_i - y_j \rvert$, and let the cost be $C_{ij} = \lambda d_{ij}$ with unknown $\lambda > 0$.

Shipments follow the regularized plan with $\varepsilon = 1$; only $\lambda / \varepsilon$ can be identified.

An analyst observes the counts $N_{ij}$ of $N$ shipments drawn independently from the plan, normalized to be a probability distribution $\pi$ over cells.

1. Show that the distributions $\pi_{ij} \propto \exp(a_i + b_j - \lambda d_{ij})$, for all vectors $a$ and $b$ and all $\lambda$, form an exponential family, and that the maximum likelihood estimate matches the observed row totals, column totals, and average distance.
2. Draw $6$ origins and $8$ destinations from the uniform distribution, totals from Dirichlet distributions, and a table of $N = 2000$ shipments with $\lambda = 4$. Estimate $\lambda$ by finding the value at which the regularized plan with the observed row and column totals has the observed average distance.
3. Holding the row and column totals fixed, the derivative of the plan's average distance with respect to $\lambda$ is minus the Fisher information for $\lambda$ per shipment. Use it to compute a standard error, and compare it with the standard deviation of the estimates across $300$ simulated tables.
```

```{solution-start} ip_ex_inverse_ot
:class: dropdown
```

Here is one solution.

For part 1, write the cell as the pair $(i, j)$; then $\log \pi_{ij} = a \cdot e_i + b \cdot e_j - \lambda d_{ij}$ minus a normalizing constant, where $e_i$ and $e_j$ are indicator vectors.

This is an exponential family with sufficient statistic $(e_i, e_j, d_{ij})$ and natural parameter $(a, b, -\lambda)$.

By the result of {doc}`exponential_families` that maximum likelihood matches moments, the estimate makes the model's means of the sufficient statistic equal to their sample means: the row totals, the column totals, and the average distance.

For given $\lambda$, Sinkhorn's algorithm finds the $a$ and $b$ that match the totals, so we find $\lambda$ by one-dimensional root finding.

The derivative in part 3 is the second derivative of the log likelihood per shipment after $a$ and $b$ have been chosen to match the observed totals, so $N$ times minus it is the information in the table.

In an exponential family the second derivatives at given parameters do not depend on the data, so at the estimates this observed information equals the Fisher information.

```{code-cell} ipython3
rng_inv = np.random.default_rng(5)
m_s, n_s = 6, 8
x_o, y_d = np.sort(rng_inv.uniform(size=m_s)), np.sort(rng_inv.uniform(size=n_s))
d = np.abs(x_o[:, None] - y_d[None, :])
p_s, q_s = rng_inv.dirichlet(5 * np.ones(m_s)), rng_inv.dirichlet(5 * np.ones(n_s))
λ_true, N_ship = 4.0, 2000
π_true, _, _ = sinkhorn(λ_true * d, p_s, q_s, 1.0)


def estimate_λ(counts, h=1e-4):
    π_hat = counts / counts.sum()
    r_hat, c_hat = π_hat.sum(axis=1), π_hat.sum(axis=0)

    def excess_distance(λ):
        π_λ, _, _ = sinkhorn(λ * d, r_hat, c_hat, 1.0)
        return np.sum(π_λ * d) - np.sum(π_hat * d)

    λ_hat = brentq(excess_distance, 0.0, 30.0)
    info = -(excess_distance(λ_hat + h) - excess_distance(λ_hat - h)) / (2 * h)
    return λ_hat, 1 / np.sqrt(counts.sum() * info)


def draw_table():
    return rng_inv.multinomial(N_ship, π_true.ravel()).reshape(m_s, n_s)


λ_hat, se = estimate_λ(draw_table())
print(f"estimate {λ_hat:.3f}, standard error {se:.3f}")

estimates = np.array([estimate_λ(draw_table()) for _ in range(300)])
print(f"300 tables: mean estimate {estimates[:, 0].mean():.3f}, "
      f"standard deviation {estimates[:, 0].std():.3f}, "
      f"mean standard error {estimates[:, 1].mean():.3f}")
```

The standard errors match the spread of the estimates across simulated tables.

```{solution-end}
```
