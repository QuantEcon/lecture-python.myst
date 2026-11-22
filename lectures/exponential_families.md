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

(exponential_families)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Exponential Family and Maximum Entropy

```{contents} Contents
:depth: 2
```

## Overview

This lecture and its sequel {doc}`information_projections` describe elements of **information geometry**, a field that uses tools from convex analysis and differential geometry to study parameterized families of probability distributions.

A key concept in information geometry is an **exponential family** of probability distributions.

To understand what an exponential family is, we must understand the definition of a **sufficient statistic** for the parameter of a family of probability distributions.

A sufficient statistic carries all the information about the parameter that a sample contains.

```{note}
{cite:t}`Darmois1935`, {cite:t}`Koopman1936`, and {cite:t}`Pitman1936` described smoothness conditions under which a family of densities whose set of possible outcomes does not depend on its parameter is an exponential family if samples of every size have a sufficient statistic of the same finite dimension.
```

The exponential family includes the Bernoulli, exponential, Poisson, normal, gamma, and Beta families, among others.

Geometry studies **invariants**, properties that do not change when we change coordinates.

To characterize such **invariants**, we ask which objects change and which do not when we relabel the parameters of a family or transform the data.

KL divergence does not change when appropriate transformations are applied to either the parameters or the data.

However, the Euclidean distance between parameter vectors *does* change when we relabel parameters.

Differential entropy *does* change when we transform the data.

A key tool used to define an exponential family is a convex function called the **log-partition function**.

The log-partition function generates

* two systems of coordinates for the models in a family, called **natural parameters** and **expectation parameters**;
* the means and covariances of the family's sufficient statistics;
* the Fisher information matrix;
* the entropy of each model, through a Legendre transform, when the family's base measure is constant; and
* the Kullback–Leibler (KL) divergence between any two models, through a **Bregman divergence**.

```{note}
Here "system of coordinates" is a synonym for "parameterization of a family."
```

Each model in an exponential family answers a question posed by {cite:t}`Jaynes1957`: what probability distribution expresses the least information, as measured by Shannon's entropy, among all distributions that have given means of some statistics?

A special case characterizes the normal distribution: among all densities on the real line with a given mean and variance, the normal density has the largest entropy.

The **Fisher information matrix** assigns lengths to paths of models and defines a distance between models, the **Fisher–Rao distance**.

The Fisher–Rao distance does not change when we relabel parameters or transform the data.

For the normal family, it is a distance of hyperbolic geometry.

The **Wasserstein distance** of optimal transport measures instead how far probability mass must move to turn one model into another; it does not change when we relabel parameters, but it does change when we transform the data.

```{note}
Economists already use several of these tools under other names.

The log-partition function and its **Legendre transform** are related in the same way as a profit function and a production function.

The formula for the derivative of the log-partition function is a version of Hotelling's lemma.

The **inclusive value** of the multinomial logit model of discrete choice is a log-partition function.

Up to a constant, it is the expected utility of the best alternative.

Its Legendre transform is minus an entropy.
```

The sequel {doc}`information_projections` uses these tools to study projections of probability distributions onto families of distributions, with applications to maximum likelihood, robust control, the updating of input–output tables, and optimal transport.

The lecture builds on {doc}`divergence_measures`, which describes entropy, cross-entropy, and KL divergence; we import some of its notation.

{doc}`mle` and {doc}`multivariate_normal` provide additional background.

{cite:t}`amari_nagaoka2000`, {cite:t}`Nielsen2020`, and chapters 2, 8, 11, and 12 of {cite:t}`CoverThomas2006` treat the material of this lecture and its sequel.

Our order of topics follows the lecture slides of {cite:t}`NielsenMLSS2015`.

We use the following imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import brentq
from scipy.special import logsumexp, digamma, polygamma, gammaln
from scipy.stats import norm, poisson, laplace, logistic, uniform, chi2
from scipy.stats import lognorm, cauchy
from scipy.stats import gamma as gamma_dist
```

## Models and families

### Models, families, and parameters

A **model** is a probability distribution for a random variable $X$ that takes values in a set $\mathcal X$.

We describe a model by a density or by a probability mass function.

A **parametric family** is a set of models

$$
\mathcal P = \{ p(\cdot\,; \theta) : \theta \in \Theta \} ,
$$

where the **parameter space** $\Theta$ is a subset of $\mathbb R^D$.

The map $\theta \mapsto p(\cdot\,; \theta)$ is a **parameterization** of $\mathcal P$; it assigns coordinates $\theta$ to the models in $\mathcal P$.

Statisticians often call a whole family a "statistical model."

We do not.

In this lecture, a model is a single distribution $p(\cdot\,; \theta)$ at a single parameter vector $\theta$, and a family is a set of models indexed by parameter vectors that range over $\Theta$.

Here are six families.

* The **Bernoulli family**: $\mathcal X = \{0, 1\}$, $p(x; \pi) = \pi^x (1 - \pi)^{1 - x}$, and $\Theta = (0, 1)$.
* The **Poisson family**: $\mathcal X = \{0, 1, 2, \ldots\}$, $p(x; \lambda) = \lambda^x e^{-\lambda} / x!$, and $\Theta = (0, \infty)$.
* The **normal family**: $\mathcal X = \mathbb R$, $p(x; \mu, \sigma) = (2 \pi \sigma^2)^{-1/2} \exp\left(-(x - \mu)^2 / (2 \sigma^2)\right)$, and $\Theta$ is the half-plane $\mathbb R \times (0, \infty)$.
* The **uniform family**: $\mathcal X = (0, \infty)$, $p(x; \theta) = 1/\theta$ for $0 < x < \theta$ and $p(x; \theta) = 0$ for $x \geq \theta$, and $\Theta = (0, \infty)$.
* The **Cauchy location family**: $\mathcal X = \mathbb R$, $p(x; \mu) = 1 / \left(\pi (1 + (x - \mu)^2)\right)$, and $\Theta = \mathbb R$.
* A **normal mixture family**: $\mathcal X = \mathbb R$, $p(x; w, \mu_1, \mu_2) = w \, \varphi(x - \mu_1) + (1 - w) \, \varphi(x - \mu_2)$, where $\varphi$ is the $N(0, 1)$ density, and $\Theta = [0, 1] \times \mathbb R^2$.

In the Bernoulli family $\pi$ is a probability; in the normal and Cauchy families it is $3.14159\ldots$.

A parameterization is **identifiable** if different parameter vectors give different models.

The parameterizations of the first five families are identifiable.

The mixture parameterization is not: $(w, \mu_1, \mu_2)$ and $(1 - w, \mu_2, \mu_1)$ give the same model, every $w$ gives the same model when $\mu_1 = \mu_2$, and every $\mu_1$ gives the same model when $w = 0$.

We assume that a parameterization is identifiable and smooth, and that the $D$ functions $\partial p(\cdot\,; \theta) / \partial \theta_i$ are linearly independent.

Then $\mathcal P$ is a $D$-dimensional surface, or **manifold**, in the set of all probability distributions on $\mathcal X$, and $\theta$ is a coordinate system on $\mathcal P$.

Harold Hotelling in 1930 and {cite:t}`Rao1945` studied a parametric family as such a space.

A family has many coordinate systems.

We can describe the normal family by $(\mu, \sigma)$, by $(\mu, \sigma^2)$, or by the pair of means $(E X, E X^2) = (\mu, \mu^2 + \sigma^2)$.

### Score and Fisher information

The **score** of a model is the gradient $\nabla_\theta \log p(X; \theta)$ of its log density with respect to the parameter vector.

When we can differentiate under the integral sign, the score has mean zero:

$$
E_\theta \left[ \nabla_\theta \log p(X; \theta) \right] = \int \nabla_\theta \, p(x; \theta) \, dx = \nabla_\theta \int p(x; \theta) \, dx = 0 .
$$

The covariance matrix of the score is the **Fisher information matrix**

$$
I(\theta) = E_\theta \left[ \nabla_\theta \log p(X; \theta) \, \nabla_\theta \log p(X; \theta)^\top \right] .
$$ (ef_fisher)

An IID sample of size $n$ has Fisher information $n I(\theta)$.

The Cramér–Rao inequality says that the covariance matrix of an unbiased estimator of $\theta$ from such a sample exceeds $I(\theta)^{-1} / n$ by a positive semidefinite matrix.

For the normal family in coordinates $(\mu, \sigma)$,

$$
I(\mu, \sigma) = \frac{1}{\sigma^2} \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix} .
$$ (ef_fisher_normal)

### Sufficient statistics

Let $\mathbf X = (X_1, \ldots, X_n)$ be an IID sample from a model $p(\cdot\,; \theta)$ in a family $\mathcal P$.

A **statistic** is a function $s(\mathbf X)$ of the sample that does not involve $\theta$.

A statistic $s$ is **sufficient** for $\theta$ in the family $\mathcal P$ if the conditional distribution of $\mathbf X$ given $s(\mathbf X)$ is the same for every $\theta \in \Theta$.

Once we know $s(\mathbf X)$, the rest of the sample tells us nothing more about $\theta$.

{cite:t}`Fisher1922` introduced sufficiency.

The Fisher–Neyman **factorization theorem** says that $s$ is sufficient if and only if the joint density factors as

$$
\prod_{i=1}^n p(x_i; \theta) = g\left(s(\mathbf x); \theta\right) a(\mathbf x)
$$ (ef_factorization)

for some functions $g$ and $a$, where $a$ does not depend on $\theta$.

By {eq}`ef_factorization`, two samples with the same value of $s$ have proportional likelihood functions.

They deliver the same maximum likelihood estimate and, under any prior, the same posterior distribution.

Sufficiency is a property of a statistic relative to a family.

We give examples of sufficient statistics after we define an exponential family.

(ef_invariants)=
## Invariants

Felix Klein's Erlangen program of 1872 defined a geometry by a group of transformations and studied the properties of figures that the transformations leave unchanged.

Information geometry studies properties of models and families that survive three kinds of transformation.

1. A **reparameterization** replaces $\theta$ by $\xi = \phi(\theta)$, where $\phi$ is smooth and one-to-one with a smooth inverse. Each model stays the same; only its coordinates change.
2. An **invertible transformation of the data** replaces $X$ by $Y = f(X)$, where $f$ is smooth and one-to-one. Each model for $X$ becomes a model for $Y$, and the family keeps its parameter space.
3. A **statistic** replaces the data by a function of the data. More generally, a **garbling** of the kind studied in {doc}`blackwell_kihlstrom` replaces the data by a random variable whose conditional distribution given the data does not depend on $\theta$. Statistics and garblings can discard information.

### Distances between parameter vectors

We usually describe a normal model by its mean $\mu$ and standard deviation $\sigma$.

The pair $(\mu, \sigma)$ is a point in a half-plane, but the ordinary distance between two such points says little about how hard it is to tell the two models apart from data.

Consider four normal models:

* $A = N(0, 0.2^2)$ and $B = N(1, 0.2^2)$, and
* $C = N(0, 2^2)$ and $D = N(1, 2^2)$.

In the $(\mu, \sigma)$ half-plane, $A$ is as far from $B$ as $C$ is from $D$.

But $A$ and $B$ barely overlap, while $C$ and $D$ overlap heavily.

We read $D_{KL}(p \| q) = \int p(x) \log (p(x) / q(x)) \, dx$ as "the KL divergence of $p$ from $q$."

The KL divergence between two normal models with a common standard deviation $\sigma$ is $(\mu_1 - \mu_2)^2 / (2 \sigma^2)$, which is $12.5$ for $A$ and $B$ and $0.125$ for $C$ and $D$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Equal parameter distances, unequal KL divergences
    name: fig-ef-four-normals
---
x = np.linspace(-6, 7, 1000)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, σ, names in zip(axes, (0.2, 2.0), (("A", "B"), ("C", "D"))):
    for μ, name in zip((0.0, 1.0), names):
        ax.plot(x, norm.pdf(x, μ, σ), lw=2, label=f"{name}: N({μ:g}, {σ:g}²)")
    ax.set_title(rf"$\sigma = {σ:g}$, KL = {1 / (2 * σ**2):.3g}")
    ax.set_xlabel("$x$")
    ax.legend()
plt.tight_layout()
plt.show()
```

A geometry of probability distributions should measure distances by how distinguishable distributions are.

For the exponential family, one convex function generates such a geometry.

### What changes and what does not

In the table below, $J = \partial \theta / \partial \xi$ is the Jacobian matrix of the inverse of a reparameterization $\phi$, and $f'$ is the derivative of a transformation $f$ of scalar data.

| object | reparameterization $\theta \to \xi = \phi(\theta)$ | transformation of the data $x \to y = f(x)$ |
|---|---|---|
| KL divergence between two models | unchanged | unchanged |
| Fisher information matrix | becomes $J^\top I(\theta) J$ | unchanged |
| Euclidean distance between parameter vectors | changes | unchanged |
| Wasserstein distance between two models | unchanged | changes |
| density of a model at a point | unchanged | divided by $\lvert f'(x) \rvert$ |
| differential entropy of a model | unchanged | changes by $E \log \lvert f'(X) \rvert$ |
| maximum likelihood estimate | becomes $\phi(\hat \theta)$ | unchanged |
| unbiasedness of an estimator | can be lost | unchanged |
| a flat prior density | generally not flat in $\xi$ | unchanged |
| the Jeffreys prior | unchanged | unchanged |

Under $y = f(x)$, each density acquires the factor $1 / \lvert f'(x) \rvert$, which does not depend on $\theta$.

Likelihood ratios and scores do not change, so neither do KL divergences, Fisher information matrices, and maximum likelihood estimates.

Differential entropy changes because it is an expectation of the log density itself: measuring income in cents instead of dollars adds $\log 100$ to it.

Under a reparameterization, the chain rule multiplies the score by $J^\top$, so the Fisher information matrix becomes $J^\top I(\theta) J$.

The quadratic form $d\theta^\top I(\theta) \, d\theta$ does not change, because $d\theta = J \, d\xi$.

{ref}`ef_fisher_rao` uses this quadratic form to define a distance between models that does not depend on coordinates.

The maximum likelihood estimate changes its coordinates from $\hat \theta$ to $\phi(\hat \theta)$ and names the same model.

Unbiasedness can be lost because $E \phi(\hat \theta) \neq \phi(E \hat \theta)$ when $\phi$ is nonlinear; {ref}`ef_estimation` gives an example.

A flat prior on $\sigma$ is not flat on $\sigma^2$.

The prior of {cite:t}`jeffreys1946invariant` has density proportional to $\sqrt{\det I(\theta)}$.

In coordinates $\xi$, its density is proportional to $\sqrt{\det (J^\top I J)} = \lvert \det J \rvert \sqrt{\det I}$, as the change-of-variables formula requires, so the Jeffreys prior assigns the same probability to each set of models in every coordinate system.

{ref}`ef_ex_jeffreys` works out the Bernoulli case.

We compute several of these objects for two normal models, $N(1, 1.5^2)$ and $N(-0.5, 1)$.

We use four coordinate systems: $(\mu, \sigma)$, $(\mu, \sigma^2)$, the pair of means $(\mu, \mu^2 + \sigma^2)$, and $(\mu / \sigma^2, -1 / (2 \sigma^2))$, which we meet again as natural parameters.

The transformation $y = e^x$ turns each normal model into a lognormal model.

The function `fisher_matrix` computes {eq}`ef_fisher` by numerical integration, with the score computed by finite differences.

```{code-cell} ipython3
def fisher_matrix(logpdf, v, a, b, h=1e-5):
    """Fisher information matrix (ef_fisher) at parameter vector v, by quadrature."""
    v = np.asarray(v, dtype=float)
    e = np.eye(v.size) * h

    def score(z):
        return np.array([(logpdf(z, v + e[i]) - logpdf(z, v - e[i])) / (2 * h)
                         for i in range(v.size)])

    return np.array([[integrate.quad(
        lambda z: score(z)[i] * score(z)[j] * np.exp(logpdf(z, v)),
        a, b, limit=200)[0] for j in range(v.size)] for i in range(v.size)])


def kl_quad(logpdf, v1, v2, a, b):
    """KL divergence of model v1 from model v2, by quadrature."""
    f = lambda z: np.exp(logpdf(z, v1)) * (logpdf(z, v1) - logpdf(z, v2))
    return integrate.quad(f, a, b, limit=200)[0]


def x_normal(x, v):        # log density of X, coordinates (μ, σ)
    return norm.logpdf(x, v[0], v[1])


def y_lognormal(y, v):     # log density of Y = exp(X), coordinates (μ, σ)
    return lognorm.logpdf(y, s=v[1], scale=np.exp(v[0]))


def x_natural(x, v):       # log density of X, coordinates (μ/σ², -1/(2σ²))
    return norm.logpdf(x, -v[0] / (2 * v[1]), np.sqrt(-1 / (2 * v[1])))


(μ1, σ1), (μ2, σ2) = (1.0, 1.5), (-0.5, 1.0)
coordinates = {
    "(μ, σ)": lambda μ, σ: np.array([μ, σ]),
    "(μ, σ²)": lambda μ, σ: np.array([μ, σ**2]),
    "(μ, μ² + σ²)": lambda μ, σ: np.array([μ, μ**2 + σ**2]),
    "(μ/σ², -1/(2σ²))": lambda μ, σ: np.array([μ / σ**2, -1 / (2 * σ**2)]),
}
print("Euclidean distance between the two parameter vectors")
for name, c in coordinates.items():
    print(f"  {name:18} {np.linalg.norm(c(μ1, σ1) - c(μ2, σ2)):.4f}")

print("\nKL divergence")
print(f"  from X:          "
      f"{kl_quad(x_normal, (μ1, σ1), (μ2, σ2), -np.inf, np.inf):.6f}")
print(f"  from Y = exp(X): "
      f"{kl_quad(y_lognormal, (μ1, σ1), (μ2, σ2), 0, np.inf):.6f}")

H_x = norm(μ1, σ1).entropy()
H_y = lognorm(s=σ1, scale=np.exp(μ1)).entropy()
print("\ndifferential entropy of the first model")
print(f"  of X: {H_x:.4f}, of Y: {H_y:.4f}, difference: {H_y - H_x:.4f}")

θ1 = coordinates["(μ/σ², -1/(2σ²))"](μ1, σ1)
cases = (("from X, coordinates (μ, σ)", x_normal, (μ1, σ1), -np.inf),
         ("from Y, coordinates (μ, σ)", y_lognormal, (μ1, σ1), 0),
         ("from X, coordinates (μ/σ², -1/(2σ²))", x_natural, θ1, -np.inf))
print("\nFisher information matrix of the first model")
for label, logpdf, v, a in cases:
    I_v = np.round(fisher_matrix(logpdf, v, a, np.inf), 4) + 0.0
    print(f"  {label}:\n{I_v}")
```

The four Euclidean distances differ.

The KL divergence computed from $X$ equals the KL divergence computed from $Y$.

The differential entropy of $Y$ exceeds that of $X$ by $E \log \lvert f'(X) \rvert = E X = 1$.

The Fisher information matrices computed from $X$ and from $Y$ in coordinates $(\mu, \sigma)$ agree with {eq}`ef_fisher_normal`.

In the coordinates $(\mu / \sigma^2, -1 / (2 \sigma^2))$, the Fisher information matrix is a different matrix, $J^\top I J$.

(ef_wasserstein_section)=
### Moving mass and comparing heights

Optimal transport measures the distance between two models in another way.

The **2-Wasserstein distance** between distributions $p$ and $q$ on the real line is

$$
W_2(p, q) = \left( \min E\left[ (X - Y)^2 \right] \right)^{1/2} ,
$$ (ef_wasserstein)

where the minimum is over joint distributions of $(X, Y)$ whose marginal distributions are $p$ and $q$.

{doc}`opt_transport` solves discrete versions of this problem.

On the real line, the minimum is attained by pairing quantiles: $X = F_p^{-1}(U)$ and $Y = F_q^{-1}(U)$, where $F_p$ and $F_q$ are the two cumulative distribution functions and $U$ is uniform on $(0, 1)$.

For two normal models, this pairing is $Y = \mu_2 + (\sigma_2 / \sigma_1)(X - \mu_1)$, and

$$
W_2\left(N(\mu_1, \sigma_1^2), N(\mu_2, \sigma_2^2)\right) = \sqrt{(\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2} ,
$$

the Euclidean distance between the parameter vectors $(\mu_1, \sigma_1)$ and $(\mu_2, \sigma_2)$.

So $W_2(A, B) = W_2(C, D) = 1$: optimal transport moves the mass of $A$ to $B$, and the mass of $C$ to $D$, by one unit, although $A$ and $B$ are much easier to tell apart than $C$ and $D$.

The two kinds of distance answer different questions.

KL divergence and Fisher information compare the heights of two densities outcome by outcome and use no distance between outcomes; the Wasserstein distance measures how far probability mass must move along the real line.

{cite:t}`KhanZhang2022` call the first comparison vertical and the second horizontal.

The Wasserstein distance is a distance between models, so relabeling parameters does not change it; of the parameterizations in this lecture, it equals the Euclidean distance between parameter vectors only in $(\mu, \sigma)$.

Because it measures displacements in the units of the data, transforming the data changes it.

It is also finite for models that put their probability on different sets: the Wasserstein distance between $A$ and a point mass at zero is $0.2$, while the KL divergence between them is infinite in either order.

The function below computes {eq}`ef_wasserstein` for two distributions that are increasing functions of one standard normal random variable $Z$, so that pairing the values of $Z$ pairs quantiles.

We compute it for the four normal models and for the lognormal models that the transformation $y = e^x$ produces.

```{code-cell} ipython3
def w2_increasing(g1, g2):
    """W2 between the distributions of g1(Z) and g2(Z), where Z is standard
    normal and g1 and g2 are increasing; |Z| > 12 has negligible probability."""
    integrand = lambda z: (g1(z) - g2(z))**2 * norm.pdf(z)
    return np.sqrt(integrate.quad(integrand, -12, 12, limit=200)[0])


four_normals = {"A, B": ((0.0, 0.2), (1.0, 0.2)), "C, D": ((0.0, 2.0), (1.0, 2.0))}
for name, ((μa, σa), (μb, σb)) in four_normals.items():
    w_x = w2_increasing(lambda z: μa + σa * z, lambda z: μb + σb * z)
    w_y = w2_increasing(lambda z: np.exp(μa + σa * z),
                        lambda z: np.exp(μb + σb * z))
    kl_x = kl_quad(x_normal, (μa, σa), (μb, σb), -np.inf, np.inf)
    kl_y = kl_quad(y_lognormal, (μa, σa), (μb, σb), 0, np.inf)
    print(f"{name}: W2 from X {w_x:.4f}, from Y = exp(X) {w_y:.4f};   "
          f"KL from X {kl_x:.4f}, from Y {kl_y:.4f}")
```

After the transformation, the Wasserstein distance is $1.79$ between $A$ and $B$ and $93.8$ between $C$ and $D$, because exponentiation stretches the upper tail, where $C$ and $D$ put much more probability than $A$ and $B$.

The KL divergences do not change.

{ref}`ef_fisher_rao` compares the shortest paths that the two kinds of distance select.

### Statistics and sufficiency

A statistic or a garbling can only lose information.

Let $X$ be the data, a single observation or a whole sample, and let $p^s(\cdot\,; \theta)$ be the distribution of a statistic $s(X)$ when $X$ has distribution $p(\cdot\,; \theta)$.

For any two parameter vectors $\theta_1$ and $\theta_2$,

$$
D_{KL}\left(p^s(\cdot\,; \theta_1) \| p^s(\cdot\,; \theta_2)\right) \leq D_{KL}\left(p(\cdot\,; \theta_1) \| p(\cdot\,; \theta_2)\right) ,
$$ (ef_dpi)

and the Fisher information matrix of $s(X)$ falls short of the Fisher information matrix of $X$ by a positive semidefinite matrix.

Inequality {eq}`ef_dpi` is the data processing inequality of {doc}`blackwell_kihlstrom`.

{cite:t}`kullback1951information` showed that equality holds in {eq}`ef_dpi` for every pair of parameter vectors if and only if $s$ is sufficient.

When the parameter space is connected, the Fisher information matrix of $s(X)$ equals that of $X$ for every $\theta$ if and only if $s$ is sufficient.

{cite:t}`chentsov1981` characterized the Fisher information by this invariance.

On the set of probability distributions over a finite set, the Fisher information metric is, up to a constant factor, the only Riemannian metric that does not change under the transformations that sufficient statistics induce.

{doc}`blackwell_kihlstrom` states Chentsov's theorem.

## The exponential family

### Definition

A family of models is an **exponential family** if its densities (or probability mass functions) can be written as

$$
p(x; \theta) = h(x) \exp\left( \theta \cdot t(x) - F(\theta) \right),
\qquad
F(\theta) = \log \int h(x) \exp\left( \theta \cdot t(x) \right) dx ,
$$ (ef_family)

with the integral replaced by a sum when $x$ is discrete, and where

* $t(x) \in \mathbb{R}^d$ is the **sufficient statistic**,
* $\theta \in \mathbb{R}^d$ is the **natural parameter**,
* $h(x) \geq 0$ is the **base measure**, and
* $F(\theta)$ is the **log-partition function**, also called the cumulant function.

The name comes from statistical mechanics, where $e^{F(\theta)}$ is the partition function of a Boltzmann–Gibbs distribution.

The natural parameters range over the set $\Theta$ of $\theta$ for which the integral defining $F$ is finite.

In the maximum entropy problems of {ref}`ef_maxent`, $h$ is constant; the function `tilt` defined there uses a base distribution $q$ as $h$.

A nonconstant $h$ arises when we maximize entropy relative to a base distribution that is not uniform, a problem that {doc}`information_projections` takes up.

We assume that the representation is **minimal**: no nontrivial linear combination $a \cdot t(x)$ is constant on the set where $h(x) > 0$.

Then different $\theta$'s give different models.

We also assume that $\Theta$ is an open set, as it is for every family in this lecture.

A minimal representation is unique up to an affine change of coordinates.

If $A$ is an invertible $d \times d$ matrix, the sufficient statistic $A t(x)$ and the natural parameter $A^{-\top} \theta$ describe the same models, because $(A^{-\top} \theta) \cdot (A t(x)) = \theta \cdot t(x)$.

If $b \in \mathbb{R}^d$, the natural parameter $\theta + b$ and the base measure $h(x) e^{-b \cdot t(x)}$ also describe the same models.

Here are some familiar families written this way.

| family | $t(x)$ | $\theta$ | $F(\theta)$ |
|---|---|---|---|
| Bernoulli($\pi$) | $x$ | $\log \frac{\pi}{1 - \pi}$ | $\log(1 + e^{\theta})$ |
| Poisson($\lambda$) | $x$ | $\log \lambda$ | $e^{\theta}$ |
| exponential($\lambda$) | $x$ | $-\lambda$ | $-\log(-\theta)$ |
| $N(\mu, \sigma^2)$ | $(x, x^2)$ | $\left(\frac{\mu}{\sigma^2}, -\frac{1}{2\sigma^2}\right)$ | $-\frac{\theta_1^2}{4 \theta_2} + \frac{1}{2}\log\left(-\frac{\pi}{\theta_2}\right)$ |
| Beta($a$, $b$) | $(\log x, \log(1 - x))$ | $(a - 1, b - 1)$ | $\log B(\theta_1 + 1, \theta_2 + 1)$ |
| categorical($\pi$) | $(1_{x = 1}, \ldots, 1_{x = K - 1})$ | $\left(\log \frac{\pi_k}{\pi_K}\right)_{k < K}$ | $\log\left(1 + \sum_{k < K} e^{\theta_k}\right)$ |

In every row $h(x) = 1$, except the Poisson row, where $h(x) = 1/x!$.

In the Bernoulli and categorical rows $\pi$ is a probability or a vector of probabilities; in the normal row it is $3.14159\ldots$.

In the Beta row, $B$ is the beta function.

The categorical family on $\{1, \ldots, K\}$ uses only $K - 1$ indicators, because all $K$ indicators sum to one; with all $K$, the representation would not be minimal.

The multivariate normal family $N(\mu, \Sigma)$ is an exponential family with $t(x) = (x, x x^\top)$ and natural parameters $(\Sigma^{-1}\mu, -\frac{1}{2}\Sigma^{-1})$, where the pairing is $\theta \cdot t(x) = \theta_1 \cdot x + x^\top \Theta_2 x$ with $\Theta_2$ a symmetric matrix.

Because $x_i x_j = x_j x_i$, a minimal representation keeps only the entries of $x x^\top$ on and above the diagonal.

A family of the form {eq}`ef_family` contains every normalized geometric mixture of two of its models: the distribution proportional to $p(x; \theta_0)^{1 - a} p(x; \theta_1)^a$ has natural parameter $(1 - a)\theta_0 + a \theta_1$, for every $a \in [0, 1]$ and for any other $a$ for which that parameter lies in $\Theta$.

The sequel {doc}`information_projections` builds a geometry on this fact.

### Sufficient statistics of fixed dimension

For a model in an exponential family, the joint density of an IID sample of size $n$ is

$$
\prod_{i=1}^n p(x_i; \theta) = \exp\left( n \left[ \theta \cdot \bar t - F(\theta) \right] \right) \prod_{i=1}^n h(x_i),
\qquad
\bar t = \frac{1}{n} \sum_{i=1}^n t(x_i) .
$$

By the factorization theorem {eq}`ef_factorization`, the average $\bar t$ is sufficient for $\theta$.

Its dimension is $d$, whatever the sample size $n$.

For the normal family, $\bar t$ is the pair of sample moments $\left(\frac{1}{n} \sum_i x_i, \frac{1}{n} \sum_i x_i^2\right)$.

For the Bernoulli family, $\bar t$ is the fraction of successes.

For the Poisson family, $\bar t$ is the sample mean.

{cite:t}`Darmois1935`, {cite:t}`Koopman1936`, and {cite:t}`Pitman1936` proved a converse.

Suppose that the models in a family have densities on the real line, that the set of outcomes $x$ with $p(x; \theta) > 0$ does not depend on $\theta$, and that for every sample size $n$ an IID sample has a sufficient statistic whose dimension does not grow with $n$.

Then, under smoothness conditions, the family is an exponential family.[^pkd]

[^pkd]: The smoothness conditions rule out statistics that pack many numbers into one, such as a statistic that interleaves the decimal digits of $x_1, \ldots, x_n$. For a single observation, $x$ itself is a sufficient statistic of finite dimension; the theorem concerns how the dimension of a sufficient statistic behaves as $n$ grows.

Two families show why the conditions matter.

The uniform family on $(0, \theta)$ has the one-dimensional sufficient statistic $\max_i x_i$ for every $n$, and it is not an exponential family: its set of possible outcomes $(0, \theta)$ depends on $\theta$.

The Cauchy location family satisfies the condition on outcomes, and no statistic of fixed dimension is sufficient for it: its smallest sufficient statistic is the set of order statistics, whose dimension grows with $n$.

We draw a sample $\mathbf x$ of size $n = 8$ from $N(1, 1.5^2)$ and multiply it by an orthogonal matrix $Q$ that maps the vector $\mathbf 1 = (1, \ldots, 1)$ into itself.

The rotated sample $Q \mathbf x$ has the same sum, $\mathbf 1^\top Q \mathbf x = \mathbf 1^\top \mathbf x$, and the same sum of squares, $\mathbf x^\top Q^\top Q \mathbf x = \mathbf x^\top \mathbf x$, so it has the same sufficient statistic for the normal family.

```{code-cell} ipython3
rng_rotate = np.random.default_rng(1)
n = 8
x_sample = rng_rotate.normal(1.0, 1.5, n)

# Q is orthogonal and maps (1, ..., 1) into itself
ones = np.ones(n)
basis = np.linalg.svd(ones[None, :])[2][1:].T     # orthonormal basis of 1's complement
rotation = np.linalg.qr(rng_rotate.normal(size=(n - 1, n - 1)))[0]
Q = np.outer(ones, ones) / n + basis @ rotation @ basis.T
x_rotated = Q @ x_sample

print("sample:        ", np.round(x_sample, 3))
print("rotated sample:", np.round(x_rotated, 3))
print(f"sums:            {x_sample.sum():.6f}  {x_rotated.sum():.6f}")
print(f"sums of squares: {x_sample @ x_sample:.6f}  {x_rotated @ x_rotated:.6f}")

μ_vals, σ_vals = np.meshgrid(np.linspace(-2, 4, 61), np.linspace(0.3, 4, 38))


def log_likelihood(dist, sample, μ, σ):
    return sum(dist(μ, σ).logpdf(xi) for xi in sample)


for name, dist in (("normal", norm), ("Cauchy", cauchy)):
    gap = np.max(np.abs(log_likelihood(dist, x_sample, μ_vals, σ_vals)
                        - log_likelihood(dist, x_rotated, μ_vals, σ_vals)))
    print(f"{name:6} family: largest gap between the two log likelihood "
          f"functions = {gap:.2e}")
```

The figure shows the two log likelihood functions of the location parameter $\mu$, with the scale parameter fixed at $1.5$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Log likelihoods before and after rotation
    name: fig-ef-rotated-sample
---
μ_line = np.linspace(-2, 4, 400)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, (name, dist) in zip(axes, (("normal", norm), ("Cauchy", cauchy))):
    for sample, label, ls in ((x_sample, "sample", "-"),
                              (x_rotated, "rotated sample", "--")):
        ax.plot(μ_line, log_likelihood(dist, sample, μ_line, 1.5), ls, lw=2,
                label=label)
    ax.set_title(f"{name} family")
    ax.set_xlabel(r"$\mu$")
    ax.legend()
axes[0].set_ylabel("log likelihood")
plt.tight_layout()
plt.show()
```

The normal family cannot distinguish the two samples; the Cauchy family can.

### The log-partition function generates moments

The log-partition function determines everything else.

Differentiate $F(\theta) = \log \int h(x) \exp(\theta \cdot t(x)) \, dx$:

$$
\nabla F(\theta) = \frac{\int t(x) h(x) \exp(\theta \cdot t(x)) \, dx}{\int h(x) \exp(\theta \cdot t(x)) \, dx} = E_\theta [t(X)] .
$$ (ef_gradient)

Differentiating again gives

$$
\nabla^2 F(\theta) = E_\theta\left[ t(X) t(X)^\top \right] - E_\theta[t(X)] \, E_\theta[t(X)]^\top = \operatorname{Cov}_\theta [t(X)] .
$$ (ef_hessian)

The function $F$ is convex because a covariance matrix is positive semidefinite; minimality makes $\operatorname{Cov}_\theta[t(X)]$ positive definite, so $F$ is strictly convex.

More generally, $\log E_\theta \exp(s \cdot t(X)) = F(\theta + s) - F(\theta)$, so the derivatives of $F$ are the cumulants of $t(X)$.

Let's verify {eq}`ef_gradient` and {eq}`ef_hessian` for the Poisson and normal families, comparing finite-difference derivatives of $F$ with moments computed by summation and integration.

```{code-cell} ipython3
def F_poisson(θ):
    return np.exp(θ)


def F_normal(θ):
    θ1, θ2 = θ
    return -θ1**2 / (4 * θ2) + 0.5 * np.log(-np.pi / θ2)


def normal_θ(μ, σ):
    return np.array([μ / σ**2, -1 / (2 * σ**2)])


def gradient(F, θ, h=1e-5):
    θ = np.atleast_1d(np.asarray(θ, dtype=float))
    e = np.eye(θ.size) * h
    return np.array([np.squeeze(F(θ + e[i]) - F(θ - e[i])) / (2 * h)
                     for i in range(θ.size)])


def hessian(F, θ, h=1e-4):
    θ = np.atleast_1d(np.asarray(θ, dtype=float))
    e = np.eye(θ.size) * h
    return np.array([(gradient(F, θ + e[i]) - gradient(F, θ - e[i])) / (2 * h)
                     for i in range(θ.size)])


λ = 3.0
k = np.arange(100)
pk = poisson.pmf(k, λ)
print("Poisson:  F'(θ), F''(θ) =",
      np.round([gradient(F_poisson, np.log(λ))[0],
                hessian(F_poisson, np.log(λ))[0, 0]], 6),
      "  mean, variance =", np.round([pk @ k, pk @ k**2 - (pk @ k)**2], 6))

μ, σ = 0.5, 1.2
moments = [integrate.quad(lambda x: x**j * norm.pdf(x, μ, σ), -15, 15)[0]
           for j in range(1, 5)]
mean_t = np.array(moments[:2])
cov_t = np.array([[moments[1] - moments[0]**2,
                   moments[2] - moments[0] * moments[1]],
                  [moments[2] - moments[0] * moments[1],
                   moments[3] - moments[1]**2]])
print("\nnormal:  ∇F(θ) =", np.round(gradient(F_normal, normal_θ(μ, σ)), 6),
      "  E[t] =", np.round(mean_t, 6))
print("∇²F(θ) =\n", np.round(hessian(F_normal, normal_θ(μ, σ)), 4))
print("Cov[t] =\n", np.round(cov_t, 4))
```

## Two coordinate systems

### Expectation parameters

By {eq}`ef_gradient`, the map $\theta \mapsto \eta = \nabla F(\theta) = E_\theta[t(X)]$ sends each natural parameter to the vector of means of the sufficient statistic.

This map is one-to-one because $F$ is strictly convex.

For the families in this lecture, its range is the interior of the convex hull of the values that $t(X)$ can take: a Bernoulli mean lies strictly between $0$ and $1$, a Poisson mean is positive, and so on.

We can label each model in the family either by its natural parameter $\theta$ or by its **expectation parameter** $\eta$.

For the normal family, $\eta = (\mu, \mu^2 + \sigma^2)$; for the Poisson family, $\eta = \lambda$; for the Bernoulli family, $\eta = \pi$.

An exponential family carries three coordinate systems: conventional parameters, such as $(\mu, \sigma)$ for the normal family; natural parameters $\theta$; and expectation parameters $\eta$.

The maximum entropy problems of {ref}`ef_maxent` fix the expectation parameters $\eta = c$; their solutions deliver the natural parameters $\theta$.

### Legendre duality

The map from $\eta$ back to $\theta$ is the gradient of another convex function.

The **convex conjugate**, or Legendre transform, of $F$ is

$$
F^*(\eta) = \sup_\theta \left\{ \theta \cdot \eta - F(\theta) \right\} .
$$ (ef_conjugate)

For $\eta$ in the range of $\nabla F$, the supremum is attained where $\eta = \nabla F(\theta)$, so

$$
F(\theta) + F^*(\eta) = \theta \cdot \eta
\quad \text{when } \eta = \nabla F(\theta) ,
$$ (ef_fenchel)

and, by the envelope theorem, $\nabla F^*(\eta) = \theta$.

The two gradient maps $\nabla F$ and $\nabla F^*$ are inverse functions, and their Jacobians are inverse matrices:

$$
\nabla^2 F^*(\eta) = \left[ \nabla^2 F(\theta) \right]^{-1} .
$$

Because $F$ is convex, applying the transform twice returns $F$:

$$
F(\theta) = \max_\eta \left\{ \theta \cdot \eta - F^*(\eta) \right\} .
$$

```{note}
Economists use the same duality between a profit function and a production function.

In {doc}`houthakker_aggregation`, the industry profit function is $\Pi(p) = \max_X \{ f(X) - p \cdot X \}$, where $X$ is a vector of inputs, $p$ is the vector of their prices, and $f$ is the aggregate production function (which that lecture calls $F$).

Writing $\theta = -p$ gives $\Pi(-\theta) = \max_X \{ \theta \cdot X - (-f(X)) \}$, which has the same form as the last display.

The dictionary is

* log-partition function $F$ $\leftrightarrow$ profit function $\Pi$,
* natural parameters $\theta$ $\leftrightarrow$ minus input prices $-p$,
* expectation parameters $\eta$ $\leftrightarrow$ input quantities $X$, and
* conjugate $F^*$ $\leftrightarrow$ minus the production function $-f$.

Hotelling's lemma, $\partial \Pi / \partial p_i = -X_i$, is the counterpart of $\nabla F(\theta) = \eta$, and recovering the production function from the profit function is the counterpart of recovering $F^*$ from $F$.
```

### The conjugate is negative entropy

Evaluate $\log p(x; \theta) = \log h(x) + \theta \cdot t(x) - F(\theta)$ at a draw $X$ from $p(\cdot; \theta)$ and take expectations.

Using {eq}`ef_fenchel`,

$$
E_\theta \log p(X; \theta) = E_\theta \log h(X) + \theta \cdot \eta - F(\theta) = E_\theta \log h(X) + F^*(\eta) .
$$

So when $h(x) = 1$ wherever $p(x; \theta) > 0$,

$$
F^*(\eta) = - H(p(\cdot; \theta)) ,
$$ (ef_negentropy)

and *the conjugate of the log-partition function is the negative of the entropy, expressed as a function of the expectation parameters.*

In general, $H(p(\cdot; \theta)) = -F^*(\eta) - E_\theta \log h(X)$.

{ref}`ef_ex_poisson` works out the Poisson case, where $h(x) = 1/x!$.

For the Bernoulli family, $h = 1$, $F(\theta) = \log(1 + e^\theta)$, and $F^*(\eta) = \eta \log \eta + (1 - \eta) \log (1 - \eta)$.

In the left panel of the figure below, the tangent line to $F$ with slope $\eta$ meets the vertical axis at $-F^*(\eta)$.

In the right panel, the tangent line to $F^*$ at $\eta$ has slope $\theta$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Bernoulli log-partition function and its conjugate
    name: fig-ef-legendre
---
def F_bern(θ):
    return np.log1p(np.exp(θ))


def F_star_bern(η):
    return η * np.log(η) + (1 - η) * np.log(1 - η)


θ0 = 2.0
η0 = 1 / (1 + np.exp(-θ0))

θ_vals = np.linspace(-4, 4, 200)
η_vals = np.linspace(0.001, 0.999, 200)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.plot(θ_vals, F_bern(θ_vals), lw=2, label=r"$F(\theta)$")
ax.plot(θ_vals, -F_star_bern(η0) + η0 * θ_vals, "--",
        label=rf"tangent line with slope $\eta$ = {η0:.2f}")
ax.plot([0], [-F_star_bern(η0)], "ko")
ax.annotate(r"$-F^*(\eta)$", (0, -F_star_bern(η0)),
            xytext=(8, -14), textcoords="offset points")
ax.plot([θ0], [F_bern(θ0)], "o", color="C1")
ax.axvline(0, color="0.7", lw=0.8)
ax.set_xlabel(r"$\theta$")
ax.legend()

ax = axes[1]
ax.plot(η_vals, F_star_bern(η_vals), lw=2, label=r"$F^*(\eta) = -H$")
ax.plot(η_vals, F_star_bern(η0) + θ0 * (η_vals - η0), "--",
        label=rf"tangent line with slope $\theta$ = {θ0:.2f}")
ax.plot([η0], [F_star_bern(η0)], "o", color="C1")
ax.set_xlabel(r"$\eta$")
ax.legend()
plt.tight_layout()
plt.show()
```

(ef_maxent)=
## Maximum entropy

### A die

Edwin Jaynes proposed the maximum entropy principle {cite}`Jaynes1957` and illustrated it in his 1962 Brandeis lectures with the following problem, now known as the Brandeis dice problem {cite}`Jaynes1963`.

A die has faces $x \in \{1, 2, \ldots, 6\}$.

We are told only that the average number of spots is $4.5$, rather than the $3.5$ of a fair die.

Which probability distribution $p = (p_1, \ldots, p_6)$ should we assign to the faces?

Many distributions have mean $4.5$.

Jaynes proposed choosing the one that is "maximally noncommittal with regard to missing information" {cite}`Jaynes1957`, which he took to mean the one that maximizes Shannon's entropy

$$
H(p) = - \sum_{x} p(x) \log p(x)
$$

subject to $\sum_x p(x) = 1$ and $\sum_x x \, p(x) = 4.5$.

### The solution is a model in an exponential family

Let's solve a more general version of this problem.

Let $t(x) = (t_1(x), \ldots, t_d(x))$ be a vector of functions of the outcome $x$, which takes values in a finite set.

We seek the distribution $p$ that maximizes $H(p)$ subject to the moment conditions

$$
\sum_x p(x) \, t(x) = c
$$ (ef_moments)

for a given vector $c$, and $\sum_x p(x) = 1$.

Attach a vector of Lagrange multipliers $\theta$ to {eq}`ef_moments`.

The first-order condition for $p(x)$ is $-\log p(x) - 1 + \theta \cdot t(x) - \nu = 0$, where $\nu$ is the multiplier on $\sum_x p(x) = 1$.

The solution has the form

$$
p(x; \theta) = \exp\left( \theta \cdot t(x) - F(\theta) \right),
\qquad
F(\theta) = \log \sum_x \exp\left( \theta \cdot t(x) \right) ,
$$ (ef_maxent_solution)

where $\theta$ satisfies {eq}`ef_moments`.

The function $F$ makes the probabilities sum to one.

Such a $\theta$ exists if and only if $c$ lies in the relative interior of the convex hull of the points $t(x)$, that is, in its interior as a subset of the smallest affine subspace that contains it; for the die below, if and only if $1 < c < 6$.

When $c$ lies on the relative boundary, as for a die whose mean is $6$, a maximizing distribution still exists.

It puts zero probability on some outcomes, and it lies outside the family, as a limit of models whose $\theta$ diverges.

The models {eq}`ef_maxent_solution` form an exponential family with sufficient statistic $t$ and constant base measure.

The Lagrange multipliers $\theta$ are natural parameters, and the constrained means $c$ are expectation parameters $\eta$.

To confirm that {eq}`ef_maxent_solution` is a maximum, let $p$ be any distribution that satisfies {eq}`ef_moments`, and let $p^*$ be the model in the exponential family that satisfies it.

Because $\log p^*(x) = \theta \cdot t(x) - F(\theta)$ is an affine function of $t(x)$, and $p$ and $p^*$ assign the same mean $c$ to $t$,

$$
\sum_x p(x) \log p^*(x) = \theta \cdot c - F(\theta) = \sum_x p^*(x) \log p^*(x) = - H(p^*) .
$$

Therefore

$$
H(p^*) - H(p) = \sum_x p(x) \log p(x) - \sum_x p(x) \log p^*(x) = D_{KL}(p \| p^*) \geq 0 ,
$$ (ef_entropy_gap)

with equality only if $p = p^*$.

Among all distributions that satisfy the moment conditions, the model in the exponential family has the largest entropy, and the entropy shortfall of any other distribution equals its KL divergence from that model.

The sequel {doc}`information_projections` shows that {eq}`ef_entropy_gap` is an instance of a Pythagorean theorem.

For Jaynes's die, $t(x) = x$ and $\theta$ is a scalar.

The function `tilt` below computes the **exponential tilt** of a **base distribution** $q$: the distribution proportional to $q(x) \exp(\theta \, t(x))$ for a statistic $t$.

For the maximum entropy problem, $q$ is uniform, and the result is {eq}`ef_maxent_solution`.

The second function finds the $\theta$ that delivers a given mean.

```{code-cell} ipython3
def tilt(q, t, θ):
    """The distribution proportional to q(x) exp(θ t(x))."""
    log_w = np.log(q) + θ * t
    return np.exp(log_w - logsumexp(log_w))


def match_mean(q, t, c):
    """Find θ such that the tilted distribution has mean c."""
    θ = brentq(lambda θ: tilt(q, t, θ) @ t - c, -50, 50)
    return θ, tilt(q, t, θ)


def entropy(p):
    return -p @ np.log(p)


def kl(p, q):
    return p @ np.log(p / q)


faces = np.arange(1, 7)
uniform_die = np.full(6, 1 / 6)
θ_die, p_die = match_mean(uniform_die, faces, 4.5)
print(f"θ = {θ_die:.4f}")
print("maximum entropy distribution:", np.round(p_die, 4))
print(f"entropy = {entropy(p_die):.4f}, compared with log 6 = {np.log(6):.4f}")
```

The probabilities rise geometrically, by the factor $e^{\theta}$ from one face to the next.

Let's check {eq}`ef_entropy_gap` by drawing other distributions with mean $4.5$.

We construct them by moving away from $p^*$ in directions that preserve both the total probability and the mean.

```{code-cell} ipython3
rng = np.random.default_rng(0)

# directions orthogonal to (1, ..., 1) and to (1, 2, ..., 6)
constraints = np.vstack([np.ones(6), faces])
null_space = np.linalg.svd(constraints)[2][2:].T

print(" H(p*) - H(p)    KL(p || p*)")
for _ in range(5):
    while True:
        p = p_die + null_space @ rng.normal(scale=0.05, size=4)
        if np.all(p > 0):
            break
    print(f"{entropy(p_die) - entropy(p):12.6f} {kl(p, p_die):14.6f}")
```

The figure shows the maximum entropy distributions for several means.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Maximum entropy die distributions by mean
    name: fig-ef-die
---
fig, ax = plt.subplots()
for c in (2.5, 3.5, 4.5, 5.5):
    θ, p = match_mean(uniform_die, faces, c)
    ax.plot(faces, p, "o-", label=rf"mean {c}, $\theta$ = {θ:.2f}")
ax.set_xlabel("face")
ax.set_ylabel("probability")
ax.legend()
plt.show()
```

### Shadow prices

*The Lagrange multipliers of the maximum entropy problem are the shadow prices of the moment constraints.*

By {eq}`ef_negentropy`, the maximized entropy is $H(p^*) = -F^*(c)$, so by the envelope theorem $\partial H(p^*) / \partial c = -\nabla F^*(c) = -\theta$.

The multiplier $\theta_j$ is the rate at which entropy falls as the constrained mean $c_j$ rises.

For Jaynes's die, $\theta > 0$ at $c = 4.5$, because pushing the mean further from $3.5$ requires a less uniform distribution.

### The normal distribution

Now let the outcome $x$ be a real number, and constrain its mean and its second moment:

$$
E[X] = \mu, \qquad E[X^2] = \mu^2 + \sigma^2 .
$$

For a density $p$ on the real line, the **differential entropy** is $H(p) = -\int p(x) \log p(x) \, dx$.

The argument that led to {eq}`ef_maxent_solution` now says that the density with the largest entropy has the form

$$
p(x; \theta) = \exp\left( \theta_1 x + \theta_2 x^2 - F(\theta) \right) ,
$$

which is integrable only if $\theta_2 < 0$.

Completing the square shows that this is the normal density with $\theta_1 = \mu / \sigma^2$ and $\theta_2 = -1/(2\sigma^2)$.

The argument that delivered {eq}`ef_entropy_gap` goes through with sums replaced by integrals, so that every density $p$ with mean $\mu$ and variance $\sigma^2$ satisfies

$$
H(\varphi) - H(p) = D_{KL}(p \| \varphi) \geq 0 ,
$$ (ef_normal_gap)

where $\varphi$ is the $N(\mu, \sigma^2)$ density.

*Among all densities on the real line with a given mean and variance, the normal density has the largest entropy.*

Its entropy is $H(\varphi) = \frac{1}{2} \log(2 \pi e \sigma^2)$.

Unlike the entropy of a discrete distribution, differential entropy can be negative, and it changes when we change the units in which $x$ is measured, as the table in {ref}`ef_invariants` records; the gap {eq}`ef_normal_gap` is a KL divergence and does neither.

We check {eq}`ef_normal_gap` for several densities with mean zero and variance one.

We compute the entropies with SciPy's formulas and the KL divergences by numerical integration.

```{code-cell} ipython3
φ = norm(0, 1)
alternatives = {
    "uniform": uniform(-np.sqrt(3), 2 * np.sqrt(3)),
    "Laplace": laplace(0, 1 / np.sqrt(2)),
    "logistic": logistic(0, np.sqrt(3) / np.pi),
}

print(f"entropy of N(0, 1): {φ.entropy():.4f}\n")
print(f"{'distribution':12} {'variance':>8} {'entropy':>8} "
      f"{'H(φ) - H(p)':>12} {'KL(p || φ)':>11}")
for name, dist in alternatives.items():
    a, b = dist.support()
    a, b = max(a, -30), min(b, 30)
    kl_p_φ = integrate.quad(
        lambda x: dist.pdf(x) * (dist.logpdf(x) - φ.logpdf(x)), a, b,
        limit=200)[0]
    print(f"{name:12} {dist.var():8.4f} {dist.entropy():8.4f} "
          f"{φ.entropy() - dist.entropy():12.4f} {kl_p_φ:11.4f}")
```

We can also watch the normal distribution emerge from a discrete maximum entropy problem.

Put the outcome on a fine grid of points and impose the two moment conditions.

The function below finds the natural parameters by Newton's method, using {eq}`ef_gradient` and {eq}`ef_hessian`: the gradient of $F$ is the vector of means of $t$, and the Hessian of $F$ is the covariance matrix of $t$.

```{code-cell} ipython3
def maxent(q, T, c, tol=1e-12, max_iter=100):
    """
    Maximize entropy relative to q subject to E[T] = c.

    T is an m × d array whose rows are t(x).  Returns θ and the
    distribution proportional to q(x) exp(θ · t(x)).  The iterations
    converge only if c is in the interior of the convex hull of the
    rows of T.
    """
    def tilted(θ):
        log_w = np.log(q) + T @ θ
        return np.exp(log_w - logsumexp(log_w))

    θ = np.zeros(T.shape[1])
    for _ in range(max_iter):
        p = tilted(θ)
        mean = p @ T
        cov = (T - mean).T @ ((T - mean) * p[:, None])
        step = np.linalg.solve(cov, c - mean)
        θ = θ + step
        if np.max(np.abs(step)) < tol:
            break
    else:
        raise RuntimeError("no convergence: is c inside the hull of t(x)?")
    return θ, tilted(θ)


μ_grid, σ_grid = 1.0, 1.2
grid = np.linspace(-6, 6, 241)
spacing = grid[1] - grid[0]
T_grid = np.column_stack([grid, grid**2])
θ_grid, p_grid = maxent(np.full(grid.size, 1 / grid.size), T_grid,
                        c=np.array([μ_grid, μ_grid**2 + σ_grid**2]))
print(f"θ = {np.round(θ_grid, 4)}")
print(f"(μ/σ², -1/(2σ²)) = "
      f"{np.round([μ_grid / σ_grid**2, -1 / (2 * σ_grid**2)], 4)}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Grid maximum entropy matches the normal
    name: fig-ef-grid-normal
---
fig, ax = plt.subplots()
ax.plot(grid, p_grid / spacing, lw=4, alpha=0.4,
        label="maximum entropy on a grid")
ax.plot(grid, norm.pdf(grid, μ_grid, σ_grid), "k--", lw=1,
        label=f"N({μ_grid:g}, {σ_grid:g}²)")
ax.set_xlabel("$x$")
ax.legend(loc="upper right")
plt.show()
```

### Other moment conditions

Different moment conditions produce different families.

| outcomes | constrained means | maximum entropy distribution |
|---|---|---|
| $\{1, \ldots, m\}$ | none | uniform |
| $\{0, 1, 2, \ldots\}$ | $E[X]$ | geometric |
| $[a, b]$ | none | uniform |
| $[0, \infty)$ | $E[X]$ | exponential |
| $(0, \infty)$ | $E[X]$, $E[\log X]$ | gamma |
| $(0, \infty)$ | $E[\log X]$, $E[(\log X)^2]$ | lognormal |
| $(0, 1)$ | $E[\log X]$, $E[\log(1 - X)]$ | Beta |
| $\mathbb{R}$ | $E[\lvert X \rvert]$ | Laplace |
| $\mathbb{R}$ | $E[X]$, $E[X^2]$ | normal |
| $\mathbb{R}^m$ | $E[X]$, $E[X X^\top]$ | multivariate normal |

The last row implies that, among all densities on $\mathbb{R}^m$ with a given mean vector and covariance matrix $\Sigma$, the multivariate normal density has the largest entropy, $\frac{1}{2} \log \det(2 \pi e \Sigma)$.

The lognormal row is the normal row applied to $\log X$: a density on $(0, \infty)$ proportional to $\exp(\theta_1 \log x + \theta_2 (\log x)^2)$ is a lognormal density.

{ref}`ef_ex_gamma` asks you to work out the gamma case.

### Logit choice probabilities

The duality between the log-partition function and entropy is familiar to economists who use the multinomial logit model of discrete choice.

A decision maker chooses one of $K$ alternatives with utilities $v_1, \ldots, v_K$ plus independent type I extreme-value (Gumbel) shocks with scale $\tau$.

The probabilities of the choices are

$$
\pi_k = \frac{\exp(v_k / \tau)}{\sum_j \exp(v_j / \tau)} ,
$$

and the expected maximized utility is $\tau \log \sum_j \exp(v_j / \tau) + \tau \gamma$, where $\gamma \approx 0.5772$ is Euler's constant.

Dropping the constant $\tau \gamma$ leaves the **inclusive value** $\tau \log \sum_j \exp(v_j / \tau)$.

The inclusive value is $\tau$ times $\log \sum_j e^{\theta_j}$ evaluated at $\theta = v / \tau$.

The function $\theta \mapsto \log \sum_j e^{\theta_j}$ is the log-partition function of the categorical family with all $K$ indicators $1_{x = k}$ as sufficient statistics.

That representation is not minimal, because the indicators sum to one.

Adding the same constant to every $v_k$ leaves the choice probabilities unchanged.

Equation {eq}`ef_gradient` still holds, so the gradient of the inclusive value with respect to $v$ is the vector of choice probabilities.

Discrete choice theorists call this fact the Williams–Daly–Zachary theorem.

It holds for any additive random utility model; the logit model is the special case with Gumbel shocks.

Because the conjugate of $\theta \mapsto \log \sum_j e^{\theta_j}$ is the negative entropy on the probability simplex,

$$
\tau \log \sum_j \exp(v_j / \tau) = \max_{\pi} \left\{ \sum_k \pi_k v_k + \tau H(\pi) \right\} ,
$$ (ef_logit)

where the maximum is over probability vectors $\pi$ and is attained at the logit probabilities.

{cite:t}`AndersonDePalmaThisse1988` interpret {eq}`ef_logit` as the indirect utility of a representative consumer whose consumption shares are the logit probabilities.

More generally, for any base distribution $q$ with $q_k > 0$ for all $k$,

$$
\tau \log \sum_k q_k \exp(v_k / \tau) = \max_{\pi} \left\{ \sum_k \pi_k v_k - \tau D_{KL}(\pi \| q) \right\} ,
$$ (ef_logit_base)

with the maximum attained at $\pi^*_k \propto q_k \exp(v_k / \tau)$.

To verify {eq}`ef_logit_base`, expand $D_{KL}(\pi \| \pi^*)$ to find that, for every probability vector $\pi$,

$$
\sum_k \pi_k v_k - \tau D_{KL}(\pi \| q) = \tau \log \sum_k q_k \exp(v_k / \tau) - \tau D_{KL}(\pi \| \pi^*) ,
$$

and the last term is zero only when $\pi = \pi^*$.

When $q$ is uniform, $D_{KL}(\pi \| q) = \log K - H(\pi)$, and {eq}`ef_logit_base` is {eq}`ef_logit` minus the constant $\tau \log K$.

{cite:t}`MattssonWeibull2002` interpret {eq}`ef_logit_base` as the problem of a decision maker who chooses a probability distribution $\pi$ over alternatives and pays a cost $\tau D_{KL}(\pi \| q)$ for departing from a default distribution $q$.

They call this cost a disutility of effort.

The maximizer $\pi^*$ is an exponential tilt of $q$.

The sequel {doc}`information_projections` finds the same tilt in the worst-case models of robust control, in the RAS method for updating input–output tables, and in the entropy-regularized optimal transport problem behind the {cite:t}`ChooSiow2006` model of the marriage market.

Let's check {eq}`ef_logit` and {eq}`ef_logit_base` numerically.

```{code-cell} ipython3
v = np.array([1.0, 2.0, 0.5, 1.5])
τ = 0.7
inclusive_value = τ * logsumexp(v / τ)
π_logit = np.exp(v / τ - logsumexp(v / τ))
print(f"{'inclusive value:':36} {inclusive_value:.6f}")
objective = π_logit @ v + τ * entropy(π_logit)
print(f"{'π·v + τ H(π) at logit π:':36} {objective:.6f}")

# other probability vectors do worse
best_other = max(π @ v + τ * entropy(π)
                 for π in rng.dirichlet(np.ones(4), size=10_000))
print(f"{'best of 10,000 random π:':36} {best_other:.6f}")

# a nonuniform base distribution
q = np.array([0.1, 0.2, 0.3, 0.4])
π_q = tilt(q, v / τ, 1.0)
print(f"\n{'τ log Σ q exp(v/τ):':36} {τ * logsumexp(v / τ, b=q):.6f}")
print(f"{'π·v - τ D(π || q) at π ∝ q e^(v/τ):':36} "
      f"{π_q @ v - τ * kl(π_q, q):.6f}")
```

## Bregman divergences and KL divergence

### Bregman divergence

A convex function lies above its tangent lines.

The **Bregman divergence** generated by a strictly convex function $F$ measures by how much:

$$
B_F(\theta_1, \theta_2) = F(\theta_1) - F(\theta_2) - \nabla F(\theta_2) \cdot (\theta_1 - \theta_2) \geq 0 .
$$ (ef_bregman)

It is the vertical gap at $\theta_1$ between $F$ and the tangent line at $\theta_2$, and it is zero only when $\theta_1 = \theta_2$.

A Bregman divergence is generally not symmetric in its two arguments.

When $F(\theta) = \frac{1}{2} \lVert \theta \rVert^2$, it is half the squared Euclidean distance.

### KL divergence is a Bregman divergence

Take two models in an exponential family, with natural parameters $\theta_1$ and $\theta_2$ and expectation parameters $\eta_1$ and $\eta_2$.

Since $\log p(x; \theta_1) - \log p(x; \theta_2) = (\theta_1 - \theta_2) \cdot t(x) - F(\theta_1) + F(\theta_2)$, taking expectations under $p(\cdot; \theta_1)$ gives

$$
D_{KL}\left( p_{\theta_1} \| p_{\theta_2} \right)
= F(\theta_2) - F(\theta_1) - \eta_1 \cdot (\theta_2 - \theta_1)
= B_F(\theta_2, \theta_1) .
$$ (ef_kl_bregman)

The arguments trade places: *the KL divergence of $p_{\theta_1}$ from $p_{\theta_2}$ is the Bregman divergence of $\theta_2$ from $\theta_1$.*

Using {eq}`ef_fenchel` to replace $F(\theta_1)$, we can also write

$$
D_{KL}\left( p_{\theta_1} \| p_{\theta_2} \right)
= F(\theta_2) + F^*(\eta_1) - \theta_2 \cdot \eta_1
= B_{F^*}(\eta_1, \eta_2) ,
$$ (ef_kl_dual)

in which the arguments appear in the same order as in the KL divergence.

KL divergence is a Bregman divergence in either coordinate system, generated by $F$ in natural parameters and by $F^*$ in expectation parameters.

The two formulas compute one invariant number in two coordinate systems.

Let's check {eq}`ef_kl_bregman` and {eq}`ef_kl_dual` for Poisson and normal distributions, computing the KL divergence directly by summation and integration.

```{code-cell} ipython3
def bregman(F, θ1, θ2):
    """B_F(θ1, θ2), using a numerical gradient of F at θ2."""
    θ1, θ2 = np.atleast_1d(θ1), np.atleast_1d(θ2)
    return np.squeeze(F(θ1) - F(θ2)) - gradient(F, θ2) @ (θ1 - θ2)


def F_star_poisson(η):
    return η * np.log(η) - η


# Poisson
λ1, λ2 = 3.0, 5.0
k = np.arange(150)
kl_direct = poisson.pmf(k, λ1) @ (poisson.logpmf(k, λ1) - poisson.logpmf(k, λ2))
print("Poisson")
print(f"  direct KL(p1 || p2)   = {kl_direct:.6f}")
print(f"  B_F(θ2, θ1)           = "
      f"{bregman(F_poisson, np.log(λ2), np.log(λ1)):.6f}")
print(f"  B_F*(η1, η2)          = {bregman(F_star_poisson, λ1, λ2):.6f}")
print(f"  B_F(θ1, θ2), reversed = "
      f"{bregman(F_poisson, np.log(λ1), np.log(λ2)):.6f}")

# normal
(μ1, σ1), (μ2, σ2) = (0.0, 1.0), (1.0, 2.0)
kl_direct = integrate.quad(
    lambda x: norm.pdf(x, μ1, σ1)
    * (norm.logpdf(x, μ1, σ1) - norm.logpdf(x, μ2, σ2)), -20, 20)[0]
print("\nnormal")
print(f"  direct KL(p1 || p2)   = {kl_direct:.6f}")
print(f"  B_F(θ2, θ1)           = "
      f"{bregman(F_normal, normal_θ(μ2, σ2), normal_θ(μ1, σ1)):.6f}")
print(f"  B_F(θ1, θ2), reversed = "
      f"{bregman(F_normal, normal_θ(μ1, σ1), normal_θ(μ2, σ2)):.6f}")
```

For the normal family, writing out $B_F(\theta_2, \theta_1)$ recovers the familiar formula

$$
D_{KL}\left( N(\mu_1, \sigma_1^2) \| N(\mu_2, \sigma_2^2) \right)
= \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2 \sigma_2^2} - \frac{1}{2} ,
$$ (ef_kl_normal)

which reduces to the expression $(\mu_1 - \mu_2)^2/(2\sigma^2)$ that we used for the four normal distributions when $\sigma_1 = \sigma_2 = \sigma$.

The multivariate version of this formula appears in {doc}`advanced:entropy`.

{doc}`approximation_errors` applies {eq}`ef_kl_normal` frequency by frequency to compute the KL divergence rate between two stationary Gaussian processes.

### Fisher information is the curvature of KL divergence

When $\theta_2$ is close to $\theta_1$, a second-order Taylor expansion of {eq}`ef_kl_bregman` gives

$$
D_{KL}\left( p_{\theta} \| p_{\theta + \delta} \right) \approx \frac{1}{2} \, \delta^\top \nabla^2 F(\theta) \, \delta .
$$

For the exponential family, $\nabla^2 F(\theta)$ is the Fisher information matrix {eq}`ef_fisher`, because the score $\nabla_\theta \log p(X; \theta) = t(X) - \nabla F(\theta)$ has covariance matrix $\operatorname{Cov}_\theta[t(X)] = \nabla^2 F(\theta)$.

*The Fisher information matrix measures how fast nearby distributions become distinguishable.*

This explains the four normal models with which we began.

With $\sigma$ fixed, the Fisher information for $\mu$ is $1/\sigma^2$, so a unit change in $\mu$ moves the distribution by $\frac{1}{2} \cdot 1^2 / \sigma^2$, which is $12.5$ when $\sigma = 0.2$ and $0.125$ when $\sigma = 2$.

For a change in the mean alone, the quadratic formula is exact.

In expectation parameters, the Fisher information matrix is $\nabla^2 F^*(\eta) = I(\theta)^{-1}$.

If $\theta$ is a function of another parameter $\xi$, with Jacobian $J = \partial \theta / \partial \xi$, then the Fisher information matrix for $\xi$ is $J^\top I(\theta) J$; with $\xi = \eta$, $J = \nabla^2 F^*(\eta)$ and $J^\top I(\theta) J = I(\theta)^{-1}$.

The quadratic approximations in the two coordinate systems agree to second order in the size of a displacement and differ for large displacements.

{doc}`specification_tests` shows that the likelihood ratio, Lagrange multiplier, and Hausman statistics of a test of independence of irrelevant alternatives are $2m$ times a KL divergence and quadratic approximations of this kind in three coordinates.

The figure shows this for the Poisson family, comparing $D_{KL}(p_{\lambda_0} \| p_\lambda)$ with its quadratic approximations in natural parameters, $\frac{1}{2} \lambda_0 (\log \lambda - \log \lambda_0)^2$, and in expectation parameters, $\frac{1}{2} (\lambda - \lambda_0)^2 / \lambda_0$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Poisson KL divergence and quadratic approximations
    name: fig-ef-fisher
---
λ0 = 4.0
λ_vals = np.linspace(1, 10, 200)
kl_vals = λ0 * np.log(λ0 / λ_vals) + λ_vals - λ0

fig, ax = plt.subplots()
ax.plot(λ_vals, kl_vals, lw=3, label=r"$D_{KL}(p_{\lambda_0} \| p_\lambda)$")
ax.plot(λ_vals, 0.5 * λ0 * np.log(λ_vals / λ0)**2, "--",
        label="quadratic in natural parameter")
ax.plot(λ_vals, 0.5 * (λ_vals - λ0)**2 / λ0, ":",
        label="quadratic in expectation parameter")
ax.set_xlabel(r"$\lambda$")
ax.set_ylim(0, 2.5)
ax.legend(loc="upper center")
plt.show()
```

Near $\lambda_0$ the three curves coincide; farther away they separate.

(ef_fisher_rao)=
### The Fisher–Rao distance

KL divergence is not symmetric, and it violates the triangle inequality.

The Fisher information matrix defines a distance.

Let $\theta(\tau)$, $0 \leq \tau \leq 1$, be a smooth path of models in a family.

Its **length** is

$$
L = \int_0^1 \sqrt{\dot \theta(\tau)^\top I(\theta(\tau)) \, \dot \theta(\tau)} \, d\tau .
$$ (ef_path_length)

The **Fisher–Rao distance** between two models is the length of the shortest path in the family that joins them {cite}`Rao1945`.

The length {eq}`ef_path_length` does not depend on coordinates: under a reparameterization, $\dot \theta = J \dot \xi$ and the Fisher information matrix becomes $J^\top I J$, so the integrand does not change.

Fisher information does not change under invertible transformations of the data or under sufficient statistics, so the Fisher–Rao distance does not change either.

For nearby models, the Fisher–Rao distance is approximately $\sqrt{\delta^\top I(\theta) \delta} \approx \sqrt{2 D_{KL}(p_\theta \| p_{\theta + \delta})}$.

For the normal family in coordinates $(\mu, \sigma)$, {eq}`ef_fisher_normal` gives the squared length element

$$
ds^2 = \frac{d\mu^2 + 2 \, d\sigma^2}{\sigma^2} .
$$

With $u = \mu / \sqrt{2}$, $ds^2 = 2 (du^2 + d\sigma^2) / \sigma^2$, twice the squared length element of the Poincaré half-plane, a model of hyperbolic geometry.

In coordinates $(u, \sigma)$, the shortest paths are vertical lines and semicircles centered on the axis $\sigma = 0$.

The Fisher–Rao distance between two normal models is {cite}`CostaSantosStrapasson2015`

$$
d_{FR}\left(N(\mu_1, \sigma_1^2), N(\mu_2, \sigma_2^2)\right)
= \sqrt{2} \, \operatorname{arccosh}\left( 1 + \frac{(\mu_1 - \mu_2)^2 / 2 + (\sigma_1 - \sigma_2)^2}{2 \sigma_1 \sigma_2} \right) .
$$ (ef_fisher_rao_normal)

We compute {eq}`ef_fisher_rao_normal` for the four normal models with which we began.

```{code-cell} ipython3
def fisher_rao_normal(μ1, σ1, μ2, σ2):
    """Fisher–Rao distance (ef_fisher_rao_normal) between two normal models."""
    return np.sqrt(2) * np.arccosh(
        1 + ((μ1 - μ2)**2 / 2 + (σ1 - σ2)**2) / (2 * σ1 * σ2))


def normal_geodesic(μ1, σ1, μ2, σ2, n_points=401):
    """
    Points (μ, σ) on the shortest path between two normal models with
    μ1 ≠ μ2: a semicircle in the (μ/√2, σ) half-plane.
    """
    u1, u2 = μ1 / np.sqrt(2), μ2 / np.sqrt(2)
    center = (u1**2 + σ1**2 - u2**2 - σ2**2) / (2 * (u1 - u2))
    radius = np.hypot(u1 - center, σ1)
    angles = np.linspace(np.arctan2(σ1, u1 - center),
                         np.arctan2(σ2, u2 - center), n_points)
    return np.sqrt(2) * (center + radius * np.cos(angles)), radius * np.sin(angles)


pairs = {"A, B": (0.0, 0.2, 1.0, 0.2), "C, D": (0.0, 2.0, 1.0, 2.0)}
for name, (μa, σa, μb, σb) in pairs.items():
    kl_ab = np.log(σb / σa) + (σa**2 + (μa - μb)**2) / (2 * σb**2) - 0.5
    μ_path, σ_path = normal_geodesic(μa, σa, μb, σb)
    print(f"{name}: Fisher–Rao distance {fisher_rao_normal(μa, σa, μb, σb):.4f}, "
          f"sqrt(2 KL) = {np.sqrt(2 * kl_ab):.4f}, "
          f"midpoint of the shortest path N({μ_path[200]:.2f}, {σ_path[200]:.3f}²)")
```

The path from $A$ to $B$ that holds $\sigma = 0.2$ fixed has length $1 / 0.2 = 5$, which equals $\sqrt{2 D_{KL}(A \| B)}$.

The shortest path has length $3.775$; it widens the distribution on the way, and at its midpoint it reaches $N(0.5, 0.406^2)$.

For $C$ and $D$, the shortest path barely bends, and the Fisher–Rao distance $0.497$ is close to $\sqrt{2 D_{KL}(C \| D)} = 0.5$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Shortest paths between normal models
    name: fig-ef-fisher-rao
---
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, (name, (μa, σa, μb, σb)) in zip(axes, pairs.items()):
    μ_path, σ_path = normal_geodesic(μa, σa, μb, σb)
    ax.plot(μ_path, σ_path, lw=2, label="shortest path")
    ax.plot([μa, μb], [σa, σb], "k--", lw=1, label=r"path with $\sigma$ fixed")
    ax.plot([μa, μb], [σa, σb], "ko")
    for μ, σ, label in ((μa, σa, name[0]), (μb, σb, name[-1])):
        ax.annotate(label, (μ, σ), xytext=(4, -14), textcoords="offset points")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma$")
    ax.set_ylim(0, 1.3 * σ_path.max())
    ax.legend(loc="lower center")
plt.tight_layout()
plt.show()
```

We measure the length of the shortest path from $A$ to $B$ in two coordinate systems: in $(\mu, \sigma)$ with {eq}`ef_fisher_normal`, and in natural parameters with the Fisher information matrix $\nabla^2 F(\theta)$.

```{code-cell} ipython3
μ_path, σ_path = normal_geodesic(0.0, 0.2, 1.0, 0.2, n_points=2001)

dμ, dσ = np.diff(μ_path), np.diff(σ_path)
σ_mid = (σ_path[1:] + σ_path[:-1]) / 2
length_μσ = np.sum(np.sqrt(dμ**2 + 2 * dσ**2) / σ_mid)

θ_path = np.array([normal_θ(μ, σ) for μ, σ in zip(μ_path, σ_path)])
dθ = np.diff(θ_path, axis=0)
θ_mid = (θ_path[1:] + θ_path[:-1]) / 2
length_θ = sum(np.sqrt(d @ hessian(F_normal, m) @ d) for d, m in zip(dθ, θ_mid))

print(f"length in (μ, σ):                {length_μσ:.4f}")
print(f"length in natural parameters:    {length_θ:.4f}")
print(f"formula (ef_fisher_rao_normal):  {fisher_rao_normal(0.0, 0.2, 1.0, 0.2):.4f}")
```

The two lengths agree with each other and with {eq}`ef_fisher_rao_normal`.

The Fisher–Rao distance depends on the family through which paths may pass.

A path between two normal models can leave the normal family and pass through other densities.

The map $p \mapsto 2 \sqrt{p}$ sends each density to a point on a sphere of radius $2$, because $\int (2 \sqrt{p})^2 \, dx = 4$.

It also turns the Fisher quadratic form $\int (dp)^2 / p \, dx$ into the squared Euclidean length $\int (d \, 2 \sqrt{p})^2 \, dx$.

So the shortest path between two densities, when every density is allowed, is an arc of a great circle, and its length is

$$
d_{FR}(p, q) = 2 \arccos \int \sqrt{p(x) \, q(x)} \, dx ,
$$ (ef_fisher_rao_all)

where $\int \sqrt{p q} \, dx$ is the Bhattacharyya coefficient of {doc}`divergence_measures`.

{cite:t}`KhanZhang2022` describe the same construction for distributions on a finite set.

```{code-cell} ipython3
for name, (μa, σa, μb, σb) in pairs.items():
    bc = integrate.quad(lambda x: np.sqrt(norm.pdf(x, μa, σa) * norm.pdf(x, μb, σb)),
                        -np.inf, np.inf)[0]
    print(f"{name}: through normal models {fisher_rao_normal(μa, σa, μb, σb):.5f}, "
          f"through all densities {2 * np.arccos(bc):.5f}")
```

For $A$ and $B$, a path through densities that are not normal is shorter, $3.054$ against $3.775$.

For $C$ and $D$, the two distances agree to four decimal places.

The Wasserstein distance {eq}`ef_wasserstein` selects other shortest paths.

In the coordinates $(\mu, \sigma)$ it is the Euclidean distance, so its shortest path between two normal models is the straight line between their parameter vectors.

For $A$ and $B$, that is the path with $\sigma$ fixed, the dashed line in the figure of shortest paths.

Each model on it is the distribution of $(1 - \tau) X + \tau Y$, $0 \leq \tau \leq 1$, where $(X, Y)$ has the joint distribution that attains {eq}`ef_wasserstein`.

{cite:t}`McCann1997` called this path **displacement interpolation**, and {cite:t}`Takatsu2011` studied it for normal models.

The figure plots the models at the midpoints of three paths from $A$ to $B$: the Wasserstein path, the Fisher–Rao path within the normal family, and the mixture path $(1 - \tau) A + \tau B$, which {doc}`information_projections` studies.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Midpoints of three interpolating paths
    name: fig-ef-midpoints
---
x_mid = np.linspace(-1, 2, 600)
μ_fr, σ_fr = normal_geodesic(0.0, 0.2, 1.0, 0.2, n_points=401)
midpoints = {
    "Wasserstein: N(0.5, 0.2²)": norm.pdf(x_mid, 0.5, 0.2),
    f"Fisher–Rao: N({μ_fr[200]:.1f}, {σ_fr[200]:.3f}²)":
        norm.pdf(x_mid, μ_fr[200], σ_fr[200]),
    "mixture: (A + B) / 2": 0.5 * (norm.pdf(x_mid, 0, 0.2) + norm.pdf(x_mid, 1, 0.2)),
}
fig, ax = plt.subplots()
for μ in (0.0, 1.0):
    ax.plot(x_mid, norm.pdf(x_mid, μ, 0.2), color="0.75", lw=1,
            label="A and B" if μ == 0 else None)
for label, density in midpoints.items():
    ax.plot(x_mid, density, lw=2, label=label)
ax.set_xlabel("$x$")
ax.set_ylim(0, 2.9)
ax.legend(loc="upper center", ncol=2, fontsize=9, frameon=False)
plt.show()
```

The Wasserstein path slides the mass of $A$ toward $B$ and keeps its shape.

The Fisher–Rao path spreads the mass out and gathers it again.

The mixture path leaves the mass where it is and changes the heights of the two humps.

(ef_estimation)=
## Estimation

### Maximum likelihood matches moments

Let $x_1, \ldots, x_n$ be IID draws from a model in an exponential family.

The average log likelihood is

$$
\frac{1}{n} \sum_{i=1}^n \log p(x_i; \theta) = \theta \cdot \bar t - F(\theta) + \frac{1}{n} \sum_{i=1}^n \log h(x_i),
\qquad
\bar t = \frac{1}{n} \sum_{i=1}^n t(x_i) .
$$

The statistic $t$ is sufficient: apart from the last term, which does not involve $\theta$, the likelihood depends on the data only through $\bar t$.

The average log likelihood is a concave function of $\theta$ whose first-order condition is $\nabla F(\hat \theta) = \bar t$.

*The maximum likelihood estimator of the expectation parameter is the sample average of the sufficient statistic*, $\hat \eta = \bar t$.

The maximum likelihood estimator of the natural parameter is $\hat \theta = \nabla F^*(\bar t)$, provided that $\bar t$ lies in the range of $\nabla F$.

When it does not, as in a Bernoulli sample with no successes, $\hat \theta$ does not exist (see {ref}`ef_ex_bernoulli`).

When both exist, the estimates $\hat \eta$ and $\hat \theta$ are two sets of coordinates of one estimated model.

The maximized average log likelihood is $F^*(\bar t)$ plus the average of $\log h(x_i)$, by {eq}`ef_fenchel`.

In expectation parameters, maximum likelihood estimation is trivial; in natural parameters, it requires inverting $\nabla F$.

For the gamma and Beta families, inverting $\nabla F$ takes a numerical method, such as the Newton iterations we used above.

### Finite-sample behavior of the two estimators

The maximum likelihood estimators in the two coordinate systems behave differently in finite samples.

Both estimators name the same estimated model; bias and covariance depend on the coordinate system, as the table in {ref}`ef_invariants` records.

The estimator $\hat \eta = \bar t$ is unbiased because it is a sample average; its covariance matrix is $\operatorname{Cov}_\theta[t(X)]/n = \nabla^2 F(\theta)/n$.

The Fisher information matrix for $\eta$ is $\nabla^2 F^*(\eta)$, so the Cramér–Rao lower bound for an unbiased estimator of $\eta$ is $\left[\nabla^2 F^*(\eta)\right]^{-1}/n = \nabla^2 F(\theta)/n$.

*In expectation parameters, the maximum likelihood estimator is unbiased and attains the Cramér–Rao bound exactly*, in every sample size.

The estimator $\hat \theta = \nabla F^*(\bar t)$ of the natural parameter is a nonlinear function of $\bar t$, and it is biased in finite samples; for the normal family, $E \hat \theta = n \theta / (n - 3)$ when $n > 3$.

The Cramér–Rao bound applies only to unbiased estimators, so for the biased $\hat \theta$ we treat $I(\theta)^{-1}/n$ as a benchmark.

In the exponential family, an unbiased estimator can attain the Cramér–Rao bound in finite samples only if the quantity it estimates is an affine function of $\eta$.

The natural parameter $\theta = \nabla F^*(\eta)$ is an affine function of $\eta$ only when $F$ is quadratic, as for a normal family with known variance.

For the normal family with unknown mean and variance, $\theta_1 = \eta_1 / (\eta_2 - \eta_1^2)$ and $\theta_2 = -1/(2(\eta_2 - \eta_1^2))$ are not affine in $\eta$, so no unbiased estimator of either attains the bound.

For the normal family, as $n \to \infty$, the bias of $\hat \theta$ vanishes and $n$ times its covariance matrix converges to $I(\theta)^{-1}$.

In general, asymptotic theory describes the distribution of $\hat \theta$, whose finite-sample moments need not exist, as {ref}`ef_ex_bernoulli` shows.

The simulation below illustrates these facts for the normal family with $\mu = 1$ and $\sigma = 1.5$, using samples of size $n = 30$.

The ellipses enclose 95 percent of the probability of normal distributions centered at the true parameters, with covariance matrix $\nabla^2 F(\theta)/n = I(\theta)/n$ in the left panel and $I(\theta)^{-1}/n$ in the right panel.

```{code-cell} ipython3
def ellipse(center, cov, level=chi2.ppf(0.95, 2), n_points=200):
    """Points on the ellipse (z - center)' cov^{-1} (z - center) = level."""
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack([np.cos(angles), np.sin(angles)])
    return center[:, None] + np.linalg.cholesky(level * cov) @ circle


μ, σ, n, R = 1.0, 1.5, 30, 20_000
θ_true = normal_θ(μ, σ)
η_true = np.array([μ, μ**2 + σ**2])
I_θ = hessian(F_normal, θ_true)        # Fisher information for θ

samples = rng.normal(μ, σ, size=(R, n))
η_hat = np.column_stack([samples.mean(axis=1), (samples**2).mean(axis=1)])
σ2_hat = η_hat[:, 1] - η_hat[:, 0]**2
θ_hat = np.column_stack([η_hat[:, 0] / σ2_hat, -1 / (2 * σ2_hat)])

print("expectation parameters")
print("  bias:", np.round(η_hat.mean(axis=0) - η_true, 4))
print("  n × sample covariance:\n", np.round(n * np.cov(η_hat.T), 3))
print("  n × Cramér–Rao bound = ∇²F(θ):\n", np.round(I_θ, 3))
print("natural parameters")
print("  bias:", np.round(θ_hat.mean(axis=0) - θ_true, 4))
print("  n × sample covariance:\n", np.round(n * np.cov(θ_hat.T), 3))
print("  n × benchmark = I(θ)^{-1}:\n", np.round(np.linalg.inv(I_θ), 3))
print(f"  E θ̂ = nθ/(n-3):", np.round(n * θ_true / (n - 3), 4),
      "  mean of θ̂:", np.round(θ_hat.mean(axis=0), 4))
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Maximum likelihood estimates in both parameterizations
    name: fig-ef-cramer-rao
---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
panels = ((axes[0], η_hat, η_true, I_θ / n,
           (r"$\hat\eta_1$", r"$\hat\eta_2$"), "expectation parameters",
           "Cramér–Rao ellipse", "lower right"),
          (axes[1], θ_hat, θ_true, np.linalg.inv(I_θ) / n,
           (r"$\hat\theta_1$", r"$\hat\theta_2$"), "natural parameters",
           r"ellipse from $I(\theta)^{-1}/n$", "lower left"))
for ax, est, truth, cov, labels, title, ell_label, loc in panels:
    ax.scatter(*est[:1000].T, s=3, alpha=0.3)
    ax.plot(*ellipse(truth, cov), "k", lw=1.5, label=ell_label)
    ax.plot(*truth, "r+", ms=14, mew=2, label="true value")
    ax.plot(*est.mean(axis=0), "kx", ms=10, mew=2, label="mean of estimates")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend(loc=loc)
plt.tight_layout()
plt.show()
```

In expectation parameters, the cloud of estimates is centered at the true value, and its covariance matrix equals the Cramér–Rao bound, up to simulation error.

In natural parameters, the cloud is skewed, its mean is displaced from the true value by the bias $3\theta/(n - 3)$, and its covariance matrix is larger than the benchmark $I(\theta)^{-1}/n$, which it approaches only as $n$ grows.

## Concluding remarks

A family is a set of models, and a parameterization gives the models coordinates.

KL divergence and the Fisher–Rao distance depend neither on the coordinates nor on invertible transformations of the data.

Sufficient statistics preserve them.

If the set of possible outcomes of a smooth family of densities does not depend on its parameter, and if its samples have sufficient statistics of a fixed finite dimension, then the family is an exponential family.

For the exponential family, one convex function $F$ delivers four facts:

* its gradient maps natural parameters $\theta$ to expectation parameters $\eta$, and the gradient of its conjugate $F^*$ maps them back;
* its Hessian is the covariance matrix of the sufficient statistic and the Fisher information matrix;
* its conjugate is the negative of entropy, when the base measure is constant; and
* the Bregman divergences of $F$ and $F^*$ are KL divergences.

Models in an exponential family solve maximum entropy problems.

The normal density maximizes entropy among densities with a given mean and variance.

The Lagrange multipliers of a maximum entropy problem are natural parameters and shadow prices.

The logit model's inclusive value is a log-partition function, and its conjugate is negative entropy.

For the normal family, the Fisher–Rao distance is a distance of hyperbolic geometry.

The Wasserstein distance of optimal transport measures how far probability mass moves; it ignores the parameterization but depends on the units of the data.

The sequel {doc}`information_projections` uses these facts to describe the geometry of KL divergence.

## Exercises

```{exercise}
:label: ef_ex_gamma

Consider densities on $(0, \infty)$.

1. Show that the density that maximizes entropy subject to given values of $E[X]$ and $E[\log X]$ is a gamma density, and write the gamma family in the form {eq}`ef_family`.
2. Compute the log-partition function $F$, and use {eq}`ef_gradient` to express $E[X]$ and $E[\log X]$ in terms of the shape and rate parameters.
3. Draw 500 observations from a gamma distribution with shape $2.5$ and rate $1.5$, and compute the maximum likelihood estimator by matching the sample means of $X$ and $\log X$, using Newton's method with the Hessian {eq}`ef_hessian`. Compare your estimates with those of `scipy.stats.gamma.fit` with the location fixed at zero.
```

```{solution-start} ef_ex_gamma
:class: dropdown
```

Here is one solution.

For part 1, the argument that led to {eq}`ef_maxent_solution` gives a density proportional to $\exp(\theta_1 x + \theta_2 \log x) = x^{\theta_2} e^{\theta_1 x}$, which is integrable on $(0, \infty)$ when $\theta_1 < 0$ and $\theta_2 > -1$.

This is the gamma density with shape $a = \theta_2 + 1$ and rate $b = -\theta_1$:

$$
p(x; a, b) = \frac{b^a}{\Gamma(a)} x^{a - 1} e^{-b x} .
$$

So $t(x) = (x, \log x)$, $\theta = (-b, a - 1)$, and $h(x) = 1$.

For part 2,

$$
F(\theta) = \log \int_0^\infty x^{\theta_2} e^{\theta_1 x} dx = \log \Gamma(\theta_2 + 1) - (\theta_2 + 1) \log(-\theta_1) .
$$

Differentiating,

$$
E[X] = \frac{\partial F}{\partial \theta_1} = \frac{\theta_2 + 1}{-\theta_1} = \frac{a}{b},
\qquad
E[\log X] = \frac{\partial F}{\partial \theta_2} = \psi(a) - \log b ,
$$

where $\psi$ is the digamma function.

The Hessian is

$$
\nabla^2 F(\theta) = \begin{bmatrix} a/b^2 & 1/b \\ 1/b & \psi'(a) \end{bmatrix} .
$$

For part 3, we apply Newton's method to $\nabla F(\theta) = \bar t$.

We start from the method-of-moments estimates, and we shorten any Newton step that would leave $\Theta$, where $a > 0$ and $b > 0$.

```{code-cell} ipython3
def gamma_mle(x, tol=1e-10, max_iter=100):
    """Match the sample means of x and log x by Newton's method in θ."""
    t_bar = np.array([x.mean(), np.log(x).mean()])
    a = x.mean()**2 / x.var()          # method-of-moments starting values
    b = a / x.mean()
    for _ in range(max_iter):
        grad_F = np.array([a / b, digamma(a) - np.log(b)])
        hess_F = np.array([[a / b**2, 1 / b],
                           [1 / b, polygamma(1, a)]])
        step = np.linalg.solve(hess_F, t_bar - grad_F)  # step in (-b, a - 1)
        s = 1.0
        while b - s * step[0] <= 0 or a + s * step[1] <= 0:
            s /= 2                                      # stay inside Θ
        b, a = b - s * step[0], a + s * step[1]
        if max(abs(s * step[0]) / b, abs(s * step[1]) / a) < tol:
            break
    else:
        raise RuntimeError("no convergence")
    return a, b


rng_gamma = np.random.default_rng(6)
x = rng_gamma.gamma(shape=2.5, scale=1 / 1.5, size=500)
a_hat, b_hat = gamma_mle(x)
a_sp, _, scale_sp = gamma_dist.fit(x, floc=0)
print(f"moment matching: shape = {a_hat:.5f}, rate = {b_hat:.5f}")
print(f"scipy fit:       shape = {a_sp:.5f}, rate = {1 / scale_sp:.5f}")
```

The two sets of estimates agree.

```{solution-end}
```

```{exercise}
:label: ef_ex_moments4

Adding powers of $x$ to the sufficient statistic produces a richer exponential family, at the cost of a log-partition function with no closed form.

Consider densities on the real line with

$$
E[X] = 0, \qquad E[X^2] = 1, \qquad E[X^3] = 0.5, \qquad E[X^4] = 3.5 .
$$

1. Show that the density with the largest entropy has the form $\exp\left(\sum_{k=1}^4 \theta_k x^k - F(\theta)\right)$, and explain why $\theta_4 < 0$.
2. Approximate the solution with the function `maxent` on a grid of points. Check that the natural parameters do not change when you widen and refine the grid.
3. Compare the entropy of the solution with the entropy of $N(0, 1)$, and verify {eq}`ef_normal_gap`.
```

```{solution-start} ef_ex_moments4
:class: dropdown
```

Here is one solution.

For part 1, the argument that led to {eq}`ef_maxent_solution` gives a density proportional to $\exp\left(\sum_{k=1}^4 \theta_k x^k\right)$.

The density is integrable only if the highest power of $x$ with a nonzero coefficient is even and its coefficient is negative.

If $\theta_4 = 0$, integrability would require $\theta_3 = 0$, and the density would be normal, with $E[X^3] = 0 \neq 0.5$.

Therefore $\theta_4 < 0$.

The integral that defines $F$ has no closed form, so we compute $\theta$ numerically.

For part 2, we solve the problem on three grids.

```{code-cell} ipython3
c4 = np.array([0.0, 1.0, 0.5, 3.5])
for bound, n_points in ((8, 641), (10, 1601), (12, 2401)):
    grid4 = np.linspace(-bound, bound, n_points)
    T4 = np.column_stack([grid4**k for k in range(1, 5)])
    θ4, p4 = maxent(np.full(grid4.size, 1 / grid4.size), T4, c4)
    print(f"grid [-{bound}, {bound}] with {n_points} points: θ = {np.round(θ4, 5)}")
```

The natural parameters agree to five decimal places.

For part 3, we compute the entropy of the solution on the last grid, its KL divergence from $N(0, 1)$, and the entropy shortfall.

```{code-cell} ipython3
spacing4 = grid4[1] - grid4[0]
density4 = p4 / spacing4
H4 = -p4 @ np.log(density4)
kl4 = p4 @ (np.log(density4) - norm.logpdf(grid4))
print(f"entropy of the solution:  {H4:.5f}")
print(f"entropy of N(0, 1):       {norm.entropy():.5f}")
print(f"H(φ) - H(p):              {norm.entropy() - H4:.5f}")
print(f"KL(p || φ):               {kl4:.5f}")

fig, ax = plt.subplots()
ax.plot(grid4, density4, lw=2, label="maximum entropy density")
ax.plot(grid4, norm.pdf(grid4), "k--", lw=1, label="N(0, 1)")
ax.set_xlim(-5, 5)
ax.set_xlabel("$x$")
ax.legend(loc="upper left")
plt.show()
```

The solution has the same mean and variance as $N(0, 1)$, so {eq}`ef_normal_gap` applies, and the entropy shortfall equals the KL divergence.

```{solution-end}
```

```{exercise}
:label: ef_ex_poisson

The Poisson family has the nonconstant base measure $h(x) = 1/x!$.

1. Use {eq}`ef_conjugate` to show that $F^*(\eta) = \eta \log \eta - \eta$, and check that $\nabla F^*$ is the inverse of $\nabla F$.
2. Show that the entropy of the Poisson($\lambda$) distribution is $H = \lambda - \lambda \log \lambda + E[\log X!]$, so that $H = -F^*(\lambda) - E \log h(X)$ and $F^* \neq -H$. Check the formula numerically for $\lambda \in \{0.5, 3, 20\}$ by computing $-\sum_x p(x) \log p(x)$ directly.
3. Use {eq}`ef_kl_dual` to derive a closed form for $D_{KL}(\text{Poisson}(\lambda_1) \| \text{Poisson}(\lambda_2))$, and check it against a direct summation.
```

```{solution-start} ef_ex_poisson
:class: dropdown
```

Here is one solution.

For part 1, $F(\theta) = e^\theta$, so the first-order condition for $\sup_\theta \{\theta \eta - e^\theta\}$ is $\eta = e^\theta$, or $\theta = \log \eta$.

Substituting, $F^*(\eta) = \eta \log \eta - \eta$.

Then $\nabla F^*(\eta) = \log \eta$, which inverts $\nabla F(\theta) = e^\theta$.

For part 2, $\log p(x) = \theta x - e^\theta - \log x!$ with $\theta = \log \lambda$, so

$$
H = -E \log p(X) = -\lambda \log \lambda + \lambda + E[\log X!] = -F^*(\lambda) - E \log h(X) .
$$

Because $E[\log X!] > 0$ whenever $\lambda > 0$, $-F^*(\lambda)$ falls short of $H$, as the last column of the table below shows.

```{code-cell} ipython3
k = np.arange(300)
print(f"{'λ':>5} {'direct H':>10} {'formula':>10} {'-F*(λ)':>10}")
for λ in (0.5, 3.0, 20.0):
    p = poisson.pmf(k, λ)
    H_direct = -p @ poisson.logpmf(k, λ)
    H_formula = λ - λ * np.log(λ) + p @ gammaln(k + 1)
    print(f"{λ:5.1f} {H_direct:10.6f} {H_formula:10.6f} "
          f"{-F_star_poisson(λ):10.6f}")
```

For part 3, {eq}`ef_kl_dual` gives

$$
D_{KL}(p_{\lambda_1} \| p_{\lambda_2})
= F^*(\lambda_1) - F^*(\lambda_2) - \log \lambda_2 \, (\lambda_1 - \lambda_2)
= \lambda_1 \log \frac{\lambda_1}{\lambda_2} - \lambda_1 + \lambda_2 ,
$$

which is the formula we used to draw {numref}`fig-ef-fisher`.

```{code-cell} ipython3
λ1, λ2 = 3.0, 5.0
direct = poisson.pmf(k, λ1) @ (poisson.logpmf(k, λ1) - poisson.logpmf(k, λ2))
print(f"direct sum:  {direct:.8f}")
print(f"closed form: {λ1 * np.log(λ1 / λ2) - λ1 + λ2:.8f}")
```

```{solution-end}
```

```{exercise}
:label: ef_ex_bernoulli

Repeat the comparison of the two coordinate systems for the Bernoulli family.

1. With $\pi = 0.2$ and $n = 25$, simulate many samples and compute $\hat \eta = \bar x$ and $\hat \theta = \log(\hat \eta / (1 - \hat \eta))$.
2. Verify that $\hat \eta$ is unbiased with variance equal to the Cramér–Rao bound $\pi(1 - \pi)/n$.
3. What fraction of samples produce an infinite $\hat \theta$? Among samples with finite $\hat \theta$, compare the mean and variance of $\hat \theta$ with $\theta$ and with the benchmark $1/(n \pi (1 - \pi))$. How do these comparisons change when $n = 400$?
```

```{solution-start} ef_ex_bernoulli
:class: dropdown
```

Here is one solution.

```{code-cell} ipython3
def bernoulli_experiment(π, n, rng, R=100_000):
    η_hat = rng.binomial(n, π, size=R) / n
    θ = np.log(π / (1 - π))
    finite = (η_hat > 0) & (η_hat < 1)
    θ_hat = np.log(η_hat[finite] / (1 - η_hat[finite]))
    print(f"n = {n}")
    print(f"  η̂: mean {η_hat.mean():.4f} (π = {π}), "
          f"variance {η_hat.var():.6f} (bound {π * (1 - π) / n:.6f})")
    print(f"  fraction with infinite θ̂: {1 - finite.mean():.4f}")
    print(f"  finite θ̂: mean {θ_hat.mean():.4f} (θ = {θ:.4f}), "
          f"variance {θ_hat.var():.4f} "
          f"(benchmark {1 / (n * π * (1 - π)):.4f})")


rng_bern = np.random.default_rng(2)
bernoulli_experiment(0.2, 25, rng_bern)
bernoulli_experiment(0.2, 400, rng_bern)
```

With $n = 25$, about $0.4$ percent of samples contain no successes.

In those samples $\bar x = 0$ lies on the boundary of the range of $\nabla F$, the interval $(0, 1)$, so the maximum likelihood estimator of $\theta$ does not exist; the formula gives $\hat \theta = -\infty$.

Among the other samples, $\hat \theta$ is biased downward, with a variance more than 20 percent above the benchmark $1/(n \pi (1 - \pi))$.

With $n = 400$, infinite estimates essentially never occur, and the bias and the gap between the variance and the benchmark are small, as asymptotic theory predicts.

Meanwhile $\hat \eta$ is unbiased and attains the Cramér–Rao bound at both sample sizes.

```{solution-end}
```

```{exercise}
:label: ef_ex_conjugate

Bayes' law takes a simple form in natural parameters.

1. Let $x_1, \ldots, x_n$ be IID Bernoulli($\pi$) draws, and let the prior for $\pi$ be Beta($a$, $b$). Using the Beta row of the table above, write the prior as a distribution in the Beta family with natural parameters $(a - 1, b - 1)$ and show that the posterior is a distribution in the same family whose natural parameters are those of the prior plus $\left(\sum_i x_i, \, n - \sum_i x_i\right)$.
2. Explain why, more generally, if the log likelihood is linear in some functions of the parameter and the prior is a distribution in an exponential family with those functions as its sufficient statistic, the posterior is in the same family and Bayes' law adds to its natural parameters.
3. Starting from a Beta(1, 1) prior and draws with $\pi = 0.3$, plot the posterior mean of $\pi$, with a band of two posterior standard deviations, as a function of $n$. After $n = 200$ draws, use {eq}`ef_gradient` and {eq}`ef_hessian` to compute the posterior mean and variance of $\log \pi$, and check them by Monte Carlo.

The Beta–Bernoulli model appears in {doc}`exchangeable`, {doc}`likelihood_bayes`, and {doc}`merging_of_opinions`.
```

```{solution-start} ef_ex_conjugate
:class: dropdown
```

Here is one solution.

For part 1, the prior density is proportional to $\pi^{a - 1}(1 - \pi)^{b - 1} = \exp\left( (a - 1) \log \pi + (b - 1) \log(1 - \pi) \right)$.

The likelihood of $n$ draws with $s = \sum_i x_i$ successes is $\pi^s (1 - \pi)^{n - s} = \exp\left( s \log \pi + (n - s) \log (1 - \pi) \right)$.

Multiplying, the posterior is proportional to $\exp\left( (a - 1 + s) \log \pi + (b - 1 + n - s) \log(1 - \pi) \right)$, a Beta($a + s$, $b + n - s$) density.

Its natural parameters are $(a - 1 + s, b - 1 + n - s)$.

For part 2, Bayes' law says that the log posterior equals the log prior plus the log likelihood, up to a constant.

If the log likelihood is linear in some functions of the parameter, as it is in $(\log \pi, \log(1 - \pi))$ here, and the log prior is linear in the same functions, then the log posterior is too, with coefficients that add.

The prior family whose sufficient statistics are those functions of the parameter is called a **conjugate** family.

This use of the word conjugate is unrelated to the convex conjugate $F^*$.

For part 3, the posterior mean of $\pi$ is $(a + s)/(a + b + n)$, and the posterior variance is $(a + s)(b + n - s) / \left((a + b + n)^2 (a + b + n + 1)\right)$.

```{code-cell} ipython3
rng_bayes = np.random.default_rng(3)
π_true, a0, b0 = 0.3, 1.0, 1.0
x = rng_bayes.binomial(1, π_true, size=500)
s = np.cumsum(x)
n_vals = np.arange(1, x.size + 1)
a_post, b_post = a0 + s, b0 + n_vals - s

post_mean = a_post / (a_post + b_post)
post_sd = np.sqrt(a_post * b_post
                  / ((a_post + b_post)**2 * (a_post + b_post + 1)))
fig, ax = plt.subplots()
ax.plot(n_vals, post_mean, label="posterior mean")
ax.fill_between(n_vals, post_mean - 2 * post_sd, post_mean + 2 * post_sd,
                alpha=0.3, label="± 2 posterior standard deviations")
ax.axhline(π_true, color="k", ls="--", lw=1, label=r"true $\pi$")
ax.set_xlabel("$n$")
ax.set_ylim(0, 1)
ax.legend()
plt.show()
```

After $n = 200$ draws the posterior is Beta($a$, $b$) with $(a, b) = (a_0 + s, b_0 + 200 - s)$.

Applying {eq}`ef_gradient` to $F(\theta) = \log B(\theta_1 + 1, \theta_2 + 1)$ gives $E[\log \pi] = \psi(a) - \psi(a + b)$, and {eq}`ef_hessian` gives $\operatorname{Var}[\log \pi] = \psi'(a) - \psi'(a + b)$.

```{code-cell} ipython3
a, b = a_post[199], b_post[199]
draws = rng_bayes.beta(a, b, size=200_000)
print(f"E[log π]:   formula {digamma(a) - digamma(a + b):.5f}, "
      f"Monte Carlo {np.log(draws).mean():.5f}")
print(f"Var[log π]: formula {polygamma(1, a) - polygamma(1, a + b):.5f}, "
      f"Monte Carlo {np.log(draws).var():.5f}")
```

```{solution-end}
```

```{exercise}
:label: ef_ex_jeffreys

The Jeffreys prior of a family has density proportional to $\sqrt{\det I(\theta)}$.

1. For the Bernoulli family in coordinates $\pi$, show that $I(\pi) = 1 / (\pi (1 - \pi))$ and that the Jeffreys prior is Beta($1/2$, $1/2$).
2. In the natural parameter $\theta = \log(\pi / (1 - \pi))$, show that $I(\theta) = \pi (1 - \pi)$. Show that the Jeffreys prior in $\theta$, with density proportional to $\sqrt{\pi (1 - \pi)}$, is the distribution of $\theta$ when $\pi$ has the Beta($1/2$, $1/2$) distribution.
3. Draw $\pi$ from the uniform distribution on $(0, 1)$ and from Beta($1/2$, $1/2$), transform each draw to $\theta$, and compare histograms of $\theta$ with the densities that the change-of-variables formula implies. Is the uniform prior on $\pi$ flat in $\theta$?
```

```{solution-start} ef_ex_jeffreys
:class: dropdown
```

Here is one solution.

For part 1, the score is $x / \pi - (1 - x) / (1 - \pi)$, and its variance is $\pi / \pi^2 + (1 - \pi) / (1 - \pi)^2 = 1 / (\pi (1 - \pi))$.

The Jeffreys prior has density proportional to $\pi^{-1/2} (1 - \pi)^{-1/2}$, the Beta($1/2$, $1/2$) density.

For part 2, $F(\theta) = \log(1 + e^\theta)$, so $I(\theta) = F''(\theta) = \pi (1 - \pi)$ by {eq}`ef_hessian`.

Because $d\pi / d\theta = \pi (1 - \pi)$, the change-of-variables formula turns the Beta($1/2$, $1/2$) density into the density

$$
\frac{\pi^{-1/2} (1 - \pi)^{-1/2}}{B(1/2, 1/2)} \, \pi (1 - \pi) = \frac{\sqrt{\pi (1 - \pi)}}{B(1/2, 1/2)}
$$

for $\theta$, which is proportional to $\sqrt{I(\theta)}$.

The same change of variables turns the uniform density for $\pi$ into the logistic density $\pi (1 - \pi) = e^\theta / (1 + e^\theta)^2$ for $\theta$.

```{code-cell} ipython3
rng_jeffreys = np.random.default_rng(4)
R_draws = 200_000
θ_vals = np.linspace(-8, 8, 400)
π_vals = 1 / (1 + np.exp(-θ_vals))
cases = {
    "uniform prior on π": (rng_jeffreys.uniform(size=R_draws),
                           π_vals * (1 - π_vals)),
    "Jeffreys prior, Beta(1/2, 1/2)": (rng_jeffreys.beta(0.5, 0.5, size=R_draws),
                                       np.sqrt(π_vals * (1 - π_vals)) / np.pi),
}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, (name, (π_draws, density)) in zip(axes, cases.items()):
    θ_draws = np.log(π_draws / (1 - π_draws))
    counts, edges = np.histogram(θ_draws, bins=80, range=(-8, 8))
    ax.stairs(counts / (R_draws * np.diff(edges)), edges, fill=True, alpha=0.4,
              label=r"histogram of $\theta$")
    ax.plot(θ_vals, density, "k", lw=1.5, label="change of variables")
    ax.set_title(name)
    ax.set_xlabel(r"$\theta$")
    ax.legend()
plt.tight_layout()
plt.show()
```

In the code, `np.pi` is $B(1/2, 1/2) = \Gamma(1/2)^2 = 3.14159\ldots$.

The uniform prior on $\pi$ is not flat in $\theta$: it puts most of its probability on $\lvert \theta \rvert < 3$.

A flat prior on $\theta$ is improper; in $\pi$ its density is proportional to $1 / (\pi (1 - \pi))$.

The Jeffreys prior is the same distribution over models in both coordinate systems.

```{solution-end}
```
