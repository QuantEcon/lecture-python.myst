---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(risk_aversion_or_mistaken_beliefs)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Risk Aversion or Mistaken Beliefs?

## Overview

This lecture explores how *risk aversion* and *mistaken beliefs* are confounded in asset pricing data.

In a rational expectations equilibrium containing a risk-averse representative investor, higher mean returns compensate for higher risks.

But in a non-rational expectations model in which a representative investor holds beliefs that differ from "the econometrician's", observed average returns depend on *both* risk aversion *and* misunderstood return distributions.

Wrong beliefs contribute what look like "stochastic discount factor shocks" when viewed from the perspective of an econometrician who trusts his model.

Those different perspectives can potentially explain observed countercyclical risk prices.

We organize a discussion of these ideas around a single mathematical device, namely, a **likelihood ratio**, a non-negative random variable with unit mean that twists one probability distribution into another.

Likelihood ratios, equivalently multiplicative martingale increments, appear in at least four distinct roles in modern asset pricing:

| Probability   | Likelihood ratio                  | Describes             |
|:--------------|:----------------------------------|:----------------------|
| Econometric   | $1$ (no twist)                    | macro risk factors    |
| Risk neutral  | $m_{t+1}^\lambda$                 | prices of risks       |
| Mistaken      | $m_{t+1}^w$                       | experts' forecasts    |
| Doubtful      | $m_{t+1} \in \mathcal{M}$         | misspecification fears|

Each likelihood ratio takes the log-normal form
$m_{t+1}^b = \exp(-b_t^\top \varepsilon_{t+1} - \frac{1}{2} b_t^\top b_t)$
with $b_t = 0$, $\lambda_t$, $w_t$, or a worst-case distortion.

The lecture draws primarily on three lines of work:

1. {cite:t}`Lucas1978` and {cite:t}`hansen1983stochastic`: a representative investor's risk
   aversion generates a likelihood ratio that prices risks.
2. {cite:t}`piazzesi2015trend`: survey data on professional forecasters
   decompose the likelihood ratio into a smaller risk price and a belief distortion.
3. {cite:t}`hansen2020twisted` and {cite:t}`szoke2022estimating`: robust control theory
   constructs twisted probability models from tilted discounted entropy balls to
   price model uncertainty, generating state-dependent uncertainty prices that
   explain puzzling term-structure movements.

We start with some standard imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from numpy.linalg import inv, eigvals, norm
from scipy.stats import norm as normal_dist
```

## Likelihood ratios and twisted densities

### The baseline model

Let $\varepsilon$ denote a vector of risks to be taken and priced. 

Under the econometrician's probability model, $\varepsilon$ has a standard multivariate normal density:

```{math}
:label: eq_baseline

\phi(\varepsilon) \propto \exp \left(-\frac{1}{2} \varepsilon^\top\varepsilon\right), \qquad \varepsilon \sim \mathcal{N}(0, I)
```

Define a **likelihood ratio**

```{math}
:label: eq_lr

m(\varepsilon) = \exp \left(-\lambda^\top\varepsilon - \frac{1}{2} \lambda^\top\lambda\right) \geq 0
```

which satisfies $E  m(\varepsilon) = 1$ when the mathematical expectation $E$ is taken with respect to the econometrician's model.

### The twisted density

The **twisted density** is

```{math}
:label: eq_twisted

\hat\phi(\varepsilon) = m(\varepsilon) \phi(\varepsilon) \propto \exp \left(-\frac{1}{2}(\varepsilon + \lambda)^\top(\varepsilon + \lambda)\right)
```

which is a $\mathcal{N}(-\lambda, I)$ density.

The likelihood ratio has shifted the mean of $\varepsilon$ from $0$ to $-\lambda$ while preserving the covariance.

````{exercise}
:label: lr_exercise_1

Verify that:

1. $E m(\varepsilon) = 1$ by computing $\int m(\varepsilon) \phi(\varepsilon) d\varepsilon$ using the moment-generating function of a standard normal.
2. The twisted density $\hat\phi(\varepsilon) = m(\varepsilon) \phi(\varepsilon)$ is indeed $\mathcal{N}(-\lambda, I)$ by combining exponents:

$$
m(\varepsilon) \phi(\varepsilon) \propto \exp \left(-\lambda^\top\varepsilon - \tfrac{1}{2}\lambda^\top\lambda\right) \exp \left(-\tfrac{1}{2}\varepsilon^\top\varepsilon\right) = \exp \left(-\tfrac{1}{2}\bigl[\varepsilon^\top\varepsilon + 2\lambda^\top\varepsilon + \lambda^\top\lambda\bigr]\right)
$$

and complete the square to obtain $-\frac{1}{2}(\varepsilon + \lambda)^\top(\varepsilon + \lambda)$.

````

````{solution} lr_exercise_1
:class: dropdown

For part 1, write $E m(\varepsilon) = \int \exp(-\lambda^\top\varepsilon - \tfrac{1}{2}\lambda^\top\lambda) \phi(\varepsilon) d\varepsilon = \exp(-\tfrac{1}{2}\lambda^\top\lambda) E[\exp(-\lambda^\top\varepsilon)]$.

The moment-generating function of $\varepsilon \sim \mathcal{N}(0,I)$ (or expectation of the log-normal random variable) gives $E[\exp(-\lambda^\top\varepsilon)] = \exp(\tfrac{1}{2}\lambda^\top\lambda)$.

So $E m(\varepsilon) = \exp(-\tfrac{1}{2}\lambda^\top\lambda)\exp(\tfrac{1}{2}\lambda^\top\lambda) = 1$. 

For part 2, combine the exponents:

$$
m(\varepsilon) \phi(\varepsilon) \propto \exp \left(-\tfrac{1}{2}\varepsilon^\top\varepsilon - \lambda^\top\varepsilon - \tfrac{1}{2}\lambda^\top\lambda\right)
$$

Recognise the argument as $-\tfrac{1}{2}(\varepsilon^\top\varepsilon + 2\lambda^\top\varepsilon + \lambda^\top\lambda) = -\tfrac{1}{2}(\varepsilon + \lambda)^\top(\varepsilon + \lambda)$.

This is the kernel of a $\mathcal{N}(-\lambda, I)$ density. 

````

### Relative entropy

The **relative entropy** (Kullback-Leibler divergence) of the twisted density with respect to the baseline density is

```{math}
:label: eq_entropy

E\bigl[m(\varepsilon)\log m(\varepsilon)\bigr] = \frac{1}{2} \lambda^\top\lambda
```

a convenient scalar measure of the statistical distance between the two models.

The vector $\lambda$ is the key object.

Depending on context it represents *risk prices*, *belief distortions*, or *worst-case mean perturbations* under model uncertainty.

For illustration, consider the scalar case $\varepsilon \in \mathbb{R}$ with $\lambda = 1.5$.

```{code-cell} ipython3
ε = np.linspace(-5, 5, 500)
λ_val = 1.5

ϕ_base  = normal_dist.pdf(ε, 0, 1)
m_lr    = np.exp(-λ_val * ε - 0.5 * λ_val**2)
ϕ_twist = m_lr * ϕ_base

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(ε, ϕ_base, 'steelblue', lw=2)
axes[0].set_xlabel(r"$\varepsilon$")

axes[1].plot(ε, m_lr, 'firebrick', lw=2)
axes[1].axhline(1, color='grey', lw=0.8, ls='--')
axes[1].set_xlabel(r"$\varepsilon$")

axes[2].plot(ε, ϕ_base, 'steelblue', lw=2,
             ls='--', alpha=0.6, label='Baseline')
axes[2].plot(ε, ϕ_twist, 'firebrick', lw=2,
             label='Twisted')
axes[2].set_xlabel(r"$\varepsilon$")
axes[2].legend()

axes[0].set_ylabel("Density")
axes[1].set_ylabel("Likelihood ratio")
axes[2].set_ylabel("Density")
plt.tight_layout()
plt.show()
```

The left panel shows the baseline $\mathcal{N}(0,1)$ density.

The middle panel shows the likelihood ratio $m(\varepsilon)$, which up-weights negative $\varepsilon$ values and down-weights positive ones when $\lambda > 0$.

The right panel shows the resulting twisted density $\hat\phi(\varepsilon) = \mathcal{N}(-\lambda, 1)$.


## The econometrician's state-space model

### State dynamics

The econometrician works with a linear Gaussian state-space system at a *monthly* frequency:

```{math}
:label: eq_state

x_{t+1} = A x_t + C \varepsilon_{t+1}
```

```{math}
:label: eq_obs

y_{t+1} = D x_t + G \varepsilon_{t+1}
```

```{math}
:label: eq_shocks

\varepsilon_{t+1} \sim \mathcal{N}(0, I)
```

Here $x_t$ is an $n \times 1$ state vector and $\varepsilon_{t+1}$ is a $k \times 1$ shock vector.

Throughout, we assume $n = k$ and that the volatility matrix $C$ is square and invertible.

This assumption is needed whenever we back out distortion coefficient matrices of the form $W = -C^{-1}(\cdot)$ from alternative transition matrices, so that the time-$t$ distortion is $w_t = W x_t$.

The observation $y_{t+1}$ represents consumption growth, $c_{t+1} - c_t = D x_t + G \varepsilon_{t+1}$.

Separately, the risk-free one-period interest rate is a linear function of the state:

$$
r_t = \delta_0 + \bar{r}^\top x_t
$$

```{figure} /_static/lecture_specific/risk_aversion_or_mistaken_beliefs/fig2_tom.png
The econometrician's model: estimated state dynamics.
```


## Asset pricing with likelihood ratios

### Risk-neutral rational expectations pricing

Under rational expectations with a risk-neutral representative investor, the stock price $p_t$ (the ex-dividend market value of a claim to the stream $\{d_{t+j}\}_{j=1}^\infty$) satisfies:

$$
p_t = \exp(-r_t) E_t(p_{t+1} + d_{t+1})
$$

The expectations theory of the term structure of interest rates prices a zero-coupon risk-free claim to one dollar at time $t+n$ as:

```{math}
:label: eq_rn_recursion

p_t(1) = \exp(-r_t), \qquad p_t(n+1) = \exp(-r_t) E_t p_{t+1}(n), \qquad p_t(n) = \exp(\bar{A}_n^{RN} + B_n^{RN} x_t)
```

The last equality states that bond prices take an **exponential-affine** form in the state.

This is a consequence of the linear Gaussian structure and can be verified by substituting the guess into the recursion and matching coefficients (see {ref}`Exercise 3 <arp_ex3>` in {doc}`Affine Models of Asset Prices <affine_risk_prices>`).

These formulas work "pretty well" for conditional means but less well for conditional variances, i.e. the Shiller *volatility puzzles*.

### Modern asset pricing: adding risk aversion

It would be convenient if versions of the same pricing formulas worked even when investors are risk averse or hold distorted beliefs.

The likelihood ratio makes this possible.

We now promote the static vector $\lambda$ from {eq}`eq_lr` to a *state-dependent* risk price vector.

With a slight abuse of notation, we now let $\lambda$ denote a $k \times n$ **matrix** of risk price coefficients, so that $\lambda_t = \lambda x_t$ is a $k \times 1$ vector at each date $t$.

In the code below, this matrix is the parameter `Λ`.

The likelihood ratio increment is

```{math}
:label: eq_sdf_lr

m_{t+1}^\lambda = \exp \left(-\lambda_t^\top \varepsilon_{t+1} - \frac{1}{2} \lambda_t^\top\lambda_t\right), \qquad \lambda_t = \lambda x_t
```

with $E_t m_{t+1}^\lambda = 1$ and $m_{t+1}^\lambda \geq 0$.

The likelihood ratio $m_{t+1}^\lambda$ distorts the conditional distribution of $\varepsilon_{t+1}$ from $\mathcal{N}(0,I)$ to $\mathcal{N}(-\lambda x_t, I)$.

Covariances of returns with $m_{t+1}^\lambda$ affect mean returns.

This is the channel through which risk aversion prices risks.

With this device, *modern asset pricing* takes the form:

For stocks (Lucas-Hansen):

```{math}
:label: eq_stock_lr

p_t = \exp(-r_t) E_t\bigl(m_{t+1}^\lambda (p_{t+1} + d_{t+1})\bigr)
```

For the term structure (Dai-Singleton-Backus-Zin):

```{math}
:label: eq_ts_lr

p_t(1) = \exp(-r_t), \qquad p_t(n+1) = \exp(-r_t) E_t\bigl(m_{t+1}^\lambda p_{t+1}(n)\bigr), \qquad p_t(n) = \exp(\bar{A}_n + B_n x_t)
```

Note that the coefficients $\bar{A}_n$, $B_n$ here differ from the risk-neutral coefficients $\bar{A}_n^{RN}$, $B_n^{RN}$ in {eq}`eq_rn_recursion` because the likelihood ratio modifies the recursion.

### Risk-neutral dynamics

The risk-neutral representation implies **twisted dynamics**.

Under the twisted measure, define $\tilde\varepsilon_{t+1} := \varepsilon_{t+1} + \lambda_t = \varepsilon_{t+1} + \lambda  x_t$.

Since $\varepsilon_{t+1} \sim \mathcal{N}(0, I)$ under the econometrician's measure, $\tilde\varepsilon_{t+1}$ has mean $\lambda x_t$ and is *not* standard normal under that measure.

However, the likelihood ratio $m_{t+1}^\lambda \propto \exp\bigl(-(\lambda x_t)^\top \varepsilon_{t+1} - \tfrac{1}{2}\|\lambda x_t\|^2\bigr)$ tilts the probability measure in exactly the right way to absorb this shift.

By the standard exponential tilting result for Gaussian models, $\tilde\varepsilon_{t+1} \sim \mathcal{N}(0, I)$ under the risk-neutral measure.

Substituting $\varepsilon_{t+1} = \tilde\varepsilon_{t+1} - \lambda  x_t$ into {eq}`eq_state` gives:

```{math}
:label: eq_rn_dynamics

x_{t+1} = (A - C\lambda) x_t + C \tilde\varepsilon_{t+1}, \qquad \tilde\varepsilon_{t+1} \sim \mathcal{N}(0,I)
```

The dependence of $\lambda_t = \lambda x_t$ on the state modifies the dynamics relative to the econometrician's model.

### Expectation under a twisted distribution

The mathematical expectation of $y_{t+1}$ under the probability distribution twisted by likelihood ratio $m_{t+1}$ is

$$
\tilde{E}_t y_{t+1} = E_t m_{t+1} y_{t+1}
$$

Under the risk-neutral dynamics, the term structure theory becomes:

$$
p_t(1) = \exp(-r_t), \qquad p_t(n+1) = \exp(-r_t) \tilde{E}_t p_{t+1}(n), \qquad p_t(n) = \exp(\tilde{\bar{A}}_n + \tilde{B}_n x_t)
$$

These are the same formulas as rational-expectations asset pricing, but expectations are taken with respect to a probability measure *twisted by risk aversion*.

The derivation of the recursive bond price coefficients is the same as in {ref}`Exercise 3 <arp_ex3>` of {doc}`Affine Models of Asset Prices <affine_risk_prices>`, applied here under the risk-neutral dynamics {eq}`eq_rn_dynamics`.

Now let's implement the state-space model and its asset pricing implications.

```{code-cell} ipython3
class LikelihoodRatioModel:
    """
    Gaussian state-space model with likelihood ratio twists.

    x_{t+1} = A x_t + C ε_{t+1},  ε ~ N(0,I)
    y_{t+1} = D x_t + G ε_{t+1}
    r_t = delta_0 + r_bar'x_t,  λ_t = Λ x_t
    """

    def __init__(self, A, C, D, G, r_bar, Λ, δ_0=0.0):
        self.A = np.atleast_2d(A).astype(float)
        self.C = np.atleast_2d(C).astype(float)
        self.D = np.atleast_2d(D).astype(float)
        self.G = np.atleast_2d(G).astype(float)
        self.r_bar = np.asarray(r_bar, dtype=float)
        self.Λ = np.atleast_2d(Λ).astype(float)
        self.δ_0 = float(δ_0)
        self.n = self.A.shape[0]
        self.k = self.C.shape[1]
        # risk-neutral dynamics
        self.A_Q = self.A - self.C @ self.Λ

    def short_rate(self, x):
        return self.δ_0 + self.r_bar @ x

    def risk_prices(self, x):
        return self.Λ @ x

    def relative_entropy(self, x):
        λ = self.risk_prices(x)
        return 0.5 * λ @ λ

    def bond_coefficients(self, n_max):
        """Bond price coefficients: log p_t(n) = A_bar_n + B_n' x_t."""
        A_bar = np.zeros(n_max + 1)
        B = np.zeros((n_max + 1, self.n))
        A_bar[1] = -self.δ_0
        B[1] = -self.r_bar
        CCt = self.C @ self.C.T
        for nn in range(1, n_max):
            A_bar[nn + 1] = A_bar[nn] + 0.5 * B[nn] @ CCt @ B[nn] - self.δ_0
            B[nn + 1] = self.A_Q.T @ B[nn] - self.r_bar
        return A_bar, B

    def yields(self, x, n_max):
        """Yield curve: y_t(n) = -(A_bar_n + B_n'x_t) / n."""
        A_bar, B = self.bond_coefficients(n_max)
        return np.array([(-A_bar[n] - B[n] @ x) / n
                         for n in range(1, n_max + 1)])

    def simulate(self, x0, T, rng=None):
        """Simulate under the econometrician's model."""
        if rng is None:
            rng = np.random.default_rng(42)
        X = np.zeros((T + 1, self.n))
        X[0] = x0
        for t in range(T):
            X[t + 1] = self.A @ X[t] + self.C @ rng.standard_normal(self.k)
        return X

    def simulate_twisted(self, x0, T, rng=None):
        """Simulate under the risk-neutral (twisted) model."""
        if rng is None:
            rng = np.random.default_rng(42)
        X = np.zeros((T + 1, self.n))
        X[0] = x0
        for t in range(T):
            X[t + 1] = self.A_Q @ X[t] + self.C @ rng.standard_normal(self.k)
        return X
```

### Example: a two-factor model

We set up a two-factor model with a persistent "level" factor and a
less persistent "slope" factor, mimicking the U.S. yield curve.

```{code-cell} ipython3
A = np.array([[0.97, -0.03],
              [0.00,  0.90]])

C = np.array([[0.007, 0.000],
              [0.000, 0.010]])

D = np.array([[0.5, 0.3]])       # consumption growth loading
G = np.array([[0.004, 0.003]])    # consumption shock loading

δ_0 = 0.004                      # short rate intercept (~4.8% annual)
r_bar = np.array([0.06, 0.04])   # short rate loading

# Risk prices
Λ = np.array([[-3.0,  0.0],
              [ 0.0, -6.0]])

model = LikelihoodRatioModel(A, C, D, G, r_bar, Λ, δ_0=δ_0)

print(f"Eigenvalues of A:   {eigvals(A).round(4)}")
print(f"Eigenvalues of A_Q: {eigvals(model.A_Q).round(4)}")
assert all(np.abs(eigvals(model.A_Q)) < 1), "A_Q must be stable!"
```

The yield curve's shape depends on the current state $x_t$.

We evaluate the model at three representative states to see how the two factors, level and slope, generate upward-sloping, relatively flat, and inverted yield curves.

```{code-cell} ipython3
n_max = 120
maturities = np.arange(1, n_max + 1)

states = {
    "Normal (upward-sloping)":  np.array([-0.005, -0.015]),
    "Relatively flat":          np.array([ 0.008, -0.005]),
    "Inverted":                 np.array([ 0.020,  0.010]),
}

fig, ax = plt.subplots(figsize=(9, 5))
for label, x in states.items():
    y = model.yields(x, n_max) * 1200    # annualise (monthly, x1200)
    ax.plot(maturities, y, lw=2, label=label)

ax.set_xlabel("Maturity (months)")
ax.set_ylabel("Yield (annualised %)")
ax.legend()
plt.tight_layout()
plt.show()
```

### Econometrician's model vs. risk-neutral model

The econometrician estimates state dynamics under the **physical measure** $P$, which governs the actual data-generating process:

$$
x_{t+1} = A x_t + C \varepsilon_{t+1}, \qquad \varepsilon_{t+1} \sim \mathcal{N}(0,I) \text{ under } P
$$

The **risk-neutral measure** $Q$ is the probability distribution twisted by the likelihood ratio $m_{t+1}^\lambda$.

Under $Q$, the state evolves as

$$
x_{t+1} = A_Q x_t + C \tilde\varepsilon_{t+1}, \qquad \tilde\varepsilon_{t+1} \sim \mathcal{N}(0,I) \text{ under } Q
$$

where $A_Q = A - C\lambda$.

The two measures agree on which events are possible but disagree on their probabilities.

The physical measure $P$ governs forecasting and estimation, while the risk-neutral measure $Q$ governs asset pricing.

```{code-cell} ipython3
print("A:\n", model.A)
print("\nA_Q = A - CΛ:\n", model.A_Q)
print(f"\nEigenvalues of A:   {eigvals(model.A).round(4)}")
print(f"Eigenvalues of A_Q: {eigvals(model.A_Q).round(4)}")
```

To see the difference in action, we simulate both models from the same initial state using the same shock sequence.

Both simulations draw the same standard normal random vectors, but the transition matrices $A$ and $A_Q$ govern how those shocks cumulate over time.

```{code-cell} ipython3
T = 300
x0 = np.array([0.01, 0.005])
rng1 = np.random.default_rng(123)
rng2 = np.random.default_rng(123)  # same seed for comparability

X_econ = model.simulate(x0, T, rng=rng1)
X_rn   = model.simulate_twisted(x0, T, rng=rng2)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for i, (ax, lab) in enumerate(zip(axes, ["Level factor", "Slope factor"])):
    ax.plot(X_econ[:, i], 'steelblue', lw=2,
            label="Econometrician (P)")
    ax.plot(X_rn[:, i], 'firebrick', lw=2,
            alpha=0.8, label="Risk-neutral (Q)")
    ax.set_ylabel(lab)
    ax.legend()

axes[1].set_xlabel("Period")
plt.tight_layout()
plt.show()
```

Both factors are more persistent under $Q$ than under $P$: the eigenvalues of $A_Q$ are closer to unity than those of $A$.

The risk-neutral paths (red) exhibit wider swings and slower mean reversion.

This is because the negative risk prices $\lambda$ in our calibration make $A_Q = A - C\lambda$ larger, slowing the rate at which factors revert to zero.

The gap between the $P$ and $Q$ dynamics is precisely what generates a term premium in bond yields, because long bonds are priced under $Q$, where risks look more persistent.

## An identification challenge

The risk price vector $\lambda_t = \lambda  x_t$ can be interpreted as either:

- a **risk price vector** expressing the representative agent's risk aversion, or
- the representative agent's **belief distortion** relative to the econometrician's
  model.

Because the pricing formulas {eq}`eq_stock_lr` to {eq}`eq_ts_lr` depend only on the composite $\lambda_t$, not on whether it reflects risk aversion or belief distortion, the two interpretations produce identical asset prices and econometric fits.

> Relative to the model of a risk-averse representative investor with rational
> expectations, a model of a risk-neutral investor with appropriately mistaken
> beliefs produces *observationally equivalent* predictions.

This insight was articulated by {cite:t}`HST_1999` and
{cite:t}`piazzesi2015trend`.

To distinguish risk aversion from belief distortion, one needs either
*more information* (the PSS approach using survey data) or *more theory*
(the Hansen-Szőke robust control approach), or both (the {cite:t}`szoke2022estimating` approach).

```{code-cell} ipython3
x_test = np.array([0.01, 0.005])
y_risk_averse = model.yields(x_test, 60) * 1200

# Mistaken belief model
model_mistaken = LikelihoodRatioModel(
    A=model.A_Q, C=C, D=D, G=G,
    r_bar=r_bar, Λ=np.zeros_like(Λ), δ_0=δ_0
)
y_mistaken = model_mistaken.yields(x_test, 60) * 1200

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.arange(1, 61), y_risk_averse, 'steelblue', lw=2,
        label='Risk averse + rational expectations')
ax.plot(np.arange(1, 61), y_mistaken, 'firebrick', lw=2, ls='--',
        label='Risk neutral + mistaken beliefs')
ax.set_xlabel("Maturity (months)")
ax.set_ylabel("Yield (annualised %)")
ax.legend()
plt.tight_layout()
plt.show()
```

The two yield curves are identical.

Without additional information (e.g., surveys of forecasters), we cannot tell them apart from asset price data alone.


## More information: experts' forecasts (PSS)

### The PSS framework

{cite:t}`piazzesi2015trend` (henceforth PSS) exploit data on professional forecasters' expectations to decompose the likelihood ratio into risk prices and belief distortions.

Their setup posits:

- The representative agent's risk aversion leads him to price risks
  $\varepsilon_{t+1}$ with prices $\lambda_t^* = \lambda^*  x_t$, where $\lambda^*$ is a $k \times n$ matrix.
- The representative agent has **twisted beliefs** $(A^*, C) = (A - C W^*, C)$
  relative to the econometrician's model $(A, C)$, where $W^*$ is a $k \times n$ matrix of belief distortion coefficients, so that $w_t^* = W^* x_t$.
- Professional forecasters use the twisted beliefs $(A^*, C)$ to answer
  survey questions about their forecasts.

### Estimation strategy

PSS proceed in four steps:

1. Use data $\{x_t\}_{t=0}^T$ to estimate the econometrician's model $A$, $C$.
2. Project experts' one-step-ahead forecasts $E_t^*[x_{t+1}]$ on $x_t$ to obtain
   $E_t^*[x_{t+1}] = A^* x_t$ and interpret $A^*$ as incorporating belief
   distortions.
3. Back out the mean distortion matrix $W^* = -C^{-1}(A^* - A)$, so that $w_t^* = W^* x_t$ is the state-dependent mean shift applied to the
   density of $\varepsilon_{t+1}$.  (This requires $C$ to be invertible.)
4. Reinterpret the $\lambda$ estimated by the rational-expectations econometrician
   as $\lambda = \lambda^* + W^*$, where $\lambda_t^* = \lambda^*  x_t$ is the
   (smaller) price of risk vector actually charged by the representative agent with
   distorted beliefs.

An econometrician who mistakenly imposes rational expectations estimates risk prices $\lambda_t = \lambda  x_t$ that sum two parts:
- *smaller risk prices* $\lambda_t^* = \lambda^*  x_t$ actually charged by the erroneous-beliefs
  representative agent, and
- *conditional mean distortions* $w_t^* = W^*  x_t$ of the risks $\varepsilon_{t+1}$ that
  the twisted-beliefs representative agent's model displays relative to the
  econometrician's.

### Numerical illustration

```{code-cell} ipython3
# Same A, C as the two-factor model above
A_econ = np.array([[0.97, -0.03],
                   [0.00,  0.90]])

A_star = np.array([[0.985, -0.025],   # experts' subjective transition
                   [0.000,  0.955]])

C_mat = np.array([[0.007, 0.000],
                  [0.000, 0.010]])

# Belief distortion
W_star = -inv(C_mat) @ (A_star - A_econ)

Λ_total = np.array([[-3.0,  0.0],
                    [ 0.0, -6.0]])
Λ_true = Λ_total - W_star   # true risk prices

print("Belief distortion W*:\n", W_star.round(3))
print("\nTotal risk prices Λ:\n", Λ_total.round(3))
print("\nTrue risk prices Λ*:\n", Λ_true.round(3))
```

```{code-cell} ipython3
x_grid = np.linspace(-0.02, 0.04, 200)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, (ax, lab) in enumerate(zip(axes,
        ["Level factor risk price", "Slope factor risk price"])):
    x_vals = np.zeros((200, 2))
    x_vals[:, i] = x_grid
    x_vals[:, 1 - i] = 0.005

    λ_total = np.array([Λ_total @ x for x in x_vals])[:, i]
    λ_true  = np.array([Λ_true @ x  for x in x_vals])[:, i]

    ax.plot(x_grid, λ_total, 'steelblue', lw=2,
            label=r"$\lambda_t$ (RE econometrician)")
    ax.plot(x_grid, λ_true, 'seagreen', lw=2,
            label=r"$\lambda^*_t$ (true risk price)")
    ax.fill_between(x_grid, λ_true, λ_total, alpha=0.15, color='firebrick',
                    label=r"$w^*_t$ (belief distortion)")
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel(f"State $x_{{{i+1},t}}$")
    ax.set_ylabel(lab)
    ax.legend()

plt.tight_layout()
plt.show()
```

PSS find that experts perceive the level and slope of the yield curve to be *more persistent* than the econometrician's estimates imply.

Subjective risk prices $\lambda^*  x_t$ vary less than the $\lambda  x_t$ estimated by the rational-expectations econometrician.

However, PSS offer no explanation for *why* beliefs are distorted.

Are they mistakes, ignorance of good econometrics, or something else?

The next two sections address this question: first by reviewing why rational expectations may be a poor approximation, and then by developing a robust control theory that *rationalises* belief distortions as optimal responses to model uncertainty.

## Rationalizing rational expectations

### The informal justification

The standard justification for rational expectations treats it as the outcome of learning from an infinite history: least-squares learning converges to rational expectations.

The argument requires that:

- agents know correct functional forms, and
- a stochastic approximation argument partitions dynamics into a fast part
  (that justifies a law of large numbers) and a slow part (that justifies an ODE).

However, long intertemporal dependencies make *rates of convergence slow*.

### Good econometricians

Good econometricians have limited data and only hunches about functional forms.

They fear that their fitted models are incorrect.

An agent who is like a good econometrician:

- has a parametric model estimated from limited data,
- acknowledges that many other specifications fit nearly as well, including other parameter
  values, other functional forms, omitted variables, neglected nonlinearities,
  and history dependencies,
- fears that one of those other models actually prevails, and
- seeks "good enough" decisions under *all* such alternative models, i.e. **robustness**.

Robust control theory formalises this idea by having the agent optimally distort probability assessments toward a worst-case scenario, producing belief distortions that look like the "mistakes" identified by PSS but that arise from a coherent response to model uncertainty rather than from ignorance.

## A theory of belief distortions: robust control

### Hansen's dubious agent

Inspired by robust control theory, consider a dubious investor who:

- shares the econometrician's model $A$, $C$, $D$, $G$,
- expresses doubts by using a continuum of likelihood ratios to form a **discounted
  entropy ball** of size $\eta$ around the econometrician's model,
- wants a valuation that is good for every model in the entropy ball, and
- constructs a *lower bound* on values and a *worst-case model* that attains it.

### Valuation under the econometrician's model

Taking the log consumption process to be linear Gaussian with shocks $\varepsilon_{t+1} \sim \mathcal{N}(0,I)$:

$$
c_{t+1} - c_t = D x_t + G \varepsilon_{t+1}, \qquad x_{t+1} = A x_t + C \varepsilon_{t+1}
$$

the dubious agent's value function is

$$
V(x_0, c_0) := E \left[\sum_{t=0}^{\infty} \beta^t c_t  \middle|  x_0, c_0\right] = c_0 + \beta E \left[V(x_1, c_1)  \middle|  x_0, c_0\right]
$$

Note that the objective is *linear* in consumption.

There is no concave utility function $u(c_t)$.

All aversion to uncertainty here comes from the *worst-case model selection* (the $\min$ over likelihood ratios below), not from utility curvature.

This separation is a key feature of the robust control approach: the agent expresses doubt through the entropy ball, rather than through a curved utility function.

### The sequence problem

The dubious agent solves a *min* problem in which a malevolent "nature" chooses the worst-case probability distortion subject to an entropy budget:

```{math}
:label: eq_hansen_seq

J(x_0, c_0 \mid \eta) := \min_{\{m_{t+1}\}} E \left[\sum_{t=0}^{\infty} \beta^t M_t c_t  \middle|  x_0, c_0\right]
```

subject to

$$
c_{t+1} - c_t = D x_t + G \varepsilon_{t+1}, \qquad x_{t+1} = A x_t + C \varepsilon_{t+1}
$$

$$
E \left[\sum_{t=0}^{\infty} \beta^t M_t E \left[m_{t+1}\log m_{t+1}  \middle|  x_t, c_t\right]  \middle|  x_0, c_0\right] \leq \eta
$$

$$
M_{t+1} = M_t m_{t+1}, \qquad E[m_{t+1} \mid x_t, c_t] = 1, \qquad M_0 = 1
$$

The cumulative likelihood ratio $M_t = \prod_{s=0}^{t-1} m_{s+1}$ converts the original probability measure into the distorted one.

The constraint bounds the *discounted entropy*.

The $M_t$ weighting ensures entropy is measured under the *distorted* measure and the $\beta^t$ discounting means future divergences are penalised less, admitting persistent alternatives.

The likelihood ratio process $\{M_t\}_{t=0}^{\infty}$ is a multiplicative **martingale**.

```{figure} /_static/lecture_specific/risk_aversion_or_mistaken_beliefs/eggs_backus.png
Discounted entropy ball around the econometrician's model.
```

### Why discounted entropy?

Discounted entropy includes models that undiscounted entropy excludes.

Undiscounted entropy over infinite sequences requires alternative models to share the same long-run averages as the baseline, thereby excluding models that differ only in persistent, low-frequency dynamics.

But those persistent alternatives are precisely the models that are hardest to distinguish from the econometrician's model with finite data and that matter most for pricing long-lived assets.

Discounted entropy, by treating future divergences less severely, admits these statistically elusive but economically important alternatives into the set of models that the dubious agent contemplates.

### Entropy and the likelihood ratio

With the log-normal likelihood ratio

$$
m_{t+1} := \exp \left(-\frac{w_t^\top w_t}{2} - w_t^\top \varepsilon_{t+1}\right)
$$

conditional entropy takes the simple form

$$
E \left[m_{t+1}\log m_{t+1}  \middle|  x_t, c_t\right] = \frac{1}{2} w_t^\top w_t
$$

Substituting into {eq}`eq_hansen_seq` and performing a change of measure (replacing $E[\cdot]$ with $E^w[\cdot]$ under the distorted model) yields the reformulated problem:

```{math}
:label: eq_hansen_reform

J(x_0, c_0 \mid \eta) := \min_{\{w_t\}} E^w \left[\sum_{t=0}^{\infty} \beta^t c_t  \middle|  x_0, c_0\right]
```

subject to

$$
c_{t+1} - c_t = D x_t + G (\tilde\varepsilon_{t+1} - w_t), \qquad x_{t+1} = A x_t + C (\tilde\varepsilon_{t+1} - w_t)
$$

$$
\frac{1}{2} E^w \left[\sum_{t=0}^{\infty} \beta^t w_t^\top w_t  \middle|  x_0, c_0\right] \leq \eta
$$

Here $\tilde\varepsilon_{t+1} \sim \mathcal{N}(0, I)$ under $E^w$, and we have substituted $\varepsilon_{t+1} = \tilde\varepsilon_{t+1} - w_t$ (so $E[\varepsilon_{t+1}] = -w_t$ under the distorted measure).

The shift $-w_t$ *reduces* expected consumption growth by $G w_t$ and shifts the state dynamics by $-C w_t$.

This is how the worst-case model makes the agent worse off.

### Outcome: constant worst-case distortion

Because the econometrician's model is linear Gaussian and the entropy constraint is a scalar bound $\eta$, the worst-case mean distortion turns out to be a *constant vector*:

$$
w_t = \bar{w}
$$

The consequence is that the contribution of $w_t$ to risk prices is *state-independent*.

This does *not* help explain countercyclical prices of risk (or prices of model uncertainty), motivating the more refined "tilted" entropy ball in the next section.

We compute $\bar{w}$ using the multiplier formulation (see {ref}`the multiplier preferences section below <mult_pref_section>`), in which the parameter $\theta$ penalises entropy: larger $\theta$ means less concern about misspecification.

In the multiplier formulation, the agent minimises

$$
E^w \left[\sum_{t=0}^\infty \beta^t \bigl(c_t + \tfrac{\theta}{2} w_t^\top w_t\bigr)\right]
$$

over $\{w_t\}$ subject to the shifted dynamics.

Since $c_t = c_0 + \sum_{s=0}^{t-1}(D x_s + G \varepsilon_{s+1})$ and $\varepsilon_{s+1} = \tilde\varepsilon_{s+1} - w_s$, the first-order condition for $w_t$ balances the entropy penalty $\theta  w_t$ against the marginal effect on discounted consumption:

$$
\theta \bar{w} = \frac{\beta}{1-\beta} G^\top + \beta C^\top v
$$

where $v$ solves $v = \frac{\beta}{1-\beta} D^\top + \beta A^\top v$, or equivalently $v = \beta (I - \beta A^\top)^{-1} D^\top / (1-\beta)$.

The vector $v$ captures the discounted cumulative effect of a unit change in $x_t$ on future consumption.

```{code-cell} ipython3
def hansen_worst_case(A, C, D, G, β, θ):
    """Constant worst-case distortion w_bar for Hansen's dubious agent."""
    n = A.shape[0]
    v = β * np.linalg.solve(np.eye(n) - β * A.T, D.T.flatten()) / (1 - β)
    w_bar = (1.0 / θ) * (β / (1 - β) * G.T.flatten() + β * C.T @ v)
    return w_bar


β = 0.995
θ_values = [0.5, 1.0, 2.0, 5.0]

print(f"{'θ':>6}  {'w_bar[0]':>10}  {'w_bar[1]':>10}  {'Entropy':>10}")
print("-" * 42)
for θ in θ_values:
    w = hansen_worst_case(A, C, D, G, β, θ)
    print(f"{θ:>6.1f}  {w[0]:>10.4f}  {w[1]:>10.4f}  {0.5 * w @ w:>10.4f}")
```

The worst-case distortion $\bar{w}$ is constant: it does not depend on the state $x_t$.

Larger $\theta$ (less concern about misspecification) yields a smaller distortion.

````{exercise}
:label: lr_exercise_3

Derive the formula for $\bar{w}$.

1. Write the discounted consumption path as $\sum_{t=0}^\infty \beta^t c_t = \frac{c_0}{1-\beta} + \sum_{t=0}^\infty \beta^t \sum_{s=0}^{t-1}(D x_s - G w_s + G \tilde\varepsilon_{s+1})$.
2. Use the state recursion $x_{t+1} = A x_t - C w_t + C \tilde\varepsilon_{t+1}$ and take first-order conditions with respect to the constant $w_t = \bar{w}$.
3. Verify that the first-order condition gives $\theta \bar{w} = \frac{\beta}{1-\beta} G^\top + \beta C^\top v$ with $v = \beta(I - \beta A^\top)^{-1} D^\top / (1-\beta)$.
4. Check numerically that larger $\theta$ brings $\bar{w}$ closer to zero.

````

````{solution} lr_exercise_3
:class: dropdown

For part 1, the consumption increment $\Delta c_{s+1} = Dx_s - G\bar{w} + G\tilde\varepsilon_{s+1}$ at date $s$ enters $c_t$ for every $t \geq s+1$, with total discounted weight $\frac{\beta^{s+1}}{1-\beta}$.

Swapping the order of summation:

$$
E \left[\sum_{t=0}^\infty \beta^t c_t\right] = \frac{c_0}{1-\beta} + \frac{1}{1-\beta}\sum_{s=0}^\infty \beta^{s+1}\bigl(D  E[x_s] - G\bar{w}\bigr)
$$

For part 2, define $S = \sum_{s=0}^\infty \beta^{s+1} E[x_s]$.

From $E[x_{s+1}] = A E[x_s] - C\bar{w}$, multiply both sides by $\beta^{s+2}$ and sum over $s = 0, 1, \ldots$:

$$
S - \beta x_0 = \beta A  S - \frac{\beta^2}{1-\beta}C\bar{w}
$$

Solving: $S = (I - \beta A)^{-1} \left(\beta x_0 - \frac{\beta^2}{1-\beta}C\bar{w}\right)$.

Substituting back, the expected objective $E[\sum \beta^t(c_t + \frac{\theta}{2}\|\bar{w}\|^2)]$ depends on $\bar{w}$ only through the term $\frac{1}{1-\beta}(D S - \frac{\beta}{1-\beta}G\bar{w}) + \frac{\theta}{2(1-\beta)}\|\bar{w}\|^2$.

For part 3, differentiate with respect to $\bar{w}$ and set to zero.

The only part of $S$ that depends on $\bar{w}$ is $-\frac{\beta^2}{1-\beta}(I-\beta A)^{-1}C\bar{w}$, so $\nabla_{\bar{w}}(D S) = -\frac{\beta^2}{1-\beta}C^\top(I - \beta A^\top)^{-1}D^\top$.

The first-order condition is:

$$
\frac{\theta}{1-\beta}\bar{w} = \frac{1}{1-\beta} \left(\frac{\beta}{1-\beta}G^\top + \frac{\beta^2}{1-\beta}C^\top(I - \beta A^\top)^{-1}D^\top\right)
$$

Simplifying: $\theta\bar{w} = \frac{\beta}{1-\beta}G^\top + \beta C^\top v$, where $v = \frac{\beta}{1-\beta}(I - \beta A^\top)^{-1}D^\top$.

Therefore $\bar{w} = \frac{1}{\theta}\bigl(\frac{\beta}{1-\beta}G^\top + \beta C^\top v\bigr)$.

For part 4, as $\theta \to \infty$, $\bar{w} = \frac{1}{\theta}(\cdots) \to 0$, which the numerical table confirms.

````

## Tilting the entropy ball

### Hansen and Szőke's more refined dubious agent

To generate *state-dependent* uncertainty prices, Hansen and Szőke introduce a more refined dubious agent who:

- shares the econometrician's model $A$, $C$, $D$, $G$,
- expresses doubts by using a continuum of likelihood ratios to form a
  discounted entropy ball around the econometrician's model, *and*
- also insists that some martingales representing particular alternative
  *parametric* models be included in the discounted entropy ball.

The inclusion of those alternative parametric models *tilts* the entropy ball, which affects the worst-case model in a way that can produce countercyclical uncertainty prices.

"Tilting" means replacing the constant entropy bound $\eta$ with a state-dependent bound $\xi(x_t)$ that is larger in states where the feared parametric alternative deviates more from the baseline.

### Concern about other parametric models

The investor wants to include particular alternative models with

$$
E_t \left[\bar{m}_{t+1}\log\bar{m}_{t+1}\right] = \frac{1}{2} \bar{w}_t^\top \bar{w}_t =: \frac{1}{2}\xi(x_t)
$$

and discounted entropy

$$
\frac{1}{2} E^{\bar{W}} \left[\sum_{t=0}^{\infty} \beta^t \xi(x_t)  \middle|  x_0, c_0\right]
$$

This is accomplished by replacing the earlier entropy constraint with

```{math}
:label: eq_tilted_constraint

\frac{1}{2} E^w \left[\sum_{t=0}^{\infty} \beta^t w_t^\top w_t  \middle|  x_0, c_0\right] \leq \frac{1}{2} E^w \left[\sum_{t=0}^{\infty} \beta^t \xi(x_t)  \middle|  x_0, c_0\right]
```

The time-$t$ contributions to the right-hand side of {eq}`eq_tilted_constraint` relax the discounted entropy constraint in states $x_t$ in which $\xi(x_t)$ is larger.

This sets the stage for *state-dependent* mean distortions in the worst-case model.

### Concern about bigger long-run risk

Inspired by {cite:t}`Bansal_Yaron_2004`, an agent fears that the true state dynamics are more persistent than the econometrician's model implies, expressed by

$$
x_{t+1} = \bar{A} x_t + C \tilde\varepsilon_{t+1}
$$

Since $\bar{A} = A - C\bar{W}$, this feared model is equivalent to shifting the mean of $\varepsilon_{t+1}$ by $-\bar{W}x_t$, giving $\bar{w}_t = \bar{W} x_t$ with

$$
\bar{W} = -C^{-1}(\bar{A} - A)
$$

(again using the assumption that $C$ is square and invertible), which implies a *quadratic* $\xi$ function:

```{math}
:label: eq_xi

\xi(x_t) := x_t^\top \bar{W}^\top\bar{W} x_t =: x_t^\top \Xi x_t
```

```{figure} /_static/lecture_specific/risk_aversion_or_mistaken_beliefs/eggs_backus2.png
Tilted discounted entropy balls. Including particular parametric alternatives with more long-run risk tilts the entropy ball and generates state-dependent worst-case distortions.
```

### The Szőke agent's sequence problem

Attaching a Lagrange multiplier $\tilde\theta \geq 0$ to the tilted entropy constraint {eq}`eq_tilted_constraint` yields the following linear-quadratic problem:

```{math}
:label: eq_szoke_seq

J(x_0, c_0 \mid \Xi) := \max_{\tilde\theta \geq 0} \min_{\{w_t\}}  E^w \left[\sum_{t=0}^{\infty} \beta^t c_t + \tilde\theta \frac{1}{2}\sum_{t=0}^{\infty} \beta^t\bigl(w_t^\top w_t - x_t^\top \Xi x_t\bigr)  \middle|  x_0, c_0\right]
```

subject to

$$
c_{t+1} - c_t = D x_t + G (\tilde\varepsilon_{t+1} - w_t), \qquad x_{t+1} = A x_t + C (\tilde\varepsilon_{t+1} - w_t)
$$

The worst-case distortion is now **affine** in the state:

$$
\tilde{w}_t = a + \tilde{W} x_t
$$

where $a$ is a constant vector determined by the linear terms in the value function.

The state-dependent component $\tilde{W} x_t$ is the new contribution of the tilted entropy ball, and is what generates countercyclical uncertainty prices.

Writing $\tilde{A} = A - C \tilde{W}$ and $\tilde{D} = D - G \tilde{W}$, the worst-case dynamics are

$$
x_{t+1} = \tilde{A} x_t - Ca + C\tilde\varepsilon_{t+1}, \qquad c_{t+1} - c_t = \tilde{D} x_t - Ga + G\tilde\varepsilon_{t+1}
$$

### Implementation: tilted entropy ball

For the inner minimisation over $\{w_t\}$ in {eq}`eq_szoke_seq`, the value function is **affine-quadratic** in the state because $c_t$ enters linearly:

$$
J(x, c) = \frac{c}{1-\beta} + v^\top x + x^\top P x + K
$$

The first-order condition for $w_t$ decomposes into a constant part (determining $a$) and a state-dependent part (determining $\tilde{W}$).

The state-dependent component satisfies (writing $\tilde\theta$ as $\theta$ in the code):

$$
(\theta I + 2\beta C^\top P C) \tilde{W} = 2\beta C^\top P A
$$

Matching quadratic terms in $x$ gives the $P$ update:

$$
P = -\tfrac{\theta}{2} \Xi + \tfrac{\theta}{2} \tilde{W}^\top \tilde{W} + \beta (A - C\tilde{W})^\top P (A - C\tilde{W})
$$

The code below iterates on the $(P, \tilde{W})$ system to convergence, solving for the state-dependent component only; the constant $a$ requires separate linear-term equations and does not generally coincide with the Hansen $\bar{w}$.

```{code-cell} ipython3
class TiltedEntropyModel:
    """
    Hansen-Szőke tilted entropy ball model.

    Given (A, C, D, G, β, θ, Ξ), computes the state-dependent
    component W_tilde of the worst-case distortion w_t = a + W_tilde x_t.
    """

    def __init__(self, A, C, D, G, β, θ, Ξ):
        self.A = np.atleast_2d(A).astype(float)
        self.C = np.atleast_2d(C).astype(float)
        self.D = np.atleast_2d(D).astype(float)
        self.G = np.atleast_2d(G).astype(float)
        self.β, self.θ = float(β), float(θ)
        self.Ξ = np.atleast_2d(Ξ).astype(float)
        self.n = self.A.shape[0]

        self.W_tilde = self._solve_worst_case()
        self.A_tilde = self.A - self.C @ self.W_tilde
        self.D_tilde = self.D - self.G @ self.W_tilde

    def _solve_worst_case(self):
        """Iterate on (P, W) system to find worst-case W_tilde."""
        n, k = self.n, self.C.shape[1]
        β, θ = self.β, self.θ

        P = np.zeros((n, n))
        converged = False
        for _ in range(2000):
            M = θ * np.eye(k) + 2 * β * self.C.T @ P @ self.C
            W = np.linalg.solve(M, 2 * β * self.C.T @ P @ self.A)
            A_w = self.A - self.C @ W
            P_new = (-(θ / 2) * self.Ξ
                    + (θ / 2) * W.T @ W
                    + β * A_w.T @ P @ A_w)
            P_new = 0.5 * (P_new + P_new.T)
            if np.max(np.abs(P_new - P)) < 1e-12:
                converged = True
                break
            P = P_new

        if not converged:
            print("Warning: (P, W) iteration did not converge")
        self._P_quad = P
        return W

    def state_dependent_distortion(self, x):
        """State-dependent component W_tilde @ x of the worst-case distortion."""
        return self.W_tilde @ x

    def state_dependent_entropy(self, x):
        """Entropy of the state-dependent component: (1/2)(W_tilde x)'(W_tilde x)."""
        w = self.state_dependent_distortion(x)
        return 0.5 * w @ w

    def xi_function(self, x):
        return x @ self.Ξ @ x
```

```{code-cell} ipython3
# Feared parametric model
A_bar = np.array([[0.995, -0.03],
                  [0.000,  0.96]])

W_bar = -inv(C) @ (A_bar - A)
Ξ = W_bar.T @ W_bar

print("Feared transition A_bar:\n", A_bar)
print("\nImplied distortion W_bar:\n", W_bar.round(3))
print("\nTilting matrix Ξ:\n", Ξ.round(1))
```

In {eq}`eq_szoke_seq` the multiplier $\tilde\theta$ is determined by the outer maximisation.  

Here we fix $\theta$ at a representative value and solve only the inner minimisation, illustrating the multiplier formulation described in {ref}`the multiplier preferences section below <mult_pref_section>`.

```{code-cell} ipython3
θ_tilt = 3.0
tilted = TiltedEntropyModel(A, C, D, G, β, θ_tilt, Ξ)

print("Worst-case distortion W_tilde:\n", tilted.W_tilde.round(4))
print("\nWorst-case transition A_tilde:\n", tilted.A_tilde.round(4))
print(f"\nEigenvalues of A:  {eigvals(A).round(4)}")
print(f"Eigenvalues of A_tilde: {eigvals(tilted.A_tilde).round(4)}")
```

````{exercise}
:label: lr_exercise_4

Derive the first-order condition for the state-dependent component of the tilted entropy problem.

1. Start from {eq}`eq_szoke_seq` and write $w_t = \bar{w} + W x_t$;
argue that the value function is affine-quadratic, $J(x,c) = c/(1-\beta) + v^\top x + x^\top P x + K$, and that the first-order condition separates into a constant part (determining $a$) and a state-dependent part (determining $W$).
2. Show that the state-dependent first-order condition gives $\theta  W = 2\beta  C^\top P (A - CW)$, which can be rearranged to the system solved in the code.
3. Derive the $P$ update by substituting the optimal $W$ back into the Bellman equation and matching quadratic terms in $x$.

````

````{solution} lr_exercise_4
:class: dropdown

For part 1, write $w_t = \bar{w} + Wx_t$.

Since $c_t$ is linear in past states and shocks, and the dynamics are linear, the value function takes the affine-quadratic form $J(x,c) = c/(1-\beta) + v^\top x + x^\top P x + K$.

The first-order condition for $w_t$ at each $t$ is:

$$
(\theta I + 2\beta C^\top P C)\, w_t = \frac{\beta}{1-\beta} G^\top + \beta C^\top v + 2\beta C^\top P A x_t
$$

The constant terms (independent of $x_t$) determine $a$; note that this differs from the Hansen $\bar{w}$ because of the $2\beta C^\top P C$ factor on the left-hand side.

The terms proportional to $x_t$ determine $W$.

For part 2, the state-dependent first-order condition from the quadratic terms is:

$$
\frac{\partial}{\partial W}\bigl[\tfrac{\theta}{2}x_t^\top W^\top W x_t + \beta x_t^\top(A-CW)^\top P(A-CW)x_t\bigr] = 0
$$

This gives $\theta W - 2\beta C^\top P(A-CW) = 0$.

Rearranging: $(\theta I + 2\beta C^\top PC) W = 2\beta C^\top PA$.

For part 3, substitute the optimal $W$ back into the Bellman equation.

The quadratic terms in $x_t$ give:

$$
P = -\tfrac{\theta}{2} \Xi + \tfrac{\theta}{2} W^\top W + \beta (A-CW)^\top P(A-CW)
$$

This is the matrix Riccati equation that the code iterates to convergence.

````

### State-dependent entropy: the key innovation

```{code-cell} ipython3
x_grid = np.linspace(-0.03, 0.04, 200)

entropy_tilted = np.array([tilted.state_dependent_entropy(np.array([x, 0.005]))
                           for x in x_grid])
ξ_vals = np.array([tilted.xi_function(np.array([x, 0.005]))
                    for x in x_grid])

# Calibrate Hansen's θ so constant entropy matches E[ξ(x_t)/2]
Σ_stat = solve_discrete_lyapunov(A, C @ C.T)
avg_ξ_half = 0.5 * np.trace(Ξ @ Σ_stat)
w_unit = hansen_worst_case(A, C, D, G, β, 1.0)
θ_hansen = norm(w_unit) / np.sqrt(2 * avg_ξ_half)
w_hansen = w_unit / θ_hansen
hansen_ent = 0.5 * w_hansen @ w_hansen

fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(hansen_ent, color='steelblue', lw=2, ls='--',
           label=rf"Hansen: constant $\frac{{1}}{{2}}\bar{{w}}^\top\bar{{w}} = {hansen_ent:.4f}$")
ax.plot(x_grid, entropy_tilted, 'firebrick', lw=2,
        label=r"Szőke (state-dep.): $\frac{1}{2}(\tilde{W}x_t)^\top(\tilde{W}x_t)$")
ax.plot(x_grid, 0.5 * ξ_vals, 'seagreen', lw=2, ls=':',
        label=r"Feared model: $\frac{1}{2}\xi(x_t)$")
ax.set_xlabel(r"Level factor $x_{1,t}$")
ax.set_ylabel("Conditional entropy")
ax.legend()
plt.tight_layout()
plt.show()
```

The key innovation of the tilted entropy ball is visible: the state-dependent component $\tilde{W} x_t$ of the worst-case distortion grows with $|x_t|$, producing *countercyclical uncertainty prices*.

By contrast, Hansen's constant distortion $\bar{w}$ has entropy $\frac{1}{2}\bar{w}^\top\bar{w}$ that does not vary with $x_t$ (shown as a horizontal line).

The Szőke parabola lies inside the feared model's entropy budget $\frac{1}{2}\xi(x_t)$ along this slice, consistent with the tilted entropy constraint {eq}`eq_tilted_constraint`.

### Three probability twisters

To summarize, three distinct probability twisters play roles in this analysis:

| Symbol         | Source                        | Describes                         |
|:---------------|:------------------------------|:----------------------------------|
| $w_t^*$        | Piazzesi, Salomão, Schneider  | Mistaken agent's beliefs          |
| $\bar{w}_t$    | Szőke's feared parametric model | Especial LRR parametric worry   |
| $\tilde{W} x_t$  | Szőke's worst-case model      | State-dependent component of worst-case distortion |

```{code-cell} ipython3
x_state = np.array([0.02, 0.008])
w_pss    = W_star @ x_state
w_feared = W_bar @ x_state
w_szoke  = tilted.state_dependent_distortion(x_state)

ε_grid = np.linspace(-4, 4, 500)
ϕ_base = normal_dist.pdf(ε_grid, 0, 1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ε_grid, ϕ_base, 'black', lw=2,
        label='Econometrician: $\\mathcal{N}(0,1)$')

for w_val, label, color, ls in [
    (w_pss[0], r"PSS mistaken $w^*_t$", 'steelblue', '-'),
    (w_feared[0], r"Feared LRR $\bar{w}_t$", 'seagreen', '--'),
    (w_szoke[0], r"Szőke worst-case $\tilde{w}_t$", 'firebrick', '-'),
]:
    ax.plot(ε_grid, normal_dist.pdf(ε_grid, -w_val, 1),
            color=color, lw=2, ls=ls, label=label)

ax.set_xlabel(r"$\varepsilon_1$")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.show()
```

Each twister shifts the econometrician's $\mathcal{N}(0,1)$ density by $-w_t$, where the direction and magnitude depend on the current state $x_t$.

For the state shown here, all three distortion vectors $w_t$ are negative in their first component, so the twisted densities $\mathcal{N}(-w_t, 1)$ are shifted slightly to the right.

The shifts are small relative to the unit variance because the state $x_t = (0.02, 0.008)$ is close to the unconditional mean.

## Empirical challenges and model performances

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: U.S. Treasury yields and yield spread
    name: fig-us-yields
---
data = pd.read_csv(
    'https://raw.githubusercontent.com/QuantEcon/lecture-python.myst/update-asset/lectures/'
    '_static/lecture_specific/risk_aversion_or_mistaken_beliefs/fred_data.csv',
    parse_dates=['DATE'], index_col='DATE'
)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                          gridspec_kw={'height_ratios': [2, 1]})

# Recession shading helper
def shade_recessions(ax, rec):
    ax.fill_between(rec.index, 0, 1,
                    where=rec.values.flatten() == 1,
                    transform=ax.get_xaxis_transform(),
                    color='grey', alpha=0.2)

rec = data['USREC'].dropna()

ax = axes[0]
shade_recessions(ax, rec)

ax.plot(data['GS1'], 'steelblue', lw=2,
        label=r'$y_{\mathrm{nom}}^{(1)}$')
ax.plot(data['GS5'], 'seagreen', lw=2,
        label=r'$y_{\mathrm{nom}}^{(5)}$')
ax.plot(data['GS10'], 'firebrick', lw=2,
        label=r'$y_{\mathrm{nom}}^{(10)}$')
ax.plot(data['DFII5'], 'seagreen', lw=2, ls='--',
        label=r'$y_{\mathrm{real}}^{(5)}$')
ax.plot(data['DFII10'], 'firebrick', lw=2, ls='--',
        label=r'$y_{\mathrm{real}}^{(10)}$')

ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('Yield (%)')
ax.legend(loc='upper right')

ax2 = axes[1]
shade_recessions(ax2, rec)

spread_10_1 = data['GS10'] - data['GS1']
ax2.plot(spread_10_1, 'steelblue', lw=2,
         label=r'$y^{(10)} - y^{(1)}$')
ax2.axhline(0, color='black', lw=0.5)
ax2.set_ylabel('Spread (%)')
ax2.set_xlabel('Year')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()
```

Several recognised patterns characterise the U.S. term structure:

- The nominal yield curve usually slopes *upward*.
- The long-minus-short yield spread *narrows before* U.S. recessions and
  *widens after* them.
- Consequently, the slope of the yield curve helps *predict* recessions and economic activity.
- Long and short yields are *almost equally volatile* (the Shiller "volatility puzzle").
- To solve the Shiller puzzle, risk prices (or something observationally equivalent)
  must *depend on volatile state variables*.

The following table summarises how various models perform:

| Model                          | Average slope | Slopes near recessions | Volatile long yield |
|:-------------------------------|:--------------|:-----------------------|:--------------------|
| {cite:t}`Lucas1978`            | no            | no                     | no                  |
| Epstein-Zin with LRR           | maybe         | yes                    | no                  |
| {cite:t}`piazzesi2015trend`    | built-in      | built-in               | yes                 |
| {cite:t}`szoke2022estimating`  | *YES*       | yes                    | yes                 |

### Why Szőke's model succeeds

Szőke's framework delivers:

1. A theory of *state-dependent belief distortions* with state-dependent component $\tilde{W} x_t$.
2. A theory about the *question that professional forecasters answer*: they
   respond with their worst-case model because they hear "tell me forecasts that
   rationalise your (max-min) decisions."
3. A way to *measure* the size of belief distortions relative to the
   econometrician's model.

```{code-cell} ipython3
model_rn = LikelihoodRatioModel(
    A, C, D, G, r_bar, Λ=np.zeros((2, 2)), δ_0=δ_0)
model_uncert = LikelihoodRatioModel(
    A, C, D, G, r_bar, Λ=tilted.W_tilde, δ_0=δ_0)

x_test = np.array([0.01, -0.03])
n_max = 120
mats = np.arange(1, n_max + 1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(mats, model_rn.yields(x_test, n_max) * 1200,
        'grey', lw=2, ls=':', label='Risk neutral')
ax.plot(mats, model.yields(x_test, n_max) * 1200,
        'steelblue', lw=2, label=r'Risk aversion ($\lambda x_t$)')
ax.plot(mats, model_uncert.yields(x_test, n_max) * 1200,
        'firebrick', lw=2, ls='--',
        label=r'Model uncertainty ($\tilde{W} x_t$)')
ax.set_xlabel("Maturity (months)")
ax.set_ylabel("Yield (annualised %)")
ax.legend()
plt.tight_layout()
plt.show()
```

The risk-aversion-only and model-uncertainty-only yield curves both slope upward, generating a term premium.

The two explanations represent *alternative channels* for the same observed term premium, reinforcing the identification challenge explored throughout this lecture.

## Cross-equation restrictions and estimation

A key appeal of the robust control approach is that it lets us deviate from rational expectations while still preserving a set of powerful **cross-equation restrictions** on decision makers' beliefs.

As {cite:t}`szoke2022estimating` puts it:

> An appealing feature of robust control theory is that it lets us deviate from
> rational expectations, but still preserves a set of powerful cross-equation
> restrictions on decision makers' beliefs. … Consequently, estimation can proceed
> essentially as with rational expectations econometrics. The main difference is
> that now restrictions through which we interpret the data emanate from the
> decision maker's best response to a worst-case model instead of to the
> econometrician's model.

### Szőke's empirical strategy

In the Szőke framework, the rational-expectations econometrician's single likelihood ratio $m_{t+1}^\lambda$ is decomposed as $\lambda_t = \tilde{w}_t + \tilde{\lambda}_t$, paralleling the PSS decomposition $\lambda = \lambda^* + W^*$.

The combined likelihood ratio is

$$
m_{t+1}^\lambda = \exp \left(-\lambda_t^\top\varepsilon_{t+1} - \frac{1}{2}\lambda_t^\top\lambda_t\right), \qquad \lambda_t = \tilde{w}_t + \tilde\lambda_t
$$

where $\tilde{w}_t = a + \tilde{W} x_t$ is the worst-case belief distortion (with state-dependent component $\tilde{W} x_t$) and $\tilde\lambda_t = \tilde\lambda x_t$ is the residual risk price.

In stage I (estimation):

1. Use $\{x_t, c_t\}_{t=0}^T$ to estimate the econometrician's $A$, $C$, $D$, $G$.
2. View $\Xi$ as a matrix of additional free parameters and estimate them
   simultaneously with risk prices $\tilde\lambda x_t$ from data
   $\{p_t(n+1)\}_{n=1}^N$, $t = 0, \ldots, T$, by imposing cross-equation
   restrictions:

$$
p_t(n+1) = \exp(-r_t) E_t \left[m_{t+1}^\lambda  p_{t+1}(n)\right]
$$

In stage II (assessment):

1. Assess improvements in predicted behaviour of the term structure.
2. Use estimated worst-case dynamics to form distorted forecasts
   $\tilde{x}_{t+1} = (A - C\tilde{W})x_t - Ca$ and compare them to those of
   professional forecasters.
3. Compute the discounted KL divergence $\frac{1}{2}E^w[\sum \beta^t w_t^\top w_t]$ of
   each twisted model relative to the econometrician's model and compare them.

```{code-cell} ipython3
def discounted_kl(W, A_w, C, x0, β, T_horizon=500):
    """Approximate discounted KL divergence (1/2) E^w [Σ β^t w_t'w_t]."""
    n_sims = 10_000
    rng = np.random.default_rng(2024)
    X = np.tile(x0, (n_sims, 1))
    total = np.zeros(n_sims)
    for t in range(T_horizon):
        w_t = X @ W.T
        total += β**t * 0.5 * np.sum(w_t**2, axis=1)
        X = X @ A_w.T + rng.standard_normal((n_sims, len(x0))) @ C.T
    return np.mean(total)

x0_test = np.array([0.01, 0.005])
kl_szoke  = discounted_kl(tilted.W_tilde, tilted.A_tilde, C, x0_test, β)
kl_feared = discounted_kl(W_bar, A_bar, C, x0_test, β)

print(f"Szőke worst-case KL: {kl_szoke:.4f}")
print(f"Feared LRR KL:       {kl_feared:.4f}")
status = ('closer to' if kl_szoke < kl_feared
          else 'farther from')
print(f"\nWorst-case model is {status} "
      f"the econometrician's model.")
```

The Szőke worst-case model has lower discounted KL divergence from the econometrician's model than the feared long-run risk model, meaning it is statistically harder to distinguish from the baseline.

Yet it still generates the state-dependent uncertainty prices needed to match term-structure dynamics.

(mult_pref_section)=
## Multiplier preferences

The constraint formulation {eq}`eq_hansen_seq` bounds discounted entropy by $\eta$, but an equivalent **multiplier** formulation replaces the constraint with a penalty term weighted by a Lagrange multiplier $\theta$.

The **multiplier preference** version of the dubious agent's problem is:

```{math}
:label: eq_mult_seq

\hat{J}(x_0, c_0 \mid \theta) := \min_{\{m_{t+1}\}} E \left[\sum_{t=0}^{\infty} \beta^t M_t\bigl(c_t + \theta m_{t+1}\log m_{t+1}\bigr)  \middle|  x_0, c_0\right]
```

with $M_{t+1} = M_t m_{t+1}$, $E[m_{t+1} \mid x_t, c_t] = 1$, $M_0 = 1$.

The recursive formulation is:

$$
\hat{J}(x_t, c_t \mid \theta) = c_t + \min_{m_{t+1}} E \left[m_{t+1}\bigl[\beta \hat{J}(x_{t+1}, c_{t+1}) + \theta\log m_{t+1}\bigr]  \middle|  x_t, c_t\right]
$$

$$
= c_t - \theta\log E \left[\exp \left(-\frac{\beta \hat{J}(x_{t+1}, c_{t+1})}{\theta}\right)  \middle|  x_t, c_t\right]
$$

$$
=: c_t + T_t \left[\beta \hat{J}(x_{t+1}, c_{t+1})\right]
$$

where the right-hand side is attained by

$$
m_{t+1}^* \propto \exp \left(-\frac{\beta \hat{J}(x_{t+1}, c_{t+1})}{\theta}\right)
$$

By Lagrange multiplier theory, for the **corresponding dual pair** $(\tilde\theta, \eta)$,

$$
\hat{J}(x_t, c_t \mid \tilde\theta) = J(x_t, c_t \mid \eta) + \tilde\theta \eta
$$

Each choice of $\tilde\theta$ in the multiplier problem corresponds to a particular entropy bound $\eta(\tilde\theta)$ in the constraint problem, so the two formulations are equivalent.

The operator $T_t$ defined above is a **risk-sensitivity operator** that maps the continuation value through an exponential tilt, downweighting good outcomes and upweighting bad ones.

```{code-cell} ipython3
def T_operator(V, θ, probs=None):
    """Risk-sensitivity operator: T[V] = -θ log E[exp(-V/θ)]."""
    if probs is None:
        probs = np.ones(len(V)) / len(V)
    V_s = -V / θ
    max_v = np.max(V_s)
    return -θ * (max_v + np.log(np.sum(probs * np.exp(V_s - max_v))))

rng = np.random.default_rng(42)
V_samples = rng.normal(loc=5.0, scale=1.0, size=10_000)
E_V = np.mean(V_samples)

θ_grid = np.logspace(-1, 3, 50)
T_vals = [T_operator(V_samples, θ) for θ in θ_grid]

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(θ_grid, T_vals, 'firebrick', lw=2, label=r"$T_\theta[V]$")
ax.axhline(E_V, color='steelblue', lw=1.5,
           ls='--', label=r"$E[V]$ (risk neutral)")
ax.set_xlabel(r"Robustness parameter $\theta$")
ax.set_ylabel("Value")
ax.legend()
ax.annotate(r"$\theta \to \infty$: risk neutral",
            xy=(500, E_V), fontsize=11, color='steelblue',
            xytext=(50, E_V - 0.8),
            arrowprops=dict(arrowstyle='->', color='steelblue'))
plt.tight_layout()
plt.show()
```

As $\theta \to \infty$, the risk-sensitivity operator converges to the ordinary expectation $E[V]$, and the agent becomes risk neutral.

As $\theta$ shrinks, the operator places more weight on bad outcomes, reflecting greater concern about model misspecification.


## Who cares?

Joint probability distributions of interest rates and macroeconomic shocks are important throughout macroeconomics:

- **Costs of aggregate fluctuations.** Welfare assessments of business cycles
  depend sensitively on how risks are priced.
- **Consumption Euler equations.** The "New Keynesian IS curve" is a log-linearised
  consumption Euler equation whose risk adjustments are controlled by the stochastic
  discount factor.
- **Optimal taxation and government debt management.** Government bond prices embed
  risk prices whose state dependence matters for optimal fiscal policy.
- **Central bank expectations management.** Forward guidance works by shifting the
  term structure, an exercise whose effects depend on the same likelihood ratios
  studied here.
- **Long-run risk and secular stagnation.** The Bansal-Yaron long-run risk
  hypothesis is difficult to detect statistically, yet an agent who fears it in
  the sense formalised above may behave very differently than one who does not.

Understanding whether observed asset prices reflect risk aversion, mistaken beliefs, or fears of model misspecification, and quantifying each component, is interesting for both positive and normative macroeconomics.


## Related lectures

This lecture connects to several others in the series:

- {ref}`Doubts or Variability? <doubts_or_variability>` studies how a preference for robustness generates worst-case likelihood ratios that look like stochastic discount factor shocks, complementing the analysis here with Hansen-Jagannathan bounds and detection-error probabilities.
- {ref}`Asset Pricing: Finite State Models <mass>` introduces stochastic discount factors and risk-neutral pricing in a finite-state Markov setting, the discrete-state counterpart of the continuous Gaussian framework used here.
- {ref}`Heterogeneous Beliefs and Bubbles <harrison_kreps>` examines how heterogeneous and possibly mistaken beliefs generate speculative asset price bubbles, providing another perspective on how beliefs affect asset prices.
- {ref}`Likelihood Ratio Processes <likelihood_ratio_process>` develops the mathematical properties of likelihood ratios, the central device organising this lecture, including their martingale structure and statistical applications.
- {ref}`Divergence Measures <divergence_measures>` covers Kullback-Leibler divergence and relative entropy in detail, providing the information-theoretic foundations for the entropy constraints used in the robust control sections.
- {ref}`Affine Models of Asset Prices <affine_risk_prices>` extends the linear Gaussian state-space framework to affine and exponential-quadratic stochastic discount factors, developing risk-neutral pricing formulas closely related to those derived here.