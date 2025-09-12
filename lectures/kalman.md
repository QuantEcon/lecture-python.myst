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

(kalman)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# A First Look at the Kalman Filter

```{index} single: Kalman Filter
```

```{contents} Contents
:depth: 2
```
```{include} _admonition/gpu.md
```
In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

## Overview

This lecture provides a simple and intuitive introduction to the Kalman filter, for those who either

* have heard of the Kalman filter but don't know how it works, or
* know the Kalman filter equations, but don't know where they come from

For additional (more advanced) reading on the Kalman filter, see

* {cite}`Ljungqvist2012`, section 2.7
* {cite}`AndersonMoore2005`

The second reference presents a comprehensive treatment of the Kalman filter.

Required knowledge: Familiarity with matrix manipulations, multivariate normal distributions, covariance matrices, etc.

We'll need the following imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
from scipy import linalg
import jax
import jax.numpy as jnp
import matplotlib.cm as cm
from quantecon import Kalman, LinearStateSpace
from scipy.stats import norm
from scipy.integrate import quad
from scipy.linalg import eigvals
```

## The Basic Idea

The Kalman filter has many applications in economics, but for now
let's pretend that we are rocket scientists.

A missile has been launched from country Y and our mission is to track it.

Let $x  \in \mathbb{R}^2$ denote the current location of the missile---a
pair indicating latitude-longitude coordinates on a map.

At the present moment in time, the precise location $x$ is unknown, but
we do have some beliefs about $x$.

One way to summarize our knowledge is a point prediction $\hat{x}$

* But what if the President wants to know the probability that the missile is currently over the Sea of Japan?
* Then it is better to summarize our initial beliefs with a bivariate probability density $p$
* $\int_E p(x)dx$ indicates the probability that we attach to the missile being in region $E$.

The density $p$ is called our **prior** for the random variable $x$.

To keep things tractable in our example, we assume that our prior is Gaussian.

In particular, we take

```{math}
:label: prior

p = N(\hat{x}, \Sigma)
```

where $\hat{x}$ is the mean of the distribution and $\Sigma$ is a
$2 \times 2$ covariance matrix.  In our simulations, we will suppose that

```{math}
:label: kalman_dhxs

\hat{x}
=
\begin{bmatrix}
    0.2 \\
    -0.2
\end{bmatrix}
,
\qquad
\Sigma
=
\begin{bmatrix}
    0.4 & 0.3 \\
    0.3 & 0.45
\end{bmatrix}
```

This density $p(x)$ is shown below as a contour map, with the center of the red ellipse being equal to $\hat{x}$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
      Prior distribution
    name: fig_prior
---
# Set up the Gaussian prior density p
Σ = jnp.array([[0.4, 0.3], 
               [0.3, 0.45]])
x_hat = jnp.array([[0.2], 
                   [-0.2]])

# Set up grid for plotting
x_grid = jnp.linspace(-1.5, 2.9, 100)
y_grid = jnp.linspace(-3.1, 1.7, 100)
X, Y = jnp.meshgrid(x_grid, y_grid)

def gen_gaussian_plot_vals(X, Y, μ, Σ):
    """
    Compute and return the probability density function of bivariate normal
    distribution

    Parameters
    ----------
    X, Y : array_like(float)
        Meshgrid arrays defining the grid over which to evaluate the PDF.

    μ : array_like(float)
        Mean vector of the distribution. Shape (2, 1) or (2,).

    Σ : array_like(float)
        Covariance matrix of the distribution. Shape (2, 2).
    
    Returns
    -------
    Z : array_like(float)
        PDF values with same shape as X and Y meshgrids.
    """
    
    # Create coordinate arrays: stack X and Y to get shape (2, M * N)
    coords = jnp.stack([X.ravel(), Y.ravel()])
    
    # Define bivariate normal p.d.f
    def bivariate_normal(x):
        """Compute PDF for a single point x"""
        x_μ = x.reshape(-1, 1) - μ  # (2, 1)
        z = x_μ.T @ jnp.linalg.inv(Σ) @ x_μ  # scalar
        denom = 2 * jnp.pi * jnp.sqrt(jnp.linalg.det(Σ))
        return jnp.exp(-0.5 * z) / denom
    
    # Apply vmap over columns (each column is a point)
    vectorized_pdf = jax.vmap(bivariate_normal, in_axes=1)

    # Compute all PDF values
    pdf_values = vectorized_pdf(coords)
    
    # Reshape back to original meshgrid shape
    return pdf_values.reshape(X.shape)

# Plot the figure

fig, ax = plt.subplots()
ax.grid()

Z = gen_gaussian_plot_vals(X, Y, x_hat, Σ)
ax.contourf(X, Y, Z, levels=6, alpha=0.6, cmap=cm.jet)
cs = ax.contour(X, Y, Z, levels=6, colors="black")
ax.clabel(cs, inline=1, fontsize=10)

plt.show()
```

### The Filtering Step

We are now presented with some good news and some bad news.

The good news is that the missile has been located by our sensors, which report that the current location is $y = (2.3, -1.9)^top$.

The next figure shows the original prior $p(x)$ and the new reported
location $y$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
      Prior distribution and observation
    name: fig_obs
---
# The observed value of y
y = jnp.array([[2.3], 
               [-1.9]])

fig, ax = plt.subplots()
ax.grid()

Z = gen_gaussian_plot_vals(X, Y, x_hat, Σ)
ax.contourf(X, Y, Z, levels=6, alpha=0.6, cmap=cm.jet)
cs = ax.contour(X, Y, Z, levels=6, colors="black")
ax.clabel(cs, inline=1, fontsize=10)
ax.text(y[0, 0], y[1, 0], "$y$", fontsize=20, color="black")

plt.show()
```

The bad news is that our sensors are imprecise.

In particular, we should interpret the output of our sensor not as $y=x$, but rather as

```{math}
:label: kl_measurement_model

y = G x + v, \quad \text{where} \quad v \sim N(0, R)
```

Here $G$ and $R$ are $2 \times 2$ matrices with $R$ positive definite.

Both are assumed known, and the noise term $v$ is assumed
to be independent of $x$.

How then should we combine our prior $p(x) = N(\hat{x}, \Sigma)$ and this
new information $y$ to improve our understanding of the location of the
missile?

As you may have guessed, the answer is to use Bayes' theorem, which tells
us to update our prior $p(x)$ to $p(x \,|\, y)$ via

$$
p(x \,|\, y) = \frac{p(y \,|\, x) \, p(x)} {p(y)}
$$

where $p(y) = \int p(y \,|\, x) \, p(x) dx$.

In solving for $p(x \,|\, y)$, we observe that

* $p(x) = N(\hat{x}, \Sigma)$.
* In view of {eq}`kl_measurement_model`, the conditional density $p(y \,|\, x)$ is $N(Gx, R)$.
* $p(y)$ does not depend on $x$, and enters into the calculations only as a normalizing constant.

Because we are in a linear and Gaussian framework, the updated density can be computed by calculating population linear regressions.

In particular, the solution is known [^f1] to be

$$
p(x \,|\, y) = N(\hat x^F, \Sigma^F)
$$

where

```{math}
:label: kl_filter_exp

\hat{x}^F := \hat{x} + \Sigma G' (G \Sigma G' + R)^{-1}(y - G \hat{x})
\quad \text{and} \quad
\Sigma^F := \Sigma - \Sigma G' (G \Sigma G' + R)^{-1} G \Sigma
```

Here $\Sigma G' (G \Sigma G' + R)^{-1}$ is the matrix of population regression coefficients of the hidden object $x - \hat{x}$ on the surprise $y - G \hat{x}$.

We can verify it by computing
$$
\begin{aligned}
\mathrm{Cov}(x - \hat{x}, y - G \hat{x})\mathrm{Var}(y - G \hat{x})^{-1}
&= \mathrm{Cov}(x - \hat{x}, G x + v - G \hat{x})\mathrm{Var}(G x + v  - G \hat{x})^{-1}\\
&= \Sigma G'(G \Sigma G' + R)^{-1}
\end{aligned}
$$

This new density $p(x \,|\, y) = N(\hat{x}^F, \Sigma^F)$ is shown in the next figure via contour lines and the color map.

The original density is left in as contour lines for comparison

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
      Updated distribution from observation
    name: fig_update_obs
---
# Define the matrices G and R from the equation y = G x + N(0, R)
G = jnp.array([[1, 0], 
               [0, 1]])
R = 0.5 * Σ

fig, ax = plt.subplots()
ax.grid()

# Density 1
Z = gen_gaussian_plot_vals(X, Y, x_hat, Σ)
cs1 = ax.contour(X, Y, Z, levels=6, colors="black")
ax.clabel(cs1, inline=1, fontsize=10)

# Density 2
M = Σ @ G.T @ linalg.inv(G @ Σ @ G.T + R)
x_hat_F = x_hat + M @ (y - G @ x_hat)
Σ_F = Σ - M @ G @ Σ
Z_F = gen_gaussian_plot_vals(X, Y, x_hat_F, Σ_F)
Z_F = Z_F.at[jnp.where(Z_F < 1e-10)].set(0.1)  # Avoid very small values

ax.contourf(X, Y, Z_F, levels=6, alpha=0.6, cmap=cm.jet)
cs2 = ax.contour(X, Y, Z_F, levels=6, colors="black")
ax.clabel(cs2, inline=1, fontsize=10)

# Observed value
ax.text(y[0, 0], y[1, 0], "$y$", fontsize=20, color="black")

plt.show()
```

Our new density twists the prior $p(x)$ in a direction determined by  the new
information $y - G \hat x$.

In generating the figure, we set $G$ to the identity matrix and $R = 0.5 \Sigma$ for $\Sigma$ defined in {eq}`kalman_dhxs`.

(kl_forecase_step)=
### The Forecast Step

What have we achieved so far?

We have obtained probabilities for the current location of the state (missile) given prior and current information.

This is called "filtering" rather than forecasting because we are filtering
out noise rather than looking into the future.

* $p(x \,|\, y) = N(\hat x^F, \Sigma^F)$ is called the **filtering distribution**

But now let's suppose that we are given another task: to predict the location of the missile after one unit of time (whatever that may be) has elapsed.

To do this we need a model of how the state evolves.

Let's suppose that we have one, and that it's linear and Gaussian. In particular,

```{math}
:label: kl_xdynam

x_{t+1} = A x_t + w_{t+1}, \quad \text{where} \quad w_t \sim N(0, Q)
```

Our aim is to combine this law of motion and our current distribution $p(x \,|\, y) = N(\hat x^F, \Sigma^F)$ to come up with a new *predictive* distribution for the location in one unit of time.

In view of {eq}`kl_xdynam`, all we have to do is introduce a random vector $x^F \sim N(\hat x^F, \Sigma^F)$ and work out the distribution of $A x^F + w$ where $w$ is independent of $x^F$ and has distribution $N(0, Q)$.

Since linear combinations of Gaussians are Gaussian, $A x^F + w$ is Gaussian.

Elementary calculations and the expressions in {eq}`kl_filter_exp` tell us that

$$
\mathbb{E} [A x^F + w]
= A \mathbb{E} x^F + \mathbb{E} w
= A \hat x^F
= A \hat x + A \Sigma G' (G \Sigma G' + R)^{-1}(y - G \hat x)
$$

and

$$
\operatorname{Var} [A x^F + w]
= A \operatorname{Var}[x^F] A' + Q
= A \Sigma^F A' + Q
= A \Sigma A' - A \Sigma G' (G \Sigma G' + R)^{-1} G \Sigma A' + Q
$$

The matrix $A \Sigma G' (G \Sigma G' + R)^{-1}$ is often written as
$K_{\Sigma}$ and called the **Kalman gain**.

* The subscript $\Sigma$ has been added to remind us that  $K_{\Sigma}$ depends on $\Sigma$, but not $y$ or $\hat x$.

Using this notation, we can summarize our results as follows.

Our updated prediction is the density $N(\hat x_{new}, \Sigma_{new})$ where

```{math}
:label: kl_mlom0

\begin{aligned}
    \hat x_{new} &:= A \hat x + K_{\Sigma} (y - G \hat x) \\
    \Sigma_{new} &:= A \Sigma A' - K_{\Sigma} G \Sigma A' + Q \nonumber
\end{aligned}
```

* The density $p_{new}(x) = N(\hat x_{new}, \Sigma_{new})$ is called the **predictive distribution**

The predictive distribution is the new density shown in the following figure, where
the update has used parameters.

$$
A
=
\begin{bmatrix}
    1.2 & 0.0 \\
    0.0 & -0.2
\end{bmatrix},
  \qquad
Q = 0.3 * \Sigma
$$

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
      Updated distribution from transition
    name: fig_update_trans
---
# The matrices A and Q
A = jnp.array([[1.2, 0], 
               [0, -0.2]])
Q = 0.3 * Σ

fig, ax = plt.subplots()
ax.grid()

# Density 1
Z = gen_gaussian_plot_vals(X, Y, x_hat, Σ)
cs1 = ax.contour(X, Y, Z, 6, colors="black")
ax.clabel(cs1, inline=1, fontsize=10)

# Density 2
M = Σ @ G.T @ linalg.inv(G @ Σ @ G.T + R)
x_hat_F = x_hat + M @ (y - G @ x_hat)
Σ_F = Σ - M @ G @ Σ
Z_F = gen_gaussian_plot_vals(X, Y, x_hat_F, Σ_F)
Z_F = Z_F.at[jnp.where(Z_F < 1e-10)].set(0.1)  # Avoid very small values
cs2 = ax.contour(X, Y, Z_F, 6, colors="black")
ax.clabel(cs2, inline=1, fontsize=10)

# Observed value
ax.text(y[0, 0], y[1, 0], "$y$", fontsize=20, color="black")

# Density 3
new_x_hat = A @ x_hat_F
new_Σ = A @ Σ_F @ A.T + Q
new_Z = gen_gaussian_plot_vals(X, Y, new_x_hat, new_Σ)
cs3 = ax.contour(X, Y, new_Z, 6, colors="black")
ax.clabel(cs3, inline=1, fontsize=10)
ax.contourf(X, Y, new_Z, 6, alpha=0.6, cmap=cm.jet)


plt.show()
```

### The Recursive Procedure

```{index} single: Kalman Filter; Recursive Procedure
```

Let's look back at what we've done.

We started the current period with a prior $p(x)$ for the location $x$ of the missile.

We then used the current measurement $y$ to update to $p(x \,|\, y)$.

Finally, we used the law of motion {eq}`kl_xdynam` for $\{x_t\}$ to update to $p_{new}(x)$.

If we now step into the next period, we are ready to go round again, taking $p_{new}(x)$
as the current prior.

Swapping notation $p_t(x)$ for $p(x)$ and $p_{t+1}(x)$ for $p_{new}(x)$, the full recursive procedure is:

1. Start the current period with prior $p_t(x) = N(\hat x_t, \Sigma_t)$.
1. Observe current measurement $y_t$.
1. Compute the filtering distribution $p_t(x \,|\, y) = N(\hat x_t^F, \Sigma_t^F)$ from $p_t(x)$ and $y_t$, applying Bayes rule and the conditional distribution {eq}`kl_measurement_model`.
1. Compute the predictive distribution $p_{t+1}(x) = N(\hat x_{t+1}, \Sigma_{t+1})$ from the filtering distribution and {eq}`kl_xdynam`.
1. Increment $t$ by one and go to step 1.

Repeating {eq}`kl_mlom0`, the dynamics for $\hat x_t$ and $\Sigma_t$ are as follows

```{math}
:label: kalman_lom

\begin{aligned}
    \hat x_{t+1} &= A \hat x_t + K_{\Sigma_t} (y_t - G \hat x_t) \\
    \Sigma_{t+1} &= A \Sigma_t A' - K_{\Sigma_t} G \Sigma_t A' + Q \nonumber
\end{aligned}
```

These are the standard dynamic equations for the Kalman filter (see, for example, {cite}`Ljungqvist2012`, page 58).

(kalman_convergence)=
## Convergence

The matrix $\Sigma_t$ is a measure of the uncertainty of our prediction $\hat x_t$ of $x_t$.

Apart from special cases, this uncertainty will never be fully resolved, regardless of how much time elapses.

One reason is that our prediction $\hat x_t$ is made based on information available at $t-1$, not $t$.

Even if we know the precise value of $x_{t-1}$ (which we don't), the transition equation {eq}`kl_xdynam` implies that $x_t = A x_{t-1} + w_t$.

Since the shock $w_t$ is not observable at $t-1$, any time $t-1$ prediction of $x_t$ will incur some error (unless $w_t$ is degenerate).

However, it is certainly possible that $\Sigma_t$ converges to a constant matrix as $t \to \infty$.

To study this topic, let's expand the second equation in {eq}`kalman_lom`:

```{math}
:label: kalman_sdy

\Sigma_{t+1} = A \Sigma_t A' -  A \Sigma_t G' (G \Sigma_t G' + R)^{-1} G \Sigma_t A' + Q
```

This is a nonlinear difference equation in $\Sigma_t$.

A fixed point of {eq}`kalman_sdy` is a constant matrix $\Sigma$ such that

```{math}
:label: kalman_dare

\Sigma = A \Sigma A' -  A \Sigma G' (G \Sigma G' + R)^{-1} G \Sigma A' + Q
```

Equation {eq}`kalman_sdy` is known as a discrete-time Riccati difference equation.

Equation {eq}`kalman_dare` is known as a [discrete-time algebraic Riccati equation](https://en.wikipedia.org/wiki/Algebraic_Riccati_equation).

Conditions under which a fixed point exists and the sequence $\{\Sigma_t\}$ converges to it are discussed in {cite}`AHMS1996` and {cite}`AndersonMoore2005`, chapter 4.

A sufficient (but not necessary) condition is that all the eigenvalues $\lambda_i$ of $A$ satisfy $|\lambda_i| < 1$ (cf. e.g., {cite}`AndersonMoore2005`, p. 77).

(This strong condition assures that the unconditional  distribution of $x_t$  converges as $t \rightarrow + \infty$.)

In this case, for any initial choice of $\Sigma_0$ that is both non-negative and symmetric, the sequence $\{\Sigma_t\}$ in {eq}`kalman_sdy` converges to a non-negative symmetric matrix $\Sigma$ that solves {eq}`kalman_dare`.

## Implementation

```{index} single: Kalman Filter; Programming Implementation
```

The class `Kalman` from the [QuantEcon.py](https://quantecon.org/quantecon-py) package implements the Kalman filter

* Instance data consists of:
    * the moments $(\hat x_t, \Sigma_t)$ of the current prior.
    * An instance of the [LinearStateSpace](https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/lss.py) class from [QuantEcon.py](https://quantecon.org/quantecon-py).

The latter represents a linear state space model of the form

$$
\begin{aligned}
    x_{t+1} & = A x_t + C w_{t+1}
    \\
    y_t & = G x_t + H v_t
\end{aligned}
$$

where the shocks $w_t$ and $v_t$ are IID standard normals.

To connect this with the notation of this lecture we set

$$
Q := CC' \quad \text{and} \quad R := HH'
$$

* The class `Kalman` from the [QuantEcon.py](https://quantecon.org/quantecon-py) package has a number of methods, some that we will wait to use until we study more advanced applications in subsequent lectures.
* Methods pertinent for this lecture are:
    * `prior_to_filtered`, which updates $(\hat x_t, \Sigma_t)$ to $(\hat x_t^F, \Sigma_t^F)$
    * `filtered_to_forecast`, which updates the filtering distribution to the predictive distribution -- which becomes the new prior $(\hat x_{t+1}, \Sigma_{t+1})$
    * `update`, which combines the last two methods
    * `stationary_values`, which computes the solution to {eq}`kalman_dare` and the corresponding (stationary) Kalman gain

You can view the program [on GitHub](https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/kalman.py).

## Exercises

```{exercise-start}
:label: kalman_ex1
```

Consider the following simple application of the Kalman filter, loosely based
on {cite}`Ljungqvist2012`, section 2.9.2.

Suppose that

* all variables are scalars
* the hidden state $\{x_t\}$ is in fact constant, equal to some $\theta \in \mathbb{R}$ unknown to the modeler

State dynamics are therefore given by {eq}`kl_xdynam` with $A=1$, $Q=0$ and $x_0 = \theta$.

The measurement equation is $y_t = \theta + v_t$ where $v_t$ is $N(0,1)$ and IID.

The task of this exercise is to simulate the model and, using the code from `kalman.py`, plot the first five predictive densities $p_t(x) = N(\hat x_t, \Sigma_t)$.

As shown in {cite}`Ljungqvist2012`, sections 2.9.1--2.9.2, these distributions asymptotically put all mass on the unknown value $\theta$.

In the simulation, take $\theta = 10$, $\hat x_0 = 8$ and $\Sigma_0 = 1$.

Your figure should -- modulo randomness -- look something like this

```{image} /_static/lecture_specific/kalman/kl_ex1_fig.png
:align: center
```

```{exercise-end}
```


```{solution-start} kalman_ex1
:class: dropdown
```

```{code-cell} ipython3
# Parameters
θ = 10  # Constant value of state x_t
A, C, G, H = 1, 0, 1, 1
ss = LinearStateSpace(A, C, G, H, mu_0=θ)

# Set prior, initialize kalman filter
x_hat_0, Σ_0 = 8, 1
kalman = Kalman(ss, x_hat_0, Σ_0)

# Draw observations of y from state space model
N = 5
seed = 1234 # Set random seed
x, y = ss.simulate(N, seed)
y = y.flatten()

# Set up plot
fig, ax = plt.subplots()
xgrid = jnp.linspace(θ - 5, θ + 2, 200)

for i in range(N):
    # Record the current predicted mean and variance
    m, v = kalman.x_hat.item(), kalman.Sigma.item()
    # Plot, update filter
    ax.plot(xgrid, norm.pdf(xgrid, loc=m, scale=jnp.sqrt(v)), label=f'$t={i}$')
    kalman.update(y[i])

ax.set_title("First 5 densities when θ=10.0")
ax.legend(loc='upper left')
plt.show()
```

```{solution-end}
```

```{exercise-start}
:label: kalman_ex2
```

The preceding figure gives some support to the idea that probability mass
converges to $\theta$.

To get a better idea, choose a small $\epsilon > 0$ and calculate

$$
z_t := 1 - \int_{\theta - \epsilon}^{\theta + \epsilon} p_t(x) dx
$$

for $t = 0, 1, 2, \ldots, T$.

Plot $z_t$ against $T$, setting $\epsilon = 0.1$ and $T = 600$.

Your figure should show error erratically declining something like this

```{image} /_static/lecture_specific/kalman/kl_ex2_fig.png
:align: center
```

```{exercise-end}
```


```{solution-start} kalman_ex2
:class: dropdown
```

```{code-cell} ipython3
ϵ = 0.1
θ = 10  # Constant value of state x_t
A, C, G, H = 1, 0, 1, 1
ss = LinearStateSpace(A, C, G, H, mu_0=θ)

x_hat_0, Σ_0 = 8, 1
kalman = Kalman(ss, x_hat_0, Σ_0)

T = 600
seed = 1234
z = jnp.empty(T)
x, y = ss.simulate(T, seed)
y = y.flatten()

for t in range(T):
    # Record the current predicted mean and variance and plot their densities
    m, v = kalman.x_hat.item(), kalman.Sigma.item()

    # Wrap parameters
    def f(x):
        return norm.pdf(x, loc=m, scale=jnp.sqrt(v))
    
    integral, error = quad(f, θ - ϵ, θ + ϵ)
    z = z.at[t].set(1 - integral)

    kalman.update(y[t])

fig, ax = plt.subplots()
ax.set_ylim(0, 1)
ax.set_xlim(0, T)
ax.plot(range(T), z)
ax.fill_between(range(T), jnp.zeros(T), z, color="blue", alpha=0.2)

ax.set_title("Probability differences")
plt.show()
```

```{solution-end}
```

```{exercise-start}
:label: kalman_ex3
```

As discussed {ref}`above <kalman_convergence>`, if the shock sequence $\{w_t\}$ is not degenerate, then it is not in general possible to predict $x_t$ without error at time $t-1$ (and this would be the case even if we could observe $x_{t-1}$).

Let's now compare the prediction $\hat x_t$ made by the Kalman filter
against a competitor who *is* allowed to observe $x_{t-1}$.

This competitor will use the conditional expectation $\mathbb E[ x_t
\,|\, x_{t-1}]$, which in this case is $A x_{t-1}$.

The conditional expectation is known to be the optimal prediction method in terms of minimizing mean squared error.

(More precisely, the minimizer of $\mathbb E \, \| x_t - g(x_{t-1}) \|^2$ with respect to $g$ is $g^*(x_{t-1}) := \mathbb E[ x_t \,|\, x_{t-1}]$)

Thus we are comparing the Kalman filter against a competitor who has more
information (in the sense of being able to observe the latent state) and
behaves optimally in terms of minimizing squared error.

Our horse race will be assessed in terms of squared error.

In particular, your task is to generate a graph plotting observations of both $\| x_t - A x_{t-1} \|^2$ and $\| x_t - \hat x_t \|^2$ against $t$ for $t = 1, \ldots, 50$.

For the parameters, set $G = I, R = 0.5 I$ and $Q = 0.3 I$, where $I$ is
the $2 \times 2$ identity.

Set

$$
A =
\begin{bmatrix}
    0.5 & 0.4 \\
    0.6 & 0.3
\end{bmatrix}
$$

To initialize the prior density, set

$$
\Sigma_0 =
\begin{bmatrix}
    0.9 & 0.3 \\
    0.3 & 0.9
\end{bmatrix}
$$

and $\hat x_0 = (8, 8)$.

Finally, set $x_0 = (0, 0)$.

You should end up with a figure similar to the following (modulo randomness)

```{image} /_static/lecture_specific/kalman/kalman_ex3.png
:align: center
```

Observe how, after an initial learning period, the Kalman filter performs quite well, even relative to the competitor who predicts optimally with knowledge of the latent state.

```{exercise-end}
```

```{solution-start} kalman_ex3
:class: dropdown
```

```{code-cell} ipython3
# Define A, C, G, H
G = jnp.eye(2)
H = jnp.sqrt(0.5) * G

A = jnp.array([[0.5, 0.4],
               [0.6, 0.3]])
C = jnp.sqrt(0.3) * G

# Set up state space model, initial value x_0 set to zero
ss = LinearStateSpace(A, C, G, H, mu_0=jnp.zeros(2))

# Define the prior density
Σ = jnp.array([[0.9, 0.3],
               [0.3, 0.9]])
x_hat = jnp.array([8, 8])

# Initialize the Kalman filter
kn = Kalman(ss, x_hat, Σ)

# Print eigenvalues of A
print("Eigenvalues of A:")
print(eigvals(A))

# Print stationary Σ
S, K = kn.stationary_values()
print("Stationary prediction error variance:")
print(S)

# Generate the plot
T = 50
seed = 1234
x, y = ss.simulate(T, seed)

e1 = jnp.empty(T-1)
e2 = jnp.empty(T-1)

for t in range(1, T):
    kn.update(y[:,t])
    diff1 = x[:, t] - kn.x_hat.flatten()
    diff2 = x[:, t] - A @ x[:, t-1]
    e1 = e1.at[t-1].set(diff1 @ diff1)
    e2 = e2.at[t-1].set(diff2 @ diff2)

fig, ax = plt.subplots()
ax.plot(range(1, T), e1, 'k-', lw=2, alpha=0.6,
        label='Kalman filter error')
ax.plot(range(1, T), e2, 'g-', lw=2, alpha=0.6,
        label='Conditional expectation error')
ax.legend()
ax.set_title("Kalman filter vs conditional expectation")
plt.show()
```

```{solution-end}
```

```{exercise}
:label: kalman_ex4

Try varying the coefficient $0.3$ in $Q = 0.3 I$ up and down.

Observe how the diagonal values in the stationary solution $\Sigma$ (see {eq}`kalman_dare`) increase and decrease in line with this coefficient.

The interpretation is that more randomness in the law of motion for $x_t$ causes more (permanent) uncertainty in prediction.
```

[^f1]: See, for example, page 93 of {cite}`Bishop2006`. To get from his expressions to the ones used above, you will also need to apply the [Woodbury matrix identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity).
