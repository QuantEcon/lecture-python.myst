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

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

## Overview

This lecture provides a simple and intuitive introduction to the Kalman filter

It is aimed at readers who either

* have heard of the Kalman filter but don't know how it works, or
* know the Kalman filter equations, but don't know where they come from

For additional (more advanced) reading on the Kalman filter, see

* {cite}`Ljungqvist2012`, section 2.7
* {cite}`AndersonMoore2005`

The second reference presents a  comprehensive treatment of the Kalman filter.

Required knowledge: Familiarity with matrix manipulations, multivariate normal distributions, covariance matrices, etc.

We'll need the following imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np
from quantecon import Kalman, LinearStateSpace
from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad
from scipy.linalg import eigvals
```

## The basic idea

The Kalman filter has many applications in economics, but for now
let's pretend that we are rocket scientists.

A missile has been launched from a hostile country and our mission is to track it.

Let $X_t  \in \mathbb{R}^2$ denote the current location of the missile---a
pair indicating latitude-longitude coordinates on a map.

At the present moment in time, the location $X_t$ is unknown, but we do have some beliefs about it.

We could certainly produce a point prediction.

For example, it could mark a point on the globe somewhere in northern Mongolia.

But the fact is that we are uncertain.

And the President wants to know: what is the probability that the missile is within 500km of Manhattan?

A point prediction doesn't address that question.

Hence it's best if we can express our current understanding via a bivariate probability density $p$.

* Now $\int_E p(x)dx$ indicates the probability that the missile is in region $E$.

We will call $p$ our **prior** for the random variable $X$.

To keep things tractable, we assume for now that our prior is Gaussian.

In particular, we take

```{math}
:label: prior

    p = N(\mu, \Sigma)
```

where $\mu$ is the (vector) mean of the distribution---a natural point prediction---and $\Sigma$ is a $2 \times 2$ covariance matrix.

In our simulations, we will suppose that

```{math}
:label: kalman_dhxs

\mu
= \left(
\begin{array}{c}
    0.2 \\
    -0.2
\end{array}
  \right),
\qquad
\Sigma
= \left(
\begin{array}{cc}
    0.4 & 0.3 \\
    0.3 & 0.45
\end{array}
  \right)
```

This density $p$ is shown below as a contour map, with the center of the red ellipse being equal to $\mu$.

```{code-cell} ipython3
:tags: [output_scroll]

# Set up the Gaussian prior density p
Σ = np.array([[0.4, 0.3],
              [0.3, 0.45]])
μ = np.array([[0.2],
              [-0.2]])
# Define the matrices G and R from the measurement equation Y = G X + v
G = np.array([[1, 0],
              [0, 1]])
R = 0.5 * Σ
# The matrices A and Q
A = np.array([[1.2, 0],
              [0, -0.2]])
Q = 0.3 * Σ
# The observed value of y
y = np.array([[2.3],
              [-1.9]])

# Set up grid for plotting
x_grid = np.linspace(-1.5, 2.9, 100)
y_grid = np.linspace(-3.1, 1.7, 100)
X, Y = np.meshgrid(x_grid, y_grid)

def gen_gaussian_plot_vals(μ, C):
    "Z values for plotting the bivariate Gaussian N(μ, C)"
    pos = np.dstack((X, Y))
    return multivariate_normal(μ.ravel(), C).pdf(pos)

# Plot the figure

fig, ax = plt.subplots(figsize=(10, 8))
ax.grid()

Z = gen_gaussian_plot_vals(μ, Σ)
ax.contourf(X, Y, Z, 6, alpha=0.6, cmap="viridis")
cs = ax.contour(X, Y, Z, 6, colors="black")
ax.clabel(cs, inline=1, fontsize=10)

plt.show()
```

### The filtering step

We are now presented with some good news and some bad news.

The good news is that the missile has been located by our sensors, which report that the current location is $Y_t = (2.3, -1.9)$.

The next figure shows the original prior $p$ and the new reported signal $Y_t$

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 8))
ax.grid()

Z = gen_gaussian_plot_vals(μ, Σ)
ax.contourf(X, Y, Z, 6, alpha=0.6, cmap="viridis")
cs = ax.contour(X, Y, Z, 6, colors="black")
ax.clabel(cs, inline=1, fontsize=10)
y_1, y_2 = y[0].item(), y[1].item()
ax.scatter(y_1, y_2, marker="o", s=50, color="black", zorder=3)
ax.text(y_1 + 0.1, y_2 + 0.1, "$Y_t$", fontsize=20, color="black")

plt.show()
```

The bad news is that our sensors are imprecise.

The sensor report is a noisy signal distorted by measurement error.

In particular, we should interpret the output of our sensor not as $Y_t=X_t$, but rather as

```{math}
:label: kl_measurement_model

Y_t = G X_t + v_t, \quad \text{where} \quad v_t \sim N(0, R)
```

Here $G$ and $R$ are $2 \times 2$ matrices, with $R$ being symmetric and positive definite.

We assume that

* $G$ and $R$ are known
* the noise term $v_t$ is unobservable and independent of $X_t$

How then should we combine our prior $X_t \sim N(\mu, \Sigma)$ and this
new information $Y_t$ to improve our understanding of the location of the
missile?

As you may have guessed, the answer is to use Bayes' theorem.

It tells us how to update the prior density $p(x)$ for $X_t$ to the
posterior density $p(x \,|\, y)$ after observing $Y_t$:

$$
p(x \,|\, Y_t) = \frac{p(Y_t \,|\, x) \, p(x)} {p(Y_t)}
$$

where $p(Y_t) = \int p(Y_t \,|\, x) \, p(x) dx$.

In solving for $p(x \,|\, Y_t)$, we observe that

* $p(x)$ is the prior density $N(\mu, \Sigma)$.
* $p(Y_t \,|\, x)$ is the conditional density of $Y_t$ given $X_t=x$.
* In view of {eq}`kl_measurement_model`, this conditional density is $N(Gx, R)$.

Due to our linear Gaussian framework, the updated density turns out to be Gaussian as well.

In particular, the solution is known to be

$$
    p(x \,|\, Y_t) = N(\mu^F, \Sigma^F)
$$

where

```{math}
:label: kl_filter_exp

\mu^F := \mu + \Sigma G^\top (G \Sigma G^\top + R)^{-1}(y - G \mu)
```

and 

```{math}
:label: kl_filter_exp2

\Sigma^F := \Sigma - \Sigma G^\top (G \Sigma G^\top + R)^{-1} G \Sigma
```

```{note}
A proof can be found in {cite}`Bishop2006`. To get from his expressions to the ones used above, you will also need to apply the [Woodbury matrix identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity).
```

Here $\Sigma G^\top (G \Sigma G^\top + R)^{-1}$ is the matrix of population
regression coefficients of the hidden state deviation $X_t - \mu$ on the
*signal surprise* $Y_t - G \mu$.

This new density $p(x \,|\, Y_t) = N(\mu^F, \Sigma^F)$ is shown in the next figure via contour lines and the color map.

The original density is left in as contour lines for comparison

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 8))
ax.grid()

Z = gen_gaussian_plot_vals(μ, Σ)
cs1 = ax.contour(X, Y, Z, 6, colors="black")
ax.clabel(cs1, inline=1, fontsize=10)
M = Σ @ G.T @ linalg.inv(G @ Σ @ G.T + R)
μ_F = μ + M @ (y - G @ μ)
Σ_F = Σ - M @ G @ Σ
new_Z = gen_gaussian_plot_vals(μ_F, Σ_F)
cs2 = ax.contour(X, Y, new_Z, 6, colors="black")
ax.clabel(cs2, inline=1, fontsize=10)
ax.contourf(X, Y, new_Z, 6, alpha=0.6, cmap="viridis")
y_1, y_2 = y[0].item(), y[1].item()
ax.scatter(y_1, y_2, marker="o", s=50, color="black", zorder=3)
ax.text(y_1 + 0.1, y_2 + 0.1, "$Y_t$", fontsize=20, color="black")

plt.show()
```

Our new density twists the prior $p(x)$ in a direction determined by the new information $Y_t - G \mu$.

In generating the figure, we set $G$ to the identity matrix and $R = 0.5 \Sigma$ for $\Sigma$ defined in {eq}`kalman_dhxs`.

(kl_forecase_step)=
### The forecast step

What have we achieved so far?

We have obtained probabilities for the current location of the state (missile) given prior and current information.

This is called "filtering" rather than forecasting because we are filtering
out noise rather than looking into the future.

The posterior $p(x \,|\, Y_t) = N(\mu^F, \Sigma^F)$ is called the **filtering distribution** for $X_t$ after observing $Y_t$

But now let's suppose that we are given another task: to predict the location of
the missile after one unit of time (whatever that may be) has elapsed.

To do this we need a model of how the state evolves.

Let's suppose that we have one, and that it's linear and Gaussian.

In particular,

```{math}
:label: kl_xdynam

X_{t+1} = A X_t + W_{t+1}, \quad \text{where} \quad W_t \sim N(0, Q)
```

Our aim is to combine this law of motion and our current filtering distribution
$N(\mu^F, \Sigma^F)$ to come up with a new **predictive** distribution for
the location in one unit of time.

In view of {eq}`kl_xdynam`, all we have to do is introduce a random vector $X^F \sim N(\mu^F, \Sigma^F)$ and work out the distribution of $A X^F + W$ where $W$ is independent of $X^F$ and has distribution $N(0, Q)$.

Since linear combinations of Gaussians are Gaussian, $A X^F + W$ is Gaussian.

Standard calculations and the expressions in {eq}`kl_filter_exp`--{eq}`kl_filter_exp2` tell us that

$$
\begin{aligned}
\mathbb{E} [A X^F + W]
&= A \mathbb{E}[X^F] + \mathbb{E}[W] \\
&= A \mu^F \\
&= A \mu + A \Sigma G^\top (G \Sigma G^\top + R)^{-1}(Y_t - G \mu)
\end{aligned}
$$

and

$$
\begin{aligned}
\operatorname{Var} [A X^F + W]
&= A \operatorname{Var}[X^F] A^\top + Q \\
&= A \Sigma^F A^\top + Q \\
&= A \Sigma A^\top + Q - A \Sigma G^\top (G \Sigma G^\top + R)^{-1} G \Sigma A^\top
\end{aligned}
$$

The matrix $A \Sigma G^\top (G \Sigma G^\top + R)^{-1}$ is often written as
$K_{\Sigma}$ and called the **Kalman gain**.

* The subscript $\Sigma$ has been added to remind us that  $K_{\Sigma}$ depends on $\Sigma$, but not $Y_t$ or $\mu$.

Using this notation, we can summarize our results as follows.

Our updated prediction is the density $N(\mu_{\mathrm{new}}, \Sigma_{\mathrm{new}})$ where

```{math}
:label: kl_mlom0

\begin{aligned}
    \mu_{\mathrm{new}} &:= A \mu + K_{\Sigma} (y - G \mu) \\
    \Sigma_{\mathrm{new}} &:= A \Sigma A^\top - K_{\Sigma} G \Sigma A^\top + Q \nonumber
\end{aligned}
```

* The density $p_{\mathrm{new}}(x) = N(\mu_{\mathrm{new}}, \Sigma_{\mathrm{new}})$ is called the **predictive distribution**

The predictive distribution is the new density shown in the following figure, where
the update has used parameters.

$$
A
= \left(
\begin{array}{cc}
    1.2 & 0.0 \\
    0.0 & -0.2
\end{array}
  \right),
  \qquad
  Q = 0.3 \Sigma
$$

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 8))
ax.grid()

# Density 1
Z = gen_gaussian_plot_vals(μ, Σ)
cs1 = ax.contour(X, Y, Z, 6, colors="black")
ax.clabel(cs1, inline=1, fontsize=10)

# Density 2
M = Σ @ G.T @ linalg.inv(G @ Σ @ G.T + R)
μ_F = μ + M @ (y - G @ μ)
Σ_F = Σ - M @ G @ Σ
Z_F = gen_gaussian_plot_vals(μ_F, Σ_F)
cs2 = ax.contour(X, Y, Z_F, 6, colors="black")
ax.clabel(cs2, inline=1, fontsize=10)

# Density 3
new_μ = A @ μ_F
new_Σ = A @ Σ_F @ A.T + Q
new_Z = gen_gaussian_plot_vals(new_μ, new_Σ)
cs3 = ax.contour(X, Y, new_Z, 6, colors="black")
ax.clabel(cs3, inline=1, fontsize=10)
ax.contourf(X, Y, new_Z, 6, alpha=0.6, cmap="viridis")
y_1, y_2 = y[0].item(), y[1].item()
ax.scatter(y_1, y_2, marker="o", s=50, color="black", zorder=3)
ax.text(y_1 + 0.1, y_2 + 0.1, "$Y_t$", fontsize=20, color="black")

plt.show()
```

### The recursive procedure

```{index} single: Kalman Filter; Recursive Procedure
```

Let's look back at what we've done.

We started the current period with a prior density $p_t(x)$ for the hidden state $X_t$.

We then observed the signal $Y_t$ and updated the prior density to the
filtering density $p_t(x \,|\, Y_t)$.

Finally, we used the law of motion {eq}`kl_xdynam` for $\{X_t\}$ to update
to the predictive density $p_{t+1}(x)$ for $X_{t+1}$.

If we now step into the next period, we are ready to go round again, taking
$p_{t+1}(x)$ as the current prior density and reading in the new observation
$Y_{t+1}$.

Using this time-indexed notation, the full recursive procedure is:

1. Start the current period with prior density $p_t(x) = N(\mu_t, \Sigma_t)$ for $X_t$.
1. Observe current signal $Y_t = y_t$.
1. Compute the filtering density $p_t(x \,|\, y_t) = N(\mu_t^F, \Sigma_t^F)$ from $p_t(x)$ and $y_t$, applying Bayes rule and the conditional distribution {eq}`kl_measurement_model`.
1. Compute the predictive density $p_{t+1}(x) = N(\mu_{t+1}, \Sigma_{t+1})$ for $X_{t+1}$ from the filtering density and {eq}`kl_xdynam`.
1. Increment $t$ by one and go to step 1.

Repeating {eq}`kl_mlom0`, the dynamics for $\mu_t$ and $\Sigma_t$ are as follows

```{math}
:label: kalman_lom

\begin{aligned}
    \mu_{t+1} &= A \mu_t + K_{\Sigma_t} (y_t - G \mu_t) \\
    \Sigma_{t+1} &= A \Sigma_t A^\top - K_{\Sigma_t} G \Sigma_t A^\top + Q \nonumber
\end{aligned}
```

These are the standard dynamic equations for the Kalman filter (see, for example, {cite}`Ljungqvist2012`, page 58).

```{note}
Here $\mu_t$ is the filter's prediction of the hidden state $X_t$. In much of the Kalman filter literature it is written $\hat x_t$, emphasizing that it is an estimate of $X_t$.
```

(kalman_convergence)=
## Convergence

The matrix $\Sigma_t$ is a measure of the uncertainty of our prediction $\mu_t$ of $X_t$.

Apart from special cases, this uncertainty will never be fully resolved, regardless of how much time elapses.

One reason is that our prediction $\mu_t$ is made based on information available at $t-1$, not $t$.

Even if we knew the precise realized value $X_{t-1}=x_{t-1}$ (which we
don't), the transition equation {eq}`kl_xdynam` implies that
$X_t = A x_{t-1} + W_t$.

Since the shock $W_t$ is not observable at $t-1$, any time $t-1$ prediction of $X_t$ will incur some error (unless $W_t$ is degenerate).

However, it is certainly possible that $\Sigma_t$ converges to a constant matrix as $t \to \infty$.

To study this topic, let's expand the second equation in {eq}`kalman_lom`:

```{math}
:label: kalman_sdy

\Sigma_{t+1} = A \Sigma_t A^\top -  A \Sigma_t G^\top (G \Sigma_t G^\top + R)^{-1} G \Sigma_t A^\top + Q
```

This is a nonlinear difference equation in $\Sigma_t$.

A fixed point of {eq}`kalman_sdy` is a constant matrix $\Sigma$ such that

```{math}
:label: kalman_dare

\Sigma = A \Sigma A^\top -  A \Sigma G^\top (G \Sigma G^\top + R)^{-1} G \Sigma A^\top + Q
```

Equation {eq}`kalman_sdy` is known as a discrete-time Riccati difference equation.

Equation {eq}`kalman_dare` is known as a [discrete-time algebraic Riccati equation](https://en.wikipedia.org/wiki/Algebraic_Riccati_equation).

Conditions under which a fixed point exists and the sequence $\{\Sigma_t\}$ converges to it are discussed in {cite}`AHMS1996` and {cite}`AndersonMoore2005`, chapter 4.

A sufficient (but not necessary) condition is that all the eigenvalues $\lambda_i$ of $A$ satisfy $|\lambda_i| < 1$ (cf. e.g., {cite}`AndersonMoore2005`, p. 77).

(This strong condition assures that the unconditional  distribution of $X_t$  converges as $t \to \infty$.)

In this case, for any symmetric nonnegative definite initial choice of $\Sigma_0$, the sequence $\{\Sigma_t\}$ in {eq}`kalman_sdy` converges to a nonnegative symmetric matrix $\Sigma$ that solves {eq}`kalman_dare`.

## Implementation

```{index} single: Kalman Filter; Programming Implementation
```

The class `Kalman` from the [QuantEcon.py](https://quantecon.org/quantecon-py/) package implements the Kalman filter

* Instance data consists of:
    * the moments $(\mu_t, \Sigma_t)$ of the current prior, stored as the attributes `x_hat` and `Sigma` (the mean $\mu_t$ is named `x_hat` because it is also written $\hat x_t$ in much of the literature).
    * An instance of the [LinearStateSpace](https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/lss.py) class from [QuantEcon.py](https://quantecon.org/quantecon-py/).

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
Q := C C^\top \quad \text{and} \quad R := H H^\top
$$

* The class `Kalman` from the [QuantEcon.py](https://quantecon.org/quantecon-py/) package has a number of methods, some that we will wait to use until we study more advanced applications in subsequent lectures.
* Methods pertinent for this lecture  are:
    * `prior_to_filtered`, which updates $(\mu_t, \Sigma_t)$ to $(\mu_t^F, \Sigma_t^F)$
    * `filtered_to_forecast`, which updates the filtering distribution to the predictive distribution -- which becomes the new prior $(\mu_{t+1}, \Sigma_{t+1})$
    * `update`, which combines the last two methods
    * a `stationary_values`, which computes the solution to {eq}`kalman_dare` and the corresponding (stationary) Kalman gain

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

The task of this exercise to simulate the model and, using the code from `kalman.py`, plot the first five predictive densities $p_t(x) = N(\mu_t, \Sigma_t)$.

As shown in {cite}`Ljungqvist2012`, sections 2.9.1--2.9.2, these distributions asymptotically put all mass on the unknown value $\theta$.

In the simulation, take $\theta = 10$, $\mu_0 = 8$ and $\Sigma_0 = 1$.

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
μ_0, Σ_0 = 8, 1
kalman = Kalman(ss, μ_0, Σ_0)

# Draw observations of y from state space model
N = 5
x, y = ss.simulate(N)
y = y.flatten()

# Set up plot
fig, ax = plt.subplots(figsize=(10,8))
xgrid = np.linspace(θ - 5, θ + 2, 200)

for i in range(N):
    # Record the current predicted mean and variance
    m, v = kalman.x_hat.item(), kalman.Sigma.item()
    # Plot, update filter
    ax.plot(xgrid, norm.pdf(xgrid, loc=m, scale=np.sqrt(v)), label=f'$t={i}$')
    kalman.update(y[i])

ax.set_title(f'First {N} densities when $\\theta = {θ:.1f}$')
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

Plot $z_t$ against $t$, setting $\epsilon = 0.1$ and $T = 600$.

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

μ_0, Σ_0 = 8, 1
kalman = Kalman(ss, μ_0, Σ_0)

T = 600
z = np.empty(T)
x, y = ss.simulate(T)
y = y.flatten()

for t in range(T):
    # Record the current predicted mean and variance and plot their densities
    m, v = kalman.x_hat.item(), kalman.Sigma.item()

    f = lambda x: norm.pdf(x, loc=m, scale=np.sqrt(v))
    integral, error = quad(f, θ - ϵ, θ + ϵ)
    z[t] = 1 - integral

    kalman.update(y[t])

fig, ax = plt.subplots(figsize=(9, 7))
ax.set_ylim(0, 1)
ax.set_xlim(0, T)
ax.plot(range(T), z)
ax.fill_between(range(T), np.zeros(T), z, color="blue", alpha=0.2)
plt.show()
```

```{solution-end}
```

```{exercise-start}
:label: kalman_ex3
```

As discussed {ref}`above <kalman_convergence>`, if the shock sequence $\{w_t\}$ is not degenerate, then it is not in general possible to predict $x_t$ without error at time $t-1$ (and this would be the case even if we could observe $x_{t-1}$).

Let's now compare the prediction $\mu_t$ made by the Kalman filter
against a competitor who **is** allowed to observe $x_{t-1}$.

This competitor will use the conditional expectation $\mathbb E[ x_t
\,|\, x_{t-1}]$, which in this case is $A x_{t-1}$.

The conditional expectation is known to be the optimal prediction method in terms of minimizing mean squared error.

(More precisely, the minimizer of $\mathbb E \, \| x_t - g(x_{t-1}) \|^2$ with respect to $g$ is $g^*(x_{t-1}) := \mathbb E[ x_t \,|\, x_{t-1}]$)

Thus we are comparing the Kalman filter against a competitor who has more
information (in the sense of being able to observe the latent state) and
behaves optimally in terms of minimizing squared error.

Our horse race will be assessed in terms of squared error.

In particular, your task is to generate a graph plotting observations of both $\| x_t - A x_{t-1} \|^2$ and $\| x_t - \mu_t \|^2$ against $t$ for $t = 1, \ldots, 49$.

For the parameters, set $G = I, R = 0.5 I$ and $Q = 0.3 I$, where $I$ is
the $2 \times 2$ identity.

Set

$$
A
= \left(
\begin{array}{cc}
    0.5 & 0.4 \\
    0.6 & 0.3
\end{array}
  \right)
$$

To initialize the prior density, set

$$
\Sigma_0
= \left(
\begin{array}{cc}
    0.9 & 0.3 \\
    0.3 & 0.9
\end{array}
  \right)
$$

and $\mu_0 = (8, 8)$.

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
G = np.identity(2)
H = np.sqrt(0.5) * np.identity(2)

A = [[0.5, 0.4],
     [0.6, 0.3]]
C = np.sqrt(0.3) * np.identity(2)

# Set up state space mode, initial value x_0 set to zero
ss = LinearStateSpace(A, C, G, H, mu_0 = np.zeros(2))

# Define the prior density
Σ = [[0.9, 0.3],
     [0.3, 0.9]]
Σ = np.array(Σ)
μ = np.array([8, 8])

# Initialize the Kalman filter
kn = Kalman(ss, μ, Σ)

# Print eigenvalues of A
print("Eigenvalues of A:")
print(eigvals(A))

# Print stationary Σ
S, K = kn.stationary_values()
print("Stationary prediction error variance:")
print(S)

# Generate the plot
T = 50
x, y = ss.simulate(T)

e1 = np.empty(T-1)
e2 = np.empty(T-1)

for t in range(1, T):
    kn.update(y[:, t-1])
    diff1 = x[:, t] - kn.x_hat.flatten()
    diff2 = x[:, t] - A @ x[:, t-1]
    e1[t-1] = diff1 @ diff1
    e2[t-1] = diff2 @ diff2

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(range(1, T), e1, 'k-', lw=2, alpha=0.6,
        label='Kalman filter error')
ax.plot(range(1, T), e2, 'g-', lw=2, alpha=0.6,
        label='Conditional expectation error')
ax.legend()
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
