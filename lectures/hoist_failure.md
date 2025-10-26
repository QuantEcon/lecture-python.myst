---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Fault Tree Uncertainties

```{contents} Contents
:depth: 2
```

## Overview

This lecture puts elementary tools to work to approximate probability distributions of the annual failure rates of a system consisting of
a number of critical parts.

We'll use log normal distributions to approximate probability distributions of critical  component parts.

To approximate the probability distribution of the *sum* of $n$ lognormal random variables (representing the system's total failure rate), we compute the convolution of these distributions.

We'll use the following concepts and tools:

* lognormal distributions
* the convolution theorem that describes the probability distribution of the sum of independent random variables
* fault tree analysis for approximating a failure rate of a multi-component system
* a hierarchical probability model for describing uncertain probabilities
* Fourier transforms and inverse Fourier transforms as efficient ways of computing convolutions of sequences

```{seealso}
For more on Fourier transforms, see {doc}`Circulant Matrices <eig_circulant>` as well as {doc}`Covariance Stationary Processes <advanced:arma>` and {doc}`Estimation of Spectra <advanced:estspec>`.
```

{cite:t}`Ardron_2018` and {cite:t}`Greenfield_Sargent_1993` applied these methods to approximate failure probabilities of safety systems in nuclear facilities.

These techniques respond to recommendations by {cite:t}`apostolakis1990` for quantifying uncertainty in safety system reliability.

We will use the following imports and settings throughout this lecture:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from tabulate import tabulate
import quantecon as qe

np.set_printoptions(precision=3, suppress=True)
```

## The lognormal distribution

If random variable $x$ follows a normal distribution with mean $\mu$ and variance $\sigma^2$, then $y = \exp(x)$ follows a **lognormal distribution** with parameters $\mu, \sigma^2$.

```{note}
We refer to $\mu$ and $\sigma^2$ as *parameters* rather than mean and variance because:
* $\mu$ and $\sigma^2$ are the mean and variance of $x = \log(y)$
* They are *not* the mean and variance of $y$
* The mean of $y$ is $\exp(\mu + \frac{1}{2}\sigma^2)$ and the variance is $(e^{\sigma^2} - 1) e^{2\mu + \sigma^2}$
```

A lognormal random variable $y$ is always nonnegative.

The probability density function for $y$ is

```{math}
:label: lognormal_pdf

f(y) = \frac{1}{y \sigma \sqrt{2 \pi}} \exp \left( \frac{- (\log y - \mu)^2 }{2 \sigma^2} \right), \quad y \geq 0
```

Important properties of a lognormal random variable are:

```{math}
:label: lognormal_properties

\begin{aligned}
 \text{Mean:} & \quad e ^{\mu + \frac{1}{2} \sigma^2} \\
 \text{Variance:}  & \quad (e^{\sigma^2} - 1) e^{2 \mu + \sigma^2} \\
  \text{Median:} & \quad e^\mu \\
 \text{Mode:} & \quad e^{\mu - \sigma^2} \\
 \text{0.95 quantile:} & \quad e^{\mu + 1.645 \sigma} \\
 \text{0.95/0.05 quantile ratio:}  & \quad e^{3.29 \sigma}
 \end{aligned}
```

### Stability properties

Recall that independent normally distributed random variables have the following stability property:

If $x_1 \sim N(\mu_1, \sigma_1^2)$ and $x_2 \sim N(\mu_2, \sigma_2^2)$ are independent, then $x_1 + x_2 \sim N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$.

Independent lognormal distributions have a different stability property: the *product* of independent lognormal random variables is also lognormal.

Specifically, if $y_1$ is lognormal with parameters $(\mu_1, \sigma_1^2)$ and $y_2$ is lognormal with parameters $(\mu_2, \sigma_2^2)$, then $y_1 y_2$ is lognormal with parameters $(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$.

```{warning}
While the product of two lognormal distributions is lognormal, the *sum* of two lognormal distributions is *not* lognormal.
```

This observation motivates the central challenge of this lecture: approximating the probability distribution of *sums* of independent lognormal random variables.

## The convolution theorem

Let $x$ and $y$ be independent random variables with probability densities $f(x)$ and $g(y)$, where $x, y \in \mathbb{R}$.

Let $z = x + y$.

Then the probability density of $z$ is

```{math}
:label: convolution_continuous

h(z) = (f * g)(z) \equiv \int_{-\infty}^\infty f(\tau) g(z - \tau) d\tau
```

where $(f*g)$ denotes the **convolution** of $f$ and $g$.

For nonnegative random variables, this specializes to

```{math}
:label: convolution_nonnegative

h(z) = (f * g)(z) \equiv \int_{0}^z f(\tau) g(z - \tau) d\tau
```

### Discrete convolution

We will use a discretized version of the convolution formula.

We replace both $f$ and $g$ with discretized counterparts, normalized to sum to 1.

The discrete convolution formula is

```{math}
:label: convolution_discrete

h_n = (f*g)_n = \sum_{m=0}^n f_m g_{n-m}, \quad n \geq 0
```

This computes the probability mass function of the sum of two discrete random variables.

### Example: discrete distributions

Consider two probability mass functions:

$$
f_j = \Pr(X = j), \quad j = 0, 1
$$

and

$$
g_j = \Pr(Y = j), \quad j = 0, 1, 2, 3
$$

The distribution of $Z = X + Y$ is given by the convolution $h = f * g$.

```{code-cell} ipython3
# Define probability mass functions
f = [0.75, 0.25]
g = [0.0, 0.6, 0.0, 0.4]

# Compute convolution using two methods
h = np.convolve(f, g)
hf = fftconvolve(f, g)

print(f"f = {f}, sum = {np.sum(f):.3f}")
print(f"g = {g}, sum = {np.sum(g):.3f}")
print(f"h = {h}, sum = {np.sum(h):.3f}")
print(f"hf = {hf}, sum = {np.sum(hf):.3f}")
```

Both `numpy.convolve` and `scipy.signal.fftconvolve` produce the same result, but `fftconvolve` is much faster for long sequences.

We will use `fftconvolve` throughout this lecture for efficiency.

## Approximating continuous distributions

We now verify that discretized distributions can accurately approximate samples from underlying continuous distributions.

We generate samples of size 25,000 from three independent lognormal random variables and compute their pairwise and triple-wise sums.

We then compare histograms of the samples with histograms of the discretized distributions.

```{code-cell} ipython3
# Set parameters for lognormal distributions
μ, σ = 5.0, 1.0
n_samples = 25000

# Generate samples
np.random.seed(1234)
s1 = np.random.lognormal(μ, σ, n_samples)
s2 = np.random.lognormal(μ, σ, n_samples)
s3 = np.random.lognormal(μ, σ, n_samples)

# Compute sums
ssum2 = s1 + s2
ssum3 = s1 + s2 + s3

# Plot histogram of s1
fig, ax = plt.subplots()
ax.hist(s1, 1000, density=True, alpha=0.6)
ax.set_xlabel('value')
ax.set_ylabel('density')
plt.show()
```

```{code-cell} ipython3
# Plot histogram of sum of two lognormal distributions
fig, ax = plt.subplots()
ax.hist(ssum2, 1000, density=True, alpha=0.6)
ax.set_xlabel('value')
ax.set_ylabel('density')
plt.show()
```

```{code-cell} ipython3
# Plot histogram of sum of three lognormal distributions
fig, ax = plt.subplots()
ax.hist(ssum3, 1000, density=True, alpha=0.6)
ax.set_xlabel('value')
ax.set_ylabel('density')
plt.show()
```

Let's verify that the sample mean matches the theoretical mean:

```{code-cell} ipython3
samp_mean = np.mean(s2)
theoretical_mean = np.exp(μ + σ**2 / 2)

print(f"Theoretical mean: {theoretical_mean:.3f}")
print(f"Sample mean: {samp_mean:.3f}")
```

## Discretizing the lognormal distribution

We define helper functions to create discretized versions of lognormal probability density functions.

```{code-cell} ipython3
def lognormal_pdf(x, μ, σ):
    """
    Compute lognormal probability density function.
    """
    p = 1 / (σ * x * np.sqrt(2 * np.pi)) \
            * np.exp(-0.5 * ((np.log(x) - μ) / σ)**2)
    return p


def discretize_lognormal(μ, σ, I, m):
    """
    Create discretized lognormal probability mass function.
    """
    x = np.arange(1e-7, I, m)
    p_array = lognormal_pdf(x, μ, σ)
    p_array_norm = p_array / np.sum(p_array)
    return p_array, p_array_norm, x
```

We set the grid length $I$ to a power of 2 to enable efficient Fast Fourier Transform computation.

```{note}
Increasing the power $p$ (e.g., from 12 to 15) improves the approximation quality but increases computational cost.
```

```{code-cell} ipython3
# Set grid parameters
p = 15
I = 2**p  # Truncation value (power of 2 for FFT efficiency)
m = 0.1   # Increment size
```

Let's visualize how well the discretized distribution approximates the continuous lognormal distribution:

```{code-cell} ipython3
# Compute discretized PDF
pdf, pdf_norm, x = discretize_lognormal(μ, σ, I, m)

# Plot discretized PDF against histogram
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, pdf, 'r-', lw=2, label='discretized PDF')
ax.hist(s1, 1000, density=True, alpha=0.6, label='sample histogram')
ax.set_xlim(0, 2500)
ax.set_xlabel('value')
ax.set_ylabel('density')
ax.legend()
plt.show()
```

Now let's verify that the discretized distribution has the correct mean:

```{code-cell} ipython3
# Compute mean from discretized PDF
mean_discrete = np.sum(x * pdf_norm)
mean_theory = np.exp(μ + 0.5 * σ**2)

print(f"Theoretical mean: {mean_theory:.3f}")
print(f"Discretized mean: {mean_discrete:.3f}")
```

## Convolving probability mass functions

Now let's use the convolution theorem to compute the probability distribution of a sum of the two lognormal random variables we have parameterized above.

We'll also compute the probability of a sum of three log normal distributions constructed above.

For long sequences, `scipy.signal.fftconvolve` is much faster than `numpy.convolve` because it uses Fast Fourier Transforms.

Let's define the Fourier transform and the inverse Fourier transform first

### The Fast Fourier Transform

The **Fourier transform** of a sequence $\{x_t\}_{t=0}^{T-1}$ is

```{math}
:label: eq:ft1

x(\omega_j) = \sum_{t=0}^{T-1} x_t \exp(-i \omega_j t)
```

where $\omega_j = \frac{2\pi j}{T}$ for $j = 0, 1, \ldots, T-1$.

The **inverse Fourier transform** of the sequence $\{x(\omega_j)\}_{j=0}^{T-1}$ is

```{math}
:label: eq:ift1

x_t = T^{-1} \sum_{j=0}^{T-1} x(\omega_j) \exp(i \omega_j t)
```

The sequences $\{x_t\}_{t=0}^{T-1}$ and $\{x(\omega_j)\}_{j=0}^{T-1}$ contain the same information.

The pair of equations {eq}`eq:ft1` and {eq}`eq:ift1` tell how to recover one series from its Fourier partner.


The program `scipy.signal.fftconvolve` deploys  the theorem that  a convolution of two sequences $\{f_k\}, \{g_k\}$ can be computed in the following way:

-  Compute Fourier transforms $F(\omega), G(\omega)$ of the $\{f_k\}$ and $\{g_k\}$ sequences, respectively
-  Form the product $H (\omega) = F(\omega) G (\omega)$
- The convolution of $f * g$ is the inverse Fourier transform of $H(\omega)$

The **fast Fourier transform** and the associated **inverse fast Fourier transform** execute these calculations very quickly.

This is the algorithm used by `fftconvolve`.

Let's do a warmup calculation that compares the times taken by `numpy.convolve` and `scipy.signal.fftconvolve`

```{code-cell} ipython3
# Discretize three lognormal distributions
_, pmf1, x = discretize_lognormal(μ, σ, I, m)
_, pmf2, x = discretize_lognormal(μ, σ, I, m)
_, pmf3, x = discretize_lognormal(μ, σ, I, m)

# Time numpy.convolve
with qe.Timer() as timer_numpy:
    conv_np = np.convolve(pmf1, pmf2)
    conv_np = np.convolve(conv_np, pmf3)
time_numpy = timer_numpy.elapsed

# Time fftconvolve
with qe.Timer() as timer_fft:
    conv_fft = fftconvolve(pmf1, pmf2)
    conv_fft = fftconvolve(conv_fft, pmf3)
time_fft = timer_fft.elapsed

print(f"Time with np.convolve: {time_numpy:.4f} seconds")
print(f"Time with fftconvolve: {time_fft:.4f} seconds")
print(f"Speedup factor: {time_numpy / time_fft:.1f}x")
```

The Fast Fourier Transform provides orders of magnitude speedup.

Now let’s plot our computed probability mass function approximation for the sum of two log normal random variables against the histogram of the sample that we formed above

```{code-cell} ipython3
# Compute convolution of two distributions for comparison
conv2 = fftconvolve(pmf1, pmf2)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, conv2[:len(x)] / m, 'r-', lw=2, label='convolution (FFT)')
ax.hist(ssum2, 1000, density=True, alpha=0.6, label='sample histogram')
ax.set_xlim(0, 5000)
ax.set_xlabel('value')
ax.set_ylabel('density')
ax.legend()
plt.show()
```

Now we present the plot for the sum of three lognormal random variables:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, conv_fft[:len(x)] / m, 'r-', lw=2, label='convolution (FFT)')
ax.hist(ssum3, 1000, density=True, alpha=0.6, label='sample histogram')
ax.set_xlim(0, 5000)
ax.set_xlabel('value')
ax.set_ylabel('density')
ax.legend()
plt.show()
```

Let's verify that the means are correct

```{code-cell} ipython3
# Mean of sum of two distributions
mean_conv2 = np.sum(x * conv2[:len(x)])
mean_theory2 = 2 * np.exp(μ + 0.5 * σ**2)

print(f"Sum of two distributions:")
print(f"  Theoretical mean: {mean_theory2:.3f}")
print(f"  Computed mean: {mean_conv2:.3f}")
```

```{code-cell} ipython3
# Mean of sum of three distributions
mean_conv3 = np.sum(x * conv_fft[:len(x)])
mean_theory3 = 3 * np.exp(μ + 0.5 * σ**2)

print(f"Sum of three distributions:")
print(f"  Theoretical mean: {mean_theory3:.3f}")
print(f"  Computed mean: {mean_conv3:.3f}")
```

## Fault tree analysis

We shall soon apply the convolution theorem to compute the probability of a **top event** in a failure tree analysis.

Before applying the convolution theorem, we first describe the model that connects constituent events to the *top* end whose failure rate we seek to quantify.

Fault tree analysis is a widely used technique for assessing system reliability, as described by {cite:t}`Ardron_2018`.

To construct the statistical model, we repeatedly use  what is called the **rare event approximation**.

### The rare event approximation

We want to compute the probability of an event $A \cup B$.

For events $A$ and $B$, the probability of the union is

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

where $A \cup B$ is the event that $A$ **or** $B$ occurs, and $A \cap B$ is the event that $A$ **and** $B$ both occur.

If $A$ and $B$ are independent, then $P(A \cap B) = P(A) P(B)$.

When $P(A)$ and $P(B)$ are both small, $P(A) P(B)$ is even smaller.

The **rare event approximation** is

$$
P(A \cup B) \approx P(A) + P(B)
$$

This approximation is widely used in system failure analysis.

### System failure probability

Consider a system with $n$ critical components where system failure occurs when *any* component fails.

We assume:

* The failure probability $P(A_i)$ of each component $A_i$ is small
* Component failures are statistically independent

We repeatedly apply a **rare event approximation** to obtain the following formula for the problem of a system failure:

$$ 
P(F) \approx P(A_1) + P (A_2) + \cdots + P (A_n) 
$$

or

```{math}
:label: eq:probtop

P(F) \approx \sum_{i=1}^n P(A_i)
```

where $P(F)$ is the system failure probability.

Probabilities for each event are recorded as failure rates per year.

## Failure Rates Unknown

Now we come to the problem that really interests us, following  {cite:t}`Ardron_2018` and
 {cite:t}`Greenfield_Sargent_1993`  in the spirit of  {cite:t}`apostolakis1990`.

The component failure rates $P(A_i)$ are not known precisely and must be estimated.

We address this problem by specifying **probabilities of probabilities** that  capture one  notion of not knowing the constituent probabilities that are inputs into a failure tree analysis.


Thus, we assume that a system analyst is uncertain about  the failure rates $P(A_i), i =1, \ldots, n$ for components of a system.

The analyst copes with this situation by regarding the systems failure probability $P(F)$ and each of the component probabilities $P(A_i)$ as  random variables.

  * dispersions of the probability distribution of $P(A_i)$ characterizes the analyst's uncertainty about the failure probability $P(A_i)$

  * the dispersion of the implied probability distribution of $P(F)$ characterizes his uncertainty about the probability of a system's failure.

This leads to what is sometimes called a **hierarchical** model in which the analyst has  probabilities about the probabilities $P(A_i)$.

The analyst formalizes his uncertainty by assuming that

 * the failure probability $P(A_i)$ is itself a log normal random variable with parameters $(\mu_i, \sigma_i)$.
 * failure rates $P(A_i)$ and $P(A_j)$ are statistically independent for all pairs with $i \neq j$.

The analyst  calibrates the parameters  $(\mu_i, \sigma_i)$ for the failure events $i = 1, \ldots, n$ by reading reliability studies in engineering papers that have studied historical failure rates of components that are as similar as possible to the components being used in the system under study.

The analyst assumes that such  information about the observed dispersion of annual failure rates, or times to failure, can inform him of what to expect about parts' performances in his system.

The analyst  assumes that the random variables $P(A_i)$   are  statistically mutually independent.

The analyst wants to approximate a probability mass function and cumulative distribution function
of the systems failure probability $P(F)$.

  * We say probability mass function because of how we discretize each random variable, as described earlier.

The analyst calculates the probability mass function for the *top event* $F$, i.e., a *system failure*,  by repeatedly applying the convolution theorem to compute the probability distribution of a sum of independent log normal random variables, as described in equation
{eq}`eq:probtop`.

## Application: waste hoist failure rate

We now analyze a real-world example with $n = 14$ components.

The application estimates the annual failure rate of a critical hoist at a nuclear waste facility.

A regulatory agency requires the system to be designed so that the top event failure rate is small with high probability.

### Model specification

We'll take close to a real world example by assuming that $n = 14$.

The example estimates the annual failure rate of a critical  hoist at a nuclear waste facility.

A regulatory agency wants the system to be designed in a way that makes the failure rate of the top event small with high probability.

This example is Design Option B-2 (Case I) described in Table 10 on page 27 of {cite:t}`Greenfield_Sargent_1993`.

The table describes parameters $\mu_i, \sigma_i$ for  fourteen log normal random variables that consist of  **seven pairs** of random variables that are identically and independently distributed.

 * Within a pair, parameters $\mu_i, \sigma_i$ are the same

 * As described in table 10 of {cite:t}`Greenfield_Sargent_1993`  p. 27, parameters of log normal distributions for  the seven unique probabilities $P(A_i)$ have been calibrated to be the values in the following Python code:


```{code-cell} ipython3
# Component failure rate parameters 
# (see Table 10 of Greenfield & Sargent 1993)
params = [
    (4.28, 1.1947),   # Component type 1
    (3.39, 1.1947),   # Component type 2
    (2.795, 1.1947),  # Component type 3
    (2.717, 1.1947),  # Component type 4
    (2.717, 1.1947),  # Component type 5
    (1.444, 1.4632),  # Component type 6
    (-0.040, 1.4632), # Component type 7 (appears 8 times)
]
```

```{note}
Since failure rates are very small, these lognormal distributions actually describe $P(A_i) \times 10^{-9}$.

So the probabilities that we'll put on the $x$ axis of the probability mass function and associated cumulative distribution function should be multiplied by $10^{-09}$
```

We define a helper function to find array indices:

```{code-cell} ipython3
def find_nearest(array, value):
    """
    Find the index of the array element nearest to the given value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
```

We compute the required thirteen convolutions in the following code.

(Please feel free to try different values of the power parameter $p$ that we use to set the number of points in our grid for constructing
the probability mass functions that discretize the continuous log normal distributions.)

```{code-cell} ipython3
# Set grid parameters
p = 15
I = 2**p
m = 0.05

# Discretize all component failure rate distributions
# First 6 components use unique parameters, last 8 share the same parameters
component_pmfs = []
for μ, σ in params[:6]:
    _, pmf, x = discretize_lognormal(μ, σ, I, m)
    component_pmfs.append(pmf)

# Add 8 copies of component type 7
μ7, σ7 = params[6]
_, pmf7, x = discretize_lognormal(μ7, σ7, I, m)
component_pmfs.extend([pmf7] * 8)

# Compute system failure distribution via sequential convolution
with qe.Timer() as timer:
    system_pmf = component_pmfs[0]
    for pmf in component_pmfs[1:]:
        system_pmf = fftconvolve(system_pmf, pmf)

print(f"Time for 13 convolutions: {timer.elapsed:.4f} seconds")
```

We now plot a counterpart to the cumulative distribution function (CDF) in  figure 5 on page 29 of {cite:t}`Greenfield_Sargent_1993`

```{code-cell} ipython3
# Compute cumulative distribution function
cdf = np.cumsum(system_pmf)

# Plot CDF
Nx = 1400
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x[:int(Nx / m)], cdf[:int(Nx / m)], 'b-', lw=2)

# Add reference lines for key quantiles
quantile_levels = [0.05, 0.10, 0.50, 0.90, 0.95]
for q in quantile_levels:
    ax.axhline(q, color='gray', linestyle='--', alpha=0.5)

ax.set_xlim(0, Nx)
ax.set_ylim(0, 1)
ax.set_xlabel(r'failure rate ($\times 10^{-9}$ per year)')
ax.set_ylabel('cumulative probability')
plt.show()
```

We also present a counterpart to their Table 11 on page 28 of {cite:t}`Greenfield_Sargent_1993`, which lists key quantiles of the system failure rate distribution


```{code-cell} ipython3
# Find quantiles
quantiles = [0.01, 0.05, 0.10, 0.50, 0.665, 0.85, 0.90, 0.95, 0.99, 0.9978]
quantile_values = [x[find_nearest(cdf, q)] for q in quantiles]

# Create table
table_data = [[f"{100*q:.2f}%", f"{val:.3f}"]
              for q, val in zip(quantiles, quantile_values)]

print("\nSystem failure rate quantiles (×10^-9 per year):")
print(tabulate(table_data, 
      headers=['Percentile', 'Failure rate'], tablefmt='grid'))
```

The computed quantiles agree closely with column 2 of Table 11 on page 28 of {cite}`Greenfield_Sargent_1993`.

Minor discrepancies may be due to differences in:
* Numerical precision of input parameters $\mu_i, \sigma_i$
* Number of grid points in the discretization
* Grid increment size

## Exercises

```{exercise-start}
:label: hoist_ex1
```

Experiment with different values of the power parameter $p$ (which determines the grid size as $I = 2^p$).

Try $p \in \{12, 13, 14, 15, 16\}$ and compare:
1. Computation time
2. Accuracy of the median (50th percentile) compared to the reference value
3. Memory usage implications

What trade-offs do you observe?
```{exercise-end}
```

```{solution-start} hoist_ex1
:class: dropdown
```

Here is one solution:

```{code-cell} ipython3
# Test different grid sizes
p_values = [12, 13, 14, 15, 16]
results = []

for p_test in p_values:
    I_test = 2**p_test
    m_test = 0.05

    # Discretize distributions
    pmfs_test = []
    for μ, σ in params[:6]:
        _, pmf, x_test = discretize_lognormal(μ, σ, I_test, m_test)
        pmfs_test.append(pmf)

    # Add 8 copies of component type 7
    μ7, σ7 = params[6]
    _, pmf7, x_test = discretize_lognormal(μ7, σ7, I_test, m_test)
    pmfs_test.extend([pmf7] * 8)

    # Time the convolutions
    with qe.Timer() as timer_test:
        system_test = pmfs_test[0]
        for pmf in pmfs_test[1:]:
            system_test = fftconvolve(system_test, pmf)

    # Compute median
    cdf_test = np.cumsum(system_test)
    median = x_test[find_nearest(cdf_test, 0.5)]

    results.append([p_test, I_test,
        f"{timer_test.elapsed:.4f}", f"{median:.7f}"])

print(tabulate(results,
               headers=['p', 'Grid size (2^p)', 'Time (s)', 'Median'],
               tablefmt='grid'))
```
The results typically show the following trade-offs:

- Larger grid sizes provide better accuracy but increase computation time
- The relationship between $p$ and computation time is roughly linear for FFT-based convolution
- Beyond $p = 13$, the accuracy gains diminish while computational cost continues to grow
- For this application, $p = 13$ provides a good balance between accuracy and efficiency

```{solution-end}
```

```{exercise-start}
:label: hoist_ex2
```

The rare event approximation assumes that $P(A_i) P(A_j)$ is negligible compared to $P(A_i) + P(A_j)$.

Using the computed distribution, calculate the expected value of the system failure rate and compare it to the sum of the expected values of the individual component failure rates.

How good is the rare event approximation in this case?
```{exercise-end}
```


```{solution-start} hoist_ex2
:class: dropdown
```

Here is one solution:

```{code-cell} ipython3
# Create extended grid for the convolution result
x_extended = np.arange(0, len(system_pmf) * m, m)
E_system = np.sum(x_extended * system_pmf)

# Compute sum of individual expected values
component_means = [np.exp(μ + 0.5 * σ**2) for μ, σ in params[:6]]
# Add 8 components of type 7
μ7, σ7 = params[6]
component_means.extend([np.exp(μ7 + 0.5 * σ7**2)] * 8)

E_sum = sum(component_means)

print(f"Expected system failure rate: {E_system:.3f} × 10^-9")
print(f"Sum of component expected failure rates: {E_sum:.3f} × 10^-9")
print(f"Relative difference: {100 * abs(E_system - E_sum) / E_sum:.2f}%")
```

The rare event approximation works well when failure probabilities are small. 

The expected value of the sum equals the sum of the expected values (by linearity of expectation), so these should match closely regardless of the rare event approximation.

```{solution-end}
```
