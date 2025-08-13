---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(divergence_measures)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Measuring Divergence Between Distributions

```{contents} Contents
:depth: 2
```

## Overview

This lecture explores various measures for quantifying the difference between probability distributions.

We'll cover three important divergence measures and their applications in statistical inference:

* **Kullback-Leibler (KL) divergence** - measures the expected excess surprisal from using a misspecified model
* **Chernoff entropy** - provides bounds on error probabilities in hypothesis testing  
* **Jensen-Shannon divergence** - a symmetric measure based on KL divergence

We'll demonstrate how these measures relate to the ability of likelihood ratio tests to distinguish between distributions, and visualize distribution overlaps to build intuition.

Let's start by importing the necessary Python tools.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from numba import vectorize, jit
from math import gamma
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_dist
import pandas as pd
from IPython.display import display, Math
```

## Setup: Beta Distributions

For our examples, we'll work with Beta distributions. Let's define the parameters and density functions:

```{code-cell} ipython3
# Parameters in the two Beta distributions
F_a, F_b = 1, 1
G_a, G_b = 3, 1.2

@vectorize
def p(x, a, b):
    r = gamma(a + b) / (gamma(a) * gamma(b))
    return r * x** (a-1) * (1 - x) ** (b-1)

# The two density functions
f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))

# Plot the distributions
x_range = np.linspace(0.001, 0.999, 1000)
f_vals = [f(x) for x in x_range]
g_vals = [g(x) for x in x_range]

plt.figure(figsize=(10, 6))
plt.plot(x_range, f_vals, 'b-', linewidth=2, label=r'$f(x) \sim \text{Beta}(1,1)$')
plt.plot(x_range, g_vals, 'r-', linewidth=2, label=r'$g(x) \sim \text{Beta}(3,1.2)$')

# Fill overlap region
overlap = np.minimum(f_vals, g_vals)
plt.fill_between(x_range, 0, overlap, alpha=0.3, color='purple', label='overlap')

plt.xlabel('x')
plt.ylabel('density')
plt.legend()
plt.show()
```

(rel_entropy)=
## Kullback–Leibler divergence

The first measure we'll study is the **Kullback–Leibler (KL) divergence**.

Let's consider two probability distributions $f$ and $g$.

We want to quantify how different they are.

In this particular case, we want to know that if we use $g$ to approximate $f$, how surprised we would be.

The **Kullback–Leibler divergence** is defined as:

$$
D_{KL}(f\|g) = KL(f,g) = \int f(x) \log\frac{f(x)}{g(x)} dx
$$

This measures the expected log difference between the two distributions, weighted by $f$.

KL divergence measures the expected excess surprisal from using model $g$ when the true model is $f$.

It has several important properties:

- $KL(f, g) \geq 0$ with equality if and only if $f = g$ almost everywhere

- In general, $KL(f, g) \neq KL(g, f)$ (not symmetric)

- It's not a proper metric but is fundamental in information theory

We will see that KL divergence playing a central role in statistical inference, particularly in model selection and hypothesis testing.

In particular {doc}`likelihood_ratio_process` discribes a link between the KL divergence and the expected log likelihood ratio.

It also underpins the performance of likelihood ratio tests as we shall see in {doc}`wald_friedman`.


Let's implement a function to compute KL divergence:

```{code-cell} ipython3
def compute_KL(f, g):
    """
    Compute KL divergence KL(f, g)
    """
    integrand = lambda w: f(w) * np.log(f(w) / g(w))
    val, _ = quad(integrand, 1e-5, 1-1e-5)
    return val
```

```{code-cell} ipython3
def compute_KL(f, g):
    """
    Compute KL divergence KL(f, g)
    """
    integrand = lambda w: f(w) * np.log(f(w) / g(w))
    val, _ = quad(integrand, 1e-5, 1-1e-5)
    return val

# Compute KL divergences between our example distributions
kl_fg = compute_KL(f, g)
kl_gf = compute_KL(g, f)

print(f"KL(f, g) = {kl_fg:.4f}")
print(f"KL(g, f) = {kl_gf:.4f}")
```

## Jensen-Shannon Divergence

Sometimes we want a symmetric measure of divergence that captures the difference between two distributions without favoring one over the other.

This often arises in applications like clustering, where we want to compare distributions without assuming one is the true model.

Another important application is in generative models, where we want to measure how well a model approximates a target distribution.

The **Jensen-Shannon (JS) divergence** is such a symmetric measure based on KL divergence:

$$
JS(f,g) = \frac{1}{2} KL(f, m) + \frac{1}{2} KL(g, m)
$$

where $m = \frac{1}{2}(f+g)$ is the mixture of $f$ and $g$.

When we take the square root of JS divergence, we get a proper metric called the **Jensen-Shannon distance**.

As we can see from the definition, JS divergence is symmetric: $JS(f,g) = JS(g,f)$.

```{code-cell} ipython3
def compute_JS(f, g):
    """
    Compute Jensen-Shannon divergence
    """
    def m(w):
        return 0.5 * (f(w) + g(w))
    
    js_div = 0.5 * compute_KL(f, m) + 0.5 * compute_KL(g, m)
    return js_div

js_div = compute_JS(f, g)
print(f"Jensen-Shannon divergence JS(f,g) = {js_div:.4f}")
```

Let's visualize the mixture distribution $m$:

```{code-cell} ipython3
def m(x):
    return 0.5 * (f(x) + g(x))

m_vals = [m(x) for x in x_range]

plt.figure(figsize=(10, 6))
plt.plot(x_range, f_vals, 'b-', linewidth=2, label=r'$f(x)$')
plt.plot(x_range, g_vals, 'r-', linewidth=2, label=r'$g(x)$')
plt.plot(x_range, m_vals, 'g--', linewidth=2, label=r'$m(x) = \frac{1}{2}(f(x) + g(x))$')

plt.xlabel('x')
plt.ylabel('density')
plt.legend()
plt.show()
```

## Chernoff Entropy

Chernoff entropy was motivated by an early application of  the [theory of large deviations](https://en.wikipedia.org/wiki/Large_deviations_theory).

```{note}
Large deviation theory provides refinements of the central limit theorem. 
```

The Chernoff entropy between probability densities $f$ and $g$ is defined as:

$$
C(f,g) = - \log \min_{\phi \in (0,1)} \int f^\phi(x) g^{1-\phi}(x) dx
$$

Chernoff entropy provides an upper bound on model selection error probability: 
the error rate is bounded above by $e^{-C(f,g)T}$ where $T$ is the sample size.

We will see such a bound in action in {doc}`likelihood_ratio_process`.

```{code-cell} ipython3
def chernoff_integrand(ϕ, f, g):
    """
    Compute the integrand for Chernoff entropy
    """
    def integrand(w):
        return f(w)**ϕ * g(w)**(1-ϕ)

    result, _ = quad(integrand, 1e-5, 1-1e-5)
    return result

def compute_chernoff_entropy(f, g):
    """
    Compute Chernoff entropy C(f,g)
    """
    def objective(ϕ):
        return chernoff_integrand(ϕ, f, g)
    
    # Find the minimum over ϕ in (0,1)
    result = minimize_scalar(objective, 
                             bounds=(1e-5, 1-1e-5), 
                             method='bounded')
    min_value = result.fun
    ϕ_optimal = result.x
    
    chernoff_entropy = -np.log(min_value)
    return chernoff_entropy, ϕ_optimal

C_fg, ϕ_optimal = compute_chernoff_entropy(f, g)
print(f"Chernoff entropy C(f,g) = {C_fg:.4f}")
print(f"Optimal ϕ = {ϕ_optimal:.4f}")
```

## Comparing Divergence Measures

Let's create a comparison of these divergence measures across 
different pairs of Beta distributions

```{code-cell} ipython3
distribution_pairs = [
    # (f_params, g_params)
    ((1, 1), (0.1, 0.2)),
    ((1, 1), (0.3, 0.3)),
    ((1, 1), (0.3, 0.4)),
    ((1, 1), (0.5, 0.5)),
    ((1, 1), (0.7, 0.6)),
    ((1, 1), (0.9, 0.8)),
    ((1, 1), (1.1, 1.05)),
    ((1, 1), (1.2, 1.1)),
    ((1, 1), (1.5, 1.2)),
    ((1, 1), (2, 1.5)),
    ((1, 1), (2.5, 1.8)),
    ((1, 1), (3, 1.2)),
    ((1, 1), (4, 1)),
    ((1, 1), (5, 1))
]

# Create comparison table
results = []
for i, ((f_a, f_b), (g_a, g_b)) in enumerate(distribution_pairs):
    # Define the density functions
    f = jit(lambda x, a=f_a, b=f_b: p(x, a, b))
    g = jit(lambda x, a=g_a, b=g_b: p(x, a, b))
    
    # Compute measures
    kl_fg = compute_KL(f, g)
    kl_gf = compute_KL(g, f)
    js_div = compute_JS(f, g)
    chernoff_ent, _ = compute_chernoff_entropy(f, g)
    
    results.append({
        'Pair (f, g)': f"\\text{{Beta}}({f_a},{f_b}), \\text{{Beta}}({g_a},{g_b})",
        'KL(f, g)': f"{kl_fg:.4f}",
        'KL(g, f)': f"{kl_gf:.4f}",
        'JS': f"{js_div:.4f}",
        'C': f"{chernoff_ent:.4f}"
    })

df = pd.DataFrame(results)

# Sort by JS divergence
df['JS_numeric'] = df['JS'].astype(float)
df = df.sort_values('JS_numeric').drop('JS_numeric', axis=1)

# Generate LaTeX table manually
columns = ' & '.join([f'\\text{{{col}}}' for col in df.columns])
rows = ' \\\\\n'.join(
    [' & '.join([f'{val}' for val in row]) 
     for row in df.values])

latex_code = rf"""
\begin{{array}}{{lcccc}}
{columns} \\
\hline
{rows}
\end{{array}}
"""

display(Math(latex_code))
```

We can clearly see how the divergence measures covary as we vary the parameters of the Beta distributions.

Let's visualize this relationship between KL divergence, JS divergence, and Chernoff entropy

```{code-cell} ipython3
kl_fg_values = [float(result['KL(f, g)']) for result in results]
js_values = [float(result['JS']) for result in results]
chernoff_values = [float(result['C']) for result in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# JS divergence and KL divergence
axes[0].scatter(kl_fg_values, js_values, alpha=0.7, s=60)
axes[0].set_xlabel('KL divergence KL(f, g)')
axes[0].set_ylabel('JS divergence')
axes[0].set_title('JS divergence and KL divergence')

# Chernoff Entropy and JS divergence
axes[1].scatter(js_values, chernoff_values, alpha=0.7, s=60)
axes[1].set_xlabel('JS divergence')
axes[1].set_ylabel('Chernoff entropy')
axes[1].set_title('Chernoff entropy and JS divergence')

plt.tight_layout()
plt.show()
```

Let's explore further by creating visualizations showing how distribution overlap relates to divergence measures

```{code-cell} ipython3
def plot_dist_diff():
    """
    Plot overlap of two distributions and divergence measures
    """
    
    # Chose a subset of Beta distribution parameters
    param_grid = [
        ((1, 1), (1, 1)),   
        ((1, 1), (1.5, 1.2)),
        ((1, 1), (2, 1.5)),  
        ((1, 1), (3, 1.2)),  
        ((1, 1), (5, 1)),
        ((1, 1), (0.3, 0.3))
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    divergence_data = []
    
    for i, ((f_a, f_b), (g_a, g_b)) in enumerate(param_grid):
        row = i // 2
        col = i % 2
        
        # Create density functions
        f = jit(lambda x, a=f_a, b=f_b: p(x, a, b))
        g = jit(lambda x, a=g_a, b=g_b: p(x, a, b))
        
        # Compute divergence measures
        kl_fg = compute_KL(f, g)
        js_div = compute_JS(f, g) 
        chernoff_ent, _ = compute_chernoff_entropy(f, g)
        
        divergence_data.append({
            'f_params': (f_a, f_b),
            'g_params': (g_a, g_b),
            'kl_fg': kl_fg,
            'js_div': js_div,
            'chernoff': chernoff_ent
        })
        
        # Plot distributions
        x_range = np.linspace(0, 1, 200)
        f_vals = [f(x) for x in x_range]
        g_vals = [g(x) for x in x_range]
        
        axes[row, col].plot(x_range, f_vals, 'b-', linewidth=2, 
                           label=f'f ~ Beta({f_a},{f_b})')
        axes[row, col].plot(x_range, g_vals, 'r-', linewidth=2, 
                           label=f'g ~ Beta({g_a},{g_b})')
        
        # Fill overlap region
        overlap = np.minimum(f_vals, g_vals)
        axes[row, col].fill_between(x_range, 0, overlap, alpha=0.3, 
                                   color='purple', label='overlap')
        
        # Add divergence information
        axes[row, col].set_title(
            f'KL(f, g)={kl_fg:.3f}, JS={js_div:.3f}, C={chernoff_ent:.3f}',
            fontsize=12)
        axes[row, col].legend(fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    return divergence_data

divergence_data = plot_dist_diff()
```

## Related Lectures

This lecture serves as a foundation for understanding tools we use to capture the information content of statistical models that underpin many of our lectures:

- For a more detailed illustration of the relationship between divergence measures and statical inference, see {doc}`likelihood_ratio_process`, {doc}`wald_friedman`, and {doc}`mix_moidel`.

- These measures plays a crucial role in capturing the heterogeneity in the beliefs of agents in a model. 
For an application of this idea, see {doc}`likelihood_ratio_process_2` where we study how agents with different beliefs interact in a dynamic setting where we discuss the role of divergence measures in Lawrence Blume and David Easley's model on heterogeneous beliefs and financial markets.
