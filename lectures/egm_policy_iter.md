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

```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`Optimal Growth IV: The Endogenous Grid Method <single: Optimal Growth IV: The Endogenous Grid Method>`

```{contents} Contents
:depth: 2
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install interpolation
```

## Overview

Previously, we solved the stochastic optimal growth model using

1. {doc}`value function iteration <optgrowth_fast>`
1. {doc}`Euler equation based time iteration <coleman_policy_iter>`

We found time iteration to be significantly more accurate and efficient.

In this lecture, we'll look at a clever twist on time iteration called the **endogenous grid method** (EGM).

EGM is a numerical method for implementing policy iteration invented by [Chris Carroll](http://www.econ2.jhu.edu/people/ccarroll/).

The original reference is {cite}`Carroll2006`.

Let's start with some standard imports:

```{code-cell} ipython
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
import numpy as np
from interpolation import interp
from numba import njit
```

## Key Idea

Let's start by reminding ourselves of the theory and then see how the numerics fit in.

### Theory

Take the model set out in {doc}`the time iteration lecture <coleman_policy_iter>`, following the same terminology and notation.

The Euler equation is

```{math}
:label: egm_euler

(u'\circ \sigma^*)(y)
= \beta \int (u'\circ \sigma^*)(f(y - \sigma^*(y)) z) f'(y - \sigma^*(y)) z \phi(dz)
```

As we saw, the Coleman-Reffett operator is a nonlinear operator $K$ engineered so that $\sigma^*$ is a fixed point of $K$.

It takes as its argument a continuous strictly increasing consumption policy $\sigma \in \Sigma$.

It returns a new function $K \sigma$,  where $(K \sigma)(y)$ is the $c \in (0, \infty)$ that solves

```{math}
:label: egm_coledef

u'(c)
= \beta \int (u' \circ \sigma) (f(y - c) z ) f'(y - c) z \phi(dz)
```

### Exogenous Grid

As discussed in {doc}`the lecture on time iteration <coleman_policy_iter>`, to implement the method on a computer, we need a numerical approximation.

In particular, we represent a policy function by a set of values on a finite grid.

The function itself is reconstructed from this representation when necessary, using interpolation or some other method.

{doc}`Previously <coleman_policy_iter>`, to obtain a finite representation of an updated consumption policy, we

* fixed a grid of income points $\{y_i\}$
* calculated the consumption value $c_i$ corresponding to each
  $y_i$ using {eq}`egm_coledef` and a root-finding routine

Each $c_i$ is then interpreted as the value of the function $K \sigma$ at $y_i$.

Thus, with the points $\{y_i, c_i\}$ in hand, we can reconstruct $K \sigma$ via approximation.

Iteration then continues...

### Endogenous Grid

The method discussed above requires a root-finding routine to find the
$c_i$ corresponding to a given income value $y_i$.

Root-finding is costly because it typically involves a significant number of
function evaluations.

As pointed out by Carroll {cite}`Carroll2006`, we can avoid this if
$y_i$ is chosen endogenously.

The only assumption required is that $u'$ is invertible on $(0, \infty)$.

Let $(u')^{-1}$ be the inverse function of $u'$.

The idea is this:

* First, we fix an *exogenous* grid $\{k_i\}$ for capital ($k = y - c$).
* Then we obtain  $c_i$ via

```{math}
:label: egm_getc

c_i =
(u')^{-1}
\left\{
    \beta \int (u' \circ \sigma) (f(k_i) z ) \, f'(k_i) \, z \, \phi(dz)
\right\}
```

* Finally, for each $c_i$ we set $y_i = c_i + k_i$.

It is clear that each $(y_i, c_i)$ pair constructed in this manner satisfies {eq}`egm_coledef`.

With the points $\{y_i, c_i\}$ in hand, we can reconstruct $K \sigma$ via approximation as before.

The name EGM comes from the fact that the grid $\{y_i\}$ is  determined **endogenously**.

## Implementation

As {doc}`before <coleman_policy_iter>`, we will start with a simple setting
where

* $u(c) = \ln c$,
* production is Cobb-Douglas, and
* the shocks are lognormal.

This will allow us to make comparisons with the analytical solutions

```{code-cell} python3
:load: _static/lecture_specific/optgrowth/cd_analytical.py
```

We reuse the `OptimalGrowthModel` class

```{code-cell} python3
:load: _static/lecture_specific/optgrowth_fast/ogm.py
```

### The Operator

Here's an implementation of $K$ using EGM as described above.

```{code-cell} python3
@njit
def K(σ_array, og):
    """
    The Coleman-Reffett operator using EGM

    """

    # Simplify names
    f, β = og.f, og.β
    f_prime, u_prime = og.f_prime, og.u_prime
    u_prime_inv = og.u_prime_inv
    grid, shocks = og.grid, og.shocks

    # Determine endogenous grid
    y = grid + σ_array  # y_i = k_i + c_i

    # Linear interpolation of policy using endogenous grid
    σ = lambda x: interp(y, σ_array, x)

    # Allocate memory for new consumption array
    c = np.empty_like(grid)

    # Solve for updated consumption value
    for i, k in enumerate(grid):
        vals = u_prime(σ(f(k) * shocks)) * f_prime(k) * shocks
        c[i] = u_prime_inv(β * np.mean(vals))

    return c
```

Note the lack of any root-finding algorithm.

### Testing

First we create an instance.

```{code-cell} python3
og = OptimalGrowthModel()
grid = og.grid
```

Here's our solver routine:

```{code-cell} python3
:load: _static/lecture_specific/coleman_policy_iter/solve_time_iter.py
```

Let's call it:

```{code-cell} python3
σ_init = np.copy(grid)
σ = solve_model_time_iter(og, σ_init)
```

Here is a plot of the resulting policy, compared with the true policy:

```{code-cell} python3
y = grid + σ  # y_i = k_i + c_i

fig, ax = plt.subplots()

ax.plot(y, σ, lw=2,
        alpha=0.8, label='approximate policy function')

ax.plot(y, σ_star(y, og.α, og.β), 'k--',
        lw=2, alpha=0.8, label='true policy function')

ax.legend()
plt.show()
```

The maximal absolute deviation between the two policies is

```{code-cell} python3
np.max(np.abs(σ - σ_star(y, og.α, og.β)))
```

How long does it take to converge?

```{code-cell} python3
%%timeit -n 3 -r 1
σ = solve_model_time_iter(og, σ_init, verbose=False)
```

Relative to time iteration, which as already found to be highly efficient, EGM
has managed to shave off still more run time without compromising accuracy.

This is due to the lack of a numerical root-finding step.

We can now solve the optimal growth model at given parameters extremely fast.
