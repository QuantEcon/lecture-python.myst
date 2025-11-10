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

(mccall_with_sep)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Job Search II: Search and Separation

```{index} single: An Introduction to Job Search
```

```{contents} Contents
:depth: 2
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon jax
```

## Overview

Previously {doc}`we looked <mccall_model>` at the McCall job search model {cite}`McCall1970` as a way of understanding unemployment and worker decisions.

One unrealistic feature of that version of the model was that every job is permanent.

In this lecture, we extend the model by introducing job separation.

Once separation enters the picture, the agent comes to view

* the loss of a job as a capital loss, and
* a spell of unemployment as an *investment* in searching for an acceptable job

The other minor addition is that a utility function will be included to make
worker preferences slightly more sophisticated.

We'll need the following imports

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple
from quantecon.distributions import BetaBinomial
```

## The Model

The model is similar to the {doc}`baseline McCall job search model <mccall_model>`.

It concerns the life of an infinitely lived worker and

* the opportunities he or she (let's say he to save one character) has to work at different wages
* exogenous events that destroy his current job
* his decision making process while unemployed

The worker can be in one of two states: employed or unemployed.

He wants to maximize

```{math}
:label: objective

{\mathbb E} \sum_{t=0}^\infty \beta^t u(y_t)
```

At this stage the only difference from the {doc}`baseline model <mccall_model>` is that we've added some flexibility to preferences by
introducing a utility function $u$.

It satisfies $u'> 0$ and $u'' < 0$.

Wage offers $\{ w_t \}$ are IID with common distribution $q$.

The set of possible wage values is denoted by $\mathbb W$.

### Timing and Decisions

At the start of each period, the agent can be either

* unemployed or
* employed at some existing wage level $w$.

At the start of a given period, the current wage offer $w_t$ is observed.

If currently employed, the worker

1. receives utility $u(w)$ and
1. is fired with some (small) probability $\alpha$.

If currently unemployed, the worker either accepts or rejects the current offer $w_t$.

If he accepts, then he begins work immediately at wage $w_t$.

If he rejects, then he receives unemployment compensation $c$.

The process then repeats.

```{note}
We do not allow for job search while employed---this topic is taken up in a {doc}`later lecture <jv>`.
```

## Solving the Model

We drop time subscripts in what follows and primes denote next period values.

Let

* $v_e(w)$ be maximum lifetime value accruing to a worker who enters the current
  period *employed* with existing wage $w$
* $v_u(w)$ be maximum lifetime value accruing to a worker who who enters the
  current period *unemployed* and receives wage offer $w$.

Here **maximum lifetime value** means the value of {eq}`objective` when
the worker makes optimal decisions at all future points in time.

As we now show, these obtaining these functions is key to solving the new model.

### The Bellman Equations

We recall that, in {doc}`the original job search model <mccall_model>`, the
value function (the value of being unemployed with a given wage offer) satisfied
a Bellman equation.

Here this function again satisfies a Bellman equation that looks very similar.


```{math}
:label: bell2_mccall

    v_u(w) = \max 
        \left\{ 
            v_e(w), \,  
            u(c) + \beta \sum_{w' \in \mathbb W} v_u(w') q(w') 
        \right\}
```

The difference is that the value of accepting is $v_e(w)$ rather than
$w/(1-\beta)$.

We have to make this change because jobs are not permanent.

Accepting transitions the worker to employment and hence yields reward $v_e(w)$.

Rejecting leads to unemployment compensation and unemployment tomorrow.

Equation {eq}`bell2_mccall` expresses the value of being unemployed with offer
$w$ in hand as a maximum over the value of two options: accept or reject
the current offer.

The function $v_e$ also satisfies a Bellman equation:

```{math}
:label: bell1_mccall

    v_e(w) = u(w) + \beta
        \left[
            (1-\alpha)v_e(w) + \alpha \sum_{w' \in \mathbb W} v_u(w') q(w')
        \right]
```

```{note}
This equation differs from a traditional Bellman equation because there is no max.

There is no max because an employed agent has no choices.

Nonetheless, in keeping with most of the literature, we also refer to it as a
Bellman equation.

```

Equation {eq}`bell1_mccall` expresses the value of being employed at wage $w$ in terms of

* current reward $u(w)$ plus
* discounted expected reward tomorrow, given the $\alpha$ probability of being fired

As we will see, equations {eq}`bell1_mccall` and {eq}`bell2_mccall` provide
enough information to solve for both $v_e$ and $v_u$.

Once we have them in hand, we will be able to make optimal choices.

(ast_mcm)=
### A Simplifying Transformation

Rather than jumping straight into solving these equations, let's see if we can
simplify them somewhat.

(This process will be analogous to our {ref}`second pass <mm_op2>` at the plain vanilla
McCall model, where we reduced the Bellman equation to an equation in an unknown
scalar value, rather than an unknown vector.)

First, let

```{math}
:label: defh_mm

h := u(c) + \beta \sum_{w' \in \mathbb W} v_u(w') q(w')
```

be the continuation value associated with unemployment (the value of rejecting the current offer).

We can now write {eq}`bell2_mccall` as

$$
v_u(w) = \max \left\{ v_e(w), \,  h \right\}
$$

or, shifting time forward one period

$$
\sum_{w' \in \mathbb W} v_u(w') q(w')
 = \sum_{w' \in \mathbb W} \max \left\{ v_e(w'), \,  h \right\} q(w')
$$

Using {eq}`defh_mm` again now gives

```{math}
:label: bell02_mccall

h = u(c) + \beta \sum_{w' \in \mathbb W} \max \left\{ v_e(w'), \,  h \right\} q(w')
```

Finally, from {eq}`defh_mm` we have

$$
\sum_{w' \in \mathbb W} v_u(w') q(w') = \frac{h - u(c)}{\beta}
$$

so {eq}`bell1_mccall` can now be rewritten as

```{math}
:label: bell01_mccall

v_e(w) = u(w) + \beta
    \left[
        (1-\alpha)v_e(w) + \alpha \frac{h - u(c)}{\beta}
    \right]
```

### Simplifying to a Single Equation

We can simplify further by solving {eq}`bell01_mccall` for $v_e$ as a function of $h$.

Rearranging {eq}`bell01_mccall` gives

$$
v_e(w) = u(w) + \beta(1-\alpha)v_e(w) + \alpha(h - u(c))
$$

or

$$
v_e(w) - \beta(1-\alpha)v_e(w) = u(w) + \alpha(h - u(c))
$$

Solving for $v_e(w)$:

```{math}
:label: v_e_closed

v_e(w) = \frac{u(w) + \alpha(h - u(c))}{1 - \beta(1-\alpha)}
```

Substituting this into {eq}`bell02_mccall` yields

```{math}
:label: bell_scalar

h = u(c) + \beta \sum_{w' \in \mathbb W} \max \left\{ \frac{u(w') + \alpha(h - u(c))}{1 - \beta(1-\alpha)}, \,  h \right\} q(w')
```

This is a single scalar equation in $h$.

### The Reservation Wage

Suppose we can use {eq}`bell_scalar` to solve for $h$.

Once we have $h$, we can obtain $v_e$ from {eq}`v_e_closed`.

We can then determine optimal behavior for the worker.

From {eq}`bell2_mccall`, we see that an unemployed agent accepts current offer
$w$ if $v_e(w) \geq h$.

This means precisely that the value of accepting is higher than the value of rejecting.

It is clear that $v_e$ is (at least weakly) increasing in $w$, since the agent is never made worse off by a higher wage offer.

Hence, we can express the optimal choice as accepting wage offer $w$ if and only if

$$
w \geq \bar w
\quad \text{where} \quad
\bar w \text{ solves } v_e(\bar w) = h
$$

### Solving the Bellman Equations

We'll use the same iterative approach to solving the Bellman equations that we
adopted in the {doc}`first job search lecture <mccall_model>`.

Since we have reduced the problem to a single scalar equation {eq}`bell_scalar`,
we only need to iterate on $h$.

The iteration rule is

```{math}
:label: bell_iter

h_{n+1} = u(c) + \beta \sum_{w' \in \mathbb W}
    \max \left\{ \frac{u(w') + \alpha(h_n - u(c))}{1 - \beta(1-\alpha)}, \,  h_n \right\} q(w')
```

starting from some initial condition $h_0$.

Once convergence is achieved, we can compute $v_e$ from {eq}`v_e_closed`:

```{math}
:label: bell_v_e_final

v_e(w) = \frac{u(w) + \alpha(h - u(c))}{1 - \beta(1-\alpha)}
```

This approach is simpler than iterating on both $h$ and $v_e$ simultaneously, as
we now only need to track a single scalar value.

(Convergence can be established via the Banach contraction mapping theorem.)

## Implementation

Let's implement this iterative process.

In the code, you'll see that we use a class to store the various parameters and other
objects associated with a given model.

This helps to tidy up the code and provides an object that's easy to pass to functions.

The default utility function is a CRRA utility function

```{code-cell} ipython3
def u(c, σ=2.0):
    return (c**(1 - σ) - 1) / (1 - σ)
```

Also, here's a default wage distribution, based around the BetaBinomial
distribution:

```{code-cell} ipython3
n = 60                                  # n possible outcomes for w
w_default = jnp.linspace(10, 20, n)     # wages between 10 and 20
a, b = 600, 400                         # shape parameters
dist = BetaBinomial(n-1, a, b)          # distribution
q_default = jnp.array(dist.pdf())       # probabilities as a JAX array
```

Here's our model class for the McCall model with separation.

```{code-cell} ipython3
class Model(NamedTuple):
    α: float = 0.2              # job separation rate
    β: float = 0.98             # discount factor
    c: float = 6.0              # unemployment compensation
    w: jnp.ndarray = w_default  # wage outcome space
    q: jnp.ndarray = q_default  # probabilities over wage offers
```

Now we iterate until successive realizations are closer together than some small tolerance level.

We then return the current iterate as an approximate solution.

First, we define a function to compute $v_e$ from $h$:

```{code-cell} ipython3
def compute_v_e(model, h):
    " Compute v_e from h using the closed-form expression. "
    α, β, c, w = model.α, model.β, model.c, model.w
    return (u(w) + α * (h - u(c))) / (1 - β * (1 - α))
```

Now we implement the iteration on $h$ only:

```{code-cell} ipython3
def update_h(model, h):
    " One update of the scalar h. "
    α, β, c, w, q = model.α, model.β, model.c, model.w, model.q
    v_e = compute_v_e(model, h)
    h_new = u(c) + β * (jnp.maximum(v_e, h) @ q)
    return h_new

@jax.jit
def solve_model(model, tol=1e-5, max_iter=2000):
    " Iterates to convergence on the Bellman equations. "

    def cond_fun(state):
        h, i, error = state
        return jnp.logical_and(error > tol, i < max_iter)

    def body_fun(state):
        h, i, error = state
        h_new = update_h(model, h)
        error_new = jnp.abs(h_new - h)
        return h_new, i + 1, error_new

    # Initial state: (h, i, error)
    h_init = u(model.c) / (1 - model.β)
    i_init = 0
    error_init = tol + 1

    init_state = (h_init, i_init, error_init)
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    h_final, _, _ = final_state

    # Compute v_e from the converged h
    v_e_final = compute_v_e(model, h_final)

    return v_e_final, h_final
```

### The Reservation Wage: First Pass

The optimal choice of the agent is summarized by the reservation wage.

As discussed above, the reservation wage is the $\bar w$ that solves
$v_e(\bar w) = h$ where $h$ is the continuation value.

Let's compare $v_e$ and $h$ to see what they look like.

We'll use the default parameterizations found in the code above.

```{code-cell} ipython3
model = Model()
v_e, h = solve_model(model)

fig, ax = plt.subplots()
ax.plot(model.w, v_e, 'b-', lw=2, alpha=0.7, label='$v_e$')
ax.plot(model.w, [h] * len(model.w),
        'g-', lw=2, alpha=0.7, label='$h$')
ax.set_xlim(min(model.w), max(model.w))
ax.legend()
plt.show()
```

The value $v_e$ is increasing because higher $w$ generates a higher wage flow conditional on staying employed.

### The Reservation Wage: Computation

Here's a function `compute_reservation_wage` that takes an instance of `Model`
and returns the associated reservation wage.

```{code-cell} ipython3
@jax.jit
def compute_reservation_wage(model):
    """
    Computes the reservation wage of an instance of the McCall model
    by finding the smallest w such that v_e(w) >= h. If no such w exists, then
    w_bar is set to np.inf.
    """

    v_e, h = solve_model(model)
    i = jnp.searchsorted(v_e, h, side='left')
    w_bar = jnp.where(i >= len(model.w), jnp.inf, model.w[i])
    return w_bar
```

Next we will investigate how the reservation wage varies with parameters.

## Impact of Parameters

In each instance below, we'll show you a figure and then ask you to reproduce it in the exercises.

### The Reservation Wage and Unemployment Compensation

First, let's look at how $\bar w$ varies with unemployment compensation.

In the figure below, we use the default parameters in the `Model` class, apart from
c (which takes the values given on the horizontal axis)

```{figure} /_static/lecture_specific/mccall_model_with_separation/mccall_resw_c.png

```

As expected, higher unemployment compensation causes the worker to hold out for higher wages.

In effect, the cost of continuing job search is reduced.

### The Reservation Wage and Discounting

Next, let's investigate how $\bar w$ varies with the discount factor.

The next figure plots the reservation wage associated with different values of
$\beta$

```{figure} /_static/lecture_specific/mccall_model_with_separation/mccall_resw_beta.png

```

Again, the results are intuitive: More patient workers will hold out for higher wages.

### The Reservation Wage and Job Destruction

Finally, let's look at how $\bar w$ varies with the job separation rate $\alpha$.

Higher $\alpha$ translates to a greater chance that a worker will face termination in each period once employed.

```{figure} /_static/lecture_specific/mccall_model_with_separation/mccall_resw_alpha.png

```

Once more, the results are in line with our intuition.

If the separation rate is high, then the benefit of holding out for a higher wage falls.

Hence the reservation wage is lower.

## Exercises

```{exercise-start}
:label: mmws_ex1
```

Reproduce all the reservation wage figures shown above.

Regarding the values on the horizontal axis, use

```{code-cell} ipython3
grid_size = 25
c_vals = jnp.linspace(2, 12, grid_size)         # unemployment compensation
β_vals = jnp.linspace(0.8, 0.99, grid_size)     # discount factors
α_vals = jnp.linspace(0.05, 0.5, grid_size)     # separation rate
```

```{exercise-end}
```

```{solution-start} mmws_ex1
:class: dropdown
```

Here's the first figure.

```{code-cell} ipython3
def compute_res_wage_given_c(c):
    model = Model(c=c)
    w_bar = compute_reservation_wage(model)
    return w_bar

w_bar_vals = jax.vmap(compute_res_wage_given_c)(c_vals)

fig, ax = plt.subplots()
ax.set(xlabel='unemployment compensation', ylabel='reservation wage')
ax.plot(c_vals, w_bar_vals, label=r'$\bar w$ as a function of $c$')
ax.legend()
plt.show()
```

Here's the second one.

```{code-cell} ipython3
def compute_res_wage_given_beta(β):
    model = Model(β=β)
    w_bar = compute_reservation_wage(model)
    return w_bar

w_bar_vals = jax.vmap(compute_res_wage_given_beta)(β_vals)

fig, ax = plt.subplots()
ax.set(xlabel='discount factor', ylabel='reservation wage')
ax.plot(β_vals, w_bar_vals, label=r'$\bar w$ as a function of $\beta$')
ax.legend()
plt.show()
```

Here's the third.

```{code-cell} ipython3
def compute_res_wage_given_alpha(α):
    model = Model(α=α)
    w_bar = compute_reservation_wage(model)
    return w_bar

w_bar_vals = jax.vmap(compute_res_wage_given_alpha)(α_vals)

fig, ax = plt.subplots()
ax.set(xlabel='separation rate', ylabel='reservation wage')
ax.plot(α_vals, w_bar_vals, label=r'$\bar w$ as a function of $\alpha$')
ax.legend()
plt.show()
```

```{solution-end}
```
