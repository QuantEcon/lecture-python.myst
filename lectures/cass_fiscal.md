---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Fiscal Policy Experiments in a Non-stochastic Model

## Introduction

This lecture studies eﬀects of technology and fiscal shocks on equilibrium outcomes in a nonstochastic growth model. 

We use the model as a laboratory to exhibit numerical techniques for approximating equilibria and to display the structure of dynamic models in which decision makers have perfect foresight about future government decisions. 

Following {cite}`hall1971dynamic`, we augment a nonstochastic version of the standard growth model with a government that purchases a stream of goods and that finances itself with an array of distorting flat-rate taxes.

Distorting taxes prevent a competitive equilibrium allocation from solving a planning problem. 

Therefore, to compute an equilibrium allocation and price system, we solve a system of nonlinear diﬀerence equations consisting of the first-order conditions for decision makers and the other equilibrium conditions.

We present two ways to solve the model:

- The first method is called shooting algorithm;

- The second method is applying a root-finding algorithm to minimize the residuals derived from the first-order conditions.

We will use the following imports

```{code-cell} ipython3
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from collections import namedtuple
from mpmath import mp, mpf
from warnings import warn

# Set the precision
mp.dps = 40
mp.pretty = True
```

Note that we will use the `mpmath` library to perform high-precision arithmetic in the shooting algorithm in cases where the solution diverges due to numerical instability.

We will use the following parameters

```{code-cell} ipython3
# Create a namedtuple to store the model parameters
Model = namedtuple("Model", 
            ["β", "γ", "δ", "α", "A"])

def create_model(β=0.95, # discount factor
                 γ=2.0,  # relative risk aversion coefficient
                 δ=0.2,  # depreciation rate
                 α=0.33, # capital share
                 A=1.0   # TFP
                 ):
    """Create a model instance."""
    return Model(β=β, γ=γ, δ=δ, α=α, A=A)

model = create_model()

# Total number of periods
S = 100
```

## The Economy

### Households

The representative household has preferences over nonnegative streams of a single consumption good $c_t$ and leisure $1-n_t$ that are ordered by:

$$
\sum_{t=0}^{\infty} \beta^t U(c_t, 1-n_t), \quad \beta \in (0, 1)
$$ (eq:utility)

where

- $U$ is twice continuously differentiable, and strictly concave with $c_t \geq 0$. In this lecture, we focus on the special case.

under the budget constraint

$$
\begin{aligned}
    \sum_{t=0}^\infty& q_t \left\{ (1 + \tau_{ct})c_t + \underbrace{[k_{t+1} - (1 - \delta)k_t]}_{\text{no tax when investing}} \right\} \\
    &\leq \sum_{t=0}^\infty q_t \left\{ \eta_t k_t - \underbrace{\tau_{kt}(\eta_t - \delta)k_t}_{\text{tax on rental return}} + (1 - \tau_{nt})w_t n_t - \tau_{ht} \right\}.
\end{aligned}
$$ (eq:house_budget)

### Technology

The economy's production technology is defined by: 

$$
g_t + c_t + x_t \leq F(k_t, n_t),
$$ (eq:tech_capital)

where 

- $g_t$ is government expenditure
- $x_t$ is gross investment, and 
- $F(k_t, n_t)$ is a linearly homogeneous production function with positive and decreasing marginal products of capital $k_t$ and labor $n_t$.

The law of motion for capital is given by:

$$
k_{t+1} = (1 - \delta)k_t + x_t,
$$

where 

- $\delta \in (0, 1)$ is depreciation rate.


### Price System

A price system is a triple of sequences $\{q_t, \eta_t, w_t\}_{t=0}^\infty$, where

- $q_t$ is the time $0$ pretax price of one unit of investment or consumption at time $t$ ($x_t$ or $c_t$),
- $\eta_t$ is the pretax price at time $t$ that the household receives from the firm for renting capital at time $t$,
- $w_t$ is the pretax price at time $t$ that the household receives for renting labor to the firm at time $t$.

The prices $w_t$ and $\eta_t$ are expressed in terms of time $t$ goods, while $q_t$ is expressed in terms of the numeraire at time 0.

### Government

Government plans $\{ g_t \}_{t=0}^\infty$ for government purchases and taxes $\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$ subject to the budget constraint

$$
\sum_{t=0}^\infty q_t g_t \leq \sum_{t=0}^\infty q_t \left\{ \tau_{ct}c_t + \tau_{kt}(\eta_t - \delta)k_t + \tau_{nt}w_t n_t + \tau_{ht} \right\}.
$$ (eq:gov_budget)

### Firm

A representative firm chooses $\{k_t, n_t\}_{t=0}^\infty$ to maximize their present value of profit:

$$
\sum_{t=0}^\infty q_t \left[ F(k_t, n_t) - w_t n_t - \eta_t k_t \right],
$$

Euler's theorem for linearly homogeneous functions states that if a function $F(k, n)$ is linearly homogeneous of degree 1, then:

$$
F(k, n) = F_k k + F_n n,
$$

where $F_k = \frac{\partial F(k, n)}{\partial k}$ and $F_n = \frac{\partial F(k, n)}{\partial n}$.


### Equilibrium

In the equilibrium, given a budget-feasible government policy $\{g_t\}_{t=0}^\infty$ and $\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$ subject to {eq}`eq:gov_budget`,

- *Household* chooses $\{c_t\}_{t=0}^\infty$, $\{n_t\}_{t=0}^\infty$, and $\{k_{t+1}\}_{t=0}^\infty$ to maximize utility{eq}`eq:utility` subject to budget constraint{eq}`eq:house_budget`, and 
- *Frim* chooses sequences of capital $\{k_t\}_{t=0}^\infty$ and $\{n_t\}_{t=0}^\infty$ to maximize profits

    $$
         \sum_{t=0}^\infty q_t [F(k_t, n_t) - \eta_t k_t - w_t n_t]
    $$ (eq:firm_profit)
  
- A **feasible allocation** is a sequence $\{c_t, x_t, n_t, k_t\}_{t=0}^\infty$ that satisfies feasibility condition {eq}`eq:tech_capital`.


```{prf:definition}
:label: com_eq_tax

A **competitive equilibrium with distorting taxes** is a **budget-feasible government policy**, **a feasible allocation**, and **a price system** for which, given the price system and the government policy, the allocation solves the household’s problem and the ﬁrm’s problem.
```

## Non-arbitrage Condition

By rearranging {eq}`eq:house_budget` and group $k_t$ at the same $t$, we can get

$$
    \begin{aligned}
    \sum_{t=0}^\infty q_t \left[(1 + \tau_{ct})c_t \right] &\leq \sum_{t=0}^\infty q_t(1 - \tau_{nt})w_t n_t - \sum_{t=0}^\infty q_t \tau_{ht} \\
    &+ \sum_{t=1}^\infty\left\{ \left[(1 - \tau_{kt})(\eta_t - \delta) + 1\right]q_t - q_{t-1}\right\}k_t \\
    &+ \left[(1 - \tau_{k0})(\eta_0 - \delta) + 1\right]q_0k_0 - \lim_{T \to \infty} q_T k_{T+1}
    \end{aligned}
$$ (eq:constrant_house)

By setting the terms multiplying $k_t$ to $0$ we have the non-arbitrage condition:

$$
\frac{q_t}{q_{t+1}} = \left[(1 - \tau_{kt})(\eta_t - \delta) + 1\right]
$$ (eq:no_arb)

Moreover, we have terminal condition

$$
-\lim_{T \to \infty} q_T k_{T+1} = 0.
$$ (eq:terminal)

Moreover, applying Euler's theorem on firm's present value gives

$$
\sum_{t=0}^\infty q_t \left[ F(k_t, n_t) - w_t n_t - \eta_t k_t \right] = \sum_{t=0}^\infty q_t \left[ (F_{kt} - \eta_t) k_t + (F_{nt} - w_t) n_t \right]
$$

and no-arbitrage analogous to the household case are

$$
\eta_t = F_{kt} \quad \text{and} \quad w_t = F_{nt}.
$$ (eq:no_arb_firms)

## Household's First Order Condition

Household maximize {eq}`eq:utility` under {eq}`eq:house_budget`. Let $U_1 = \frac{\partial U}{\partial c}, U_2 = \frac{\partial U}{\partial (1-n)} = -\frac{\partial U}{\partial n} = -U_n.$, we can derive FOC from the Lagrangian

$$
\mathcal{L} = \sum_{t=0}^\infty \beta^t U(c_t, 1 - n_t) + \mu \left( \sum_{t=0}^\infty q_t \left[(1 + \tau_{ct})c_t - (1 - \tau_{nt})w_t n_t + \ldots \right] \right),
$$

Hence we have FOC:

$$
\frac{\partial \mathcal{L}}{\partial c_t} = \beta^t U_1(c_t, 1 - n_t) - \mu q_t (1 + \tau_{ct}) = 0
$$ (eq:foc_c_1)

which gives $\mu q_t = \beta^t \frac{U_1(c_t, 1 - n_t)}{(1 + \tau_{ct})}$

and 

$$
\frac{\partial \mathcal{L}}{\partial n_t} = \beta^t \left(-U_2(c_t, 1 - n_t)\right) - \mu q_t (1 - \tau_{nt}) w_t = 0
$$ (eq:foc_n_1)

Rearranguing {eq}`eq:foc_c_1` and {eq}`eq:foc_n_1`, we have

$$
\begin{aligned}
\beta^t U_1(c_t, 1 - n_t)  = \beta^t U_{1t} = \mu q_t (1 + \tau_{ct}),
\end{aligned}
$$ (eq:foc_c)

$$
\begin{aligned}
\beta^t U_2(c_t, 1 - n_t) = \beta^t U_{2t} = \mu q_t (1 - \tau_{nt}) w_t.
\end{aligned}
$$ (eq:foc_n)


Plugging {eq}`eq:foc_c` into {eq}`eq:terminal` by replacing $q_t$, we get terminal condition

$$
-\lim_{T \to \infty} \beta^T \frac{U_{1T}}{(1 + \tau_{cT})} k_{T+1} = 0.
$$ (eq:terminal_final)

## Computing Equilibria

To compute an equilibrium we solve a price system $\{q_t, \eta_t, w_t\}$, a budget feasible government policy $\{g_t, \tau_t\} \equiv \{g_t, \tau_{ct}, \tau_{nt}, \tau_{kt}, \tau_{ht}\}$, and an allocation $\{c_t, n_t, k_{t+1}\}$ that solve the system of nonlinear difference equations consisting of 

- feasibility condition {eq}`eq:tech_capital`, no-arbitrage condition for household {eq}`eq:no_arb` and firms {eq}`eq:no_arb_firms`, household's first order conditions {eq}`eq:foc_c` and {eq}`eq:foc_n`.
- initial condition $k_0$, and terminal condition {eq}`eq:terminal_final`.

### Inelastic Labor Supply

In this lecture, we consider the special case where $U(c, 1-n) = u(c)$ and $f(k) := F(k, 1)$.

We rewrite {eq}`eq:tech_capital` with $f(k) := F(k, 1)$,

$$
k_{t+1} = f(k_t) + (1 - \delta) k_t - g_t - c_t.
$$ (eq:feasi_capital)

```{code-cell} ipython3
def next_k(k_t, g_t, c_t, model):
    """
    Capital next period: k_{t+1} = f(k_t) + (1 - δ) * k_t - c_t - g_t
    """
    return f(k_t, model) + (1 - model.δ) * k_t - g_t - c_t
```

By the properties of a linearly homogeneous production function, we have $F_k(k, n) = f'(k)$ and $F_n(k, 1) = f(k, 1) - f'(k)k$.

Substituting {eq}`eq:foc_c`, {eq}`eq:no_arb_firms`, and {eq}`eq:feasi_capital` into {eq}`eq:no_arb`, we obtain:

$$
\begin{aligned}
&\frac{u'(f(k_t) + (1 - \delta) k_t - g_t - k_{t+1})}{(1 + \tau_{ct})} \\
&- \beta \frac{u'(f(k_{t+1}) + (1 - \delta) k_{t+1} - g_{t+1} - k_{t+2})}{(1 + \tau_{ct+1})} \\
&\times [(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1] = 0.
\end{aligned}
$$

This can be simplified to:

$$
\begin{aligned}
u'(c_t) = \beta u'(c_{t+1}) \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} [(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1].
\end{aligned}
$$ (eq:diff_second)

which represents the Euler equation for the household.

This equation is instrumental in solving for the equilibrium sequence of consumption and capital, as demonstrated in the second method.

### Steady state

Tax rates and government expenditures act as forcing functions for the difference equations {eq}`eq:feasi_capital` and {eq}`eq:diff_second`.

Define $z_t = [g_t, \tau_{kt}, \tau_{ct}]'$. We can express the second-order difference equation as:

$$
H(k_t, k_{t+1}, k_{t+2}; z_t, z_{t+1}) = 0.
$$ (eq:second_ord_diff)

We assume that the government policy reaches a steady state such that $\lim_{t \to \infty} z_t = \bar z$ and that the steady state holds for $t > T$. The terminal steady-state capital stock $\bar{k}$ satisfies:

$$
H(\bar{k}, \bar{k}, \bar{k}, \bar{z}, \bar{z}) = 0.
$$

From the difference equation {eq}`eq:diff_second`, we can derive the steady-state condition:

$$
\begin{aligned}
u'(\bar{c}) &= \beta u'(\bar{c}) \frac{(1 + \bar{\tau}_{c})}{(1 + \bar{\tau}_{c})} [(1 - \bar{\tau}_{k})(f'(\bar{k}) - \delta) + 1]. \\
&\implies 1 = \beta[(1 - \bar{\tau}_{k})(f'(\bar{k}) - \delta) + 1].
\end{aligned}
$$ (eq:diff_second_steady)

### Other equilibrium quantities and prices

*Price:*

$$
q_t = \frac{\beta^t u'(c_t)}{u'(c_0)}
$$ (eq:equil_q)

```{code-cell} ipython3
def compute_q_path(c_path, model, S=100):
    """
    Compute q path: q_t = (β^t * u'(c_t)) / u'(c_0)
    """
    q_path = np.zeros_like(c_path)
    for t in range(S):
        q_path[t] = (model.β ** t * 
                     u_prime(c_path[t], model)) / u_prime(c_path[0], model)
    return q_path
```

*Capital rental rate*

$$
\eta_t = f'(k_t)  
$$

```{code-cell} ipython3
def compute_η_path(k_path, model, S=100):
    """
    Compute η path: η_t = f'(k_t)
    """
    η_path = np.zeros_like(k_path)
    for t in range(S):
        η_path[t] = f_prime(k_path[t], model)
    return η_path
```

*Labor rental rate:*

$$
w_t = f(k_t) - k_t f'(k_t)    
$$

```{code-cell} ipython3
def compute_w_path(k_path, η_path, model, S=100):
    """
    Compute w path: w_t = f(k_t) - k_t * f'(k_t)
    """
    A, α = model.A, model.α, model.δ
    w_path = np.zeros_like(k_path)
    for t in range(S):
        w_path[t] = f(k_path[t], model) - k_path[t] * η_path[t]
    return w_path
```

*Gross one-period return on capital:*

$$
\bar{R}_{t+1} = \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1\right] =  \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} R_{t, t+1}
$$ (eq:gross_rate)

```{code-cell} ipython3
def compute_R_bar(τ_ct, τ_ctp1, τ_ktp1, k_tp1, model):
    """
    Gross one-period return on capital:
    R̄ = [(1 + τ_c_t) / (1 + τ_c_{t+1})] 
        * { [1 - τ_k_{t+1}] * [f'(k_{t+1}) - δ] + 1 }
    """
    A, α, δ = model.A, model.α, model.δ
    return  ((1 + τ_ct) / (1 + τ_ctp1)) * (
        (1 - τ_ktp1) * (f_prime(k_tp1, model) - δ) + 1) 

def compute_R_bar_path(shocks, k_path, model, S=100):
    """
    Compute R̄ path over time.
    """
    A, α, δ = model.A, model.α, model.δ
    R_bar_path = np.zeros(S + 1)
    for t in range(S):
        R_bar_path[t] = compute_R_bar(
            shocks['τ_c'][t], shocks['τ_c'][t + 1], shocks['τ_k'][t + 1],
            k_path[t + 1], model)
    R_bar_path[S] = R_bar_path[S - 1]
    return R_bar_path
```

*One-period discount factor:*

$$
R^{-1}_{t, t+1} = \frac{q_t}{q_{t-1}} = m_{t, t+1} = \beta \frac{u'(c_{t+1})}{u'(c_t)} \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})}
$$ (eq:equil_R)


*Net one-period rate of interest:*

$$
r_{t, t+1} \equiv R_{t, t+1} - 1 = (1 - \tau_{k, t+1})(f'(k_{t+1}) - \delta)
$$ (eq:equil_r)

By {eq}`eq:equil_R`, we have

$$
R_{t, t+s} = e^{s \cdot r_{t, t+s}}.
$$

Then by {eq}`eq:equil_r`, we have 

$$
\frac{q_{t+s}}{q_t} = e^{-s \cdot r_{t, t+s}}.
$$

Rearranging the above equation, we have

$$
r_{t, t+s} = -\frac{1}{s} \ln\left(\frac{q_{t+s}}{q_t}\right).
$$

```{code-cell} ipython3
def compute_rts_path(q_path, S, t):
    """
    Compute r path:
    r_t,t+s = - (1/s) * ln(q_{t+s} / q_t)
    """
    s = np.arange(1, S + 1) 
    q_path = np.array([float(q) for q in q_path]) 
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rts_path = - np.log(q_path[t + s] / q_path[t]) / s
    return rts_path
```

## Specifications of the model

In our model, the representative household has the following CRRA preferences over consumption: 

$$
u(c) = \frac{c^{1 - \gamma}}{1 - \gamma}
$$

```{code-cell} ipython3
def u_prime(c, model):
    """
    Marginal utility: u'(c) = c^{-γ}
    """
    return c ** (-model.γ)
```

By substituting {eq}`eq:gross_rate` into {eq}`eq:diff_second`, we have

$$
c_{t+1} = c_t \left[ \beta \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{k, t+1})(f'(k_{t+1}) - \delta) + 1 \right] \right]^{\frac{1}{\gamma}} = c_t \left[ \beta \overline{R}_{t+1} \right]^{\frac{1}{\gamma}}
$$ (eq:consume_R)

```{code-cell} ipython3
def next_c(c_t, R_bar, model):
    """
    Consumption next period: c_{t+1} = c_t * (β * R̄)^{1/γ}
    """
    β, γ = model.β, model.γ
    return c_t * (β * R_bar) ** (1 / γ)
```

The production function is given by the Cobb-Douglas form with inelastic labor supply:

$$
F(k, 1) = A k^\alpha
$$

```{code-cell} ipython3
def f(k, model): 
    """
    Production function: f(k) = A * k^{α}
    """

    A, α = model.A, model.α
    return A * k ** α

def f_prime(k, model):
    """
    Marginal product of capital: f'(k) = α * A * k^{α - 1}
    """
    A, α = model.A, model.α
    return α * A * k ** (α - 1)
```

## Computation

In the following sections, we will apply two methods to solve the model: the shooting algorithm and residual minimization using the Euler equation ({eq}`eq:diff_second`) and feasibility condition ({eq}`eq:feasi_capital`).

### Method 1: Shooting Algorithm

1. Solve the equation {eq}`eq:diff_second_steady` for the terminal steady-state capital $\bar{k}$ that corresponds to the permanent policy vector $\bar{z}$.

2. Select a large time index $S \gg T$, guess an initial consumption rate $c_0$, and use the equation {eq}`eq:feasi_capital` to solve for $k_1$.

3. Use the equation {eq}`eq:consume_R` to determine $c_{t+1}$. Then, apply the equation {eq}`eq:feasi_capital` to compute $k_{t+2}$.

4. Iterate step 3 to compute candidate values $\hat{k}_t$ for $t = 1, \dots, S$.

5. Compute the difference $\hat{k}_S - \bar{k}$. If $\left| \hat{k}_S - \bar{k} \right| > \epsilon$ for some small $\epsilon$, adjust $c_0$ and repeat steps 2-5.

6. Adjust $c_0$ iteratively using the bisection method to find a value that ensures $\left| \hat{k}_S - \bar{k} \right| < \epsilon$.

```{code-cell} ipython3
# Steady-state calculation
def steady_states(model, g_ss, τ_k_ss=0.0):
    """
    Steady-state values:
    - Capital: (1 - τ_k_ss) * [α * A * k_ss^{α - 1} - δ] = (1 / β) - 1
    - Consumption: c_ss = A * k_ss^{α} - δ * k_ss - g_ss
    """
    β, δ, α, A = model.β, model.δ, model.α, model.A
    numerator = δ + (1 / β - 1) / (1 - τ_k_ss)
    denominator = α * A
    k_ss = (numerator / denominator) ** (1 / (α - 1))
    c_ss = A * k_ss ** α - δ * k_ss - g_ss
    return k_ss, c_ss

def shooting_algorithm(c0, k0, shocks, S, model):
    """
    Shooting algorithm for given initial c0 and k0.
    """
    # Convert shocks to high-precision
    g_path, τ_c_path, τ_k_path = (
        list(map(mpf, shocks[key])) for key in ['g', 'τ_c', 'τ_k']
    )

    # Initialize paths with initial values
    c_path = [mpf(c0)] + [mpf(0)] * S
    k_path = [mpf(k0)] + [mpf(0)] * S

    # Generate paths for k_t and c_t
    for t in range(S):
        k_t, c_t, g_t = k_path[t], c_path[t], g_path[t]

        # Calculate next period's capital
        k_tp1 = next_k(k_t, g_t, c_t, model)
        # Failure due to negative capital
        if k_tp1 < mpf(0):
            return None, None 
        k_path[t + 1] = k_tp1

        # Calculate next period's consumption
        R_bar = compute_R_bar(τ_c_path[t], τ_c_path[t + 1], 
                              τ_k_path[t + 1], k_tp1, model)
        c_tp1 = next_c(c_t, R_bar, model)
        # Failure due to negative consumption
        if c_tp1 < mpf(0):
            return None, None
        c_path[t + 1] = c_tp1

    return k_path, c_path


def bisection_c0(c0_guess, k0, shocks, S, model, 
                 tol=mpf('1e-6'), max_iter=1000, verbose=False):
    """
    Bisection method to find optimal initial consumption c0.
    """
    k_ss_final, _ = steady_states(model, 
                                  mpf(shocks['g'][-1]), 
                                  mpf(shocks['τ_k'][-1]))
    c0_lower, c0_upper = mpf(0), f(k_ss_final, model)

    c0 = c0_guess
    for iter_count in range(max_iter):
        k_path, _ = shooting_algorithm(c0, k0, shocks, S, model)
        if k_path is None:
            if verbose:
                print(f"Iteration {iter_count + 1}: shooting failed with c0 = {c0}")
            # Adjust upper bound when shooting fails
            c0_upper = c0
        else:
            error = k_path[-1] - k_ss_final
            if verbose and iter_count % 100 == 0:
                print(f"Iteration {iter_count + 1}: c0 = {c0}, error = {error}")

            # Check for convergence
            if abs(error) < tol:
                # Converged successfully
                print(f"Converged successfully on iteration {iter_count + 1}")
                return c0 

            # Update bounds based on the error
            if error > mpf(0):
                c0_lower = c0
            else:
                c0_upper = c0

        # Calculate the new midpoint for bisection
        c0 = (c0_lower + c0_upper) / mpf('2')

    # Return the last computed c0 if convergence was not achieved
    # Send a Warning message when this happens
    warn(f"Converged failed. Returning the last c0 = {c0}", stacklevel=2)
    return c0

def run_shooting(shocks, S, model, c0_func=bisection_c0, shooting_func=shooting_algorithm):
    """
    Runs the shooting algorithm.
    """
    # Compute initial steady states
    k0, c0 = steady_states(model, mpf(shocks['g'][0]), mpf(shocks['τ_k'][0]))
    
    # Find the optimal initial consumption
    optimal_c0 = c0_func(c0, k0, shocks, S, model)
    print(f"Parameters: {model}")
    print(f"Optimal initial consumption c0: {mp.nstr(optimal_c0, 7)} \n")
    
    # Simulate the model
    k_path, c_path = shooting_func(optimal_c0, k0, shocks, S, model)
    
    # Combine and return the results
    return np.column_stack([k_path, c_path])
```

### Experiments

We will run a series of experiments and analyze the transition path for the equilibrium in each scenario:

1. A foreseen once-and-for-all increase in $g$ from 0.2 to 0.4 occurring in period 10.
2. A foreseen once-and-for-all increase in $\tau_c$ from 0.0 to 0.2 occurring in period 10.
3. A foreseen once-and-for-all increase in $\tau_k$ from 0.0 to 0.2 occurring in period 10.
4. A foreseen one-time increase in $g$ from 0.2 to 0.4 in period 10, after which $g$ reverts to 0.2 permanently.

+++

Next we prepare the sequence of variables that will be used to initialize the simulation. 

We will start from the steady state and then apply the shocks at the appropriate time.

```{code-cell} ipython3
def plot_results(solution, k_ss, c_ss, shocks, shock_param, 
                 axes, model, label='', linestyle='-', T=40):
    
    k_path = solution[:, 0]
    c_path = solution[:, 1]

    axes[0].plot(k_path[:T], linestyle=linestyle, label=label)
    axes[0].axhline(k_ss, linestyle='--', color='black')
    axes[0].set_title('k')

    # Plot for c
    axes[1].plot(c_path[:T], linestyle=linestyle, label=label)
    axes[1].axhline(c_ss, linestyle='--', color='black')
    axes[1].set_title('c')

    # Plot for g
    R_bar_path = compute_R_bar_path(shocks, k_path, model, S)

    axes[2].plot(R_bar_path[:T], linestyle=linestyle, label=label)
    axes[2].set_title('$\overline{R}$')
    axes[2].axhline(1 / model.β, linestyle='--', color='black')
    
    η_path = compute_η_path(k_path, model, S=T)
    η_ss = model.α * model.A * k_ss ** (model.α - 1)
    
    axes[3].plot(η_path[:T], linestyle=linestyle, label=label)
    axes[3].axhline(η_ss, linestyle='--', color='black')
    axes[3].set_title(r'$\eta$')
    
    axes[4].plot(shocks[shock_param][:T], linestyle=linestyle, label=label)
    axes[4].axhline(shocks[shock_param][0], linestyle='--', color='black')
    axes[4].set_title(rf'${shock_param}$')
```

**Experiment 1: Foreseen once-and-for-all increase in $g$ from 0.2 to 0.4 in period 10**

The experiment replicates the Figure 12.9.1 in RMT5 under $\gamma = 2$.

```{code-cell} ipython3
# Define shocks as a dictionary
shocks = {
    'g': np.concatenate((np.repeat(0.2, 10), np.repeat(0.4, S - 9))),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

k_ss_initial, c_ss_initial = steady_states(model, 
                                           shocks['g'][0], 
                                           shocks['τ_k'][0])

print(f"Steady-state capital: {k_ss_initial:.4f}")
print(f"Steady-state consumption: {c_ss_initial:.4f}")

solution = run_shooting(shocks, S, model)

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

plot_results(solution, k_ss_initial, 
             c_ss_initial, shocks, 'g', axes, model, T=40)

for ax in axes[5:]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()
```

Let's write the procedures above into a function that runs the solver and draw the plots for a given model

```{code-cell} ipython3
:tags: [hide-input]

def experiment_model(shocks, S, model, solver, plot_func, policy_shock, T=40):
    """
    Run the shooting algorithm given a model and plot the results.
    """

    k0, c0 = steady_states(model, shocks['g'][0], shocks['τ_k'][0])
    
    print(f"Steady-state capital: {k0:.4f}")
    print(f"Steady-state consumption: {c0:.4f}")
    print('-'*64)
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()

    solution = solver(shocks, S, model)
    plot_func(solution, k0, c0, 
              shocks, policy_shock, axes, model, T=T)

    for ax in axes[5:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
```

The experiment replicates the Figure 12.9.2 in RMT5 under $\gamma = 2$ and $\gamma = 0.2$.

```{code-cell} ipython3
# Solve the model using shooting
solution = run_shooting(shocks, S, model)

# Compute the initial steady states
k_ss_initial, c_ss_initial = steady_states(model, 
                                           shocks['g'][0], 
                                           shocks['τ_k'][0])

# Plot the solution for γ=2
fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

label = fr"$\gamma = {model.γ}$"
plot_results(solution, k_ss_initial, c_ss_initial, 
             shocks, 'g', axes, model, label=label, 
             T=40)

# Solve and plot the result for γ=0.2
model_γ2 = create_model(γ=0.2)
solution = run_shooting(shocks, S, model_γ2)

plot_results(solution, k_ss_initial, c_ss_initial, 
             shocks, 'g', axes, model_γ2, 
             label=fr"$\gamma = {model_γ2.γ}$", 
             linestyle='-.', T=40)

handles, labels = axes[0].get_legend_handles_labels()  
fig.legend(handles, labels, loc='lower right', 
           ncol=3, fontsize=14, bbox_to_anchor=(1, 0.1))  

for ax in axes[5:]:
    fig.delaxes(ax)
    
plt.tight_layout()
plt.show()
```

Let's write another function that runs the solver and draw the plots for two models as we did above

```{code-cell} ipython3
:tags: [hide-input]

def experiment_two_models(shocks, S, model_1, model_2, solver, plot_func, 
                          policy_shock, legend_label_fun=None, T=40):
    """
    Compares and plots results of the shooting algorithm for two models.
    """
    
    k0, c0 = steady_states(model, shocks['g'][0], shocks['τ_k'][0])
    print(f"Steady-state capital: {k0:.4f}")
    print(f"Steady-state consumption: {c0:.4f}")
    print('-'*64)
    
    # Use a default legend labeling function if none is provided
    if legend_label_fun is None:
        legend_label_fun = lambda model: fr"$\gamma = {model.γ}$"

    # Set up the figure and axes
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()

    # Function to run and plot for each model
    def run_and_plot(model, linestyle='-'):
        solution = solver(shocks, S, model)
        plot_func(solution, k0, c0, shocks, policy_shock, axes, model, 
                  label=legend_label_fun(model), linestyle=linestyle, T=T)

    # Plot for both models
    run_and_plot(model_1)
    run_and_plot(model_2, linestyle='-.')

    # Set legend using labels from the first axis
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', ncol=3, 
               fontsize=14, bbox_to_anchor=(1, 0.1))

    # Remove extra axes and tidy up the layout
    for ax in axes[5:]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    plt.show()
```

Now we plot other equilibrium quantities:

```{code-cell} ipython3
def plot_prices(solution, c_ss, shock_param, axes,
                model, label='', linestyle='-', T=40):
    
    α, β, δ, γ, A = model.α, model.β, model.δ, model.γ, model.A
    
    k_path = solution[:, 0]
    c_path = solution[:, 1]

    # Plot for c
    axes[0].plot(c_path[:T], linestyle=linestyle, label=label)
    axes[0].axhline(c_ss, linestyle='--', color='black')
    axes[0].set_title('c')
    
    # Plot for q
    q_path = compute_q_path(c_path, model, S=S)
    axes[1].plot(q_path[:T], linestyle=linestyle, label=label)
    axes[1].plot(β**np.arange(T), linestyle='--', color='black')
    axes[1].set_title('q')
    
    # Plot for r_{t,t+1}
    R_bar_path = compute_R_bar_path(shocks, k_path, model, S)
    axes[2].plot(R_bar_path[:T] - 1, linestyle=linestyle, label=label)
    axes[2].axhline(1 / β - 1, linestyle='--', color='black')
    axes[2].set_title('$r_{t,t+1}$')

    # Plot for r_{t,t+s}
    for style, s in zip(['-', '-.', '--'], [0, 10, 60]):
        rts_path = compute_rts_path(q_path, T, s)
        axes[3].plot(rts_path, linestyle=style, 
                     color='black' if style == '--' else None,
                     label=f'$t={s}$')
        axes[3].set_xlabel('s')
        axes[3].set_title('$r_{t,t+s}$')

    # Plot for g
    axes[4].plot(shocks[shock_param][:T], linestyle=linestyle, label=label)
    axes[4].axhline(shocks[shock_param][0], linestyle='--', color='black')
    axes[4].set_title(shock_param)
```

```{code-cell} ipython3
solution = run_shooting(shocks, S, model)

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

plot_prices(solution, c_ss_initial, 'g', axes, model, T=40)

for ax in axes[5:]:
    fig.delaxes(ax)

handles, labels = axes[3].get_legend_handles_labels()  
fig.legend(handles, labels, title=r"$r_{t,t+s}$ with ", loc='lower right', ncol=3, fontsize=10, bbox_to_anchor=(1, 0.1))  
plt.tight_layout()
plt.show()
```

**Experiment 2: Foreseen once-and-for-all increase in $\tau_c$ from 0.0 to 0.2 in period 10**

The experiment replicates the Figure 12.9.4.

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, run_shooting, plot_results, 'τ_c')
```

**Experiment 3: Foreseen once-and-for-all increase in $\tau_k$ from 0.0 to 0.2 in period 10**

The experiment replicates the Figure 12.9.5.

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))) 
}

experiment_two_models(shocks, S, model, model_γ2, 
                run_shooting, plot_results, 'τ_k')
```

**Experiment 4: Foreseen one-time increase in $g$ from 0.2 to 0.4 in period 10, after which $g$ returns to 0.2 forever**

The experiment replicates the Figure 12.9.6.

```{code-cell} ipython3
g_path = np.repeat(0.2, S + 1)
g_path[10] = 0.4

shocks = {
    'g': g_path,
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, run_shooting, plot_results, 'g')
```

### Method 2: Residual Minimization Using the Euler Equation and Feasibility Condition

The second method involves minimizing the residuals of the following equations:

- *The Euler equation* {eq}`eq:diff_second`:
  $$
  1 = \beta \left(\frac{c_{t+1}}{c_t}\right)^{-\gamma} \frac{(1+\tau_{ct})}{(1+\tau_{ct+1})} \left[(1 - \tau_{kt+1})(\alpha A k_{t+1}^{\alpha-1} - \delta) + 1 \right]
  $$

- *The feasibility condition* {eq}`eq:feasi_capital`:
  $$
  k_{t+1} = A k_{t}^{\alpha} + (1 - \delta) k_t - g_t - c_t.
  $$

+++

The algorithm is described as follows:

1. *Calculate the initial state $k_0$*:
   - Derive $k_0$ based on the given initial government plan $z_0$.

2. *Initialize a sequence of initial guesses* $\{\hat{c}_t, \hat{k}_t\}_{t=0}^{S}$.

3. *Compute the residuals* $l_a$ and $l_k$ for $t = 0, \dots, S$, as well as $l_{k_0}$ for $t = 0$ and $l_{k_S}$ for $t = S$:
   - Compute the *Euler's equation* residual for $t = 0, \dots, S$ using {eq}`eq:diff_second`:

     $$
     l_{ta} = \beta u'(c_{t+1}) \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1 \right] - 1
     $$

   - Compute the *feasibility condition* residual for $t = 1, \dots, S-1$ using {eq}`eq:feasi_capital`:

     $$
     l_{tk} = k_{t+1} - f(k_t) - (1 - \delta)k_t + g_t + c_t
     $$

   - Compute the residual for the *initial condition for $k_0$* using {eq}`eq:diff_second_steady` and the initial capital $k_0$:

     $$
     l_{k_0} = 1 - \beta \left[ (1 - \tau_{k0}) \left(f'(k_0) - \delta \right) + 1 \right]
     $$

   - Compute the residual for the *terminal condition* for $t = S$ using {eq}`eq:diff_second` under the assumptions $c_t = c_{t+1} = c_S$, $k_t = k_{t+1} = k_S$, $\tau_{ct} = \tau_{ct+1} = \tau_{cS}$, and $\tau_{kt} = \tau_{kt+1} = \tau_{kS}$:
     
     $$
     l_{k_S} = \beta u'(c_S) \frac{(1 + \tau_{cS})}{(1 + \tau_{cS})} \left[(1 - \tau_{kS})(f'(k_S) - \delta) + 1 \right] - 1
     $$

4. *Residual Minimization*:
   - Adjust the guesses for $\{\hat{c}_t, \hat{k}_t\}_{t=0}^{S}$ to minimize the residuals $l_{k_0}$, $l_{ta}$, $l_{tk}$, and $l_{k_S}$ for $t = 0, \dots, S$.

```{code-cell} ipython3
# Euler's equation and feasibility condition 
def euler_residual(c_t, c_tp1, τ_c_t, τ_c_tp1, τ_k_tp1, k_tp1, model):
    """
    Computes the residuals for Euler's equation.
    """
    β, γ, δ, α, A = model.β, model.γ, model.δ, model.α, model.A
    η_tp1 = α * A * k_tp1 ** (α - 1)
    return β * (c_tp1 / c_t) ** (-γ) * (1 + τ_c_t) / (1 + τ_c_tp1) * (
        (1 - τ_k_tp1) * (η_tp1 - δ) + 1) - 1

def feasi_residual(k_t, k_tm1, c_tm1, g_t, model):
    """
    Computes the residuals for feasibility condition.
    """
    α, A, δ = model.α, model.A, model.δ
    return k_t - (A * k_tm1 ** α + (1 - δ) * k_tm1 - c_tm1 - g_t)

# Computing residuals as objective function to minimize
def compute_residuals(vars_flat, k_init, S, shocks, model):
    """
    Compute a vector of residuals under Euler's equation, feasibility condition, 
    and boundary conditions.
    """
    k, c = vars_flat.reshape((S + 1, 2)).T
    residuals = np.zeros(2 * S + 2)

    # Initial condition for capital
    residuals[0] = k[0] - k_init

    # Compute residuals for each time step
    for t in range(S):
        residuals[2 * t + 1] = euler_residual(
            c[t], c[t + 1],
            shocks['τ_c'][t], shocks['τ_c'][t + 1], shocks['τ_k'][t + 1],
            k[t + 1], model
        )
        residuals[2 * t + 2] = feasi_residual(
            k[t + 1], k[t], c[t],
            shocks['g'][t], model
        )

    # Terminal condition
    residuals[-1] = euler_residual(
        c[S], c[S],
        shocks['τ_c'][S], shocks['τ_c'][S], shocks['τ_k'][S],
        k[S], model
    )
    
    return residuals

# Root-finding Algorithm to minimize the residual
def run_min(shocks, S, model):
    """
    Root-finding algorithm to minimize the vector of residuals.
    """
    k_ss, c_ss = steady_states(model, shocks['g'][0], shocks['τ_k'][0])
    
    # Initial guess for the solution path
    initial_guess = np.column_stack(
        (np.full(S + 1, k_ss), np.full(S + 1, c_ss))).flatten()

    # Solve the system using root-finding
    sol = root(compute_residuals, initial_guess, 
               args=(k_ss, S, shocks, model), tol=1e-8)

    # Reshape solution to get time paths for k and c
    return sol.x.reshape((S + 1, 2))
```

Below are the results for the same experiments using the second method.

This method does not have numerical stability issues, so `mp.mpf` is not necessary.

**Experiment 1: Foreseen once-and-for-all increase in $g$ from 0.2 to 0.4 in period 10**

The experiment replicates Figure 12.9.1 in RMT5 under the parameter $\gamma = 2$.

```{code-cell} ipython3
# Define the shocks for the simulation
S = 100
shocks = {
    'g': np.concatenate((np.repeat(0.2, 10), np.repeat(0.4, S - 9))),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, run_min, plot_results, 'g')
```

The experiment replicates the Figure 12.9.2 in RMT5 under $\gamma = 2$ and $\gamma = 0.2$.

```{code-cell} ipython3
experiment_two_models(shocks, S, model, model_γ2, 
                run_min, plot_results, 'g')
```

Below replicates the graph 12.9.3:

```{code-cell} ipython3
solution = run_min(shocks, S, model)

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

plot_prices(solution, c_ss_initial, 'g', axes, model, T=40)

for ax in axes[5:]:
    fig.delaxes(ax)

handles, labels = axes[3].get_legend_handles_labels()  
fig.legend(handles, labels, title=r"$r_{t,t+s}$ with ", loc='lower right', ncol=3, fontsize=10, bbox_to_anchor=(1, 0.1))  
plt.tight_layout()
plt.show()
```

**Experiment 2: Foreseen once-and-for-all increase in $\tau_c$ from 0.0 to 0.2 in period 10.**

The experiment replicates the Figure 12.9.4.

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, run_min, plot_results, 'τ_c')
```

**Experiment 3: Foreseen once-and-for-all increase in $\tau_k$ from 0.0 to 0.2 in period 10.**

The experiment replicates the Figure 12.9.5.

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))) 
}

experiment_two_models(shocks, S, model, model_γ2, 
                run_min, plot_results, 'τ_k')
```

**Experiment 4: Foreseen one-time increase in $g$ from 0.2 to 0.4 in period 10, after which $g$ returns to 0.2 forever**

The experiment replicates the Figure 12.9.6.

```{code-cell} ipython3
g_path = np.repeat(0.2, S + 1)
g_path[10] = 0.4

shocks = {
    'g': g_path,
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, run_min, plot_results, 'g')
```
