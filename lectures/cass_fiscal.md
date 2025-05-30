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

# Cass-Koopmans Model with Distorting Taxes 

## Overview

This lecture studies effects of foreseen   fiscal and technology shocks on competitive equilibrium prices and quantities in a nonstochastic version of a Cass-Koopmans  growth model with features described in this QuantEcon lecture {doc}`cass_koopmans_2`.

This model is discussed in more  detail  in chapter 11 of   {cite}`Ljungqvist2012`.

We use the model as a laboratory to experiment with  numerical techniques for approximating equilibria and to display the structure of dynamic models in which decision makers have perfect foresight about future government decisions. 

Following a classic paper by Robert E. Hall {cite}`hall1971dynamic`, we augment a nonstochastic version of the Cass-Koopmans optimal  growth model with a government that purchases a stream of goods and that finances its purchases  with an sequences of several  distorting flat-rate taxes.

Distorting taxes prevent a competitive equilibrium allocation from solving a planning problem. 

Therefore, to compute an equilibrium allocation and price system, we solve a system of nonlinear difference equations consisting of the first-order conditions for decision makers and the other equilibrium conditions.

We present two ways to approximate an equilibrium:

- The first is a shooting algorithm like the one that we deployed  in {doc}`cass_koopmans_2`.

- The second method is a root-finding algorithm that  minimizes residuals from the  first-order conditions of the consumer and   representative firm.

After studying the behavior of the closed one-country model, we study a two-country version of the model that is closely related to  {cite:t}`mendoza1998international`.


(cs_fs_model)=
## The Economy


### Technology

Feasible allocations satisfy  

$$
g_t + c_t + x_t \leq F(k_t, n_t),
$$ (eq:tech_capital)

where 

- $g_t$ is government purchases of the time $t$ good
- $x_t$ is gross investment, and 
- $F(k_t, n_t)$ is a linearly homogeneous production function with positive and decreasing marginal products of capital $k_t$ and labor $n_t$.

Physical capital evolves according to

$$
k_{t+1} = (1 - \delta)k_t + x_t,
$$

where $\delta \in (0, 1)$ is a depreciation rate.

It is sometimes convenient to eliminate $x_t$ from {eq}`eq:tech_capital` and to represent it as 
as 

$$
g_t + c_t + k_{t+1} \leq F(k_t, n_t) + (1 - \delta)k_t.
$$ 

### Components of a competitive equilibrium

All trades occurring at time $0$.

The representative  household owns capital, makes investment decisions, and rents capital and labor to a representative production firm.

The representative firm uses capital and labor to produce goods with the production function $F(k_t, n_t)$.

A **price system** is a triple of sequences $\{q_t, \eta_t, w_t\}_{t=0}^\infty$, where

- $q_t$ is the time $0$ pretax price of one unit of investment or consumption at time $t$ ($x_t$ or $c_t$),
- $\eta_t$ is the pretax price at time $t$ that the household receives from the firm for renting capital at time $t$, and
- $w_t$ is the pretax price at time $t$ that the household receives for renting labor to the firm at time $t$.

The prices $w_t$ and $\eta_t$ are expressed in terms of time $t$ goods, while $q_t$ is expressed in terms of a numeraire at time $0$, as in {doc}`cass_koopmans_2`.

The presence of a government  distinguishes this lecture from
{doc}`cass_koopmans_2`.

Government purchases of goods at time $t$ are $g_t \geq 0$.

A government expenditure plan is a sequence $g = \{g_t\}_{t=0}^\infty$. 

A government tax plan is a $4$-tuple of sequences $\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$, 
where 

- $\tau_{ct}$ is a tax rate on consumption at time $t$, 
- $\tau_{kt}$ is a tax rate on rentals of capital at time $t$, 
- $\tau_{nt}$ is a tax rate on wage earnings at time $t$, and 
- $\tau_{ht}$ is a lump sum tax on a consumer at time $t$.

Because  lump-sum taxes $\tau_{ht}$ are available, the government actually should not
use any distorting taxes. 

Nevertheless, we include all of these taxes because, like {cite}`hall1971dynamic`,
they allow us to analyze how various taxes distort production and consumption decisions.

In the [experiment section](cf:experiments), we shall see how variations in government tax plan affect 
the transition path and equilibrium.


### Representative Household

A representative household has preferences over nonnegative streams of a single consumption good $c_t$ and leisure $1-n_t$ that are ordered by:

$$
\sum_{t=0}^{\infty} \beta^t U(c_t, 1-n_t), \quad \beta \in (0, 1),
$$ (eq:utility)

where

- $U$ is strictly increasing in $c_t$, twice continuously differentiable, and strictly concave with $c_t \geq 0$ and $n_t \in [0, 1]$.


The representative hßousehold maximizes {eq}`eq:utility` subject to  the single budget constraint:

$$
\begin{aligned}
    \sum_{t=0}^\infty& q_t \left\{ (1 + \tau_{ct})c_t + \underbrace{[k_{t+1} - (1 - \delta)k_t]}_{\text{no tax when investing}} \right\} \\
    &\leq \sum_{t=0}^\infty q_t \left\{ \eta_t k_t - \underbrace{\tau_{kt}(\eta_t - \delta)k_t}_{\text{tax on rental return}} + (1 - \tau_{nt})w_t n_t - \tau_{ht} \right\}.
\end{aligned}
$$ (eq:house_budget)

Here we have assumed that the government gives a depreciation allowance $\delta k_t$
from the gross rentals on capital $\eta_t k_t$ and so collects taxes $\tau_{kt} (\eta_t - \delta) k_t$
on rentals from capital.

### Government 

Government plans $\{ g_t \}_{t=0}^\infty$ for government purchases and taxes $\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$ must respect the budget constraint

$$
\sum_{t=0}^\infty q_t g_t \leq \sum_{t=0}^\infty q_t \left\{ \tau_{ct}c_t + \tau_{kt}(\eta_t - \delta)k_t + \tau_{nt}w_t n_t + \tau_{ht} \right\}.
$$ (eq:gov_budget)



Given a budget-feasible government policy $\{g_t\}_{t=0}^\infty$ and $\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$ subject to {eq}`eq:gov_budget`,

- *Household* chooses $\{c_t\}_{t=0}^\infty$, $\{n_t\}_{t=0}^\infty$, and $\{k_{t+1}\}_{t=0}^\infty$ to maximize utility{eq}`eq:utility` subject to budget constraint{eq}`eq:house_budget`, and 
- *Frim* chooses sequences of capital $\{k_t\}_{t=0}^\infty$ and $\{n_t\}_{t=0}^\infty$ to maximize profits

    $$
         \sum_{t=0}^\infty q_t [F(k_t, n_t) - \eta_t k_t - w_t n_t]
    $$ (eq:firm_profit)
  
- A **feasible allocation** is a sequence $\{c_t, x_t, n_t, k_t\}_{t=0}^\infty$ that satisfies feasibility condition {eq}`eq:tech_capital`.

## Equilibrium

```{prf:definition}
:label: com_eq_tax

A **competitive equilibrium with distorting taxes** is a **budget-feasible government policy**, 
**a feasible allocation**, and **a price system** for which, given the price system and the government 
policy, the allocation solves the household's problem and the firm's problem.
```

## No-arbitrage Condition

A no-arbitrage argument implies a restriction on prices and tax rates across time.


By rearranging {eq}`eq:house_budget` and group $k_t$ at the same $t$, we can get

$$
    \begin{aligned}
    \sum_{t=0}^\infty q_t \left[(1 + \tau_{ct})c_t \right] &\leq \sum_{t=0}^\infty q_t(1 - \tau_{nt})w_t n_t - \sum_{t=0}^\infty q_t \tau_{ht} \\
    &+ \sum_{t=1}^\infty\left\{ \left[(1 - \tau_{kt})(\eta_t - \delta) + 1\right]q_t - q_{t-1}\right\}k_t \\
    &+ \left[(1 - \tau_{k0})(\eta_0 - \delta) + 1\right]q_0k_0 - \lim_{T \to \infty} q_T k_{T+1}
    \end{aligned}
$$ (eq:constrant_house)

The household inherits a given $k_0$ that it takes as initial condition and is free to choose $\{ c_t, n_t, k_{t+1} \}_{t=0}^\infty$.

The household's budget constraint {eq}`eq:house_budget` must be bounded in equilibrium due to finite resources. 

This imposes a restriction on price and tax sequences. 

Specifically, for  $t \geq 1$, the terms multiplying $k_t$ must equal zero.

If they were strictly positive (negative), the household could arbitrarily increase (decrease) the right-hand side of {eq}`eq:house_budget` by selecting an arbitrarily large positive (negative) $k_t$, leading to unbounded profit or arbitrage opportunities:

- For strictly positive terms, the household could purchase large capital stocks $k_t$ and profit from their rental services and undepreciated value. 

- For strictly negative terms, the household could engage in "short selling" synthetic units of capital. Both cases would make {eq}`eq:house_budget` unbounded.

Hence, by setting the terms multiplying $k_t$ to $0$ we have the non-arbitrage condition:

$$
\frac{q_t}{q_{t+1}} = \left[(1 - \tau_{kt+1})(\eta_{t+1} - \delta) + 1\right].
$$ (eq:no_arb)

Moreover, we have terminal condition:

$$
-\lim_{T \to \infty} q_T k_{T+1} = 0.
$$ (eq:terminal)



Zero-profit conditions for the representative firm impose additional restrictions on equilibrium prices and quantities. 

The present value of the firm's profits is

$$
\sum_{t=0}^\infty q_t \left[ F(k_t, n_t) - w_t n_t - \eta_t k_t \right].
$$

Applying Euler's theorem on linearly homogeneous functions to $F(k, n)$, the firm's present value is:

$$
\sum_{t=0}^\infty q_t \left[ (F_{kt} - \eta_t)k_t + (F_{nt} - w_t)n_t \right].
$$

No-arbitrage (or zero-profit) conditions are:

$$
\eta_t = F_{kt}, \quad w_t = F_{nt}.
$$(eq:no_arb_firms)

## Household's First Order Condition

Household maximize {eq}`eq:utility` under {eq}`eq:house_budget`.

Let $U_1 = \frac{\partial U}{\partial c}, U_2 = \frac{\partial U}{\partial (1-n)} = -\frac{\partial U}{\partial n}.$, we can derive FOC from the Lagrangian

$$
\mathcal{L} = \sum_{t=0}^\infty \beta^t U(c_t, 1 - n_t) + \mu \left( \sum_{t=0}^\infty q_t \left[(1 + \tau_{ct})c_t - (1 - \tau_{nt})w_t n_t + \ldots \right] \right),
$$

First-order necessary conditions for the representative household's problem are 

$$
\frac{\partial \mathcal{L}}{\partial c_t} = \beta^t U_{1}(c_t, 1 - n_t) - \mu q_t (1 + \tau_{ct}) = 0
$$ (eq:foc_c_1)

and 

$$
\frac{\partial \mathcal{L}}{\partial n_t} = \beta^t \left(-U_{2t}(c_t, 1 - n_t)\right) - \mu q_t (1 - \tau_{nt}) w_t = 0
$$ (eq:foc_n_1)

Rearranging {eq}`eq:foc_c_1` and {eq}`eq:foc_n_1`, we have

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


Plugging {eq}`eq:foc_c` into {eq}`eq:terminal` and replacing $q_t$, we get terminal condition

$$
-\lim_{T \to \infty} \beta^T \frac{U_{1T}}{(1 + \tau_{cT})} k_{T+1} = 0.
$$ (eq:terminal_final)

## Computing Equilibria

To compute an equilibrium,  we seek a  price system $\{q_t, \eta_t, w_t\}$, a budget feasible government policy $\{g_t, \tau_t\} \equiv \{g_t, \tau_{ct}, \tau_{nt}, \tau_{kt}, \tau_{ht}\}$, and an allocation $\{c_t, n_t, k_{t+1}\}$ that solve a system of nonlinear difference equations consisting of 

- feasibility condition {eq}`eq:tech_capital`, no-arbitrage condition for household {eq}`eq:no_arb` and firms {eq}`eq:no_arb_firms`, household's first order conditions {eq}`eq:foc_c` and {eq}`eq:foc_n`.
- an initial condition $k_0$ and a terminal condition {eq}`eq:terminal_final`.

(cass_fiscal_shooting)=
## Python Code

We require the following imports

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

We  use the `mpmath` library to perform high-precision arithmetic in the shooting algorithm in cases where the solution diverges due to numerical instability.

```{note}
In the functions below, we include routines to handle the growth component, which will be discussed further in the section {ref}`growth_model`.

We include them here to avoid code duplication.
```


We set the following parameters

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

### Inelastic Labor Supply

In this lecture, we consider the special case where $U(c, 1-n) = u(c)$ and $f(k) := F(k, 1)$.

We rewrite {eq}`eq:tech_capital` with $f(k) := F(k, 1)$,

$$
k_{t+1} = f(k_t) + (1 - \delta) k_t - g_t - c_t.
$$ (eq:feasi_capital)

```{code-cell} ipython3
def next_k(k_t, g_t, c_t, model, μ_t=1):
    """
    Capital next period: k_{t+1} = f(k_t) + (1 - δ) * k_t - c_t - g_t
    with optional growth adjustment: k_{t+1} = (f(k_t) + (1 - δ) * k_t - c_t - g_t) / μ_{t+1}
    """
    return (f(k_t, model) + (1 - model.δ) * k_t - g_t - c_t) / μ_t
```

By the properties of a linearly homogeneous production function, we have $F_k(k, n) = f'(k)$ and $F_n(k, 1) = f(k, 1) - f'(k)k$.

Substituting {eq}`eq:foc_c`, {eq}`eq:no_arb_firms`, and {eq}`eq:feasi_capital` into {eq}`eq:no_arb`, we obtain the Euler equation

$$
\begin{aligned}
&\frac{u'(f(k_t) + (1 - \delta) k_t - g_t - k_{t+1})}{(1 + \tau_{ct})} \\
&- \beta \frac{u'(f(k_{t+1}) + (1 - \delta) k_{t+1} - g_{t+1} - k_{t+2})}{(1 + \tau_{ct+1})} \\
&\times [(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1] = 0.
\end{aligned}
$$(eq:euler_house)

This can be simplified to:

$$
\begin{aligned}
u'(c_t) = \beta u'(c_{t+1}) \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} [(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1].
\end{aligned}
$$ (eq:diff_second)


Equation {eq}`eq:diff_second` will appear prominently in our equilibrium computation algorithms.
 

### Steady state

Tax rates and government expenditures act as **forcing functions** for  difference equations {eq}`eq:feasi_capital` and {eq}`eq:diff_second`.

Define $z_t = [g_t, \tau_{kt}, \tau_{ct}]'$. 

Represent  the second-order difference equation as:

$$
H(k_t, k_{t+1}, k_{t+2}; z_t, z_{t+1}) = 0.
$$ (eq:second_ord_diff)

We assume that a government policy reaches a steady state such that $\lim_{t \to \infty} z_t = \bar z$ and that the steady state prevails for $t > T$. 

The terminal steady-state capital stock $\bar{k}$ satisfies:

$$
H(\bar{k}, \bar{k}, \bar{k}, \bar{z}, \bar{z}) = 0.
$$

From  difference equation {eq}`eq:diff_second`, we can infer a restriction on  the steady-state:

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
def compute_q_path(c_path, model, S=100, A_path=None):
    """
    Compute q path: q_t = (β^t * u'(c_t)) / u'(c_0)
    with optional A_path for growth models.
    """
    A = np.ones_like(c_path) if A_path is None else np.asarray(A_path)
    q_path = np.zeros_like(c_path)
    for t in range(S):
        q_path[t] = (model.β ** t * 
                         u_prime(c_path[t], model, A[t])) / u_prime(c_path[0], model, A[0])
    return q_path
```

*Capital rental rate*

$$
\eta_t = f'(k_t)  
$$

```{code-cell} ipython3
def compute_η_path(k_path, model, S=100, A_path=None):
    """
    Compute η path: η_t = f'(k_t)
    with optional A_path for growth models.
    """
    A = np.ones_like(k_path) if A_path is None else np.asarray(A_path)
    η_path = np.zeros_like(k_path)
    for t in range(S):
        η_path[t] = f_prime(k_path[t], model, A[t])
    return η_path
```

*Labor rental rate:*

$$
w_t = f(k_t) - k_t f'(k_t)    
$$

```{code-cell} ipython3
def compute_w_path(k_path, η_path, model, S=100, A_path=None):
    """
    Compute w path: w_t = f(k_t) - k_t * f'(k_t)
    with optional A_path for growth models.
    """
    A = np.ones_like(k_path) if A_path is None else np.asarray(A_path)
    w_path = np.zeros_like(k_path)
    for t in range(S):
        w_path[t] = f(k_path[t], model, A[t]) - k_path[t] * η_path[t]
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
    R_bar = [(1 + τ_c_t) / (1 + τ_c_{t+1})] 
        * { [1 - τ_k_{t+1}] * [f'(k_{t+1}) - δ] + 1 }
    """
    return ((1 + τ_ct) / (1 + τ_ctp1)) * (
        (1 - τ_ktp1) * (f_prime(k_tp1, model) - model.δ) + 1)
```

```{code-cell} ipython3
def compute_R_bar_path(shocks, k_path, model, S=100):
    """
    Compute R_bar path over time.
    """
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
R^{-1}_{t, t+1} = \frac{q_{t+1}}{q_{t}} = m_{t, t+1} = \beta \frac{u'(c_{t+1})}{u'(c_t)} \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})}
$$ (eq:equil_bigR)


*Net one-period rate of interest:*

$$
r_{t, t+1} \equiv R_{t, t+1} - 1 = (1 - \tau_{k, t+1})(f'(k_{t+1}) - \delta)
$$ (eq:equil_r)

By {eq}`eq:equil_bigR` and $r_{t, t+1} = - \ln(\frac{q_{t+1}}{q_t})$, we have

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

## Some functional forms

We assume that  the representative household' period utility has  the following CRRA (constant relative risk aversion) form  

$$
u(c) = \frac{c^{1 - \gamma}}{1 - \gamma}
$$

```{code-cell} ipython3
def u_prime(c, model, A_t=1):
    """
    Marginal utility: u'(c) = c^{-γ}
    with optional technology adjustment: u'(cA) = (cA)^{-γ}
    """
    return (c * A_t) ** (-model.γ)
```

By substituting {eq}`eq:gross_rate` into {eq}`eq:diff_second`, we obtain

$$
c_{t+1} = c_t \left[ \beta \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{k, t+1})(f'(k_{t+1}) - \delta) + 1 \right] \right]^{\frac{1}{\gamma}} = c_t \left[ \beta \overline{R}_{t+1} \right]^{\frac{1}{\gamma}}
$$ (eq:consume_R)

```{code-cell} ipython3
def next_c(c_t, R_bar, model, μ_t=1):
    """
    Consumption next period: c_{t+1} = c_t * (β * R̄)^{1/γ}
    with optional growth adjustment: c_{t+1} = c_t * (β * R_bar)^{1/γ} * μ_{t+1}^{-1}
    """
    return c_t * (model.β * R_bar) ** (1 / model.γ) / μ_t
```

For the production function we assume a Cobb-Douglas form:

$$
F(k, 1) = A k^\alpha
$$

```{code-cell} ipython3
def f(k, model, A=1): 
    """
    Production function: f(k) = A * k^{α}
    """
    return A * k ** model.α

def f_prime(k, model, A=1):
    """
    Marginal product of capital: f'(k) = α * A * k^{α - 1}
    """
    return model.α * A * k ** (model.α - 1)
```

## Computation

We describe  two ways to compute an equilibrium: 

 * a shooting algorithm
 * a residual-minimization method that focuses on imposing  Euler equation {eq}`eq:diff_second` and the  feasibility condition {eq}`eq:feasi_capital`.

### Shooting Algorithm

This algorithm deploys the following steps.

1. Solve the equation {eq}`eq:diff_second_steady` for the terminal steady-state capital $\bar{k}$ that corresponds to the permanent policy vector $\bar{z}$.

2. Select a large time index $S \gg T$, guess an initial consumption rate $c_0$, and use the equation {eq}`eq:feasi_capital` to solve for $k_1$.

3. Use the equation {eq}`eq:consume_R` to determine $c_{t+1}$. Then, apply the equation {eq}`eq:feasi_capital` to compute $k_{t+2}$.

4. Iterate step 3 to compute candidate values $\hat{k}_t$ for $t = 1, \dots, S$.

5. Compute the difference $\hat{k}_S - \bar{k}$. If $\left| \hat{k}_S - \bar{k} \right| > \epsilon$ for some small $\epsilon$, adjust $c_0$ and repeat steps 2-5.

6. Adjust $c_0$ iteratively using the bisection method to find a value that ensures $\left| \hat{k}_S - \bar{k} \right| < \epsilon$.

The following code implements these steps.

```{code-cell} ipython3
# Steady-state calculation
def steady_states(model, g_ss, τ_k_ss=0.0, μ_ss=None):
    """
    Calculate steady state values for capital and 
    consumption with optional A_path for growth models.
    """

    β, δ, α, γ = model.β, model.δ, model.α, model.γ

    A = model.A or 1.0

    # growth‐adjustment in the numerator: μ^γ or 1
    μ_eff = μ_ss**γ if μ_ss is not None else 1.0

    num = δ + (μ_eff/β - 1) / (1 - τ_k_ss)
    k_ss = (num / (α * A)) ** (1 / (α - 1))

    c_ss = (
        A * k_ss**α - δ * k_ss - g_ss
        if μ_ss is None
        else k_ss**α + (1 - δ - μ_ss) * k_ss - g_ss
    )

    return k_ss, c_ss

def shooting_algorithm(
    c0, k0, shocks, S, model, A_path=None):
    """
    Shooting algorithm for given initial c0 and k0
    with optional A_path for growth models.
    """
    # unpack & mpf‐ify shocks, fill μ with ones if missing
    g = np.array(list(map(mpf, shocks['g'])), dtype=object)
    τ_c = np.array(list(map(mpf, shocks['τ_c'])), dtype=object)
    τ_k = np.array(list(map(mpf, shocks['τ_k'])), dtype=object)
    μ = (np.array(list(map(mpf, shocks['μ'])), dtype=object)
              if 'μ' in shocks else np.ones_like(g))
    A = np.ones_like(g) if A_path is None else A_path

    k_path = np.empty(S+1, dtype=object)
    c_path = np.empty(S+1, dtype=object)
    k_path[0], c_path[0] = mpf(k0), mpf(c0)

    for t in range(S):
        k_t, c_t = k_path[t], c_path[t]
        k_tp1 = next_k(k_t, g[t], c_t, model, μ[t+1])
        if k_tp1 < 0:
            return None, None
        k_path[t+1] = k_tp1

        R_bar = compute_R_bar(
            τ_c[t], τ_c[t+1], τ_k[t+1], k_tp1, model
        )
        c_tp1 = next_c(c_t, R_bar, model, μ[t+1])
        if c_tp1 < 0:
            return None, None
        c_path[t+1] = c_tp1

    return k_path, c_path


def bisection_c0(
    c0_guess, k0, shocks, S, model, tol=mpf('1e-6'), 
    max_iter=1000, verbose=False, A_path=None):
    """
    Bisection method to find initial c0
    """
    # steady‐state uses last shocks (μ=1 if missing)
    g_last    = mpf(shocks['g'][-1])
    τ_k_last  = mpf(shocks['τ_k'][-1])
    μ_last    = mpf(shocks['μ'][-1]) if 'μ' in shocks else mpf('1')
    k_ss_fin, _ = steady_states(model, g_last, τ_k_last, μ_last)

    c0_lo, c0_hi = mpf('0'), f(k_ss_fin, model)
    c0 = mpf(c0_guess)

    for i in range(1, max_iter+1):
        k_path, _ = shooting_algorithm(c0, k0, shocks, S, model, A_path)
        if k_path is None:
            if verbose:
                print(f"[{i}] shoot failed at c0={c0}")
            c0_hi = c0
        else:
            err = k_path[-1] - k_ss_fin
            if verbose and i % 100 == 0:
                print(f"[{i}] c0={c0}, err={err}")
            if abs(err) < tol:
                if verbose:
                    print(f"Converged after {i} iter")
                return c0
            # update bounds in one line
            c0_lo, c0_hi = (c0, c0_hi) if err > 0 else (c0_lo, c0)
        c0 = (c0_lo + c0_hi) / mpf('2')

    warn(f"bisection did not converge after {max_iter} iters; returning c0={c0}")
    return c0


def run_shooting(
    shocks, S, model, A_path=None, 
    c0_finder=bisection_c0, shooter=shooting_algorithm):
    """
    Compute initial SS, find c0, and return [k,c] paths
    with optional A_path for growth models.
    """
    # initial SS at t=0 (μ=1 if missing)
    g0    = mpf(shocks['g'][0])
    τ_k0  = mpf(shocks['τ_k'][0])
    μ0    = mpf(shocks['μ'][0]) if 'μ' in shocks else mpf('1')
    k0, c0 = steady_states(model, g0, τ_k0, μ0)

    optimal_c0 = c0_finder(c0, k0, shocks, S, model, A_path=A_path)
    print(f"Model: {model}\nOptimal initial consumption c0 = {mpf(optimal_c0)}")

    k_path, c_path = shooter(optimal_c0, k0, shocks, S, model, A_path)
    return np.column_stack([k_path, c_path])
```

(cf:experiments)=
### Experiments

Let's run some  experiments.

1. A foreseen once-and-for-all increase in $g$ from 0.2 to 0.4 occurring in period 10,
2. A foreseen once-and-for-all increase in $\tau_c$ from 0.0 to 0.2 occurring in period 10,
3. A foreseen once-and-for-all increase in $\tau_k$ from 0.0 to 0.2 occurring in period 10, and
4. A foreseen one-time increase in $g$ from 0.2 to 0.4 in period 10, after which $g$ reverts to 0.2 permanently.

+++

To start, we  prepare  sequences that we'll  used to initialize our iterative algorithm. 

We will start from an initial  steady state and  apply shocks at an the indicated  time.

```{code-cell} ipython3
def plot_results(
    solution, k_ss, c_ss, shocks, shock_param, axes, model,
    A_path=None, label='', linestyle='-', T=40):
    """
    Plot simulation results (k, c, R, η, and a policy shock)
    with optional A_path for growth models.
    """
    k_path = solution[:, 0]
    c_path = solution[:, 1]
    T = min(T, k_path.size)

    # handle growth parameters
    μ0 = shocks['μ'][0] if 'μ' in shocks else 1.0
    A0 = A_path[0] if A_path is not None else (model.A or 1.0)

    # steady‐state lines
    R_bar_ss = (1 / model.β) * (μ0**model.γ)
    η_ss     = model.α * A0 * k_ss**(model.α - 1)

    # plot k
    axes[0].plot(k_path[:T], linestyle=linestyle, label=label)
    axes[0].axhline(k_ss, linestyle='--', color='black')
    axes[0].set_title('k')

    # plot c
    axes[1].plot(c_path[:T], linestyle=linestyle, label=label)
    axes[1].axhline(c_ss, linestyle='--', color='black')
    axes[1].set_title('c')

    # plot R bar
    S_full    = k_path.size - 1
    R_bar_path = compute_R_bar_path(shocks, k_path, model, S_full)
    axes[2].plot(R_bar_path[:T], linestyle=linestyle, label=label)
    axes[2].axhline(R_bar_ss, linestyle='--', color='black')
    axes[2].set_title(r'$\bar{R}$')

    # plot η
    η_path = compute_η_path(k_path, model, S_full)
    axes[3].plot(η_path[:T], linestyle=linestyle, label=label)
    axes[3].axhline(η_ss, linestyle='--', color='black')
    axes[3].set_title(r'$\eta$')

    # plot shock
    shock_series = np.array(shocks[shock_param], dtype=object)
    axes[4].plot(shock_series[:T], linestyle=linestyle, label=label)
    axes[4].axhline(shock_series[0], linestyle='--', color='black')
    axes[4].set_title(rf'${shock_param}$')

    if label:
        for ax in axes[:5]:
            ax.legend()
```

**Experiment 1: Foreseen once-and-for-all increase in $g$ from 0.2 to 0.4 in period 10**

The figure below shows consequences of a foreseen permanent increase in $g$ at $t = T = 10$ that is financed by an increase in lump-sum taxes

```{code-cell} ipython3
# Define shocks as a dictionary
shocks = {
    'g': np.concatenate(
        (np.repeat(0.2, 10), np.repeat(0.4, S - 9))
    ),
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

The above figures indicate that an equilibrium **consumption smoothing** mechanism is at work, driven by the representative consumer's preference for smooth consumption paths coming from the curvature of its one-period utility function. 

- The steady-state value of the capital stock remains unaffected:
  - This follows from the fact that $g$ disappears from the steady state version of the Euler equation ({eq}`eq:diff_second_steady`).

- Consumption begins to decline gradually before time $T$ due to increased government consumption:
  - Households reduce consumption to offset government spending, which is financed through increased lump-sum taxes.
  - The competitive economy signals households to consume less through an increase in the stream of lump-sum taxes.
  - Households, caring about the present value rather than the timing of taxes, experience an adverse wealth effect on consumption, leading to an immediate response.
  
- Capital gradually accumulates between time $0$ and $T$ due to increased savings and reduces gradually after time $T$:
    - This temporal variation in capital stock smooths consumption over time, driven by the representative consumer's consumption-smoothing motive.

Let's collect the procedures used above into a function that runs the solver and draws  plots for a given experiment

```{code-cell} ipython3
:tags: [hide-input]

def experiment_model(
    shocks, S, model, A_path=None, solver=run_shooting, 
    plot_func=plot_results, policy_shock='g', T=40):
    """
    Run the shooting algorithm and plot results.
    """
    # initial steady state (μ0=None if no growth)
    g0   = mpf(shocks['g'][0])
    τk0  = mpf(shocks['τ_k'][0])
    μ0   = mpf(shocks['μ'][0]) if 'μ' in shocks else None
    k_ss, c_ss = steady_states(model, g0, τk0, μ0)

    print(f"Steady-state capital: {float(k_ss):.4f}")
    print(f"Steady-state consumption: {float(c_ss):.4f}")
    print('-'*64)

    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()

    sol = solver(shocks, S, model, A_path)
    plot_func(
        sol, k_ss, c_ss, shocks, policy_shock, axes, model,
        A_path=A_path, T=T
    )

    # remove unused axes
    for ax in axes[5:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
```

The following figure compares responses to a foreseen increase in $g$ at $t = 10$ for
two economies:
 
 *  our original economy with $\gamma = 2$, shown in the solid line, and
 *  an otherwise identical economy with $\gamma = 0.2$.

This comparison interest us because the  utility curvature parameter $\gamma$ governs the household's willingness to substitute consumption over time, and thus it preferences about smoothness of consumption paths over time.

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

The outcomes indicate that lowering $\gamma$ affects both the consumption and capital stock paths because - it increases the representative consumer's willingness to substitute consumption across time:

- Consumption path:
  - When $\gamma = 0.2$, consumption becomes less smooth compared to $\gamma = 2$.
  - For $\gamma = 0.2$, consumption mirrors the government expenditure path more closely, staying higher until $t = 10$.

- Capital stock path:
  - With $\gamma = 0.2$, there are smaller build-ups and drawdowns of capital stock.
  - There are also smaller fluctuations in $\bar{R}$ and $\eta$.

Let's write another function that runs the solver and draws plots for these two experiments

```{code-cell} ipython3
:tags: [hide-input]

def experiment_two_models(
    shocks, S, model_1, model_2, solver=run_shooting, plot_func=plot_results, 
    policy_shock='g', legend_label_fun=None, T=40, A_path=None):
    """
    Compare and plot the shooting algorithm paths for two models.
    """
    is_growth = 'μ' in shocks
    μ0 = mpf(shocks['μ'][0]) if is_growth else None

    # initial steady states for both models
    g0   = mpf(shocks['g'][0])
    τk0  = mpf(shocks['τ_k'][0])
    k_ss1, c_ss1 = steady_states(model_1, g0, τk0, μ0)
    k_ss2, c_ss2 = steady_states(model_2, g0, τk0, μ0)

    # print both    
    print(f"Model 1 (γ={model_1.γ}): steady state k={float(k_ss1):.4f}, c={float(c_ss1):.4f}")
    print(f"Model 2 (γ={model_2.γ}): steady state k={float(k_ss2):.4f}, c={float(c_ss2):.4f}")
    print('-'*64)

    # default legend labels
    if legend_label_fun is None:
        legend_label_fun = lambda m: fr"$\gamma = {m.γ}$"

    # prepare figure
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()

    # loop over (model, steady‐state, linestyle)
    for model, (k_ss, c_ss), ls in [
        (model_1, (k_ss1, c_ss1), '-'),
        (model_2, (k_ss2, c_ss2), '-.')
    ]:
        sol = solver(shocks, S, model, A_path)
        plot_func(sol, k_ss, c_ss, shocks, policy_shock, axes, 
                  model, A_path=A_path, 
                  label=legend_label_fun(model), 
                  linestyle=ls, T=T)

    # shared legend in lower‐right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower right', ncol=2, 
        fontsize=12, bbox_to_anchor=(1, 0.1))

    # drop the unused subplot
    for ax in axes[5:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
```

Now we plot other equilibrium quantities:

```{code-cell} ipython3
def plot_prices(solution, c_ss, shock_param, axes,
                model, label='', linestyle='-', T=40):
    """
    Compares and plots prices
    """
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

For $\gamma = 2$ again, the next figure describes the response of $q_t$ and the term
structure of interest rates to a foreseen increase in $g_t$ at $t = 10$

```{code-cell} ipython3
solution = run_shooting(shocks, S, model)

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

plot_prices(solution, c_ss_initial, 'g', axes, model, T=40)

for ax in axes[5:]:
    fig.delaxes(ax)

handles, labels = axes[3].get_legend_handles_labels()  
fig.legend(handles, labels, title=r"$r_{t,t+s}$ with ", loc='lower right', 
           ncol=3, fontsize=10, bbox_to_anchor=(1, 0.1))  
plt.tight_layout()
plt.show()
```

The second panel on the top compares $q_t$ for the initial steady state with $q_t$ after the
increase in $g$ is foreseen at $t = 0$, while the third panel compares the implied
short rate $r_t$.

The fourth panel shows the term structure of interest rates at $t=0$, $t=10$, and $t=60$.

Notice that, by $t = 60$, the system has converged to the new steady state, and the term structure of interest rates becomes flat.

At $t = 10$, the term structure of interest rates is upward sloping.

This upward slope reflects the expected increase in the rate of growth of consumption over time, as shown in the consumption panel.

At $t = 0$, the term structure of interest rates exhibits a "U-shaped" pattern:

- It declines until maturity at $s = 10$.
- After $s = 10$, it increases for longer maturities.
    
This pattern aligns with the pattern of consumption growth in the first two figures, 
which declines at an increasing rate until $t = 10$ and then declines at a decreasing rate afterward.

+++

**Experiment 2: Foreseen once-and-for-all increase in $\tau_c$ from 0.0 to 0.2 in period 10**

With an inelastic labor supply, the Euler equation {eq}`eq:euler_house` 
and the other equilibrium conditions show that
- constant consumption taxes do not distort decisions, but 
- anticipated changes in them do. 

Indeed, {eq}`eq:euler_house` or {eq}`eq:diff_second` indicates that a foreseen in-
crease in $\tau_{ct}$ (i.e., a decrease in $(1+\tau_{ct})$
$(1+\tau_{ct+1})$) operates like an increase in $\tau_{kt}$.

The following figure portrays the response to a foreseen increase in the consumption tax $\tau_c$.

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, 
                 solver=run_shooting, 
                 plot_func=plot_results,  
                 policy_shock='τ_c')
```

Evidently  all variables in the figures above eventually return to their initial
steady-state values.

The anticipated increase in $\tau_{ct}$ leads to variations in consumption 
and capital stock across time:

- At $t = 0$:
    - Anticipation of the increase in $\tau_c$ causes an *immediate jump in consumption*.
    - This is followed by a *consumption binge* that sends the capital stock downward until $t = T = 10$.
- Between $t = 0$ and $t = T = 10$:
    - The decline in the capital stock raises $\bar{R}$ over time.
    - The equilibrium conditions require the growth rate of consumption to rise until $t = T$.
- At $t = T = 10$:
    - The jump in $\tau_c$ depresses $\bar{R}$ below $1$, causing a *sharp drop in consumption*.
- After $T = 10$:
    - The effects of anticipated distortion are over, and the economy gradually adjusts to the lower capital stock.
    - Capital must now rise, requiring *austerity* —consumption plummets after $t = T$,  indicated by  lower levels of consumption.
    - The interest rate gradually declines, and consumption grows at a diminishing rate along the path to the terminal steady-state.

+++

**Experiment 3: Foreseen once-and-for-all increase in $\tau_k$ from 0.0 to 0.2 in period 10**

For the two $\gamma$ values 2 and 0.2, the next figure shows the
response to a foreseen permanent jump in $\tau_{kt}$ at $t = T = 10$.

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))) 
}

experiment_two_models(shocks, S, model, model_γ2, 
                solver=run_shooting, 
                 plot_func=plot_results,  
                 policy_shock='τ_k')
```

The path of government expenditures remains fixed
- the increase in $\tau_{kt}$ is offset by a reduction in the present value of lump-sum taxes to keep the budget balanced.

The  figure shows that:

- Anticipation of the increase in $\tau_{kt}$ leads to immediate decline in capital stock due to increased current consumption and a growing consumption flow.
- $\bar{R}$ starts rising at $t = 0$ and peaks at $t = 9$, and at $t = 10$, $\bar{R}$ drops sharply due to the tax change.
    - Variations in $\bar{R}$ align with  the impact of the tax increase at $t = 10$ on consumption across time.
- Transition dynamics push $k_t$ (capital stock) toward a new, lower steady-state level. In the new steady state:
    - Consumption is lower due to reduced output from the lower capital stock.
    - Smoother consumption paths occur when $\gamma = 2$ than when $\gamma = 0.2$.

+++

So far we have explored consequences of foreseen once-and-for-all changes
in government policy. Next we describe some experiments in which there is a
foreseen one-time change in a policy variable (a "pulse").

**Experiment 4: Foreseen one-time increase in $g$ from 0.2 to 0.4 in period 10, after which $g$ returns to 0.2 forever**

```{code-cell} ipython3
g_path = np.repeat(0.2, S + 1)
g_path[10] = 0.4

shocks = {
    'g': g_path,
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model,
                 solver=run_shooting, 
                 plot_func=plot_results,  
                 policy_shock='g')
```

The figure indicates how:

- Consumption:
    - Drops immediately upon  announcement of the policy and continues to decline over time in anticipation of the one-time surge in $g$.
    - After the shock at $t = 10$, consumption begins to recover, rising at a diminishing rate toward its steady-state value.
    
- Capital and $\bar{R}$:
    - Before $t = 10$, capital accumulates as interest rate changes induce households to prepare for the anticipated increase in government spending.
    - At $t = 10$, the capital stock sharply decreases as the government consumes part of it.
    - $\bar{R}$ jumps above its steady-state value due to the capital reduction and then gradually declines toward its steady-state level.

+++

### Method 2: Residual Minimization 

The second method involves minimizing residuals (i.e., deviations from equalities) of the following equations:

- The Euler equation {eq}`eq:diff_second`:

  $$
  1 = \beta \left(\frac{c_{t+1}}{c_t}\right)^{-\gamma} \frac{(1+\tau_{ct})}{(1+\tau_{ct+1})} \left[(1 - \tau_{kt+1})(\alpha A k_{t+1}^{\alpha-1} - \delta) + 1 \right]
  $$

- The feasibility condition {eq}`eq:feasi_capital`:

  $$
  k_{t+1} = A k_{t}^{\alpha} + (1 - \delta) k_t - g_t - c_t.
  $$

```{code-cell} ipython3
# Euler's equation and feasibility condition 
def euler_residual(c_t, c_tp1, τ_c_t, τ_c_tp1, τ_k_tp1, k_tp1, model, μ_tp1=1):
    """
    Computes the residuals for Euler's equation 
    with optional growth model parameters μ_tp1.
    """
    R_bar = compute_R_bar(τ_c_t, τ_c_tp1, τ_k_tp1, k_tp1, model)
    
    c_expected = next_c(c_t, R_bar, model, μ_tp1)

    return c_expected / c_tp1 - 1.0

def feasi_residual(k_t, k_tm1, c_tm1, g_t, model, μ_t=1):
    """
    Computes the residuals for feasibility condition 
    with optional growth model parameter μ_t.
    """
    k_t_expected = next_k(k_tm1, g_t, c_tm1, model, μ_t)
    return k_t_expected - k_t
```

The algorithm proceeds follows:

1. Find initial steady state $k_0$ based on the government plan at $t=0$.

2. Initialize a sequence of initial guesses $\{\hat{c}_t, \hat{k}_t\}_{t=0}^{S}$.

3. Compute residuals $l_a$ and $l_k$ for $t = 0, \dots, S$, as well as $l_{k_0}$ for $t = 0$ and $l_{k_S}$ for $t = S$:
   - Compute the Euler equation residual for $t = 0, \dots, S$ using {eq}`eq:diff_second`:

     $$
     l_{ta} = \beta u'(c_{t+1}) \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1 \right] - 1
     $$

   - Compute the feasibility condition residual for $t = 1, \dots, S-1$ using {eq}`eq:feasi_capital`:

     $$
     l_{tk} = k_{t+1} - f(k_t) - (1 - \delta)k_t + g_t + c_t
     $$

   - Compute the residual for the initial condition for $k_0$ using {eq}`eq:diff_second_steady` and the initial capital $k_0$:

     $$
     l_{k_0} = 1 - \beta \left[ (1 - \tau_{k0}) \left(f'(k_0) - \delta \right) + 1 \right]
     $$

   - Compute the residual for the terminal condition for $t = S$ using {eq}`eq:diff_second` under the assumptions $c_t = c_{t+1} = c_S$, $k_t = k_{t+1} = k_S$, $\tau_{ct} = \tau_{ct+1} = \tau_{c_s}$, and $\tau_{kt} = \tau_{kt+1} = \tau_{k_s}$:
     
     $$
     l_{k_S} = \beta u'(c_S) \frac{(1 + \tau_{c_s})}{(1 + \tau_{c_s})} \left[(1 - \tau_{k_s})(f'(k_S) - \delta) + 1 \right] - 1
     $$

4. Iteratively adjust  guesses for $\{\hat{c}_t, \hat{k}_t\}_{t=0}^{S}$ to minimize  residuals $l_{k_0}$, $l_{ta}$, $l_{tk}$, and $l_{k_S}$ for $t = 0, \dots, S$.

```{code-cell} ipython3
def compute_residuals(vars_flat, k_init, S, shock_paths, model):
    """
    Compute the residuals for the Euler equation and feasibility condition.
    """
    g, τ_c, τ_k, μ = (shock_paths[key] for key in ('g','τ_c','τ_k','μ'))
    k, c = vars_flat.reshape((S+1, 2)).T
    res = np.empty(2*S+2, dtype=float)

    # boundary condition on initial capital
    res[0] = k[0] - k_init

    # interior Euler and feasibility
    for t in range(S):
        res[2*t + 1] = euler_residual(
            c[t],    c[t+1],
            τ_c[t],  τ_c[t+1],
            τ_k[t+1],k[t+1],
            model, μ[t+1])
        res[2*t + 2] = feasi_residual(
            k[t+1], k[t], c[t],
            g[t],  model,
            μ[t+1])

    # terminal Euler condition at t=S
    res[-1] = euler_residual(
        c[S],   c[S],
        τ_c[S], τ_c[S],
        τ_k[S], k[S],
        model,
        μ[S])

    return res


def run_min(shocks, S, model, A_path=None):
    """
    Solve for the full (k,c) path by root‐finding the residuals.
    """
    shocks['μ'] = shocks['μ'] if 'μ' in shocks else np.ones_like(shocks['g'])

    # compute the steady‐state to serve as both initial capital and guess
    k_ss, c_ss = steady_states(
        model,
        shocks['g'][0],
        shocks['τ_k'][0],
        shocks['μ'][0]  # =1 if no growth
    )

    # initial guess: flat at the steady‐state
    guess = np.column_stack([
        np.full(S+1, k_ss),
        np.full(S+1, c_ss)
    ]).flatten()

    sol = root(
        compute_residuals,
        guess,
        args=(k_ss, S, shocks, model),
        tol=1e-8
    )

    return sol.x.reshape((S+1, 2))
```

We found that  method 2 did  not encounter numerical stability issues, so using  `mp.mpf` is not necessary.

We leave as exercises replicating some of our experiments by using the second method.

```{exercise}
:label: cass_fiscal_ex1

Replicate the plots of our four experiments  using the second method of residual minimization:
1. A foreseen once-and-for-all increase in $g$ from 0.2 to 0.4 occurring in period 10,
2. A foreseen once-and-for-all increase in $\tau_c$ from 0.0 to 0.2 occurring in period 10,
3. A foreseen once-and-for-all increase in $\tau_k$ from 0.0 to 0.2 occurring in period 10, and
4. A foreseen one-time increase in $g$ from 0.2 to 0.4 in period 10, after which $g$ reverts to 0.2 permanently,
```

```{solution-start} cass_fiscal_ex1
:class: dropdown
```

Here is one solution:

**Experiment 1: Foreseen once-and-for-all increase in $g$ from 0.2 to 0.4 in period 10**

```{code-cell} ipython3
shocks = {
    'g': np.concatenate((np.repeat(0.2, 10), np.repeat(0.4, S - 9))),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, solver=run_min, 
                 plot_func=plot_results,  
                 policy_shock='g')
```

```{code-cell} ipython3
experiment_two_models(shocks, S, model, model_γ2, 
                run_min, plot_results, 'g')
```

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

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, solver=run_min, 
                 plot_func=plot_results,  
                 policy_shock='τ_c')
```

**Experiment 3: Foreseen once-and-for-all increase in $\tau_k$ from 0.0 to 0.2 in period 10.**

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))) 
}

experiment_two_models(shocks, S, model, model_γ2, 
                solver=run_min, 
                 plot_func=plot_results,  
                 policy_shock='τ_k')
```

**Experiment 4: Foreseen one-time increase in $g$ from 0.2 to 0.4 in period 10, after which $g$ returns to 0.2 forever**

```{code-cell} ipython3
g_path = np.repeat(0.2, S + 1)
g_path[10] = 0.4

shocks = {
    'g': g_path,
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, solver=run_min, 
                 plot_func=plot_results,  
                 policy_shock='g')
```

```{solution-end}
```


```{exercise}
:label: cass_fiscal_ex2

Design a new experiment where the government expenditure $g$ increases from $0.2$ to $0.4$ in period $10$, 
and then decreases to $0.1$ in period $20$ permanently.
```

```{solution-start} cass_fiscal_ex2
:class: dropdown
```

Here is one solution:

```{code-cell} ipython3
g_path = np.repeat(0.2, S + 1)
g_path[10:20] = 0.4
g_path[20:] = 0.1

shocks = {
    'g': g_path,
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, solver=run_min, 
                 plot_func=plot_results,  
                 policy_shock='g')
```

```{solution-end}
```

(growth_model)=
## Exogenous growth

In the previous section, we considered a model without exogenous growth.

We set the term $A_t$ in the production function to a constant by
setting $A_t = 1$ for all $t$.

Now we are ready to consider growth.

To incorporate growth, we modify the production function to be

$$
Y_t = F(K_t, A_tn_t)
$$

where $Y_t$ is aggregate output, $N_t$ is total employment, $A_t$ is labor-augmenting technical change,
and $F(K, AN)$ is the same linearly homogeneous production function as before.

We assume that $A_t$ follows the process

$$
A_{t+1} = \mu_{t+1}A_t
$$ (eq:growth)

and that $\mu_{t+1}=\bar{\mu}>1$.

```{code-cell} ipython3
# Set the constant A parameter to None
model = create_model(A=None)
```

```{code-cell} ipython3
def compute_A_path(A0, shocks, S=100):
    """
    Compute A path over time.
    """
    A_path = np.full(S + 1, A0)
    for t in range(1, S + 1):
        A_path[t] = A_path[t-1] * shocks['μ'][t-1]
    return A_path
```

### Inelastic Labor Supply

By linear homogeneity, the production function can be expressed as

$$
y_t=f(k_t)
$$

where $f(k)=F(k,1) = k^\alpha$ and $k_t=\frac{K_t}{n_tA_t}$, $y_t=\frac{Y_t}{n_tA_t}$.

$k_t$ and $y_t$ are measured per unit of "effective labor" $A_tn_t$.

We also let $c_t=\frac{C_t}{A_tn_t}$ and $g_t=\frac{G_t}{A_tn_t}$, where $C_t$ and $G_t$ are total consumption and total government expenditures.

We continue to consider the case of inelastic labor supply.

Based on this, feasibility can be summarized by the following modified version
of equation {eq}`eq:feasi_capital`:

$$
k_{t+1}=\mu_{t+1}^{-1}[f(k_t)+(1-\delta)k_t-g_t-c_t]
$$ (eq:feasi_mod)


Again, by the properties of a linearly homogeneous production function, we have 

$$ 
\eta_t = F_k(k_t, 1) = f'(k_t), w_t = F_n(k_t, 1) = f(k_t) - f'(k_t)k_t 
$$

Since per capita consumption is now $c_tA_t$, the counterpart to the Euler equation {eq}`eq:diff_second` is

$$
\begin{aligned}
u'(c_tA_t) = \beta u'(c_{t+1}A_{t+1}) \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} [(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1].
\end{aligned} 
$$ (eq:diff_mod)

$\bar{R}_{t+1}$ continues to be defined by {eq}`eq:gross_rate`, except that now $k_t$ is capital per effective unit of labor. 

Thus, substituting {eq}`eq:gross_rate`, {eq}`eq:diff_mod` becomes

$$
u'(c_tA_t) = \beta u'(c_{t+1}A_{t+1})\bar{R}_{t+1}
$$

Assuming that the household's utility function is the same as before, we have

$$
(c_tA_t)^{-\gamma} = \beta (c_{t+1}A_{t+1})^{-\gamma} \bar{R}_{t+1}
$$

Thus, the counterpart to {eq}`eq:consume_R` is

$$
c_{t+1} = c_t \left[ \beta \bar{R}_{t+1} \right]^{\frac{1}{\gamma}}\mu_{t+1}^{-1}
$$ (eq:consume_r_mod)

### Steady State

In a steady state, $c_{t+1} = c_t$. Then {eq}`eq:diff_mod` becomes

$$
1=\mu^{-\gamma}\beta[(1-\tau_k)(f'(k)-\delta)+1] 
$$ (eq:diff_mod_st)

from which we can compute that the steady-state level of capital per unit of effective labor satisfies

$$
f'(k)=\delta + (\frac{\frac{1}{\beta}\mu^{\gamma}-1}{1-\tau_k})
$$ (eq:cap_mod_st)  

and that

$$
\bar{R}=\frac{\mu^{\gamma}}{\beta}
$$ (eq:Rbar_mod_st)

The steady-state level of consumption per unit of effective labor can be found using {eq}`eq:feasi_mod`:

$$
c = f(k)+(1-\delta-\mu)k-g
$$

Since the algorithm and plotting routines are the same as before, we include the steady-state calculations and 
shooting routine in the section {ref}`cass_fiscal_shooting`.

### Shooting Algorithm

Now we can apply the shooting algorithm to compute equilibrium. We augment the vector of shock variables by including $\mu_t$, then proceed as before.

### Experiments

Let's run some experiments:

1. A foreseen once-and-for-all increase in $\mu$ from 1.02 to 1.025 in period 10
2. An unforeseen once-and-for-all increase in $\mu$ to 1.025 in period 0

+++

#### Experiment 1: A foreseen increase in $\mu$ from 1.02 to 1.025 at t=10

The figures below show the effects of a permanent increase in productivity growth $\mu$ from 1.02 to 1.025 at t=10. 

They now measure $c$ and $k$ in effective units of labor.

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1),
    'μ': np.concatenate((np.repeat(1.02, 10), np.repeat(1.025, S - 9)))
}

A_path = compute_A_path(1.0, shocks, S)

k_ss_initial, c_ss_initial = steady_states(model, 
                                         shocks['g'][0],
                                         shocks['τ_k'][0],
                                         shocks['μ'][0]
                                        )

print(f"Steady-state capital: {k_ss_initial:.4f}")
print(f"Steady-state consumption: {c_ss_initial:.4f}")

# Run the shooting algorithm with the A_path parameter
solution = run_shooting(shocks, S, model, A_path)

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

plot_results(solution, k_ss_initial, 
             c_ss_initial, shocks, 'μ', axes, model, 
             A_path, T=40)

for ax in axes[5:]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()
```

The results in the figures are mainly driven by {eq}`eq:diff_mod_st`
and imply that a permanent increase in
$\mu$ will lead to a decrease in the steady-state value of capital per unit of effective
labor.

The figures indicate the following:

- As capital becomes more efficient, even with less of it, consumption per
capita can be raised.
- Consumption smoothing drives an *immediate jump in consumption* in anticipation of the increase in $\mu$.
- The increased productivity of capital leads to an increase in the gross return
$\bar R$. 
- Perfect foresight makes the effects of the increase in the growth of capital
precede it, with the effect visible at $t=0$.

#### Experiment 2: An unforeseen increase in $\mu$ from 1.02 to 1.025 at t=0

The figures below show the effects of an immediate jump in $\mu$ to 1.025 at t=0.

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1),
    'μ': np.concatenate((np.repeat(1.02, 1), np.repeat(1.025, S)))
}

A_path = compute_A_path(1.0, shocks, S)

k_ss_initial, c_ss_initial = steady_states(model, 
                                           shocks['g'][0],
                                           shocks['τ_k'][0],
                                           shocks['μ'][0]
                                          )

print(f"Steady-state capital: {k_ss_initial:.4f}")
print(f"Steady-state consumption: {c_ss_initial:.4f}")

# Run the shooting algorithm with the A_path parameter
solution = run_shooting(shocks, S, model, A_path)

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

plot_results(solution, k_ss_initial, 
             c_ss_initial, shocks, 'μ', axes, model, A_path, T=40)

for ax in axes[5:]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()
```

Again, we can collect the procedures used above into a function that runs the solver and draws plots for a given experiment.

```{code-cell} ipython3
def experiment_model(shocks, S, model, A_path, solver, plot_func, policy_shock, T=40):
    """
    Run the shooting algorithm given a model and plot the results.
    """
    k0, c0 = steady_states(model, shocks['g'][0], shocks['τ_k'][0], shocks['μ'][0])
    
    print(f"Steady-state capital: {k0:.4f}")
    print(f"Steady-state consumption: {c0:.4f}")
    print('-'*64)
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()

    solution = solver(shocks, S, model, A_path)
    plot_func(solution, k0, c0, 
              shocks, policy_shock, axes, model, A_path, T=T)

    for ax in axes[5:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
```

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1),
    'μ': np.concatenate((np.repeat(1.02, 1), np.repeat(1.025, S)))
}

experiment_model(shocks, S, model, A_path, run_shooting, plot_results, 'μ')
```

The figures show that:

- The paths of all variables are now smooth due to the absence of feedforward effects.
- Capital per effective unit of labor gradually declines to a lower steady-state level.
- Consumption per effective unit of labor jumps immediately and then declines smoothly toward its lower steady-state value.
- The after-tax gross return $\bar{R}$ once again co-moves with the consumption growth rate, verifying the Euler equation {eq}`eq:diff_mod_st`.

```{exercise}
:label: cass_fiscal_ex3

Replicate the plots of our two experiments using the second method of residual minimization:
1. A foreseen increase in $\mu$ from 1.02 to 1.025$ at t=10
2. An unforeseen increase in $\mu$ from 1.02 to 1.025$ at t=0
```

```{solution-start} cass_fiscal_ex3
:class: dropdown
```

Here is one solution:

**Experiment 1:  A foreseen increase in $\mu$ from 1.02 to 1.025 at $t=10$**

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1),
    'μ': np.concatenate((np.repeat(1.02, 10), np.repeat(1.025, S - 9)))
}

A_path = compute_A_path(1.0, shocks, S)

experiment_model(shocks, S, model, A_path, run_min, plot_results, 'μ')
```

**Experiment 2:  An unforeseen increase in $\mu$ from 1.02 to 1.025 at $t=0$**

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1),
    'μ': np.concatenate((np.repeat(1.02, 1), np.repeat(1.025, S)))
}

experiment_model(shocks, S, model, A_path, run_min, plot_results, 'μ')
```

```{solution-end}
```

## A two-country model

This section describes a two-country version of the basic model of {ref}`cs_fs_model`.

The model has a structure similar to ones used in the international real business cycle literature and is 
in the spirit of an analysis of distorting taxes by {cite:t}`mendoza1998international`.

We allow two countries to trade goods and claims on future goods, but not labor. 

Both countries have production technologies, and consumers in each country can hold capital in either country, subject to different tax treatments. 

We denote variables in the second country with asterisks (*).

Households in both countries maximize lifetime utility:

$$
\sum_{t=0}^{\infty} \beta^t u(c_t) \quad \text{and} \quad \sum_{t=0}^{\infty} \beta^t u(c_t^*),
$$

where $u(c) = \frac{c^{1-\gamma}}{1-\gamma}$ with $\gamma > 0$.

Production follows a Cobb-Douglas function with identical technology parameters across countries.

The global resource constraint for this two-country economy is:

$$
(c_t+c_t^*)+(g_t+g_t^*)+(k_{t+1}-(1-\delta)k_t)+(k_{t+1}^*-(1-\delta)k_t^*) = f(k_t)+f(k_t^*)
$$

which combines the feasibility constraints for the two countries. 

Later, we will use this constraint as a global feasibility constraint in our computation.

To connect the two countries, we need to specify how capital flows across borders and how taxes are levied in different jurisdictions.

### Capital Mobility and Taxation

A consumer in country one can hold capital in either country but pays taxes on rentals from foreign holdings of capital at the rate set by the foreign country. 

Residents in both countries can purchase consumption at date $t$ at a common Arrow-Debreu price $q_t$. We assume capital markets are complete.

Let $B_t^f$ be the amount of time $t$ goods that the representative domestic consumer raises by issuing a one-period IOU to the representative foreign consumer. 

So $B_t^f > 0$ indicates the domestic consumer is borrowing from abroad at $t$, and $B_t^f < 0$ indicates the domestic consumer is lending abroad at $t$.

Hence, the budget constraint of a representative consumer in country one is:

$$
\begin{aligned}
\sum_{t=0}^{\infty} q_t \left( c_t + (k_{t+1} - (1-\delta)k_t) + (\tilde{k}_{t+1} - (1-\delta)\tilde{k}_t) + R_{t-1,t}B_{t-1}^f \right) \leq \\
\sum_{t=0}^{\infty} q_t \left( (\eta_t - \tau_{kt}(\eta_t - \delta))k_t + (\eta_t^* - \tau_{kt}^*(\eta_t^* - \delta))\tilde{k}_t + (1 - \tau_{nt})w_t n_t - \tau_{ht} + B_t^f \right).
\end{aligned}
$$

No-arbitrage conditions for $k_t$ and $\tilde{k}_t$ for $t \geq 1$ imply

$$
\begin{aligned}
q_{t-1} &= [(1 - \tau_{kt})(\eta_t - \delta) + 1] q_t, \\
q_{t-1} &= [(1 - \tau^*_{kt})(\eta^*_t - \delta) + 1] q_t,
\end{aligned}
$$

which together imply that after-tax rental rates on capital are equalized across the two countries:

$$
(1 - \tau^*_{kt})(\eta^*_t - \delta) = (1 - \tau_{kt})(\eta_t - \delta).
$$

The no-arbitrage conditions for $B_t^f$ for $t \geq 0$ are $q_t = q_{t+1} R_{t+1,t}$, which implies that

$$
q_{t-1} = q_t R_{t-1,t}
$$

for $t \geq 1$.

Since domestic capital, foreign capital, and consumption loans bear the same rates of return, portfolios are indeterminate. 

We can set holdings of foreign capital equal to zero in each country if we allow $B_t^f$ to be nonzero. 

This way of resolving portfolio indeterminacy is convenient because it reduces the number of initial conditions we need to specify. 

Therefore, we set holdings of foreign capital equal to zero in both countries while allowing international lending.

Given an initial level $B_{-1}^f$ of debt from the domestic country to the foreign country, and where $R_{t-1,t} = \frac{q_{t-1}}{q_t}$, international debt dynamics satisfy

$$
B^f_t = R_{t-1,t} B^f_{t-1} + c_t + (k_{t+1} - (1 - \delta)k_t) + g_t - f(k_t)
$$

```{code-cell} ipython3
def Bf_path(k, c, g, model):
    """
    Compute B^{f}_t:
      Bf_t = R_{t-1} Bf_{t-1} + c_t + (k_{t+1}-(1-δ)k_t) + g_t - f(k_t)
    with Bf_0 = 0.
    """
    S = len(c) - 1                       
    R = c[:-1]**(-model.γ) / (model.β * c[1:]**(-model.γ))

    Bf = np.zeros(S + 1) 
    for t in range(1, S + 1):
        inv = k[t] - (1 - model.δ) * k[t-1]         
        Bf[t] = (
            R[t-1] * Bf[t-1] + c[t] + inv + g[t-1] 
            - f(k[t-1], model))
    return Bf

def Bf_ss(c_ss, k_ss, g_ss, model):
    """
    Compute the steady-state B^f
    """
    R_ss   = 1.0 / model.β  
    inv_ss = model.δ * k_ss 
    num    = c_ss + inv_ss + g_ss - f(k_ss, model)
    den    = 1.0 - R_ss
    return num / den
```

and

$$
c^*_t + (k^*_{t+1} - (1 - \delta)k^*_t) + g^*_t - R_{t-1,t} B^f_{t-1} = f(k^*_t) - B^f_t.
$$

The firms' first-order conditions in the two countries are:

$$
\begin{aligned}
\eta_t &= f'(k_t), \quad w_t = f(k_t) - k_t f'(k_t) \\
\eta^*_t &= f'(k^*_t), \quad w^*_t = f(k^*_t) - k^*_t f'(k^*_t).
\end{aligned}
$$

International trade in goods establishes:

$$
\frac{q_t}{\beta^t} = \frac{u'(c_t)}{1 + \tau_{ct}} = \mu^* \frac{u'(c^*_t)}{1 + \tau^*_{ct}},
$$

where $\mu^*$ is a nonnegative number that is a function of the Lagrange multiplier 
on the budget constraint for a consumer in country $*$. 

We have normalized the Lagrange multiplier on the budget constraint of the domestic country 
to set the corresponding $\mu$ for the domestic country to unity.

```{code-cell} ipython3
def compute_rs(c_t, c_tp1, c_s_t, c_s_tp1, τc_t, 
               τc_tp1, τc_s_t, τc_s_tp1, model):
    """
    Compute international risk sharing after trade starts.
    """

    return (c_t**(-model.γ)/(1+τc_t)) * ((1+τc_s_t)/c_s_t**(-model.γ)) - (
        c_tp1**(-model.γ)/(1+τc_tp1)) * ((1+τc_s_tp1)/c_s_tp1**(-model.γ))
```

Equilibrium requires that the following two national Euler equations be satisfied for $t \geq 0$:

$$
\begin{aligned}
u'(c_t) &= \beta u'(c_{t+1}) \left[ (1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1 \right] \left[ \frac{1 + \tau_{ct+1}}{1 + \tau_{ct}} \right], \\
u'(c^*_t) &= \beta u'(c^*_{t+1}) \left[ (1 - \tau^*_{kt+1})(f'(k^*_{t+1}) - \delta) + 1 \right] \left[ \frac{1 + \tau^*_{ct+1}}{1 + \tau^*_{ct}} \right].
\end{aligned}
$$

The following code computes both the domestic and foreign Euler equations.

Since they have the same form but use different variables, we can write a single function that handles both cases.

```{code-cell} ipython3
def compute_euler(c_t, c_tp1, τc_t, 
                    τc_tp1, τk_tp1, k_tp1, model):
    """
    Compute the Euler equation.
    """
    Rbar = (1 - τk_tp1)*(f_prime(k_tp1, model) - model.δ) + 1
    return model.β * (c_tp1/c_t)**(-model.γ) * (1+τc_t)/(1+τc_tp1) * Rbar - 1
```

### Initial condition and steady state

For the initial conditions, we choose the pre-trade allocation of capital ($k_0, k_0^*$) and the
initial level $B_{-1}^f$ of international debt owed
by the unstarred (domestic) country to the starred (foreign) country.

### Equilibrium steady state values

The steady state of the two-country model is characterized by two sets of equations.

First, the following equations determine the steady-state capital-labor ratios $\bar k$ and $\bar k^*$ in each country:

$$
f'(\bar{k}) = \delta + \frac{\rho}{1 - \tau_k}
$$ (eq:steady_k_bar)

$$
f'(\bar{k}^*) = \delta + \frac{\rho}{1 - \tau_k^*}
$$ (eq:steady_k_star)

Given these steady-state capital-labor ratios, the domestic and foreign consumption values $\bar c$ and $\bar c^*$ are determined by:

$$
(\bar{c} + \bar{c}^*) = f(\bar{k}) + f(\bar{k}^*) - \delta(\bar{k} + \bar{k}^*) - (\bar{g} + \bar{g}^*)
$$ (eq:steady_c_k_bar)

$$
\bar{c} = f(\bar{k}) - \delta\bar{k} - \bar{g} - \rho\bar{B}^f
$$ (eq:steady_c_kB)

Equation {eq}`eq:steady_c_k_bar` expresses feasibility at the steady state, while equation {eq}`eq:steady_c_kB` represents the trade balance, including interest payments, at the steady state.

The steady-state level of debt $\bar{B}^f$ from the domestic country to the foreign country influences the consumption allocation between countries but not the total world capital stock.

We assume $\bar{B}^f = 0$ in the steady state, which gives us the following function to compute the steady-state values of capital and consumption

```{code-cell} ipython3
def compute_steady_state_global(model, g_ss=0.2):
    """
    Calculate steady state values for capital, consumption, and investment.
    """
    k_ss = ((1/model.β - (1-model.δ)) / (model.A * model.α)) ** (1/(model.α-1))
    c_ss = f(k_ss, model) - model.δ * k_ss - g_ss
    return k_ss, c_ss
```

Now, we can apply the residual minimization method to compute the steady-state values of capital and consumption.

Again, we minimize the residuals of the Euler equation, the global resource constraint, and the no-arbitrage condition.

```{code-cell} ipython3
def compute_residuals_global(z, model, shocks, T, k0_ss, k_star, Bf_star):
    """
    Compute residuals for the two-country model.
    """
    k, c, k_s, c_s = z.reshape(T+1, 4).T
    g, gs = shocks['g'], shocks['g_s']
    τc, τk = shocks['τ_c'], shocks['τ_k']
    τc_s, τk_s = shocks['τ_c_s'], shocks['τ_k_s']
    
    res = [k[0] - k0_ss, k_s[0] - k0_ss]

    for t in range(T):
        e_d = compute_euler(
            c[t], c[t+1], 
            τc[t], τc[t+1], τk[t+1], 
            k[t+1], model)
        
        e_f = compute_euler(
            c_s[t], c_s[t+1], 
            τc_s[t], τc_s[t+1], τk_s[t+1], 
            k_s[t+1], model)
        
        rs = compute_rs(
            c[t], c[t+1], c_s[t], c_s[t+1], 
            τc[t], τc[t+1], τc_s[t], τc_s[t+1], 
            model)
        
        # Global resource constraint
        grc = k[t+1] + k_s[t+1] - (
            f(k[t], model) + f(k_s[t], model) +
            (1-model.δ)*(k[t] + k_s[t]) -
            c[t] - c_s[t] - g[t] - gs[t]
        )
        
        res.extend([e_d, e_f, rs, grc])

    Bf_term = Bf_path(k, c, shocks['g'], model)[-1]
    res.append(k[T] - k_star)
    res.append(Bf_term - Bf_star)
    return np.array(res)
```

Now we plot the results

```{code-cell} ipython3
# Function to plot global two-country model results
def plot_global_results(k, k_s, c, c_s, shocks, model, 
                        k0_ss, c0_ss, g_ss, S, T=40, shock='g',
                        # a dictionary storing sequence for lower left panel
                        ll_series='None'):
    """
    Plot results for the two-country model.
    """
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    x = np.arange(T)
    τc, τk = shocks['τ_c'], shocks['τ_k']
    Bf = Bf_path(k, c, shocks['g'], model)
    
    # Compute derived series
    R_ratio = c[:-1]**(-model.γ) / (model.β * c[1:]**(-model.γ)) \
    *(1+τc[:-1])/(1+τc[1:])
    inv = k[1:] - (1-model.δ)*k[:-1]
    inv_s = k_s[1:] - (1-model.δ)*k_s[:-1]

    # Add initial conditions into the series
    R_ratio = np.append(1/model.β, R_ratio)
    c = np.append(c0_ss, c)
    c_s = np.append(c0_ss, c_s)
    k = np.append(k0_ss, k)
    k_s = np.append(k0_ss, k_s)

    # Capital
    axes[0,0].plot(x, k[:T], '-', lw=1.5)
    axes[0,0].plot(x, np.full(T, k0_ss), 'k-.', lw=1.5)
    axes[0,0].plot(x, k_s[:T], '--', lw=1.5)
    axes[0,0].set_title('k')
    axes[0,0].set_xlim(0, T-1)
    
    # Consumption
    axes[0,1].plot(x, c[:T], '-', lw=1.5)
    axes[0,1].plot(x, np.full(T, c0_ss), 'k-.', lw=1.5)
    axes[0,1].plot(x, c_s[:T], '--', lw=1.5)
    axes[0,1].set_title('c')
    axes[0,1].set_xlim(0, T-1)
    
    # Interest rate
    axes[0,2].plot(x, R_ratio[:T], '-', lw=1.5)
    axes[0,2].plot(x, np.full(T, 1/model.β), 'k-.', lw=1.5)
    axes[0,2].set_title(r'$\bar{R}$')
    axes[0,2].set_xlim(0, T-1)
    
    # Investment
    axes[1,0].plot(x, np.full(T, model.δ * k0_ss), 
    'k-.', lw=1.5)
    axes[1,0].plot(x, np.append(model.δ*k0_ss, inv[:T-1]), 
    '-', lw=1.5)
    axes[1,0].plot(x, np.append(model.δ*k0_ss, inv_s[:T-1]), 
    '--', lw=1.5)
    axes[1,0].set_title('x')
    axes[1,0].set_xlim(0, T-1)
    
    # Shock
    axes[1,1].plot(x, shocks[shock][:T], '-', lw=1.5)
    axes[1,1].plot(x, np.full(T, shocks[shock][0]), 'k-.', lw=1.5)
    axes[1,1].set_title(f'${shock}$')
    axes[1,1].set_ylim(-0.1, 0.5)
    axes[1,1].set_xlim(0, T-1)
    
    # Capital flow
    axes[1,2].plot(x, np.append(0, Bf[1:T]), lw=1.5)
    axes[1,2].plot(x, np.zeros(T), 'k-.', lw=1.5)
    axes[1,2].set_title(r'$B^{f}$')
    axes[1,2].set_xlim(0, T-1)

    plt.tight_layout()
    return fig, axes
```

#### Experiment 1: A foreseen increase in $g$ from 0.2 to 0.4 at t=10

The figure below presents transition dynamics after an increase in $g$ in the domestic economy from 0.2 to 0.4 that is announced ten periods in advance.

We start both economies from a steady state with $B_0^f = 0$.

In the figure below, the blue lines represent the domestic economy and orange dotted lines represent the foreign economy.

```{code-cell} ipython3
Model = namedtuple("Model", ["β", "γ", "δ", "α", "A"])
model = Model(β=0.95, γ=2.0, δ=0.2, α=0.33, A=1.0)

shocks_global = {
    'g': np.concatenate((np.full(10, 0.2), np.full(S-9, 0.4))),
    'g_s': np.full(S+1, 0.2),
    'τ_c': np.zeros(S+1),
    'τ_k': np.zeros(S+1),
    'τ_c_s': np.zeros(S+1),
    'τ_k_s': np.zeros(S+1)
}
g_ss = 0.2
k0_ss, c0_ss = compute_steady_state_global(model, g_ss)

k_star = k0_ss
Bf_star = Bf_ss(c0_ss, k_star, g_ss, model)

init_glob = np.tile([k0_ss, c0_ss, k0_ss, c0_ss], S+1)
sol_glob = root(
    lambda z: compute_residuals_global(z, model, shocks_global,
                                        S, k0_ss, k_star, Bf_star),
    init_glob, tol=1e-12
)
k, c, k_s, c_s = sol_glob.x.reshape(S+1, 4).T

# Plot global results via function
plot_global_results(k, k_s, c, c_s,
                        shocks_global, model,
                        k0_ss, c0_ss, g_ss,
                        S)
plt.show()
```

At time 1, the government announces that domestic government purchases $g$ will rise ten periods later, cutting into future private resources.

To smooth consumption, domestic households immediately increase saving, offsetting the anticipated hit to their future wealth.

In a closed economy, they would save solely by accumulating extra domestic capital; with open capital markets, they can also lend to foreigners.

Once the capital flow opens up at time $1$, the no-arbitrage conditions connect adjustments of both types of saving: the increase in savings by domestic households will reduce the equilibrium return on bonds and capital in the foreign economy to prevent arbitrage opportunities.

Because no-arbitrage equalizes the ratio of marginal utilities, the resulting paths of consumption and capital are synchronized across the two economies.

Up to the date the higher $g$ takes effect, both countries continue to build their capital stocks.

When government spending finally rises 10 periods later, domestic households begin to draw down part of that capital to cushion consumption.

Again by no-arbitrage conditions, when $g$ actually increases, both countries reduce their investment rates.

The domestic economy, in turn, starts running current-account deficits partially to fund the increase in $g$.

This means that foreign households begin repaying part of their external debt by reducing their capital stock.


#### Experiment 2: A foreseen increase in $g$ from 0.2 to 0.4 at t=10

We now explore the impact of an increase in capital taxation in the domestic economy $10$ periods after its announcement at $t = 1$.

Because the change is anticipated, households in both countries adjust immediately—even though the tax does not take effect until period $t = 11$.

```{code-cell} ipython3
shocks_global = {
    'g': np.full(S+1, g_ss),
    'g_s': np.full(S+1, g_ss),
    'τ_c': np.zeros(S+1),
    'τ_k': np.concatenate((np.zeros(10), np.full(S-9, 0.2))),
    'τ_c_s': np.zeros(S+1),
    'τ_k_s': np.zeros(S+1),
}
    
k0_ss, c0_ss = compute_steady_state_global(model, g_ss)
k_star = k0_ss
Bf_star = Bf_ss(c0_ss, k_star, g_ss, model)

init_glob = np.tile([k0_ss, c0_ss, k0_ss, c0_ss], S+1)

sol_glob = root(
    lambda z: compute_residuals_global(z, model, 
            shocks_global, S, k0_ss, k_star, Bf_star),
            init_glob, tol=1e-12)

k, c, k_s, c_s = sol_glob.x.reshape(S+1, 4).T

# plot 
fig, axes = plot_global_results(k, k_s, c, c_s, shocks_global, model, 
                                k0_ss, c0_ss, g_ss, S, shock='τ_k')
plt.tight_layout()
plt.show()
```

After the tax increase is announced, domestic households foresee lower after-tax returns on capital, so they shift toward higher present consumption and allow the domestic capital stock to decline.

This shrinkage of the world capital supply drives the global real interest rate upward, prompting foreign households to raise current consumption as well.

Prior to the actual tax hike, the domestic economy finances part of its consumption by importing capital, generating a current-account deficit.

When $\tau_k$ finally rises, international arbitrage leads investors to reallocate capital quickly toward the untaxed foreign market, compressing the yield on bonds everywhere.

The bond-rate drop reflects the lower after-tax return on domestic capital and the higher foreign capital stock, which depresses its marginal product.

Foreign households fund their capital purchases by borrowing abroad, creating a pronounced current-account deficit and a buildup of external debt.

After the policy change, both countries move smoothly toward a new steady state in which:

  * Consumption levels in each economy settle below their pre-announcement paths.
  * Capital stocks differ just enough to equalize after-tax returns across borders.
  
Despite carrying positive net liabilities, the foreign country enjoys higher steady-state consumption because its larger capital stock yields greater output.

The episode demonstrates how open capital markets transmit a domestic tax shock internationally: capital flows and interest-rate movements share the burden, smoothing consumption adjustments in both the taxed and untaxed economies over time.

+++

```{exercise}
:label: cass_fiscal_ex4

In this exercise, replace the plot for ${x_t}$ with $\eta_t$ to replicate the figure in {cite}`Ljungqvist2012`.

Compare the figures for ${k_t}$ and $\eta_t$ and discuss the economic intuition.
```
```{solution-start} cass_fiscal_ex4
:class: dropdown
```

Here is one solution.

```{code-cell} ipython3
fig, axes = plot_global_results(k, k_s, c, c_s, shocks_global, model, 
                                k0_ss, c0_ss, g_ss, S, shock='τ_k')

# Clear the plot for x_t
axes[1,0].cla()

# Plot η_t
axes[1,0].plot(compute_η_path(k, model)[:40])
axes[1,0].plot(compute_η_path(k_s, model)[:40], '--')
axes[1,0].plot(np.full(40, f_prime(k_s, model)[0]), 'k-.', lw=1.5)
axes[1,0].set_title(r'$\eta$')

plt.tight_layout()
plt.show()
```

When capital ${k_t}$ decreases in the domestic country after the tax shock, the rental rate $\eta_t$ increases in that country.

This reflects that as capital becomes scarcer, its marginal product rises.

```{solution-end}
```
