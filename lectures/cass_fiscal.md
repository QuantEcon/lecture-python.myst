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

# Fiscal Policy Experiments in a Non-stochastic Model

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
# Create a named tuple to store the model parameters
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
The representative household has preferences over nonnegative streams of a single consumption good $c_t$:

$$
\sum_{t=0}^{\infty} \beta^t U(c_t), \quad \beta \in (0, 1)
$$ (eq:utility)
where  
- $U(c_t)$ is twice continuously differentiable, and strictly concave with $c_t \geq 0$.

under the budget constraint

$$
\begin{aligned}
    \sum_{t=0}^\infty& q_t \left\{ (1 + \tau_{ct})c_t + \underbrace{[k_{t+1} - (1 - \delta)k_t]}_{\text{no tax when investing}} \right\} \\
    &\leq \sum_{t=0}^\infty q_t \left\{ \tau_{kt} - \underbrace{\tau_{kt}(\eta_t - \delta)k_t}_{\text{tax on rental return}} + (1 - \tau_{nt})w_t n_t - \tau_{ht} \right\}.
\end{aligned}
$$ (eq:house_budget)


In our model, the representative household has the following CRRA preferences over consumption: 

$$
U(c) = \frac{c^{1 - \gamma}}{1 - \gamma}
$$

where $c$ is consumption and $\gamma$ is the coefficient of relative risk aversion. 


### Technology

The economy's production technology is defined by: 

$$
g_t + c_t + x_t \leq F(k_t, n_t),
$$ (eq:tech_capital)

- $g_t$ is government expenditure
- $x_t$ is gross investment, and 
- $F(k_t, n_t)$ a linearly homogeneous production function with positive and decreasing marginal products of capital $k_t$ and labor $n_t$.

The law of motion for capital is given by:
$$
      k_{t+1} = (1 - \delta)k_t + x_t,
$$

where 

- $\delta \in (0, 1)$ is depreciation rate, $k_t$ is the stock of physical capital, and $x_t$ is gross investment.


### Price System

A price system is a triple of sequences $\{q_t, \eta_t, w_t\}_{t=0}^\infty$, where

- $q_t$ is the time 0 pretax price of one unit of investment or consumption at time $t$ ($x_t$ or $c_t$),
- $\eta_t$ is the pretax price at time $t$ that the household receives from the firm for renting capital at time $t$,
- $w_t$ is the pretax price at time $t$ that the household receives for renting labor to the firm at time $t$.

The prices $w_t$ and $\eta_t$ are expressed in terms of time $t$ goods, while $q_t$ is expressed in terms of the numeraire at time 0.

### Government

Government plans $\{ g_t \}_{t=0}^\infty$ for government purchases and taxes $\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$ subject to the budget constraint

$$
\sum_{t=0}^\infty q_t g_t \leq \sum_{t=0}^\infty q_t \left\{ \tau_{ct}c_t + \tau_{kt}(\eta_t - \delta)k_t + \tau_{nt}w_t n_t + \tau_{ht} \right\}.
$$ (eq:gov_budget)

### Firm

Firms maximize their present value of profit:

$$
\sum_{t=0}^\infty q_t \left[ F(k_t, n_t) - w_t n_t - \eta_t k_t \right],
$$
Euler's theorem for linearly homogeneous functions states that if a function $F(k, n)$ is linearly homogeneous (degree 1), then:

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
    &+ \left[(1 - \tau_{k0})(\eta_0 - \delta) + 1\right]q_0k_0 - \lim_{T \to \infty} q_T r k_{T+1}
    \end{aligned}
$$ (eq:constrant_house)

By setting the terms multiplying $k_t$ to $0$ we have the non-arbitrage condition:

$$
\frac{q_t}{q_{t+1}} = \left[(1 - \tau_{kt})(\eta_t - \delta) + 1\right]
$$ (eq:no_arb)

Moreover, we have terminal condition

$$
-\lim_{T \to \infty} q_T r k_{T+1} = 0.
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
-\lim_{T \to \infty} \frac{U_{1t}}{(1 + \tau_{ct})} r k_{T+1} = 0.
$$ (eq:terminal_final)

$F(k_t, n_t) = A k_t^\alpha n_t^{1 - \alpha}$

$f'(k_t) = \alpha A k_t^{\alpha - 1}$

## Computing Equilibria

To compute an equilibrium we solve a price system $\{q_t, \eta_t, w_t\}$, a budget feasible government policy $\{g_t, \tau_t\} \equiv \{g_t, \tau_{ct}, \tau_{nt}, \tau_{kt}, \tau_{ht}\}$, and an allocation $\{c_t, n_t, k_{t+1}\}$ that solve the system of nonlinear difference equations consisting of 

- feasibility condition {eq}`eq:tech_capital`, no-arbitrage condition for household {eq}`eq:no_arb` and firms {eq}`eq:no_arb_firms`, household's first order conditions {eq}`eq:foc_c` and {eq}`eq:foc_n`.
- initial condition $k_0$, and terminal condition {eq}`eq:terminal_final`.

### Inelastic Labor Supply

First, we consider the special case where $U(c, 1-n) = u(c)$.

First we rewrite {eq}`eq:tech_capital` with $f(k) := F(k, 1)$,

$$
k_{t+1} = f(k_t) + (1 - \delta) k_t - g_t - c_t.
$$ (eq:feasi_capital)

By the properties of linearly homogeneous production function, we have $F_k(k, n) = f'(k)$, and $F_n(k, 1) = f(k, 1)  - f'(k)k$.

Substitute {eq}`eq:foc_c`, {eq}`eq:no_arb_firms`, and {eq}`eq:feasi_capital` into {eq}`eq:no_arb`, we have 

$$
\begin{aligned}
&\frac{u'(f(k_t) + (1 - \delta) k_t - g_t - k_{t+1})}{(1 + \tau_{ct})} \\
&- \beta \frac{u'(f(k_{t+1}) + (1 - \delta) k_{t+1} - g_{t+1} - k_{t+2})}{(1 + \tau_{ct+1})} \\
&\times [(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1] = 0. 
\end{aligned}
$$

which can be written as 

$$
\begin{aligned}
u'(c_t) = \beta u'(c_{t+1}) \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} [(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1]. 
\end{aligned}
$$ (eq:diff_second)

which is the Euler equation for the household.

This equation can be used to solve for the equilibrium sequence of consumption and capital as we will see in the second method.

### Steady state

Tax rates and government expenditures serve as forcing functions for the difference equations {eq}`eq:feasi_capital` and {eq}`eq:diff_second`. 

Let $z_t = [g_t, \tau_{kt}, \tau_{ct}]'$. We can write the second-order difference equation into 

$$
H(k_t, k_{t+1}, k_{t+2}; z_t, z_{t+1}) = 0.
$$ (eq:second_ord_diff)

We assume that the government policy is in the steady state satisfying $\lim_{t \to \infty} z_t = \bar z$. We assume the steady state is reached for $t > T$. A terminal  steady-state capital stock $\bar k$ solves 

$$H(\bar{k}, \bar{k}, \bar{k}, \bar{z}, \bar{z}) = 0$$

Hencer we can derive the steady-state from the difference equation {eq}`eq:diff_second`

$$
\begin{aligned}
u'(\bar c) &= \beta u'(\bar c) \frac{(1 + \bar \tau_{c})}{(1 + \bar \tau_{c})} [(1 - \bar \tau_{k})(f'(\bar k) - \delta) + 1]. \\
&\implies 1 = \beta[(1 - \bar \tau_{k})(f'(\bar k) - \delta) + 1]
\end{aligned}
$$ (eq:diff_second_steady)


### Other equilibrium quantities

**Consumption**
$$
c_t = f(k_t) + (1 - \delta)k_t - k_{t+1} - g_t  
$$ (eq:equil_c)

**Price of the good:**
$$
q_t = \beta^t \frac{u'(c_t)}{1 + \tau_{ct}}
$$ (eq:equil_q)

**Marginal product of capital**
$$
\eta_t = f'(k_t)  
$$

**Wage:**
$$
w_t = f(k_t) - k_t f'(k_t)    
$$

**Gross one-period return on capital:**
$$
\bar{R}_{t+1} = \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1\right] =  \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} R_{t, t+1}
$$ (eq:gross_rate)

**One-period discount factor:**
$$
R^{-1}_{t, t+1} = \frac{q_t}{q_{t-1}} = m_{t, t+1} = \beta \frac{u'(c_{t+1})}{u'(c_t)} \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})}
$$ (eq:equil_R)

**Net one-period rate of interest:**
$$
r_{t, t+1} \equiv R_{t, t+1} - 1 = (1 - \tau_{k, t+1})(f'(k_{t+1}) - \delta)
$$ (eq:equil_r)

### Shooting Algorithm

In the following sections, we will experiment apply two methods to solve the model: shooting algorithm and minimization of Euler residual and law of motion capital.

### Experiments

+++

We will do a number of experiments and analyze the transition path for the equilibrium in each case:

1. A foreseen once-and-for-all increase in $g$ from 0.2 to 0.4 in period 10.
2. A foreseen once-and-for-all increase in $\tau_c$ from 0.0 to 0.2 in period 10.
3. A foreseen once-and-for-all increase in $\tau_k$ from 0.0 to 0.2 in period 10.
3. A foreseen one-time increase in $g$ from 0.2 to 0.4 in period 10, after which $g$ returns to 0.2 forever

+++

Below we write the formulas for the shooting algorithm:

$u(c) = \frac{c^{1 - \gamma}}{1 - \gamma}$

$u'(c_t) = \beta u'(c_{t+1}) \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[ (1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1 \right]$

$R_{t,t+1} = \left[ (1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1 \right] \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})}$

$k_{t+1} = f(k_t) + (1 - \delta)k_t - g_t - c_t$

$u'(c_t) = \beta R_{t,t+1} u'(c_{t+1}) \iff c_t^{-\gamma} = \beta R_{t,t+1} c_{t+1}^{-\gamma} \iff c_{t+1} = c_t \left( \beta R_{t,t+1} \right)^{1/\gamma}$

```{code-cell} ipython3
def u_prime(c, γ):
    """
    Marginal utility: u'(c) = c^{-γ}
    """
    return c ** (-γ)

def f(A, k, α): 
    """
    Production function: f(k) = A * k^{α}
    """
    return A * k ** α

def f_prime(A, k, α):
    """
    Marginal product of capital: f'(k) = α * A * k^{α - 1}
    """
    return α * A * k ** (α - 1)

def compute_R_bar(A, τ_ct, τ_ctp1, τ_ktp1, k_tp1, α, δ):
    """
    Gross one-period return on capital:
    R̄ = [(1 + τ_c_t) / (1 + τ_c_{t+1})] * { [1 - τ_k_{t+1}] * [f'(k_{t+1}) - δ] + 1 }
    """
    return ((1 + τ_ct) / (1 + τ_ctp1)) * (
        (1 - τ_ktp1) * (f_prime(A, k_tp1, α) - δ) + 1)

def next_k(A, k_t, g_t, c_t, α, δ):
    """
    Capital next period: k_{t+1} = f(k_t) + (1 - δ) * k_t - c_t - g_t
    """
    return f(A, k_t, α) + (1 - δ) * k_t - g_t - c_t

def next_c(c_t, R_bar, γ, β):
    """
    Consumption next period: c_{t+1} = c_t * (β * R̄)^{1/γ}
    """
    return c_t * (β * R_bar) ** (1 / γ)

# Compute other equilibrium quantities
def compute_R_bar_path(shocks, k_path, model, S):
    """
    Compute R̄ path over time.
    """
    A, α, δ = model.A, model.α, model.δ
    R_bar_path = np.zeros(S + 1)
    for t in range(S):
        R_bar_path[t] = compute_R_bar(
            A,
            shocks['τ_c'][t], shocks['τ_c'][t + 1], shocks['τ_k'][t + 1],
            k_path[t + 1],
            α, δ
        )
    R_bar_path[S] = R_bar_path[S - 1]
    return R_bar_path

def compute_η_path(k_path, α, A, S=100):
    """
    Compute η path: η_t = f'(k_t) = α * A * k_t^{α - 1}
    """
    η_path = np.zeros_like(k_path)
    for t in range(S):
        η_path[t] = α * A * k_path[t] ** (α - 1)
    return η_path

def compute_w_path(k_path, η_path, α, A, S=100):
    """
    Compute w path: w_t = f(k_t) - k_t * f'(k_t)
    """
    w_path = np.zeros_like(k_path)
    for t in range(S):
        w_path[t] = A * k_path[t] ** α - k_path[t] * η_path[t]
    return w_path

def compute_q_path(c_path, β, γ, S=100):
    """
    Compute q path: q_t = (β^t * u'(c_t)) / u'(c_0)
    """
    q_path = np.zeros_like(c_path)
    for t in range(S):
        q_path[t] = (β ** t * u_prime(c_path[t], γ)) / u_prime(c_path[0], γ)
    return q_path

def compute_rts_path(q_path, T, t):
    """
    Compute r path:
    r_t(s) = - (1/s) * ln(q_{t+s} / q_t)
    """
    s = np.arange(T)
    q_path = np.array([float(q) for q in q_path])

    with np.errstate(divide='ignore', invalid='ignore'):
        rts_path = - np.log(q_path[t + s] / q_path[t]) / s
    return rts_path

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
```

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
    
    η_path = compute_η_path(k_path, model.α, model.A, S=T)
    η_ss = model.α * model.A * k_ss ** (model.α - 1)
    
    axes[3].plot(η_path[:T], linestyle=linestyle, label=label)
    axes[3].axhline(η_ss, linestyle='--', color='black')
    axes[3].set_title(r'$\eta$')
    
    axes[4].plot(shocks[shock_param][:T], linestyle=linestyle, label=label)
    axes[4].axhline(shocks[shock_param][0], linestyle='--', color='black')
    axes[4].set_title(rf'${shock_param}$')
```

```{code-cell} ipython3
def shooting_algorithm(c0, k0, shocks, S, model):
    """
    Shooting algorithm for given initial c0 and k0.
    """
    # High-precision parameters
    β, γ, δ, α, A = map(mpf, (model.β, model.γ, model.δ, model.α, model.A))
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
        k_tp1 = next_k(A, k_t, g_t, c_t, α, δ)
        # Failure due to negative capital
        if k_tp1 < mpf(0):
            return None, None 
        k_path[t + 1] = k_tp1

        # Calculate next period's consumption
        R_bar = compute_R_bar(A, τ_c_path[t], τ_c_path[t + 1], 
                              τ_k_path[t + 1], k_tp1, α, δ)
        c_tp1 = next_c(c_t, R_bar, γ, β)
        # Failure due to negative consumption
        if c_tp1 < mpf(0):
            return None, None
        c_path[t + 1] = c_tp1

    return k_path, c_path


def bisection_c0(k0, c0_guess, shocks, S, model, 
                 tol=mpf('1e-6'), max_iter=1000, verbose=False):
    """
    Bisection method to find optimal initial consumption c0.
    """
    β, γ, δ, α, A = map(mpf, (model.β, model.γ, model.δ, model.α, model.A))
    k_ss_final, _ = steady_states(model, 
                                  mpf(shocks['g'][-1]), mpf(shocks['τ_k'][-1]))
    c0_lower, c0_upper = mpf(0), A * k_ss_final ** α

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

def run_shooting(shocks, S, model, c0_function=bisection_c0, shooting_func=shooting_algorithm):
    """
    Runs the shooting algorithm.
    """
    # Compute initial steady states
    k0, c0 = steady_states(model, mpf(shocks['g'][0]), mpf(shocks['τ_k'][0]))
    
    # Find the optimal initial consumption
    optimal_c0 = c0_function(k0, c0, shocks, S, model)
    print(f"Parameters: {model}")
    print(f"Optimal initial consumption c0: {mp.nstr(optimal_c0, 7)} \n")
    
    # Simulate the model
    k_path, c_path = shooting_func(optimal_c0, k0, shocks, S, model)
    
    # Combine and return the results
    return np.column_stack([k_path, c_path])
```

## Experiments with Shooting Algorithm

### Experiment 1: Foreseen once-and-for-all increase in $g$ from 0.2 to 0.4 in period 10.

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
def experiment_model(shocks, S, model, solver, plot_func, policy_shock, T=40):
    """
    Plots the results of running the shooting algorithm given a model
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
solution = run_shooting(shocks, S, model)
k_ss_initial, c_ss_initial = steady_states(model, 
                                           shocks['g'][0], 
                                           shocks['τ_k'][0])
fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

label = fr"$\gamma = {model.γ}$"
plot_results(solution, k_ss_initial, c_ss_initial, 
             shocks, 'g', axes, model, label=label, 
             T=40)

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
def experiment_two_models(shocks, S, model_1, model_2, solver, plot_func, 
                          policy_shock, legend_label_fun=None, T=40):
    """
    Compares and plots results of the shooting algorithm for two models.
    """
    
    k0, c0 = steady_states(model, shocks['g'][0], shocks['τ_k'][0])
    print(f"Steady-state capital: {k0:.4f}")
    print(f"Steady-state consumption: {c0:.4f}")
    print('-'*64)
    
    # Use a default labeling function if none is provided
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
    
    q_path = compute_q_path(c_path, 
                            β, γ, S=S)
    axes[1].plot(q_path[:T], linestyle=linestyle, label=label)
    axes[1].plot(β**np.arange(T), linestyle='--', color='black')
    axes[1].set_title('q')
    
    R_bar_path = compute_R_bar_path(shocks, k_path, model, S)
        
    axes[2].plot(R_bar_path[:T] - 1, linestyle=linestyle, label=label)
    axes[2].axhline(1 / β - 1, linestyle='--', color='black')
    axes[2].set_title('$r_{t,t+1}$')

    
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

### Experiment 2: Foreseen once-and-for-all increase in $\tau_c$ from 0.0 to 0.2 in period 10.

The experiment replicates the Figure 12.9.4.

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, run_shooting, plot_results, 'τ_c')
```

### Experiment 3: Foreseen once-and-for-all increase in $\tau_k$ from 0.0 to 0.2 in period 10.

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

### Experiment 4: Foreseen one-time increase in $g$ from 0.2 to 0.4 in period 10, after which $g$ returns to 0.2 forever

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

## Method 2: Minimization of Euler Residual and Law of Motion Capital


$$1 = \beta \left(\frac{c_{t+1}}{c_t}\right)^{-\gamma} \frac{(1+\tau_{ct})}{(1+\tau_{ct+1})} \left[(1 - \tau_{kt+1})(\alpha A k_{t+1}^{\alpha-1} - \delta) + 1 \right]$$

and a law of motion for capital

$$k_{t+1} = A k_{t}^{\alpha} + (1 - \delta) k_t - g_t - c_t.$$

+++

### Algorithm for minimization approach (Method 2)

1. **Calculate initial state $k_0$**:
   - Based on the given initial government plan $z_0$.

2. **Initialize a sequence of initial guesses** $\{\hat{c}_t, \hat{k}_t\}_{t=0}^{S}$

3. **Compute the residuals** $R_a$, $R_k$ for $t = 0, \dots, S$, and $R_{tk_0}$ for $t=0$:
   - **Arbitrage condition** residual for $t = 0, \dots, S$:
     $$
     R_{ta} = \beta u'(c_{t+1}) \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1 \right] - u'(c_t)
     $$
   - **Feasibility condition** residual for $t = 1, \dots, S-1$:
     $$
     R_{tk} = k_{t+1} - f(k_t) + (1 - \delta)k_t - g_t - c_t
     $$
   - **Initial condition for $k_0$**:
     $$
     R_{k_0} = 1 - \beta \left[ (1 - \tau_{k0}) \left(f'(k_0) - \delta \right) + 1 \right]
     $$

4. **Root-finding**:
   - Adjust the guesses for $k_t$ and $c_t$ to minimize the residuals $R_{k_0}$, $R_{ta}$, and $R_{tk}$ for $t = 0, \dots, S$.

5. **Output**:
   - The solution $\{c_t, k_t\}_{t=0}^{S}$.

```{code-cell} ipython3
# Arbitrage and Transition Equations for method 2
def arbitrage(c_t, c_tp1, τ_c_t, τ_c_tp1, τ_k_tp1, k_tp1, model):
    """
    Computes the arbitrage condition.
    """
    β, γ, δ, α, A = model.β, model.γ, model.δ, model.α, model.A
    η_tp1 = α * A * k_tp1 ** (α - 1)
    return β * (c_tp1 / c_t) ** (-γ) * (1 + τ_c_t) / (1 + τ_c_tp1) * (
        (1 - τ_k_tp1) * (η_tp1 - δ) + 1) - 1

def transition(k_t, k_tm1, c_tm1, g_t, model):
    """
    Computes the capital transition.
    """
    α, A, δ = model.α, model.A, model.δ
    return k_t - (A * k_tm1 ** α + (1 - δ) * k_tm1 - c_tm1 - g_t)

# Residuals for method 2
def compute_residuals(vars_flat, k_init, S, shocks, model):
    """
    Compute residuals for arbitrage and transition equations.
    """
    k, c = vars_flat.reshape((S + 1, 2)).T
    residuals = np.zeros(2 * S + 2)

    # Initial condition for capital
    residuals[0] = k[0] - k_init

    # Compute residuals for each time step
    for t in range(S):
        residuals[2 * t + 1] = arbitrage(
            c[t], c[t + 1],
            shocks['τ_c'][t], shocks['τ_c'][t + 1], shocks['τ_k'][t + 1],
            k[t + 1], model
        )
        residuals[2 * t + 2] = transition(
            k[t + 1], k[t], c[t],
            shocks['g'][t], model
        )

    # Terminal condition for arbitrage
    residuals[-1] = arbitrage(
        c[S], c[S],
        shocks['τ_c'][S], shocks['τ_c'][S], shocks['τ_k'][S],
        k[S], model
    )
    
    return residuals

# Root-finding Algorithm to minimize the residual
def run_min(shocks, S, model):
    """
    Root-finding algorithm to minimize the residuals.
    """
    k_ss, c_ss = steady_states(model, shocks['g'][0], shocks['τ_k'][0])
    
    # Initial guess for the solution path
    initial_guess = np.column_stack(
        (np.full(S + 1, k_ss), np.full(S + 1, c_ss))).flatten()

    # Solve the system using root-finding
    sol = root(compute_residuals, initial_guess, args=(k_ss, S, shocks, model), tol=1e-8)

    # Reshape solution to get time paths for k and c
    return sol.x.reshape((S + 1, 2))
```

Below are the results for the same experiments using the method of minimization of Euler residual and law of motion capital.

This method does not have numerical stability issues so `mp.mpf` is not necessary.

### Experiment 1: Foreseen once-and-for-all increase in $g$ from 0.2 to 0.4 in period 10.

The experiment replicates the Figure 12.9.1 in RMT5 under $\gamma = 2$.

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

### Experiment 2: Foreseen once-and-for-all increase in $\tau_c$ from 0.0 to 0.2 in period 10.

The experiment replicates the Figure 12.9.4.

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, run_min, plot_results, 'τ_c')
```

### Experiment 3: Foreseen once-and-for-all increase in $\tau_k$ from 0.0 to 0.2 in period 10.

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

### Experiment 4: Foreseen one-time increase in $g$ from 0.2 to 0.4 in period 10, after which $g$ returns to 0.2 forever

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
