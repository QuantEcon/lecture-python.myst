---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(kalman)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# A Second Look at the Kalman Filter

```{index} single: Kalman Filter 2
```

```{contents} Contents
:depth: 2
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

The purpose is to watch how the firm ends of learning a worker's "type" and how the worker's pay evolves as the firm learns. 

The $h_t$ part of the worker's "type" moves over time, but the effort type $u_t$ is fixed, so for this part the firm is in effect "learning a parameter".

To conduct simulations, we want to bring in these imports, as in the "first looks" lecture

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
import numpy as np
from quantecon import Kalman, LinearStateSpace
from collections import namedtuple
```

## A filtering example


A representative worker has productivity described by the following dynamic process:

```{math}
:label: worker_model

\begin{align}
h_{t+1} &= \alpha h_t + \beta u_t + c w_{t+1}, \quad c_{t+1} \sim {\mathcal N}(0,1) \\
u_{t+1} & = u_t \\
y_t & = g h_t + v_t , \quad v_t \sim {\mathcal N} (0, R)
\end{align}
```

Here 

* $h_t$ is the logarithm of human capital at time time $t$
* $u_t$ is the worker's effort at accumulating human capital at $t$ 
* $y_t$ is the worker's output at time $t$
* $h_0 \sim {\mathcal N}(\hat h_0, \sigma_{h,0})$
* $u_0 \sim {\mathcal N}(\hat u_0, \sigma_{u,0})$


At time $t\geq 1$, the  firm where the worker is permanently employed has observed  $y^{t-1} = [y_{t-1}, y_{t-2}, \ldots, y_0]$.

The firm does not observe the  worker's "type" $h_0, u_0$, but does observe his/her output $y_t$ at time $t$ before the worker gets paid at time
$t$. 

At time $t \geq 0$, the firm pays the worker log wage  

$$
w_t = g  E [ h_t | y_{t-1} ]
$$

Parameters of the model are $\alpha, \beta, c, R, g, \hat h_0, \hat u_0, \sigma_h, \sigma_u$.

First we create a `namedtuple` to store the parameters of the model

```{code-cell} ipython3
WorkerModel = namedtuple("WorkerModel", ('α', 'β', 'c', 'g', 'R',
                                        'hhat_0', 'uhat_0', 'σ_h', 'σ_u'))

def create_worker(α=0.5, β=0.3, c=0.2,
                    R=4, g=0.5, hhat_0=1, uhat_0=1, σ_h=1, σ_u=4):
    return WorkerModel(α=α, β=β, c=c, g=g, R=R, hhat_0=hhat_0,
                         uhat_0=uhat_0, σ_h=σ_h, σ_u=σ_u)
```

## Steps to get answer

We can write system [](worker_model) in the state-space form

```{math}
\begin{align}
\begin{bmatrix} h_{t+1} \cr u_{t+1} \end{bmatrix} &= \begin{bmatrix} \alpha & \beta \cr 0 & 1 \end{bmatrix}\begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} + \begin{bmatrix} c & 0 \end{bmatrix} w_{t+1} \cr
y_t & = \begin{bmatrix} g & 0 \end{bmatrix} \begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} + v_t
\end{align}
```

We can further summarize the system as

```{math}
\begin{align}
x_{t+1} & = A x_t + C w_{t+1} \cr
y_t & = G x_t + v_t \cr
x_0 & \sim {\mathcal N}(\hat x_0, \Sigma_0) \end{align}
```
where

```{math}
\begin{align}
x_t &= \begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} \cr
\hat x_0 & = \begin{bmatrix} \hat h_0 \cr \hat u_0 \end{bmatrix} \cr
\Sigma_0 & = \begin{bmatrix} \sigma_{h,0} & 0 \cr
                     0 & \sigma_{u,0} \end{bmatrix}
\end{align}
```

Here we compute them using [`LinearStateSpace`](https://quanteconpy.readthedocs.io/en/latest/tools/lss.html) class.

We simulate the system for $T = 100$ periods

```{code-cell} ipython3
# Define A, C, G
worker = create_worker()
A = np.array([[worker.α, worker.β], 
              [0, 1]])
C = np.array([[worker.c], 
              [0]])
G = np.array([worker.g, 0])

# Create LinearStateSpace instance with H = sqrt(R)
ss = LinearStateSpace(A, C, G, np.sqrt(worker.R))

T = 100
x, y = ss.simulate(T)
y = y.flatten()
```

We can now compute the Kalman filter for this system

Here we use the [`Kalman`](https://quanteconpy.readthedocs.io/en/latest/tools/kalman.html) class to compute the Kalman filter.

```{code-cell} ipython3
# Define initial state and covariance matrix
xhat_0 = np.array([[np.random.normal(worker.hhat_0, worker.σ_h)], 
                   [np.random.normal(worker.uhat_0, worker.σ_u)]])

print('initial state: \n', xhat_0)

Σ_0 = np.array([[worker.σ_h, 0],
                [0, worker.σ_u]])

print('initial covariance matrix: \n', Σ_0)

# Compute Kalman filter
kalman = Kalman(ss, xhat_0, Σ_0)
```

For a draw of $h_0, u_0$,  we plot $E y_t = G \hat x_t $ where $\hat x_t = E [x_t | y^{t-1}]$.

We also plot $E [u_0 | y^{t-1}]$, where the firm is inferring a worker's hard-wired "work ethic" $u_0$.

```{code-cell} ipython3
Σ_t = []
y_hat_t = np.zeros(T)
u_hat_t = np.zeros(T)

x_t_sum = 0
for i in range(T):
    kalman.update(y[i])
    x_hat, Σ = kalman.x_hat, kalman.Sigma
    Σ_t.append(Σ)
    y_hat_t[i] = G @ x_hat
    u_hat_t[i] = x_hat[1]
```

```{code-cell} ipython3
:tags: []

fig, ax = plt.subplots(1, 2)

ax[0].plot(y_hat_t, label=r'$E[y_t]$')
ax[0].set_xlabel('Time')
ax[0].set_ylabel(r'$E[y_t]$')
ax[0].set_title(r'$E[y_t]$ over time')

ax[1].plot(u_hat_t, label=r'$E[u_t|y^{t-1}]$')
ax[1].set_xlabel('Time')
ax[1].set_ylabel(r'$E[u_0|y^{t-1}]$')
ax[1].set_title(r'Inferred work ethic ($u_0$) over time')

fig.tight_layout()
plt.show()
```

Now we check the $\Sigma_0$ and $\Sigma_T$

```{code-cell} ipython3
:tags: []

print(Σ_t[0])
```

```{code-cell} ipython3
:tags: []

print(Σ_t[-1])
```

We can draw multiple initial points to see the trends converges after a long time $T=5000$

```{code-cell} ipython3
:tags: [hide-input]

def simulate_workers(ss, T, ax):
    x, y = ss.simulate(T)
    y = y.flatten()

    xhat_0 = np.array([[np.random.normal(worker.hhat_0, 
                                         worker.σ_h)], 
                       [np.random.normal(worker.uhat_0, 
                                         worker.σ_u)]])

    Σ_0 = np.array([[worker.σ_h, 0],
                    [0, worker.σ_u]])

    # Compute Kalman filter
    kalman = Kalman(ss, xhat_0, Σ_0)
    Σ_t = []
    
    y_hat_t = np.zeros(T)
    u_hat_t = np.zeros(T)

    x_t_sum = 0
    for i in range(T):
        kalman.update(y[i])
        x_hat, Σ = kalman.x_hat, kalman.Sigma
        Σ_t.append(Σ)
        y_hat_t[i] = G @ x_hat
        u_hat_t[i] = x_hat[1]
        
    ax[0].plot(y_hat_t, label=r'$E[y_t]$', alpha=0.5)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel(r'$E[y_t]$')
    ax[0].set_title(r'$E[y_t]$ over time')

    ax[1].plot(u_hat_t, label=r'$E[u_t|y^{t-1}]$', alpha=0.5)
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel(r'$E[u_0|y^{t-1}]$')
    ax[1].set_title(r'Inferred work ethic ($u_0$) over time')
```

```{code-cell} ipython3
:tags: []

iteration = 5
fig, ax = plt.subplots(1, 2)

for i in range(iteration):
    simulate_workers(ss, 5_000, ax)
    
fig.tight_layout()
plt.show()
```

## Coding steps:

(Note to Tom: A large proportion of the text in this section has been integrated into the previous sections. Please kindly review if there is anything else that we can move to the previous sections. Many thanks in advance.)

For given parameter values, write a program (a class?) to compute the following objects by using the Kalman class and 
consulting the "First Look ... " quantecon lecture:

* All of the objects in the "innovation representation"
    \begin{align}
    \hat x_{t+1} & = A \hat x_t + K_t a_t \cr
    y_{t} & = G \hat x_t + a_t
    \end{align}
    (The Kalman gain $K_t$ is computed in the Kalman class)
* Please also compute $\Sigma_t$ -- the conditional covariance matrix
    
* For a draw of $h_0, u_0$,  please prepare graphs of $E y_t = G \hat x_t $ where $\hat x_t = E [x_t | y^{t-1}]$. Please also prepare a graph of $E [u_t | y^{t-1}]$. (Here the firm is inferring a worker's hard-wired "work ethic" $u_0$.)
