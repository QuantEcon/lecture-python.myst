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

This is a sequel to this quantecon lecture   {doc}`A First Look at the Kalman filter <kalman>`.

Instead of using a Kalman filter to  track a rocket as we did in that lecture, here we'll use it 
make inferences about a worker's  human capital and a worker's  effort input into accumulating and maintaining
human capital, both of which are unobserved to a firm that learns about those things only be observing a history
of the output that the worker generates for the firm.

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

To conduct simulations, we want to bring in these imports, as in the "first looks" lecture

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
import numpy as np
from quantecon import Kalman, LinearStateSpace
from collections import namedtuple
```

## A worker's output 


A representative worker output at a firm where he or she is permanently employed is  described by the following dynamic process:

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
* $u_t$ is the logarithm of the worker's effort at accumulating human capital at $t$ 
* $y_t$ is the logarithm of the worker's output at time $t$
* $h_0 \sim {\mathcal N}(\hat h_0, \sigma_{h,0})$
* $u_0 \sim {\mathcal N}(\hat u_0, \sigma_{u,0})$

Parameters of the model are $\alpha, \beta, c, R, g, \hat h_0, \hat u_0, \sigma_h, \sigma_u$.

At time $0$, a firm has hired the worker.

The worker is permanently attached to the firm and so works for the  firm at dates $t =0, 1, 2, \ldots$.

At the beginning of time $0$, the firm observes neither the worker's innate initial human capitl $h_0$ nor its hard-wired permanent effort level $u_0$.



The $h_t$ part of the worker's "type" moves over time, but the effort type $u_t = u_0$, so it in effectively a fixed ``parameter'' that the firm does not know.


At time $t\geq 1$, the  firm where the worker is permanently employed has observed  $y^{t-1} = [y_{t-1}, y_{t-2}, \ldots, y_0]$.

The firm does not observe the  worker's "type" $h_0, u_0$.

But the firm  does observe the worker's  output $y_t$ at time $t$ and remembers the worker's past outputs $y^{t-1}$.




## A firm's wage-setting policy



Based on   information about the worker that the firm has at time $t \geq 1$, the firm pays the worker log wage  

$$
w_t = g  E [ h_t | y^{t-1} ], \quad t \geq 1
$$

and at time $0$ pays the  worker a log wage equal to  the unconditional mean of $y_0$:

$$
w_0 = g \hat h_0 . 
$$



## Forming a state-space representation

We write system [](worker_model) in the state-space form

```{math}
\begin{align}
\begin{bmatrix} h_{t+1} \cr u_{t+1} \end{bmatrix} &= \begin{bmatrix} \alpha & \beta \cr 0 & 1 \end{bmatrix}\begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} + \begin{bmatrix} c \cr 0 \end{bmatrix} w_{t+1} \cr
y_t & = \begin{bmatrix} g & 0 \end{bmatrix} \begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} + v_t
\end{align}
```

which is equivalent with

```{math}
\begin{align}
x_{t+1} & = A x_t + C w_{t+1} \cr
y_t & = G x_t + v_t \cr
x_0 & \sim {\mathcal N}(\hat x_0, \Sigma_0) \end{align}
```
where

```{math}
\begin{equation}
x_t  = \begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} , \quad
\hat x_0  = \begin{bmatrix} \hat h_0 \cr \hat u_0 \end{bmatrix} , \quad
\Sigma_0  = \begin{bmatrix} \sigma_{h,0} & 0 \cr
                     0 & \sigma_{u,0} \end{bmatrix}
\end{equation}
```

To prepare for computing the firm's wage setting policy, we first we create a `namedtuple` to store the parameters of the model

```{code-cell} ipython3
WorkerModel = namedtuple("WorkerModel", ('A', 'C', 'G', 'R', 'xhat_0', 'Σ_0'))

def create_worker(α=.8, β=.2, c=.2,
                  R=.5, g=1.0, hhat_0=4, uhat_0=4, 
                  σ_h=4, σ_u=4):
    
    A = np.array([[α, β], 
                  [0, 1]])
    C = np.array([[c], 
                  [0]])
    G = np.array([g, 1])

    # Define initial state and covariance matrix
    xhat_0 = np.array([[hhat_0], 
                       [uhat_0]])
    
    Σ_0 = np.array([[σ_h, 0],
                    [0, σ_u]])
    
    return WorkerModel(A=A, C=C, G=G, R=R, xhat_0=xhat_0, Σ_0=Σ_0)
```

Now we  form the state space system we want by using the [`LinearStateSpace`](https://quanteconpy.readthedocs.io/en/latest/tools/lss.html) class.

Let's simulate a worker  for $T = 100$ periods

```{code-cell} ipython3
# Define A, C, G, R, xhat_0, Σ_0
worker = create_worker()
A, C, G, R = worker.A, worker.C, worker.G, worker.R
xhat_0, Σ_0 = worker.xhat_0, worker.Σ_0

# Create a LinearStateSpace object
ss = LinearStateSpace(A, C, G, np.sqrt(R), mu_0=xhat_0, Sigma_0=Σ_0)

T = 100
x, y = ss.simulate(T)
y = y.flatten()

h_0, u_0 = x[0, 0], x[1, 0]
print('h_0 =', h_0)
print('u_0 =', u_0)
```

To compute the firm's policy for setting the log wage given the information it has about the worker,
we want to use the Kalman filter described in this quantecon lecture   {doc}`A First Look at the Kalman filter <kalman>`.

Thus, we want to compute all of the objects in the "innovation representation"

```{math}
    \begin{align}
    \hat x_{t+1} & = A \hat x_t + K_t a_t \cr
    y_{t} & = G \hat x_t + a_t
    \end{align}
```
where $K_t$ is the Kalman gain matrix at time $t$.


We accomplish this by using the [`Kalman`](https://quanteconpy.readthedocs.io/en/latest/tools/kalman.html) class.

```{code-cell} ipython3
kalman = Kalman(ss, xhat_0, Σ_0)
Σ_t = []
y_hat_t = np.zeros(T-1)
u_hat_t = np.zeros(T-1)

for t in range(1, T):
    kalman.update(y[t])
    x_hat, Σ = kalman.x_hat, kalman.Sigma
    Σ_t.append(Σ)
    y_hat_t[t-1] = worker.G @ x_hat
    u_hat_t[t-1] = x_hat[1]
```

For a draw of $h_0, u_0$,  we plot $E y_t = G \hat x_t $ where $\hat x_t = E [x_t | y^{t-1}]$.

We also plot $E [u_0 | y^{t-1}]$, which is  the firm inference about  a worker's hard-wired "work ethic" $u_0$, conditioned on information $y^{t-1}$ that it has about him or her coming into period $t$.

We can watch as the the firm's inference of the worker's work ethic $E [u_0 | y^{t-1}]$ converges toward the hidden  (to the firm) value $u_0$.

```{code-cell} ipython3
:tags: []

fig, ax = plt.subplots(1, 2)

ax[0].plot(y_hat_t, label=r'$E[y_t| y^{t-1}]$')
ax[0].set_xlabel('Time')
ax[0].set_ylabel(r'$E[y_t]$')
ax[0].set_title(r'$E[y_t]$ over time')
ax[0].legend()

ax[1].plot(u_hat_t, label=r'$E[u_t|y^{t-1}]$')
ax[1].axhline(y=u_0, color='grey', linestyle='dashed', label=fr'$u_0={u_0:.2f}$')
ax[1].set_xlabel('Time')
ax[1].set_ylabel(r'$E[u_t|y^{t-1}]$')
ax[1].set_title('Inferred work ethic over time')
ax[1].legend()

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

NEW REQUEST FOR HUMPHREY AND SMIT:

HUMPHREY AND/OR SMIT: YOU HAVE DONE A WONDERFUL JOB.  AS A ``REWARD'' FOR YOUR EXCELLENT WORK, I'D LIKE TO ASK YOU TO TWEAK YOUR CODE TO ALLOW US TO DO THE FOLLOWING THINGS:

* LET ME ARBITRARILY SET THE WORKER'S INITIAL $h_0, u_0$ PAIR INSTEAD OF DRAWING IT FROM THE INITIAL DISTRIBUTION THAT THE FIRM HAS IN ITS HEAD.  THAT WILL LET ME GENERATE SOME PATHS WITH HIDDEN STATES AT SET VALUES THAT I ARBITRARILY PUT AT VARIOUS SPOTS IN THE PRIOR DISTRIBUTION OF THESE TWO OBJECTS. IT WILL HELP ME GENERATE SOME INTERESTING GRAPHS.

```{code-cell} ipython3
:tags: []

# For example, we might want h_0 = 3 and u_0 = 4
mu_0 = np.array([3.0, 4.0])

# Create a LinearStateSpace object with Sigma_0 as a matrix of zeros
ss_example = LinearStateSpace(A, C, G, np.sqrt(R), mu_0=mu_0, 
                              Sigma_0=np.zeros((2, 2)) # This line forces exact h_0=3 and u_0=4
                             )

T = 100
x, y = ss_example.simulate(T)
y = y.flatten()

# Now h_0 and u_0 will be exactly hhat_0
h_0, u_0 = x[0, 0], x[1, 0]
print('h_0 =', h_0)
print('u_0 =', u_0)

# This shows h_t and u_t over time.
print(x)
```

```{code-cell} ipython3
:tags: []

# If we want to set the initial 
# h_0 = hhat_0 = 7 and u_0 = uhhat_0 = 3.5:
worker = create_worker(hhat_0=7.0, uhat_0=3.5)

ss_example = LinearStateSpace(A, C, G, np.sqrt(R), 
                              mu_0=worker.xhat_0, # This line takes h_0=hhat_0 and u_0 = uhhat_0
                              Sigma_0=np.zeros((2, 2)) # This line forces exact h_0=hhat_0 and u_0=uhhat_0
                             )

T = 100
x, y = ss_example.simulate(T)
y = y.flatten()

# Now h_0 and u_0 will be exactly hhat_0
h_0, u_0 = x[0, 0], x[1, 0]
print('h_0 =', h_0)
print('u_0 =', u_0)

# This shows h_t and u_t over time.
print(x)
```

```{code-cell} ipython3
:tags: []

# Code to generate a plot like figure 1

# First we get the Kalman filter with initial xhat_0 and Σ_0 
kalman = Kalman(ss, xhat_0, Σ_0)
Σ_t = []
y_hat_t = np.zeros(T-1)
u_hat_t = np.zeros(T-1)

# Then we iteratively update the kalman filter class using 
# observation y based on the linear state model above:
for t in range(1, T):
    kalman.update(y[t])
    x_hat, Σ = kalman.x_hat, kalman.Sigma
    Σ_t.append(Σ)
    y_hat_t[t-1] = worker.G @ x_hat
    u_hat_t[t-1] = x_hat[1]


# Generate plots for y_hat_t and u_hat_t
fig, ax = plt.subplots(1, 2)

ax[0].plot(y_hat_t, label=r'$E[y_t| y^{t-1}]$')
ax[0].set_xlabel('Time')
ax[0].set_ylabel(r'$E[y_t]$')
ax[0].set_title(r'$E[y_t]$ over time')
ax[0].legend()

ax[1].plot(u_hat_t, label=r'$E[u_t|y^{t-1}]$')
ax[1].axhline(y=u_0, color='grey', linestyle='dashed', label=fr'$u_0={u_0:.2f}$')
ax[1].set_xlabel('Time')
ax[1].set_ylabel(r'$E[u_t|y^{t-1}]$')
ax[1].set_title('Inferred work ethic over time')
ax[1].legend()

fig.tight_layout()
plt.show()

# Code to generate a plot like figure 2 is shown below
```

* TEACH ME HOW TO GENERATE WORKERS CHARACTERIZED BY DIFFERENT PARAMETER VECTORS, I.E., DIFFERENT VALUES OF $\alpha, \beta$ AND SO ON.  THAT WILL ALLOW US TO DO SOME EXPERIMENTS AND GENERATE GRAPHS THAT TEACH THE READER HOW "LEARNING RATES" AND "PAY PROFILES" DEPEND ON THOSE PARAMETERS AS WELL AS ON THE INITIAL HIDDEN $h_0, w_0$.  

```{code-cell} ipython3
:tags: []

# We can set these parameters when creating a worker -- just like classes!
hard_working_worker =  create_worker(α=.4, β=.8, hhat_0=7.0, uhat_0=100, σ_h=2.5, σ_u=3.2)

print(hard_working_worker)
```

* MAKE A GRAPH THAT SHOWS THE EVOLUTION OF THE CONDITIONAL VARIANCES OF THE FIRM'S ESTIMATES OF $u_t$ and $h_t$.  THESE CAN BE EXTRACTED FROM THE FORMULA $ G \Sigma_t G^\top$ OR SOMETHING LIKE THAT. I CAN GIVE YOU CORRECTED VERSION OF THAT FORMULA ONCE YOU GET STARTED.

+++

THANKS SO MUCH!

We can also simulate the system for $T = 50$ periods for different workers.

The difference between the inferred work ethics and true work ethics converges to $0$ over time.

This shows that the filter is gradually teaching the worker and firm about the worker's effort.

```{code-cell} ipython3
:tags: [hide-input]

def simulate_workers(worker, T, ax, mu_0=None, Sigma_0=None, diff=True, name=None, title=None):
    A, C, G, R = worker.A, worker.C, worker.G, worker.R
    xhat_0, Σ_0 = worker.xhat_0, worker.Σ_0
    
    if isinstance(mu_0, type(None)):
        mu_0 = xhat_0
    if isinstance(Sigma_0, type(None)):
        Sigma_0 = worker.Σ_0
        
    ss = LinearStateSpace(A, C, G, np.sqrt(R), mu_0=mu_0, Sigma_0=Sigma_0)

    x, y = ss.simulate(T)
    y = y.flatten()

    u_0 = x[1, 0]
    
    # Compute Kalman filter
    kalman = Kalman(ss, xhat_0, Σ_0)
    Σ_t = []
    
    y_hat_t = np.zeros(T)
    u_hat_t = np.zeros(T)

    for i in range(T):
        kalman.update(y[i])
        x_hat, Σ = kalman.x_hat, kalman.Sigma
        Σ_t.append(Σ)
        y_hat_t[i] = worker.G @ x_hat
        u_hat_t[i] = x_hat[1]

    if diff == True:
        title = ('Difference between inferred and true work ethic over time' 
                 if title == None else title)
        
        ax.plot(u_hat_t - u_0, alpha=.5)
        ax.axhline(y=0, color='grey', linestyle='dashed')
        ax.set_xlabel('Time')
        ax.set_ylabel(r'$E[u_t|y^{t-1}] - u_0$')
        ax.set_title(title)
        
    else:
        label_line = (r'$E[u_t|y^{t-1}]$' if name == None 
                      else name)
        title = ('Inferred work ethic over time' 
                if title == None else title)
        
        u_hat_plot = ax.plot(u_hat_t, label=label_line)
        ax.axhline(y=u_0, color=u_hat_plot[0].get_color(), linestyle='dashed', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel(r'$E[u_t|y^{t-1}]$')
        ax.set_title(title)
```

```{code-cell} ipython3
:tags: []

num_workers = 3
T = 50
fig, ax = plt.subplots(figsize=(7, 7))

for i in range(num_workers):
    worker = create_worker(uhat_0=4+2*i)
    simulate_workers(worker, T, ax)
ax.set_ylim(ymin=-2, ymax=2)
plt.show()
```

```{code-cell} ipython3
:tags: []

# We can also generate plots of u_t:

T = 50
fig, ax = plt.subplots(figsize=(7, 7))

uhat_0s = [2, -2, 1]
αs = [0.2, 0.3, 0.5]
βs = [0.1, 0.9, 0.3]

for i, (uhat_0, α, β) in enumerate(zip(uhat_0s, αs, βs)):
    worker = create_worker(uhat_0=uhat_0, α=α, β=β)
    simulate_workers(worker, T, ax,
                    diff=False, name=r'$u_{{{}, t}}$'.format(i)) # By setting diff=False, it will give u_t
    
ax.axhline(y=u_0, xmin=0, xmax=0, color='grey', 
           linestyle='dashed', label=r'$u_{i, 0}$')
ax.legend(bbox_to_anchor=(1, 0.5))
plt.show()
```

```{code-cell} ipython3
:tags: []

# We can also use exact u_0=1 and h_0=2 for all workers

T = 50
fig, ax = plt.subplots(figsize=(7, 7))

# These two lines set u_0=1 and h_0=2 for all workers
mu_0 = np.array([[1],
                 [2]])
Sigma_0 = np.zeros((2,2))

uhat_0s = [2, -2, 1]
αs = [0.2, 0.3, 0.5]
βs = [0.1, 0.9, 0.3]

for i, (uhat_0, α, β) in enumerate(zip(uhat_0s, αs, βs)):
    worker = create_worker(uhat_0=uhat_0, α=α, β=β)
    simulate_workers(worker, T, ax, mu_0=mu_0, Sigma_0=Sigma_0, 
                     diff=False, name=r'$u_{{{}, t}}$'.format(i))
    
# This controls the boundary of plots
ax.set_ylim(ymin=-3, ymax=3)
ax.axhline(y=u_0, xmin=0, xmax=0, color='grey', 
           linestyle='dashed', label=r'$u_{i, 0}$')
ax.legend(bbox_to_anchor=(1, 0.5))
plt.show()
```

```{code-cell} ipython3
:tags: []

# We can generate a plot for only one of the workers:

T = 50
fig, ax = plt.subplots(figsize=(7, 7))

mu_0_1 = np.array([[1],
                 [100]])
mu_0_2 = np.array([[1],
                 [30]])
Sigma_0 = np.zeros((2,2))

uhat_0s = 100
αs = 0.5
βs = 0.3

worker = create_worker(uhat_0=uhat_0, α=α, β=β)
simulate_workers(worker, T, ax, mu_0=mu_0_1, Sigma_0=Sigma_0, 
                 diff=False, name=r'Hard-working worker')
simulate_workers(worker, T, ax, mu_0=mu_0_2, Sigma_0=Sigma_0, 
                 diff=False, 
                 title='A hard-working worker and a less hard-working worker',
                 name=r'Normal worker')
ax.axhline(y=u_0, xmin=0, xmax=0, color='grey', 
           linestyle='dashed', label=r'$u_{i, 0}$')
ax.legend(bbox_to_anchor=(1, 0.5))
plt.show()
```
