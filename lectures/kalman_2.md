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

(kalman)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# A Second Look at the Kalman Filter


The purpose is to watch how the firm ends of learning a worker's "type" and how the worker's pay evolves as the firm learns. 

The $h_t$ part of the worker's "type" moves over time, but the effort type $u_t$ is fixed, so for this part the firm is in effect "learning a parameter".

To conduct simulations, we want to bring in these imports, as in the "first looks" lecture

```{code-cell} ipython
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
from scipy import linalg
import numpy as np
import matplotlib.cm as cm
from quantecon import Kalman, LinearStateSpace
from scipy.stats import norm
from scipy.integrate import quad
from scipy.linalg import eigvals
```

## A filtering example


A representative worker has productivity described by the following dynamic process:

```{math}
\begin{align}
h_{t+1} &= \alpha h_t + \beta h_t + c w_{t+1}, \quad c_{t+1} \sim {\mathcal N}(0,1) \\
u_t & = u_t \\
y_t & = g h_t + v_t , \quad v_t \sim {\mathcal N} (0, R) \tag{1}
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


## Steps to get answer

Write system (1) in the state-space form

```{math}
\begin{align}
\begin{bmatrix} h_{t+1} \cr u_{t+1} \end{bmatrix} &= \begin{bmatrix} \alpha & \beta \cr 0 & 1 \end{bmatrix}\begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} + \begin{bmatrix} c & 0 \end{bmatrix} w_{t+1} \cr
y_t & = \begin{bmatrix} g & 0 \end{bmatrix} \begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} + v_t
\end{align}
```

or

```{math}
\begin{align}
x_{t+1} & = A x_t + C w_{t+1} \cr
y_t & = G x_t + v_t \cr
x_0 & \sim {\mathcal N}(\hat x_0, \Sigma_0) \end{align}
```
where

```{math}
\begin{align}
h_t &= \begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} \cr
\hat x_0 & = \begin{bmatrix} \hat h_0 \cr \hat u_0 \end{bmatrix} \cr
\hat \Sigma_0 & = \begin{bmatrix} \sigma_{h,0} & 0 \cr
                     0 & \sigma_{u,0} \end{bmatrix}
\end{align}
```

Parameters of the model are $\alpha, \beta, c, R, \hat h_0, \hat u_0, \sigma_h, \sigma_u$.


## Coding steps:

TODO:
For given parameter values, write a program (a class?) to compute the following objects by using the Kalman class and 
consulting the "First Look ... " quantecon lecture:

* All of the objects in the "innovation representation"
    \begin{align}
    \hat x_{t+1} & = A \hat x_t + K_t a_t \cr
    y_{t} & = G \hat x_t + a_t
    \end{align}
    (The Kalman gain $K_t$ are is computed in the Kalman class)
* Please also compute $\Sigma_t$ -- the conditional covariance matrix
    
* For a draw of $h_0, u_0$,  please prepare graphs of $E y_t = G \hat x_t $ where $\hat x_t = E [x_t | y^{t-1}] $. Please also prepare a graph of $E u_0 | y^{t-1}$. (Here the firm is inferring a worker's hard-wired "work ethic" $u_0$.)