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

(time_series_with_matrices)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Univariate Time Series with Matrix Algebra

```{contents} Contents
:depth: 2
```

## Overview

This lecture uses matrices to solve some linear difference equations.

As a running example, we’ll study a **second-order linear difference
equation** that was the key technical tool in Paul Samuelson’s 1939
article {cite}`Samuelson1939` that introduced the **multiplier-accelerator** model.

This model became the workhorse that powered early econometric versions of
Keynesian macroeconomic models in the United States.

You can read about the details of that model in {doc}`this <samuelson>`
QuantEcon lecture.

(That lecture also describes some technicalities about second-order linear difference equations.)

In this lecture, we'll also learn about an **autoregressive** representation and a **moving average** representation of a  non-stationary
univariate time series $\{y_t\}_{t=0}^T$.

We'll also study a "perfect foresight" model of stock prices that involves solving
a "forward-looking" linear difference equation.

We will use the following imports:

```{code-cell} ipython
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
```

## Samuelson's model

Let $t = 0, \pm 1, \pm 2, \ldots$ index time.

For $t = 1, 2, 3, \ldots, T$ suppose that

```{math}
:label: tswm_1

y_{t} = \alpha_{0} + \alpha_{1} y_{t-1} + \alpha_{2} y_{t-2}
```

where we assume that $y_0$ and $y_{-1}$ are given numbers
that we take as **initial conditions**.

In Samuelson's model, $y_t$ stood for **national income** or perhaps a different
measure of aggregate activity called **gross domestic product** (GDP) at time $t$.

Equation {eq}`tswm_1` is called a **second-order linear difference equation**.

But actually, it is a collection of $T$ simultaneous linear
equations in the $T$ variables $y_1, y_2, \ldots, y_T$.

```{note}
To be able to solve a second-order linear difference
equation, we require two **boundary conditions** that can take the form
either of two **initial conditions** or two **terminal conditions** or
possibly one of each.
```

Let’s write our equations as a stacked system

$$
\underset{\equiv A}{\underbrace{\left[\begin{array}{cccccccc}
1 & 0 & 0 & 0 & \cdots & 0 & 0 & 0\\
-\alpha_{1} & 1 & 0 & 0 & \cdots & 0 & 0 & 0\\
-\alpha_{2} & -\alpha_{1} & 1 & 0 & \cdots & 0 & 0 & 0\\
0 & -\alpha_{2} & -\alpha_{1} & 1 & \cdots & 0 & 0 & 0\\
\vdots & \vdots & \vdots & \vdots & \cdots & \vdots & \vdots & \vdots\\
0 & 0 & 0 & 0 & \cdots & -\alpha_{2} & -\alpha_{1} & 1
\end{array}\right]}}\left[\begin{array}{c}
y_{1}\\
y_{2}\\
y_{3}\\
y_{4}\\
\vdots\\
y_{T}
\end{array}\right]=\underset{\equiv b}{\underbrace{\left[\begin{array}{c}
\alpha_{0}+\alpha_{1}y_{0}+\alpha_{2}y_{-1}\\
\alpha_{0}+\alpha_{2}y_{0}\\
\alpha_{0}\\
\alpha_{0}\\
\vdots\\
\alpha_{0}
\end{array}\right]}}
$$

or

$$
A y = b
$$

where

$$
y = \begin{bmatrix} y_1 \cr y_2 \cr \vdots \cr y_T \end{bmatrix}
$$

Evidently $y$ can be computed from

$$
y = A^{-1} b
$$

The vector $y$ is a complete time path $\{y_t\}_{t=1}^T$.

Let’s put Python to work on an example that captures the flavor of
Samuelson’s multiplier-accelerator model.

We'll set parameters equal to the same values we used in {doc}`this QuantEcon lecture <samuelson>`.

```{code-cell} python3
T = 80

# parameters
𝛼0 = 10.0
𝛼1 = 1.53
𝛼2 = -.9

y_1 = 28. # y_{-1}
y0 = 24.
```

Now we construct $A$ and $b$.

```{code-cell} python3
A = np.identity(T)  # The T x T identity matrix

for i in range(T):

    if i-1 >= 0:
        A[i, i-1] = -𝛼1

    if i-2 >= 0:
        A[i, i-2] = -𝛼2

b = np.full(T, 𝛼0)
b[0] = 𝛼0 + 𝛼1 * y0 + 𝛼2 * y_1
b[1] = 𝛼0 + 𝛼2 * y0
```

Let’s look at the matrix $A$ and the vector $b$ for our
example.

```{code-cell} python3
A, b
```

Now let’s solve for the path of $y$.

If $y_t$ is GNP at time $t$, then we have a version of
Samuelson’s model of the dynamics for GNP.

To solve $y = A^{-1} b$ we can either invert $A$ directly, as in 

```{code-cell} python3
A_inv = np.linalg.inv(A)

y = A_inv @ b
```

or we can use `np.linalg.solve`: 


```{code-cell} python3
y_second_method = np.linalg.solve(A, b)
```

Here make sure the two methods give the same result, at least up to floating
point precision:

```{code-cell} python3
np.allclose(y, y_second_method)
```

```{note}
In general, `np.linalg.solve` is more numerically stable than using
`np.linalg.inv` directly. 
However, stability is not an issue for this small example. Moreover, we will
repeatedly use `A_inv` in what follows, so there is added value in computing
it directly.
```

Now we can plot.

```{code-cell} python3
plt.plot(np.arange(T)+1, y)
plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

The **steady state** value $y^*$ of $y_t$ is obtained by setting $y_t = y_{t-1} =
y_{t-2} = y^*$ in {eq}`tswm_1`, which yields

$$
y^* = \frac{\alpha_{0}}{1 - \alpha_{1} - \alpha_{2}}
$$

If we set the initial values to $y_{0} = y_{-1} = y^*$, then $y_{t}$ will be
constant:

```{code-cell} python3
y_star = 𝛼0 / (1 - 𝛼1 - 𝛼2)
y_1_steady = y_star # y_{-1}
y0_steady = y_star

b_steady = np.full(T, 𝛼0)
b_steady[0] = 𝛼0 + 𝛼1 * y0_steady + 𝛼2 * y_1_steady
b_steady[1] = 𝛼0 + 𝛼2 * y0_steady
```

```{code-cell} python3
y_steady = A_inv @ b_steady
```

```{code-cell} python3
plt.plot(np.arange(T)+1, y_steady)
plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

## Adding a Random Term

To generate some excitement, we'll follow in the spirit of the great economists
Eugen Slutsky and Ragnar Frisch and replace our original second-order difference
equation with the following **second-order stochastic linear difference
equation**:

```{math}
:label: tswm_2

y_{t} = \alpha_{0} + \alpha_{1} y_{t-1} + \alpha_{2} y_{t-2} + u_t
```

where $u_{t} \sim N\left(0, \sigma_{u}^{2}\right)$ and is IID,
meaning **independent** and **identically** distributed.

We’ll stack these $T$ equations into a system cast in terms of
matrix algebra.

Let’s define the random vector

$$
u=\left[\begin{array}{c}
u_{1}\\
u_{2}\\
\vdots\\
u_{T}
\end{array}\right]
$$

Where $A, b, y$ are defined as above, now assume that $y$ is
governed by the system

$$
A y = b + u
$$ (eq:eqar)

The solution for $y$ becomes

$$
y = A^{-1} \left(b + u\right)
$$ (eq:eqma)

Let’s try it out in Python.

```{code-cell} python3
𝜎u = 2.
```

```{code-cell} python3
u = np.random.normal(0, 𝜎u, size=T)
y = A_inv @ (b + u)
```

```{code-cell} python3
plt.plot(np.arange(T)+1, y)
plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

The above time series looks a lot like (detrended) GDP series for a
number of advanced countries in recent decades.

We can simulate $N$ paths.

```{code-cell} python3
N = 100

for i in range(N):
    col = cm.viridis(np.random.rand())  # Choose a random color from viridis
    u = np.random.normal(0, 𝜎u, size=T)
    y = A_inv @ (b + u)
    plt.plot(np.arange(T)+1, y, lw=0.5, color=col)

plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

Also consider the case when $y_{0}$ and $y_{-1}$ are at
steady state.

```{code-cell} python3
N = 100

for i in range(N):
    col = cm.viridis(np.random.rand())  # Choose a random color from viridis
    u = np.random.normal(0, 𝜎u, size=T)
    y_steady = A_inv @ (b_steady + u)
    plt.plot(np.arange(T)+1, y_steady, lw=0.5, color=col)

plt.xlabel('t')
plt.ylabel('y')

plt.show()
```



## Computing Population Moments


We can apply standard formulas for multivariate normal distributions to compute the mean vector and covariance matrix
for our time series model

$$
y = A^{-1} (b + u) .
$$

You can read about multivariate normal distributions in this lecture {doc}`Multivariate Normal Distribution <multivariate_normal>`.

Let's write our  model as 

$$ 
y = \tilde A (b + u)
$$

where $\tilde A = A^{-1}$.

Because  linear combinations of normal random variables are normal, we know that

$$
y \sim {\mathcal N}(\mu_y, \Sigma_y)
$$

where

$$ 
\mu_y = \tilde A b
$$

and 

$$
\Sigma_y = \tilde A (\sigma_u^2 I_{T \times T} ) \tilde A^T
$$

Let's write a Python  class that computes the mean vector $\mu_y$ and covariance matrix $\Sigma_y$.



```{code-cell} ipython3
class population_moments:
    """
    Compute population moments mu_y, Sigma_y.
    ---------
    Parameters:
    alpha0, alpha1, alpha2, T, y_1, y0
    """
    def __init__(self, alpha0, alpha1, alpha2, T, y_1, y0, sigma_u):

        # compute A
        A = np.identity(T)

        for i in range(T):
            if i-1 >= 0:
                A[i, i-1] = -alpha1

            if i-2 >= 0:
                A[i, i-2] = -alpha2

        # compute b
        b = np.full(T, alpha0)
        b[0] = alpha0 + alpha1 * y0 + alpha2 * y_1
        b[1] = alpha0 + alpha2 * y0

        # compute A inverse
        A_inv = np.linalg.inv(A)

        self.A, self.b, self.A_inv, self.sigma_u, self.T = A, b, A_inv, sigma_u, T
    
    def sample_y(self, n):
        """
        Give a sample of size n of y.
        """
        A_inv, sigma_u, b, T = self.A_inv, self.sigma_u, self.b, self.T
        us = np.random.normal(0, sigma_u, size=[n, T])
        ys = np.vstack([A_inv @ (b + u) for u in us])

        return ys

    def get_moments(self):
        """
        Compute the population moments of y.
        """
        A_inv, sigma_u, b = self.A_inv, self.sigma_u, self.b

        # compute mu_y
        self.mu_y = A_inv @ b
        self.Sigma_y = sigma_u**2 * (A_inv @ A_inv.T)

        return self.mu_y, self.Sigma_y


my_process = population_moments(
    alpha0=10.0, alpha1=1.53, alpha2=-.9, T=80, y_1=28., y0=24., sigma_u=1)
    
mu_y, Sigma_y = my_process.get_moments()
A_inv = my_process.A_inv
```

It is enlightening  to study the $\mu_y, \Sigma_y$'s implied by  various parameter values.

Among other things, we can use the class to exhibit how  **statistical stationarity** of $y$ prevails only for very special initial conditions. 

Let's begin by generating $N$ time realizations of $y$ plotting them together with  population  mean $\mu_y$ .

```{code-cell} ipython3
# plot mean
N = 100

for i in range(N):
    col = cm.viridis(np.random.rand())  # Choose a random color from viridis
    ys = my_process.sample_y(N)
    plt.plot(ys[i,:], lw=0.5, color=col)
    plt.plot(mu_y, color='red')

plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

Visually, notice how the  variance across realizations of $y_t$ decreases as $t$ increases.

Let's plot the population variance of $y_t$ against $t$.

```{code-cell} ipython3
# plot variance
plt.plot(Sigma_y.diagonal())
plt.show()
```

Notice how the population variance increases and asymptotes

+++

Let's print out the covariance matrix $\Sigma_y$ for a  time series $y$

```{code-cell} ipython3
my_process = population_moments(alpha0=0, alpha1=.8, alpha2=0, T=6, y_1=0., y0=0., sigma_u=1)
    
mu_y, Sigma_y = my_process.get_moments()
print("mu_y = ",mu_y)
print("Sigma_y = ", Sigma_y)
```

Notice that  the covariance between $y_t$ and $y_{t-1}$ -- the elements on the superdiagonal -- are **not** identical.

This is is an indication that the time series respresented by our $y$ vector is not **stationary**.  

To make it stationary, we'd have to alter our system so that our **initial conditions** $(y_1, y_0)$ are not fixed numbers but instead a jointly normally distributed random vector with a particular mean and  covariance matrix.

We describe how to do that in another lecture in this lecture {doc}`Linear State Space Models <linear_models>`.

But just to set the stage for that analysis, let's  print out the bottom right corner of $\Sigma_y$.

```{code-cell} ipython3
mu_y, Sigma_y = my_process.get_moments()
print("bottom right corner of Sigma_y = \n", Sigma_y[72:,72:])
```

Please notice how the sub diagonal and super diagonal elements seem to have converged.

This is an indication that our process is asymptotically stationary.

You can read  about stationarity of more general linear time series models in this lecture {doc}`Linear State Space Models <linear_models>`.

There is a lot to be learned about the process by staring at the off diagonal elements of $\Sigma_y$ corresponding to different time periods $t$, but we resist the temptation to do so here.

+++


## Moving Average Representation

Let's print out  $A^{-1}$ and stare at  its structure 

  *  is it triangular or almost triangular or $\ldots$ ?

To study the structure of $A^{-1}$, we shall print just  up to $3$ decimals.

Let's begin by printing out just the upper left hand corner of $A^{-1}$

```{code-cell} ipython3
with np.printoptions(precision=3, suppress=True):
    print(A_inv[0:7,0:7])
```




Evidently, $A^{-1}$ is a lower triangular matrix. 


Let's print out the lower right hand corner of $A^{-1}$ and stare at it.

```{code-cell} ipython3
with np.printoptions(precision=3, suppress=True):
    print(A_inv[72:,72:])
```


Notice how  every row ends with the previous row's pre-diagonal entries.



 

Since $A^{-1}$ is lower triangular,  each  row represents  $ y_t$ for a particular $t$ as the sum of 
- a time-dependent function $A^{-1} b$ of the initial conditions incorporated in $b$, and 
- a weighted sum of  current and past values of the IID shocks $\{u_t\}$

Thus,  let $\tilde{A}=A^{-1}$. 

Evidently,  for $t\geq0$,

$$
y_{t+1}=\sum_{i=1}^{t+1}\tilde{A}_{t+1,i}b_{i}+\sum_{i=1}^{t}\tilde{A}_{t+1,i}u_{i}+u_{t+1}
$$

This is  a **moving average** representation with time-varying coefficients.

Just as system {eq}`eq:eqma` constitutes  a 
**moving average** representation for $y$, system  {eq}`eq:eqar` constitutes  an **autoregressive** representation for $y$.




## A Forward Looking Model

Samuelson’s model is **backwards looking** in the sense that we give it **initial conditions** and let it
run.

Let’s now turn to model  that is **forward looking**.

We apply similar linear algebra machinery to study a **perfect
foresight** model widely used as a benchmark in macroeconomics and
finance.

As an example, we suppose that $p_t$ is the price of a stock and
that $y_t$ is its dividend.

We assume that $y_t$ is determined by second-order difference
equation that we analyzed just above, so that

$$
y = A^{-1} \left(b + u\right)
$$

Our **perfect foresight** model of stock prices is

$$
p_{t} = \sum_{j=0}^{T-t} \beta^{j} y_{t+j}, \quad \beta \in (0,1)
$$

where $\beta$ is a discount factor.

The model asserts that the price of the stock at $t$ equals the
discounted present values of the (perfectly foreseen) future dividends.

Form

$$
\underset{\equiv p}{\underbrace{\left[\begin{array}{c}
p_{1}\\
p_{2}\\
p_{3}\\
\vdots\\
p_{T}
\end{array}\right]}}=\underset{\equiv B}{\underbrace{\left[\begin{array}{ccccc}
1 & \beta & \beta^{2} & \cdots & \beta^{T-1}\\
0 & 1 & \beta & \cdots & \beta^{T-2}\\
0 & 0 & 1 & \cdots & \beta^{T-3}\\
\vdots & \vdots & \vdots & \vdots & \vdots\\
0 & 0 & 0 & \cdots & 1
\end{array}\right]}}\left[\begin{array}{c}
y_{1}\\
y_{2}\\
y_{3}\\
\vdots\\
y_{T}
\end{array}\right]
$$

```{code-cell} python3
𝛽 = .96
```

```{code-cell} python3
# construct B
B = np.zeros((T, T))

for i in range(T):
    B[i, i:] = 𝛽 ** np.arange(0, T-i)
```

```{code-cell} python3
B
```

```{code-cell} python3
𝜎u = 0.
u = np.random.normal(0, 𝜎u, size=T)
y = A_inv @ (b + u)
y_steady = A_inv @ (b_steady + u)
```

```{code-cell} python3
p = B @ y
```

```{code-cell} python3
plt.plot(np.arange(0, T)+1, y, label='y')
plt.plot(np.arange(0, T)+1, p, label='p')
plt.xlabel('t')
plt.ylabel('y/p')
plt.legend()

plt.show()
```

Can you explain why the trend of the price is downward over time?

Also consider the case when $y_{0}$ and $y_{-1}$ are at the
steady state.

```{code-cell} python3
p_steady = B @ y_steady

plt.plot(np.arange(0, T)+1, y_steady, label='y')
plt.plot(np.arange(0, T)+1, p_steady, label='p')
plt.xlabel('t')
plt.ylabel('y/p')
plt.legend()

plt.show()
```

