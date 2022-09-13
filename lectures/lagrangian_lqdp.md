---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Lagrangian for LQ Control

```{code-cell} ipython3
:tags: [hide-output]
!pip install quantecon
```

```{code-cell} ipython3
import numpy as np
from quantecon import LQ
from scipy.linalg import schur
```

+++

## Overview

This is a sequel to this lecture {doc}`linear quadratic dynamic programming <lqcontrol>`

It can also be regarded as presenting **invariant subspace** techniques that extend ones that 
we encountered earlier in this lecture {doc}`stability in linear rational expectations models<re_with_feedback>`

We present a Lagrangian formulation of an infinite horizon linear quadratic undiscounted dynamic programming problem.

Such a problem is  also sometimes called an  optimal linear regulator problem.

A Lagrangian formulation 

* carries insights about connections between stability and optimality
 
* is the basis for fast algorithms for solving Riccati equations
  
* opens the way to constructing solutions of dynamic systems that don't come directly from an  intertemporal optimization problem

A key tool in this lecture is the concept of an $n \times n$ **symplectic** matrix.

A symplectic matrix has eigenvalues that occur in **reciprocal pairs**, meaning that if $\lambda_i \in (-1,1)$ is an eigenvalue, then so is $\lambda_i^{-1}$.  

This reciprocal pairs property of the eigenvalues of a matrix  is a tell-tale sign that the matrix describes the joint dynamics of a system of 
equations describing the **states** and **costates** that constitute   first-order necessary  conditions for solving an undiscounted linear-quadratic  infinite-horizon optimization problem. 

The symplectic matrix that will interest us describes the first-order dynamics of **state** and **co-state** vectors of an optimally controlled system.

In focusing on eigenvalues and eigenvectors of this matrix, we capitalize on an analysis of 
**invariant subspaces.** 

These invariant subspace formulations of LQ dynamic programming problems provide a bridge between recursive
(i.e., dynamic programming) formulations and classical formulations of linear control and linear filtering problems that make use of related matrix decompositions (see for example [this lecture](https://python-advanced.quantecon.org/lu_tricks.html) and [this lecture](https://python-advanced.quantecon.org/classical_filtering.html)).

While most of this lecture focuses on undiscounted problems, later sections describe handy ways of transforming discounted problems to undiscounted ones.

The techniques in this lecture will prove useful when we study Stackelberg and Ramsey problem in
[this lecture](https://python-advanced.quantecon.org/dyn_stack.html). 


  
## Undiscounted LQ DP Problem


The problem is to choose a sequence of controls  $\{u_t\}_{t=0}^\infty$ to maximize the criterion

$$ 
- \sum_{t=0}^\infty \{x'_t Rx_t+u'_tQu_t\} 
$$

subject to $x_{t+1}=Ax_t+Bu_t$, where  $x_0$ is a  given initial state vector. 

Here $x_t$ is an $(n\times 1)$ vector of state variables, $u_t$ is a $(k\times 1)$
vector of controls, $R$ is a positive semidefinite symmetric matrix,
$Q$ is a positive definite symmetric matrix, $A$ is an $(n\times n)$
matrix, and $B$ is an $(n\times k)$ matrix.

The optimal 
value function  turns out to be  quadratic, $V(x)= - x'Px$, where $P$ is a positive
semidefinite symmetric matrix.

Using the transition law to eliminate next period's state, the Bellman
equation becomes

$$ 
- x'Px=\max_u \{- x' Rx-u'Qu-(Ax+Bu)' P(Ax+Bu)\}
$$ (bellman0)

The first-order necessary conditions for the maximum problem on the
right side of equation {eq}`bellman0` are

```{note} 
We use the following rules for differentiating quadratic and bilinear matrix forms: 
${\partial x' A x \over \partial x} = (A + A') x; {\partial y' B z \over \partial y} = B z, {\partial
y' B z \over \partial z} = B' y$.
```

$$
(Q+B'PB)u=-B'PAx,
$$

which implies that an optimal decision rule for $u$ is 

$$
u=-(Q+B'PB)^{-1} B'PAx
$$ 

or 

$$
u=-Fx,
$$

where 

$$ 
F=(Q+B'PB)^{-1}B'PA.
$$

Substituting $u = - (Q+B'PB)^{-1}B'PAx$  into
the right side of equation {eq}`bellman0` and rearranging gives

$$
P=R+A'PA-A'PB(Q+B'PB)^{-1} B'PA.
$$ (riccati)

Equation {eq}`riccati` is called an **algebraic matrix Riccati** equation.

There are multiple solutions of equation {eq}`riccati`.

But only one of them is positive definite.  

The positive define solution is associated with the maximum of our problem.

It expresses the matrix $P$ as an implicit function of the matrices
$R,Q,A,B$. 

Notice that the **gradient of the value function** is

$$
\frac{\partial V(x)}{\partial x} = - 2 P x 
$$ (eqn:valgrad)

We shall use fact {eq}`eqn:valgrad` later.

+++

## Lagrangian

For the undiscounted optimal linear regulator problem, form the Lagrangian

$$
{\cal L} = - \sum^\infty_{t=0} \biggl\{ x^\prime_t R x_t + u_t^\prime Q u_t +
                                 2 \mu^\prime_{t+1} [A x_t + B u_t - x_{t+1}]\biggr\}
$$ (lag-lqdp-eq1)

where $2 \mu_{t+1}$ is a vector of Lagrange multipliers on the time $t$ transition law $x_{t+1} = A x_t + B u_t$.

(We put the $2$ in front of $\mu_{t+1}$ to make things match up nicely with equation {eq}`eqn:valgrad`.)

First-order conditions for maximization with respect to $\{u_t,x_{t+1}\}_{t=0}^\infty$ are

$$
\begin{aligned}
2 Q u_t &+ 2B^\prime \mu_{t+1} = 0 \ ,\ t \geq 0 \cr \mu_t &= R x_t + A^\prime \mu_{t+1}\ ,\ t\geq 1.\cr
\end{aligned}
$$ (lag-lqdp-eq2)

Define $\mu_0$ to be  a vector of shadow prices of $x_0$ and apply an envelope condition to {eq}`lag-lqdp-eq1`
 to deduce that

$$
\mu_0 = R x_0 + A' \mu_1,
$$

which is a time $t=0 $ counterpart to the second equation of system {eq}`lag-lqdp-eq2`.

An important fact is  that  

$$ 
\mu_{t+1} = P x_{t+1}
$$ (eqn:muPx)

where $P$ is a positive define  matrix that solves  the algebraic Riccati equation {eq}`riccati`. 

Thus, from equations {eq}`eqn:valgrad` and  {eq}`eqn:muPx`,  $- 2 \mu_{t}$ is
the gradient of the value function with respect to $x_t$. 

The Lagrange multiplier vector $\mu_{t}$ is often called the **costate** vector that 
corresponds to the **state** vector $x_t$.

It is useful to proceed with the following steps:

* solve the first equation of {eq}`lag-lqdp-eq2`  for $u_t$ in terms of $\mu_{t+1}$.

* substitute the result into the law of motion $x_{t+1} = A x_t + B u_t$.

* arrange the resulting equation and the second equation of {eq}`lag-lqdp-eq2`  into the form

$$
L\ \begin{pmatrix}x_{t+1}\cr \mu_{t+1}\cr\end{pmatrix}\ = \ N\ \begin{pmatrix}x_t\cr \mu_t\cr\end{pmatrix}\
,\ t \geq 0,
$$ (eq:systosolve)

where

$$
L = \ \begin{pmatrix}I & BQ^{-1} B^\prime \cr 0 & A^\prime\cr\end{pmatrix}, \quad N = \
\begin{pmatrix}A & 0\cr -R & I\cr\end{pmatrix}.
$$

When $L$ is of full rank (i.e., when $A$ is of full rank), we can write
system {eq}`eq:systosolve` as

$$
\begin{pmatrix}x_{t+1}\cr \mu_{t+1}\cr\end{pmatrix}\ = M\ \begin{pmatrix}x_t\cr\mu_t\cr\end{pmatrix}
$$ (eq4orig)

where

$$
M\equiv L^{-1} N = \begin{pmatrix}A+B Q^{-1} B^\prime A^{\prime-1}R &
-B Q^{-1} B^\prime A^{\prime-1}\cr -A^{\prime -1} R & A^{\prime -1}\cr\end{pmatrix}.
$$ (Mdefn)

+++

## State-Costate Dynamics


We seek to solve the difference equation system  {eq}`eq4orig` for a sequence $\{x_t\}_{t=0}^\infty$
that satisfies 

* an initial condition for $x_0$
* a terminal condition $\lim_{t \rightarrow +\infty} x_t =0$ 

This  terminal condition reflects our desire for a **stable** solution, one that does not diverge as $t \rightarrow \infty$.


We inherit our wish for stability of the $\{x_t\}$ sequence from a desire to maximize

$$ 
-\sum_{t=0}^\infty \bigl[ x_t ' R x_t + u_t' Q u_t \bigr],
$$

which requires that $x_t' R x_t$ converge to zero as $t \rightarrow + \infty$.

+++

## Reciprocal Pairs Property

To proceed, we study properties of the $(2n \times 2n)$ matrix $M$ defined in {eq}`Mdefn`. 

It helps to introduce a $(2n \times 2n)$ matrix

$$
J = \begin{pmatrix}0 & -I_n\cr I_n & 0\cr\end{pmatrix}.
$$

The rank of $J$ is $2n$.

**Definition:**  A matrix $M$ is called **symplectic** if

$$
MJM^\prime = J.
$$ (lag-lqdp-eq3)

Salient properties of symplectic matrices that are readily verified include:

  * If $M$ is symplectic, then $M^2$ is symplectic
  * The determinant of a symplectic, then $\textrm{det}(M) = 1$

It can be verified directly that $M$ in equation {eq}`Mdefn` is symplectic.

It follows from equation {eq}`lag-lqdp-eq3` and from the fact $J^{-1} = J^\prime = -J$ that for any symplectic
matrix $M$,

$$
M^\prime = J^{-1} M^{-1} J.
$$ (lag-lqdp-eq4)

Equation {eq}`lag-lqdp-eq4` states that $M^\prime$ is related to the inverse of $M$
by a **similarity transformation**.

For square matrices, recall that  
  
* similar matrices share eigenvalues

*  eigenvalues of the inverse of a matrix are  inverses of  eigenvalues of the matrix

* a matrix and its transpose share eigenvalues

It then follows from equation {eq}`lag-lqdp-eq4`  that
the eigenvalues of $M$ occur in reciprocal pairs: if $\lambda$ is an
eigenvalue of $M$, so is $\lambda^{-1}$.

Write equation {eq}`eq4orig` as 

$$
y_{t+1} = M y_t
$$ (eq658)

where $y_t = \begin{pmatrix}x_t\cr \mu_t\cr\end{pmatrix}$. 

Consider a **triangularization** of $M$

$$
V^{-1} M V= \begin{pmatrix}W_{11} & W_{12} \cr 0 & W_{22}\cr\end{pmatrix}
$$ (eqn:triangledecomp)

where 

* each block on the right side is $(n\times n)$
* $V$ is nonsingular
* all eigenvalues of $W_{22}$ exceed $1$ in modulus
* all  eigenvalues of $W_{11}$ are  less than $1$ in modulus 

## Schur decomposition

The **Schur decomposition** and the **eigenvalue decomposition**
are two  decompositions of the form {eq}`eqn:triangledecomp`. 

Write equation {eq}`eq658`  as

$$
y_{t+1} = V W V^{-1} y_t.
$$ (eq659)

A solution of equation {eq}`eq659`  for arbitrary initial condition $y_0$ is
evidently

$$
y_{t} = V \left[\begin{matrix}W^t_{11} & W_{12,t}\cr 0 & W^t_{22}\cr\end{matrix}\right]
\ V^{-1} y_0
$$ (eq6510)

where $W_{12,t} = W_{12}$ for $t=1$ and  for $t \geq 2$ obeys the recursion

$$
W_{12, t} = W^{t-1}_{11} W_{12,t-1} + W_{12,t-1} W^{t-1}_{22}
$$ 

and where $W^t_{ii}$ is $W_{ii}$ raised to the $t$th  power.

Write equation {eq}`eq6510` as

$$
\begin{pmatrix}y^\ast_{1t}\cr y^\ast_{2t}\cr\end{pmatrix}\ =\ \left[\begin{matrix} W^t_{11} &
W_{12, t}\cr 0 & W^t_{22}\cr\end{matrix}\right]\quad \begin{pmatrix}y^\ast_{10}\cr
y^\ast_{20}\cr\end{pmatrix}
$$

where $y^\ast_t = V^{-1} y_t$, and in particular where

$$
y^\ast_{2t} = V^{21} x_t + V^{22} \mu_t,
$$ (eq6511)

and where $V^{ij}$ denotes the $(i,j)$ piece of
the partitioned $V^{-1}$ matrix.

Because $W_{22}$ is an unstable matrix, $y^\ast_t$ will diverge unless $y^\ast_{20} = 0$.

Let $V^{ij}$ denote the $(i,j)$ piece of the partitioned $V^{-1}$ matrix.

To attain stability, we must impose $y^\ast_{20} =0$, which from equation {eq}`eq6511`  implies

$$
V^{21} x_0 + V^{22} \mu_0 = 0
$$

or

$$
\mu_0 = - (V^{22})^{-1} V^{21} x_0.
$$

This equation replicates itself over
time in the sense that it implies

$$
\mu_t = - (V^{22})^{-1} V^{21} x_t.
$$

But notice that because $(V^{21}\ V^{22})$ is the second row block of
the inverse of $V,$ it follows that

$$
(V^{21} \ V^{22})\quad \begin{pmatrix}V_{11}\cr V_{21}\cr\end{pmatrix} = 0
$$

which implies

$$
V^{21} V_{11} + V^{22} V_{21} = 0.
$$

Therefore,

$$
-(V^{22})^{-1} V^{21} = V_{21} V^{-1}_{11}.
$$

So we can write

$$
\mu_0 = V_{21} V_{11}^{-1} x_0
$$

and

$$
\mu_t = V_{21} V^{-1}_{11} x_t.
$$

However, we know  that $\mu_t = P x_t$,
where $P$ occurs in the matrix that solves the Riccati equation.


Thus, the preceding argument establishes that

$$
P = V_{21} V_{11}^{-1}.
$$ (eqn:Pvaughn)

Remarkably, formula {eq}`eqn:Pvaughn` provides us with a computationally 
efficient way of computing the positive definite  matrix $P$ that solves the algebraic Riccati equation {eq}`riccati` that emerges
from dynamic programming.

This same method can be applied to compute the solution of
any system of the form {eq}`eq4orig` if a solution exists, even
if eigenvalues of $M$ fail to occur in reciprocal pairs.

The method
will typically work so long as the eigenvalues of $M$ split   half
inside and half outside the unit circle.

Systems in which  eigenvalues (properly adjusted for discounting) fail
to occur in reciprocal pairs arise when the system being solved
is an equilibrium of a model in which there are distortions that
prevent there being any optimum problem that the equilibrium
solves. See {cite}`Ljungqvist2012`,  ch 12.  

## Application

Here we demonstrate the computation with an example which is the deterministic version of an example borrowed from this [quantecon lecture](https://python.quantecon.org/lqcontrol.html).

```{code-cell} ipython3
# Model parameters
r = 0.05
c_bar = 2
μ = 1

# Formulate as an LQ problem
Q = np.array([[1]])
R = np.zeros((2, 2))
A = [[1 + r, -c_bar + μ],
     [0,              1]]
B = [[-1],
     [0]]

# Construct an LQ instance
lq = LQ(Q, R, A, B)
```

Given matrices $A$, $B$, $Q$, $R$, we can then compute $L$, $N$, and $M=L^{-1}N$.

```{code-cell} ipython3
def construct_LNM(A, B, Q, R):

    n, k = lq.n, lq.k

    # construct L and N
    L = np.zeros((2*n, 2*n))
    L[:n, :n] = np.eye(n)
    L[:n, n:] = B @ np.linalg.inv(Q) @ B.T
    L[n:, n:] = A.T

    N = np.zeros((2*n, 2*n))
    N[:n, :n] = A
    N[n:, :n] = -R
    N[n:, n:] = np.eye(n)

    # compute M
    M = np.linalg.inv(L) @ N

    return L, N, M
```

```{code-cell} ipython3
L, N, M = construct_LNM(lq.A, lq.B, lq.Q, lq.R)
```

```{code-cell} ipython3
M
```

Let's verify that $M$ is symplectic.

```{code-cell} ipython3
n = lq.n
J = np.zeros((2*n, 2*n))
J[n:, :n] = np.eye(n)
J[:n, n:] = -np.eye(n)

M @ J @ M.T - J
```

We can compute the eigenvalues of $M$ using `np.linalg.eigvals`, arranged in ascending order.

```{code-cell} ipython3
eigvals = sorted(np.linalg.eigvals(M))
eigvals
```

When we apply Schur decomposition such that $M=V W V^{-1}$, we want 

* the upper left block of $W$, $W_{11}$, to have all of its eigenvalues   less than 1 in modulus, and 
* the lower right block $W_{22}$ to have  eigenvalues that  exceed 1 in modulus. 

To get what we want, let's define a sorting function that tells `scipy.schur` to sort the corresponding eigenvalues with modulus smaller than 1 to the upper left.

```{code-cell} ipython3
stable_eigvals = eigvals[:n]

def sort_fun(x):
    "Sort the eigenvalues with modules smaller than 1 to the top-left."

    if x in stable_eigvals:
        stable_eigvals.pop(stable_eigvals.index(x))
        return True
    else:
        return False

W, V, _ = schur(M, sort=sort_fun)
```

```{code-cell} ipython3
W
```

```{code-cell} ipython3
V
```

We can check the modulus of eigenvalues of $W_{11}$ and $W_{22}$. 

Since they are both triangular matrices,  eigenvalues are the  diagonal elements. 

```{code-cell} ipython3
# W11
np.diag(W[:n, :n])
```

```{code-cell} ipython3
# W22
np.diag(W[n:, n:])
```

The following functions wrap  $M$ matrix construction, Schur decomposition, and stability-imposing computation of $P$.

```{code-cell} ipython3
def stable_solution(M, verbose=True):
    """
    Given a system of linear difference equations

        y' = |a b| y
        x' = |c d| x

    which is potentially unstable, find the solution
    by imposing stability.

    Parameter
    ---------
    M : np.ndarray(float)
        The matrix represents the linear difference equations system.
    """
    n = M.shape[0] // 2
    stable_eigvals = list(sorted(np.linalg.eigvals(M))[:n])

    def sort_fun(x):
        "Sort the eigenvalues with modules smaller than 1 to the top-left."

        if x in stable_eigvals:
            stable_eigvals.pop(stable_eigvals.index(x))
            return True
        else:
            return False

    W, V, _ = schur(M, sort=sort_fun)
    if verbose:
        print('eigenvalues:\n')
        print('    W11: {}'.format(np.diag(W[:n, :n])))
        print('    W22: {}'.format(np.diag(W[n:, n:])))

    # compute V21 V11^{-1}
    P = V[n:, :n] @ np.linalg.inv(V[:n, :n])

    return W, V, P

def stationary_P(lq, verbose=True):
    """
    Computes the matrix :math:`P` that represent the value function

         V(x) = x' P x

    in the infinite horizon case. Computation is via imposing stability
    on the solution path and using Schur decomposition.

    Parameters
    ----------
    lq : qe.LQ
        QuantEcon class for analyzing linear quadratic optimal control
        problems of infinite horizon form.

    Returns
    -------
    P : array_like(float)
        P matrix in the value function representation.
    """

    Q = lq.Q
    R = lq.R
    A = lq.A * lq.beta ** (1/2)
    B = lq.B * lq.beta ** (1/2)

    n, k = lq.n, lq.k

    L, N, M = construct_LNM(A, B, Q, R)
    W, V, P = stable_solution(M, verbose=verbose)

    return P
```

```{code-cell} ipython3
# compute P
stationary_P(lq)
```

Note that the matrix $P$ computed in this way is close to what we get from the routine in quantecon that solves an algebraic  Riccati equation by iterating to convergence on a Riccati difference equation. 

The small difference comes from computational errors and will decrease as we increase the maximum number of iterations or decrease the tolerance for convergence.

```{code-cell} ipython3
lq.stationary_values()
```

Using a Schur decomposition is much more efficient.

```{code-cell} ipython3
%%timeit
stationary_P(lq, verbose=False)
```

```{code-cell} ipython3
%%timeit
lq.stationary_values()
```


## Other Applications

The preceding approach to imposing stability on a system  of potentially unstable linear difference equations is not limited to  linear quadratic dynamic optimization problems. 

For example, the same method is used in our [Stability in Linear Rational Expectations Models](https://python.quantecon.org/re_with_feedback.html#another-perspective) lecture.


Let's try to solve the model described in that lecture by applying the `stable_solution` function defined in this lecture above.

```{code-cell} ipython3
def construct_H(ρ, λ, δ):
    "contruct matrix H given parameters."

    H = np.empty((2, 2))
    H[0, :] = ρ,δ
    H[1, :] = - (1 - λ) / λ, 1 / λ

    return H

H = construct_H(ρ=.9, λ=.5, δ=0)
```

```{code-cell} ipython3
W, V, P = stable_solution(H)
P
```

## Discounted Problems 

+++



### Transforming States and Controls to Eliminate Discounting

A pair of useful transformations allows us to convert a discounted problem into an undiscounted one.

Thus, suppose that we have a discounted problem with objective 


$$
 - \sum^\infty_{t=0} \beta^t \biggl\{ x^\prime_t R x_t + u_t^\prime Q u_t \biggr\}
$$ 


and that the state transition equation 
is again $x_{t +1 }=Ax_t+Bu_t$.

Define the transformed state and control variables

* $\hat x_t = \beta^{\frac{t}{2}} x_t $
* $\hat u_t = \beta^{\frac{t}{2}} u_t$
  
and the transformed transition equation
matrices
* $\hat A = \beta^{\frac{1}{2}} A$
* $\hat B =  \beta^{\frac{1}{2}} B  $
  
so that the adjusted state and control variables
obey the transition law

$$
\hat x_{t+1} = \hat A \hat x_t + \hat B \hat u_t. 
$$ 

Then a discounted optimal control problem
defined by $A, B, R, Q, \beta$ having  optimal policy characterized by $P, F$ is associated with an equivalent
undiscounted problem defined by $\hat A, \hat B, Q, R$ having  optimal policy characterized by $\hat F, \hat P$ that satisfy
the following   equations:

$$
\hat F=(Q+B'\hat PB)^{-1}\hat B'P \hat A
$$

and

$$
\hat P=R+\hat A'P \hat A-\hat A'P \hat B(Q+B'\hat P \hat B)^{-1} \hat B'P \hat A
$$

It follows immediately from the definitions of $\hat A, \hat B$ that $\hat F = F$ and $\hat P = P$.

By exploiting these transformations,  we can solve a discounted problem by solving an associated undiscounted problem.




In particular, we can first transform a discounted LQ problem to an undiscounted one and then solve that  discounted optimal regulator problem using the Lagrangian and invariant subspace methods described above.

+++

For example, when $\beta=\frac{1}{1+r}$, we can solve for $P$ with $\hat{A}=\beta^{1/2} A$ and $\hat{B}=\beta^{1/2} B$. 

These settings are adopted by default in the function `stationary_P` defined above.

```{code-cell} ipython3
β = 1 / (1 + r)
lq.beta = β
```

```{code-cell} ipython3
stationary_P(lq)
```

We can verify that the solution agrees with one that comes from applying the  routine `LQ.stationary_values` in the  quantecon package.

```{code-cell} ipython3
lq.stationary_values()
```


### Lagrangian for Discounted Problem

For several purposes, it is useful  explicitly briefly to describe
a Lagrangian for a discounted problem. 

Thus, for the discounted optimal linear regulator problem,
form the Lagrangian

$$
{\cal{L}} = - \sum^\infty_{t=0} \beta^t \biggl\{ x^\prime_t R x_t + u_t^\prime Q u_t
+ 2 \beta \mu^\prime_{t+1} [A x_t + B u_t - x_{t+1}]\biggr\}
$$ (eq661)

where $2 \mu_{t+1}$ is a vector of Lagrange multipliers on the state vector $x_{t+1}$.

First-order conditions for maximization with respect
to $\{u_t,x_{t+1}\}_{t=0}^\infty$ are

$$
\begin{aligned}
2 Q u_t &+ 2  \beta B^\prime \mu_{t+1} = 0 \ ,\ t \geq 0 \cr \mu_t &= R x_t + \beta A^\prime \mu_{t+1}\ ,\ t\geq 1.\cr
\end{aligned}
$$ (eq662)

Define $2 \mu_0$ to be the vector of shadow prices of $x_0$ and apply an envelope condition to
{eq}`eq661` to  deduce that

$$
\mu_0 = R x_0 + \beta A' \mu_1 ,
$$

which is a time $t=0 $ counterpart to the second equation of system {eq}`eq662`. 

Proceeding as we did above with  the undiscounted system  {eq}`lag-lqdp-eq2`, we can rearrange the first-order conditions into the
system

$$
\left[\begin{matrix} I & \beta B Q^{-1} B' \cr
             0 & \beta A' \end{matrix}\right]
\left[\begin{matrix} x_{t+1} \cr \mu_{t+1} \end{matrix}\right] =
\left[\begin{matrix} A & 0 \cr
             - R & I \end{matrix}\right] 
\left[\begin{matrix} x_t \cr \mu_t \end{matrix}\right]
$$ (eq663)

which in the special case that $\beta = 1$ agrees with equation {eq}`lag-lqdp-eq2`, as expected.

+++

By staring at system {eq}`eq663`, we can infer  identities that shed light on the structure of optimal linear regulator problems, some of which will be useful in [this lecture](https://python-advanced.quantecon.org/dyn_stack.html) when we apply and  extend the methods of this lecture to study Stackelberg and Ramsey problems.

First, note that the first block of equation system {eq}`eq663` asserts that when  $\mu_{t+1} = P x_{t+1}$, then   

$$ 
(I + \beta Q^{-1} B' P B P ) x_{t+1} = A x_t, 
$$
 
which  can be rearranged to sbe

$$
x_{t+1} = (I + \beta B Q^{-1} B' P)^{-1}  A x_t .
$$

This expression for the optimal closed loop dynamics of the state  must agree with an alternative expression that we had derived with dynamic programming, namely,

$$
x_{t+1} = (A - BF) x_t .
$$

But using  

$$
F=\beta (Q+\beta B'PB)^{-1} B'PA 
$$ (eqn:optimalFformula)

it follows that 

$$ 
A- B F = (I - \beta B (Q+ \beta B' P B)^{-1} B' P) A .
$$ 

Thus, our two expressions for the
closed loop dynamics  agree if and only if

$$ 
(I + \beta B Q^{-1} B' P )^{-1} =    (I - \beta B (Q+\beta  B' P B)^{-1} B' P) .
$$ (eqn:twofeedbackloops)

Matrix  equation {eq}`eqn:twofeedbackloops` can be verified by applying a partitioned inverse formula. 

```{note}
Just use the formula $(a - b d^{-1} c)^{-1} = a^{-1} + a^{-1} b (d - c a^{-1} b)^{-1} c a^{-1}$ for appropriate choices of the matrices $a, b, c, d$.
```



Next, note that for *any*  fixed $F$ for which eigenvalues of $A- BF$ are less than $\frac{1}{\beta}$ in modulus, the value function associated with using this rule forever is $- x_0 \tilde P x_0$ where  $\tilde P$ obeys the following  matrix  equation:

$$
\tilde P = (R + F' Q F) + \beta (A - B F)' P (A - BF) .
$$ (eq666)

Evidently, $\tilde P = P $ only when $F $ obeys formula {eq}`eqn:optimalFformula`. 

Next, note that  the second equation of system {eq}`eq663` implies the "forward looking" equation for the Lagrange multiplier

$$ 
\mu_t = R x_t + \beta A' \mu_{t+1}
$$

whose solution is 

$$
\mu_t = P x_t ,
$$

where

$$
P = R + \beta A' P (A - BF)  
$$ (eq667)

where we must require that $F$ obeys equation {eq}`eqn:optimalFformula`.

Equations {eq}`eq666` and {eq}`eq667` provide different perspectives on the optimal value function.
