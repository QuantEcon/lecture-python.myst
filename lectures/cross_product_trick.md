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

# Eliminating Cross Products 

## Overview

This lecture describes formulas for eliminating 

  * cross products between states and control in linear-quadratic dynamic programming  problems
  
  * covariances between state and measurement noises in  Kalman filtering  problems


For a linear-quadratic dynamic programming problem, the idea involves these steps

 * transform  states and controls in a way that leads to an equivalent problem with no cross-products between transformed states and controls
 * solve the transformed problem using standard formulas for problems with no cross-products between states and controls  presented in this lecture {doc}`Linear Control: Foundations <lqcontrol>`
 * transform the optimal decision rule for the altered problem into the optimal decision rule for the original problem with cross-products between states and controls

+++

## Undiscounted Dynamic Programming Problem

Here is a nonstochastic undiscounted LQ dynamic programming with cross products between
states and controls in the objective function.



The problem is defined by the 5-tuple of matrices $(A, B, R, Q, H)$
where  $R$ and $Q$ are positive definite symmetric matrices and 
$A \sim m \times m, B \sim m \times k,  Q \sim k \times k, R \sim m \times m$ and $H \sim k \times m$.


The problem is to choose $\{x_{t+1}, u_t\}_{t=0}^\infty$ to maximize 

$$
 - \sum_{t=0}^\infty (x_t' R x_t + u_t' Q u_t + 2 u_t H x_t) 
$$

subject to the linear constraints 

$$ x_{t+1} = A x_t + B u_t,  \quad t \geq 0 $$

where $x_0$ is a given initial condition. 

The solution to this undiscounted infinite-horizon problem is a time-invariant feedback rule  

$$ u_t  = -F x_t $$

where

$$ F = -(Q + B'PB)^{-1} B'PA $$

and  $P \sim m \times m $ is a positive definite solution of the algebraic matrix Riccati equation

$$
P = R + A'PA - (A'PB + H')(Q + B'PB)^{-1}(B'PA + H).
$$


+++

It can be verified that an **equivalent** problem without cross-products between states and controls
is  defined by  a 4-tuple of matrices : $(A^*, B, R^*, Q) $. 

That the omitted matrix $H=0$ indicates that there are no cross products between states and controls
in the equivalent problem. 

The matrices $(A^*, B, R^*, Q) $ defining the  equivalent problem and the value function, policy function matrices $P, F^*$ that solve it are  related to the matrices $(A, B, R, Q, H)$ defining the original problem  and the  value function, policy function matrices $P, F$ that solve the original problem by 

\begin{align*}
A^* & = A - B Q^{-1} H, \\
R^* & = R - H'Q^{-1} H, \\
P & = R^* + {A^*}' P A - ({A^*}' P B) (Q + B' P B)^{-1} B' P A^*, \\
F^* & = (Q + B' P B)^{-1} B' P A^*, \\
F & = F^* + Q^{-1} H.
\end{align*}

+++

## Kalman Filter

The **duality** that prevails  between a linear-quadratic optimal control and a Kalman filtering problem means that there is an analogous transformation that allows us to transform a Kalman filtering problem
with non-zero covariance matrix  between between shocks to states and shocks to measurements to an equivalent Kalman filtering problem with zero covariance between shocks to states and measurments.

Let's look at the appropriate transformations.


First, let's recall the Kalman filter with covariance between noises to states and measurements.

The hidden Markov model is 

\begin{align*}
x_{t+1} & = A x_t + B w_{t+1},  \\
z_{t+1} & = D x_t + F w_{t+1},  
\end{align*}

where $A \sim m \times m, B \sim m \times p $ and $D \sim k \times m, F \sim k \times p $,
and $w_{t+1}$ is the time $t+1$ component of a sequence of i.i.d. $p \times 1$ normally distibuted
random vectors with mean vector zero and covariance matrix equal to a $p \times p$ identity matrix. 

Thus, $x_t$ is $m \times 1$ and $z_t$ is $k \times 1$. 

The Kalman  filtering formulas are 


\begin{align*}
K(\Sigma_t) & = (A \Sigma_t D' + BF')(D \Sigma_t D' + FF')^{-1}, \\
\Sigma_{t+1}&  = A \Sigma_t A' + BB' - (A \Sigma_t D' + BF')(D \Sigma_t D' + FF')^{-1} (D \Sigma_t A' + FB').
\end{align*} (eq:Kalman102)
 

Define   tranformed matrices

\begin{align*}
A^* & = A - BF' (FF')^{-1} D, \\
B^* {B^*}' & = BB' - BF' (FF')^{-1} FB'.
\end{align*}

### Algorithm

A consequence of  formulas {eq}`eq:Kalman102} is that we can use the following algorithm to solve Kalman filtering problems that involve  non zero covariances between state and signal noises. 

First, compute $\Sigma, K^*$ using the ordinary Kalman filtering  formula with $BF' = 0$, i.e.,
with zero covariance matrix between random shocks to  states and  random shocks to measurements. 

That is, compute  $K^*$ and $\Sigma$ that  satisfy

\begin{align*}
K^* & = (A^* \Sigma D')(D \Sigma D' + FF')^{-1} \\
\Sigma & = A^* \Sigma {A^*}' + B^* {B^*}' - (A^* \Sigma D')(D \Sigma D' + FF')^{-1} (D \Sigma {A^*}').
\end{align*}

The Kalman gain for the original problem **with non-zero covariance** between shocks to states and measurements is then

$$
K = K^* + BF' (FF')^{-1},
$$

The state reconstruction covariance matrix $\Sigma$ for the original problem equals the state reconstrution covariance matrix for the transformed problem.

+++

## Duality table

Here is a handy table to remember how the Kalman filter and dynamic program are related.


| Dynamic Program | Kalman Filter |
| :-------------: | :-----------: |
|       $A$       |     $A'$      |
|       $B$       |     $D'$      |
|       $H$       |     $FB'$     |
|       $Q$       |     $FF'$     |
|       $R$       |     $BB'$     |
|       $F$       |     $K'$      |
|       $P$       |   $\Sigma$    |

+++


```{code-cell} ipython3

```
