---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Optimal Transport 

## Overview

The **transportation** or **optimal transport** problem is interesting both because of its many applications and its important role in the history of economic theory.

In this lecture, we describe the problem, tell how **linear programming** is a key tool for solving it,
then provide  some examples.  

We will provide other applications in followup lectures.

The optimal transport problem was studied in early work about linear programming, as summarized for example by {cite}`DoSSo`.  A modern reference about applications in economics is {cite}`Galichon_2016`.

We shall  solve our problems first by using the scipy function *linprog* and then the quantecon program *linprog_simplex*.

```{code-cell} ipython3
:tags: [hide-output]
!pip install --upgrade quantecon
```

Let's start with some imports.

```{code-cell} ipython3
import numpy as np
from scipy.optimize import linprog
from quantecon.optimize import linprog_simplex
```

## The Linear Programming Problem

Suppose that $m$ factories produce goods that must be sent to $n$ locations.

Let 

* $x_{ij}$ denote the quantity  shipped from factory $i$ to location  $j$

* $c_{ij}$ denote the cost of shipping one unit  from factory $i$ to location $j$

* $p_i$ denote the capacity of factory $i$ and $q_j$ denote the amount required at location $j$. 

*  $i = 1, 2, \dots, m$ and $j = 1, 2, \dots, n$.

A planner wants to minimize  total transportation costs subject to the following constraints:

* The amount shipped **from** each factory must equal  its capacity.

* The amount shipped **to** each location must equal the quantity required there.

The planner's problem can be expressed as the following constrained minimization problem:

$$
\begin{aligned}
\min_{x_{ij}} \ & \sum_{i=1}^m \sum_{j=1}^n c_{ij} x_{ij} \\
\mbox{subject to } \ & \sum_{j=1}^n x_{ij} = p_i, & i = 1, 2, \dots, m \\
& \sum_{i=1}^m x_{ij} = q_j, & j = 1, 2, \dots, n \\
& x_{ij} \ge 0 \\
\end{aligned}
$$ (plannerproblem)

This is an **optimal transport problem** with

* $mn$ decision variables, namely, the entries $x_{ij}$   and

* $m+n$ constraints.

Summing the $q_j$'s across all $j$'s and the $p_i$'s across all $i$'s indicates that the total capacity of all the factories  equals  total requirements at all locations: 

$$
\sum_{j=1}^n q_j = \sum_{j=1}^n \sum_{i=1}^m x_{ij} = \sum_{i=1}^m \sum_{j=1}^n x_{ij} = \sum_{i=1}^m p_i
$$ (sumconstraints)

The presence of the restrictions in {eq}`sumconstraints` will be the source of one redundancy in the complete set of restrictions that we describe below.  

More about this later.

### Vectorizing a Matrix of Decision Variables

A **matrix** of decision variables $x_{ij}$ appears in problem {eq}`plannerproblem`. 

The Scipy function *linprog* expects to see a **vector** of decision variables. 

This situation impels us to want to  rewrite our problem in terms of a **vector** of decision variables.

Let 

* $X, C$ be $m \times n$ matrices with entries $x_{ij}, c_{ij}$,

* $p$ be $m$-dimensional vector with entries $p_i$,

* $q$ be $n$-dimensional vector with entries $q_j$. 

Where $\mathbf{1}_n$ denotes $n$-dimensional column vector $(1, 1, \dots, 1)'$, our  problem can now be expressed compactly as:

$$
\begin{aligned}
\min_{X} \ & \operatorname{tr} (C' X) \\
\mbox{subject to } \ & X \ \mathbf{1}_n = p \\
& X' \ \mathbf{1}_m = q \\
& X \ge 0 \\
\end{aligned}
$$

We can convert the matrix $X$ into a vector by stacking all of its columns into a  column vector. 

Doing this is called **vectorization**, an operation that we denote  $\operatorname{vec}(X)$. 

Similarly, we convert the matrix $C$ into an $mn$-dimensional vector $\operatorname{vec}(C)$.

The objective function can be expressed as the inner product between $\operatorname{vec}(C)$ and $\operatorname{vec}(X)$:

$$
\operatorname{vec}(C)' \cdot \operatorname{vec}(X).
$$

To express the constraints in terms of $\operatorname{vec}(X)$, we use a **Kronecker product** denoted by $\otimes$ and defined as follows.

Suppose $A$ is an $m \times s$ matrix with entries $(a_{ij})$ and that $B$ is an $n \times t$ matrix.

A **Kronecker product** of $A$ and $B$ is defined by

$$
A \otimes B = 
\begin{bmatrix}
a_{11}B & a_{12}B & \dots & a_{1s}B \\ 
a_{21}B & a_{22}B & \dots & a_{2s}B \\ 
  &   & \vdots &   \\ 
a_{m1}B & a_{m2}B & \dots & a_{ms}B \\ 
\end{bmatrix}.
$$

$A \otimes B$ is an $mn \times st$ matrix.

It has the property that for any $m \times n$ matrix $X$

$$
\operatorname{vec}(A'XB) = (B' \otimes A') \operatorname{vec}(X).
$$ (kroneckerprop)

We can now express our constraints in terms of $\operatorname{vec}(X)$.

Let $A = \mathbf{I}_m', B = \mathbf{1}_n$.

By equation {eq}`kroneckerprop`

$$
X \ \mathbf{1}_n = \operatorname{vec}(X \ \mathbf{1}_n) = \operatorname{vec}(\mathbf{I}_m X \ \mathbf{1}_n) = (\mathbf{1}_n' \otimes \mathbf{I}_m) \operatorname{vec}(X).
$$

where  $\mathbf{I}_m$ denotes the $m \times m$ identity matrix.

Constraint $X \ \mathbf{1}_n = p$ can now be written as:

$$
(\mathbf{1}_n' \otimes \mathbf{I}_m) \operatorname{vec}(X) = p.
$$

Similarly, the constraint $X' \ \mathbf{1}_m = q$ can be rewriten as:

$$
(\mathbf{I}_n \otimes \mathbf{1}_m') \operatorname{vec}(X) = q.
$$

Our problem can now be expressed in terms of an $mn$-dimensional vector of decision variables:

$$
\begin{aligned}
\min_{z} \ & \operatorname{vec}(C)' z \\
\mbox{subject to } \ & A z = b \\
& z \ge 0 \\
\end{aligned}
$$ (decisionvars)

where

$$
A = 
\begin{bmatrix}
\mathbf{1}_n' \otimes \mathbf{I}_m \\ 
\mathbf{I}_n \otimes \mathbf{1}_m' \\ 
\end{bmatrix},
b = \begin{bmatrix} p \\ q \\ \end{bmatrix}
$$

where $z = \operatorname{vec}(X)$.

**Example:**

We now provide an example that takes the form {eq}`decisionvars` that we'll solve by deploying the function *linprog*.

The table below provides numbers for the requirements vector $q$, the capacity vector $p$,
and entries $c_{ij}$  of the cost-of-shipping matrix $C$.


```{raw} html
<table>
    <tr>
	    <th> </th>
        <th colspan="3"><center>Factory</center></th>
	    <th rowspan="2">Requirement</th>  
	</tr >
    <tr>
        <th> Location </th> <th>1</th> <th>2</th> <th>3</th>
	</tr>
    <tr>
	    <td>1</td>  <td>10</td> <td>20</td> <td>30</td> <td>25</td>
	</tr>
    <tr>
	    <td>2</td> <td>15</td> <td>40</td> <td>35</td> <td>115</td>
	</tr>
    <tr>
	    <td>3</td> <td>20</td> <td>15</td> <td>40</td> <td>60</td>
	</tr>
    <tr>
	    <td>4</td> <td>20</td> <td>30</td> <td>55</td> <td>30</td>
	</tr>
    <tr>
	    <td>5</td> <td>40</td> <td>30</td> <td>25</td> <td>70</td>
	</tr>
    <tr>
	    <td>Capacity</td> <td>50</td> <td>100</td> <td>150</td> <td>300</td>
	</tr>
</table>
```

The numbers in the above table tell us to construct the following objects:

$$
m = 3, n = 5, \\
p = (50,100,150)', q = (25,115,60,30,70)', \\ 
C = 
\begin{bmatrix}
10 &15 &20 &20 &40 \\
20 &40 &15 &30 &30 \\
30 &35 &40 &55 &25 \\
\end{bmatrix}.
$$

Let's write Python code that sets up the problem and solves it.

```{code-cell} ipython3
# Define parameters
m = 3
n = 5

p = np.array([50, 100, 150])
q = np.array([25, 115, 60, 30, 70])

C = np.array([[10, 15, 20, 20, 40],
              [20, 40, 15, 30, 30],
              [30, 35, 40, 55, 25]])

# Vectorize matrix C
C_vec = C.reshape((m*n, 1), order='F')

# Construct matrix A by Kronecker product
A1 = np.kron(np.ones((1, n)), np.identity(m))
A2 = np.kron(np.identity(n), np.ones((1, m)))
A = np.vstack([A1, A2])

# Construct vector b
b = np.hstack([p, q])

# Solve the primal problem
res = linprog(C_vec, A_eq=A, b_eq=b, method='Revised simplex')

# Print results
print("message:", res.message)
print("nit:", res.nit)
print("fun:", res.fun)
print("z:", res.x)
print("X:", res.x.reshape((m,n), order='F'))
```

```{code-cell} ipython3
C.reshape((m*n, 1), order='F')
```

```{code-cell} ipython3
C.reshape((m*n, 1), order='C')
```

```{code-cell} ipython3
C.reshape((m*n, 1), order='A')
```

**Interpreting the warning:**  

The above warning message from scipy pointing out that A is not full rank.

This indicates that the problem has been set up to include one or more  redundant constraints.

Here, the source of the redundancy is that the set of restrictions {eq}`sumconstraints`.

Let's explore this further by printing out $A$ and staring at it.

```{code-cell} ipython3
A
```

The singularity of $A$ reflects that the  first three constraints and the last five constraints  both require  that "total requirements equal total capacities" expressed in {eq}`sumconstraints`.

One  equality constraint here is redundant.

Below we drop one of the equality constraints, and use only  7 of them.

After doing this, we attain the same minimized cost. 

However, we find a  different transportation plan.

Though it is a different plan, it attains the same cost!

```{code-cell} ipython3
linprog(C_vec, A_eq=A[:-1], b_eq=b[:-1], method='Revised simplex')
```

```{code-cell} ipython3
%timeit linprog(C_vec, A_eq=A[:-1], b_eq=b[:-1], method='Revised simplex')
```

```{code-cell} ipython3
%timeit linprog(C_vec, A_eq=A, b_eq=b, method='Revised simplex')
```

Evidently, it is slightly quicker to work with the system that removed a redundant constraint.

Let's drill down and do some more calculations to help us understand whether or not our finding **two** different optimal transport plans reflects our having dropped a redundant equality constraint.

```{admonition} Hint
It will turn out that dropping a redundant equality isn't really what mattered.
```

To verify our hint, we shall simply use **all** of  the original equality constraints (including a redundant one), but we'll just shuffle the order of the constraints.

```{code-cell} ipython3
arr = np.arange(m+n)
```

```{code-cell} ipython3
sol_found = []
cost = []

# simulate 1000 times
for i in range(1000):

    np.random.shuffle(arr)
    res_shuffle = linprog(C_vec, A_eq=A[arr], b_eq=b[arr], method='Revised simplex')

    # if find a new solution
    sol = tuple(res_shuffle.x)
    if sol not in sol_found:
        sol_found.append(sol)
        cost.append(res_shuffle.fun)
```

```{code-cell} ipython3
for i in range(len(sol_found)):
    print(f"transportation plan {i}: ", sol_found[i])
    print(f"     minimized cost {i}: ", cost[i])
```

**Ah hah!** As you can see, putting constraints in different orders in this case uncovers two optimal transportation plans that achieve the same minimized cost.

These are the same two plans computed early.

Next, we show that leaving out the first constraint "accidentally" leads to the initial plan that we computed.

```{code-cell} ipython3
linprog(C_vec, A_eq=A[1:], b_eq=b[1:], method='Revised simplex')
```

Let's compare this transport plan with

```{code-cell} ipython3
res.x
```

Here the matrix $X$ contains entries $x_{ij}$ that tell amounts shipped **from** factor $i = 1, 2, 3$
**to** location $j=1,2, \ldots, 5$.

The vector $z$ evidently equals $\operatorname{vec}(X)$.

The minimized cost from the optimal transport plan is given by the $fun$ variable.

We can also solve an optimal transportation problem using a powerful tool from `quantecon`, namely,`quantecon.optimize.linprog_simplex`. 

It uses the same simplex algorithm as `scipy.optimize.linprog`, but the program is accelerated by using `numba`.

As you will see very soon, by using `scipy.optimize.linprog` the time required to solve an optimal transportation problem can be reduced significantly.

```{code-cell} ipython3
# construct matrices/vectors for linprog_simplex
c = C.flatten()

# Equality constraints
A_eq = np.zeros((m+n, m*n))
for i in range(m):
    for j in range(n):
        A_eq[i, i*n+j] = 1
        A_eq[m+j, i*n+j] = 1

b_eq = np.hstack([p, q])
```

Since `quantecon.optimize.linprog_simplex` does maximization instead of minimization, we need to put a negative sign before vector `c`.

```{code-cell} ipython3
res_qe = linprog_simplex(-c, A_eq=A_eq, b_eq=b_eq)
```

Since the two LP solvers use the same simplex algorithm, we expect to get exactly the same solutions

```{code-cell} ipython3
res_qe.x.reshape((m, n), order='C')
```

```{code-cell} ipython3
res.x.reshape((m, n), order='F')
```

Let's do a speed comparison between `scipy.optimize.linprog` and `quantecon.optimize.linprog_simplex`.

```{code-cell} ipython3
# scipy.optimize.linprog
%timeit res = linprog(C_vec, A_eq=A[:-1, :], b_eq=b[:-1], method='Revised simplex')
```

```{code-cell} ipython3
# quantecon.optimize.linprog_simplex
%timeit out = linprog_simplex(-c, A_eq=A_eq, b_eq=b_eq)
```

As you can see, the `quantecon.optimize.linprog_simplex` is almost 200 times faster.

## The Dual Problem

Let $u, v$ denotes vectors of dual decision variables with entries $(u_i), (v_j)$.

The **dual** to  **minimization** problem {eq}`plannerproblem` is the **maximization** problem:

$$
\begin{aligned}
\max_{u_i, v_j} \ & \sum_{i=1}^m p_i u_i + \sum_{j=1}^n q_j v_j \\
\mbox{subject to } \ & u_i + v_j \le c_{ij}, \ i = 1, 2, \dots, m;\ j = 1, 2, \dots, n \\
\end{aligned}
$$ (dualproblem)

The dual problem is also a linear programming problem.

It has $m+n$ dual variables and $mn$ constraints. 

Vectors $u$ and $v$ of **values** are attached to the first and the second sets of primal constraits, respectively.

Thus, $u$ is attached to the constraints 

* $(\mathbf{1}_n' \otimes \mathbf{I}_m) \operatorname{vec}(X) = p$ 
 
and  $v$ is attached to constraints 
 
* $(\mathbf{I}_n \otimes \mathbf{1}_m') \operatorname{vec}(X) = q.$

Components of the vectors $u$ and $v$ of **values**  are **shadow prices** of the quantities appearing on the right sides of those constraints.

We can write the dual problem as 

$$
\begin{aligned}
\max_{u_i, v_j} \ & p u + q v \\
\mbox{subject to } \ & A' \begin{bmatrix} u \\ v \\ \end{bmatrix} = \operatorname{vec}(C) \\
\end{aligned}
$$ (dualproblem2)

For the same numerical example described above, let's solve the dual problem.

```{code-cell} ipython3
# Solve the dual problem
res_dual = linprog(-b, A_ub=A.T, b_ub=C_vec,
                   bounds=[(None, None)]*(m+n), method='Revised simplex')

#Print results
print("message:", res_dual.message)
print("nit:", res_dual.nit)
print("fun:", res_dual.fun)
print("u:", res_dual.x[:m])
print("v:", res_dual.x[-n:])
```

We can also solve the dual problem using [quantecon.optimize.linprog_simplex](https://quanteconpy.readthedocs.io/en/latest/optimize/linprog_simplex.html).

```{code-cell} ipython3
res_dual_qe = linprog_simplex(b_eq, A_ub=A_eq.T, b_ub=c)
```

And the shadow prices computed by the two programs are identical.

```{code-cell} ipython3
res_dual_qe.x
```

```{code-cell} ipython3
res_dual.x
```

We can compare computational times from using our two tools.

```{code-cell} ipython3
%timeit linprog(-b, A_ub=A.T, b_ub=C_vec, bounds=[(None, None)]*(m+n), method='Revised simplex')
```

```{code-cell} ipython3
%timeit linprog_simplex(b_eq, A_ub=A_eq.T, b_ub=c)
```

`quantecon.optimize.linprog_simplex` solves the dual problem 10 times faster.

Just for completeness, let's  solve the dual problems with nonsingular $A$ matrices that we create by dropping a redundant equality constraint.

Try first leaving out the first constraint:

```{code-cell} ipython3
linprog(-b[1:], A_ub=A[1:].T, b_ub=C_vec,
        bounds=[(None, None)]*(m+n-1), method='Revised simplex')
```

Not let's instead leave out the last constraint:

```{code-cell} ipython3
linprog(-b[:-1], A_ub=A[:-1].T, b_ub=C_vec,
        bounds=[(None, None)]*(m+n-1), method='Revised simplex')
```

### Interpretation of dual problem

By **strong duality**, we know that:

$$
\sum_{i=1}^m \sum_{j=1}^n c_{ij} x_{ij}  = \sum_{i=1}^m p_i u_i + \sum_{j=1}^n q_j v_j
$$

One unit more  capacity in factory $i$, i.e. $p_i$,   results in $u_i$ more transportation costs.

Thus, $u_i$ describes the  cost of shipping one unit  **from** factory $i$. 

Call this  the ship-out cost of one unit  shipped from factory $i$.

Similarly, $v_j$ is the  cost of shipping one unit **to** location $j$.

Call this  the ship-in cost of one unit  to location $j$.

Strong duality implies that  total transprotation costs  equals   total ship-out costs **plus**   total ship-in costs.

It is reasonable that, for one unit of a product, ship-out cost $u_i$ **plus** ship-in cost $v_j$ should  equal transportation cost $c_{ij}$.

This equality is assured by   **complementary slackness** conditions that state that whenever $x_{ij} > 0$, meaning that there are positive shipments  from factory $i$ to location $j$,    it must be true that  $u_i + v_j = c_{ij}$.
