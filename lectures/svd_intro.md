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

# Singular Value Decomposition (SVD)

## Overview

The **singular value decomposition** (SVD) is a work-horse in applications of least squares projection that
form  foundations for many statistical and  machine learning methods.

After defining the SVD, we'll describe how it connects to

* **four fundamental spaces** of linear algebra
* under-determined and over-determined **least squares regressions**
* **principal components analysis** (PCA)

Like principal components analysis (PCA), DMD can be thought of as a data-reduction procedure that  represents salient patterns by projecting data onto a limited set of factors.

In a sequel to this lecture about  {doc}`Dynamic Mode Decompositions <var_dmd>`, we'll describe how SVD's provide ways rapidly to compute reduced-order approximations to first-order Vector Autoregressions (VARs).

##  The Setting

Let $X$ be an $m \times n$ matrix of rank $p$.

Necessarily, $p \leq \min(m,n)$.

In  much of this lecture, we'll think of $X$ as a matrix of **data** in which

* each column is an **individual** -- a time period or person, depending on the application

* each row is a **random variable** describing an attribute of a time period or a person, depending on the application


We'll be interested in  two  situations

* A **short and fat** case in which $m << n$, so that there are many more columns (individuals) than rows (attributes).

* A  **tall and skinny** case in which $m >> n$, so that there are many more rows  (attributes) than columns (individuals).


We'll apply a **singular value decomposition** of $X$ in both situations.

In the $ m < < n$ case  in which there are many more individuals $n$ than attributes $m$, we can calculate sample moments of  a joint distribution  by taking averages  across observations of functions of the observations.

In this $ m < < n$ case,  we'll look for **patterns** by using a **singular value decomposition** to do a **principal components analysis** (PCA).

In the $m > > n$  case in which there are many more attributes $m$ than individuals $n$ and when we are in a time-series setting in which $n$ equals the number of time periods covered in the data set $X$, we'll proceed in a different way.

We'll again use a **singular value decomposition**,  but now to construct a **dynamic mode decomposition** (DMD)

## Singular Value Decomposition

A **singular value decomposition** of an $m \times n$ matrix $X$ of rank $p \leq \min(m,n)$ is

$$
X  = U \Sigma V^\top
$$ (eq:SVD101)

where

$$
\begin{aligned}
UU^\top  &  = I  &  \quad U^\top  U = I \cr
VV^\top  & = I & \quad V^\top  V = I
\end{aligned}
$$

and

* $U$ is an $m \times m$ orthogonal  matrix of **left singular vectors** of $X$
* Columns of $U$ are eigenvectors of $X X^\top $
* $V$ is an $n \times n$ orthogonal matrix of **right singular vectors** of $X$
* Columns of $V$  are eigenvectors of $X^\top  X$
* $\Sigma$ is an $m \times n$ matrix in which the first $p$ places on its main diagonal are positive numbers $\sigma_1, \sigma_2, \ldots, \sigma_p$ called **singular values**; remaining entries of $\Sigma$ are all zero

* The $p$ singular values are positive square roots of the eigenvalues of the $m \times m$ matrix  $X X^\top $ and also of the $n \times n$ matrix $X^\top  X$

* We adopt a convention that when $U$ is a complex valued matrix, $U^\top $ denotes the **conjugate-transpose** or **Hermitian-transpose** of $U$, meaning that
$U_{ij}^\top $ is the complex conjugate of $U_{ji}$.

* Similarly, when $V$ is a complex valued matrix, $V^\top $ denotes the **conjugate-transpose** or **Hermitian-transpose** of $V$


The matrices $U,\Sigma,V$ entail linear transformations that reshape in vectors in the following ways:

* multiplying vectors  by the unitary matrices $U$ and $V$ **rotates** them, but leaves **angles between vectors** and **lengths of vectors** unchanged.
* multiplying vectors by the diagonal  matrix $\Sigma$ leaves **angles between vectors** unchanged but **rescales** vectors.

Thus, representation {eq}`eq:SVD101` asserts that multiplying an $n \times 1$  vector $y$ by the $m \times n$ matrix $X$
amounts to performing the following three multiplications of $y$ sequentially:

* **rotating** $y$ by computing $V^\top  y$
* **rescaling** $V^\top  y$ by multiplying it by $\Sigma$
* **rotating** $\Sigma V^\top  y$ by multiplying it by $U$

This structure of the $m \times n$ matrix  $X$ opens the door to constructing systems
of data **encoders** and **decoders**.

Thus,

* $V^\top  y$ is an encoder
* $\Sigma$ is an operator to be applied to the encoded data
* $U$ is a decoder to be applied to the output from applying operator $\Sigma$ to the encoded data

We'll apply this circle of ideas  later in this lecture when we study Dynamic Mode Decomposition.

**Road Ahead**

What we have described above  is called a **full** SVD.

In a **full** SVD, the  shapes of $U$, $\Sigma$, and $V$ are $\left(m, m\right)$, $\left(m, n\right)$, $\left(n, n\right)$, respectively.

Later we'll also describe an **economy** or **reduced** SVD.

Before we study a **reduced** SVD we'll say a little more about properties of a **full** SVD.

## Four Fundamental Subspaces

Let  ${\mathcal C}$ denote a column space, ${\mathcal N}$ denote a null space, and ${\mathcal R}$ denote a row space.

Let's start by recalling the four fundamental subspaces of an $m \times n$
matrix $X$ of rank $p$.

* The **column space** of $X$, denoted ${\mathcal C}(X)$, is the span of the  columns of  $X$, i.e., all vectors $y$ that can be written as linear combinations of columns of $X$. Its dimension is $p$.
* The **null space** of $X$, denoted ${\mathcal N}(X)$ consists of all vectors $y$ that satisfy
$X y = 0$. Its dimension is $n-p$.
* The **row space** of $X$, denoted ${\mathcal R}(X)$ is the column space of $X^\top $. It consists of all
vectors $z$ that can be written as  linear combinations of rows of $X$. Its dimension is $p$.
* The **left null space** of $X$, denoted ${\mathcal N}(X^\top )$, consist of all vectors $z$ such that
$X^\top  z =0$.  Its dimension is $m-p$.

For a  full SVD of a matrix $X$, the matrix $U$ of left singular vectors  and the matrix $V$ of right singular vectors contain orthogonal bases for all four subspaces.

They form two pairs of orthogonal subspaces
that we'll describe now.

Let $u_i, i = 1, \ldots, m$ be the $m$ column vectors of $U$ and let
$v_i, i = 1, \ldots, n$ be the $n$ column vectors of $V$.

Let's write the full SVD of X as

$$
X = \begin{bmatrix} U_L & U_R \end{bmatrix} \begin{bmatrix} \Sigma_p & 0 \cr 0 & 0 \end{bmatrix}
     \begin{bmatrix} V_L & V_R \end{bmatrix}^\top
$$ (eq:fullSVDpartition)

where  $ \Sigma_p$ is  a $p \times p$ diagonal matrix with the $p$ singular values on the diagonal and

$$
\begin{aligned}
U_L & = \begin{bmatrix}u_1 & \cdots  & u_p \end{bmatrix},  \quad U_R  = \begin{bmatrix}u_{p+1} & \cdots u_m \end{bmatrix}  \cr
V_L & = \begin{bmatrix}v_1 & \cdots  & v_p \end{bmatrix} , \quad U_R  = \begin{bmatrix}v_{p+1} & \cdots u_n \end{bmatrix}
\end{aligned}
$$


Representation {eq}`eq:fullSVDpartition` implies that

$$
X \begin{bmatrix} V_L & V_R \end{bmatrix} = \begin{bmatrix} U_L & U_R \end{bmatrix} \begin{bmatrix} \Sigma_p & 0 \cr 0 & 0 \end{bmatrix}
$$

or

$$
\begin{aligned}
X V_L & = U_L \Sigma_p \cr
X V_R & = 0
\end{aligned}
$$ (eq:Xfour1a)

or

$$
\begin{aligned}
X v_i & = \sigma_i u_i , \quad i = 1, \ldots, p \cr
X v_i & = 0 ,  \quad i = p+1, \ldots, n
\end{aligned}
$$ (eq:orthoortho1)

Equations {eq}`eq:orthoortho1` tell how the transformation $X$ maps a pair of orthonormal  vectors $v_i, v_j$ for $i$ and $j$ both less than or equal to the rank $p$ of $X$ into a pair of orthonormal vectors $u_i, u_j$.

Equations {eq}`eq:Xfour1a` assert that

$$
\begin{aligned}
{\mathcal C}(X) & = {\mathcal C}(U_L) \cr
{\mathcal N}(X) & = {\mathcal C} (V_R)
\end{aligned}
$$


Taking transposes on both sides of representation {eq}`eq:fullSVDpartition` implies


$$
X^\top  \begin{bmatrix} U_L & U_R \end{bmatrix} = \begin{bmatrix} V_L & V_R \end{bmatrix} \begin{bmatrix} \Sigma_p & 0 \cr 0 & 0 \end{bmatrix}
$$

or

$$
\begin{aligned}
X^\top  U_L & = V_L \Sigma_p \cr
X^\top  U_R & = 0
\end{aligned}
$$  (eq:Xfour1b)

or

$$
\begin{aligned}
X^\top  u_i & = \sigma_i v_i, \quad i=1, \ldots, p \cr
X^\top  u_i & = 0 \quad i= p+1, \ldots, m
\end{aligned}
$$ (eq:orthoortho2)

Notice how equations {eq}`eq:orthoortho2` assert that  the transformation $X^\top $ maps a pair of distinct orthonormal  vectors $u_i, u_j$  for $i$ and $j$ both less than or equal to the rank $p$ of $X$ into a pair of distinct orthonormal vectors $v_i, v_j$ .


Equations {eq}`eq:Xfour1b` assert that

$$
\begin{aligned}
{\mathcal R}(X) & \equiv  {\mathcal C}(X^\top ) = {\mathcal C} (V_L) \cr
{\mathcal N}(X^\top ) & = {\mathcal C}(U_R)
\end{aligned}
$$



Thus, taken together, the systems of equations {eq}`eq:Xfour1a` and {eq}`eq:Xfour1b`
describe the  four fundamental subspaces of $X$ in the following ways:

$$
\begin{aligned}
{\mathcal C}(X) & = {\mathcal C}(U_L) \cr
{\mathcal N}(X^\top ) & = {\mathcal C}(U_R) \cr
{\mathcal R}(X) & \equiv  {\mathcal C}(X^\top ) = {\mathcal C} (V_L) \cr
{\mathcal N}(X) & = {\mathcal C} (V_R) \cr

\end{aligned}
$$ (eq:fourspaceSVD)

Since $U$ and $V$ are both orthonormal matrices, collection {eq}`eq:fourspaceSVD` asserts that

* $U_L$ is an orthonormal basis for the column space of $X$
* $U_R$ is an orthonormal basis for the null space of $X^\top $
* $V_L$ is an orthonormal basis for the row space of $X$
* $V_R$ is an orthonormal basis for the null space of $X$


We have verified the four claims in {eq}`eq:fourspaceSVD` simply  by performing the multiplications called for by the right side of {eq}`eq:fullSVDpartition` and reading them.

The claims in {eq}`eq:fourspaceSVD` and the fact that $U$ and $V$ are both unitary (i.e, orthonormal) matrices  imply
that

* the column space of $X$ is orthogonal to the null space of $X^\top $
* the null space of $X$ is orthogonal to the row space of $X$

Sometimes these properties are described with the following two pairs of orthogonal complement subspaces:

* ${\mathcal C}(X)$ is the orthogonal complement of $ {\mathcal N}(X^\top )$
* ${\mathcal R}(X)$ is the orthogonal complement  ${\mathcal N}(X)$

Let's do an example.


```{code-cell} ipython3
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
```

Having imported these modules, let's do the example.

```{code-cell} ipython3
np.set_printoptions(precision=2)

# Define the matrix
A = np.array([[1, 2, 3, 4, 5],
              [2, 3, 4, 5, 6],
              [3, 4, 5, 6, 7],
              [4, 5, 6, 7, 8],
              [5, 6, 7, 8, 9]])

# Compute the SVD of the matrix
U, S, V = np.linalg.svd(A,full_matrices=True)

# Compute the rank of the matrix
rank = np.linalg.matrix_rank(A)

# Print the rank of the matrix
print("Rank of matrix:\n", rank)
print("S: \n", S)

# Compute the four fundamental subspaces
row_space = U[:, :rank]
col_space = V[:, :rank]
null_space = V[:, rank:]
left_null_space = U[:, rank:]


print("U:\n", U)
print("Column space:\n", col_space)
print("Left null space:\n", left_null_space)
print("V.T:\n", V.T)
print("Row space:\n", row_space.T)
print("Right null space:\n", null_space.T)
```

## Eckart-Young Theorem

Suppose that we want to construct  the best rank $r$ approximation of an $m \times n$ matrix $X$.

By best, we mean a  matrix $X_r$ of rank $r < p$ that, among all rank $r$ matrices, minimizes

$$ 
|| X - X_r || 
$$

where $ || \cdot || $ denotes a norm of a matrix $X$ and where $X_r$ belongs to the space of all rank $r$ matrices
of dimension $m \times n$.

Three popular **matrix norms**  of an $m \times n$ matrix $X$ can be expressed in terms of the singular values of $X$

* the **spectral** or $l^2$ norm $|| X ||_2 = \max_{||y|| \neq 0} \frac{||X y ||}{||y||} = \sigma_1$
* the **Frobenius** norm $||X ||_F = \sqrt{\sigma_1^2 + \cdots + \sigma_p^2}$
* the **nuclear** norm $ || X ||_N = \sigma_1 + \cdots + \sigma_p $

The Eckart-Young theorem states that for each of these three norms, same rank $r$ matrix is best and that it equals

$$
\hat X_r = \sigma_1 U_1 V_1^\top  + \sigma_2 U_2 V_2^\top  + \cdots + \sigma_r U_r V_r^\top
$$ (eq:Ekart)

This is a very powerful theorem that says that we can take our $ m \times n $ matrix $X$ that in not full rank, and we can best approximate it by a full rank $p \times p$ matrix through the SVD. 

Moreover, if some of these $p$ singular values carry more information than others, and if we want to have the most amount of information with the least amount of data, we can take $r$ leading singular values ordered by magnitude.

We'll say more about this later when we present Principal Component Analysis.

You can read about the Eckart-Young theorem and some of its uses [here](https://en.wikipedia.org/wiki/Low-rank_approximation).

We'll make use of this theorem when we discuss principal components analysis (PCA) and also dynamic mode decomposition (DMD).

## Full and Reduced SVD's

Up to now we have described properties of a **full** SVD in which shapes of $U$, $\Sigma$, and $V$ are $\left(m, m\right)$, $\left(m, n\right)$, $\left(n, n\right)$, respectively.

There is  an alternative bookkeeping convention called an **economy** or **reduced** SVD in which the shapes of $U, \Sigma$ and $V$ are different from what they are in a full SVD.

Thus, note that because we assume that $X$ has rank $p$, there are only $p$ nonzero singular values, where $p=\textrm{rank}(X)\leq\min\left(m, n\right)$.

A **reduced** SVD uses this fact to express $U$, $\Sigma$, and $V$ as matrices with shapes $\left(m, p\right)$, $\left(p, p\right)$, $\left( n, p\right)$.

You can read about reduced and full SVD here
<https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html>

For a full SVD,

$$
\begin{aligned}
UU^\top  &  = I  &  \quad U^\top  U = I \cr
VV^\top  & = I & \quad V^\top  V = I
\end{aligned}
$$

But not all these properties hold for a  **reduced** SVD.

Which properties hold depend on whether we are in a **tall-skinny** case or a **short-fat** case.

 * In a **tall-skinny** case in which $m > > n$, for a **reduced** SVD

$$
\begin{aligned}
UU^\top  &  \neq I  &  \quad U^\top  U = I \cr
VV^\top  & = I & \quad V^\top  V = I
\end{aligned}
$$

* In a **short-fat** case in which $m < < n$, for a **reduced** SVD

$$
\begin{aligned}
UU^\top  &  = I  &  \quad U^\top  U = I \cr
VV^\top  & = I & \quad V^\top  V \neq I
\end{aligned}
$$

When we study Dynamic Mode Decomposition below, we shall want to remember these properties when we use a  reduced SVD to compute some DMD representations.


Let's do an  exercise  to compare **full** and **reduced** SVD's.

To review,


* in a **full** SVD

  -  $U$ is $m \times m$
  -  $\Sigma$ is $m \times n$
  -  $V$ is $n \times n$

* in a **reduced** SVD

  -  $U$ is $m \times p$
  - $\Sigma$ is $p\times p$
  -  $V$ is $n \times p$

First, let's study a case in which $m = 5 > n = 2$.

(This is a small example of the **tall-skinny** case that will concern us when we study **Dynamic Mode Decompositions** below.)

```{code-cell} ipython3
import numpy as np
X = np.random.rand(5,2)
U, S, V = np.linalg.svd(X,full_matrices=True)  # full SVD
Uhat, Shat, Vhat = np.linalg.svd(X,full_matrices=False) # economy SVD
print('U, S, V =')
U, S, V
```

```{code-cell} ipython3
print('Uhat, Shat, Vhat = ')
Uhat, Shat, Vhat
```

```{code-cell} ipython3
rr = np.linalg.matrix_rank(X)
print(f'rank of X = {rr}')
```


**Properties:**

* Where $U$ is constructed via a full SVD, $U^\top  U = I_{m\times m}$ and  $U U^\top  = I_{m \times m}$
* Where $\hat U$ is constructed via a reduced SVD, although $\hat U^\top  \hat U = I_{p\times p}$, it happens that  $\hat U \hat U^\top  \neq I_{m \times m}$

We illustrate these properties for our example with the following code cells.

```{code-cell} ipython3
UTU = U.T@U
UUT = U@U.T
print('UUT, UTU = ')
UUT, UTU
```


```{code-cell} ipython3
UhatUhatT = Uhat@Uhat.T
UhatTUhat = Uhat.T@Uhat
print('UhatUhatT, UhatTUhat= ')
UhatUhatT, UhatTUhat
```




**Remarks:**

The cells above illustrate the application of the  `full_matrices=True` and `full_matrices=False` options.
Using `full_matrices=False` returns a reduced singular value decomposition.

The **full** and **reduced** SVD's both accurately  decompose an $m \times n$ matrix $X$

When we study Dynamic Mode Decompositions below, it  will be important for us to remember the preceding properties of full and reduced SVD's in such tall-skinny cases.





Now let's turn to a short-fat case.

To illustrate this case,  we'll set $m = 2 < 5 = n $ and compute both full and reduced SVD's.

```{code-cell} ipython3
import numpy as np
X = np.random.rand(2,5)
U, S, V = np.linalg.svd(X,full_matrices=True)  # full SVD
Uhat, Shat, Vhat = np.linalg.svd(X,full_matrices=False) # economy SVD
print('U, S, V = ')
U, S, V
```

```{code-cell} ipython3
print('Uhat, Shat, Vhat = ')
Uhat, Shat, Vhat
```
Let's verify that our reduced SVD accurately represents $X$

```{code-cell} ipython3
SShat=np.diag(Shat)
np.allclose(X, Uhat@SShat@Vhat)
```

## Polar Decomposition

A **reduced** singular value decomposition (SVD) of $X$ is related to a **polar decomposition** of $X$

$$
X  = SQ
$$

where

$$
\begin{aligned}
 S & = U\Sigma U^\top  \cr
Q & = U V^\top
\end{aligned}
$$

Here

* $S$ is  an $m \times m$  **symmetric** matrix
* $Q$ is an $m \times n$  **orthogonal** matrix

and in our reduced SVD

* $U$ is an $m \times p$ orthonormal matrix
* $\Sigma$ is a $p \times p$ diagonal matrix
* $V$ is an $n \times p$ orthonormal

## Application: Principal Components Analysis (PCA)

Let's begin with a case in which $n >> m$, so that we have many  more individuals $n$ than attributes $m$.

The  matrix $X$ is **short and fat**  in an  $n >> m$ case as opposed to a **tall and skinny** case with $m > > n $ to be discussed later.

We regard  $X$ as an  $m \times n$ matrix of **data**:

$$
X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_n\end{bmatrix}
$$

where for $j = 1, \ldots, n$ the column vector $X_j = \begin{bmatrix}x_{1j}\\x_{2j}\\\vdots\\x_{mj}\end{bmatrix}$ is a  vector of observations on variables $\begin{bmatrix}X_1\\X_2\\\vdots\\X_m\end{bmatrix}$.

In a **time series** setting, we would think of columns $j$ as indexing different __times__ at which random variables are observed, while rows index different random variables.

In a **cross-section** setting, we would think of columns $j$ as indexing different __individuals__ for  which random variables are observed, while rows index different **attributes**.

As we have seen before, the SVD is a way to decompose a matrix into useful components, just like polar decomposition, eigendecomposition, and many others. 

PCA, on the other hand, is a method that builds on the SVD to analyze data. The goal is to apply certain steps, to help better visualize patterns in data, using statistical tools to capture the most important patterns in data.

**Step 1: Standardize the data:** 

Because our data matrix may hold variables of different units and scales, we first need to standardize the data. 

First by computing the average of each row of $X$.

$$
\bar{X_i}= \frac{1}{n} \sum_{j = 1}^{n} x_{ij}
$$

We then create an average matrix out of these means:


$$
\bar{X} =  \begin{bmatrix} \bar{X_1} \\ \bar{X_2} \\ \ldots \\ \bar{X_m}\end{bmatrix}\begin{bmatrix}1 \mid 1 \mid \cdots \mid 1 \end{bmatrix}
$$

And subtract out of the original matrix to create a mean centered matrix:

$$
B = X - \bar{X}
$$


**Step 2: Compute the covariance matrix:** 

Then because we want to extract the relationships between variables rather than just their magnitude, in other words, we want to know how they can explain each other, we compute the covariance matrix of $B$.

$$
C = \frac{1}{n} BB^{\top}
$$

**Step 3: Decompose the covariance matrix and arrange the singular values:**

Since the matrix $C$ is positive definite, we can eigendecompose it, find its eigenvalues, and rearrange the eigenvalue and eigenvector matrices in a decreasing order.

The eigendecomposition of $C$ can be found by decomposing $B$ instead. Since $B$ is not a square matrix, we obtain an SVD of $B$:

$$
\begin{aligned}
B B^\top &= U \Sigma V^\top (U \Sigma V^{\top})^{\top}\\
&= U \Sigma V^\top V \Sigma^\top U^\top\\
&= U \Sigma \Sigma^\top U^\top
\end{aligned}
$$

$$
C = \frac{1}{n} U \Sigma \Sigma^\top U^\top
$$

We can then rearrange the columns in the matrices $U$ and $\Sigma$ so that the singular values are in decreasing order.


**Step 4: Select singular values, (optional) truncate the rest:**

We can now decide how many singular values to pick, based on how much variance you want to retain. (e.g., retaining 95% of the total variance). 

We can obtain the percentage by calculating the variance contained in the leading $r$ factors divided by the variance in total:

$$
\frac{\sum_{i = 1}^{r} \sigma^2_{i}}{\sum_{i = 1}^{p} \sigma^2_{i}}
$$

**Step 5: Create the Score Matrix:**

$$
\begin{aligned}
T&= BV \cr
&= U\Sigma V^\top V \cr
&= U\Sigma
\end{aligned}
$$


## Relationship of PCA to SVD

To relate an SVD to a PCA of data set $X$, first construct the SVD of the data matrix $X$:

Let’s assume that sample means of all variables are zero, so we don't need to standardize our matrix.

$$
X = U \Sigma V^\top  = \sigma_1 U_1 V_1^\top  + \sigma_2 U_2 V_2^\top  + \cdots + \sigma_p U_p V_p^\top
$$ (eq:PCA1)

where

$$
U=\begin{bmatrix}U_1|U_2|\ldots|U_m\end{bmatrix}
$$

$$
V^\top  = \begin{bmatrix}V_1^\top \\V_2^\top \\\ldots\\V_n^\top \end{bmatrix}
$$

In equation {eq}`eq:PCA1`, each of the $m \times n$ matrices $U_{j}V_{j}^\top $ is evidently
of rank $1$.

Thus, we have

$$
X = \sigma_1 \begin{pmatrix}U_{11}V_{1}^\top \\U_{21}V_{1}^\top \\\cdots\\U_{m1}V_{1}^\top \\\end{pmatrix} + \sigma_2\begin{pmatrix}U_{12}V_{2}^\top \\U_{22}V_{2}^\top \\\cdots\\U_{m2}V_{2}^\top \\\end{pmatrix}+\ldots + \sigma_p\begin{pmatrix}U_{1p}V_{p}^\top \\U_{2p}V_{p}^\top \\\cdots\\U_{mp}V_{p}^\top \\\end{pmatrix}
$$ (eq:PCA2)

Here is how we would interpret the objects in the  matrix equation {eq}`eq:PCA2` in
a time series context:

* $  \textrm{for each} \   k=1, \ldots, n $, the object $\lbrace V_{kj} \rbrace_{j=1}^n$ is a time series   for the $k$th **principal component**

* $U_j = \begin{bmatrix}U_{1k}\\U_{2k}\\\ldots\\U_{mk}\end{bmatrix} \  k=1, \ldots, m$
is a vector of **loadings** of variables $X_i$ on the $k$th principal component,  $i=1, \ldots, m$

* $\sigma_k $ for each $k=1, \ldots, p$ is the strength of $k$th **principal component**, where strength means contribution to the overall covariance of $X$.

## PCA with Eigenvalues and Eigenvectors

We now  use an eigen decomposition of a sample covariance matrix to do PCA.

Let $X_{m \times n}$ be our $m \times n$ data matrix.

Let's assume that sample means of all variables are zero.

We can assure  this  by **pre-processing** the data by subtracting sample means.

Define a sample covariance matrix $\Omega$ as

$$
\Omega = XX^\top
$$

Then use an eigen decomposition to represent $\Omega$ as follows:

$$
\Omega =P\Lambda P^\top
$$

Here

* $P$ is $m×m$ matrix of eigenvectors of $\Omega$

* $\Lambda$ is a diagonal matrix of eigenvalues of $\Omega$

We can then represent $X$ as

$$
X=P\epsilon
$$

where

$$
\epsilon = P^{-1} X
$$

and

$$
\epsilon\epsilon^\top =\Lambda .
$$

We can verify that

$$
XX^\top =P\Lambda P^\top  .
$$ (eq:XXo)

It follows that we can represent the data matrix $X$  as

\begin{equation*}
X=\begin{bmatrix}X_1|X_2|\ldots|X_m\end{bmatrix} =\begin{bmatrix}P_1|P_2|\ldots|P_m\end{bmatrix}
\begin{bmatrix}\epsilon_1\\\epsilon_2\\\ldots\\\epsilon_m\end{bmatrix}
= P_1\epsilon_1+P_2\epsilon_2+\ldots+P_m\epsilon_m
\end{equation*}


To reconcile the preceding representation with the PCA that we had obtained earlier through the SVD, we first note that $\epsilon_j^2=\lambda_j\equiv\sigma^2_j$.

Now define  $\tilde{\epsilon_j} = \frac{\epsilon_j}{\sqrt{\lambda_j}}$,
which  implies that $\tilde{\epsilon}_j\tilde{\epsilon}_j^\top =1$.

Therefore

$$
\begin{aligned}
X&=\sqrt{\lambda_1}P_1\tilde{\epsilon_1}+\sqrt{\lambda_2}P_2\tilde{\epsilon_2}+\ldots+\sqrt{\lambda_m}P_m\tilde{\epsilon_m}\\
&=\sigma_1P_1\tilde{\epsilon_2}+\sigma_2P_2\tilde{\epsilon_2}+\ldots+\sigma_mP_m\tilde{\epsilon_m} ,
\end{aligned}
$$

which  agrees with

$$
X=\sigma_1U_1{V_1}^{T}+\sigma_2 U_2{V_2}^{T}+\ldots+\sigma_{r} U_{r}{V_{r}}^{T}
$$

provided that  we set

* $U_j=P_j$ (a vector of  loadings of variables on principal component $j$)

* ${V_k}^{T}=\tilde{\epsilon_k}$ (the $k$th principal component)

Because  there are alternative algorithms for  computing  $P$ and $U$ for  given a data matrix $X$, depending on  algorithms used, we might have sign differences or different orders of eigenvectors.

We can resolve such ambiguities about  $U$ and $P$ by

1. sorting eigenvalues and singular values in descending order
2. imposing positive diagonals on $P$ and $U$ and adjusting signs in $V^\top $ accordingly

## Connections

To pull things together, it is useful to assemble and compare some formulas presented above.

First, consider an  SVD of an $m \times n$ matrix:

$$
X = U\Sigma V^\top
$$

Compute:

$$
\begin{aligned}
XX^\top &=U\Sigma V^\top V\Sigma^\top  U^\top \cr
&\equiv U\Sigma\Sigma^\top U^\top \cr
&\equiv U\Lambda U^\top
\end{aligned}
$$  (eq:XXcompare)

Compare representation {eq}`eq:XXcompare` with equation {eq}`eq:XXo` above.

Evidently, $U$ in the SVD is the matrix $P$  of
eigenvectors of $XX^\top $ and $\Sigma \Sigma^\top $ is the matrix $\Lambda$ of eigenvalues.

Second, let's compute

$$
\begin{aligned}
X^\top X &=V\Sigma^\top  U^\top U\Sigma V^\top \\
&=V\Sigma^\top {\Sigma}V^\top
\end{aligned}
$$



Thus, the matrix $V$ in the SVD is the matrix of eigenvectors of $X^\top X$

Summarizing and fitting things together, we have the eigen decomposition of the sample
covariance matrix

$$
X X^\top  = P \Lambda P^\top
$$

where $P$ is an orthogonal matrix.

Further, from the SVD of $X$, we know that

$$
X X^\top  = U \Sigma \Sigma^\top  U^\top
$$

where $U$ is an orthogonal matrix.

Thus, $P = U$ and we have the representation of $X$

$$
X = P \epsilon = U \Sigma V^\top
$$

It follows that

$$
U^\top  X = \Sigma V^\top  = \epsilon
$$

Note that the preceding implies that

$$
\epsilon \epsilon^\top  = \Sigma V^\top  V \Sigma^\top  = \Sigma \Sigma^\top  = \Lambda ,
$$

so that everything fits together.

Below we define a class `DecomAnalysis` that wraps  PCA and SVD for a given a data matrix `X`.

```{code-cell} ipython3
class DecomAnalysis:
    """
    A class for conducting PCA and SVD.
    X: data matrix
    r_component: chosen rank for best approximation
    """

    def __init__(self, X, r_component=None):

        self.X = X

        self.Ω = (X @ X.T)

        self.m, self.n = X.shape
        self.r = LA.matrix_rank(X)

        if r_component:
            self.r_component = r_component
        else:
            self.r_component = self.m

    def pca(self):

        𝜆, P = LA.eigh(self.Ω)    # columns of P are eigenvectors

        ind = sorted(range(𝜆.size), key=lambda x: 𝜆[x], reverse=True)

        # sort by eigenvalues
        self.𝜆 = 𝜆[ind]
        P = P[:, ind]
        self.P = P @ diag_sign(P)

        self.Λ = np.diag(self.𝜆)

        self.explained_ratio_pca = np.cumsum(self.𝜆) / self.𝜆.sum()

        # compute the N by T matrix of principal components
        self.𝜖 = self.P.T @ self.X

        P = self.P[:, :self.r_component]
        𝜖 = self.𝜖[:self.r_component, :]

        # transform data
        self.X_pca = P @ 𝜖

    def svd(self):

        U, 𝜎, VT = LA.svd(self.X)

        ind = sorted(range(𝜎.size), key=lambda x: 𝜎[x], reverse=True)

        # sort by eigenvalues
        d = min(self.m, self.n)

        self.𝜎 = 𝜎[ind]
        U = U[:, ind]
        D = diag_sign(U)
        self.U = U @ D
        VT[:d, :] = D @ VT[ind, :]
        self.VT = VT

        self.Σ = np.zeros((self.m, self.n))
        self.Σ[:d, :d] = np.diag(self.𝜎)

        𝜎_sq = self.𝜎 ** 2
        self.explained_ratio_svd = np.cumsum(𝜎_sq) / 𝜎_sq.sum()

        # slicing matrices by the number of components to use
        U = self.U[:, :self.r_component]
        Σ = self.Σ[:self.r_component, :self.r_component]
        VT = self.VT[:self.r_component, :]

        # transform data
        self.X_svd = U @ Σ @ VT

    def fit(self, r_component):

        # pca
        P = self.P[:, :r_component]
        𝜖 = self.𝜖[:r_component, :]

        # transform data
        self.X_pca = P @ 𝜖

        # svd
        U = self.U[:, :r_component]
        Σ = self.Σ[:r_component, :r_component]
        VT = self.VT[:r_component, :]

        # transform data
        self.X_svd = U @ Σ @ VT

def diag_sign(A):
    "Compute the signs of the diagonal of matrix A"

    D = np.diag(np.sign(np.diag(A)))

    return D
```

We also define a function that prints out information so that we can compare  decompositions
obtained by different algorithms.

```{code-cell} ipython3
def compare_pca_svd(da):
    """
    Compare the outcomes of PCA and SVD.
    """

    da.pca()
    da.svd()

    print('Eigenvalues and Singular values\n')
    print(f'λ = {da.λ}\n')
    print(f'σ^2 = {da.σ**2}\n')
    print('\n')

    # loading matrices
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    plt.suptitle('loadings')
    axs[0].plot(da.P.T)
    axs[0].set_title('P')
    axs[0].set_xlabel('m')
    axs[1].plot(da.U.T)
    axs[1].set_title('U')
    axs[1].set_xlabel('m')
    plt.show()

    # principal components
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    plt.suptitle('principal components')
    axs[0].plot(da.ε.T)
    axs[0].set_title('ε')
    axs[0].set_xlabel('n')
    axs[1].plot(da.VT[:da.r, :].T * np.sqrt(da.λ))
    axs[1].set_title('$V^\top *\sqrt{\lambda}$')
    axs[1].set_xlabel('n')
    plt.show()
```

## Exercises

```{exercise}
:label: svd_ex1

In Ordinary Least Squares (OLS), we learn to compute $ \hat{\beta} = (X^\top X)^{-1} X^\top y $, but there are cases such as when we have colinearity or an underdetermined system: **short fat** matrix.

In these cases, the $ (X^\top X) $ matrix is not not invertible (its determinant is zero) or ill-conditioned (its determinant is very close to zero).

What we can do instead is to create what is called a [pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse), a full rank approximation of the inverted matrix so we can compute $ \hat{\beta} $ with it.

Thinking in terms of the Eckart-Young theorem, build the pseudoinverse matrix $ X^{+} $ and use it to compute $ \hat{\beta} $.

```

```{solution-start} svd_ex1
:class: dropdown
```

We can use SVD to compute the pseudoinverse:

$$
X  = U \Sigma V^\top
$$

inverting $X$, we have:

$$
X^{+}  = V \Sigma^{+} U^\top
$$

where:

$$
\Sigma^{+} = \begin{bmatrix}
\frac{1}{\sigma_1} & 0 & \cdots & 0 & 0 \\
0 & \frac{1}{\sigma_2} & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & \frac{1}{\sigma_p} & 0 \\
0 & 0 & \cdots & 0 & 0 \\
\end{bmatrix}
$$

and finally:

$$
\hat{\beta} = X^{+}y = V \Sigma^{+} U^\top y 
$$

```{solution-end}
```


For an example  PCA applied to analyzing the structure of intelligence tests see this lecture {doc}`Multivariable Normal Distribution <multivariate_normal>`.

Look at  parts of that lecture that describe and illustrate the classic factor analysis model.

As mentioned earlier, in a sequel to this lecture about  {doc}`Dynamic Mode Decompositions <var_dmd>`, we'll describe how SVD's provide ways rapidly to compute reduced-order approximations to first-order Vector Autoregressions (VARs).
