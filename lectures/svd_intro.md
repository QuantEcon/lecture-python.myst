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

In addition to regular packages contained in Anaconda by default, this lecture also requires:

```{code-cell} ipython3
:tags: [hide-output]
!pip install quandl
```

```{code-cell} ipython3
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
%matplotlib inline
import quandl as ql
import pandas as pd
```

## Overview

The **singular value decomposition** is a work-horse in applications of least squares projection that
form a foundation for  some important machine learning methods.

This lecture describes the singular value decomposition and two of its uses:

 * principal components analysis (PCA)

 * dynamic mode decomposition (DMD)

 Each of these can be thought of as a data-reduction procedure  designed to capture salient patterns by projecting data onto a limited set of factors.

##  The Setup

Let $X$ be an $m \times n$ matrix of rank $r$.

Necessarily, $r \leq \min(m,n)$.

In this lecture, we'll think of $X$ as a matrix of **data**.

  * each column is an **individual** -- a time period or person, depending on the application
  
  * each row is a **random variable** measuring an attribute of a time period or a person, depending on the application
  
  
We'll be interested in  two  cases

  * A **short and fat** case in which $m << n$, so that there are many more columns than rows.

  * A  **tall and skinny** case in which $m >> n$, so that there are many more rows than columns. 
    
   
We'll apply a **singular value decomposition** of $X$ in both situations.

In the first case in which there are many more observations $n$ than random variables $m$, we learn about a joint distribution  by taking averages  across observations of functions of the observations. 

Here we'll look for **patterns** by using a **singular value decomposition** to do a **principal components analysis** (PCA).

In the second case in which there are many more random variables $m$ than observations $n$, we'll proceed in a different way. 

We'll again use a **singular value decomposition**,  but now to do a **dynamic mode decomposition** (DMD)

## Singular Value Decomposition

A **singular value decomposition** of an $m \times n$ matrix $X$ of rank $r \leq \min(m,n)$ is

$$
X  = U \Sigma V^T
$$

where 

\begin{align*}
UU^T &  = I  &  \quad U^T U = I \cr    
VV^T & = I & \quad V^T V = I
\end{align*}
 
where 
 
* $U$ is an $m \times m$ matrix whose columns are eigenvectors of $X^T X$

* $V$ is an $n \times n$ matrix whose columns are eigenvectors of $X X^T$

* $\Sigma$ is an $m \times n$ matrix in which the first $r$ places on its main diagonal are positive numbers $\sigma_1, \sigma_2, \ldots, \sigma_r$ called **singular values**; remaining entries of $\Sigma$ are all zero

* The $r$ singular values are square roots of the eigenvalues of the $m \times m$ matrix  $X X^T$ and the $n \times n$ matrix $X^T X$

* When $U$ is a complex valued matrix, $U^T$ denotes the **conjugate-transpose** or **Hermitian-transpose** of $U$, meaning that 
$U_{ij}^T$ is the complex conjugate of $U_{ji}$. 

* Similarly, when $V$ is a complex valued matrix, $V^T$ denotes the **conjugate-transpose** or **Hermitian-transpose** of $V$

In what is called a **full** SVD, the  shapes of $U$, $\Sigma$, and $V$ are $\left(m, m\right)$, $\left(m, n\right)$, $\left(n, n\right)$, respectively. 



There is also an alternative shape convention called an **economy** or **reduced** SVD .

Thus, note that because we assume that $A$ has rank $r$, there are only $r $ nonzero singular values, where $r=\textrm{rank}(A)\leq\min\left(m, n\right)$.  

A **reduced** SVD uses this fact to express $U$, $\Sigma$, and $V$ as matrices with shapes $\left(m, r\right)$, $\left(r, r\right)$, $\left(r, n\right)$.

Sometimes, we will use a full SVD 

At other times, we'll use a reduced SVD  in which $\Sigma$ is an $r \times r$  diagonal matrix.

## Digression:  Polar Decomposition

A singular value decomposition (SVD) is related to the **polar decomposition** of $X$

$$
X  = SQ   
$$

where

\begin{align*}
 S & = U\Sigma U^T \cr
Q & = U V^T 
\end{align*}

and $S$ is evidently a symmetric matrix and $Q$ is an orthogonal matrix.

## Principle Components Analysis (PCA)

Let's begin with a case in which $n >> m$, so that we have many  more observations $n$ than random variables $m$.

The  matrix $X$ is **short and fat**  in an  $n >> m$ case as opposed to a **tall and skinny** case with $m > > n $ to be discussed later.

We regard  $X$ as an  $m \times n$ matrix of **data**:

$$
X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_n\end{bmatrix}
$$

where for $j = 1, \ldots, n$ the column vector $X_j = \begin{bmatrix}X_{1j}\\X_{2j}\\\vdots\\X_{mj}\end{bmatrix}$ is a  vector of observations on variables $\begin{bmatrix}x_1\\x_2\\\vdots\\x_m\end{bmatrix}$.

In a **time series** setting, we would think of columns $j$ as indexing different __times__ at which random variables are observed, while rows index different random variables.

In a **cross section** setting, we would think of columns $j$ as indexing different __individuals__ for  which random variables are observed, while rows index different **random variables**.

The number of positive singular values equals the rank of  matrix $X$.

Arrange the singular values  in decreasing order.

Arrange   the positive singular values on the main diagonal of the matrix $\Sigma$ of into a vector $\sigma_R$.

Set all other entries of $\Sigma$ to zero.

## Relationship of PCA to SVD

To relate a SVD to a PCA (principal component analysis) of data set $X$, first construct  the  SVD of the data matrix $X$:

$$
X = U \Sigma V^T = \sigma_1 U_1 V_1^T + \sigma_2 U_2 V_2^T + \cdots + \sigma_r U_r V_r^T
$$ (eq:PCA1)

where

$$
U=\begin{bmatrix}U_1|U_2|\ldots|U_m\end{bmatrix}
$$

$$
V^T = \begin{bmatrix}V_1^T\\V_2^T\\\ldots\\V_n^T\end{bmatrix}
$$

In equation {eq}`eq:PCA1`, each of the $m \times n$ matrices $U_{j}V_{j}^T$ is evidently
of rank $1$. 

Thus, we have 

$$
X = \sigma_1 \begin{pmatrix}U_{11}V_{1}^T\\U_{21}V_{1}^T\\\cdots\\U_{m1}V_{1}^T\\\end{pmatrix} + \sigma_2\begin{pmatrix}U_{12}V_{2}^T\\U_{22}V_{2}^T\\\cdots\\U_{m2}V_{2}^T\\\end{pmatrix}+\ldots + \sigma_r\begin{pmatrix}U_{1r}V_{r}^T\\U_{2r}V_{r}^T\\\cdots\\U_{mr}V_{r}^T\\\end{pmatrix}
$$ (eq:PCA2)

Here is how we would interpret the objects in the  matrix equation {eq}`eq:PCA2` in 
a time series context:

* $ V_{k}^T= \begin{bmatrix}V_{k1} &  V_{k2} & \ldots & V_{kn}\end{bmatrix}  \quad \textrm{for each} \   k=1, \ldots, n $ is a time series  $\lbrace V_{kj} \rbrace_{j=1}^n$ for the $k$th **principal component**

* $U_j = \begin{bmatrix}U_{1k}\\U_{2k}\\\ldots\\U_{mk}\end{bmatrix} \  k=1, \ldots, m$
is a vector of **loadings** of variables $X_i$ on the $k$th principle component,  $i=1, \ldots, m$

* $\sigma_k $ for each $k=1, \ldots, r$ is the strength of $k$th **principal component**

## Reduced Versus Full SVD

Earlier, we mentioned **full** and **reduced** SVD's.


You can read about reduced and full SVD here
<https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html>

In a **full** SVD

  * $U$ is $m \times m$
  * $\Sigma$ is $m \times n$
  * $V$ is $n \times n$

In a **reduced** SVD

  * $U$ is $m \times r$
  * $\Sigma$ is $r \times r$
  * $V$ is $n \times r$ 

 
Let's do a some  small exerecise  to compare **full** and **reduced** SVD's.

First, let's study a case in which $m = 5 > n = 2$.

```{code-cell} ipython3
import numpy as np
X = np.random.rand(5,2)
U, S, V = np.linalg.svd(X,full_matrices=True)  # full SVD
Uhat, Shat, Vhat = np.linalg.svd(X,full_matrices=False) # economy SVD
print('U, S, V ='), U, S, V
```

```{code-cell} ipython3
print('Uhat, Shat, Vhat = '), Uhat, Shat, Vhat
```

```{code-cell} ipython3
rr = np.linalg.matrix_rank(X)
rr
```

**Remark:** The cells above illustrate application of the  `fullmatrices=True` and `full-matrices=False` options.
Using `full-matrices=False` returns a reduced singular value decomposition. This option implements
an optimal reduced rank approximation of a matrix, in the sense of  minimizing the Frobenius
norm of the discrepancy between the approximating matrix and the matrix being approximated.
Optimality in this sense is  established in the celebrated Eckart–Young theorem. See <https://en.wikipedia.org/wiki/Low-rank_approximation>.

Let's do another exercise, but now we'll set $m = 2 < 5 = n $

```{code-cell} ipython3
import numpy as np
X = np.random.rand(2,5)
U, S, V = np.linalg.svd(X,full_matrices=True)  # full SVD
Uhat, Shat, Vhat = np.linalg.svd(X,full_matrices=False) # economy SVD
print('U, S, V ='), U, S, V
```

```{code-cell} ipython3
print('Uhat, Shat, Vhat = '), Uhat, Shat, Vhat
```

```{code-cell} ipython3
rr = np.linalg.matrix_rank(X)
rr
```

## PCA with Eigenvalues and Eigenvectors

We now  use an eigen decomposition of a sample covariance matrix to do PCA.

Let $X_{m \times n}$ be our $m \times n$ data matrix.

Let's assume that sample means of all variables are zero.

We can assure  this  by **pre-processing** the data by subtracting sample means.

Define the sample covariance matrix $\Omega$ as 

$$
\Omega = XX^T
$$

Then use an eigen decomposition to represent $\Omega$ as follows:

$$
\Omega =P\Lambda P^T
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
\epsilon\epsilon^T=\Lambda .
$$ 

We can verify that

$$
XX^T=P\Lambda P^T .
$$

It follows that we can represent the data matrix as 

\begin{equation*}
X=\begin{bmatrix}X_1|X_2|\ldots|X_m\end{bmatrix} =\begin{bmatrix}P_1|P_2|\ldots|P_m\end{bmatrix}
\begin{bmatrix}\epsilon_1\\\epsilon_2\\\ldots\\\epsilon_m\end{bmatrix} 
= P_1\epsilon_1+P_2\epsilon_2+\ldots+P_m\epsilon_m
\end{equation*}

where 

$$
\epsilon\epsilon^T=\Lambda .
$$

To reconcile the preceding representation with the PCA that we obtained through the SVD above, we first note that $\epsilon_j^2=\lambda_j\equiv\sigma^2_j$.

Now define  $\tilde{\epsilon_j} = \frac{\epsilon_j}{\sqrt{\lambda_j}}$, 
which evidently implies that $\tilde{\epsilon}_j\tilde{\epsilon}_j^T=1$.

Therefore 

$$
\begin{aligned}
X&=\sqrt{\lambda_1}P_1\tilde{\epsilon_1}+\sqrt{\lambda_2}P_2\tilde{\epsilon_2}+\ldots+\sqrt{\lambda_m}P_m\tilde{\epsilon_m}\\
&=\sigma_1P_1\tilde{\epsilon_2}+\sigma_2P_2\tilde{\epsilon_2}+\ldots+\sigma_mP_m\tilde{\epsilon_m} ,
\end{aligned}
$$

which evidently agrees with 

$$
X=\sigma_1U_1{V_1}^{T}+\sigma_2 U_2{V_2}^{T}+\ldots+\sigma_{r} U_{r}{V_{r}}^{T}
$$

provided that  we set 

* $U_j=P_j$ (the loadings of variables on principal components) 

* ${V_k}^{T}=\tilde{\epsilon_k}$ (the principal components)

Since there are several possible ways of computing  $P$ and $U$ for  given a data matrix $X$, depending on  algorithms used, we might have sign differences or different orders between eigenvectors.

We can resolve such ambiguities about  $U$ and $P$ by

1. sorting eigenvalues and singular values in descending order
2. imposing positive diagonals on $P$ and $U$ and adjusting signs in $V^T$ accordingly

## Connections

To pull things together, it is useful to assemble and compare some formulas presented above.

First, consider the following SVD of an $m \times n$ matrix:

$$
X = U\Sigma V^T
$$

Compute:

\begin{align*}
XX^T&=U\Sigma V^TV\Sigma^T U^T\cr
&\equiv U\Sigma\Sigma^TU^T\cr
&\equiv U\Lambda U^T
\end{align*}
  
Thus, $U$ in the SVD is the matrix $P$  of
eigenvectors of $XX^T$ and $\Sigma \Sigma^T$ is the matrix $\Lambda$ of eigenvalues.

Second, let's compute

\begin{align*}
X^TX &=V\Sigma^T U^TU\Sigma V^T\\
&=V\Sigma^T{\Sigma}V^T
\end{align*} 

Thus, the matrix $V$ in the SVD is the matrix of eigenvectors of $X^TX$

Summarizing and fitting things together, we have the eigen decomposition of the sample
covariance matrix

$$
X X^T = P \Lambda P^T
$$

where $P$ is an orthogonal matrix.

Further, from the SVD of $X$, we know that

$$
X X^T = U \Sigma \Sigma^T U^T
$$

where $U$ is an orthonal matrix.  

Thus, $P = U$ and we have the representation of $X$

$$
X = P \epsilon = U \Sigma V^T
$$

It follows that 

$$
U^T X = \Sigma V^T = \epsilon
$$

Note that the preceding implies that

$$
\epsilon \epsilon^T = \Sigma V^T V \Sigma^T = \Sigma \Sigma^T = \Lambda ,
$$

so that everything fits together.

Below we define a class `DecomAnalysis` that wraps  PCA and SVD for a given a data matrix `X`.

```{code-cell} ipython3
class DecomAnalysis:
    """
    A class for conducting PCA and SVD.
    """

    def __init__(self, X, n_component=None):

        self.X = X

        self.Ω = (X @ X.T)

        self.m, self.n = X.shape
        self.r = LA.matrix_rank(X)

        if n_component:
            self.n_component = n_component
        else:
            self.n_component = self.m

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

        P = self.P[:, :self.n_component]
        𝜖 = self.𝜖[:self.n_component, :]

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
        U = self.U[:, :self.n_component]
        Σ = self.Σ[:self.n_component, :self.n_component]
        VT = self.VT[:self.n_component, :]

        # transform data
        self.X_svd = U @ Σ @ VT

    def fit(self, n_component):

        # pca
        P = self.P[:, :n_component]
        𝜖 = self.𝜖[:n_component, :]

        # transform data
        self.X_pca = P @ 𝜖

        # svd
        U = self.U[:, :n_component]
        Σ = self.Σ[:n_component, :n_component]
        VT = self.VT[:n_component, :]

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
    axs[1].set_title('$V^T*\sqrt{\lambda}$')
    axs[1].set_xlabel('n')
    plt.show()
```


For an example  PCA applied to analyzing the structure of intelligence tests see this lecture {doc}`Multivariable Normal Distribution <multivariate_normal>`.

Look at the parts of that lecture that describe and illustrate the classic factor analysis model.

## Dynamic Mode Decomposition (DMD)

We turn to the case in which $m >>n$ in which an $m \times n$  data matrix $\tilde X$ contains many more random variables $m$ than observations $n$.

This  **tall and skinny** case is associated with **Dynamic Mode Decomposition**.

You can read about Dynamic Mode Decomposition here {cite}`DMD_book` and here {cite}`DDSE_book` (section 7.2).

We start  with an $m \times n $ matrix of data $\tilde X$ of the form 


$$
 \tilde X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_n\end{bmatrix}
$$ 

where for $t = 1, \ldots, n$,  the $m \times 1 $ vector $X_t$ is

$$ X_t = \begin{bmatrix}  X_{1,t} & X_{2,t} & \cdots & X_{m,t}     \end{bmatrix}^T $$

where $T$ again denotes complex transposition and $X_{i,t}$ is an observation on variable $i$ at time $t$.

From $\tilde X$,   form two matrices 

$$
 X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_{n-1}\end{bmatrix}
$$ 

and

$$
X' =  \begin{bmatrix} X_2 \mid X_3 \mid \cdots \mid X_n\end{bmatrix}
$$

Here $'$ does not denote matrix transposition but instead is part of the name of the matrix $X'$.

In forming $ X$ and $X'$, we have in each case  dropped a column from $\tilde X$,  the last column in the case of $X$, and  the first column in the case of $X'$.

Evidently, $ X$ and $ X'$ are both $m \times \tilde n$ matrices where $\tilde n = n - 1$.

We denote the rank of $X$ as $p \leq \min(m, \tilde n) = \tilde n$.

We start with a system consisting of $m$ least squares regressions of **everything** on one lagged value of **everything**:

$$
 X' = A  X + \epsilon
$$ 

where $\epsilon$ is an $m \times m$ matrix of least squares residuals satisfying

$$
\epsilon X^+ = 0
$$

and 

$$
A =  X'  X^{+} .
$$ (eq:Afullformula)

Here the (possibly huge) $m \times m $ matrix $X^{+}$ is the Moore-Penrose generalized inverse of $X$.

The $i$th the row of $A$ is an $m \times 1$ vector of regression coefficients of $X_{i,t+1}$ on $X_{j,t}, j = 1, \ldots, m$.


Consider the (reduced) singular value decomposition 

  $$ 
  X =  U \Sigma  V^T
  $$ (eq:SVDforDMD)


  
where $U$ is $m \times p$, $\Sigma$ is a $p \times p$ diagonal  matrix, and $ V^T$ is a $p \times m$ matrix.

Here $p$ is the rank of $X$, where necessarily $p \leq \tilde n$. 

(We  described and illustrated a **reduced** singular value decomposition above, and compared it with a **full** singular value decomposition.)  

We could construct the generalized inverse $X^+$  of $X$ by using
a singular value decomposition  $X = U \Sigma V^T$ to compute

$$
X^{+} =  V \Sigma^{-1}  U^T
$$ (eq:Xpinverse)

where the matrix $\Sigma^{-1}$ is constructed by replacing each non-zero element of $ \Sigma$ with $\sigma_j^{-1}$.

We could use formula {eq}`eq:Xpinverse`   together with formula {eq}`eq:Afullformula` to compute the matrix  $A$ of regression coefficients.

Instead of doing that, we'll eventually use **dynamic mode decomposition** to compute a rank $r$ approximation to $A$,
where $r <  p$.  


The idea behind **dynamic mode decomposition** is to construct this low rank  approximation to $A$ that  


* constructs an $m \times r$ matrix $\Phi$ that captures effects  on all $m$ variables of $r \leq p$  **modes** that are associated with the $r$ largest eigenvalues of $A$

   
* uses $\Phi$, the current value of $X_t$, and  powers of the $r$ largest eigenvalues of $A$ to forecast *future* $X_{t+j}$'s




## Preliminary Analysis

We'll put basic ideas on the table by starting with the special case in which $r = p$ so that we retain
all $p$ singular values of $X$.

(Later, we'll retain only $r < p$ of them)

When $r = p$,  formula
{eq}`eq:Xpinverse` implies that 


$$
A = X' V \Sigma^{-1}  U^T
$$ (eq:Aformbig)

where $V$ is an $\tilde n \times p$ matrix, $\Sigma^{-1}$ is a $p \times p$ matrix,  $U$ is a $p \times m$ matrix,
and  $U^T  U = I_p$ and $V V^T = I_m $.

We use the $p$  columns of $U$, and thus the $p$ rows of $U^T$,  to define   a $p \times 1$  vector $\tilde X_t$ as follows


$$
\tilde X_t = U^T X_t .
$$ (eq:tildeXdef2)

Since $U U^T$ is an $m \times m$ identity matrix, it follows from equation {eq}`eq:tildeXdef2` that we can recover $X_t$ from $\tilde X_t$ by using 

$$
X_t = U \tilde X_t .
$$ (eq:Xdecoder)


 * Equation {eq}`eq:tildeXdef2` serves as an **encoder** that  summarizes the $m \times 1$ vector $X_t$ by a $p \times 1$ vector $\tilde X_t$ 
  
 * Equation {eq}`eq:Xdecoder` serves as a **decoder** that recovers the $m \times 1$ vector $X_t$ from the $p \times 1$ vector $\tilde X_t$ 

The following  $p \times p$ transition matrix governs the motion of $\tilde X_t$:

$$
 \tilde A = U^T A U = U^T X' V \Sigma^{-1} .
$$ (eq:Atilde0)

Evidently, 

$$
\tilde X_{t+1} = \tilde A \tilde X_t 
$$ (eq:xtildemotion)

Notice that if we multiply both sides of {eq}`eq:xtildemotion` by $U$ 
we get

$$
U \tilde X_t = U \tilde A \tilde X_t =  U \tilde A U^T X_t 
$$

which by virtue of decoder equation {eq}`eq:xtildemotion` recovers

$$
X_{t+1} = A X_t .
$$





### Lower Rank Approximations


Instead of using all $p$ modes $\tilde X_t$  calculated according to formula {eq}`eq:tildeXdef2`, we can use just the $r<p$ largest of them. 

These are the ones that are most important in shaping
the dynamics of $X$.   

We can accomplish this by   computing the $r$ largest singular values of $X$ and  forming  matrices $\tilde V, \tilde U$ corresponding to those $r$ singular values. 
  
We can  then construct  a reduced-order system of dimension $r$ by forming an  $r \times r$ transition matrix
$\tilde A$ redefined by  

$$
 \tilde A = \tilde U^T A \tilde U 
$$ (eq:tildeA_1)

Here we now use $\tilde U$ rather than $U$ as we did earlier in equation {eq}`eq:Atilde0`.

This redefined  $\tilde A$ matrix governs the dynamics of a redefined  $r \times 1$ vector $\tilde X_t $
according to

$$ 
    \tilde X_{t+1} = \tilde A \tilde X_t
$$

where now 

$$
\tilde X_t = \tilde U^T X_t 
$$

and 

$$ 
X_t = \tilde U \tilde X_t.
$$

From equation {eq}`eq:tildeA_1` and {eq}`eq:Aformbig` it follows that


$$
  \tilde A = \tilde U^T X' \tilde V \Sigma^{-1}
$$ (eq:tildeAform)

  
Next, we'll construct an eigencomposition of $\tilde A$:  

$$ 
  \tilde A W =  W \Lambda
$$ (eq:tildeAeigen)
  
where $\Lambda$ is a $r \times r$ diagonal matrix of eigenvalues and the columns of $W$ are corresponding eigenvectors
of $\tilde A$.   

Both $\Lambda$ and $W$ are $r \times r$ matrices.
  
Construct the $m \times r$ matrix

$$
  \Phi = X' \tilde  V  \tilde \Sigma^{-1} W
$$ (eq:Phiformula)


  
The following very useful proposition was established by Tu et al. {cite}`tu_Rowley`. 

**Proposition** The $r$ columns of $\Phi$ are eigenvectors of $A$ that correspond to the largest $r$ eigenvalues of $A$. 

**Proof:** From formula {eq}`eq:Phiformula` we have

$$  
\begin{aligned}
  A \Phi & =  (X' \tilde V \tilde \Sigma^{-1} \tilde U^T) (X' \tilde V \tilde \Sigma^{-1} W) \cr
  & = X' \tilde V \Sigma^{-1} \tilde A W \cr
  & = X' \tilde V \tilde \Sigma^{-1} W \Lambda \cr
  & = \Phi \Lambda 
  \end{aligned}
$$ 

Thus, we can conclude that

$$  
A \Phi = \Phi \Lambda
$$ (eq:APhiLambda)

Let $\phi_i$ be the the $i$the column of $\Phi$ and $\lambda_i$ be the corresponding $i$ eigenvalue of $\tilde A$ from decomposition {eq}`eq:tildeAeigen`. 

Writing out the $m \times r$ vectors on both sides of  equation {eq}`eq:APhiLambda` and equating them gives


$$
A \phi_i = \lambda_i \phi_i .
$$

Thus, $\phi_i$ is an eigenvector of $A$ that corresponds to eigenvalue  $\lambda_i$ of $A$.

This concludes the proof. 


Also see {cite}`DDSE_book` (p. 238)







## Some Refinements

The following argument from {cite}`DDSE_book` (page 240) provides a computationally efficient way
to compute projections of the time $t$ data onto  $r$ dominant **modes** at time $t$.  

For convenience, we'll do this first for time $t=1$.



Define  a projection  of $X_1$ onto  $r$ dominant **modes**  $b$ at time $1$  by

$$ 
   X_1 = \Phi b 
$$ (eq:X1proj)

where $b$ is an $r \times 1$ vector. 

Since $X_1 = \tilde U \tilde X_1$, it follows that 
 
$$ 
  \tilde U \tilde X_1 = X' \tilde V \tilde \Sigma^{-1} W b
$$

and

$$ 
  \tilde X_1 = \tilde U^T X' \tilde V \tilde \Sigma^{-1} W b
$$

Recall from formula {eq}`eq:tildeAform` that $ \tilde A = \tilde U^T X' \tilde V \tilde \Sigma^{-1}$ so that
  
$$ 
  \tilde X_1 = \tilde A W b
$$

and therefore, by the eigendecomposition  {eq}`eq:tildeAeigen` of $\tilde A$, we have

$$ 
  \tilde X_1 = W \Lambda b
$$ 

Therefore, 
  
$$ 
  b = ( W \Lambda)^{-1} \tilde X_1
$$ 

or 


$$ 
  b = ( W \Lambda)^{-1} \tilde U^T X_1
$$ (eq:beqnsmall)



which is  computationally more efficient than the following alternative equation for computing the initial vector $b$ of $r$ dominant
modes:

$$
  b= \Phi^{+} X_1
$$ (eq:bphieqn)


Conditional on $X_t$, we can construct forecasts $\check X_{t+j} $ of $X_{t+j}, j = 1, 2, \ldots, $  from 
either 

$$
\check X_{t+j} = \Phi \Lambda^j \Phi^{+} X_t
$$ (eq:checkXevoln)


or  the following equation 

$$ 
  \check X_{t+j} = \Phi \Lambda^j (W \Lambda)^{-1}  \tilde U^T X_t
$$ (eq:checkXevoln2)



### Putting Things Together
    
With $\Lambda, \Phi, \Phi^{+}$ in hand, our least-squares fitted dynamics fitted to the $r$  modes
are governed by

$$
X_{t+1}^{(r)} = \Phi \Lambda \Phi^{+} X_t^{(r)} 
$$ (eq:Xdynamicsapprox)

where $X_t^{(r)}$ is an $m \times 1$ vector.

By virtue of equation {eq}`eq:APhiLambda`, it follows that **if we had kept $r = p$**,  this equation would be equivalent with

$$
X_{t+1} = A X_t .
$$ (eq:Xdynamicstrue)

When $r < p $, equation {eq}`eq:Xdynamicsapprox` is an approximation (of reduced  order $r$) to the $X$ dynamics in equation
{eq}`eq:Xdynamicstrue`.

 
Conditional on $X_t$, we construct forecasts $\check X_{t+j} $ of $X_{t+j}, j = 1, 2, \ldots, $  from {eq}`eq:checkXevoln`.




## Reduced-order VAR

DMD  is a natural tool for estimating a **reduced order vector autoregression**,
an object that we define in terms of the population regression equation

$$
X_{t+1} = \check A X_t + C \epsilon_{t+1}
$$ (eq:VARred)

where 

* $X_t$ is an $m \times 1$ vector
* $\check A$ is an $m \times m$ matrix of rank $r$ whose eigenvalues are all less than $1$ in modulus
* $\epsilon_{t+1} \sim {\mathcal N}(0, I)$ is an $m \times 1$ vector of i.i.d. shocks
* $E \epsilon_{t+1} X_t^T = 0$, so that all shocks are orthogonal to all regressors

To link this model to a dynamic mode decomposition (DMD), again take

$$ 
X = [ X_1 \mid X_2 \mid \cdots \mid X_{n-1} ]
$$

$$
X' =  [ X_2 \mid X_3 \mid \cdots \mid X_n ]
$$

so that according to  model {eq}`eq:VARred` 


$$
X' = \begin{bmatrix} \check A X_1 + C \epsilon_2  \mid \check A X_2 + C \epsilon_3 \mid \cdots \mid \check A X_{n-1} +  C 
\epsilon_n \end{bmatrix}
$$

To illustrate some useful calculations, assume that $n =3 $ and form

$$
X' X^T = \begin{bmatrix} \check A X_1 + C \epsilon_2  &  \check A X_2 + C \epsilon_3 \end{bmatrix} 
   \begin{bmatrix} X_1^T \cr X_2^T \end{bmatrix} 
$$

or 

$$
X' X^T = \check A ( X_1 X_1^T + X_2 X_2^T) + C( \epsilon_2 X_1^T + \epsilon_3 X_2^T) 
$$

but because 

$$
E ( \epsilon_2 X_1^T + \epsilon_3 X_2^T)  = 0 
$$

we have

$$
X' X^T = \check A ( X_1 X_1^T + X_2 X_2^T)
$$

Evidently,

$$
X X^T = ( X_1 X_1^T + X_2 X_2^T)
$$

so that our  matrix  $\check A$ of least squares regression coefficients is

$$
\check A = (X' X^T)  (X X^T)^+
$$

Our **assumption** that $\check A$ is a matrix of rank $r$ leads us to represent it as

$$
\check A = \Phi \Lambda \Phi^{+}
$$

where $\Phi$ and $\Lambda$ are computed with the DMD algorithm described above.

Associated with the VAR representation {eq}`eq:VARred`
is the usual moving average representation

$$
X_{t+j} = \check A^j X_t + C \epsilon_{t+j} + \check A C \epsilon_{t+j-1} + \cdots \check A^{j-1} \epsilon_{t+1}
$$

After computing $\check A$, we can construct sample versions
of

$$ 
C \epsilon_{t+1} = X_{t+1} - \check A X_t , \quad t =1, \ldots, n-1
$$

and check whether they are serially uncorrelated as assumed in {eq}`eq:VARred`.

For example, we can compute spectra and cross-spectra of components of $C \epsilon_{t+1}$
and check for serial-uncorrelatedness in the usual ways.

We can also estimate the covariance matrix of $C \epsilon_{t+1}$
from

$$
\frac{1}{n-1} \sum_{t=1}^{n-1} (C \epsilon_{t+1} )( C \epsilon_{t+1})^T 
$$

It can be enlightening to diagonize  our reduced order VAR {eq}`eq:VARred` by noting that it can 
be written
 

$$
X_{t+1} = \Phi \Lambda \Phi^{+} X_t + C \epsilon_{t+1}
$$


and then writing it as 

$$
\Phi^+ X_{t+1} = \Lambda  \Phi^{+} X_t +  \Phi^+ C \epsilon_{t+1}
$$

or

$$
\bar X_{t+1} = \Lambda \bar X_t + \bar \epsilon_{t+1} 
$$ (eq:VARmodes)

where $\bar X_t $ is an $r \times 1$ **mode** and $\bar \epsilon_{t+1}$ is an $r \times 1$
shock.

The $r$ modes $\bar X_t$ obey the  first-order VAR {eq}`eq:VARmodes` in which $\Lambda$ is an $r \times r$ diagonal matrix.  

Note that while $\Lambda$ is diagonal, the contemporaneous covariance matrix of $\bar \epsilon_{t+1}$ need not be.


**Remark:** It is permissible for $X_t$ to contain lagged values of  observables.

 For example, we might have a setting in which 

$$
X_t = \begin{bmatrix}
y_{1t} \cr
y_{1,t-1} \cr
\vdots \cr
y_{1, t-k}\cr
y_{2,t} \cr
y_{2, t-1} \cr
\vdots
\end{bmatrix}
$$

+++

## Source for Some Python Code

You can find a Python implementation of DMD here:

https://mathlab.github.io/PyDMD/
