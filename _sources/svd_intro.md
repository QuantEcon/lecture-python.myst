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
form the backbone of important parts of modern machine learning methods.

This lecture describes the singular value decomposition and two of its uses:

 * principal components analysis (PCA)

 * dynamic mode decomposition (DMD)

 Each of these can be thought of as data-reduction methods that are designed to capture principal patterns in data by projecting data onto a limited set of factors.

##  The Setup

Let $X$ be an $m \times n$ matrix of rank $r$.

In this lecture, we'll think of $X$ as a matrix of **data**.

  * each column is an **individual** -- a time period or person, depending on the application
  
  * each row is a **random variable** measuring an attribute of a time period or a person, depending on the application
  
  
We'll be interested in  two distinct cases

  * A **short and fat** case in which $m << n$, so that there are many more columns than rows.

  * A  **tall and skinny** case in which $m >> n$, so that there are many more rows than columns. 
    
   
We'll apply a **singular value decomposition** of $X$ in both situations.

In the first case in which there are many more observations $n$ than random variables $m$, we learn about the joint distribution of the  random variables by taking averages  across observations of functions of the observations. 

Here we'll look for **patterns** by using a **singular value decomosition** to do a **principal components analysis** (PCA).

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

* $\Sigma$ is an $m \times r$ matrix in which the first $r$ places on its main diagonal are positive numbers $\sigma_1, \sigma_2, \ldots, \sigma_r$ called **singular values**; remaining entries of $\Sigma$ are all zero

* The $r$ singular values are square roots of the eigenvalues of the $m \times m$ matrix  $X X^T$ and the $n \times n$ matrix $X^T X$

* When $U$ is a complex valued matrix, $U^T$ denotes the **conjugate-transpose** or **Hermitian-transpose** of $U$, meaning that 
$U_{ij}^T$ is the complex conjugate of $U_{ji}$. 

* Similarly, when $V$ is a complex valued matrix, $V^T$ denotes the **conjugate-transpose** or **Hermitian-transpose** of $V$

The shapes of $U$, $\Sigma$, and $V$ are $\left(m, m\right)$, $\left(m, n\right)$, $\left(n, n\right)$, respectively. 

Below, we shall assume these shapes.

However, though we chose not to, there is an alternative shape convention that we could have used.

Thus, note that because we assume that $A$ has rank $r$, there are only $r $ nonzero singular values, where $r=\textrm{rank}(A)\leq\min\left(m, n\right)$.  

Therefore,  we could also write $U$, $\Sigma$, and $V$ as matrices with shapes $\left(m, r\right)$, $\left(r, r\right)$, $\left(r, n\right)$.

Sometimes, we will choose the former one to be consistent with what is adopted by `numpy`.

At other times, we'll use the latter convention in which $\Sigma$ is an $r \times r$  diagonal matrix.

Also, when we discuss the **dynamic mode decomposition** below, we'll use a special case of the latter  convention in which it is understood that
$r$ is just a pre-specified small number of leading singular values that we think capture the  most interesting  dynamics.

## Digression:  Polar Decomposition

 Through  the following identities, the singular value decomposition (SVD) is related to the **polar decomposition** of $X$

\begin{align*}
X & = SQ  \cr  
S & = U\Sigma U^T \cr
Q & = U V^T 
\end{align*}

where $S$ is evidently a symmetric matrix and $Q$ is an orthogonal matrix.

## Principle Components Analysis (PCA)

Let's begin with a case in which $n >> m$, so that we have many  more observations $n$ than random variables $m$.

The data matrix $X$ is **short and fat**  in an  $n >> m$ case as opposed to a **tall and skinny** case with $m > > n $ to be discussed later in this lecture.

We regard  $X$ as an  $m \times n$ matrix of **data**:

$$
X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_n\end{bmatrix}
$$

where for $j = 1, \ldots, n$ the column vector $X_j = \begin{bmatrix}X_{1j}\\X_{2j}\\\vdots\\X_{mj}\end{bmatrix}$ is a  vector of observations on variables $\begin{bmatrix}x_1\\x_2\\\vdots\\x_m\end{bmatrix}$.

In a **time series** setting, we would think of columns $j$ as indexing different __times__ at which random variables are observed, while rows index different random variables.

In a **cross section** setting, we would think of columns $j$ as indexing different __individuals__ for  which random variables are observed, while rows index different **random variables**.

The number of singular values equals the rank of  matrix $X$.

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

* $ V_{k}^T= \begin{bmatrix}V_{k1} &  V_{k2} & \ldots & V_{kn}\end{bmatrix}  \quad \textrm{for each} \   k=1, \ldots, n $ is a time series  $\lbrace V_{kj} \rbrace_{j=1}^n$ for the $k$th principal component

* $U_j = \begin{bmatrix}U_{1k}\\U_{2k}\\\ldots\\U_{mk}\end{bmatrix} \  k=1, \ldots, m$
is a vector of loadings of variables $X_i$ on the $k$th principle component,  $i=1, \ldots, m$

* $\sigma_k $ for each $k=1, \ldots, r$ is the strength of $k$th **principal component**

## Reduced Versus Full SVD

You can read about reduced and full SVD here
<https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html>
 
Let's do a small experiment to see the difference

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


## PCA with Eigenvalues and Eigenvectors

We now  turn to using the eigen decomposition of a sample covariance matrix to do PCA.

Let $X_{m \times n}$ be our $m \times n$ data matrix.

Let's assume that sample means of all variables are zero.

We can make sure that this is true by **pre-processing** the data by substracting sample means appropriately.

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
\epsilon\epsilon^T=\Lambda
$$ 

We can verify that

$$
XX^T=P\Lambda P^T
$$

It follows that we can represent the data matrix as 

\begin{equation*}
X=\begin{bmatrix}X_1|X_2|\ldots|X_m\end{bmatrix} =\begin{bmatrix}P_1|P_2|\ldots|P_m\end{bmatrix}
\begin{bmatrix}\epsilon_1\\\epsilon_2\\\ldots\\\epsilon_m\end{bmatrix} 
= P_1\epsilon_1+P_2\epsilon_2+\ldots+P_m\epsilon_m
\end{equation*}

where 

$$
\epsilon\epsilon^T=\Lambda
$$

To reconcile the preceding representation with the PCA that we obtained through the SVD above, we first note that $\epsilon_j^2=\lambda_j\equiv\sigma^2_j$.

Now define  $\tilde{\epsilon_j} = \frac{\epsilon_j}{\sqrt{\lambda_j}}$
which evidently implies that $\tilde{\epsilon}_j\tilde{\epsilon}_j^T=1$.

Therefore 

$$
\begin{aligned}
X&=\sqrt{\lambda_1}P_1\tilde{\epsilon_1}+\sqrt{\lambda_2}P_2\tilde{\epsilon_2}+\ldots+\sqrt{\lambda_m}P_m\tilde{\epsilon_m}\\
&=\sigma_1P_1\tilde{\epsilon_2}+\sigma_2P_2\tilde{\epsilon_2}+\ldots+\sigma_mP_m\tilde{\epsilon_m}
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

We want a way that leads to the same $U$ and $P$. 

In the following, we accomplish this by

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

## Dynamic Mode Decomposition (DMD)

We now turn to the case in which $m >>n $ so that there are many more random variables $m$ than observations $n$.

This is the **tall and skinny** case associated with **Dynamic Mode Decomposition**.

You can read about Dynamic Mode Decomposition here {cite}`DMD_book`.

We start  with an $m \times n $ matrix of data $\tilde X$ of the form 


$$
 \tilde X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_n\end{bmatrix}
$$ 

where for $t = 1, \ldots, n$,  the $m \times 1 $ vector $X_t$ is

$$ X_t = \begin{bmatrix}  X_{1,t} & X_{2,t} & \cdots & X_{m,t}     \end{bmatrix}^T $$

where $T$ denotes transposition and $X_{i,t}$ is an observations on variable $i$ at time $t$.

From $\tilde X$,   form two matrices 

$$
 X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_{n-1}\end{bmatrix}
$$ 

and

$$
X' =  \begin{bmatrix} X_2 \mid X_3 \mid \cdots \mid X_n\end{bmatrix}
$$

(Note that here $'$ does not denote matrix transposition but instead is part of the name of the matrix $X'$.)

In forming $ X$ and $X'$, we have in each case  dropped a column from $\tilde X$.

Evidently, $ X$ and $ X'$ are both $m \times \tilde n$ matrices where $\tilde n = n - 1$.

We start with a system consisting of $m$ least squares regressions of **everything** on one lagged value of **everything**:

$$
 X' = A  X + \epsilon
$$

where 

$$
A =  X'  X^{+}
$$

and where the (huge) $m \times m $ matrix $X^{+}$ is the Moore-Penrose generalized inverse of $X$ that we could compute
as 

$$
X^{+} = V \Sigma^{-1} U^T
$$

where the matrix $\Sigma^{-1}$ is constructed by replacing each non-zero element of $\Sigma$ with $\sigma_j^{-1}$.

The idea behind **dynamic mode decomposition** is to construct an approximation that  

* sidesteps computing the generalized inverse $X^{+}$

* retains only the largest  $\tilde r< < r$ eigenvalues and associated eigenvectors of $U$ and $V^T$ 

* constructs an $m \times \tilde r$ matrix $\Phi$ that captures effects  on all $m$ variables of $r$ dynamic modes

* uses $\Phi$ and  powers of $\tilde r$ leading singular values to forecast *future* $X_t$'s

The magic of **dynamic mode decomposition** is that we accomplish this without ever computing the regression coefficients $A = X' X^{+}$.

To construct a DMD, we deploy the following steps:

* Compute the singular value decomposition 

  $$ 
  X = U \Sigma V^T
  $$
  
  where $U$ is $m \times r$, $\Sigma$ is an $r \times r$ diagonal  matrix, and $V^T$ is an $r \times \tilde n$ matrix. 
  
  
* Notice that (though it would be costly), we could compute $A$ by solving 

  $$
  A = X' V \Sigma^{-1} U^T
  $$
  
  But we won't do that.  
  
  Instead we'll proceed as follows.
  
  Note that since,  $X' = A U \Sigma V^T$, we know that 
  
  $$
  A U  =  X' V \Sigma^{-1}
  $$
    
  so that 
  
  $$
  U^T X' V \Sigma^{-1} = U^T A U \equiv \tilde A
  $$ (eq:tildeAform)
    
* At this point,  we  deploy a reduced-dimension version of formula {eq}`eq:tildeAform} by
* using only the  columns of $U$ that correspond to the $\tilde r$ largest singular values.  
  
  Tu et al. {cite}`tu_Rowley` verify that eigenvalues and eigenvectors of $\tilde A$ equal the leading eigenvalues and associated eigenvectors of $A$.

* Construct an eigencomposition of $\tilde A$ that satisfies

  $$ 
  \tilde A W =  W \Lambda
  $$
  
  where $\Lambda$ is a $\tilde r \times \tilde r$ diagonal matrix of eigenvalues and the columns of $W$ are corresponding eigenvectors
  of $\tilde A$.   Both $\Lambda$ and $W$ are $\tilde r \times \tilde r$ matrices.
  
* Construct the $m \times \tilde r$ matrix

  $$
  \Phi = X' V \Sigma^{-1} W
  $$
  
  Let $\Phi^{+}$ be a generalized inverse of $\Phi$; $\Phi^{+}$ is an $\tilde r \times m$ matrix. 
  
* Define an initial vector $b$ of dominant modes by

  $$
  b= \Phi^{+} X_1
  $$
  
  where evidently $b$ is an $\tilde r \times 1$ vector.
    
With $\Lambda, \Phi, \Phi^{+}$ in hand, our least-squares fitted dynamics fitted to the $\tilde r$ dominant modes
are governed by

$$
X_{t+1} = \Phi \Lambda \Phi^{+} X_t
$$

 
Conditional on $X_t$, we construct forecasts $\check X_{t+j} $ of $X_{t+j}, j = 1, 2, \ldots, $  from 

$$
\check X_{t+j} = \Phi \Lambda^j \Phi^{+} X_t
$$


## Reduced-order VAR

DMD  is a natural tool for estimating a **reduced order vector autoregression**,
an object that we define in terms of the populations regression equation

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
\tilde X_{t+1} = \Lambda \tilde X_t + \tilde \epsilon_{t+1} 
$$ (eq:VARmodes)

where $\tilde X_t $ is an $r \times 1$ **mode** and $\tilde \epsilon_{t+1}$ is an $r \times 1$
shock.

The $r$ modes $\tilde X_t$ obey the  first-order VAR {eq}`eq:VARmodes` in which $\Lambda$ is an $r \times r$ diagonal matrix.  

Note that while $\Lambda$ is diagonal, the contemporaneous covariance matrix of $\tilde \epsilon_{t+1}$ need not be.


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
