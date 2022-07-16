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
form  foundations for important machine learning methods.

This lecture describes the singular value decomposition and two of its uses:

 * principal components analysis (PCA)

 * dynamic mode decomposition (DMD)

 Each of these can be thought of as a data-reduction procedure  designed to capture salient patterns by projecting data onto a limited set of factors.

##  The Setup

Let $X$ be an $m \times n$ matrix of rank $p$.

Necessarily, $p \leq \min(m,n)$.

In this lecture, we'll think of $X$ as a matrix of **data**.

  * each column is an **individual** -- a time period or person, depending on the application
  
  * each row is a **random variable** describing an attribute of a time period or a person, depending on the application
  
  
We'll be interested in  two  cases

  * A **short and fat** case in which $m << n$, so that there are many more columns (individuals) than rows (attributes).

  * A  **tall and skinny** case in which $m >> n$, so that there are many more rows  (attributes) than columns (individuals). 
    
   
We'll apply a **singular value decomposition** of $X$ in both situations.

In the first case in which there are many more individuals $n$ than attributes $m$, we learn sample moments of  a joint distribution  by taking averages  across observations of functions of the observations. 

In this $ m < < n$ case,  we'll look for **patterns** by using a **singular value decomposition** to do a **principal components analysis** (PCA).

In the $m > > n$  case in which there are many more attributes $m$ than individuals $n$, we'll proceed in a different way. 

We'll again use a **singular value decomposition**,  but now to construct a **dynamic mode decomposition** (DMD)

## Singular Value Decomposition

A **singular value decomposition** of an $m \times n$ matrix $X$ of rank $p \leq \min(m,n)$ is

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

* $\Sigma$ is an $m \times n$ matrix in which the first $p$ places on its main diagonal are positive numbers $\sigma_1, \sigma_2, \ldots, \sigma_p$ called **singular values**; remaining entries of $\Sigma$ are all zero

* The $p$ singular values are square roots of the eigenvalues of the $m \times m$ matrix  $X X^T$ and the $n \times n$ matrix $X^T X$

* When $U$ is a complex valued matrix, $U^T$ denotes the **conjugate-transpose** or **Hermitian-transpose** of $U$, meaning that 
$U_{ij}^T$ is the complex conjugate of $U_{ji}$. 

* Similarly, when $V$ is a complex valued matrix, $V^T$ denotes the **conjugate-transpose** or **Hermitian-transpose** of $V$

In what is called a **full** SVD, the  shapes of $U$, $\Sigma$, and $V$ are $\left(m, m\right)$, $\left(m, n\right)$, $\left(n, n\right)$, respectively. 



There is also an alternative shape convention called an **economy** or **reduced** SVD .

Thus, note that because we assume that $X$ has rank $p$, there are only $p$ nonzero singular values, where $p=\textrm{rank}(X)\leq\min\left(m, n\right)$.  

A **reduced** SVD uses this fact to express $U$, $\Sigma$, and $V$ as matrices with shapes $\left(m, p\right)$, $\left(p, p\right)$, $\left( n, p\right)$.



## Properties of Full and Reduced SVD's



You can read about reduced and full SVD here
<https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html>

For a full SVD, 

\begin{align*}
UU^T &  = I  &  \quad U^T U = I \cr    
VV^T & = I & \quad V^T V = I
\end{align*}

But these properties don't hold for a  **reduced** SVD.

Which properties hold depend on whether we are in a **tall-skinny** case or a **short-fat** case.

 * In a **tall-skinny** case in which $m > > n$, for a **reduced** SVD


\begin{align*}
UU^T &  \neq I  &  \quad U^T U = I \cr    
VV^T & = I & \quad V^T V = I
\end{align*}

 * In a **short-fat** case in which $m < < n$, for a **reduced** SVD

\begin{align*}
UU^T &  = I  &  \quad U^T U = I \cr    
VV^T & = I & \quad V^T V \neq I
\end{align*}

When we study Dynamic Mode Decomposition below, we shall want to remember this caveat because sometimes we'll be using reduced SVD's to compute key objects.


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
print('U, S, V ='), U, S, V
```

```{code-cell} ipython3
print('Uhat, Shat, Vhat = '), Uhat, Shat, Vhat
```

```{code-cell} ipython3
rr = np.linalg.matrix_rank(X)
print('rank of X - '), rr
```


**Properties:**

* Where $U$ is constructed via a full SVD, $U^T U = I_{p\times p}$ and  $U U^T = I_{m \times m}$ 
* Where $\hat U$ is constructed via a reduced SVD, although $\hat U^T \hat U = I_{p\times p}$ it happens that  $\hat U \hat U^T \neq I_{m \times m}$ 

We illustrate these properties for our example with the following code cells.

```{code-cell} ipython3
UTU = U.T@U
UUT = U@U.T
print('UUT, UTU = '), UUT, UTU 
```


```{code-cell} ipython3
UhatUhatT = Uhat@Uhat.T
UhatTUhat = Uhat.T@Uhat
print('UhatUhatT, UhatTUhat= '), UhatUhatT, UhatTUhat
```




**Remarks:** 

The cells above illustrate application of the  `fullmatrices=True` and `full-matrices=False` options.
Using `full-matrices=False` returns a reduced singular value decomposition.

This option implements an optimal reduced rank approximation of a matrix, in the sense of  minimizing the Frobenius
norm of the discrepancy between the approximating matrix and the matrix being approximated.


Optimality in this sense is  established in the celebrated Eckart–Young theorem. See <https://en.wikipedia.org/wiki/Low-rank_approximation>.

When we study Dynamic Mode Decompositions below, it  will be important for us to remember the preceding properties of full and reduced SVD's in such tall-skinny cases.  





Now let's turn to a short-fat case.

To illustrate this case,  we'll set $m = 2 < 5 = n $

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
print('rank X = '), rr
```
## Digression:  Polar Decomposition

A singular value decomposition (SVD) of $X$ is related to a **polar decomposition** of $X$

$$
X  = SQ   
$$

where

\begin{align*}
 S & = U\Sigma U^T \cr
Q & = U V^T 
\end{align*}

and $S$ is evidently a symmetric matrix and $Q$ is an orthogonal matrix.

## Principal Components Analysis (PCA)

Let's begin with a case in which $n >> m$, so that we have many  more individuals $n$ than attributes $m$.

The  matrix $X$ is **short and fat**  in an  $n >> m$ case as opposed to a **tall and skinny** case with $m > > n $ to be discussed later.

We regard  $X$ as an  $m \times n$ matrix of **data**:

$$
X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_n\end{bmatrix}
$$

where for $j = 1, \ldots, n$ the column vector $X_j = \begin{bmatrix}X_{1j}\\X_{2j}\\\vdots\\X_{mj}\end{bmatrix}$ is a  vector of observations on variables $\begin{bmatrix}x_1\\x_2\\\vdots\\x_m\end{bmatrix}$.

In a **time series** setting, we would think of columns $j$ as indexing different __times__ at which random variables are observed, while rows index different random variables.

In a **cross section** setting, we would think of columns $j$ as indexing different __individuals__ for  which random variables are observed, while rows index different **attributes**.

The number of positive singular values equals the rank of  matrix $X$.

Arrange the singular values  in decreasing order.

Arrange   the positive singular values on the main diagonal of the matrix $\Sigma$ of into a vector $\sigma_R$.

Set all other entries of $\Sigma$ to zero.

## Relationship of PCA to SVD

To relate a SVD to a PCA (principal component analysis) of data set $X$, first construct  the  SVD of the data matrix $X$:

$$
X = U \Sigma V^T = \sigma_1 U_1 V_1^T + \sigma_2 U_2 V_2^T + \cdots + \sigma_p U_p V_p^T
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
X = \sigma_1 \begin{pmatrix}U_{11}V_{1}^T\\U_{21}V_{1}^T\\\cdots\\U_{m1}V_{1}^T\\\end{pmatrix} + \sigma_2\begin{pmatrix}U_{12}V_{2}^T\\U_{22}V_{2}^T\\\cdots\\U_{m2}V_{2}^T\\\end{pmatrix}+\ldots + \sigma_p\begin{pmatrix}U_{1p}V_{p}^T\\U_{2p}V_{p}^T\\\cdots\\U_{mp}V_{p}^T\\\end{pmatrix}
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
\epsilon = P^{-1} X
$$

and

$$
\epsilon\epsilon^T=\Lambda .
$$ 

We can verify that

$$
XX^T=P\Lambda P^T .
$$ (eq:XXo)

It follows that we can represent the data matrix $X$  as 

\begin{equation*}
X=\begin{bmatrix}X_1|X_2|\ldots|X_m\end{bmatrix} =\begin{bmatrix}P_1|P_2|\ldots|P_m\end{bmatrix}
\begin{bmatrix}\epsilon_1\\\epsilon_2\\\ldots\\\epsilon_m\end{bmatrix} 
= P_1\epsilon_1+P_2\epsilon_2+\ldots+P_m\epsilon_m
\end{equation*}


To reconcile the preceding representation with the PCA that we had obtained earlier through the SVD, we first note that $\epsilon_j^2=\lambda_j\equiv\sigma^2_j$.

Now define  $\tilde{\epsilon_j} = \frac{\epsilon_j}{\sqrt{\lambda_j}}$, 
which  implies that $\tilde{\epsilon}_j\tilde{\epsilon}_j^T=1$.

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

Because  there are alternative algorithms for  computing  $P$ and $U$ for  given a data matrix $X$, depending on  algorithms used, we might have sign differences or different orders between eigenvectors.

We can resolve such ambiguities about  $U$ and $P$ by

1. sorting eigenvalues and singular values in descending order
2. imposing positive diagonals on $P$ and $U$ and adjusting signs in $V^T$ accordingly

## Connections

To pull things together, it is useful to assemble and compare some formulas presented above.

First, consider an  SVD of an $m \times n$ matrix:

$$
X = U\Sigma V^T
$$

Compute:

$$
\begin{align}
XX^T&=U\Sigma V^TV\Sigma^T U^T\cr
&\equiv U\Sigma\Sigma^TU^T\cr
&\equiv U\Lambda U^T
\end{align}
$$  (eq:XXcompare)

Compare representation {eq}`eq:XXcompare` with equation {eq}`eq:XXo` above.

Evidently, $U$ in the SVD is the matrix $P$  of
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

Look at  parts of that lecture that describe and illustrate the classic factor analysis model.

## Dynamic Mode Decomposition (DMD)



We turn to the **tall and skinny** case  associated with **Dynamic Mode Decomposition**, the case in  which $ m >>n $.

Here an $ m \times n $  data matrix $ \tilde X $ contains many more attributes $ m $ than individuals $ n $.

This  

Dynamic mode decomposition was introduced by {cite}`schmid2010`,

You can read more about Dynamic Mode Decomposition here [[KBBWP16](https://python.quantecon.org/zreferences.html#id24)] and here [[BK19](https://python.quantecon.org/zreferences.html#id25)] (section 7.2).


We want to fit a **first-order vector autoregression**

$$
X_{t+1} = A X_t + C \epsilon_{t+1}
$$ (eq:VARfirstorder)

where $\epsilon_{t+1}$ is the time $t+1$ instance of an i.i.d. $m \times 1$ random vector with mean vector
zero and identity  covariance matrix and

where 
the $ m \times 1 $ vector $ X_t $ is

$$
X_t = \begin{bmatrix}  X_{1,t} & X_{2,t} & \cdots & X_{m,t}     \end{bmatrix}^T
$$ (eq:Xvector)

and where $ T $ again denotes complex transposition and $ X_{i,t} $ is an observation on variable $ i $ at time $ t $.



We want to fit equation {eq}`eq:VARfirstorder`. 


Our data are organized in   an $ m \times (n+1) $ matrix  $ \tilde X $ 

$$
\tilde X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_n \mid X_{n+1} \end{bmatrix}
$$

where for $ t = 1, \ldots, n +1 $,  the $ m \times 1 $ vector $ X_t $ is given by {eq}`eq:Xvector`. 

Thus, we want to estimate a  system  {eq}`eq:VARfirstorder` that consists of $ m $ least squares regressions of **everything** on one lagged value of **everything**.

The $i$'th equation of {eq}`eq:VARfirstorder` is a regression of $X_{i,t+1}$ on the vector $X_t$.


We proceed as follows. 


From $ \tilde X $,  we  form two $m \times n$ matrices

$$
X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_{n}\end{bmatrix}
$$

and

$$
X' =  \begin{bmatrix} X_2 \mid X_3 \mid \cdots \mid X_{n+1}\end{bmatrix}
$$

Here $ ' $ does not indicate matrix transposition but instead is part of the name of the matrix $ X' $.

In forming $ X $ and $ X' $, we have in each case  dropped a column from $ \tilde X $,  the last column in the case of $ X $, and  the first column in the case of $ X' $.

Evidently, $ X $ and $ X' $ are both $ m \times  n $ matrices.

We denote the rank of $ X $ as $ p \leq \min(m, n)  $.

Two possible cases are 

 *  $ n > > m$, so that we have many more time series  observations $n$ than variables $m$
 *  $m > > n$, so that we have many more variables $m $ than time series observations $n$

At a general level that includes both of these special cases, a common formula describes the least squares estimator $\hat A$ of $A$ for both cases.

But some important  details differ.

The common formula is

$$ 
\hat A = X' X^+ 
$$ (eq:commonA)

where $X^+$ is the pseudo-inverse of $X$.

Applicable formulas for the pseudo-inverse differ for our two cases.

**Short-Fat Case:**

When $ n > > m$, so that we have many more time series  observations $n$ than variables $m$ and when
$X$ has linearly independent **rows**, $X X^T$ has an inverse and the pseudo-inverse $X^+$ is

$$
X^+ = X^T (X X^T)^{-1} 
$$

Here $X^+$ is a **right-inverse** that verifies $ X X^+ = I_{m \times m}$.

In this case, our formula {eq}`eq:commonA` for the least-squares estimator of the population matrix of regression coefficients  $A$ becomes

$$ 
\hat A = X' X^T (X X^T)^{-1}
$$ (eq:Ahatform101)


This  formula for least-squares regression coefficients widely used in econometrics.

For example, it is used  to estimate vector autorgressions.   

The right side of formula {eq}`eq:Ahatform101` is proportional to the empirical cross second moment matrix of $X_{t+1}$ and $X_t$ times the inverse
of the second moment matrix of $X_t$.



**Tall-Skinny Case:**

When $m > > n$, so that we have many more attributes $m $ than time series observations $n$ and when $X$ has linearly independent **columns**,
$X^T X$ has an inverse and the pseudo-inverse $X^+$ is

$$
X^+ = (X^T X)^{-1} X^T
$$

Here  $X^+$ is a **left-inverse** that verifies $X^+ X = I_{n \times n}$.

In this case, our formula  {eq}`eq:commonA` for a least-squares estimator of $A$ becomes

$$
\hat A = X' (X^T X)^{-1} X^T
$$ (eq:hatAversion0)

Please compare formulas {eq}`eq:Ahatform101` and {eq}`eq:hatAversion0` for $\hat A$.

Here we are interested in formula {eq}`eq:hatAversion0`.

The $ i $th  row of $ \hat A $ is an $ m \times 1 $ vector of regression coefficients of $ X_{i,t+1} $ on $ X_{j,t}, j = 1, \ldots, m $.


If we use formula {eq}`eq:hatAversion0` to calculate $\hat A X$ we find that

$$
\hat A X = X'
$$

so that the regression equation **fits perfectly**.

This is the usual outcome in an **underdetermined least-squares** model.


To reiterate, in our **tall-skinny** case  in which we have a number $n$ of observations   that is small relative to the number $m$ of
attributes that appear in the vector $X_t$,  we want to fit equation {eq}`eq:VARfirstorder`.


To  offer  ideas about how we can efficiently calculate the pseudo-inverse $X^+$, as our  estimator $\hat A$ of $A$ we form an  $m \times m$ matrix that  solves the least-squares best-fit problem

$$ 
\hat A = \textrm{argmin}_{\check A} || X' - \check  A X ||_F   
$$ (eq:ALSeqn)

where $|| \cdot ||_F$ denotes the Frobenius (or Euclidean) norm of a matrix.

The Frobenius norm is defined as

$$
 ||A||_F = \sqrt{ \sum_{i=1}^m \sum_{j=1}^m |A_{ij}|^2 }
$$


The minimizer of the right side of equation {eq}`eq:ALSeqn` is

$$
\hat A =  X'  X^{+}  
$$ (eq:hatAform)

where the (possibly huge) $ n \times m $ matrix $ X^{+} = (X^T X)^{-1} X^T$ is again a pseudo-inverse of $ X $.




For some situations that we are interested in, $X^T X $ can be close to singular, a situation that can make some numerical algorithms  be error-prone.

To acknowledge that possibility, we'll use  efficient algorithms for computing and for constructing reduced rank approximations of  $\hat A$ in formula {eq}`eq:hatAversion0`.
 

The $ i $th  row of $ \hat A $ is an $ m \times 1 $ vector of regression coefficients of $ X_{i,t+1} $ on $ X_{j,t}, j = 1, \ldots, m $.

An efficient way to compute the pseudo-inverse $X^+$ is to start with  a singular value decomposition



$$
X =  U \Sigma  V^T 
$$ (eq:SVDDMD)

where we remind ourselves that for a **reduced** SVD, $X$ is an $m \times n$ matrix of data, $U$ is an $m \times p$ matrix, $\Sigma$  is a $p \times p$ matrix, and $V is an $n \times p$ matrix.  

We can    efficiently  construct the pertinent pseudo-inverse $X^+$
by recognizing the following string of equalities.  

$$
\begin{aligned}
X^{+} & = (X^T X)^{-1} X^T \\
  & = (V \Sigma U^T U \Sigma V^T)^{-1} V \Sigma U^T \\
  & = (V \Sigma \Sigma V^T)^{-1} V \Sigma U^T \\
  & = V \Sigma^{-1} \Sigma^{-1} V^T V \Sigma U^T \\
  & = V \Sigma^{-1} U^T 
\end{aligned}
$$ (eq:efficientpseudoinverse)


(Since we are in the $m > > n$ case in which $V^T V = I_{p \times p}$ in a reduced SVD, we can use the preceding
string of equalities for a reduced SVD as well as for a full SVD.)

Thus, we shall  construct a pseudo-inverse $ X^+ $  of $ X $ by using
a singular value decomposition of $X$ in equation {eq}`eq:SVDDMD`  to compute


$$
X^{+} =  V \Sigma^{-1}  U^T 
$$ (eq:Xplusformula)

where the matrix $ \Sigma^{-1} $ is constructed by replacing each non-zero element of $ \Sigma $ with $ \sigma_j^{-1} $.

We can  use formula {eq}`eq:Xplusformula`   together with formula {eq}`eq:hatAform` to compute the matrix  $ \hat A $ of regression coefficients.

Thus, our  estimator $\hat A = X' X^+$ of the $m \times m$ matrix of coefficients $A$    is

$$
\hat A = X' V \Sigma^{-1}  U^T 
$$ (eq:AhatSVDformula)

We’ll eventually use **dynamic mode decomposition** to compute a rank $ r $ approximation to $ \hat A $,
where $ r <  p $.
  
**Remark:** In our Python code, we'll sometimes use  a reduced SVD.


Next, we describe alternative representations of our first-order linear dynamic system.

+++

## Representation 1
 
In this representation, we shall use a **full** SVD of $X$.

We use the $m$  **columns** of $U$, and thus the $m$ **rows** of $U^T$,  to define   a $m \times 1$  vector $\tilde b_t$ as 


$$
\tilde b_t = U^T X_t .
$$ (eq:tildeXdef2)

The original  data $X_t$ can be represented as

$$ 
X_t = U \tilde b_t
$$ (eq:Xdecoder)

(Here we use $b$ to remind ourselves that we are creating a **basis** vector.)

Since we are now using a **full** SVD, $U U^T = I_{m \times m}$.

So it follows from equation {eq}`eq:tildeXdef2` that we can reconstruct  $X_t$ from $\tilde b_t$.

In particular,  



 * Equation {eq}`eq:tildeXdef2` serves as an **encoder** that  **rotates** the $m \times 1$ vector $X_t$ to become an $m \times 1$ vector $\tilde b_t$ 
  
 * Equation {eq}`eq:Xdecoder` serves as a **decoder** that **reconstructs** the $m \times 1$ vector $X_t$ by rotating  the $m \times 1$ vector $\tilde b_t$ 



Define a  transition matrix for an $m \times 1$ basis vector  $\tilde b_t$ by

$$ 
\tilde A = U^T \hat A U 
$$ (eq:Atilde0)

We can evidently recover $\hat A$ from

$$
\hat A = U \tilde A U^T 
$$

Dynamics of the rotated $m \times 1$ state $\tilde b_t$ are governed by

$$
\tilde b_{t+1} = \tilde A \tilde b_t 
$$

To construct forecasts $\overline X_t$ of  future values of $X_t$ conditional on $X_1$, we can apply  decoders
(i.e., rotators) to both sides of this 
equation and deduce

$$
\overline X_{t+1} = U \tilde A^t U^T X_1
$$

where we use $\overline X_t$ to denote a forecast.

+++

## Representation 2


This representation is related to  one originally proposed by  {cite}`schmid2010`.

It can be regarded as an intermediate step to  a related and perhaps more useful  representation 3.


As with Representation 1, we continue to

* use a **full** SVD and **not** a reduced SVD



As we observed and illustrated  earlier in this lecture, for a full SVD
$U U^T$ and $U^T U$ are both identity matrices; but under a reduced SVD of $X$, $U^T U$ is not an identity matrix.  

As we shall see, a full SVD is  too confining for what we ultimately want to do, namely,  situations in which  $U^T U$ is **not** an identity matrix because we  use a reduced SVD of $X$.

But for now, let's proceed under the assumption that both of the  preceding two  requirements are satisfied.

 

Form an eigendecomposition of the $m \times m$ matrix $\tilde A = U^T \hat A U$ defined in equation {eq}`eq:Atilde0`:

$$
\tilde A = W \Lambda W^{-1} 
$$ (eq:tildeAeigen)

where $\Lambda$ is a diagonal matrix of eigenvalues and $W$ is an $m \times m$
matrix whose columns are eigenvectors  corresponding to rows (eigenvalues) in 
$\Lambda$.

When $U U^T = I_{m \times m}$, as is true with a full SVD of $X$, it follows that 

$$ 
\hat A = U \tilde A U^T = U W \Lambda W^{-1} U^T 
$$ (eq:eqeigAhat)

Evidently, according to equation {eq}`eq:eqeigAhat`, the diagonal matrix $\Lambda$ contains eigenvalues of 
$\hat A$ and corresponding eigenvectors of $\hat A$ are columns of the matrix $UW$. 


Thus, the systematic (i.e., not random) parts of the $X_t$ dynamics captured by our first-order vector autoregressions   are described by

$$
X_{t+1} = U W \Lambda W^{-1} U^T  X_t 
$$

Multiplying both sides of the above equation by $W^{-1} U^T$ gives

$$ 
W^{-1} U^T X_{t+1} = \Lambda W^{-1} U^T X_t 
$$

or 

$$
\hat b_{t+1} = \Lambda \hat b_t
$$

where now our encoder is

$$ 
\hat b_t = W^{-1} U^T X_t
$$

and our decoder is

$$
X_t = U W \hat b_t
$$

We can use this representation to construct a predictor $\overline X_{t+1}$ of $X_{t+1}$ conditional on $X_1$  via: 

$$
\overline X_{t+1} = U W \Lambda^t W^{-1} U^T X_1 
$$ (eq:DSSEbookrepr)


In effect, 
{cite}`schmid2010` defined an $m \times m$ matrix $\Phi_s$ as

$$ 
\Phi_s = UW 
$$ (eq:Phisfull)

and represented equation {eq}`eq:DSSEbookrepr` as

$$
\overline X_{t+1} = \Phi_s \Lambda^t \Phi_s^+ X_1 
$$ (eq:schmidrep)

Components of the  basis vector $ \hat b_t = W^{-1} U^T X_t \equiv \Phi_s^+$ are often  called DMD **modes**, or sometimes also
DMD **projected nodes**.    




We turn next  to an alternative  representation suggested by  Tu et al. {cite}`tu_Rowley`, one that is more appropriate to use when, as in practice is typically the case, we use a reduced SVD.




## Representation 3

Departing from the procedures used to construct  Representations 1 and 2, each of which deployed a **full** SVD, we now use a **reduced** SVD.  

Again, we let  $p \leq \textrm{min}(m,n)$ be the rank of $X$.

Construct a **reduced** SVD

$$
X = \tilde U \tilde \Sigma \tilde V^T, 
$$

where now $U$ is $m \times p$ and $\Sigma$ is $ p \times p$ and $V^T$ is $p \times n$. 

Our minimum-norm least-squares estimator  approximator of  $A$ now has representation 

$$
\hat A = X' \tilde V \tilde \Sigma^{-1} \tilde U^T
$$


Paralleling a step in Representation 1, define a  transition matrix for a rotated $p \times 1$ state $\tilde b_t$ by

$$ 
\tilde A =\tilde  U^T \hat A \tilde U 
$$ (eq:Atildered)

Because we are now working with a reduced SVD, so that $\tilde U \tilde U^T \neq I$, since $\hat A \neq \tilde U \tilde A \tilde U^T$, we can't simply  recover $\hat A$ from  $\tilde A$ and $\tilde U$. 


Nevertheless, hoping for the best, we persist and construct an eigendecomposition of what  is now a 
$p \times p$ matrix $\tilde A$:

$$
 \tilde A =  W  \Lambda  W^{-1}
$$ (eq:tildeAeigenred)


Mimicking our procedure in Representation 2, we cross our fingers and compute the $m \times p$ matrix

$$
\tilde \Phi_s = \tilde U W
$$ (eq:Phisred)

that  corresponds to {eq}`eq:Phisfull` for a full SVD.  

At this point, it is interesting to compute $\hat A \tilde  \Phi_s$:

$$
\begin{aligned}
\hat A \tilde \Phi_s & = (X' \tilde V \tilde \Sigma^{-1} \tilde U^T) (\tilde U W) \\
  & = X' \tilde V \tilde \Sigma^{-1} W \\
  & \neq (\tilde U W) \Lambda \\
  & = \tilde \Phi_s \Lambda
  \end{aligned}
$$
 
That 
$ \hat A \tilde \Phi_s \neq \tilde \Phi_s \Lambda $ means, that unlike the  corresponding situation in Representation 2, columns of $\tilde \Phi_s = \tilde U W$
are **not** eigenvectors of $\hat A$ corresponding to eigenvalues  $\Lambda$.

But in a quest for eigenvectors of $\hat A$ that we *can* compute with a reduced SVD,  let's define 

$$
\Phi \equiv \hat A \tilde \Phi_s = X' \tilde V \tilde \Sigma^{-1} W
$$

It turns out that columns of $\Phi$ **are** eigenvectors of $\hat A$,
 a consequence of a  result established by Tu et al. {cite}`tu_Rowley`.

To present their result, for convenience we'll drop the tilde $\tilde \cdot$ above $U, V,$ and $\Sigma$
and adopt the understanding that each of them is  computed with a reduced SVD.  


Thus, we now use the notation
that the  $m \times p$ matrix $\Phi$  is defined as

$$
  \Phi = X'   V  \Sigma^{-1} W
$$ (eq:Phiformula)


  
**Proposition** The $p$ columns of $\Phi$ are eigenvectors of $\check A$.

**Proof:** From formula {eq}`eq:Phiformula` we have

$$  
\begin{aligned}
  \hat A \Phi & =  (X' V \Sigma^{-1} U^T) (X' V \Sigma^{-1} W) \cr
  & = X' V \Sigma^{-1} \tilde A W \cr
  & = X' V \Sigma^{-1} W \Lambda \cr
  & = \Phi \Lambda 
  \end{aligned}
$$ 

Thus, we  have deduced  that

$$  
\hat A \Phi = \Phi \Lambda
$$ (eq:APhiLambda)

Let $\phi_i$ be the the $i$the column of $\Phi$ and $\lambda_i$ be the corresponding $i$ eigenvalue of $\tilde A$ from decomposition {eq}`eq:tildeAeigenred`. 

Writing out the $m \times 1$ vectors on both sides of  equation {eq}`eq:APhiLambda` and equating them gives


$$
\hat A \phi_i = \lambda_i \phi_i .
$$

Thus, $\phi_i$ is an eigenvector of $\hat A$ that corresponds to eigenvalue  $\lambda_i$ of $\tilde A$.

This concludes the proof. 

Also see {cite}`DDSE_book` (p. 238)


### Decoder of  $X$ as linear projection






From  eigendecomposition {eq}`eq:APhiLambda` we can represent $\hat A$ as 

$$ 
\hat A = \Phi \Lambda \Phi^+ .
$$ (eq:Aform12)


From formula {eq}`eq:Aform12` we can deduce the reduced dimension dynamics

$$ 
\check b_{t+1} = \Lambda \check b_t 
$$

where

$$
\check b_t  = \Phi^+ X_t  
$$ (eq:decoder102)


Since $\Phi$ has $p$ linearly independent columns, the generalized inverse of $\Phi$ is

$$
\Phi^{+} = (\Phi^T \Phi)^{-1} \Phi^T
$$

and so

$$ 
\check b = (\Phi^T \Phi)^{-1} \Phi^T X
$$ (eq:checkbform)

The matrix $\check b$  is recognizable as the  matrix of least squares regression coefficients of the matrix
$X$ on the matrix $\Phi$ and 

$$
\check X = \Phi \check b
$$ (eq:Xcheck_)

is the least squares projection of $X$ on $\Phi$.

 

By virtue of least-squares projection theory discussed here <https://python-advanced.quantecon.org/orth_proj.html>, 
we can represent $X$ as the sum of the projection $\check X$ of $X$ on $\Phi$  plus a matrix of errors.


To verify this, note that the least squares projection $\check X$ is related to $X$ by


$$ 
X = \check X + \epsilon 
$$

or

$$
X = \Phi \check b + \epsilon
$$

where $\epsilon$ is an $m \times n$ matrix of least squares errors satisfying the least squares
orthogonality conditions $\epsilon^T \Phi =0 $ or

$$ 
(X - \Phi \check b)^T \Phi = 0_{m \times p}
$$ (eq:orthls)

Rearranging  the orthogonality conditions {eq}`eq:orthls` gives $X^T \Phi = \check b \Phi^T \Phi$,
which implies formula {eq}`eq:checkbform`. 





### Alternative algorithm



There is a better way to compute the $p \times 1$ vector $\check b_t$ than provided by formula
{eq}`eq:decoder102`.

In particular, the following argument from {cite}`DDSE_book` (page 240) provides a computationally efficient way
to compute $\check b_t$.  

For convenience, we'll do this first for time $t=1$.



For $t=1$, we have  

$$ 
   X_1 = \Phi \check b_1
$$ (eq:X1proj)

where $\check b_1$ is an $r \times 1$ vector. 

Recall from representation 1 above that  $X_1 =  U \tilde b_1$, where $\tilde b_1$ is the time $1$  basis vector for representation 1.

It  then follows from equation {eq}`eq:Phiformula` that 
 
$$ 
  U \tilde b_1 = X' V \Sigma^{-1} W \check b_1
$$

and consequently

$$ 
  \tilde b_1 = U^T X' V \Sigma^{-1} W \check b_1
$$

Recall that  from equation {eq}`eq:AhatSVDformula`,  $ \tilde A = U^T X' V \Sigma^{-1}$.

It then follows  that
  
$$ 
  \tilde  b_1 = \tilde A W \check b_1
$$

and therefore, by the  eigendecomposition  {eq}`eq:tildeAeigen` of $\tilde A$, we have

$$ 
  \tilde b_1 = W \Lambda \check b_1
$$ 

Consequently, 
  
$$ 
  \check b_1 = ( W \Lambda)^{-1} \tilde b_1
$$ 

or 


$$ 
  \check b_1 = ( W \Lambda)^{-1} U^T X_1 ,
$$ (eq:beqnsmall)



which is  computationally more efficient than the following instance of  equation {eq}`eq:decoder102` for computing the initial vector $\check b_1$:

$$
  \check b_1= \Phi^{+} X_1
$$ (eq:bphieqn)


Users of  DMD sometimes call  components of the  basis vector $\check b_t  = \Phi^+ X_t \equiv (W \Lambda)^{-1} U^T X_t$  the  **exact** DMD modes.  

Conditional on $X_t$, we can compute our decoded $\check X_{t+j},   j = 1, 2, \ldots $  from 
either 

$$
\check X_{t+j} = \Phi \Lambda^j \Phi^{+} X_t
$$ (eq:checkXevoln)


or  

$$ 
  \check X_{t+j} = \Phi \Lambda^j (W \Lambda)^{-1}  U^T X_t .
$$ (eq:checkXevoln2)

We can then use $\check X_{t+j}$ to forcast $X_{t+j}$.



## Using Fewer Modes

Some of the preceding formulas assume that we have retained all $p$ modes associated with the positive
singular values of $X$.  

We can  adjust our  formulas to describe a situation in which we instead retain only
the $r < p$ largest singular values.  

In that case, we simply replace $\Sigma$ with the appropriate $p\times p$ matrix of singular values,
$U$ with the $m \times p$ matrix of whose columns correspond to the $r$ largest singular values,
and $V$ with the $n \times p$ matrix whose columns correspond to the $r$ largest  singular values.

Counterparts of all of the salient formulas above then apply.



## Source for Some Python Code

You can find a Python implementation of DMD here:

https://mathlab.github.io/PyDMD/
