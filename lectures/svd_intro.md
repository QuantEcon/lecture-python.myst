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



We turn to the case in which $ m >>n $ in which an $ m \times n $  data matrix $ \tilde X $ contains many more random variables $ m $ than observations $ n $.

This  **tall and skinny** case is associated with **Dynamic Mode Decomposition**.

You can read about Dynamic Mode Decomposition here [[KBBWP16](https://python.quantecon.org/zreferences.html#id24)] and here [[BK19](https://python.quantecon.org/zreferences.html#id25)] (section 7.2).


We want to fit a **first-order vector autoregression**

$$
X_{t+1} = A X_t + C \epsilon_{t+1}
$$ (eq:VARfirstorder)

where 
the $ m \times 1 $ vector $ X_t $ is

$$
X_t = \begin{bmatrix}  X_{1,t} & X_{2,t} & \cdots & X_{m,t}     \end{bmatrix}^T
$$ (eq:Xvector)

and where $ T $ again denotes complex transposition and $ X_{i,t} $ is an observation on variable $ i $ at time $ t $.



We want to fit equation {eq}`eq:VARfirstorder`. 


Our data is assembled in the form of  an $ m \times n $ matrix  $ \tilde X $ 

$$
\tilde X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_n\end{bmatrix}
$$

where for $ t = 1, \ldots, n $,  the $ m \times 1 $ vector $ X_t $ is given by {eq}`eq:Xvector`. 

We want to estimate system  {eq}`eq:VARfirstorder` consisting of $ m $ least squares regressions of **everything** on one lagged value of **everything**.




We proceed as follows. 


From $ \tilde X $,  we  form two matrices

$$
X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_{n-1}\end{bmatrix}
$$

and

$$
X' =  \begin{bmatrix} X_2 \mid X_3 \mid \cdots \mid X_n\end{bmatrix}
$$

Here $ ' $ does not denote matrix transposition but instead is part of the name of the matrix $ X' $.

In forming $ X $ and $ X' $, we have in each case  dropped a column from $ \tilde X $,  the last column in the case of $ X $, and  the first column in the case of $ X' $.

Evidently, $ X $ and $ X' $ are both $ m \times \tilde n $ matrices where $ \tilde n = n - 1 $.

We denote the rank of $ X $ as $ p \leq \min(m, \tilde n)  $.

Two possible cases are when

 *  $ \tilde n > > m$, so that we have many more time series  observations $\tilde n$ than variables $m$
 *  $m > > \tilde n$, so that we have many more variables $m $ than time series observations $\tilde n$

At a general level that includes both of these special cases, a common formula describes the least squares estimator $\hat A$ of $A$ for both cases, but important  details differ.

The common formula is

$$ \hat A = X' X^+ $$

where $X^+$ is the pseudo-inverse of $X$.

Formulas for the pseudo-inverse differ for our two cases.

When $ \tilde n > > m$, so that we have many more time series  observations $\tilde n$ than variables $m$ and when
$X$ has linearly independent **rows**, $X X^T$ has an inverse and the pseudo-inverse $X^+$ is

$$
X^+ = X^T (X X^T)^{-1} 
$$

Here $X^+$ is a **right-inverse** that verifies $ X X^+ = I_{m \times m}$.

In this case, our formula for the least-squares estimator of $A$ becomes

$$ 
\hat A = X' X^T (X X^T)^{-1}
$$

This is formula is widely used in economics to estimate vector autorgressions.   

The left side is proportional to the empirical cross second moment matrix of $X_{t+1}$ and $X_t$ times the inverse
of the second moment matrix of $X_t$, the least-squares formula widely used in econometrics.



When $m > > \tilde n$, so that we have many more variables $m $ than time series observations $\tilde n$ and when $X$ has linearly independent **columns**,
$X^T X$ has an inverse and the pseudo-inverse $X^+$ is

$$
X^+ = (X^T X)^{-1} X^T
$$

Here  $X^+$ is a **left-inverse** that verifies $X^+ X = I_{\tilde n \times \tilde n}$.

In this case, our formula for a least-squares estimator of $A$ becomes

$$
\hat A = X' (X^T X)^{-1} X^T
$$ (eq:hatAversion0)

This is the case that we are interested in here. 


Thus, we want to fit equation {eq}`eq:VARfirstorder` in a situation in which we have a number $n$ of observations  that is small relative to the number $m$ of
variables that appear in the vector $X_t$.

We'll use  efficient algorithms for computing and for constructing reduced rank approximations of  $\hat A$ in formula {eq}`eq:hatAversion0`.
 




To reiterate and supply more  detail about how we can efficiently calculate the pseudo-inverse $X^+$, as our  estimator $\hat A$ of $A$ we form an  $m \times m$ matrix that  solves the least-squares best-fit problem

$$ 
\hat A = \textrm{argmin}_{\check A} || X' - \check  A X ||_F   
$$ (eq:ALSeqn)

where $|| \cdot ||_F$ denotes the Frobeneus norm of a matrix.

The solution of the problem on the right side of equation {eq}`eq:ALSeqn` is

$$
\hat A =  X'  X^{+} . 
$$ (eq:hatAform)

Here the (possibly huge) $ \tilde n \times m $ matrix $ X^{+} $ is the pseudo-inverse of $ X $.

The $ i $th  row of $ \hat A $ is an $ m \times 1 $ vector of pseudo-regression coefficients of $ X_{i,t+1} $ on $ X_{j,t}, j = 1, \ldots, m $.

An efficient way to compute the pseudo-inverse $X^+$ is to start with  the (reduced) singular value decomposition



$$
X =  U \Sigma  V^T 
$$ (eq:SVDDMD)

where $ U $ is $ m \times p $, $ \Sigma $ is a $ p \times p $ diagonal  matrix, and $ V^T $ is a $ p \times \tilde n $ matrix.

Here $ p $ is the rank of $ X $, where necessarily $ p \leq \tilde n $.

(We  described and illustrated a **reduced** singular value decomposition above, and compared it with a **full** singular value decomposition.)

We can construct a pseudo-inverse $ X^+ $  of $ X $ by using
a singular value decomposition of $X$ in equation {eq}`eq:SVDDMD`  to compute


$$
X^{+} =  V \Sigma^{-1}  U^T 
$$ (eq:Xplusformula)

where the matrix $ \Sigma^{-1} $ is constructed by replacing each non-zero element of $ \Sigma $ with $ \sigma_j^{-1} $.

We can  use formula {eq}`eq:Xplusformula`   together with formula {eq}`eq:hatAform` to compute the matrix  $ \hat A $ of regression coefficients.

Thus, our  estimator $\hat A = X' X^+$ of the $m \times m$ matrix of coefficients $A$    is

$$
\hat A = X' V \Sigma^{-1}  U^T 
$$

In addition to doing that, we’ll eventually use **dynamic mode decomposition** to compute a rank $ r $ approximation to $ A $,
where $ r <  p $.
  


Next, we turn to two alternative __reduced order__ representations of our dynamic system.

+++

## Representation 1

We use the $p$  columns of $U$, and thus the $p$ rows of $U^T$,  to define   a $p \times 1$  vector $\tilde X_t$ as follows


$$
\tilde b_t = U^T X_t 
$$ (eq:tildeXdef2)

and 

$$ 
X_t - U \tilde b_t
$$ (eq:Xdecoder)

(Here we use $b$ to remind us that we are creating a **basis** vector.)

Since $U U^T$ is an $m \times m$ identity matrix, it follows from equation {eq}`eq:tildeXdef2` that we can reconstruct  $X_t$ from $\tilde b_t$ by using 



 * Equation {eq}`eq:tildeXdef2` serves as an **encoder** that  summarizes the $m \times 1$ vector $X_t$ by a $p \times 1$ vector $\tilde b_t$ 
  
 * Equation {eq}`eq:Xdecoder` serves as a **decoder** that recovers the $m \times 1$ vector $X_t$ from the $p \times 1$ vector $\tilde b_t$ 



Define the reduced transition matrix 

$$ 
\tilde A = U^T \hat A U 
$$

We can evidently recover $A$ from

$$
\hat A = U \tilde A U^T 
$$

Dynamics of the reduced $p \times 1$ state $\tilde b_t$ are governed by

$$
\tilde b_{t+1} = \tilde A \tilde b_t 
$$

To construct forecasts $\overline X_t$ of  future values of $X_t$ conditional on $X_1$, we can apply  decoders to both sides of this 
equation and deduce

$$
\overline X_{t+1} = U \tilde A^t U^T X_1
$$

where we use $\overline X_t$ to denote a forecast.

+++

## Representation 2

Form an eigendecomposition of $\tilde A$:

$$
\tilde A = W \Lambda W^{-1} 
$$

where $\Lambda$ is a diagonal matrix of eigenvalues and $W$ is a $p \times p$
matrix whose columns are eigenvectors  corresponding to rows (eigenvalues) in 
$\Lambda$.

Note that

$$ 
A = U \tilde U^T = U W \Lambda W^{-1} U^T 
$$

Thus, the systematic part of the $X_t$ dynamics captured by our first-order vector autoregressions   are described by

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

where now our endoder is

$$ 
\hat b_t = W^{-1} U^T X_t
$$

and our decoder is

$$
X_t = U W \hat b_t
$$

We can use this representation to predict future $X_t$'s via: 

$$
\overline X_{t+1} = U W \Lambda^t W^{-1} U^T X_1 
$$



## Using Fewer Modes

The preceding formulas assume that we have retained all $p$ modes associated with the positive
singular values of $X$.  

We can easily adapt all of the formulas to describe a situation in which we instead retain only
the $r < p$ largest singular values.  

In that case, we simply replace $\Sigma$ with the appropriate $r \times r$ matrix of singular values,
$U$ with the $m \times r$ matrix of whose columns correspond to the $r$ largest singular values,
and $V$ with the $\tilde n \times r$ matrix whose columns correspond to the $r$ largest  singular values.

Counterparts of all of the salient formulas above then apply.



## Source for Some Python Code

You can find a Python implementation of DMD here:

https://mathlab.github.io/PyDMD/
