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

# QR Decomposition

## Overview

This lecture describes the QR decomposition and how it relates to

 * Orthogonal projection and least squares

 * A Gram-Schmidt process

 * Eigenvalues and eigenvectors


We'll write some Python code to help consolidate our understandings.

## Matrix factorization

In this lecture, we consider a real matrix $A \in \mathbb{R}^{n \times m}$ with $m \geq n$.

A QR decomposition (also called a QR factorization) of $A$ takes the form

$$
A = QR, \qquad
Q \in \mathbb{R}^{n \times n}, \qquad
R \in \mathbb{R}^{n \times m}, \qquad
Q^\top Q = I_n.
$$

The matrix $R$ has zeros below its main diagonal.

Hence, $R$ is upper triangular when $m = n$ and upper trapezoidal when $m > n$.

We'll use a *Gram-Schmidt process* to compute a  QR decomposition.

Because doing so is so educational, we'll  write our own Python code to do the job.

## Gram-Schmidt process

We'll start with a *square* matrix $A$.

If a square matrix $A$ is nonsingular, then a QR factorization is unique up to signs.

We'll deal with a rectangular matrix $A$ later.

Actually, our algorithm will work with a rectangular $A$ that is not square.

### Gram-Schmidt process for square $A$

Here we apply a Gram-Schmidt  process to the  *columns*  of matrix $A$.

In particular, let

$$
A= \begin{bmatrix} a_1 & a_2 & \cdots & a_n \end{bmatrix}
$$

Let $|| \cdot ||$ denote the L2 norm.

The Gram-Schmidt algorithm repeatedly combines the following  two steps in a particular order

*  **normalize** a vector to have unit norm

*  **orthogonalize** the next vector

To begin, we set $u_1 = a_1$ and then *normalize*:

$$
u_1=a_1, \ \ \ e_1=\frac{u_1}{||u_1||}
$$

We *orthogonalize* first to compute $u_2$ and then *normalize* to create $e_2$:

$$
u_2=a_2-(a_2\cdot e_1)e_1, \ \ \  e_2=\frac{u_2}{||u_2||}
$$

We invite the reader to verify that $e_1$ is orthogonal to $e_2$ by checking that
$e_1 \cdot e_2 = 0$.

The Gram-Schmidt procedure continues iterating.

Thus,  for $k= 2, \ldots, n-1$ we construct

$$
u_{k+1}=a_{k+1}-(a_{k+1}\cdot e_1)e_1-\cdots-(a_{k+1}\cdot e_k)e_k, \ \ \ e_{k+1}=\frac{u_{k+1}}{||u_{k+1}||}
$$


Here $(a_j \cdot e_i)$ can be interpreted as the linear least squares *regression coefficient* of $a_j$ on $e_i$

* it is the inner product of $a_j$ and $e_i$ divided by the inner product of $e_i$ where 
    $e_i \cdot e_i = 1$, as *normalization* has assured us.
    
* this regression coefficient has an interpretation as being  a *covariance* divided by a *variance*
   

It can  be verified that

$$
A= \begin{bmatrix} a_1 & a_2 & \cdots & a_n \end{bmatrix}=
\begin{bmatrix} e_1 & e_2 & \cdots & e_n \end{bmatrix}
\begin{bmatrix} a_1\cdot e_1 & a_2\cdot e_1 & \cdots & a_n\cdot e_1\\ 0 & a_2\cdot e_2 & \cdots & a_n\cdot e_2
\\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & a_n\cdot e_n \end{bmatrix}
$$

Thus, we have constructed the decomposition

$$ 
A = Q R
$$

where 

$$ 
Q = \begin{bmatrix} e_1 & e_2 & \cdots & e_n \end{bmatrix}
$$

and 

$$
R = \begin{bmatrix} a_1\cdot e_1 & a_2\cdot e_1 & \cdots & a_n\cdot e_1\\ 0 & a_2\cdot e_2 & \cdots & a_n\cdot e_2
\\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & a_n\cdot e_n \end{bmatrix}
$$

### $A$ not square 

Now suppose that $A$ is an $n \times m$ matrix where $m > n$.  

Then a QR decomposition is

$$
A= \begin{bmatrix} a_1 & a_2 & \cdots & a_m \end{bmatrix}=\begin{bmatrix} e_1 & e_2 & \cdots & e_n \end{bmatrix}
\begin{bmatrix} a_1\cdot e_1 & a_2\cdot e_1 & \cdots & a_n\cdot e_1 & a_{n+1}\cdot e_1 & \cdots & a_{m}\cdot e_1 \\
0 & a_2\cdot e_2 & \cdots & a_n\cdot e_2 & a_{n+1}\cdot e_2 & \cdots & a_{m}\cdot e_2 \\ \vdots & \vdots & \ddots & \quad  \vdots & \vdots & \ddots & \vdots
\\ 0 & 0 & \cdots & a_n\cdot e_n & a_{n+1}\cdot e_n & \cdots & a_{m}\cdot e_n \end{bmatrix}
$$

Here, $R$ is an $n \times m$ upper trapezoidal matrix.

This implies that

\begin{align*}
a_1 & = (a_1\cdot e_1) e_1 \cr
a_2 & = (a_2\cdot e_1) e_1 + (a_2\cdot e_2) e_2 \cr
\vdots & \quad \vdots \cr
a_n & = (a_n\cdot e_1) e_1 + (a_n\cdot e_2) e_2 + \cdots + (a_n \cdot e_n) e_n  \cr
a_{n+1} & = (a_{n+1}\cdot e_1) e_1 + (a_{n+1}\cdot e_2) e_2 + \cdots + (a_{n+1}\cdot e_n) e_n  \cr
\vdots & \quad \vdots \cr
a_m & = (a_m\cdot e_1) e_1 + (a_m\cdot e_2) e_2 + \cdots + (a_m \cdot e_n) e_n  \cr
\end{align*}

## Some code

Now let's write some homemade Python code to implement a QR decomposition by deploying the  Gram-Schmidt process described above.

```{code-cell} ipython3
import numpy as np
from scipy.linalg import qr

rng = np.random.default_rng(1234)
```

The `qr_decomposition` function below illustrates the square case and its extension to wide $n \times m$ matrices with $m \geq n$, assuming that the first $n$ columns are linearly independent.

```{code-cell} ipython3
def qr_decomposition(A):
    n, m = A.shape # get the shape of A

    Q = np.empty((n, n)) # initialize matrix Q
    u = np.empty((n, n)) # initialize matrix u

    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

    for i in range(1, n):

        u[:, i] = A[:, i]
        for j in range(i):
            u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j] # get each u vector

        Q[:, i] = u[:, i] / np.linalg.norm(u[:, i]) # compute each e vector

    R = np.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i]

    return Q, R
```

The preceding code is fine but can benefit from some further housekeeping.

We want to do this because later in this notebook we want to compare results from using our homemade code above with the code for a QR that the Python SciPy package delivers.

There can be sign differences between the $Q$ and $R$ matrices produced by different numerical algorithms.

All of these are valid QR decompositions because of how the  sign differences cancel out when we compute $QR$.

To make the results from our homemade function and the QR module in SciPy comparable, we'll require the diagonal entries of $R$ to be nonnegative.

We do this by adjusting the signs of the columns in $Q$ and the corresponding rows in $R$ appropriately.

To accomplish this we'll define a pair of functions.

```{code-cell} ipython3
def diag_sign(A):
    """Compute a diagonal sign matrix, mapping zero signs to one."""

    signs = np.where(np.diag(A) < 0, -1.0, 1.0)
    D = np.diag(signs)

    return D

def adjust_sign(Q, R):
    """
    Adjust the signs of the columns of Q and rows of R
    so that R has a nonnegative diagonal.
    """

    D = diag_sign(R)

    Q[:, :] = Q @ D
    R[:, :] = D @ R

    return Q, R
```

## Example

Now let's do an example.

```{code-cell} ipython3
A = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
# A = np.array([[1.0, 0.5, 0.2], [0.5, 0.5, 1.0], [0.0, 1.0, 1.0]])
# A = np.array([[1.0, 0.5, 0.2], [0.5, 0.5, 1.0]])

A
```

```{code-cell} ipython3
Q, R = adjust_sign(*qr_decomposition(A))
```

```{code-cell} ipython3
Q
```

```{code-cell} ipython3
R
```

Let's compare outcomes  with what the SciPy package produces

```{code-cell} ipython3
Q_scipy, R_scipy = adjust_sign(*qr(A))
```

```{code-cell} ipython3
print('Our Q: \n', Q)
print('\n')
print('SciPy Q: \n', Q_scipy)
```

```{code-cell} ipython3
print('Our R: \n', R)
print('\n')
print('SciPy R: \n', R_scipy)
```

The above outcomes give us the good news that our homemade function agrees with what
SciPy produces.


Now let's do a QR decomposition for a rectangular matrix $A$ that is $n \times m$ with 
$m > n$.

```{code-cell} ipython3
A = np.array([[1, 3, 4], [2, 0, 9]])
```

```{code-cell} ipython3
Q, R = adjust_sign(*qr_decomposition(A))
Q, R
```

```{code-cell} ipython3
Q_scipy, R_scipy = adjust_sign(*qr(A))
Q_scipy, R_scipy
```

## Using QR decomposition to compute eigenvalues

Now for a useful  fact about the QR algorithm.  

The following iterations on the QR decomposition can be used to compute *eigenvalues*
of a *square* matrix $A$.

Here is the algorithm:

1. Set $A_0 = A$ and form $A_0 = Q_0 R_0$

2. Form $A_1 = R_0 Q_0 $ . Note that $A_1$ is similar to $A_0$ (easy to verify) and so has the same eigenvalues.

3. Form $A_1 = Q_1 R_1$ (i.e., form the QR decomposition of $A_1$).

4. Form $ A_2 = R_1 Q_1 $ and then $A_2 = Q_2 R_2$  .

5. Iterate to convergence.

6. Compute eigenvalues of $A$ and compare them to the diagonal values of the limiting $A_n$ found from this process.

```{prf:remark}
This algorithm is close to one of the most efficient ways of computing eigenvalues!
```

Let's write some Python code to try out the algorithm

```{code-cell} ipython3
def qr_eigvals(A, tol=1e-12, maxiter=1000):
    "Find the eigenvalues of A using QR decomposition."

    A_old = np.copy(A)
    A_new = np.copy(A)

    diff = np.inf
    i = 0
    while (diff > tol) and (i < maxiter):
        A_old[:, :] = A_new
        Q, R = qr_decomposition(A_old)

        A_new[:, :] = R @ Q

        diff = np.abs(A_new - A_old).max()
        i += 1

    eigvals = np.diag(A_new)

    return eigvals
```

Now let's try the code and compare the results with `np.linalg.eigvals`.

Here goes

```{code-cell} ipython3
# experiment this with one random A matrix
A = rng.random((3, 3))
```

```{code-cell} ipython3
sorted(qr_eigvals(A))
```

Compare with NumPy's implementation.

```{code-cell} ipython3
sorted(np.linalg.eigvals(A))
```

## QR and PCA

There are interesting connections between the QR decomposition and principal components analysis (PCA).

Here are  some.

1.  Let $X^\top$ be a $k \times n$ random matrix where the $j$th column is a random draw
from $N(\mu, \Sigma)$ where $\mu$ is $k \times 1$ vector of means and $\Sigma$ is a $k \times k$
covariance matrix.  We want $n \gg k$ -- this is an "econometrics example".

2. Form $X^\top = Q R $ where $Q $ is $k \times k$ and $R$ is $k \times n$.

3. Form the eigenvalues of $ R R^\top$, i.e., we'll compute $R R^\top = \tilde P \Lambda \tilde P^\top $.

4. Form $X^\top X = Q \tilde P \Lambda \tilde P^\top Q^\top$ and compare it with the eigen decomposition
$ X^\top X = P \hat \Lambda P^\top$.

5. It will turn out that $\Lambda = \hat \Lambda$ and that $P = Q \tilde P$.


Let's verify conjecture 5 with some Python code.

Start by simulating a random $\left(n, k\right)$ matrix $X$.

```{code-cell} ipython3
k = 5
n = 1000

# generate some random moments
𝜇 = rng.random(size=k)
C = rng.random((k, k))
Σ = C.T @ C
```

```{code-cell} ipython3
# Each row of X is a draw from the multivariate normal distribution
X = rng.multivariate_normal(𝜇, Σ, size=n)
```

```{code-cell} ipython3
X.shape
```

Let's apply the QR decomposition to $X^\top$.

```{code-cell} ipython3
Q, R = adjust_sign(*qr_decomposition(X.T))
```

Check the shapes of $Q$ and $R$.

```{code-cell} ipython3
Q.shape, R.shape
```

Now we can construct $R R^\top=\tilde{P} \Lambda \tilde{P}^\top$ and form an eigen decomposition.

```{code-cell} ipython3
RR = R @ R.T

𝜆, P_tilde = np.linalg.eigh(RR)
Λ = np.diag(𝜆)
```

We can also apply the decomposition to $X^\top X=P \hat{\Lambda} P^\top$.

```{code-cell} ipython3
XX = X.T @ X

𝜆_hat, P = np.linalg.eigh(XX)
Λ_hat = np.diag(𝜆_hat)
```

Compare the eigenvalues that are on the diagonals of $\Lambda$ and $\hat{\Lambda}$.

```{code-cell} ipython3
𝜆, 𝜆_hat
```

Let's compare $P$ and $Q \tilde{P}$. 

Again we need to be careful about sign differences between the columns of $P$ and $Q\tilde{P}$. 

```{code-cell} ipython3
QP_tilde = Q @ P_tilde

np.abs(P @ diag_sign(P) - QP_tilde @ diag_sign(QP_tilde)).max()
```

Let's verify that $X^\top X$ can be decomposed as $Q \tilde{P} \Lambda \tilde{P}^\top Q^\top$.

```{code-cell} ipython3
QPΛPQ = Q @ P_tilde @ Λ @ P_tilde.T @ Q.T
```

```{code-cell} ipython3
np.abs(QPΛPQ - XX).max()
```
