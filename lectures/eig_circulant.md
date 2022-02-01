---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Circulant Matrices

## Overview

This lecture describes circulant matrices and some of their properties.

Circulant matrices have a special structure that connects them to  useful concepts
including

  * convolution
  * Fourier transforms
  * permutation matrices

Because of these connections, circulant matrices are widely used  in machine learning, for example, in image processing.


We begin by importing some Python packages

```{code-cell} ipython3
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
%matplotlib inline
```

```{code-cell} ipython3
np.set_printoptions(precision=3, suppress=True)
```

## Constructing a Circulant Matrix

To construct an $N \times N$ circulant matrix, we  need only the first row, say,  

$$ \begin{bmatrix} c_{0} & c_{1} & c_{2} & c_{3} & c_{4} & \cdots & c_{N-1} \end{bmatrix} .$$

After setting entries in the first row, the remaining rows of a circulant matrix are determined as
follows:

$$
C=\left[\begin{array}{ccccccc}
c_{0} & c_{1} & c_{2} & c_{3} & c_{4} & \cdots & c_{N-1}\\
c_{N-1} & c_{0} & c_{1} & c_{2} & c_{3} & \cdots & c_{N-2}\\
c_{N-2} & c_{N-1} & c_{0} & c_{1} & c_{2} & \cdots & c_{N-3}\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
c_{3} & c_{4} & c_{5} & c_{6} & c_{7} & \cdots & c_{2}\\
c_{2} & c_{3} & c_{4} & c_{5} & c_{6} & \cdots & c_{1}\\
c_{1} & c_{2} & c_{3} & c_{4} & c_{5} & \cdots & c_{0}
\end{array}\right]
$$ (eqn:circulant)

It is also possible to construct a circulant matrix by creating the transpose of the above matrix, in which case only the
first column needs to be specified.

Let's write some Python code to generate a circulant matrix.

```{code-cell} ipython3
@njit
def construct_cirlulant(row):

    N = row.size
    
    C = np.empty((N, N))

    for i in range(N):

        C[i, i:] = row[:N-i]
        C[i, :i] = row[N-i:]

    return C
```

```{code-cell} ipython3
# a simple case when N = 3
construct_cirlulant(np.array([1., 2., 3.]))
```

### Some Properties of Circulant Matrices

Here are some useful properties:

Suppose that $A$ and $B$ are both circulant matrices. Then it can be verified that

 * The transpose of a circulant matrix is a circulant matrix.


 
  * $A + B$ is a circulant matrix
  * $A B$ is a circulant matrix
  * $A B = B A$ 

Now consider a circulant matrix with first row 

  $$  c = \begin{bmatrix} c_0 & c_1 & \cdots & c_{N-1} \end{bmatrix} $$

 and consider a vector 

 $$ a = \begin{bmatrix} a_0 & a_1 & \cdots  &  a_{N-1} \end{bmatrix} $$

 The **convolution** of  vectors $c$ and $a$ is defined   as the vector $b = c * a $  with components

$$
 b_k = \sum_{i=0}^{n-1} c_{k-i} a_i  
$$ (eqn:conv)

We use $*$ to denote **convolution** via the calculation described in equation {eq}`eqn:conv`.

It can be verified that the vector $b$ satisfies

$$ b = C^T a  $$

where $C^T$ is the transpose of the circulant matrix  defined in equation {eq}`eqn:circulant`.  





## Connection to Permutation Matrix

A good way to construct a circulant matrix is to use a **permutation matrix**.

Before defining a permutation **matrix**, we'll define a **permutation**.

A **permutation** of a set of the set of non-negative integers $\{0, 1, 2, \ldots \}$ is a one-to-one mapping of the set into itself.

A permutation of a set $\{1, 2, \ldots, n\}$ rearranges the $n$ integers in the set.  


A [permutation matrix](https://mathworld.wolfram.com/PermutationMatrix.html) is obtained by permuting the rows of an $n \times n$ identity matrix according to a permutation of the numbers $1$ to $n$. 


Thus, every row and every column contain precisely a single $1$ with $0$ everywhere else.

Every permutation corresponds to a unique permutation matrix.

For example, the $N \times N$ matrix

$$
P=\left[\begin{array}{cccccc}
0 & 1 & 0 & 0 & \cdots & 0\\
0 & 0 & 1 & 0 & \cdots & 0\\
0 & 0 & 0 & 1 & \cdots & 0\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
0 & 0 & 0 & 0 & \cdots & 1\\
1 & 0 & 0 & 0 & \cdots & 0
\end{array}\right]
$$ (eqn:exampleP)

serves as  a **cyclic shift**  operator that, when applied to an $N \times 1$ vector $h$, shifts entries in rows $2$ through $N$ up one row and shifts the entry in row $1$ to row $N$. 


Eigenvalues of  the cyclic shift permutation matrix $P$ defined in equation {eq}`eqn:exampleP` can be computed  by constructing

$$
P-\lambda I=\left[\begin{array}{cccccc}
-\lambda & 1 & 0 & 0 & \cdots & 0\\
0 & -\lambda & 1 & 0 & \cdots & 0\\
0 & 0 & -\lambda & 1 & \cdots & 0\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
0 & 0 & 0 & 0 & \cdots & 1\\
1 & 0 & 0 & 0 & \cdots & -\lambda
\end{array}\right]
$$

and solving 

$$
\textrm{det}(P - \lambda I) = (-1)^N \lambda^{N}-1=0
$$


Eigenvalues $\lambda_i$  can be complex.

Magnitudes $\mid \lambda_i \mid$  of these  eigenvalues $\lambda_i$ all equal  $1$.

Thus, **singular values** of the  permutation matrix $P$ defined in equation {eq}`eqn:exampleP` all equal $1$.

It can be verified that permutation matrices are orthogonal matrices:

$$
P P' = I 
$$




## Examples with Python

Let's write some Python code to illustrate these ideas.

```{code-cell} ipython3
@njit
def construct_P(N):

    P = np.zeros((N, N))

    for i in range(N-1):
        P[i, i+1] = 1
    P[-1, 0] = 1

    return P
```

```{code-cell} ipython3
P4 = construct_P(4)
P4
```

```{code-cell} ipython3
# compute the eigenvalues and eigenvectors
𝜆, Q = np.linalg.eig(P4)
```

```{code-cell} ipython3
for i in range(4):
    print(f'𝜆{i} = {𝜆[i]:.1f} \nvec{i} = {Q[i, :]}\n')
```

In graphs  below, we shall portray eigenvalues of a shift  permutation matrix   in the complex plane. 

These eigenvalues are uniformly distributed along the unit circle.

They are the **$n$ roots of unity**, meaning they are the $n$  numbers  $z$  that solve $z^n =1$, where $z$ is a complex number.

In particular, the $n$ roots of unity are

$$
z = \exp\left(\frac{2 \pi j k }{N} \right) , \quad k = 0, \ldots, N-1
$$

where $j$ denotes the purely imaginary unit number.

```{code-cell} ipython3
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

for i, N in enumerate([3, 4, 6, 8]):

    row_i = i // 2
    col_i = i % 2

    P = construct_P(N)
    𝜆, Q = np.linalg.eig(P)

    circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
    ax[row_i, col_i].add_patch(circ)

    for j in range(N):
        ax[row_i, col_i].scatter(𝜆[j].real, 𝜆[j].imag, c='b')

    ax[row_i, col_i].set_title(f'N = {N}')
    ax[row_i, col_i].set_xlabel('real')
    ax[row_i, col_i].set_ylabel('imaginary')

plt.show()
```
For a vector of  coefficients $\{c_i\}_{i=0}^{n-1}$, eigenvectors of $P$ are also  eigenvectors of 

$$
C = c_{0} I + c_{1} P + c_{2} P^{2} +\cdots + c_{N-1} P^{N-1}.
$$

Consider an example in which  $N=8$ and let $w = e^{-2 \pi j / N}$.

It can be verified that the matrix $F_8$ of eigenvectors of $P_{8}$  is

$$
F_{8}=\left[\begin{array}{ccccc}
1 & 1 & 1 & \cdots & 1\\
1 & w & w^{2} & \cdots & w^{7}\\
1 & w^{2} & w^{4} & \cdots & w^{14}\\
1 & w^{3} & w^{6} & \cdots & w^{21}\\
1 & w^{4} & w^{8} & \cdots & w^{28}\\
1 & w^{5} & w^{10} & \cdots & w^{35}\\
1 & w^{6} & w^{12} & \cdots & w^{42}\\
1 & w^{7} & w^{14} & \cdots & w^{49}
\end{array}\right]
$$

The matrix $F_8$ defines a  [Discete Fourier Transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform).

To convert it into an orthogonal eigenvector matrix, we can simply normalize it by dividing every entry  by $\sqrt{8}$. 

 *  stare at the first column of $F_8$ above to convince yourself of this fact 

The eigenvalues corresponding to each eigenvector are $\{w^{j}\}_{j=0}^{7}$ in order.

```{code-cell} ipython3
def construct_F(N):

    w = np.e ** (-np.complex(0, 2*np.pi/N))

    F = np.ones((N, N), dtype=np.complex)
    for i in range(1, N):
        F[i, 1:] = w ** (i * np.arange(1, N))

    return F, w
```

```{code-cell} ipython3
F8, w = construct_F(8)
```

```{code-cell} ipython3
w
```

```{code-cell} ipython3
F8
```

```{code-cell} ipython3
# normalize
Q8 = F8 / np.sqrt(8)
```

```{code-cell} ipython3
# verify the orthogonality (unitarity)
Q8 @ np.conjugate(Q8)
```

Let's verify that $k$th column of $Q_{8}$ is an eigenvector of $P_{8}$ with an eigenvalue $w^{k}$.

```{code-cell} ipython3
P8 = construct_P(8)
```

```{code-cell} ipython3
diff_arr = np.empty(8, dtype=np.complex)
for j in range(8):
    diff = P8 @ Q8[:, j] - w ** j * Q8[:, j]
    diff_arr[j] = diff @ diff.T
```

```{code-cell} ipython3
diff_arr
```

## Associated Permutation Matrix 


Next, we execute calculations to verify that the circulant matrix $C$ defined  in equation {eq}`eqn:circulant` can be written as 


$$
C = c_{0} I + c_{1} P + \cdots + c_{n-1} P^{n-1}
$$

and that every eigenvector of $P$ is also an eigenvector of $C$.

```{code-cell} ipython3

```

We illustrate this for $N=8$ case.

```{code-cell} ipython3
c = np.random.random(8)
```

```{code-cell} ipython3
c
```

```{code-cell} ipython3
C8 = construct_cirlulant(c)
```

Compute $c_{0} I + c_{1} P + \cdots + c_{n-1} P^{n-1}$.

```{code-cell} ipython3
N = 8

C = np.zeros((N, N))
P = np.eye(N)

for i in range(N):
    C += c[i] * P
    P = P8 @ P
```

```{code-cell} ipython3
C
```

```{code-cell} ipython3
C8
```

Now let's compute the difference between two circulant matrices that we have  constructed in two different ways.

```{code-cell} ipython3
np.abs(C - C8).max()
```

The  $k$th column of $P_{8}$ associated with eigenvalue $w^{k-1}$ is an eigenvector of $C_{8}$ associated with an eigenvalue $\sum_{h=0}^{7} c_{j} w^{h k}$.

```{code-cell} ipython3
𝜆_C8 = np.zeros(8, dtype=np.complex)

for j in range(8):
    for k in range(8):
        𝜆_C8[j] += c[k] * w ** (j * k)
```

```{code-cell} ipython3
𝜆_C8
```

We can verify this by comparing `C8 @ Q8[:, j]` with `𝜆_C8[j] * Q8[:, j]`.

```{code-cell} ipython3
# verify
for j in range(8):
    diff = C8 @ Q8[:, j] - 𝜆_C8[j] * Q8[:, j]
    print(diff)
```

## Discrete Fourier Transform

The **Discrete Fourier Transform** (DFT) allows us to  represent a  discrete time sequence as a weighted sum of complex sinusoids.

Consider a sequence of $N$ real number $\{x_j\}_{j=0}^{N-1}$. 

The **Discrete Fourier Transform** maps $\{x_j\}_{j=0}^{N-1}$ into a sequence of complex numbers $\{X_k\}_{k=0}^{N-1}$

where

$$
X_{k}=\sum_{n=0}^{N-1}x_{n}e^{-2\pi\frac{kn}{N}i}
$$

```{code-cell} ipython3
def DFT(x):
    "The discrete Fourier transform."

    N = len(x)
    w = np.e ** (-np.complex(0, 2*np.pi/N))

    X = np.zeros(N, dtype=np.complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * w ** (k * n)

    return X
```

Consider the following example.

$$
x_{n}=\begin{cases}
1/2 & n=0,1\\
0 & \text{otherwise}
\end{cases}
$$

```{code-cell} ipython3
x = np.zeros(10)
x[0:2] = 1/2
```

```{code-cell} ipython3
x
```

Apply a discrete fourier transform.

```{code-cell} ipython3
X = DFT(x)
```

```{code-cell} ipython3
X
```

We can plot  magnitudes of a sequence of numbers and the  associated discrete Fourier transform.

```{code-cell} ipython3
def plot_magnitude(x=None, X=None):

    data = []
    names = []
    xs = []
    if (x is not None):
        data.append(x)
        names.append('x')
        xs.append('n')
    if (X is not None):
        data.append(X)
        names.append('X')
        xs.append('j')

    num = len(data)
    for i in range(num):
        n = data[i].size
        plt.figure(figsize=(8, 3))
        plt.scatter(range(n), np.abs(data[i]))
        plt.vlines(range(n), 0, np.abs(data[i]), color='b')

        plt.xlabel(xs[i])
        plt.ylabel('magnitude')
        plt.title(names[i])
        plt.show()
```

```{code-cell} ipython3
plot_magnitude(x=x, X=X)
```

The **inverse Fourier transform**  transforms a Fourier transform  $X$ of $x$  back to $x$.

The inverse Fourier transform is defined as

$$
x_{n} = \sum_{k=0}^{N-1} \frac{1}{N} X_{k} e^{2\pi\left(\frac{kn}{N}\right)i}, \quad n=0, 1, \ldots, N-1
$$

```{code-cell} ipython3
def inverse_transform(X):

    N = len(X)
    w = np.e ** (np.complex(0, 2*np.pi/N))

    x = np.zeros(N, dtype=np.complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * w ** (k * n) / N

    return x
```

```{code-cell} ipython3
inverse_transform(X)
```

Another example is

$$
x_{n}=2\cos\left(2\pi\frac{11}{40}n\right),\ n=0,1,2,\cdots19
$$

Since $N=20$, we cannot use an integer multiple of $\frac{1}{20}$ to represent a frequency $\frac{11}{40}$.

To handle this,  we shall end up using all $N$ of the availble   frequencies in the DFT.

Since $\frac{11}{40}$ is in between $\frac{10}{40}$ and $\frac{12}{40}$ (each of which is an integer multiple of $\frac{1}{20}$), the complex coefficients in the DFT   have their  largest magnitudes at $k=5,6,15,16$, not just at a single frequency.

```{code-cell} ipython3
N = 20
x = np.empty(N)

for j in range(N):
    x[j] = 2 * np.cos(2 * np.pi * 11 * j / 40)
```

```{code-cell} ipython3
X = DFT(x)
```

```{code-cell} ipython3
plot_magnitude(x=x, X=X)
```

What happens if we change the last example to $x_{n}=2\cos\left(2\pi\frac{10}{40}n\right)$? 

Note that $\frac{10}{40}$ is an integer multiple of $\frac{1}{20}$.

```{code-cell} ipython3
N = 20
x = np.empty(N)

for j in range(N):
    x[j] = 2 * np.cos(2 * np.pi * 10 * j / 40)
```

```{code-cell} ipython3
X = DFT(x)
```

```{code-cell} ipython3
plot_magnitude(x=x, X=X)
```

If we represent the discrete Fourier transform as a matrix, we discover that it equals the  matrix $F_{N}$ of eigenvectors  of the permutation matrix $P_{N}$.

We can use the example where $x_{n}=2\cos\left(2\pi\frac{11}{40}n\right),\ n=0,1,2,\cdots19$ to illustrate this.

```{code-cell} ipython3
N = 20
x = np.empty(N)

for j in range(N):
    x[j] = 2 * np.cos(2 * np.pi * 11 * j / 40)
```

```{code-cell} ipython3
x
```

First use the summation formula to transform $x$ to $X$.

```{code-cell} ipython3
X = DFT(x)
X
```

Now let's evaluate the outcome  of postmultiplying  the eigenvector matrix  $F_{20}$ by the vector $x$, a product that we claim should equal the Fourier tranform of the sequence $\{x_n\}_{n=0}^{N-1}$.

```{code-cell} ipython3
F20, _ = construct_F(20)
```

```{code-cell} ipython3
F20 @ x
```

Similarly, the inverse DFT can be expressed as a inverse DFT matrix $F^{-1}_{20}$.

```{code-cell} ipython3
F20_inv = np.linalg.inv(F20)
F20_inv @ X
```

```{code-cell} ipython3

```
