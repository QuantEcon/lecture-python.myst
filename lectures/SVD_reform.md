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

## SVD reformulation

+++


We turn to the case in which $ m >>n $ in which an $ m \times n $  data matrix $ \tilde X $ contains many more random variables $ m $ than observations $ n $.

This  **tall and skinny** case is associated with **Dynamic Mode Decomposition**.

You can read about Dynamic Mode Decomposition here [[KBBWP16](https://python.quantecon.org/zreferences.html#id24)] and here [[BK19](https://python.quantecon.org/zreferences.html#id25)] (section 7.2).


We want to fit a first order vector autoregression

$$
X_{t+1} = A X_t + C \epsilon_{t+1}
$$

where 
the $ m \times 1 $ vector $ X_t $ is

$$
X_t = \begin{bmatrix}  X_{1,t} & X_{2,t} & \cdots & X_{m,t}     \end{bmatrix}^T
$$

and where $ T $ again denotes complex transposition and $ X_{i,t} $ is an observation on variable $ i $ at time $ t $.

We  an $ m \times n $ matrix of data $ \tilde X $ of the form

$$
\tilde X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_n\end{bmatrix}
$$

where for $ t = 1, \ldots, n $,  the $ m \times 1 $ vector $ X_t $ is

$$
X_t = \begin{bmatrix}  X_{1,t} & X_{2,t} & \cdots & X_{m,t}     \end{bmatrix}^T
$$


From $ \tilde X $,   form two matrices

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

We denote the rank of $ X $ as $ p \leq \min(m, \tilde n) = \tilde n $.

We start with a system consisting of $ m $ least squares regressions of **everything** on one lagged value of **everything**:

$$
X' = A  X 
$$

where the $m \times m$ matrix $A$ solves the best-fit problem

$$ 
A = \textrm{argmin}_{\tilde A} || X' - \tilde A ||_F   
$$

Here $|| \cdot ||_F$ denotes the Frobeneus norm of a matrix.

It turns out that 

$$
A =  X'  X^{+} . \tag{6.3}
$$

Here the (possibly huge) $ \tilde n \times m $ matrix $ X^{+} $ is the pseudo-inverse of $ X $.

The $ i $th the row of $ A $ is an $ m \times 1 $ vector of pseudo-regression coefficients of $ X_{i,t+1} $ on $ X_{j,t}, j = 1, \ldots, m $.

Consider the (reduced) singular value decomposition


<a id='equation-eq-svdfordmd'></a>
$$
X =  U \Sigma  V^T \tag{6.4}
$$

where $ U $ is $ m \times p $, $ \Sigma $ is a $ p \times p $ diagonal  matrix, and $ V^T $ is a $ p \times \tilde n $ matrix.

Here $ p $ is the rank of $ X $, where necessarily $ p \leq \tilde n $.

(We  described and illustrated a **reduced** singular value decomposition above, and compared it with a **full** singular value decomposition.)

We can construct a pseudo-inverse $ X^+ $  of $ X $ by using
a singular value decomposition  $ X = U \Sigma V^T $ to compute


<a id='equation-eq-xpinverse'></a>
$$
X^{+} =  V \Sigma^{-1}  U^T \tag{6.5}
$$

where the matrix $ \Sigma^{-1} $ is constructed by replacing each non-zero element of $ \Sigma $ with $ \sigma_j^{-1} $.

We could use formula [(6.5)](#equation-eq-xpinverse)   together with formula [(6.3)](#equation-eq-afullformula) to compute the matrix  $ A $ of regression coefficients.

In addition to doing that, we’ll eventually use **dynamic mode decomposition** to compute a rank $ r $ approximation to $ A $,
where $ r <  p $.
  

+++

### Review of key formulas



$$ A = X' X^+ $$


where the singular value decomposition of $X$ is


$$
X  = U \Sigma V^T
$$

We  represent the pseudo-inverse $X^+$ as

$$ X^+ = V \Sigma^{-1} U^T $$

Thus, our DMD estimator of the $m \times m$ matrix of coefficients   $A = X' X^+$ is

$$
A = X' V \Sigma^{-1}  U^T 
$$

Next, we turn to two alternative __reduced order__ representations of our dynamic system.

+++

## Representation 1

Define an encoder

$$
\tilde b_t = U^T X_t 
$$

and a decoder

$$ 
X_t - U \tilde b_t
$$

Define the reduced transition matrix 

$$ 
\tilde A = U^T A U 
$$

We can evidently recover $A$ from

$$
A = U \tilde A U^T 
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

Thus, the $X_t$ dynamics are described by

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
