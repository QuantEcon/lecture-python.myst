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

# VARs and DMDs

This lecture applies computational methods  that we learned about in this lecture 
{doc}`Singular Value Decomposition <svd_intro>` to 

* first-order vector autoregressions (VARs)
* dynamic mode decompositions (DMDs)
* connections between DMDs and first-order VARs 

## First-Order Vector Autoregressions 


We want to fit a **first-order vector autoregression**

$$
X_{t+1} = A X_t + C \epsilon_{t+1}, \quad \epsilon_{t+1} \perp X_t 
$$ (eq:VARfirstorder)

where $\epsilon_{t+1}$ is the time $t+1$ component  of a sequence of  i.i.d. $m \times 1$ random vectors with mean vector
zero and identity  covariance matrix and where 
the $ m \times 1 $ vector $ X_t $ is

$$
X_t = \begin{bmatrix}  X_{1,t} & X_{2,t} & \cdots & X_{m,t}     \end{bmatrix}^\top 
$$ (eq:Xvector)

and where $\cdot ^\top $ again denotes complex transposition and $ X_{i,t} $ is  variable $ i $ at time $ t $.



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

Here $ ' $  is part of the name of the matrix $ X' $ and does not indicate matrix transposition.

We  use  $\cdot^\top $ to denote matrix transposition or its extension to complex matrices. 

In forming $ X $ and $ X' $, we have in each case  dropped a column from $ \tilde X $,  the last column in the case of $ X $, and  the first column in the case of $ X' $.

Evidently, $ X $ and $ X' $ are both $ m \times  n $ matrices.

We denote the rank of $ X $ as $ p \leq \min(m, n)  $.

Two  cases that interest us are

 *  $ n > > m$, so that we have many more time series  observations $n$ than variables $m$
 *  $m > > n$, so that we have many more variables $m $ than time series observations $n$

At a general level that includes both of these special cases, a common formula describes the least squares estimator $\hat A$ of $A$.

But important  details differ.

The common formula is

$$ 
\hat A = X' X^+ 
$$ (eq:commonA)

where $X^+$ is the pseudo-inverse of $X$.

To read about the **Moore-Penrose pseudo-inverse** please see [Moore-Penrose pseudo-inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)

Applicable formulas for the pseudo-inverse differ for our two cases.

**Short-Fat Case:**

When $ n > > m$, so that we have many more time series  observations $n$ than variables $m$ and when
$X$ has linearly independent **rows**, $X X^\top $ has an inverse and the pseudo-inverse $X^+$ is

$$
X^+ = X^\top  (X X^\top )^{-1} 
$$

Here $X^+$ is a **right-inverse** that verifies $ X X^+ = I_{m \times m}$.

In this case, our formula {eq}`eq:commonA` for the least-squares estimator of the population matrix of regression coefficients  $A$ becomes

$$ 
\hat A = X' X^\top  (X X^\top )^{-1}
$$ (eq:Ahatform101)


This  formula for least-squares regression coefficients is widely used in econometrics.

It is used  to estimate vector autorgressions.   

The right side of formula {eq}`eq:Ahatform101` is proportional to the empirical cross second moment matrix of $X_{t+1}$ and $X_t$ times the inverse
of the second moment matrix of $X_t$.



**Tall-Skinny Case:**

When $m > > n$, so that we have many more attributes $m $ than time series observations $n$ and when $X$ has linearly independent **columns**,
$X^\top  X$ has an inverse and the pseudo-inverse $X^+$ is

$$
X^+ = (X^\top  X)^{-1} X^\top 
$$

Here  $X^+$ is a **left-inverse** that verifies $X^+ X = I_{n \times n}$.

In this case, our formula  {eq}`eq:commonA` for a least-squares estimator of $A$ becomes

$$
\hat A = X' (X^\top  X)^{-1} X^\top 
$$ (eq:hatAversion0)

Please compare formulas {eq}`eq:Ahatform101` and {eq}`eq:hatAversion0` for $\hat A$.

Here we are especially interested in formula {eq}`eq:hatAversion0`.

The $ i $th  row of $ \hat A $ is an $ m \times 1 $ vector of regression coefficients of $ X_{i,t+1} $ on $ X_{j,t}, j = 1, \ldots, m $.


If we use formula {eq}`eq:hatAversion0` to calculate $\hat A X$ we find that

$$
\hat A X = X'
$$

so that the regression equation **fits perfectly**.

This is a typical outcome in an **underdetermined least-squares** model.


To reiterate, in the  **tall-skinny** case (described in {doc}`Singular Value Decomposition <svd_intro>`)  in which we have a number $n$ of observations   that is small relative to the number $m$ of
attributes that appear in the vector $X_t$,  we want to fit equation {eq}`eq:VARfirstorder`.

We  confront the facts that the least squares estimator is underdetermined and that the regression equation fits perfectly.  


To proceed, we'll want efficiently to calculate the pseudo-inverse $X^+$.

The pseudo-inverse $X^+$ will be a component of our estimator of $A$.

As our  estimator $\hat A$ of $A$ we want to form an  $m \times m$ matrix that  solves the least-squares best-fit problem

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

where the (possibly huge) $ n \times m $ matrix $ X^{+} = (X^\top  X)^{-1} X^\top $ is again a pseudo-inverse of $ X $.




For some situations that we are interested in, $X^\top  X $ can be close to singular, a situation that  makes some numerical algorithms  be inaccurate.

To acknowledge that possibility, we'll use  efficient algorithms to  constructing 
a **reduced-rank approximation** of  $\hat A$ in formula {eq}`eq:hatAversion0`.

Such an approximation to our vector autoregression will no longer fit perfectly.
 

The $ i $th  row of $ \hat A $ is an $ m \times 1 $ vector of regression coefficients of $ X_{i,t+1} $ on $ X_{j,t}, j = 1, \ldots, m $.

An efficient way to compute the pseudo-inverse $X^+$ is to start with  a singular value decomposition



$$
X =  U \Sigma  V^\top  
$$ (eq:SVDDMD)

where we remind ourselves that for a **reduced** SVD, $X$ is an $m \times n$ matrix of data, $U$ is an $m \times p$ matrix, $\Sigma$  is a $p \times p$ matrix, and $V$ is an $n \times p$ matrix.  

We can    efficiently  construct the pertinent pseudo-inverse $X^+$
by recognizing the following string of equalities.  

$$
\begin{aligned}
X^{+} & = (X^\top  X)^{-1} X^\top  \\
  & = (V \Sigma U^\top  U \Sigma V^\top )^{-1} V \Sigma U^\top  \\
  & = (V \Sigma \Sigma V^\top )^{-1} V \Sigma U^\top  \\
  & = V \Sigma^{-1} \Sigma^{-1} V^\top  V \Sigma U^\top  \\
  & = V \Sigma^{-1} U^\top  
\end{aligned}
$$ (eq:efficientpseudoinverse)


(Since we are in the $m > > n$ case in which $V^\top  V = I_{p \times p}$ in a reduced SVD, we can use the preceding
string of equalities for a reduced SVD as well as for a full SVD.)

Thus, we shall  construct a pseudo-inverse $ X^+ $  of $ X $ by using
a singular value decomposition of $X$ in equation {eq}`eq:SVDDMD`  to compute


$$
X^{+} =  V \Sigma^{-1}  U^\top  
$$ (eq:Xplusformula)

where the matrix $ \Sigma^{-1} $ is constructed by replacing each non-zero element of $ \Sigma $ with $ \sigma_j^{-1} $.

We can  use formula {eq}`eq:Xplusformula`   together with formula {eq}`eq:hatAform` to compute the matrix  $ \hat A $ of regression coefficients.

Thus, our  estimator $\hat A = X' X^+$ of the $m \times m$ matrix of coefficients $A$    is

$$
\hat A = X' V \Sigma^{-1}  U^\top  
$$ (eq:AhatSVDformula)



## Dynamic Mode Decomposition (DMD)



We turn to the $ m >>n $ **tall and skinny** case  associated with **Dynamic Mode Decomposition**.

Here an $ m \times n+1 $  data matrix $ \tilde X $ contains many more attributes (or variables) $ m $ than time periods  $ n+1 $.


Dynamic mode decomposition was introduced by {cite}`schmid2010`,

You can read  about Dynamic Mode Decomposition {cite}`DMD_book` and {cite}`Brunton_Kutz_2019` (section 7.2).


**Dynamic Mode Decomposition** (DMD) computes a rank $ r < p  $ approximation to the least squares regression coefficients $ \hat A $  described by formula {eq}`eq:AhatSVDformula`.

  
We'll  build up gradually  to a formulation that is useful  in applications.


We'll do this by describing three  alternative representations of our first-order linear dynamic system, i.e., our vector autoregression. 

**Guide to three representations:** In practice, we'll mainly be interested in Representation 3. 

We use the first two representations  to present some useful  intermediate steps that  help us to appreciate what is under the hood of Representation 3.  

In applications, we'll use only a small  subset of **DMD modes** to approximate dynamics. 

We use  such a small subset of DMD modes to  construct a reduced-rank approximation to $A$.

To do that, we'll want to use the  **reduced**  SVD's affiliated with representation 3, not the **full** SVD's affiliated with representations 1 and 2. 


**Guide to impatient reader:** In our applications, we'll be using Representation 3. 

You might want to skip the stage-setting representations 1 and 2 on first reading.

+++

## Representation 1
 
In this representation, we shall use a **full** SVD of $X$.

We use the $m$  **columns** of $U$, and thus the $m$ **rows** of $U^\top $,  to define   a $m \times 1$  vector $\tilde b_t$ as 


$$
\tilde b_t = U^\top  X_t .
$$ (eq:tildeXdef2)

The original  data $X_t$ can be represented as

$$ 
X_t = U \tilde b_t
$$ (eq:Xdecoder)

(Here we use $b$ to remind ourselves that we are creating a **basis** vector.)

Since we are now using a **full** SVD, $U U^\top  = I_{m \times m}$.

So it follows from equation {eq}`eq:tildeXdef2` that we can reconstruct  $X_t$ from $\tilde b_t$.

In particular,  



 * Equation {eq}`eq:tildeXdef2` serves as an **encoder** that  **rotates** the $m \times 1$ vector $X_t$ to become an $m \times 1$ vector $\tilde b_t$ 
  
 * Equation {eq}`eq:Xdecoder` serves as a **decoder** that **reconstructs** the $m \times 1$ vector $X_t$ by rotating  the $m \times 1$ vector $\tilde b_t$ 



Define a  transition matrix for an $m \times 1$ basis vector  $\tilde b_t$ by

$$ 
\tilde A = U^\top  \hat A U 
$$ (eq:Atilde0)

We can  recover $\hat A$ from

$$
\hat A = U \tilde A U^\top  
$$

Dynamics of the  $m \times 1$ basis vector $\tilde b_t$ are governed by

$$
\tilde b_{t+1} = \tilde A \tilde b_t 
$$

To construct forecasts $\overline X_t$ of  future values of $X_t$ conditional on $X_1$, we can apply  decoders (i.e., rotators) to both sides of this equation and deduce

$$
\overline X_{t+1} = U \tilde A^t U^\top  X_1
$$

where we use $\overline X_{t+1}, t \geq 1 $ to denote a forecast.

+++

## Representation 2


This representation is related to  one originally proposed by  {cite}`schmid2010`.

It can be regarded as an intermediate step on the way  to obtaining  a related   representation 3 to be presented later


As with Representation 1, we continue to

* use a **full** SVD and **not** a reduced SVD



As we observed and illustrated   in a lecture about the {doc}`Singular Value Decomposition <svd_intro>`

  * (a) for a full SVD $U U^\top  = I_{m \times m} $ and $U^\top  U = I_{p \times p}$ are both identity matrices
 
  * (b)  for  a reduced SVD of $X$, $U^\top  U $ is not an identity matrix.  

As we shall see later, a full SVD is  too confining for what we ultimately want to do, namely,  cope with situations in which  $U^\top  U$ is **not** an identity matrix because we  use a reduced SVD of $X$.

But for now, let's proceed under the assumption that we are using a full SVD so that  requirements (a) and (b) are both satisfied.

 

Form an eigendecomposition of the $m \times m$ matrix $\tilde A = U^\top  \hat A U$ defined in equation {eq}`eq:Atilde0`:

$$
\tilde A = W \Lambda W^{-1} 
$$ (eq:tildeAeigen)

where $\Lambda$ is a diagonal matrix of eigenvalues and $W$ is an $m \times m$
matrix whose columns are eigenvectors  corresponding to rows (eigenvalues) in 
$\Lambda$.

When $U U^\top  = I_{m \times m}$, as is true with a full SVD of $X$, it follows that 

$$ 
\hat A = U \tilde A U^\top  = U W \Lambda W^{-1} U^\top  
$$ (eq:eqeigAhat)

According to equation {eq}`eq:eqeigAhat`, the diagonal matrix $\Lambda$ contains eigenvalues of $\hat A$ and corresponding eigenvectors of $\hat A$ are columns of the matrix $UW$. 

It follows that the systematic (i.e., not random) parts of the $X_t$ dynamics captured by our first-order vector autoregressions   are described by

$$
X_{t+1} = U W \Lambda W^{-1} U^\top   X_t 
$$

Multiplying both sides of the above equation by $W^{-1} U^\top $ gives

$$ 
W^{-1} U^\top  X_{t+1} = \Lambda W^{-1} U^\top  X_t 
$$

or 

$$
\hat b_{t+1} = \Lambda \hat b_t
$$

where our **encoder**  is 

$$ 
\hat b_t = W^{-1} U^\top  X_t
$$

and our **decoder** is

$$
X_t = U W \hat b_t
$$

We can use this representation to construct a predictor $\overline X_{t+1}$ of $X_{t+1}$ conditional on $X_1$  via: 

$$
\overline X_{t+1} = U W \Lambda^t W^{-1} U^\top  X_1 
$$ (eq:DSSEbookrepr)


In effect, 
{cite}`schmid2010` defined an $m \times m$ matrix $\Phi_s$ as

$$ 
\Phi_s = UW 
$$ (eq:Phisfull)

and a generalized inverse

$$
\Phi_s^+ = W^{-1}U^\top  
$$ (eq:Phisfullinv)

{cite}`schmid2010` then  represented equation {eq}`eq:DSSEbookrepr` as

$$
\overline X_{t+1} = \Phi_s \Lambda^t \Phi_s^+ X_1 
$$ (eq:schmidrep)

Components of the  basis vector $ \hat b_t = W^{-1} U^\top  X_t \equiv \Phi_s^+ X_t$ are   
DMD **projected modes**.    

To understand why they are called **projected modes**, notice that

$$ 
\Phi_s^+ = ( \Phi_s^\top  \Phi_s)^{-1} \Phi_s^\top 
$$

so that the $m \times p$ matrix 

$$
\hat b =  \Phi_s^+ X
$$ 

is a matrix of regression coefficients of the $m \times n$ matrix $X$ on the $m \times p$ matrix $\Phi_s$.

We'll say more about this interpretation in a related context when we discuss representation 3, which was suggested by  Tu et al. {cite}`tu_Rowley`.

It is more appropriate to use  representation 3  when, as is often the case  in practice, we want to use a reduced SVD.




## Representation 3

Departing from the procedures used to construct  Representations 1 and 2, each of which deployed a **full** SVD, we now use a **reduced** SVD.  

Again, we let  $p \leq \textrm{min}(m,n)$ be the rank of $X$.

Construct a **reduced** SVD

$$
X = \tilde U \tilde \Sigma \tilde V^\top , 
$$

where now $\tilde U$ is $m \times p$, $\tilde \Sigma$ is $ p \times p$, and $\tilde V^\top $ is $p \times n$. 

Our minimum-norm least-squares approximator of  $A$ now has representation 

$$
\hat A = X' \tilde V \tilde \Sigma^{-1} \tilde U^\top 
$$ (eq:Ahatwithtildes)


**Computing Dominant Eigenvectors of $\hat A$**

We begin by paralleling a step used to construct  Representation 1, define a  transition matrix for a rotated $p \times 1$ state $\tilde b_t$ by

$$ 
\tilde A =\tilde  U^\top  \hat A \tilde U 
$$ (eq:Atildered)


**Interpretation as projection coefficients**


{cite}`DDSE_book` remark that $\tilde A$  can be interpreted in terms of a projection of $\hat A$ onto the $p$ modes in $\tilde U$. 

To verify this, first note that, because  $ \tilde U^\top  \tilde U = I$, it follows that 

$$
\tilde A = \tilde U^\top  \hat A \tilde U = \tilde U^\top  X' \tilde V \tilde \Sigma^{-1} \tilde U^\top  \tilde U 
= \tilde U^\top  X' \tilde V \tilde \Sigma^{-1} \tilde U^\top 
$$ (eq:tildeAverify)


 

Next, we'll just  compute the regression coefficients in a projection of $\hat A$ on $\tilde U$ using a standard least-squares formula

$$
(\tilde U^\top  \tilde U)^{-1} \tilde U^\top  \hat A = (\tilde U^\top  \tilde U)^{-1} \tilde U^\top  X' \tilde V \tilde \Sigma^{-1} \tilde U^\top  = 
\tilde U^\top  X' \tilde V \tilde \Sigma^{-1} \tilde U^\top   = \tilde A .
$$

Thus, we have verified that $\tilde A$ is a least-squares projection of $\hat A$ onto $\tilde U$.

**An Inverse Challenge**


Because we are using  a reduced SVD,  $\tilde U \tilde U^\top  \neq I$.

Consequently, 

$$
\hat A \neq \tilde U \tilde A \tilde U^\top ,
$$

so we can't simply  recover $\hat A$ from  $\tilde A$ and $\tilde U$. 

**A Blind Alley**

We can start by   hoping for the best and proceeding to construct an eigendecomposition of the $p \times p$ matrix $\tilde A$:

$$
 \tilde A =  \tilde  W  \Lambda \tilde  W^{-1} 
$$ (eq:tildeAeigenred)

where $\Lambda$ is a diagonal matrix of $p$ eigenvalues and the columns of $\tilde W$
are corresponding eigenvectors. 


Mimicking our procedure in Representation 2, we cross our fingers and compute an $m \times p$ matrix

$$
\tilde \Phi_s = \tilde U \tilde W
$$ (eq:Phisred)

that  corresponds to {eq}`eq:Phisfull` for a full SVD.  

At this point, where $\hat A$ is given by formula {eq}`eq:Ahatwithtildes` it is interesting to compute $\hat A \tilde  \Phi_s$:

$$
\begin{aligned}
\hat A \tilde \Phi_s & = (X' \tilde V \tilde \Sigma^{-1} \tilde U^\top ) (\tilde U \tilde W) \\
  & = X' \tilde V \tilde \Sigma^{-1} \tilde  W \\
  & \neq (\tilde U \tilde  W) \Lambda \\
  & = \tilde \Phi_s \Lambda
  \end{aligned}
$$
 
That 
$ \hat A \tilde \Phi_s \neq \tilde \Phi_s \Lambda $ means that, unlike the  corresponding situation in Representation 2, columns of $\tilde \Phi_s = \tilde U \tilde  W$
are **not** eigenvectors of $\hat A$ corresponding to eigenvalues  on the diagonal of matix $\Lambda$.

**An Approach That Works**

Continuing our quest for eigenvectors of $\hat A$ that we **can** compute with a reduced SVD,  let's define  an $m \times p$ matrix
$\Phi$ as

$$
\Phi \equiv \hat A \tilde \Phi_s = X' \tilde V \tilde \Sigma^{-1}  \tilde  W
$$ (eq:Phiformula)

It turns out that columns of $\Phi$ **are** eigenvectors of $\hat A$.

This is  a consequence of a  result established by Tu et al. {cite}`tu_Rowley` that we now present.




  
**Proposition** The $p$ columns of $\Phi$ are eigenvectors of $\hat A$.

**Proof:** From formula {eq}`eq:Phiformula` we have

$$  
\begin{aligned}
  \hat A \Phi & =  (X' \tilde  V \tilde  \Sigma^{-1} \tilde  U^\top ) (X' \tilde  V \Sigma^{-1} \tilde  W) \cr
  & = X' \tilde V \tilde  \Sigma^{-1} \tilde A \tilde  W \cr
  & = X' \tilde  V \tilde  \Sigma^{-1}\tilde  W \Lambda \cr
  & = \Phi \Lambda 
  \end{aligned}
$$ 

so that

$$  
\hat A \Phi = \Phi \Lambda .
$$ (eq:APhiLambda)

  

Let $\phi_i$ be the $i$th  column of $\Phi$ and $\lambda_i$ be the corresponding $i$ eigenvalue of $\tilde A$ from decomposition {eq}`eq:tildeAeigenred`. 

Equating the $m \times 1$ vectors that appear on the two  sides of  equation {eq}`eq:APhiLambda`  gives


$$
\hat A \phi_i = \lambda_i \phi_i .
$$

This equation confirms that  $\phi_i$ is an eigenvector of $\hat A$ that corresponds to eigenvalue  $\lambda_i$ of both  $\tilde A$ and $\hat A$.

This concludes the proof. 

Also see {cite}`DDSE_book` (p. 238)


### Decoder of  $\check b$ as a linear projection






From  eigendecomposition {eq}`eq:APhiLambda` we can represent $\hat A$ as 

$$ 
\hat A = \Phi \Lambda \Phi^+ .
$$ (eq:Aform12)


From formula {eq}`eq:Aform12` we can deduce  dynamics of the $p \times 1$ vector $\check b_t$:

$$ 
\check b_{t+1} = \Lambda \check b_t 
$$

where

$$
\check b_t  = \Phi^+ X_t  
$$ (eq:decoder102)


Since the $m \times p$ matrix $\Phi$ has $p$ linearly independent columns, the generalized inverse of $\Phi$ is

$$
\Phi^{+} = (\Phi^\top  \Phi)^{-1} \Phi^\top 
$$

and so

$$ 
\check b = (\Phi^\top  \Phi)^{-1} \Phi^\top  X
$$ (eq:checkbform)

The $p \times n$  matrix $\check b$  is recognizable as a  matrix of least squares regression coefficients of the $m \times n$  matrix
$X$ on the $m \times p$ matrix $\Phi$ and consequently

$$
\check X = \Phi \check b
$$ (eq:Xcheck_)

is an $m \times n$ matrix of least squares projections of $X$ on $\Phi$.

 **Variance Decomposition of $X$**

By virtue of the least-squares projection theory discussed in  this quantecon lecture  <https://python-advanced.quantecon.org/orth_proj.html>, we can represent $X$ as the sum of the projection $\check X$ of $X$ on $\Phi$  plus a matrix of errors.


To verify this, note that the least squares projection $\check X$ is related to $X$ by


$$ 
X = \check X + \epsilon 
$$

or

$$
X = \Phi \check b + \epsilon
$$ (eq:Xbcheck)

where $\epsilon$ is an $m \times n$ matrix of least squares errors satisfying the least squares orthogonality conditions $\epsilon^\top  \Phi =0 $ or

$$ 
(X - \Phi \check b)^\top  \Phi = 0_{m \times p}
$$ (eq:orthls)

Rearranging  the orthogonality conditions {eq}`eq:orthls` gives $X^\top  \Phi = \check b \Phi^\top  \Phi$, which implies formula {eq}`eq:checkbform`. 





### An Approximation



We now describe a way to approximate  the $p \times 1$ vector $\check b_t$ instead of using  formula {eq}`eq:decoder102`.

In particular, the following argument adapted from {cite}`DDSE_book` (page 240) provides a computationally efficient way to approximate $\check b_t$.  

For convenience, we'll apply the method at  time $t=1$.



For $t=1$, from equation {eq}`eq:Xbcheck` we have  

$$ 
   \check X_1 = \Phi \check b_1
$$ (eq:X1proj)

where $\check b_1$ is a $p \times 1$ vector. 

Recall from representation 1 above that  $X_1 =  U \tilde b_1$, where $\tilde b_1$ is a time $1$  basis vector for representation 1 and $U$ is from the full SVD  $X = U \Sigma V^\top$.  

It  then follows from equation {eq}`eq:Xbcheck` that 

 
$$ 
  U \tilde b_1 = X' \tilde V \tilde \Sigma^{-1} \tilde  W \check b_1 + \epsilon_1
$$

where $\epsilon_1$ is a least-squares error vector from equation {eq}`eq:Xbcheck`. 

It follows that 

$$
\tilde b_1 = U^\top  X' V \tilde \Sigma^{-1} \tilde W \check b_1 + U^\top  \epsilon_1
$$


Replacing the error term $U^\top  \epsilon_1$ by zero, and replacing $U$ from a **full** SVD of $X$ with $\tilde U$ from a **reduced** SVD,  we obtain  an approximation $\hat b_1$ to $\tilde b_1$:



$$ 
  \hat b_1 = \tilde U^\top  X' \tilde V \tilde \Sigma^{-1} \tilde  W \check b_1
$$

Recall that  from equation {eq}`eq:tildeAverify`,  $ \tilde A = \tilde U^\top  X' \tilde V \tilde \Sigma^{-1}$.

It then follows  that
  
$$ 
  \hat  b_1 = \tilde   A \tilde W \check b_1
$$

and therefore, by the  eigendecomposition  {eq}`eq:tildeAeigenred` of $\tilde A$, we have

$$ 
  \hat b_1 = \tilde W \Lambda \check b_1
$$ 

Consequently, 
  
$$ 
  \hat b_1 = ( \tilde W \Lambda)^{-1} \tilde b_1
$$ 

or 


$$ 
   \hat b_1 = ( \tilde W \Lambda)^{-1} \tilde U^\top  X_1 ,
$$ (eq:beqnsmall)



which is a computationally efficient approximation to  the following instance of  equation {eq}`eq:decoder102` for  the initial vector $\check b_1$:

$$
  \check b_1= \Phi^{+} X_1
$$ (eq:bphieqn)


(To highlight that {eq}`eq:beqnsmall` is an approximation, users of  DMD sometimes call  components of   basis vector $\check b_t  = \Phi^+ X_t $  the  **exact** DMD modes and components of $\hat b_t = ( \tilde W \Lambda)^{-1} \tilde U^\top  X_t$ the **approximate** modes.)  

Conditional on $X_t$, we can compute a decoded $\check X_{t+j},   j = 1, 2, \ldots $  from the exact modes via

$$
\check X_{t+j} = \Phi \Lambda^j \Phi^{+} X_t
$$ (eq:checkXevoln)


or  use compute a decoded $\hat X_{t+j}$ from  approximate modes via

$$ 
  \hat X_{t+j} = \Phi \Lambda^j (\tilde W \Lambda)^{-1}  \tilde U^\top  X_t .
$$ (eq:checkXevoln2)

We can then use  a decoded $\check X_{t+j}$ or $\hat X_{t+j}$ to forecast $X_{t+j}$.



### Using Fewer Modes

In applications, we'll actually  use only  a few modes, often  three or less.  

Some of the preceding formulas assume that we have retained all $p$ modes associated with  singular values of $X$.  

We can  adjust our  formulas to describe a situation in which we instead retain only
the $r < p$ largest singular values.  

In that case, we simply replace $\tilde \Sigma$ with the appropriate $r\times r$ matrix of singular values, $\tilde U$ with the $m \times r$ matrix  whose columns correspond to the $r$ largest singular values, and $\tilde V$ with the $n \times r$ matrix whose columns correspond to the $r$ largest  singular values.

Counterparts of all of the salient formulas above then apply.



## Source for Some Python Code

You can find a Python implementation of DMD here:

https://mathlab.sissa.it/pydmd
