
## Old Stuff -- Pre March 26

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

Here the (possibly huge) $\tilde n \times m $ matrix $X^{+}$ is the Moore-Penrose generalized inverse of $X$.

The $i$th the row of $A$ is an $m \times 1$ vector of regression coefficients of $X_{i,t+1}$ on $X_{j,t}, j = 1, \ldots, m$.


Consider the (reduced) singular value decomposition 

  $$ 
  X =  U \Sigma  V^T
  $$ (eq:SVDforDMD)


  
where $U$ is $m \times p$, $\Sigma$ is a $p \times p$ diagonal  matrix, and $ V^T$ is a $p \times \tilde n$ matrix.

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




##  Analysis

We'll put basic ideas on the table by starting with the special case in which $r = p$ so that we retain
all $p$ singular values of $X$.

(Later, we'll retain only $r < p$ of them)

When $r = p$,  formula
{eq}`eq:Xpinverse`  for $X^+$ implies that 


$$
A = X' V \Sigma^{-1}  U^T
$$ (eq:Aformbig)

where $V$ is an $\tilde n \times p$ matrix, $\Sigma^{-1}$ is a $p \times p$ matrix,  $U^T$ is a $p \times m$ matrix,
and  $U^T  U = I_p$ and $V V^T = I_m $.


It is convenient to represent $A$ as computed in  equation {eq}`eq:Aformbig` as

$$
A = U \tilde A U^T
$$ (eq:Afactortilde)

where the   $p \times p$ transition matrix $\tilde A$ can be recovered from 

$$
 \tilde A = U^T A U = U^T X' V \Sigma^{-1} .
$$ (eq:Atilde0)

We use the $p$  columns of $U$, and thus the $p$ rows of $U^T$,  to define   a $p \times 1$  vector $\tilde X_t$ as follows


$$
\tilde X_t = U^T X_t .
$$ (eq:tildeXdef2)

Since $U U^T$ is an $m \times m$ identity matrix, it follows from equation {eq}`eq:tildeXdef2` that we can reconstruct  $X_t$ from $\tilde X_t$ by using 

$$
X_t = U \tilde X_t .
$$ (eq:Xdecoder)


 * Equation {eq}`eq:tildeXdef2` serves as an **encoder** that  summarizes the $m \times 1$ vector $X_t$ by a $p \times 1$ vector $\tilde X_t$ 
  
 * Equation {eq}`eq:Xdecoder` serves as a **decoder** that recovers the $m \times 1$ vector $X_t$ from the $p \times 1$ vector $\tilde X_t$ 



Because $U^T U = I_p$, we have

$$
\tilde X_{t+1} = \tilde A \tilde X_t 
$$ (eq:xtildemotion)

Notice that if we multiply both sides of {eq}`eq:xtildemotion` by $U$ 
we get

$$
U \tilde X_{t+1} = U \tilde A \tilde X_t =  U \tilde A U^T X_t 
$$

which by virtue of decoder equation {eq}`eq:xtildemotion` recovers

$$
X_{t+1} = A X_t .
$$





  
It is useful to  construct an eigencomposition of the $p \times p$ transition matrix  $\tilde A$ defined 
in equation in {eq}`eq:Atilde0` above:  

$$ 
  \tilde A W =  W \Lambda
$$ (eq:tildeAeigen)
  
where $\Lambda$ is a $r \times r$ diagonal matrix of eigenvalues and the columns of $W$ are corresponding eigenvectors
of $\tilde A$.   

Both $\Lambda$ and $W$ are $p \times p$ matrices.
  
Construct the $m \times p$ matrix

$$
  \Phi = X'  V  \Sigma^{-1} W
$$ (eq:Phiformula)


  
Tu et al. {cite}`tu_Rowley` established the following  

**Proposition** The $r$ columns of $\Phi$ are eigenvectors of $A$ that correspond to the largest $r$ eigenvalues of $A$. 

**Proof:** From formula {eq}`eq:Phiformula` we have

$$  
\begin{aligned}
  A \Phi & =  (X' V \Sigma^{-1} U^T) (X' V \Sigma^{-1} W) \cr
  & = X' V \Sigma^{-1} \tilde A W \cr
  & = X' V \Sigma^{-1} W \Lambda \cr
  & = \Phi \Lambda 
  \end{aligned}
$$ 

Thus, we  have deduced  that

$$  
A \Phi = \Phi \Lambda
$$ (eq:APhiLambda)

Let $\phi_i$ be the the $i$the column of $\Phi$ and $\lambda_i$ be the corresponding $i$ eigenvalue of $\tilde A$ from decomposition {eq}`eq:tildeAeigen`. 

Writing out the $m \times p$ vectors on both sides of  equation {eq}`eq:APhiLambda` and equating them gives


$$
A \phi_i = \lambda_i \phi_i .
$$

Thus, $\phi_i$ is an eigenvector of $A$ that corresponds to eigenvalue  $\lambda_i$ of $A$.

This concludes the proof. 


Also see {cite}`DDSE_book` (p. 238)


### Two Representations of $A$

We  have constructed  two representations of (or approximations to) $A$.

One from equation {eq}`eq:Afactortilde` is 

$$ 
A = U \tilde A U^T  
$$ (eq:Aform11)

while from equation the eigen decomposition {eq}`eq:APhiLambda` the other  is 

$$ 
A = \Phi \Lambda \Phi^+ 
$$ (eq:Aform12)


From formula {eq}`eq:Aform11` we can deduce 

$$
\tilde X_{t+1}  = \tilde A \tilde X_t 
$$

where 

$$
\begin{aligned}
\tilde X_t & = U^T X_t \cr
X_t & = U \tilde X_t
\end{aligned}
$$


From formula {eq}`eq:Aform12` we can deduce 

$$ 
b_{t+1} = \Lambda b_t 
$$

where

$$
\begin{aligned}
b_t & = \Phi^+ X_t \cr 
X_t & = \Phi b_t 
\end{aligned}
$$


There is better formula for the $p \times 1$ vector $b_t$

In particular, the following argument from {cite}`DDSE_book` (page 240) provides a computationally efficient way
to compute $b_t$.  

For convenience, we'll do this first for time $t=1$.



For $t=1$, we have  

$$ 
   X_1 = \Phi b_1
$$ (eq:X1proj)

where $b_1$ is a $p \times 1$ vector. 

Since $X_1 =  U \tilde X_1$, it follows that 
 
$$ 
  U \tilde X_1 = X' V \Sigma^{-1} W b_1
$$

and

$$ 
  \tilde X_1 = U^T X' V \Sigma^{-1} W b_1
$$

Recall  that $ \tilde A = U^T X' V \Sigma^{-1}$ so that
  
$$ 
  \tilde X_1 = \tilde A W b_1
$$

and therefore, by the eigendecomposition  {eq}`eq:tildeAeigen` of $\tilde A$, we have

$$ 
  \tilde X_1 = W \Lambda b_1
$$ 

Therefore, 
  
$$ 
  b_1 = ( W \Lambda)^{-1} \tilde X_1
$$ 

or 


$$ 
  b_1 = ( W \Lambda)^{-1} U^T X_1
$$ (eq:beqnsmall)



which is  computationally more efficient than the following instance of our earlier equation for computing the initial vector $b_1$:

$$
  b_1= \Phi^{+} X_1
$$ (eq:bphieqn)


Conditional on $X_t$, we can construct forecasts $\check X_{t+j} $ of $X_{t+j}, j = 1, 2, \ldots, $  from 
either 

$$
\check X_{t+j} = \Phi \Lambda^j \Phi^{+} X_t
$$ (eq:checkXevoln)


or  the following equation 

$$ 
  \check X_{t+j} = \Phi \Lambda^j (W \Lambda)^{-1}  U^T X_t
$$ (eq:checkXevoln2)



### Using Fewer Modes

The preceding formulas assume that we have retained all $p$ modes associated with the positive
singular values of $X$.  

We can easily adapt all of the formulas to describe a situation in which we instead retain only
the $r < p$ largest singular values.  

In that case, we simply replace $\Sigma$ with the appropriate $r \times r$ matrix of singular values,
$U$ with the $m \times r$ matrix of whose columns correspond to the $r$ largest singular values,
and $V$ with the $\tilde n \times r$ matrix whose columns correspond to the $r$ largest  singular values.

Counterparts of all of the salient formulas above then apply.





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
