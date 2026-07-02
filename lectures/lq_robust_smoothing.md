---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(lq_robust_smoothing)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Robust Consumption Smoothing and Precautionary Savings

```{contents} Contents
:depth: 2
```

```{index} single: Robust Control; permanent income
```

```{index} single: Precautionary Savings; robustness
```

## Overview

This lecture studies a robust version of the LQ permanent income model due to {cite:t}`HST_1999` and {cite:t}`HansenSargent2008`.

It is the third of three lectures on the LQ permanent income model.

It builds on {doc}`lq_permanent_income`, which develops the standard model, and {doc}`lq_bewley_complete_markets`, which studies its cross-section and market-structure implications.

A consumer who distrusts his specification of the labor income process engages in a form of precautionary savings.

Our description of the model with concerns about robustness includes

- how (for quantities) a concern for robustness is observationally equivalent to an increase in
  impatience
- how the worst-case model that the consumer uses to shape his decision rule distorts the baseline model's endowment process toward greater persistence
- a frequency-domain representation of the effects of concerns about misspecification of the endowment process
- a detection-error-probability characterization of the amount of model uncertainty

The lecture concludes by combining the Bewley economy of {doc}`lq_bewley_complete_markets` with the robustness machinery.

Using tools from {cite:t}`HansenSargent2008`, we show:

- how a continuum of consumers $i$ who use identical decision rules can nevertheless differ in their robustness parameters $\sigma_i \leq 0$ and
  their discount factors $\beta_i$, provided that the  pair $(\sigma_i, \beta_i)$ lies on an observational-equivalence locus
  derived below
- how every such consumer chooses the **same consumption-saving rule** as a baseline
  plain-vanilla $(\sigma = 0, \beta)$ agent with no concerns about misspecification of the endowment process
- how the equilibrium interest rate $R = \beta^{-1}$ and all aggregate dynamics therefore
  coincide with those of a benchmark Bewley model
- how distinct $(\sigma_i, \beta_i)$ agents act as if they have  different subjective  models of their non-financial income process

We first present the HST model in its general form, which includes physical capital and investment $i_t$.

When we return to the Bewley economy of {doc}`lq_bewley_complete_markets`, we specialise to a pure endowment economy with no capital, so investment plays no role there.

Let's begin with some imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

```

## A brief review

We recall the essentials from {doc}`lq_permanent_income` and {doc}`lq_bewley_complete_markets`.

A consumer with quadratic utility and discount factor $\beta$ faces the endowment process

$$
\begin{aligned}
z_{t+1} &= \check{A}\, z_t + \check{C}\, w_{t+1} \\
y_t &= \check{G}\, z_t
\end{aligned}
$$ (eq:pi-endowment)

The optimal decision rule has a state-space representation in which the state is current consumption $c_t$ and the exogenous endowment state $z_t$:

$$
\begin{aligned}
c_{t+1} &= c_t + (1-\beta)\,\check{G}(I-\beta\check{A})^{-1}\check{C}\, w_{t+1} \\
b_t &= \check{G}(I-\beta\check{A})^{-1} z_t - \frac{1}{1-\beta}\,c_t \\
y_t &= \check{G}\, z_t \\
z_{t+1} &= \check{A}\, z_t + \check{C}\, w_{t+1}
\end{aligned}
$$ (eq:pi-crep)

We again use the two-factor endowment $y_t = z_{1t} + z_{2t}$,

$$
\begin{pmatrix}z_{1,t+1}\\z_{2,t+1}\end{pmatrix}
=
\begin{pmatrix}1 & 0\\0 & 0\end{pmatrix}
\begin{pmatrix}z_{1t}\\z_{2t}\end{pmatrix}
+
\begin{pmatrix}\sigma_1 & 0\\0 & \sigma_2\end{pmatrix}
\begin{pmatrix}w_{1,t+1}\\w_{2,t+1}\end{pmatrix}
$$ (eq:pi-twofactor)

with $z_{1t}$ a permanent component and $z_{2t}$ a purely transitory component.

The following cell fixes the calibration used below.

```{code-cell} ipython3
# Parameters (as in the preceding lectures)
β = 0.95       # discount factor
σ1 = 0.15      # std of permanent shock
σ2 = 0.30      # std of transitory shock
```

## A robust permanent income model

```{index} single: Robust Control; permanent income
```

```{index} single: Precautionary Savings; robustness
```

### Robustness and precautionary savings

We now study a consumer who *distrusts* his specification of the stochastic process governing his labor income.

The model is due to {cite:t}`HST_1999` (HST), who estimated
it on US quarterly consumption and investment data.

For a fuller treatment of the HST model and its asset-pricing implications, see {doc}`robust_permanent_income`.

A consumer who fears model misspecification engages in a form of **precautionary savings** that is
distinct from the usual precautionary motive (which requires a convex marginal utility).

Here, the
precautionary motive arises because the consumer wants to protect against misspecification of the
**conditional means** of income shocks, and it operates even with quadratic preferences.

HST showed an important **observational equivalence** result: for quantities $(c_t, i_t)$ alone,
a concern for robustness is indistinguishable from an increase in impatience (a decrease in
$\beta$).

We develop this result carefully below.

```{index} single: Observational Equivalence; robustness and discounting
```

### The HST model

```{index} single: Hansen Sargent Tallarini; model
```

HST's model features a planner with preferences over consumption streams $\{c_t\}$, mediated
through **service streams** $\{s_t\}$.

Let $b$ be a preference shifter (utility bliss point).

The **Bellman equation for the robust planner** is

$$
-x^\top P x - p =
\sup_c \inf_w \Bigl\{-(s-b)^2 + \beta\bigl(\theta (w^*)^\top w^* - \mathbb{E}\,(x^*)^\top P x^* - p\bigr)\Bigr\}
$$ (eq:income1)

subject to the household technology, capital accumulation, endowment dynamics, and the state law:

$$
\begin{aligned}
s &= (1+\lambda)c - \lambda h \\
h^* &= \delta_h h + (1-\delta_h) c \\
k^* &= \delta_k k + i \\
c + i &= \gamma k + d \\
\begin{pmatrix}d\\b\end{pmatrix} &= U z \\
z^* &= A_{22} z + C_2(\epsilon^* + w^*)
\end{aligned}
$$ (eq:income1a)

Here $^*$ denotes the next-period value; $c$ is consumption; $s$ is the scalar service measure;
$h$ is a habit stock; $k$ is the capital stock; $i$ is investment; $d$ is an endowment/technology
shock; $b$ is a **preference shock** (bliss-point shifter, distinct from the bond/debt variable
$b_t$ used above); $\epsilon^* \sim N(0,I)$ is the baseline shock; and
$w^*$ is a **distortion** to the conditional mean of $\epsilon^*$ chosen by a minimizing agent.

The penalty parameter $\theta > 0$ governs the consumer's concern about robustness.

We use the transformation

$$
\sigma = -\theta^{-1} \leq 0
$$

so $\sigma = 0$ corresponds to no robustness concern and $\sigma < 0$ to an increasing concern.

When $\lambda > 0$ and $\delta_h \in (0,1)$, the technology
{eq}`eq:income1a` accommodates **habit persistence** (positive $\lambda$) or durability.

The stock
$h_t$ is a geometric weighted average of current and past consumption.

Equation $c_t + k_t = Rk_{t-1} + d_t$ with
$R = \delta_k + \gamma$ combines capital accumulation with a linear production technology.

$R$ is
the physical gross return on capital.

Let $x_t^\top = [h_{t-1},\, k_{t-1},\, z_t^\top]$.

The state transition equations are:

$$
x_{t+1} = A\, x_t + B\, u_t + C(\epsilon_{t+1} + w_{t+1})
$$ (eq:law0)

where $u_t = c_t$ and $w_{t+1}$ is the distortion to the conditional mean of $\epsilon_{t+1}$.

HST estimated the model on U.S. quarterly data (1970Q1-1996Q3) using
nondurables plus services for consumption and durable consumption plus gross private investment for
investment.

Key estimates are summarised in the following table (reported in Appendix A of HST):

| Parameter | Habit | No Habit |
|-----------|-------|----------|
| Risk-free rate | 0.025 | 0.025 |
| $\beta$ | 0.997 | 0.997 |
| $\delta_h$ | 0.682 | — |
| $\lambda$ | 2.443 | 0 |
| $\alpha_1$ | 0.813 | 0.900 |
| $\alpha_2$ | 0.189 | 0.241 |
| $\phi_1$ | 0.998 | 0.995 |
| $\phi_2$ | 0.704 | 0.450 |
| $2 \times \log L$ | 779.05 | 762.55 |

HST imposed $\beta R = 1$ and $\delta_k = 0.975$, so $\gamma$ is pinned down once $\beta$ is
estimated.

An annual real interest rate of 2.5% corresponds to $\beta = 0.997$.

### Solution when $\sigma = 0$

When $\sigma = 0$ the objective reduces to

$$
\mathbb{E}_0\sum_{t=0}^{\infty}\beta^t\bigl\{-(s_t - b_t)^2\bigr\}
$$ (eq:income5)

Formulating a Lagrangian and deriving first-order conditions yields:

$$
\begin{aligned}
\mu_{st} &= b_t - s_t \\
\mu_{ct} &= (1+\lambda)\mu_{st} + (1-\delta_h)\mu_{ht} \\
\mu_{ht} &= \beta \mathbb{E}_t[\delta_h \mu_{h,t+1} - \lambda \mu_{s,t+1}] \\
\mu_{ct} &= \beta R\, \mathbb{E}_t\mu_{c,t+1}
\end{aligned}
$$ (eq:foc)

Here $\mu_{st}$ is the **marginal valuation of consumption services**, which summarises the
endogenous state variables $h_{t-1}$ and $k_{t-1}$.

Equation {eq}`eq:foc` (last line) implies
$\mathbb{E}_t\mu_{c,t+1} = (\beta R)^{-1}\mu_{ct}$, so $\mu_{st}$ is a martingale
when $\beta R = 1$:

$$
\mu_{st} = \mu_{s,t-1} + \nu^\top \epsilon_t
$$ (eq:martingale)

for some vector $\nu$.

Solving forward and substituting gives

$$
\mu_{st} = \Psi_1 k_{t-1} + \Psi_2 h_{t-1} + \Psi_3\sum_{j=0}^{\infty} R^{-j} \mathbb{E}_t b_{t+j}
            + \Psi_4\sum_{j=0}^{\infty} R^{-j} \mathbb{E}_t d_{t+j}
$$ (eq:income10)

where

$$
\Psi_1 = -(1+\lambda)R(1-R^{-2}\beta^{-1})\!\left[\frac{1-R^{-1}\tilde\delta_h}{1-R^{-1}\tilde\delta_h+\lambda(1-\tilde\delta_h)}\right], \quad
\Psi_4 = R^{-1}\Psi_1
$$ (eq:income100a)

and $\tilde\delta_h = (\delta_h + \lambda)/(1+\lambda)$.

In the widely-studied special case $\lambda = \delta_h = 0$, so $s_t = c_t$ and
$\mu_{st} = b_t - c_t$, the marginal propensity to consume out of **non-human wealth** $Rk_{t-1}$
equals that out of **human wealth** $\sum_{j=0}^{\infty}R^{-j}\mathbb{E}_t d_{t+j}$, a well-known feature of
the LQ model.

The formula for $\mu_{st}$ can be written as $\mu_{st} = M_s x_t$ where $x_t$ follows {eq}`eq:law0`.

It follows that

$$
\nu^\top = M_s C, \qquad \alpha = \sqrt{\nu^\top \nu} = \sqrt{M_s C C^\top M_s^\top}
$$ (eq:hsoffset2)

The scalar $\alpha$ plays a central role in the observational equivalence result below.

### Observational equivalence

```{index} single: Observational Equivalence; Theorem 1
```

HST state an observational-equivalence theorem.

````{prf:theorem} Observational Equivalence, I
:label: thm-lqcs-oe1

Fix all parameters except $(\sigma, \beta)$ and suppose $\beta R = 1$ when $\sigma = 0$.

There exists $\underline\sigma < 0$ such that for any
$\sigma \in (\underline\sigma, 0)$, the optimal consumption-investment plan for $(0,\beta)$ is also
chosen by a robust decision maker with parameters $(\sigma, \hat\beta(\sigma))$, where

$$
\hat\beta(\sigma) = \frac{1}{R} + \frac{\sigma\alpha^2}{R-1}
$$ (eq:obseq)

and $\hat\beta(\sigma) < \beta$.
````

Since $R > 1$ and $\alpha^2 > 0$, a more negative $\sigma$ (stronger robustness
concern) lowers $\hat\beta$.

A robust consumer wants to save more because his alter ego, a utility-minimizing agent, makes future income look worse than the approximating model predicts.

A lower discount factor makes a consumer less patient and therefore reduces saving.

When these two forces are balanced according to {eq}`eq:obseq`, consumption plans are identical across $(\sigma, \hat\beta(\sigma))$ pairs.

````{prf:proof}
When $\beta R = 1$ and $\sigma = 0$, the marginal utility $\mu_{st}$ obeys the martingale

$$
\mu_{st} = \mu_{s,t-1} + \alpha\,\tilde\epsilon_t
$$ (eq:reversee1)

where $\tilde\epsilon_t$ is scalar IID with mean zero and unit variance.

Activating a concern about robustness ($\sigma < 0$) implies the utility minimizing alter ego sets 

$$
\tilde w_t = K(\sigma,\hat\beta)\,\mu_{s,t-1}
$$

making the worst-case model for $\mu_{st}$:

$$
\mu_{st} = (1 + \alpha\,K(\sigma,\hat\beta))\,\mu_{s,t-1} + \alpha\,\tilde\epsilon_t
$$ (eq:reversee3)

For the allocation to remain the same, we require the robust Euler equation
$\hat\beta R\,\hat{\mathbb{E}}_t\mu_{s,t+1} = \mu_{st}$ to hold under the worst-case model, which gives

$$
(\hat\beta R)^{-1} = 1 + \alpha\, K(\sigma,\hat\beta)
$$ (eq:eulerdist)

The minimizing agent's Bellman equation, a pure forecasting problem, yields

$$
\hat\zeta(\hat\beta) \equiv 1 + \alpha K(\sigma,\hat\beta) = \frac{1}{1 - \sigma\alpha^2 P(\hat\beta)}
$$ (eq:distort2)

where $P(\hat\beta)$ solves the scalar Bellman equation:

$$
-P(\hat\beta) = \frac{\hat\beta - 1 + \sigma\alpha^2 + \sqrt{(\hat\beta-1+\sigma\alpha^2)^2 + 4\sigma\alpha^2}}{-2\sigma\alpha^2}
$$ (eq:distortcons)

Solving {eq}`eq:eulerdist`-{eq}`eq:distortcons` for $\hat\beta$ gives exactly {eq}`eq:obseq`.
````

Equation {eq}`eq:obseq` is the useful numerical object because it gives a straight-line map from the robustness parameter to the observationally equivalent discount factor.

### Precautionary savings interpretation

```{index} single: Precautionary Savings; robustness vs convex marginal utility
```

The consumer's concern about model misspecification activates the precautionary savings motive that underlies the observational-equivalence theorem.

A concern about robustness makes the consumer save *more*.

Decreasing $\beta$ makes the consumer save *less*.

The observational-equivalence theorem says that these two forces can be made to offset each other exactly.

In the special case $\lambda = \delta_h = 0$, $s_t = c_t$ and the consumption rule is

$$
c_t = (1 - R^{-2}\beta^{-1})\!\left[Rk_{t-1} + \mathbb{E}_t\sum_{j=0}^{\infty}R^{-j}d_{t+j}\right]
      + \left(\frac{(R\beta)^{-1}-1}{R-1}\right)\!b
$$ (eq:consfunction)

The **marginal propensity to consume** out of non-human wealth $Rk_{t-1}$ *equals* that out of
human wealth $\mathbb{E}_t\sum R^{-j}d_{t+j}$.

This equal-propensity property is a hallmark of the LQ model and *persists* when a concern for robustness is present, in contrast to usual precautionary-savings models with convex marginal utility.

{prf:ref}`thm-lqcs-oe1` says that with $\sigma < 0$, the observationally equivalent
$\hat\beta$ satisfies $\hat\beta < \beta$.

If the starting point has $\beta R = 1$, then
$\hat\beta R < 1$.

For a non-robust consumer with discount factor $\hat\beta$ at the same
interest rate, the Euler equation implies $\mathbb{E}_t c_{t+1} < c_t$: expected consumption
declines over time.

This downward drift is the impatience offset in {prf:ref}`thm-lqcs-oe1`.

It cancels the robust consumer's precautionary-savings motive, leaving the consumption and investment quantities unchanged.

The upward-drift comparison appears in {prf:ref}`thm-lqcs-oe2`, which asks the reverse observational-equivalence question.

The classical precautionary motive arises because:

$$
u'''(c) > 0 \;\Rightarrow\; \mathbb{E}_t u'(c_{t+1}) > u'(\mathbb{E}_t c_{t+1}) \;\Rightarrow\; \mathbb{E}_t c_{t+1} > c_t
$$

This channel requires *convexity of marginal utility* and is absent with quadratic preferences.

In contrast, the robustness-based precautionary motive operates through distortions of **conditional means** of shocks, shifting the first moment of the innovation to non-financial income.

### Observational equivalence and distorted expectations

```{index} single: Distorted Expectations; Stackelberg multiplier game
```

The observational-equivalence result can be interpreted using a **Stackelberg multiplier game**.

After the minimizing agent has committed to a distortion process $\{w_{t+1}\}$, the maximizing consumer faces the following worst-case law of motion for the state $X_t$:

$$
\begin{aligned}
X_{t+1} &= \bigl(A - BF(\sigma,\hat\beta) + CK(\sigma,\hat\beta)\bigr) X_t + C\tilde\epsilon_{t+1} \\
\begin{pmatrix}b_t\\d_t\end{pmatrix} &= S X_t
\end{aligned}
$$ (eq:sys2)

A robust consumer with concerns about possible misspecification of the approximating model's stochastic process for non-financial income forms expectations of future income using the **distorted transition matrix**
$A - BF + CK$ rather than the approximating transition matrix $A - BF$.

The distorted expectations operator $\hat{\mathbb{E}}_t$ satisfies

$$
\hat{\mathbb{E}}_t X_{t+j} = (A - BF(\sigma,\hat\beta) + CK(\sigma,\hat\beta))^j X_t
$$

Observational equivalence requires that the modified human-wealth formula

$$
\hat\Psi_4 \sum_{j=0}^{\infty} R^{-j}\hat{\mathbb{E}}_t d_{t+j}
$$

equals its benchmark counterpart $\Psi_4 \sum_{j=0}^{\infty} R^{-j} \mathbb{E}_t d_{t+j}$.

This is achieved by a mutual adjustment of the coefficients $\hat\Psi_j$ through $\hat\beta$ and the distorted expectation operator $\hat{\mathbb{E}}_t$ through $\sigma$.

The worst-case eigenvalue of $A - BF + CK$ exceeds that of $A - BF$ in modulus, so the worst-case distortions make the income process *more persistent* than under the approximating model.

This is the precautionary motive in state-space form: the minimizing agent makes future income look more risky by introducing low-frequency persistence.

### Frequency domain interpretation

```{index} single: Frequency Domain; permanent income model
```

The LQ permanent income framework has a natural frequency-domain interpretation.

The consumer's concave utility makes him dislike **high-frequency** fluctuations in consumption, which he smooths by adjusting savings.

High-frequency fluctuations are easier to smooth, so the consumer is automatically robust to misspecification of high-frequency features of the income process.

**Low-frequency** fluctuations are harder to smooth because they are more persistent.

In the frequency-domain notation of HST, the transfer function from shocks $\epsilon_t$ to the
target $s_t - b_t$ is $G(\zeta)$, and the frequency decomposition of the $H_2$ criterion is

$$
H_2 = -\frac{1}{2\pi}\int_{-\pi}^{\pi} \operatorname{trace}\!\bigl[G(\sqrt\beta\, e^{i\omega})^\top\,G(\sqrt\beta\, e^{i\omega})\bigr]\, d\omega
$$

The integrand $G^\top G$ is *largest at low frequencies* $\omega \approx 0$, where the consumer's welfare is most sensitive to income variability.

Recognizing this, the minimizing agent concentrates the worst-case distortions at low frequencies.

The distortion process has spectral density $W(\zeta)^\top W(\zeta)$ that is concentrated near $\omega = 0$.

The variance of the worst-case shocks grows as $|\sigma|$ increases.

### Detection error probabilities

```{index} single: Detection Error Probabilities
```

A natural way to discipline the choice of $\sigma$ (or $\theta$) is to ask: **how difficult would
it be to statistically distinguish the approximating model from the worst-case model?**

For a sample of length $T$, one can use a **log-likelihood ratio test** to compare the two
hypotheses.

The **detection error probability** (DEP) is the probability of making the wrong
decision using the log-likelihood ratio statistic when one does not know which model generated the
data.

Specifically:

$$
\text{DEP}(\sigma) = \frac{1}{2}\bigl[\mathbb{P}\{\text{prefer approx.} \mid \text{worst-case is true}\}
                                    + \mathbb{P}\{\text{prefer worst-case} \mid \text{approx. is true}\}\bigr]
$$

When $\sigma = 0$ the two models are identical and DEP $= 0.5$.

As $|\sigma|$ increases the
models diverge and the DEP falls toward zero.

The full DEP calculation requires a specified approximating model, its worst-case counterpart, and the sample length used in the likelihood-ratio experiment.

We compute such a DEP for a robust Bewley model below.

```{note}
HST suggested that a DEP above 0.2 is "plausible", meaning the models are still hard enough to distinguish statistically that a concern for robustness is warranted.

Values of $\sigma$ corresponding to DEP $\geq 0.2$ define a set of plausible worst-case models.
```

### Robustness of decision rules

```{index} single: Robustness; payoff evaluation
```

To evaluate whether robust decision rules perform better than the non-robust rule when the data are
generated by a distorted model, define the **payoff** when the decision rule is designed for
robustness parameter $\sigma_2$ and the data are generated by the distorted model associated with
$\sigma_1$:

$$
\pi(\sigma_1;\sigma_2) = -\mathbb{E}_{0,\sigma_1}\sum_{t=0}^{\infty}\beta^t\, x_t^\top H(\sigma_2)^\top H(\sigma_2)\, x_t
$$ (eq:soln3)

where the state evolves under decision rule $F(\sigma_2)$ and worst-case shocks $K(\sigma_1)$:

$$
x_{t+1} = \bigl(A - BF(\sigma_2) + CK(\sigma_1)\bigr)x_t + C\epsilon_{t+1}
$$ (eq:soln2)

For $\sigma_1 = 0$ (approximating model generates data), the non-robust rule ($\sigma_2 = 0$) is
optimal by construction.

As $\sigma_1$ decreases (the data are generated by increasingly
distorted models), the payoff of the $\sigma_2 = 0$ rule deteriorates faster than that of robust
rules.

Computing the payoff comparison requires solving the full HST matrix problem for $F(\sigma_2)$ and $K(\sigma_1)$.

### Another observational equivalence result

```{index} single: Observational Equivalence; Theorem 2
```

````{prf:theorem} Observational Equivalence, II
:label: thm-lqcs-oe2

Fix all parameters except $(\sigma,\beta)$ and consider a consumption-investment allocation for $(\hat\sigma, \hat\beta)$ where $\hat\beta R = 1$ and $\hat\sigma < 0$.

Then there exists $\tilde\beta > \hat\beta$ such that the $(\hat\sigma, \hat\beta)$ allocation also solves the $(0, \tilde\beta)$ problem.
````

{prf:ref}`thm-lqcs-oe1` showed that starting from a benchmark with $\beta R = 1$, activating
robustness ($\sigma < 0$) is equivalent to *reducing* $\beta$.

{prf:ref}`thm-lqcs-oe2` goes in the opposite direction: it shows that the effects of activating a concern for robustness from a starting point with $\beta R = 1$ are replicated by *increasing* $\beta$ while setting $\sigma = 0$.

In other words, when $\beta R = 1$, a concern for robustness operates like an *increase* in the discount factor, pushing $\beta R > 1$ and imparting an *upward drift* to the expected consumption profile.

````{prf:proof}
With $\hat\beta R = 1$ and $\hat\sigma < 0$, the robust Euler equation implies

$$
\hat{\mathbb{E}}_t \mu_{c,t+1} = \mu_{ct}
$$

One seeks $\tilde\beta > \hat\beta$ and $\sigma = 0$ such that the same allocation solves the
non-robust problem with discount factor $\tilde\beta$.

The key step is to observe that the worst-case distortion $K(\hat\sigma, \hat\beta)$ introduces a
drift in the marginal utility process that is equivalent to the drift produced by raising the
discount factor above $\hat\beta$.

Equating the two drifts and solving the scalar Bellman equation for $K$ yields

$$
\tilde\beta(\hat\sigma) = \frac{\hat\beta(1+\hat\beta)}{2(1+\hat\sigma\alpha^2)}
\left[1 + \sqrt{1 - 4\hat\beta\,\frac{1+\hat\sigma\alpha^2}{(1+\hat\beta)^2}}\right]
$$ (eq:obsequivn2)

The solution satisfies $\tilde\beta > \hat\beta$ when $\hat\sigma < 0$.
````

The map {eq}`eq:obsequivn2` is a closed form, so we can plot it directly.

The next figure compares the two observational-equivalence loci for the two-factor calibration, using $\alpha^2 = \sigma_1^2 + (1-\beta)^2\sigma_2^2$ (derived below in {eq}`eq:bew_alpha2`).

We start from a benchmark with $\hat\beta R = 1$, so $\hat\beta = \beta$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The two observational-equivalence loci
    name: fig-lqcs-oe-loci
---
β_bench = β                                 # benchmark with β̂ R = 1
α2 = σ1**2 + (1 - β)**2 * σ2**2             # two-factor α² (see eq:bew_alpha2)

σ_hat_vals = np.linspace(0.0, -0.16, 60)

# Locus II (eq:obsequivn2): robustness ⟺ an *increase* in β (σ = 0)
disc = 1 - 4 * β_bench * (1 + σ_hat_vals * α2) / (1 + β_bench)**2
β_tilde = (β_bench * (1 + β_bench)) / (2 * (1 + σ_hat_vals * α2)) \
          * (1 + np.sqrt(disc))

# Locus I (eq:obseq / eq:bew_locus): robustness ⟺ a *decrease* in β (σ = 0)
β_hat = β_bench + σ_hat_vals * α2 * β_bench / (1 - β_bench)

fig, ax = plt.subplots()
ax.plot(-σ_hat_vals, β_tilde, lw=2, color='C0',
        label=r'locus II: $\tilde\beta(\hat\sigma)$ (upward drift)')
ax.plot(-σ_hat_vals, β_hat, lw=2, color='C3',
        label=r'locus I: $\hat\beta(\sigma)$ (downward drift)')
ax.axhline(β_bench, color='k', linestyle=':', lw=1,
           label=r'benchmark $\beta$ ($\beta R = 1$)')
ax.set_xlabel(r'robustness concern $-\hat\sigma$')
ax.set_ylabel('observationally equivalent discount factor')
ax.legend()
plt.show()

print(f"at σ̂ = {σ_hat_vals[-1]:.3f}:  β̃ = {β_tilde[-1]:.4f} > β = {β_bench}")
print(f"                       β̂ = {β_hat[-1]:.4f} < β = {β_bench}")
```

The two loci pass through the benchmark $\beta$ at $\hat\sigma = 0$ and separate as the robustness concern grows.

Locus I, from {prf:ref}`thm-lqcs-oe1`, lies *below* $\beta$: activating robustness looks like an increase in impatience, which imparts a downward drift to expected consumption.

Locus II, from {prf:ref}`thm-lqcs-oe2`, lies *above* $\beta$: the same robustness concern, viewed from a benchmark with $\beta R = 1$, looks like an increase in patience, which imparts an upward drift.

### A robust LQ Bewley model

```{index} single: Robust Bewley Model
```

We now synthesise the lecture by embedding the Bewley economy of {doc}`lq_bewley_complete_markets` into the HST framework and applying the observational-equivalence theorem.

In this way, we construct a family of **robust Bewley economies**, parameterised by a robustness level $\sigma \leq 0$, whose equilibrium quantities are identical to those of the plain vanilla Bewley model.

We first map the Bewley economy into HST notation, specialising the robust model to
$\lambda = \delta_h = 0$ (no habits, no durable goods) and to a
pure endowment economy (no physical capital, $k_t = 0$).

In this case:

Services equal consumption: $s_t = c_t$.

The only traded security is the one-period risk-free bond, and we write the household's net asset position as $a_t=-b_t$ so that positive $a_t$ denotes wealth rather than debt.

The endowment process follows the state-space representation {eq}`eq:pi-endowment`.

The household's augmented state vector is $x_t = [a_t,\; z_t^\top]^\top$, and the law of motion
{eq}`eq:law0` specialises to

$$
\begin{pmatrix} a_{t+1} \\ z_{t+1} \end{pmatrix}
=
\underbrace{\begin{pmatrix} R & R\check{G} \\ 0 & \check{A} \end{pmatrix}}_{A}
\begin{pmatrix} a_t \\ z_t \end{pmatrix}
+
\underbrace{\begin{pmatrix} -R \\ 0 \end{pmatrix}}_{B}
c_t
+
\underbrace{\begin{pmatrix} 0 \\ \check{C} \end{pmatrix}}_{C}
\epsilon_{t+1}
$$ (eq:bew_law)

The objective is $\mathbb{E}_0 \sum_{t=0}^\infty \beta^t [-(c_t - \gamma)^2/2]$, which is the HST
criterion {eq}`eq:income5` with $\sigma = 0$ and $b_t \equiv \gamma$ (a fixed bliss level).

The robust Bellman equation {eq}`eq:income1` with $\sigma = 0$ therefore reduces exactly to
the LQ problem of {doc}`lq_permanent_income`, confirming that the HST framework nests the Bewley model.

We next compute the robustness parameter $\alpha^2$.

From the $(c_t,z_t)$ representation {eq}`eq:pi-crep`, the consumption innovation is

$$
c_{t+1} - c_t = h\, w_{t+1}, \qquad
h = (1-\beta)\,\check{G}(I-\beta\check{A})^{-1}\check{C}
$$ (eq:bew_cinno)

The vector $h$ plays the role of $\nu^\top = M_s C$ in the HST scalar $\alpha$ formula
{eq}`eq:hsoffset2`.

Consequently,

$$
\alpha^2 = h h^\top = (1-\beta)^2\,
\check{G}(I-\beta\check{A})^{-1}\check{C}\check{C}^\top(I-\beta\check{A}^\top)^{-1}\check{G}^\top
$$ (eq:bew_alpha)

For the two-factor model {eq}`eq:pi-twofactor` with $\check{A} = \mathrm{diag}(1,0)$ and
$\check{C} = \mathrm{diag}(\sigma_1,\sigma_2)$ this simplifies to

$$
\alpha^2 = \sigma_1^2 + (1-\beta)^2\,\sigma_2^2
$$ (eq:bew_alpha2)

The permanent shock variance $\sigma_1^2$ enters with coefficient 1 because a unit permanent
shock is *fully* capitalised into consumption.

The transitory shock variance $\sigma_2^2$
enters with the small coefficient $(1-\beta)^2$ because only its annuity value is consumed.

Applying {prf:ref}`thm-lqcs-oe1` {eq}`eq:obseq` with equilibrium interest rate $R = \beta_0^{-1}$ and
$\alpha^2$ from {eq}`eq:bew_alpha2` gives the **Bewley observational equivalence locus**:

$$
\hat\beta(\sigma) = \beta_0 + \frac{\sigma\,\alpha^2\,\beta_0}{1-\beta_0}
$$ (eq:bew_locus)

For $\sigma < 0$, we have $\hat\beta(\sigma) < \beta_0$.

An agent with the pair
$(\sigma, \hat\beta(\sigma))$ on this locus is more concerned about model misspecification
(lower $\sigma$) but also more impatient (lower $\hat\beta$); the two forces cancel exactly,
leaving the consumption decision rule unchanged.

These ingredients combine into a robust Bewley equilibrium.

````{prf:proposition}
:label: prop-lqcs-bewley

Suppose all agents in the Bewley economy share a common pair
$(\sigma, \hat\beta(\sigma))$ lying on the locus {eq}`eq:bew_locus`, with $R = \beta_0^{-1}$.

Then every agent's optimal consumption plan is identical to that of the plain vanilla
$(\sigma = 0,\, \beta_0)$ economy, and the equilibrium interest rate remains $R = \beta_0^{-1}$.
````

````{prf:proof}
By {prf:ref}`thm-lqcs-oe1`, each agent's consumption-saving rule is identical to the benchmark.

The goods-market clearing condition $\int c_t^i\, di = Y$ is therefore satisfied at
$R = \beta_0^{-1}$ for the same reason as in the benchmark Bewley economy.
````

#### Heterogeneous $(\beta_i, \sigma_i)$ preferences

A richer extension populates the economy with a **continuum of types**, each indexed by a
robustness parameter $\sigma_i \in [\underline\sigma, 0]$, with discount factor

$$
\beta_i = \hat\beta(\sigma_i) = \beta_0 + \frac{\sigma_i\,\alpha^2\,\beta_0}{1-\beta_0}
$$ (eq:bew_heterog)

Since all pairs $(\sigma_i, \beta_i)$ lie on {eq}`eq:bew_locus`, every agent adopts the **same consumption rule** as the benchmark.

Aggregate dynamics are unchanged because the cross-section mean of consumption equals $Y$ and the cross-section variance grows at rate $\alpha^2$ per period.

The equilibrium interest rate is unchanged: $R = \beta_0^{-1}$.

Agents are observationally indistinguishable to an outside econometrician because data on $(c_t^i, a_t^i)$ cannot reveal whether agent $i$ has $\sigma_i = 0$ or $\sigma_i < 0$.

Agents differ in their internal model because an agent with $\sigma_i < 0$ applies a worst-case distortion $w_{t+1}^i = K(\sigma_i, \beta_i)\,\mu_{s,t}^i$ to her conditional expectations, while an agent with $\sigma_i = 0$ takes the approximating model at face value.

This sets the stage for a Bewley model with **heterogeneous ambiguity aversion**: although
every agent acts identically in terms of observable choices, they hold different subjective
models of the income process and have different attitudes toward model uncertainty.

#### Computation

```{code-cell} ipython3
# Bewley parameters
β0_bew = β       # 0.95
σ1_bew = σ1      # 0.15
σ2_bew = σ2      # 0.30
R_bew = 1.0 / β0_bew

# Two-factor Bewley α^2
α2_bew = σ1_bew**2 + (1 - β0_bew)**2 * σ2_bew**2

print(f"α^2  (Bewley, two-factor)        = {α2_bew:.6f}")
print(f"  permanent component  σ1^2     = {σ1_bew**2:.6f}  "
      f"({100*σ1_bew**2/α2_bew:.1f} % of α^2)")
print(f"  transitory component (1-β)^2σ2^2= {(1-β0_bew)**2*σ2_bew**2:.6f}  "
      f"({100*(1-β0_bew)**2*σ2_bew**2/α2_bew:.1f} % of α^2)")
```

The calculation shows why permanent shocks dominate $\alpha^2$ in this calibration.

We now solve the scalar robust forecasting problem attached to this $\alpha^2$.

The solution selects the Bellman-equation root that satisfies the observational-equivalence Euler equation.

```{code-cell} ipython3
def robust_scalar_solution(σ, β0, α2):
    """
    Solve the scalar robust marginal-utility problem on the
    observational-equivalence locus.
    """
    α = np.sqrt(α2)
    R = 1.0 / β0

    if np.isclose(σ, 0.0):
        return β0, np.nan, 1.0, 0.0

    β_hat = β0 + σ * α2 * β0 / (1 - β0)
    disc = (β_hat - 1 + σ * α2)**2 + 4 * σ * α2
    root_disc = np.sqrt(max(disc, 0.0))
    target_ζ = 1 / (β_hat * R)

    candidates = []
    for sign in (1.0, -1.0):
        P = (β_hat - 1 + σ * α2 + sign * root_disc) / (-2 * σ * α2)
        ζ = 1 / (1 - σ * α2 * P)
        K = (ζ - 1) / α
        candidates.append((abs(ζ - target_ζ), P, ζ, K))

    _, P, ζ, K = min(candidates, key=lambda x: x[0])
    return β_hat, P, ζ, K


def log_likelihood_ratio(paths, ζ, α):
    """
    Return log p_worst(path) - log p_approx(path).
    """
    lag = paths[:, :-1]
    lead = paths[:, 1:]
    ll_worst = -0.5 * np.sum(((lead - ζ * lag) / α)**2, axis=1)
    ll_approx = -0.5 * np.sum(((lead - lag) / α)**2, axis=1)
    return ll_worst - ll_approx


def simulate_scalar_paths(ζ, α, T, n_paths, seed):
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_paths, T + 1))
    shocks = rng.standard_normal((n_paths, T))

    for t in range(T):
        paths[:, t + 1] = ζ * paths[:, t] + α * shocks[:, t]

    return paths


def detection_error_probability(ζ, α, T=40, n_paths=10_000, seed=1234):
    """
    Finite-sample DEP for the approximating and worst-case scalar laws.
    """
    if np.isclose(ζ, 1.0):
        return 0.5

    approx_paths = simulate_scalar_paths(1.0, α, T, n_paths, seed)
    worst_paths = simulate_scalar_paths(ζ, α, T, n_paths, seed + 1)

    llr_approx = log_likelihood_ratio(approx_paths, ζ, α)
    llr_worst = log_likelihood_ratio(worst_paths, ζ, α)

    return 0.5 * (np.mean(llr_worst < 0) + np.mean(llr_approx > 0))
```

The next figure reports worst-case dynamics and model-detection probabilities implied by this solved scalar problem.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Solved robust scalar model
    name: fig-lqcs-robust-scalar
---
α_bew = np.sqrt(α2_bew)
β_min = 0.88
σ_min = (β_min - β0_bew) * (1 - β0_bew) / (α2_bew * β0_bew)
σ_vals = np.linspace(0.0, σ_min, 31)

solutions = np.array([robust_scalar_solution(σ, β0_bew, α2_bew) for σ in σ_vals])
β_hat_vals = solutions[:, 0]
ζ_vals = solutions[:, 2]
K_vals = solutions[:, 3]
dep_vals = np.array([
    detection_error_probability(ζ, α_bew)
    for ζ in ζ_vals
])

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))

horizons = np.arange(31)
for σ in [0.0, σ_vals[10], σ_vals[20]]:
    β_hat, P, ζ, K = robust_scalar_solution(σ, β0_bew, α2_bew)
    label = rf'$\sigma={σ:.3f}$, $\zeta={ζ:.3f}$'
    axes[0].plot(horizons, ζ**horizons, lw=2, label=label)

axes[0].set_xlabel('horizon')
axes[0].set_ylabel(r'response of $\mu_{s,t+h}$')
axes[0].set_title('worst-case impulse response')
axes[0].legend(fontsize=8.5)

axes[1].plot(-σ_vals, dep_vals, lw=2, color='C0')
axes[1].axhline(0.2, color='C3', linestyle='--', lw=1.2,
                label='DEP = 0.2')
axes[1].set_xlabel(r'robustness concern $-\sigma$')
axes[1].set_ylabel('detection error probability')
axes[1].set_ylim(0.0, 0.52)
axes[1].set_title('finite-sample detectability')
axes[1].legend(fontsize=8.5)

fig.tight_layout()
plt.show()
```

The left panel shows that the solved worst-case law makes marginal utility more persistent as $\sigma$ becomes more negative.

The right panel computes the DEP from the exact likelihood ratio between the approximating scalar law $\mu_{t+1}=\mu_t+\alpha\epsilon_{t+1}$ and the solved worst-case law $\mu_{t+1}=\zeta(\sigma)\mu_t+\alpha\epsilon_{t+1}$.

### Concluding remarks

We close with a summary of the key messages from all three lectures.

The LQ permanent income model, a rational-expectations version of Friedman's permanent income hypothesis, has two complementary state-space representations:

1. **$(b_t, z_t)$ representation**: emphasises that the consumer's optimal borrowing is history
   dependent and cointegrated with consumption.

2. **$(c_t, z_t)$ representation**: emphasises that consumption is a martingale (random walk)
   and that assets $b_t$ are encoded in consumption, so the impulse response function of
   consumption is "box-shaped": a permanent shift in the level.

We embedded this single-agent model in a Bewley equilibrium with a continuum of ex-post
heterogeneous consumers.

The equilibrium gross interest rate $R = \beta^{-1}$ is supported by
constant average consumption, though the cross-section variance of consumption grows linearly with
age.

A complete-markets version of the same model achieves full risk sharing and a time-invariant
consumption distribution at the cost of more complex financial arrangements (Arrow securities).

A concern for model misspecification, parameterised by $\sigma = -\theta^{-1} \leq 0$, alters the permanent income model.

A concern for robustness generates a precautionary savings motive even under quadratic preferences by distorting the conditional means of income shocks.

The distorted worst-case model makes the income process **more persistent**, shifting power toward low frequencies where the permanent income consumer is most vulnerable.

The observational equivalence theorem {prf:ref}`thm-lqcs-oe1` shows that for quantities $(c_t, i_t)$ alone, a concern for robustness is indistinguishable from a reduction in $\beta$.

The reverse theorem {prf:ref}`thm-lqcs-oe2` shows that, starting from $\beta R = 1$, robustness is observationally equivalent to an *increase* in $\beta$, which imparts an upward drift to expected consumption.

Detection error probabilities provide a principled way to calibrate $\sigma$: choose $|\sigma|$ small enough that the approximating and worst-case models remain difficult to distinguish statistically.

The observationally equivalent $(\sigma, \hat\beta)$ pairs **do** have different implications for asset prices, a point explored further by HST in the asset-pricing context.

The robust Bewley economy shows how agents can have the same consumption decision rule and support the same equilibrium interest rate $R = \beta_0^{-1}$ while differing in their worst-case subjective income dynamics.

## Exercises

```{exercise-start}
:label: lqcs_ex1
```

We translate from the benchmark Bewley economy to HST notation.

Specialise the robust-control setup to the no-habit, no-capital LQ Bewley environment
($\lambda = \delta_h = 0$, $k_t = 0$), and let the endowment process be the two-factor model in
{eq}`eq:pi-twofactor`.

1. Write the household state as $x_t = [a_t, z_t^\top]^\top$, where $a_t=-b_t$ is net assets, and derive matrices $(A, B, C)$ for the law of motion {eq}`eq:law0`.

2. Show that when $\sigma = 0$, the Bellman problem coincides with the LQ permanent-income
  problem.

3. Derive $\alpha^2$ and verify

$$
\alpha^2 = \sigma_1^2 + (1-\beta)^2\sigma_2^2.
$$

Interpret economically why the permanent and transitory components enter with different weights.

```{exercise-end}
```

```{solution-start} lqcs_ex1
:class: dropdown
```

Here is one solution:

1. With $x_t = [a_t, z_t^\top]^\top$ and budget law $a_{t+1} = R(a_t + y_t - c_t)$, $y_t = \check G z_t$, and $z_{t+1} = \check A z_t + \check C \epsilon_{t+1}$, the stacked law is

$$
\begin{pmatrix} a_{t+1} \\ z_{t+1} \end{pmatrix}
=
\underbrace{\begin{pmatrix} R & R\check G \\ 0 & \check A \end{pmatrix}}_{A}
\begin{pmatrix} a_t \\ z_t \end{pmatrix}
+
\underbrace{\begin{pmatrix} -R \\ 0 \end{pmatrix}}_{B} c_t
+
\underbrace{\begin{pmatrix} 0 \\ \check C \end{pmatrix}}_{C}\epsilon_{t+1}.
$$

  The sign of $B$ is negative because higher $c_t$ reduces asset accumulation $a_{t+1}$.

2. At $\sigma=0$, the robust Bellman problem collapses to the ordinary LQ objective with no minimizing distortion term, so the planner/consumer problem is exactly the permanent-income problem with quadratic utility and linear constraints.

3. From the $(c_t,z_t)$ representation,

$$
\Delta c_{t+1} = h\,\epsilon_{t+1},
\qquad h = (1-\beta)\check G (I-\beta\check A)^{-1}\check C.
$$

  In HST notation, $\alpha^2 = h h^\top$, and for the two-factor calibration $\check A=\mathrm{diag}(1,0)$ and $\check C=\mathrm{diag}(\sigma_1,\sigma_2)$, so

$$
\alpha^2 = \sigma_1^2 + (1-\beta)^2\sigma_2^2.
$$

  Permanent shocks get unit weight because they shift lifetime resources one-for-one, while
  transitory shocks are annuitised and therefore scaled by $(1-\beta)$ in consumption growth.

```{solution-end}
```

```{exercise-start}
:label: lqcs_ex2
```

This exercise studies a continuum of robust but observationally equivalent Bewley consumers.

Fix a benchmark pair $(\beta_0, \sigma = 0)$ with $R = \beta_0^{-1}$ and define

$$
\beta(\sigma) = \beta_0 + \frac{\sigma\alpha^2\beta_0}{1-\beta_0},
\qquad \sigma \in [-\bar\sigma, 0].
$$

Suppose a unit interval of consumers is indexed by $i$ with type $\sigma_i \in [-\bar\sigma, 0]$
and discount factor $\beta_i = \beta(\sigma_i)$.

1. Use {prf:ref}`thm-lqcs-oe1` to show that each type has the same consumption rule as the benchmark
  $(\beta_0, 0)$ agent.

2. Prove that aggregate consumption and bond-market clearing imply the same equilibrium interest
  rate $R = \beta_0^{-1}$ as in the plain-vanilla Bewley model.

3. Explain why agents can be observationally equivalent in quantities while still holding different
  worst-case subjective models.

```{exercise-end}
```

```{solution-start} lqcs_ex2
:class: dropdown
```

Here is one solution:

1. {prf:ref}`thm-lqcs-oe1` implies that if $(\sigma_i, \beta_i)$ lies on

$$
\beta_i = \beta_0 + \frac{\sigma_i\alpha^2\beta_0}{1-\beta_0},
$$

  then type $i$ chooses the same decision rule as the benchmark $(0,\beta_0)$ agent and all types share the same consumption policy function $c_t = \mathcal C(a_t,z_t)$.

2. Since all individual policy rules coincide with benchmark Bewley policies, aggregating over consumers gives the same goods- and bond-market clearing conditions and supports the same equilibrium $R=\beta_0^{-1}$.

3. Observational equivalence concerns quantities generated by optimal rules, so distinct $(\sigma_i,\beta_i)$ can generate the same $\{c_t^i,a_t^i\}$ while implying different internal worst-case beliefs.

```{solution-end}
```

```{exercise-start}
:label: lqcs_ex3
```

This exercise separates quantities from beliefs without introducing an additional calibration.

Consider two agents $a$ and $b$ in the robust Bewley economy with $\sigma^a < \sigma^b \leq 0$ and
$\beta^j = \beta_0 + \sigma^j\alpha^2\beta_0/(1-\beta_0)$ for $j \in \{a,b\}$.

1. Use {eq}`eq:bew_cinno` and {eq}`eq:bew_locus` to show that the two agents have the same consumption innovation $h\epsilon_{t+1}$.

2. Show that if the two agents start from the same $(a_t,z_t)$ and observe the same shock
  $\epsilon_{t+1}$, then their next-period choices of consumption and assets coincide.

3. Explain why the two agents can nevertheless disagree about the worst-case conditional mean of
  $\epsilon_{t+1}$.

Summarise what is and is not identified by data on quantities alone.

```{exercise-end}
```

```{solution-start} lqcs_ex3
:class: dropdown
```

Here is one solution:

1. Equation {eq}`eq:bew_locus` places both agents on the observational-equivalence locus, so
{prf:ref}`thm-lqcs-oe1` implies that both use the benchmark consumption rule and therefore the same
innovation vector $h$ in {eq}`eq:bew_cinno`.

2. With a common state and common shock, both agents apply the same policy function and the same law
of motion, so $c_{t+1}^a=c_{t+1}^b$ and $a_{t+1}^a=a_{t+1}^b$.

3. The minimizing feedback $K(\sigma^j,\beta^j)$ can differ across $j$, so the agents can attach
different worst-case conditional means to the same shock process even though their observable
choices coincide.

Conclusion: quantities identify the equilibrium decision rule but not the decomposition between
impatience ($\beta$) and robustness ($\sigma$) along the observational-equivalence locus.

```{solution-end}
```
