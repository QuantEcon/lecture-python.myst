---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(social_security_depletion)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Social Security Trust Fund Depletion

```{contents} Contents
:depth: 2
```

## Overview

The Social Security trustees project that the reserves of the Old-Age and Survivors Insurance trust fund will run out during $2032$ {cite}`Trustees2026`.

The program cannot borrow.

Unless Congress acts, current law then reduces every benefit by a common factor $\phi_t \le 1$ that makes the program's cash flow balance: a worker entitled to a scheduled benefit $b^{sch}$ collects $\phi_t b^{sch}$.

We call $\phi_t$ the **payable factor**.

It is not a policy parameter that somebody chooses.

It is an equilibrium object, because the quantity it scales depends on decisions that workers take while looking ahead at it:

* a worker who expects a smaller pension works longer and saves more
* a worker inside the claiming window just before the cut has a reason to file early, while benefits are still paid in full
* a worker who expects to live longer has a reason to file late, because the actuarial reward for waiting is fixed by statute

Those decisions determine taxable payroll and the scheduled benefit bill, which are exactly the two quantities that determine $\phi_t$.

This lecture studies that fixed point, following {cite:t}`SargentShi2026`.

We write down the ledger that current law prescribes, define the map that sends a path for the payable factor into another path for the payable factor, and compute its fixed point in a quantitative overlapping generations model with

* a life cycle from age $20$ to age $100$ with projected mortality
* permanent heterogeneity in earnings ability and in the taste for work
* the statutory benefit formula, the actuarial adjustment for the claim age, the taxable maximum, and the taxation of benefits
* retirement and benefit claiming as **separate** irreversible decisions

The exercises then put the machinery to work: on the oscillation that a decreasing map produces, on raising the full retirement age, on what it would cost to guarantee scheduled benefits instead, and on how much of the long-run cut is the work of rising life expectancy.

Several symbols recur, so we record them here.

| Symbol | Meaning |
|---|---|
| $\phi_t$ | payable factor: the common fraction of scheduled benefits actually paid |
| $F_t$, $r^F$ | trust fund balance and the rate credited on it |
| $N_t$ | population; every other aggregate is per capita |
| $\mathit{Pay}_t$, $B^{sch}_t$ | per capita taxable payroll and the per capita scheduled benefit bill |
| $\tau^{ss}$, $\kappa_b$, $\bar y$ | payroll tax rate, tax rate on benefits received, taxable maximum |
| $\mathrm{AIME}$, $\mathrm{PIA}$, $\psi$ | the earnings record, the benefit it buys, and the rate at which the record accrues |
| $\mathrm{AF}(\chi)$, FRA, $\varsigma$ | actuarial factor at claim age $\chi$, the full retirement age, and a benefit scale |
| $R$, $\chi$ | a worker's retirement age and claim age |
| $j$, $t$ | age index ($0$ at age $20$) and calendar year |
| $e$, $h$ | permanent earnings ability and taste for work |
| $\ell_j$, $\Omega$ | survival from age $20$ to age $20+j$, and $\sum_j \beta^j\ell_j$ |
| $\tilde R$, $a_0$, $\mathcal W$ | after-tax gross return, assets on hand, and lifetime resources |

This lecture is a companion to {doc}`two_computation`, which studies two ways of moving from an unfunded to a fully funded system in the model of {cite:t}`huang1997two`.

Here nobody moves to a funded system: we ask what current law does on its own, and what it would cost to stop it.

Two other lectures supply background: {doc}`ak2` introduces the {cite:t}`auerbach1987dynamic` overlapping generations framework, and {doc}`ak_aiyagari` adds uninsurable idiosyncratic risk to it.

We use the following imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.optimize import fsolve, brentq
```

## The institution

Social Security's ledger is a stock, $F_t$, and two flows.

The fund earns an administered rate $r^F$ on the special issue Treasury securities it holds, it receives the payroll tax, and it pays benefits.

Writing $N_t$ for the population, $\mathit{Pay}_t$ for per capita taxable payroll, and $B^{sch}_t$ for the per capita scheduled benefit bill, the ledger evolves as

```{math}
:label: ssd_fund
F_{t+1} = (1+r^F)F_t + N_t\left(\mathit{Rev}_t - \phi_t B^{sch}_t\right),
\qquad F_t \ge 0 ,
```

where program revenue is the payroll tax plus the tax that returns to the program on the benefits it pays,

```{math}
:label: ssd_rev
\mathit{Rev}_t = \tau^{ss}\mathit{Pay}_t + \kappa_b \phi_t B^{sch}_t .
```

The restriction $F_t \ge 0$ in {eq}`ssd_fund` is the law: the program may not borrow.

It is what forces $\phi_t$ below one.

The program pays the largest common fraction of scheduled benefits that its resources allow,

```{math}
:label: ssd_phi
\phi_t = \min\left\{1, \;
\frac{(1+r^F)F_t + N_t\tau^{ss}\mathit{Pay}_t}
     {N_t(1-\kappa_b)B^{sch}_t}\right\} .
```

The numerator is what the program can draw on: the balance it carries in plus the payroll tax it collects.

The denominator is what it owes, net of the benefit tax that comes back to it.

We date depletion at the first year the fund is empty,

$$
t^{dep} = \min\{t : F_t = 0\} ,
$$

and we call $t^{dep}-1$ the **first shortfall year**, the year in which the program pays out its last reserves and the first year in which $\phi_t < 1$.

```{note}
The trustees date "reserve depletion" to the year the reserves run out, which is the first shortfall year here.

Their projected $2032$ is therefore our $t^{dep}-1$, and the first year in which the whole benefit is cut is $2033$.
```

From $t^{dep}$ on the fund is empty and {eq}`ssd_phi` collapses to a cash flow condition,

```{math}
:label: ssd_phiflow
\phi_t = \frac{\tau^{ss}\mathit{Pay}_t}{(1-\kappa_b)B^{sch}_t},
\qquad t \ge t^{dep} .
```

### The payable factor is a fixed point

Equation {eq}`ssd_phiflow` looks like a formula for $\phi_t$, but both $\mathit{Pay}_t$ and $B^{sch}_t$ are built from decisions that workers make while looking at the whole future path $\{\phi_s\}$.

Let $\Phi$ be the map that takes a candidate path $\{\phi_s\}_{s\ge 0}$, hands it to workers, collects the taxable payroll and scheduled benefits that their optimal decisions generate, runs the ledger {eq}`ssd_fund`, and returns the payable path {eq}`ssd_phi` that the ledger implies.

An equilibrium payable path is a fixed point,

```{math}
:label: ssd_fixedpoint
\{\phi_t\} = \Phi\big(\{\phi_t\}\big) .
```

```{important}
$\Phi$ is a **decreasing** map.

Raise the candidate path and benefits become more valuable, so workers claim earlier and retire earlier; taxable payroll falls and the scheduled bill rises; {eq}`ssd_phi` then returns a *smaller* number.

Successive guesses therefore alternate around the fixed point rather than approaching it from one side.

Whether the alternation dies out depends on how strongly workers respond, and {ref}`ssd_ex1` finds dates at which it does not: the undamped iteration settles into a two-cycle and stays there.

Every computation below therefore takes a damped step.
```

One consequence of {eq}`ssd_phiflow` is worth recording before we compute anything.

In a long run in which the fund is empty and the economy has settled down, the payable factor is pinned by the program's **cost rate**, the ratio of scheduled benefits to taxable payroll:

```{math}
:label: ssd_longrun
\phi_\infty = \frac{\tau^{ss}}{(1-\kappa_b)}\cdot
\frac{\mathit{Pay}_\infty}{B^{sch}_\infty} .
```

Everything demographic and behavioral that matters for the long-run benefit cut reaches it through that one ratio.

## A model we can solve quickly

{cite:t}`SargentShi2026` compute general equilibrium transition paths of an economy with idiosyncratic earnings risk, a neoclassical firm, a consolidated government budget constraint, and a cross-section of $80{,}000$ workers per age.

That calculation takes about $22$ hours on an A100 GPU.

The model below keeps the institution exactly and strips the economy down until the whole fixed point runs in about a minute on a laptop.

```{admonition} What this lecture leaves out
:class: warning

* **Prices are fixed.** The interest rate and the wage do not respond. In the paper, capital deepens as workers save against a smaller pension, the capital--output ratio rises from $3.00$ to $3.19$, and the interest rate falls from $6.0$ to $5.3$ percent.
* **There is no government budget constraint outside the program.** In the paper, depletion removes the Social Security deficit from the consolidated budget, so the labor tax *falls* from $0.220$ to $0.190$: current law delivers a benefit cut to retirees and a tax cut to workers in the same year.
* **There is no idiosyncratic earnings risk and no borrowing constraint**, so the life cycle has a closed form. Workers differ only in permanent ability and in their taste for work.
* **Annuity markets are complete**, so there are no accidental bequests and no bequest motive.
* **Benefits are scaled rather than extended.** The program's spousal and survivor benefits go to additional beneficiaries; here a single scale factor on each worker's own benefit matches the aggregate cost instead, which raises individual replacement rates by about a seventh.
* **Demography is stylized**, a smooth birth path and a Gompertz mortality table calibrated to the trustees' dependency ratios, rather than the published population projection.

What survives is the part this lecture is about: an endogenous payable factor, and claiming and retirement decisions that respond to it.
```

### Parameters

Units are the average wage index of $2025$, so one model unit is $\$72{,}025$ and mean earnings per worker are one.

The statutory amounts -- the taxable maximum, the two bend points of the benefit formula, the payroll tax rate, the actuarial factors, the trust fund balance -- take their legislated or recorded $2025$ values.

The tax rates on labor, capital and consumption follow {cite:t}`kitao2014`, and the discount factor, the interest rate, the dispersion of earnings ability and the rate credited on the fund are those of {cite:t}`SargentShi2026`.

Two parameters are chosen inside this model, and we say which below.

```{code-cell} ipython3
AWI = 72025.07          # average wage index, 2025: dollars per model unit
AGE0, J = 20, 81        # ages 20 ... 100
Y0, T = 2026, 120       # first transition year, horizon
YEARS = np.arange(Y0, Y0+T+1)

Params = namedtuple('Params', [
    'β', 'η', 'r', 'τ_l', 'τ_k', 'τ_c', 'τ_ss', 'κ_b', 'r_F',
    'ycap', 'bend', 'prate', 'FRA', 'ψ', 'σ_a', 'σ_η', 'g_m', 'bscale', 'F_ratio'])

par = Params(
    β=0.9739,            # discount factor
    η=0.81,              # disutility of work (calibrated)
    r=0.060,             # interest rate
    τ_l=0.22, τ_k=0.30, τ_c=0.05,          # labor, capital, consumption taxes
    τ_ss=0.106,          # OASI payroll tax rate
    κ_b=0.03891,         # tax on benefits received
    r_F=0.02,            # rate credited on the trust fund
    ycap=176100/AWI,     # taxable maximum
    bend=np.array([1226.0, 7391.0])*12/AWI,   # PIA bend points
    prate=np.array([0.90, 0.32, 0.15]),       # PIA replacement rates
    FRA=67,              # full retirement age
    ψ=0.055,             # AIME accrual (a 35 year moving average)
    σ_a=0.315,           # dispersion of earnings ability
    σ_η=0.16,            # dispersion of the taste for work
    g_m=0.007,           # annual mortality improvement
    bscale=1.145,        # benefit scale ς (calibrated)
    F_ratio=1.6255)      # 2025 reserves relative to benefit payments

def after_tax_R(p):
    "The gross after-tax return on saving."
    return 1 + p.r*(1 - p.τ_k)
```

The benefit formula has two pieces.

The **primary insurance amount** is a concave, piecewise linear function of a worker's average indexed earnings, with marginal replacement rates $0.90$, $0.32$ and $0.15$ on the three segments.

The **actuarial factor** $\mathrm{AF}(\chi)$ multiplies it, and depends on the age $\chi$ at which the worker files: the statutory rule reduces benefits by $5/9$ of one percent per month for the first $36$ months of early claiming and $5/12$ of one percent per month beyond that, and adds a delayed retirement credit of $8$ percent a year after the full retirement age.

```{code-cell} ipython3
def claim_factors(fra=67, lo=62, hi=70):
    "Statutory actuarial factors AF(χ) for claim ages lo ... hi."
    out = []
    for a in range(lo, hi+1):
        if a <= fra:
            m = (fra - a)*12
            out.append(round(1 - (min(m, 36)*(5/9) + max(m-36, 0)*(5/12))/100, 4))
        else:
            out.append(round(1 + 0.08*(a - fra), 4))
    return np.array(out)

def pia(aime, p):
    "The primary insurance amount: a concave piecewise-linear function of AIME."
    s0 = np.minimum(aime, p.bend[0])
    s1 = np.clip(aime - p.bend[0], 0.0, p.bend[1] - p.bend[0])
    s2 = np.maximum(aime - p.bend[1], 0.0)
    return p.prate[0]*s0 + p.prate[1]*s1 + p.prate[2]*s2

X_GRID = np.arange(62, 71)
print("claim age     ", X_GRID)
print("AF(χ) at FRA 67", claim_factors(67))
print("AF(χ) at FRA 69", claim_factors(69))
```

{numref}`fig-ssd-benefit-rules` shows the progressivity of the program on the left and its price of waiting on the right.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The benefit formula is progressive and the reward for waiting is statutory
    name: fig-ssd-benefit-rules
---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

grid = np.linspace(0, 3, 400)
axes[0].plot(grid, pia(grid, par), lw=2)
for b in par.bend:
    axes[0].axvline(b, color='k', ls=':', lw=1)
axes[0].set_xlabel('AIME, multiples of the average wage')
axes[0].set_ylabel('primary insurance amount')
axes[0].set_title('the benefit formula, with its bend points')

axes[1].plot(X_GRID, claim_factors(67), 'o-', lw=2, label='full retirement age 67')
axes[1].plot(X_GRID, claim_factors(69), 's-', lw=2, label='full retirement age 69')
axes[1].axhline(1.0, color='k', lw=0.8)
axes[1].set_xlabel('claim age $\\chi$')
axes[1].set_ylabel('actuarial factor $\\mathrm{AF}(\\chi)$')
axes[1].set_title('the reward for waiting')
axes[1].legend()
plt.tight_layout()
plt.show()
```

A worker whose earnings record is below the first bend point replaces $90$ cents of every dollar; one above the second replaces $15$.

Waiting from $62$ to $70$ raises the benefit by $77$ percent, and the schedule is statutory: it does not move when mortality improves, and it does not move when the payable factor falls.

{ref}`ssd_ex2` raises the full retirement age, which is the orange line.

### Demography

Mortality follows a Gompertz--Makeham force of mortality that improves at $0.7$ percent a year until $2085$ and is frozen afterwards, so that the economy settles into a stationary equilibrium inside the horizon.

Cohort sizes grow at one rate before $1990$ and another after, and we pick the two rates so that the model reproduces the trustees' aged dependency ratio in $2025$ and in $2085$: the population aged $65$ and over per person aged $20$ to $64$ rises from $0.32$ to $0.50$.

```{code-cell} ipython3
def mortality(age, year, p):
    "Force of mortality by age and calendar year; improvement stops in 2085."
    yr = np.minimum(year, 2085)
    return (5e-4 + 2.5e-5*np.exp(0.095*age))*np.exp(-p.g_m*(yr - 2025))

def cohort_survival(entry_years, p):
    "ℓ[c, j]: probability of reaching age 20+j, cohort by cohort."
    ent = np.asarray(entry_years, float).reshape(-1, 1)
    j = np.arange(J).reshape(1, -1)
    s = np.exp(-mortality(AGE0 + j, ent + j, p))
    s[:, -1] = 0.0
    return np.concatenate([np.ones((len(ent), 1)),
                           np.cumprod(s[:, :-1], axis=1)], axis=1)

YS = 1990          # the year the growth rate of entering cohorts changes

def entry_sizes(g, years):
    "Size of the cohort that turns 20 in each year."
    y = np.asarray(years, float)
    return np.where(y <= YS, np.exp(g[0]*(y - YS)), np.exp(g[1]*(y - YS)))

def population(g, years, p):
    "N[j, t]: people of age 20+j in each calendar year."
    years = np.atleast_1d(years)
    out = np.zeros((J, len(years)))
    for j in range(J):
        ent = years - j
        out[j] = entry_sizes(g, ent)*cohort_survival(ent, p)[:, j]
    return out

def dependency(g, t, p):
    "People aged 65 and over per person aged 20 to 64."
    n = population(g, [t], p)[:, 0]
    return n[45:].sum()/n[:45].sum()

g_pop = fsolve(lambda g: [dependency(g, 2025, par) - 0.32,
                          dependency(g, 2085, par) - 0.50], [0.02, 0.0])

print(f"entry cohorts grow at {g_pop[0]:.4f} before {YS} and {g_pop[1]:.4f} after")
for t in (2025, 2035, 2055, 2085):
    print(f"   dependency ratio in {t}: {dependency(g_pop, t, par):.3f}")
print(f"   life expectancy at 20 of the cohort of 2026: "
      f"{20 + cohort_survival([2026], par)[0].sum():.1f}")
```

{numref}`fig-ssd-demography` is the engine of everything that follows, so it is worth looking at.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Cohorts stopped growing and lives keep lengthening
    name: fig-ssd-demography
---
span = np.arange(2000, 2121)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(span, [dependency(g_pop, t, par) for t in span], lw=2)
axes[0].plot([2025, 2085], [0.32, 0.50], 'o', ms=7, color='C1',
             label='trustees')
axes[0].set_xlabel('year')
axes[0].set_ylabel('people 65 and over per person 20 to 64')
axes[0].legend()

for t, style in ((2026, '-'), (2085, '--')):
    n_t = population(g_pop, [t], par)[:, 0]
    axes[1].plot(AGE0 + np.arange(J), n_t/n_t.sum(), style, lw=2, label=str(t))
axes[1].set_xlabel('age')
axes[1].set_ylabel('share of the population')
axes[1].legend()
plt.tight_layout()
plt.show()
```

Two forces push the dependency ratio up, and the model carries both.

Cohorts entering the labor force stopped growing, so the base of the pyramid stops widening.

Mortality keeps improving, so its top keeps filling in.

{ref}`ssd_ex4` separates them.

### Earnings and types

Earnings are the wage times a hump shaped age-efficiency profile that peaks at age $50$, times a permanent ability draw.

Workers also differ in their taste for work.

Both are five point discretizations of lognormal distributions, so there are $25$ types.

```{code-cell} ipython3
def efficiency():
    "Age-efficiency profile, normalized to mean one over ages 20 to 64."
    j = np.arange(J)
    e = np.exp(0.0583*j - 0.000972*j**2)
    return e/e[:45].mean()

ε = efficiency()

def types(p, n=5):
    "Permanent earnings ability and taste for work, on a tensor grid."
    x, w = np.polynomial.hermite_e.hermegauss(n)
    w = w/w.sum()
    e = np.exp(p.σ_a*x); e = e/(w*e).sum()
    h = np.exp(p.σ_η*x)
    E, H = np.meshgrid(e, h, indexing='ij')
    return E.ravel(), H.ravel(), np.outer(w, w).ravel()
```

Earnings rise with experience, peak at age $50$, and decline thereafter.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Earnings peak at age 50 and decline thereafter
    name: fig-ssd-efficiency
---
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(AGE0 + np.arange(J), ε, lw=2)
ax.set_xlabel('age')
ax.set_ylabel('efficiency units of labor')
ax.set_xlim(20, 80)
plt.tight_layout()
plt.show()
```

### The worker's problem

A worker chooses when to stop working and when to file for benefits.

Both decisions are irreversible, and they are separate: {cite:t}`coileDiamondGruberJousten2002` and the literature that follows them establish that leaving the labor force and claiming a benefit are distinct events.

Write $R$ for the retirement age, $\chi$ for the claim age, and $\ell_j$ for the probability of surviving from age $20$ to age $20+j$.

A worker of type $(e, h)$ ranks plans by

```{math}
:label: ssd_utility
\sum_{j} \beta^j \ell_j
\big[\log c_j - \eta\, h\, n_j \big],
\qquad n_j = \mathbf 1\{j < R - 20\} ,
```

where $e$ is permanent earnings ability, $h$ is the taste for work, and $\eta h$ is therefore the worker's own disutility of a year of work.

We normalize the wage to one.

While working, a worker earns $y_j = \varepsilon_j e$, pays the labor tax and the payroll tax on earnings up to the taxable maximum $\bar y$, and accumulates the earnings record

$$
\mathrm{AIME}_{j+1} = \mathrm{AIME}_j
 + \psi\,\mathbf 1\{j < R-20\}\big(\min(y_j, \bar y) - \mathrm{AIME}_j\big),
$$

an exponentially weighted moving average whose effective window of $2/\psi - 1 \approx 35$ years approximates the statutory highest-35-years rule.

Once filed, the worker collects

$$
b_j = \varsigma\,\phi_t\, \mathrm{AF}(\chi)\,\mathrm{PIA}(\mathrm{AIME}_j) ,
$$

taxed at $\kappa_b$, where $\varsigma$ is a benefit scale that the calibration below sets.

It stands in for the spousal and survivor benefits that a model of single workers leaves out.

With complete annuity markets and log utility the consumption problem has a closed form.

Annuities pay the gross return $\tilde R/s_j$ to a survivor, where $\tilde R = 1 + r(1-\tau_k)$ and $s_j$ is the one-period survival probability, because the balances of those who die are shared among those who live.

Compounding those returns turns survival into a discount factor: a unit of income received at age $20+j$ is worth $\ell_j \tilde R^{-j}$ at age $20$, and the lifetime budget constraint is

```{math}
:label: ssd_budget
(1+\tau_c)\sum_j \ell_j \tilde R^{-j} c_j
= a_0 + \sum_j \ell_j \tilde R^{-j} m_j(R, \chi) \equiv \mathcal W(R,\chi) ,
```

where $m_j$ is net income at age $j$: earnings net of the two taxes while working, and the payable benefit net of its own tax once filed.

Because the annuity return and the survival probability cancel, the Euler equation is the one from a model without mortality,

$$
c_j = c_0 (\beta \tilde R)^j .
$$

Substituting it into {eq}`ssd_budget` gives $(1+\tau_c)c_0\Omega = \mathcal W$ with $\Omega = \sum_j \beta^j\ell_j$, and substituting it into {eq}`ssd_utility` gives

$$
\sum_j \beta^j \ell_j \log c_j
= \Omega \log c_0 + \log(\beta\tilde R)\sum_j j\,\beta^j \ell_j .
$$

The second term does not depend on the plan, and neither does the $\log[(1+\tau_c)\Omega]$ that comes out of $c_0 = \mathcal W/[(1+\tau_c)\Omega]$.

Dropping both leaves a two-dimensional discrete choice:

```{math}
:label: ssd_objective
\max_{R,\;\chi}\;\;
\Omega \log \mathcal W(R,\chi) \;-\; \eta\, h \sum_{j < R-20} \beta^j \ell_j .
```

The first term is the present value of everything the plan delivers, including the whole future path of the payable factor.

The second is the cost of the extra years of work.

A worker who is already alive when the news arrives solves the same problem from where it stands.

Write $j_0$ for its age index in $2026$.

Then both sums in {eq}`ssd_objective` run from $j_0$ rather than from $0$, with $\ell_j$ and $\beta^j$ measured from $j_0$; $a_0$ in {eq}`ssd_budget` is the wealth it has accumulated by then; and the decisions it has already taken are not up for reconsideration.

That is what the arguments `j0`, `a0` and `plan0` carry in the code below.

```{note}
The paper also models the **retirement earnings test**, which withholds the benefits of a claimant who keeps working before the full retirement age and then credits them back by advancing the effective claim age.

For a worker who leaves work at or before the full retirement age the credit matches the withholding one for one, so filing earlier than retirement is payoff neutral in this deterministic model.

We therefore impose $\chi \ge \min(R, 70)$ and drop the test.

Claiming remains a separate decision: a worker may file at retirement or wait, up to age $70$.

The restriction does bind on one margin.

A worker who means to work past the full retirement age could file at $62$, have the benefits withheld until the test stops applying, and then collect at the credited factor while still working.

That option is not quite equivalent to filing later, and this model rules it out.
```

We enumerate the admissible plans and index every cohort alive along the transition.

```{code-cell} ipython3
R_GRID = np.arange(58, 73)
# a worker files no earlier than it stops working, and no later than 70
PLANS = np.array([(R, X) for R in R_GRID for X in X_GRID if X >= min(R, 70)])
R_P, X_P = PLANS[:, 0], PLANS[:, 1]
NP = len(PLANS)

ENTRY = np.arange(Y0-J+1, Y0+T+1)      # entry year of every cohort ever alive
NC = len(ENTRY)
ENT_IX = {y: i for i, y in enumerate(ENTRY)}
IDX = np.array([[ENT_IX[t-j] for t in YEARS] for j in range(J)])

print(f"{NP} admissible plans, {NC} cohorts")
```

The retirement grid runs from $58$ to $72$, and it binds for the two most extreme tastes for work, which carry $1.1$ percent of the population each.

Widening it to $50$ to $80$ moves those two types to $56$ and $73$ and moves nothing else: the depletion year, both mean ages, and the payable factor to four decimals are the same.

We keep the narrow grid, which is about twice as fast.

Everything that does not depend on the payable path is built once.

```{code-cell} ipython3
class World:
    "Demography, types, survival and discounting: fixed along the iteration."

    def __init__(self, p, g):
        self.p = p
        e, h, w = types(p)
        self.NE = len(e)
        self.C = NC*self.NE
        self.entry = np.repeat(ENTRY, self.NE)
        self.e = np.tile(e, NC)
        self.η = np.tile(h, NC)
        self.w = np.tile(w, NC)
        self.j0 = np.clip(Y0 - self.entry, 0, J-1)      # age index in 2026
        self.ell = np.repeat(cohort_survival(ENTRY, p), self.NE, axis=0)
        self.disc = self.ell/after_tax_R(p)**np.arange(J)
        self.AF = claim_factors(p.FRA)
        bet_ell = (p.β**np.arange(J))*self.ell
        self.tail = np.cumsum(bet_ell[:, ::-1], axis=1)[:, ::-1]
        self.pre = np.concatenate([np.zeros((self.C, 1)),
                                   np.cumsum(bet_ell, axis=1)], axis=1)
        self.N = population(g, YEARS, p)
        self.Npop = self.N.sum(axis=0)
        self.share = self.N/self.Npop

def phi_of(φ, year):
    "The payable factor: one before 2026, constant beyond the horizon."
    k = np.clip(year - Y0, 0, len(φ)-1)
    return np.where(year < Y0, 1.0, np.asarray(φ)[k])
```

The life cycle itself is one loop over ages.

```{code-cell} ipython3
def lifecycle(W, φ, Rj, Xa, j0, disc, keep=False):
    """Discounted net income from age index j0 on, for plans (Rj, Xa).

    With keep=True also returns taxable payroll and scheduled benefits at
    every age, which is what aggregation needs.
    """
    p = W.p
    nd = np.ndim(Rj)
    e_row = W.e.reshape(-1, *[1]*(nd-1))
    ent = W.entry.reshape(-1, *[1]*(nd-1))
    shape = np.broadcast(Rj, Xa, e_row).shape
    aime = np.zeros(shape)
    S = np.zeros(shape)
    AFx = np.interp(Xa, X_GRID, W.AF)*np.ones(shape)
    pay, ben = [], []
    for j in range(J):
        work = (j < Rj)*np.ones(shape)
        y = ε[j]*e_row*work
        capped = np.minimum(y, p.ycap)*work
        b = p.bscale*AFx*pia(aime, p)*((AGE0 + j >= Xa)*np.ones(shape))
        inc = ((1-p.τ_l)*y - p.τ_ss*capped
               + (1-p.κ_b)*phi_of(φ, ent + j)*b)
        S = S + np.where(j >= j0, disc[..., j]*inc, 0.0)
        if keep:
            pay.append(capped)
            ben.append(b)
        aime = aime + p.ψ*work*(capped - aime)
    return (S, np.array(pay), np.array(ben)) if keep else S

def solve_plans(W, φ, j0, plan0=None, a0=0.0):
    """The plan that maximizes the worker's objective, for every cohort and type."""
    p, C = W.p, W.C
    j0 = np.asarray(j0)
    S = lifecycle(W, φ, (R_P-AGE0).reshape(1, NP), X_P.reshape(1, NP),
                  j0.reshape(C, 1), W.disc.reshape(C, 1, J))
    scale = (after_tax_R(p)**j0)/W.ell[np.arange(C), j0]
    wealth = a0 + S*scale.reshape(C, 1)
    norm = (p.β**j0)*W.ell[np.arange(C), j0]
    Om = W.tail[np.arange(C), j0]/norm
    Dw = (W.pre[np.arange(C)[:, None], np.clip(R_P-AGE0, 0, J)[None, :]]
          - W.pre[np.arange(C), j0][:, None])/norm[:, None]
    V = (Om[:, None]*np.log(np.maximum(wealth, 1e-9))
         - (p.η*W.η)[:, None]*np.maximum(Dw, 0.0))
    if plan0 is not None:
        # decisions already taken by 2026 cannot be undone
        R0, X0 = plan0
        age26 = W.j0 + AGE0
        alive = W.entry < Y0
        ok = np.where((alive & (R0 <= age26))[:, None],
                      R_P[None, :] == R0[:, None],
                      R_P[None, :] >= np.where(alive, age26, 0)[:, None])
        ok &= np.where((alive & (X0 <= age26))[:, None],
                       X_P[None, :] == X0[:, None],
                       X_P[None, :] >= np.where(alive, np.maximum(62, age26), 62)[:, None])
        V = np.where(ok, V, -np.inf)
    k = V.argmax(axis=1)
    return R_P[k], X_P[k], wealth[np.arange(C), k], Om, Dw[np.arange(C), k]

def flows(W, φ, R, X, j0):
    "Taxable payroll and scheduled benefits at the chosen plans, shape (J, C)."
    _, pay, ben = lifecycle(W, φ, (R-AGE0), X, np.asarray(j0), W.disc, keep=True)
    return pay, ben
```

## Aggregation and the ledger

Aggregates are population weighted averages across ages, and the operator $\Phi$ is the fund law {eq}`ssd_fund` together with {eq}`ssd_phi`.

```{code-cell} ipython3
def aggregate(W, pay, ben):
    "Per capita taxable payroll and scheduled benefits in each calendar year."
    wt = W.w.reshape(NC, W.NE)[0]           # the type weights, tiled across cohorts
    Pj = pay.reshape(J, NC, W.NE) @ wt
    Bj = ben.reshape(J, NC, W.NE) @ wt
    Pay = (np.take_along_axis(Pj, IDX, axis=1)*W.share).sum(axis=0)
    Ben = (np.take_along_axis(Bj, IDX, axis=1)*W.share).sum(axis=0)
    return Pay, Ben

def fund_and_phi(W, Pay, Ben, F0):
    "The ledger and the payable factor: one pass of the operator Φ."
    p = W.p
    F = np.zeros(len(YEARS)+1)
    F[0] = F0
    φ = np.ones(len(YEARS))
    for t in range(len(YEARS)):
        obligations = W.Npop[t]*(1-p.κ_b)*Ben[t]
        resources = (1+p.r_F)*F[t] + W.Npop[t]*p.τ_ss*Pay[t]
        φ[t] = min(1.0, resources/obligations) if obligations > 0 else 1.0
        rev = p.τ_ss*Pay[t] + p.κ_b*φ[t]*Ben[t]
        F[t+1] = max((1+p.r_F)*F[t] + W.Npop[t]*(rev - φ[t]*Ben[t]), 0.0)
    return F, φ
```

## The economy before the news

Workers alive in $2026$ made their plans expecting benefits to be paid in full forever.

We solve that economy first.

It supplies two things that the transition needs: the retirement and claim decisions that cohorts have **already executed** by $2026$, and the assets they carry into $2026$.

It also supplies the scale for the third, the opening trust fund balance, which we set below to the recorded end-of-$2025$ reserves of $1.63$ times annual benefit payments.

```{code-cell} ipython3
def income_path(W, φ, R, X, rows=slice(None)):
    "Net income at every age at the chosen plans, for all rows or a subset."
    p = W.p
    e, entry, R, X = W.e[rows], W.entry[rows], R[rows], X[rows]
    aime = np.zeros(len(e))
    out = np.zeros((J, len(e)))
    AFx = np.interp(X, X_GRID, W.AF)
    for j in range(J):
        work = (j < R - AGE0).astype(float)
        y = ε[j]*e*work
        capped = np.minimum(y, p.ycap)*work
        b = p.bscale*AFx*pia(aime, p)*((AGE0 + j >= X).astype(float))
        out[j] = ((1-p.τ_l)*y - p.τ_ss*capped
                  + (1-p.κ_b)*phi_of(φ, entry + j)*b)
        aime = aime + p.ψ*work*(capped - aime)
    return out

def pre_transition(W):
    "Plans, assets at 2026, and aggregates when φ ≡ 1 forever."
    p = W.p
    φ1 = np.ones(T+1)
    z = np.zeros(W.C, int)
    R, X, wealth, Om, _ = solve_plans(W, φ1, z)
    pay, ben = flows(W, φ1, R, X, z)
    Pay, Ben = aggregate(W, pay, ben)
    c0 = wealth/((1+p.τ_c)*Om)
    inc = income_path(W, φ1, R, X)
    a = np.zeros(W.C)
    A = np.zeros((J, W.C))
    for j in range(J-1):
        A[j] = a
        s = W.ell[:, j+1]/np.maximum(W.ell[:, j], 1e-12)
        a = (after_tax_R(p)/np.maximum(s, 1e-12))*(
            a + inc[j] - (1+p.τ_c)*c0*(p.β*after_tax_R(p))**j)
    A[J-1] = a
    return R, X, A[W.j0, np.arange(W.C)], Pay, Ben, wealth, Om

W = World(par, g_pop)
R_pre, X_pre, a26, Pay_pre, Ben_pre, wealth_pre, Om_pre = pre_transition(W)

mask = W.entry == 2026
wts = W.w[mask]/W.w[mask].sum()
print(f"cohort entering in 2026:  mean retirement age {(R_pre[mask]*wts).sum():.2f}, "
      f"mean claim age {(X_pre[mask]*wts).sum():.2f}")
print(f"cost rate in 2026: {Ben_pre[0]/Pay_pre[0]:.4f}"
      f"   (trustees for 2025: 0.1362)")
```

Two parameters were chosen to make those two numbers come out.

The disutility of work $\eta$ sets the retirement age: the cohort entering in $2026$ retires at $64.8$ on average, against the median retirement age of $64.5$ that {cite:t}`SargentShi2026` target.

The benefit scale $\varsigma$ sets the cost rate, and brings it within about a tenth of a percentage point of the trustees' $2025$ figure.

The claim age is not targeted, and the model misses it by more: this cohort files at $66.5$ on average, a year and a half later than the $65.0$ the paper matches.

Claiming later than the data is a bias to keep in mind when reading the transition below, where the claim age is one of the two margins doing the work.

Two more statistics that nothing was fitted to are worth printing, because both bear on how hard the model pushes the two margins.

```{code-cell} ipython3
def replacement_rate(W, R, X):
    "The first benefit received, relative to earnings in the last year of work."
    p = W.p
    aime, last, first = (np.zeros(W.C) for _ in range(3))
    AFx = np.interp(X, X_GRID, W.AF)
    for j in range(J):
        work = (j < R - AGE0).astype(float)
        y = ε[j]*W.e*work
        capped = np.minimum(y, p.ycap)*work
        last = np.where(R - AGE0 - 1 == j, y, last)
        first = np.where(X - AGE0 == j, p.bscale*AFx*pia(aime, p), first)
        aime = aime + p.ψ*work*(capped - aime)
    return first/np.maximum(last, 1e-12)

rr = (replacement_rate(W, R_pre, X_pre)[mask]*wts).sum()
print(f"replacement rate, cohort entering in 2026:      {rr:.3f}")
print(f"   at the statutory formula, without ς:         {rr/par.bscale:.3f}")

shares = {int(a): float(W.w[mask][X_pre[mask] == a].sum()/W.w[mask].sum())
          for a in X_GRID}
print("   claim ages chosen:",
      {a: round(v, 3) for a, v in shares.items() if v > 0.005})
```

The first number exposes something that the benefit scale $\varsigma$ does quietly.

Spousal and survivor benefits are paid to *additional people*; here the whole $14.5$ percent lands on each worker's own benefit.

So a worker in this model collects about a seventh more than the statutory formula would pay it, and the incentives to retire and to file are correspondingly stronger.

Matching the program's aggregate cost with a model of single workers has to distort something, and this is where it goes.

The second number exposes the other limitation.

Claiming piles up on a single age, and nobody files at $62$.

In the data, and in the paper, there are large spikes at $62$ and at the full retirement age; the forces that produce the spike at $62$ -- liquidity constraints, health, the bad earnings draw that ends a career -- are exactly the ones this version leaves out.

## Solving for the equilibrium payable path

Now give workers the news.

Starting from $\phi_t \equiv 1$, we apply $\Phi$, take a damped step toward the result, and repeat.

The step length starts at one half and anneals toward a floor, as in the solver of {cite:t}`SargentShi2026`.

```{code-cell} ipython3
def transition(W, F0, plan0, a0, iters=40, tol=1e-4,
               ν0=0.5, anneal=0.92, floor=0.08, verbose=True):
    "Damped iteration on the payable path."
    φ = np.ones(T+1)
    for k in range(iters):
        R, X, _, _, _ = solve_plans(W, φ, W.j0, plan0=plan0,
                                    a0=a0.reshape(W.C, 1))
        pay, ben = flows(W, φ, R, X, W.j0)
        Pay, Ben = aggregate(W, pay, ben)
        F, φ_new = fund_and_phi(W, Pay, Ben, F0)
        gap = np.abs(φ_new - φ).max()
        ν = max(ν0*anneal**k, floor)
        φ = (1-ν)*φ + ν*φ_new
        if verbose and k % 5 == 0:
            print(f"   iteration {k:>3}   step {ν:.3f}   gap {gap:.2e}"
                  f"   φ in 2033 {φ[2033-Y0]:.4f}")
        if gap < tol:
            break
    return dict(φ=φ, F=F, R=R, X=X, Pay=Pay, Ben=Ben)

F0 = par.F_ratio*W.Npop[0]*Ben_pre[0]
base = transition(W, F0, (R_pre, X_pre), a26)
```

The converged path gives the two dates and the size of the cut.

```{code-cell} ipython3
φ, F = base['φ'], base['F']
depletion = YEARS[np.argmax(F[:-1] <= 1e-12)]
shortfall = YEARS[np.argmax(φ < 1)]
print(f"first shortfall year: {shortfall}")
print(f"depletion year:       {depletion}")
print(f"φ in {shortfall}: {φ[shortfall-Y0]:.3f},  in {depletion}: {φ[depletion-Y0]:.3f},"
      f"  in the long run: {φ[-1]:.3f}")
```

The dates are the trustees' dates, and the first full-year cut is close to theirs: they project that continuing income will cover about $78$ percent of scheduled benefits in the first year the fund is empty.

Neither date was targeted.

The calibration fixed the cost rate of a single year, $2025$, and took the opening balance from the record; the year in which that balance runs out is then whatever the cash flows of the intervening years make it, and those depend on the claiming and retirement decisions that the falling payable factor itself provokes.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The reserves run out in 2032 and the payable factor keeps falling after the cut
    name: fig-ssd-fund-phi
---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(YEARS, F[:-1]/(W.Npop*base['Ben']), lw=2)
axes[0].set_title('trust fund reserves / annual scheduled benefits')
axes[1].plot(YEARS, φ, lw=2)
axes[1].set_title('payable factor $\\phi_t$')
axes[1].set_ylim(0, 1.05)
for ax in axes:
    ax.axvline(depletion, color='k', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('year')
    ax.set_xlim(Y0, 2120)
plt.tight_layout()
plt.show()
```

The payable factor does not settle at the level it reaches on the day the fund empties.

It falls further for the rest of the century, because the dependency ratio keeps rising.

By {eq}`ssd_longrun` the long-run cut is governed entirely by the program's cost rate, and we can check that identity on the computed path.

```{code-cell} ipython3
cost = base['Ben']/base['Pay']
print(f"terminal cost rate: {cost[-1]:.4f}")
print(f"τ_ss/[(1-κ_b)·cost rate] = {par.τ_ss/((1-par.κ_b)*cost[-1]):.4f}"
      f"   against φ_∞ = {φ[-1]:.4f}")
```

## Claiming and retirement along the transition

Both decisions respond to the path of $\phi_t$, and both reverse direction at the cut.

```{code-cell} ipython3
def flow_ages(W, R, X):
    "Population weighted mean age of new retirees and of new claimants."
    wt = W.w.reshape(NC, W.NE)[0]
    age = (AGE0 + np.arange(J)).reshape(J, 1, 1)
    hitR = (R.reshape(1, NC, W.NE) == age)*wt
    hitX = (X.reshape(1, NC, W.NE) == age)*wt
    nR = np.take_along_axis(hitR.sum(axis=2), IDX, axis=1)*W.share
    nX = np.take_along_axis(hitX.sum(axis=2), IDX, axis=1)*W.share
    ages = (AGE0 + np.arange(J)).reshape(J, 1)
    return ((nR*ages).sum(axis=0)/np.maximum(nR.sum(axis=0), 1e-12),
            (nX*ages).sum(axis=0)/np.maximum(nX.sum(axis=0), 1e-12))

mean_R, mean_X = flow_ages(W, base['R'], base['X'])
```

The two ages move together, and both turn at the cut.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Claiming and retirement rush forward before the cut and reverse after it
    name: fig-ssd-flow-ages
---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(YEARS, mean_R, lw=2, label='mean age of new retirees')
ax.plot(YEARS, mean_X, lw=2, label='mean age of new claimants')
ax.axvline(depletion, color='k', ls='--', lw=1, alpha=0.7)
ax.set_xlabel('year')
ax.set_ylabel('age')
ax.set_xlim(Y0, 2075)
ax.legend()
plt.tight_layout()
plt.show()
```

As the fund runs down there is a **rush**.

Both ages fall year after year from $2026$, and the mean age of new claimants reaches $62.2$ in the first year the fund is empty, against $64.7$ at the start.

Workers inside the claiming window file as early as they can, because filing fixes the actuarial factor while the payments made in those years are still paid in full or nearly so.

A worker who waits instead collects under the reduced factors of later years.

The year after, the pattern reverses: there is nothing left to rush for, and both ages jump back above $64.5$ and then drift up, to $65.3$ and $66.6$ in the long run.

The pension is smaller, so careers lengthen; and mortality keeps improving while the actuarial schedule stays at its statutory values, so a later claim buys the higher benefit for more years.

Past about $2075$ the mean claim age chatters between two adjacent integers, which is why {numref}`fig-ssd-flow-ages` stops there; the note on computation below explains why.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Payroll gains arrive at once, the benefits of later claiming decades later
    name: fig-ssd-cost-rate
---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(YEARS, base['Pay'], lw=2, label='taxable payroll')
axes[0].plot(YEARS, base['Ben'], lw=2, label='scheduled benefits')
axes[0].legend()
axes[0].set_title('per capita, model units')
axes[1].plot(YEARS, cost, lw=2)
axes[1].set_title('cost rate: scheduled benefits / taxable payroll')
for ax in axes:
    ax.axvline(depletion, color='k', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('year')
    ax.set_xlim(Y0, 2120)
plt.tight_layout()
plt.show()
```

The benefit formula runs on two clocks, and the right panel of {numref}`fig-ssd-cost-rate` shows both.

Longer careers add contributors to taxable payroll **immediately**.

Later filing lowers the scheduled bill immediately, by reducing the number of claimants, and raises it only **decades later**, when the delayed filers collect benefits carrying higher actuarial factors.

The cost rate therefore bumps up in the years before the cut, as the rush adds claimants, and then falls back for a year or two afterwards, as filing is deferred again, before resuming its climb.

Once the fund is empty, {eq}`ssd_phiflow` makes the payable factor exactly proportional to the inverse of the cost rate, so that dip is a benefit cut that does not happen.

At the cost rate of $2031$ the formula would pay $\phi = 0.702$; what workers' own responses deliver in $2033$ is $0.755$.

{cite:t}`SargentShi2026` find the asymmetry strong enough that the payable factor recovers for a few years after the cut before resuming its decline.

Here the recovery shows up in the cost rate but not in $\phi_t$ itself: the dip is not large enough to outrun a dependency ratio that is still rising.

### Who rushes

The flow means hide who is doing the rushing, so it is worth looking at the decisions one cohort at a time.

{numref}`fig-ssd-who-rushes` plots, against a worker's age in $2026$, the plan it had made before the news and the plan it chooses after.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Irreversibility binds the old, while the young work longer but file sooner
    name: fig-ssd-who-rushes
---
ages26 = np.arange(21, 91)

def by_age26(vals):
    "Weighted mean of a decision across the types of each cohort alive in 2026."
    out = np.zeros(len(ages26))
    for i, a in enumerate(ages26):
        m = W.entry == Y0 - (a - AGE0)
        out[i] = (vals[m]*W.w[m]).sum()/W.w[m].sum()
    return out

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, before, after, title in (
        (axes[0], R_pre, base['R'], 'retirement age'),
        (axes[1], X_pre, base['X'], 'claim age')):
    ax.plot(ages26, by_age26(before), lw=2, label='planned before the news')
    ax.plot(ages26, by_age26(after), lw=2, ls='--', label='chosen after')
    ax.set_xlabel('age in 2026')
    ax.set_ylabel('age')
    ax.set_title(title)
axes[0].legend()
plt.tight_layout()
plt.show()
```

Three groups appear in {numref}`fig-ssd-who-rushes`.

Workers who have already retired and filed by $2026$ sit on top of their old plans, because both decisions are irreversible: the two lines coincide at the right of each panel.

Workers in or near the claiming window move first and move earliest: they file sooner than they had meant to, and some of them leave work sooner too, to collect while benefits are still paid in full.

Workers in their forties and younger go the other way on one margin and with the crowd on the other.

They work longer, because their pension is smaller.

But they file **sooner** than they had planned, by more than a year: with the payable factor falling from one year to the next, a benefit deferred is a benefit collected at a lower factor, and that headwind works against the actuarial reward for waiting.

### A worker's life cycle

What the news does to one worker is easier to see than what it does to an aggregate.

{numref}`fig-ssd-life-cycle` follows the median type of the cohort entering in $2026$ through its whole life, before and after.

```{code-cell} ipython3
R_b, X_b, wealth_b, Om_b, _ = solve_plans(W, φ, W.j0, plan0=(R_pre, X_pre),
                                          a0=a26.reshape(W.C, 1))

def life_profile(W_, φ_, R, X, wealth, Om, row):
    "Net income, consumption and assets by age for a single worker."
    p = W_.p
    m = income_path(W_, φ_, R, X, rows=[row])[:, 0]
    c = wealth[row]/((1+p.τ_c)*Om[row])*(p.β*after_tax_R(p))**np.arange(J)
    a = np.zeros(J)
    for j in range(J-1):
        s = W_.ell[row, j+1]/W_.ell[row, j]
        a[j+1] = (after_tax_R(p)/s)*(a[j] + m[j] - (1+p.τ_c)*c[j])
    return m, c, a

row = np.where(W.entry == 2026)[0][12]          # the median type
before = life_profile(W, np.ones(T+1), R_pre, X_pre, wealth_pre, Om_pre, row)
after = life_profile(W, φ, R_b, X_b, wealth_b, Om_b, row)
```

Three panels: what the worker receives, what it spends, and what it holds.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: A smaller pension means less consumption, more saving, and no bridge year
    name: fig-ssd-life-cycle
---
age = AGE0 + np.arange(J)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for k, ttl in enumerate(('net income', 'consumption', 'assets')):
    axes[k].plot(age, before[k], lw=2, label='before the news')
    axes[k].plot(age, after[k], lw=2, ls='--', label='after')
    axes[k].set_title(ttl)
    axes[k].set_xlabel('age')
    axes[k].set_xlim(20, 90)
axes[0].legend()
plt.tight_layout()
plt.show()

print(f"retirement age {R_pre[row]} -> {R_b[row]},  claim age {X_pre[row]} -> {X_b[row]}")
print(f"peak assets    {before[2].max():.2f} -> {after[2].max():.2f}")
```

This worker's retirement age does not move -- the choice is an integer, and an aggregate that moves by a year moves because different workers switch at different dates -- but it files a year earlier, it consumes less at every age, and it carries about a third more assets into retirement.

The income panel of {numref}`fig-ssd-life-cycle` shows one more thing that the plan before the news was willing to pay for.

Retiring at $65$ and filing at $66$ leaves a year with no income at all, bridged out of savings, in exchange for a permanently higher actuarial factor.

That bridge is the first thing the news takes away.

The upward tilt of consumption is $\beta\tilde R > 1$: this worker is patient enough, at an after-tax return of $4.2$ percent, to want a rising profile.

The third panel is the saving response that, in general equilibrium, would deepen the capital stock and raise the wage, and with it taxable payroll: the channel this lecture shuts off and the paper keeps.

## How this compares with the full model

The table collects what the stripped-down model delivers next to the general equilibrium numbers of {cite:t}`SargentShi2026`.

```{code-cell} ipython3
rows = [("first shortfall year",            f"{shortfall}",            "2032"),
        ("fund empty from",                 f"{depletion}",            "2033"),
        ("φ in the first full-cut year",    f"{φ[depletion-Y0]:.3f}",  "0.773"),
        ("φ in the long run",               f"{φ[-1]:.3f}",            "0.546"),
        ("mean claim age, 2026",            f"{mean_X[0]:.1f}",        "65.0"),
        ("mean claim age, long run",        f"{mean_X[-1]:.1f}",       "69.2"),
        ("mean retirement age, 2026",       f"{mean_R[0]:.1f}",        "62.7"),
        ("mean retirement age, long run",   f"{mean_R[-1]:.1f}",       "66.2"),
        ("cost rate, 2026",                 f"{cost[0]:.3f}",          "0.136")]

print(f"{'':34}{'this lecture':>14}{'the paper':>12}")
for name, mine, theirs in rows:
    print(f"{name:34}{mine:>14}{theirs:>12}")
```

The dates agree and the direction of every response agrees.

Two things do not, and both are worth seeing plainly.

The long-run cut is deeper here, $0.482$ against $0.546$.

Prices are fixed in this lecture, so the capital deepening that raises the wage, and with it taxable payroll, is missing; and the demography here is a smooth stylization rather than the trustees' projection.

The behavioral responses are also **smaller** here than in the paper, not merely centered differently.

Over the transition the mean retirement age rises by $0.9$ years against the paper's $3.5$, and the mean claim age by $1.9$ against $4.2$.

Three features of the stripped-down model mute them: workers face no idiosyncratic earnings risk, so nobody is pushed across a threshold by a bad draw; each worker's choice is an integer, so a type responds only when the gain crosses a whole year; and prices do not move, so the wage never rewards the longer careers.

The retirement *levels* need one more caution: the paper's $62.7$ is a mean pulled down by a left tail of early retirements that idiosyncratic shocks produce, and its median, which is what the calibration here was matched to, is $64.5$.

```{admonition} What the general equilibrium model adds
:class: tip

**A tax cut arrives with the benefit cut.** Depletion removes the Social Security deficit from the consolidated federal budget, so the labor tax falls from $0.220$ to $0.190$ in the long run. Current law cuts retirees' benefits and workers' taxes in the same year.

**Capital deepens.** Every working cohort saves against a smaller pension, the capital--output ratio rises from $3.00$ to $3.19$, the interest rate falls from $6.0$ to $5.3$ percent, and the wage rises.

**Financing a bailout is expensive, and the ranking is not the ranking of capital.** Paying every scheduled benefit in full costs a newborn in the terminal steady state $4.8$ percent of lifetime consumption under consumption taxation, $8.7$ percent under capital taxation and $11.6$ percent under wage taxation. The capital tax contracts capital the most and is still cheaper than the wage tax, which distorts the retirement margin through which the economy had relieved the pressure on the program, and shrinks its own base.

**One program reform changes the arithmetic.** Raising the full retirement age from $67$ to $69$ delays depletion by ten years and raises the long-run payable factor from $0.546$ to $0.621$. Raising the taxable maximum and cutting the top replacement rate move the incidence and leave the depletion year where it is.

**The announcement date redistributes among the living.** A bailout is worth about $+4.5$ percent of remaining lifetime consumption to a worker aged $60$ in $2026$ and about $-2$ percent to a worker aged $25$.
```

```{admonition} Computing the full model
:class: note

The paper's solver has the same shape as the one above and differs in scale.

The worker's problem is a dynamic program with assets, an idiosyncratic productivity shock, permanent ability, the earnings record and the claim status as states, solved by backward induction over $81$ ages and $150$ calendar years; the cross-section is a panel of $80{,}000$ workers per age cell, simulated forward with common random numbers so that every scenario faces the same workers and the same shocks.

Stationary equilibria are found with Anderson acceleration, and the transition is a damped iteration on prices, the bequest transfer, the payable path and a transitory tax rate, as here.

The code is written in JAX and runs on an A100 GPU: about $22$ hours from a cold start for the full set of experiments, or about an hour to replay the stored converged paths.

The paper also reports a residual that damping does not remove.

Because the period is a year, the iteration can cycle between two adjacent first shortfall years, each implying the other.

Our own iteration stalls in the same way, for a related reason: retirement and claim ages are integers, so a mass of workers switches discontinuously, and $\Phi$ is not continuous.

In the output above, the gap stops falling at about $10^{-2}$ while the payable factor printed beside it settles to the fourth decimal.
```

## Exercises

```{exercise}
:label: ssd_ex1

The box above claims that $\Phi$ is a decreasing map, and that is the reason every iteration here is damped.

Check it, in two steps.

1. Take the converged path, raise it by five percentage points in every year from $2033$ on, and apply $\Phi$ once. Does the image rise or fall?
1. Run the iteration **undamped**, that is, with $\phi_{k+1} = \Phi(\phi_k)$, for twenty steps starting from $\phi \equiv 1$, and plot the payable factor in $2040$ and in $2080$ against the iteration count. Compare both with a damped iteration. Does the undamped iteration converge?
```

```{solution-start} ssd_ex1
:class: dropdown
```

For part 1, one application of the operator is one call each to `solve_plans`, `flows`, `aggregate` and `fund_and_phi`.

```{code-cell} ipython3
def Phi(W, φ, F0, plan0, a0):
    "One application of the operator Φ."
    R, X, _, _, _ = solve_plans(W, φ, W.j0, plan0=plan0, a0=a0.reshape(W.C, 1))
    pay, ben = flows(W, φ, R, X, W.j0)
    Pay, Ben = aggregate(W, pay, ben)
    return fund_and_phi(W, Pay, Ben, F0)[1]

φ_up = φ.copy()
φ_up[depletion-Y0:] += 0.05

im_0 = Phi(W, φ, F0, (R_pre, X_pre), a26)
im_up = Phi(W, φ_up, F0, (R_pre, X_pre), a26)

k = 2050 - Y0
print(f"φ in 2050:            {φ[k]:.4f}  ->  raised to {φ_up[k]:.4f}")
print(f"Φ(φ) in 2050:         {im_0[k]:.4f}  ->  {im_up[k]:.4f}")
print(f"mean change in Φ(φ) over 2033-2100: "
      f"{(im_up - im_0)[depletion-Y0:2100-Y0].mean():+.4f}")
```

The image moves the other way: a more generous promise brings earlier claiming and earlier retirement, which lowers taxable payroll and raises the scheduled bill, so the program can pay a smaller fraction.

For part 2, an iteration on a decreasing map alternates around its fixed point, and whether the alternation dies out depends on how strong the response is.

```{code-cell} ipython3
def trace(ν, years=(2040, 2080), n=20):
    "Payable factor at selected dates along an iteration with step ν."
    φ_k = np.ones(T+1)
    out, gaps = [], []
    for _ in range(n):
        φ_new = Phi(W, φ_k, F0, (R_pre, X_pre), a26)
        gaps.append(np.abs(φ_new - φ_k).max())
        φ_k = (1-ν)*φ_k + ν*φ_new
        out.append([φ_k[y-Y0] for y in years])
    return np.array(out), np.array(gaps)

un, gap_un = trace(1.0)
dp, gap_dp = trace(0.3)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for k, y in enumerate((2040, 2080)):
    axes[k].plot(un[:, k], 'o-', lw=2, label='undamped')
    axes[k].plot(dp[:, k], 's-', lw=2, label='damped, step 0.3')
    axes[k].axhline(φ[y-Y0], color='k', ls='--', lw=1, label='fixed point')
    axes[k].set_title(f'$\\phi$ in {y}')
    axes[k].set_xlabel('iteration')
axes[0].legend()
plt.tight_layout()
plt.show()

print(f"undamped, largest residual after 20 steps: {gap_un[-1]:.3f}")
print(f"damped,   largest residual after 20 steps: {gap_dp[-1]:.3f}")
```

In $2040$ the undamped iteration alternates and settles: the response of claiming and retirement to the payable factor is weak enough at that date that $\Phi$ is a contraction there.

In $2080$ it is not.

The undamped sequence locks into a two-cycle between about $0.48$ and $0.60$ and stays there: each payable factor calls forth decisions that justify the other one, and the iteration never stops swinging between them.

The damped sequence settles between the two.

This is why every solver in this lecture takes a fractional step, and why {cite:t}`SargentShi2026` anneal theirs toward a floor.

```{solution-end}
```

```{exercise}
:label: ssd_ex2

Of the reforms that {cite:t}`SargentShi2026` study, only one moves the depletion date: raising the full retirement age from $67$ to $69$.

The reform is announced in $2026$ and applies to everyone at once, current beneficiaries included.

It works entirely through the actuarial factor: at a full retirement age of $69$ the factor is about $13$ percent lower at every claim age, which is the gap between the two lines in the right panel of {numref}`fig-ssd-benefit-rules`, and the claim window still closes at $70$, so a worker can recover only part of the cut by filing later.

Compute the reformed transition and report the depletion year and the long-run payable factor.

By how many years is depletion delayed, and how much of the long-run benefit cut does the reform undo?
```

```{solution-start} ssd_ex2
:class: dropdown
```

Only the actuarial schedule changes, so we build a new `World` with `FRA=69` and run the same transition from the same initial conditions.

```{code-cell} ipython3
W69 = World(par._replace(FRA=69), g_pop)
base69 = transition(W69, F0, (R_pre, X_pre), a26, verbose=False)

φ69, F69 = base69['φ'], base69['F']
dep69 = YEARS[np.argmax(F69[:-1] <= 1e-12)]

print(f"{'':28}{'FRA 67':>10}{'FRA 69':>10}")
print(f"{'depletion year':28}{depletion:>10}{dep69:>10}")
print(f"{'φ in the long run':28}{φ[-1]:>10.3f}{φ69[-1]:>10.3f}")
print(f"{'cost rate in the long run':28}"
      f"{cost[-1]:>10.3f}{(base69['Ben']/base69['Pay'])[-1]:>10.3f}")
```

The reform lifts the whole payable path, and the gap between the two never closes.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(YEARS, φ, lw=2, label='current law')
ax.plot(YEARS, φ69, lw=2, label='full retirement age 69')
ax.set_xlabel('year')
ax.set_ylabel('payable factor $\\phi_t$')
ax.set_xlim(Y0, 2120)
ax.legend()
plt.tight_layout()
plt.show()
```

Depletion is delayed by seven years, from $2033$ to $2040$, against the ten years the paper reports.

The long-run cut falls from $51.8$ to $43.2$ percent of scheduled benefits, so the reform undoes about a sixth of it.

It does that work because it cuts the scheduled bill at once, for current beneficiaries as well as future ones, so the fund lasts longer and the cash flow condition {eq}`ssd_phiflow` clears at a higher payable factor forever after.

It does not restore scheduled benefits; it lowers them.

A worker's *paid* benefit is the product of the two factors, and the reform raises one by cutting the other.

```{solution-end}
```

```{exercise}
:label: ssd_ex3

Suppose that instead of letting the cut happen, Congress guarantees every scheduled benefit, $\phi_t = 1$ forever, and pays for it by raising the payroll tax $\tau^{ss}$ to a new constant from $2026$ on.

1. Find the rate that balances the program's cash flow in the long run, taking into account that workers respond to it.
1. Compute the consumption equivalent variation of the guarantee for cohorts entering between $2026$ and $2096$. Under the closed form {eq}`ssd_objective`, a cohort's lifetime value is $V = \Omega\log\mathcal W - \eta h D$, so the compensation that equates the two economies is $\lambda = \exp\big(\sum_i w_i \Delta V_i/\Omega\big) - 1$.
1. Do the same for the workers who are **already alive** in $2026$, as a function of their age. Who gains from the guarantee and who pays for it?
```

```{solution-start} ssd_ex3
:class: dropdown
```

For part 1, with $\phi_t \equiv 1$ there is no fixed point in $\phi$ left to solve: we need only the payroll tax at which the program's long-run cash flow balances, and workers' responses to that tax are inside $\mathit{Pay}$ and $B^{sch}$.

```{code-cell} ipython3
def program_gap(τ, plan0, a0):
    "Long-run cash flow of the program when every benefit is paid in full."
    W_τ = World(par._replace(τ_ss=τ), g_pop)
    R, X, _, _, _ = solve_plans(W_τ, np.ones(T+1), W_τ.j0,
                                plan0=plan0, a0=a0.reshape(W_τ.C, 1))
    pay, ben = flows(W_τ, np.ones(T+1), R, X, W_τ.j0)
    Pay, Ben = aggregate(W_τ, pay, ben)
    return τ*Pay[-1] - (1-par.κ_b)*Ben[-1]

τ_star = brentq(lambda τ: program_gap(τ, (R_pre, X_pre), a26), 0.10, 0.40, xtol=1e-4)
print(f"payroll tax that funds the guarantee: {τ_star:.4f}"
      f"   (current law: {par.τ_ss})")
```

For part 2, we read the lifetime values off `solve_plans` in the two economies.

```{code-cell} ipython3
def cohort_values(W_, φ_, plan0, a0):
    "Lifetime value of every row, and the weights and Ω needed to aggregate."
    R, X, wealth, Om, Dw = solve_plans(W_, φ_, W_.j0, plan0=plan0,
                                       a0=a0.reshape(W_.C, 1))
    V = Om*np.log(wealth) - (W_.p.η*W_.η)*Dw
    return V, Om

V_dep, Om_dep = cohort_values(W, φ, (R_pre, X_pre), a26)
W_bail = World(par._replace(τ_ss=τ_star), g_pop)
V_bail, _ = cohort_values(W_bail, np.ones(T+1), (R_pre, X_pre), a26)

def cev(entry_year):
    "Consumption equivalent variation of the guarantee, cohort by cohort."
    m = W.entry == entry_year
    wts = W.w[m]/W.w[m].sum()
    return np.exp(((V_bail[m] - V_dep[m])*wts).sum()/Om_dep[m][0]) - 1

entries = np.arange(2026, 2097)
λ_new = np.array([cev(y) for y in entries])
ages = np.arange(21, 96)
λ_old = np.array([cev(Y0 - (a - AGE0)) for a in ages])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(entries, 100*λ_new, lw=2)
axes[0].set_xlabel('year the cohort enters')
axes[0].set_title('cohorts not yet born')
axes[1].plot(ages, 100*λ_old, lw=2)
axes[1].set_xlabel('age in 2026')
axes[1].set_title('workers already alive')
for ax in axes:
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('consumption equivalent variation, %')
plt.tight_layout()
plt.show()

print(f"entering in 2026: {100*λ_new[0]:+.2f} percent")
print(f"entering in 2096: {100*λ_new[-1]:+.2f} percent")
for a in (25, 45, 60, 65, 80):
    print(f"   aged {a} in 2026: {100*λ_old[a-21]:+.2f} percent")
```

Every entering cohort is worse off, by about thirteen percent of lifetime consumption, and the cost barely depends on the entry date.

An entering cohort pays the higher payroll tax across its whole working life, to restore the benefits of the retirees of those years, and collects its own restored benefit only decades later, by which time the number of retirees per worker is higher still.

In {cite:t}`SargentShi2026` the losses grow with the entry date, because their financing tax steps up from a transitory to a higher terminal rate partway through the transition, so later cohorts spend more of their lives under the higher rate.

Here the financing rate is one constant from $2026$ on, so every entering cohort faces the same schedule and the profile is nearly flat.

That is the finding of {cite:t}`SargentShi2026` for all three of their financing instruments, and the mechanism here is the same one.

The right panel is the other half of the story, and it is the paper's result on timing and incidence.

The cohorts already alive in $2026$ divide sharply.

A worker aged $25$ loses about as much as a cohort that has yet to enter.

A worker in the late fifties or sixties **gains**: the restored benefit arrives within a few years, and the higher payroll tax falls on the few working years that remain.

Retirees gain without paying at all, except through the general equilibrium channels this model suppresses.

The crossover is a little above age $50$.

{cite:t}`SargentShi2026` find the same pattern with a smaller spread, because their wage tax is a broader instrument than a doubled payroll tax, and because their general equilibrium prices move.

What the lecture model cannot say is which instrument is cheapest, because that comparison runs through those prices and through the taxes the guarantee displaces.

```{solution-end}
```

```{exercise}
:label: ssd_ex4

Two forces raise the dependency ratio in this model: cohorts entering the labor force stopped growing, and mortality keeps improving.

Separate them.

Hold the birth path fixed and switch the mortality improvement off, `par._replace(g_m=0.0)`, so that the life table of $2025$ applies forever.

1. What happens to the dependency ratio in $2085$, and to the long-run payable factor?
1. What happens to the depletion year? Explain why it moves so much less than the long-run cut does.
1. What happens to the claim age, and what does that tell you about why claiming drifts later in the baseline?
```

```{solution-start} ssd_ex4
:class: dropdown
```

We rerun the transition in a `World` that keeps the 2025 life table forever.

```{code-cell} ipython3
W0 = World(par._replace(g_m=0.0), g_pop)       # same cohorts, the 2025 life table forever
flat = transition(W0, F0, (R_pre, X_pre), a26, verbose=False)
dep0 = YEARS[np.argmax(flat['F'][:-1] <= 1e-12)]
mR0, mX0 = flow_ages(W0, flat['R'], flat['X'])

print(f"{'':28}{'baseline':>11}{'no gains':>11}")
print(f"{'life expectancy at 20, 2086':28}"
      f"{20+cohort_survival([2086], par)[0].sum():>11.1f}"
      f"{20+cohort_survival([2086], par._replace(g_m=0.0))[0].sum():>11.1f}")
print(f"{'dependency ratio, 2085':28}{dependency(g_pop, 2085, par):>11.3f}"
      f"{dependency(g_pop, 2085, par._replace(g_m=0.0)):>11.3f}")
print(f"{'depletion year':28}{depletion:>11}{dep0:>11}")
print(f"{'φ in the long run':28}{φ[-1]:>11.3f}{flat['φ'][-1]:>11.3f}")
print(f"{'cost rate in the long run':28}{cost[-1]:>11.3f}"
      f"{(flat['Ben']/flat['Pay'])[-1]:>11.3f}")
print(f"{'mean claim age, long run':28}{mean_X[-1]:>11.2f}{mX0[-1]:>11.2f}")
```

The two paths separate only slowly, which is the point: longevity does its work in the long run.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(YEARS, φ, lw=2, label='baseline')
ax.plot(YEARS, flat['φ'], lw=2, label='no mortality improvement')
ax.set_xlabel('year')
ax.set_ylabel('payable factor $\\phi_t$')
ax.set_xlim(Y0, 2120)
ax.legend()
plt.tight_layout()
plt.show()
```

Longer lives are a large part of the long-run problem.

Holding the birth path fixed, they account for about a quarter of the long-run benefit cut: the payable factor settles at $0.60$ rather than $0.48$.

The depletion year hardly moves, and the reason is visible in {eq}`ssd_phi`.

The date at which the reserves run out is set by the cash flow of the next few years, which the cohorts already alive determine; improvements in mortality accumulate slowly, and most of what they will do to the cost rate still lies decades ahead when the fund empties.

The long run is where they do their work.

The claim age settles at $65.0$ rather than $66.6$.

That is the mechanism the lecture asserted for later claiming and had not tested.

The actuarial schedule $\mathrm{AF}(\chi)$ is statutory and frozen, so when mortality improves, waiting buys the higher benefit for more years, and workers wait.

Switch the improvement off and the drift disappears.

```{solution-end}
```
