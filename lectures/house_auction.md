---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Multiple Good Allocation Mechanisms

##  Overview

This lecture describes two mechanisms for allocating $n$ private goods ("houses")  to $m$ people ("buyers").

We assume that  $m > n$ so that there are more potential buyers than there are houses.  

Prospective buyers regard the houses  as **substitutes**.

Buyer $j$ attaches  value $v_{ij}$ to house $i$.  

These  values are **private**

  * $v_{ij}$ is  known only to person $j$ unless person $j$ chooses to tell someone.

We require that a mechanism allocate **at most** one house to one prospective buyer.


We describe two distinct mechanisms

 * A multiple rounds, ascending bid auction
 
 * A special case of a Groves-Clarke ({cite}`Groves_73`, {cite}`Clarke_71`) mechanism with a benevolent social planner


**Note:** In 1994, the multiple rounds, ascending bid auction was actually used by Stanford University to sell leases to 9 lots on the Stanford campus to eligible faculty members.
 
We begin with  overviews of the two mechanisms.

## Ascending Bids Auction for Multiple Goods

An auction is administered by an **auctioneer** 

The auctioneer has an $n \times 1$ vector $r$ of reservation prices on the $n$ houses.

The auctioneer sells house $i$ only if the final price bid for it exceeds $r_i$

The auctioneer  allocates all $n$ houses **simultaneously** 

The auctioneer does not know bidders' private values $v_{ij}$ 

There are multiple **rounds**



 - during each round, active participants can submit bids on any of the  $n$ houses  
 
 - each bidder can bid on only one house during one round
 
 - a person who was high bidder on a particular house in one round  is understood to submit  that same bid for the same  house in the next round
 
 - between rounds, a bidder who was not a high bidder can change the house on which he/she chooses to bid
 
 - the auction ends when the price of no house changes from one round to the next
 
 - all $n$ houses are allocated after the final round

 - house $i$  is retained by the auctioneer if not prospective buyer offers more that $r_i$ for the house 
 
In this auction,  person $j$ never tells anyone else his/her private values $v_{ij}$




## A Benevolent Planner

This mechanism is designed so that all prospective buyers voluntarily choose to reveal their private values to a **social planner** who uses them to construct a socially optimal allocation.

Among all feasible allocations,  a **socially optimal allocation** maximizes the sum of  private values across all prospective buyers.

The planner tells everyone in advance how he/she will allocate houses based on the matrix of values that prospective buyers report.

The mechanism provide every prospective buyer an incentive to reveal his vector of private values to the planner.

After the planner receives everyone's vector of private values, the planner deploys a **sequential** algorithm to determine an **allocation** of houses and a set of **fees** that he charges awardees  for the negative **externality** that their presence impose on other prospective buyers. 




## Equivalence of Allocations

Remarkably, these two mechanisms can produce virtually identical allocations.

We construct Python code for both mechanism.

We also work out some examples by hand or almost by hand.


Next, let's dive down into the details.


## Ascending Bid Auction


### Basic Setting


We start with  a more detailed description of the setting. 


* A seller owns $n$ houses that he wants to sell for the maximum possible amounts to a  set of $m$ prospective eligible buyers.

* The seller wants to sell at most one house to each potential  buyer.

* There are $m$ potential eligible buyers, identified by $j = [1, 2, \ldots, m]$ 

    * Each potential  buyer is permitted  to buy at most  one house.  

    * Buyer $j$ would be willing to pay at most $v_{ij}$ for house $i$. 
    
    * Buyer $j$  knows $v_{ij}, i= 1, \ldots , n$, but no one else does.

    * If buyer $j$ pays $p_i$ for house $i$, he enjoys surplus value $v_{ij} - p_i$.

    * Each buyer $j$ wants to choose the $i$ that maximizes his/her surplus value $v_{ij} - p_i$.

    * The seller wants to maximize $\sum_i p_i$.  

The seller conducts a **simultaneous, multiple goods, ascending bid auction**. 

Outcomes of the  auction  are 

  * An $n \times 1$ vector $p$ of sales prices $p = [p_1, \ldots, p_n]$ for the 
  $n$ houses.

  * An $n \times m$ matrix $Q$ of $0$'s and $1$'s, where $Q_{ij} = 1$ if and only if person $j$ bought house $i$. 

  * An $n \times m$ matrix $S$ of surplus values consisting of all zeros unless
  person $j$ bought house $i$, in which case $S_{ij} = v_{ij} - p_i$

+++

We describe  rules for the auction it terms of  **pseudo  code**.

The pseudo code will provide a road map for writing Python code to implement the auction.

+++

## Pseudocode 

Here is a quick sketch of a possible simple structure for our Python code

**Inputs:**

- $n, m$.
- an $n \times m$ non-negative matrix $v$ of  private values
- an $n \times 1$ vector $r$  of seller-specified reservation prices
- the seller will not accept a price less than $r_i$ for house $i$
- we are free to think of these reservation prices as private values of a fictitious $m +1$ th buyer who does not actually participate in the auction
- initial bids can be thought of starting at $r$
- a scalar $\epsilon$ of seller-specified minimum price-bid increments


For each round of the auction, new bids on a house  must be at least the prevailing highest bid so far **plus** $\epsilon$


**Auction Protocols**

- the auction consists of a  finite number of **rounds**
- in each round, a prospective buyer can bid on one and only one house
- after each round,  a  house is temporarily awarded to the person who made the  highest bid for that house
    - temporarily winning bids on each house are announced
    - this sets the stage to move on to the next round
- a new round is held
    - bids for temporary winners from the previous round are again attached to the houses on which they bid; temporary winners of the last round  leave their bids from the previous round unchanged
    - all other active  prospective buyers must submit a new bid on some house
    - new bids on a house must be at least equal to the prevailing temporary price that won the last round **plus** $\epsilon$
    - if a person does not submit a new bid and was also not a temporary winner from the previous round, that  person must  drop out of the auction permanently
    - for each house, the highest bid, whether it is a new bid or was the temporary winner from the previous round, is announced, with the person who made that new (temporarily) winning bid being (temporarily) awarded the house to start the next round
- rounds continue until no price on **any** house changes from the previous round
- houses are sold to the winning bidders from the final round at the prices that they bid

**Outputs:**
- an $n \times 1$ vector $p$ of sales prices
- an $n \times m$ matrix $S$ of surplus values consisting of all zeros unless
person $j$ bought house $i$, in which case $S_{ij} = v_{ij} - p_i$
- an $n \times (m+1)$  matrix $Q$ of $0$'s and $1$'s that tells which buyer bought which  house.  (The last column  accounts for unsold houses.)


**Proposed buyer strategy:**

In this pseudo code and the actual Python code below, we'll assume that all buyers choose to use the following  strategy

   * The strategy is optimal  for each buyer 

Each buyer $j = 1, \ldots, m$ uses the same strategy.

The strategy has the form:
- Let $\check p^t$ be the $n \times 1$ vector of  prevailing highest-bid prices  at the beginning of round $t$
- Let $\epsilon>0$ be the minimum bid increment specified by the seller
- For each prospective buyer $j$, compute the index of the best house to bid on during round $t$, namely 
$\hat i_t = \textrm{argmax}_i\{  [  v_{ij} - \check p^t_i - \epsilon  ]\}$
- If $\max_i\{  [  v_{ij} - \check p^t_i - \epsilon  ]\} $  $\leq$</font> $0$, person $j$ permanently drops out of the auction at round $t$
- If  $v_{\hat i_t, j} - \check p^t_i - \epsilon>0$, person $j$ bids $\check p^t_i + \epsilon$ on house $j$


**Resolving ambiguities**: The protocols  we have described so far leave open two possible sources of ambiguity. 

(1) **The optimal bid choice for buyers in each round.** It is possible that a buyer has the same surplus value for multiple houses. The  argmax function in Python always returns the first argmax element. We instead  prefer to randomize among such winner. For that reason,  we write our own argmax function below.

(2) **Seller's choice of winner if same price bid cast by several buyers.** To resolve  this ambiguity, we use the np.random.choice function below.

Given the randomness in outcomes, it is possible that different  allocations  of houses could emerge from the same inputs. 

However, this will happen only when the bid price increment $\epsilon$ is nonnegligible.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import prettytable as pt

np.random.seed(100)
```


```{code-cell} ipython3
np.set_printoptions(precision=3, suppress=True)
```

## An Example

+++

Before building a Python class, let's step by step solve things almost "by hand"  to grasp  how the auction proceeds.

A step-by-step procedure also helps reduce bugs, especially when the value matrix is peculiar (e.g. the differences between values are negligible, a column containing identical values or multiple buyers have the same valuation etc.). 

Fortunately, our auction  behaves well and  robustly with various peculiar matrices.

We provide some examples later in this lecture. 



```{code-cell} ipython3
v = np.array([[8, 5, 9, 4],
              [4, 11, 7, 4],
              [9, 7, 6, 4]])
n, m = v.shape
r = np.array([2, 1, 0])
ϵ = 1
p = r.copy()
buyer_list = np.arange(m)
house_list = np.arange(n)
```

```{code-cell} ipython3
v
```

Remember that column indexes $j$ indicate buyers and row indexes $i$ indicate houses.

The above value matrix $v$ is peculiar in the sense that Buyer 3 (indexed from 0) puts the same value $4$ on  every house being sold.

Maybe buyer 3 is a bureaucrat who purchases these house simply by following  instructions from his superior.

```{code-cell} ipython3
r
```

```{code-cell} ipython3
def find_argmax_with_randomness(v):
    """
    We build our own verion of argmax function such that the argmax index will be returned randomly 
    when there are multiple maximum values. This function is similiar to np.argmax(v,axis=0)

    Parameters:
    ----------
    v: 2 dimensional np.array
    
    """
    
    n, m = v.shape
    index_array = np.arange(n)
    result=[]
    
    for ii in range(m):
        max_value = v[:,ii].max()
        result.append(np.random.choice(index_array[v[:,ii] == max_value]))
        
    return np.array(result)
```

```{code-cell} ipython3
def present_dict(dt):
    """
    A function that present the information in table.

    Parameters:
    ----------
    dt: dictionary.
    
    """
    
    ymtb = pt.PrettyTable()
    ymtb.field_names = ['House Number', *dt.keys()]
    ymtb.add_row(['Buyer', *dt.values()])
    print(ymtb)
```

**Check Kick Off Condition**

```{code-cell} ipython3
def check_kick_off_condition(v, r, ϵ):
    """
    A function that checks whether the auction could be initiated given the reservation price and value matrix.
    To avoid the situation that the reservation prices are so high that no one would even bid in the first round. 

    Parameters:
    ----------
    v : value matrix of the shape (n,m).

    r: the reservation price

    ϵ: the minimun price increment in each round 
    
    """
    
    # we convert the price vector to a matrix in the same shape as value matrix to facilitate subtraction
    p_start = (ϵ+r)[:,None] @ np.ones(m)[None,:]
    
    surplus_value = v - p_start
    buyer_decision = (surplus_value > 0).any(axis = 0)
    return buyer_decision.any()
```

```{code-cell} ipython3
check_kick_off_condition(v, r, ϵ)
```

### round 1 

+++

**submit bid**

```{code-cell} ipython3
def submit_initial_bid(p_initial, ϵ, v):
    """
    A function that describes the bid information in the first round. 

    Parameters:
    ----------
    p_initial: the price (or the reservation prices) at the beginning of auction.

    v: the value matrix

    ϵ: the minimun price increment in each round 
    
    Returns:
    ----------
    p: price array after this round of bidding
    
    bid_info: a dictionary that contains bidding information (house number as keys and buyer as values).
    
    """
    
    p = p_initial.copy()
    p_start_mat = (ϵ + p)[:,None] @ np.ones(m)[None,:]
    surplus_value = v - p_start_mat
    
    # we only care about active buyers who have positve surplus values
    active_buyer_diagnosis = (surplus_value > 0).any(axis = 0)
    active_buyer_list = buyer_list[active_buyer_diagnosis]
    active_buyer_surplus_value = surplus_value[:,active_buyer_diagnosis]
    active_buyer_choice = find_argmax_with_randomness(active_buyer_surplus_value)
    # choice means the favourite houses given the current price and ϵ
    
    # we only retain the unique house index because prices increase once at one round
    house_bid =  list(set(active_buyer_choice))
    p[house_bid] += ϵ
    
    bid_info = {}
    for house_num in house_bid:
        bid_info[house_num] = active_buyer_list[active_buyer_choice == house_num]
    
    return p, bid_info
```

```{code-cell} ipython3
p, bid_info = submit_initial_bid(p, ϵ, v)
```

```{code-cell} ipython3
p
```

```{code-cell} ipython3
present_dict(bid_info)
```

**check terminal condition**

+++

Notice that  two buyers bid for house 2 (indexed from 0).

Because the auction protocol does not specify  a selection rule in this case, we simply select a winner **randomly**. 

This is reasonable because the seller can't distinguish these buyers and  doesn't know the valuation of each buyer.

It is both convenient and practical for him to just pick a winner randomly. 

There is a  50% probability that Buyer 3 is chosen as the winner for house 2, although he values it less than buyer 0.

In this case, buyer 0 has to bid one more time with a higher price, which crowds out Buyer 3.

Therefore, final price could be 3 or 4, depending on the winner in the last round.  

```{code-cell} ipython3
def check_terminal_condition(bid_info, p, v):
    """
    A function that checks whether the auction ends.
    
    Recall that the auction ends when either losers have non-positive surplus values for each house
    or there is no loser (every buyer gets a house).

    Parameters:
    ----------
    bid_info: a dictionary that contains bidding information of house numbers (as keys) and buyers (as values).

    p: np.array. price array of houses

    v: value matrix 
    
    Returns:
    ----------
    allocation: a dictionary that descirbe how the houses bid are assigned.
    
    winner_list: a list of winners
    
    loser_list: a list of losers
    
    """
    
    # there may be several buyers bidding one house, we choose a winner randomly
    winner_list=[np.random.choice(bid_info[ii]) for ii in bid_info.keys()]
    
    allocation = {house_num:winner for house_num,winner in zip(bid_info.keys(),winner_list)}
    
    loser_set = set(buyer_list).difference(set(winner_list))
    loser_list = list(loser_set)
    loser_num = len(loser_list)
    
    if loser_num == 0:
        print('The auction ends because every buyer gets one house.')
        return allocation,winner_list,loser_list
    
    p_mat = (ϵ + p)[:,None] @ np.ones(loser_num)[None,:]
    loser_surplus_value = v[:,loser_list] - p_mat
    loser_decision = (loser_surplus_value > 0).any(axis = 0)
    
    print(~(loser_decision.any()))
    
    return allocation,winner_list,loser_list
```

```{code-cell} ipython3
allocation,winner_list,loser_list = check_terminal_condition(bid_info, p, v)
```

```{code-cell} ipython3
present_dict(allocation)
```

```{code-cell} ipython3
winner_list
```

```{code-cell} ipython3
loser_list
```

### round 2


+++

From the second round on, the auction proceeds differently from the first round.

Now only active losers (those who have positive surplus values) have  an incentive to submit bids to displace temporary winners from the previous round.

```{code-cell} ipython3
def submit_bid(loser_list, p, ϵ, v, bid_info):
    """
    A function that executes the bid operation after the first round.
    After the first round, only active losers would cast a new bid with price as old price + increment.
    By such bid, winners of last round are replaced by the active losers.

    Parameters:
    ----------
    loser_list: a list that includes the indexes of losers

    p: np.array. price array of houses
    
    ϵ: minimum increment of bid price

    v: value matrix 
    
    bid_info: a dictionary that contains bidding information of house numbers (as keys) and buyers (as values).
    
    Returns:
    ----------
    p_end: a price array after this round of bidding
    
    bid_info: a dictionary that contains updated bidding information.
    
    """
    
    p_end=p.copy()
    
    loser_num = len(loser_list)
    p_mat = (ϵ + p_end)[:,None] @ np.ones(loser_num)[None,:]
    loser_surplus_value = v[:,loser_list] - p_mat
    loser_decision = (loser_surplus_value > 0).any(axis = 0)

    active_loser_list = np.array(loser_list)[loser_decision]
    active_loser_surplus_value = loser_surplus_value[:,loser_decision]
    active_loser_choice = find_argmax_with_randomness(active_loser_surplus_value)

    # we retain the unique house index and increasing the corresponding bid price
    house_bid = list(set(active_loser_choice))  
    p_end[house_bid] += ϵ

    # we record the bidding information from active losers
    bid_info_active_loser = {}
    for house_num in house_bid:
        bid_info_active_loser[house_num] = active_loser_list[active_loser_choice == house_num]

    # we update the bidding information according to the bidding from actice losers
    for house_num in bid_info_active_loser.keys():
        bid_info[house_num] = bid_info_active_loser[house_num]
    
    return p_end,bid_info
```

```{code-cell} ipython3
p,bid_info = submit_bid(loser_list, p, ϵ, v, bid_info)
```

```{code-cell} ipython3
p
```

```{code-cell} ipython3
present_dict(bid_info)
```

```{code-cell} ipython3
allocation,winner_list,loser_list = check_terminal_condition(bid_info, p, v)
```

```{code-cell} ipython3
present_dict(allocation)
```

### round 3

```{code-cell} ipython3
p,bid_info = submit_bid(loser_list, p, ϵ, v, bid_info)
```

```{code-cell} ipython3
p
```

```{code-cell} ipython3
present_dict(bid_info)
```

```{code-cell} ipython3
allocation,winner_list,loser_list = check_terminal_condition(bid_info, p, v)
```

```{code-cell} ipython3
present_dict(allocation)
```

### round 4

```{code-cell} ipython3
p,bid_info = submit_bid(loser_list, p, ϵ, v, bid_info)
```

```{code-cell} ipython3
p
```

```{code-cell} ipython3
present_dict(bid_info)
```

Notice that  Buyer 3 now switches  to bid for house 1 having recongized that  house 2 is no longer his best option.

```{code-cell} ipython3
allocation,winner_list,loser_list = check_terminal_condition(bid_info, p, v)
```

```{code-cell} ipython3
present_dict(allocation)
```

### round 5

```{code-cell} ipython3
p,bid_info = submit_bid(loser_list, p, ϵ, v, bid_info)
```

```{code-cell} ipython3
p
```

```{code-cell} ipython3
present_dict(bid_info)
```

Now Buyer 1 bids for house 1 again with price at 4, which crowds out Buyer 3, marking the end of the auction.

```{code-cell} ipython3
allocation,winner_list,loser_list = check_terminal_condition(bid_info, p, v)
```

```{code-cell} ipython3
present_dict(allocation)
```

```{code-cell} ipython3
# as for the houses unsold

house_unsold_list = list(set(house_list).difference(set(allocation.keys())))
house_unsold_list
```

```{code-cell} ipython3
total_revenue = p[list(allocation.keys())].sum()
total_revenue
```

## A Python Class

+++

Above we simulated an ascending bid auction step by step. 

When defining  functions, we repeatedly computed some intermediate objects because our Python function loses track of variables once the  function is executed.

That of course led  to redundancy in our code 

It is much more efficient  to collect all of the aforementioned code into a class that  records information about all rounds. 

```{code-cell} ipython3
class ascending_bid_auction:
    
    def __init__(self, v, r, ϵ):
        """
        A class that simulates an ascending bid auction for houses. 
        
        Given buyers' value matrix, sellers' reservation prices and minimum increment of bid prices,
        this class can execute an ascending bid auction and present information round by round until the end.

        Parameters:
        ----------
        v: 2 dimensional value matrix 

        r: np.array of reservation prices

        ϵ: minimum increment of bid price

        """
        
        self.v = v.copy()
        self.n,self.m = self.v.shape
        self.r = r
        self.ϵ = ϵ
        self.p = r.copy()
        self.buyer_list = np.arange(self.m)
        self.house_list = np.arange(self.n)
        self.bid_info_history = []
        self.allocation_history = []
        self.winner_history = []
        self.loser_history = []
        
        
    def find_argmax_with_randomness(self, v):
        n,m = v.shape
        index_array = np.arange(n)
        result=[]

        for ii in range(m):
            max_value = v[:,ii].max()
            result.append(np.random.choice(index_array[v[:,ii] == max_value]))

        return np.array(result)
    
    
    def check_kick_off_condition(self):
        # we convert the price vector to a matrix in the same shape as value matrix to facilitate subtraction
        p_start = (self.ϵ + self.r)[:,None] @ np.ones(self.m)[None,:]
        self.surplus_value = self.v - p_start
        buyer_decision = (self.surplus_value > 0).any(axis = 0)
        return buyer_decision.any()
    
    
    def submit_initial_bid(self):
        # we intend to find the optimal choice of each buyer 
        p_start_mat = (self.ϵ + self.p)[:,None] @ np.ones(self.m)[None,:]
        self.surplus_value = self.v - p_start_mat

        # we only care about active buyers who have positve surplus values
        active_buyer_diagnosis = (self.surplus_value > 0).any(axis = 0)
        active_buyer_list = self.buyer_list[active_buyer_diagnosis]
        active_buyer_surplus_value = self.surplus_value[:,active_buyer_diagnosis]
        active_buyer_choice = self.find_argmax_with_randomness(active_buyer_surplus_value)

        # we only retain the unique house index because prices increase once at one round
        house_bid =  list(set(active_buyer_choice))
        self.p[house_bid] += self.ϵ

        bid_info = {}
        for house_num in house_bid:
            bid_info[house_num] = active_buyer_list[active_buyer_choice == house_num]
        self.bid_info_history.append(bid_info)
        
        print('The bid information is')
        ymtb = pt.PrettyTable()
        ymtb.field_names = ['House Number', *bid_info.keys()]
        ymtb.add_row(['Buyer', *bid_info.values()])
        print(ymtb)
        
        print('The bid prices for houses are')
        ymtb = pt.PrettyTable()
        ymtb.field_names = ['House Number', *self.house_list]
        ymtb.add_row(['Price', *self.p])
        print(ymtb)
        
        self.winner_list=[np.random.choice(bid_info[ii]) for ii in bid_info.keys()]
        self.winner_history.append(self.winner_list)
        
        self.allocation = {house_num:[winner] for house_num,winner in zip(bid_info.keys(),self.winner_list)}
        self.allocation_history.append(self.allocation)
        
        loser_set = set(self.buyer_list).difference(set(self.winner_list))
        self.loser_list = list(loser_set)
        self.loser_history.append(self.loser_list)
        
        print('The winners are')
        print(self.winner_list)
        
        print('The losers are')
        print(self.loser_list)
        print('\n')
        
    
    def check_terminal_condition(self):
        loser_num = len(self.loser_list)

        if loser_num == 0:
            print('The auction ends because every buyer gets one house.')
            print('\n')
            return True

        p_mat = (self.ϵ + self.p)[:,None] @ np.ones(loser_num)[None,:]
        self.loser_surplus_value = self.v[:,self.loser_list] - p_mat
        self.loser_decision = (self.loser_surplus_value > 0).any(axis = 0)

        return ~(self.loser_decision.any())
    
    
    def submit_bid(self):
        bid_info = self.allocation_history[-1].copy()  # we only record the bid info of winner
        
        loser_num = len(self.loser_list)
        p_mat = (self.ϵ + self.p)[:,None] @ np.ones(loser_num)[None,:]
        self.loser_surplus_value = self.v[:,self.loser_list] - p_mat
        self.loser_decision = (self.loser_surplus_value > 0).any(axis = 0)

        active_loser_list = np.array(self.loser_list)[self.loser_decision]
        active_loser_surplus_value = self.loser_surplus_value[:,self.loser_decision]
        active_loser_choice = self.find_argmax_with_randomness(active_loser_surplus_value)

        # we retain the unique house index and increasing the corresponding bid price
        house_bid = list(set(active_loser_choice))  
        self.p[house_bid] += self.ϵ

        # we record the bidding information from active losers
        bid_info_active_loser = {}
        for house_num in house_bid:
            bid_info_active_loser[house_num] = active_loser_list[active_loser_choice == house_num]

        # we update the bidding information according to the bidding from actice losers
        for house_num in bid_info_active_loser.keys():
            bid_info[house_num] = bid_info_active_loser[house_num]
        self.bid_info_history.append(bid_info)
        
        print('The bid information is')
        ymtb = pt.PrettyTable()
        ymtb.field_names = ['House Number', *bid_info.keys()]
        ymtb.add_row(['Buyer', *bid_info.values()])
        print(ymtb)
        
        print('The bid prices for houses are')
        ymtb = pt.PrettyTable()
        ymtb.field_names = ['House Number', *self.house_list]
        ymtb.add_row(['Price', *self.p])
        print(ymtb)
        
        self.winner_list=[np.random.choice(bid_info[ii]) for ii in bid_info.keys()]
        self.winner_history.append(self.winner_list)
        
        self.allocation = {house_num:[winner] for house_num,winner in zip(bid_info.keys(),self.winner_list)}
        self.allocation_history.append(self.allocation)
        
        loser_set = set(self.buyer_list).difference(set(self.winner_list))
        self.loser_list = list(loser_set)
        self.loser_history.append(self.loser_list)
        
        print('The winners are')
        print(self.winner_list)
        
        print('The losers are')
        print(self.loser_list)
        print('\n')
        
        
    def start_auction(self):
        print('The Ascending Bid Auction for Houses')
        print('\n')
        
        print('Basic Information: %d houses, %d buyers'%(self.n, self.m))
        
        print('The valuation matrix is as follows')
        ymtb = pt.PrettyTable()
        ymtb.field_names = ['Buyer Number', *(np.arange(self.m))]
        for ii in range(self.n):
            ymtb.add_row(['House %d'%(ii), *self.v[ii,:]])
        print(ymtb)
        
        print('The reservation prices for houses are')
        ymtb = pt.PrettyTable()
        ymtb.field_names = ['House Number', *self.house_list]
        ymtb.add_row(['Price', *self.r])
        print(ymtb)
        print('The minimum increment of bid price is %.2f' % self.ϵ)
        print('\n')
        
        ctr = 1
        if self.check_kick_off_condition():
            print('Auction starts successfully')
            print('\n')
            print('Round %d'% ctr)
            
            self.submit_initial_bid()
            
            while True:
                if self.check_terminal_condition():
                    print('Auction ends')
                    print('\n')
                    
                    print('The final result is as follows')
                    print('\n')
                    print('The allocation plan is')
                    ymtb = pt.PrettyTable()
                    ymtb.field_names = ['House Number', *self.allocation.keys()]
                    ymtb.add_row(['Buyer', *self.allocation.values()])
                    print(ymtb)
                    
                    print('The bid prices for houses are')
                    ymtb = pt.PrettyTable()
                    ymtb.field_names = ['House Number', *self.house_list]
                    ymtb.add_row(['Price', *self.p])
                    print(ymtb)
                    
                    print('The winners are')
                    print(self.winner_list)

                    print('The losers are')
                    print(self.loser_list)
                    
                    self.house_unsold_list = list(set(self.house_list).difference(set(self.allocation.keys())))
                    print('The houses unsold are')
                    print(self.house_unsold_list)
                    
                    self.total_revenue = self.p[list(self.allocation.keys())].sum()
                    print('The total revenue is %.2f' % self.total_revenue)
                    
                    break
                    
                ctr += 1
                print('Round %d'% ctr)
                self.submit_bid()
            
            # we compute the surplus matrix S and the quantity matrix X as required in 1.1
            self.S = np.zeros((self.n, self.m))
            for ii,jj in zip(self.allocation.keys(),self.allocation.values()):
                self.S[ii,jj] = self.v[ii,jj] - self.p[ii]
            
            self.Q = np.zeros((self.n, self.m + 1))  # the last column records the houses unsold
            for ii,jj in zip(self.allocation.keys(),self.allocation.values()):
                self.Q[ii,jj] = 1
            for ii in self.house_unsold_list:
                self.Q[ii,-1] = 1
            
            # we sort the allocation result by the house number
            house_sold_list = list(self.allocation.keys())
            house_sold_list.sort()
            
            dict_temp = {}
            for ii in house_sold_list:
                dict_temp[ii] = self.allocation[ii]
            self.allocation = dict_temp
            
        else:
            print('The auction can not start because of high reservation prices')
```

Let's use our class to conduct the auction described in one of the above examples.

```{code-cell} ipython3
v = np.array([[8,5,9,4],[4,11,7,4],[9,7,6,4]])
r = np.array([2,1,0])
ϵ = 1

auction_1 = ascending_bid_auction(v, r, ϵ)

auction_1.start_auction()
```

```{code-cell} ipython3
# the surplus matrix S

auction_1.S
```

```{code-cell} ipython3
# the quantity matrix X

auction_1.Q
```

## Robustness Checks

Let's do some stress testing of our code by applying it to  auctions characterized by different matrices of private values.

**1. number of houses = number of buyers**

```{code-cell} ipython3
v2 = np.array([[8,5,9],[4,11,7],[9,7,6]])

auction_2 = ascending_bid_auction(v2, r, ϵ)

auction_2.start_auction()
```

**2. multilple excess buyers**

```{code-cell} ipython3
v3 = np.array([[8,5,9,4,3],[4,11,7,4,6],[9,7,6,4,2]])

auction_3 = ascending_bid_auction(v3, r, ϵ)

auction_3.start_auction()
```

**3. more houses than buyers**

```{code-cell} ipython3
v4 = np.array([[8,5,4],[4,11,7],[9,7,9],[6,4,5],[2,2,2]])
r2 = np.array([2,1,0,1,1])

auction_4 = ascending_bid_auction(v4, r2, ϵ)

auction_4.start_auction()
```

**4. some houses have extremely high reservation prices**

```{code-cell} ipython3
v5 = np.array([[8,5,4],[4,11,7],[9,7,9],[6,4,5],[2,2,2]])
r3 = np.array([10,1,0,1,1])

auction_5 = ascending_bid_auction(v5, r3, ϵ)

auction_5.start_auction()
```

**5. reservation prices are so high that the auction can't start**

```{code-cell} ipython3
r4 = np.array([15,15,15])

auction_6 = ascending_bid_auction(v, r4, ϵ)

auction_6.start_auction()
```



+++

## A Groves-Clarke Mechanism

+++

We now decribe an alternative way for society to allocate $n$  houses to $m$ possible buyers in a way that maximizes
 total value across all potential buyers.
 
We continue to assume that each buyer can purchase at most one house.

The mechanism  is a very special case of a Groves-Clarke mechanism({cite}`Groves_73`, {cite}`Clarke_71`). 

Its special structure substantially simplifies writing Python code to find an optimal allocation.

Our mechanims works like this.

* The values $V_{ij}$ are private information to person $j$

* The mechanism makes each person $j$ willing to  tell a social planner his private values $V_{i,j}$ for all $i = 1, \ldots, n$. 

* The social planner  asks all potential bidders to tell the planner  their private values $V_{ij}$

* The social planner tells no one these, but uses them to allocate houses and set prices

* The mechanism is designed in a way that makes all prospective buyers want to tell the planner their private values 
  
   - truth telling is a dominant strategy for each potential buyer

* The planner finds a house, bidder pair with highest private value by computing 
   $(\tilde i, \tilde j) = \operatorname{argmax} (V_{ij})$

* The planner assigns house $\tilde i $ to buyer $\tilde j$

* The planner charges buyer $\tilde j$ the price $\max_{- \tilde j} V_{\tilde i,  j}$, where   $- \tilde j$ means all $j$'s except $\tilde j$. 

* The planner creates a   matrix of private values for the remaining houses $-\tilde i$ by deleting row (i.e., house) $\tilde i$ and column (i.e., buyer) $\tilde j$ from $V$.  
  - (But in doing this, the planner keeps track of the  real names of the bidders and the houses).
  

* The planner returns  to the original step and repeat it.

* The planner iterates until all $n$  houses  are allocated and the charges for  all $n$ houses are set.

+++

## An Example Solved by Hand

+++

Let's see how our Groves-Clarke algorithm would work for the following simple  matrix $V$ matrix of private values

$$
V =\begin{bmatrix} 10 & 9 & 8 & 7 & 6 \cr
                    9 & 9 & 7 & 6 & 6 \cr
                    8 & 6 & 6 & 9 & 4 \cr
                    7 & 5 & 6 & 4 & 9 \end{bmatrix}
$$

**Remark:** In the first step, when the highest private value corresponds to more than one house, bidder pairs, we choose the pair with the highest sale price. If a highest sale price corresponds to two or more pairs with highest private values, we randomly choose one.

```{code-cell} ipython3
np.random.seed(666)

V_orig = np.array([[10, 9, 8, 7, 6],  # record the origianl values
                   [9, 9, 7, 6, 6],
                   [8, 6, 6, 9, 4],
                   [7, 5, 6, 4, 9]])
V = np.copy(V_orig)  # used iteratively
n, m = V.shape
p = np.zeros(n) # prices of houses
Q = np.zeros((n, m)) # keep record the status of houses and buyers
```

**First assignment**

First, we find house, bidder pair with highest private value.

```{code-cell} ipython3
i, j = np.where(V==np.max(V))
i, j
```

So, house 0 will be sold to buyer 0 at a price of 9. We then update the sale price of house 0 and the status matrix Q.

```{code-cell} ipython3
p[i] = np.max(np.delete(V[i, :], j))
Q[i, j] = 1
p, Q
```

Then we remove row 0 and column 0 from $V$. To keep the real number of houses and buyers, we set this row and this column to -1, which will have the same result as removing them since $V \geq 0$.

```{code-cell} ipython3
V[i, :] = -1
V[:, j] = -1
V
```

**Second assignment**

We find house, bidder pair with the highest private value again.

```{code-cell} ipython3
i, j = np.where(V==np.max(V))
i, j
```

In this special example, there are three pairs (1, 1), (2, 3) and (3, 4) with the highest private value. To solve this problem, we choose the one with highest sale price.

```{code-cell} ipython3
p_candidate = np.zeros(len(i))
for k in range(len(i)):
    p_candidate[k] = np.max(np.delete(V[i[k], :], j[k]))
k, = np.where(p_candidate==np.max(p_candidate))
i, j = i[k], j[k]
i, j
```

So, house 1 will be sold to buyer 1 at a price of 7. We update matrices.

```{code-cell} ipython3
p[i] = np.max(np.delete(V[i, :], j))
Q[i, j] = 1
V[i, :] = -1
V[:, j] = -1
p, Q, V
```

**Third assignment**

```{code-cell} ipython3
i, j = np.where(V==np.max(V))
i, j
```

In this special example, there are two pairs (2, 3) and (3, 4) with the highest private value. 

To resolve the  assignment, we choose the one with highest sale price.

```{code-cell} ipython3
p_candidate = np.zeros(len(i))
for k in range(len(i)):
    p_candidate[k] = np.max(np.delete(V[i[k], :], j[k]))
k, = np.where(p_candidate==np.max(p_candidate))
i, j = i[k], j[k]
i, j
```

The two pairs even have the same sale price. 

We randomly choose one pair.

```{code-cell} ipython3
k = np.random.choice(len(i))
i, j = i[k], j[k]
i, j
```

Finally, house 2 will be sold to buyer 3.

We update matrices accordingly.

```{code-cell} ipython3
p[i] = np.max(np.delete(V[i, :], j))
Q[i, j] = 1
V[i, :] = -1
V[:, j] = -1
p, Q, V
```

**Fourth assignment**

```{code-cell} ipython3
i, j = np.where(V==np.max(V))
i, j
```

House 3 will be sold to buyer 4. 

The final outcome  follows.

```{code-cell} ipython3
p[i] = np.max(np.delete(V[i, :], j))
Q[i, j] = 1
V[i, :] = -1
V[:, j] = -1
S = V_orig*Q - np.diag(p)@Q
p, Q, V, S
```

##  Another Python Class

It is efficient to assemble our calculations in a single Python Class.

```{code-cell} ipython3
class GC_Mechanism:
    
    def __init__(self, V):
        """
        Implementation of the special Groves Clarke Mechanism for house auction. 
        
        Parameters:
        ----------
        V: 2 dimensional private value matrix 

        """
        
        self.V_orig = V.copy()
        self.V = V.copy()
        self.n, self.m = self.V.shape
        self.p = np.zeros(self.n)
        self.Q = np.zeros((self.n, self.m))
        self.S = np.copy(self.Q)
        
    def find_argmax(self):
        """
        Find the house-buyer pair with the highest value.
        When the highest private value corresponds to more than one house, bidder pairs, 
        we choose the pair with the highest sale price. 
        Moreoever, if the highest sale price corresponds to two or more pairs with highest private value, 
        We randomly choose one.

        Parameters:
        ----------
        V: 2 dimensional private value matrix with -1 indicating revomed rows and columns
        
        Returns:
        ----------
        i: the index of the sold house

        j: the index of the buyer

        """
        i, j = np.where(self.V==np.max(self.V))
        
        if (len(i)>1):
            p_candidate = np.zeros(len(i))
            for k in range(len(i)):
                p_candidate[k] = np.max(np.delete(self.V[i[k], :], j[k]))
            k, = np.where(p_candidate==np.max(p_candidate))
            i, j = i[k], j[k]
            
            if (len(i)>1):
                k = np.random.choice(len(i))
                k = np.array([k])
                i, j = i[k], j[k]
        return i, j
    
    def update_status(self, i, j):
        self.p[i] = np.max(np.delete(self.V[i, :], j))
        self.Q[i, j] = 1
        self.V[i, :] = -1
        self.V[:, j] = -1
        
    def calculate_surplus(self):
        self.S = self.V_orig*self.Q - np.diag(self.p)@self.Q
        
    def start(self):
        while (np.max(self.V)>=0):
            i, j = self.find_argmax()
            self.update_status(i, j)
            print("House %i is sold to buyer %i at price %i"%(i[0], j[0], self.p[i[0]]))
            print("\n")
        self.calculate_surplus()
        print("Prices of house:\n", self.p)
        print("\n")
        print("The status matrix:\n", self.Q)
        print("\n")
        print("The surplus matrix:\n", self.S)
    
```

```{code-cell} ipython3
np.random.seed(666)

V_orig = np.array([[10, 9, 8, 7, 6], 
                   [9, 9, 7, 6, 6],
                   [8, 6, 6, 9, 4],
                   [7, 5, 6, 4, 9]])
gc_mechanism = GC_Mechanism(V_orig)
gc_mechanism.start()
```

### Elaborations

Here we use some additional notation designed to conform with standard notation in parts of the VCG literature.

We want to verify that our pseudo code is indeed a **pivot mechanism**, also called a **VCG** (Vickrey-Clarke-Groves) mechanism.

  * The mechanism is named after {cite}`Groves_73`, {cite}`Clarke_71`, and {cite}`Vickrey_61`.

To prepare for verifying this, we add some notation.

Let $X$ be the set of feasible allocations of houses under the protocols above (i.e., at most one house to each person).

Let $X(v)$ be the allocation that the mechanism chooses for matrix $v$ of private values.

The mechanism maps a matrix $v$ of private values into an $x \in X$.

Let $v_j(x)$ be the value that person $j$ attaches to allocation $x \in X$.

Let $\check t_j(v)$ the payment that the mechanism charges person $j$.

The  VCG mechanism chooses the allocation

$$
X(v)  = \operatorname{argmax}_{x \in X} \sum_{j=1}^m v_j(x)  
$$ (eq:GC1)

and charges person $j$ a "social cost"

$$
\check t_j(v) = \max_{x \in  X} \sum_{k \neq j} v_k(x) -  \sum_{k \neq j} v_k(X(v)) 
$$ (eq:GC2)

In our setting, equation {eq}`eq:GC1` says that the VCG allocation allocates houses to maximize the total value of the successful prospective buyers.

In our setting, equation {eq}`eq:GC2` says that the mechanism charges people for the externality that their presence in society imposes on other prospective buyers.

Thus, notice that according to equation {eq}`eq:GC2`:

- unsuccessful prospective buyers pay $0$ because removing  them from "society" would not affect the allocation chosen by the mechanim

- successful prospective buyers pay the difference between the total value society could achieve without them present and the total value that others present in society do achieve under the mechanism.

The generalized second-price auction described in our pseudo code above does indeed satisfy (1).
We want to compute $\check t_j$ for $j = 1, \ldots, m$ and compare with $p_j$ from the second price auction.

+++

###  Social Cost

Using the GC_Mechanism class, we can  calculate the social cost of each buyer.

Let's see a simpler example with private value matrix

$$
V =\begin{bmatrix} 10 & 9 & 8 & 7 & 6 \cr
                    9 & 8 & 7 & 6 & 6 \cr
                    8 & 7 & 6 & 5 & 4 \end{bmatrix}
$$

To begin with, we implement the GC mechanism and see the outcome.

```{code-cell} ipython3
np.random.seed(666)

V_orig = np.array([[10, 9, 8, 7, 6], 
                   [9, 8, 7, 6, 6],
                   [8, 7, 6, 5, 4]])
gc_mechanism = GC_Mechanism(V_orig)
gc_mechanism.start()
```

We exclude buyer 0 and calculate the allocation.

```{code-cell} ipython3
V_exc_0 = np.copy(V_orig)
V_exc_0[:, 0] = -1
V_exc_0
gc_mechanism_exc_0 = GC_Mechanism(V_exc_0)
gc_mechanism_exc_0.start()
```

Calculate the social cost of buyer 0.

```{code-cell} ipython3
print("The social cost of buyer 0:", 
     np.sum(gc_mechanism_exc_0.Q*gc_mechanism_exc_0.V_orig)-np.sum(np.delete(gc_mechanism.Q*gc_mechanism.V_orig, 0, axis=1)))
```

Repeat this process for buyer 1 and buyer 2

```{code-cell} ipython3
V_exc_1 = np.copy(V_orig)
V_exc_1[:, 1] = -1
V_exc_1
gc_mechanism_exc_1 = GC_Mechanism(V_exc_1)
gc_mechanism_exc_1.start()

print("\nThe social cost of buyer 1:", 
     np.sum(gc_mechanism_exc_1.Q*gc_mechanism_exc_1.V_orig)-np.sum(np.delete(gc_mechanism.Q*gc_mechanism.V_orig, 1, axis=1)))
```

```{code-cell} ipython3
V_exc_2 = np.copy(V_orig)
V_exc_2[:, 2] = -1
V_exc_2
gc_mechanism_exc_2 = GC_Mechanism(V_exc_2)
gc_mechanism_exc_2.start()

print("\nThe social cost of buyer 2:", 
     np.sum(gc_mechanism_exc_2.Q*gc_mechanism_exc_2.V_orig)-np.sum(np.delete(gc_mechanism.Q*gc_mechanism.V_orig, 2, axis=1)))
```
