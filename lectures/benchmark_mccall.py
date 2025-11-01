import matplotlib.pyplot as plt
import numpy as np
import numba
import jax
import jax.numpy as jnp
from typing import NamedTuple
import quantecon as qe
from quantecon.distributions import BetaBinomial
import time

# Setup default parameters
n, a, b = 50, 200, 100
q_default = np.array(BetaBinomial(n, a, b).pdf())
q_default_jax = jnp.array(BetaBinomial(n, a, b).pdf())

w_min, w_max = 10, 60
w_default = np.linspace(w_min, w_max, n+1)
w_default_jax = jnp.linspace(w_min, w_max, n+1)

# McCall model for JAX
class McCallModel(NamedTuple):
    c: float = 25
    β: float = 0.99
    w: jnp.ndarray = w_default_jax
    q: jnp.ndarray = q_default_jax

def compute_reservation_wage_two(model, max_iter=500, tol=1e-5):
    c, β, w, q = model.c, model.β, model.w, model.q
    h = (w @ q) / (1 - β)
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        s = jnp.maximum(w / (1 - β), h)
        h_next = c + β * (s @ q)
        error = jnp.abs(h_next - h)
        h = h_next
        i += 1

    return (1 - β) * h

# =============== NUMBA SOLUTION ===============
cdf_numba = np.cumsum(q_default)

@numba.jit
def compute_stopping_time_numba(w_bar, seed=1234):
    np.random.seed(seed)
    t = 1
    while True:
        w = w_default[qe.random.draw(cdf_numba)]
        if w >= w_bar:
            stopping_time = t
            break
        else:
            t += 1
    return stopping_time

@numba.jit
def compute_mean_stopping_time_numba(w_bar, num_reps=100000):
    obs = np.empty(num_reps)
    for i in range(num_reps):
        obs[i] = compute_stopping_time_numba(w_bar, seed=i)
    return obs.mean()

# =============== JAX SOLUTION ===============
cdf_jax = jnp.cumsum(q_default_jax)

@jax.jit
def compute_stopping_time_jax(w_bar, key):
    def update(state):
        t, key, done = state
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey)
        w = w_default_jax[jnp.searchsorted(cdf_jax, u)]
        done = w >= w_bar
        t = jnp.where(done, t, t + 1)
        return t, key, done

    def cond(state):
        t, _, done = state
        return jnp.logical_not(done)

    initial_state = (1, key, False)
    t_final, _, _ = jax.lax.while_loop(cond, update, initial_state)
    return t_final

def compute_mean_stopping_time_jax(w_bar, num_reps=100000, seed=1234):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, num_reps)
    compute_fn = jax.jit(jax.vmap(compute_stopping_time_jax, in_axes=(None, 0)))
    obs = compute_fn(w_bar, keys)
    return jnp.mean(obs)

# =============== BENCHMARKING ===============
def benchmark_numba():
    c_vals = np.linspace(10, 40, 25)
    stop_times = np.empty_like(c_vals)

    # Warmup
    mcm = McCallModel(c=25.0)
    w_bar = compute_reservation_wage_two(mcm)
    _ = compute_mean_stopping_time_numba(float(w_bar), num_reps=1000)

    # Actual benchmark
    start = time.time()
    for i, c in enumerate(c_vals):
        mcm = McCallModel(c=float(c))
        w_bar = compute_reservation_wage_two(mcm)
        stop_times[i] = compute_mean_stopping_time_numba(float(w_bar))
    end = time.time()

    return end - start, stop_times

def benchmark_jax():
    c_vals = jnp.linspace(10, 40, 25)
    stop_times = np.empty_like(c_vals)

    # Warmup - compile the functions
    model = McCallModel(c=25.0)
    w_bar = compute_reservation_wage_two(model)
    _ = compute_mean_stopping_time_jax(w_bar, num_reps=1000).block_until_ready()

    # Actual benchmark
    start = time.time()
    for i, c in enumerate(c_vals):
        model = McCallModel(c=c)
        w_bar = compute_reservation_wage_two(model)
        stop_times[i] = compute_mean_stopping_time_jax(w_bar).block_until_ready()
    end = time.time()

    return end - start, stop_times

if __name__ == "__main__":
    print("Benchmarking Numba vs JAX solutions for ex_mm1...")
    print("=" * 60)

    print("\nRunning Numba solution...")
    numba_time, numba_results = benchmark_numba()
    print(f"Numba time: {numba_time:.2f} seconds")

    print("\nRunning JAX solution...")
    jax_time, jax_results = benchmark_jax()
    print(f"JAX time: {jax_time:.2f} seconds")

    print("\n" + "=" * 60)
    print(f"Speedup: {numba_time/jax_time:.2f}x faster with {'JAX' if jax_time < numba_time else 'Numba'}")
    print("=" * 60)

    # Verify results are similar
    max_diff = np.max(np.abs(numba_results - jax_results))
    print(f"\nMaximum difference in results: {max_diff:.6f}")
    print(f"Results are {'similar' if max_diff < 1.0 else 'different'}")
