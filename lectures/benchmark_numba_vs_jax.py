"""
Benchmark comparing parallel Numba vs optimized JAX for ex_mm1
"""

import time
import numpy as np
import numba
import jax
import jax.numpy as jnp
from functools import partial
import quantecon as qe
from typing import NamedTuple

# Setup model parameters
class McCallModel(NamedTuple):
    c: float = 25          # unemployment compensation
    β: float = 0.99        # discount factor
    w: jnp.ndarray = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    q: jnp.ndarray = jnp.array([0.1, 0.15, 0.2, 0.25, 0.2, 0.1])

# Default values
q_default = jnp.array([0.1, 0.15, 0.2, 0.25, 0.2, 0.1])
w_default = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

# ============================================================================
# PARALLEL NUMBA VERSION
# ============================================================================

q_default_np = np.array(q_default)
w_default_np = np.array(w_default)
cdf_np = np.cumsum(q_default_np)

@numba.jit
def compute_stopping_time_numba(w_bar, seed=1234):
    np.random.seed(seed)
    t = 1
    while True:
        w = w_default_np[qe.random.draw(cdf_np)]
        if w >= w_bar:
            stopping_time = t
            break
        else:
            t += 1
    return stopping_time

@numba.jit(parallel=True)
def compute_mean_stopping_time_numba(w_bar, num_reps=100000):
    obs = np.empty(num_reps)
    for i in numba.prange(num_reps):
        obs[i] = compute_stopping_time_numba(w_bar, seed=i)
    return obs.mean()

# ============================================================================
# OPTIMIZED JAX VERSION
# ============================================================================

cdf_jax = jnp.cumsum(q_default)

@jax.jit
def compute_stopping_time_jax(w_bar, key):
    """
    Optimized version with better state management.
    Key improvement: Check acceptance condition before incrementing t,
    avoiding redundant jnp.where operation.
    """
    def update(loop_state):
        t, key, accept = loop_state
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey)
        w = w_default[jnp.searchsorted(cdf_jax, u)]
        accept = w >= w_bar
        t = t + 1
        return t, key, accept

    def cond(loop_state):
        _, _, accept = loop_state
        return jnp.logical_not(accept)

    initial_loop_state = (0, key, False)
    t_final, _, _ = jax.lax.while_loop(cond, update, initial_loop_state)
    return t_final

@partial(jax.jit, static_argnames=('num_reps',))
def compute_mean_stopping_time_jax(w_bar, num_reps=100000, seed=1234):
    """
    Generate a mean stopping time over `num_reps` repetitions by repeatedly
    drawing from `compute_stopping_time`.
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, num_reps)
    # Vectorize compute_stopping_time and evaluate across keys
    compute_fn = jax.vmap(compute_stopping_time_jax, in_axes=(None, 0))
    obs = compute_fn(w_bar, keys)
    return jnp.mean(obs)

# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark(num_trials=5, num_reps=100000):
    """
    Benchmark parallel Numba vs optimized JAX.
    """
    w_bar = 35.0

    print("="*70)
    print("Benchmark: Parallel Numba vs Optimized JAX (ex_mm1)")
    print("="*70)
    print(f"Number of MC replications: {num_reps:,}")
    print(f"Number of benchmark trials: {num_trials}")
    print(f"Reservation wage: {w_bar}")
    print(f"Number of CPU threads: {numba.config.NUMBA_NUM_THREADS}")
    print()

    # Warm-up runs
    print("Warming up...")
    _ = compute_mean_stopping_time_numba(w_bar, num_reps=1000)
    _ = compute_mean_stopping_time_jax(w_bar, num_reps=1000).block_until_ready()
    print("Warm-up complete.\n")

    results = {}

    # Benchmark Numba (Parallel)
    print("Benchmarking Numba (Parallel)...")
    numba_times = []
    for i in range(num_trials):
        start = time.perf_counter()
        result = compute_mean_stopping_time_numba(w_bar, num_reps=num_reps)
        elapsed = time.perf_counter() - start
        numba_times.append(elapsed)
        print(f"  Trial {i+1}: {elapsed:.4f} seconds")

    numba_mean = np.mean(numba_times)
    numba_std = np.std(numba_times)
    results['Numba (Parallel)'] = (numba_mean, numba_std, result)
    print(f"  Mean: {numba_mean:.4f} ± {numba_std:.4f} seconds")
    print(f"  Result: {result:.4f}\n")

    # Benchmark JAX (Optimized)
    print("Benchmarking JAX (Optimized)...")
    jax_times = []
    for i in range(num_trials):
        start = time.perf_counter()
        result = compute_mean_stopping_time_jax(w_bar, num_reps=num_reps).block_until_ready()
        elapsed = time.perf_counter() - start
        jax_times.append(elapsed)
        print(f"  Trial {i+1}: {elapsed:.4f} seconds")

    jax_mean = np.mean(jax_times)
    jax_std = np.std(jax_times)
    results['JAX (Optimized)'] = (jax_mean, jax_std, float(result))
    print(f"  Mean: {jax_mean:.4f} ± {jax_std:.4f} seconds")
    print(f"  Result: {float(result):.4f}\n")

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Implementation':<25} {'Time (s)':<20} {'Relative Performance'}")
    print("-"*70)

    for name, (mean_time, std_time, _) in results.items():
        print(f"{name:<25} {mean_time:>6.4f} ± {std_time:<6.4f}")

    print("-"*70)

    # Determine winner
    if numba_mean < jax_mean:
        speedup = jax_mean / numba_mean
        print(f"\n🏆 WINNER: Numba (Parallel)")
        print(f"   Numba is {speedup:.2f}x faster than JAX")
    else:
        speedup = numba_mean / jax_mean
        print(f"\n🏆 WINNER: JAX (Optimized)")
        print(f"   JAX is {speedup:.2f}x faster than Numba")

    print("="*70)

if __name__ == "__main__":
    benchmark()
