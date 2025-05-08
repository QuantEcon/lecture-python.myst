import jax
import jax.numpy as jnp

devices = jax.devices()
print(f"The available devices are: {devices}")

@jax.jit
def matrix_multiply(a, b):
    return jnp.dot(a, b)

# Example usage:
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1000, 1000))
y = jax.random.normal(key, (1000, 1000))
z = matrix_multiply(x, y)

# Now the function is JIT compiled and will likely run on GPU (if available)
print(z)

devices = jax.devices()
print(f"The available devices are: {devices}")