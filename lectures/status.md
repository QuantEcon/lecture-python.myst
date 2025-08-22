---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Execution Statistics

This table contains the latest execution statistics.

```{nb-exec-table}
```

(status:machine-details)=

These lectures are built on `linux` instances through `github actions`. 

These lectures are using the following python version

```{code-cell} ipython
!python --version
```

and the following package versions

```{code-cell} ipython
:tags: [hide-output]
!conda list
```

This lecture series has access to the following GPU

```{code-cell} ipython
!nvidia-smi
```

You can check the backend used by JAX using:

```{code-cell} ipython3
import jax
# Check if JAX is using GPU
print(f"JAX backend: {jax.devices()[0].platform}")
```