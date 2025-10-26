# python
# numpy.sin(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
# ただし、引数xはラジアン単位なのに注意
import numpy as np
import matplotlib.pyplot as plt
def periodic_func(x, period = 2*np.pi):
    return np.sin(2 * np.pi * x / period)

x = np.linspace(0, 4*np.pi, 500)
y = periodic_func(x)

plt.plot(x, y)
plt.title("Periodic Function (Numpy)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()


import jax.numpy as jnp
from jax import grad, jit

# 周期関数を定義
def periodic_func(x, period=2*jnp.pi):
    return jnp.sin(2 * jnp.pi * x / period)

# JITコンパイル
periodic_func_jit = jit(periodic_func)

# 自動微分（勾配）
grad_periodic = grad(periodic_func)

x = jnp.linspace(0, 4*jnp.pi, 500)
y = periodic_func_jit(x)
dy = grad_periodic(x)

plt.plot(x, y, label="f(x)")
plt.plot(x, dy, label="f'(x)")
plt.title("Periodic Function (JAX)")
plt.legend()
plt.show()