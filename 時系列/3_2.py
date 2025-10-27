import numpy as np
import matplotlib.pyplot as plt

"""
離散時間データを三角関数の線形和で生成し、
尤度最大化（= ガウス雑音仮定下の最小二乗）で係数を推定する最小構成。
- モデル: x[n] = β0 + β1 cos(ω1 n) + α1 sin(ω1 n) + β2 cos(ω2 n) + α2 sin(ω2 n) + ε[n]
- 推定: β^ = argmin ||x - Φβ||^2  (NumPyの最小二乗で解く)
"""

# -----------------------------
# 1) データ生成
# -----------------------------
N = 128
n = np.arange(N)
fs = 128.0

# 周波数（Hz）
f1 = 5.0
f2 = 10.0
w1 = 2 * np.pi * f1 / fs
w2 = 2 * np.pi * f2 / fs

# 真の係数
beta0_true = 0.5
beta1_true, alpha1_true = 1.0, 0.8
beta2_true, alpha2_true = 0.6, -0.4

# 信号生成
rng = np.random.default_rng(42)
x = (
    beta0_true
    + beta1_true * np.cos(w1 * n)
    + alpha1_true * np.sin(w1 * n)
    + beta2_true * np.cos(w2 * n)
    + alpha2_true * np.sin(w2 * n)
    + 0.1 * rng.standard_normal(N)
)

# -----------------------------
# 2) デザイン行列 Φ の構築
#    Φ = [1, cos(w1 n), sin(w1 n), cos(w2 n), sin(w2 n)]
# -----------------------------
Phi = np.column_stack([
    np.ones(N),
    np.cos(w1 * n),
    np.sin(w1 * n),
    np.cos(w2 * n),
    np.sin(w2 * n),
])

# -----------------------------
# 3) 最尤推定（最小二乗）
#    lstsq は数値的に安定: β^, residuals, rank, s
# -----------------------------
beta_hat, residuals, rank, s = np.linalg.lstsq(Phi, x, rcond=None)

# 予測
x_hat = Phi @ beta_hat

# ガウス雑音の分散推定
# MLE (β, σ^2 同時): σ^2_MLE = RSS / N
# 不偏推定量: σ^2_unbiased = RSS / (N - p)
RSS = np.sum((x - x_hat) ** 2)
p = Phi.shape[1]
sigma2_mle = RSS / N
sigma2_unbiased = RSS / (N - p)

print("真の係数 [β0, β1, α1, β2, α2] =",
      [beta0_true, beta1_true, alpha1_true, beta2_true, alpha2_true])
print("推定係数 β^                 =", beta_hat.round(4).tolist())
print(f"RSS = {RSS:.4f}")
print(f"σ^2_MLE = {sigma2_mle:.6f},  σ^2_unbiased = {sigma2_unbiased:.6f}")

# -----------------------------
# 4) 可視化（最小限）
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(n, x, label="observed x[n]", linewidth=1, alpha=0.75)
plt.plot(n, x_hat, label="fitted $\\hat{x}[n]$", linewidth=2)
plt.xlabel("n")
plt.ylabel("amplitude")
plt.title("Sinusoidal Linear Model: MLE (Least Squares) Fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
