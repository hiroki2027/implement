# 時系列データの生成観測メカニズム
# 同時確率分布による表現

# 時系列データの生成観測メカニズムの一つである確率家庭についての理解
# 時系列データはデータ生成観測メカニズムに従って得られる

# 時系列データ　x = (x_1, ,,, , x_n)が確率分布p(X_1, ,, , X_n)に従い発生

import numpy as np
import matplotlib.pyplot as plt

# AR(1)モデルによる時系列の生成
# AR(1)（自己回帰モデル・一次）は1時点前の値が次の値に影響を与える確率モデル」
# x_t = Φ * x_{t-1} + ε_t

np.random.seed(42)
N = 100
phi = 0.8
sigma = 0.5

x = np.zeros(N)
for t in range(1, N):
    x[t] = phi * x[t-1] + np.random.normal(0, sigma)

plt.plot(x, label="x_tを生成")
plt.xlabel("t")
plt.ylabel("x_t")
plt.title("AR(1) model")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 周辺分布（定常分布）の確認
# -----------------------------
# 理論上の定常分布: x_t ~ N(0, σ^2 / (1 - φ^2))
mean_theoretical = 0
var_theoretical = sigma**2 / (1 - phi**2)
std_theoretical = np.sqrt(var_theoretical)

# ヒストグラムと理論分布の比較
from scipy.stats import norm

plt.figure(figsize=(8, 5))
count, bins, _ = plt.hist(x[int(N/5):], bins=20, density=True, alpha=0.6, label="Empirical (simulated)")
pdf = norm.pdf(bins, mean_theoretical, std_theoretical)
plt.plot(bins, pdf, 'r-', lw=2, label=f"Theoretical N(0, {std_theoretical**2:.3f})")
plt.title("Marginal Distribution of AR(1) process")
plt.xlabel("x_t")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
