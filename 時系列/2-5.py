# データ観測メカニズム
# 時系列データ x = (x_1, ,,, , x_n)：かくx_tがp(x_t| t)に従い発生
# 代表値　x_t = f(t) + ε_t
# ε_t は確率分に従う確率変数

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 4*np.pi, 1000)
x_t = np.sin(t)

plt.plot(x_t)
plt.title("周期関数")
plt.show()

# 時系列データx のp生成観測メカニズム
# x_t = f(t) + ε_t
# f(t): 複数の関数Φ_i(t)(i=0,,,,m)の重み付線形和で表される (係数はβ_i)
# ε_t: N(o, (σ_ε)**2)に従う
# 時間tを説明変数、x_tを目的変数とした回帰分析と同様に考えることができる

# 時間軸
t = np.linspace(0, 10, 200)

# f(t)の基底関数
phi1 = np.sin(t)
phi2 = np.cos(2*t)
phi3 = 0.1*t

# 各基底関数の係数 
beta = np.array([1.2, -0.8, 0.5])

# 真の関数
f_t = beta[0]*phi1 + beta[1]*phi2 + beta[2]*phi3

# ノイズ
epsilon = np.random.normal(0, 0.2, len(t))

# 観測値
x_t = f_t + epsilon

# --- 可視化
plt.figure(figsize=(10,6))
plt.plot(t, f_t, label="真の関数 f(t)", color="black", linewidth=2)
plt.scatter(t, x_t, label="観測データ x_t = f(t) + ε_t", color="red", s=10, alpha=0.6)
plt.plot(t, phi1, "--", label="基底関数 Φ₁(t)=sin(t)", alpha=0.5)
plt.plot(t, phi2, "--", label="基底関数 Φ₂(t)=cos(2t)", alpha=0.5)
plt.plot(t, phi3, "--", label="基底関数 Φ₃(t)=0.1t", alpha=0.5)

plt.title("時系列データ生成観測メカニズム: xₜ = f(t) + εₜ", fontsize=13)
plt.xlabel("時間 t")
plt.ylabel("値")
plt.legend()
plt.grid(True)
plt.show()

"""
設定関数 x_t で基底関数は既知、係数betaは未知とする
評価基準：尤度最大
色々計算すると、最小二乗誤差を最小にするbetaを求めたいことがわかる
"""

import torch
import torch.nn as nn
import torch.optim as optim

t = torch.linspace(0, 10, 200).reshape(-1, 1)

phi1 = torch.sin(t)
phi2 = torch.cos(2*t)
phi3 = 0.1*t

Phi = torch.cat([phi1, phi2, phi3], dim=1)

beta_true = torch.tensor([[1.2], [-0.8], [0.5]])
x_t = Phi @ beta_true + 0.2 * torch.randn_like(t)

# betaを学習対象パラメータとする
beta = torch.randn((3, 1), requires_grad=True)
optimizer = optim.Adam([beta], lr=0.05)

for epoch in range(1000):
    optimizer.zero_grad()
    loss = torch.mean((Phi @ beta - x_t)**2)
    loss.backward()
    optimizer.step()

print("推定された β =", beta.detach().flatten().numpy())