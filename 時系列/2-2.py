import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 仮の時系列データを生成
np.random.seed(0)
x = np.random.randn(100).cumsum()  # 累積和でトレンドっぽく

# 移動平均 
window = 3
x_sma = pd.Series(x).rolling(window=window).mean()

# プロット
plt.figure(figsize=(8,4))
plt.plot(x, label='original')
plt.plot(x_sma, label=f'{window}-point moving average', linewidth=2)
plt.legend()
plt.title('Moving Average for Trend Extraction')
plt.show()

# 数値的可視化
# 標本平均
x_hat = np.mean(x)
y_hat = np.mean(x_sma)
# 分散
v_x = np.var(x, ddof=0)
v_x_sma = np.var(x_sma, ddof=0)

print(f'平均 (mean): {x_hat}')
print(f'分散 (variance): {v_x}')

print(f'移動平均の平均: {y_hat}')
print(f'移動平均の分散: {v_x_sma}')

# 時系列データの可視化
# 標本平均や標本分散、ヒストグラム：iの序列構造を考慮しない数値化、視覚化
# i の序列構造に着目した数値化。視覚化
# 標本自己分散、標本自己相関、散布図：離れた時点での系列との関係を数値化、視覚化

# 標本自己分散：共分散の量変数を同一時系列にして、片方をhステップずらしたもの
def C(x, h):
    x = np.array(x)
    n = len(x)
    x_mean = np.mean(x)
    return np.sum((x[:n-h] - x.mean()) * (x[h:] - x.mean())) / n

# 時系列データの数値的可視化
# 自己共分散：二つの変数列の共分散
def var(x):
    return C(x, 0)

# 標本自己相関係数
def p(x, h):
    return C(x, h) / var(x)

# 自己共分散・自己相関を計算し表示
print("\nLag h | 自己共分散 C(x, h) | 自己相関 p(x, h)")
print("----------------------------------------------")
for h in range(1, 6):
    cov = C(x, h)
    corr = p(x, h)
    print(f"{h:5d} | {cov:16.5f} | {corr:16.5f}")

# 自己相関が1に近い = 線形的なトレンドを持っている
# 自己相関が周期的に + - + - = 周期構造をもつ
# 自己相関が急激に減衰して0に近づく　= ランダム性が強い
