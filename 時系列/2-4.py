import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.randn(100).cumsum()

# window = 3 の中央値
x_median = pd.Series(x).rolling(window=3).median()

plt.plot(x, label="original")
plt.plot(x_median, label="移動平均")
plt.show()

# 重みづけ移動平均
def weighted_moving_average_matrix(x, weights):
    k = len(weights)
    X = np.column_stack([x[i:len(x)-k+i+1] for i in range(k)])
    return X @ weights[::-1]

x = np.arange(1, 6)
weights = np.array([0.1, 0.3, 0.6])
y = weighted_moving_average_matrix(x, weights)
print(y)