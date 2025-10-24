# 目的: 画素がどう変換されるかを理解
# カーネうががそのエッジや模様を検出する様子を見る

# 各出力画素は周囲ピクセルとフィルタの積和でもとまる

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.randn(1, 1, 5, 5)
print(x)
kernel = torch.tensor([[1., 0., -1.],
                       [1., 0., -1.],
                       [1., 0., -1.]]).unsqueeze(0).unsqueeze(0)

y = F.conv2d(x, kernel)
print(y)
print(y.shape)