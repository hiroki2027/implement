# 活性化関数（非線形変換)

# CNNを線型写像の連続ではなく、”表現力のある非線形関数"にする

import torch
import torch.nn as nn

act = nn.ReLU()
print(act(torch.tensor([-2.0, -1.0, 0.0, 2.0])))