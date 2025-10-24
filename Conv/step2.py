# Pooling層 (特徴量抽出とダウンさんぷりんぐ)

import torch
import torch.nn as nn

x = torch.arange(1, 17, dtype=torch.float32).view(1, 1, 4, 4)
print(x)

pool = nn.MaxPool2d(kernel_size=2, stride=2)
print(pool(x))

"""
nn.MaxPool2d(
    kernel_size=2  一回に含める領域のサイズ
    stride=2 次の領域に進む間隔
    padding=0 入力の端にバディングする量
    return_indices=False, Trueにすると、最大値の位置インデックスも返す
    ceil_mode = False Trueにすると橋の余部分を切り上げる
    )

"""