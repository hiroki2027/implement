import torch 
import torch.nn as nn

conv = nn.Conv2d(
    in_channels = 3, # 入力チャンネル数
    out_channels = 16, # 出力チャンネル（フィルタ数)
    kernel_size = 3, # カーネルサイズ
    stride = 1, # ストライド
    padding = 0, # バディング
    dilation = 1, # カーネルの膨張率
    groups = 1, # グループ数
    bias = True, #バイアスを使うかどうか
    padding_mode = 'zeros' # バディングモード
)

# 入力: (N, C_in, H, W)
x = torch.rand(1, 3, 32, 32)

# 出力
y = conv(x)

print(y)