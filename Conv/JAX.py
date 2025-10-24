# Flaxを導入
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random

# Convそうの定義
class ConvLayer(nn.Module):
    out_channels: int = 16
    kernel_size: int = 3
    stride: int = 1
    padding: str = "VALID"
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        return nn.Conv(
            features=self.out_channels,
            kernel_size = (self.kernel_size, self.kernel_size),
            strides = (self.stride, self.stride),
            padding = self.padding,
            use_bias = self.use_bias
              ) (x)
    
    # 🔹 入力データ（PyTorchと同じ形）
# Flaxは (N, H, W, C) なので転置が必要
key = random.PRNGKey(0)
x = random.normal(key, (1, 32, 32, 3))  # (N, H, W, C)

# 🔹 モデル初期化
model = ConvLayer(out_channels=16, kernel_size=3)
params = model.init(key, x)  # パラメータを初期化

# 🔹 順伝播（forward）
y = model.apply(params, x)

print("出力のshape:", y.shape)
print("出力:", y)