import jax
import jax.numpy as jnp
from Conv2d import conv2d_forward

# ===== 1. ダミー入力を作る =====
# N: バッチサイズ, C_in: 入力チャネル, H: 高さ, W: 幅
N, C_in, H, W = 1, 1, 5, 5
x_demo = jnp.arange(N * C_in * H * W)
print("入力 x:\n", x_demo)
x = jnp.arange(N * C_in * H * W).reshape(N, C_in, H, W)
print("入力 x:\n", x)
print("x.shape =", x.shape)

# ===== 2. フィルタとバイアスを定義 =====
C_out, K_H, K_W = 1, 3, 3
w = jnp.ones((C_out, C_in, K_H, K_W))  # 3x3のフィルタ（全要素1）
b = jnp.zeros((C_out,))                # バイアス0
print(w)
print("\nフィルタ w.shape =", w.shape)
print("バイアス b.shape =", b.shape)

# ===== 3. 畳み込みの順伝播 =====
y = conv2d_forward(x, w, b, stride=1, padding=0)

# ===== 4. 結果を出力 =====
print("\n出力 y:\n", y)
print("y.shape =", y.shape)

# ===== 5. デバッグ：中身の流れを確認（例として最初の出力要素） =====
stride = 1
padding = 0
kh, kw = K_H, K_W
i, j = 0, 0  # 最初の出力位置
x_padded = jnp.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')

x_slice = x_padded[:, :, i*stride:i*stride+kh, j*stride:j*stride+kw]
print("\n最初の局所領域 x_slice:\n", x_slice)
print("対応するフィルタ w:\n", w)
print("積和結果:", jnp.tensordot(x_slice, w, axes=([1,2,3],[1,2,3])))