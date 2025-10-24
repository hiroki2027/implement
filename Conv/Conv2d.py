import jax
import jax.numpy as jnp

"""

このコードは、出力特徴マップの各位置に「入力画像の対応する領域とフィルタを掛けた積和結果」を1つずつ代入しており、
出力の空間サイズはカーネルサイズ・ストライド・パディングによって決まるため、
通常は入力画像サイズとは異なる（ただし適切なパディングを使えば一致させることも可能）

"""

def conv2d_forward(x, w, b, stride=1, padding=0):
    """
    2D畳み込み層の順伝播計算
    x: 入力 (N, C_in, H, W)
    w: フィルタ (C_out, C_in, K_H, K_W)
    b: バイアス (C_out,)
    """

    n, c_in, h, w_in = x.shape
    c_out, _, kh, kw = w.shape

    # パディング処理
    if padding > 0:
        x = jnp.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')

    # 出力サイズ計算　バディング処理で+2Pは処理ずみ
    h_out = (x.shape[2] - kh) // stride + 1
    w_out = (x.shape[3] - kw) // stride + 1

    # 出力特徴マップ初期化
    y = jnp.zeros((n, c_out, h_out, w_out))

    # 数式に忠実な三重ループ
    for i in range(h_out):
        for j in range(w_out):
            # 入力の局所領域を切り出し
            x_slice = x[:, :, i*stride:i*stride+kh, j*stride:j*stride+kw]

            # Σ x*w + b
            # 行列の積ではないことに注意
            y = y.at[:, :, i, j].set(jnp.tensordot(x_slice, w, axes=([1,2,3],[1,2,3])) + b)

    return y