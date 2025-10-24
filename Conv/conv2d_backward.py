import jax
import jax.numpy as jnp

def conv2d_backward(dy, x, w, stride=1, padding=0):
    """
    畳み込み層の逆誤差伝播（Backpropagation）
    -------------------------------------------
    dy: 出力側の勾配 (N, C_out, H_out, W_out)
    x:  順伝播時の入力 (N, C_in, H, W)
    w:  フィルタ (C_out, C_in, K_H, K_W)
    stride: ストライド幅
    padding: ゼロパディング幅

    戻り値:
        dx: 入力に対する勾配 (N, C_in, H, W)
        dw: フィルタに対する勾配 (C_out, C_in, K_H, K_W)
        db: バイアスに対する勾配 (C_out,)
    """

    # --- 数式的背景 ---
    # 順伝播: y = x * w + b
    # 損失 L に対する勾配:
    # ∂L/∂b = Σ_{n,i,j} dy_{n,c,i,j}
    # ∂L/∂w = Σ_{n,i,j} dy_{n,c_out,i,j} * x_{n,c_in,i*stride+m,j*stride+n}
    # ∂L/∂x = Σ_{c_out,m,n} dy_{n,c_out,i,j} * w_{c_out,c_in,m,n} を逆畳み込みで計算
    #
    # ∂L/∂w は入力の該当パッチとdyの積和
    # ∂L/∂x は dy をフィルタを180度回転させたものと畳み込み（転置畳み込み）したもの
    #
    # ここでは明示的ループで計算し、理解を深める。

    N, C_in, H, W = x.shape
    C_out, _, K_H, K_W = w.shape

    # パディング処理（forwardと対応）
    if padding > 0:
        x_padded = jnp.pad(
            x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant'
        )
    else:
        x_padded = x

    # 勾配の初期化
    dx = jnp.zeros_like(x_padded)
    dw = jnp.zeros_like(w)
    db = jnp.zeros((C_out,))

    # バイアス勾配: 出力勾配をチャンネル方向で合計
    db = jnp.sum(dy, axis=(0, 2, 3))

    # 出力サイズ
    H_out, W_out = dy.shape[2], dy.shape[3]

    # 逆伝播ループ
    for i in range(H_out):
        for j in range(W_out):
            # 入力の局所領域 (N, C_in, K_H, K_W)
            x_slice = x_padded[:, :, i*stride:i*stride+K_H, j*stride:j*stride+K_W]

            # ∂L/∂w の更新
            # dy[:, :, i, j]: (N, C_out)
            # x_slice: (N, C_in, K_H, K_W)
            # tensordotでN軸を縮約し、(C_out, C_in, K_H, K_W) を得る
            dw += jnp.tensordot(dy[:, :, i, j], x_slice, axes=([0], [0]))

            # ∂L/∂x の更新
            # dy[:, :, i, j]: (N, C_out)
            # w: (C_out, C_in, K_H, K_W)
            # tensordotでC_out軸を縮約し、(N, C_in, K_H, K_W) を得る
            dx = dx.at[:, :, i*stride:i*stride+K_H, j*stride:j*stride+K_W].add(
                jnp.tensordot(dy[:, :, i, j], w, axes=([1], [0]))
            )

    # パディングを除去（元の入力サイズに戻す）
    if padding > 0:
        dx = dx[:, :, padding:-padding, padding:-padding]

    return dx, dw, db


# --- 数式での勾配定義 ---
# ∂L/∂b_c = Σ_{n,i,j} dy_{n,c,i,j}
# ∂L/∂w_{c_out,c_in,m,n} = Σ_{n,i,j} dy_{n,c_out,i,j} * x_{n,c_in,i*stride+m,j*stride+n}
# ∂L/∂x_{n,c_in,i,j} = Σ_{c_out,m,n} dy_{n,c_out,i−m,j−n} * w_{c_out,c_in,m,n} (転置畳み込み)   