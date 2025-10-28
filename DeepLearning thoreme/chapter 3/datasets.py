# ダミーデータ生成
import numpy as np

def generate_linear_data(n_samples=100, noise_std=0.1, seed=42):
    """
    y = 2x + 1にノイズを加えたダミーデータの作成
    """
    np.random.seed(seed)
    X  = np.linspace(-1,1, n_samples)
    y_true = 2*X + 1
    noise = np.random.normal(0, noise_std, n_samples)
    y_noisy = y_true + noise
    return X, y_true, y_noisy

