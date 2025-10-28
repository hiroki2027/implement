import numpy as np
from cost_function import CostFunctions
from datasets import generate_linear_data

def run_loss_comparison():
    cf = CostFunctions()
    X, y_true, y_noisy = generate_linear_data()
    
    # 仮のモデル出力：y_pred = y_noisy にオフセットを加える
    y_pred = y_noisy + np.random.normal(0, 0.05, len(X))
    
    # 各誤差の計算
    loss_sup = cf.supremum_error(y_pred, y_true)
    loss_l2 = cf.l2_error(y_pred, y_true)
    loss_mse = cf.MSE(y_pred, y_true)
    
    return {
        "Supremum": loss_sup,
        "L2": loss_l2,
        "MSE": loss_mse
    }, X, y_true, y_pred