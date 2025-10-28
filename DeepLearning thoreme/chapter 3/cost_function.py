# コスト関数群の実装
import numpy as np

class CostFunctions:
    """
    コスト関数群をまとめたクラス
    インスタンス化して利用する
    """

    def supremum_error(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """ 
        上限誤差関数
        最も大きな誤差を返す
        """
        return np.max(np.abs(y_pred - y_true))

    def l2_error(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        L2誤差関数
        """
        diff = y_pred - y_true
        return np.sum(diff**2)
    
    def MSE(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        平均二乗誤差
        """
        diff = y_pred - y_true
        return  np.mean((diff)**2)
    
    def Cross_Entropy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        クロスエントロピー H(p, q) = - Σp_i * log(q_i)
        y_true はone-hotベクトルであることを想定
        
        - np.clip: log(0)を避けるため、y_predの値を [1e-12, 1-1e-12] に制限
        - 出力平均を取る理由: 各サンプルの損失の期待値（平均）を近似するため。
          期待値 E[...] を有限サンプルで推定すると平均になる。
          np.clip()はnp.log(0)を防いでいる
        """
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        ce = -np.sum(y_true * np.log(y_pred))
        return ce / y_true.shape[0]
    
    def shannon_Entropy(self, p) -> float:
        """
        シャノンエントロピーの実装　S(p,q) = - Σ p_i * log(p_i)
        """

        epsilon = 1e-12
        p = np.clip(p, epsilon, 1. - epsilon)
        entropy= -np.sum(p * np.log(p))
        return entropy/p.shape[0]