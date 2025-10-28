# 実行スクリプト
from experiments import run_loss_comparison
from visualizer import plot_predictions
from datasets import generate_linear_data
from visualizer import plot_loss_surface
if __name__ == "__main__":
    results, X, y_true, y_pred = run_loss_comparison()
    
    print("=== Loss Comparison ===")
    for name, val in results.items():
        print(f"{name}: {val:.6f}")
    
    # 結果の可視化
    plot_predictions(X, y_true, y_pred, title="True vs Predicted")
    plot_loss_surface()