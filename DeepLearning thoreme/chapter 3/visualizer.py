# 損失局面や誤差の可視化
import numpy as np
import matplotlib.pyplot as plt
def plot_predictions(X, y_true, y_pred, title="モデル比較"):
    plt.figure(figsize=(6, 4))
    plt.scatter(X, y_true, label="True", color="blue", alpha=0.6)
    plt.scatter(X, y_pred, label="Predited", color="red", alpha=0.6)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_surface(loss_values, titles):
    plt.figsize(figsize=(6, 4))
    x = np.arrage(len(loss_values))
    for name, vals in loss_values.itmes():
        plt.plot(x, vals, label=name)
    plt.xlabel("sample index")
    plt.ylabel("loss value")
    plt.title(titles)
    plt.legend()
    plt.show()


from cost_function import CostFunctions
from datasets import generate_linear_data

def plot_loss_surface():
    X, y_true, y_noisy = generate_linear_data()
    cf = CostFunctions()

    w1_vals = np.linspace(0, 4, 50)
    w2_vals = np.linspace(0, 2, 50)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    
    losses = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            y_pred = W1[i, j] * X + W2[i, j]  # f(x; w1, w2)
            losses[i, j] = cf.MSE(y_pred, y_true)
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W1, W2, losses, cmap='viridis')
    ax.set_xlabel("w1")
    ax.set_ylabel("w2 (bias)")
    ax.set_zlabel("Loss (MSE)")
    plt.title("3D Loss Surface")
    plt.show()