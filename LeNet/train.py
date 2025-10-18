import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from LeNet.models.lenet import LeNet

def main():
    # データセ,ット
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256)

    # モデル、損失関数、最適化
    model = LeNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    # 学習ループ
    for epoch in range(5):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                pred = model(X).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            print(f"Epoch {epoch+1}, Accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    main()
