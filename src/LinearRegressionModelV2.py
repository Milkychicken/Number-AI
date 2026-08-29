""" A new and improved LinearRegressPionModel implementing a linear layer """

import matplotlib.pyplot as plt
import torch as t
from torch import nn
from pathlib import Path

device = "cuda:0" if t.cuda.is_available() else "cpu"

weight = 2
bias = 0.7

data_X = t.arange(-2, 1, 0.05, device=device).unsqueeze(1)
data_y = data_X * weight + bias
test_split = int(len(data_X)*0.8)

train_X = data_X[:test_split].to(device)
train_y = data_y[:test_split].to(device)
test_X = data_X[test_split:].to(device)
test_y = data_y[test_split:].to(device)

MODELS_PATH = Path("resources/models")

def main():
    model = LinearRegressionModelV2().to(device)
    loss_fn = nn.L1Loss().to(device)
    optim = t.optim.Adam(model.parameters(), lr=0.03)

    train_model(model, train_X, train_y, loss_fn, optim, 100)
    print(model.state_dict())

    test_model(model, data_X, data_y)
    model_save_path = MODELS_PATH / "LinearRegressionModeV2.pth"
    t.save(obj=model.state_dict(), f=model_save_path)


class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)

    def forward(self, x: t.Tensor):
        return self.linear_layer(x)

def train_model(model: nn.Module, training_data_X: t.Tensor, training_data_y: t.Tensor,
                loss_fn, optim, epochs: int):
    # tracking variables
    epoch_history = []
    loss_history = []

    model.train()
    for i in range(epochs):
        optim.zero_grad()
        output = model(training_data_X)
        loss = loss_fn(output, training_data_y)
        loss.backward()
        optim.step()

        epoch_history.append(i)
        loss_history.append(loss.item())

    plt.figure(1)
    plt.title(f"Loss history over {epochs} epochs")
    plt.scatter(epoch_history, loss_history, c="b")
    plt.show()

def test_model(model: nn.Module, test_data_X: t.Tensor, test_data_y: t.Tensor):
    model.eval()
    with t.inference_mode():
        y_predictions = model(test_data_X)
    plot_data(test_data_X, test_data_y, y_predictions)

def plot_data(x_input: t.Tensor, y_data: t.Tensor, y_preds: t.Tensor):
    plt.figure(2)
    plt.title("y_predictions vs. y_data")
    plt.scatter(x_input.cpu(), y_preds.cpu(), c="r")
    plt.scatter(x_input.cpu(), y_data.cpu(), c="g")
    plt.show()


main()