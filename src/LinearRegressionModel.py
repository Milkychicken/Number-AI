import torch as t
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

#Static variables
weight = 0.7
bias = 0.1

data = t.arange(0.0, 10.0, 0.2).unsqueeze(1)
data_label = data*weight+bias
test_split = int(len(data) * 0.8)

X_train = data[:test_split]
y_train = data_label[:test_split]
X_test = data[test_split:]
y_test = data_label[test_split:]


def main():
    t.manual_seed(60)
    model = LinearRegressionModel()
    loss_fn = nn.modules.loss.MSELoss()
    #After brief testing, MSELoss is the best for range 0-10
    #The best for range 0-1 is still L1Loss
    optimizer = optim.SGD(model.parameters(),
                          lr=0.02)
    trainModel(model, loss_fn, optimizer, epochs=3)
    plotData(predictions=testModel(model, X_test))



class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(t.rand(1))
        self.bias = nn.Parameter(t.rand(1))
        
    def forward(self, X):
        return self.weight * X + self.bias


def trainModel(trainingModel, loss_fn, optimizer, trainData = X_train, trainLabels = y_train, epochs = 5):
    for epoch in range(epochs):
        trainingModel.train()
        y_predictions = trainingModel(trainData)
        loss = loss_fn(y_predictions, trainLabels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def testModel(model, testData):
    model.eval()
    with t.inference_mode():
        y_predictions = model(testData)
    return y_predictions

def plotData(training_data=X_train,
             training_label=y_train,
             testing_data=X_test,
             testing_label=y_test,
             predictions=None):
    plt.figure(0, figsize=(10, 8))
    plt.xlabel('Input')
    plt.ylabel('(desired) Output')

    plt.scatter(training_data, training_label, c='b', label='training data', s=6)

    plt.scatter(testing_data, testing_label, color='g', label='testing data', s=6)

    if predictions is not None:
        plt.scatter(testing_data, predictions, color='r', label='predictions', s=6)

    plt.legend(prop={'size': 10})
    plt.show()

if __name__ == '__main__':
    main()

