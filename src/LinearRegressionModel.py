import torch as t
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

#Static variables
weight = 3
bias = 4

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
t.cuda.set_device(0)

data = t.arange(0.0, 10.0, 0.2).unsqueeze(1).to(device)
data_label = data*weight+bias
data_label = data_label.to(device)
test_split = int(len(data) * 0.8)

X_train = data[:test_split].to(device)
y_train = data_label[:test_split].to(device)
X_test = data[test_split:].to(device)
y_test = data_label[test_split:].to(device)


def main():
    t.manual_seed(60)
    model = LinearRegressionModel().to(device)
    loss_fn = nn.modules.loss.MSELoss().to(device)
    #After brief testing, MSELoss is the best for range 0-10
    #The best for range 0-1 is still L1Loss
    optimizer = optim.ASGD(model.parameters(),
                          lr=0.005)

    trainModel(model, loss_fn, optimizer, epochs=500)

    model.eval()    #turns off different settings in the model not needed for evaluation (dropout/batch norm layers)
    with t.inference_mode():    #turns off gradient tracking and a few more things not needed for evaluation
        y_results = testModel(model, data)

    plotData(predictions=y_results)
    print(model.state_dict())




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

        trainingModel.eval()  # turns off different settings in the model not needed for evaluation (dropout/batch norm layers)
        with t.inference_mode():  # turns off gradient tracking and a few more things not needed for evaluation
            test_results = testModel(trainingModel, X_test)
            test_loss = loss_fn(test_results, y_test)
        print(f"epoch {epoch}, test loss: {test_loss.item()}")

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

    plt.scatter(training_data.to("cpu"), training_label.to("cpu"), c='b', label='training data', s=6)

    plt.scatter(testing_data.to("cpu"), testing_label.to("cpu"), color='g', label='testing data', s=6)

    if predictions is not None:
        plt.scatter(data.to("cpu"), predictions.to("cpu"), color='r', label='predictions', s=6)

    plt.legend(prop={'size': 10})
    plt.show()

if __name__ == '__main__':
    main()

