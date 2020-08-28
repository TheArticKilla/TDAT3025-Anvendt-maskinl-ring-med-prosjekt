import torch
import matplotlib.pyplot as plt
import pandas as pd

learning_rate = 0.000001
epochs = 100000

file = pd.read_csv("datasets/day_head_circumference.csv")
x_train = torch.tensor(file.day).reshape(-1, 1)
y_train = torch.tensor(file.head_circumference).reshape(-1, 1)

class NonLinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return 20*torch.sigmoid(x @ self.W.double() + self.b.double()) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x.double()), y.double())


if __name__ == '__main__':
    model = NonLinearRegressionModel()

    # Optimize: adjust W and b to minimize loss using stochastic gradient descent
    optimizer = torch.optim.SGD([model.W, model.b], learning_rate)
    for epoch in range(epochs):
        model.loss(x_train, y_train).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b

        optimizer.zero_grad()  # Clear gradients for next step

    # Print model variables and loss
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

    # Visualize result
    plt.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
    plt.xlabel('day')
    plt.ylabel('head circumference')
    x, indices = torch.sort(x_train, 0)
    plt.plot(x, model.f(x).detach(), label='$y = f(x) = 20*σ(xW+b)+31$')
    plt.legend()
    plt.show()
