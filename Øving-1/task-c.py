import torch
import matplotlib.pyplot as plt
import pandas as pd

learning_rate = 0.0001
epochs = 200000

file = pd.read_csv("datasets/day_head_circumference.csv")
print(file.head[1])
x_train = torch.tensor(file.day).reshape(-1, 1)
y_train = torch.tensor(file.head_circumference).reshape(-1, 1)

class NonLinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
        self.sigma = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return 20*self.sigma.double()*(x @ self.W.double() + self.b.double()) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x.double()) - y.double()))


if __name__ == '__main__':
    model = NonLinearRegressionModel()

    model = LinearRegressionModel()

    # Optimize: adjust W and b to minimize loss using stochastic gradient descent
    optimizer = torch.optim.SGD([model.W, model.b], learning_rate)
    for epoch in range(epochs):
        model.loss(x_train, y_train).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b

        optimizer.zero_grad()  # Clear gradients for next step

    # Print model variables and loss
    print("W = %s, b = %s, sigma = %s, loss = %s" % (model.W, model.b, model.sigma, model.loss(x_train, y_train)))

    # Visualize result
    plt.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
    plt.xlabel('x')
    plt.ylabel('y')
    x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])  # x = [[1], [6]]]
    plt.plot(x, model.f(x).detach(), label='$y = f(x) = xW+b$')
    plt.legend()
    plt.show()
