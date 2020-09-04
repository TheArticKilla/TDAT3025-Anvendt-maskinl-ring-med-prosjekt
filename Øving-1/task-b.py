import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

learning_rate = 0.0001
epochs = 200000

file = pd.read_csv("datasets/day_length_weight.csv")
x_train = torch.tensor([file.length, file.weight]).reshape(-1, 2)
y_train = torch.tensor(file.day).reshape(-1, 1)

class LinearRegressionModel:
    def __init__(self):
        # Variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    #Predicator
    def f(self, x):
        return x @ self.W.double() + self.b.double()

    # Uses Mean Squered Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x.double()), y.double())

if __name__ == '__main__':
    model = LinearRegressionModel()

    # Optimize: adjust W and b to minimize loss using stochastic gradient descent
    optimizer = torch.optim.SGD([model.W, model.b], learning_rate)
    for epoch in range(epochs):
        model.loss(x_train, y_train).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,

        optimizer.zero_grad()  # Clear gradients for next step

    # Print model variables and loss
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

    #Visualize the result
    fig = plt.figure()
    plot = fig.add_subplot(111, projection='3d')

    plot.scatter(file.length, file.weight, file.day)

    #X, Y = np.meshgrid(file.length, file.weight)
    #Z = model.f(x_train).detach()
    #plot.plot_wireframe(X, Y, Z, label='$y = f(x) = xW+b$', color="green")
    x1_grid, x2_grid = torch.tensor(np.meshgrid(np.linspace(-1000.0, 1000.0, 10),
                                                np.linspace(-1000.0, 1000.0, 10)), dtype=torch.double)
    y_grid = torch.tensor(np.empty([10, 10]), dtype=torch.double)
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            y_grid[i, j] = model.f(torch.tensor([[x1_grid[i, j], x2_grid[i, j]]])).detach()
    plot_f = plot.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")
    #plot.scatter(file.length, file.weight, model.f(x_train).detach(), label='$y = f(x) = xW+b$', c="red")

    plot.set_xlabel('Length')
    plot.set_ylabel('Weight')
    plot.set_zlabel('Age')

    plt.show()
