import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d import axes3d, art3d

learning_rate = 0.1
epochs = 10000

x_train = torch.tensor([[1.0, 1.0],[1.0, 0.0],[0.0, 1.0],[0.0, 0.0]])
y_train = torch.tensor([[0.0],[1.1],[1.1],[0.0]])

class SigmoidModel:
    def __init__(self):
        # Model variables

        # Failing values:
        # self.W1 = torch.tensor([[0.0], [0.0]], requires_grad=True)
        # self.W2 = torch.tensor([[0.0], [0.0]], requires_grad=True)
        # self.b1 = torch.tensor([[0.0, 0.0]], requires_grad=True)
        # self.b2 = torch.tensor([[0.0]], requires_grad=True)

        # Successful values:
        self.W1 = torch.tensor([[random.uniform(-1, 1)], [random.uniform(-1, 1)]], requires_grad=True)
        self.W2 = torch.tensor([[random.uniform(-1, 1)], [random.uniform(-1, 1)]], requires_grad=True)
        self.b1 = torch.tensor([[random.uniform(-1, 1), random.uniform(-1, 1)]], requires_grad=True)
        self.b2 = torch.tensor([[random.uniform(-1, 1)]], requires_grad=True)

    def logits1(self, x):
        return x @ self.W1 + self.b1

    def logits2(self, x):
        return x @ self.W2 + self.b2

    # First layer function
    def f1(self, x):
        return torch.sigmoid(self.logits1(x))

    # Second layer function
    def f2(self, x):
        return torch.sigmoid(self.logits2(x))

    # Predictor
    def f(self, x):
        return self.f2(self.f1(x))

    # Cross Entropy Loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits2(self.f1(x)), y)

if __name__ == '__main__':
    model = SigmoidModel()

    optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2], learning_rate)
    for epoch in range(epochs):
        model.loss(x_train, y_train).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b

        optimizer.zero_grad()  # Clear gradients for next step

    # Print model variables and loss
    print("W1 = %s, W2 = %s, b1 = %s, b2 = %s, loss = %s" % (model.W1, model.W2, model.b1, model.b2, model.loss(x_train, y_train)))

    #Visualize the result
    fig = plt.figure()
    plot = fig.add_subplot(111, projection='3d')

    plot.scatter(x_train[:,0], x_train[:,1], y_train)

    x1_grid, x2_grid = torch.tensor(np.meshgrid(np.linspace(-0.25, 1.25, 10),
                                                np.linspace(-0.25, 1.25, 10)), dtype=torch.float)
    y_grid = torch.tensor(np.empty([10, 10]), dtype=torch.float)
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            y_grid[i, j] = model.f(torch.tensor([[x1_grid[i, j], x2_grid[i, j]]])).detach()
    plot_f = plot.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")
    plt.show()
