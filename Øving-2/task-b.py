import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, art3d

learning_rate = 0.1
epochs = 10000

x_train = torch.tensor([[1.0, 1.0],[1.0, 0.0],[0.0, 1.0],[0.0, 0.0]])
y_train = torch.tensor([[0.0],[1.1],[1.1],[1.1]])

class SigmoidModel:
    def __init__(self):
        #Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    # Cross Entropy Loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)

if __name__ == '__main__':
    model = SigmoidModel()

    optimizer = torch.optim.SGD([model.W, model.b], learning_rate)
    for epoch in range(epochs):
        model.loss(x_train, y_train).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b

        optimizer.zero_grad()  # Clear gradients for next step

    # Print model variables and loss
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

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
