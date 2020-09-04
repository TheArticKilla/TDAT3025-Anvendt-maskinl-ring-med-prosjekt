import torch
import matplotlib.pyplot as plt
import numpy as np

learning_rate = 0.1
epochs = 10000

x_train = torch.tensor([[1.],[0.]])
y_train = torch.tensor([[0.],[1.]])

class SigmoidModel:
    def __init__(self):
        #Model variables
        self.W = torch.tensor([[0.]], requires_grad=True)
        self.b = torch.tensor([[0.]], requires_grad=True)

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

    # Visualize result
    fig = plt.figure("NOT operator")

    plot1 = fig.add_subplot()
    plt.plot(x_train.detach(), y_train.detach(), 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
    plt.xlabel('x')
    plt.ylabel('y')
    out = torch.reshape(torch.tensor(np.linspace(0, 1, 100).tolist()), (-1, 1))
    plot1.set_xticks([0, 1])  # x range from 0 to 1
    plot1.set_yticks([0, 1])  # y range from 0 to 1
    x, indices = torch.sort(out, 0)
    # Plot sigmoid regression curve.
    plt.plot(x, model.f(x).detach())

    plt.show()
