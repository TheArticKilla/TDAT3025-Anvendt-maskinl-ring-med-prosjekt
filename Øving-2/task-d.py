import torch
import torchvision
import matplotlib.pyplot as plt

learning_rate = 0.1
epochs = 0
accuracy_requirement = 0.9

class SoftmaxModel:
    def __init__(self):
        #Model variables
        self.W = torch.rand((784, 10), requires_grad=True)
        self.b = torch.rand((1, 10), requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy Loss
    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # The accuracy of the model
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

if __name__ == '__main__':
    model = SoftmaxModel()

    # Load observations from the mnist dataset. The observations are divided into a training set and a test set
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
    x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
    y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
    y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
    x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
    y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
    y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

    optimizer = torch.optim.SGD([model.W, model.b], learning_rate)

    while model.accuracy(x_test, y_test).item() < accuracy_requirement:
        model.loss(x_train, y_train).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b
        optimizer.zero_grad()  # Clear gradients for next step
        epochs += 1

    print("W = %s, b = %s, loss = %s, accuracy = %s, epochs = %s" % (model.W, model.b, model.loss(x_test, y_test), model.accuracy(x_test, y_test), epochs))

    for i in range(10):
        plt.imsave('image-%s.png' % i, model.W[:, i].reshape(28, 28).detach())
