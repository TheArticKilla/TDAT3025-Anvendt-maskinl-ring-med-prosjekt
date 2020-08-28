import torch
import matplotlib.pyplot as plt
import pandas as pd

learning_rate = 0.0001
epochs = 200000

file = pd.read_csv("datasets/day_length_weight.csv")
x_train = torch.tensor(file.day).reshape(-1, 1)
y_train = torch.tensor(file.length).reshape(-1, 1)
z_train = torch.tensor(file.weight).reshape(-1, 1)

class LinearRegressionModel:


if __name__ == '__main__':
    model = LinearRegressionModel()
