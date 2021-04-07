# TDAT3025 - Fall 2020
This repo contains all the exercises for the NTNU course [TDAT3025 - Applied Machine Learning with Project](https://www.ntnu.no/studier/emner/TDAT3025#tab=omEmnet), during the fall semester 2020.
## [Exercise 1 - Linear Recursion](/Øving-1)
This exercise uses PyTorch to train the machine on different datasets, to create a regression model corresponding to the values.
### [Task A](/Øving-1/task-a.py)
Linear 2d recursion of a dataset containing length to width values.
### [Task B](/Øving-1/task-b.py)
Linear 3d recursion of dataset using length and weight values corresponding to different days.
### [Task C](/Øving-1/task-c.py)
Non linear 2d recursion of a dataset of the head circumference of newborn babies by day, using the following predicator: f(x) = 20σ(xW + b) + 31, where σ is the sigmoid function.
## [Exercise 2 - Artificial Neural Networks](/Øving-2)
Exercise in neural networks, where task a-c visualizes different operators, and task d classifies hand written numbers, trained on the MNIST dataset.
### [Task A](/Øving-2/task-a.py)
Visualization of the NOT operator as a 2d graph.
### [Task B](/Øving-2/task-b.py)
Visualization of the NAND operator in 3d space.
### [Task C](/Øving-2/task-c.py)
Visualization of the XOR operator in 3d space, with both non converging starting values, as well as functioning values.
### [Task D](/Øving-2/task-d.py)
Training and testing on the MNIST dataset, with an accuracy guarantee of at least 90%, generating images visualizing the optimized model.
## [Exercise 3 - Convolutional Neural Networks](/Øving-3)
An exercise utilizing convolutional neural networks, with tasks a-c gradually expands a convoluted neural network on the MNIST dataset, and task d uses the same techniques on the Fashion MNIST set. All tasks are made to be able to utilize CUDA cores.
### [Task A](/Øving-3/task-a.py)
Expands an 1 layer deep convoluted neural network, into a 2 layer deep network. Achieving a 98.7% accuracy.
### [Task B](/Øving-3/task-b.py)
Adds an extra dense layer to the previous model. Now achieving 98.0% accuracy.
### [Task C](/Øving-3/task-c.py)
Further expands the model form a and b, by adding a Dropout layer. Now achieving an accuracy of 98.0%.
### [Task D](/Øving-3/task-d.py)
This time, we use a convoluted neural network on the Fashion MNIST dataset, instead of the standard MNIST set. This model achieves an accuracy of 88.0%
## [Exercise 4 - Recurrent Neural Networks](/Øving-4)
Exercise in recurrent neural networks, training a recurrent network model on learning to generate and recognize words.
### [Task A](/Øving-4/task-a.py)
Task A trains a model to generate the sentence: "hello world".
### [Task B](/Øving-4/task-b.py)
Task B trains the same model to recognize different words representing emojis.
## [Exercise 5 - Jupyter Notebook, Pandas and feature extraction](/Øving-5)
Exercise in which we use Jupyter Notebook and Pandas to analyze and extract features from [a dataset containing
data about mushrooms](https://archive.ics.uci.edu/ml/datasets/Mushroom)
## [Exercise 6 - Dimension Reduction in Jupyter](Øving-6)
This exercise looks at the reduction of dimensions in [a dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom), to select the most prevalent features.
## [Exercise 7 - Unsupervised Learning with Clustering in Jupyter](Øving-7)
Using unsupervised learning to cluster [a dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom) using K-means and Silhouette metric scoring.
## [Execise 8 - Reinforcement Learning](Øving-8)
Reinforcement learning algorithms on the [Gym CartPole environment][https://gym.openai.com/envs/CartPole-v1/].
### [Task 1](Øving-8/task-1.py)
First task in creating a simple Q-learing algorithm to solve the [CartPole environment](https://gym.openai.com/envs/CartPole-v1/)
### [Task 2B](Øving-8/task-2b.py)
Expands the solution from task 1 to use a DQN to solve the CartPole instead.
