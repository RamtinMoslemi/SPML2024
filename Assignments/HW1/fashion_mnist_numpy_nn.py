import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import numpy_nn as nn

np.random.seed(123)


def get_data():
    fashion_mnist = fetch_openml("Fashion-MNIST", parser='auto')
    x, y = fashion_mnist['data'], fashion_mnist['target']
    # Normalization:
    x = ((x / 255.) - .5) * 2
    # Remove classes 2-6 and 8-9
    y = y.astype(int)
    filter_classes = [0, 1, 7]
    filtered_indices = np.isin(y, filter_classes)
    x = x[filtered_indices].to_numpy()
    y = y[filtered_indices]
    y[y == 7] = 2  # change label 7 to 2
    # Do the train-test split
    return train_test_split(x, y, test_size=10000)


def onehot_enc(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary


if __name__ == '__main__':
    # Download the dataset and split it into training and testing sets
    x_train, x_test, y_train, y_test = get_data()

    # One-hot encode the target labels
    y_train = onehot_enc(y_train, 3)

    # Define the layers of the neural network
    layers = [nn.Linear(784, 50),
              nn.ReLU(),
              nn.Linear(50, 50),
              nn.ReLU(),
              nn.layers.Linear(50, 3),
              # numpy_nn.activation_functions.Sigmoid(),
              nn.Softmax()]

    # Create the model
    model = nn.MLP(layers, nn.CrossEntropy(), nn.GradientDescent(0.001))

    # Train the model
    losses = model.train(x_train, y_train, 30, 64)

    # Plot the loss
    plt.plot(losses)
    plt.title("Training loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    # Test the model
    y_prediction = np.argmax(model.forward(x_test), axis=1)
    acc = np.mean(y_prediction == y_test)
    print(f'Test accuracy is {100 * acc:.2f}%')
