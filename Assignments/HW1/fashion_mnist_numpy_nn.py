import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import numpy_nn as nn

np.random.seed(123)
class_names = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
               5:  'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
# Include all the classes you want to see in training
kept_classes = [0, 1, 7]


def get_data(filter_classes):
    fashion_mnist = fetch_openml("Fashion-MNIST", parser='auto')
    x, y = fashion_mnist['data'], fashion_mnist['target'].astype(int)
    # Remove classes
    filtered_indices = np.isin(y, filter_classes)
    x, y = x[filtered_indices].to_numpy(), y[filtered_indices]
    # Normalize the pixels to be in [-1, +1] range
    x = ((x / 255.) - .5) * 2
    removed_class_count = 0
    for i in range(10):  # Fix the labels
        if i in filter_classes and removed_class_count != 0:
            y[y == i] = i - removed_class_count
        elif i not in filter_classes:
            removed_class_count += 1
    # Do the train-test split
    return train_test_split(x, y, test_size=10_000)


def onehot_encoder(y, num_labels):
    one_hot = np.zeros(shape=(y.size, num_labels), dtype=int)
    one_hot[np.arange(y.size), y] = 1
    return one_hot


if __name__ == '__main__':
    # Download the dataset and split it into training and testing sets
    x_train, x_test, y_train, y_test = get_data(kept_classes)
    # One-hot encode the target labels of the training set
    y_train = onehot_encoder(y_train, len(kept_classes))

    # Define the layers of the neural network
    layers = [nn.Linear(784, 50),
              nn.ReLU(),
              nn.Linear(50, 50),
              nn.ReLU(),
              nn.Linear(50, len(kept_classes)),
              # nn.Sigmoid(),
              nn.Softmax()]
    # Create the model
    model = nn.MLP(layers, nn.CrossEntropy(), nn.GradientDescent(0.001))

    # Train the model
    losses = model.train(x_train, y_train, 30, 64)

    # Plot the loss
    plt.plot(losses)
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Test the model
    y_prediction = np.argmax(model.forward(x_test), axis=1)
    acc = 100 * np.mean(y_prediction == y_test)
    print(f'Test accuracy with {len(y_train)} training examples on {len(y_test)} test samples is {acc:.2f}%')
