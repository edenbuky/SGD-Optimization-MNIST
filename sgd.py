#################################
# Your name: Eden Buky
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    w = np.zeros(data.shape[1])  # Initialize weight vector to zeros
    for t in range(1, T + 1):
        eta_t = eta_0 / t  # Learning rate at iteration t
        i = np.random.randint(0, data.shape[0])  # Randomly sample index i
        x_i = data[i]  # Sample data
        y_i = labels[i]  # Corresponding label

        # Check condition for updating weights
        if y_i * np.dot(w, x_i) < 1:
            w = (1 - eta_t) * w + eta_t * C * y_i * x_i
        else:
            w = (1 - eta_t) * w

    return w

from scipy.special import expit
def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    n, d = data.shape  # n is the number of samples, d is the number of features
    w = np.zeros(d)  # Initialize weight vector to zeros

    # Perform T iterations of SGD
    for t in range(1, T + 1):
        # Randomly select an example
        i_t = np.random.randint(0, n)
        x_i = data[i_t]
        y_i = labels[i_t]

        # Compute the gradient for the logistic loss
        # expit(x) = 1 / (1 + exp(-x)) is the logistic (sigmoid) function
        gradient = -y_i * x_i * (1 - expit(y_i * np.dot(w, x_i)))

        # Update the learning rate
        eta_t = eta_0 / t

        # Update the weights
        w -= eta_t * gradient

    return w

#################################

# Place for additional code
def cross_validate_eta(train_data, train_labels, validation_data, validation_labels,T=1000, eta_range=np.logspace(-5, 5, num=11),C = None):
    accuracies = []  # Store the average accuracies for each eta
    eta_values = []  # Store the eta values for plotting

    for eta_0 in eta_range:
        run_accuracies = []  # Store accuracies for the current eta across runs
        for _ in range(10):  # Perform 10 runs for each eta
            if not C:
              weights = SGD_log(train_data, train_labels, eta_0, T)  # Train the model
            else:
              weights = SGD_hinge(train_data, train_labels, C, eta_0, T)  # Train the model
            predictions = np.sign(np.dot(validation_data, weights))  # Predict on validation set
            accuracy = np.mean(predictions == validation_labels)  # Calculate accuracy
            run_accuracies.append(accuracy)

        # Calculate the average accuracy for the current eta and store it
        avg_accuracy = np.mean(run_accuracies)
        accuracies.append(avg_accuracy)
        eta_values.append(eta_0)

        print(f"eta_0: {eta_0}, Average Accuracy: {avg_accuracy}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.semilogx(eta_values, accuracies, marker='o')
    plt.xlabel('Learning Rate (eta_0)')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy vs Learning Rate (eta_0)')
    plt.grid(True)
    plt.show()

    # Return the best eta_0 based on the highest average accuracy
    best_eta = eta_values[np.argmax(accuracies)]
    return best_eta
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
print(cross_validate_eta(train_data, train_labels, validation_data, validation_labels, 1000, np.logspace(-5, 5, num=11)), 1)
eta_0_range = np.arange(0, 2.1, 0.1)
print(cross_validate_eta(train_data, train_labels, validation_data, validation_labels, 1000, eta_0_range), 1)

def cross_validate_C(train_data, train_labels, validation_data, validation_labels, eta_0=0.7, C_range=np.logspace(-4, 4, num=9), T=1000):
    accuracies = []  # Store the average accuracies for each C
    C_values = []  # Store the C values for plotting

    for C in C_range:
        run_accuracies = []  # Store accuracies for the current C across runs
        for _ in range(10):  # Perform 10 runs for each C
            weights = SGD_hinge(train_data, train_labels, C, eta_0, T)  # Train the model
            predictions = np.sign(np.dot(validation_data, weights))  # Predict on validation set
            accuracy = np.mean(predictions == validation_labels)  # Calculate accuracy
            run_accuracies.append(accuracy)

        # Calculate the average accuracy for the current C and store it
        avg_accuracy = np.mean(run_accuracies)
        accuracies.append(avg_accuracy)
        C_values.append(C)

        print(f"C: {C}, Average Accuracy: {avg_accuracy}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.semilogx(C_values, accuracies, marker='o')
    plt.xlabel('Regularization Strength (C)')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy vs Regularization Strength (C)')
    plt.grid(True)
    plt.show()

    # Return the best C based on the highest average accuracy
    best_C = C_values[np.argmax(accuracies)]
    return best_C
print(cross_validate_C(train_data, train_labels, validation_data, validation_labels))

def train_and_visualize_w(train_data, train_labels, eta_0=0.7, C=0.0001, T=20000):
    # Train the classifier
    w = SGD_hinge(train_data, train_labels, C, eta_0, T)

    # Reshape the weight vector into a 28x28 image
    w_image = np.reshape(w, (28, 28))

    # Visualize the image
    plt.imshow(w_image, interpolation='nearest')
    plt.colorbar()
    plt.title('Visualization of the weight vector')
    plt.show()

    return w

# Assuming train_data and train_labels are already defined
print(train_and_visualize_w(train_data, train_labels))

def evaluate_classifier_on_test(train_data, train_labels, test_data, test_labels, eta_0=0.7, C=0.0001, T=20000):
    # Train the classifier with the best hyperparameters on the training data
    w_best = SGD_hinge(train_data, train_labels, C, eta_0, T)

    # Predict labels for the test data
    test_predictions = np.sign(np.dot(test_data, w_best))

    # Calculate the accuracy on the test set
    test_accuracy = np.mean(test_predictions == test_labels)

    return test_accuracy

# Assuming test_data and test_labels are already defined and preprocessed
test_accuracy = evaluate_classifier_on_test(train_data, train_labels, test_data, test_labels)
print(f"Accuracy of the best classifier on the test set: {test_accuracy}")


train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
print(cross_validate_eta(train_data, train_labels, validation_data, validation_labels, 1000, np.logspace(-5, 5, num=11)))
eta_0_range_fine = np.linspace(1e-6, 1e-5, num=10)
print(cross_validate_eta(train_data, train_labels, validation_data, validation_labels, 1000, eta_0_range_fine))


def train_SGD_log_and_visualize_w(train_data, train_labels, eta_0, T):
    """
    Train SGD with logistic loss and visualize the resulting weight vector as an image.

    Parameters:
    - train_data: The data to train on.
    - train_labels: The correct labels for the training data.
    - eta_0: The optimal learning rate found.
    - T: The number of iterations to train for.

    Returns:
    - w: The weight vector after training.
    """
    w = SGD_log(train_data, train_labels, eta_0, T)  # Train the classifier

    # Reshape the weight vector into a 28x28 image
    w_image = np.reshape(w, (28, 28))

    # Visualize the image
    plt.imshow(w_image, interpolation='nearest', cmap='viridis')
    plt.colorbar()
    plt.title('Visualization of the weight vector w')
    plt.show()

    return w

# Assuming train_data and train_labels are defined and SGD_log is implemented correctly
print(train_SGD_log_and_visualize_w(train_data, train_labels, 3e-6, 20000))


def train_SGD_log_and_visualize_w(train_data, train_labels, eta_0, T):
    """
    Train SGD with logistic loss and visualize the resulting weight vector as an image.

    Parameters:
    - train_data: The data to train on.
    - train_labels: The correct labels for the training data.
    - eta_0: The optimal learning rate found.
    - T: The number of iterations to train for.

    Returns:
    - w: The weight vector after training.
    """
    w = SGD_log(train_data, train_labels, eta_0, T)  # Train the classifier

    # Reshape the weight vector into a 28x28 image
    w_image = np.reshape(w, (28, 28))

    # Visualize the image
    plt.imshow(w_image, interpolation='nearest', cmap='viridis')
    plt.colorbar()
    plt.title('Visualization of the weight vector w')
    plt.show()

    return w


# Assuming train_data and train_labels are defined and SGD_log is implemented correctly
w = train_SGD_log_and_visualize_w(train_data, train_labels, 3e-6, 20000)

def compute_accuracy(w, test_data, test_labels):
    # Compute predictions: if the dot product of w and x is positive, predict 1; otherwise, predict -1.
    predictions = np.sign(np.dot(test_data, w))
    # Compute accuracy as the fraction of predictions that match labels.
    accuracy = np.mean(predictions == test_labels)
    return accuracy

# Assuming test_data and test_labels are available
accuracy = compute_accuracy(w, test_data, test_labels)
print(f"The accuracy of the best classifier on the test set is: {accuracy}")

def train_and_plot_norm(train_data, train_labels, eta_0, T):
    n, d = train_data.shape
    w = np.zeros(d)
    norms = []

    for t in range(1, T + 1):
        i = np.random.randint(0, n)
        x_i = train_data[i]
        y_i = train_labels[i]

        # Compute the gradient for the logistic loss
        gradient = -y_i * x_i * (1 - expit(y_i * np.dot(w, x_i)))

        # Update the weights
        w -= (eta_0 / t) * gradient

        # Calculate and record the norm of w
        norms.append(np.linalg.norm(w))

    # Plotting
    plt.plot(range(1, T + 1), norms)
    plt.xlabel('Iteration')
    plt.ylabel('||w||')
    plt.title('Norm of w over iterations')
    plt.show()

    return w
best_eta_0 = 3e-6  # Replace with the best eta_0 you found
iterations = 20000

# Assuming train_data and train_labels are already defined
w_final = train_and_plot_norm(train_data, train_labels, best_eta_0, iterations)
print(w_final)

#################################