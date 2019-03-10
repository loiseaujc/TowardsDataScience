# --> Import standard Python libraries.
import numpy as np

# --> Import sklearn utility functions to create derived-class objects.
from sklearn.base import BaseEstimator, ClassifierMixin

# --> Redefine the Heavisde function.
H = lambda x: np.heaviside(x, 1).astype(np.int)

class Rosenblatt(BaseEstimator, ClassifierMixin):
    """
    Implementation of Rosenblatt's Perceptron using sklearn BaseEstimator and
    ClassifierMixin.
    """

    def __init__(self):
        return

    def predict(self, X):
        return H( X.dot(self.weights) + self.bias )

    def fit(self, X, y, epochs=100):

        # --> Number of features.
        n = X.shape[1]

        # --> Initialize the weights and bias.
        self.weights = np.zeros((n, ))
        self.bias = 0.0

        # --> List to store the number of errors.
        self.errors_ = list()

        # --> Perceptron algorithm loop.
        for _ in range(epochs):

            # --> Current number of errors.
            errors = 0

            # --> Loop through the examples.
            for xi, y_true in zip(X, y):

                # --> Compute error.
                error = y_true - self.predict(xi)

                if error != 0:
                    # --> Update the weights and bias.
                    self.weights += error * xi
                    self.bias += error

                    # --> Current number of errors.
                    errors += 1

            # --> Total number of the i-th epoch.
            self.errors_.append(errors)

            # --> If no error is made, exit the outer  loop.
            if errors == 0:
                break

        return self

if __name__ == "__main__":

    # --> Generate toy problem.
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples = 100,
        n_features = 2,
        n_informative = 2,
        n_redundant = 0,
        n_clusters_per_class = 1,
        flip_y = 0,                 # No noise.
        class_sep = 1.5,
        random_state = 999,         # Fix random seed for reproducibility.
    )

    # --> Plot the problem.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    ax.scatter(
        X[:, 0], X[:, 1],
        c = y,
        cmap = plt.cm.coolwarm,
        s = 40,
        edgecolors="k",
        alpha = 0.5,
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    # --> Classify data using Rosenblatt's perceptron.
    model = Rosenblatt()
    model.fit(X, y)

    # --> Decision boundary.
    d = lambda x: -(model.weights[0]* x + model.bias)/model.weights[1]

    x0, x1 = ax.get_xlim()
    x = np.linspace(x0, x1)

    # --> Plot the decision boundary.
    ax.plot(
        x, d(x),
        color = "k"
    )

    ax.set_xlim(x0, x1)

    plt.show()
