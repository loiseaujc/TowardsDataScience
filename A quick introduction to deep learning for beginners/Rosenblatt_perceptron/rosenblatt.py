"""
Rosenblatt's Perceptron
-----------------------

This module implements Rosenblatt's Perceptron for linear classification
problems. It is part of the supplementary material associated to a
TowardsDataScience blog post on the subject [1] as well as of an
introductory course to deep learning for beginners.

[1] Link to the TDS post : https://tinyurl.com/y2ccfyrc

"""

# Author : Jean-Christophe Loiseau <loiseau.jc@gmail.com>
# Data : March 2019
# Licence : ...

# --> Miscellaneous
import warnings

# --> Import standard Python libraries.
import numpy as np

# --> Setup matplotlib
import matplotlib.pyplot as plt

# --> Import sklearn utility functions.
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_X_y, check_array


def H(x): return np.heaviside(x, 1).astype(np.int)


class Rosenblatt(BaseEstimator, ClassifierMixin):
    """
    Implementation of Rosenblatt's Perceptron based on sklearn BaseEstimator and
    ClassifierMixin.
    """

    def __init__(self):
        # --> Weights of the model.
        self.weights = None

        # --> Bias.
        self.bias = None

        # --> Number of errors made at each stage of the training procedure.
        self.errors_ = list()

    def decision_function(self, X):
        """Predict the signed distance from the decision hyperplane for each
        sample.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples for which we aim to predict the class.

        Returns
        -------
        array, shape = (n_samples, )
            Signed distance from the decision hyperplane for each sample. If
            this distance is positive, the sample belongs to class 1, otherwise
            it belongs to class 0.

        """

        # --> Check if model has already been fitted.
        if not hasattr(self, "weights") or self.weights is None:
            raise NotFittedError(
                "This %(name)s instance is not fitted yet."
                % {"name": type(self).__name__}
            )

        # --> Sanity check for X.
        X = check_array(X)

        # --> Check that X has the correct number of features.
        n_features = self.weights.shape[0]
        if X.shape[1] != n_features:
            raise ValueError(
                "X has %d features per samples; expecting %d."
                % (X.shape[1], n_features)
            )

        # --> If everything is OK, compute the score.
        scores = X.dot(self.weights) + self.bias

        return scores

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features).
            Samples for which we aim to predict the class.

        Returns
        -------
        array-like, shape = (n_samples, )
            Predicted class label per sample.

        """
        return H(self.decision_function(X))

    def fit(self, X, y, maxiter=100):
        """Fit the Rosenblatt perceptron using the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features).
            Training data. Each row correspond to one training example and each
            column to a different feature.

        y : array-like, shape = (n_samples, ).
            Labels associated to each training example in X.

        maxiter : integer
            Maximum number of iterations before it stops.

        Returns
        -------
        self : object.
               The trained model.

        """

        # --> Sanity check for X and y.
        X, y = check_X_y(X, y)

        # --> Number of features.
        n_features = X.shape[1]

        # --> Initialize the weights and bias.
        self.weights = np.zeros((n_features, ))
        self.bias = 0.0

        # --> Perceptron algorithm loop.
        for _ in range(maxiter):

            # --> Current number of errors.
            errors = 0

            # --> Loop through the examples.
            for xi, y_true in zip(X, y):

                # --> Compute error.
                error = y_true - self.predict(xi.reshape((1, -1)))

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

        # --> Raise warning if perceptron has not converged.
        if errors != 0:
            warnings.warn(
                "Perceptron learning did not converge using the maximum number"
                "of iterations given."
            )

        return self


def main(cmap="coolwarm"):

    # --> Generate toy problem.
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0,                 # No noise.
        class_sep=1.5,
        random_state=999,         # Fix random seed for reproducibility.
    )

    # --> Plot the problem.
    _, ax = plt.subplots(1, 1, figsize=(3, 3))

    ax.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap=plt.get_cmap(cmap),
        s=40,
        edgecolors="k",
        alpha=0.5,
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    # --> Classify data using Rosenblatt's perceptron.
    model = Rosenblatt()
    model.fit(X, y)

    # --> Decision boundary.
    def decision(x): return -(model.weights[0] * x + model.bias)/model.weights[1]

    x = np.linspace(*ax.get_xlim())

    # --> Plot the decision boundary.
    ax.plot(x, decision(x), color="k")

    ax.set_xlim(x.min(), x.max())

    plt.show()


if __name__ == "__main__":

    main()
