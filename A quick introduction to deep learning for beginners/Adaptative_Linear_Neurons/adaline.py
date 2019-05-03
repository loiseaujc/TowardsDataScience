"""
Adaptive Linear Neurons
-----------------------

This module implements the Adaptive Linear Neuron for linear classification
problems. It is part of the supplementary material associated to a
TowardsDataScience blog post on the subject [1] as well as of an
introductory course to deep learning for beginners.

[1] Link to the TDS post :

"""

# Author : Jean-Christophe Loiseau <loiseau.jc@gmail.com>
# Date : May 2019
# Licence : GPL-3.0

# --> Import standard Python libraries.
import numpy as np

# --> Setup Matplotlib
import matplotlib.pyplot as plt

# --> Import SciPy optimization functions.
from scipy.optimize import minimize

# --> Import sklearn utility functions.
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_X_y, check_array

# --> Import sklearn utility functions.
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# --> Import Rosenblatt's perceptron class
from rosenblatt import Rosenblatt


def H(z): return np.heaviside(z-0.5, 1).astype(np.int)


class Adaline(BaseEstimator, ClassifierMixin):
    """
    Implementation of Adaline based on sklearn BaseEstimator and
    ClassifierMixin.
    """

    def __init__(self, solver="CG", tol=1e-8):
        """Initialize adaline object.

        Parameters
        ----------
        solver : string
            Solver to be used. See `scipy.optimize.minimize` for more details.
            (the default is "CG").
        tol : float
            Tolerance for the minimization solver (the default is 1e-8).

        Returns
        -------
        self
            Description of returned object.

        """

        # --> scipy.optimize solver to be used to train the model.
        self.solver = solver

        # --> Tolerance for the minimization solver.
        self.tol = tol

        # --> Weights of the model.
        self.weights = None

        # --> Bias.
        self.bias = None

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

        # --> Check if adaline has already been fitted.
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

    def fit(self, X, y):
        """Fit the Adaline using the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features).
            Training data. Each row correspond to one training example and each
            column to a different feature.

        y : array-like, shape = (n_samples, ).
            Labels associated to each training example in X.

        Returns
        -------
        self : object.
               The trained adaline.

        """

        # --> Sanity check for X and y.
        X, y = check_X_y(X, y)

        # --> Add intercept column.
        X = np.insert(X, 0, 1, axis=1)

        # --> Number of features.
        n_features = X.shape[1]

        # --> Initialize the weights.
        w = np.zeros((n_features, ))  # NOTE: Bias is w[0].

        # --> Train the adaline.
        output = minimize(
            self.loss_function,     # Function to be minimized.
            w,                      # Initial guess for the weights.
            jac=True,               # Specify that the function also computed its gradient.
            args=(X, y),            # Addiotional arguments for the loss function.
            tol=self.tol,           # Tolerance for the optimization.
            method=self.solver,     # Solver for the optimizer.
        )

        # --> Return the optimal weights and bias.
        self.bias = output.x[0]
        self.weights = output.x[1:]

        return self

    def loss_function(self, w, X, y):
        """Loss function for Adaline : Mean Squared Error (MSE).

        Parameters
        ----------
        w : array, shape = (n_features, )
            Current estimate of the bias (w[0]) and weights (w[1:]).
        X : array-like, shape = (n_samples, n_features)
            Traning samples.
        y : array, shape = (n_samples, )
            Labels associated to each training example in X.

        Returns
        -------
        loss : float
               Current value of the loss function.
        grad : array, shape = (n_features, )
               Current gradient.

        """

        # --> Set the weights and bias of the adaline.
        self.bias = w[0]
        self.weights = w[1:]

        # --> Compute the adaline's prediction.
        y_pred = self.decision_function(X[:, 1:])

        # --> Compute the loss function.
        loss = 0.5*np.mean((y_pred - y)**2)

        # --> Compute the gradient.
        grad = (y_pred - y).dot(X)
        grad /= len(y)

        return loss, grad


def main(cmap="coolwarm"):

    #####################################
    #####                           #####
    #####     FIT ADALINE model     #####
    #####                           #####
    #####################################

    # --> Generate toy problem.
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

    # --> Split between training and testing.
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.5,
        random_state=0,
    )

    # --> Plot the problem.
    _, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)

    ax[0].scatter(
        x_train[:, 0], x_train[:, 1],
        c=y_train,
        cmap=cmap,
        s=40,
        edgecolors="k",
        alpha=0.5,
    )

    ax[1].scatter(
        x_test[:, 0], x_test[:, 1],
        c=y_test,
        cmap=cmap,
        s=40,
        edgecolors="k",
        alpha=0.5,
    )

    ax[1].set_xlabel(r"$x_1$")

    # --> Classify data using Adaline.
    adaline = Adaline()
    adaline.fit(X, y)

    # --> Decision boundary.
    def adaline_decision(x):
        return (0.5 - adaline.weights[0] * x - adaline.bias)/adaline.weights[1]

    x = np.linspace(*ax[0].get_xlim())

    # --> Plot the decision boundary.
    ax[0].plot(x, adaline_decision(x), color="k", label=r"Adaline")

    ax[1].plot(x, adaline_decision(x), color="k")

    ########################################
    #####                              #####
    #####     FIT PERCEPTRON MODEL     #####
    #####                              #####
    ########################################

    # --> Fit the perceptron model.
    perceptron = Rosenblatt()
    perceptron.fit(x_train, y_train)

    # --> Decision boundary for the perceptron.
    def perceptron_decision(x):
        return (-perceptron.weights[0] * x - perceptron.bias)/perceptron.weights[1]

    # --> Plot the decision boundary.
    ax[0].plot(x, perceptron_decision(x), color="gray", ls="--", label=r"Perceptron")

    ax[1].plot(x, perceptron_decision(x), color="gray", ls="--")

    # --> Add decorators to the figure.
    ax[0].set_xlim(x.min(), x.max())

    ax[0].legend(loc="lower center", bbox_to_anchor=(1, 1), ncol=2)

    ax[0].set_xlabel(r"$x_1$")
    ax[0].set_ylabel(r"$x_2$")

    ax[0].set_title(r"(a) Training dataset", y=-0.33)
    ax[1].set_title(r"(b) Testing dataset", y=-0.33)

    plt.show()


if __name__ == "__main__":

    main()
