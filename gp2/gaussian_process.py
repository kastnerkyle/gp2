import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scipy import linalg
from sklearn.utils import check_array
import matplotlib.pyplot as plt


def plot_gp_confidence(gp, show_gp_points=True, X_low=-1, X_high=1, X_step=.01, xlim=None, ylim=None):
    xpts = np.arange(X_low, X_high, step=X_step).reshape((-1, 1))
    try:
        y_pred = gp.predict(xpts)
        var = gp.predicted_var_
        if gp.predicted_mean_.shape[1] > 1:
            raise ValueError("plot_gp_confidence only works for 1 dimensional Gaussian processes!")
    except TypeError:
        y_pred = xpts * 0
        var = gp.predicted_var_ * np.ones((xpts.shape[0], xpts.shape[0]))

    plt.errorbar(xpts.squeeze(), y_pred.squeeze(), yerr=np.diag(var), capsize=0, color='steelblue')
    if show_gp_points:
        plt.plot(gp._X, gp._y, color='darkred', marker='o', linestyle='')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()


class SimpleGaussianProcessRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, kernel_function, copy=True):
        self.kernel_function = kernel_function
        self.copy = copy
        self.predicted_mean_ = 0
        self.predicted_var_ = self._covariance(np.zeros((1, 1)), np.zeros((1, 1)))
        self._X = None
        self._y = None

    def _covariance(self, x1, x2):
        return self.kernel_function(x1, x2)

    def fit(self, X, y):
        self._X = None
        self._y = None
        return self.partial_fit(X, y)

    def partial_fit(self, X, y):
        X = check_array(X, copy=self.copy)
        y = check_array(y, copy=self.copy)
        if self._X is None:
            self._X = X
            self._y = y
        else:
            self._X = np.vstack((self._X, X))
            self._y = np.vstack((self._y, y))

    def predict(self, X, y=None):
        X = check_array(X, copy=self.copy)
        cov_xxn = self._covariance(X, self._X)
        cov_x = self._covariance(self._X, self._X)
        cov_xn = self._covariance(X, X)
        cov_x_inv = linalg.pinv(cov_x)
        mean = cov_xxn.dot(cov_x_inv).dot(self._y)
        var = cov_xn - cov_xxn.dot(cov_x_inv).dot(cov_xxn.T)
        self.predicted_mean_ = mean
        self.predicted_var_ = var
        return mean

if __name__ == "__main__":
    from pairwise_distances import exponential_kernel
    gp = SimpleGaussianProcessRegressor(exponential_kernel)
    plt.title('Initial GP Confidence')
    plot_gp_confidence(gp, X_low=-3, X_high=3, X_step=.01, xlim=(-3, 3), ylim=(-3, 3))

    rng = np.random.RandomState(1999)
    n_samples = 200
    X = rng.rand(n_samples, 1)
    y = np.sin(20 * X) + .95 * rng.randn(X.shape[0], 1)
    plt.title('Noisy Data')
    plt.scatter(X, y, color='steelblue')
    plt.show()

    gp.fit(X, y)
    X_new = rng.rand(5, 1)
    gp.predict(X_new)
    plt.title('Final GP Confidence')
    plot_gp_confidence(gp, show_gp_points=False, X_low=0, X_high=1, X_step=.01)
