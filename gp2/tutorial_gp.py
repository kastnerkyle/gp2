import numpy as np
import matplotlib.pyplot as plt


# from mrmartin.ner/?p=223
def exponential_kernel(x1, x2):
    # Broadcasting tricks to get every pairwise distance.
    return np.exp(-(x1[np.newaxis, :, :] - x2[:, np.newaxis, :])[:, :, 0] ** 2).T


# correlation matrix for MV normal
def correlation(kernel, x1, x2):
    return kernel(x1, x2)


def conditional(x_new, x, y, kernel):
    corr_xxn = correlation(kernel, x_new, x)
    corr_x = correlation(kernel, x, x)
    corr_xn = correlation(kernel, x_new, x_new)
    mu = corr_xxn.dot(np.linalg.pinv(corr_x)).dot(y)
    sigma = corr_xn - corr_xxn.dot(np.linalg.pinv(corr_x)).dot(corr_xxn.T)
    return mu, sigma


rng = np.random.RandomState(1999)

# Initial guess
kernel = exponential_kernel
init = np.zeros((1, 1))
sigma = kernel(init, init)
xpts = np.arange(-3, 3, step=0.01).reshape((-1, 1))
plt.errorbar(xpts.squeeze(), np.zeros(len(xpts)), yerr=sigma.squeeze(), capsize=0, color='steelblue')
plt.ylim(-3, 3)
plt.figure()

# First point estimate
x_new = np.atleast_2d(1.)
# No conditional, this is the first value!
y_new = np.atleast_2d(0 + sigma * rng.randn())
x = x_new
y = y_new

# Plotting
y_pred, sigma_pred = conditional(xpts, x, y, kernel=kernel)
plt.errorbar(xpts.squeeze(), y_pred.squeeze(), yerr=np.diag(sigma_pred), capsize=0, color='steelblue')
plt.plot(x, y, color='darkred', marker='o', linestyle='')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.figure()

# Second point estimate
x_new = np.atleast_2d(-0.7)
mu, s = conditional(x_new, x, y, kernel=kernel)
y_new = np.atleast_2d(mu + np.diag(s)[:, np.newaxis] * rng.randn(*x_new.shape))
x = np.vstack((x, x_new))
y = np.vstack((y, y_new))

# Plotting
y_pred, sigma_pred = conditional(xpts, x, y, kernel=kernel)
plt.errorbar(xpts.squeeze(), y_pred.squeeze(), yerr=np.diag(sigma_pred), capsize=0, color='steelblue')
plt.plot(x, y, color='darkred', marker='o', linestyle='')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.figure()

# Multipoint estimate
x_new = 3 * rng.rand(3, 1) - 1.5
mu, s = conditional(x_new, x, y, kernel=kernel)
y_new = mu + np.diag(s)[:, np.newaxis] * rng.randn(*x_new.shape)
x = np.vstack((x, x_new))
y = np.vstack((y, y_new))

# Plotting
y_pred, sigma_pred = conditional(xpts, x, y, kernel=kernel)
plt.errorbar(xpts.squeeze(), y_pred.squeeze(), yerr=np.diag(sigma_pred), capsize=0, color='steelblue')
plt.plot(x, y, color='darkred', marker='o', linestyle='')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()
