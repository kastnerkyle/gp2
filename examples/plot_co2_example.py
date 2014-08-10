import matplotlib.pyplot as plt
from gp2.gaussian_process import SimpleGaussianProcessRegressor
from gp2.pairwise_distances import init_spectral_mixture_kernel, spectral_mixture_kernel
from gp2.datasets import load_co2

data = load_co2()
X_train = data['xtrain']
X_test = data['xtest']
y_train = data['ytrain']
y_test = data['ytest']
w, m, v = init_spectral_mixture_kernel(X_train, y_train)

def mykern(x, z, weights=w, means=m, variances=v):
    return spectral_mixture_kernel(x, z, weights, means, variances)
#gp = SimpleGaussianProcessRegressor(spectral_mixture_kernel)
gp = SimpleGaussianProcessRegressor(mykern)
gp.fit(X_train, y_train)
y_pred = gp.predict(X_test)
plt.plot(X_train, y_train)
plt.plot(X_test, y_test)
plt.plot(X_test, y_pred)
plt.show()
