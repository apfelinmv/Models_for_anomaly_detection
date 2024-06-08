import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def plot_isolation_example(X, xi, xo):

    clf = IsolationForest(contamination=0.1)
    clf.fit(X)

    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    axs[0].scatter(X[:, 0], X[:, 1], c='grey', s=20, edgecolor='k')
    axs[0].scatter(xi[0], xi[1], c='blue', s=100, edgecolor='k', label='x1')
    axs[0].set_title('(a) regular point')
    axs[0].legend()

    axs[1].contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    axs[1].scatter(X[:, 0], X[:, 1], c='grey', s=20, edgecolor='k')
    axs[1].scatter(xo[0], xo[1], c='red', s=100, edgecolor='k', label='x0')
    axs[1].set_title('(b) anomalous point')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# Generate a synthetic dataset
np.random.seed(42)
X = np.random.randn(300, 2)

xi = np.array([0.5, 0.5])
xo = np.array([-2, -2])

plot_isolation_example(X, xi, xo)
