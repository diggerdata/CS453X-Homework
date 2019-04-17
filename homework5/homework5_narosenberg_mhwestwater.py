import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load data and convert y from 1 hot to the int value
    X = np.load('small_mnist_test_images.npy')
    y = np.argmax(np.load('small_mnist_test_labels.npy'), axis=1)

    # PCA algorithm
    Xtilde = X - np.mean(X, axis=0)  # X - Xbar
    eigval, eigvec = np.linalg.eig(Xtilde.dot(Xtilde.T))
    idx = np.argsort(eigval)[::-1]  # sort by eigenvalues

    # plot top 2 components
    plt.scatter(eigvec[:, idx[0]], eigvec[:, idx[1]], s=0.5, c=y)
    plt.show()
