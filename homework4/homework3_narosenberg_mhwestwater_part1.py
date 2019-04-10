from cvxopt import solvers, matrix
import numpy as np
import sklearn.svm
from math import copysign


class SVM453X:
    def __init__(self):
        self.w = 0
        self.b = 0

    # Expects each *row* to be an m-dimensional row vector. X should
    # contain n rows, where n is the number of examples.
    # y should correspondingly be an n-vector of labels (-1 or +1).
    def fit(self, X, y):
        # add ones to each data point for bias term
        Xtilde = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        m, n = Xtilde.shape

        G = -y[np.newaxis].T * Xtilde
        P = np.eye(n)
        q = np.zeros(n)
        h = -np.ones((m, 1))

        # Solve -- if the variables above are defined correctly, you can call this as-is:
        sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))

        # Fetch the learned hyperplane and bias parameters out of sol['x']
        alpha = np.array(sol['x'])
        self.w = alpha[:-1].reshape((-1,))  # get w and convert from a (3, 1) to (3, )
        self.b = alpha[-1]

    # Given a 2-D matrix of examples X, output a vector of predicted class labels
    def predict(self, x):
        # type: (np.ndarray) -> np.ndarray
        ans = np.zeros(x.shape[0])
        for i, val in enumerate(x):
            ans[i] = copysign(1, np.dot(val.reshape(1, -1), self.w) + self.b)
        return ans


def test1():
    # Set up toy problem
    X = np.array([[1, 1], [2, 1], [1, 2], [2, 3], [1, 4], [2, 4]])
    y = np.array([-1, -1, -1, 1, 1, 1])

    # Train your model
    svm453X = SVM453X()
    svm453X.fit(X, y)
    print(svm453X.w, svm453X.b)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard-margin
    svm.fit(X, y)
    print(svm.coef_, svm.intercept_)

    acc = np.mean(svm453X.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))


def test2(seed):
    np.random.seed(seed)

    # Generate random data
    X = np.random.rand(20,3)
    # Generate random labels based on a random "ground-truth" hyperplane
    while True:
        w = np.random.rand(3)
        y = 2*(X.dot(w) > 0.5) - 1
        # Keep generating ground-truth hyperplanes until we find one
        # that results in 2 classes
        if len(np.unique(y)) > 1:
            break

    svm453X = SVM453X()
    svm453X.fit(X, y)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard margin
    svm.fit(X, y)
    diff = np.linalg.norm(svm.coef_ - svm453X.w) + np.abs(svm.intercept_ - svm453X.b)
    print(diff)

    acc = np.mean(svm453X.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

    if acc == 1 and diff < 1e-1:
        print("Passed")


if __name__ == "__main__": 
    test1()
    for seed in range(5):
        test2(seed)
