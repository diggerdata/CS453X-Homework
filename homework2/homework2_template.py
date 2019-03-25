import numpy as np
from tqdm import tqdm

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    return np.append(np.array([a.flatten() for a in faces]), np.ones((len(faces), 1)), axis=1).T

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):
    return np.mean(((np.dot(Xtilde.T, w)) - y)**2) / 2.0

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    return (Xtilde.dot(Xtilde.T.dot(w) - y))/np.shape(Xtilde)[1] + (alpha/(2*np.shape(Xtilde)[1])) * w.T.dot(w)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    return np.linalg.solve(Xtilde.dot(Xtilde.T),np.eye(len(Xtilde))).dot(Xtilde).dot(y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    w = np.random.randn(np.shape(Xtilde)[0], 1) * 0.01
    for i in tqdm(range(5000)):
        grad = gradfMSE(w, Xtilde, y)
        new_w = w - grad * 0.003
        w = new_w
    return w

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    pass

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")[np.newaxis].T
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    # w1 = method1(Xtilde_tr, ytr)
    # print(fMSE(w1, Xtilde_tr, ytr))
    # print(fMSE(w1, Xtilde_te, yte))
    w2 = method2(Xtilde_tr, ytr)
    print(fMSE(w2, Xtilde_tr, ytr))
    print(fMSE(w2, Xtilde_te, yte))
    # w3 = method3(Xtilde_tr, ytr)

    # Report fMSE cost using each of the three learned weight vectors
    # ...
