import numpy as np
import matplotlib.pyplot as plt
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
    return gradientDescent(Xtilde, y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde, y, alpha=ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations
    w = np.random.randn(np.shape(Xtilde)[0], 1) * 0.01
    for i in tqdm(range(T)):
        grad = gradfMSE(w, Xtilde, y, alpha=alpha)
        new_w = w - grad * EPSILON
        w = new_w
    return w

def visualizeWeights(w1, w2, w3):
    fig,(ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3)
    new_w1 = np.reshape(w1[:-1], (48,48))
    new_w2 = np.reshape(w2[:-1], (48,48))
    new_w3 = np.reshape(w3[:-1], (48,48))
    ax1.imshow(new_w1)
    ax1.set_title('Part A')
    ax2.imshow(new_w2)
    ax2.set_title('Part B')
    ax3.imshow(new_w3)
    ax3.set_title('Part C')
    fig.show()

def visualizeErrors(w, Xtilde, y, k=5):
    errors = (np.dot(Xtilde.T, w)) - y
    idx = errors.argsort(axis=0)[-k:][::-1]
    top_errors = Xtilde[:, idx]
    top_predictions = errors[idx]
    top_labels = y[idx]
    fig, ax = plt.subplots(1,k, figsize=(15,9))
    for i in range(np.shape(top_errors)[1]):
        rs_img = top_errors[:-1, i].reshape((48,48))
        ax[i].imshow(rs_img)
        ax[i].set_title('Pred: ' + str(round(top_predictions[i,0,0], 2)) + ' Act: ' + str(top_labels[i,0,0]))
    plt.show()

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")[np.newaxis].T
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")[np.newaxis].T

    print('Part A')
    w1 = method1(Xtilde_tr, ytr)
    print(fMSE(w1, Xtilde_tr, ytr))
    print(fMSE(w1, Xtilde_te, yte))

    print('Part B')
    w2 = method2(Xtilde_tr, ytr)
    print(fMSE(w2, Xtilde_tr, ytr))
    print(fMSE(w2, Xtilde_te, yte))

    print('Part C')
    w3 = method3(Xtilde_tr, ytr)
    print(fMSE(w3, Xtilde_tr, ytr))
    print(fMSE(w3, Xtilde_te, yte))
    
    visualizeWeights(w1, w2, w3)
    visualizeErrors(w3, Xtilde_te, yte)

