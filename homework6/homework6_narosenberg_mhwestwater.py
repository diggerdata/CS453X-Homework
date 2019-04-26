import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize
from tqdm import tqdm, trange

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 50  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    W1 = w[:(784*NUM_HIDDEN)].reshape(784, NUM_HIDDEN)
    b1 = w[(784*NUM_HIDDEN):(784*NUM_HIDDEN+NUM_HIDDEN)].reshape(NUM_HIDDEN)
    W2 = w[(784*NUM_HIDDEN+NUM_HIDDEN):(784*NUM_HIDDEN+NUM_HIDDEN+NUM_HIDDEN*10)].reshape(NUM_HIDDEN, 10)
    b2 = w[(784*NUM_HIDDEN+NUM_HIDDEN+NUM_HIDDEN*10):].reshape(10)
    return W1, b1, W2, b2

def unpackCache(w):
    return w[0], w[1], w[2], w[3]

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    w= np.concatenate((W1,b1,W2,b2), axis=None)
    return w

def packCache(Z1, A1, Z2, A2):
    return [Z1, A1, Z2, A2]

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("mnist_{}_images.npy".format(which))
    labels = np.load("mnist_{}_labels.npy".format(which))
    return images, labels

def plotSGDPath (trainX, trainY, ws):
    def toyFunction (x1, x2):
        return np.sin((2 * x1**2 - x2) / 10.)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    axis2 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    Yaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    Zaxis = toyFunction(Xaxis, Yaxis)
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1.0 * (x > 0)

def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

def forward(x, w):
    W1, b1, W2, b2 = unpack(w)
    Z1 = np.dot(x, W1) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    cache = packCache(Z1, A1, Z2, A2)
    return cache

def backward(X, y, w, cache):
    W1, _, W2, _ = unpack(w)
    Z1, A1, Z2, A2 = unpackCache(cache)
    g = np.multiply((A2 - y).T * W2, dReLU(Z1.T))
    dW2 = (A2 - y) * ReLU(Z1).T
    db2 = A2 -y
    dW1 = g * X.T
    db1 = g
    return pack(dW1, db1, dW2, db2)

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE (X, y, w):
    p = forward(X, w)[3]
    m = y.shape[0]
    return -np.sum(np.sum(y * np.log(p), axis=1), axis=0) / m

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, y, w, lambd=0.7):
    W1, _, W2, _ = unpack(w)
    m = X.shape[0]
    Z1, A1, Z2, A2 = unpackCache(forward(X, w))
    g = (np.multiply(np.dot(W2, (A2 - y).T), dReLU(Z1.T))).T
    dW2 = (1/m) * np.dot(ReLU(Z1).T, (A2 - y)) + (lambd/m)*W2
    db2 = np.mean(A2 - y, axis=0)
    dW1 = (1/m) * np.dot(X.T, g) + (lambd/m)*W1
    db1 = np.mean(g, axis=0)
    return pack(dW1, db1, dW2, db2)

def updateWeights(w, grads, learning_rate=1e-4):
    W1, b1, W2, b2 = unpack(w)
    dW1, db1, dW2, db2 = unpack(grads)
    W1 += -learning_rate * dW1
    b1 += -learning_rate * db1
    W2 += -learning_rate * dW2
    b2 += -learning_rate * db2
    return pack(W1, b1, W2, b2)

def next_batch(X, y, batch_size):
    """Get a batch of `batch_size` from training data."""
    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])

def predict(X, y, w):
    """Get predictions from a dataset and weigts. The lables (y) need to be in 
    one-hot form.
    """
    A2 = forward(X, w)[3]
    return A2 > 0.5

def fPC(X, y, w):
    """Compute the fPC."""
    y_hat = predict(X, y, w)
    return np.mean(y_hat == y)

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN. Then return the sequence of w's obtained during SGD.
def train (trainX, trainY, testX, testY, w, epochs=300, batch_size=256, learning_rate=0.1, print_loss=True):
    ws = np.copy(w)
    for epoch in trange(epochs, desc="Training"):
        for batch_x, batch_y in next_batch(trainX, trainY, batch_size): 
            grads = gradCE(batch_x, batch_y, ws)
            ws = updateWeights(ws, grads, learning_rate=learning_rate)
        if print_loss and epoch % 10:
            tqdm.write("Loss: {}".format(fCE(testX, testY, ws)))
    print("Final loss: {}".format(fCE(testX, testY, ws)))
    print("Final accuracy: {}".format(fPC(testX, testY, ws)))
    return ws

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_INPUT, NUM_HIDDEN))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    w = pack(W1, b1, W2, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_, lambd=0.0), \
                                    w))

    # Train the network and obtain the sequence of w's obtained using SGD.
    ws = train(trainX, trainY, testX, testY, w)

    # Plot the SGD trajectory
    plotSGDPath(trainX, trainY, ws)
