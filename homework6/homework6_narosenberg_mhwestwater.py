import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize
from copy import deepcopy
from tqdm import tqdm, trange

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    W1 = w['W1']
    b1 = w['b1']
    W2 = w['W2']
    b2 = w['b2']
    return W1, b1, W2, b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("small_mnist_{}_images.npy".format(which)).T
    labels = np.load("small_mnist_{}_labels.npy".format(which)).T
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

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(X), Y) + np.multiply((1 - Y), np.log(1 - X))
    cost = - np.sum(logprobs) / m    
    
    cost = np.squeeze(cost)     
    return cost

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    dW1, db1, dW2, db2 = unpack(grads)
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return pack(W1, b1, W2, b2)

def sigmoid(X):
    return 1/(1+np.exp(-X))

def softmax(x):
    return np.exp(x) / np.exp(x).sum()

def relu(x, derivative=False):
    if derivative:
        return 1*(x > 0)
    else:
        return x*(x > 0)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def forwardProp(X, w):
    W1, _, W2, _ = unpack(w)
    h = x * W1
    h = relu(h)
    prob = softmax(h * W2)
    return h, prob

def backProp(X, y, w, cache):
    m = X.shape[1]
    W1, _, W2, _ = unpack(w)
    _, A1, _, A2 = unpack(cache)
    dZ2 = A2 - y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return pack(dW1, db1, dW2, db2)

def next_batch(X, y, batch_size):
    """Get a batch of `batch_size` from training data."""
    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])

def updateWeights(w, grads, learning_rate=0.01):
    W1, b1, W2, b2 = unpack(w)
    dW1, db1, dW2, db2 = unpack(grads)
    for batch_x, batch_y in next_batch(X, y, batch_size):
            loss = fCE(batch_x, batch_y, w)
            gradient = gradCE(batch_x, batch_y, w)
            epoch_loss.append(loss)
            w += -alpha * gradient
    return   

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN. Then return the sequence of w's obtained during SGD.
def train (trainX, trainY, testX, testY, w, epochs=500, print_cost=True, lr=0.01):
    ws = deepcopy(w)
    for i in trange(epochs):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forwardProp(trainX, ws)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = fCE(A2, trainY, ws)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backProp(trainX, trainY, ws, cache)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        ws = updateWeights(ws, grads, lr=lr)
        
        # Print the cost every 1000 iterations
        if print_cost and i % 10 == 0:
            tqdm.write("Cost after iteration %i: %f" %(i, cost))
    return ws

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones((NUM_HIDDEN,1))
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones((NUM_OUTPUT,1))
    w = pack(W1, b1, W2, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    # print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
    #                                 lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
    #                                 w))

    # Train the network and obtain the sequence of w's obtained using SGD.
    ws = train(trainX, trainY, testX, testY, w, lr=0.05)

    # Plot the SGD trajectory
    plotSGDPath(trainX, trainY, ws)
