import numpy as np
from tqdm import tqdm

def preprocess(data):
    return np.append(data, np.ones((len(data), 1)), axis=1)

def sigmoid_activation(x):
	# compute and return the sigmoid activation value for a
	# given input value
	return 1.0 / (1 + np.exp(-x))
 
def next_batch(X, y, batchSize):
	# loop over our dataset `X` in mini-batches of size `batchSize`
	for i in np.arange(0, X.shape[0], batchSize):
		# yield a tuple of the current batched data and labels
		yield (X[i:i + batchSize], y[i:i + batchSize])

def fSGD(X, y, num_epochs, batch_size, alpha=0.01):
    W = np.random.uniform(size=(10,X.shape[1])).T
    # initialize a list to store the loss value for each epoch
    loss_history = []
    # loop over the desired number of epochs
    for epoch in tqdm(np.arange(0, num_epochs)):
        # initialize the total loss for the epoch
        epoch_loss = []
    
        # loop over our data in batches
        for (batch_x, batch_y) in next_batch(X, y, batch_size):
            # take the dot product between our current batch of
            # features and weight matrix `W`, then pass this value
            # through the sigmoid activation function
            preds = sigmoid_activation(batch_x.dot(W))
    
            # now that we have our predictions, we need to determine
            # our `error`, which is the difference between our predictions
            # and the true values
            error = preds - batch_y
    
            # given our `error`, we can compute the total loss value on
            # the batch as the sum of squared loss
            loss = np.sum(error ** 2)
            epoch_loss.append(loss)
    
            # the gradient update is therefore the dot product between
            # the transpose of our current batch and the error on the
            # # batch
            gradient = batch_x.T.dot(error) / batch_x.shape[0]
    
            # use the gradient computed on the current batch to take
            # a "step" in the correct direction
            W += -alpha * gradient
    
        # update our loss history list by taking the average loss
        # across all batches
        loss_history.append(np.average(epoch_loss))
    return W

def predict(X, y, w):
    a = X.dot(w)
    b = np.zeros_like(a)
    b[np.arange(len(a)), a.argmax(1)] = 1
    return b

def softmax(X, y):
    exps = np.exp(X)
    return exps / np.sum(exps)

def fCE(X, y, w):
    # y = y.argmax(axis=1)
    print(y.shape)
    p = softmax(X.dot(w), y)
    # p = X.dot(w)
    m = y.shape[0]
    log_likelihood = np.log(p[range(m),:]) * y
    return -1.0 * np.sum(np.sum(log_likelihood, axis=1), axis=0) / m

def fPC(X, y, w):
    y_hat = predict(X, y, w)
    return np.mean(y_hat == y)

if __name__ == '__main__':
    X_tr = preprocess(np.load("small_mnist_train_images.npy"))
    y_tr = np.load("small_mnist_train_labels.npy")
    X_te = preprocess(np.load("small_mnist_test_images.npy"))
    y_te = np.load("small_mnist_test_labels.npy")

    print('Starting training')
    w = fSGD(X_tr, y_tr, 500, 100)
    print('Training accuracy: ', fPC(X_tr, y_tr, w))
    print('Training CE: ', fCE(X_tr, y_tr, w))
    print('Testing accuracy: ', fPC(X_te, y_te, w))
    print('Testing CE: ', fCE(X_te, y_te, w))