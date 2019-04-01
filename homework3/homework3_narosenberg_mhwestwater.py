import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess(data):
    return np.append(data, np.ones((len(data), 1)), axis=1)

def softmax(X):
    X -= np.max(X)
    return (np.exp(X).T / np.sum(np.exp(X), axis=1)).T

def fGrad(X, y, w):
    m = X.shape[0]
    p = softmax(X.dot(w))
    return -np.dot(X.T,(y - p)) / m

def fCE(X, y, w):
    p = softmax(X.dot(w))
    m = y.shape[0]
    return -np.sum(y * np.log(p)) / m
 
def next_batch(X, y, batch_size):
	for i in np.arange(0, X.shape[0], batch_size):
		yield (X[i:i + batch_size], y[i:i + batch_size])

def fSGD(X, y, num_epochs, batch_size, alpha=0.1):
    w = np.random.randn(X.shape[1], 10) * 0.1
    loss_history = []

    for epoch in tqdm(np.arange(0, num_epochs), desc='Training'):
        epoch_loss = []
        for batch_x, batch_y in next_batch(X, y, batch_size):
            loss = fCE(batch_x, batch_y, w)
            gradient = fGrad(batch_x, batch_y, w)
            epoch_loss.append(loss)
            w += -alpha * gradient
        loss_history.append(np.average(epoch_loss))
    return w, loss_history

def predict(X, y, w):
    p = X.dot(w)
    y_hat = np.zeros_like(p)
    y_hat[np.arange(p.shape[0]), p.argmax(1)] = 1
    return y_hat

def fPC(X, y, w):
    y_hat = predict(X, y, w)
    return np.mean(y_hat == y)

if __name__ == '__main__':
    X_tr = preprocess(np.load("small_mnist_train_images.npy"))
    y_tr = np.load("small_mnist_train_labels.npy")
    X_te = preprocess(np.load("small_mnist_test_images.npy"))
    y_te = np.load("small_mnist_test_labels.npy")

    w, loss_history = fSGD(X_tr, y_tr, 100, 100)
    print('Training accuracy: ', fPC(X_tr, y_tr, w))
    print('Training CE: ', fCE(X_tr, y_tr, w))
    print('Testing accuracy: ', fPC(X_te, y_te, w))
    print('Testing CE: ', fCE(X_te, y_te, w))

    plt.plot(loss_history)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss History')
    plt.show()