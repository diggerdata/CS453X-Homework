import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import skimage.transform
import random as rng


##########
# Part 1 #
##########

def permutate(X, y):
    """Shuffle data and lables."""
    permutation = np.random.permutation(X.shape[0])
    shuffled_X = X[permutation]
    shuffled_y = y[permutation]
    return shuffled_X, shuffled_y

def preprocess(X):
    """Preprocess data by appending ones to each sample."""
    return np.append(X, np.ones((X.shape[0], 1)), axis=1)

def softmax(X):
    """Softmax function."""
    X -= np.max(X)
    return (np.exp(X).T / np.sum(np.exp(X), axis=1)).T

def fGrad(X, y, w):
    """Gradient of the error."""
    m = X.shape[0]
    p = softmax(X.dot(w))
    return -np.dot(X.T,(y - p)) / m

def fCE(X, y, w):
    """Compute the cross entropy of the prediction."""
    p = softmax(X.dot(w))
    m = y.shape[0]
    return -np.sum(y * np.log(p)) / m
 
def next_batch(X, y, batch_size):
    """Get a batch of `batch_size` from training data."""
    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])

def fSGD(X, y, num_epochs, batch_size, alpha=0.1):
    """Compute the stochastic gradient descent (SGD)."""
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
    """Get predictions from a dataset and weigts. The lables (y) need to be in 
    one-hot form.
    """
    p = X.dot(w)
    y_hat = np.zeros_like(p)
    y_hat[np.arange(p.shape[0]), p.argmax(1)] = 1
    return y_hat

def fPC(X, y, w):
    """Compute the fPC."""
    y_hat = predict(X, y, w)
    return np.mean(y_hat == y)

##########
# Part 2 #
##########


def translate_up(img, w, n=28):
    img_out = np.zeros(n**2)
    img_out[0:w*n] = img[(n-w)*n:]
    return img_out


def translate_down(img, w, n=28):
    img_out = np.zeros(n**2)
    img_out[(n-w)*n:] = img[0:w*n]
    return img_out


def translate_right(img, w, n=28):
    p = (n - w) // 2
    img_square = img.reshape((n, n))
    img_cut = img_square.T[p:n - p].T
    img_out = np.zeros((n, n))
    for i, val in enumerate(img_cut):
        img_out[i, n-w:] = val
    return img_out.flatten()


def translate_left(img, w, n=28):
    p = (n - w) // 2
    img_square = img.reshape((n, n))
    img_cut = img_square.T[p:n - p].T
    img_out = np.zeros((n, n))
    for i, val in enumerate(img_cut):
        img_out[i, 0:w] = val
    return img_out.flatten()


def rotate(img, angle):
    return skimage.transform.rotate(img.reshape(28, 28), angle).flatten()


def add_noise(img, stdv=0.03, mu=0.03, n=28):
    noise = stdv * np.random.randn(n**2) + mu
    return img + noise


def sharpen(img_in, threshold=0.005, add=0.25):
    img_out = np.zeros(img_in.shape)
    idx = img_in > threshold
    img_out[idx] = add
    img_in += img_out
    img_in[img_in > 1] = 1
    return img_in


def augment_data(X, y):
    imgs_out = X
    labels_out = y
    rng.seed(123458)
    for j in range(2):
        transformed_imgs = np.zeros(X.shape)
        transformed_labels = np.zeros(y.shape)
        for i, img in tqdm(enumerate(X), desc='Augmenting'):
            label = y[i]
            label_digit = label.nonzero()[0][0]
            if j == 0:
                img = translate_down(img, 27)
                img = translate_right(img, 26)
            else:
                img = translate_up(img, 27)
                img = translate_left(img, 26)

            if label_digit in (0, 8):
                img = rotate(img, 180)

            img = sharpen(img)
            img = add_noise(img)
            img = rotate(img, rng.randint(1, 4) * (1 if rng.getrandbits(1) else -1))
            transformed_imgs[i, :] = img
            transformed_labels[i, :] = label

        imgs_out = np.concatenate([imgs_out, transformed_imgs])
        labels_out = np.concatenate([labels_out, transformed_labels])

    return imgs_out, labels_out


def run_and_print(Xtr, ytr, Xte, yte, title=""):
    w, loss_history = fSGD(Xtr, ytr, 100, 100)
    print('Training accuracy: ', fPC(Xtr, ytr, w))
    print('Training CE: ', fCE(Xtr, ytr, w))
    print('Testing accuracy: ', fPC(Xte, yte, w))
    print('Testing CE: ', fCE(Xte, yte, w))

    plt.plot(loss_history)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title(f'Loss History: {title}')
    plt.show()


if __name__ == '__main__':
    X_tr_raw = np.load("small_mnist_train_images.npy")
    X_tr = preprocess(X_tr_raw)
    y_tr_raw = np.load("small_mnist_train_labels.npy")
    # X_tr_raw = np.load("full_train.npy")
    # X_tr = preprocess(X_tr_raw)
    # y_tr = np.load("full_train_labels.npy")
    X_te = preprocess(np.load("small_mnist_test_images.npy"))
    y_te = np.load("small_mnist_test_labels.npy")
    # Shuffle the training data
    X_tr, y_tr = permutate(X_tr, y_tr_raw)

    print("Accuracy with raw data set")
    run_and_print(X_tr, y_tr, X_te, y_te, "Raw Data")

    # Augment data and run again
    print("\nAccuracy with augmented data set")
    X_tr_augmented_raw, y_tr_augmented = augment_data(X_tr_raw, y_tr_raw)
    X_tr_augmented = preprocess(X_tr_augmented_raw)
    X_tr_augmented, y_tr_augmented = permutate(X_tr_augmented, y_tr_augmented)

    run_and_print(X_tr_augmented, y_tr_augmented, X_te, y_te, "Augmented Data")

