import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
from tqdm import tqdm
import random as rng


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


def sharpen(img_in, threshold=0.01, add=0.25):
    img_out = np.zeros(img_in.shape)
    idx = img_in > threshold
    img_out[idx] = add
    img_in += img_out
    img_in[img_in > 1] = 1
    return img_in


if __name__ == "__main__":
    # Load data
    train = np.load("small_mnist_train_images.npy")
    train_label = np.load("small_mnist_train_labels.npy")

    imgs_out = train
    labels_out = train_label
    rng.seed(123458)

    SHARP = sharpen(train[5])
    plt.imshow(train[5].reshape((28, 28)))
    plt.show()
    plt.imshow(SHARP.reshape((28, 28)))
    plt.show()

    for j in range(2):
        transformed_imgs = np.zeros(train.shape)
        transformed_labels = np.zeros(train_label.shape)
        for i, img in tqdm(enumerate(train)):
            label = train_label[i]
            label_digit = label.nonzero()[0][0]
            if j == 0:
                img = translate_down(img, 27)
                img = translate_right(img, 26)
            else:
                img = translate_up(img, 27)
                img = translate_left(img, 26)

            if label_digit in (0, 8):
                img = rotate(img, 180)

            # if label_digit in (6, 9):
            #     img = rotate(img, 180)
            #     label = np.zeros(10)
            #     label[label_digit] = 1

            img = sharpen(img)
            # img = add_noise(img)
            img = rotate(img, rng.randint(1, 4) * (1 if rng.getrandbits(1) else -1))
            transformed_imgs[i, :] = img
            transformed_labels[i, :] = label

        imgs_out = np.concatenate([imgs_out, transformed_imgs])
        labels_out = np.concatenate([labels_out, train_label])

    i = 3004

    print(imgs_out.shape)
    print(train_label.shape)
    print(imgs_out.shape)
    print(labels_out.shape)

    np.save("full_train.npy", imgs_out)
    np.save("full_train_labels.npy", labels_out)

    plt.imshow(imgs_out[i].reshape((28, 28)))
    plt.show()
    plt.imshow(imgs_out[i + 5000].reshape((28, 28)))
    plt.show()
    plt.imshow(imgs_out[i + 10000].reshape((28, 28)))
    plt.show()

    print(train_label[i])


