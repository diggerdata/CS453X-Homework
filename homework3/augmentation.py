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


if __name__ == "__main__":
    # Load data
    train = np.load("small_mnist_train_images.npy")
    train_label = np.load("small_mnist_train_labels.npy")

    # img_out = translate_left(train[5], 12)
    # print(train[5].shape)
    # print(img_out.shape)
    # plt.imshow(img_out.reshape((28, 28)))
    # plt.show()


    imgs = train
    labels = train_label
    rng.seed(123458)

    for j in range(2):
        transformed_imgs = np.zeros(train.shape)
        for i, img in tqdm(enumerate(train)):
            if j == 0:
                transformed = translate_down(img, 27)
                transformed = translate_right(transformed, 26)
            else:
                # transformed = translate_up(img, 27)
                # transformed = translate_left(transformed, 26)
                transformed = add_noise(img)

            rotated = rotate(transformed, rng.randint(1, 4) * (1 if rng.getrandbits(1) else -1))
            # noised = add_noise(transformed)
            transformed_imgs[i, :] = rotated

        imgs = np.concatenate([imgs, transformed_imgs])
        labels = np.concatenate([labels, train_label])

    i = 3002

    print(imgs.shape)
    print(train_label.shape)
    print(imgs.shape)
    print(labels.shape)

    np.save("full_train.npy", imgs)
    np.save("full_train_labels.npy", labels)

    plt.imshow(imgs[i].reshape((28, 28)))
    plt.show()
    plt.imshow(imgs[i+5000].reshape((28, 28)))
    plt.show()
    plt.imshow(imgs[i+10000].reshape((28, 28)))
    plt.show()

    print(train_label[i])


