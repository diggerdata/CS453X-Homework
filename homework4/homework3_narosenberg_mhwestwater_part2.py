import sklearn.svm
import sklearn.metrics
import sklearn.model_selection as skms
import numpy as np
import pandas


def train_test_split(X, y, test_size=0.5, random_state=42):
    num_items = len(X)
    rng_idx = np.arange(num_items)
    np.random.shuffle(rng_idx)
    split_idx = np.split(rng_idx, 2)
    X_train = X[split_idx[0]]
    X_test = X[split_idx[1]]
    y_train = y[split_idx[0]]
    y_test = y[split_idx[1]]
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Load data
    d = pandas.read_csv('train.csv')
    y = np.array(d.target)  # Labels
    X = np.array(d.iloc[:, 2:])  # Features

    # Split into train/test folds
    X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.5, random_state=42)

    # Linear SVM
    svm_linear = sklearn.svm.SVC(kernel='linear', C=1e15, verbose=1)
    svm_linear.fit(X_train, y_train)

    # Non-linear SVM (polynomial kernel)
    svm_poly = sklearn.svm.SVC(kernel='poly', degree=3, C=1e15, verbose=1)
    svm_poly.fit(X_train, y_train)

    # Apply the SVMs to the test set
    yhat_linear = svm_linear.decision_function(X_test)  # Linear kernel
    yhat_poly = svm_poly.decision_function(X_test)  # Non-linear kernel

    # Compute AUC
    auc_linear = sklearn.svm.metrics.roc_auc(y_test, yhat_linear)
    auc_poly = sklearn.svm.metrics.roc_auc(y_test, yhat_poly)

    print(auc_linear)
    print(auc_poly)
