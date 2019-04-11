import sklearn.svm
import sklearn.metrics
import sklearn.model_selection as skms
import numpy as np
import pandas
from tqdm import tqdm

# accuracy numbers
# linear: 0.8565898024372915
# poly:   0.8349351974958529


# split a set of data and labels in half
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


# fit a linear and poly svc and return the decision function for each model
def compute_yhat(x_train, x_test, y_train):
    # Linear SVM
    svm_linear = sklearn.svm.LinearSVC(dual=False, C=1e15, verbose=0)
    svm_linear.fit(x_train, y_train)

    # Non-linear SVM (polynomial kernel)
    svm_poly = sklearn.svm.SVC(kernel='poly', degree=3, gamma='auto', verbose=0)
    svm_poly.fit(x_train, y_train)

    # Apply the SVMs to the test set
    yhat_linear = svm_linear.decision_function(x_test)  # Linear kernel
    yhat_poly = svm_poly.decision_function(x_test)  # Non-linear kernel

    return yhat_linear, yhat_poly


# split the data up into a number of bags, run the svns, and average the results
def bag(x_train, x_test, y_train, num_bags=50):
    # create an array with all indexes to permeate and split into different groups
    rng_idx = np.arange(len(x_train))
    np.random.shuffle(rng_idx)
    bag_list = np.split(rng_idx, num_bags)
    # create arrays to store the data
    pred_linear = np.empty((0, x_test.shape[0]))
    pred_poly = np.empty((0, x_test.shape[0]))

    # iterate over all the different bags of indexes and run the svns
    for idx in tqdm(bag_list):
        linear, poly = compute_yhat(x_train[idx], x_test, y_train[idx])
        pred_linear = np.append(pred_linear, [linear], axis=0)
        pred_poly = np.append(pred_poly, [poly], axis=0)

    # take the average of all the different test sets
    yhat_linear = np.mean(pred_linear, axis=0)
    yhat_poly = np.mean(pred_poly, axis=0)

    return yhat_linear, yhat_poly


if __name__ == '__main__':
    # Load data
    d = pandas.read_csv('train.csv')
    y = np.array(d.target)  # Labels
    X = np.array(d.iloc[:, 2:])  # Features

    # Split into train/test folds
    # X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.5, random_state=42)  # built in version
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    yhat_linear, yhat_poly = bag(X_train, X_test, y_train)

    # Compute AUC
    print(y_test.shape)
    print(yhat_linear.shape)
    auc_linear = sklearn.metrics.roc_auc_score(y_test, yhat_linear)
    auc_poly = sklearn.metrics.roc_auc_score(y_test, yhat_poly)

    print(f"AUC linear: {auc_linear}")
    print(f"AUC poly:   {auc_poly}")


