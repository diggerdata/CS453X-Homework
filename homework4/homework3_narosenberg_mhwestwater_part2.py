import sklearn.svm
import sklearn.metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas

# Load data
d = pandas.read_csv('train.csv')
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:, 2:])  # Features

# Split into train/test folds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
print(X.shape)
print(X_train.shape)
# Linear SVM
# TODO

# Non-linear SVM (polynomial kernel)
# TODO

# Apply the SVMs to the test set
# yhat1 = ...  # Linear kernel
# yhat2 = ...  # Non-linear kernel

# Compute AUC
#auc1 = ...
#auc2 = ...

print(auc1)
print(auc2)
