import numpy as np

def problem1 (A, B):
    """Given matrices A and B, compute and return an expression for A + B.
    
    Parameters
    ----------
    A : Numpy array
    B : Numpy array
    
    Returns
    -------
    Numpy array
        The numpy array resulting from A + B.
    """

    return A + B

def problem2 (A, B, C):
    """Given matrices A, B, and C, compute and return AB − C.
    
    Parameters
    ----------
    A : Numpy array
    B : Numpy array
    C : Numpy array
    
    Returns
    -------
    Numpy array
        The numpy array resulting from AB + C.
    """

    return np.dot(A, B) - C

def problem3 (A, B, C):
    return A*B + C.T

def problem4 (x, y):
    return x.T*y

def problem5 (A):
    return np.zeros(*A.shape)

def problem6 (A):
    return np.ones(A.shape[0])

def problem7 (A, alpha):
    return A + alpha*np.eye(*A.shape)

def problem8 (A, i, j):
    return A[i,j]

def problem9 (A, i):
    return np.sum(A[i,:])

def problem10 (A, c, d):
    return np.mean(np.where(c <= A <= d, A))

def problem11 (A, k):
    return ...

def problem12 (A, x):
    return ...

def problem13 (A, x):
    return ...
