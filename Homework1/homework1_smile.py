#%%
import itertools
from multiprocessing import Pool
from os import cpu_count

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#%%
def fPC (y, yhat):
    return np.mean(y == yhat)
#%%
def measureAccuracyOfPredictors (predictors, X, y):
    predictions = np.zeros(len(X))
    for i, face in enumerate(X):
        score = np.zeros(len(predictors))
        for j, p in enumerate(predictors):
            score[j] = face[p[0], p[1]] > face[p[2], p[3]]
        predictions[i] = round(np.mean(score))
    return fPC(predictions, y)

#%%
def permutate(L, N=2):
    return np.array([a for a in itertools.permutations(enumerate(L),N)])

#%%
def process_combinations(face):
    combinations = permutate(face.flatten())
    thresh = (combinations[:,0,1] > combinations[:,1,1])[np.newaxis]
    return thresh
#%%
def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels, num_predictors=5):
    # multithread code for faster training
    with Pool(cpu_count()) as p:
        all_thresh = np.array(list(tqdm(p.imap(process_combinations, trainingFaces), total=len(trainingFaces))))
    
    # find the best `num_predictors` in all possible combinations
    correct = np.zeros((1,np.shape(all_thresh)[2]))
    for i in range(np.shape(all_thresh)[2]):
        correct[0,i] = fPC(all_thresh[:,0,i], trainingLabels)
    i_permutation = permutate(trainingFaces[0].flatten())[:,:,0].T
    correct = np.append(correct, i_permutation, axis=0)
    idx = correct.argsort()[0,-num_predictors:][::-1]
    sorted_correct = correct[:,idx]

    # get c1,r1,c2,r2 from flattened indexes
    predictors = np.array([np.unravel_index(a, (24,24)) for a in sorted_correct[1,:].astype(int)])
    predictors = np.append(predictors, np.array([np.unravel_index(a, (24,24)) for a in sorted_correct[2,:].astype(int)]), axis=1)
    
    print("Predictors:")
    print(np.shape(predictors))
    print(predictors)
    print("-------------------------")
    print("Accuracy on training set:")
    print(measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels))
    print("Accuracy on testing set:")
    print(measureAccuracyOfPredictors(predictors, testingFaces, testingLabels))

    show = True
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[1,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        # Show r1,c1 for all predictors
        for i in range(np.shape(predictors)[0]):
            rect = patches.Rectangle((predictors[i,0]-0.5,predictors[i,1]-0.5),1,1,linewidth=2,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            # Show r2,c2
            rect = patches.Rectangle((predictors[i,2]-0.5,predictors[i,3]-0.5),1,1,linewidth=2,edgecolor='b',facecolor='none')
            ax.add_patch(rect)
        # Display the merged result
        plt.show()

#%%
def loadData (which, amount=2000):
    faces = np.load("Homework1/{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("Homework1/{}ingLabels.npy".format(which))
    return faces[:amount,:,:], labels[:amount]

#%%
if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels)
