#%%
import copy
import itertools

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
    score = np.zeros((len(predictors), len(X)))
    for j, p in enumerate(predictors):
        score[j,:] = X[:, p[0], p[1]] > X[:, p[2], p[3]]
    predictions = np.mean(score, axis=0) > 0.5
    return fPC(y, predictions)

#%%
def permutate(L, N=2):
    return np.array([a for a in itertools.permutations(enumerate(L),N)])[:,:,0]

#%%
def process_combinations(face):
    combinations = permutate(face.flatten())
    thresh = combinations[:,0,1] > combinations[:,1,1]
    return thresh
#%%
def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels, num_predictors=5):
    i_permutation = permutate(trainingFaces[0].flatten())[:,0:2].T
    predict = np.array([np.unravel_index(a, (24,24)) for a in i_permutation[0,:].astype(int)])
    predict = np.append(predict, np.array([np.unravel_index(a, (24,24)) for a in i_permutation[1,:].astype(int)]), axis=1)
    results = "n,trainingAccuracy,testingAccuracy\r\n"
    # loop through each amount of training data
    predictors = []
    with tqdm(total=5*num_predictors*len(predict), desc='Training...') as pbar:
        for j in range(400, 2400, 400):
            predictors = []
            for k in range(num_predictors):
                best_score = 0.0
                best_predictor = None
                for p in predict:
                    test_predictors = copy.copy(predictors)
                    test_predictors.append(p.tolist())
                    accuracy = measureAccuracyOfPredictors(np.array(test_predictors), trainingFaces[:j], trainingLabels[:j])
                    if accuracy > best_score and p.tolist() not in predictors:
                        best_predictor = p.tolist()
                        best_score = accuracy
                    pbar.update(1)
                predictors.append(best_predictor)
            predictors = np.array(predictors)
            results = results + (str(j) + "," \
                + str(measureAccuracyOfPredictors(predictors, trainingFaces[:j], trainingLabels[:j])) \
                + "," + str(measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)) + "\r\n")
    print("------------------------------")
    print(results)

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
def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

#%%
if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels)
