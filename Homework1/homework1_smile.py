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
    for i in range(np.shape(predictors)[0]):
        # print(np.shape(predictors[i]))
        c1 = predictors[i,0]
        r1 = predictors[i,1]
        c2 = predictors[i,2]
        r2 = predictors[i,3]
        predictions = np.append(predictions, X[:,c1,r1] > X[:,c2,r2])
    yhat = np.mean(predictions, axis=0)
    return fPC(y, yhat)

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
    # multithread code for faster training
    results = "n    trainingAccuracy        testingAccuracy\r\n"
    for j in range(400, 401, 1):
        # # with Pool(cpu_count()) as p:
        # #     all_thresh = np.array(list(tqdm(p.imap(process_combinations, trainingFaces[:j]), total=len(trainingFaces[:j]))))
        # print(np.shape(all_thresh))
        # # find the best `num_predictors` in all possible combinations
        # predictions = np.zeros((1,np.shape(all_thresh)[1]))
        # for i in range(np.shape(all_thresh)[1]):
        #     predictions[0,i] = fPC(all_thresh[:,i], trainingLabels[:j])
        # print(i_permutation)
        # predictions = np.append(predictions, i_permutation, axis=0)
        # idx = predictions.argsort(axis=1)[0,-num_predictors:][::-1]
        # sorted_predictions = predictions[:,idx]
        # print(sorted_predictions)

        # get c1,r1,c2,r2 from flattened indexes
        i_permutation = permutate(trainingFaces[0].flatten())[:,0:2].T
        predict = np.array([np.unravel_index(a, (24,24)) for a in i_permutation[0,:].astype(int)])
        predict = np.append(predict, np.array([np.unravel_index(a, (24,24)) for a in i_permutation[1,:].astype(int)]), axis=1)
        predictors = []
        for k in tqdm(range(num_predictors)):
            best_score = 0
            best_predictor = None
            def _process_predictors(p):
                # print(p)
                test_predictors = predictors
                test_predictors.append(p)
                # print(test_predictors)
                accuracy = measureAccuracyOfPredictors(np.array(test_predictors), trainingFaces[:j], trainingLabels[:j])
                if accuracy > best_score:
                   best_predictor = p
                   best_score = accuracy
            predictors = np.array(predictors.append(best_predictor))
        # for i in range(np.shape(i_permutation)[0]):
        #     predictions[0,i] = fPC(all_thresh[:,i], trainingLabels[:j])
               
        # print(predictors)
        results = results + (str(j) + " " + str(measureAccuracyOfPredictors(predictors, trainingFaces[:j], trainingLabels[:j])) + "  " + str(measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)) + "\r\n")

    print(results)

    show = False
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
    faces = np.load("Homework1/{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("Homework1/{}ingLabels.npy".format(which))
    return faces, labels

#%%
if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels)
