from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as ran
import operator as op
import time


class KMeansClusterND(object):

    def __init__(self, numClusters, tol=1e-4):
        '''
            Arguments:
        numClusters  = the number of groups into which to cluster the data
        tol          = the tolerance of the stopping criteria during optimization
        '''

        self.numClusters = numClusters
        self.tol = tol


    def init_centroids(self, X):
        '''
        Provides randomly-assigned initial cluster centroids.
        '''

        if type(X) != np.ndarray:
            X = np.array(X)

        centroids = []
        dimensions = int(X.shape[1])

        ## find minimum and maximum of each dimension
        xMins = [min(X[:,dim]) for dim in range(dimensions)]
        xMaxs = [max(X[:,dim]) for dim in range(dimensions)]

        ## assign random centroids for each cluster
        for i in range(self.numClusters):
            xCents = [ran.random()*(xMaxs[j] - xMins[j]) + xMins[j] for j in range(dimensions)]
            print xCents
            centroids.append(xCents)

        return np.array(centroids)




if __name__ == '__main__':

    #X = pd.read_csv('c:\\winpy\\python\\Siamese\\example_data\\k_means_data.csv')
    X = np.random.rand(300,4)
    X = np.array(X)

    numClusters = 4
    tol = 1e-4

    km = KMeansClusterND(numClusters, tol)

    centroids = km.init_centroids(X)













