## k_nearest_neighbors.py

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os

class KNearestNeighbor(object):

    def __init__(self, X, x):
        '''
            Arguments:
        X  = Training set of data. X is assumed to be an array object with
             first axis being the number of samples and the length of the
             second axis being the dimension.
        x  = Test data point
        '''

        self.x = x
        self.X = X


    def distance(self, a, b):

        return sum((a -b)**2)**0.5


    def distances_from_origin(self, X):

        dimensions = X.shape[1]
        origin = np.zeros([dimensions])

        return sorted([self.distance(x, origin) for x in X])


    def closest_neighbors(self, X):

        originDist = self.distances_from_origin(X)
        originDist = np.array([originDist]).reshape(len(originDist), 1)

        print originDist.shape


    




if __name__ == '__main__':

    X = np.random.rand(10,2)

    knn = KNearestNeighbor(X, X[0])

    #plt.plot(X, color='k', ls='', marker='o', alpha=0.60)
    #plt.show()

    knn.closest_neighbors(knn.X)











