from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as ran


class KMeansCluster(object):

    def __init__(self, numClusters, tol=1e-3):
        '''
            Arguments:
        numClusters  = the number of groups into which to cluster the data
        tol          = the tolerance of the stopping criteria during optimization
        '''

        self.numClusters = numClusters
        self.tol = tol


    def plot_data(self, X, centroids=[]):
        '''
        Visualizes the 2-D input data.
        '''

        if len(X.shape) > 2:
            print 'Error: Can not plot the {}-D input data on a 2-D plot'.format(len(X.shape))

        if type(X) != np.ndarray:
            X = np.array(X)

        ## determine axis limits
        xMin, xMax = min(X[:,0]), max(X[:,0]) 
        yMin, yMax = min(X[:,1]), max(X[:,1])

        xRange, yRange = abs(xMin - xMax), abs(yMin - yMax)

        xMin, xMax = xMin - 0.05*xRange, xMax + 0.05*xRange
        yMin, yMax = yMin - 0.05*yRange, yMax + 0.05*yRange 

        ## plot the data
        plt.plot(X[:,0], X[:,1], 'ko', alpha=0.75)
        if centroids.shape[0] > 0 and len(centroids.shape) == 2:
            plt.plot(centroids[:,0], centroids[:,1], 'yo', alpha=0.9)

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Raw Data')
        plt.axis([xMin, xMax, yMin, yMax])

        plt.show()

        return None


    def init_centroids(self, X):
        '''
        Provides randomly-assigned initial cluster centroids.
        '''

        if type(X) != np.ndarray:
            X = np.array(X)
        centroids = np.empty((0,2), float)

        x1Min, x1Max = min(X[:,0]), max(X[:,0]) 
        x2Min, x2Max = min(X[:,1]), max(X[:,1])

        for i in range(self.numClusters):

            x1Cent = ran.random()*(x1Max - x1Min) + x1Min
            x2Cent = ran.random()*(x2Max - x2Min) + x2Min

            centroids = np.append(centroids, np.array([[x1Cent, x2Cent]]), axis=0)

        return centroids


    def assign_clusters(self, X, centroids):
        '''
        Assigns each member of X to a centroid
        '''




if __name__ == '__main__':

    X = pd.read_csv('c:\\winpy\\python\\Siamese\\example_data\\k_means_data.csv')

    km = KMeansCluster(numClusters=2, tol=1e-3)

    #km.plot_data(X)

    centroids = km.init_centroids(X)
    print centroids

















        
