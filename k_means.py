from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as ran
import operator as op


class KMeansCluster(object):

    def __init__(self, numClusters, tol=1e-4):
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
            plt.plot(centroids[:,0], centroids[:,1], 'ro', alpha=0.8, ms=10,)

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

        centroids = []

        x1Min, x1Max = min(X[:,0]), max(X[:,0]) 
        x2Min, x2Max = min(X[:,1]), max(X[:,1])

        for i in range(self.numClusters):

            x1Cent = ran.random()*(x1Max - x1Min) + x1Min
            x2Cent = ran.random()*(x2Max - x2Min) + x2Min

            centroids.append([x1Cent, x2Cent])

        return np.array(centroids)


    def assign_clusters(self, X, centroids):
        '''
        Assigns each member of X to a centroid. Returns a list containing
        the index of the centroid to which each sample in X is the closest.
        '''

        if type(X) != np.ndarray:
            X = np.array(X)

        indexes = []

        ## for each sample, assign the closest centroid
        for x in X:
            squaredDists = [self.squared_distance(x, c) for c in centroids]
            index, value = min(enumerate(squaredDists), key=op.itemgetter(1))
            indexes.append(index)

        return np.array(indexes).reshape(len(indexes), 1)


    def adjust_centroids(self, X, indexes):
        '''
        Re-computes the location of each centroid.
        '''

        if type(X) != np.ndarray:
            X = np.array(X)

        X = np.append(X, indexes, axis=1)

        centroids = []

        ## for each centroid, re-compute its location by averaging over all
        ## samples which are assigned to it
        for i in range(self.numClusters):            
            centroids.append([np.mean(X[X[:,2]==i,0]), np.mean(X[X[:,2]==i,1])])

        return np.array(centroids)


    def converge(self, X, oldCentroids, plotting=False):
        '''
        Randomly initializes centroids and finds the optimal
        locations for each.
        '''

        residual = 1.0
        while residual > self.tol:
            indexes       = self.assign_clusters(X, oldCentroids)
            newCentroids  = self.adjust_centroids(X, indexes)
            J = self.cost_function(X, oldCentroids, indexes)
            print J
            residual      = self.absolute_change(oldCentroids, newCentroids)
            oldCentroids  = newCentroids

        if plotting:
            self.plot_data(X, newCentroids)

        return newCentroids


    def best_fit(self, X, numIter=100):
        '''
        Finds the centroids which yield the lowest cost function
        by initializing 'numIter' different initial centroids.
        '''

        if type(X) != np.ndarray:
            X = np.array(X)

        print '\nFinding the best set of centroids over {} random initializations:\n\n'.format(numIter)

        jMin = 1e100
        bestCentroids = None
        for i in range(numIter):

            print 'Iteration {} of {}'.format(i+1, numIter)

            initCentroids  = self.init_centroids(X)
            centroids      = self.converge(X, initCentroids, plotting=False)
            indexes        = self.assign_clusters(X, centroids)
            jNew           = self.cost_function(X, centroids, indexes)

            if jNew < jMin:
                bestCentroids = centroids
                print '\nNew minimum J value = {}\n'.format(jNew)
                jMin = jNew

        self.centroids = bestCentroids

        return None


    def cost_function(self, X, centroids, indexes):
        '''
        Computes the cost function at the current cluster state
        defined by 'X', 'indexes', and 'centroids'.
        '''

        return sum([self.squared_distance(x[0:2], indexes[i]) for i, x in enumerate(X)])


    def squared_distance(self, x, centroid):
        '''
        Returns the squared distance between the two points 'x'
        and 'centroid'.
        '''

        return sum((x - centroid)**2)**0.5


    def absolute_change(self, oldCentroids, newCentroids):
        '''
        Returns the sum of the distances between the elements in
        oldCentroids and those in newCentroids.
        '''

        return np.sum(abs(oldCentroids - newCentroids))




if __name__ == '__main__':

    X = pd.read_csv('c:\\winpy\\python\\Siamese\\example_data\\k_means_data.csv')

    km = KMeansCluster(numClusters=3, tol=1e-4)

    km.best_fit(X, numIter=1)
    km.plot_data(X, km.centroids)















        
