from __future__ import division

from scipy.optimize import minimize
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

import sklearn.datasets as datasets
import numpy as np
import pandas as pd

class NeuralNetwork(object):

    def __init__(self, lambda_=0, maxIter=500, smallInitValue=0.1, hiddenLayerSizes=[25], optimizeAlgo='BFGS'):
        '''
            Arguments:
        lambda_           = regularization parameter
        maxIter           = max # of iterations for optimization algorithm to compute
        smallInitValue    = parameter by which to normalize the initial random weights
        hiddenLayerSizes  = list containing the number of hidden units in each hidden layer
        optimizeAlgo      = the optimization algorithm to use (see scipy.optimize.minimize)
        '''

        self.lambda_ = lambda_
        self.maxIter = maxIter
        self.smallInitValue = smallInitValue
        self.numHiddenLayers = len(hiddenLayers)
        self.hiddenLayerSizes = hiddenLayers
        self.optimizeAlgo = optimizeAlgo


    def learn(self, X, y):
        '''
        The learning routine.
            Arguments:
        X  = 2-dimensional training set array
        y  = 1-dimensional classification array
        '''

        self.inputLayerSize = X.shape[1]
        self.numClassifyValues = len(set(y))

        thetasInit = []
        ## initialize the input layer theta array
        thetasInit.append(self.initialize_weights(self.inputLayerSize, self.hiddenLayerSizes[0]))

        ## initialize the hidden layer theta arrays
        for i in range(0, self.numHiddenLayers-1):
            thetasInit.append(self.initialize_weights(self.hiddenLayerSizes[i], self.hiddenLayerSizes[i+1]))

        ## initialize the output layer theta array
        thetasInit.append(self.initialize_weights(self.hiddenLayerSizes[self.numHiddenLayers-1], self.numClassifyValues))

        ## vectorize the theta arrays
        thetasInitVec = self.vec_thetas(thetasInit)

        ## run the optimization routine
        _res = minimize(self.cost_function, thetasInitVec, jac=self.cost_function_deriv, method=self.optimizeAlgo, \
                        args=(X, y), options={'maxiter': self.maxIter})

        ## store the fitted weights as an attribute - a list of the theta arrays
        self.thetas = self.unvec_thetas(_res.x)


    def classify(self, X):
        '''
        Given an input array X, return an array containing the classification 
        values of each of the samples contained within X.
        '''

        _, _, _, _, hypothesis = self._forward(X, self.thetas)
        return hypothesis.argmax(0)
    

    def predict_probability(self, X):
        '''
        Given an input array X, returns an array containing the classification
        probability for each of the possible classification values for each of
        the samples contained within X.
        '''

        _, _, _, _, hypothesis = self._forward(X, self.thetas)
        return hypothesis


    def cost_function(self, thetas, X, y):
        '''
        Computes the cost function J(theta).
            Arguments:
        thetas  = the vectorized array of weights
        X       = the input array of samples
        y       = the input array of classifications
        '''

        thetas = self.unvec_thetas(thetas)

        m = X.shape[0]
        Y = np.eye(self.numClassifyValues)[y]
        
        _, _, _, _, h = self._forward(X, thetas)

        cost = (-Y * np.log(h).T) - ((1 - Y) * np.log(1 - h).T)
        J = np.sum(cost) / m
        
        ## apply regularization
        if self.lambda_ != 0:
            t1f = t1[:, 1:]
            t2f = t2[:, 1:]
            reg = (self.lambda_ / (2 * m)) * (self.sum_square(t1f) + self.sum_square(t2f))
            J += reg
        return J
        

    def cost_function_deriv(self, thetas, X, y):
        '''
        Computes the derivative of the cost function J(theta).
            Arguments:
        thetas  = the vectorized array of weights
        X       = the input array of samples
        y       = the input array of classifications
        '''

        thetas = self.unvec_thetas(thetas)
        
        m = X.shape[0]
        Y = np.eye(self.numClassifyValues)[y]

        tNoBias = [t[:, 1:] for t in thetas]        
        deltas = [0 for x in range(self.numHiddenLayers + 1)]

        for i, row in enumerate(X):
            ## forward propagation
            aIn, zH, aH, zOut, aOut = self._forward(row, thetas)
            
            ## back propagation
            for j in range(self.numHiddenLayers, -1, -1):
                if j == self.numHiddenLayers:
                    d = aOut - Y[i, :].T
                else:
                    d = np.dot(tNoBias[j+1].T, d) * self.sigmoid_deriv(zH[j])

                if j == 0:
                    deltas[j] += np.dot(d[np.newaxis].T, aIn[np.newaxis])
                else:
                    deltas[j] += np.dot(d[np.newaxis].T, aH[j-1][np.newaxis])

        thetaGradients = [((1 / m) * delta) for delta in deltas]
        
        ## apply regularization
        if self.lambda_ != 0:
            for i in range(self.numHiddenLayers + 1):
                #thetaGradients[i][:, 1:] = thetaGradients[i][:, 1:] + (self.lambda_m / m) * tNoBias[i]
                thetaGradients[i][:, 1:] += (self.lambda_m / m) * tNoBias[i]

        return self.vec_thetas(thetaGradients)


    def _forward(self, X, thetas):
        '''
        Computes the forward propagation of the neural network.
            Arguments:
        X       = sample data set
        thetas  = list of arrays of theta weights
        '''

        m = X.shape[0]
        if len(X.shape) == 1:
            bias = np.array(1).reshape(1,)
        else:
            bias = np.ones(m).reshape(m,1)
        
        ## input layer
        aIn = np.hstack((bias, X))
        
        ## hidden layers
        aH = []
        zH = []
        for i in range(0, self.numHiddenLayers):
            t = thetas[i]
            if i == 0:
                z = np.dot(t, aIn.T)
            else:
                z = np.dot(t, a.T)
            a = self.sigmoid(z)
            a = np.hstack((bias, a.T))
            aH.append(a)
            zH.append(z)
        
        ## output layers
        tOut = thetas[len(thetas)-1]
        zOut = np.dot(tOut, aH[len(aH)-1].T)
        aOut = self.sigmoid(zOut)

        return aIn, zH, aH, zOut, aOut
    

    def sigmoid(self, z):
        '''
        Returns the sigmoid of the input array 'z'.
        '''

        return 1 / (1 + np.exp(-z))
    

    def sigmoid_deriv(self, z):
        '''
        Returns the derivative of the sigmoid of the array 'z'.
        '''

        sig = self.sigmoid(z)

        return sig * (1 - sig)
    

    def sum_squre(self, a):
        '''
        Returns the sum of the square of each element in the input array 'a'.
        '''

        return np.sum(a ** 2)
    

    def initialize_weights(self, l_in, l_out):
        '''
        Returns a l_in x l_out array of random floats within 
        +/- self.smallInitValue
        '''

        return np.random.rand(l_out, l_in + 1) * 2 * self.smallInitValue - self.smallInitValue


    def vec_thetas(self, thetas):
        '''
        Returns a "vectorized" 1 x n of the input list of arrays, 'thetas'.
        '''

        return np.concatenate([x.reshape(-1) for x in thetas])
    

    def unvec_thetas(self, thetas):
        '''
        Reconstitutes the input 1 x n "vectorized" array into the original
        list of arrays.
        '''

        thetasReturn = []

        ## input layer theta array
        endIndex = self.hiddenLayerSizes[0] * (self.inputLayerSize + 1)
        thetasReturn.append(thetas[0:endIndex].reshape((self.hiddenLayerSizes[0], self.inputLayerSize + 1)))

        ## hidden layer theta arrays
        for i in range(0, self.numHiddenLayers-1):
            startIndex = endIndex
            endIndex += self.hiddenLayerSizes[i+1] * (self.hiddenLayerSizes[i] + 1)
            thetasReturn.append(thetas[startIndex:endIndex].reshape((self.hiddenLayerSizes[i+1], self.hiddenLayerSizes[i] + 1)))

        ## output layer theta array
        thetasReturn.append(thetas[endIndex:].reshape((self.numClassifyValues, self.hiddenLayerSizes[self.numHiddenLayers-1] + 1)))

        return thetasReturn




if __name__ == '__main__':

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)

    hiddenLayers = [25, 10]

    nn = NeuralNetwork(lambda_=0, maxIter=500, smallInitValue=0.1, hiddenLayerSizes=hiddenLayers, optimizeAlgo='BFGS')

    nn.learn(X_train, y_train)

    score = accuracy_score(y_test, nn.classify(X_test))
    
    print score












