from __future__ import division
import numpy as np
from scipy.optimize import minimize

import sklearn.datasets as datasets
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

import time

# change

class NeuralNetwork(object):
    

    def __init__(self, lambda_=0, maxIter=500, smallInitValue=0.1, hiddenLayerSizes=[25], optimizeAlgo='BFGS'):
        '''
            Arguments:
        lambda_           = regularization parameter
        maxIter           = maximum # of iterations for optimization algorithm to compute
        smallInitValue    = parameter by which to normalize the random weights initialization
        hiddenLayerSizes  = a list containing the number of hidden units in each hidden layer
        optimizeAlgo      = the optimization algorithm to use (see scipy.optimize.minimize for options)
        '''

        self.lambda_ = lambda_
        self.maxIter = maxIter
        self.smallInitValue = smallInitValue
        self.numHiddenLayers = len(hiddenLayers)
        self.hiddenLayerSizes = hiddenLayers
        self.optimizeAlgo = optimizeAlgo

        # old stuff - for comparison testing
        self.hiddenLayerSize = 25


    def fit(self, X, y):
        '''
        The learning routine

            Arguments:
        X - 2-dimensional training set array
        y - 1-dimensional ______ array
        '''

        inputLayerSize = X.shape[1]
        numClassifyValues = len(set(y))

        thetasInit = []
        thetasInit.append(self.initialize_weights(inputLayerSize, self.hiddenLayerSizes[0]))

        for i in range(0, self.numHiddenLayers-1):
            thetasInit.append(self.initialize_weights(self.hiddenLayerSizes[i], self.hiddenLayerSizes[i+1]))

        thetasInit.append(self.initialize_weights(self.hiddenLayerSizes[self.numHiddenLayers-1], numClassifyValues))

        thetasInitVec = self.pack_thetas(thetasInit)

        _res = minimize(self.function, thetasInitVec, jac=self.function_prime, method=self.optimizeAlgo, \
                                 args=(inputLayerSize, self.hiddenLayerSize, numClassifyValues, X, y, 0), \
                                 options={'maxiter': self.maxIter} )

        ## store the final weights in the object
        self.thetas = self.unpack_thetas(_res.x, inputLayerSize, numClassifyValues)


    def predict(self, X):

        _, _, _, _, hypothesis = self._forward(X, self.t1, self.t2)
        return hypothesis.argmax(0)
    

    def predict_probability(self, X):

        _, _, _, _, hypothesis = self._forward(X, self.t1, self.t2)
        return hypothesis


    def function(self, thetas, inputLayerSize, hiddenLayerSize, numClassifyValues, X, y, lambda_):

        ## compute J(theta)
        thetas = self.unpack_thetas(thetas, inputLayerSize, numClassifyValues)

        m = X.shape[0]
        Y = np.eye(numClassifyValues)[y]
        
        _, _, _, _, h = self._forward(X, thetas)

        costPositive = -Y * np.log(h).T
        costNegative = (1 - Y) * np.log(1 - h).T
        cost = costPositive - costNegative
        J = np.sum(cost) / m
        
        ## apply regularization
        if lambda_ != 0:
            t1f = t1[:, 1:]
            t2f = t2[:, 1:]
            reg = (self.lambda_ / (2 * m)) * (self.sumsqr(t1f) + self.sumsqr(t2f))
            J = J + reg
        return J
        

    def function_prime(self, thetas, inputLayerSize, hiddenLayerSize, numClassifyValues, X, y, lambda_):
        thetas = self.unpack_thetas(thetas, inputLayerSize, numClassifyValues)
        ## compute dJ(theta)/dtheta
        
        m = X.shape[0] ## number of training examples
        t1f = t1[:, 1:]
        t2f = t2[:, 1:]
        Y = np.eye(numClassifyValues)[y]
        
        Delta1, Delta2 = 0, 0
        for i, row in enumerate(X):
            a1, z2, a2, z3, a3 = self._forward(row, thetas)
            
            ## back-propagation
            d3 = a3 - Y[i, :].T
            d2 = np.dot(t2f.T, d3) * self.sigmoid_deriv(z2)           
            Delta2 += np.dot(d3[np.newaxis].T, a2[np.newaxis])
            Delta1 += np.dot(d2[np.newaxis].T, a1[np.newaxis])
            
        Theta1_grad = (1 / m) * Delta1
        Theta2_grad = (1 / m) * Delta2
        
        ## apply regularization
        if lambda_ != 0:
            Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * t1f
            Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * t2f
        
        return self.pack_thetas(Theta1_grad, Theta2_grad)


    def _forward(self, X, thetas):

        m = X.shape[0]
        #bias = None
        if len(X.shape) == 1:
            bias = np.array(1).reshape(1,)
        else:
            bias = np.ones(m).reshape(m,1)
        
        # Input layer
        a = np.hstack((bias, X))
        
        aH = []
        zH = []
        # Hidden Layers
        for i in range(0, self.numHiddenLayers):
        t = thetas[i]
        z = np.dot(t, a.T)
        a = self.sigmoid(z)
        a = np.hstack((bias, a.T))
        aH.append(a)
        zH.append(z)
        
        # Output layer
        tOut = thetas[len(thetas)-1]
        zOut = np.dot(tOut, aH[len(aH)-1].T)
        aOut = self.sigmoid(zOut)

        return a1, zH, aH, zOut, aOut
    

    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))
    

    def sigmoid_deriv(self, z):

        sig = self.sigmoid(z)
        return sig * (1 - sig)
    

    def sumsqr(self, a):

        return np.sum(a ** 2)
    

    def initialize_weights(self, l_in, l_out):

        return np.random.rand(l_out, l_in + 1) * 2 * self.smallInitValue - self.smallInitValue
    
    '''
    def pack_thetas(self, t1, t2):

        return np.concatenate((t1.reshape(-1), t2.reshape(-1)))
    '''

    def pack_thetas(self, thetas):
        '''
        Vectorizing the theta arrays

        thetas - the list of array objects to be vectorized
        '''

        return np.concatenate([x.reshape(-1) for x in thetas])
    
    '''
    def unpack_thetas_old(self, thetas, inputLayerSize, hiddenLayerSize, numClassifyValues):

        ## unvectorizing thetas
        t1_start = 0
        t1_end = hiddenLayerSize * (inputLayerSize + 1)
        t1 = thetas[t1_start:t1_end].reshape((hiddenLayerSize, inputLayerSize + 1))
        t2 = thetas[t1_end:].reshape((numClassifyValues, hiddenLayerSize + 1))
        return t1, t2
    '''

    def unpack_thetas(self, thetas, inputLayerSize, numClassifyValues):

        ## unvectorizing thetas

        thetasReturn = []

        tEnd = self.hiddenLayerSizes[0] * (inputLayerSize + 1) # done
        meep = thetas[0:tEnd].reshape((self.hiddenLayerSizes[0], inputLayerSize + 1))
        thetasReturn.append(meep)

        for i in range(0, self.numHiddenLayers-1):
            tStart = tEnd
            tEnd += self.hiddenLayerSizes[i+1] * (self.hiddenLayerSizes[i] + 1)
            meep = thetas[tStart:tEnd].reshape((self.hiddenLayerSizes[i+1], self.hiddenLayerSizes[i] + 1))
            thetasReturn.append(meep)

        meep = thetas[tEnd:].reshape((numClassifyValues, self.hiddenLayerSizes[self.numHiddenLayers-1] + 1))
        thetasReturn.append(meep)

        return thetasReturn




if __name__ == '__main__':

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)

    hiddenLayers = [25,10]

    nn = NeuralNetwork(lambda_=0, maxIter=500, smallInitValue=0.1, hiddenLayerSizes=hiddenLayers, optimizeAlgo='BFGS')

    nn.fit(X_train, y_train)










