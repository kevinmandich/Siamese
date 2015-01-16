from __future__ import division
import numpy as np
from scipy.optimize import minimize

import sklearn.datasets as datasets
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

import time, pickle

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

        _res = minimize(self.function, thetasInitVec, jac=self.function_deriv, method=self.optimizeAlgo, \
                                 args=(inputLayerSize, self.hiddenLayerSize, numClassifyValues, X, y, 0), \
                                 options={'maxiter': self.maxIter} )

        ## store the final weights in the object
        self.thetas = self.unpack_thetas(_res.x, inputLayerSize, numClassifyValues)


    def predict(self, X):

        _, _, _, _, hypothesis = self._forward(X, self.thetas)
        return hypothesis.argmax(0)
    

    def predict_probability(self, X):

        _, _, _, _, hypothesis = self._forward(X, self.thetas)
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
        

    def function_deriv(self, thetas, inputLayerSize, hiddenLayerSize, numClassifyValues, X, y, lambda_, speak=1):
        thetas = self.unpack_thetas(thetas, inputLayerSize, numClassifyValues)
        ## compute dJ(theta)/dtheta
        
        m = X.shape[0] ## number of training examples
        tf = []
        for t in thetas:
            tf.append(t[:, 1:])
        Y = np.eye(numClassifyValues)[y]
        
        deltas = [0 for x in range(self.numHiddenLayers + 1)]

        for i, row in enumerate(X):

            aIn, zH, aH, zOut, aOut = self._forward(row, thetas)
            
            ## back-propagation
            for j in range(self.numHiddenLayers, -1, -1):
                if j == self.numHiddenLayers:
                    d = aOut - Y[i, :].T
                else:
                    d = np.dot(tf[j+1].T, d) * self.sigmoid_deriv(zH[j])

                if j == 0:
                    deltas[j] += np.dot(d[np.newaxis].T, aIn[np.newaxis])
                else:
                    deltas[j] += np.dot(d[np.newaxis].T, aH[j-1][np.newaxis])

        thetaGradients = []
        for delta in deltas:
            thetaGradients.append((1 / m) * delta)
        
        ## apply regularization
        if lambda_ != 0:
            for i in range(self.numHiddenLayers + 1):
                thetaGradients[i][:, 1:] = thetaGradients[i][:, 1:] + (lambda_m / m) * tf[i]

        return self.pack_thetas(thetaGradients)


    def _forward(self, X, thetas):
        '''
        Computes the forward propagation of the neural network.

            Arguments:
        X       = sample data set
        thetas  = list of arrays of theta weights
        '''

        m = X.shape[0]
        #bias = None
        if len(X.shape) == 1:
            bias = np.array(1).reshape(1,)
        else:
            bias = np.ones(m).reshape(m,1)
        
        # Input layer
        aIn = np.hstack((bias, X))
        
        # Hidden Layers
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
        
        # Output layer
        tOut = thetas[len(thetas)-1]
        zOut = np.dot(tOut, aH[len(aH)-1].T)
        aOut = self.sigmoid(zOut)

        ########## do we ever use z3 outside of this function? check the original code.

        return aIn, zH, aH, zOut, aOut
    

    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))
    

    def sigmoid_deriv(self, z):

        sig = self.sigmoid(z)
        return sig * (1 - sig)
    

    def sumsqr(self, a):

        return np.sum(a ** 2)
    

    def initialize_weights(self, l_in, l_out):

        return np.random.rand(l_out, l_in + 1) * 2 * self.smallInitValue - self.smallInitValue


    def pack_thetas(self, thetas):
        '''
        Vectorizing the theta arrays

        thetas - the list of array objects to be vectorized
        '''

        return np.concatenate([x.reshape(-1) for x in thetas])
    

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

    score = accuracy_score(y_test, nn.predict(X_test))
    
    print score

    #nn.fit(X_train, y_train)

    '''
    fileName = 'c:\\test\\thetas.pkl'
    f = open(fileName, 'r')
    allToGrab = pickle.load(f)
    f.close()

    X_train = allToGrab[0]
    X_test = allToGrab[1]
    y_train = allToGrab[2]
    y_test = allToGrab[3]
    thetas = [allToGrab[4], allToGrab[5]]
    nn.thetas = thetas
    thetas = nn.pack_thetas(thetas)

    #hypo = nn.predict_probability(X_test)
    #print 'hypo = {}'.format(hypo)


    print thetas.shape

    inputLayerSize = X_test.shape[1]
    numClassifyValues = len(set(y_test))

    J = nn.function(thetas, inputLayerSize, 5, numClassifyValues, X_test, y_test, 0)
    J_t  = nn.function_deriv(thetas, inputLayerSize, 5, numClassifyValues, X_test, y_test, 0, 0)

    thetas2 = nn.unpack_thetas(J_t, inputLayerSize, numClassifyValues)

    print '\nJ = {}\n'.format(J)
    print thetas2[0].reshape(-1)
    print thetas2[1].reshape(-1)


    #nn.thetas = thetas

    '''













