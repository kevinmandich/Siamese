from __future__ import division
import numpy as np
from scipy.optimize import minimize

import sklearn.datasets as datasets
from sklearn import cross_validation
from sklearn.metrics import accuracy_score


class NeuralNetwork(object):
    

    def __init__(self, lambda_=0, maxIter=500, smallInitValue=0.1, hiddenLayerSize=25, optimizeAlgo='BFGS'):
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
        self.hiddenLayerSize = hiddenLayerSize
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
        
        ## initialize the theta arrays
        theta1Init = self.initialize_weights(self.inputLayerSize, self.hiddenLayerSize)
        theta2Init = self.initialize_weights(self.hiddenLayerSize, self.numClassifyValues)

        ## vectorize the theta arrays
        thetasInitVec = self.vec_thetas(theta1Init, theta2Init)
        
        ## run the optimization routine
        result = minimize(self.cost_function, thetasInitVec, jac=self.cost_function_deriv, method=self.optimizeAlgo, \
                        args=(X, y), options={'maxiter': self.maxIter} )

        ## store the final weights in the object
        self.tIn, self.tOut = self.unvec_thetas(result.x)


    def classify(self, X):
        '''
        Given an input array X, return an array containing the classification 
        values of each of the samples contained within X.
        '''

        _, _, _, _, hypothesis = self.forward_prop(X, self.tIn, self.tOut)
        return hypothesis.argmax(0)
    

    def predict_probability(self, X):
        '''
        Given an input array X, returns an array containing the classification
        probability for each of the possible classification values for each of
        the samples contained within X.
        '''

        _, _, _, _, hypothesis = self.forward_prop(X, self.tIn, self.tOut)
        return hypothesis


    def cost_function(self, thetas, X, y):
        '''
        Computes the cost function J(theta).
            Arguments:
        thetas  = the vectorized array of weights
        X       = the input array of samples
        y       = the input array of classifications
        '''

        tIn, tOut = self.unvec_thetas(thetas)

        m = X.shape[0]
        Y = np.eye(self.numClassifyValues)[y]
        
        _, _, _, _, h = self.forward_prop(X, tIn, tOut)

        ## compute J(theta)
        cost = (-Y * np.log(h).T) - ((1 - Y) * np.log(1 - h).T)
        J = np.sum(cost) / m
        
        ## apply regularization
        if self.lambda_ != 0:
            t1f = tIn[:, 1:]
            t2f = tOut[:, 1:]
            reg = (self.lambda_ / (2 * m)) * (self.sum_square(t1f) + self.sum_square(t2f))
            J = J + reg
        return J
        

    def cost_function_deriv(self, thetas, X, y):
        tIn, tOut = self.unvec_thetas(thetas)
        ## compute dJ(theta)/dtheta

        m = X.shape[0] ## number of training examples
        t1NoBias = tIn[:, 1:]
        t2NoBias = tOut[:, 1:]

        Y = np.eye(self.numClassifyValues)[y]
        
        deltaIn, deltaH = 0, 0
        for i, row in enumerate(X):
            ## forward propagation
            aIn, zH, aH, zOut, aOut = self.forward_prop(row, tIn, tOut)
            
            ## back propagation
            d3 = aOut - Y[i, :].T
            d2 = np.dot(t2NoBias.T, d3) * self.sigmoid_deriv(zH)

            deltaH += np.dot(d3[np.newaxis].T, aH[np.newaxis])
            deltaIn += np.dot(d2[np.newaxis].T, aIn[np.newaxis])
            
        thetaInGrad = (1 / m) * deltaIn
        thetaHGrad = (1 / m) * deltaH
        
        ## apply regularization
        if self.lambda_ != 0:
            thetaInGrad[:, 1:] = thetaInGrad[:, 1:] + (self.lambda_ / m) * t1NoBias
            thetaHGrad[:, 1:] = thetaHGrad[:, 1:] + (self.lambda_ / m) * t2NoBias
        
        return self.vec_thetas(thetaInGrad, thetaHGrad)


    def forward_prop(self, X, tIn, tOut):
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
        
        # Hidden Layer
        zH = np.dot(tIn, aIn.T)
        aH = self.sigmoid(zH)
        aH = np.hstack((bias, aH.T))
        
        # Output layer
        zOut = np.dot(tOut, aH.T)
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
    

    def sum_square(self, a):
        '''
        Returns the sum of the square of each element in the input array 'a'.
        '''

        return np.sum(a ** 2)
    

    def initialize_weights(self, indexIn, indexOut):
        '''
        Returns a l_in x l_out array of random floats within 
        +/- self.smallInitValue
        '''

        return np.random.rand(indexOut, indexIn + 1) * 2 * self.smallInitValue - self.smallInitValue
    

    def vec_thetas(self, tIn, tOut):
        '''
        Returns a "vectorized" 1 x n of the input list of arrays, 'thetas'.
        '''

        return np.concatenate((tIn.reshape(-1), tOut.reshape(-1)))
    

    def unvec_thetas(self, thetas):
        '''
        Reconstitutes the input 1 x n "vectorized" array into the original
        list of arrays.
        '''

        startIndex = 0
        endIndex = self.hiddenLayerSize * (self.inputLayerSize + 1)
        tIn = thetas[startIndex:endIndex].reshape((self.hiddenLayerSize, self.inputLayerSize + 1))
        tOut = thetas[endIndex:].reshape((self.numClassifyValues, self.hiddenLayerSize + 1))

        return tIn, tOut
    





if __name__ == '__main__':

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)

    nn = NeuralNetwork()
    nn.learn(X_train, y_train)

    score = accuracy_score(y_test, nn.classify(X_test))
    
    print score












