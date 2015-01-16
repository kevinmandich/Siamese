from __future__ import division
import numpy as np
from scipy.optimize import minimize

import sklearn.datasets as datasets
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

import time, pickle

# change

class NeuralNetwork(object):
    

    def __init__(self, lambda_=0, maxIter=500, smallInitValue=0.1, hiddenLayerSize=25, optimizeAlgo='BFGS'):

        self.lambda_ = lambda_
        self.maxIter = maxIter
        self.smallInitValue = smallInitValue
        self.hiddenLayerSize = hiddenLayerSize
        self.optimizeAlgo = optimizeAlgo


    def fit(self, X, y):
        ## the actual learning routine

        #num_features = X.shape[0]
        inputLayerSize = X.shape[1]
        numClassifyValues = len(set(y))
        
        theta1Init = self.initialize_weights(inputLayerSize, self.hiddenLayerSize)
        theta2Init = self.initialize_weights(self.hiddenLayerSize, numClassifyValues)
        thetasInitVec = self.pack_thetas(theta1Init, theta2Init)
        
        _res = minimize(self.function, thetasInitVec, jac=self.function_prime, method=self.optimizeAlgo, \
                                 args=(inputLayerSize, self.hiddenLayerSize, numClassifyValues, X, y, 0), \
                                 options={'maxiter': self.maxIter} )

        ## store the final weights in the object
        self.t1, self.t2 = self.unpack_thetas(_res.x, inputLayerSize, self.hiddenLayerSize, numClassifyValues)


    def predict(self, X):

        _, _, _, _, hypothesis = self._forward(X, self.t1, self.t2)
        return hypothesis.argmax(0)
    

    def predict_probability(self, X):

        _, _, _, _, hypothesis = self._forward(X, self.t1, self.t2)
        return hypothesis


    def function(self, thetas, inputLayerSize, hiddenLayerSize, numClassifyValues, X, y, lambda_):

        ## compute J(theta)
        t1, t2 = self.unpack_thetas(thetas, inputLayerSize, hiddenLayerSize, numClassifyValues)

        m = X.shape[0]
        Y = np.eye(numClassifyValues)[y]
        
        _, _, _, _, h = self._forward(X, t1, t2)

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
        

    def function_prime(self, thetas, inputLayerSize, hiddenLayerSize, numClassifyValues, X, y, lambda_, speak=0):
        t1, t2 = self.unpack_thetas(thetas, inputLayerSize, hiddenLayerSize, numClassifyValues)
        ## compute dJ(theta)/dtheta

        m = X.shape[0] ## number of training examples
        t1f = t1[:, 1:]
        t2f = t2[:, 1:]

        Y = np.eye(numClassifyValues)[y]
        
        Delta1, Delta2 = 0, 0
        for i, row in enumerate(X):
            a1, z2, a2, z3, a3 = self._forward(row, t1, t2)
            
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


    def _forward(self, X, t1, t2):

        m = X.shape[0]
        #bias = None
        if len(X.shape) == 1:
            bias = np.array(1).reshape(1,)
        else:
            bias = np.ones(m).reshape(m,1)
        
        # Input layer
        a1 = np.hstack((bias, X))
        
        # Hidden Layer
        z2 = np.dot(t1, a1.T)
        a2 = self.sigmoid(z2)
        a2 = np.hstack((bias, a2.T))
        
        # Output layer
        z3 = np.dot(t2, a2.T)
        a3 = self.sigmoid(z3)

        return a1, z2, a2, z3, a3
    

    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))
    

    def sigmoid_deriv(self, z):

        sig = self.sigmoid(z)
        return sig * (1 - sig)
    

    def sumsqr(self, a):

        return np.sum(a ** 2)
    

    def initialize_weights(self, l_in, l_out):

        return np.random.rand(l_out, l_in + 1) * 2 * self.smallInitValue - self.smallInitValue
    

    def pack_thetas(self, t1, t2):

        ## vectorizing thetas
        return np.concatenate((t1.reshape(-1), t2.reshape(-1)))
    

    def unpack_thetas(self, thetas, inputLayerSize, hiddenLayerSize, numClassifyValues):

        ## unvectorizing thetas
        t1_start = 0
        t1_end = hiddenLayerSize * (inputLayerSize + 1)
        t1 = thetas[t1_start:t1_end].reshape((hiddenLayerSize, inputLayerSize + 1))
        t2 = thetas[t1_end:].reshape((numClassifyValues, hiddenLayerSize + 1))
        return t1, t2
    





if __name__ == '__main__':

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)

    nn = NeuralNetwork()
    nn.fit(X_train, y_train)

    score = accuracy_score(y_test, nn.predict(X_test))
    
    print score












