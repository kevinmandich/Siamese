from __future__ import division

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import random, time, copy

class SupportVectorMachine(object):

    def __init__(self, C, kernel='gaussian', sigma=1.0, tol=1e-3, maxIter=5):
        '''
            Arguments:
        C          = the regularization parameter
        kernel     = the kernel type to use for training
        sigma      = the parameter used for the Gaussian kernel
        tol        = the tolerance value used during the optimization routine
        maxIter  = the maximum number of runs through the optimization routine when
                     there are no further changes to 'alphas' (see self.learn)
        '''

        self.C = C
        self.kernel = kernel
        self.sigma = sigma
        self.tol = tol
        self.maxIter = maxIter


    def linear_kernel(self, x1, x2):
        '''
        Returns the linear kernal of the inputs 'x1' and 'x2'.
        '''

        return np.dot(x1, x2)


    def gaussian_kernel(self, x1, x2, sigma):
        '''
        Returns the Gaussian (or RBF) kernel of the inputs 'x1' and 'x2',
        subject to the parameter 'sigma'.
        '''

        return np.exp( -self.sum_square(x1 - x2) / (2 * sigma**2) )


    def sum_square(self, a):
        '''
        Returns the sum of the square of each element in the input array 'a'.
        '''

        return np.sum(a ** 2)


    def learn(self, X, y):
        '''
        The learning algorighm. Inputs X and y must have the same size
        along the first axis. Returns None.
        '''

        ## input data parameters
        m = X.shape[0]
        n = X.shape[1]

        ## replace 0 with -1 in y
        X = np.array(X)
        y = np.array(y)
        y[y == 0] = -1

        ## optimization parameters
        alphas = np.zeros([m,1])
        E = np.zeros([m, 1])
        b, L, H, eta, iter_ = 0.0, 0.0, 0.0, 0.0, 0.0

        ## pre-compute kernel matrix
        x1 = X[:,0]
        x2 = X[:,1]

        if self.kernel == 'gaussian':

            X2 = x1**2 + x2**2
            K = X2.reshape(len(X2), 1) + X2 - 2*np.dot(X, X.T)
            K = self.gaussian_kernel(1, 0, self.sigma) ** K

        elif self.kernel == 'linear':

            K = self.linear_kernel(x1, x2)

        ## begin training
        while iter_ < self.maxIter:

            alphaChanges = 0
            for i in range(0, m):

                E[i] = b + np.sum(alphas*y*K[:][i].reshape(m, 1)) - y[i]

                if (y[i]*E[i] < -self.tol and alphas[i] < self.C) or (y[i]*E[i] > self.tol and alphas[i] > 0):

                    j = random.randint(0, m-1)
                    while j == i:
                        j = random.randint(0,m-1)

                    E[j] = b + np.sum(alphas*y*K[:][j].reshape(m, 1)) - y[j]


                    alphaIOld = copy.copy(alphas[i])
                    alphaJOld = copy.copy(alphas[j])

                    if y[i] == y[j]:
                        L = max(0, alphas[j] + alphas[i] - self.C)
                        H = min(self.C, alphas[j] + alphas[i])
                    else:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])

                    if L == H:
                        continue

                    eta = 2*K[i][j] - K[i][i] - K[j][j]
                    if eta >= 0:
                        continue

                    alphas[j] = alphas[j] - (y[j]*(E[i] - E[j])) / eta

                    alphas[j] = min(H, alphas[j])
                    alphas[j] = max(L, alphas[j])

                    if abs(alphas[j] - alphaJOld) < self.tol:
                        alphas[j] = alphaJOld
                        continue

                    alphas[i] = alphas[i] + y[i]*y[j]*(alphaJOld - alphas[j])

                    b1 = b - E[i] - y[i] * K[i][j] * (alphas[i] - alphaIOld) \
                                  - y[j] * K[i][j] * (alphas[j] - alphaJOld)

                    b2 = b - E[j] - y[i] * K[i][j] * (alphas[i] - alphaIOld) \
                                  - y[j] * K[j][j] * (alphas[j] - alphaJOld)

                    if self.C > alphas[i] > 0:
                        b = b1
                    elif self.C > alphas[j] > 0:
                        b = b2
                    else:
                        b = 0.5*(b1 + b2)

                    alphaChanges += 1

            if alphaChanges == 0:
                iter_ += 1
            else:
                iter_ = 0

            print '.',

        print '\nFinished training the support vector machine.\n\n'

        ## write the model as attributes
        indexes = np.where(alphas > 0)[0]
        self.X = X[indexes,:]
        self.y = y[indexes]
        self.b = b
        self.alphas = alphas[indexes]
        self.w = (np.matrix((alphas*y).reshape(1,m))*X).T

        ## print alphas (for debugging)
        #print self.X
        print self.b
        print self.w

        return None


    def predict(self, X):
        '''
        Returns a m x 1 array of predictions of (0, 1) values based
        on the input array X.
        '''

        try:
            self.w
        except:
            print 'WARNING: no model parameters exist.'
            print 'The SVM must be trained before it can make predictions.\n'
            return None

        X = np.array(X)
        if X.shape[0] == 1:
            X.reshape(1, 2)

        m = X.shape[0]
        #p = np.zeros(m)
        pad = np.zeros(m)

        if self.kernel = 'linear':
            p = X * model.w + model.b

        else:
            Xtemp1 = sum(X**2, axis=1)
            Xtemp2 = sum(model.X**2, axis=1)
            K = Xtemp1.reshape(len(Xtemp1), 1) + Xtemp2 - 2*np.dot(X, model.X.T)
            K = model.gaussian_function(1, 0, sigma) ** K



    def plot_data(self, data):
        '''
        Plots the positive and negative examples included in 'data'. The last
        column of 'data' must contain the classification values in (0,1).
        '''

        ## split the data into positives and negatives
        positives = data[data[:, 2] == 1]
        negatives = data[data[:, 2] == 0]

        ## plot
        plt.plot( positives[:, 0], positives[:, 1], 'bo' )
        plt.plot( negatives[:, 0], negatives[:, 1], 'ro' )

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('SVM Example')

        return None




if __name__ == '__main__':

    print 'Computing the SVM:\n'

    X = pd.read_csv('c:\\winpy\\python\\Siamese\\example_data\\svm_linear_X_values_2.csv')
    y = pd.read_csv('c:\\winpy\\python\\Siamese\\example_data\\svm_linear_y_values_2.csv')


    svm = SupportVectorMachine(C=1.0, kernel='gaussian', sigma=0.1, tol=1e-3, maxPasses=5)

    #svm.plot_data(np.hstack([X, y]))
    #plt.show()

    #svm.learn(X, y)

    svm.predict(X)











