from __future__ import division

import numpy as np
import pandas as pd
import random, time

class SupportVectorMachine(object):

    def __init__(self, C, kernel='gaussian', sigma=1.0, tol=1e-3, maxPasses=5):

        self.C = C
        self.kernel = kernel
        self.sigma = sigma
        self.tol = tol
        self.maxPasses = maxPasses


    def linear_kernel(self, x1, x2):

        return np.dot(x1, x2)


    def gaussian_kernel(self, x1, x2, sigma):

        return np.exp( -self.sum_square(x1 - x2) / (2 * sigma**2) )


    def sum_square(self, a):
        '''
        Returns the sum of the square of each element in the input array 'a'.
        '''

        return np.sum(a ** 2)


    def learn(self, X, y):

        ## input data parameters
        m = X.shape[0]
        n = X.shape[1]

        ## replace 0 with -1 in y
        print y.shape
        y = np.array(y)
        y[y == 0] = -1

        ## optimization parameters
        alphas = np.zeros([m,1])
        E = np.zeros([m, 1])
        b, L, H, eta, passes = 0, 0, 0, 0, 0

        ## pre-compute kernel matrix
        x1 = np.array(X[X.columns[0]])
        x2 = np.array(X[X.columns[1]])

        if self.kernel == 'gaussian':

            X2 = x1**2 + x2**2
            K = X2.reshape(len(X2), 1) + X2 - 2*np.dot(X, X.T)
            K = self.gaussian_kernel(1, 0, self.sigma) ** K

        elif self.kernel == 'linear':

            K = self.linear_kernel(x1, x2)

        ## begin training
        while passes < self.maxPasses:

            numAlphaChanges = 0
            for i in range(0, m):

                E[i] = b + np.sum(alphas*y*K[:][i].reshape(m, 1)) - y[i]

                if (y[i]*E[i] < -self.tol and alphas[i] < self.C) or (y[i]*E[i] > self.tol and alphas[i] > 0):

                    j = random.randint(0, m-1)
                    while j == i:
                        j = random.randint(0,m-1)

                    E[j] = b + np.sum(alphas*y*K[:][j].reshape(m, 1)) - y[j]

                    alphaIOld = alphas[i]
                    alphaJOld = alphas[j]

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

                    alphas[j] = min(alphas[j], H)
                    alphas[j] = max(alphas[j], L)

                    if abs(alphas[j] - alphaJOld) < self.tol:
                        alphas[j] = alphaJOld
                        continue

                    alphas[i] = alphas[i] + y[i]*y[j]*(alphaJOld - alphas[j])

                    b1 = b - E[i] - y[i] * K[i][j] * (alphas[i] - alphaIOld) \
                                  - y[j] * K[i][j] * (alphas[j] - alphaJOld)

                    b1 = b - E[j] - y[i] * K[i][j] * (alphas[i] - alphaIOld) \
                                  - y[j] * K[j][j] * (alphas[j] - alphaJOld)

                    if self.C > alphas[i] > 0:
                        b = b1
                    elif self.C > alphas[j] > 0:
                        b = b2
                    else:
                        b = 0.5*(b1 + b2)

            if numAlphaChanges == 0:
                passes += 1
            else:
                passes = 0

            print '.'

        print '\nFinished training the support vector machine.\n\n'

        ## write the model as attributes
        indexes = np.where(alphas > 0)[0]
        print indexes
        self.X = X[indexes,:]
        self.y = y[indexes]
        self.b = b
        self.alphas = alphas[indexes]
        self.w = (((alphas*y).T)*X).T

        print self.w




if __name__ == '__main__':

    print 'Computing the SVM:\n'

    X = pd.read_csv('c:\\winpy\\python\\Siamese\\example_data\\svm_linear_X_values.csv')
    y = pd.read_csv('c:\\winpy\\python\\Siamese\\example_data\\svm_linear_y_values.csv')

    #values = np.array([[1,0],[2,4],[1,-1]])
    #X = pd.DataFrame(values, columns=['x1','x2'])

    #x1 = np.array(X[X.columns[0]])
    #x2 = np.array(X[X.columns[1]])

    svm = SupportVectorMachine(C=1.0, kernel='gaussian', sigma=0.1, tol=1e-3, maxPasses=5)

    svm.learn(X, y)











