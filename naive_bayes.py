from __future__ import division

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


class NaiveBayes(object):

    def __init__(self, X, className):
        '''
        Initialize the class with the training data 'X' and the name
        of the classification variable, 'className'
        '''

        ## store the className and the set of features
        self.X = X
        self.className = className
        self.classes = X.groupby(className).size()
        self.features = set(X.columns) - set([className])

        ## store all possible values for each feature
        self.featureValues = {}
        for feature in self.features:
            self.featureValues[feature] = [ind for ind in X.groupby(feature).size().index]

        self.evidence = {}
        self.priors = {}
        self.likelihood = {}


    def compute_priors(self, X):
        '''
        Returns a dictionary with classification values as keys and their
        respective prior probability as values.
        '''

        total = len(X)

        return {class_:self.classes[class_]/total for class_ in self.classes.index}


    def compute_evidence(self, X):
        '''
        Returns a dictionary of dictionaries corresponding to the 
        marginal probability, or evidence, of the training set features.
        '''

        evidence = {}
        total = len(X)

        ## compute evidence propabailities:
        for feature in self.features:
            features = X.groupby(feature).size()
            evidence[feature] = {key:features[key]/total for key in features.index}

        return evidence


    def compute_likelihood(self, X):
        '''
        Returns a dictionary of dictionaries of dictionaries corresponding
        to the likelihood probabilities of each feature value.
        '''

        likelihood = {}

        ## compute likelihoods:
        for class_ in self.classes.index:

            likelihood[class_] = {}

            ## separate by class
            classData = X[X[self.className]==class_]
            classTotal = len(classData)

            for feature in self.features:

                likelihood[class_][feature] = {}
                classFeatures = classData.groupby(feature).size()

                ## initialize all feature value likelihoods has having a probability of 0.0
                for index in self.featureValues[feature]:
                    likelihood[class_][feature][index] = 0.0

                ## compute the likelihood probability
                for index in classFeatures.index:
                    likelihood[class_][feature][index] = classFeatures[index]/classTotal

        return likelihood


    def train(self):
        '''
        The training routine which learns the prior, marginal, and
        likelihood probabilities of the NaiveBayes object's training data.
        '''

        self.priors     = self.compute_priors(self.X)
        self.evidence   = self.compute_evidence(self.X)
        self.likelihood = self.compute_likelihood(self.X)

        return None


    def predict(self, xTest):
        '''
        Returns the class corresponding to the highest posterior
        proability computed for the input sample 'xTest'
        '''

        if len(self.features) == 0:
            print '\nYou must train the Naive Bayes classifier before using it.\n'
            return None

        maxProb = 0
        for class_ in self.classes.index:

            pPrior = self.priors[class_]
            pEvidence = 1.0
            pLikelihood = 1.0

            for feature in self.features:
                featureValue = xTest[feature][int(xTest.index)]
                pEvidence *= self.evidence[feature][featureValue]
                pLikelihood *= self.likelihood[class_][feature][featureValue]

            ## note: don't actually need to use pEvidence here
            pPosterior = pLikelihood*pPrior/pEvidence

            if pPosterior > maxProb:
                maxProb = pPosterior
                bestClass = class_

        return bestClass




if __name__ == '__main__':


    data = sm.datasets.fair.load_pandas().data

    ## modify data
    data['affairs'] = (data.affairs > 0).astype(int)

    cd = data[data['affairs']==1]
    cf = cd.groupby('rate_marriage').size()

    nb = NaiveBayes(data, 'affairs')
    nb.train()

    xTest = data[5000:5001]

    predictedClass = nb.predict(xTest)
    print xTest
    print 'Predicted class = {}'.format(predictedClass)
    '''

    data = pd.read_csv('c:\\winpy\\python\\Siamese\\example_data\\fruit.csv')

    nb = NaiveBayes(data, 'fruit')
    nb.train()

    print data[0:1]
    print nb.predict(data[0:1])
    '''











