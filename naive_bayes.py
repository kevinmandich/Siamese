from __future__ import division

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time, os

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

class NaiveBayes(object):

    def __init__(self, X, className):
        '''
        Initialize the class with the training data 'X' and the name
        of the classification variable, 'className'. Here, 'X' is assumed
        to be a Pandas dataframe object.
        '''

        ## store the className and the set of features
        self.className = className
        self.classes = X.groupby(className).size()
        self.features = set(X.columns) - set([className])

        ## store all possible values for each feature
        self.featureValues = {}
        for feature in self.features:
            self.featureValues[feature] = [ind for ind in X.groupby(feature).size().index]

        ## initialize probability dictionaries
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


    def train(self, X):
        '''
        The training routine which learns the prior, marginal, and
        likelihood probabilities of the NaiveBayes object's training data.
        '''

        self.priors     = self.compute_priors(X)
        self.evidence   = self.compute_evidence(X)
        self.likelihood = self.compute_likelihood(X)

        return None


    def predict(self, xTest, evidenceFlag=False):
        '''
        Returns the class corresponding to the highest posterior
        proability computed for the input sample 'xTest'

        evidenceFlag - when True, includes the P(evidence) as part of the 
                       computation for the posterior probability.
        '''

        if len(self.features) == 0:
            print '\nYou must train the Naive Bayes classifier before using it.\n'
            return None

        predicted = []

        for i in range(len(xTest)):

            X = xTest[i:i+1]            
            maxProb = 0

            for class_ in self.classes.index:

                pPrior = self.priors[class_]
                pEvidence = 1.0
                pLikelihood = 1.0

                for feature in self.features:

                    featureValue = X[feature][int(X.index[0])]
                    if evidenceFlag:
                        pEvidence *= self.evidence[feature][featureValue]

                    try:
                        pLikelihood *= self.likelihood[class_][feature][featureValue]
                    except:
                        print 'Warning: the feature value {} for feature {}'.format(featureValue, feature) \
                             +'does not exist in the training data'
                        pLikelihood = 0
                        break

                if evidenceFlag:
                    pPosterior = pLikelihood*pPrior/pEvidence
                else:
                    pPosterior = pLikelihood*pPrior

                if pPosterior > maxProb:
                    maxProb = pPosterior
                    bestClass = class_

            predicted.append(bestClass)

        return predicted




if __name__ == '__main__':

    data = pd.read_csv('c:\\winpy\\python\\Siamese\\example_data\\sample2.csv')
    print data

    nb = NaiveBayes(data, 'goodBad')
    nb.train(data)

    predicted = nb.predict(data)










