from __future__ import division

import statsmodels.api as sm
import numpy as np
import pandas as pd
import time


class NaiveBayes(object):

    def __init__(self):

        ## initialize all object attributes
        self.features = set()
        self.className = ''

        self.evidence = {}
        self.priors = {}
        self.likelihood = {}


    def compute_priors(self, X):

        ## compute class priors:
        total = len(X)

        return {class_:self.classes[class_]/total for class_ in self.classes.index}


    def compute_evidence(self, X):

        evidence = {}
        total = len(X)

        ## compute evidence propabailities:
        for feature in X.columns:
            if feature != self.className:
                features = X.groupby(feature).size()
                evidence[feature] = {key:features[key]/total for key in features.index}

        return evidence


    def compute_likelihood(self, X):

        likelihood = {}

        ## compute likelihoods:
        for class_ in self.classes.index:

            likelihood[class_] = {}

            ## separate by class
            classData = X[X[self.className]==class_]
            classTotal = len(classData)

            for feature in X.columns:

                likelihood[class_][feature] = {}

                if feature != self.className:
                    classFeatures = classData.groupby(feature).size()

                    for index in classFeatures.index:
                        likelihood[class_][feature][index] = classFeatures[index]/classTotal

        return likelihood


    def train(self, X, className):

        ## store the className and the set of features
        self.className = className
        self.classes = X.groupby(className).size()
        self.features = set(X.columns) - set([className])

        self.priors = self.compute_priors(X)
        self.evidence = self.compute_evidence(X)
        self.likelihood = self.compute_likelihood(X)

        return None


    def predict(self, xTest):

        if len(self.features) == 0:
            print '\nYou must train the Naive Bayes classifier before using it.\n'
            return None

        



if __name__ == '__main__':

    data = sm.datasets.fair.load_pandas().data

    ## modify data
    data['affairs'] = (data.affairs > 0).astype(int)

    cd = data[data['affairs']==1]
    cf = cd.groupby('rate_marriage').size()

    nb = NaiveBayes()
    nb.train(data, 'affairs')

    dataTest = data[0:1]







