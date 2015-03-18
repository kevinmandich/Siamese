import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import random as rnd
import math
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


if __name__ == '__main__':

    dta = sm.datasets.fair.load_pandas().data

    ## re-cast any affairs with value > 0 as 1
    dta['affair'] = (dta.affairs > 0).astype(int)

    y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + religious + educ + occupation + occupation_husb', dta, return_type="dataframe")

    y = np.ravel(y)

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=0)

    model = LogisticRegression()
    model.fit(xTrain, yTrain)

    scoreTrain = model.score(xTrain, yTrain)
    scoreTest = model.score(xTest, yTest)
    print scoreTrain, scoreTest, '\n'

    coeffs = pd.DataFrame(zip(X.columns, np.transpose(model.coef_.tolist()[0])))

    predicted = model.predict(xTest)
    print predicted, '\n'

    probs = model.predict_proba(xTest)
    print probs, '\n'

    print metrics.accuracy_score(yTest, predicted)
    print metrics.roc_auc_score(yTest, predicted), '\n'

    print metrics.confusion_matrix(yTest, predicted)
    print metrics.classification_report(yTest, predicted), '\n'

    ## ten-fold cross-validation:

    scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
    print 'Scores:\n', scores
    print scores.mean(), '\n'



    ## ************
    ## checking error as a function of training set size
    ## ************

    m = 100
    testSize = int(math.floor(len(X)*0.2))
    
    mValues = []
    trainScores = []
    testScores = []

    ## define testing set
    xTest = X[(len(X)-testSize):]
    yTest = y[(len(X)-testSize):]

    xSample = X[:(len(X)-testSize)]
    ySample = y[:(len(X)-testSize)]

    while m < len(X) - testSize:

        ## define training set
        rows = rnd.sample(xSample.index, m)
        xTrain = xSample.ix[rows]
        yTrain = ySample[rows]

        model.fit(xTrain, yTrain)

        trainScores.append(model.score(xTrain, yTrain))
        testScores.append(model.score(xTest, yTest))

        mValues.append(m)
        m += 100

    plt.plot(mValues, trainScores, color='b', label='Training Error')
    plt.plot(mValues, testScores, color='r', label='Test Error')
    plt.title('Train and Test Errors vs. Size of Training Set')
    plt.xlabel('Size of Training Set, m')
    plt.ylabel('Train and Test Errors')
    plt.legend()

    plt.show()

    ## ************
    ## checking error as a function of the regularization coefficient, lambda
    ## ************


    lambda_ = 0.00001
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.5, random_state=rnd.randint(1,100))

    lambdaValues = []
    trainScores = []
    testScores = []

    while lambda_ < 1:

        model = LogisticRegression(penalty='l2', C=lambda_**(-1))
        model.fit(xTrain, yTrain)

        trainScores.append(1.0 - model.score(xTrain, yTrain))
        testScores.append(1.0 - model.score(xTest, yTest))

        lambdaValues.append(lambda_)
        lambda_ *= 2

    plt.plot(lambdaValues, trainScores, color='b', marker='o', label='Training Error')
    plt.plot(lambdaValues, testScores, color='r', marker='o', label='Test Error')
    plt.title('Train and Test Errors vs. Regularization Parameter')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Train and Test Errors')
    plt.legend()

    plt.show()















