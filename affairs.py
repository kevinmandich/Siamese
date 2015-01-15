import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

dta = sm.datasets.fair.load_pandas().data

dta['affair'] = (dta.affairs > 0).astype(int)

dta.groupby('affair').mean()

'''
pd.crosstab(dta['rate_marriage'], dta['affair'].astype(bool)).plot(kind='bar')
plt.title('Marriage Rating Distribution by Affair Status')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')

affair_yrs_married = pd.crosstab(dta.yrs_married, dta.affair.astype(bool))
affair_yrs_married.div(affair_yrs_married.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Affair Percentage by Years Married')
plt.xlabel('Years Married')
plt.ylabel('Percentage')
plt.show()
'''

y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + religious + educ + C(occupation) + C(occupation_husb)', dta, return_type="dataframe")

X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
                        'C(occupation)[T.3.0]':'occ_3',
                        'C(occupation)[T.4.0]':'occ_4',
                        'C(occupation)[T.5.0]':'occ_5',
                        'C(occupation)[T.6.0]':'occ_6',
                        'C(occupation_husb)[T.2.0]':'occ_husb_2',
                        'C(occupation_husb)[T.3.0]':'occ_husb_3',
                        'C(occupation_husb)[T.4.0]':'occ_husb_4',
                        'C(occupation_husb)[T.5.0]':'occ_husb_5',
                        'C(occupation_husb)[T.6.0]':'occ_husb_6'})

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

# ten-fold cross-validation:

scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print scores
print scores.mean(), '\n'














