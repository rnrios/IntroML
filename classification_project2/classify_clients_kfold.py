import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import shuffle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import numpy as np


def fit_and_predict(model, X_train, Y_train, nome, k=10):
    scores = cross_val_score(model, X_train, Y_train, cv=10)
    mean = np.mean(scores)

    msg = 'Mean Correct classification rate {}: {:.2f}'.format(nome, mean)
    print(msg)
    return mean

data = pd.read_csv('clients.csv')
#data = shuffle(data)
X = data[['last_visit', 'frequency', 'weeks_subscribed']]

X_df = pd.get_dummies(X)
Y_df = data['situation']

X = X_df.values
Y = Y_df.values

train_len = int(len(X)*.8)
X_train = X[:train_len]
Y_train = Y[:train_len]

X_test = X[train_len:]
Y_test = Y[train_len:]

results = {}
modelOVR = OneVsRestClassifier(LinearSVC(random_state=0, max_iter=1e4))
correctOVR = fit_and_predict(modelOVR, X_train, Y_train, 'OVR')
results[correctOVR] = modelOVR

modelMNB = MultinomialNB()
correctMNB  = fit_and_predict(modelMNB, X_train, Y_train, 'MNB')
results[correctMNB] = modelMNB

modelAB = AdaBoostClassifier()
correctAB = fit_and_predict(modelAB, X_train, Y_train, 'AdaBoost')
results[correctAB] = modelAB

base_rate = max(Counter(Y_test).values())/len(Y_test)
print('Baseline correct classification rate: %.2f' %base_rate)

best_model = results[max(results)]
print('\nBest model: ',best_model)
best_model.fit(X_train, Y_train)

result = best_model.predict(X_test)
correct_cls = (result == Y_test)

total_correct = sum(correct_cls)
correct_rate = total_correct/len(Y_test)
print('\nCorrect classification rate on test dataset: {:.2f}'.format(correct_rate))