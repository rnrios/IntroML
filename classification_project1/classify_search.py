import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import shuffle


def fit_and_predict(model, X_train, Y_train,
                    X_test, Y_test, nome):
    model.fit(X_train, Y_train)

    result = model.predict(X_test)
    correct = (result == Y_test)

    total_correct = sum(correct)
    correct_rate = 100.0*total_correct/len(Y_test)

    msg = 'Correct classification rate {0}: {1}'.format(nome, correct_rate)
    print(msg)
    return correct_rate

data = pd.read_csv('search.csv')
data = shuffle(data)
X = data[['home', 'search', 'logged']]

X_df = pd.get_dummies(X)
Y_df = data['bought']

X = X_df.values
Y = Y_df.values

train_len = int(len(X)*.8)
test_len = int(len(X)*.1)

X_train = X[:train_len]
Y_train = Y[:train_len]

X_val = X[train_len:train_len+test_len]
Y_val = Y[train_len:train_len+test_len]

X_test = X[-test_len:]
Y_test = Y[-test_len:]

taxa_base = 100*max(Counter(Y_test).values())/len(Y_test)
print('Baseline correct classification rate on test dataset: %.2f' %taxa_base)

modelMNB = MultinomialNB()
correctMNB  = fit_and_predict(modelMNB, X_train, Y_train,
                    X_val, Y_val, 'MN Bayes')
modelAB = AdaBoostClassifier()
correctAB = fit_and_predict(modelAB, X_train, Y_train,
                    X_val, Y_val, 'AdaBoost')

model = modelAB if correctAB > correctMNB else modelMNB
model.fit(X_train, Y_train)

result = model.predict(X_test)
correct = (result == Y_test)

total_correct = sum(correct)
correct_rate = 100.0*total_correct/len(Y_val)
print('Correct classification rate on test dataset: {0}'.format(correct_rate))