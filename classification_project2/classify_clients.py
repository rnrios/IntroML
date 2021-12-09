import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import shuffle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


def fit_and_predict(model, X_train, Y_train,
                    X_test, Y_test, nome):
    model.fit(X_train, Y_train)

    result = model.predict(X_test)
    correct = (result == Y_test)

    total_correct = sum(correct)
    correct_rate = 100.0*total_correct/len(Y_test)

    msg = 'Correct classification rate {}: {:.2f}'.format(nome, correct_rate)
    print(msg)
    return correct_rate

data = pd.read_csv('clients.csv')
#data = shuffle(data)
X = data[['last_visit', 'frequency', 'weeks_subscribed']]

X_df = pd.get_dummies(X)
Y_df = data['situation']

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

modelo = OneVsRestClassifier(LinearSVC(random_state=0))
scores = cross_val_score(X_train, Y_train, cv=3)
print(scores)





results = {}
modelOVR = OneVsRestClassifier(LinearSVC(random_state=0, max_iter=1e5))
correctOVR = fit_and_predict(modelOVR, X_train, Y_train,
                    X_val, Y_val
                    , 'One Vs Rest')
results[correctOVR] = modelOVR

modelMNB = MultinomialNB()
correctMNB  = fit_and_predict(modelMNB, X_train, Y_train,
                    X_val, Y_val
                    , 'MN Bayes')
results[correctMNB] = modelMNB

modelAB = AdaBoostClassifier()
correctAB = fit_and_predict(modelAB, X_train, Y_train,
                    X_val, Y_val
                    , 'AdaBoost')
results[correctAB] = modelAB

base_rate = 100*max(Counter(Y_test).values())/len(Y_test)
print('Baseline correct classification rate: %.2f' %base_rate)

best_model = results[max(results)]
best_model.fit(X_train, Y_train)

result = best_model.predict(X_test)
correct_cls = (result == Y_test)

total_correct = sum(correct_cls)
correct_rate = 100.0*total_correct/len(Y_test)
print('Correct classification rate on test dataset: {:.2f}'.format(correct_rate))