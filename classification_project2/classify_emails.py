#!-*- coding:utf8 -*-

import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier


def count_word_frequencies(text, word_dict):
    vector = [0]*len(word_dict)
    for word in text:
        if word in word_dict:
            position = word_dict[word]
            vector[position] += 1
    return vector 


def fit_and_predict(model, X_train, Y_train, nome, k=10):
    scores = cross_val_score(model, X_train, Y_train, cv=10)
    mean = np.mean(scores)

    msg = 'Mean Correct classification rate {}: {:.2f}'.format(nome, mean)
    print(msg)
    return mean


data = pd.read_csv('emails.csv')
data_only_text = data['email']
data_only_text = data_only_text.str.lower().str.split(' ')
word_set = set()
for list_of_words in data_only_text:
    word_set.update(list_of_words)

len_dict = len(word_set)
word_tuples = zip(word_set, range(len_dict))
word_dict = {word:index for word, index in word_tuples}



list_of_text_frequencies = [count_word_frequencies(text, word_dict)
                            for text in data_only_text]

X = np.array(list_of_text_frequencies)
Y = np.array(data['classificacao'].tolist())

train_len = int(.8*len(Y))

X_train = X[:train_len]
Y_train = Y[:train_len]

X_test = X[train_len:]
Y_test = Y[train_len:]


results = {}
modelOVR = OneVsRestClassifier(LinearSVC(random_state=0, max_iter=1e4))
correctOVR = fit_and_predict(modelOVR, X_train, Y_train, 'OVR')
results[correctOVR] = modelOVR

modelOVO = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=1e4))
correctOVO = fit_and_predict(modelOVO, X_train, Y_train, 'OVO')
results[correctOVO] = modelOVO

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