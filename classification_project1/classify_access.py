from process_data import load_access
from sklearn.naive_bayes import MultinomialNB


X,Y = load_access()

X_train = X[:90]
Y_train = Y[:90]

X_test = X[90:]
Y_test = Y[90:]

model = MultinomialNB()
model.fit(X_train, Y_train)

result = model.predict(X_test)
difference = result - Y_test

correct = [d for d in difference if d==0]
total_correct = len(correct)
correct_rate = 100.0*total_correct/len(X_test)

print('Correct classification rate: %.2f'%correct_rate)

 