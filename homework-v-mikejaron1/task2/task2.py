from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.grid_search import GridSearchCV
import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

seed = 4

test = False

# Get data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# for testing
if test:
	X_train = X_train[:1000]
	X_test = X_test[:100]
	y_train = y_train[:1000]
	y_test = y_test[:100]


print(X_train.shape, 'train data')
print(X_test.shape, 'test data')

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# make optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


# try grid searching for the best model
def make_model(optimizer=sgd, hidden_size=64, drop_out=False):
	model = Sequential()
	model.add(Dense(hidden_size, activation='relu', input_dim=784))
	if drop_out:
		model.add(Dropout(0.5))
	model.add(Dense(hidden_size, activation='relu'))
	if drop_out:
		model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
	              optimizer=optimizer,
	              metrics=['accuracy'])
	return model

# for vanilla model
# clf = KerasClassifier(make_model)

# # grid search
param_grid = {'epochs': [10, 20, 30],
			'hidden_size': [32, 64, 256]}

# grid = GridSearchCV(clf, param_grid=param_grid, cv=5)

# grid.fit(X_train, y_train)

# res = pd.DataFrame(grid.cv_results_)
# print('vanilla')
# print(res.pivot_table(index=["param_epochs", "param_hidden_size"], 
					# values=['mean_train_score', "mean_test_score"]))
# print('vanilla')
# best_params1 = grid.best_params_


# with dropout
clf = KerasClassifier(make_model, drop_out=True)

grid = GridSearchCV(clf, param_grid=param_grid, cv=5)

grid.fit(X_train, y_train)

# res = pd.DataFrame(grid.cv_results_)
# print('vanilla')
# print(res.pivot_table(index=["param_epochs", "param_hidden_size"],
				# values=['mean_train_score', "mean_test_score"]))

print('with drop out')
best_params = grid.best_params_


# Vanilla
model = make_model(optimizer=sgd, hidden_size=best_params['hidden_size'])
history_callback1 = model.fit(X_train, y_train, verbose=2,
          epochs=best_params['epochs'], batch_size=32, validation_split=.1)
score1 = model.evaluate(X_test, y_test, batch_size=32)

# with drop out
model = make_model(optimizer=sgd, hidden_size=best_params['hidden_size'], drop_out=True)
history_callback = model.fit(X_train, y_train, verbose=2,
          epochs=best_params['epochs'], batch_size=32, validation_split=.1)
score = model.evaluate(X_test, y_test, batch_size=32)

print()
print('vanilla')
print("Test lost: {:.3f}".format(score1[0]))
print("Test Accuracy: {:.3f}".format(score1[1]))
print()
print('with drop out')
print("Test lost: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))

# write to file
f = open('results.txt', 'w')
f.write('Vanilla:\n')
f.write("Test lost: {:.3f}\n".format(score1[0]))
f.write("Test Accuracy: {:.3f}\n".format(score1[1]))
f.write('With Drop Out:\n')
f.write("Test lost: {:.3f}\n".format(score[0]))
f.write("Test Accuracy: {:.3f}\n".format(score1[1]))

# make plots
df = pd.DataFrame(history_callback1.history)
df.to_csv('vanilla.csv')


df = pd.DataFrame(history_callback.history)
df.to_csv('dropOut.csv')





