from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.advanced_activations import LeakyReLU

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import pandas as pd

seed = 4

# Get data
data = load_iris()
y = data.target
y = to_categorical(y, num_classes=3)
X = data.data
feauture_names = list(data.target_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
	random_state=seed)

# make optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 4-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=4))
model.add(Dropout(0.5))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=20, batch_size=32)
score = model.evaluate(X_test, y_test, batch_size=32)
print("Test lost: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))

quit()
# try grid searching for the best model
def make_model(optimizer=sgd, hidden_size=64):
	model = Sequential()
	model.add(Dense(hidden_size, activation='relu', input_dim=4))
	model.add(Dropout(0.5))
	model.add(Dense(hidden_size, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(3, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
	              optimizer=optimizer,
	              metrics=['accuracy'])
	return model

clf = KerasClassifier(make_model)

param_grid = {'epochs': [1, 5, 10],
			'hidden_size': [32, 64, 256]}

grid = GridSearchCV(clf, param_grid=param_grid, cv=5)

grid.fit(X_train, y_train)

res = pd.DataFrame(grid.cv_results_)
print(res.pivot_table(index=["param_epochs", "param_hidden_size"],
				values=['mean_train_score', "mean_test_score"]))

print(grid.best_params_)

# for top values
# epoch = 10
# hidden size = 256

# run the model with the new parameters
model = make_model(optimizer=sgd, hidden_size=grid.best_params_['hidden_size'])
model.fit(X_train, y_train,
          epochs=grid.best_params_['epochs'], batch_size=32)
score = model.evaluate(X_test, y_test, batch_size=32)
print("With best params")
print("Test lost: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))


