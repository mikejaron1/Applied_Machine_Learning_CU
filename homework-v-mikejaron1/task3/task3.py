from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, \
	Dense, Dropout, Input
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
import scipy.io as sio
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

test = False
seed = 4

image_ind = 20
train_data = sio.loadmat('train_32x32.mat')
test_data = sio.loadmat('test_32x32.mat')

# access to the dict
X_train = train_data['X']
y_train = train_data['y']
X_test = test_data['X']
y_test = test_data['y']
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# import scipy.misc
# scipy.misc.imsave('outfile1.jpg', X_train[:,:,:,0])

batch_size = 32
num_classes = 10
epochs = 10

img_rows, img_cols, channel, total_images = X_train.shape
input_shape = (img_rows, img_cols, channel)

# transpose saves imgae better, found out after saving the matrix as an image
X_train = X_train.transpose(3, 0, 1, 2)
X_test = X_test.transpose(3, 0, 1, 2)
# X_train = X_train.reshape(X_train.shape[3], img_rows, img_cols, channel)
# X_test = X_test.reshape(X_test.shape[3], img_rows, img_cols, channel)

# the labels are 1-10, 0 = 10 and needs to be 0
y_train[y_train == 10] = 0
y_test[y_test == 10] = 0

y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

if test:
	X_train = X_train[:800]
	X_test = X_test[:80]
	y_train = y_train[:800]
	y_test = y_test[:80]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


# try grid searching for the best model
def make_model(optimizer=sgd, kern=3, hidden_size=64):
	cnn = Sequential()
	cnn.add(Conv2D(32, (kern, kern), activation='relu', 
		input_shape=input_shape))
	# cnn.add(Conv2D(64, (kern, kern), activation='relu'))
	cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
	# cnn.add(Dropout(0.25))

	cnn.add(Conv2D(32, (kern, kern), activation='relu'))
	cnn.add(Conv2D(32, (kern, kern), activation='relu'))
	cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

	# cnn.add(Conv2D(256, (kern, kern), activation='relu'))
	# cnn.add(Conv2D(256, (kern, kern), activation='relu'))
	# cnn.add(Conv2D(256, (kern, kern), activation='relu'))
	# cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# cnn.add(Conv2D(512, (kern, kern), activation='relu'))
	# cnn.add(Conv2D(512, (kern, kern), activation='relu'))
	# cnn.add(Conv2D(512, (kern, kern), activation='relu'))
	# cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	cnn.add(Flatten())
	cnn.add(Dense(32, activation='relu'))
	# cnn.add(Dense(32, activation='relu'))
	# cnn.add(Dropout(0.5))
	cnn.add(Dense(num_classes, activation='softmax'))

	cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, 
		metrics=['accuracy'])
	return cnn

# for vanilla model
cnn = KerasClassifier(make_model)

# # grid search
param_grid = {'epochs': [10, 20, 30, 60],
			'hidden_size': [32, 64, 256],
			'conv1': [3, 5, 6, 10, 20],
			# 'conv2': [3, 5, 6, 10],
			'optimizer': ['adam', sgd]}


# grid = GridSearchCV(cnn, param_grid=param_grid, cv=2)
# grid.fit(X_train, y_train)

# best_params = grid.best_params_

# ran the grid search before
best_params = {'epochs': 10, 'hidden_size': 32, 'optimizer': 'adam',
	'conv1': 3}

print('w/o batch')
print(best_params)

cnn = make_model(optimizer=best_params['optimizer'], 
	kern=best_params['conv1'], 
	hidden_size=best_params['hidden_size'])
history_cnn = cnn.fit(X_train, y_train, batch_size=batch_size, 
	epochs=best_params['epochs'], verbose=2, validation_split=.1)
base_score = cnn.evaluate(X_test, y_test, batch_size=batch_size)

# if test:
# 	print(pd.DataFrame(history_cnn.history))
# 	print('base', base_score)
# 	print(best_params)
# 	quit()


def make_batch_model(optimizer=sgd, hidden_size=64):
	# start batch normalization
	model = Sequential()
	model.add(Conv2D(hidden_size, kernel_size=(3, 3),
					activation='relu',
					input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	# model.add(Dense(hidden_size, activation='softmax'))
	# model.add(BatchNormalization())
	# model.add(Activation('softmax'))
	model.add(Dense(num_classes, activation='softmax'))

	# running the fitting
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
		metrics=['accuracy'])
	
	return model

# model = KerasClassifier(make_batch_model)

# grid search
# param_grid = {'epochs': [10, 20, 30],
# 			'hidden_size': [32, 64, 256]}

# grid = GridSearchCV(model, param_grid=param_grid, cv=2)
# grid.fit(X_train, y_train)

# best_params_batch = grid.best_params_

# ran before 
best_params_batch = {'epochs': 20, 'hidden_size': 64}
# print('w/ batch')
# print(best_params_batch)

model = make_batch_model(optimizer='adam', 
	hidden_size=best_params_batch['hidden_size'])

history_batch = model.fit(X_train, y_train, batch_size=batch_size, 
	epochs=best_params_batch['epochs'], verbose=2, validation_split=.1)
batch_score = model.evaluate(X_test, y_test)

# print('base', base_score)
# print('batch', batch_score)

f = open('results_task3.txt', 'w')
f.write('base model:\n')
f.write('best params =' + str(best_params))
f.write("Test lost: {:.3f}\n".format(base_score[0]))
f.write("Test Accuracy: {:.3f}\n".format(base_score[1]))
f.write('With Batch Normalization:\n')
f.write('best params =' + str(best_params_batch))
f.write("Test lost: {:.3f}\n".format(batch_score[0]))
f.write("Test Accuracy: {:.3f}\n".format(batch_score[1]))
f.close()


print pd.DataFrame(history_cnn.history)
print pd.DataFrame(history_batch.history)

print('base', base_score)
print('batch', batch_score)



