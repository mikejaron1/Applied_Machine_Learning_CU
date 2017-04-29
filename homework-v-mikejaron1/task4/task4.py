from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
import pandas as pd

test = False
seed = 4
# num_classes = 37

# don't need since index does not match up
# labels_list = open('./annotations/list.txt').readlines()
# y = [i.split()[1] for i in labels_list[6:]]

# get class labels from image names
name_list = [i[:i.rfind('_')] for i in os.listdir('./images/') if 'jpg' in i]
img_list = [i for i in os.listdir('./images/') if '.jpg' in i]
print 'class names', len(name_list), len(set(name_list))
print 'image names', len(img_list), len(set(img_list))

if test:
	# get random indexs and grab those spots
	# random_idx = np.random.random_integers(0, len(img_list), 10)
	# random index didnt work since ther are so many i manually coded to get
	# 2 different classes
	img_list = np.array(img_list)[[0, 1, 2, 3, 4, 5, 500, 501, 502, 503, 504,
	 								505]]
	name_list = np.array(name_list)[[0, 1, 2, 3, 4, 5, 500, 501, 502, 503, 504, 
									505]]

model = VGG16(weights='imagenet', include_top=False)

images = [image.load_img('./images/' + i, target_size=(224, 224)) 
			for i in img_list]
X = np.array([image.img_to_array(i) for i in images])

# X = np.expand_dims(X, axis=0)
X = preprocess_input(X)

features = model.predict(X)
print features.shape

# not sure if necessary
y = pd.factorize(name_list)[0]
print y.shape
print X.shape

features_ = features.reshape(features.shape[0], -1)
print features_.shape

X_train, X_test, y_train, y_test = train_test_split(features_, y, 
	test_size=0.20, 
	stratify=y,
	random_state=seed)

print('starting LR')
lr = LogisticRegression().fit(X_train, y_train)
print lr.score(X_train, y_train)
print lr.score(X_test, y_test)


