## Michael Jaron, mj2776

# 2.2
from __future__ import absolute_import, division, print_function

def div(x):
	return x/8


import numpy as np

def div2(x):
	return np.array(x)/8


# 2.3
import io
def inp():
	f = io.open('./task2/input.txt', 'rt')
	return len(f.read())

# 2.4

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

def pred():
	KNN = KNeighborsClassifier()
	iris = load_iris()
	score = cross_val_score(KNN, iris.data, iris.target, cv=5).mean()
	return score



def test_answer():
	assert div(2) == .25   ## 2.2
	assert div2(2) == .25  ## 2.2
	assert inp() == 7
	assert pred() >= .7
