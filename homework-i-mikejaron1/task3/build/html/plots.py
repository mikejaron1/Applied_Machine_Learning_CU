import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


iris = load_iris()
feat_names = iris['feature_names']

plt.figure(1, figsize=(15,15)) 
i = 1
for item in list(range(4)):
    for j in list(range(4)):
        plt.subplot(4,4,i)
        if item != j:
            plt.scatter(iris.data[:,j], iris.data[:,item], c = iris.target, marker='o', s=60, alpha=.6)
        else:
            plt.hist(iris.data[:,item], bins=20)
        if i in [1,5,9,13]:
            plt.ylabel(feat_names[item])
        else:
            plt.yticks([])
        if i >= 13:
            plt.xlabel(feat_names[j])
        else:
            plt.xticks([])
        
        i += 1

## for tagerts 0 = setosa, 1 = versicolor, 2 = virginica

plt.scatter([], [], c = '#800000', marker='o', s=60, alpha=.6, label='setosa')
plt.scatter([], [], c = '#000080', marker='o', s=60, alpha=.7, label='versicolor')
plt.scatter([], [], c = '#009933', marker='o', s=60, alpha=.6, label='virginica')

plt.subplots_adjust(hspace=0, wspace=0)
plt.legend()
plt.show()