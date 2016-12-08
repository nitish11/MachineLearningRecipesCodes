#Defining good features
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree


#Metadata of the dataset
greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24  + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked=True, color=['r','b'])
plt.show()
