import random

from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)


class CustomKNN():
    def fit(self, X_train, Y_train):
		self.X_train = X_train
		self.Y_train = Y_train
		
    def predict(self, X_test):
		predictions = []
		for row in X_test:
		    label = random.choice(self.Y_train)
		    predictions.append(label)
		return predictions
		

from sklearn import datasets
iris = datasets.load_iris()

#Metadata of the dataset
x = iris.data
y = iris.target

#prepare train and test data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.5)

#Classifier-1
#from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()

#Classifier-2
#from sklearn.neighbors import KNeighborsClassifier
#my_classifier = KNeighborsClassifier()

#Custom Classifier
my_classifier = CustomKNN()

#Training Classifier, Prediction and Accuracy Calculation
my_classifier.fit(x_train, y_train)
predictions = my_classifier.predict(x_test) 
#print predictions
from sklearn.metrics import accuracy_score
print accuracy_score(y_test,predictions)

