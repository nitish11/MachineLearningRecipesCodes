from sklearn import datasets
iris = datasets.load_iris()

#Metadata of the dataset
x = iris.data
y = iris.target

#Example 1 of the dataset
#print iris.data[0]
#print iris.target[0]


#prepare train and test data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.5)

#Classifier-1
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

#Classifier-2
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

#Training Classifier, Prediction and Accuracy Calculation
my_classifier.fit(x_train, y_train)
predictions = my_classifier.predict(x_test) 
#print predictions
from sklearn.metrics import accuracy_score
print accuracy_score(y_test,predictions)
