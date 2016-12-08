from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()
test_idx = [0,50,100]

#Metadata of the dataset
print iris.feature_names
print iris.target_names

#Example 1 of the dataset
print iris.data[0]
print iris.target[0]

#Print 150 example of dataset
for i in range(len(iris.target)):
	print "Example %d: label %s, features %s" %(i, iris.target[i],iris.data[i])

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#classifier 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data,train_target)

print test_target
print clf.predict(test_data) 


#visulaization code 
from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=iris.feature_names,class_names=iris.target_names, filled=True, rounded=True, impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

