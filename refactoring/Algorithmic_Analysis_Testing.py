import Algorithmic_Analysis
from sklearn import datasets, model_selection

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=5)
target = Algorithmic_Analysis.getTarget(X_test, y_test, 4, [0,1,2])
print(target)
