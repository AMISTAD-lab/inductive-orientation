import Inductive_Generator
from sklearn import datasets, model_selection
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 5)
KNN10 = KNeighborsClassifier(n_neighbors=10)

KNN10_generator = Inductive_Generator.Inductive_Generator("sparse", KNN10, [0,1,2], X_train, y_train)

# testing save_state
KNN10_generator.get_LDM(X_test, 10, 3, 0.3, "generate_subset")
KNN10_generator.compute_PD()
KNN10_generator.save_state("./test/trial1.json", "iris")

list_of_10_PD = []
for i in range(10):
  KNN10_generator.get_LDM(X_test, 10, 3, 0.3, "generate_subset")
  KNN10_generator.compute_PD()
  list_of_10_PD.append(KNN10_generator.PD)


