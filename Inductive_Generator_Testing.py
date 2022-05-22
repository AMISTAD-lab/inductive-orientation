import Inductive_Generator
from sklearn import datasets, model_selection
from sklearn.neighbors import KNeighborsClassifier
import json
from Fully_Synthetic import generate_fully_synethic


# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 5)
# KNN10 = KNeighborsClassifier(n_neighbors=10)

# KNN10_generator = Inductive_Generator.Inductive_Generator("sparse", KNN10, [0,1,2], X_train, y_train)

# # testing save_state
# KNN10_generator.get_LDM(X_test, 10, 3, 0.3, "generate_subset")
# KNN10_generator.compute_PD()
# KNN10_generator.save_state("./test/trial2.json", "KNN10", "iris")
# with open("./test/trial2.json") as logs:
#   saved_state = json.loads(logs.read(), cls=Inductive_Generator.Inductive_Generator_Decoder)

# list_of_10_PD = []
# for i in range(10):
#   KNN10_generator.get_LDM(X_test, 10, 3, 0.3, "generate_subset")
#   KNN10_generator.compute_PD()
#   list_of_10_PD.append(KNN10_generator.PD)

# testing new generate_subset_plus_fixed
# KNN10_generator = Inductive_Generator.Inductive_Generator("sparse", KNN10, [0,1,2], X_train, y_train, X_test, y_test)
# KNN10_generator.get_LDM(X_test, 10, 3, 0.3, "generate_subset_plus_fixed")
# KNN10_generator.compute_PD()
# print(KNN10_generator.PD)

# testing larger test set
X, y = generate_fully_synethic(4, 2000, 100, 2)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 10, random_state=42)
KNN10 = KNeighborsClassifier(n_neighbors=10)
KNN10_generator = Inductive_Generator.Inductive_Generator("sparse", KNN10, [0,1], X_train, y_train, X_test, y_test)
KNN10_generator.get_LDM(X_test, 500, 10, 0.3, "generate_subset_plus_fixed")
KNN10_generator.compute_PD()