import Algorithmic_Analysis
import Inductive_Generator
from sklearn import datasets, model_selection
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=5)

# generating target vector - 4 out of 5 correct with 3 classes
target = Algorithmic_Analysis.getTarget(X_test, y_test, 4, [0,1,2])
print(target)

# testing algorithmic bias
KNN10 = KNeighborsClassifier(n_neighbors=10)
KNN10_generator = Inductive_Generator.Inductive_Generator("sparse", KNN10, [0,1,2], X_train, y_train)
KNN10_generator.get_LDM(X_test, 2000, 3, 0.25, "generate_subset")
KNN10_generator.compute_PD()
PD_25 = KNN10_generator.PD
print(f"KNN10 PD vector trained on 25 percent of the Iris training data set: {PD_25}")
KNN10_Iris_Bias = Algorithmic_Analysis.computeAlgorithmicBias(target, PD_25)
print(KNN10_Iris_Bias)

# KNN10_generator.get_LDM(X_test, 2000, 3, 0.1, "generate_subset")
# KNN10_generator.compute_PD()
# PD_10 = KNN10_generator.PD
# KNN10_Iris_Bias = Algorithmic_Analysis.computeAlgorithmicBias(target, PD_10)
# print(f"KNN10 PD vector trained on 10 percent of the Iris training data set: {PD_10}")
# print(KNN10_Iris_Bias)

KNN10_Iris_Entropy = Algorithmic_Analysis.computeEntropy(KNN10_generator.PD)
print("The Entropic Expressivity of KNN10 is ", KNN10_Iris_Entropy)

KNN10_Iris_Capacity = Algorithmic_Analysis.computeAlgorithmicCapacity(KNN10_generator.LDM, KNN10_generator.PD)
print("The Algorithmic Capacity of KNN10 is ", KNN10_Iris_Capacity)
