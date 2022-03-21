# Testing on the server
import Inductive_Generator
from Fully_Synthetic import generate_fully_synethic
from sklearn import model_selection

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier

# Decision Tree up to 21
decisionTreeClassifier17 = DecisionTreeClassifier(max_depth= 3)

# k-Nearest Neighbors up to 21
KNN17 = KNeighborsClassifier(n_neighbors=3)

# Random Forest up to 21
randomForest17 = RandomForestClassifier(n_estimators=3)

# Adaboost
adaboostClassifier50 = AdaBoostClassifier()

# Quadratic Discriminant
quadraticDiscriminantAnalysis = QuadraticDiscriminantAnalysis()

# Gaussian Process Classifier  
gaussianProcessClassifier = GaussianProcessClassifier()

# Gaussian Naive Bayes
naiveBayesClassifier = GaussianNB()

# Linear Support Vector Machine
linearSVC = LinearSVC()

# Logistic Regression
logisticRegression = LogisticRegression()

# Getting Data
X, y = generate_fully_synethic(2000, 4, 100)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 5)

# Generating inductive orientation vectors
decisionTreeClassifier17_generator = Inductive_Generator.Inductive_Generator("sparse", decisionTreeClassifier17, [0,1], X_train, y_train)
decisionTreeClassifier17_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
decisionTreeClassifier17_generator.compute_PD()
decisionTreeClassifier17_generator.save_state("./logs/trial1_DecisionTree17")

KNN17_generator = Inductive_Generator.Inductive_Generator("sparse", KNN17, [0,1], X_train, y_train)
KNN17_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
KNN17_generator.compute_PD()
KNN17_generator.save_state("./logs/trial1_KNN17")

naiveBayesClassifier_generator = Inductive_Generator.Inductive_Generator("sparse", naiveBayesClassifier, [0,1], X_train, y_train)
naiveBayesClassifier_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
naiveBayesClassifier_generator.compute_PD()
naiveBayesClassifier_generator.save_state("./logs/trial1_naiveBayes")

linearSVC_generator = Inductive_Generator.Inductive_Generator("sparse", linearSVC, [0,1], X_train, y_train)
linearSVC_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
linearSVC_generator.compute_PD()
linearSVC_generator.save_state("./logs/trial1_linearSVC")
