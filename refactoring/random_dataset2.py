# corresponds to trial 3, same as trial 2, but this time it's not a random seed

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

from time import time

# Decision Tree up to 21
decisionTreeClassifier3 = DecisionTreeClassifier(max_depth= 3)

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
X, y = generate_fully_synethic(4, 2000, 100, 2)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 5, random_state=42)

# Generating inductive orientation vectors

def decisionTreeSetup():

    start = time()

    for i in range(1,22):
        trial_start = time()
        decisionTreeClassifier = DecisionTreeClassifier(max_depth= i)
        decisionTreeClassifier_generator = Inductive_Generator.Inductive_Generator("sparse",decisionTreeClassifier, [0,1], X_train, y_train)
        decisionTreeClassifier_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
        decisionTreeClassifier_generator.compute_PD()
        decisionTreeClassifier_generator.save_state(f"./trial3/trial3_DecisionTree{i}.json", f"DecisionTree{i}", "FullySynthetic")
        trial_end = time()
        print(f"Decision Tree with Depth {i} finished. Time elapsed: {(trial_end - trial_start)/60}.")

    end = time()
    print(f"All Decision Trees finished. Time elapsed: {(end - start)/60}.")


def kNNSetup():

    start = time()

    for i in range(1,22):
        trial_start = time()
        kNeighborsClassifier = KNeighborsClassifier(n_neighbors=i)
        kNeighborsClassifier_generator = Inductive_Generator.Inductive_Generator("sparse",kNeighborsClassifier, [0,1], X_train, y_train)
        kNeighborsClassifier_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
        kNeighborsClassifier_generator.compute_PD()
        kNeighborsClassifier_generator.save_state(f"./trial3/trial3_KNN{i}.json", f"KNN{i}", "FullySynthetic")
        trial_end = time()
        print(f"KNN with {i} Neighbors finished. Time elapsed: {(trial_end - trial_start)/60}.")

    end = time()
    print(f"All KNNs finished. Time elapsed: {(end - start)/60}.")

def randomForestSetup():

    start = time()

    for i in range(1,22):
        trial_start = time()
        randomForest = RandomForestClassifier(n_estimators=i)
        randomForest_generator = Inductive_Generator.Inductive_Generator("sparse",randomForest, [0,1], X_train, y_train)
        randomForest_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
        randomForest_generator.compute_PD()
        randomForest_generator.save_state(f"./trial3/trial3_RandomForest{i}.json", f"RandomForest{i}", "FullySynthetic")
        trial_end = time()
        print(f"Random Forest with {i} estimators finished. Time elapsed: {(trial_end - trial_start)/60}.")

    end = time()
    print(f"All Random Forests finished. Time elapsed: {(end - start)/60}.")

def adaboostSetup():

    start = time()
    adaboostClassifier50 = AdaBoostClassifier()
    adaboostClassifier50_generator = Inductive_Generator.Inductive_Generator("sparse",adaboostClassifier50, [0,1], X_train, y_train)
    adaboostClassifier50_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
    adaboostClassifier50_generator.compute_PD()
    adaboostClassifier50_generator.save_state(f"./trial3/trial3_Adaboost50.json", f"Adaboost50", "FullySynthetic")
    end = time()
    print(f"All Adaboost finished. Time elapsed: {(end - start)/60}.")

def QDASetup():

    start = time()
    quadraticDiscriminantAnalysis = QuadraticDiscriminantAnalysis()
    quadraticDiscriminantAnalysis_generator = Inductive_Generator.Inductive_Generator("sparse",quadraticDiscriminantAnalysis, [0,1], X_train, y_train)
    quadraticDiscriminantAnalysis_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
    quadraticDiscriminantAnalysis_generator.compute_PD()
    quadraticDiscriminantAnalysis_generator.save_state(f"./trial3/trial3_QuadraticDiscriminantAnalysis.json", f"QuadraticDiscriminantAnalysis", "FullySynthetic")
    end = time()
    print(f"All QDA finished. Time elapsed: {(end - start)/60}.")

def gaussianProcessSetup():

    start = time()
    gaussianProcessClassifier = GaussianProcessClassifier()
    gaussianProcessClassifier_generator = Inductive_Generator.Inductive_Generator("sparse",gaussianProcessClassifier, [0,1], X_train, y_train)
    gaussianProcessClassifier_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
    gaussianProcessClassifier_generator.compute_PD()
    gaussianProcessClassifier_generator.save_state(f"./trial3/trial3_GaussianProcessClassifier.json", f"GaussianProcessClassifier", "FullySynthetic")
    end = time()
    print(f"All GaussianProcessClassifier finished. Time elapsed: {(end - start)/60}.")

def naiveBayesClassifierSetup():

    start = time()
    naiveBayesClassifier = GaussianNB()
    naiveBayesClassifier_generator = Inductive_Generator.Inductive_Generator("sparse",naiveBayesClassifier, [0,1], X_train, y_train)
    naiveBayesClassifier_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
    naiveBayesClassifier_generator.compute_PD()
    naiveBayesClassifier_generator.save_state(f"./trial3/trial3_NaiveBayesClassifier.json", f"NaiveBayesClassifier", "FullySynthetic")
    end = time()
    print(f"All Naive Bayes Classifier finished. Time elapsed: {(end - start)/60}.")

def linearSVCSetup():

    start = time()
    linearSVC = LinearSVC()
    linearSVC_generator = Inductive_Generator.Inductive_Generator("sparse",linearSVC, [0,1], X_train, y_train)
    linearSVC_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
    linearSVC_generator.compute_PD()
    linearSVC_generator.save_state(f"./trial3/trial3_LinearSVC.json", f"LinearSVC", "FullySynthetic")
    end = time()
    print(f"All Linear SVC finished. Time elapsed: {(end - start)/60}.")

def logisticRegressionSetup():

    start = time()
    logisticRegression = LogisticRegression()
    logisticRegression_generator = Inductive_Generator.Inductive_Generator("sparse",logisticRegression, [0,1], X_train, y_train)
    logisticRegression_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
    logisticRegression_generator.compute_PD()
    logisticRegression_generator.save_state(f"./trial3/trial3_LogisticRegression.json", f"LogisticRegression", "FullySynthetic")
    end = time()
    print(f"All Logistic Regression finished. Time elapsed: {(end - start)/60}.")

decisionTreeSetup()
kNNSetup()
randomForestSetup()
adaboostSetup()
QDASetup()
gaussianProcessSetup()
naiveBayesClassifierSetup()
linearSVCSetup()
logisticRegressionSetup()
randomForestSetup()
