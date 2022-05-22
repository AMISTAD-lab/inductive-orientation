# corresponds to trial 4, uses the same random binary data as trials 2 and 3, but this time
# the testing set is guaranteed to be seend by the model during training.

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
        decisionTreeClassifier_generator = Inductive_Generator.Inductive_Generator("sparse",decisionTreeClassifier, [0,1], X_train, y_train, X_test, y_test)
        decisionTreeClassifier_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset_plus_fixed")
        decisionTreeClassifier_generator.compute_PD()
        decisionTreeClassifier_generator.save_state(f"./trial4/trial4_DecisionTree{i}.json", f"DecisionTree{i}", "SemiRandom")
        trial_end = time()
        print(f"Decision Tree with Depth {i} finished. Time elapsed: {(trial_end - trial_start)/60}.")

    end = time()
    print(f"All Decision Trees finished. Time elapsed: {(end - start)/60}.")


def kNNSetup():

    start = time()

    for i in range(1,22):
        trial_start = time()
        kNeighborsClassifier = KNeighborsClassifier(n_neighbors=i)
        kNeighborsClassifier_generator = Inductive_Generator.Inductive_Generator("sparse",kNeighborsClassifier, [0,1], X_train, y_train, X_test, y_test)
        kNeighborsClassifier_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset_plus_fixed")
        kNeighborsClassifier_generator.compute_PD()
        kNeighborsClassifier_generator.save_state(f"./trial4/trial4_KNN{i}.json", f"KNN{i}", "SemiRandom")
        trial_end = time()
        print(f"KNN with {i} Neighbors finished. Time elapsed: {(trial_end - trial_start)/60}.")

    end = time()
    print(f"All KNNs finished. Time elapsed: {(end - start)/60}.")

def randomForestSetup():

    start = time()

    for i in range(1,22):
        trial_start = time()
        randomForest = RandomForestClassifier(n_estimators=i)
        randomForest_generator = Inductive_Generator.Inductive_Generator("sparse",randomForest, [0,1], X_train, y_train, X_test, y_test)
        randomForest_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset_plus_fixed")
        randomForest_generator.compute_PD()
        randomForest_generator.save_state(f"./trial4/trial4_RandomForest{i}.json", f"RandomForest{i}", "SemiRandom")
        trial_end = time()
        print(f"Random Forest with {i} estimators finished. Time elapsed: {(trial_end - trial_start)/60}.")

    end = time()
    print(f"All Random Forests finished. Time elapsed: {(end - start)/60}.")

def adaboostSetup():

    start = time()
    adaboostClassifier50 = AdaBoostClassifier()
    adaboostClassifier50_generator = Inductive_Generator.Inductive_Generator("sparse",adaboostClassifier50, [0,1], X_train, y_train, X_test, y_test)
    adaboostClassifier50_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset_plus_fixed")
    adaboostClassifier50_generator.compute_PD()
    adaboostClassifier50_generator.save_state(f"./trial4/trial4_Adaboost50.json", f"Adaboost50", "SemiRandom")
    end = time()
    print(f"All Adaboost finished. Time elapsed: {(end - start)/60}.")

def QDASetup():

    start = time()
    quadraticDiscriminantAnalysis = QuadraticDiscriminantAnalysis()
    quadraticDiscriminantAnalysis_generator = Inductive_Generator.Inductive_Generator("sparse",quadraticDiscriminantAnalysis, [0,1], X_train, y_train, X_test, y_test)
    quadraticDiscriminantAnalysis_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset_plus_fixed")
    quadraticDiscriminantAnalysis_generator.compute_PD()
    quadraticDiscriminantAnalysis_generator.save_state(f"./trial4/trial4_QuadraticDiscriminantAnalysis.json", f"QuadraticDiscriminantAnalysis", "SemiRandom")
    end = time()
    print(f"All QDA finished. Time elapsed: {(end - start)/60}.")

def gaussianProcessSetup():

    start = time()
    gaussianProcessClassifier = GaussianProcessClassifier()
    gaussianProcessClassifier_generator = Inductive_Generator.Inductive_Generator("sparse",gaussianProcessClassifier, [0,1], X_train, y_train, X_test, y_test)
    gaussianProcessClassifier_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset_plus_fixed")
    gaussianProcessClassifier_generator.compute_PD()
    gaussianProcessClassifier_generator.save_state(f"./trial4/trial4_GaussianProcessClassifier.json", f"GaussianProcessClassifier", "SemiRandom")
    end = time()
    print(f"All GaussianProcessClassifier finished. Time elapsed: {(end - start)/60}.")

def naiveBayesClassifierSetup():

    start = time()
    naiveBayesClassifier = GaussianNB()
    naiveBayesClassifier_generator = Inductive_Generator.Inductive_Generator("sparse",naiveBayesClassifier, [0,1], X_train, y_train, X_test, y_test)
    naiveBayesClassifier_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset_plus_fixed")
    naiveBayesClassifier_generator.compute_PD()
    naiveBayesClassifier_generator.save_state(f"./trial4/trial4_NaiveBayesClassifier.json", f"NaiveBayesClassifier", "SemiRandom")
    end = time()
    print(f"All Naive Bayes Classifier finished. Time elapsed: {(end - start)/60}.")

def linearSVCSetup():

    start = time()
    linearSVC = LinearSVC()
    linearSVC_generator = Inductive_Generator.Inductive_Generator("sparse",linearSVC, [0,1], X_train, y_train, X_test, y_test)
    linearSVC_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset_plus_fixed")
    linearSVC_generator.compute_PD()
    linearSVC_generator.save_state(f"./trial4/trial4_LinearSVC.json", f"LinearSVC", "SemiRandom")
    end = time()
    print(f"All Linear SVC finished. Time elapsed: {(end - start)/60}.")

def logisticRegressionSetup():

    start = time()
    logisticRegression = LogisticRegression()
    logisticRegression_generator = Inductive_Generator.Inductive_Generator("sparse",logisticRegression, [0,1], X_train, y_train, X_test, y_test)
    logisticRegression_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset_plus_fixed")
    logisticRegression_generator.compute_PD()
    logisticRegression_generator.save_state(f"./trial4/trial4_LogisticRegression.json", f"LogisticRegression", "SemiRandom")
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
