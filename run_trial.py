"""Discontinued file. See Trial_Setup_Utils"""

# important file
import Inductive_Generator
from Fully_Synthetic import generate_fully_synethic
from sklearn import model_selection
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier

from time import time
import os
import sys

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
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 10, random_state=42)

# Generating inductive orientation vectors

def decisionTreeSetup(max_branches):
    start = time()

    for i in range(1,max_branches):
        print(f"Starting Decision Tree with Depth {i}...")
        trial_start = time()
        decisionTreeClassifier = DecisionTreeClassifier(max_depth= i)
        decisionTreeClassifier_generator = Inductive_Generator.Inductive_Generator("sparse",decisionTreeClassifier, [0,1], X_train, y_train, X_test, y_test)
        decisionTreeClassifier_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
        decisionTreeClassifier_generator.compute_PD()
        decisionTreeClassifier_generator.save_state(f"logs/trial{TRIAL_NUM}/trial{TRIAL_NUM}_DecisionTree{i}.json", f"DecisionTree{i}", dataset_name)
        trial_end = time()
        print(f"Decision Tree with Depth {i} finished. Time elapsed: {(trial_end - trial_start)/60}.")

    end = time()
    print(f"All Decision Trees finished. Time elapsed: {(end - start)/60}.")


def kNNSetup(max_neighbors, num_dataset):

    start = time()

    for i in range(1,max_neighbors+1):
        print(f"Starting KNN with {i} Neighbors...")
        trial_start = time()

        # TODO: add option to use downloaded model
        kNeighborsClassifier = KNeighborsClassifier(n_neighbors=i)
        os.mkdir(f"logs/trial{TRIAL_NUM}/KNN{i}")
        kNeighborsClassifier_generator = Inductive_Generator.Inductive_Generator("sparse",kNeighborsClassifier, [0,1], f"logs/trial{TRIAL_NUM}/KNN{i}", X_train, y_train, X_test, y_test)
        kNeighborsClassifier_generator.get_LDM(X_test, num_dataset, 5, 0.15, "generate_subset")
        kNeighborsClassifier_generator.compute_PD()
        kNeighborsClassifier_generator.save_state(f"logs/trial{TRIAL_NUM}/trial{TRIAL_NUM}_KNN{i}.json", f"KNN{i}", dataset_name)
        trial_end = time()
        print(f"KNN with {i} Neighbors finished. Time elapsed: {(trial_end - trial_start)/60}.")

    end = time()
    print(f"All KNNs finished. Time elapsed: {(end - start)/60}.")

def randomForestSetup(max_estimators):
    """Random Forest"""
    start = time()

    for i in range(1,max_estimators):
        print(f"Starting Random Forest with {i} estimators...")
        trial_start = time()


        randomForest = RandomForestClassifier(n_estimators=i)
        randomForest_generator = Inductive_Generator.Inductive_Generator("sparse",randomForest, [0,1], X_train, y_train, X_test, y_test)
        randomForest_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
        randomForest_generator.compute_PD()
        randomForest_generator.save_state(f"logs/trial{TRIAL_NUM}/trial{TRIAL_NUM}_RandomForest{i}.json", f"RandomForest{i}", dataset_name)
        trial_end = time()
        print(f"Random Forest with {i} estimators finished. Time elapsed: {(trial_end - trial_start)/60}.")

    end = time()
    print(f"All Random Forests finished. Time elapsed: {(end - start)/60}.")

def randomForestSetupDepth(n_estimators, max_depth):
    """Brute force random forest w/ num_estimators metric (??)"""
    start = time()

    for i in range(1,max_depth):
        print(f"Starting Random Forest with {i} estimators...")
        trial_start = time()
        randomForest = RandomForestClassifier(n_estimators=n_estimators, max_depth=i)
        randomForest_generator = Inductive_Generator.Inductive_Generator("sparse",randomForest, [0,1], X_train, y_train, X_test, y_test)
        randomForest_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
        randomForest_generator.compute_PD()
        randomForest_generator.save_state(f"logs/trial{TRIAL_NUM}/trial{TRIAL_NUM}_RandomForest{n_estimators}EstDepth{i}.json", f"RandomForest{i}", dataset_name)
        trial_end = time()
        print(f"Random Forest with {i} estimators finished. Time elapsed: {(trial_end - trial_start)/60}.")

    end = time()
    print(f"All Random Forests finished. Time elapsed: {(end - start)/60}.")


def generate_iterable_model(model_name):
    """Helper to interable_model_setup -- """
def iterable_model_setup(model, model_name:str, from_download:bool = False, max_metric:int = 200):
    """
    Experiments + model generation OR experiments from pre-generated models that iterate over some metric;
    i.e., random forest, kNN, decisionTreeSetup
    """
    start = time()

    for i in range(1)
    print(f"Starting {model_name} with {i} estimators...")



def adaboostSetup():

    start = time()
    print(f"Starting Adaboost...")
    adaboostClassifier50 = AdaBoostClassifier()
    adaboostClassifier50_generator = Inductive_Generator.Inductive_Generator("sparse",adaboostClassifier50, [0,1], X_train, y_train, X_test, y_test)
    adaboostClassifier50_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
    adaboostClassifier50_generator.compute_PD()
    adaboostClassifier50_generator.save_state(f"logs/trial{TRIAL_NUM}/trial{TRIAL_NUM}_Adaboost50.json", f"Adaboost50", dataset_name)
    end = time()
    print(f"All Adaboost finished. Time elapsed: {(end - start)/60}.")

def QDASetup():

    start = time()
    print(f"Starting QDA...")
    quadraticDiscriminantAnalysis = QuadraticDiscriminantAnalysis()
    quadraticDiscriminantAnalysis_generator = Inductive_Generator.Inductive_Generator("sparse",quadraticDiscriminantAnalysis, [0,1], X_train, y_train, X_test, y_test)
    quadraticDiscriminantAnalysis_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
    quadraticDiscriminantAnalysis_generator.compute_PD()
    quadraticDiscriminantAnalysis_generator.save_state(f"logs/trial{TRIAL_NUM}/trial{TRIAL_NUM}_QuadraticDiscriminantAnalysis.json", f"QuadraticDiscriminantAnalysis", dataset_name)
    end = time()
    print(f"All QDA finished. Time elapsed: {(end - start)/60}.")

def gaussianProcessSetup():

    start = time()
    print(f"Starting GaussianProcessClassifier...")
    gaussianProcessClassifier = GaussianProcessClassifier()
    print("1")
    gaussianProcessClassifier_generator = Inductive_Generator.Inductive_Generator("sparse",gaussianProcessClassifier, [0,1], X_train, y_train, X_test, y_test)
    print("2")
    gaussianProcessClassifier_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
    print("3")
    gaussianProcessClassifier_generator.compute_PD()
    print("4")
    gaussianProcessClassifier_generator.save_state(f"logs/trial{TRIAL_NUM}/trial{TRIAL_NUM}_GaussianProcessClassifier.json", f"GaussianProcessClassifier", dataset_name)
    print("5")
    end = time()
    print(f"All GaussianProcessClassifier finished. Time elapsed: {(end - start)/60}.")

def naiveBayesClassifierSetup():

    start = time()
    print(f"Starting Naive Bayes Classifier...")
    naiveBayesClassifier = GaussianNB()
    print("1")
    naiveBayesClassifier_generator = Inductive_Generator.Inductive_Generator("sparse",naiveBayesClassifier, [0,1], X_train, y_train, X_test, y_test)
    print("2")
    naiveBayesClassifier_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
    print("3")
    naiveBayesClassifier_generator.compute_PD()
    print("4")
    naiveBayesClassifier_generator.save_state(f"logs/trial{TRIAL_NUM}/trial{TRIAL_NUM}_NaiveBayesClassifier.json", f"NaiveBayesClassifier", dataset_name)
    print("5")
    end = time()
    print(f"All Naive Bayes Classifier finished. Time elapsed: {(end - start)/60}.")

def linearSVCSetup():

    start = time()
    print(f"Starting Linear SVC...")
    linearSVC = LinearSVC()
    linearSVC_generator = Inductive_Generator.Inductive_Generator("sparse",linearSVC, [0,1], X_train, y_train, X_test, y_test)
    linearSVC_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
    linearSVC_generator.compute_PD()
    linearSVC_generator.save_state(f"logs/trial{TRIAL_NUM}/trial{TRIAL_NUM}_LinearSVC.json", f"LinearSVC", dataset_name)
    end = time()
    print(f"All Linear SVC finished. Time elapsed: {(end - start)/60}.")

def logisticRegressionSetup():

    start = time()
    print(f"Starting Logistic Regression...")
    logisticRegression = LogisticRegression()
    logisticRegression_generator = Inductive_Generator.Inductive_Generator("sparse",logisticRegression, [0,1], X_train, y_train, X_test, y_test)
    logisticRegression_generator.get_LDM(X_test, 500, 5, 0.15, "generate_subset")
    logisticRegression_generator.compute_PD()
    logisticRegression_generator.save_state(f"logs/trial{TRIAL_NUM}/trial{TRIAL_NUM}_LogisticRegression.json", f"LogisticRegression", dataset_name)
    end = time()
    print(f"All Logistic Regression finished. Time elapsed: {(end - start)/60}.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage error: require dataset name and size of holdout set")
    
    logs = os.listdir("logs")
    try:
        TRIAL_NUM = max([int(log.split("l")[1]) for log in logs])+1
    except:
        TRIAL_NUM = 1
    os.mkdir(os.path.join("logs", f"trial{TRIAL_NUM}"))

    print(sys.argv)
    if sys.argv[1] in "Abalone":
        dataset = "Abalone.csv"
    elif sys.argv[1] in "Bank_Marketing":
        dataset = "Bank_Marketing.csv"
    elif sys.argv[1] in "Car_Evaluation":
        dataset = "Car_Evaluation.csv"
    elif sys.argv[1] in "EEG_Eye_State.csv":
        dataset = "EEG_Eye_State.csv"
    elif sys.argv[1] in "Letter_Recognition":
        dataset = "Letter_Recognition.csv"
    elif sys.argv[1] in "Obesity":
        dataset = "Obesity.csv"
    elif sys.argv[1] in "Shopper_Intention":
        dataset = "Shopper_Intention.csv"        
    elif sys.argv[1] in "Spam":
        dataset = "Spam.csv"
    elif sys.argv[1] in "Wine_Quality":
        dataset = "Wine_Quality.csv"
    elif sys.argv[1] in "Semi_Random":
        dataset = "Semi_Random"
    else:
        sys.exit("We don't have that dataset. All that we have is ", os.listdir("datasets"))
    
    if dataset == "Semi_Random":  
        dataset_name = "SemiRandom"       
        X, y = generate_fully_synethic(4, 100, 100, 2)
    else:
        dataset_name = dataset.split(".")[0]
        dataset = os.path.join("datasets", dataset)
        data = pd.read_csv(dataset)
        X = data[data.columns[:-1]]
        X = X.iloc[:,:].values
        y = data[data.columns[-1]]
        y = y.values
        

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = int(sys.argv[2]), random_state=42)

    # decisionTreeSetup(50)
    kNNSetup(2, num_dataset=10)
    # randomForestSetup(50)
    # adaboostSetup()
    # QDASetup()
    # naiveBayesClassifierSetup()
    # linearSVCSetup()
    # logisticRegressionSetup()
    # gaussianProcessSetup()
    # randomForestSetupDepth(1, 50)
    # randomForestSetupDepth(11, 50)


# eeg 14979 total entries 6723 positive clases (44.88 percent positive class)
# shopper 12245 total entries 1892 positive classes (15.45 percent positive class)