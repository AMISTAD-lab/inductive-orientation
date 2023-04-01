# important file
import Inductive_Generator
from Fully_Synthetic import generate_fully_synethic
from sklearn import model_selection
import pandas as pd
import numpy as np
import pdb
import re

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


# model specific generator functions
def decision_tree_max_depth(max_depth):
    return DecisionTreeClassifier(max_depth=max_depth) #random_state=42)

def knn_n_neighbors(n_neigbors):
    return KNeighborsClassifier(n_neighbors =n_neigbors)

def random_forest_n_estimators(n_estimators):
    return RandomForestClassifier(n_estimators=n_estimators)

#result_number
#model_number

def model_training_loop(model, model_name, metric_range, metric_type, 
                     num_dataset, num_repeat, model_num, dataset_info, proportion_of_dataset):
    """"Trains and downloads LDMs"""
    start = time()
    for i in metric_range:
        print(f"Training {model_name} with {i} {metric_type}")
        trial_start = time()
        saving_dir = f"models/trial{model_num}/{model_name}{i}"
        maybe_mkdir(".", saving_dir) # make folder to store the models
        model_iter = model(i) # classifier
        model_generator = Inductive_Generator.Inductive_Generator("sparse", model_iter, [0,1], saving_dir, dataset_info, holdout_size=None, num_holdouts=0)
        model_generator.get_LDM(None, num_dataset, num_repeat, proportion_of_dataset, "generate_subset", from_download=False)
    
        trial_end = time()
        print(f"Training of {model_name} with {i} {metric_type} finished. Time elapsed: {(trial_end - trial_start)/60}.")

    end = time()
    print(f"All {model_name}s finished training. Time elapsed: {(end - start)/60}.")



def model_load_loop(model, model_name:str, metric_range:tuple, metric_type:str, 
                     num_dataset:int, num_repeat:int, model_num:int, 
                    result_num:int, dataset_info:dict, num_holdouts:int, holdout_size:int):#from_download:bool = False):
    """
    model (method) = a method that returns a model
    model_name (string) = actual name of the model, only for logging
    metric_range (Range) = range of values for a experimental variable like num_neighbors
    metric_type (string) = the name of the experimental variable, only for logging
    num_dataset (int) = the number of dataset to draw from the data distribution
    num_repeat (int) = the number of repeated ones on a particular subdataset
    model_num (int) = which model/trial number we are using (determines whether we use saved models)
    dataset_name (string) = only for logging
    result_number (int) = where to store our result, like a result trial number

    num_holdouts (int) = number of holdout sets gathered from
    holdout_size (int) = size of each holdout set
    """
    start = time()
    for i in metric_range:
        print(f"Starting result {result_num} using models from model {model_name} with {i} {metric_type}")
        trial_start = time()
        maybe_mkdir(".", f"results/trial{result_num}/model{model_num}/{model_name}{i}") # make folder to store the LDMs
        model_iter = model(i) # classifier
        model_generator = Inductive_Generator.Inductive_Generator("sparse", model_iter, [0,1], f"models/trial{model_num}/{model_name}{i}", dataset_info, holdout_size, num_holdouts)
        model_generator.getN_LDM_Pf(dataset_info["X_test"], dataset_info["y_test"], num_dataset, num_repeat, 0.15, from_download= True)
        model_generator.save_state(f"results/trial{result_num}/model{model_num}/{model_name}{i}/trial{model_num}_{model_name}{i}.json", f"{model_name}{i}", dataset_info["dataset_name"])
        trial_end = time()
        print(f"{model_name} with {i} {metric_type} finished. Time elapsed: {(trial_end - trial_start)/60}.")

    end = time()
    print(f"All {model_name}s finished. Time elapsed: {(end - start)/60}.")


def maybe_mkdir(basepath, folder_name):
    '''
        creates folder along a path
        input:
            basepath - starting directory to add the folders, must exist already
            folder_name - path to the final directory beyond the basepath, folders along the way do not need to exist
        return:
            exist_path - path to the final created directory
    '''
    path_to_add = folder_name.split("/")
    exist_path = basepath
    for path in path_to_add:
        current_path = os.path.join(exist_path, path)
        if path not in os.listdir(exist_path):
            os.mkdir(current_path)
        exist_path = current_path
    return exist_path

def next_trial_num(dir, default):
    logs = os.listdir(dir)
    log_nums = [int(log.split("l")[1]) for log in logs]
    if log_nums == []:
        return default
    else:
        return max(log_nums) + 1
    
if __name__ == "__main__":
    if len(sys.argv) != 6 and len(sys.argv) != 5:
        sys.exit("Usage error: require dataset name, size of holdout set, model to test, input model number, and whether to train/inference")
    
    # update trial number
    if os.path.exists("models") == False:
        maybe_mkdir(".", "models")
    current_model_num = next_trial_num("models", default=1)

    if os.path.exists("results") == False:
        maybe_mkdir(".", "results")
    current_result_num = next_trial_num("results", default=1)
    print(sys.argv)

    # load and configure dataset
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
        dataset = "SemiRandom.csv"
    else:
        sys.exit("We don't have that dataset. All that we have is ", os.listdir("datasets"))
    
    dataset_name = dataset.split(".")[0]
    dataset = os.path.join("datasets", dataset)
    data = pd.read_csv(dataset)
    X = data[data.columns[:-1]]
    X = X.iloc[:,:].values
    y = data[data.columns[-1]]
    y = y.values
    
    #TODO: new input of test_train_ratio --> test section of data will be pool for randomly selected holdout sets
    test_train_ratio = 0.20 # set for now
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_train_ratio, random_state=42)
    # since X_train is way bigger, now need to implement thing in Inductive_Generator to randomly select and download different holdout sets

    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = int(sys.argv[2]), random_state=42) # <- old code, whole X_train/X_test is tiny
                                                                                                                        # holdout set (maybe lead to anomaly cases)
    #NOTE: old code set test_size = 5, but sklearn documentation says that it should be a ratio from 0.0 to 1.0 (?)

    # choose which model to test
    is_loop= False
    if sys.argv[3].upper() in "DECISION_TREE":
        model = decision_tree_max_depth
        model_name = "Decision Tree" # TODO: implement enum system for model names so we don't get into annoying errors
        metric_type = "Depth"
        is_loop = True # whether model must loop over metric
    elif sys.argv[3].upper() in "KNN":
        model = knn_n_neighbors
        model_name = "KNN"
        metric_type = "Neighbors"
        is_loop = True
    elif sys.argv[3].upper() in "RANDOM_FOREST":
        model = randomForestSetupDepth
        model_name = "Random Forest"
        metric_type = "Estimators"
        is_loop = True
    elif sys.argv[3].upper() in "ADABOOST":
        model = adaboostSetup
        model_name = "Adaboost"
    elif sys.argv[3].upper() in "QDA":
        model = adaboostSetup
        model_name = "QDA"
    elif sys.argv[3].upper() in "GAUSSIAN":
        model = gaussianProcessSetup
        model_name = "Gaussian"
    elif sys.argv[3].upper() in "NAIVE_BAYES":
        model = naiveBayesClassifierSetup
        model_name = "Naive Bayes"
    elif sys.argv[3].upper() in "LINEAR_SVC":
        model = linearSVCSetup
        model_name = "Linear SVC"
    elif sys.argv[3].upper() in "LOGISTIC_REGRESSION":
        model = logisticRegressionSetup
        model_name = "Logistic Regression"
    else:
        print("Running all")
        #TODO: implement thing to run all models
    
    trial_mode = sys.argv[4]
    if trial_mode == "inference":
        model_num = sys.argv[5] # which trial number to look for model in
 
    
        

    if trial_mode == "training":
        dataset_info = {"dataset_name": dataset_name, "X_train":X_train, "X_test": X_test, "y_train":y_train, "y_test":y_test}
        model_training_loop(model=model, model_name=model_name, metric_range=range(1,7), 
                        metric_type=metric_type, num_dataset=100, num_repeat=5, model_num=current_model_num, 
                        dataset_info=dataset_info, proportion_of_dataset=0.15)
    elif trial_mode == "inference":
        dataset_info = {"dataset_name": dataset_name, "X_train":X_train, "X_test": X_test, "y_train":y_train, "y_test":y_test}
        model_load_loop(model=model, model_name=model_name, metric_range=range(1,7), 
                        metric_type=metric_type, num_dataset=100, num_repeat=5, model_num=model_num,
                        dataset_info=dataset_info, result_num=current_result_num, num_holdouts=2, holdout_size=5)
    else:
        print("Not yet implemented")
        #TODO: implement non-looping generic setup



    # randomForestSetup(50)
    # adaboostSetup()
    # QDASetup()
    # naiveBayesClassifierSetup()
    # linearSVCSetup()
    # logisticRegressionSetup()
    # gaussianProcessSetup()
    # randomForestSetupDepth(1, 50)
    # randomForestSetupDepth(11, 50)

        # decisionTreeSetup(50)
    #kNNSetup(2, num_dataset=10)
    


# eeg 14979 total entries 6723 positive clases (44.88 percent positive class)
# shopper 12245 total entries 1892 positive classes (15.45 percent positive class)

"""
Run with:
python Trial_Setup_Utils.py EEG_Eye_State.csv {size of the holdout set=5}

"""