# important file
import Inductive_Generator
from Fully_Synthetic import generate_fully_synethic
from sklearn import model_selection
import pandas as pd
import numpy as np
import pdb
import re
from models import *

from time import time
import os
import sys
from models import *
from constants import ModelNamesMetrics



def model_training_loop(model, model_name, metric_range, metric_type, 
                     num_dataset, num_repeat, model_num, dataset_info, proportion_of_dataset):
    """"Trains and downloads LDMs"""
    start = time()
    for i in metric_range:
        print(f"Training {model_name} with {i} {metric_type}")
        trial_start = time()
        saving_dir = f"models/model{model_num}/{model_name}{i}"
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
        maybe_mkdir(".", f"results/trial{result_num}") # make folder to store the LDMs
        model_iter = model(i) # classifier
        model_generator = Inductive_Generator.Inductive_Generator("sparse", model_iter, [0,1], f"models/model{model_num}/{model_name}{i}", dataset_info, holdout_size, num_holdouts)
        model_generator.getN_LDM_Pf(dataset_info["X_test"], dataset_info["y_test"], num_dataset, num_repeat, 0.15, from_download= True)
        model_generator.save_state(f"results/trial{result_num}/model_{model_num}_{model_name}_{i}.json", f"{model_name}{i}", dataset_info["dataset_name"])
        trial_end = time()
        print(f"{model_name} with {i} {metric_type} finished. Time elapsed: {(trial_end - trial_start)/60}.")

    end = time()
    print(f"All {model_name}s finished. Time elapsed: {(end - start)/60}.")


def maybe_mkdir(basepath, folder_name):
    '''
        Helper, creates folder along a path
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
    if len(sys.argv) != 9 and len(sys.argv) != 8:
        sys.exit("Usage error: 1. require dataset name, \n\
                                2. number of holdout sets, \n\
                                3. size of holdout set, \n\
                                4. model to test, \n\
                                5. whether to train/inference \n\
                                6. lower range of metrics to test \n\
                                7. upper range of metrics to test \n\
                                8. input model number, \n")
    argv_dataset = sys.argv[1]
    num_holdout_sets = int(sys.argv[2])
    holdout_set_size = int(sys.argv[3])
    argv_model = sys.argv[4]
    trial_mode = sys.argv[5]
    if trial_mode == "inference":
        model_num = sys.argv[8] # which model to load
    lower_range = int(sys.argv[6])
    upper_range = int(sys.argv[7])    

    
    # update trial number
    if os.path.exists("models") == False:
        maybe_mkdir(".", "models")
    current_model_num = next_trial_num("models", default=1) # current_model_num is the trial number

    if os.path.exists("results") == False:
        maybe_mkdir(".", "results")
    current_result_num = next_trial_num("results", default=1)
    print(sys.argv)

    # load and configure dataset
    if argv_dataset in "Abalone":
        dataset = "Abalone.csv"
    elif argv_dataset in "Bank_Marketing":
        dataset = "Bank_Marketing.csv"
    elif argv_dataset in "Car_Evaluation":
        dataset = "Car_Evaluation.csv"
    elif argv_dataset in "EEG_Eye_State.csv":
        dataset = "EEG_Eye_State.csv"
    elif argv_dataset in "Letter_Recognition":
        dataset = "Letter_Recognition.csv"
    elif argv_dataset in "Obesity":
        dataset = "Obesity.csv"
    elif argv_dataset in "Shopper_Intention":
        dataset = "Shopper_Intention.csv"        
    elif argv_dataset in "Spam":
        dataset = "Spam.csv"
    elif argv_dataset in "Wine_Quality":
        dataset = "Wine_Quality.csv"
    elif argv_dataset in "Semi_Random":
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

    #NOTE: old code set test_size = 5, but sklearn documentation says that it should be a ratio from 0.0 to 1.0 (?)

    
    # choose which model to test
    is_loop= False
    if argv_model.upper() in ModelNamesMetrics.DECISION_TREE_MAX_DEPTH.value.upper():
        model = decision_tree_max_depth
        model_name = ModelNamesMetrics.DECISION_TREE_MAX_DEPTH.value # TODO: implement enum system for model names so we don't get into annoying errors
        metric_type = "Depth"
        is_loop = True # whether model must loop over metric
    elif argv_model.upper() in "KNN":
        model = knn_n_neighbors
        model_name = "KNN"
        metric_type = "Neighbors"
        is_loop = True
    elif argv_model.upper() in ModelNamesMetrics.RANDOM_FOREST_DEPTH.value.upper():
        model = random_forest_depth_setup # default is 100, alter depth
        model_name = ModelNamesMetrics.RANDOM_FOREST_DEPTH.value
        metric_type = "Depth"
        is_loop = True
    elif argv_model.upper() in ModelNamesMetrics.RANDOM_FOREST_ESTIMATORS.value.upper():
        model = random_forest_n_estimators
        model_name = ModelNamesMetrics.RANDOM_FOREST_ESTIMATORS.value
        metric_type = "Estimators"
        is_loop = True
    elif argv_model.upper() in ModelNamesMetrics.ADABOOST_ESTIMATORS.value.upper():
        model = adaboost_n_estimators
        model_name = ModelNamesMetrics.ADABOOST_ESTIMATORS.value
        metric_type = "Estimators"
        is_loop = True
    elif argv_model.upper() in "QDA":
        model = QDA
        model_name = "QDA"
        #TODO: no metric
    elif argv_model.upper() in "GAUSSIAN":
        model = gaussian_process_max_iter_predict
        metric_type = "Max iterations"
        model_name = "Gaussian"
    elif argv_model.upper() in "NAIVE_BAYES":
        model = nb_gaussian # no metric
        model_name = "Naive Bayes"
    elif argv_model.upper() in "LINEAR_SVC":
        model = linear_SVC_max_iter
        model_name = "Linear SVC"
        metric_type = "Max iterations"
    elif argv_model.upper() in "C_SVC":
        model = c_SVC_max_iter
        model_name = "C-support SVC"
        metric_type = "Max iterations"
    elif argv_model.upper() in "LOGISTIC_REGRESSION":
        model = logistic_regression_max_iter
        model_name = "Logistic Regression"
        metric_type = "Max iterations"
    else:
        print("Running all")
        #TODO: implement thing to run all models
    
    
    if trial_mode == "training":
        dataset_info = {"dataset_name": dataset_name, "X_train":X_train, "X_test": X_test, "y_train":y_train, "y_test":y_test}
        model_training_loop(model=model, model_name=model_name, metric_range=range(lower_range, upper_range), 
                        metric_type=metric_type, num_dataset=100, num_repeat=5, model_num=current_model_num, 
                        dataset_info=dataset_info, proportion_of_dataset=0.15)
    elif trial_mode == "inference":
        dataset_info = {"dataset_name": dataset_name, "X_train":X_train, "X_test": X_test, "y_train":y_train, "y_test":y_test}
        model_load_loop(model=model, model_name=model_name, metric_range=range(lower_range,upper_range), 
                        metric_type=metric_type, num_dataset=100, num_repeat=5, model_num=model_num,
                        dataset_info=dataset_info, result_num=current_result_num, num_holdouts=num_holdout_sets, holdout_size=holdout_set_size)
    else:
        print("Must enter 'training' or 'inference'")
        #TODO: implement non-looping generic setup


    # notify trial number
    print(f"{trial_mode} complete!\n\
          Trial mode: {trial_mode}\n\
          Trial number: {current_model_num}\n\
          Model type: {argv_model.upper()}\n\
          Measure range: ({lower_range}, {upper_range})\n\
          Dataset name: {dataset_name}\n")
    
    if trial_mode != "training":
        print(f"inference of model {model_num}\n")
    


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