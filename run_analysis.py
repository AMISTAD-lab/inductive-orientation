"""Makes data to graph capacity, expressivity, bias"""

# important
import argparse
import Algorithmic_Analysis
import Inductive_Generator
import Data_Generator
from sklearn import datasets, model_selection
from sklearn.neighbors import KNeighborsClassifier
from Fully_Synthetic import generate_fully_synethic
import json
import os
import numpy as np
import pandas as pd
import sys
import math
import pdb
import Trial_Setup_Utils
from constants import RESULTS_FOLDER

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs analysis functions to find algorithmic capacity, bias, and entropic expressivity.")

    parser.add_argument('--trial_num', type=int, help='provide number of trial in results folder for analysis')
    parser.add_argument('--model_num', type=int, help='provide number of model in models folder for analysis')
    parser.add_argument('--holdout_size', type=int, help='provide size of the holdout set (program will analyze target sizes of 1 to holdout set size)')
    # parser.add_argument('--target_size', type=int, help='provide size of the target set (program will analyze target sizes of 1 to holdout set size)')

    args = parser.parse_args()
    TRIAL_NUM = args.trial_num
    MODEL_NUM = args.model_num
    SIZE_HOLDOUT = args.holdout_size
    # if len(sys.argv) != 5:
    #     sys.exit("Usage error: require trial number, size of holdout set and size of target set")
    # TRIAL_NUM = int(sys.argv[1])
    # MODEL_NUM = int(sys.argv[2])
    # SIZE_HOLDOUT = int(sys.argv[3])
    # SIZE_TARGET = int(sys.argv[4])
    LOWER_RANGE_TARGET_SIZE = 1 # placeholder
    UPPER_RANGE_TARGET_SIZE = SIZE_HOLDOUT # NOTE: PLACEHOLDER for input


    # UPPER_RANGE_TARGET_SIZE = int(sys.argv[4])


    
    
    # if SIZE_TARGET > SIZE_HOLDOUT:
    #     sys.exit("target set must be a subset of the holdout set.")
    
    with open(os.path.join(f"{RESULTS_FOLDER}/results/trial{TRIAL_NUM}", os.listdir(f"{RESULTS_FOLDER}/results/trial{TRIAL_NUM}")[0])) as logs:
        SAVED_STATE = json.loads(logs.read(), cls = Inductive_Generator.Inductive_Generator_Decoder)
        DATASET_NAME = SAVED_STATE["dataset"] + ".csv"
        HOLDOUT_SET_SEEDS = SAVED_STATE["holdout_set_seeds"]



    print(sys.argv)
    # loading data
    if DATASET_NAME == "SemiRandom.csv":
        X, y = generate_fully_synethic(4, 2000, 100, 2)
    else:
        dataset = os.path.join("datasets", DATASET_NAME)
        data = pd.read_csv(dataset)
        X = data[data.columns[:-1]]
        X = X.iloc[:,:].values
        y = data[data.columns[-1]]
        y = y.values

    
    # same seed = 42 for test train split
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,  test_size = 0.2, random_state=42) 
    
    # obtain holdout_sets_x, and then holdout_sets_y using seeds
    _, holdout_sets_y = Data_Generator.generateN_holdout_sets(num_holdouts = len(HOLDOUT_SET_SEEDS),
                                                                            holdout_size = int(math.log2(len(SAVED_STATE["PDs"][0]))), # hardcoded for 2 classes (should eventually generalize)
                                                                            X_test = X_test, y_test = y_test, 
                                                                            holdout_set_seeds = HOLDOUT_SET_SEEDS)


    # Gather target sets (of multiple sizes)
    target_sets = [] # each target list in target_sets corresponds to specific holdout set

    for holdout_set_y in holdout_sets_y:
        targets_ = {} # all targets for this particular holdout set
        for target_set_size in range(LOWER_RANGE_TARGET_SIZE, UPPER_RANGE_TARGET_SIZE+1):
            targets_[target_set_size] = Algorithmic_Analysis.getTarget(holdout_set_y, target_set_size, [0,1])
        target_sets.append(targets_)


    # summary = Algorithmic_Analysis.singleAnalysis(saved_state= SAVED_STATE,targets=targets) # oblig
    longer_summary = Algorithmic_Analysis.runAnalysis(f"{RESULTS_FOLDER}/results/trial{TRIAL_NUM}",target_sets)
    Trial_Setup_Utils.maybe_mkdir("./", f"{RESULTS_FOLDER}/analysis/trial{TRIAL_NUM}")
    longer_summary.to_pickle(f"{RESULTS_FOLDER}/analysis/trial{TRIAL_NUM}/{DATASET_NAME[:-4]}.pkl")
    print(f"PICKLED {RESULTS_FOLDER}/analysis/trial{TRIAL_NUM}/{DATASET_NAME[:-4]}.pkl")

    