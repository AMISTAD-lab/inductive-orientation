"""Makes data to graph capacity, expressivity, bias"""

# important
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

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage error: require trial number, size of holdout set and size of target set")
    TRIAL_NUM = int(sys.argv[1])
    MODEL_NUM = int(sys.argv[2])
    SIZE_HOLDOUT = int(sys.argv[3])
    SIZE_TARGET = int(sys.argv[4])
    
    
    if SIZE_TARGET > SIZE_HOLDOUT:
        sys.exit("target set must be a subset of the holdout set.")

    with open(os.path.join(f"results/trial{TRIAL_NUM}", os.listdir(f"results/trial{TRIAL_NUM}")[0])) as logs:
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


    # Gather target sets
    targets = [] 
    for holdout_set_y in holdout_sets_y:
        targets.append(Algorithmic_Analysis.getTarget(holdout_set_y, SIZE_TARGET, [0,1]))


    summary = Algorithmic_Analysis.singleAnalysis(saved_state= SAVED_STATE,targets=targets) # oblig
    longer_summary = Algorithmic_Analysis.runAnalysis(f"results/trial{TRIAL_NUM}",targets)
    Trial_Setup_Utils.maybe_mkdir("./", f"analysis/trial{TRIAL_NUM}")
    longer_summary.to_pickle(f"analysis/trial{TRIAL_NUM}/{DATASET_NAME[:-4]}.pkl")

    