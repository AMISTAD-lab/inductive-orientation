"""TODO: ADD DOCUMENTATION ABOUT WHAT THIS DOES"""

# important
import Algorithmic_Analysis
import Inductive_Generator
from sklearn import datasets, model_selection
from sklearn.neighbors import KNeighborsClassifier
from Fully_Synthetic import generate_fully_synethic
import json
import os
import numpy as np
import pandas as pd
import sys

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage error: require trial number, size of holdout set and size of target set")
    TRIAL_NUM = int(sys.argv[1])
    SIZE_HOLDOUT = int(sys.argv[2])
    SIZE_TARGET = int(sys.argv[3])
    if SIZE_TARGET > SIZE_HOLDOUT:
        sys.exit("target set must be a subset of the holdout set.")
    with open(os.path.join(f"logs/trial{TRIAL_NUM}", os.listdir(f"logs/trial{TRIAL_NUM}")[0])) as logs:
        saved_state = json.loads(logs.read(), cls = Inductive_Generator.Inductive_Generator_Decoder)
        DATASET_NAME = saved_state["dataset"] + ".csv"

    print(sys.argv)
    if DATASET_NAME == "SemiRandom.csv":
        X, y = generate_fully_synethic(4, 2000, 100, 2)
    else:
        dataset = os.path.join("datasets", DATASET_NAME)
        data = pd.read_csv(dataset)
        X = data[data.columns[:-1]]
        X = X.iloc[:,:].values
        y = data[data.columns[-1]]
        y = y.values

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = SIZE_HOLDOUT, random_state=42)

    target = Algorithmic_Analysis.getTarget(X_test, y_test, SIZE_TARGET, [0,1])

    summary = Algorithmic_Analysis.runAnalysis(f"logs/trial{TRIAL_NUM}", target=target)
    summary.sort_values(by=['model_name'])
    summary.to_csv(f"trial{TRIAL_NUM}_target{SIZE_TARGET}.csv")
