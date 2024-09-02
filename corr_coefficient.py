# Small file for determining correlation coefficient between (expressivity-capacity)/(test-train)

from constants import MODEL_TO_TRIAL_NUMS, RESULTS_FOLDER, ModelNamesMetrics, DatasetNames
import pandas as pd
import json
import numpy as np
import pdb

analysis_folder = f"{RESULTS_FOLDER}/analysis"

df = pd.read_csv(f"{analysis_folder}/trial124/test_train_accuracies.csv")


def get_corr_coef(trial_folder:str, algorithm_name:str, dataset_name:str) -> float:
    """
    Given analysis/trial{num} path, export json storing dataset name, algorithm name, and
    correlation coefficient
    """
    # pdb.set_trace()
    # get (expressivity-capacity) and (test-train) array-like objects 
    try:
        df_test_train = pd.read_csv(f"{trial_folder}/test_train_accuracies_CORRECTED.csv")
    except:
        df_test_train = pd.read_csv(f"{trial_folder}/test_train_accuracies.csv")
        print(f"\n\n{trial_folder} is using BACKUP")
    df_exp_cap = pd.read_pickle(f"{trial_folder}/{dataset_name}.pkl")
    test_train_diff = np.array(df_test_train["train mean"] - df_test_train["test mean"])

    # sort df_exp_cap by parameters
    df_exp_cap["parameter"] = df_exp_cap["model_name"].apply(lambda text: int(text.split("_")[-1]))
    df_exp_cap.sort_values(by="parameter", inplace=True)

    # get expresisity capacity averages
    df_exp_cap["entropic_expressivity_mean"] = df_exp_cap["entropic_expressivity"].apply(np.mean)
    df_exp_cap["algorithmic_capacity_mean"] = df_exp_cap["algorithmic_capacity"].apply(np.mean)
    exp_cap_diff = np.array(df_exp_cap["entropic_expressivity_mean"] - df_exp_cap["algorithmic_capacity_mean"])

    # get correlation coefficient
    try:
        corr_coeff = np.corrcoef(x=exp_cap_diff, y=test_train_diff)[0,1]
    except:
        corr_coeff = np.nan

    # make json
    with open(f"{trial_folder}/correlation_coefficient.json", "w") as outfile:
        json.dump({"algorithm_name":algorithm_name, "dataset_name":dataset_name, 
                   "corr_coeff":corr_coeff}, outfile)
    
    return {"algorithm_name":algorithm_name, "dataset_name":dataset_name, 
                   "corr_coeff":corr_coeff}


if __name__=="__main__":
    corr_coef_lines = []
    for alg in ModelNamesMetrics:
        for dataset in DatasetNames:
            for trial_num in MODEL_TO_TRIAL_NUMS[alg.value][dataset.value]: 
                # print(alg.value, dataset.value, t)
                if trial_num < 400: continue
                trial_folder = f"{RESULTS_FOLDER}/analysis/trial{trial_num}"
                print(f"trial folder: {trial_folder}")

                json_line = get_corr_coef(trial_folder=trial_folder, algorithm_name=alg.value, 
                              dataset_name=dataset.value)
                json_line["trial_num"] = trial_num
                corr_coef_lines.append(json_line)
                break # only do first trial (later trials test a subset of the parameter range)

    with open("correlation_coefficients.jsonl", "w") as f:
        for item in corr_coef_lines:
            f.write(json.dumps(item) + '\n')


