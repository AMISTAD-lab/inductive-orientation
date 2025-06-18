# Small file for determining correlation coefficient between (expressivity-capacity)/(test-train)

from constants import MODEL_TO_TRIAL_NUMS, RESULTS_FOLDER, ModelNamesMetrics, DatasetNames
import pandas as pd
import json
import numpy as np
import pdb
import math
import scipy.stats

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
    # pdb.set_trace()

    # sort df_exp_cap by parameters
    df_exp_cap["parameter"] = df_exp_cap["model_name"].apply(lambda text: int(text.split("_")[-1]))
    df_exp_cap.sort_values(by="parameter", inplace=True)

    # get expresisity capacity averages
    df_exp_cap["entropic_expressivity_mean"] = df_exp_cap["entropic_expressivity"].apply(np.mean)
    df_exp_cap["algorithmic_capacity_mean"] = df_exp_cap["algorithmic_capacity"].apply(np.mean)
    # pdb.set_trace()
    df_exp_cap["e_c_differences"] = df_exp_cap.apply(lambda row: np.subtract(row["entropic_expressivity"], row["algorithmic_capacity"]), axis=1)

    exp_cap_diff = np.array(df_exp_cap["entropic_expressivity_mean"] - df_exp_cap["algorithmic_capacity_mean"])
    

    exp_cap_diff_2d = np.stack(df_exp_cap["e_c_differences"].values)
    # train_test_diff_2d = np.stacked(df_exp_cap["e_c_differences"].values)
    # print("test train:",test_train_diff)
    # print("exp cap:",exp_cap_diff)
    # get correlation coefficient
    # corr_coeffs = []
    # for i in range(len(df_exp_cap["e_c_differences"])):
    #     exp_cap_diff = df_exp_cap["e_c_differences"].apply(lambda arr: arr[i])
    #     try:
            
    #         corr_coeff = np.corrcoef(x=test_train_diff+0.001, y=exp_cap_diff)[0,1]
    #         corr_coeffs.append(corr_coeff)
    #     except Exception as e:
    #         print(f"exception for {i}, {algorithm_name}, {dataset_name}, {trial_folder}")
        

    
    try:
        corr_coeff, p_val = np.corrcoef(x=exp_cap_diff, y=test_train_diff)[0,1], 0#scipy.stats.spearmanr(exp_cap_diff[:5], test_train_diff[:5], axis=1,nan_policy='propagate')#exp_cap_diff, test_train_diff, axis=0, nan_policy='propagate') #np.corrcoef(x=exp_cap_diff, y=test_train_diff)[0,1]
    except:
        corr_coeff, p_val = np.nan, p_val
        print("BRU")

    # find corr. coefs of train accuracy against all 
    # alg_column_names = ['algorithmic_bias_size_1','algorithmic_bias_size_2','algorithmic_bias_size_3',
    #                     'algorithmic_bias_size_4', 'algorithmic_bias_size_5']
    # alg_bias_corrs = {} 
    # for alg_bias_col in alg_column_names:
    #     # get alg average
    #     df_exp_cap[f"{alg_bias_col}_mean"] = df_exp_cap[alg_bias_col].apply(np.mean)

    #     alg_corr_coeff = np.corrcoef(x=df_exp_cap[f"{alg_bias_col}_mean"], y=test_train_diff)[0,1]
    #     alg_bias_corrs[f"{alg_bias_col}_corr_coeff"] = alg_corr_coeff

    json_dict = {"algorithm_name":algorithm_name, "dataset_name":dataset_name, 
                   "exp_cap_corr_coeff":corr_coeff , "exp_cap_corr_coeff_p":p_val} #| alg_bias_corrs
    
    # json_dict = {"algorithm_name":algorithm_name, "dataset_name":dataset_name, 
    #                "exp_cap_corr_coeff_ave":np.mean(corr_coeffs), "exp_cap_corr_coeff_std":np.std(corr_coeffs)} #| alg_bias_corrs

    # make json
    with open(f"{trial_folder}/correlation_coefficient_pearson_dec_tree.json", "w") as outfile:
        print(json_dict)
        json.dump(json_dict, outfile)
    
    return json_dict


if __name__=="__main__":
    # pass
    corr_coef_lines = []
    corr_df = pd.DataFrame() # base dataframe for summarizing corr coefficients

    for alg in MODEL_TO_TRIAL_NUMS:
        for dataset in MODEL_TO_TRIAL_NUMS[alg]:
            for trial_num in MODEL_TO_TRIAL_NUMS[alg][dataset]:
                # if alg != "Decision_Tree_max_depth":
                #     continue
                # if trial_num != 301:
                #      continue
                # # if trial_num != 456:
                # #     continue 
                # # print(alg.value, dataset.value, t)
                if trial_num < 512: continue
                # # try:
                to_break = False
                trial_folder = f"{RESULTS_FOLDER}/analysis/trial{trial_num}"
                print(f"trial folder: {trial_folder}")

                try:
                    json_line = get_corr_coef(trial_folder=trial_folder, algorithm_name=alg, 
                                dataset_name=dataset)
                    # to_break = True

                except: 
                    json_line = {"algorithm_name":alg, "dataset_name":dataset}
                    print(f"error with {trial_num}")
                json_line["trial_num"] = trial_num

                corr_coef_lines.append(json_line)
                # if to_break:
                #     break
                 # only do first trial (later trials test a subset of the parameter range)
                # except:
                #     print("Issue with trial num", trial_num)

    with open("correlation_coefficients_pearson_dec_tree", "w") as f:
        for item in corr_coef_lines:
            f.write(json.dumps(item) + '\n')


