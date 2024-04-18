# imports
import numpy as np
import pandas as pd
import os
import pdb
import constants
from collections import defaultdict
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from Trial_Setup_Utils import maybe_mkdir
import argparse

def generate_benchmark_dataframe(paths:dict):
    bds = None
    for dataset_name in paths:
        path = os.path.join(constants.RESULTS_FOLDER, "analysis", f"trial{paths[dataset_name]}")
        # load data
        # pdb.set_trace()
        df = pd.read_pickle(os.path.join(path, f"{dataset_name}.pkl"))

        # pandas stuff, assign parameter
        df["parameter"] = df["model_name"].apply(lambda string: int(string.split("_")[-1]))
        
        # assign means for capacity and expressivity
        for metric_type in ["algorithmic_capacity", "entropic_expressivity"]:
            metric_mean = metric_type + "_mean" # column names
            df[metric_mean] = df[metric_type].apply(lambda x: np.mean(x))

        # assign means for bias for each target set size
        algorithmic_bias_types = [title for title in df.columns if "algorithmic_bias" in title]
        for metric_type in algorithmic_bias_types:
            metric_mean = metric_type + "_mean" # column names
            df[metric_mean] = df[metric_type].apply(lambda x: np.mean(x))

        
        # make column for final value stored in bds
        final_value_column = []
        for i, _ in df.iterrows():
            # print()
            # expressivity and capacity
            values = {"algorithmic_capacity_mean": df["algorithmic_capacity_mean"][i], "entropic_expressivity_mean" : df["entropic_expressivity_mean"][i]}
            
            # bias values (for each target set)
            # algorithmic_bias_types = [title for title in df.columns if "algorithmic_bias" in title]
            for metric_type in algorithmic_bias_types:
                metric_mean = metric_type + "_mean" # column names
                values[metric_mean] = df[metric_mean][i] 
            
            final_value_column.append(values)
        
        df[f"&{dataset_name}"] = final_value_column

        temp_df =  df[[f"&{dataset_name}", "parameter"]]
        temp_df = temp_df.set_index("parameter")

        if bds is not None:
            bds = pd.merge(bds, temp_df, left_index=True, right_index=True)
        else:
            bds = temp_df
        
    current_columns = bds.columns

    # get rid of extra "&" in columns
    bds = bds.rename(columns={dataset_name:dataset_name[1:] for dataset_name in current_columns})

    # sort by parameter 
    return bds

def split_dataframe_dictionaries(df) -> dict:
    # This will store our resulting DataFrames
    result_dfs = {}

    
    # Iterate over the DataFrame's items
    for column in df:
        for index, value in df[column].items():
            # Proceed only if the value is a dictionary
            if isinstance(value, dict):
                # For each key in the dictionary, we create or update a DataFrame
                for key, val in value.items():
                    metric_name = key.split("_")[:-1]
                    metric_name = "_".join(metric_name)
                    # If the DataFrame for this key does not exist, create it
                    if metric_name not in result_dfs:
                        result_dfs[metric_name] = pd.DataFrame(index=df.index, columns=df.columns)
                    result_dfs[metric_name].at[index, column] = val
    # loop over dfs and sort by parameter value
    for key in result_dfs.keys():
        df = result_dfs[key]
        result_dfs[key] = df.sort_index(ascending=False)
        
    return result_dfs
    # # Convert result_dfs values from dict to list to maintain order
    # result_dfs_list = list(result_dfs.values())
    
    # # Replace None with np.nan
    # for result_df in result_dfs_list:
    #     result_df.fillna(value=np.nan, inplace=True)
    
    # return result_dfs_list

class BDS_Vector:
    def __init__(self):
        self.algorithmic_capacity = []
        self.entropic_expresstivity = []
        self.algorithmic_bias = []
        self.dataset_keys = {}

def bds_dataframe(bds):
    # change it to a dictionary by the index
    bds_index = bds.to_dict('index')
    bds_matrix = {}
    metrics = bds.iat[0,0].keys()
    # for each index, create three vectors
    for index in bds_index.keys():
        bds_row = bds_index[index]
        bds_vector = {}    
        for metric in metrics:    
            bds_vector[metric] = [bds_row[dataset][metric] for dataset in bds_row.keys()]
        bds_matrix[index] = bds_vector
    
    return bds_matrix

def visualize_bds(bds_dfs:dict[str, pd.DataFrame], saving_directory:str, metric:str, model_name:str = None): # import model_name and metrics
    # if metric is provided create a 2d vector for that metric based on matrix in bds_dfs[metric]

    df = bds_dfs[metric].astype(float) # cast to floats

    ax = sns.heatmap(df, annot=True, annot_kws={"size": 7})#, vmin=None, vmax=None, cmap=None, center=None, robust=False, 
                # annot=True, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', 
                # cbar=True, cbar_kws=None, cbar_ax=None, square=F
                # alse, xticklabels='auto', 
                # yticklabels='auto', mask=None, ax=None)
    # plt.show()
    
    plt.xticks(rotation=0, fontsize=5)
    plt.yticks(fontsize=7)

    plt.savefig(f"{saving_directory}/{model_name}_{metric}.png", dpi=400)
    
    plt.close()
    # ax.savefig(f"{model_name}/{metric}.png", dpi=400)

    return

# def save_bds(bds_dfs, saving_path):
#     if 
#     with open(f'{saving_path}.pkl', 'wb') as handle:
#         pickle.dump(bds_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs Benchmark Dataset Suite for a Model for all three methods")
    parser.add_argument('--model_name', type=str, help='provide both the name of model and parameter to vary, check constants.py', required=True)

    args = parser.parse_args()
    MODEL_NAME = args.model_name
    # save data to BDS/constants.ModelNamesMetrics.KNN_NEIGHBORS.value/
    # - "bds_dfs"
    # - paths.json (save the paths object)
    

    paths = constants.MODEL_TO_TRIAL_NUMS[MODEL_NAME]
    # paths = {constants.DatasetNames.EEG_EYE_STATE.value : "/data/big/erchen/inductive_orientation/analysis/trial100",
    #          constants.DatasetNames.SEMIRANDOM.value : "/data/big/erchen/inductive_orientation/analysis/trial109", 
    #          constants.DatasetNames.SHOPPER_INTENTION.value : "/data/big/erchen/inductive_orientation/analysis/trial118",
    #          constants.DatasetNames.SHOPPER_INTENTION_BALANCED.value : "/data/big/erchen/inductive_orientation/analysis/trial200"}
    
    saving_dir = maybe_mkdir(constants.RESULTS_FOLDER, f"heatmaps/{MODEL_NAME}")
    # saving_dir = os.join(RESULTS_FOLDER, "heatmaps", MODEL_NAME)
    bds_df = generate_benchmark_dataframe(paths)
    bds_dfs = split_dataframe_dictionaries(bds_df)

    
    # pickle bds dataframes
    with open(f'{saving_dir}/{MODEL_NAME}_BDF_dfs', 'wb') as handle:
        pickle.dump(bds_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    # get and download heatmap for each metric
    for metric in bds_dfs:
        visualize_bds(bds_dfs, saving_dir, metric, MODEL_NAME)

    # visualize_bds(bds_dfs, constants.ModelNamesMetrics.KNN_NEIGHBORS.value, constants.MetricNames.ALGORITHMIC_CAPACITY.value, saving_dir)



