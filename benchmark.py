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

# generate 
def load_merge_dfs(dataset_name:str, trial_nums:list[int]):
    main_df = pd.DataFrame()

    # accumulate dfs
    for trial_num in trial_nums:
        path = os.path.join(constants.RESULTS_FOLDER, "analysis", f"trial{trial_num}")
        df = pd.read_pickle(os.path.join(path, f"{dataset_name}.pkl"))

        # add parameter column
        df["parameter"] = df["model_name"].apply(lambda string: int(string.split("_")[-1]))
        
        # main_df = pd.concat([main_df, df], ignore_index=False)
        main_df = main_df._append(df,ignore_index=True)

    # get rid of parameter duplicates
    main_df=main_df.drop_duplicates(subset=['parameter'], keep="first")
    
    # sort values
    return main_df
    

def generate_benchmark_dataframe(paths:dict):
    bds = None
    for dataset_name in paths:
        # import pdb
        # pdb.set_trace()
        df = load_merge_dfs(dataset_name, paths[dataset_name])
        # pdb.set_trace()
        # path = os.path.join(constants.RESULTS_FOLDER, "analysis", f"trial{paths[dataset_name][0]}")
        # # load data
        # # pdb.set_trace()
        # df = pd.read_pickle(os.path.join(path, f"{dataset_name}.pkl"))

        # pandas stuff, assign parameter
        # df["parameter"] = df["model_name"].apply(lambda string: int(string.split("_")[-1]))
        
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

def split_dataframe_dictionaries(df:pd.DataFrame) -> dict:
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
    # import pdb
    # pdb.set_trace()
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


### benchmark utils

def load_bds_dfs_from_pickle(saving_path: str) -> dict:
    with open(saving_path, "rb") as input_file:
        bds_dfs = pickle.load(input_file)
    return bds_dfs

def analyze_bds(bds: pd.DataFrame, aggregate_fn, sort: bool) -> pd.Series:
    """
    Takes a bds which looks like a matrix and creates a vector, each value is the aggregate
    result of a row in the DataFrame, across various datasets

    Inputs:
        bds: a Pandas Dataframe. The Columns are names of datasets and rows are the paramter values
        aggregate_fn: This is an higher order functin (such as max)
        sort: sorts in ascending order, default is no sorting
    
    Outputs:
        aggregate_vals: a Pandas Series containing a value corresponding to each parameter of a model
    """
    aggregate_vals = bds.apply(aggregate_fn, axis=1)
    if sort:
        aggregate_vals = aggregate_vals.sort_values() # key = lambda x: -1*x if reverse
    return aggregate_vals
    

# loads multiple datasets and then calls analyze_bds on all of them
# and then performs the sorting

def rank_models(bds_folders, aggregate_fn, saving_path, aggregate_name):
    # an empty series (should be a dictionary, key will be metric type, value will be series)
    series_dict = defaultdict(pd.Series) # str: pd.Series
    # loop through bds_folders
    # bds_folders = '/data/big/erchen/inductive_orientation/heatmaps'
    for model_name in os.listdir(bds_folders):
        bds_dfs_saving_path = os.path.join(bds_folders, model_name, f'{model_name}_BDF_dfs.pkl')

        # calls load_bds_dfs_from_pickle
        bds_dfs = load_bds_dfs_from_pickle(bds_dfs_saving_path)
        
        # loop through the dfs aka different metrics
        for metric in bds_dfs.keys():
            bds = bds_dfs[metric]
            # calls analyze_bds
            aggregate_vals = analyze_bds(bds, aggregate_fn, sort=False)

            # appends the model name to the series indexes
            new_index = [f'{model_name}-{parameter}' for parameter in aggregate_vals.index]
            aggregate_vals.index = new_index
            
            # appends the above series to the empty series corresponding to the right metric type
            if metric in series_dict.keys():
                series_dict[metric] = series_dict[metric]._append(aggregate_vals)
            else:
                series_dict[metric] = aggregate_vals
            
    # loop through the metrics aka dictionary keys
    for metric in series_dict.keys():
        current_series = series_dict[metric]

        # order the current series 
        current_series = current_series.sort_values()
        
        # prints out as a csv that we can read
        current_series.to_csv(os.path.join(saving_path, f'{metric}_{aggregate_name}.csv'), index=True)


# load the bds 
#   def load_bds_dfs_from_pickle(saving_location: str) -> dict:
# analyze it
#   def load_bds_dfs_from_pickle(saving_location: str) -> dict:
# aggregate across the datasets
# do a ranking


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs Benchmark Dataset Suite for a Model for all three methods")
    parser.add_argument('--model_name', type=str, help='provide both the name of model and parameter to vary, check constants.py', required=True)

    # parse inputs
    args = parser.parse_args()
    MODEL_NAME = args.model_name
    paths = constants.MODEL_TO_TRIAL_NUMS[MODEL_NAME] # actually trial number
    

    saving_dir = maybe_mkdir(constants.RESULTS_FOLDER, f"heatmaps/{MODEL_NAME}")
    
    # generate the BDS
    bds_df = generate_benchmark_dataframe(paths)
    bds_dfs = split_dataframe_dictionaries(bds_df)

    
    # pickle bds dataframes
    with open(f'{saving_dir}/{MODEL_NAME}_BDF_dfs.pkl', 'wb') as handle:
        pickle.dump(bds_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    # get and download heatmap for each metric
    for metric in bds_dfs:
        visualize_bds(bds_dfs, saving_dir, metric, MODEL_NAME)

    # visualize_bds(bds_dfs, constants.ModelNamesMetrics.KNN_NEIGHBORS.value, constants.MetricNames.ALGORITHMIC_CAPACITY.value, saving_dir)


