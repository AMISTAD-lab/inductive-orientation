# code to graph trade-off bounds between algorithmic bias and entropic expressivity
import pandas as pd
import scipy
import os
import constants
import math
import pdb
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from functools import reduce
import numpy as np
import pickle
import sys
import pdb
from constants import RESULTS_FOLDER
import time
OG_DIR = os.getcwd()
LABELS = {
            "algorithmic_capacity":"Algorithmic Capacity (bits)",
            "entropic_expressivity":"Entropic Expressivity (bits)",
            # "algorithmic_bias":"Algorithmic Bias (%)"
            "algorithmic_bias_size_1":"Threshold 1 (%)",
            "algorithmic_bias_size_2":"Threshold 2 (%)",
            "algorithmic_bias_size_3":"Threshold 3 (%)",
            "algorithmic_bias_size_4":"Threshold 4 (%)",
            "algorithmic_bias_size_5":"Threshold 5 (%)",

            }

Z_SCORE = 1.96 # 1.96 is z-score for 95% confidence bound


PATH_TO_ANALYSIS = "../../../big/erchen/inductive_orientation/analysis"
OMEGA_SIZE = 32
def verify_bounds_df(folder_path, df_name):
    os.chdir(folder_path)
    df = pd.read_pickle(f"{df_name}.pkl")
    df["exp_cap_diff"] = df['entropic_expressivity']-df['algorithmic_capacity']
    df["param_value"] = df["model_name"].apply(lambda s: int(s.split('_')[-1]))
    df["bias_upper"] = df['exp_cap_diff'].apply(lambda h: math.sqrt(0.5*(math.log2(OMEGA_SIZE) - np.mean(h)))) # general bound from expressivity
    for k in range(1,6):
        df[f"exp_upper_{k}"] = df[f"algorithmic_bias_size_{k}"].apply(lambda bias: math.log2(OMEGA_SIZE) - 2*(np.mean(bias)**2))
        df[f"bias_upper_{k}"] = 1-1/32*(sum([math.comb(5,k_) for k_ in range(k,6)]))
    df.to_csv(f"df_w_bounds.csv")
    os.chdir(OG_DIR)
    return df

def individual_plot_mult_WITH_EXP_BOUND(df, dataset_name, model_name, xlabel, folder_path, export_name_prefix, plot_with_bounds=True, title=True):

    """
    x-label
    """

    tick_spread_factor = 3
    fontsize = 14
    legend_fontsize = 10
    sns.set(rc={'legend.fontsize': 10})

    print(os.listdir())
    os.chdir(folder_path)


    # throw away the part of the name that corresponds to the model trial number
    df["model_name"] = df["model_name"].apply(lambda x: x.split("_")[-1])

    # changes the model name to be a number respresenting its parameter
  
    df = df.sort_values("param_value")
    exp_cap_fig = plt.figure()
    exp_cap_fig_ax = exp_cap_fig.add_subplot()
    legend = {}
    for i, metric_type in enumerate(["algorithmic_capacity", "entropic_expressivity"]):
        metric_mean = metric_type + "_mean" # column names
        metric_std = metric_type + "_std"


        df[metric_mean] = df[metric_type].apply(lambda x: np.mean(x))
        df[metric_std] = df[metric_type].apply(lambda x: np.std(x)/np.sqrt(np.size(x))) # actually the SEM 
        sns.lineplot(ax=exp_cap_fig_ax, data=df, x="param_value", y=metric_mean, label=LABELS[metric_type])
        exp_cap_fig_ax.fill_between(df["param_value"], df[metric_mean] - Z_SCORE*df[metric_std], df[metric_mean] + Z_SCORE*df[metric_std], alpha=0.2)
    
    # PLOT UPPER BOUNDS
    if plot_with_bounds:
        for k in range(1,6):
            sns.lineplot(ax=exp_cap_fig_ax, data=df,x="param_value",y=f"exp_upper_{k}",label=f"Threshold {k} Expressivity Bound")
    # END ADDITION
    
    exp_cap_fig_ax.legend(handles=exp_cap_fig_ax.get_lines()[:],fontsize=legend_fontsize)
    if title: exp_cap_fig_ax.set_title(f"{xlabel} vs. Algorithmic Capacity & Entropic Expressivity (95% interval) \n{model_name} on {dataset_name}")
    exp_cap_fig_ax.set_xlabel(xlabel,fontsize=fontsize)
    exp_cap_fig_ax.set_ylabel("Bits",fontsize=fontsize)

    num_ticks = min(20, max(df["param_value"])-min(df["param_value"]))
    tick_distance = (max(df["param_value"])-min(df["param_value"]))//(num_ticks)
    plt.xticks(np.arange(min(df["param_value"]), max(df["param_value"])+1, tick_distance*tick_spread_factor), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"{export_name_prefix}_exp_cap_sem.pdf")
    plt.close()

#   Plot algorithmic bias
    alg_fig = plt.figure()
    alg_fig_ax = alg_fig.add_subplot()
    algorithmic_bias_types = [title for title in df.columns if "algorithmic_bias" in title]
    for i, metric_type in enumerate(algorithmic_bias_types):
        metric_mean = metric_type + "_mean" # column names
        metric_std = metric_type + "_std"

        df[metric_mean] = df[metric_type].apply(lambda x: np.mean(x))
        df[metric_std] = df[metric_type].apply(lambda x: np.std(x)/np.sqrt(np.size(x)))
        sns.lineplot(ax=alg_fig_ax, data=df, x="param_value", y=metric_mean, label=LABELS[metric_type])
        alg_fig_ax.fill_between(df["param_value"], df[metric_mean] - Z_SCORE*df[metric_std], df[metric_mean] + Z_SCORE*df[metric_std], alpha=0.2)
    # PLOT BIAS UPPER BOUNDS
    if plot_with_bounds:
        # sns.lineplot(ax=alg_fig_ax, data=df,x="param_value",y="bias_upper",label="Bound")
        for k in range(1,6):
            sns.lineplot(ax=alg_fig_ax, data=df,x="param_value",y=f"bias_upper_{k}",label=f"Threshold {k} Maximum")
    # END ADDITION
    alg_fig_ax.legend(handles=alg_fig_ax.get_lines()[:],fontsize=legend_fontsize)
    if title: alg_fig_ax.set_title(f"{xlabel} vs. Algorithmic Bias (95% interval) \n{model_name} on {dataset_name}")
    alg_fig_ax.set_xlabel(xlabel,fontsize=fontsize)
    alg_fig_ax.set_ylabel("Bias",fontsize=fontsize)
    num_ticks = min(20, max(df["param_value"])-min(df["param_value"]))
    tick_distance = (max(df["param_value"])-min(df["param_value"]))//(num_ticks)
    plt.xticks(np.arange(min(df["param_value"]), max(df["param_value"])+1, tick_distance*tick_spread_factor), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"{export_name_prefix}_alg_sem.pdf")
    plt.close()

    try:
        # now plot test and train accuracies
        df_acc = pd.read_csv("test_train_accuracies_CORRECTED.csv")
        acc_fig = plt.figure()
        acc_fig_ax = acc_fig.add_subplot()
        sns.lineplot(ax=acc_fig_ax,data=df_acc,x="parameters", y="train mean", label="Train Accuracy")
        sns.lineplot(ax=acc_fig_ax,data=df_acc,x="parameters", y="test mean", label="Test Accuracy")
        plt.ylabel("Accuracy", fontsize=fontsize)
        tick_distance = (max(df_acc["parameters"])-min(df_acc["parameters"]))//(num_ticks)
        plt.xticks(np.arange(min(df_acc["parameters"]), max(df_acc["parameters"])+1, tick_distance*tick_spread_factor), fontsize=fontsize)
        #tick_distance = (max(df_acc["test mean"])-min(df_acc["train mean"]))//5#(num_ticks)
        plt.yticks(fontsize=fontsize)
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.savefig(f"{export_name_prefix}_accuracies.pdf")
        plt.close()
    except Exception:
        print("NO ACCURACY PLOT")



    os.chdir(OG_DIR)

if __name__=="__main__":
    # os.chdir(PATH_TO_ANALYSIS)
    


    for model in constants.MODEL_TO_TRIAL_NUMS:
        if model != "Nearest_Centroid_shrink_threshold":
            continue
        for dataset in constants.MODEL_TO_TRIAL_NUMS[model]:
            for trial_num in constants.MODEL_TO_TRIAL_NUMS[model][dataset]:

                os.chdir(OG_DIR)
                # time.sleep(1)
                #if model != constants.ModelNamesMetrics.ADABOOST_ESTIMATORS.value: continue
                # if trial_num != 427: continue # temp for testing
                try:
                    folder_path = f"{PATH_TO_ANALYSIS}/trial{trial_num}"
                    df_path = f"{PATH_TO_ANALYSIS}/trial{trial_num}/{dataset}.pkl"
                    #os.chdir(folder_path)
                    df = verify_bounds_df(folder_path,dataset)
                    # os.chdir("./")
                    individual_plot_mult_WITH_EXP_BOUND(df=df,
                                                        dataset_name=constants.READABLE_DATASETS[dataset],
                                                        xlabel=constants.MODEL_TO_X_PARAM[model],
                                                        model_name=constants.READABLE_MODELS[model], 
                                                        folder_path= folder_path, 
                                                        export_name_prefix="_bounds", 
                                                        plot_with_bounds=True) 
                    individual_plot_mult_WITH_EXP_BOUND(df=df,
                                                        dataset_name=constants.READABLE_DATASETS[dataset],
                                                        xlabel=constants.MODEL_TO_X_PARAM[model],
                                                        model_name=constants.READABLE_MODELS[model], 
                                                        folder_path= folder_path, 
                                                        export_name_prefix="_bounds_no_title", 
                                                        plot_with_bounds=True,
                                                        title=False) 
                    individual_plot_mult_WITH_EXP_BOUND(df=df,
                                                        dataset_name=constants.READABLE_DATASETS[dataset],
                                                        xlabel=constants.MODEL_TO_X_PARAM[model],
                                                        model_name=constants.READABLE_MODELS[model], 
                                                        folder_path= folder_path, 
                                                        export_name_prefix="_no_bounds", 
                                                        plot_with_bounds=False,
                                                        title=False) 
                except Exception:
                    print(F"EXCEPTION WITH TRIAL {trial_num}, dataset: {dataset}, model: {model}")

    
    