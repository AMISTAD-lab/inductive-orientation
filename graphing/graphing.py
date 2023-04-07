import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from functools import reduce
import numpy as np

def all_plots(path, analysis_type, saving_path, plot_info):
    df = pd.read_csv(path)
    fig = plt.figure()
    ax = fig.add_subplot()
    legend_dict = {}
    order = 1
    model_sequence = ["KNN", "DecisionTree", "RandomForest"]
    for model in model_sequence:
        if reduce(lambda x, y: x or y, list(map(lambda x: model in x, df["model_name"]))):
            temp_df = df[list(map(lambda x: model in x, df["model_name"]))]
            temp_df = temp_df[temp_df["model_name"].apply(lambda x: model in x)]
            temp_df["model_name"] = temp_df["model_name"].apply(lambda x: x.split("_")[1])
            temp_df["model_name"] = temp_df["model_name"].apply(lambda x: re.split("(\d+)",x)[-2])
            temp_df["model_name"] = temp_df["model_name"].astype("int8")
            sns.lineplot(ax=ax, data=temp_df, x="model_name", y=analysis_type)
            legend_dict[model] = order
            order += 1
            plt.xticks(temp_df["model_name"])
    df["model_name"] = df["model_name"].apply(lambda x: x.split("_")[1])
    for model in df["model_name"]:
        if "KNN" not in model and "DecisionTree" not in model and "RandomForest" not in model:
            temp_df = df[df["model_name"] == model]
            ax.plot(range(1,22),list(temp_df[analysis_type])*21)
            # ax.axhline(temp_df[analysis_type])
            # sns.lineplot(ax=ax, data = temp_df, y=analysis_type)
            legend_dict[model] = order
            order += 1
    
    ax.legend(legend_dict)
    ax.set_title(plot_info["title"])
    ax.set_xlabel(plot_info["xlabel"])
    ax.set_ylabel(plot_info["ylabel"])
    plt.savefig(saving_path)
    plt.show()
    

def individual_plot(path, saving_path, model_name, plot_info):
    df = pd.read_csv(path)
    df = df[df["model_name"].apply(lambda x: model_name in x)]
    df["model_name"] = df["model_name"].apply(lambda x: x.split("_")[1])
    df["model_name"] = df["model_name"].apply(lambda x: re.split('(\d+)',x)[-2])
    df["model_name"]=df["model_name"].astype("int8")
    fig = plt.figure()
    ax1 = fig.add_subplot()
    sns.lineplot(ax=ax1, data=df, x="model_name", y="algorithmic_capacity")
    sns.lineplot(ax=ax1, data=df, x="model_name", y="entropic_expressivity")
    sns.lineplot(ax=ax1, data=df, x="model_name", y="algorithmic_bias")
    ax1.legend({"algorithmic_capacity":1, "entropic_expressivity":2, "algorithmic_bias":3})
    ax1.set_title(plot_info["title"])
    ax1.set_xlabel(plot_info["xlabel"])
    ax1.set_ylabel(plot_info["ylabel"])
    plt.xticks(df["model_name"])
    plt.savefig(saving_path)
    plt.show()


def individual_plot_mult(path, saving_path, model_name, plot_info):
    # might need to change to
    df = pd.read_csv(path)
    # pickle.load(path)
    df = df[df["model_name"].apply(lambda x: model_name in x)] # row of table corresponding to the model
    df["model_name"] = df["model_name"].apply(lambda x: x.split("_")[1])
    df["model_name"] = df["model_name"].apply(lambda x: re.split('(\d+)',x)[-2])
    df["model_name"]=df["model_name"].astype("int8")
    fig = plt.figure()
    ax1 = fig.add_subplot()
    legend = {}
    for i, metric_type in enumerate(["algorithmic_capacity", "entropic_expressivity", "algorithmic_bias"]):
        metric_mean = metric_type + "_mean"
        metric_std = metric_type + "_std"
        df[metric_mean] = df[metric_type].apply(lambda x: np.mean(x))
        df[metric_std] = df[metric_type].apply(lambda x: np.std(x))
        sns.lineplot(ax=ax1, data=df, x="model_name", y=metric_type)
        ax1.errorbar(df.index, df[metric_mean], yerr=df[metric_std], fmt='-o')
        legend[metric_type] = i+1
    ax1.legend(legend)
    ax1.set_title(plot_info["title"])
    ax1.set_xlabel(plot_info["xlabel"])
    ax1.set_ylabel(plot_info["ylabel"])
    plt.xticks(df["model_name"])
    plt.savefig(saving_path)
    plt.show()
# path_to_log = "./../trial21_target8.csv"
# df = pandas.read_csv(path_to_log)
# df = df[df["model_name"].apply(lambda x: "RandomForest" in x)]
# df["model_name"] = df["model_name"].apply(lambda x: x.split("_")[1])
# df["model_name"] = df["model_name"].apply(lambda x: re.split('(\d+)',x)[-2])
# df["model_name"]=df["model_name"].astype("int8")
# fig = plt.figure()
# ax1 = fig.add_subplot()
# sns.lineplot(ax=ax1, data=df, x="model_name", y="algorithmic_capacity")
# sns.lineplot(ax=ax1, data=df, x="model_name", y="entropic_expressivity")
# sns.lineplot(ax=ax1, data=df, x="model_name", y="algorithmic_bias")
# ax1.legend({"algorithmic_capacity":1, "entropic_expressivity":2,"algorithmic_bias":3})
# ax1.set_title("Random Forest")
# ax1.set_xlabel("number of estimators")
# ax1.set_ylabel("")
# plt.xticks(df["model_name"])

if __name__ == "__main__":
    # FILE_NAME = "./../trial21_target8.csv"
    # BASE_SAVING_PATH = "./../plots/trial21_target8_"

    # KNN
    # plot_info = {"title": "K Nearest Neighbors", "xlabel": "number of neighbors", "ylabel":""}
    # individual_plot("./../trial41_target8.csv", "./../trial41_target8_KNN.pdf", "KNN", plot_info)
    
    # Decision Tree
    # plot_info = {"title": "Decision Trees", "xlabel": "max depth", "ylabel":""}
    # individual_plot("./../trial33_target8.csv", "./../trial33_target8_DecisionTree.pdf", "DecisionTree", plot_info)

    # Random Forest
    plot_info = {"title": "Random Forests", "xlabel": "number of estimators", "ylabel":""}
    individual_plot("./../trial36_target8.csv", "./../trial36_target8_RandomForest.pdf", "RandomForest", plot_info)

    # plot_info = {"title": "All Models on Shoppers Intention Dataset", "xlabel": "", "ylabel":""}
    # all_plots("./../trial19_target8.csv", "algorithmic_capacity", "./../plots/trial19_target8_capacity.pdf", plot_info)
    # all_plots("./../trial19_target8.csv", "entropic_expressivity", "./../plots/trial19_target8_expressivity.pdf", plot_info)
    # all_plots("./../trial19_target8.csv", "algorithmic_bias", "./../plots/trial19_target8_bias.pdf", plot_info)

    # plot_info = {"title": "All Models on Random Dataset", "xlabel": "", "ylabel":""}
    # all_plots("./../trial21_target8.csv", "algorithmic_capacity", "./../plots/trial21_target8_capacity.pdf", plot_info)
    # all_plots("./../trial21_target8.csv", "entropic_expressivity", "./../plots/trial21_target8_expressivity.pdf", plot_info)
    # all_plots("./../trial21_target8.csv", "algorithmic_bias", "./../plots/trial21_target8_bias.pdf", plot_info)

    # plot_info = {"title": "All Models on EEG Eye State Dataset", "xlabel": "", "ylabel":""}
    # all_plots("./../trial27_target8.csv", "algorithmic_capacity", "./../plots/trial27_target8_capacity.pdf", plot_info)
    # all_plots("./../trial27_target8.csv", "entropic_expressivity", "./../plots/trial27_target8_expressivity.pdf", plot_info)
    # all_plots("./../trial27_target8.csv", "algorithmic_bias", "./../plots/trial27_target8_bias.pdf", plot_info)
