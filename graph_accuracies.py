"""
Plots test and train accuracy scores given models/modelnum folder
"""
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import os
import re
from constants import RESULTS_FOLDER
from Trial_Setup_Utils import maybe_mkdir
from tqdm import tqdm
import argparse
import pdb



def get_single_accuracy(model_path, X_train, X_test, y_train, y_test):
    model = pickle.load(open(model_path, "rb"))
    
    # get test accuracy
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    # get train accuracy
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)

    return train_accuracy, test_accuracy
    
def parse_model_name(model_path):
    # Pattern to match an integer or decimal number at the end of the string
    pattern = r'\d+(\.\d+)?$'
    
    # Search for the pattern in the string
    match = re.search(pattern, model_path)
    
    # Extract and return the number if a match is found
    if match:
        return float(match.group())
    else:
        return None

"""Take a directory example models/modelnum folder, and then run inference on x_test, y_test."""
def get_accuracies(model_directory, X, y, saving_dir=None):
    # split data
    test_train_ratio = 0.20 # set for now
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_train_ratio, random_state=42)
    
    # initialize returns
    parameters = []
    test_accuracies_averages, train_accuracies_averages = [], []
    test_accuracies_std, train_accuracies_std = [], []
    counts = [] # number of samples
    
    for path_name in tqdm(os.listdir(model_directory)):
        # read model
        models_on_same_dataset = os.path.join(model_directory, path_name)
        temp_test_accuracies, temp_train_accuracies = [], []
        parameters.append(parse_model_name(models_on_same_dataset))
        
        for raw_model_path in os.listdir(models_on_same_dataset):
            model_path = os.path.join(models_on_same_dataset, raw_model_path)
            # get test and train accuracy
            train_accuracy, test_accuracy = get_single_accuracy(model_path, X_train, X_test, y_train, y_test)

            temp_test_accuracies.append(test_accuracy)
            temp_train_accuracies.append(train_accuracy)
        
        # update main lists of averages
        test_accuracies_averages.append(np.mean(temp_test_accuracies))
        train_accuracies_averages.append(np.mean(temp_train_accuracies))
        test_accuracies_std.append(np.std(temp_test_accuracies))
        train_accuracies_std.append(np.std(temp_train_accuracies))
        counts.append(len(temp_train_accuracies))
    
    if saving_dir:
        data = {"parameters" : parameters, "test mean" : test_accuracies_averages, \
                "train mean" : train_accuracies_averages, "test std" : test_accuracies_std, \
                "train std" : train_accuracies_std, "counts" : counts}
        df=pd.DataFrame(data)
        df = df.sort_values(by = "parameters")
        df.to_csv(os.path.join(saving_dir, "test_train_accuracies_CORRECTED.csv"))

    return df


def plot_accuracies(model_name:str, data_path:str, saving_dir): #model_name, model_parameters, test_accuracies_averages, train_accuracies_averages, test_accuracies_std, train_accuracies_std, saving_path):
    # read data
    df = pd.read_csv(data_path)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    counts = df["counts"]
    parameters = df["parameters"]
    test_mean = df["test mean"]
    train_mean = df["train mean"]

    test_sem = df["test std"]/np.sqrt(counts) # find standard error
    train_sem = df["train std"]/np.sqrt(counts)
    
    # Plot the test accuracies
    ax.plot(parameters, test_mean, label='Test Accuracies', marker='o')

    # Plot the train accuracies
    ax.plot(parameters, train_mean, label='Train Accuracies', marker='o')

    # Fill the "between" part with seaborn.fill_between
    ax.fill_between(parameters, test_mean - test_sem, test_mean + test_sem, alpha=0.3)
    ax.fill_between(parameters, train_mean - train_sem, train_mean + train_sem, alpha=0.3)

    # Set the labels and title
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Accuracy')
    ax.set_title(f"Test and Train accuracies for {model_name}")

    # Add a legend
    ax.legend()

    # Show the plot
    plt.savefig(os.path.join(saving_dir, "test_train_accuracies_CORRECTED.pdf"))
    plt.close()

def get_chars_before_first_number_re(s):
    """
    This function uses regular expressions to find all characters before the first number in a string.
    
    :param s: The input string
    :return: The substring before the first number
    """
    # Use regular expression to match any character until a digit is found
    match = re.search(r'^[^0-9]*', s)
    if match:
        return match.group()
    return s

def get_dataset(name):
    return f"{name}.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains models and Creates LDM')
    parser.add_argument('--model_num', type=int, help='the past experiment that generated models (only used during inference)', required=True)
    parser.add_argument('--dataset', type=str, help='provide the name of the dataset', required=True)
    
    args = parser.parse_args()
    trial_num = args.model_num
    dataset = get_dataset(args.dataset)
        
    # print(f"got arguments")
    # define inputs X and outputs y
    dataset_name = dataset.split(".")[0]
    dataset = os.path.join("datasets", dataset)
    data = pd.read_csv(dataset)
    X = data[data.columns[:-1]]
    X = X.iloc[:,:].values
    y = data[data.columns[-1]]
    y = y.values
    # print("Defined X and y")

    model_directory = f"{RESULTS_FOLDER}/models/model{trial_num}"
    # figure_saving_path = f"{RESULTS_FOLDER}"
    
    figure_saving_dir = maybe_mkdir(RESULTS_FOLDER, f"analysis/trial{trial_num}")
    # parameters, test_accuracies_averages, train_accuracies_averages, test_accuracies_std, train_accuracies_std \
    #     = get_accuracies(model_directory, X, y)
    
    train_accuracies_df = get_accuracies(model_directory, X, y, saving_dir=figure_saving_dir)
    
    # print("Called get_accuracies\n\n")
    model_name = get_chars_before_first_number_re(os.listdir(model_directory)[0])
    data_path = os.path.join(figure_saving_dir, "test_train_accuracies_CORRECTED.csv")
    plot_accuracies(model_name, data_path, figure_saving_dir)

        
    