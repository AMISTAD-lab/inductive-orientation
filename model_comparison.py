"""PCA, Heirarchical clustering, etc. methods to compare BDS vectors of different models"""

import constants
from typing import List, Tuple
import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pdb
import seaborn as sns

# load benchmark data for a single model (for one parameter)
def load_model_BDS_info(model_name:constants.ModelNamesMetrics, metric_type:constants.MetricNames) -> Tuple[pd.DataFrame, List[str]]:
    with open(f"{constants.RESULTS_FOLDER}/heatmaps/{model_name}/{model_name}_BDF_dfs.pkl", "rb") as file:
        d = pickle.load(file) # dictionary of dfs
    df = d[metric_type]
    parameters = df.index
    parameters = [f"{model_name}-{parameter}" for parameter in parameters]
    # df = df.drop(constants.PARAMETER, axis=1)
    return df, parameters

# load benchmark data for list of models (for one parameter)
def load_and_merge_bds(model_names:List[constants.ModelNamesMetrics], metric_type:constants.MetricNames) -> Tuple[pd.DataFrame, List[str]]:
    merged_bds, merged_labels = pd.DataFrame(), []
    for model_name in model_names:
        df, parameters = load_model_BDS_info(model_name, metric_type)
        merged_bds = merged_bds._append(df)
        merged_labels += parameters
    merged_bds = merged_bds.astype('float')
    return merged_bds, merged_labels

def perform_pca(data):
    # Standardizing the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # list of list of pca
    pca = PCA(n_components=2)  # Reduce dimensions to 2
    principal_components = pca.fit_transform(data_scaled)
    return principal_components

def plot_and_save(data: pd.DataFrame, labels: List[str], saving_directory:str):


    pca_df = pd.DataFrame(data=data, columns=['Principal Component 1', 'Principal Component 2'])
    pca_df['Label'] = labels

    # Plotting using seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Label', 
                    data=pca_df, palette='bright', s=100, marker='o', alpha=0.5).set_title('PCA of Dataset with Color Coded Labels')
    plt.grid(True)
    # plt.show()


    # plt.figure(figsize=(10, 8))
    # for i, (x, y) in enumerate(data):
    #     plt.scatter(x, y, c='red', marker='o')  # Plot the point
    #     plt.text(x + 0.05, y + 0.05, labels[i])  # Annotate the point

    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.title('PCA of Dataset with Labels')
    # plt.grid(True)
    # plt.show()
    plt.savefig(f"{saving_directory}/test.png", dpi=400)
    plt.close()


if __name__ == "__main__":
    # input: all the bds

    metric_type = constants.MetricNames.ALGORITHMIC_CAPACITY.value 
    
    models_names = [constants.ModelNamesMetrics.KNN_NEIGHBORS.value,
                    constants.ModelNamesMetrics.DECISION_TREE_MAX_DEPTH.value,
                    constants.ModelNamesMetrics.RANDOM_FOREST_DEPTH.value,
                    constants.ModelNamesMetrics.ADABOOST_ESTIMATORS.value,
                    constants.ModelNamesMetrics.C_SUPPORT_SVC_MAX_ITER.value,
                    constants.ModelNamesMetrics.LINEAR_SVC_MAX_ITER.value,
                    constants.ModelNamesMetrics.LOGISTIC_REGRESSION_MAX_ITER.value,
                    constants.ModelNamesMetrics.RANDOM_FOREST_ESTIMATORS.value]

    bds, labels = load_and_merge_bds(models_names, metric_type)
    
    model_type_label = [label.split("-")[0] for label in labels]
    # apply pca
    bds_2d = perform_pca(bds)
    
    
    # plot
    plot_and_save(bds_2d, model_type_label, saving_directory=".")
