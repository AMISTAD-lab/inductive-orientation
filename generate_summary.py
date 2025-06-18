"""For each dataset, generate summary statistics on parameter value and metric values when maximums for bias (threshold 4)"""

import constants
import os
import pandas as pd
import statistics 
import pdb
import numpy as np
import re

import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.metrics import r2_score
# from collections import defaultdict
from sklearn.cluster import KMeans,HDBSCAN

import umap


def extract_algorithm_key(model_name):
    """
    Extracts the algorithm name part from a model string.
    e.g. 'model_107_X_Y_Z_181' -> 'X_Y_Z'
          'model_108_X_Y_96'   -> 'X_Y'
    
    Args:
        model_name (str): input string with format like 'model_XXX_algo_estimators_YYY'
    
    Returns:
        str: extracted algorithm name (e.g., 'X_Y_Z'), or None if no match found
    """
    match = re.match(r'model_\d+_([A-Za-z0-9_]+)_\d+', model_name)
    model = match.group(1)
    return constants.READABLE_MODELS[model]#match.group(1) if match else None

def load_trial_data(model_name:constants.ModelNamesMetrics, dataset_name:constants.DatasetNames):
    trial_num = constants.MODEL_TO_TRIAL_NUMS[model_name][dataset_name][0]
    location = f"{constants.RESULTS_FOLDER}/analysis/trial{trial_num}/{dataset_name}.pkl"
    df = pd.read_pickle(location)
    return df


def get_expcap_corr_w_metric(model_name:constants.ModelNamesMetrics, dataset_name:constants.DatasetNames):
    "Get dict of linear regression curves of parameter with difference between average exp and cap(log scale and not log scaled)"
    # pdb.set_trace()
    df = load_trial_data(model_name, dataset_name)
    df["param"] = df['model_name'].apply(lambda s:  int(re.findall(r'\d+', s)[-1]))
    df = df.sort_values(by='param', ascending=True)

    exp_avgs = np.array(df['entropic_expressivity'].apply(lambda l: statistics.mean(l)))
    cap_avgs = np.array(df['algorithmic_capacity'].apply(lambda l: statistics.mean(l)))
    parameters = np.array(df['model_name'].apply(lambda s:  int(re.findall(r'\d+', s)[-1])))

    exp_cap_diffs = exp_avgs - cap_avgs

    x_data = parameters
    y_data = cap_avgs#exp_avgs#exp_cap_diffs

    # Define exponential decay function
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Fit the curve
    popt, pcov = curve_fit(exp_decay, x_data, y_data, p0=(5, 1, 1))


    # metrics
    y_pred = exp_decay(x_data, *popt)

    r2 = r2_score(y_data, y_pred)
    print(f"R² of the fit: {r2:.4f}")

    rmse = np.sqrt(mean_squared_error(y_data, y_pred))
    print(f"RMSE of the fit: {rmse:.4f}")

    # Extract parameter values and standard errors
    a_fit, b_fit, c_fit = popt
    a_err, b_err, c_err = np.sqrt(np.diag(pcov))

    print(f"Fitted parameters with standard errors:")
    print(f"a = {a_fit:.3f} ± {a_err:.3f}")
    print(f"b = {b_fit:.3f} ± {b_err:.3f}")
    print(f"c = {c_fit:.3f} ± {c_err:.3f}")

 

    sns.set_theme(style="darkgrid")
    
    plt.figure(figsize=(8, 5))

    plt.scatter(x_data, y_data, label='Data', alpha=0.6, edgecolor='k')

    plt.plot(x_data, exp_decay(x_data, *popt), 'r-', label=(
        f'Fit: a={a_fit:.2f}±{a_err:.2f}, '
        f'b={b_fit:.2f}±{b_err:.2f}, '
        f'c={c_fit:.2f}±{c_err:.2f}\n'
        f'R²: {r2:.4f}, '
        f'RMSE: {rmse:.4f}'
    ))
    plt.xlabel('Number of Estimators (n)')
    plt.ylabel("Algorithmic Capacity")#r"$\mathbb{E}_\mathcal{D}[H(\overline{\mathbf{P}}_{F})]$")#'Expressivity - Capacity')
    # plt.title('Exponential Decay Fit with Errors')

    # plt.legend()
    plt.ylim(0, 5)

    plt.tight_layout()

    plt.savefig(f"random_forest_exponential_fits/{dataset_name}_cap_only_decay.png")

    #return f'{a_fit:.2f}±{a_err:.2f}', f'{b_fit:.2f}±{b_err:.2f}', f'{c_fit:.2f}±{c_err:.2f}', f'{r2:.4f}', f'{rmse:.4f}'
    return a_fit, b_fit, c_fit, r2, rmse
from sklearn.linear_model import LinearRegression

def get_expcap_corr_w_metric_l(model_name: constants.ModelNamesMetrics, dataset_name: constants.DatasetNames, offset=0):
    "Get dict of linear regression curves of parameter with difference between average exp and cap (log scale on y)"
    df = load_trial_data(model_name, dataset_name)
    df["param"] = df['model_name'].apply(lambda s: int(re.findall(r'\d+', s)[-1]))
    df = df.sort_values(by='param', ascending=True)

    exp_avgs = np.array(df['entropic_expressivity'].apply(lambda l: statistics.mean(l)))
    cap_avgs = np.array(df['algorithmic_capacity'].apply(lambda l: statistics.mean(l)))
    parameters = np.array(df['param'])

    exp_cap_diffs = exp_avgs - cap_avgs

    x_data = parameters.reshape(-1, 1)
    # pdb.set_trace()

    y_data = exp_cap_diffs - min(exp_cap_diffs)  # Subtract offset to shift the curve

    # Make sure y_data is positive for log transform, add a small epsilon if needed
    epsilon = 1e-7
    y_data_positive = y_data + epsilon
    if np.any(y_data_positive <= 0):
        raise ValueError("y_data contains non-positive values; log transform not possible.")

    log_y = np.log(y_data_positive).reshape(-1, 1)

    # Fit linear regression: log(y) = m*x + b
    lin_reg = LinearRegression()
    lin_reg.fit(x_data, log_y)

    m = lin_reg.coef_[0][0]
    b = lin_reg.intercept_[0]

    # Predicted values on log scale
    log_y_pred = lin_reg.predict(x_data)
    y_pred = np.exp(log_y_pred).flatten()

    # Metrics on original scale y vs predicted y
    r2 = r2_score(y_data, y_pred)
    rmse = np.sqrt(mean_squared_error(y_data, y_pred))

    print(f"Linear regression on log(y) fit:")
    print(f"Slope (m) = {m:.4f}")
    print(f"Intercept (b) = {b:.4f}")
    print(f"R² (on original y) = {r2:.4f}")
    print(f"RMSE (on original y) = {rmse:.4f}")

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 5))

    plt.scatter(x_data, y_data, label='Data', alpha=0.6, edgecolor='k')
    plt.plot(x_data, y_pred, 'r-', label=(
        f'Fit: y = exp({m:.2f}*x + {b:.2f})\n'
        f'R²: {r2:.4f}, RMSE: {rmse:.4f}'
    ))

    plt.xlabel('Number of Estimators (n)')
    plt.ylabel('Expressivity - Capacity')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"random_forest_linear_log_fits/{dataset_name}_rf_log_linear.png")

    return f'{m:.4f}', f'{b:.4f}', f'{r2:.4f}', f'{rmse:.4f}'

def generate_summary_csv(n=5, selection_method="even_spaced"):
    # pdb.set_trace()
    summary = {}#defaultdict(lambda: [])
    tree_models = []#[constants.ModelNamesMetrics.RANDOM_FOREST_DEPTH.value.upper(), constants.ModelNamesMetrics.RANDOM_FOREST_ESTIMATORS.value.upper(),
                  #  constants.ModelNamesMetrics.DECISION_TREE_MAX_DEPTH.value.upper(), constants.ModelNamesMetrics.ADABOOST_ESTIMATORS.value.upper()]
    # pdb.set_trace()
    for model in constants.ModelNamesMetrics:
        if model.name == "GAUSSIAN_PROCESS_MAX_ITER":# or model.name in tree_models: 
            print(f"skip:{model.name}")
            continue
        flag = False
        
        for dataset in constants.DatasetNames:
            if dataset.value == "Shopper_Intention_Balanced":
                continue

            # pdb.set_trace()
            df = load_trial_data(model.value, dataset.value)
            # try:

            df["param"] = df['model_name'].apply(lambda s: int(re.findall(r'\d+', s)[-1]))
            # except Exception:
        
            #     df["param"] = list(df['model_name'].apply(lambda s: int(re.findall(r'\d+', s)[-1])))

            df = df.sort_values(by='param', ascending=True)

            # get top 10
            if selection_method=="highest":
                top5 = df.nlargest(n, 'param')
            elif selection_method=="even_spaced":
                indices = np.linspace(1, len(df) - 1, n, dtype=int)
                # Select those rows
                top5 = df.iloc[indices]
            elif selection_method=="lowest":
                top5 = df.nsmallest(n, 'param')

            if not flag:
                try: 
                    print(list(top5["model_name"]))
                    summary["model_name"]+=list(top5["model_name"])
                except Exception:
                    print(f"Model name issue ({model.value})")
                    summary["model_name"]=list(top5["model_name"])
                
                flag=True

            exp_avgs = list(top5['entropic_expressivity'].apply(lambda l: statistics.mean(l)))
            cap_avgs = list(top5['algorithmic_capacity'].apply(lambda l: statistics.mean(l)))
            parameters = np.array(top5['param'])

            try:
                summary[f"{dataset.value}_entropic_expressivity"]+= exp_avgs
            except Exception:
                print(f"EXCEPTION ent ({model.name})")
                summary[f"{dataset.value}_entropic_expressivity"] = exp_avgs

            try: 
                summary[f"{dataset.value}_algorithmic_capacity"]+=cap_avgs
            except Exception:
                print(f"EXCEPTION ca ({model.name})")
                summary[f"{dataset.value}_algorithmic_capacity"] =cap_avgs
            # summary[f"{dataset.value}_{model.value}_param"] = parameters

            bias_names = ['algorithmic_bias_size_1', 'algorithmic_bias_size_2', 'algorithmic_bias_size_3', 'algorithmic_bias_size_4','algorithmic_bias_size_5']
            for bias_name in bias_names:
                alg_bias_avgs = list(top5[bias_name].apply(lambda l: statistics.mean(l)))
                try:
                    summary[f"{dataset.value}_{bias_name}"] += alg_bias_avgs
                except Exception:
                    print(f"EXCEPTION {bias_name} ({model.name})")
                    summary[f"{dataset.value}_{bias_name}"] = alg_bias_avgs
        flag = False

    
    return pd.DataFrame(summary)
            # summary []

def generate_PCA_meta(path="summary_all_results.csv", n_clusters=4, min_cluster_size=5, cluster_method="kmeans", reduction_method = "PCA", no_trees=False):
    df = pd.read_csv(path)
    
    

    # Optional: exclude tree algs
    if no_trees:
        words_to_exclude = ['Forest', 'Tree', 'boost']

        # Build regex pattern like 'foo|test'
        pattern = '|'.join(words_to_exclude)

        # Drop rows where 'text' column contains any of the words
        df = df[~df['model_name'].str.contains(pattern, case=False, na=False)]
        df = df.reset_index(drop=True)

    model_names = df['model_name']
    # pdb.set_trace()
    X = df.drop(columns='model_name')

    # Optional: scale the features (recommended for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    pca = PCA(n_components=10)
    X = pca.fit_transform(X_scaled)
    print(f"initial reduction total variance ratio: {sum(list(pca.explained_variance_ratio_))}")


    # KMeans clustering in original feature space
    if cluster_method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        print(f"kmeans wcss: {kmeans.inertia_}")

    if cluster_method == "hdbscan":
        hdb = HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels=hdb.fit_predict(X)
        print(f"list of cluster labels: {np.unique(cluster_labels).tolist()}")


    # Perform PCA
    if reduction_method == "PCA":
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        print(f"explained var: {pca.explained_variance_}")
    if reduction_method == "UMAP":
        reducer = umap.UMAP()
        X_pca = reducer.fit_transform(X_scaled)
    if reduction_method == 'TSNE':
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_pca = tsne.fit_transform(X)


    # Put PCA results in a DataFrame
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['model_name'] = model_names
    # pdb.set_trace()
    pca_df['algorithm'] = pca_df['model_name'].apply(extract_algorithm_key)
    pca_df['cluster'] = cluster_labels


    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='algorithm',
        symbol='cluster',
        hover_name='model_name',  # Optional: shows model name on hover
        title=f'{reduction_method} Projection of Models',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        width=1000,
        height=700
    )

    # Improve layout
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')), textposition='top right')
    fig.update_layout(
        legend_title_text='Model Name / Cluster',
        legend=dict(itemsizing='constant'),
        margin=dict(l=40, r=250, t=40, b=40)
    )

    #fig.show()
    fig.write_html(f"meta_clustering_{cluster_method}_{reduction_method}_notrees{str(no_trees)}_{n_clusters}.html")
    print(f"meta_clustering_{cluster_method}_{reduction_method}_notrees{str(no_trees)}_{n_clusters}.html")
    return
    # Plot the PCA results
    # plt.figure(figsize=(8, 6))
    # plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['cluster'], cmap='tab10', alpha=0.7)

    # # Annotate each point with the model name
    # for _, row in pca_df.iterrows():
    #     plt.text(row['PC1'] + 0.05, row['PC2'] + 0.05, row['model_name'], fontsize=9)

    # plt.title(f'{reduction_method} Projection of Models')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"meta_clustering_{reduction_method}.png")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

# def generate_PCA_meta():
#     df = pd.read_csv("summary_all_results.csv")
    
#     model_names = df['model_name']
#     X = df.drop(columns='model_name')

#     # Optional: scale the features (recommended for PCA)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Perform PCA with 3 components
#     pca = PCA(n_components=3)
#     X_pca = pca.fit_transform(X_scaled)
#     print(f"Explained variance: {pca.explained_variance_}")
#     print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

#     # Create PCA result DataFrame
#     pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
#     pca_df['model_name'] = model_names

#     # 3D Plot
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], alpha=0.7)

#     # Annotate each point
#     for _, row in pca_df.iterrows():
#         ax.text(row['PC1'] + 0.05, row['PC2'] + 0.05, row['PC3'] + 0.05,
#                 row['model_name'], fontsize=8)

#     ax.set_title('3D PCA Projection of Models')
#     ax.set_xlabel('PC1')
#     ax.set_ylabel('PC2')
#     ax.set_zlabel('PC3')

#     plt.tight_layout()
#     plt.savefig("test_pca_3d.png")
#     plt.show()


# def bias_v_expressivity_scatter(bias_level=4):
#     """Plots scatterplot of bias vs expressivity"""
#     bias_column = f"algorithmic_bias_size_{bias_level}"
#     for model in constants.ModelNamesMetrics:
#         readable_model = constants.READABLE_MODELS[model.value] # readable version of model
#         print(readable_model)
#         if model.name == "GAUSSIAN_PROCESS_MAX_ITER": # arbitrary skip
#             print(f"skip:{model.name}")
#             continue        
#         for dataset in constants.DatasetNames:
#             if dataset.value == "Shopper_Intention_Balanced": # arbitrary skip
#                 continue
#             df = load_trial_data(model.value, dataset.value)

#             df[bias_column] # plot this on the axis of the scatter plot
#             df["entropic_expressivity"] # plot this on the y axis
#             # color all the points added in this model the same, label as readable_model
def bias_v_expressivity_scatter(bias_level=4, save_path="bias_vs_expressivity.png"):
    """Plots scatterplot of bias vs expressivity and saves the figure"""
    sns.set_style("darkgrid", {"axes.facecolor": ".1"})  # Dark background
    plt.figure(figsize=(12, 8))
    
    bias_column = f"algorithmic_bias_size_{bias_level}"

    all_exps = []
    all_bias = []
    all_model_names = []
    
    for model in constants.ModelNamesMetrics:
        readable_model = constants.READABLE_MODELS[model.value]
        print(readable_model)
        
        if model.name == "GAUSSIAN_PROCESS_MAX_ITER":
            print(f"skip: {model.name}")
            continue        
        
        for dataset in constants.DatasetNames:
            if dataset.value == "Shopper_Intention_Balanced":
                continue

            df = load_trial_data(model.value, dataset.value)

            exp_avgs = list(df['entropic_expressivity'].apply(lambda l: statistics.mean(l)))
            bias_avgs = list(df[bias_column].apply(lambda l: statistics.mean(l)))
            model_names = [readable_model for _ in range(len(exp_avgs))]
            
            all_exps += exp_avgs
            all_bias += bias_avgs
            all_model_names += model_names
    

    df = pd.DataFrame({"Model Name":all_model_names, "Entropic Expressivity": all_exps, "Algorithmic Bias":all_bias})
    df["Expressivity Bound"] = df["Algorithmic Bias"].apply(lambda bias: 5 - 2*(bias**2))
    df = df.sort_values(by="Expressivity Bound")
    
    sns.set_style("darkgrid")
    # plt.axhline(y=0, color='darkgrey', linewidth=1)
    # plt.axvline(x=0, color='darkgrey', linewidth=1)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, y="Entropic Expressivity", x="Algorithmic Bias", hue="Model Name", s=40, edgecolor='black', alpha=0.7)
    # sns.lineplot(data=df, x="Expressivity Bound", y='Algorithmic Bias', color='firebrick', lw=2)
    sns.lineplot(
        data=df,
        y="Expressivity Bound",
        x="Algorithmic Bias",
        color='lightcoral',       # lighter red
        linestyle='--',           # dashed line
        # linewidth=0.5,
        lw=2,
        label='Expressivity bound'

    )

    plt.legend(
    # bbox_to_anchor=(1.05, 1),
    loc='best',
    #loc='center right',
    borderaxespad=0.
    )

    if bias_level==1:
        l, u = -0.9688, 0.0312
    elif bias_level==2:
        l, u = -0.8125, 0.1875
    elif bias_level==3:
        l, u = -0.5000, 0.5000
    elif bias_level==4:
        l, u = -0.1875, 0.8125
    else:
        l, u = -0.0313, 0.9688
    # plt.xlim(l-0.05,u+0.05)#l-0.05, u+0.05)
    # plt.ylim(-0.1,5.05)
    

    plt.xlabel(f"Algorithmic bias (threshold {bias_level})")
    plt.ylabel("Entropic expressivity")  # Optional: hide x-label if desired
    plt.tight_layout()

    plt.savefig(f"{save_path}_{bias_level}_bound.png", dpi=300, bbox_inches='tight', transparent=False)

    # plt.show()
                

            
            
            
    

if __name__ == "__main__":
    pass
    # df = generate_summary_csv(n=10)
    
# model = constants.ModelNamesMetrics.RANDOM_FOREST_ESTIMATORS.value
# dataset = constants.DatasetNames.EEG_EYE_STATE.value
# df = load_trial_data(model, dataset)
    
    # data = {"dataset": [],"r2": [], "rmse":[], "a_fit": [], "b_fit":[], "c_fit":[]}
    # data = {"dataset": [], "m_fit": [], "b_fit": [], "r2": [], "rmse": []}
    # y_offsets = [0.39434007688525674,0.6787820046855395,0.1902910797891422,0.017547266930235318,0.3632025623465173,0.31370926150401784, 0.08730534784489476,0.05838263010430627,0.10587842241313686,0.1542574085825844,0.02983391698096553]

    # for i, dataset_obj in enumerate(constants.DatasetNames):
    #     dataset = dataset_obj.value

    #     df = load_trial_data(model, dataset)
    #     a,b,c,r2,rmse =get_expcap_corr_w_metric(model, dataset)
        # m, b, r2, rmse = get_expcap_corr_w_metric_l(model, dataset, offset=y_offsets[i])
        # data["dataset"].append(constants.READABLE_DATASETS[dataset])
        # data["m_fit"].append(m)
        # data["b_fit"].append(b)
        # data["r2"].append(r2)
        # data["rmse"].append(rmse)
    #     data["dataset"].append(constants.READABLE_DATASETS[dataset])
    #     data["a_fit"].append(a)
    #     data["b_fit"].append(b)
    #     data["c_fit"].append(c)
    #     data["r2"].append(r2)
    #     data["rmse"].append(rmse)

    # df = pd.DataFrame(data)
    # df.to_csv(f"random_forest_exponential_fits/{model}_cap_corr.csv", index=False)

