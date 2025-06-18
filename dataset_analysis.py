"""10/31/2024
Analysis to determine entropy/repetitiveness of a dataset. We need to estimate how much each bootstrapped sample changes, 
as our model of an inductive orientation vector assumes changes in the training subset.
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy
from collections import Counter
import pdb
import constants
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def bootstrap_analysis(data:pd.DataFrame, n_bootstraps = 1500, fraction = 0.12):
    "Returns standard deviation of bootstrap means"
    # Get min and max for each 

    bootstrap_samples = []
    for _ in range(n_bootstraps):
    # Sample with replacement
        sample = data.sample(frac=fraction, replace=True)
        bootstrap_samples.append(sample)

    bootstrap_means = [sample.mean(axis=0) for sample in bootstrap_samples]
    std_by_col = np.std(bootstrap_means, axis=0) # per-feature variance of bootstrapped means
    ave_std = np.mean(std_by_col[:-1]) # overall variance of bootstrapped means throughout all features, ignore last label column

    return std_by_col, ave_std
    # print(bootstrap_means)

def calculate_entropy_categorical(feature):
    """Calculate entropy of a categorical feature."""
    counts = Counter(feature)
    probabilities = [count / len(feature) for count in counts.values()]
    return entropy(probabilities, base=2)  # base 2 for bits

def calculate_entropy_numerical(feature, bins=10):
    """Calculate entropy of a numerical feature by binning."""
    histogram, bin_edges = np.histogram(feature, bins=bins, density=True)
    probabilities = histogram / histogram.sum()
    return entropy(probabilities, base=2)  # base 2 for bits

# Calculate entropy for each feature
def calculate_average_entropy(data:pd.DataFrame):
    entropies = {}
    for column in data.columns:
        if data[column].dtype == 'object':  # Categorical feature
            entropies[column] = calculate_entropy_categorical(data[column])
        else:  # Numerical feature
            entropies[column] = calculate_entropy_numerical(data[column], bins=10)

    # Total entropy of the dataset (sum or average of feature entropies)
    total_entropy = sum(entropies.values())
    average_entropy = total_entropy / len(entropies)
    return entropies, average_entropy
def svm_overlap_measure(data:pd.DataFrame,dataset_name):
    # Separate features (X) and class labels (y)
    X = data.iloc[:, :-1].values  # All columns except the last
    y = data.iloc[:, -1].values   # The last column
    # Train an SVM classifier
    svm_model = SVC(kernel='linear', C=1)  # Using a linear kernel
    svm_model.fit(X, y)

    # Predict class labels
    y_pred = svm_model.predict(X)

    # Measure accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"SVM Classification Accuracy: {accuracy}")

    # Classification report for detailed metrics
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    # Decision function for margin visualization
    decision_function = svm_model.decision_function(X)
    print(f"\nMean decision function values per class:\n"
      f"Class 0: {decision_function[y == 0].mean():.2f}, "
      f"Class 1: {decision_function[y == 1].mean():.2f}")
    # Reduce dimensions for visualization (if needed)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Train SVM on 2D data
    svm_model_2d = SVC(kernel='linear', C=1)
    svm_model_2d.fit(X_pca, y)

    # Create a mesh grid for decision boundary
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))
    Z = svm_model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label="Class 0", alpha=0.8)
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label="Class 1", alpha=0.8)
    plt.title("SVM Decision Boundary")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    # plt.show()
    output_path = f"pca_svm_{dataset_name}.png"
    plt.savefig(output_path, dpi=300)  # Higher DPI for better resolution
    plt.close()


if __name__=="__main__":
    
    for dataset in constants.DatasetNames:
        # if dataset.value != "Letter_Recognition": continue
        print(dataset)
        df = pd.read_csv(f"datasets/{dataset.value}.csv")
        svm_overlap_measure(df, dataset.value)
        # std_by_col, ave_std = bootstrap_analysis(df)
        # entropies, average_entropy = calculate_average_entropy(df)
        # print(f"\n\nDataset: {dataset.value}\nave_stdev (boot): {ave_std}\nPer-feature stdev (boot): {std_by_col}")
        # print(f"Per-feature entropy: {entropies}\nAverage entropy: {average_entropy}")
        # print(f"Per-feature stdev (overall):\n {df.std()}")
