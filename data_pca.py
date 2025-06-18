"""Dataset analysis + PCA on the letter recognition dataset"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Generate sample data (replace this with your actual dataset)
# Assuming `X` is your feature matrix and `y` is the class label vector
np.random.seed(42)

data = pd.read_csv("datasets/EEG_Eye_State.csv")

# Separate features (X) and class labels (y)
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values   # The last column

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA results
plt.figure(figsize=(8, 6))
for label in np.unique(y):
    plt.scatter(
        X_pca[y == label, 0], 
        X_pca[y == label, 1], 
        label=f"Class {label}"
    )

plt.title("PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
# plt.show()

# Save the figure
output_path = "pca_visualization_eeg.png"
plt.savefig(output_path, dpi=300)  # Higher DPI for better resolution
plt.close()

print(f"PCA plot saved as {output_path}")