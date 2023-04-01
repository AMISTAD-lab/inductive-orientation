
#Important
# 2 to 12 dimensions; size of dataset vary between 100 to 5000 samples

import numpy as np
import pandas

def generate_fully_synethic(n_features, n_samples, max_value, num_classes, seed=42):
    """
    Generates random dataset
    n_features (int): number of features in the X
    n_samples (int): number of samples aka size of the dataset.
    max_value (int): how big each of the feautres in the X vector can be
    num_class (int): usually just put as 2 since it's binary
    
    returns x_data (n_samples, n_features) y_data (n_samples)
    x_data[1, :]. y_data[1]
    """
    rng = np.random.default_rng(seed)
    x_data = rng.integers(low=0, high=max_value, size=(n_samples, n_features))
    y_data = rng.integers(low=0, high=num_classes, size=n_samples)
    return x_data, y_data

# should be useless function, will delete after confirming
# def generate_semirandom(n_features, n_samples, max_value, num_classes, seed=42):
#     rng = np.random.default_rng(seed)
#     x_data = rng.integers(low=0, high=max_value, size=(n_samples, n_features))
#     y_data = rng.integers(low=0, high=num_classes, size=n_samples)
    
if __name__ == "__main__":
    X, y = generate_fully_synethic(4, 2000, 100, 2, seed=42)
    df = pandas.DataFrame({"x0":X[:,0], "x1":X[:,1], "x2":X[:,2], "x3":X[:,3], "y":y})
    df.to_csv("datasets/SemiRandom.csv", index=False)