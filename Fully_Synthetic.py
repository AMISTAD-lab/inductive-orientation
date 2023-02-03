#Important
# 2 to 12 dimensions; size of dataset vary between 100 to 5000 samples

import numpy as np

def generate_fully_synethic(n_features, n_samples, max_value, num_classes, seed=42):
    rng = np.random.default_rng(seed)
    x_data = rng.integers(low=0, high=max_value, size=(n_samples, n_features))
    y_data = rng.integers(low=0, high=num_classes, size=n_samples)
    return x_data, y_data

# should be useless function, will delete after confirming
# def generate_semirandom(n_features, n_samples, max_value, num_classes, seed=42):
#     rng = np.random.default_rng(seed)
#     x_data = rng.integers(low=0, high=max_value, size=(n_samples, n_features))
#     y_data = rng.integers(low=0, high=num_classes, size=n_samples)
    
