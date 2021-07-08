import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


dataset = pd.read_csv("online_shoppers_intention.csv", header = None)
dataset = dataset.sample(frac=0.75)
values = dataset.values
X, y = values[:, :-1], values[:, -1]


def convert_to_np(dataset):
    values = dataset.values
    X, y = values[:, :-1], values[:, -1]
    points = np.array(X)
    return points


def get_avg_dist(dataset):
    points = convert_to_np(dataset)
    #print("points ", points)
    pairwise_Dist = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis = -1))
    #print("dist ", pairwise_Dist)
    return get_min_dist(pairwise_Dist)


def get_min_dist(pairwise_Dist, same=True):
    if same:
      np.fill_diagonal(pairwise_Dist, np.inf)
    min_dist = np.min(pairwise_Dist, axis=1)
    avg_dist = np.average(min_dist)
    if avg_dist == np.inf:
        avg_dist = 0

    return avg_dist

print("distance to any class label ", get_avg_dist(dataset))

#get dist same class label
dataset0 = dataset.loc[y == 0]
dataset1 = dataset.loc[y == 1]


avg_dist_same = (get_avg_dist(dataset0) + get_avg_dist(dataset1))/2
print("distance to same class label ", avg_dist_same)


points0 = convert_to_np(dataset0)
points1 = convert_to_np(dataset1)

#get dist different class label
dist_diff = cdist(points0, points1)
avg_diff_dist = get_min_dist(dist_diff, same=False)
print("distance to different class label ", avg_diff_dist)