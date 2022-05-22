import Algorithmic_Analysis
import Inductive_Generator
from sklearn import datasets, model_selection
from sklearn.neighbors import KNeighborsClassifier
from Fully_Synthetic import generate_fully_synethic
import json
import os
import numpy as np
import pandas as pd

# Getting Data
X, y = generate_fully_synethic(4, 2000, 100, 2)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 5, random_state = 42)

target4 = Algorithmic_Analysis.getTarget(X_test, y_test, 4, [0,1])
target3 = Algorithmic_Analysis.getTarget(X_test, y_test, 3, [0,1])


# summary_4 = Algorithmic_Analysis.runAnalysis("./trial3", target=target4)
# summary_4.sort_values(by=['model_name'])
# summary_4.to_csv("trial3_target4.csv")


summary_3 = Algorithmic_Analysis.runAnalysis("./trial3", target=target3)
summary_3.sort_values(by=['model_name'])
summary_3.to_csv("trial3_target3.csv")




# create a blank dataframe, and each create 4 empty columns, 
# name_column = np.array([])
# bias_column = np.array([])
# entropic_expressivity_column = np.array([])
# algorithmic_capacity_column = np.array([])

# bias, entropic_expressivity, algorithmic_capacity = Algorithmic_Analysis.singleAnalysis("./logs/trial2_Adaboost50.json", target4)


# def runAnalysis(file, target=None, name_column=[], bias_column=[], entropic_expressivity_column = [], algorithmic_capacity_column = []):

#     if file[-4:] == "json":
#         return singleAnalysis(file, target)

#     else:
#         logs = os.listdir(file)
#         logs = [os.path.join(file, log) for log in logs]
#         for log_file in logs:
#             bias, entropic_expressivity, algorithmic_capacity = singleAnalysis(log_file, target)
#             name_column.append(file.split("/")[-1].split(".")[0])
#             bias_column.append(bias)
#             entropic_expressivity_column.append(entropic_expressivity)
#             algorithmic_capacity_column.append(algorithmic_capacity)
#         summary = pd.DataFrame(name=name_column, algorithmic_bias = bias_column, \
#             entropic_expressivity = entropic_expressivity_column, algorithmic_capacity = algorithmic_capacity_column)
#         return summary
    