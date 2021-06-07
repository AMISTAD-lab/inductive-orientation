from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import new_ldm_inductive as ldm_inductive
import matplotlib.pyplot as plt
import pandas as pd

# Initialize model specific variables
#dataset = datasets.load_iris()
dataset = pd.read_csv("BankNote_Authentication.csv")
X = dataset[dataset.columns[0:4]]
y = dataset[dataset.columns[4]]
model = KNeighborsClassifier(n_neighbors=1)
model3 = KNeighborsClassifier(n_neighbors=3)
model10 = KNeighborsClassifier(n_neighbors=10)
adaboostClassifier = AdaBoostClassifier()
holdout_set_percentage = 0.003 #i think this is useless now rt
num_datasets = 5
proportion_of_dataset = 0.3
classes = [0,1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.003) #gives us about 5 things in the holdout set
#X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = holdout_set_percentage)

LDM = ldm_inductive.getLDM(model, X_train, X_test, y_train, classes=classes)
print(LDM)
Pd = ldm_inductive.computePD(LDM)
adjusted_Pd = ldm_inductive.computeGoodTuring(LDM)

list_of_PD_sparse = ldm_inductive.computeNPD(5, model, X_train, X_test, y_train, classes= classes)
list_of_PD = ldm_inductive.computeNPD(5, model, X_train, X_test, y_train, sparse=False, classes=classes)

print("sparse")
ldm_inductive.varianceUpToN(list_of_PD_sparse)
print("not sparse")
ldm_inductive.varianceUpToN(list_of_PD)
