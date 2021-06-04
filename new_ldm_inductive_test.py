from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import new_ldm_inductive as ldm_inductive
import matplotlib.pyplot as plt

# Initialize model specific variables
dataset = datasets.load_iris()
model = KNeighborsClassifier(n_neighbors=1)
model3 = KNeighborsClassifier(n_neighbors=3)
model10 = KNeighborsClassifier(n_neighbors=10)
adaboostClassifier = AdaBoostClassifier()
holdout_set_percentage = 0.03
num_datasets = 5
proportion_of_dataset = 0.3

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = holdout_set_percentage)

LDM = ldm_inductive.getLDM(model, X_train, X_test, y_train, classes=[0,1,2], num_columns = num_datasets)
Pd = ldm_inductive.computePD(LDM)