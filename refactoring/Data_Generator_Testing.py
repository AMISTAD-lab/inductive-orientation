from sklearn import datasets
import Data_Generator as Data_Generator
from Fully_Synthetic import generate_fully_synethic
from sklearn import model_selection

iris = datasets.load_iris()
X = iris.data
y = iris.target

generator = Data_Generator.Data_Generator(X, y, 42)
random_subset_X, random_subset_y = generator.generate_subset(100)

n_subset_X, n_subset_y = generator.first_n_elements(100)

X, y = generate_fully_synethic(4, 2000, 100, 2)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 5, random_state=42)

generator2 = Data_Generator.Data_Generator(X_train, y_train, 42, X_test, y_test)
fixed_x, fixed_y = generator2.generate_subset_plus_fixed(10)
print("x: ", fixed_x)
print("y: ", fixed_y)