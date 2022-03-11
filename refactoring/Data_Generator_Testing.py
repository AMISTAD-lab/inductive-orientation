from sklearn import datasets
import Data_Generator as Data_Generator

iris = datasets.load_iris()
X = iris.data
y = iris.target

generator = Data_Generator.Data_Generator(X, y, 42)
random_subset_X, random_subset_y = generator.generate_subset(100)

n_subset_X, n_subset_y = generator.first_n_elements(100)
