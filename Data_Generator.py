# Important
from random import random
import numpy as np
class Data_Generator: #TODO: add documentation for all methods
  def __init__(self, dataset_x, dataset_y, seed, fixed_dataset_x=None, fixed_dataset_y=None):
    self.dataset_x = dataset_x # TODO: change all the code here and in Inductive_Generator to work with X_train as the pool of potential holdout sets, not just a definitive holdout set
    self.dataset_y = dataset_y
    self.seed = seed
    self.fixed_dataset_x = fixed_dataset_x
    self.fixed_dataset_y = fixed_dataset_y
    self.rng = np.random.default_rng(self.seed)
  
  def set_seed(self, seed):
    self.seed = seed
    self.rng = np.random.default_rng(self.seed)
  
  def randomize_seed(self):
    self.seed = int(np.floor(np.random.rand()*100))
    self.rng = np.random.default_rng(self.seed)

  def generate_subset(self, num_entries, do_replace=False):
    """Generates training subset for each P_f in LDM"""
    indices = np.array(len(self.dataset_x))
    indices = self.rng.choice(indices, num_entries, do_replace)
    # TODO: if it's numpy we can index into everything
    random_subset_X = np.array([self.dataset_x[i] for i in indices]) 
    random_subset_y = np.array([self.dataset_y[i] for i in indices])
    return random_subset_X, random_subset_y
  
  

  def first_n_elements(self, num_entries):
    subset_X = self.dataset_x[:num_entries]
    subset_y = self.dataset_y[:num_entries]
    return subset_X, subset_y

  def generate_subset_plus_fixed(self, num_entries, do_replace=False):
    """Not used"""
    random_subset_X, random_subset_y = self.generate_subset(num_entries-len(self.fixed_dataset_x), do_replace)
    random_subset_X = np.concatenate((random_subset_X, self.fixed_dataset_x))
    random_subset_y = np.concatenate((random_subset_y, self.fixed_dataset_y))
    return random_subset_X, random_subset_y

  def split_dataset(self, num_entries):
    raise Exception("Function not implemented.")

# end of the object

def generate_holdout_set(holdout_size, X_test, y_test, do_replace:bool = True, seed_num=42):
  """Helper. Generate num_holdouts holdout sets from X_tests of holdout_size"""
  # indices = np.array(len(X_test))
  seed = np.random.default_rng(seed_num)
  indices = seed.choice(len(X_test), holdout_size, do_replace)
  # TODO: if it's numpy we can index into everything
  holdout_set_X = np.array([X_test[i] for i in indices]) 
  holdout_set_y = np.array([y_test[i] for i in indices])
  return holdout_set_X, holdout_set_y

def generateN_holdout_sets(num_holdouts, holdout_size,  X_test, y_test, holdout_set_seeds, do_replace:bool = True):
  """Generate num_holdouts holdout sets from X_tests of holdout_size. Returns two 2D arrays of each holdout set (each of x and y)"""
  holdout_sets_x = []
  holdout_sets_y = []

  if len(holdout_set_seeds) != num_holdouts:
    raise Exception("Must have holdout_set_seeds have the length of num holdouts")

  for i in range(num_holdouts): # generate and append holdout sets
    new_set_x, new_set_y = generate_holdout_set(holdout_size, X_test, y_test, do_replace, holdout_set_seeds[i]) # CHANGED HERE FROM SINGLE SEED TO MULTIPLE HOLDOUT SET SEEDS
    holdout_sets_x.append(new_set_x)
    holdout_sets_y.append(new_set_y)

  return holdout_sets_x, holdout_sets_y