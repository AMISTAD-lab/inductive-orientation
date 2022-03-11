import numpy as np
class Data_Generator:
  def __init__(self, dataset_x, dataset_y, seed):
    self.dataset_x = dataset_x
    self.dataset_y = dataset_y
    self.seed = seed
    self.rng = np.random.default_rng(self.seed)
  
  def set_seed(self, seed):
    self.seed = seed
    self.rng = np.random.default_rng(self.seed)
  
  def randomize_seed(self):
    self.seed = int(np.floor(np.random.rand()*100))
    self.rng = np.random.default_rng(self.seed)

  def generate_subset(self, num_entries, do_replace=False):
    indices = np.array(len(self.dataset_x))
    indices = self.rng.choice(indices, num_entries, do_replace)
    random_subset_X = [self.dataset_x[i] for i in indices]
    random_subset_y = [self.dataset_y[i] for i in indices]
    return random_subset_X, random_subset_y

  def first_n_elements(self, num_entries):
    subset_X = self.dataset_x[:num_entries]
    subset_y = self.dataset_y[:num_entries]
    return subset_X, subset_y

  def split_dataset(self, num_entries):
    raise Exception("Function not implemented.")
