import benchmark
import constants
from Trial_Setup_Utils import maybe_mkdir
import os

if __name__ == '__main__':
    bds_folders = os.path.join(constants.RESULTS_FOLDER, 'heatmaps')

    #aggregate_fn = max
    #aggregate_name = constants.AggregateNames.MAX.value
    aggregate_fn = lambda x: x.mean()  
    aggregate_name = constants.AggregateNames.AVG.value
    saving_dir = maybe_mkdir(constants.RESULTS_FOLDER, 'rankings')
    
    benchmark.rank_models(bds_folders, aggregate_fn, saving_dir, aggregate_name)
