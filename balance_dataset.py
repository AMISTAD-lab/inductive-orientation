import pandas as pd
from typing import List

def balance_data(dataset_path: str):
    df = pd.read_csv(dataset_path)
    
    # Getting the rightmost column
    right_column_name = df.columns[-1]

    # Perform a value_counts and convert it to dictionary
    value_counts_dict = df[right_column_name].value_counts().to_dict()

    # Assuming you have a DataFrame 'df' with a column 'label' representing the classes

    # Count the occurrences of each class
    class_counts = df[right_column_name].value_counts()

    # Find the class with the fewest occurrences
    min_count = class_counts.min()

    # Sample the same number of instances for each class
    balanced_df = df.groupby(right_column_name).apply(lambda x: x.sample(min_count)).reset_index(drop=True)

    saving_path = dataset_path.split(".")[0]+"_balanced"+".csv"
    balanced_df.to_csv(saving_path)
    return balanced_df

if __name__ == "__main__":
    dataset_paths = []
    for dataset_path in dataset_paths:
        balance_data(dataset_path)
