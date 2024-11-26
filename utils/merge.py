# This file contains utility functions for the purpose of 
# merging the directory of CSV files into a single CSV file, 
# for ease of use.
#
# It is not vital if data is formatted as a single CSV file.

import pandas as pd
import os

from utils.argparser import parse_args
from utils.path import expand_path

if __name__ == "__main__":
    args = parse_args('merge')
    
    data_dir = expand_path(args.data_dir)
    save_path = expand_path(args.save_path)
    
    # List to hold individual dataframes
    dataframes = []

    # Iterate over each file in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            # Read the CSV file
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)
            
            dataframes.append(df)
            print(f'Loaded and appended file: {file_path}')
            
    # Merge all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(save_path, index=False)
    print(f'Merged dataset saved to {save_path}')
    

    
    