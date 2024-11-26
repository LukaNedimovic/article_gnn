# This file contains a single utility function,
# that expands the environment variables at the respective path.

import os

def expand_path(path: str):
    """ 
    Expand environment variables present in the path, 
    e.g: DATA_DIR/test.csv -> article_gnn/data/test.csv
    """
    
    return os.path.expandvars(path)