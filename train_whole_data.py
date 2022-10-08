# Importing the feature engineering pipeline 
from utils import ft_engineering_pipeline

# Data wrangling
import pandas as pd

# Memory tracking
from memory_profiler import profile

# Command line arguments 
import argparse

# Model creation 
from model import create_model

# Using CPU 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Typehinting tuple
from typing import Tuple

# Defining the training function
def train(x, y, epochs: int = 10, batch_size: int = 128, learning_rate: float = 0.001): 
    # Defining a simple feed forward network 
    model = create_model(x.shape[1], 128, 'adam', learning_rate)

    # Fitting the model
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1)

    # Returning the model
    return model

def load_data(data_path: str, sample_size: int = None) -> pd.DataFrame:
    """
    Loads the data from the given data_path. 

    If sample_size is not None - returning a random subsample of data of size sample_size
    """
    # Reading the data
    data = pd.read_csv(data_path, nrows=sample_size)

    # Returning the data
    return data

def prep_data(
    df: pd.DataFrame, 
    numeric_features: list, 
    dummy_features: list, 
    target_name: str
    ) -> Tuple: 
    """
    Creates the x and y for deep learning
    """
    # Applying the feature engineering pipeline
    x, y, _ = ft_engineering_pipeline(df, numeric_features, dummy_features, target_name)

    return x, y

@profile
def pipeline():
    """
    Function that wraps everything together
    """
    # Parsing the number of rows to use 
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=None)
    args = parser.parse_args()

    # Reading the data 
    df = load_data('data/train.csv', args.rows)

    # Defining the final feature list 
    numeric_features = [
        'distance',
        'passenger_count', 
        'pickup_hour_sin',
        'pickup_hour_cos',
        'pickup_dayofyear_sin',
        'pickup_dayofyear_cos',
    ]

    # Defining the target variable
    target = 'fare_amount'

    # Defining the dummy features
    dummy_features = [
        'pickup_dayofweek'
    ]

    # Applying the feature engineering pipeline
    x, y = prep_data(df, numeric_features, dummy_features, target)
    del df

    # Defining the hps 
    batch_size = 512
    epochs = 10 

    # Training the model
    model = train(x, y, epochs=epochs, batch_size=batch_size)

    return model

if __name__ == '__main__':
    pipeline()
