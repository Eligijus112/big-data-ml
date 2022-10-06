# Importing the feature engineering pipeline 
from utils import ft_engineering_pipeline

# Deep learning 
import tensorflow as tf
import keras

# Data wrangling
import pandas as pd

# Memory tracking
from memory_profiler import profile

# Command line arguments 
import argparse

# SYS import 
import sys 

# Defining the training function
@profile
def train(x, y, epochs: int = 10, batch_size: int = 128): 
    # Defining a simple feed forward network 
    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(x.shape[1],)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    # Compiling the model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    # Fitting the model
    model.fit(x, y, epochs=epochs, batch_size=batch_size)

    # Returning the model
    return model

if __name__ == '__main__':
    # Parsing the number of rows to use 
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=None)
    args = parser.parse_args()

    # Reading the data 
    df = pd.read_csv('data/train.csv')

    # If the number of rows is specified we use a sample of the same amount
    if args.rows:
        df = df.sample(args.rows)
        df.reset_index(inplace=True, drop=True)

    # Getting the size of the object in memory
    print(f"The main dataframe takes: {sys.getsizeof(df) / 10**6} MB in memory")

    # Defining the hps 
    batch_size = 512
    epochs = 10 

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
        #'vendor_id',
        #'store_and_fwd_flag',
        'pickup_dayofweek'
    ]

    # Applying the feature engineering pipeline
    x, y, _ = ft_engineering_pipeline(df, numeric_features, dummy_features, target)

    # Training the model
    model = train(x, y, epochs=epochs, batch_size=batch_size)
