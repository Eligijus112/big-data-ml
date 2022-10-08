# Data wrangling 
import pandas as pd

# Deep learning 
import tensorflow as tf
import keras 

# Import feature engineering functions
from utils import create_date_vars, distance_calculation, custom_transform

# Iteration tracking 
from tqdm import tqdm 

# Array math 
import numpy as np 

# Object RAM tracking
import sys 
from memory_profiler import profile

# One hot encoder
from sklearn.preprocessing import OneHotEncoder

# Argument parsing
import argparse

# Model creation 
from model import create_model

# Defining the class for the batches creation 
class DataGenerator(keras.utils.Sequence):
    def __init__(
        self, 
        csv_generator: pd.io.parsers.readers.TextFileReader,
        n_batches: int,
        dummy_features: list,
        cat_encoders: dict,
        y_var: str,
        y_min: float, 
        y_max: float,
        final_features: list
        ):
        """
        Method to create an iterator in memory 

        Arguments
        ---------
        csv_generator: pd.io.parsers.readers.TextFileReader
            The csv generator from pandas
        n_batches: int
            The number of batches that are available in the csv_generator
        dummy_features: list
            The list of categorical features that need to be one hot encoded
        cat_encoders: dict
            The dictionary of the one hot encoders for the categorical features used for transformation
        y_var: str
            The name of the target variable
        y_min: float
            The minimum value of the target variable (used in min max scaling)
        y_max: float
            The maximum value of the target variable (used in min max scaling)
        final_features: list
            The list of the final features that are used for training
        """
        self.csv_generator = csv_generator
        self.n_batches = n_batches
        self.dummy_features = dummy_features
        self.cat_encoders = cat_encoders
        self.y_var = y_var
        self.y_min = y_min
        self.y_max = y_max
        self.final_features = final_features

    def __len__(self):
        """
        The total length of the iterator
        """
        return self.n_batches

    def __getitem__(self, idx):
        """
        The batch generator 
        """
        # Getting the batch
        chunk = self.csv_generator.get_chunk()

        # Reseting the index
        chunk = chunk.reset_index(drop=True)

        # Creating the date variables
        chunk = create_date_vars(chunk, verbose=False)

        # Creating the distance variable
        chunk = distance_calculation(chunk) 

        # Creating the dummy variables
        for cat_feature in self.dummy_features:
            # Extracting the values
            x = chunk[cat_feature].values

            # Transforming the data
            out = custom_transform(self.cat_encoders[cat_feature], x, cat_feature)

            # Concatenating the data
            chunk = pd.concat([chunk, out], axis=1)

            # Deleting the out, x from memory
            del out, x

        # Getting the target var 
        y = chunk[self.y_var].values

        # Min max transforming the y 
        y = (y - self.y_min) / (self.y_max - self.y_min)

        # If any of the final features are missing we fill them with 0
        missing_cols = set(self.final_features) - set(chunk.columns)
        for c in missing_cols:
            chunk[c] = 0

        # Extracting the final features
        x = chunk[self.final_features].values

        # Deleting the chunk from memory
        del chunk

        # Returning x and y 
        return x, y

@profile
def train_generator(
    path_to_csv,
    n_batches,
    final_features,
    dummy_features,
    cat_encoders,
    y_var,
    y_min,
    y_max,
    epochs: int = 10,
    batch_size: int = 128
    ): 
    # Defining a simple feed forward network 
    model = create_model(len(final_features), 128, 'adam', 0.001)

    for _ in range(epochs):
        # Creating the generator
        d = pd.read_csv(path_to_csv, chunksize=batch_size, iterator=True)
        generator = DataGenerator(
            csv_generator=d, 
            n_batches=n_batches,
            dummy_features=dummy_features,
            cat_encoders=cat_encoders,
            y_var=y_var,
            y_min=y_min,
            y_max=y_max,
            final_features=final_features
            )

        # Fitting the model
        model.fit(generator, epochs=1, verbose=1, batch_size=batch_size)

    # Returning the model
    return model

if __name__ == '__main__':
    # Parsing the number of rows to use 
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=None)
    args = parser.parse_args()

    # Defining the hps 
    batch_size = 2056
    epochs = 10

    # Reading the data 
    d = pd.read_csv('data/train.csv', chunksize=batch_size, iterator=True) 

    # Defining a list of dummy features 
    dummy_features = [
        'pickup_dayofweek',
    ]

    # Defining the target name 
    target = 'fare_amount'

    # Getting the size of the object in memory
    print(f"The main dataframe takes: {sys.getsizeof(d) / 10**6} MB in memory")

    # Iterating over the chunks to get the final number of batches 
    n_batches = 0
    predefined_batch_size = False

    # If the rows are not none 
    if args.rows is not None:
        # Calculating the amount of chunks to call to cover the rows
        n_batches = int(args.rows / batch_size)
        predefined_batch_size = True

    # Creating the min-max constants for y
    min = np.inf
    max = -np.inf

    # Creating a dictionary for the categorical features that will store unique values
    cat_dict = {}

    for i, chunk in tqdm(enumerate(d)):
        # Searching for the min and max values of y
        if chunk[target].min() < min:
            min = chunk[target].min()
        if chunk[target].max() > max:
            max = chunk[target].max()

        # Creating the date variables
        chunk = create_date_vars(chunk, verbose=False)

        # Iterating over the cate features and getting the unique values
        for cat in dummy_features:
            if cat not in cat_dict.keys():
                cat_dict[cat] = list(set(chunk[cat].unique()))
            else:
                # Extracting the current unique values
                current_unique = list(set(chunk[cat].unique()))

                # Getting the new unique values
                new_unique = list(set(current_unique) - set(cat_dict[cat]))

                # Adding the new unique values to the dictionary
                cat_dict[cat].extend(new_unique)

        if predefined_batch_size:
            if i == n_batches:
                break
        else:
            n_batches += 1

    print(f"The number of batches is: {n_batches}")

    # Creating a one hot encoder for the categorical features
    cat_encoders = {}
    for cat_feature in cat_dict.keys():
        # Creating the one hot encoder
        one_hot = OneHotEncoder(categories='auto')

        # Fitting the one hot encoder
        one_hot.fit(np.array(cat_dict[cat_feature]).reshape(-1, 1))

        # Saving the encoder to the dictionary
        cat_encoders[cat_feature] = one_hot

    # Defining the final feature list 
    final_features = [
        'distance',
        'passenger_count', 
        'pickup_hour_sin',
        'pickup_hour_cos',
        'pickup_dayofyear_sin',
        'pickup_dayofyear_cos',
    ]

    # Adding the final features from the one hot encoders
    for cat_feature in cat_encoders.keys():
        # Extracting all original values
        original_values = cat_dict[cat_feature]

        # Getting the transformed values
        out_values = cat_encoders[cat_feature].get_feature_names_out().tolist()

        # Adding the names of the feature as a prefix
        new_features = [f"{cat_feature}_{value.split('_')[-1]}" for value in out_values]

        # Adding the new features to the list
        final_features.extend(new_features)

    # Training the model
    model = train_generator(
        path_to_csv='data/train.csv',
        n_batches=n_batches - 1,
        final_features=final_features,
        dummy_features=dummy_features,
        cat_encoders=cat_encoders,
        y_var=target,
        y_min=min,
        y_max=max,
        epochs=epochs,
        batch_size=batch_size
    )