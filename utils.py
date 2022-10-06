# Data wrangling 
import pandas as pd 

# Array math 
import numpy as np 

# Min Max scaling
from sklearn.preprocessing import MinMaxScaler

# Datetime wrangling
from datetime import datetime

# To datetime conversion 
def to_datetime(x):
    try:
        return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    except:
        return pd.to_datetime(x)

def create_date_vars(d):
    """
    Creates the datetime variables
    """
    # Infering the day of the week from pickup_datetime
    d['pickup_datetime'] = [to_datetime(x) for x in d['pickup_datetime']]
    d['pickup_dayofweek'] = d['pickup_datetime'].dt.dayofweek

    # Infering the hour of the day from pickup_datetime
    d['pickup_hour'] = d['pickup_datetime'].dt.hour

    # Creating a new variable for the day of the year
    d['pickup_dayofyear'] = d['pickup_datetime'].dt.dayofyear

    # Ensuring a monotonic relationship between pickup_hour and pickup_dayofyear
    d['pickup_hour_sin'] = np.sin(2 * np.pi * d['pickup_hour']/23.0)
    d['pickup_hour_cos'] = np.cos(2 * np.pi * d['pickup_hour']/23.0)

    d['pickup_dayofyear_sin'] = np.sin(2 * np.pi * d['pickup_dayofyear']/365.0)
    d['pickup_dayofyear_cos'] = np.cos(2 * np.pi * d['pickup_dayofyear']/365.0)

    return d

# Defining the function for distance calculation
def distance_calculation(df):
    """
    Calculates the distance between two points on the earth's surface.

    The distance is in meters
    """
    R = 6373.0

    lat1 = np.radians(df['pickup_latitude'])
    lon1 = np.radians(df['pickup_longitude'])
    lat2 = np.radians(df['dropoff_latitude'])
    lon2 = np.radians(df['dropoff_longitude'])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    # Saving the distance to the dataframe
    df['distance'] = distance * 1000 # Converting to meters
    return df 

# Defining the function for dummy creation 
def create_dummy(df, dummy_var_list):
    # Placeholder for the dummy variables
    added_features = []
    for var in dummy_var_list:
        dummy = pd.get_dummies(df[var], prefix=var, drop_first=True)
        
        # Adding the new features to list 
        added_features.extend(dummy.columns)

        # Adding the dummy variables to the dataframe
        df = pd.concat([df, dummy], axis=1)
        df.drop(var, axis=1, inplace=True)

    # Returning the dataframe 
    return df, added_features

# Defining a custom label encoding function 
def custom_transform(enc, x, prefix):
    """
    Applies a custom transformation to the data
    """
    # Transforming the data
    out = enc.transform(x.reshape(-1, 1))

    # Getting the transformed values
    out_values = enc.get_feature_names_out().tolist()

    # Adding the names of the feature as a prefix
    out_values = [f"{prefix}_{value.split('_')[-1]}" for value in out_values]

    # Converting to a dataframe
    out = pd.DataFrame(out.toarray(), columns=out_values)

    # Changing the datatype to uint8
    out = out.astype('uint8')

    # Returning the transformed data
    return out

# Defining the ft engineering pipeline 
def ft_engineering_pipeline(
    df, 
    numeric_features, 
    dummy_features,
    target
    ):
    """
    Applies the feature engineering pipeline to the data
    """
    # Creating the date variables
    df = create_date_vars(df)

    # Creating the dummy variables
    df, new_features = create_dummy(df, dummy_features)

    # Appending the distance
    df = distance_calculation(df) 

    # Appending the new features to the numeric features
    final_features = numeric_features + new_features

    # Creating the x matrix 
    x = df[final_features].values

    # Creating the y vector
    y = df[target].values

    # Mean max scaling the y matrix 
    y = y.reshape(-1, 1)
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y)

    # Returning the x and y matrices
    return x, y, final_features