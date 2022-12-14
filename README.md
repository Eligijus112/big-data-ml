# Training on big data 

Project to showcase how to fit a model on big data using TF data generators and Keras.

# Data link 

To download a large chunk of data, follow the link: https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data

The data should be unzipped into the `data/` directory.

```
├── data
│   └── train.csv
```

# Running the script 

To run the training when all the data is RAM, run the following command:

```
python -m train_whole_data
```

To run the training when all the data is provided to the model in chunks, run the following command:

```
python -m train_iterator
```
