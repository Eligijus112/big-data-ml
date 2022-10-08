# Deep learning 
import tensorflow as tf
import keras

def create_model(
    input_size: int, 
    hidden_neuron_count: int, 
    optimizer_name: str = 'sgd', 
    learning_rate: float = 0.01
    ) -> keras.Sequential:
    """
    Function to initiate a model in RAM

    Arguments
    ---------
    input_size: int
        The size of the input layer
    hidden_neuron_count: int
        The number of neurons in the hidden layer
    optimizer_name: str
        The optimizer to use; Available options are: 'adam', 'sgd', 'rmsprop'
    learning_rate: float
        The learning rate to use

    Returns
    -------
    model: keras.Sequential
        The model in RAM
    """
    # Defining a simple feed forward network 
    model = keras.Sequential([
        keras.layers.Dense(hidden_neuron_count, activation=tf.nn.relu, input_shape=(input_size,)),
        keras.layers.Dense(hidden_neuron_count, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate) if optimizer_name == 'adam' else \
        keras.optimizers.SGD(learning_rate=learning_rate) if optimizer_name == 'sgd' else \
        keras.optimizers.RMSprop(learning_rate=learning_rate) if optimizer_name == 'rmsprop' else None

    # Compiling the model
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    # Returning the model
    return model

if __name__ == '__main__':
    # Initiating the model in memory 
    model = create_model(18, 128, 'adam', 0.001)