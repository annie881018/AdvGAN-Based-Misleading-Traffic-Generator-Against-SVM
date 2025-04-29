import tensorflow as tf

def build_generator(feature_dim):
    
    def custom_activation(x):
        
        return x
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(100,)))  # Indicate input(noise) shape

    # The first hidden layer
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.LeakyReLU(negative_slope=2.5))  # the slope of LeakyReLU influence the range of output

    # The second hidden layer
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.LeakyReLU(negative_slope=2.5))

    # The third hidden layer
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.LeakyReLU(negative_slope=2.5))

    # The last layer: output features
    model.add(tf.keras.layers.Dense(feature_dim))
    return model
