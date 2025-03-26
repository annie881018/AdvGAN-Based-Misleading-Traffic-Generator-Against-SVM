import tensorflow as tf

def build_generator(feature_dim):
    
    # def custom_activation(x):
        
    #     return x
    
    inputs = tf.keras.Input(shape=(feature_dim,))
    x = tf.keras.layers.Dense(256)(inputs)  # ✅ 增加神經元數量
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
    
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)  # 加入 BatchNorm

    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # 加入 Dropout，避免 G 過度擬合輸入

    outputs = tf.keras.layers.Dense(feature_dim, activation='tanh')(x)
    # outputs = layers.Activation(custom_activation)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
