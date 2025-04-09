import tensorflow as tf

def build_generator(feature_dim):
    
    def custom_activation(x):
        
        return x
    # noise
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(100,)))  # 明確指定輸入形狀

    # 第一層：增加神經元數量
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.LeakyReLU(negative_slope=0.2))  # 改進 LeakyReLU 的斜率

    # 第二層
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.LeakyReLU(negative_slope=0.2))

    # 第三層
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.LeakyReLU(negative_slope=2))

    # 最後一層：輸出10個特徵值
    model.add(tf.keras.layers.Dense(feature_dim))
    return model
