import tensorflow as tf
import numpy as np
import pandas as pd
import os
import Generator
import Modified_GAN_Discriminator
from datetime import datetime

class AdvGAN_attack:
    def __init__(self,
                 epochs,
                 features,
                 lr=0.00001,
                 thresh=0.3,
                 alpha=1,
                 beta=5,
                 num=0):
        self.epochs = epochs
        self.features = features
        self.feature_dim = len(self.features)
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.thresh = thresh
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.generator = Generator.build_generator(feature_dim=self.feature_dim)
        self.discriminator = Modified_GAN_Discriminator.build_discriminator()
        self.num = num
        self.w = tf.constant(self.discriminator.coef_[0].reshape(-1, 1), dtype=tf.float32)  # shape: (feature_dim, 1)
        self.b = tf.constant(self.discriminator.intercept_[0], dtype=tf.float32)                 # scalar
        self.ATTACKED_FEATURES_MIN = [0.0, 2.0, 60.0, 54.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ATTACKED_FEATURES_MAX = [49962.0, 16.0, 640.0, 590.0, 49962.0, 0.0, 2.0, 4.0, 8.0, 12.0]
    # loss function for influencing the output close to label 0
    def generator_loss_zero_like(self, d_predict):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.zeros_like(d_predict), d_predict)
    
    def generator_loss(self, d_output):
        # determine the margin of decision boundary
        w_norm = np.linalg.norm(self.w)
        return tf.reduce_mean(tf.maximum(0.0, d_output + 1.0))    # loss=0 if predict label 0
        # return tf.reduce_mean(tf.maximum(0.0, d_output + margin)) # loss=0 if predict to 0 less than margin
    
    # loss function for influencing the output close to margin
    def margin_loss(self, x):
        # Caculate |w^T x + b - target_plane| / ||w||
        dot = tf.tensordot(x, self.w, axes=1)
        distance = tf.abs(dot + self.b - (-1))
        normalized = distance / tf.norm(self.w)
        return tf.reduce_mean(normalized)
    
    # loss function to influence the output close to boundary
    def boundary_distance_loss(self, x, thresh):
        x = tf.reshape(x, (1, -1))
        logits = tf.matmul(x, self.w) + self.b               # shape: (batch_size, 1)
        distance = tf.abs(logits) / tf.norm(self.w)          # shape: (batch_size, 1)
        # 只懲罰 logits 遠離邊界（比如 margin 超過 1.0 就加壓）
        # penalty = tf.maximum(0.0, distance - thresh)
        # return -tf.reduce_mean(penalty)
        return -tf.reduce_mean(distance)


    # 訓練步驟
    #@tf.function
    def train_step(self, X):
        predict = -1
        noise = tf.random.normal([1, 100])
        with tf.GradientTape() as gen_tape:
            
            # Use generator to generate perturbation
            perturbation = self.generator(X)
            generated_data = tf.maximum(X + perturbation, 0)  # Assure generated_data >= 0
            # generated_data = tf.stop_gradient(tf.clip_by_value(generated_data, self.ATTACKED_FEATURES_MIN, self.ATTACKED_FEATURES_MAX))
            # generated_data = tf.round(generated_data)
            # generated_data = self.ATTACKED_FEATURES_MIN + (self.ATTACKED_FEATURES_MAX - self.ATTACKED_FEATURES_MIN) * tf.tanh(generated_data)
            # generated_data = self.generator(noise)
            generated_data = tf.clip_by_value(generated_data, self.ATTACKED_FEATURES_MIN, self.ATTACKED_FEATURES_MAX)
            
            # Pass generated data to Discriminator to get predict label,
            # the range of output is float between -1(label 0) and +1(label 1), 
            # predict is 0 or 1.
            output = self.discriminator.decision_function(generated_data)
            predict = self.discriminator.predict(generated_data)

            # Caculate the loss
            g_loss = self.generator_loss_zero_like(predict)
            # g_loss = self.margin_loss(generated_data)
            # g_loss = self.generator_loss(output)
            
            bd_loss = self.boundary_distance_loss(generated_data, self.thresh)
            # perturb_loss = self.perturb_loss(perturbation, self.thresh)
            # loss = self.alpha * g_loss + self.beta * perturb_loss
            loss = self.alpha * g_loss + self.beta * bd_loss

        # Caculate gradients
        gradients_of_generator = gen_tape.gradient(loss, self.generator.trainable_variables)
        # print("Generator gradients:", gradients_of_generator)
        
        if any(g is None for g in gradients_of_generator):
            raise ValueError("One or more generator gradients are None.")

            # 應用梯度更新模型
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return loss, generated_data, predict, g_loss, bd_loss


    # 訓練過程
    def train(self, X):
        num_train = f'{self.num}_lr_{self.lr}_a_{self.alpha}_b_{self.beta}_thsh_{self.thresh}'
        # print(f"Start to train data {num_train}----------------------")
        formatted_date = datetime.today().strftime("%m%d")
        # set output directory
        output_dir = f"result/{formatted_date}"
        # if not exist, create one directory
        os.makedirs(output_dir, exist_ok=True)
        # Complete output path
        filename_output = os.path.join(output_dir, f"generated_data_{num_train}.csv")
        # filename_output = f"result/{formatted_date}/generated_data_{num_train}.csv"
        
        # Complete features fields 
        fields = self.features
        fields = fields + ["predict", "loss", "gen_loss", "bd_loss"]
        print(fields)
        
        loss = 0
        generated_data = []
        predict = -999
        
        # Original X
        append = np.array([1, -999, -999, -999], dtype=np.float32).reshape(1, -1)
        input_data = np.hstack((X, append))
        all_data = []
        all_data.append(input_data)
        label_0_data = []
        print(self.generator.summary())
        for epoch in range(self.epochs):
            # Training Step
            loss, generated_data, predict, g_loss, bd_loss = self.train_step(X)

            # if epoch % 1000 == 0:
            #     print(f'Epoch {epoch}, Loss: {loss:.4f}, G_Loss: {g_loss:.4f}')
            #     print(f'predict: {predict}')
            #     #print(f'Epoch {epoch}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}')
            #     # Print the values of generated_features
            #     print("generated_data:", generated_data)
            #     print("----------------------------------------------------------------------")

            # Record generated data and loss
            # transform generated_data and predict to numpy array
            generated_data_np = generated_data.numpy()
            predict_np = np.array(predict, dtype=np.float32).reshape(1, -1)
            losses_np = np.array([loss, g_loss, bd_loss], dtype=np.float32).reshape(1, -1)
            # merge to shape (1, 10)
            merged_array = np.hstack((generated_data_np, predict_np))
            merged_array = np.hstack((merged_array, losses_np))
            all_data.append(merged_array)
            if predict == 0:
               label_0_data.append(merged_array)
        
        # print(all_data)
        # print(label_0_data)
        
        # Write to csv
        all_data_np = np.vstack(all_data)
        df = pd.DataFrame(all_data_np, columns=fields)
        df.to_csv(filename_output, mode='w', header=fields, index=False, encoding="utf-8")
        if label_0_data:
            label_0_data_np = np.vstack(label_0_data)
            df = pd.DataFrame(label_0_data_np, columns=fields)
            df.to_csv(os.path.join(output_dir, f"data_{self.num}_label_0.csv"), mode='w', header=fields, index=False, encoding="utf-8")
        # print(f'Epoch {epoch}, Gen Loss: {gen_loss:.4f}')
        # print(f'predict: {predict}')
        # print("----------------------------------------------------------------------")
        # print(f'cnt_one: {cnt_one}')
        # print(f'cnt_zero: {cnt_zero}')
        # return epoch, gen_loss, predict

