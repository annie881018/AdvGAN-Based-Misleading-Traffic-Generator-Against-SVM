import tensorflow as tf
import numpy as np
import pandas as pd
import os
import Generator_Against_SVM
import Discriminator_SVM
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
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.5, beta_2=0.999)
        self.generator = Generator_Against_SVM.build_generator(feature_dim=self.feature_dim)
        self.discriminator = Discriminator_SVM.build_discriminator()
        self.num = num

    def generator_loss(self, d_output):
        # determine the margin of decision boundary
        w = self.discriminator.coef_[0]
        w_norm = np.linalg.norm(w)
        margin = 2 / w_norm
        # 使用 hinge loss，目標是讓 d_output < 0
        return tf.reduce_mean(tf.maximum(0.0, d_output + margin))  # margin = 0.1

    # loss function to influence the output close to boundary
    def boundary_distance_loss(self, x):
        x = tf.reshape(x, (1, -1))
        # get weigth(w) and bias(b) of discriminator
        w = self.discriminator.coef_[0]
        b = self.discriminator.intercept_[0]
        # 轉為 TensorFlow tensor
        w_tf = tf.constant(w.reshape(-1, 1), dtype=tf.float32)  # shape: (feature_dim, 1)
        b_tf = tf.constant(b, dtype=tf.float32)                 # scalar
        logits = tf.matmul(x, w_tf) + b_tf                 # shape: (batch_size, 1)
        distance = tf.abs(logits) / tf.norm(w_tf)          # shape: (batch_size, 1)
        return -tf.reduce_mean(distance)
        return tf.reduce_mean(tf.square(preds))  # L2 範數平方，懲罰過大的微擾

    ATTACKED_FEATURES_MIN = [0.0, 2.0, 60.0, 54.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ATTACKED_FEATURES_MAX = [49962.0, 16.0, 640.0, 590.0, 49962.0, 0.0, 2.0, 4.0, 8.0, 12.0]

    # 訓練步驟
    #@tf.function
    def train_step(self, X):
        predict = -1
        noise = tf.random.normal([1, 100])
        with tf.GradientTape() as gen_tape:
            
            # Use generator to generate perturbation
            # perturbation = self.generator(X)
            # generated_data = tf.maximum(X + perturbation, 0)  # 確保 generated_data >= 0
            # generated_data = tf.stop_gradient(tf.clip_by_value(generated_data, self.ATTACKED_FEATURES_MIN, self.ATTACKED_FEATURES_MAX))
            # generated_data = tf.round(generated_data)
            # generated_data = self.ATTACKED_FEATURES_MIN + (self.ATTACKED_FEATURES_MAX - self.ATTACKED_FEATURES_MIN) * tf.tanh(generated_data)
            generated_data = self.generator(noise)
            generated_data = tf.clip_by_value(generated_data, self.ATTACKED_FEATURES_MIN, self.ATTACKED_FEATURES_MAX)
            # 判別器判斷真實和假數據
            output = self.discriminator.decision_function(generated_data)
            predict = self.discriminator.predict(generated_data)
            # output = self.discriminator.decision_function(generated_data)
            # print(f'output:{output}')

            # 計算損失
            g_loss = self.generator_loss(output)
            bd_loss = self.boundary_distance_loss(generated_data)
            # perturb_loss = self.perturb_loss(perturbation, self.thresh)
            # loss = self.alpha * g_loss + self.beta * perturb_loss
            loss = self.alpha * g_loss + self.beta * bd_loss

        # 計算梯度
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
        # 設定輸出目錄
        output_dir = f"result/{formatted_date}"

        # 如果目錄不存在，則創建它
        os.makedirs(output_dir, exist_ok=True)

        # 生成完整的輸出檔案路徑
        filename_output = os.path.join(output_dir, f"generated_data_{num_train}.csv")
        # filename_output = f"result/{formatted_date}/generated_data_{num_train}.csv"
        fields = self.features
        fields = fields + ["predict", "loss", "gen_loss", "db_loss"]
        print(fields)
        mode = "a"
        
        loss = 0
        generated_data = []
        predict = -999
        for epoch in range(self.epochs):
            # epoch += 1
            # 執行訓練步驟
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
            # 轉換為 DataFrame
            df = pd.DataFrame(merged_array)
            # Write to csv
            df.to_csv(filename_output, mode=mode, header=fields, index=False, encoding="utf-8")
            if predict == 0:
                df.to_csv(os.path.join(output_dir, "label_0.csv"), mode='a', header=fields, index=False, encoding="utf-8")
            fields = False
        # print(f'Epoch {epoch}, Gen Loss: {gen_loss:.4f}')
        # print(f'predict: {predict}')
        # print("----------------------------------------------------------------------")
        # print(f'cnt_one: {cnt_one}')
        # print(f'cnt_zero: {cnt_zero}')
        # return epoch, gen_loss, predict

