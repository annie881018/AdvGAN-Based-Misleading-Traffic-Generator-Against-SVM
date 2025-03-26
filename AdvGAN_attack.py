import tensorflow as tf
import numpy as np
import pandas as pd
import os
import Generator_Against_SVM
import Discriminator_SVM

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
        return self.cross_entropy(tf.zeros_like(d_output), d_output)
        # return -tf.reduce_mean(tf.zeros_like(d_output), d_output)

    # loss function to influence the perturbation to be as close to 0 as possible
    def perturb_loss(self, preds, thresh=0.3):
        zeros = tf.zeros(tf.shape(preds)[0])
        norm = tf.norm(preds, axis=1)
        return tf.reduce_mean(tf.maximum(zeros, norm - thresh))

    ATTACKED_FEATURES_MIN = [0.0, 2.0, 60.0, 54.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ATTACKED_FEATURES_MAX = [49962.0, 16.0, 640.0, 590.0, 49962.0, 0.0, 2.0, 4.0, 8.0, 12.0]

    # 訓練步驟
    #@tf.function
    def train_step(self, X):
        predict = -1
        with tf.GradientTape() as gen_tape:
            
            # Use generator to generate perturbation
            perturbation = self.generator(X)
            X_perturbed = tf.maximum(X + perturbation, 0)  # 確保 X_perturbed >= 0
            # X_perturbed = tf.stop_gradient(tf.clip_by_value(X_perturbed, self.ATTACKED_FEATURES_MIN, self.ATTACKED_FEATURES_MAX))
            # X_perturbed = tf.round(X_perturbed)
            # X_perturbed = self.ATTACKED_FEATURES_MIN + (self.ATTACKED_FEATURES_MAX - self.ATTACKED_FEATURES_MIN) * tf.tanh(X_perturbed)
            
            # 判別器判斷真實和假數據
            output = self.discriminator.decision_function(X_perturbed)
            predict = self.discriminator.predict(X_perturbed)
            # output = self.discriminator.decision_function(X_perturbed)
            # print(f'output:{output}')

            # 計算損失
            g_loss = self.generator_loss(output)
            
            perturb_loss = self.perturb_loss(perturbation, self.thresh)
            loss = self.alpha * g_loss + self.beta * perturb_loss

        # 計算梯度
        gradients_of_generator = gen_tape.gradient(loss, self.generator.trainable_variables)
        # print("Generator gradients:", gradients_of_generator)
        
        if any(g is None for g in gradients_of_generator):
            raise ValueError("One or more generator gradients are None.")

            # 應用梯度更新模型
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return loss, X_perturbed, predict, g_loss, perturb_loss


    # 訓練過程
    def train(self, X):
        num_train = f'{self.num}_lr_{self.lr}_a_{self.alpha}_b_{self.beta}_thsh_{self.thresh}'
        print(f"Start to train data {num_train}----------------------")
        filename_output = f"generated_data_{num_train}.csv"
        fields = self.features.append("label")
        # 先寫入欄位名稱
        df = pd.DataFrame(X, columns=fields)
        df.to_csv(filename_output, mode="w", index=False, encoding="utf-8")
        filename_losses = f"losses_{num_train}.txt"
        mode = "a"
        if os.path.exists(filename_losses):
            mode = "w"          # overwrite the old data
        file_losses = open(filename_losses, mode, encoding="utf-8")
        
        loss = 0
        generated_data = []
        predict = -999
        for epoch in range(self.epochs):
            # epoch += 1
            # 執行訓練步驟
            loss, generated_data, predict, g_loss, perturb_loss = self.train_step(X)

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}, G_Loss: {g_loss:.4f}')
                print(f'predict: {predict}')
                #print(f'Epoch {epoch}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}')
                # Print the values of generated_features
                print("generated_data:", generated_data)
                print("----------------------------------------------------------------------")

            # Record generated data and loss
            # transform generated_data and predict to numpy array
            generated_data_np = generated_data.numpy()
            predict_np = np.array(predict, dtype=np.float32).reshape(1, -1)
            # merge to shape (1, 10)
            merged_array = np.hstack((generated_data_np, predict_np))
            # 轉換為 DataFrame
            df = pd.DataFrame(merged_array, columns=fields)
            # Write to csv
            df.to_csv(filename_output, mode="a", header=False, index=False, encoding="utf-8")
            
            file_losses.write(f"loss: {loss},  G_Loss: {g_loss:.4f}  l_perturb: {perturb_loss:.4f}\n")
            
        file_losses.close()    
        # print(f'Epoch {epoch}, Gen Loss: {gen_loss:.4f}')
        # print(f'predict: {predict}')
        # print("----------------------------------------------------------------------")
        # print(f'cnt_one: {cnt_one}')
        # print(f'cnt_zero: {cnt_zero}')
        # return epoch, gen_loss, predict

