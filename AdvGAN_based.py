import tensorflow as tf
import numpy as np
import pandas as pd
import os
import Generator
import AdvGAN_Target_Model
import AdvGAN_Discriminator
from datetime import datetime
import joblib
import logging
import matplotlib.pyplot as plt
    
class AdvGAN_attack:
    def __init__(self,
                 epochs,
                 features,
                 lr=0.00001,
                 thresh=0.3,
                 alpha=1,
                 beta=5,
                 num=0,
                 batch_size=32):
        self.epochs = epochs
        self.features = features
        self.feature_dim = len(self.features)
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.thresh = thresh
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.generator = Generator.build_generator(feature_dim=self.feature_dim)
        self.discriminator = AdvGAN_Discriminator.build_discriminator(feature_dim=self.feature_dim)
        self.target_model = AdvGAN_Target_Model.build_discriminator()
        self.num = num
        self.batch_size = batch_size
        # self.w = tf.constant(self.discriminator.coef_[0].reshape(-1, 1), dtype=tf.float32)  # shape: (feature_dim, 1)
        # self.b = tf.constant(self.discriminator.intercept_[0], dtype=tf.float32)                 # scalar
        self.ATTACKED_FEATURES_MIN = [0.0, 2.0, 60.0, 54.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ATTACKED_FEATURES_MAX = [49962.0, 16.0, 640.0, 590.0, 49962.0, 0.0, 2.0, 4.0, 8.0, 12.0]
    
    # loss function to encourage misclassification after perturbation
    def adv_loss(self, probs):
        # return tf.reduce_sum(tf.maximum(0.0, probs))
        # 假設 probs 是 (batch_size, 2)，第二個維度是 [label=0, label=1]
        probs = tf.convert_to_tensor(probs, dtype=tf.float32)
        return tf.reduce_mean(probs[:, 0])  # 越低越好

    # loss function to influence the perturbation to be as close to 0 as possible
    def perturb_loss(self, perturb, thresh=3.0):
        norm = tf.norm(perturb, axis=1)
        return tf.reduce_mean(tf.maximum(0.0, norm - thresh))
    
    # loss function for influencing the output close to label 0
    def generator_loss_zero_like(self, d_predict):
        return self.cross_entropy(tf.zeros_like(d_predict), d_predict)
    
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
        penalty = tf.maximum(0.0, distance - thresh)
        # return -tf.reduce_mean(penalty)
        return -tf.reduce_mean(distance)

    # 訓練步驟
    #@tf.function
    def train_step(self, X):
        predict = -1
        noise = tf.random.normal([1, 100])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            # Use generator to generate perturbation
            X = tf.cast(X, tf.float32)  # 轉型，確保 X 是 float32
            perturbation = self.generator(X)
            generated_data = tf.maximum(X + perturbation, 0)  # 確保 generated_data >= 0
            generated_data = tf.clip_by_value(generated_data, self.ATTACKED_FEATURES_MIN, self.ATTACKED_FEATURES_MAX)
            
            # pass origin and perturbed image to discriminator and the target model
            d_origin_logits, d_origin_probs = self.discriminator(X)
            d_perturbed_logits, d_perturbed_probs = self.discriminator(generated_data)
            t_perturbed_logits = self.target_model.decision_function(generated_data)
            t_perturbed_probs = self.target_model.predict_proba(generated_data)
            
            # Cal loss
            # perturb loss
            l_perturb = tf.reduce_mean(self.perturb_loss(perturbation, self.thresh))
            
            # adversary loss
            l_adv = tf.reduce_mean(self.adv_loss(t_perturbed_probs))
            
            # generator loss
            g_loss_perturb = tf.reduce_mean(self.mse(tf.zeros_like(d_perturbed_probs), d_perturbed_probs))
            gen_loss = l_adv + self.alpha * g_loss_perturb + self.beta * l_perturb
            
            # discriminator loss
            d_loss_origin = self.mse(tf.ones_like(d_origin_probs), d_origin_probs)
            d_loss_perturb = self.mse(tf.zeros_like(d_perturbed_probs), d_perturbed_probs)
            disc_loss = d_loss_origin + d_loss_perturb
            
        predict = self.target_model.predict(generated_data)
        # 計算梯度
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        # print("Generator gradients:", gradients_of_generator)
        
        if any(g is None for g in gradients_of_generator):
            raise ValueError("One or more generator gradients are None.")

        # 應用梯度更新模型
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return generated_data, predict, gen_loss, disc_loss


    # 訓練過程
    def train(self, X, y):
        # num_train = f'{self.num}_lr_{self.lr}_a_{self.alpha}_b_{self.beta}_thsh_{self.thresh}'
        # num_train = f'{self.num}'
        # # print(f"Start to train data {num_train}----------------------")
        # # formatted_date = datetime.today().strftime("%m%d")
        # formatted_date = f'lr_{self.lr}_a_{self.alpha}_b_{self.beta}_thsh_{self.thresh}'
        # # 設定輸出目錄
        # output_dir = f"result/{formatted_date}"

        # # 如果目錄不存在，則創建它
        # os.makedirs(output_dir, exist_ok=True)

        # # 生成完整的輸出檔案路徑
        # filename_output = os.path.join(output_dir, f"generated_data_{num_train}.csv")
        # # filename_output = f"result/{formatted_date}/generated_data_{num_train}.csv"
        # fields = self.features
        # fields = fields + ["predict", "gen_loss", "disc_loss"]
        # print(fields)
        
        # batch process
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(self.batch_size)
        
        generated_data = []
        batch_predict = -999
        
        # append = np.array([1, -999, -999], dtype=np.float32).reshape(1, -1)
        # input_data = np.hstack((X, append))
        # all_data = []
        # all_data.append(input_data)
        label_0_data = []
        success_rates_per_epoch = []
        # gen_loss_per_epoch = []
        # disc_loss_per_epoch = []
        print(self.generator.summary())
        
        logging.basicConfig(
            filename="model.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.info(f"Train Num: {self.num}, Thresh: {self.thresh}, Alpha: {self.alpha}, Beta: {self.beta}.")
        logging.info("Start training...")
        print(f"Total batches: {len(dataset)}")
        print(f"Total X: {len(X)}")
        for epoch in range(self.epochs):
            epoch_success_rates = []
            # epoch_gen_loss = []
            # epoch_disc_loss = []
            print(f"[Epoch {epoch}]")
            for step, (batch_x, batch_y) in enumerate(dataset):
                # epoch += 1
                # 執行訓練步驟
                generated_data, batch_predict, gen_loss, disc_loss = self.train_step(batch_x)
                
                # calculate accuracy
                y_true = batch_y.numpy()
                y_pred = batch_predict
                mask_attack = (y_true == 1)
                mask_success = (y_true == 1) & (y_pred == 0)
                
                if np.sum(mask_attack) > 0:
                    success = np.sum(mask_success) / np.sum(mask_attack)
                    epoch_success_rates.append(success)    
            
            
            # 記錄這個 epoch 的平均攻擊成功率
            if epoch_success_rates:
                mean_success = np.mean(epoch_success_rates)
            else:
                mean_success = 0.0            
            success_rates_per_epoch.append(mean_success)
            
            
            # 記錄這個 epoch 的平均gen_loss
            # if epoch_gen_loss:
            #     mean_gen_loss = np.mean(epoch_gen_loss)
            # else:
            #     mean_gen_loss = 0.0            
            # gen_loss_per_epoch.append(mean_gen_loss)
            
            # # 記錄這個 epoch 的平均disc_loss
            # if epoch_disc_loss:
            #     mean_disc_loss = np.mean(epoch_disc_loss)
            # else:
            #     mean_disc_loss = 0.0            
            # disc_loss_per_epoch.append(mean_disc_loss)

            if epoch % 10 == 0:
                print(f"[Epoch {epoch}] Adv Attack Success Rate: {mean_success:.2%}")
                print(f'Epoch {epoch}, gen_loss: {gen_loss:.4f}, disc_loss: {disc_loss:.4f}')
                print(f'predict: {batch_predict}')
                #print(f'Epoch {epoch}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}')
                # Print the values of generated_features
                # print("generated_data:", generated_data)
                print("----------------------------------------------------------------------")

                # Record generated data and loss
                # transform generated_data and predict to numpy array
                # generated_data_np = generated_data.numpy()
                # predict_np = np.array(predict, dtype=np.float32).reshape(1, -1)
                # losses_np = np.array([gen_loss, disc_loss], dtype=np.float32).reshape(1, -1)
                # # merge to shape (1, 10)
                # merged_array = np.hstack((generated_data_np, predict_np))
                # merged_array = np.hstack((merged_array, losses_np))
                # all_data.append(merged_array)
                # if predict == 0:
                #     label_0_data.append(merged_array)
        accuracy = np.mean(success_rates_per_epoch)
        # loss_gen = np.mean(gen_loss_per_epoch)
        logging.info(f"Finish training.\nAccuracy: {accuracy:.4f}, Gen loss: {gen_loss:.4f}, Disc loss: {disc_loss:.4f}")
        plt.plot(success_rates_per_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Attack Success Rate")
        plt.title("AdvGAN Attack Success Rate Over Epochs")
        plt.grid(True)

        os.makedirs("fig", exist_ok=True)  # 若資料夾不存在則建立
        plt.savefig(os.path.join("fig", "Attack_Success_Rate.jpg"))
        
        print(f"Save Generator...")
        joblib.dump(self.generator, "Generator.pkl")
        logging.info(f"Save generator.")
        # print(f"Save Disciminator...")
        # joblib.dump(self.discriminator, "Discriminator.pkl")
        # logging.info(f"Save discriminator.")
        # all_data_np = np.vstack(all_data)
        # df = pd.DataFrame(all_data_np, columns=fields)
        # df.to_csv(filename_output, mode='w', header=fields, index=False, encoding="utf-8")
        # if label_0_data:
        #     label_0_data_np = np.vstack(label_0_data)
        #     df = pd.DataFrame(label_0_data_np, columns=fields)
        #     df.to_csv(os.path.join(output_dir, f"data_{self.num}_label_0.csv"), mode='w', header=fields, index=False, encoding="utf-8")

