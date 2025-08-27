import tensorflow as tf
import numpy as np
import pandas as pd
import os
import Modified_GAN_Generator
import Modified_GAN_Target_Model
from Filter import filtering
from datetime import datetime
import joblib
import logging
import matplotlib.pyplot as plt

class AdvGAN_attack:
    def __init__(self,
                 epochs,
                 features,
                 MIN,
                 MAX,
                 lr=0.000001,
                 thresh=0.3,
                 alpha=0.1,
                 beta=0.1,
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
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.generator = Modified_GAN_Generator.build_generator(feature_dim=self.feature_dim)
        self.discriminator = Modified_GAN_Target_Model.build_discriminator()
        self.num = num
        self.w = tf.constant(self.discriminator.coef_[0].reshape(-1, 1), dtype=tf.float32)  # shape: (feature_dim, 1)
        self.b = tf.constant(self.discriminator.intercept_[0], dtype=tf.float32)                 # scalar
        self.ATTACKED_FEATURES_MIN = MIN
        self.ATTACKED_FEATURES_MAX = MAX
        self.batch_size = batch_size
    
    # loss function for influencing the output close to label 0
    def generator_loss_zero_like(self, d_predict):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.zeros_like(d_predict), d_predict)
    
    # loss function for influencing the output close to 0
    def generator_loss(self, d_output):
        # determine the margin of decision boundary
        w = self.discriminator.coef_[0]
        w_norm = np.linalg.norm(w)
        margin = 2 / w_norm
        return tf.reduce_mean(tf.maximum(0.0, d_output + 1.0))    # loss=0 if predict label 0
        # return tf.reduce_mean(tf.maximum(0.0, d_output + margin)) # loss=0 if predict to 0 less than margin
    
    # loss function for influencing the output close to margin
    def margin_loss(self, x):
        # Caculate |w^T x + b - target_plane| / ||w||
        dot = tf.tensordot(x, self.w, axes=1)
        distance = tf.abs(dot + self.b - (-1))
        normalized = distance / tf.norm(self.w)
        return tf.reduce_mean(normalized)
    
    # loss function for influencing the output close to boundary
    def boundary_distance_loss(self, x, thresh):
        # Caculate |w^T x + b| / ||w||
        # x = tf.reshape(x, (1, -1))
        logits = tf.matmul(x, self.w) + self.b               # shape: (batch_size, 1)
        distance = tf.abs(logits) / tf.norm(self.w)          # shape: (batch_size, 1)
        # 只懲罰 logits 遠離邊界（比如 margin 超過 1.0 就加壓）
        # penalty = tf.maximum(0.0, distance - thresh)
        # return -tf.reduce_mean(penalty)
        return -tf.reduce_mean(distance)

    # Training steps
    #@tf.function
    def train_step(self, noise_batch):
        predict = -1
        with tf.GradientTape() as gen_tape:

            generated_data = self.generator(noise_batch)
            # 拆出第 0（flow duration）與第 4（max iat）維
            flow_duration = generated_data[:, 0:1]
            max_iat = generated_data[:, 4:5]

            # 其他部分：
            part_1_to_3 = generated_data[:, 1:4]   # index 1~3
            part_5_to_9 = generated_data[:, 5:]    # index 5~9

            # 放大比例：z[:, 0] ∈ [-1, 1] → scale_factor ∈ [0, 25]
            scale_factor = (noise_batch[:, 0:1] + 1.0) * 25.0

            # 放大
            flow_duration_scaled = flow_duration * scale_factor
            max_iat_scaled = max_iat * scale_factor

            # 重新組合：
            generated_data = tf.concat([
                flow_duration_scaled,   # index 0
                part_1_to_3,            # index 1~3
                max_iat_scaled,         # index 4
                part_5_to_9             # index 5~9
            ], axis=1)
            generated_data = tf.clip_by_value(generated_data, self.ATTACKED_FEATURES_MIN, self.ATTACKED_FEATURES_MAX)

            # Pass generated data to Discriminator to get predict label,
            # the range of output is float between -1(label 0) and +1(label 1),
            # predict is 0 or 1.
            output = self.discriminator.decision_function(generated_data)
            predict = self.discriminator.predict(generated_data)

            # Caculate the loss
            # g_loss = self.generator_loss(output)
            g_loss = self.margin_loss(generated_data)
            # g_loss = self.generator_loss_zero_like(predict)

            bd_loss = self.boundary_distance_loss(generated_data, self.thresh)
            # perturb_loss = self.perturb_loss(perturbation, self.thresh)
            # 取出第 0 維特徵（flow duration）
            flow_duration = generated_data[:, 0]
            max_iat = generated_data[:, 4]
            # 我們希望 flow_duration 越大越好，所以用 -mean()
            duration_loss = -tf.reduce_mean(flow_duration)
            max_iat_loss = -tf.reduce_mean(max_iat)
            # loss = self.alpha * g_loss + self.beta * perturb_loss
            loss = self.alpha * g_loss + self.beta * bd_loss + 0.5 * duration_loss + 0.2 * max_iat_loss
        # Caculate gradients
        gradients_of_generator = gen_tape.gradient(loss, self.generator.trainable_variables)
        # print("Generator gradients:", gradients_of_generator)
        
        if any(g is None for g in gradients_of_generator):
            raise ValueError("One or more generator gradients are None.")

        # Apply gradients to optimize model
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return loss, generated_data, predict, g_loss, bd_loss


    # Traing process
    def train(self):
        # Set output csv
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
        fields = fields + ["predict"]

        print(fields)

        loss = 0
        generated_data = []
        predict = -999

        all_data = pd.DataFrame([], columns=fields)
        label_0_data = []
        accuracy = []
        batches = self.num // self.batch_size
        print(self.generator.summary())
        logging.basicConfig(
            filename="model.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.info(f"Train Num: {self.num}, Thresh: {self.thresh}, Alpha: {self.alpha}, Beta: {self.beta}.")
        logging.info("Start training...")
        for epoch in range(self.epochs):
            # Start to train
            epoch_data = []
            passed = -1
            total = -1
            acc = -1
            successful_attack = -1
            attack_success_rate = -1
            generated_df = []
            for step in range(batches):
              noise = tf.random.normal([self.batch_size, 100])
              loss, generated_data, predict, g_loss, bd_loss = self.train_step(noise)
              # collect data only in the last epoch


              generated_data_np = generated_data.numpy()
              predict_np = np.array(predict, dtype=np.float32).reshape(-1, 1)
              merged = np.hstack([generated_data_np, predict_np])
              epoch_data.append(merged)
            generated_df = pd.DataFrame(np.vstack(epoch_data), columns=fields)
            filtered_df = filtering(generated_df)
            total = len(generated_df)
            passed = len(filtered_df)
            successful_attack = filtered_df[filtered_df["predict"] == 0]
            if total > 0:
                acc = passed / total
                # all_data = pd.concat([all_data, successful_attack], ignore_index=True)
                attack_success_rate = len(successful_attack) / len(generated_df)
                accuracy.append(attack_success_rate)
            else:
              print("Error: generated_df")
            if epoch % 100 == 0 or epoch == self.epochs - 1:
                print(f"[Epoch {epoch}] 條件篩選後通過比例: {passed}/{total} = {acc:.2%}")
                # print(f'gen_loss: {g_loss:.4f}, disc_loss: {bd_loss:.4f}')
                # print(f'predict: {predict}')
                print(f"[Epoch {epoch}] 條件篩選後成功攻擊比例: {len(successful_attack)}/{len(generated_df)} = {attack_success_rate:.2%}")
                # print(f"成功攻擊樣本數：{len(all_data)}")
                print(f"Avg 成功攻擊比例: {(sum(accuracy) / len(accuracy)) * 100}%")
                print(f"loss: {loss:.4f}, g_loss: {g_loss:.4f}, bd_loss: {bd_loss:.4f}")
                print(f'predict: {predict}')

                if epoch == self.epochs - 1:
                  successful_attack.to_csv(filename_output, mode='w', header=fields, index=False, encoding="utf-8")
                  # print(accuracy)
                  print(len(accuracy))
                  print(f"Avg accuracy: {(sum(accuracy) / len(accuracy)) * 100}%")
                  print(f"Save Generator...")
                  logging.info(f"Finish training.\nAvg accuracy: {(sum(accuracy) / len(accuracy)) * 100}%\nloss: {loss:.4f}, g_loss: {g_loss:.4f}, bd_loss: {bd_loss:.4f}")
                  joblib.dump(self.generator, "Generator.pkl")
                print('-' * 10)
        plt.plot(accuracy)
        plt.xlabel("Epoch")
        plt.ylabel("Attack Success Rate")
        plt.title("AdvGAN Attack Success Rate Over Epochs")
        plt.grid(True)

        os.makedirs("fig", exist_ok=True)  # 若資料夾不存在則建立
        plt.savefig(os.path.join("fig", "Attack_Success_Rate.jpg"))
        