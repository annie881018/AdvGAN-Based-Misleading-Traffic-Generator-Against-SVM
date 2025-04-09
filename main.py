
import argparse
import numpy as np
import pandas as pd
import AdvGAN_attack


# Load data
def load_data(dataset):
    df = pd.read_csv(dataset)
    df_selected = df[df['label'] == 1]
    df_selected = df_selected.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    df_selected = df_selected.loc[(df_selected != 0).any(axis=1)]
    #df_selected = shuffle(df_selected)
    X_train = np.array(df_selected[features])
    return X_train

if __name__ == "__main__":
    ATTACKED_FEATURES_MIN = [0.0, 2.0, 60.0, 54.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ATTACKED_FEATURES_MAX = [49962.0, 16.0, 640.0, 590.0, 49962.0, 0.0, 2.0, 4.0, 8.0, 12.0]
    features=[
            'flow duration', 'packet count', 'max pkt_length', 'min pkt_length',
            'max iat', 'min iat', 'fin count', 'syn count', 'psh count', 
            'ack count'
            ]
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresh", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num_data", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10000)
    args = parser.parse_args()
    thresh = args.thresh
    alpha = args.alpha
    beta = args.beta
    epochs = args.epochs
    learning_rate = args.lr
    num_data = args.num_data



    Attack = AdvGAN_attack.AdvGAN_attack(epochs=epochs, features=features, lr=learning_rate,thresh=thresh, alpha=alpha, beta=beta, num=num_data)
    # print(f"Load Data...")
    X = load_data('dataset_slowloris_normal_0225.csv')
    # print(f"Original Data: {X.shape}\n")

    # Select one original attacked features
    X_train = X[num_data:num_data+1]
    # print(X_train)
    # print(f"Start to train data {args.num_data}----------------------")
    Attack.train(X_train)