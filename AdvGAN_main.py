
import argparse
import numpy as np
import pandas as pd
import AdvGAN_based
from sklearn.utils import shuffle

ATTACKED_FEATURES_MIN = [0.0, 2.0, 60.0, 54.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ATTACKED_FEATURES_MAX = [49962.0, 16.0, 640.0, 590.0, 49962.0, 0.0, 2.0, 4.0, 8.0, 12.0]
features=[
                'flow duration', 'packet count', 'max pkt_length', 'min pkt_length',
                'max iat', 'min iat', 'fin count', 'syn count', 'psh count', 
                'ack count'
                ]
# features=[
#         'flow duration', 'packet count', 'max pkt_length', 'min pkt_length',
#         'max iat', 'min iat', 'fin count', 'syn count', 'psh count', 
#         'ack count', 'rst count', 'urg count'
#         ]

# Load data
def load_data(dataset):
    df = pd.read_csv(dataset)
    df_selected = df[df['label'] == 1]
    df_selected = df_selected.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    df_selected = df_selected.loc[(df_selected != 0).any(axis=1)]
    df_selected = shuffle(df_selected)
    y_train = np.array(df_selected['label'])
    X_train = np.array(df_selected[features])
    return X_train, y_train

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--thresh", type=float, default=10.0)
parser.add_argument("--alpha", type=float, default=10.0)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--num_data", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()
thresh = args.thresh
alpha = args.alpha
beta = args.beta
epochs = args.epochs
learning_rate = args.lr
num_data = args.num_data
batch_size = args.batch_size



Attack = AdvGAN_based.AdvGAN_attack(epochs=epochs, 
                                    features=features, 
                                    lr=learning_rate,
                                    thresh=thresh, 
                                    alpha=alpha, 
                                    beta=beta, 
                                    num=num_data,
                                    batch_size=batch_size)
# print(f"Load Data...")
# use dataset not for target model
X, y = load_data('dataset_slowloris_normal_0225.csv')
# print(f"Original Data: {X.shape}\n")

# Select one original attacked features
# X_train = X
# print(X)
X_train = X[:num_data]
y_train = y[:num_data]
# print(f"Start to train data {args.num_data}----------------------")
Attack.train(X_train, y_train)