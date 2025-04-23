
import argparse
import numpy as np
import pandas as pd
import AdvGAN_attack

if __name__ == "__main__":
    ATTACKED_FEATURES_MIN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 60.0, 0.0, 0.0, 0.0]
    ATTACKED_FEATURES_MAX = [2.0, 4.0, 8.0, 12.0, 8.0, 12.0, 16.0, 1843.0, 640.0, 49962.0, 0.0, 49962.0]
#     features=[
#             'flow duration', 'packet count', 'max pkt_length', 'min pkt_length',
#             'max iat', 'min iat', 'fin count', 'syn count', 'psh count', 
#             'ack count', 'rst count', 'urg count'
#             ]
    features=[
        'fin count', 'syn count', 'psh count', 'ack count',
        'rst count', 'urg count', 'packet count', 'pkt_length', 'max pkt_length', 'flow duration', 'min iat',
        'max iat'
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
    Attack.train()