"""
Steps to setup EMNIST
1. use LEAF setup script
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample
from 
https://github.com/TalwalkarLab/leaf/blob/master/data/femnist/README.md

move the generated train/ and test/ folders of json files to DATA_PATH/emnist

2. run script below on directory containing generated json files
"""
from pathlib import Path
import os
import pickle
import numpy as np
import json

data_path = Path(os.environ["DATA_PATH"]) / "emnist"
train_path = Path(os.environ["DATA_PATH"]) / "emnist" / "train"
test_path = Path(os.environ["DATA_PATH"]) / "emnist" / "test"

def setup_dataset():
    """
    json files from LEAF repo script --> individual npy files for each client
    """
    client_names = []
    for train_json_fname in os.listdir(train_path):
        with open(train_path / train_json_fname) as f:
            j = json.load(f)
            for client_name in j["users"]:
                client_names.append(client_name)
                client_data = j["user_data"][client_name]
                # data is originally a list of size 784 which has been normalized to [0,1]
                X_tr = np.stack([np.uint8(np.array(client_data["x"][i]).reshape(28,28)*255) for i in range(len(client_data["y"]))])
                y_tr = np.array(client_data["y"])
                np.save(train_path / f"{client_name}_X_tr.npy", X_tr)
                np.save(train_path / f"{client_name}_y_tr.npy", y_tr)
    for test_json_fname in os.listdir(test_path):
        with open(test_path / test_json_fname) as f:
            j = json.load(f)
            for client_name in j["users"]:
                client_data = j["user_data"][client_name]
                # data is originally a list of size 784 which has been normalized to [0,1]
                X_te = np.stack([np.uint8(np.array(client_data["x"][i]).reshape(28,28)*255) for i in range(len(client_data["y"]))])
                y_te = np.array(client_data["y"])
                np.save(test_path / f"{client_name}_X_te.npy", X_te)
                np.save(test_path / f"{client_name}_y_te.npy", y_te)
    with open(data_path / "client_names.json", "w") as f:
        json.dump(client_names, f, indent=4)

if __name__ == "__main__":
    setup_dataset()