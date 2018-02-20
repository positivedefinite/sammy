import data_helpers_neutrals
import numpy as np
#import pandas as pd

    
def load_data(datasets, howMuchToLoad, training_split_size):
    print("Load data from the following datasets: "+str(datasets))
    x, y, vocabulary, vocabulary_inv_list, z = data_helpers_neutrals.load_data(howMuchToLoad,datasets)
    print("Data_helpers_neutral.py has finished loading data. Preparing training/test split.")
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    y = y.argmax(axis=1)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * training_split_size)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]
    print("Training and test data ready. ETL complete.")
    return x_train, y_train, x_test, y_test, vocabulary_inv, z

def main(datasets, howMuchToLoad, training_split_size):
    x_train, y_train, x_test, y_test, vocabulary_inv, neutral_tweets = load_data(datasets, howMuchToLoad, training_split_size)
    return x_train, y_train, x_test, y_test, vocabulary_inv, neutral_tweets