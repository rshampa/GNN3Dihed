# Some utility functions
import numpy as np

def print_sample_shape(sample):
    feature_names = [
        "atomic feature vectors",
        "bond feature vectors",
        "bond indices",
        "angle features",
        "angle indices",
        "global molecular features"
    ]

    for i in range(len(sample[0])):
        print(feature_names[i] + ":", sample[0][i].size())

def one_hot_decode(one_hot_vector, options):
    return options[np.argmax(one_hot_vector)]

def one_hot_encode(selection, options):
    if selection not in options:
        selection = options[-1]

    return [int(boolean_value) for boolean_value in list(map(lambda s: selection == s, options))]

#####===========================================
def split_dataset(dataset, ratio):
    np.random.seed(120734)
    shuffled_indices = np.random.permutation(len(dataset))
    split_index = int(ratio * len(dataset))
    train_indices = shuffled_indices[:split_index]
    test_indices = shuffled_indices[split_index:]
    train_dataset = dataset.iloc[train_indices].reset_index(drop=True)
    test_dataset = dataset.iloc[test_indices].reset_index(drop=True)
    return train_dataset, test_dataset
#####===========================================
