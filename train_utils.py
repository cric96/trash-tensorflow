from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import color, transform, feature
from sklearn.metrics import confusion_matrix
import itertools
import warnings

# Configurazione di Matplotlib
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
warnings.filterwarnings("ignore")

def load_dataset(path_dataset):
    paths = []
    for dirpath, dirnames, filenames in os.walk(path_dataset):
        paths.extend(map(lambda filename: os.path.join(dirpath, filename), filenames))
    return paths
    
def split_set(dataset, validation_percentage, test_percentage):
    validation_set_size = int(len(dataset) * validation_percentage)
    test_set_size = int(len(dataset) * test_percentage)
    # shuffle the data set
    training_set_shuffle = np.array(dataset)
    np.random.shuffle(training_set_shuffle)
    # extracting sets
    print("Total size: ", len(training_set_shuffle))
    
    # extract training set
    training_set_size = len(dataset) - validation_set_size - test_set_size
    training_set = training_set_shuffle[0:training_set_size]
    print("Training set size: ", len(training_set))
    
    # extract validation set
    validation_set_last_index = training_set_size + validation_set_size + 1
    validation_set = training_set_shuffle[training_set_size:validation_set_last_index]
    print("Validation set size: ", len(validation_set))
    
    # extract test set
    test_set_start_index = validation_set_last_index
    test_set = training_set_shuffle[test_set_start_index:]
    print("Test set size: ", len(test_set))
    
    return np.array([training_set, validation_set, test_set])

def load_label_sets(dataset_path, validation_percentage, test_percentage):
    labels = os.listdir(dataset_path)
    complete_set = np.array(labels, (4,))
    # for each load and split its dataset
    for label in labels:
        dataset = load_dataset(os.path.join(dataset_path, label))
        sets = split_set(dataset, validation_percentage, test_percentage)
        complete_set[label] = sets
    return complete_set

def add_label(array, label):
    length = len(array)
    new_arr = array.reshape(length, 1)
    return np.append(new_arr, np.full((length, 1), label), axis=1)


def make_dataframe(dataset_path, validation_percentage, test_percentage=0):
    labels = os.listdir(dataset_path)
    train = []
    validation = []
    test = []
    for label in labels:
        dataset = load_dataset(os.path.join(dataset_path, label))
        sets = split_set(dataset, validation_percentage, test_percentage)
        train.extend(add_label(sets[0], label))
        validation.extend(add_label(sets[1], label))
        if(test_percentage == 0):
            test.extend(add_label(sets[1], label))
        else:
            test.extend(add_label(sets[2], label))
    return [
        pd.DataFrame(train, columns=['path', 'label']),
        pd.DataFrame(validation, columns=['path', 'label']),
        pd.DataFrame(test, columns=['path', 'label'])
    ]

def plot_confusion_matrix(y_true, y_pred, classes,title='Confusion matrix',cmap=plt.cm.Blues,figsize=(8,6)):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(dpi=96,figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()