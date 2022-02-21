"""
Learning on Sets - ALTEGRAD - Jan 2022
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    ##################
    X_train, y_train = [], []
    
    for i in range(n_train):
        x = np.random.randint(1, max_train_card, size=np.random.randint(1,max_train_card))
        x = np.concatenate((np.random.randint(1, size=max_train_card-len(x)),x))
        X_train.append(x)
        y_train.append(sum(x))
        
    ##################

    return np.array(X_train), np.array(y_train)


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    n_test = 200000
    X_test, y_test = [], []
    
    for length in range(5,105,5):
        x = np.random.randint(1,10, size=(10000, length))
        X_test.append(x)
        y_test.append(np.sum(x, axis=1))
        
    ##################

    return X_test, y_test

#print(create_train_dataset())
#print(len(create_test_dataset()[0]))