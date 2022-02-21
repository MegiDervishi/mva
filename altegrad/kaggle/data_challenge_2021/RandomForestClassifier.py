from tabnanny import verbose
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import csv
import ipdb
import pandas as pd
import matplotlib.pyplot as plt

X_train = np.load('.\\save\\train\\X_train3.npy', allow_pickle= True)
for i in range(len(X_train)):
    X_train[i][-3] = X_train[i][-3][0, 0]
y_train = np.load('.\\save\\train\\y_train3.npy', allow_pickle= True)
X_train = np.clip(X_train, -3e38, 3e38)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = np.load('.\\save\\test\\X_test3.npy', allow_pickle= True)
for i in range(len(X_test)):
    X_test[i][-3] = X_test[i][-3][0, 0]
X_test = scaler.fit_transform(X_test)

model =RandomForestClassifier(n_estimators=100, n_jobs = -1, verbose=3)
model.fit(X_train, y_train)

'''
ipdb.set_trace()
'''

preds = model.predict_proba(X_test)
preds = preds[:, 1]
predictions = zip(range(len(preds)), preds)
with open("submission.csv","w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','predicted'])
    for row in predictions:
        csv_out.writerow(row)

ipdb.set_trace()