from asyncio import base_tasks
from tabnanny import verbose
from sklearn import preprocessing
import torch
from sklearn.metrics import log_loss
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.cuda.amp import autocast
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import ipdb
from MLPModel import *
import csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train = np.load('.\\save\\train\\X_train3.npy', allow_pickle=True)
for i in range(len(X_train)):
    X_train[i][-3] = X_train[i][-3][0, 0]


n_samples, d_in = X_train.shape

y_train = np.load('.\\save\\train\\y_train3.npy', allow_pickle=True)
print(X_train[0])
X_train = np.clip(X_train, -3e38, 3e38)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1, shuffle = True)

def run_classifier(classifier, classifier_name, compute_loss = True):
    print(f'Fitting classifier {classifier_name}.')
    classifier.fit(X_train, y_train)
    print(f'Computing predictions of {classifier_name}.')
    y_pred = list(classifier.predict(X_train))
    if compute_loss:
        y_logits = classifier.predict_proba(X_train)
        nll = log_loss(y_train, y_logits[:, 1])
    else:
        nll = 0
    acc = sum(y_pred==y_train)/len(y_pred)
    y_pred_test = list(classifier.predict(X_test))
    if compute_loss:
        y_logits_test = classifier.predict_proba(X_test)
        nll_test = log_loss(y_test, y_logits_test[:, 1])
    else:
        nll_test = 0
    acc_test = sum(y_pred_test==y_test)/len(y_pred_test)
    print(f"NLL Training Loss for {classifier_name}: {nll}")
    print(f"Training Accuracy for {classifier_name}: {acc}")
    print(f"NLL Testing Loss for {classifier_name}: {nll_test}")
    print(f"Testing Accuracy for {classifier_name}: {acc_test}")
    return nll, acc, nll_test, acc_test

def run_MLP(model):
    y_logits = np.exp(model(torch.from_numpy(X_train).float().to(device)).cpu().detach().numpy())
    y_pred = np.argmax(y_logits, axis = 1)
    y_logits = np.clip(y_logits[:, 1], 1e-6, 1 - 1e-6)
    nll = log_loss(y_train, y_logits)
    acc = np.sum(y_pred==y_train)/len(y_pred)
    y_logits_test = np.exp(model(torch.from_numpy(X_test).float().to(device)).cpu().detach().numpy())
    y_pred_test = np.argmax(y_logits_test, axis = 1)
    y_logits_test = np.clip(y_logits_test[:, 1], 1e-6, 1 - 1e-6)
    nll_test = log_loss(y_test, y_logits_test)
    acc_test = np.sum(y_pred_test==y_test)/len(y_pred_test)
    print(f"NLL Training Loss for MLP: {nll}")
    print(f"Training Accuracy for MLP: {acc}")
    print(f"NLL Testing Loss for MLP: {nll_test}")
    print(f"Testing Accuracy for MLP: {acc_test}")
    return nll, acc, nll_test, acc_test

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = MLP(d_in, [64, 32], 2, 0.11524674093760068, 0.28241827263058633)
model = model.to(device)
best_epoch = 130
model.load_state_dict(torch.load(f'.\\save\\models\\epoch{best_epoch}.pt'))
model.eval()

res = {}
#res["MLP"] = run_MLP(model)
#res["Random Forest"] = run_classifier(RandomForestClassifier(n_estimators=100, n_jobs = -1), "Random Forest")
#res["Logistic Regression"] = run_classifier(LogisticRegression(n_jobs = -1), "Logistic Regression")
#res["NN"] = run_classifier(MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(64, 32), random_state=1), "NN MLP with Adam")
#res["KNN"] = run_classifier(KNeighborsClassifier(3, n_jobs = -1), "KNN k=3")
#res["SVM"] = run_classifier(svm.LinearSVC(verbose=3), "SVM", compute_loss = False)
#res["Gaussian"] = run_classifier(GaussianNB(verbose=3), "Naive Bayes")
#res["AdaBoost"] = run_classifier(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=300, algorithm='SAMME.R', verbose=3), "AdaBoost")

ipdb.set_trace()