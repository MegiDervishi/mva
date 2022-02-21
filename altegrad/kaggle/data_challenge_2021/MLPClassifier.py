from asyncio import base_tasks
from tabnanny import check
from sklearn import preprocessing
import torch
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

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
X_train = np.load('.\\save\\train\\X_train3.npy', allow_pickle=True)
for i in range(len(X_train)):
    X_train[i][-3] = X_train[i][-3][0, 0]

y_train = np.load('.\\save\\train\\y_train3.npy', allow_pickle=True)
print(X_train[0])
X_train = np.clip(X_train, -3e38, 3e38)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)

#X_train = SelectKBest(f_classif, k=8).fit_transform(X_train, y_train)

X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
X_train, y_train = X_train.float(), y_train.type(torch.LongTensor)
X_train, y_train = X_train.to(device), y_train.to(device)

n_samples, d_in = X_train.shape
batch_size = 512
val_batch_size = 512

print(X_train.shape, y_train.shape)
dataset = TensorDataset(X_train, y_train)
train_samples = 90 * n_samples//100
val_samples = n_samples - train_samples
train_dataset, val_dataset = random_split(dataset, [train_samples, val_samples])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

classifier = MLP(d_in, [64, 32], 2, 0.11524674093760068, 0.28241827263058633)
classifier = classifier.to(device)
print(classifier)

load_checkpoint = True
checkpoint = 99
if load_checkpoint:
    print(f'Loading checkpoint {checkpoint}')
    classifier.load_state_dict(torch.load(f'.\\save\\models\\epoch{checkpoint}.pt'))
    print('Loaded Checkpoint.')
else:
    checkpoint = 0

criterion = nn.NLLLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.0001743680699626652, weight_decay=1.3918554102301297e-05, momentum = 0.9)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader))
epochs = 100

train = True
best_epoch = 7
best_val_loss = np.inf
train_losses = []
val_losses = []

if train:
    #torch.autograd.set_detect_anomaly(True)
    for epoch in tqdm(range(checkpoint, checkpoint + epochs)):  # loop over the dataset multiple times
        running_loss = 0.0
        classifier.train()
        for i, data in tqdm(enumerate(train_dataloader), leave = False, total = len(train_dataloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #if torch.isnan(inputs).any():
            #    raise RuntimeError('Found nan input.')
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            with autocast():
                outputs = classifier(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            # print statistics
            running_loss += loss.item()/len(train_dataloader)
            #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader))
        print(f'\nEpoch {epoch+1}, average training loss: {running_loss}.\n')
        train_losses.append(running_loss)
        classifier.eval()
        val_loss = 0
        for i, data in enumerate(val_dataloader):
            inputs, labels = data
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()/len(val_dataloader)
        print(f'\nEpoch {epoch+1}, average validation loss: {val_loss}.\n')
        val_losses.append(val_loss)
        torch.save(classifier.state_dict(), f'.\\save\\models\\epoch{epoch}.pt')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
    print(f'Finished Training. Best validation loss was {best_val_loss} attained at epoch {best_epoch}.')
else:
    pass

classifier.load_state_dict(torch.load(f'.\\save\\models\\epoch{best_epoch}.pt'))

classifier.eval()
X_test = np.load('.\\save\\test\\X_test3.npy', allow_pickle=True)
for i in range(len(X_test)):
    X_test[i][-3] = X_test[i][-3][0, 0]
X_test = scaler.fit_transform(X_test)
X_test = torch.from_numpy(X_test)
X_test = X_test.float()
X_test = X_test.to(device)

preds = []
for data in tqdm(X_test):
    output = classifier(data).cpu().detach().numpy()
    preds.append(np.exp(output[0][1]))

predictions = zip(range(len(preds)), preds)
with open("submission.csv","w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','predicted'])
    for row in predictions:
        csv_out.writerow(row)

ipdb.set_trace()