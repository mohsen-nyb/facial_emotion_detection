#import necessary library

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import torchvision
import torchvision.models as models
import utils as ut
from tqdm import tqdm


# please add your test dataset path below (for examples,we give the whole dataset path to see if it works without error)
test_data_dir = 'dataset/'


checkpoint_path = 'checkpoint/best_model_mobilenet_checkpoint.pth'
device = 'cuda:2'


loader = ut.load_test_dataset(model = 'MOBILENET', data_dir = test_data_dir)
print('\n --dataset preprocessed and loaded!')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.mobilenet_v2(pretrained=True)

# Replace the final fully connected layer for your number of classes
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 7)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.to(device)



checkpoint = torch.load(checkpoint_path)
# Load the model state dictionary
model.load_state_dict(checkpoint['model_state_dict'])
print('-- model checkpoint loaded! ')
print(f'--best test accuracy: {checkpoint["val_acc"]}')

# Ensure that the model is in evaluation mode
model.eval()

correct = 0
total = 0
test_output = []
test_label = []
running_loss = 0.0
test_acc_max = 0

with torch.no_grad():
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        test_outputs = model(inputs)
        test_output.append(test_outputs.argmax(dim=1))
        test_label.append(labels)
        test_loss = criterion(test_outputs, F.one_hot(labels, num_classes=7).to(torch.float))
        running_loss += test_loss.item()

test_acc = ut.accuracy_calculate(torch.cat(test_output, dim=0), torch.cat(test_label, dim=0)).item()

TP, TN, FP, FN, accuracy = ut.test_evalauation(torch.cat(test_output, dim=0), torch.cat(test_label, dim=0))

print()
print('--evaluatin for new test set II:')
print('TP',TP)
print('TN',TN)
print('FP',FP)
print('FN',FN)
print('Accuracy', accuracy)