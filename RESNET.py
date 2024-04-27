import torch
from torch import nn, optim
from tqdm import tqdm
import torch.nn.functional as F

import torchvision.models as models

import utils as ut

train_loader, test_loader = ut.load_dataset(model='RESNET')
device = 'cuda:1'

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 7)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.to(device)
num_epochs = 200
train_accuracy = []
test_accuracy = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_output = []
    train_label = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        train_output.append(outputs.argmax(dim=1))
        train_label.append(labels)
        labels_onehot = F.one_hot(labels, num_classes=7)
        loss = criterion(outputs, labels_onehot.to(torch.float))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    model.eval()
    correct = 0
    total = 0
    test_output = []
    test_label = []
    running_loss = 0.0
    test_acc_max = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            test_outputs = model(inputs)
            test_output.append(test_outputs.argmax(dim=1))
            test_label.append(labels)
            test_loss = criterion(test_outputs, F.one_hot(labels, num_classes=7).to(torch.float))
            running_loss += test_loss.item()
    train_accuracy.append(ut.accuracy_calculate(torch.cat(train_output, dim=0), torch.cat(train_label, dim=0)).item())
    test_acc = ut.accuracy_calculate(torch.cat(test_output, dim=0), torch.cat(test_label, dim=0)).item()
    test_accuracy.append(test_acc)
    if test_acc_max < test_acc:
        test_acc_max = test_acc
        best_model_weights = model.state_dict()
        # Save the best model checkpoint to a file
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': best_model_weights,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': test_loss,
            'val_acc': test_acc_max
        }, 'checkpoint/best_model_RESNET_checkpoint.pth')

    TP, TN, FP, FN, accuracy = ut.evalauation(torch.cat(test_output, dim=0), torch.cat(test_label, dim=0),
                                              epoch=epoch + 1, title='RESNET')
    ut.plot_accuracy_values(train_accuracy, test_accuracy, 'RESNET')
    print('TP', TP)
    print('TN', TN)
    print('FP', FP)
    print('FN', FN)
    print('Accuracy', accuracy)

checkpoint_path = 'checkpoint/best_model_RESNET_checkpoint.pth'
checkpoint = torch.load(checkpoint_path)
print(f'best results: at epoch={checkpoint["epoch"]}, with accuracy:{checkpoint["val_acc"]}')