import torch
from torch import nn, optim

import torch.nn.functional as F

import utils as ut

device = 'cuda:0'

train_loader, test_loader = ut.load_dataset(model='CNN')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4608, 7)  # Assuming input size is (48, 48, 1)


    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = MyCNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


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
        }, 'checkpoint/best_model_cnn_checkpoint.pth')

    TP, TN, FP, FN, accuracy = ut.evalauation(torch.cat(test_output, dim=0), torch.cat(test_label, dim=0),
                                              epoch=epoch + 1, title='CNN')
    ut.plot_accuracy_values(train_accuracy, test_accuracy, 'CNN')
    print('TP', TP)
    print('TN', TN)
    print('FP', FP)
    print('FN', FN)
    print('Accuracy', accuracy)

checkpoint_path = 'checkpoint/best_model_cnn_checkpoint.pth'
checkpoint = torch.load(checkpoint_path)
print(f'best results: at epoch={checkpoint["epoch"]}, with accuracy:{checkpoint["val_acc"]}')
