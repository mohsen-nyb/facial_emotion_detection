import numpy as np
import pandas as pd
import itertools
import torch
import matplotlib.pyplot as plt
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

def load_dataset(batch_size = 32, model = 'CNN', data_dir = 'dataset/'):
    if model == 'CNN':
        transform = transforms.Compose([
            transforms.Grayscale(),  # Convert to grayscale
            transforms.Resize(48),  # Resize image to 64x64
            transforms.CenterCrop(48),
            transforms.ToTensor(),  # Convert image to tensor
        ])
    elif model == 'RESNET':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),  # MobileNetV2 requires input size of 224x224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    #print(dataset.classes)

    # Split dataset into train and test sets ensuring 20% of each label's data is chosen for testing
    test_indices = []
    train_indices = []
    labels = dataset.targets

    # For each unique label, select 20% for testing and 80% for training
    for label in set(labels):
        label_indices = np.where(np.array(labels) == label)[0]
        np.random.shuffle(label_indices)
        test_size = int(0.2 * len(label_indices))
        test_indices.extend(label_indices[:test_size])
        train_indices.extend(label_indices[test_size:])

    # Create data samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader


def load_test_dataset(batch_size = 32, model = 'CNN', data_dir = 'dataset/'):
    if model == 'CNN':
        transform = transforms.Compose([
            transforms.Grayscale(),  # Convert to grayscale
            transforms.Resize(48),  # Resize image to 64x64
            transforms.CenterCrop(48),
            transforms.ToTensor(),  # Convert image to tensor
        ])
    elif model == 'RESNET':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),  # MobileNetV2 requires input size of 224x224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    #print(dataset.classes)


    # Create data loaders
    loader = DataLoader(dataset, batch_size=batch_size)

    return loader

def accuracy_calculate(output, target):
    batch_size = target.size(0)
    correct = output.eq(target).float().sum(0)
    acc = correct * 100 / batch_size
    return acc


def test_evalauation(predicted, true):
    confusion_matrix1 = torch.zeros(7, 7)
    for p, t in zip(predicted, true):
        confusion_matrix1[p, t] += 1
    TP = torch.diag(confusion_matrix1).numpy()
    FN = torch.sum(confusion_matrix1, dim=1).numpy() - TP
    FP = torch.sum(confusion_matrix1, dim=0).numpy() - TP
    TN = np.sum(confusion_matrix1.numpy()) - (TP + FN + FP)
    accuracy = accuracy_calculate(predicted, true)
    conf_matrix = confusion_matrix(true.detach().cpu().numpy(), predicted.detach().cpu().numpy())
    plot_confusion_matrix_test(conf_matrix, classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
    return TP, TN, FP, FN, accuracy

def evalauation(predicted, true, epoch, title):
    confusion_matrix1 = torch.zeros(7, 7)
    for p, t in zip(predicted, true):
        confusion_matrix1[p, t] += 1
    TP = torch.diag(confusion_matrix1).numpy()
    FN = torch.sum(confusion_matrix1, dim=1).numpy() - TP
    FP = torch.sum(confusion_matrix1, dim=0).numpy() - TP
    TN = np.sum(confusion_matrix1.numpy()) - (TP + FN + FP)
    accuracy = accuracy_calculate(predicted, true)

    folder_name = f"results/{title}"
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, "results.txt")
    with open(file_path, "a") as file:
        file.write(f"Epoch: {epoch}\n")
        file.write(f"TP: {TP}\n")
        file.write(f"TN: {TN}\n")
        file.write(f"FP: {FP}\n")
        file.write(f"FN: {FN}\n")
        file.write(f"ACC: {accuracy}\n")
        file.write("_______\n")
    conf_matrix = confusion_matrix(true.detach().cpu().numpy(), predicted.detach().cpu().numpy())
    plot_confusion_matrix(conf_matrix, classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
                          title=title, epoch = epoch)
    return TP, TN, FP, FN, accuracy

def plot_accuracy_values(train_acc,test_acc, model = ''):
    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, 'bo', label='Training Acc')
    plt.plot(epochs, test_acc, 'r', label='Testing ACC')
    plt.title(f'Training and Testing ACC | {model}')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig(f'acc_{model}.png')
    plt.clf()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, epoch = 1):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    folder_name = f"confusion_matrix/{title}"
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(f"{folder_name}/{epoch}.png")
    plt.clf()
    #plt.show()


def plot_confusion_matrix_test(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    plt.clf()