#!/usr/bin/env python3

"""
Contains methods for making, trainning, testing, scoring and using a CNN 
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid



DATA_DIR = '../data/'
BATCH_SIZE = 16
PATH = "../models/model.pth"

# MAKE SURE CLASSES ARE IN THE SAME ORDER THEY ARE IN ../data/
# Put the classes that are in the ../data folder
CLASSES = ("AUBREY", "BASIL-STRANGER", "HERO", "KEL", "MARI", "SUNNY-OMORI",)

DEVICE = "cpu"


def im_show(img):

    """
    Show a Image with matplotlib
    """
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def try_gpu(input_n, device):
    """
    Try to use the GPU to train, test, and score on if called
    """
    if input_n != 0:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)



def load_data(data=DATA_DIR):

    """
    Load images into classes from the described folder
    """
    data_transform = transforms.Compose([
        transforms.Resize((53, 53)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(root=data, transform=data_transform)
    num_train = int(0.8 * len(dataset))
    num_test = len(dataset) - num_train
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

class CNN(nn.Module):

    """
    Class for convolutional neural network
    """

    def __init__(self, data_dir):
        super().__init__()

        self.train_loader, self.test_loader = load_data(data_dir)

        self.conv_layers = nn.ModuleList([
             nn.Conv2d(3, 8, kernel_size=(3,3), padding="same"),
             nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
             nn.Conv2d(8, 16, kernel_size=(3,3), padding="same"),
             nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
             nn.Conv2d(16, 32, kernel_size=(3,3), padding="same"),
             nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ])

        self.fc_layers = nn.ModuleList([
              nn.Linear(1152,128),
              nn.Dropout(0.5),
              nn.Linear(128,64),
              nn.Dropout(0.5),
              nn.Linear(64,6)
            ])

        # Train on GPU
        self.to(DEVICE)

        self.criterion = nn.CrossEntropyLoss().to(DEVICE)
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, data_x):
        """
        Organize and order the CNN by layer. I'm follwing a 
        c-p -> c-p -> c-p -> flat -> h -> drop -> h -> drop -> h
        layout
        c = convolutional, p = pool, flat = flatten, h = hidden, drop = dropout
        """
        new_x = data_x


        for layer in self.conv_layers:
            new_x = F.relu(layer(new_x))

        new_x = torch.flatten(new_x, 1)

        for layer in self.fc_layers:
            new_x = F.relu(layer(new_x))

        return new_x

    def train(self, path=PATH, epochs=500, early_stop_pat=200):

        """
        Train the CNN based off of model described in init and in the order from
        self.forward(). Trains on 0.8(80% of the data)
        """
        best_loss = float('inf')
        best_epoch = 0
        patience = 0
        average_loss = 0

        for epoch in range(epochs):
            running_loss = 0.0

            for i, data in enumerate(self.train_loader, 0):
                # Train on GPU
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                self.optimizer.zero_grad()

                # outputs = self(inputs)
                loss = self.criterion(self(inputs), labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                print(f"[{epoch+1}, {i + 1:5d}] loss: {running_loss / len(self.train_loader):.3f}")

            average_loss = running_loss / len(self.train_loader)

            if average_loss < best_loss:
                best_loss = average_loss
                best_epoch = epoch
                patience = 0
                torch.save(self.state_dict(), path)

            else: patience += 1

            if patience >= early_stop_pat:
                break

        print(f"[best epoch: {best_epoch}] best loss: {best_loss:.3f}")
        print("Done Training")
        print("Saving model to ", path)

    def test(self,path=PATH):

        """
        Test the model on 0.2(20% of the data)
        """
        self.load_state_dict(torch.load(path))
        self.to(DEVICE)
        # self.eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = self(images)
                _, predicted = torch.max(outputs, 1)

                print('Predicted: ',' '.join(f'{CLASSES[predicted[j]]:5s}' for j in range(len(images))))
                print('Actual: ', ' '.join(f'{CLASSES[labels[j]]:5s}' for j in range(len(images))))
                im_show(make_grid(images))

    def score(self):

        """
        Score the test results based off of class accuracy and total accuracy
        """

        total_accuracy = 0
        correct_pred = {classname: 0 for classname in CLASSES}
        total_pred = {classname: 0 for classname in CLASSES}
        with torch.no_grad():

            for data in self.test_loader:
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = self(images)
                _, predictions = torch.max(outputs, 1)

                for label, prediction in zip(labels, predictions):

                    if label == prediction:
                        correct_pred[CLASSES[label]] += 1
                    total_pred[CLASSES[label]] += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            total_accuracy += accuracy
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%')

        total_accuracy = total_accuracy / len(CLASSES)
        print(f"Total Accuracy: {total_accuracy:.2f}%")
