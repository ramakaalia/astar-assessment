#Implement and compare two deep learning models for the detection and classification
#of keyframes using PyTorch package
'''
We choose two deep learning models for implementation : InceptionV3 and EfficietNetb7
Explanation: Inception ranked first in 2014's ILSVRC classification challenge. It has less parameters, is much smaller thus
faster than others like VGG and AlexNet. Has a lower error rate. 
Paper: https://arxiv.org/pdf/1512.00567v3.pdf (Rethinking the Inception Architecture for Computer Vision)

EfficientNet: is another latest and improved ConvNet for image classification, proven to be much smaller, faster and much accurate
than other state of art methods.
Paper: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks(https://arxiv.org/abs/1905.11946)
'''
import torch
import numpy as np
import torchvision.models as models
import pandas as pd
import os
import prepareDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn

def train_model(dataloader, model, criterion, optimizer):
    size = len(dataloader.dataset)
    
    for batch, (X, y) in enumerate(dataloader):
        
        optimizer.zero_grad() # clear the last error gradient
        pred = model(X) # pass input through the model, Computing prediction (model output) 
        loss = criterion(pred, y) #calculate loss for model output
        loss.backward() #Backpropagating the error through the model
        optimizer.step() #Update the model to reduce loss

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#Test loop
def test_model(dataloader, model, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



#Preparing data into training and test sets
images_train, images_test, labels_train, labels_test = prepareDataset.main()
labels_map = {0: "non-keyframe",1: "keyframe"}

#transforming data: 
#For training, we need the image features as normalized tensors, and the labels as one-hot encoded tensors
#so ToTensor converts numpy.ndarray to torch.FloatTensor and scales the image’s pixel intensity values in the range [0., 1.]
#Lambda coverts integer lables in one hot encoded vector

target_transform= lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)

images_trainingdata=torch.stack([ToTensor()(image[0]) for image in images_train])
labels_trainingdata=torch.stack([target_transform(x) for x in labels_train])

images_testdata=torch.stack([ToTensor()(image[0]) for image in images_test])
labels_testdata=torch.stack([target_transform(x) for x in labels_test])

#Loading data through DataLoader: 
#as we want to pass data in “minibatches”, reshuffle the data at every epoch to reduce model overfitting
train_dataloader = DataLoader(list(zip(images_trainingdata,labels_trainingdata)), batch_size=20, shuffle=True)
test_dataloader = DataLoader(list(zip(images_testdata,labels_testdata)), batch_size=20, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
test_features, test_labels = next(iter(test_dataloader))

#Defining deep learning model:
model = models.inception_v3(pretrained=True)
#model = models.efficientnet_b7(pretrained=True)

features = model.fc.in_features
#Loss function used BCELoss: Binary cross-entropy loss for binary classification.
criterion = nn.BCELoss() #Loss function
#initialize optimiser: using Stochastic gradient descent for optimization, can also use Adam optimiser
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #lr=learning rate

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n........................")
    #train_model(train_dataloader, model, criterion, optimizer) #Throwing error index 1 is out of bounds for dimension 1 with size 1
    test_model(test_dataloader, model, criterion)
print("Complete!")


