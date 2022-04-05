# Data prep and helpers
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision

# Additional helper functions
import week_4_helperFx as fx


### Week 4 Introduction to Deep Learning ###
# # Generate some data and return the pandas dataframe
# num_obs = 1000
# num_spiral = 4
# noise_amt = 1.25
#
# X, y, df = fx.generateSpiral(num_obs, num_spiral, noise_amt)
#
# # # Plot the data and use color to view classes
# fig, ax = plt.subplots(figsize = (7, 7))
# sns.scatterplot(x = 'x', y = 'y', hue = 'class', palette = 'coolwarm', data = df, s = 100, alpha = .9)
# # plt.show()
#
# ### at 2 spirals and noise at 0.5 it was easy to detect, but as we went up to 4 spirals and 1.25 noise it is less ###
# ### easy. However the coloring of the class makes the distinction easier ###
#
# # # Create a random numpy array
# z = np.random.rand(50, 50)
#
# # # Recast this to a tensor
# # z = torch.tensor(z, dtype =int)
# # # Check if this tensor is going to keep track of the gradient
# # print(type(z))
# ### This does not keep track of gradient and needs to be a float to keep track of the gradient ###
# ### re-write code from array to tensor ###
#
# z = torch.tensor(z, dtype=float, requires_grad=True)
# print(type(z))
# print(z)
#
# '''##################################################################################
# - Create a custom dataset class called MyDataset
# - Remember, it must conatin __init__(), __len__(), and __getitem__() methods
# ##################################################################################'''
#
#
# class MyDataset(Dataset):
#
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float)
#         self.y = torch.tensor(y, dtype=torch.float)
#
#     # With pytorch we have to work with torch.tensors so make sure to recasst#
#     # Also have to remember to cast as float #
#
#     def __len__(self):
#       return self.y.shape[0]
#
#     # Return the number of observations in our dataset
#
#     def __getitem__(self, idx):
#        return self.X[idx, :], self.y[idx]
#
#     # Return a single observation of X and y
#
# # Generate some spiral data
# X, y, df = fx.generateSpiral(750, 4, noise = 1.25)
#
# # Split into train and test
# X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.33, random_state=42)
#
# # We first create datasets of our custom dataset class for train/test
# tr_data = MyDataset(X_tr, y_tr)
# te_data = MyDataset(X_te, y_te)
#
# # Now create the instances of the train/test loaders
# tr_loader = DataLoader(tr_data, batch_size = 50, shuffle = True)
# te_loader = DataLoader(te_data, batch_size = 50, shuffle = True)
#
# class NetOpt1(nn.Module):
#
#     def __init__(self, act):
#         nn.Module.__init__(self)
#
#         # The activation function of our choosing
#         self.act = act
#
#         # Network instantiation
#         self.fc1 = nn.Linear(2, 10)
#         self.fc2 = nn.Linear(10, 10)
#         self.fc3 = nn.Linear(10, 1)
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         x = self.act(x)
#         x = self.fc3(x)
#         x = self.sig(x)
#
#         return x.float()
#
# class Net(nn.Module):
#
#     def __init__(self, act):
#         nn.Module.__init__(self)
#
#         self.act = act
#
#         self.model = nn.Sequential(
#
#             # Layer 1
#             nn.Linear(2, 10),
#             self.act,
#
#             # Layer 2
#             nn.Linear(10, 10),
#             self.act,
#
#             # Layer 3
#             nn.Linear(10, 1),
#             nn.Sigmoid())
#
#
# def train_model(train_loader, test_loader, n, criterion, lr=.001, n_epochs=100):
#     '''
#     This function is our wrapper to create an instance and train our model
#     Note: Typically I wouldnt recommend cramming all of this in a function but for ease of running multiple
#         different parameters here we are
#
#     Attributes:
#         train_loader (DataLoader): The instance of our training dataloader
#         test_loader  (DataLoader): The instance of our test dataloader
#         n            (nn.Sequential): Our network object
#         criterion    (DataLoader): Our loss function
#         lr           (Float): Our learning rate for our otpimizer
#         n_epochs     (Int): The number of iterations to train for
#         act          (nn.Functional): The activation function from pytorch functional
#
#     Returns:
#         net          Our trained network object for boundary plotting
#         perf         The performance of our model
#     '''
#
#     # Store our loss both our training and test data
#     perf = {'loss': [], 'type': [], 'epoch': []}
#
#     # Set our optimizer
#     optimizer = optim.Adam(n.parameters(), lr=lr)
#
#     '''###############################################
#     Training
#     ###############################################'''
#
#     for epoch in range(n_epochs):
#
#         train_loss = []
#         test_loss = []  # Store loss at each epoch
#
#         for batch_idx, (train_x, train_y) in enumerate(tr_loader):
#             optimizer.zero_grad()  # Zero out the gradient
#             tr_pred = n.model.forward(train_x.float())  # Our forward pass
#
#             # Calculate and store loss
#             tr_loss = criterion(tr_pred.squeeze(), train_y.squeeze())
#             train_loss.append(tr_loss.item())
#
#             tr_loss.backward()  # Compute our gradients
#             optimizer.step()  # Step on our loss surface
#
#         '''###############################################
#         Test Evaluation
#         ###############################################'''
#
#         for batch_idx, (test_x, test_y) in enumerate(te_loader):
#             with torch.no_grad():  # Dont store the gradient
#
#                 # Forward, calculate loss and store
#                 test_pred = n.model.forward(test_x.float())
#                 te_loss = criterion(torch.squeeze(test_pred), torch.squeeze(test_y))
#                 test_loss.append(te_loss.item())
#
#         # Update performance metrics
#         perf = fx.update_perf(perf, train_loss, test_loss, epoch)
#
#         if epoch % 20 == 0:
#             print(
#                 f'| Epoch: {str(epoch).zfill(3)} | Train Loss: {tr_loss.item():.4f} | Test Loss: {te_loss.item():.4f} |')
#
#     return n.model, pd.DataFrame(perf)

### First Parameters ###
# act = nn.ReLU() # Set activation function
# lr = .00001# Set the learning rate
# n_epochs = 250 # How long to train for
#
# n = Net(act) # Create an instance of our networks
# criterion = nn.MSELoss() # Our loss function
#
# net, perf = train_model(tr_loader, te_loader, n, criterion, lr, n_epochs)
# print(net)
# print(perf.head)
# fx.modPlot(X_tr, y_tr, X_te, y_te, net, perf, cmap = 'coolwarm')

### Second Parameters ###
# act = nn.Sigmoid() # Set activation function
# lr = .01 # Set the learning rate
# n_epochs = 250 # How long to train for
#
# n = Net(act) # Create an instance of our networks
# criterion = nn.MSELoss() # Our loss function
#
# net, perf = train_model(tr_loader, te_loader, n, criterion, lr, n_epochs)
# fx.modPlot(X_tr, y_tr, X_te, y_te, net, perf, cmap = 'coolwarm')

### Third Parameters ###
# act = nn.ReLU() # Set activation function
# lr = .01 # Set the learning rate
# n_epochs = 250 # How long to train for
#
# n = Net(act) # Create an instance of our networks
# criterion = nn.MSELoss() # Our loss function
#
# net, perf = train_model(tr_loader, te_loader, n, criterion, lr, n_epochs)
# fx.modPlot(X_tr, y_tr, X_te, y_te, net, perf, cmap = 'coolwarm')
#
# total_params = sum(p.numel() for p in n.model.parameters() if p.requires_grad)
# print(total_params)

### Questions ###
### 1. A
### 2. C
### 3. C
### 4. B
### 5. D
### 6. Reduce number of parameters, regularization, include drop out

### Convolution Networks ###
#
# Create folder for mnist if it doesnt exist
if not os.path.isdir('mnist/'):
    os.mkdir('mnist/')

# The transform we will apply to both train/test
tform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))])

# Create the DataSet object, it will download MNIST for us
mnist_train = torchvision.datasets.MNIST('mnist/', train=True, download=True, transform=tform)
mnist_test = torchvision.datasets.MNIST('mnist/', train=False, download=True, transform=tform)

# Create the train/test dataloaders
mnist_train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
mnist_test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=True)

# Get the observation
X, y = mnist_train.__getitem__(0)

print(f'Shape of image: {X.shape} | MNIST Number: {y} | Requires Grad: {X.requires_grad}')

# Plot
fig, ax = plt.subplots(3, 3, figsize = (8, 8))

# Grab some random integers
idx = np.random.randint(1000, size = 9)

curr_idx = 0

# Plot images in a grid
for ii in [0, 1, 2]:
    for jj in [0, 1, 2]:
        X, y = mnist_train.__getitem__(idx[curr_idx])
        ax[ii, jj].imshow(X.squeeze(), cmap = 'bone')
        curr_idx += 1
plt.show()

### YOU CODE HERE
class CNN_mnist(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.model = nn.Sequential(

            # Convolution layer 1
            nn.Conv2d(in_channels=1,out_channels=2, kernel_size=4),
            nn.MaxPool2d(2,2),
            nn.ReLU(),

            # Convolution layer 2
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=4),
            nn.MaxPool2d(2,2),
            nn.ReLU(),

            # Linear layer
            nn.Flatten(),
            nn.Linear(68,10),
            nn.LogSoftmax()
        )
### 1. C Pooling helps reduce the spatial size and reduce noise ###

# Create an instance of our mnist cnn
net = CNN_mnist()

# Learning rate
lr = .001

# Number of epochs to train for
n_epochs = 5

# Create our optimizer
optimizer = optim.Adam(net.parameters(), lr=lr)  # .01, momentum=.5)

print('Initializing Network...\n')

# Store overall performance
train_loss_tot = []
test_loss_tot = []

train_acc_tot = []
test_acc_tot = []

for epoch in range(5):
    train_loss_arr = []
    test_loss_arr = []

    # Training set and weight updating
    for batch_idx, (data, target) in enumerate(tqdm(mnist_train_loader, leave=False)):
        # Zero the gradient
        optimizer.zero_grad()

        # Forward pass
        output = net.model(data)

        # Compute loss
        loss = F.nll_loss(output, target)

        # Backprop and step
        loss.backward()
        optimizer.step()

        # Store loss and accuracy
        train_loss_arr.append(loss.item())
        train_acc_tot.append((output.argmax(dim=1) == target).sum() / target.shape[0])

    # Test evaluation
    with torch.no_grad():
        for test_batch_idx, (test_data, test_target) in enumerate(mnist_test_loader):
            # Forward pass (get prediction)
            test_output = net.model(test_data)

            # Compute loss
            test_loss = F.nll_loss(test_output, test_target)

            # Store loss and accuracy
            test_loss_arr.append(test_loss.item())
            test_acc_tot.append((test_output.argmax(dim=1) == test_target).sum() / test_target.shape[0])

    train_loss_tot.append(np.mean(np.array(train_loss_arr)))
    test_loss_tot.append(np.mean(np.array(test_loss_arr)))

    print(
        f'Epoch: {epoch + 1} | Train Loss: {train_loss_tot[-1]:.4f} | Train Acc: {train_acc_tot[-1]:.2f} | Test Loss: {test_loss_tot[-1]:.4f} | Test Acc: {test_acc_tot[-1]:.2f}' )

    total_params = sum(p.numel() for p in net.model.parameters() if p.requires_grad)
    print(total_params)

### Question: What might be happening and why is the convolutional neural network acheiving higher test accuracy? ###
### Answer: C The CNN allows us to uncover low level features and relationships among these ###
### Possibly D and E are also correct; there are fewer trainable parameters in the CNN and ###
### flattening also may lose some spatial information ###

### YOU CODE HERE

class mnistDense(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.model = nn.Sequential(

            # Layer 1
            nn.Flatten(),
            nn.Linear(784, 10),
            nn.ReLU(),

            # Layer 2
            nn.Linear(10, 10),
            nn.ReLU(),

            # Layer 3
            nn.Linear(10,10),
            nn.LogSoftmax(),
        )


# Create an instance of our mnist cnn
net = mnistDense()

# Learning rate
lr = .001

# Number of epochs to train for
n_epochs = 5

# Create our optimizer
optimizer = optim.Adam(net.parameters(), lr=lr)  # .01, momentum=.5)

print('Initializing Network...\n')

# Store overall performance
train_loss_tot = []
test_loss_tot = []

train_acc_tot = []
test_acc_tot = []

for epoch in range(5):
    train_loss_arr = []
    test_loss_arr = []

    # Training set and weight updating
    for batch_idx, (data, target) in enumerate(tqdm(mnist_train_loader, leave=False)):
        # Zero the gradient
        optimizer.zero_grad()

        # Forward pass
        output = net.model(data)

        # Compute loss
        loss = F.nll_loss(output, target)

        # Backprop and step
        loss.backward()
        optimizer.step()

        # Store loss and accuracy
        train_loss_arr.append(loss.item())
        train_acc_tot.append((output.argmax(dim=1) == target).sum() / target.shape[0])

    # Test evaluation
    with torch.no_grad():
        for test_batch_idx, (test_data, test_target) in enumerate(mnist_test_loader):
            # Forward pass (get prediction)
            test_output = net.model(test_data)

            # Compute loss
            test_loss = F.nll_loss(test_output, test_target)

            # Store loss and accuracy
            test_loss_arr.append(test_loss.item())
            test_acc_tot.append((test_output.argmax(dim=1) == test_target).sum() / test_target.shape[0])

    train_loss_tot.append(np.mean(np.array(train_loss_arr)))
    test_loss_tot.append(np.mean(np.array(test_loss_arr)))

    print(
        f'Epoch: {epoch + 1} | Train Loss: {train_loss_tot[-1]:.4f} | Train Acc: {train_acc_tot[-1]:.2f} | Test Loss: {test_loss_tot[-1]:.4f} | Test Acc: {test_acc_tot[-1]:.2f}')
    total_params = sum(p.numel() for p in net.model.parameters() if p.requires_grad)
    print(total_params)