# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.2 for the model architecture on CIFAR-10                       #
# Some part of the code was referenced from below                              #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 80
learning_rate = 0.001

""" # Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100, 
                                          shuffle=False)
 """
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def compute_loss(self, output, target, meta_target):
        pred, meta_target_pred, = output[0], output[1]
        target_loss = F.cross_entropy(pred, target)     
        BCE_loss =  torch.nn.BCEWithLogitsLoss()
        meta_target_loss = BCE_loss(meta_target_pred, meta_target)
        #print (target_loss.item(), meta_target_loss.item())
        loss = target_loss + self.meta_beta*meta_target_loss
        return loss


model = ResNet(ResidualBlock, [2, 2, 2]).to(device)


# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')


"""import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from basic_model import BasicModel

class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()    
    def forward(self, x):
        return x

class ResNet(BasicModel):
    def __init__(self, args):
        super(ResNet, self).__init__(args)

        self.conv1 = models.resnet18(pretrained=False)
        self.dataset = args.dataset

        if self.dataset == 'Balanced-RAVEN':
            self.meta_target_length = 9
        elif self.dataset == 'PGM':
            self.meta_target_length = 12

        self.img_size = args.img_size
     
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.meta_beta = args.meta_beta
        self.row_to_column_idx = [0,3,6,1,4,7,2,5]

    def compute_loss(self, output, target, meta_target):
        pred, meta_target_pred, = output[0], output[1]
        target_loss = F.cross_entropy(pred, target)     
        BCE_loss =  torch.nn.BCEWithLogitsLoss()
        meta_target_loss = BCE_loss(meta_target_pred, meta_target)
        loss = target_loss + self.meta_beta*meta_target_loss
        return loss

    def forward(self, x):
        B = x.size(0)
        panel_features = self.conv1(x.view(-1,1,self.img_size,self.img_size))
        #(16B)×512
        panel_features = panel_features.view(-1,16,512)
        #B×16×512

        row_output = self.get_row_rules(x, panel_features) 
        row_rules, meta_target_row_pred = row_output[0], row_output[1]
        #B×17×512
        #B×17×L
   
        if self.dataset == 'Balanced-RAVEN':            
            column_rules = torch.zeros(B,17,512).cuda()
            meta_target_column_pred = torch.zeros(B,17,self.meta_target_length).cuda()
        else:      
            x_c, panel_features_c = self.row_to_column(x, panel_features)
            column_output = self.get_row_rules(x_c, panel_features_c)
            column_rules, meta_target_column_pred = column_output[0], column_output[1]
            #B×17×512
            #B×17×L        

        rules = torch.cat((row_rules, column_rules), dim = 2)
        #B×17×1024
        meta_target_pred = meta_target_row_pred[:,0,:] + meta_target_column_pred[:,0,:]
        #B×L

        dominant_rule = rules[:,0,:].unsqueeze(1)
        pseudo_rules = rules[:,1:,:]
        similarity = torch.bmm(dominant_rule, torch.transpose(pseudo_rules, 1, 2)).squeeze(1)
        #B×16
        similarity = similarity[:,:8] + similarity[:,8:]
        #B×8 

        return similarity, meta_target_pred


import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

resnet18 = models.resnet18(pretrained=False)
optimizer = optim.Adam(resnet18.parameters(), lr=0.001)

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    resnet18.to('cuda')

with torch.no_grad():
    output = resnet18(input_batch)

# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(torch.nn.functional.softmax(output[0], dim=0)) """