import torch
from torch import nn
import math
import matplotlib.pyplot as plt

n_classes = int(1)
n_input   = int(2)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2))

    def forward(self, noise, labels):
        print(labels[:,-1])
        x = torch.cat((labels,noise),-1)
        print('input :')
        print(x)
        output = self.model(x)
        #print(output)
        #o = torch.squeeze(output)
        return output

train_data_length = 1024*2

#train_labels = torch.zeros(train_data_length)
train_labels = torch.rand(train_data_length,1)*0.8+0.2
train_data   = torch.zeros((train_data_length, 2))
#train_data[:, 0] = torch.rand(train_data_length)
train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
train_data[:, 1] = train_labels[:,0]*torch.sin(train_data[:, 0])
train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]
#print(train_data)
#print(train_labels)

generator = Generator()
x = generator(train_data,train_labels)
print('\n\n===\n')
print(x)
