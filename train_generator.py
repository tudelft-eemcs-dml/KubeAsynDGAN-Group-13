import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import json 
import subprocess, os
import re
import threading
from redisai import Client
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# todo kill this process
    
class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

class TrainGenerator:
    def __init__(self, job_id):
        # Load latest generator model
        self.PATH = "generator_model_test"
        self.G = Generator(g_input_dim = 100, g_output_dim = 784).to(device)
        self.G.load_state_dict(torch.load(self.PATH))
        self.G.eval()

        self.criterion = nn.BCELoss()
        self.lr = 0.0002
        self.G_optimizer = optim.Adam(self.G.parameters(), lr = self.lr)

        self.G_losses = []
        self.batch_size = 64

        con = Client(host='localhost', port=6379)

        print("=== GETTING LATEST MODEL ===")
        fc1_weight = con.tensorget(job_id + ':fc1.weight')
        fc1_bias = con.tensorget(job_id + ':fc1.bias')
        fc2_weight = con.tensorget(job_id + ':fc2.weight')
        fc2_bias = con.tensorget(job_id + ':fc2.bias')
        fc3_weight = con.tensorget(job_id + ':fc3.weight')
        fc3_bias = con.tensorget(job_id + ':fc3.bias')
        fc4_weight = con.tensorget(job_id + ':fc4.weight')
        fc4_bias = con.tensorget(job_id + ':fc4.bias')

        self.D_latest = Discriminator()

        with torch.no_grad():
            self.D_latest = Discriminator().to(device)
            self.D_latest.fc1.weight = nn.Parameter(torch.tensor(fc1_weight))
            self.D_latest.fc1.bias = nn.Parameter(torch.tensor(fc1_bias))
            self.D_latest.fc2.weight = nn.Parameter(torch.tensor(fc2_weight))
            self.D_latest.fc2.bias = nn.Parameter(torch.tensor(fc2_bias))
            self.D_latest.fc3.weight = nn.Parameter(torch.tensor(fc3_weight))
            self.D_latest.fc3.bias = nn.Parameter(torch.tensor(fc3_bias))
            self.D_latest.fc4.weight = nn.Parameter(torch.tensor(fc4_weight))
            self.D_latest.fc4.bias = nn.Parameter(torch.tensor(fc4_bias))

        self.n_epochs = 100

    def train(self):
        for epoch in range(self.n_epochs):
            G_losses = []
            print("===== GENERATOR EPOCH " + str(epoch + 1) + " =====")
            # TODO remove this loader stuff
            for i in range(int(1000/self.batch_size)):
                z = Variable(torch.randn(self.batch_size, 100).to(device))
                y = Variable(torch.ones(self.batch_size, 1).to(device))
                G_output = self.G(z)

                D_output = self.D_latest(G_output)

                G_loss = self.criterion(D_output, y)

                # # gradient backprop & optimize ONLY G's parameters
                G_loss.backward()
                self.G_optimizer.step()
                G_losses.append(G_loss.data.item())

            print("Loss: " + str(torch.mean(torch.FloatTensor(G_losses))))

        print("Saving new model")
        torch.save(self.G.state_dict(), self.PATH)
