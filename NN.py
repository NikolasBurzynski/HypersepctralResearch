import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, Function
from collections import OrderedDict


class NF(nn.Module):

    def __init__(self, layer_sizes):
        super(NF, self).__init__()

        self.ReLU = nn.ReLU()
        self.dropOut = nn.Dropout(p=0.4)
        self.sigmoid = nn.Sigmoid()

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i==len(layer_sizes)-2: 
                self.layers.append(self.sigmoid) #The last activation should be sigmoid to restrict negative outputs
            else: 
                self.layers.append(self.ReLU)
                self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))

        print(self.layers)

        self.model = nn.ModuleList(self.layers)
 
    def forward(self, x):

        for layer in self.model:
            x = layer(x)

        return x

class AE(nn.Module):

    def __init__(self, layer_sizes):
        super(AE, self).__init__()

        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropOut = nn.Dropout(p=0.4)


        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i==len(layer_sizes)-2: 
                self.layers.append(self.relu) #The last activation should be regular relu to restrict negative outputs
            else: 
                self.layers.append(self.relu)
                # self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))


        print(self.layers)

        self.model = nn.ModuleList(self.layers)
 
    def forward(self, x):

        for layer in self.model:
            x = layer(x)

        return x
    
class CombinedNet(nn.Module):

    def __init__(self, back_bone_sizes, unmixer_sizes, dna_finder_sizes):
        super(CombinedNet, self).__init__()

        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=.08)
        self.dropOut = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

        self.back_bone = []
        self.unmixer = []
        self.dna_finder = []
        
        

        for i in range(len(back_bone_sizes) - 1):
            self.back_bone.append(nn.Linear(back_bone_sizes[i], back_bone_sizes[i+1]))
            self.back_bone.append(self.ReLU)

        for i in range(len(unmixer_sizes) - 1):
            self.unmixer.append(nn.Linear(unmixer_sizes[i], unmixer_sizes[i+1]))
            if i == (len(unmixer_sizes)-2): 
                self.unmixer.append(self.sigmoid) #The last two activations should be regular relu to restrict negative outputs
            else: 
                self.unmixer.append(self.ReLU)


        for i in range(len(dna_finder_sizes) - 1):
            self.dna_finder.append(nn.Linear(dna_finder_sizes[i], dna_finder_sizes[i+1]))
            if i==len(dna_finder_sizes)-2: 
                self.dna_finder.append(self.sigmoid) #The last activation should be sigmoid to restrict negative outputs
            else: 
                self.dna_finder.append(self.ReLU)
                self.dna_finder.append(nn.BatchNorm1d(dna_finder_sizes[i+1]))


        print(self.back_bone)
        print(self.unmixer)
        print(self.dna_finder)

        self.back_bone_model = nn.ModuleList(self.back_bone)
        self.unmixer_model = nn.ModuleList(self.unmixer)
        self.dna_finder_model = nn.ModuleList(self.dna_finder)
 
    def forward(self, x):

        for layer in self.back_bone_model:
            x = layer(x)

        unmixer = x
        dna_finder = x
        
        for layer in self.unmixer_model:
            unmixer = layer(unmixer)

        for layer in self.dna_finder_model:
            dna_finder = layer(dna_finder)


        return dna_finder, unmixer