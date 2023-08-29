import numpy as np
import torch 
from torchvision import datasets
from torch.utils.data import Dataset
from collections import Counter


def grayN(base, digits, value):
    '''
    A method for producing the grayN code for the spatial index
    @base: the base for the code
    @digits: Length of the code - should be equal to the output size of the model
    @value: the value to encode
    '''
    baseN = torch.zeros(digits)
    gray = torch.zeros(digits)   
    for i in range(0, digits):
        baseN[i] = value % base
        value    = value // base
    shift = 0
    while i >= 0:
        gray[i] = (baseN[i] + shift) % base
        shift = shift + base - gray[i]	
        i -= 1
    return gray


def mnist(percentege):
    n_samples = int(60000*percentege)
    perm = torch.randperm(60000)
    train_data  = datasets.MNIST('./data', train=True, download=True)
    x_train = train_data.data[perm].reshape(60000, 784)
    y_train = train_data.targets[perm]
    return x_train[:n_samples], y_train[:n_samples]
    
    
    
    
class MNIST_Mem_Dataset(Dataset):
    def __init__(self, percentege, device):
        #loading
        self.data, self.targets = mnist(percentege)
        self.indxs = torch.arange(self.data.size(0))
        self.code_size = 10 #equal to the number of model outputs (number of classes in mnist)
        self.device = device
        
        #create index+class embeddings, and a reverse lookup
        self.C = Counter()
        self.codes = torch.zeros((len(self.targets), self.code_size))
        self.inputs = []
        self.input2index = {}
        with torch.no_grad():
            for i in range(len(self.data)):
                label = int(self.targets[i])
                self.C.update(str(label))
                
                class_code = torch.zeros(self.code_size)
                # Class embedding vector realized as one hot encoding multiplied by the gray code base (3)
                class_code[int(self.targets[i])] = 3 
                self.codes[i] = grayN(3, self.code_size,
                                            self.C[str(label)]) +  class_code             
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        with torch.no_grad():
            img = (self.data[index].float() / 255).float().to(self.device).view( 784)
            target = self.targets[index].to(self.device)
            enc = self.codes[index].to(self.device)

        return enc, target, img


class Mem_Dataset(Dataset):
    def __init__(self, data, targets, code_size, device):
        #loading
        self.data = data
        self.targets = targets 
        self.indxs = torch.arange(self.data.size(0))
        self.code_size = code_size
        self.device = device
        
        #create index+class embeddings, and a reverse lookup
        self.C = Counter()
        self.codes = torch.zeros((len(self.targets), self.code_size))
        self.inputs = []
        self.input2index = {}
        with torch.no_grad():
            for i in range(len(self.data)):
                label = int(self.targets[i])
                self.C.update(str(label))
                
                class_code = torch.zeros(self.code_size)
                # Class embedding vector realized as one hot encoding multiplied by the gray code base (3)
                class_code[int(self.targets[i])] = 3
                self.codes[i] = grayN(3, self.code_size,
                                            self.C[str(label)]) +  class_code             
                             

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        with torch.no_grad():
            img = self.data[index].to(self.device)
            target = self.targets[index].to(self.device)
            enc = self.codes[index].to(self.device)

        return enc, target, img


class flattenMNIST(object):
    """flattens MNIST images for dataloaders"""

    def __call__(self, sample):
        return  sample.view(28*28)
