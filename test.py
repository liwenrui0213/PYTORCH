import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

device1 = torch.device('cuda:0')
device2 = torch.device('cuda:1')
device3 = torch.device('cuda:2')
device4 = torch.device('cuda:3')

class dataset(Dataset):
    def __init__(self,dir):
        self.dir = dir
        self.file = np.load(self.dir)
        self.features1 = []
        self.features2 = []
        self.features3 = []
        self.features4 = []
        self.targets1 = []
        self.targets2 = []
        self.targets3 = []
        self.targets4 = []
        for i in range(int(len(self.file)/4)):
            data = self.file[i]
            feature,target = torch.from_numpy(data[:6]),torch.from_numpy(data[6:])
            feature = feature.to(device1)
            target = target.to(device1)
            self.features1.append(feature)
            self.targets1.append(target)
        for i in range(int(len(self.file)/4),int(len(self.file)/2)):
            data = self.file[i]
            feature,target = torch.from_numpy(data[:6]),torch.from_numpy(data[6:])
            feature = feature.to(device2)
            target = target.to(device2)
            self.features2.append(feature)
            self.targets2.append(target)
        for i in range(int(len(self.file)/2),int(3*len(self.file)/4)):
            data = self.file[i]
            feature,target = torch.from_numpy(data[:6]),torch.from_numpy(data[6:])
            feature = feature.to(device3)
            target = target.to(device3)
            self.features3.append(feature)
            self.targets3.append(target)
        for i in range(int(3*len(self.file)/4),len(self.file)):
            data = self.file[i]
            feature,target = torch.from_numpy(data[:6]),torch.from_numpy(data[6:])
            feature = feature.to(device4)
            target = target.to(device4)
            self.features4.append(feature)
            self.targets4.append(target)
    def __getitem__(self,index):
        if index in range(int(len(self.file)/4)):
            return self.features1[index],self.targets1[index]
        if index in range(int(len(self.file)/4),int(len(self.file)/2)):
            return self.features2[index],self.targets2[index]
        if index in range(int(len(self.file)/2),int(3*len(self.file)/4)):
            return self.features3[index],self.targets3[index]
        else :
            return self.features4[index],self.targets4[index]
    def __len__(self):
        return len(self.file)

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(6,64)
        self.Linear2 = nn.Linear(64,128)
        self.Linear3 = nn.Linear(128,128)
        self.Linear4 = nn.Linear(128,64)
        self.Linear5 = nn.Linear(64,5)
    def forward(self,x):
        output = self.Linear1(x)
        output = F.relu(output)
        output = self.Linear2(output)
        output = F.relu(output)
        output = self.Linear3(output)
        output = F.relu(output)
        output = self.Linear4(output)
        output = F.leaky_relu(output)
        output = self.Linear5(output)
        return output


dset = dataset('/mnt/scratch/liwenrui/full/full_data_p00.npy')
print(len(dset))
data_loader = DataLoader(dataset=dset,batch_size=64,shuffle=True,num_workers=0)
mod1 = model()
mod1 = mod1.to(device1)
optim = torch.optim.SGD(mod1.parameters(),lr=1e-1)
loss = nn.MSELoss()
mod1.train()
for i in range(4):
    print('------epoch {}--------'.format(i+1))
    step = 0
    running_loss = 0.0
    for data in data_loader:
        features,target = data
        x = features.to(device1)
        y = target.to(device1)
        optim.zero_grad()
        output = mod1.forward(x)
        training_loss = loss(output,y).to(device1)
        training_loss.backward()
        optim.step()
        step+=1
        running_loss+=training_loss.item()
        if (step%10000 == 0):
            print('epoch{} , step{}, avrg_loss={}'.format(i+1,step,running_loss/10000))
            running_loss = 0.0

print('FINISH')
torch.save(mod1.state_dict(),'/mnt/scratch/liwenrui/state_dict.pt')


