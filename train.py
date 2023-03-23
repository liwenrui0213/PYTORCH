import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

device = torch.device('cuda:0')


class dataset(Dataset):
    def __init__(self,dir):
        self.dir = dir
        self.file = np.load(self.dir)

    def __getitem__(self,index):
        target = torch.from_numpy(self.file[index][:5])
        feature = torch.from_numpy(self.file[index][5:])
        return feature, target
    def __len__(self):
        return len(self.file)
class ResNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.linear1 = nn.Linear(n,n)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(n,n)
        self.relu2 = nn.ReLU()
    def forward(self,x):
        id = x
        out = x
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)+id
        out = self.relu2(out)
        return out

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(6,32)
        self.resblock1 = ResNet(32)
        self.Linear2 = nn.Linear(32,128)
        self.resblock2 = ResNet(128)
        self.resblock3 = ResNet(128)
        self.Linear3 = nn.Linear(128,32)
        self.resblock4 = ResNet(32)
        self.Linear4 = nn.Linear(32,5)
    def forward(self,x):
        out = x
        out = self.Linear1(out)
        out = F.relu(out)
        out = self.resblock1(out)
        out = self.Linear2(out)
        out = F.relu(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.Linear3(out)
        out = F.relu(out)
        out = self.resblock4(out)
        out = self.Linear4(out)
        return out



dset = dataset('/mnt/scratch/liwenrui/full/full_data_p00.npy')
print(len(dset))
data_loader = DataLoader(dataset=dset,batch_size=128,shuffle=False,num_workers=0)
mod1 = model()
#mod1 = nn.DataParallel(mod1,device_ids=[0,1,2,3,4,5,6,7])
mod1.to(device)
optim = torch.optim.Adam(mod1.parameters(),lr=1e-2)
loss = nn.MSELoss()
mod1.train()
for i in range(4):
    print('------epoch {}--------'.format(i+1))
    step = 0
    running_loss = 0.0
    for data in data_loader:
        features,target = data
        x = features.to(device)
        y = target.to(device)
        optim.zero_grad()
        output = mod1.forward(x)
        training_loss = loss(output,y).to(device)
        training_loss.backward()
        optim.step()
        step+=1
        running_loss+=training_loss.item()
        if (step%10000 == 0):
            print('epoch{} , step{}, avrg_loss={}'.format(i+1,step,running_loss/10000))
            running_loss = 0.0

print('FINISH')
torch.save(mod1.state_dict(),'/mnt/scratch/liwenrui/state_dict2.pt')


