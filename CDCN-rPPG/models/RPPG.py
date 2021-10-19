import torch
from torch import nn


class RPPG(nn.Module):
    def __init__(self, drop_rate=0.25):
        super(RPPG,self).__init__()
        self.batch_normaliztion = nn.BatchNorm1d(3)

        self.relu = nn.ReLU(inplace=True)
        self.unflat = nn.Unflatten(1, torch.Size([32, 32]))

        self.depthmap = nn.Sequential(
            nn.Linear(3, 64),
            self.relu,
            nn.Linear(64, 256),
            self.relu,
            nn.Linear(256, 1024),
            self.relu,
            self.unflat, # [1, 32, 32]
        )

    def forward(self,x):
        x = self.batch_normaliztion(x)
        x = self.relu(x)
        out = self.depthmap(x)
        
        return out


class HYPER(nn.Module):
    def __init__(self, cdc_network, num_classes=2, drop_rate=0.25):
        super(HYPER,self).__init__()
        self.cdc_network = cdc_network()
        self.rPPG = RPPG()

    def forward(self,x1,x2):
        cdc_out = self.cdc_network(x1)  # x [1, 32, 32] 
        rppg_out = self.rPPG(x2)  # x [1, 32, 32] 

        #depth = torch.cat((cdc_out[0],rppg_out), dim=1)
        depth = cdc_out[0] + rppg_out

        return tuple([depth]+list(cdc_out)[1:]) # 模仿 CDCN 的返回


if __name__ == "__main__":
    #model = RPPG()
    #inputs = torch.randn((2,3))
    #print(model(inputs))

    from CDCNs import CDCN, CDCNpp
    
    model = HYPER(cdc_network=CDCN)
    input1 = torch.randn(2,3,256,256)
    input2 = torch.randn((2,3))
    print(model(input1, input2))