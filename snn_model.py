import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class FranciszekSCNN(nn.Module):
    def __init__(self, beta=0.9, threshold=1.0, num_classes=101, population=10, dropout_p=0.1, slope=25):
        super().__init__()
        
        spike_grad = surrogate.fast_sigmoid(slope=slope)
        lif_params = {'learn_beta': True, 'learn_threshold': True, 'spike_grad': spike_grad}
        
        self.conv1 = nn.Conv2d(2, 24, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, **lif_params)
        
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, **lif_params)
        
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, **lif_params)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(96 * 4 * 4, 512) 
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold, **lif_params)
        
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.fc2 = nn.Linear(512, num_classes * population)
        self.lif5 = snn.Leaky(beta=beta, threshold=threshold, **lif_params)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        
        spk_rec = []
        
        for step in range(x.size(0)):
            cur1 = self.pool1(self.conv1(x[step]))
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.pool3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            
            cur4 = self.fc1(self.flatten(self.adaptive_pool(spk3)))
            spk4, mem4 = self.lif4(cur4, mem4)
            
            spk4_dropped = self.dropout(spk4)
            
            cur5 = self.fc2(spk4_dropped)
            spk5, mem5 = self.lif5(cur5, mem5)
            
            spk_rec.append(spk5)
            
        return torch.stack(spk_rec, dim=0), mem5