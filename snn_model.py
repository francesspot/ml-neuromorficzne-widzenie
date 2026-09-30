import os
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from spikingjelly.clock_driven import functional

import sew_resnet
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


class SCNNetwork(nn.Module):
    def __init__(self, inp_ch=2, outp_ch=101, kern_s=3, strd=1, pd=1, bs=False, bet=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=5)
        self.conv1 = nn.Conv2d(inp_ch, inp_ch * 8, kernel_size=kern_s, stride=strd, padding=pd, bias=bs)
        self.conv2 = nn.Conv2d(inp_ch * 8, inp_ch * 16, kernel_size=kern_s, stride=strd, padding=pd)
        self.flatten = nn.Flatten()
        
        self.linear = nn.Linear(inp_ch * 16 * 20 * 20, outp_ch)
        self.leaky1 = snn.Leaky(beta=bet, spike_grad=spike_grad)
        self.leaky2 = snn.Leaky(beta=bet, spike_grad=spike_grad, output=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        mem1 = self.leaky1.init_leaky()
        mem2 = self.leaky1.init_leaky()
        mem3 = self.leaky2.init_leaky()
        spk3_rec = []
        
        for step in range(x.size(0)):
            cur1 = self.pool(self.conv1(x[step]))
            spk1, mem1 = self.leaky1(cur1, mem1)
            
            cur2 = self.pool(self.conv2(spk1))
            spk2, mem2 = self.leaky1(cur2, mem2)
            
            out = self.linear(self.flatten(spk2))
            spk3, mem3 = self.leaky2(out, mem3)
            spk3_rec.append(spk3)
            
        return torch.stack(spk3_rec, dim=0), mem3

class AppModelManager:

    WEIGHT_FILES = {
        "franciszek_scnn": "franciszek_scnn_53.pth", 
        "liudmyla_scnn": "L_scnn_model.pth",
        "weronika_sew18": "sew_resnet18_best.pth",
        "weronika_t_sew18": "t_sew_resnet18_best.pth"
    }

    def __init__(self, device):
        self.device = device
        self.loaded_models = {}

    def _load_weights(self, model, model_name):
        path = self.WEIGHT_FILES.get(model_name)
        if path and os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"])
                elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                print(f"[{model_name}] Wczytano wagi z: {path}")
            except Exception as err:
                print(f"[{model_name}] Błąd ładowania pliku {path}: {err}")
        else:
            print(f"[{model_name}] Brak pliku {path} na dysku. Inicjalizacja z losowymi wagami.")

    def get_model(self, model_name):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        if model_name == "franciszek_scnn":
            model = FranciszekSCNN(
                beta=0.8417870100435023,
                threshold=0.8033307250431925,
                dropout_p=0.213169938848635,
                slope=15,
                population=10
            ).to(self.device)
            self._load_weights(model, model_name)
            for module in (model.conv1, model.lif1, model.conv2, model.lif2, model.conv3, model.lif3):
                for param in module.parameters():
                    param.requires_grad = False

        elif model_name == "liudmyla_scnn":
            model = SCNNetwork(inp_ch=2, outp_ch=101).to(self.device)
            self._load_weights(model, model_name)

        elif model_name == "weronika_sew18":
            model = sew_resnet.sew_resnet18(
                zero_init_residual=False,
                T=20,
                connect_f='ADD',
                num_classes=101
            ).to(self.device)
            self._load_weights(model, model_name)

        elif model_name == "weronika_t_sew18":
            model = sew_resnet.t_sew_resnet18(
                zero_init_residual=False,
                T=20,
                connect_f='ADD',
                num_classes=101
            ).to(self.device)
            self._load_weights(model, model_name)

        else:
            raise ValueError(f"Nieznany model: {model_name}")

        model.eval()
        self.loaded_models[model_name] = model
        return model

    def get_logits(self, model_name, tensor_data, tensor_data_flipped):
        model = self.get_model(model_name)

        with torch.no_grad():
            if model_name == "franciszek_scnn":
                spk_out, _ = model(tensor_data)
                spk_out_flipped, _ = model(tensor_data_flipped)
                spike_count = spk_out.sum(dim=0) + spk_out_flipped.sum(dim=0)
                spike_count_population = spike_count.view(tensor_data.size(1), 101, 10).sum(dim=2)
                return spike_count_population[0]

            elif model_name == "liudmyla_scnn":
                spk_out, _ = model(tensor_data)
                spk_out_flipped, _ = model(tensor_data_flipped)
                return spk_out.sum(dim=0)[0] + spk_out_flipped.sum(dim=0)[0]

            elif model_name == "weronika_t_sew18":
                spk_out = model(tensor_data)
                functional.reset_net(model)
                spk_out_flipped = model(tensor_data_flipped)
                functional.reset_net(model)
                return spk_out.sum(dim=0)[0] + spk_out_flipped.sum(dim=0)[0]

            elif model_name == "weronika_sew18":
                out = model(tensor_data)
                functional.reset_net(model)
                out_flipped = model(tensor_data_flipped)
                functional.reset_net(model)
                if out.dim() == 3:
                    out = out.mean(dim=0)
                    out_flipped = out_flipped.mean(dim=0)
                return out[0] + out_flipped[0]