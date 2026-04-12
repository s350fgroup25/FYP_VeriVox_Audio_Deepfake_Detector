import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

class SSLModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        pre_trained_model_id = './wav2vec2-xls-r-300m'
        
        self.processing = Wav2Vec2FeatureExtractor.from_pretrained(pre_trained_model_id)
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(pre_trained_model_id)
        self.out_dim = self.wav2vec2_model.config.hidden_size
        
        # 确保所有子模块都是 eval 模式
        self.wav2vec2_model.eval()
        for param in self.wav2vec2_model.parameters():
            param.requires_grad = False

    def extract_feat(self, input_data):
        with torch.no_grad():  # 确保无梯度
            emb = self.processing(input_data, sampling_rate=16000, padding=True, return_tensors="pt").input_values[0].to(self.device)
            embb = self.wav2vec2_model(emb).last_hidden_state
        return embb

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResNet(nn.Module):
    def __init__(self, f_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(f_dim, f_dim, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(f_dim)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(f_dim, f_dim, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(f_dim)
        self.act2 = nn.ReLU()
        self.se = SELayer(f_dim)

    def forward(self, x):
        shortcut = x
        x = self.act1(self.bn1(x))
        x = self.conv1(x)
        x = self.act2(self.bn2(x))
        x = self.conv2(x)
        x = self.se(x)
        return x + shortcut

class HFReadyModel(nn.Module):
    def __init__(self, device, num_labels=2):
        super().__init__()
        self.device = device
        self.num_labels = num_labels

        self.ssl = SSLModel(device)

        self.emb_mid_len = 256
        self.emb_dim = 64
        self.channel_num = 32
        self.divide = 8

        self.ada_pool = nn.AdaptiveAvgPool2d((200, self.ssl.out_dim))
        self.fc1 = nn.Linear(self.ssl.out_dim, self.emb_mid_len)
        self.bn1 = nn.BatchNorm1d(self.emb_mid_len)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(self.emb_mid_len, self.emb_dim)
        self.resnet_2 = ResNet(self.emb_dim)

        self.conv1 = nn.Conv2d(1, 1, (1,3), padding=(0,1))
        self.conv2 = nn.Conv2d(1, self.channel_num, (3,1), padding=(1,0))
        self.downsample = nn.Conv2d(self.channel_num, self.channel_num // self.divide, 1)
        self.relu3 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(self.channel_num // self.divide, self.channel_num // self.divide, (3,1), padding=(1,0))
        self.upsample = nn.Conv2d(self.channel_num // self.divide, self.channel_num, 1)

        self.conv4 = nn.Conv2d(self.channel_num, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.dense1 = nn.Linear(self.emb_dim, 1)
        self.dense2 = nn.Linear(200, self.num_labels)

    def forward(self, input_values):
        emb = self.ssl.extract_feat(input_values)
        x = self.ada_pool(emb)
        x = self.fc1(x)

        B, T, D = x.size()
        x = x.view(-1, D)
        x = self.bn1(x)
        x = x.view(B, T, D)
        x = self.relu1(x)

        x = self.fc2(x)
        x = x.transpose(1, 2)
        x = self.resnet_2(x)
        x = x.transpose(1, 2)

        x_4d = x.unsqueeze(1)
        x_tplus = self.conv1(x_4d)

        Dg = x_tplus[:, :, 1:, :] - x_4d[:, :, :-1, :]
        Dg = F.pad(Dg, (0, 0, 0, 1), value=0.0)

        p1 = self.conv2(Dg)
        p2 = self.downsample(p1)
        p2 = self.relu3(p2)
        p2 = self.conv3(p2)
        p2 = self.upsample(p2)

        gate = self.conv4(p1 + p2).squeeze(1)
        gate = self.sigmoid(gate)

        x = gate * x
        x = self.dense1(x)
        logits = self.dense2(x.squeeze(-1))
        return logits
