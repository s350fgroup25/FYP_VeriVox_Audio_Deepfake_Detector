import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# ===== SSL Backbone =====
class SSLModel(nn.Module):
    def __init__(self, device, pretrained_id=None):
        super().__init__()
        self.device = device

        # Local folders on Raspberry Pi (no internet needed)
        feature_dir = "/home/carmen/asvspoof/program/wav2vec2-xls-r-300m-feature"
        model_dir   = "/home/carmen/asvspoof/program/wav2vec2-xls-r-300m-model"

        self.processing = Wav2Vec2FeatureExtractor.from_pretrained(feature_dir)
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(model_dir)
        self.out_dim = self.wav2vec2_model.config.hidden_size  # 1024 for xls-r-300m

    def extract_feat(self, input_data):
        emb = self.processing(
            input_data,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        ).input_values[0].to(self.device)
        embb = self.wav2vec2_model(emb).last_hidden_state  # [batch, T, 1024]
        del emb
        torch.cuda.empty_cache()
        return embb

# ===== SE / ResNet =====
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
        self.in_planes = f_dim
        self.conv1 = nn.Conv1d(f_dim, f_dim, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(f_dim)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(f_dim, f_dim, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(f_dim)
        self.bn3 = nn.BatchNorm1d(f_dim)  # 占位
        self.act2 = nn.ReLU()
        self.se = SELayer(f_dim)

    def forward(self, x):
        shortcut = x
        x = self.act1(self.bn1(x))
        x = self.conv1(x)
        x = self.act2(self.bn2(x))
        x = self.conv2(x)
        x = self.se(x)
        x = x + shortcut
        return x

# ===== 主干模型 =====
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

        self.conv1 = nn.Conv2d(1, 1, (1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(1, self.channel_num, (3, 1), padding=(1, 0))
        self.downsample = nn.Conv2d(self.channel_num, self.channel_num // self.divide, 1)
        self.relu3 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(
            self.channel_num // self.divide,
            self.channel_num // self.divide,
            (3, 1),
            padding=(1, 0)
        )
        self.upsample = nn.Conv2d(self.channel_num // self.divide, self.channel_num, 1)

        self.conv4 = nn.Conv2d(self.channel_num, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.dense1 = nn.Linear(self.emb_dim, 1)
        self.dense2 = nn.Linear(200, self.num_labels)

        class_weights = torch.tensor([0.1, 0.9], dtype=torch.float32)
        self.register_buffer("ce_weights", class_weights)
        self.loss_fn = nn.CrossEntropyLoss(weight=self.ce_weights)

    def forward(self, input_values, labels=None):
        # (1) SSL features: (B, T, C)
        emb = self.ssl.extract_feat(input_values)

        # (2) 下游頭
        x = self.ada_pool(emb)            # (B, 200, C)
        x = self.fc1(x)                   # (B, 200, 256)

        B, T, D = x.size()
        x = x.view(-1, D)                 # (B*T, 256)
        x = self.bn1(x)
        x = x.view(B, T, D)
        x = self.relu1(x)

        x = self.fc2(x)                   # (B, 200, 64)
        x = x.transpose(1, 2)             # (B, 64, 200)
        x = self.resnet_2(x)              # (B, 64, 200)
        x = x.transpose(1, 2)             # (B, 200, 64)

        x_4d = x.unsqueeze(1)             # (B, 1, 200, 64)
        x_tplus = self.conv1(x_4d)        # (B, 1, 200, 64)

        Dg = x_tplus[:, :, 1:, :] - x_4d[:, :, :-1, :]
        Dg = F.pad(Dg, (0, 0, 0, 1), value=0.0)  # (B, 1, 200, 64)

        p1 = self.conv2(Dg)               # (B, C, 200, 64)
        p2 = self.downsample(p1)
        p2 = self.relu3(p2)
        p2 = self.conv3(p2)
        p2 = self.upsample(p2)

        gate = self.conv4(p1 + p2).squeeze(1)  # (B, 200, 64)
        gate = self.sigmoid(gate)

        x = gate * x                       # (B, 200, 64)

        x = self.dense1(x)                 # (B, 200, 1)
        logits = self.dense2(x.squeeze(-1))  # (B, num_labels)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits

