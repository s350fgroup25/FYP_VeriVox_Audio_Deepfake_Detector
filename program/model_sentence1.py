import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from IPython.lib.tests.test_backgroundjobs import sleeper
from transformers import Wav2Vec2Model,Wav2Vec2Config, HubertModel


#  double loss

class Attentive_Statistics_Pooling(nn.Module):
    def __init__(self, input_dim):
        super(Attentive_Statistics_Pooling, self).__init__()
        self.attention = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.eps = 1e-10

    def forward(self, x):
        attn = self.attention(x)
        attn = torch.tanh(attn)
        attn = self.softmax(attn)

        mean = torch.sum(attn * x, dim=1) / torch.sum(attn, dim=1).clamp(min=self.eps)
        x_centered = x - mean.unsqueeze(1)
        var = torch.sum(attn * (x_centered ** 2), dim=1) / torch.sum(attn, dim=1).clamp(min=self.eps)
        std_dev = torch.sqrt(var + self.eps)

        pooled_stats = torch.cat((mean, std_dev), dim=1)
        return pooled_stats


class SSLModel(nn.Module):
    def __init__(self, model_type,device,dtype=torch.float32):
        super(SSLModel, self).__init__()
        # self.model = Wav2Vec2Model.from_pretrained('microsoft/wavlm-base')
        # config = Wav2Vec2Config()
        # print("Default Feature Extractor Config:")
        # print(config.conv_stride)
        # config.conv_stride=[5, 2, 2, 2, 2, 2, 1]
        # print(config.conv_stride)
        # model = Wav2Vec2Model(config)
        self.device = device
        self.dtype = dtype

        if model_type == 'distilhubert':
            # DistilHuBERT: reduces HuBERT size by ~75% and speeds up by ~73% :contentReference[oaicite:0]{index=0}
            self.model = HubertModel.from_pretrained('ntu-spml/distilhubert',conv_stride=[5, 2, 2, 2, 2, 2, 1])
            self.out_dim = self.model.config.hidden_size

        elif model_type == 'distil-wav2vec2':
            # DistilWav2Vec2: ~45% size of base, ~2× faster than original wav2vec2 :contentReference[oaicite:1]{index=1}
            self.model = Wav2Vec2Model.from_pretrained('OthmaneJ/distil-wav2vec2',conv_stride=[5, 2, 2, 2, 2, 2, 1])
            self.out_dim = self.model.config.hidden_size
            
        elif model_type == 'wav2vec2':
            # DistilWav2Vec2: ~45% size of base, ~2× faster than original wav2vec2 :contentReference[oaicite:1]{index=1}
            self.model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base',conv_stride=[5, 2, 2, 2, 2, 2, 1])
            self.out_dim = self.model.config.hidden_size

        elif model_type == 'hubert':
            # DistilWav2Vec2: ~45% size of base, ~2× faster than original wav2vec2 :contentReference[oaicite:1]{index=1}
            self.model = HubertModel.from_pretrained('facebook/hubert-base-ls960',conv_stride=[5, 2, 2, 2, 2, 2, 1])
            self.out_dim = self.model.config.hidden_size

        else:
            # Default to WavLM
            self.model = Wav2Vec2Model.from_pretrained('microsoft/wavlm-base',conv_stride=[5, 2, 2, 2, 2, 2, 1])
            self.out_dim = self.model.config.hidden_size
            
        self.model.to(device=self.device, dtype=self.dtype)
        # 打印参数数量，调试用
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[SSLModel] Loaded model '{model_type}' with {n_params} parameters on {self.device} ({self.dtype})")

    def extract_feat(self, input_data):
        # param = next(self.model.parameters())
        # if param.device != input_data.device or param.dtype != input_data.dtype:
        #     self.model.to(input_data.device, dtype=input_data.dtype)
            
        self.model.train()
        input_tmp = input_data.squeeze(-1) if input_data.ndim == 3 else input_data
        output = self.model(input_tmp, output_hidden_states=False)
        selected_hidden_states = output.last_hidden_state
        return selected_hidden_states


class Model(nn.Module):
    def __init__(self, model_type,device):
        global num_layers
        super().__init__()
        self.device = device
        self.model_type=model_type
        self.ssl_model = SSLModel(model_type=self.model_type,device=self.device)
        if self.model_type == 'distilhubert':
            self.num_layers = 2

        elif self.model_type == 'distil-wav2vec2':
            self.num_layers = 6

        else:
            self.num_layers = 12

        self.additional_fc = nn.Linear(768, 512)
        self.projector = nn.Linear(512, 256)
        self.classifier = nn.Linear(512, 2)
        #self.self_attention_pool = SelfAttentionPooling(256)
        #self.Self_Attentive_Pooling=Self_Attentive_Pooling(256)
        self.Attentive_Statistics_Pooling=Attentive_Statistics_Pooling(256)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, input_values,  labels=None):
        # Extract features from multiple hidden states
        hidden_states = self.ssl_model.extract_feat(input_values)

        hidden_states = self.additional_fc(hidden_states)
        hidden_states = self.projector(hidden_states)

        # 应用自注意力池化
        #pooled_output = self.self_attention_pool(hidden_states)
        #pooled_output = torch.max(hidden_states, dim=1)[0]  # max返回一个元组(max, max_indices)，这里只取max值    最大池化
        #pooled_output = torch.mean(hidden_states, dim=1)          # 使用平均池化
        #pooled_output = self.Self_Attentive_Pooling(hidden_states)
        pooled_output = self.Attentive_Statistics_Pooling(hidden_states)
        logits = self.classifier(pooled_output)
        # logits2= self.classifier2(pooled_output)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits
