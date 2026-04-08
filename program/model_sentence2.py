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
        output = self.model(input_tmp, output_hidden_states=True)
        selected_hidden_states = output.hidden_states[1:13]
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

        # Initialize Attentive Statistics Pooling layers for each selected hidden state
        self.pooling_layers = nn.ModuleList([
            Attentive_Statistics_Pooling(self.ssl_model.out_dim) for _ in range(self.num_layers)
        ])

        # Linear layers after each pooling layer, following the diagram structure
        self.fc_layers = nn.ModuleList([
            nn.Linear( self.ssl_model.out_dim, 512) for _ in range(self.num_layers)
        ])

        # Final layers after pooling
        self.final_fc = nn.Linear(512, 256)
        self.classifier = nn.Linear(512, 2)
        self.fc = nn.Linear(self.num_layers, 1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.Attentive_Statistics_Pooling = Attentive_Statistics_Pooling(256)

    def forward(self, input_values,  labels=None):
        # Extract features from multiple hidden states
        hidden_states = self.ssl_model.extract_feat(input_values)

        pooled_outputs = []
        for i, hidden_state in enumerate(hidden_states):
            # pooled_output = self.pooling_layers[i](hidden_state)
            pooled_output = self.fc_layers[i](hidden_state)
            pooled_outputs.append(pooled_output)

        # Aggregate pooled outputs (e.g., concatenation or mean pooling)
        combined_output = torch.stack(pooled_outputs)
        combined = self.fc(combined_output.view(self.num_layers, -1).transpose(0, 1))
        # combined_output = torch.mean(torch.stack(pooled_outputs), dim=0)
        #
        # input_tensor_reshaped = input_tensor.view(12, -1).transpose(0, 1)


        # Additional processing layers
        # average_vector = self.projector(combined.view(hidden_state.size(0), hidden_state.size(1), 512))

        # pooled_output = self.Attentive_Statistics_Pooling(average_vector)
        final_hidden = self.final_fc(combined.view(hidden_state.size(0), hidden_state.size(1), 512))
        final_hidden = self.Attentive_Statistics_Pooling(final_hidden)
        logits = self.classifier(final_hidden)
        # logits2= self.classifier2(pooled_output)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits
