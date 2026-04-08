import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
class Self_Attentive_Pooling(nn.Module):
    def __init__(self, dim):
        super(Self_Attentive_Pooling, self).__init__()
        self.sap_linear = nn.Linear(dim, dim)
        self.attention = nn.Parameter(torch.randn(dim, 1) * 0.01)  # 改进的初始化

    def forward(self, x):
        h = torch.tanh(self.sap_linear(x))  # 激活
        w = torch.matmul(h, self.attention).squeeze(dim=2)  # 计算权重

        # 在应用 softmax 前，处理数值稳定性问题
        w = w - torch.max(w, dim=1, keepdim=True)[0]  # 减去每行的最大值以防止梯度爆炸
        w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)  # 应用 softmax

        # 加权求和
        x = torch.sum(x * w, dim=1)  # 加权平均
        return x
class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.attention = nn.Linear(input_dim, 1)  # 用一个线性层生成注意力得分

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        attention_scores = self.attention(x)  # [batch_size, seq_len, 1]
        attention_scores = torch.softmax(attention_scores, dim=1)  # 对序列长度维度进行softmax
        weighted_output = x * attention_scores  # [batch_size, seq_len, input_dim]
        pooled_output = weighted_output.sum(dim=1)  # [batch_size, input_dim]
        return pooled_output
class Attentive_Statistics_Pooling(nn.Module):
    def __init__(self, input_dim):
        super(Attentive_Statistics_Pooling, self).__init__()
        self.input_dim = input_dim
        self.attention = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.eps = 1e-10  # 防止除以零

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        
        # Step 1: Compute attention weights
        attn = self.attention(x)  # [batch_size, seq_len, input_dim]
        attn = torch.tanh(attn)   # 激活函数
        attn = self.softmax(attn) # Softmax 归一化获取权重 [batch_size, seq_len, input_dim]

        # Step 2: Compute weighted mean
        mean = torch.sum(attn * x, dim=1) / torch.sum(attn, dim=1).clamp(min=self.eps)  # [batch_size, input_dim]

        # Step 3: Compute weighted std deviation
        x_centered = x - mean.unsqueeze(1)  # 减去均值中心化
        var = torch.sum(attn * (x_centered ** 2), dim=1) / torch.sum(attn, dim=1).clamp(min=self.eps)
        std_dev = torch.sqrt(var + self.eps)  # 标准差 [batch_size, input_dim]

        # Step 4: Concatenate mean and std deviation
        pooled_stats = torch.cat((mean, std_dev), dim=1)  # [batch_size, 2 * input_dim]

        return pooled_stats
        
class Model(Wav2Vec2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.additional_fc = nn.Linear(config.hidden_size, 512)
        self.projector = nn.Linear(512, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size*2, config.num_labels)
        #self.self_attention_pool = SelfAttentionPooling(256)
        #self.Self_Attentive_Pooling=Self_Attentive_Pooling(256)
        self.Attentive_Statistics_Pooling=Attentive_Statistics_Pooling(256)
    def forward(self, input_values, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=None, labels=None    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]  # 获取最后一层的隐藏状态
        hidden_states = self.additional_fc(hidden_states)
        hidden_states = self.projector(hidden_states)

        # 应用自注意力池化
        #pooled_output = self.self_attention_pool(hidden_states)
        #pooled_output = torch.max(hidden_states, dim=1)[0]  # max返回一个元组(max, max_indices)，这里只取max值    最大池化
        #pooled_output = torch.mean(hidden_states, dim=1)          # 使用平均池化
        #pooled_output = self.Self_Attentive_Pooling(hidden_states)
        pooled_output = self.Attentive_Statistics_Pooling(hidden_states)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
