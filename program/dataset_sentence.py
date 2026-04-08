import os
import numpy as np
import torch
from sympy.codegen.ast import continue_
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor
import soundfile as sf          #tencent segment
# from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
import  math
import argparse
from types import SimpleNamespace
from typing import List, Tuple
def repeat_samples(filename, target_samples=64000):
    # 读取音频文件
    waveform, sample_rate = sf.read(filename)

    # 获取当前样本长度
    current_samples = waveform.shape[0]

    if current_samples >= target_samples:
        # 如果音频样本数多于或等于目标样本数，进行裁剪
        waveform = waveform[:target_samples]
    else:
        # 如果音频样本数少于目标样本数，进行重复填充
        # 计算至少需要重复多少次才能达到或超过目标样本数
        repeat_times = target_samples // current_samples + 1
        # 重复波形
        waveform = np.tile(waveform, repeat_times)
        # 裁剪到精确的目标样本数
        waveform = waveform[:target_samples]

    # 返回修改后的音频数据和采样率
    return waveform, sample_rate

class ASVspoof2019Dataset(Dataset):
    def __init__(self, root_dir, labels_dir,max_samples=None):
        #self.file_paths = glob.glob(os.path.join(root_dir, '*.pckl'))
        # Load the data from the text file
        data = np.genfromtxt(labels_dir, dtype='str', delimiter=' ')
        # Extract the last column which contains the labels
        labels = data[:, -1].tolist()
        files= data[:, 1].tolist()   #modified
        self.file_paths = [os.path.join(root_dir, file + '.flac') if not file.endswith('.flac') else os.path.join(root_dir, file) for file in files]   #modified
        self.labels = labels
        if max_samples:
            self.file_paths = self.file_paths[:max_samples]
            self.labels = self.labels[:max_samples]

    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        # audio, sample_rate = sf.read(self.file_paths[idx])
        audio, sample_rate = repeat_samples(self.file_paths[idx])
        label_str = self.labels[idx]   #modified
        label = '0' if label_str == 'spoof' else '1'
        return audio, label

# def collate_fn(batch):
#     batch = [sample for sample in batch if sample is not None]
#     audios, segment_labels, frame_labels = zip(*batch)
#     feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('TencentGameMate/chinese-wav2vec2-base')
#     inputs = feature_extractor(audios, return_tensors="pt", padding=True, sampling_rate=16000)
#     # inputs现在是一个字典，包含了'input_values'和'attention_mask'
#     input_values = inputs.input_values  # 特征值
#     # attention_mask = inputs.attention_mask  # 注意力掩码
#     # Convert labels from 'bonafide'/'spoof' to 0/1 integers if they are not already integers
#     segment_labels = torch.tensor(segment_labels, dtype=torch.long)
#     frame_labels=np.array(frame_labels)
#     all_frame_labels =torch.from_numpy(frame_labels).long()
#     # all_frame_labels = torch.tensor(frame_labels,
#     #                                 dtype=torch.long)  # Ensure labels are torch.long for classification
#     return {"input_values": input_values, "labels": segment_labels,
#             "all_frame_labels": all_frame_labels}

# def collate_fn(batch):
#     audios, labels = zip(*batch)
#     feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base')
#     inputs = feature_extractor(audios, return_tensors="pt", padding=True, sampling_rate=16000)

#     # inputs现在是一个字典，包含了'input_values'和'attention_mask'
#     input_values = inputs.input_values  # 特征值
#     # attention_mask = inputs.attention_mask  # 注意力掩码
#     # Convert labels from 'bonafide'/'spoof' to 0/1 integers if they are not already integers
#     labels = [0 if label == '0' else 1 for label in labels]
#     labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are torch.long for classification
#     return {"input_values": input_values,"labels": labels}

class AudioCollator:
    """
    可注入的 data collator：把特征提取器在主进程创建一次，再传进来；
    collate 时只做张量拼接，不再访问 Hugging Face Hub。
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch: List[Tuple[np.ndarray, str]]):
        audios, labels = zip(*batch)  # audios: list[np.ndarray], labels: list[str]
        # 统一采样率为 16k（wavlm-base 训练采样率），你在 repeat_samples 已经保证长度
        inputs = self.feature_extractor(
            audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
        )
        input_values = inputs.input_values

        # 你上游把 'spoof'->'0'、'bonafide'->'1' 了；这里把字符变成 int
        labels = [0 if lab == '0' else 1 for lab in labels]
        labels = torch.tensor(labels, dtype=torch.long)

        return {"input_values": input_values, "labels": labels}




