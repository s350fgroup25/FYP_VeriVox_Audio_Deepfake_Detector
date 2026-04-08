import sys
import os
import json
import torch
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor
from safetensors.torch import load_file
from model_sentence1 import Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
safetensors_path = '/home/carmen/asvspoof/program/wavlm-epoch50.safetensors'

# 載入模型
model = Model(model_type='wavlm', device=device).to(device)
state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict)
model.eval()

# 載入特徵擷取器
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base')

def infer_single_audio(file_path):
    # 用 soundfile 讀音訊，避免 torchcodec 問題
    waveform, sr = sf.read(file_path)
    # 多聲道平均
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    # waveform 為 np.ndarray，傳給 feature_extractor
    inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values=input_values)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    prob_real = float(probs[0, 1])  # 類別1代表真實
    prob_fake = float(probs[0, 0])  # 類別0代表偽造
    label = prob_real >= 0.5

    return {
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "label": bool(label)
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No audio file path provided"}))
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(json.dumps({"error": f"File not found: {file_path}"}))
        sys.exit(1)

    result = infer_single_audio(file_path)
    print(json.dumps(result))

