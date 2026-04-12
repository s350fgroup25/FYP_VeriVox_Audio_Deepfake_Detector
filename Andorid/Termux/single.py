import time
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from safetensors.torch import load_file
from model import HFReadyModel

device = 'cpu'
print(f"Using device: {device}")

# 1. Load model
print("Loading model...")
t0 = time.perf_counter()
model = HFReadyModel(device).to(device)
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict, strict=False)  # ← 关键修改
model.eval()
t1 = time.perf_counter()
print(f"Model loaded in {t1-t0:.2f}s")

# 2. Load feature extractor
print("Loading feature extractor...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('./wav2vec2-xls-r-300m')

def test_audio(file_path):
    print(f"\nProcessing: {file_path}")
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0).numpy()
    
    if sr != 16000:
        ratio = 16000 / sr
        new_len = int(len(waveform) * ratio)
        indices = np.linspace(0, len(waveform) - 1, new_len)
        waveform = np.interp(indices, np.arange(len(waveform)), waveform)
        waveform = waveform.astype(np.float32)
    
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors='pt', padding=True)
    input_values = inputs.input_values.to(device)
    
    with torch.no_grad():
        logits = model(input_values=input_values)
    
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return float(probs[0, 1])

if __name__ == '__main__':
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else 'test_audio.wav'
    prob = test_audio(audio_file)
    print(f"\nPositive class probability: {prob:.4f}")
    print(f"Result: {'REAL' if prob > 0.5 else 'FAKE'}")
