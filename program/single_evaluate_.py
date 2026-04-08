import os
os.environ['DISABLE_TORCHCODEC'] = '1'  # 強制禁用 torchcodec
import time
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
from safetensors.torch import load_file
from model_sentence1 import Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
safetensors_path = 'wavlm-epoch50.safetensors'

# 1. Load model and checkpoint
t0 = time.perf_counter()
model = Model(model_type='wavlm',device=device).to(device)
state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict)
model.eval()
t1 = time.perf_counter()
print(f"[1] Model init & load: {t1-t0:.3f} s")

# 2. Load feature extractor
t2_start = time.perf_counter()
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base')
t2 = time.perf_counter()
print(f"[2] FeatureExtractor load: {t2-t2_start:.3f} s")

def infer_single_audio(file_path):
    # 3. Read audio
    t3_start = time.perf_counter()
    waveform, sr = torchaudio.load(file_path)        # (channels, time)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0).numpy()
    t3 = time.perf_counter()
    print(f"[3] Audio loading: {t3-t3_start:.3f} s")

    # 4. Feature extraction
    t4_start = time.perf_counter()
    inputs = feature_extractor(
        waveform,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs.input_values.to(device)
    t4 = time.perf_counter()
    print(f"[4] Feature extraction: {t4-t4_start:.3f} s")

    # 5. Forward + softmax
    t5_start = time.perf_counter()
    with torch.no_grad():
        logits = model(input_values=input_values)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    t5 = time.perf_counter()
    print(f"[5] Inference & softmax: {t5-t5_start:.3f} s")

    return float(probs[0, 1])

if __name__ == '__main__':
    test_file = '/home/carmen/asvspoof/testSet/covent/covent1.wav'
    prob = infer_single_audio(test_file)
    print(f"Positive class probability: {prob:.4f}")
