import sys
import os
import json
import torch
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor
from safetensors.torch import load_file
from model_sentence1 import Model
import warnings
import logging
#
# Suppress all warnings (or tune this to only suppress specific ones)
warnings.filterwarnings("ignore")

# Set logging level to WARNING or ERROR for transformers and torch
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
safetensors_path = '/home/carmen/asvspoof/program/wavlm-epoch50.safetensors'
# Suppress stdout
sys.stdout = open(os.devnull, 'w')
# Load model
model = Model(model_type='wavlm', device=device).to(device)
state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict)
model.eval()
# Restore stdout
sys.stdout = sys.__stdout__
# Load feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base')

def infer_single_audio(file_path):
    # Use soundfile to read audio to avoid torch codec issues
    waveform, sr = sf.read(file_path)
    # Average multi-channel to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    # waveform is np.ndarray, give to feature extractor
    inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values=input_values)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    prob_real = float(probs[0, 1])  # class 1 = real
    prob_fake = float(probs[0, 0])  # class 0 = fake
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
    # Print only JSON result to stdout
    print(json.dumps(result))

