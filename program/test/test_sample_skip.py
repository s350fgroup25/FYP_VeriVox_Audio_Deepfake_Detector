import os
import time
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Trainer, TrainingArguments, Wav2Vec2FeatureExtractor
from torch.utils.data import Subset

from dataset_sentence import ASVspoof2019Dataset, AudioCollator
from model_sentence1 import Model
# Avoid importing eer since we're skipping EER calculation

# Set CUDA device and PyTorch memory config
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

asv2019_path = '/home/carmen/asvspoof/datasets/LA/'
safetensors_path = 'wavlm-epoch50.safetensors'

# Measure model load time
start_time = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model(model_type='wavlm', device=device)
model.to(device)
state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict)
end_time = time.time()
print(f"Model loaded successfully in {end_time - start_time:.3f} seconds.")

# Initialize feature extractor and collator
processor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base')
data_collator = AudioCollator(processor)

# Load eval dataset and restrict to only one sample (index 0)
eval_dataset_full = ASVspoof2019Dataset(
    os.path.join(asv2019_path, 'ASVspoof2019_LA_eval/flac/'),
    os.path.join(asv2019_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'),
    max_samples=None
)
eval_dataset = Subset(eval_dataset_full, [0])  # only first sample
# eval_dataset = Subset(eval_dataset_full, list(range(100)))

# Setup training arguments for evaluation
training_args = TrainingArguments(
    output_dir='/home/carmen/asvspoof/datasets/models/wavlm-epoch50/',
    per_device_eval_batch_size=1,
    do_train=False,
    do_eval=True,
    dataloader_num_workers=8,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Run evaluation
output = trainer.predict(eval_dataset)
predictions = output.predictions
labels = output.label_ids

# Convert logits to probabilities (softmax)
scores = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
positive_class_scores = scores[:, 1]  # assuming class 1 is positive

# Skip EER calculation - just print scores and labels
print("Skipping EER calculation due to single sample evaluation.")
print("Positive class scores:", positive_class_scores)
print("Labels:", labels)
