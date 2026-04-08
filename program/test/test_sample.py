import os
import time
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Trainer, TrainingArguments, Wav2Vec2FeatureExtractor
from torch.utils.data import Subset

# Custom dataset and collator imports
from dataset_sentence import ASVspoof2019Dataset, AudioCollator
from model_sentence1 import Model
from eer1 import eer

# Set CUDA device and PyTorch memory management
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

# Feature extractor and collator
processor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base')
data_collator = AudioCollator(processor)

# Load the evaluation dataset
eval_dataset = ASVspoof2019Dataset(
    os.path.join(asv2019_path, 'ASVspoof2019_LA_eval/flac/'),
    os.path.join(asv2019_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'),
    max_samples=None
)
# Restrict to first sample only
# eval_dataset = Subset(eval_dataset, [0])
eval_dataset = Subset(eval_dataset, list(range(100)))

# Set up evaluation arguments
training_args = TrainingArguments(
    output_dir='/home/carmen/asvspoof/datasets/models/wavlm-epoch50/',
    per_device_eval_batch_size=1,
    do_train=False,
    do_eval=True,
    dataloader_num_workers=8,
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Evaluate
output = trainer.predict(eval_dataset)
predictions = output.predictions
labels = output.label_ids

# Calculate scores (assuming binary classification)
scores = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
positive_class_scores = scores[:, 1]  # class 1 as positive

# Calculate Equal Error Rate (EER)
eer_, the = eer(positive_class_scores, labels)
print("Equal Error Rate (EER):", eer_)
