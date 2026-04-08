import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"                     # 只用物理 GPU1
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from transformers import Trainer, TrainingArguments, Wav2Vec2FeatureExtractor
from dataset_sentence import ASVspoof2019Dataset,AudioCollator
import numpy as np
import torch
from collections import defaultdict
# from eval import eer_score
# from model_large import Model  #model1  segment model2 frame
from model_sentence1 import Model  #model1  segment model2 frame
from safetensors.torch import load_file
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier, export_text
from eer1 import eer

asv2019_path='/home/carmen/asvspoof/datasets/LA/'
max_eval_samples = 2500  # Number of samples to evaluate

# Initialize the feature extractor and load the pre-trained model
# Path to your safetensors file
# safetensors_path = 'tencent-add2023model-segment-anti_spoofing.safetensors'
safetensors_path = 'wavlm-epoch50.safetensors'#add2023-noise10-0.05
# safetensors_path ='add2023-large-aug2.safetensors'
# Initialize your model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model(  model_type='wavlm',device=device)
model.to(device)

state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict)
processor =Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base')
data_collator = AudioCollator(processor)
print("Model loaded successfully.")



# Load the evaluation dataset
# eval_dataset=ASVspoof2019Dataset('/home/jupyter-fjc/dataset/partialspoof/database/eval/con_wav','partialspoof_eval.txt',max_samples=None)
eval_dataset=ASVspoof2019Dataset(os.path.join(asv2019_path,'ASVspoof2019_LA_eval/flac/'),os.path.join(asv2019_path,'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'),max_samples=None)
# Setup training arguments for evaluation
training_args = TrainingArguments(
    output_dir='/home/carmen/asvspoof/datasets/models/wavlm-epoch50/',  # Directory for saving evaluation logs
    per_device_eval_batch_size=128,  # Batch size for evaluation
    do_train=False,
    do_eval=True,
    dataloader_num_workers=8,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,  # Evaluation dataset
    data_collator=data_collator
)

output = trainer.predict(eval_dataset)
predictions = output.predictions
labels = output.label_ids

# Assuming the output is logits; convert to scores
scores = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
positive_class_scores = scores[:, 1]  # Assuming class 1 is the positive class

# Calculate EER
eer_,the = eer(positive_class_scores,labels)
print("Equal Error Rate (EER):", eer_)

