import os
import time
import numpy as np
import torch
import random
from transformers import Trainer, TrainingArguments, Wav2Vec2FeatureExtractor
from dataset_sentence import ASVspoof2019Dataset, AudioCollator
from model_sentence1 import Model
from safetensors.torch import load_file
from eer1 import eer
from contextlib import redirect_stdout

## Configuration (keep your existing settings)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

asv2019_path = '/home/carmen/asvspoof/datasets/LA/'
safetensors_path = 'wavlm-epoch50.safetensors'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model(model_type='wavlm', device=device)
model.to(device)


state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict)

processor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base')
data_collator = AudioCollator(processor)
print("Model loaded successfully.")

## Load the evaluation dataset (fetch all first, we'll sub-select later)

eval_dataset_full = ASVspoof2019Dataset(
    os.path.join(asv2019_path, 'ASVspoof2019_LA_eval/flac/'),
    os.path.join(asv2019_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'),
    max_samples=None
)

## Randomly select 10 samples if there are more than 10
num_samples = len(eval_dataset_full)
print(f"Total samples in full dataset: {num_samples}")
if num_samples > 10:
    indices = random.sample(range(num_samples), 10)
    print(f"Randomly chosen indices: {indices}")
    eval_dataset = torch.utils.data.Subset(eval_dataset_full, indices)
else:
    eval_dataset = eval_dataset_full

## Setup training arguments for evaluation

training_args = TrainingArguments(
    output_dir='/home/carmen/asvspoof/datasets/models/wavlm-epoch50/',
    per_device_eval_batch_size=128,
    do_train=False,
    do_eval=True,
    dataloader_num_workers=8,
)

## Initialize the Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

## Add output redirection here
output_dir = 'test_output'
os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist

output_path = os.path.join(output_dir, 'test_output.txt')

with open(output_path, 'w') as fout, redirect_stdout(fout):

    ## Start timing total evaluation

    total_start = time.time()
    start_eval = time.time()

    # Run evaluation

    output = trainer.predict(eval_dataset)
    end_eval = time.time()
    total_end = time.time()

    predictions = output.predictions
    labels = output.label_ids

    ## Convert logits to probabilities and take the positive class score

    scores = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
    positive_class_scores = scores[:, 1] # Assuming class 1 is the positive class

    ## Compute EER

    eer_value, the = eer(positive_class_scores, labels)
    print("Equal Error Rate (EER):", eer_value)

    ## Attempt to map per-sample file names and durations (adjust as per your dataset API)

    try:
        file_names = []
        durations_ms = []
        if hasattr(eval_dataset, "get_file_name_by_index"):
            for idx in range(len(predictions)):
                file_names.append(eval_dataset.get_file_name_by_index(idx))
                if hasattr(eval_dataset, "get_duration_by_index"):
                    durations_ms.append(eval_dataset.get_duration_by_index(idx))
                else:
                    durations_ms.append(None)
        elif hasattr(eval_dataset, "files"):
            for fp in eval_dataset.files:
                file_names.append(fp)
                durations_ms.append(None)
        else:
            file_names = [f"sample_{i:06d}.wav" for i in range(len(predictions))]
            durations_ms = [None] * len(predictions)
    except Exception:
        file_names = [f"sample_{i:06d}.wav" for i in range(len(predictions))]
        durations_ms = [None] * len(predictions)

    # Ground truth list
    ground_truth = [int(x) for x in labels.tolist()]

    ## Print per-sample details

    print("Per-sample details:")
    total_samples = len(predictions)
    runtime_total_ms = (total_end - total_start) * 1000.0

    # Distribute total inference time evenly per sample for a simple estimate

    per_sample_runtime_ms = [runtime_total_ms / total_samples] * total_samples

    min_len = min(len(file_names), len(predictions), len(ground_truth))
    for i in range(min_len):
        fname = file_names[i]
        pred = int(predictions[i]) if isinstance(predictions[i], (int, np.integer)) else int(predictions[i].argmax(-1))
        gt = int(ground_truth[i])
        correct = (pred == gt)
        runtime_ms = per_sample_runtime_ms[i]
        duration_ms = durations_ms[i]
        print(f"Sample {i+1}:")
        print(f" file_name: {fname}")
        print(f" predicted_label: {pred}")
        print(f" ground_truth_label: {gt}")
        print(f" correct: {correct}")
        print(f" runtime_ms: {runtime_ms:.3f}")
        print(f" audio_duration_ms: {duration_ms}")

    print(f"Total samples evaluated: {min_len}")
    print(f"Total evaluation time (s): {(end_eval - start_eval):.3f}")

