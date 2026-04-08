import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"                     # 只用物理 GPU1
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 如需完全离线（已将模型下载到本地），再打开下面两行：
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# 如需固定 HF 缓存目录（可选）
# os.environ["HF_HOME"] = "/home/asvspoof/hf-cache"

# ===== 再 import 依赖 =====
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import Wav2Vec2FeatureExtractor
from dataset_sentence import ASVspoof2019Dataset, AudioCollator  # tencent has no attention_mask
from model_sentence1 import Model  # model1 segment model2 frame model3 traditional feature model4 frame bce loss
from torch import nn, Tensor
import torch
from safetensors.torch import save_file
from torch.utils.data import ConcatDataset, Subset
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子


# ===== 把重逻辑放进 main()，避免子进程重复执行 =====
def main():
    setup_seed(20)
    
    max_train_samples = 10000
    max_eval_samples = 2500

    asv2019_path = '/home/carmen/asvspoof/datasets/LA/'

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    processor =Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base')
    data_collator = AudioCollator(processor)
    model = Model(model_type='wavlm', device=device).to(device)

    # 从所有样本中随机抽取 ten_percent 个index（你这里 max_samples=None，所以当前不抽样）
    random.seed(42)
    train_dataset = ASVspoof2019Dataset(
        os.path.join(asv2019_path, 'ASVspoof2019_LA_train/flac/'),
        os.path.join(asv2019_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),
        max_samples=None
    )

    eval_dataset = ASVspoof2019Dataset(
        os.path.join(asv2019_path, 'ASVspoof2019_LA_dev/flac/'),
        os.path.join(asv2019_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),
        max_samples=None
    )

    training_args = TrainingArguments(
        output_dir="/home/carmen/asvspoof/datasets/models/wavlm-epoch50/",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        num_train_epochs=50,
        save_steps=1000,
        evaluation_strategy="steps",     # ← 修正：用 evaluation_strategy，而不是 eval_strategy
        eval_steps=1000,
        logging_dir='./logs',
        logging_steps=100,
        gradient_accumulation_steps=4,
        fp16=True,
        learning_rate=1e-5,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        dataloader_num_workers=8,
        # 如需进一步省显存，可开启：
        # gradient_checkpointing=True,
        # 以及在评估时降低峰值显存：
        # eval_accumulation_steps=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.evaluate()

    model_state_dict = model.state_dict()
    save_file(model_state_dict, "wavlm-epoch50.safetensors")


if __name__ == "__main__":
    main()
