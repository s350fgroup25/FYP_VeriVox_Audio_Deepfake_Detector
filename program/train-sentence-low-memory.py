import os
import numpy as np
import random
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import Wav2Vec2FeatureExtractor
from dataset_sentence import ASVspoof2019Dataset, AudioCollator  # tencent has no attention_mask
from model_sentence1 import Model  # model1 segment model2 frame model3 traditional feature model4 frame bce loss
from safetensors.torch import save_file

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Uncomment to restrict GPU usage

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    setup_seed(20)

    asv2019_path = '/home/carmen/asvspoof/datasets/LA/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    processor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base')
    data_collator = AudioCollator(processor)
    model = Model(model_type='wavlm', device=device).to(device)

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
        eval_strategy="steps",
        eval_steps=1000,
        logging_dir='./logs',
        logging_steps=100,
        gradient_accumulation_steps=4,
        fp16=False,
        learning_rate=1e-5,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        dataloader_num_workers=8,
    )

    from torch.amp import GradScaler, autocast

    scaler = GradScaler(device="cuda")

    class AMPTrainer(Trainer):
        def training_step(self, model, inputs, return_outputs=False):
            model.train()
            inputs = self._prepare_inputs(inputs)
            with autocast(device_type='cuda'):
                loss = self.compute_loss(model, inputs)
                if loss.ndim > 0:
                    loss = loss.mean()  # make sure loss is scalar
            scaler.scale(loss).backward()
            if return_outputs:
                return loss.detach(), None
            return loss.detach()


        def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_closure=None,
                           on_tpu=None, using_native_amp=None, using_lbfgs=None):
            if optimizer is None:
                optimizer = self.optimizer
            if optimizer_closure is None:
                def optimizer_closure():
                    return None
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    trainer = AMPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.scaler = scaler

    trainer.train()
    trainer.evaluate()

    model_state_dict = model.state_dict()
    save_file(model_state_dict, "wavlm-epoch50.safetensors")

if __name__ == "__main__":
    main()
