import pandas as pd
import numpy as np
import argparse
import os
from transformers import ViTImageProcessor, Trainer
from transformers import ViTForImageClassification, TrainingArguments
from datasets import load_from_disk, Dataset
from sklearn.metrics import accuracy_score, classification_report, det_curve
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch
from evaluate import load

model_name = 'google/vit-base-patch16-224-in21k'

##### ARGS #####
parser = argparse.ArgumentParser(description='Train the VIT')

parser.add_argument('--out-path', '-o', metavar="O", dest='out_path', type=str, help='Job out path')
parser.add_argument('--epochs', '-e', metavar='E', dest="epochs" ,type=int, help='Number of epochs')
parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, help='Learning rate', dest='lr')
parser.add_argument('--data-set', '-d', dest='dataset', metavar='D', type=str, help='Dataset location')
parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, help='Batch size')

args = parser.parse_args()

print(f"Output:  {args.out_path}")
print(f"Epochs:  {args.epochs}")
print(f"Learning rate:  {args.lr}")
print(f"Dataset:  {args.dataset}")
print(f"Batch size:  {args.batch_size}")

##### LOAD dataset ####
dataset_processed = load_from_disk(f"{os.environ['HOME']}/biometrics/black-hole/processed_datasets/{args.dataset}/")
labels = dataset_processed['train'].features['label'].names
print(f"Labels of Dataset: {labels}")

#### AUX functions ####
metric = load("accuracy")
print("Loaded accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

#### LOAD processor and model ####
processor = ViTImageProcessor.from_pretrained(model_name)
print(processor)

model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={str(i): label for i, label in enumerate(labels)},
    label2id={label: str(i) for i, label in enumerate(labels)},
)

model.train() ## set the model into train mode

print(model)
#### Training arguments ####
training_args = TrainingArguments(
  output_dir=f"{args.out_path}/../../../../training_args",
  per_device_train_batch_size=args.batch_size,
  eval_strategy="steps",
  num_train_epochs=args.epochs,
  save_steps=50,
  eval_steps=50,
  logging_steps=10,
  learning_rate=args.lr,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn, 
    compute_metrics=compute_metrics,
    train_dataset=dataset_processed["train"],
    eval_dataset=dataset_processed["test"],
    tokenizer=processor,
)

#### Training
train_results = trainer.train()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

trainer.save_model(f"{args.out_path}/model-saved")