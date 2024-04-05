import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_scheduler
from tqdm import tqdm
from wandb.keras import WandbCallback

# Load DNA sequence data (FASTA files)
df = pd.read_csv('path/to/your/data.fa', sep='>', names=['sequence'])
df = df[~df.sequence.str.contains(">", case=False)] # Remove header lines
dna_sequences = df.sequence.tolist()

# Split data into training and validation sets
train_seq, val_seq, train_labels, val_labels = train_test_split(dna_sequences, dna_sequences, test_size=0.2)

# Load pretrained model and tokenizer
model_name = "path/to/your/pretrained/model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Preprocess data and create training dataset
def encode_sequences(sequences):
    return tokenizer.batch_encode_plus(
        sequences,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='tf'
    )

train_encodings = encode_sequences(train_seq)
val_encodings = encode_sequences(val_seq)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',  # Path to save the model and other outputs
    overwrite_output_dir=True,
    num_train_epochs=10,  # Total number of training epochs
    per_device_train_batch_size=4,  # Batch size (per device) for training
    per_device_eval_batch_size=4,  # Batch size (per device) for evaluation
    warmup_steps=500,  # Number of steps for the warmup phase
    weight_decay=0.01,  # Strength of weight decay
    logging_dir='./logs',  # Path to save logs and checkpoints
    logging_steps=10,  # Log every X updates
    load_best_model_at_end=True,  # Load the best checkpoint at the end of training
    metric_for_best_model='accuracy',  # Metric to use for model selection
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=val_encodings,
    compute_metrics=compute_metrics,
    callbacks=[WandbCallback()],  # WandB integration for logging metrics and checkpoints
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./results/')
tokenizer.save_pretrained('./results/')
