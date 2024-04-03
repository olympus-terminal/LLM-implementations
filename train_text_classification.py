import os
import zipfile
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

# Load dataset from a zip file and preprocess it
def load_and_preprocess_data(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as f:
        data_dir = f.namelist()[0]  # assume the first .txt file is in the root of the zip file
        os.makedirs('unzipped_data', exist_ok=True)
        with f.open(data_dir, 'r') as data:
            with open(os.path.join('unzipped_data', data_dir), 'w') as unzipped_file:
                unzipped_file.write(data.read().decode())

    # Load the preprocessed dataset
    dataset = load_dataset('text_classification', data_files='unzipped_data/')

    return dataset

# Preprocess the dataset for training and evaluation
def prepare_dataset(dataset):
    def encode_function(example):
        return tokenizer(example['text'], truncation=True)

    encoded_datasets = dataset.map(encode_function, batched=True)

    return encoded_datasets

# Define the model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load evaluation metric
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Define trainer and train the model
def train(dataset):
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.evaluate()

# Load and preprocess the dataset
zip_path = 'path/to/your/data.zip'  # replace with your zip file path
dataset = load_and_preprocess_data(zip_path)

# Preprocess the dataset for training and evaluation
dataset = prepare_dataset(dataset)

# Train the model
train(dataset)
