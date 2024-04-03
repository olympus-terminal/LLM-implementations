from transformers import DataCollatorForLanguageModeling

# Tokenize the text data
def tokenize_function(examples):
    return tokenizer(examples["text"], return_tensors='pt', padding=True, truncation=True, max_length=512)

# Apply the tokenization function to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator will dynamically pad the batches so that all are the same length
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Update the TrainingArguments to include gradient accumulation if needed and set the remove_unused_columns to False
training_args = TrainingArguments(
    output_dir="./results-Apr2nd2024b",
    num_train_epochs=11,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,  # Adjust based on your GPU memory and batch size
    logging_dir='./logs-Apr2nd2024b-11e',
    logging_steps=10,
    learning_rate=2e-3,
    report_to="wandb",  # Integrate Weights & Biases for tracking
    remove_unused_columns=False  # Important to keep the dataset structure during training
)

# Initialize the trainer with the new data collator
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=tokenized_datasets,
    data_collator=data_collator
)

trainer.train()
