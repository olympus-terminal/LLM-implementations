import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW

# Custom dataset class for FASTA sequences
class FASTADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.sequences = []
        with open(file_path, 'r') as file:
            sequence = ""
            for line in file:
                if line.startswith('>'):
                    if sequence:
                        self.sequences.append(sequence)
                        sequence = ""
                else:
                    sequence += line.strip()
            if sequence:
                self.sequences.append(sequence)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequence = self.sequences[index]
        encoded = self.tokenizer.encode(sequence, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
        return torch.tensor(encoded)

# Hyperparameters
epoch_num = 10
batch_size = 4
max_length = 512
learning_rate = 1e-4

# Load FASTA file
fasta_file = 'path/to/your/fasta/file.fasta'

# Load pre-trained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add special tokens for FASTA sequences
special_tokens = {'pad_token': '<PAD>', 'unk_token': '<UNK>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
num_added_tokens = tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# Prepare dataset and dataloader
dataset = FASTADataset(fasta_file, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Fine-tuning loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

for epoch in range(epoch_num):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch.to(device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{epoch_num} | Loss: {loss.item():.4f}')

# Save the fine-tuned model
model.save_pretrained('path/to/save/fine-tuned/model')
