##test-infer_arg1_2.py 

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import sys

# Define the model name (e.g., 'gpt2')
model_name = sys.argv[1]

# Load the model and tokenizer from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

# Define the input text
input_text = sys.argv[2:]

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors='pt').to('cuda')

# Generate next tokens
next_token = model.generate(**inputs, max_length=1000).tolist()

# Print the generated text
print("Input: ", input_text)
print("Generated Text: ", tokenizer.decode(next_token[0], skip_special_tokens=False))
