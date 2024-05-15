import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def chunked_generate_output(input_texts, model, tokenizer, max_new_tokens=1):
    # Tokenize input texts in batch with left padding
    batch = tokenizer(
        input_texts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    # Move input_ids and attention_mask to GPU
    input_ids = batch['input_ids'].to('cuda')
    attention_mask = batch['attention_mask'].to('cuda')

    # Generate output in batch
    with torch.no_grad():  # Disable gradient calculation for inference
        output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)

    # Decode the output tokens back to texts
    output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return output_texts

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_name_or_path> <input_file>")
        sys.exit(1)

    model_name_or_path = sys.argv[1]
    input_file = sys.argv[2]

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Configure tokenizer for left padding
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to('cuda')  # Move the model to GPU

    # Optimize by processing inputs in batches
    batch_size = 64  # Adjust this based on your GPU memory
    input_texts = []
    with open(input_file, 'r') as file:
        for line in file:
            input_texts.append(line.strip())
            if len(input_texts) == batch_size:
                output_texts = chunked_generate_output(input_texts, model, tokenizer)
                for output_text in output_texts:
                    print(output_text)
                input_texts = []

        # Process the last batch if it's not empty
        if input_texts:
            output_texts = chunked_generate_output(input_texts, model, tokenizer)
            for output_text in output_texts:
                print(output_text)
