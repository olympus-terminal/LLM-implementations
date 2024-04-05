### To create a production-ready LLM using the outputs from model fine-tuning results

### 1. Save the pretrained model and tokenizer using `AutoModelForSequenceClassification` and `AutoTokenizer` classes:

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("path/to/your/finetuned/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/your/finetuned/model")


### Save the model and tokenizer in a format that can be easily loaded by other applications, such as ONNX or TensorRT formats^[f:

###For ONNX format:

from transformers import TFTrainer, TFTrainingArguments

# Load training arguments from trainer_state.json
training_args = TrainingArguments.from_pretrained("path/to/your/finetuned/model")
trainer = TFTrainer(
    model=model,
    args=training_^[fargs,
    train_dataset=train_encodings,
    eval_dataset=val_encodings,
    compute_metrics=compute_metrics,
)

# Export the model to ONNX format
input_example = tokenizer("This is an example sentence.", return_tensors="tf")
trainer.model.save_pretrained("./results/onnx-model", from_pt=True)
converter = TF2Onnx()
model_inputs = trainer.model.inputs[0].shape
input_signature = [Input(name='input_ids', shape=[None, 512], dtype=TensorProto.INT32)]
onnx_path = "path/to/your/onnx-model"
converter.convert(tf_session=model, inputs=[input_example], output_path=onnx_path, input_signature=input_signature)
