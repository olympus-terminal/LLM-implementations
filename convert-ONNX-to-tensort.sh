
## pip install tensorrt onnxruntime-gpu
```

# Convert ONNX model to TensorRT plan
onnx_path = "path/to/your/onnx-model"
tensorrt_path = "path/to/your/finetuned/model.trt"
onnx2trt -o $tensorrt_path --workspace=1073741824 $onnx_path

