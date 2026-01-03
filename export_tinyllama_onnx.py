from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
onnx_dir = "./tinyllama-onnx"

print("Exporting TinyLlama to ONNX...")

# This downloads and exports in one go
model = ORTModelForCausalLM.from_pretrained(
    model_id,
    export=True,
    provider="CUDAExecutionProvider",  # we want GPU kernels
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(onnx_dir)
tokenizer.save_pretrained(onnx_dir)

print("Saved ONNX model to", onnx_dir)
