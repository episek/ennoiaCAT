from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
onnx_dir = "./tinyllama-onnx-nocache"

print("Exporting TinyLlama to ONNX (no KV cache)...")

# Export with use_cache=False to avoid past_key_values.* inputs
model = ORTModelForCausalLM.from_pretrained(
    model_id,
    export=True,
    use_cache=False,
    provider="CPUExecutionProvider",  # keep CPU for export
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(onnx_dir)
tokenizer.save_pretrained(onnx_dir)

print("Saved ONNX model (no cache) to", onnx_dir)
