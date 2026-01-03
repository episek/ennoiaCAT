import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

onnx_dir = "./tinyllama-onnx-nocache"
onnx_model_path = onnx_dir + "/model.onnx"  # adjust if name differs

print("Available providers:", ort.get_available_providers())

# Force CPU provider for now to avoid CUDA / cuDNN issues
session = ort.InferenceSession(
    onnx_model_path,
    providers=["CPUExecutionProvider"],
)

# Inspect required inputs
input_names = [i.name for i in session.get_inputs()]
print("Model inputs:", input_names)

tokenizer = AutoTokenizer.from_pretrained(onnx_dir)

prompt = "Explain 5G O-RAN fronthaul in one short paragraph."
encoded = tokenizer(prompt, return_tensors="np")

# Base inputs
input_ids = encoded["input_ids"].astype(np.int64)        # [1, seq_len]
seq_len = input_ids.shape[1]

# attention_mask: create if missing
if "attention_mask" in encoded:
    attention_mask = encoded["attention_mask"].astype(np.int64)
else:
    attention_mask = np.ones_like(input_ids, dtype=np.int64)

# position_ids: 0..seq_len-1
position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

max_new_tokens = 64

output_names = [o.name for o in session.get_outputs()]
logits_output_name = output_names[0]
print("Using ONNX output:", logits_output_name)

for _ in range(max_new_tokens):
    feed = {}

    if "input_ids" in input_names:
        feed["input_ids"] = input_ids

    if "attention_mask" in input_names:
        # ensure same length as input_ids
        if attention_mask.shape[1] != input_ids.shape[1]:
            pad_len = input_ids.shape[1] - attention_mask.shape[1]
            attention_mask = np.pad(
                attention_mask, ((0, 0), (0, pad_len)), constant_values=1
            )
        feed["attention_mask"] = attention_mask

    if "position_ids" in input_names:
        # ensure same length as input_ids
        if position_ids.shape[1] != input_ids.shape[1]:
            # extend position_ids sequentially
            last_pos = position_ids[0, -1]
            extra = np.arange(last_pos + 1, last_pos + 1 + (input_ids.shape[1] - position_ids.shape[1]),
                              dtype=np.int64).reshape(1, -1)
            position_ids = np.concatenate([position_ids, extra], axis=-1)
        feed["position_ids"] = position_ids

    outputs = session.run([logits_output_name], feed)
    logits = outputs[0]  # [1, seq_len, vocab_size]

    next_token_logits = logits[:, -1, :]
    next_token_id = np.argmax(next_token_logits, axis=-1).astype(np.int64)  # [1]

    # Append token to all sequences
    input_ids = np.concatenate([input_ids, next_token_id.reshape(1, 1)], axis=-1)

    # attention_mask: 1 for the new token
    attention_mask = np.concatenate(
        [attention_mask, np.ones((1, 1), dtype=np.int64)], axis=-1
    )

    # position_ids: increment last id by 1
    next_pos = position_ids[0, -1] + 1
    position_ids = np.concatenate(
        [position_ids, np.array([[next_pos]], dtype=np.int64)], axis=-1
    )

# Decode the final sequence
text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print("\n=== MODEL OUTPUT ===\n")
print(text)
