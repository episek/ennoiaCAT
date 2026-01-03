import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class TinyLlamaONNX:
    def __init__(self, model_dir="./tinyllama-onnx-nocache", provider=None):
        self.model_dir = model_dir

        # Choose execution provider
        available = ort.get_available_providers()
        if provider is None:
            provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else "CPUExecutionProvider"

        self.session = ort.InferenceSession(
            f"{model_dir}/model.onnx",
            providers=[provider],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.logits_name = self.session.get_outputs()[0].name

    def generate(self, prompt, max_new_tokens=64):
        encoded = self.tokenizer(prompt, return_tensors="np")
        input_ids = encoded["input_ids"].astype(np.int64)

        # Prepare attention mask
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
        attention_mask = attention_mask.astype(np.int64)

        # Prepare position_ids
        seq_len = input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        # Token loop
        for _ in range(max_new_tokens):
            feed = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }

            logits = self.session.run([self.logits_name], feed)[0]
            next_token_id = np.argmax(logits[:, -1, :], axis=-1).astype(np.int64)

            # Append new token
            input_ids = np.concatenate([input_ids, next_token_id.reshape(1, 1)], axis=-1)
            attention_mask = np.concatenate([attention_mask, np.ones((1, 1), dtype=np.int64)], axis=-1)

            # Update position ids
            next_pos = position_ids[0, -1] + 1
            position_ids = np.concatenate([position_ids, np.array([[next_pos]], dtype=np.int64)], axis=-1)

        # Decode
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
