# map_api_vi.py
import json
import re
from typing import List, Dict

import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    HF_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    OPENAI_AVAILABLE = False


class MapAPI:
    """
    Unified MapAPI that supports:

    - backend = "openai"  → uses OpenAI Chat API
    - backend = "slm"     → uses local TinyLlama SLM (optionally injected LoRA model)

    The Viavi app only calls:
        map_api.generate_response(chat: list[dict]) -> str
    """

    def __init__(
        self,
        backend: str = "openai",
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        openai_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        injected_model=None,
        injected_tokenizer=None,
    ) -> None:
        """
        Args:
            backend: "openai" or "slm"
            model_name: HF model name (for SLM backend when not injected)
            openai_model: OpenAI chat model for OpenAI backend
            temperature: sampling temperature
            max_new_tokens: generation length
            injected_model: optional pretrained HF model (e.g. LoRA-merged TinyLlama)
            injected_tokenizer: optional tokenizer corresponding to injected_model
        """
        self.backend = backend
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.injected_model = injected_model
        self.injected_tokenizer = injected_tokenizer

        if backend == "slm":
            if injected_model is not None and injected_tokenizer is not None:
                # Use externally-provided LoRA model/tokenizer
                self.model = injected_model
                self.tokenizer = injected_tokenizer
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                # Fallback: load base HF model
                if not HF_AVAILABLE:
                    raise RuntimeError(
                        "Transformers not available for SLM backend. "
                        "Install `transformers` and `torch`."
                    )
                self._init_slm(model_name)

        elif backend == "openai":
            if not OPENAI_AVAILABLE:
                raise RuntimeError(
                    "OpenAI client not available. Install `openai` and set OPENAI_API_KEY."
                )
            self._init_openai(openai_model)
        else:
            raise ValueError(f"Unsupported backend '{backend}'. Use 'openai' or 'slm'.")

    # ------------------------------------------------------------------
    # BACKEND INITIALIZERS
    # ------------------------------------------------------------------
    def _init_slm(self, model_name: str) -> None:
        """Initialize local TinyLlama-style SLM (HF)."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.use_default_system_prompt = False

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map={"": self.device},
        )
        self.model.eval()

    def _init_openai(self, openai_model: str) -> None:
        """Initialize OpenAI Chat backend."""
        self.client = OpenAI()
        self.openai_model = openai_model

    # ------------------------------------------------------------------
    # OPTIONAL: SYSTEM PROMPT + FEW-SHOT (for dict updating)
    # ------------------------------------------------------------------
    def get_system_prompt(self, original_dict: dict, user_input: str) -> str:
        system_prompt = (
            "You are a helpful assistant that only returns valid JSON dictionaries.\n"
            "Your task is to update an existing JSON dictionary based on user input.\n"
            "Follow these strict rules:\n"
            "- Do NOT add or remove any keys.\n"
            "- Only modify the values of existing keys if the user's input is relevant.\n"
            "- If the user's input does not imply any valid update, return the original dictionary unchanged.\n"
            "- Your response must be valid JSON with the same schema and keys — no text, explanations, or formatting outside the JSON block.\n\n"
            f"Here is the dictionary:\n{json.dumps(original_dict, indent=2)}\n\n"
            f"The user's input is:\n\"\"\"{user_input}\"\"\"\n\n"
            "Respond ONLY with the updated dictionary as valid JSON. Nothing else."
        )
        return system_prompt

    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        few_shot_examples = [
            {
                "role": "user",
                "content": "Set the start frequency to 300 MHz",
            },
            {
                "role": "assistant",
                "content": (
                    "{'plot': True, 'scan': False, 'start': 300000000.0, "
                    "'stop': 900000000.0, 'points': 101, 'port': None, 'device': None, "
                    "'verbose': False, 'capture': None, 'command': None, 'save': None}"
                ),
            },
        ]
        return few_shot_examples

    # ------------------------------------------------------------------
    # CORE CHAT → TEXT (USED BY VIAVI APP)
    # ------------------------------------------------------------------
    def generate_response(self, chat: List[Dict[str, str]]) -> str:
        """
        Main entry point used by the Viavi app.

        Args:
            chat: list of {"role": "system" | "user" | "assistant", "content": str}

        Returns:
            Assistant text response as string.
        """
        if self.backend == "openai":
            completion = self.client.chat.completions.create(
                model=self.openai_model,
                messages=chat,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            reply = completion.choices[0].message.content or ""
            return reply.strip()

        elif self.backend == "slm":
            tokenizer = self.injected_tokenizer or self.tokenizer
            model = self.injected_model or self.model

            prompt = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                )

            text = tokenizer.decode(output[0], skip_special_tokens=True)
            if "<|assistant|>" in text:
                text = text.split("<|assistant|>")[-1].strip()
            return text.strip()

        else:
            raise RuntimeError(f"Unknown backend '{self.backend}'")

    # ------------------------------------------------------------------
    # DICT UPDATE USING LLM (uses generate_response internally)
    # ------------------------------------------------------------------
    def update_dict_from_user_input(self, user_input: str, original_dict: dict) -> dict:
        """
        Uses LLM (OpenAI or SLM) to update a dictionary based on user input.
        """
        system_prompt = self.get_system_prompt(original_dict, user_input)
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Update the dictionary and return JSON only."},
        ]

        response = self.generate_response(chat)
        print("Raw LLM response for dict update:\n", response)

        cleaned = re.sub(
            r"^```(?:json)?|```$", "", response.strip(), flags=re.IGNORECASE
        ).strip()

        try:
            candidate_dict = json.loads(cleaned)
            if isinstance(candidate_dict, dict):
                updated_dict = {**original_dict, **candidate_dict}
                return updated_dict
            else:
                print("Extracted content is not a dictionary. Returning original.")
                return original_dict
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}. Returning original.")
            return original_dict

    def get_defaults_opts(self) -> dict:
        """
        Default options dictionary (kept for compatibility with old code).
        """
        opts = {
            "plot": True,
            "scan": False,
            "start": 300000000.0,
            "stop": 900000000.0,
            "points": 101,
            "port": None,
            "device": None,
            "verbose": False,
            "capture": None,
            "command": None,
            "save": None,
        }
        return opts

    # ------------------------------------------------------------------
    # OPTIONAL: LEGACY-LIKE PARSE LOGIC (if you want to reuse it)
    # ------------------------------------------------------------------
    def parse_user_input(self, user_input: str, opts: dict) -> dict:
        """
        Uses LLM to map free text → updated options dict.
        """
        prompt_s = (
            f"Given the following dictionary: {opts}, modify its values based on "
            f"the user's input below.\n\n"
            f"User Input: \"{user_input}\"\n\n"
            "Instructions:\n"
            "- Extract relevant key-value pairs from the user input.\n"
            "- Update one or more values within the dictionary based on the extracted information.\n"
            "- Maintain the dictionary format and ensure the updated version follows the same structure.\n"
            "- If no relevant changes are found, return the original dictionary unchanged.\n"
            "- Respond ONLY in JSON with the same keys as the original dictionary."
        )

        chat = [
            {"role": "system", "content": "You update JSON dictionaries only."},
            {"role": "user", "content": prompt_s},
        ]

        response = self.generate_response(chat)
        cleaned = re.sub(
            r"^```(?:json)?|```$", "", response.strip(), flags=re.IGNORECASE
        ).strip()

        try:
            updated = json.loads(cleaned)
            if isinstance(updated, dict):
                print(f"Updated dictionary: {updated}")
                return updated
            else:
                print("The LLM response does not contain a valid dictionary.")
                return opts
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            print(f"Original LLM response: {response}")
            return opts
        except Exception as e:
            print(f"Error parsing input: {e}")
            return opts
