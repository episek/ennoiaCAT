import json
import torch
import re
from transformers import pipeline


class MapAPI:
    def __init__(self, model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", tokenizer={}, max_new_tokens=512, temperature=0.7, do_sample=True):
        """
        Initializes the MapAPI class and creates a text generation pipeline.

        Args:
            model: The language model to use.
            tokenizer: The tokenizer corresponding to the model.
            max_new_tokens (int): Max number of tokens to generate.
            temperature (float): Sampling temperature.
            do_sample (bool): Whether to sample or use greedy decoding.
        """
        if tokenizer:
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
            self.model = model
            self.tokenizer = tokenizer
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_system_prompt(self, original_dict, user_input):

        system_prompt = (
            
        "You are a helpful assistant that only returns valid JSON dictionaries.\n"
        "Your task is to update an existing JSON dictionary based on user input.\n"
        "Follow these strict rules:\n"
        "- Do NOT add or remove any keys.\n"
        "- Only modify the values of existing keys if the user's input is relevant.\n"
        "- If the user's input does not imply any valid update, return the original dictionary unchanged.\n"
        "- Your response must be valid JSON with the same schema and keys — no text, explanations, or formatting outside the JSON block.\n\n"
        f" Here is the dictionary:\n{json.dumps(original_dict, indent=2)}\n\n"
        f"The user's input is:\n\"\"\"{user_input}\"\"\"\n\n"
        "Respond ONLY with the updated dictionary as valid JSON. Nothing else.."
    
        )
        
        return system_prompt

    
    def get_few_shot_examples(self):
        # === Few-Shot Examples ===
        few_shot_examples = [
            {
                "role": "user",
                "content": "Set the start frequency to 300 MHz"
            },
            {
                "role": "assistant",
                "content": (
                     "{'plot': True, 'scan': False, 'start': 300000000.0, 'stop': 900000000.0, 'points': 101, 'port': None, 'device': None, 'verbose': False, 'capture': None, 'command': None, 'save': None}"
                )
            },
            {
                "role": "user",
                "content": "Set the stop frequency to 850 MHz"
            },
            {
                "role": "assistant",
                 "content": (
                     "{'plot': True, 'scan': False, 'start': 300000000.0, 'stop': 850000000.0, 'points': 101, 'port': None, 'device': None, 'verbose': False, 'capture': None, 'command': None, 'save': None}"
                )
            },
            {
                "role": "user",
                "content": "Set the start frequency to 249 MHz and stop frequency to 366 MHz"
            },
            {
                "role": "assistant",
                 "content": (
                     "{'plot': True, 'scan': False, 'start': 249000000.0, 'stop': 366000000.0, 'points': 101, 'port': None, 'device': None, 'verbose': False, 'capture': None, 'command': None, 'save': None}"
                )
            }, 
            {
                "role": "user",
                "content": "How to configure tinySA"
            },
            {
                "role": "assistant",
                 "content": (
                     "{'plot': True, 'scan': False, 'start': 300000000.0, 'stop': 900000000.0, 'points': 101, 'port': None, 'device': None, 'verbose': False, 'capture': None, 'command': None, 'save': None}"
                )
            }, 
        ]
        return few_shot_examples      


    def update_dict_from_user_input(self,user_input, original_dict):
        """
        Uses an LLM to update a dictionary based on natural language user input.

        Args:
            user_input (str): Natural language instruction from the user.
            original_dict (dict): The dictionary to be updated.
            pipe (callable): Text-generation pipeline.

        Returns:
            dict: Updated dictionary in same structure as original_dict.
        """

        # Format the prompt using LLaMA 2-style chat template
        prompt = f"""<|system|>
        You are a helpful assistant. Only return valid JSON that updates the dictionary based on the user's input.
        You are not allowed to add or remove any keys from the dictionary. Only update the values of existing keys.
        <|user|>
        You are given this dictionary:
        {json.dumps(original_dict, indent=2)}

        Update ONLY the relevant dictionary values based on the user's input below:
        {user_input}

        Return ONLY the updated dictionary in valid JSON with the same schema and keys.
        <|assistant|>""".strip()


        # Run the model
        response = ""
        for chunk in self.pipe(prompt, max_new_tokens=200, temperature=0.7, return_full_text=False):
            response += chunk["generated_text"].strip()

        print("Raw LLM response:\n", response)

        # Clean up common code block wrappers (e.g. ```json ... ```)
        cleaned = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.IGNORECASE).strip()

        try:
            # Parse and validate
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

 
    
    def get_defaults_opts(self):
         # Define default options dictionary
        opts = {
            "plot": True,
            "scan": True,
            "start": 300000000.0,
            "stop": 900000000.0,
            "points": 101,
            "port": False,
            "device": None,
            "verbose": False,
            "capture": False,
            "command": None,
            "save": None
        }
        return opts
        
    def generate_response(self, chat):
        """
        Generate a response from a chat history using the model and tokenizer.

        Args:
            chat (list): A list of chat messages formatted for tokenizer's chat template.

        Returns:
            str: The generated assistant response.
        """
        # === Prepare Prompt ===
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        print("Prompt for LLM:\n", prompt)
        # === Tokenize Prompt ===
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # === Generate Output ===
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # === Decode Output ===
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # === Extract Assistant Reply ===
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        else:
            response = response.strip()

        print("\nEnnoia:\n" + response)
        return response


    
    def query_local_llm(self, prompt):
        full_prompt = f"User: {prompt}\nAssistant:"
        response = ""

        for chunk in self.pipe(full_prompt, max_new_tokens=200, temperature=0.7, return_full_text=False):
            chunk_text = chunk["generated_text"]
            response_part = chunk_text.replace(full_prompt, "").strip()
            yield response_part
            response += response_part

        return response.strip()

    def parse_user_input(self, user_input, opts):
        """
        Parses user input using an LLM and updates the opts dictionary dynamically.
        """
        prompt_S = f"""
        Given the following dictionary: {opts}, modify its values based on the user's input below.

        User Input: "{user_input}"

        Instructions:
        - Extract relevant key-value pairs from the user input.
        - Update one or more values within {opts} based on the extracted information.
        - Maintain the dictionary format and ensure the updated version follows the same structure.
        - If no relevant changes are found, return the original dictionary unchanged.
        - Respond in same **JSON format** as {opts} without additional explanations.
        """

        try:
            response_S = ""
            for part in self.query_local_llm(prompt_S):
                response_S += part

            extracted_data = response_S.strip()
            updated_opts = json.loads(extracted_data)

            if isinstance(updated_opts, dict):
                print(f"Updated dictionary: {updated_opts}")
                return updated_opts
            else:
                print("The LLM response does not contain a valid dictionary.")
                return opts

        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            print(f"Original LLM response: {response_S}")
            return opts

        except Exception as e:
            print(f"Error parsing input: {e}")
            return opts