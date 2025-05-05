# This script is used to interact with the LLaMA 2 model using the MapAPI and TinySAHelper classes.
# It initializes the model, sets up the system prompt and few-shot examples, and enters an interactive chat loop.
import json
import ast
from types import SimpleNamespace
from tinySA_config import TinySAHelper
from map_api import MapAPI

tokenizer, peft_model, device = TinySAHelper.load_lora_model()
helper = TinySAHelper()
map_api = MapAPI(peft_model, tokenizer)

system_prompt = helper.get_system_prompt()
print(f"System prompt: {system_prompt}")
few_shot_examples = helper.get_few_shot_examples()

def_dict = map_api.get_defaults_opts()
pipe = map_api.pipe

few_shot_examples2 = map_api.get_few_shot_examples()

# === Interactive Chat Loop ===
print("Ask Ennoia (type 'exit' to quit):")
while True:
    user_input = input("\n>> ").strip()
    if user_input.lower() == "exit":
        print("Exiting.")
        break

    if not user_input:
        continue  # Skip empty input and re-prompt

    # === Construct LLaMA 2 chat prompt ===
    chat1 = [{"role": "system", "content": system_prompt}] + few_shot_examples + [{"role": "user", "content": user_input}]
    response = map_api.generate_response(chat1)
    print(f"\nAssistant: {response}")
    system_prompt2 = map_api.get_system_prompt(def_dict,user_input)
    chat2 = [{"role": "system", "content": system_prompt2}] + few_shot_examples2 + [{"role": "user", "content": user_input}]
    api_str = map_api.generate_response(chat2)
    print(f"\nAPI response: {api_str}")
        # Parse response safely into a dictionary
    api_dict = def_dict
    try:
        parsed = json.loads(api_str)
        if isinstance(parsed, dict):
            api_dict = parsed
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(api_str)
            if isinstance(parsed, dict):
                api_dict = parsed
        except Exception:
            print("Warning: Failed to parse response as a valid dictionary. Using default options.")

    print(f"\nParsed API options:\n{api_dict}")
  