
import json
import ast
import streamlit as st
from tinySA_config import TinySAHelper
from map_api import MapAPI
from types import SimpleNamespace

    # Define option descriptions for reference
options_descriptions = {
    "plot": "plot rectangular",
    "scan": "scan by script",
    "start": "start frequency",
    "stop": "stop frequency",
    "points": "scan points",
    "port": "specify port number",
    "device": "define device node",
    "verbose": "enable verbose output",
    "capture": "capture current display to file",
    "command": "send raw command",
    "save": "write output to CSV file"
}

st.set_page_config(page_title="Ennoia Technologies", page_icon="🤖")
st.sidebar.image('ennoia.jpg')
st.title("Ennoia Technologies")
st.markdown(
    """ 
    Chat and Test with Ennoia Connect Platform ©. All rights reserved. 
    """
)

# STEP 4: Load Local Model
st.write("\n⏳ Working in OFFLINE mode. Loading local model... (might take a minute)")
# --- Caching the model and tokenizer ---
@st.cache_resource
def load_model_and_tokenizer():
    return TinySAHelper.load_lora_model()

tokenizer, peft_model, device = load_model_and_tokenizer()

st.write(f"Device set to use {device}")

helper = TinySAHelper()
system_prompt = helper.get_system_prompt()
few_shot_examples = helper.get_few_shot_examples()

map_api = MapAPI(peft_model, tokenizer)

@st.cache_data
def get_default_options():
    return map_api.get_defaults_opts()

def_dict = get_default_options()

few_shot_examples2 = map_api.get_few_shot_examples()


#os.environ["STREAMLIT_WATCH_FILES"] = "false"

# --- Get and cache the TinySA port ---
if "tinySA_port" not in st.session_state:
    st.session_state.tinySA_port = helper.getport()


st.write(f"\n✅ Local model {peft_model.config.name_or_path} loaded! Let's get to work.\n")


# Initialize TinySA device

st.write(f"Found TinySA device on: {st.session_state.tinySA_port}")
st.write("Continuing with the device...")

st.write("Hi. I am Ennoia, your AI assistant. How can I help you today?")

# Initialize session state

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



prompt = st.chat_input("Ask Ennoia:")

if prompt:
    # Store the user message in session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Construct the full prompt with context (system + user message)
        user_input = st.session_state.messages[-1]["content"]
        
        chat1 = [{"role": "system", "content": system_prompt}] + few_shot_examples + [{"role": "user", "content": user_input}]
        response = map_api.generate_response(chat1)

        # Display the streamed response from the assistant
        st.markdown(response)
        
        # Save the assistant's response in session state
        st.session_state.messages.append({"role": "assistant", "content": response})


    system_prompt2 = map_api.get_system_prompt(def_dict,user_input)
    chat2 = [{"role": "system", "content": system_prompt2}] + few_shot_examples2 + [{"role": "user", "content": user_input}]
    api_str = map_api.generate_response(chat2)
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

    # Ensure it's a dict before using SimpleNamespace
    if isinstance(api_dict, dict):
        opt = SimpleNamespace(**api_dict)
        print(f"opt = {opt}")
        gcf = helper.configure_tinySA(opt)
        st.pyplot(gcf)
    else:
        st.error("API response is not a valid dictionary.")
 

      