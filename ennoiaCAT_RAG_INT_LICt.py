
import ennoia_client_lic as lic 
import argparse

parser = argparse.ArgumentParser(description="Ennoia License Client")
parser.add_argument(
    "--action",
    choices=["activate", "verify"],
    default="verify",
    help="Action to perform (default: verify)"
)
parser.add_argument("--key", help="Ennoia License key for activation")
args = parser.parse_args()


if args.action == "activate":
    if not args.key:
        print("❗ Please provide a license key with --key")
    else:
        success = lic.request_license(args.key)
elif args.action == "verify":
    success = lic.verify_license_file()
else:
    success = lic.verify_license_file()
  
if not success:
    print("❌ License verification failed. Please check your license key or contact support.")
    exit()
    
import json
import ast
import streamlit as st
from tinySA_config import TinySAHelper
from map_api import MapAPI
from types import SimpleNamespace
import pandas as pd
from timer import Timer, timed, fmt_seconds

    # Define option descriptions for reference
options_descriptions = {
    "plot": "plot rectangular",
    "scan": "scan by script",
    "start": "start frequency",
    "stop": "stop frequency",
    "center": "center frequency",
    "span": "span",
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

if not success:
    st.error("Ennoia License verification failed. Please check your license key or contact support.")
    st.stop()
else:
    st.success("Ennoia License verified successfully.")

# --- App logic starts here ---
selected_options = TinySAHelper.select_checkboxes()
st.success(f"You selected: {', '.join(selected_options) if selected_options else 'nothing'}")


# --- Caching the model and tokenizer ---

if "SLM" in selected_options:
    @st.cache_resource
    def load_model_and_tokenizer():
        return TinySAHelper.load_lora_model()

    st.write("\n⏳ Working in OFFLINE mode. Loading local model... (might take a minute)")
    tokenizer, peft_model, device = load_model_and_tokenizer()
    st.write(f"Device set to use {device}")
    map_api = MapAPI(peft_model, tokenizer)
else:
    st.write("\n⏳ Working in ONLINE mode.")  
    client, ai_model = TinySAHelper.load_OpenAI_model()
    map_api = MapAPI() 

helper = TinySAHelper()
system_prompt = helper.get_system_prompt()
few_shot_examples = helper.get_few_shot_examples()



@st.cache_data
def get_default_options():
    return map_api.get_defaults_opts()

def_dict = get_default_options()

few_shot_examples2 = map_api.get_few_shot_examples()

# --- Get and cache the TinySA port ---
if "tinySA_port" not in st.session_state:
    st.session_state.tinySA_port = helper.getport()

if "SLM" in selected_options:
    st.write(f"\n✅ Local SLM model {peft_model.config.name_or_path} loaded & device found! Let's get to work.\n")
else:
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = ai_model
    st.write(f"\n✅ Online LLM model {ai_model} loaded & device! Let's get to work.\n")
# Initialize TinySA device

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
    t = Timer()
    t.start()

    # Store the user message in session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Construct the full prompt with context (system + user message)
        user_input = st.session_state.messages[-1]["content"]
        
        chat1 = [{"role": "system", "content": system_prompt}] + few_shot_examples + [{"role": "user", "content": user_input}]
        if "SLM" in selected_options:
            response = map_api.generate_response(chat1)
        else:
            openAImessage = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=chat1,
                temperature=0,
                max_tokens=200,
                frequency_penalty=1,
                stream=False
            )
            response = openAImessage.choices[0].message.content
        # Display the streamed response from the assistant
        st.markdown(response)
        
        # Save the assistant's response in session state
        st.session_state.messages.append({"role": "assistant", "content": response})


    system_prompt2 = map_api.get_system_prompt(def_dict,user_input)
    chat2 = [{"role": "system", "content": system_prompt2}] + few_shot_examples2 + [{"role": "user", "content": user_input}]
    if "SLM" in selected_options:
        api_str = map_api.generate_response(chat2)
    else:
        openAImessage = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=chat2,
                temperature=0,
                max_tokens=200,
                frequency_penalty=1,
                stream=False
            )
        api_str = openAImessage.choices[0].message.content
    #st.markdown(api_str)
    # Parse response safely into a dictionary
    def_dict["save"] = True
    print(f"\nSave output response:\n{def_dict}")
    api_dict = def_dict
    try:
        parsed = json.loads(api_str)
        if isinstance(parsed, dict):
            api_dict = parsed
            api_dict["save"] = True
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(api_str)
            if isinstance(parsed, dict):
                api_dict = parsed
                api_dict["save"] = True
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
        st.error("API response is not a valid dictionary. Setting default options.")
 
    try:
        result = helper.read_signal_strength('max_signal_strengths.csv')
        if not result:
            st.error("Could not read signal strength data.")

        sstr, freq = result
        freq_mhz = [x / 1e6 for x in freq]
        print(f"\nSignal strengths: {sstr}")
        print(f"\nFrequencies: {freq_mhz}")
        
        operator_table = helper.get_operator_frequencies()
        if not operator_table:
            st.error("Operator table could not be loaded.")

        frequency_report_out = helper.analyze_signal_peaks(sstr, freq_mhz, operator_table)
        print(f"\nFrequency report: {frequency_report_out}")
        if not frequency_report_out:
            st.write("No strong trained frequency band seen.")

    except Exception as e:
        st.error(f"Failed to process request: {str(e)}")
      
  
    # Convert to Pandas DataFrame
    df = pd.DataFrame(frequency_report_out)

    # Display as a table in Streamlit
    st.dataframe(df)  # Interactive table
    
    t.stop()
    #print("elapsed:", fmt_seconds(t.elapsed()))
    st.write(f"elapsed: {fmt_seconds(t.elapsed())}")
    t.reset()  # reset to zero