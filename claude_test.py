import os
from anthropic import Anthropic

# Read your API key from environment variable
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("Please set ANTHROPIC_API_KEY environment variable first.")

client = Anthropic(api_key=api_key)

response = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=200,
    messages=[
        {"role": "user", "content": "Say hello from my Streamlit Conda environment"}
    ],
)

print(response.content[0].text)
