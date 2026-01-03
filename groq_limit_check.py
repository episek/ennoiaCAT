from groq import Groq

# Initialize the Groq client with your API key
client = Groq(api_key="sk-proj-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")

# Make a sample API request (e.g., chat completion)
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    model="mixtral-8x7b-32768"
)

# Access the raw HTTP response headers
headers = response._response.headers

# Extract rate limit information from headers
rate_limit_limit = headers.get("x-ratelimit-limit-requests")  # Total request limit
rate_limit_remaining = headers.get("x-ratelimit-remaining-requests")  # Remaining requests
rate_limit_reset = headers.get("x-ratelimit-reset-requests")  # Time when limit resets (Unix timestamp)
token_limit = headers.get("x-ratelimit-limit-tokens")  # Total token limit
token_remaining = headers.get("x-ratelimit-remaining-tokens")  # Remaining tokens
token_reset = headers.get("x-ratelimit-reset-tokens")  # Time when token limit resets

# Print the rate limit details
print(f"Request Limit: {rate_limit_limit}")
print(f"Requests Remaining: {rate_limit_remaining}")
print(f"Request Limit Reset Time: {rate_limit_reset}")
print(f"Token Limit: {token_limit}")
print(f"Tokens Remaining: {token_remaining}")
print(f"Token Limit Reset Time: {token_reset}")
