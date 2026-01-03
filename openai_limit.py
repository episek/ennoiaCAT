import openai
import time
#from openai.error import RateLimitError

# try:
    # response = openai.ChatCompletion.create(
        # model="gpt-4o-mini",
        # messages=[{"role": "user", "content": "Hello!"}]
        
    # )
    # print(response['choices'][0]['message']['content'])
    # time.sleep(1)  # wait 1 second between requests
# except RateLimitError as e:
    # print("Rate limit exceeded. Check your billing or slow down requests.")

response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Summarize this:"}
    ],
    max_tokens=3000,  # Limit the response length
)
