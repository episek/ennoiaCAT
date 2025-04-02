from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from pydantic import BaseModel
import json
import os
import csv
from typing import List
import subprocess

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def read_signal_strength(filename):
    # Lists to store the data
    frequencies = []
    strengths = []
    
    #generate batch CSV file
    subprocess.run(["python", "genCSV.py"])

    #generate max CSV file 
    subprocess.run(["python", "highest_strength.py"])

    # Read the CSV file
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                freq = int(float(row['Frequency']))  # Convert to int for cleaner numbers
                strength = float(row['Signal_Strength'])
                frequencies.append(freq)
                strengths.append(strength)
                
    except FileNotFoundError:
        print(f"Could not find file: {filename}")
        return
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")
        return
    
    # return the signal strength
    return  strengths, frequencies


# Initialize the FastAPI app once
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the Groq client (assuming API key is set in environment)
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))  # Set your API key in env or here directly
router = APIRouter()

# Define the data schema
class FrequencyReport(BaseModel):
    operator: str
    strength: float
    technology: str
    service: str
    frequency_range: str
    
class FrequencyInput(BaseModel):
    user_message: str

class FinalResult(BaseModel):
    signal_strengths: List[float]
    frequency: List[float]
    frequency_report: FrequencyReport    



@router.post("/prompt")
async def generate_report(request: FrequencyInput):  # Made async
    """
    Check compliance for the user message.
    """
    try:
        
        # Write code here 
        print("AID Frequency Analyzer")
        sstr, freq = read_signal_strength('max_signal_strengths.csv')
        freq_mhz = [x / 1e6 for x in freq]
        
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in frequency analysis of wireless signals. "
                       # ""
                       # "Your goal is to answer the users questions as CONCISE and DIRECT as possible. Always be proactive when you can, i.e. suggest next steps or offer to actually solve the problem. If need be, ask requisite follow up questions beforehand."
                        f"I will provide two arrays: `sstr` containing signal strengths and `freq_mhz` containing corresponding frequencies from a frequency scan. "
                        f"Here is the data: signal strengths = {sstr}, frequencies = {freq_mhz}. "
                        f"The `freq_mhz` array represents frequency values (in MHz), and the `sstr` array represents the signal strength (in dBm) for each corresponding frequency. "
                        "Your task is to analyze this data and provide structured insights based solely on the values in `sstr` and `freq_mhz`. "
                        f"Identify distinct frequency ranges in `freq_mhz` based on signal strengths"
                        f"Do not extrapolate frequency range beyond maximum value in `freq_mhz`"
                        "Make sure the frequency ranges don't use more than 2 decimal places"
                        "For each distinct frequency range identified, calculate the average signal strength from the corresponding `sstr` values. "
                        "For each distinct frequency range identified, identify potential names of services, names of network operators in Austin TX, and technologies that typically operate in each frequency range. "
                        f"Ensure that the response is in the following JSON format: {json.dumps(FrequencyReport.model_json_schema(), indent=2)}"

                    ),
                },
                {
                    "role": "user",
                    "content": "Analyze the provided signal strength and frequency data, and return the results in the specified JSON format.",
                },
            ],
            model="llama-3.3-70b-versatile",
            #model="whisper-large-v3-turbo",
            temperature=0,
            top_p=1,
            response_format={"type": "json_object"},
        )

        
        # Return the content from the completion
        # return json.loads(chat_completion.choices[0].message.content)  # Parse JSON string to dict
        return {
            "signal_strengths": sstr,
            "frequency": freq_mhz,
            "frequency_report": json.loads(chat_completion.choices[0].message.content)

        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Include the router in the app
app.include_router(router)

# Optional: Add a root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Server is running"}