import json
from typing import Boolean

from fastapi import APIRouter, HTTPException
from groq import AsyncGroq, Groq
from pydantic import BaseModel

client = AsyncGroq()
router = APIRouter()


# example of the response we want. 
# [
#   {
#     "provider": "Local AM stations",
#     "strength": 0.34,
#     "technology": "Analog AM",
#     "service": "AM Radio Broadcasting",
#     "frequency_range": "1 MHz"
#   },
#   {
#     "provider": "Local FM stations (e.g., NPR, iHeart)",
#     "strength": -61.53,
#     "technology": "Analog FM, HD Radio",
#     "service": "FM Radio Broadcasting",
#     "frequency_range": "89–107 MHz"
#   },
#   {
#     "provider": "Government, Businesses",
#     "strength": -66.43,
#     "technology": "Analog FM, P25, DMR",
#     "service": "Land Mobile Radio (LMR)",
#     "frequency_range": "175–179 MHz"
#   },
#   {
#     "provider": "Businesses, Public Safety",
#     "strength": -77.07,
#     "technology": "Analog FM, DMR, P25",
#     "service": "Land Mobile Radio (LMR)",
#     "frequency_range": "451–475 MHz"
#   },
#   {
#     "provider": "T-Mobile",
#     "strength": -65.70,
#     "technology": "4G LTE, 5G NR",
#     "service": "Cellular (600 MHz, Band 71)",
#     "frequency_range": "505–645 MHz"
#   },
#   {
#     "provider": "Verizon",
#     "strength": -62.40,
#     "technology": "4G LTE, 5G NR",
#     "service": "Cellular (700 MHz, Band 13)",
#     "frequency_range": "729–756 MHz"
#   },
#   {
#     "provider": "AT&T (FirstNet)",
#     "strength": -74.73,
#     "technology": "4G LTE, 5G NR",
#     "service": "Cellular/Public Safety (700 MHz, Band 14)",
#     "frequency_range": "757–767 MHz"
#   },
#   {
#     "provider": "Verizon, AT&T",
#     "strength": -74.95,
#     "technology": "4G LTE",
#     "service": "Cellular (800 MHz, Band 5)",
#     "frequency_range": "851–894 MHz"
#   }
# ]

#define the data schemas you want here. 
class FrequencyReport(BaseModel):
    is_compliant: Boolean
    message: str


class ComplianceInput(BaseModel):
    user_message: str = ""


@router.post("/prompt")
def check_compliance(request: ComplianceInput):
    """
    Check compliance for the user message.
    """
    chat_completion = client.chat.completions.create(
        #
        # Required parameters
        #
        messages=[
            {
                "role": "system",
                "content": "You are an expert in frequency analysis. You will generate a report about the included data. "
                "Compliance Rules: 1. Don't share phone number. 2. Don't talk about about Collectron AI."
                f" The JSON object must use the schema: {json.dumps(FrequencyReport.model_json_schema(), indent=2)}",
            },
            {
                "role": "user",
                "content": request.user_message,
            },
        ],
        # The language model which will generate the completion.
        model="llama-3.3-70b-versatile",
        temperature=0,
        top_p=1,
        response_format={"type": "json_object"},
    )

    # Print the completion returned by the LLM.
    return chat_completion.choices[0].message.content
