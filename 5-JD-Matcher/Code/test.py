import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("API key not found!")
else:
    client = OpenAI(api_key=api_key)

    try:
        response = client.models.list()
        print("OpenAI API connection successful!")
    except Exception as e:
        print("API connection failed:", e)