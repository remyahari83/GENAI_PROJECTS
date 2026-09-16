import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load env variables from config/.env
env_path = os.path.join(os.path.dirname(__file__), "..", "config", ".env")
env_path = os.path.abspath(env_path)
load_dotenv(env_path)

# print for confirmation of .env
print(f"Loaded .env from: {env_path}")


def get_embedding_model(provider: str):
    provider = provider.lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        print("Loading OpenAI embedding model...")
        return OpenAIEmbeddings(api_key = api_key)
    elif provider in ["gemini", "google"]:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env")
        print("Loading Google embedding model...")
        return GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-2", google_api_key = api_key)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")