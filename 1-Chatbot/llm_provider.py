import os
import google.generativeai as genai
from openai import OpenAI

class LLMProvider:
    def __init__(self, config: dict):
        self.provider = config["provider"]
        self.model = config["models"][self.provider]

        if self.provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY env variable was not set.")
            self.client = OpenAI(api_key=key)
        elif self.provider == "gemini":
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("GEMINI_API_KEY env variable was not set.")
            genai.configure(api_key=key)
            self.client = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        print(f"{self.provider.title()} client initiated successfully.✅")

    def chat(self, user_message:str) -> str:
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": user_message}]
                )
                return response.choices[0].message.content
            elif self.provider == "gemini":
                response = self.client.generate_content(user_message)
                return response.text
            
        except Exception as e:
            raise Exception(f"Error in {self.provider} chat: {str(e)}")