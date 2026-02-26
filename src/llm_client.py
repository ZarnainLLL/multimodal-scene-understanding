# Sends structured scene data to Gemini and returns a natural language summary.

import os
from google import genai
from src.prompt_builder import build_prompt


class LLMClient:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not found in .env")
        self.client = genai.Client(api_key=api_key)

    def generate_summary(self, scene_data):
        system_prompt, user_prompt = build_prompt(scene_data)
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{system_prompt}\n\n{user_prompt}",
            )
            return response.text.strip()
        except Exception as e:
            return f"[Gemini Error] {e}"