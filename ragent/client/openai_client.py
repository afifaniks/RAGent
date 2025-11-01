import re
from openai import OpenAI

class OpenAIClient:
    def __init__(self):
        self.client = OpenAI()

    def chat_completion(self, model_name: str, prompt: str, system_prompt: str, temperature: float):
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return {
            "message": {
                "content": response.choices[0].message.content
            }
        }