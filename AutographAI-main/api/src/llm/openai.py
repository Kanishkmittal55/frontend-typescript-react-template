from typing import (
    Callable,
    List,
)

import openai
import tiktoken
from llm.basellm import BaseLLM
from retry import retry


class OpenAIChat(BaseLLM):
    """Wrapper around OpenAI Chat large language models."""

    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> None:
        openai.api_key = openai_api_key
        self.model = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    @retry(tries=3, delay=1)
    def generate(
        self,
        messages: List[str],
    ) -> str:
        try:
            completions = openai.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=messages,
            )
            return completions.choices[0].message.content
        # catch context length / do not retry
        except openai.error.InvalidRequestError as e:
            return str(f"Error: {e}")
        # catch authorization errors / do not retry
        except openai.error.AuthenticationError as e:
            return "Error: The provided OpenAI API key is invalid"
        except Exception as e:
            print(f"Retrying LLM call {e}")
            raise Exception()

    async def generateStreaming(
        self,
        messages: List[str],
        onTokenCallback=Callable[[str], None],
    ) -> str:
        full_output = ""
        stream = openai.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
            stream=True,
        )
        print("The stream is : ", stream)
        for chunk in stream:
            delta = chunk.choices[0].delta 
            # If there's new text, append it
            content = getattr(delta, "content", None)
            if content:
                full_output += content
                # Invoke your websocket token handler
                await onTokenCallback(chunk)
        return full_output

    def num_tokens_from_string(self, string: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def max_allowed_token_length(self) -> int:
        # TODO: list all models and their max tokens from api
        return 2049
