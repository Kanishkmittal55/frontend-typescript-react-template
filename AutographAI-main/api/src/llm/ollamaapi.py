from typing import Callable, List, Dict, Any
from llm.basellm import BaseLLM
import requests
import logging
import tiktoken

class OllamaChat(BaseLLM):
    """Wrapper around Ollama's llama3.x large language model."""

    # Constructor
    def __init__(
        self,
        model_name: str = "llama3.1", # We are setting default values for the passable parameters
        max_tokens: int = 5000,
        temperature: float = 0.0,
        host: str = "http://localhost:7860/ollama/chat",
        headers: Dict[str, str] = None,
    ) -> None: # This indicates the return type of the method. Since constructors don't return anything explicitly (other than the newly created object), it is specified as None.
        """
        Initialize the Ollama model.
        :param model_name: The name of the model to use (default: llama3.2).
        :param max_tokens: Maximum number of tokens to generate.
        :param temperature: Sampling temperature for the model.
        :param host: The host where the Ollama server is running (default: localhost).
        :param headers: Optional headers for the HTTP client.
        """
        if headers is None:
            headers = {}  # Ensure headers is a dictionary
        # headers.setdefault('Content-Type', 'application/json')  # Add default header
        self.model = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.host = host  # Store the endpoint
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def generate(self, message:List[str]) -> str:
        """
        Generate a response from the model.
        :param messages: A list of strings representing the conversation history.
        :return: The generated response as a string.
        """
        try:
            # Prepare the request payload with only the chunk (prompt)
            payload = {
                "prompt": message[0]  # Assuming the API expects a field named "chunk" for the prompt
            }

            # Make the POST request to the Ollama endpoint
            logging.debug(f"Sending request to {self.host} with payload: {payload}")
            response = requests.post(self.host, json=payload)

            # Raise an exception for HTTP errors
            response.raise_for_status()

            # Parse and return the generated response
            result = response.json()
            return result.get("generated_text", "")
        except requests.RequestException as e:
            logging.error(f"Error communicating with Ollama: {e}")
            return f"Error: {e}"
        except ValueError as e:  # Handle JSON decoding errors
            logging.error(f"Invalid JSON received from Ollama: {response.text}")
            return f"Invalid JSON received: {response.text}"

    async def generateStreaming(
        self, messages: List[str], onTokenCallback: Callable[[str], None]
    ) -> List[str]:
        """
        Generate a response from the model in streaming mode.
        :param messages: A list of dictionaries representing the conversation history.
        :param onTokenCallback: A callback function to handle each token during streaming.
        :return: The complete response as a list of tokens.
        """
        try:
            # Convert list of strings to the format expected by the model
            formatted_messages = "\n".join(messages)

            print("Formatted messages : ", formatted_messages)
            stream = chat(
                model=self.model,
                messages=formatted_messages,
                options={
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                stream=True,
            )


            result = []
            for chunk in stream:
                content = chunk["message"]["content"]
                result.append(content)
                await onTokenCallback(content)
            return result
        
        except Exception as e:
            return [str(f"Error: {e}")]

    def num_tokens_from_string(self, string: str) -> int: 
        """
        Estimate the number of tokens in a string using the LLaMA tokenizer.
        :param string: The input string.
        :return: The number of tokens in the string.
        """

        # Use tiktoken's cl100k_base tokenizer to encode the input string
        tokens = self.tokenizer.encode(string)
        return len(tokens)

    def max_allowed_token_length(self) -> int:
        """
        Return the maximum number of tokens the model can handle.
        :return: The maximum allowed token length.
        """
        return 4096
    

# if __name__ == "__main__":
#     # Path to your tokenizer model (update this with your actual path)
#     tokenizer_model_path = "/path/to/llama/tokenizer/model_file"

#     # Initialize the OllamaChat instance
#     ollama_chat = OllamaChat(
#         model_name="llama3.2", 
#         max_tokens=1000, 
#         temperature=0.7, 
#         tokenizer_model_path=tokenizer_model_path
#     )

#     # Example input for synchronous generation
#     messages = [
#         "Why is the earth round?",
#         "Why is the sky blue?"
#     ]

#     # Generate response synchronously
#     print("\nSynchronous Response:")
#     response = ollama_chat.generate(messages)
#     print(response)

#     # Test token counting
#     print("\nToken Count Test:")
#     for msg in messages:
#         token_count = ollama_chat.num_tokens_from_string(msg)
#         print(f"Message: {msg}\nToken Count: {token_count}")

#     # Example input for streaming generation
#     async def handle_token(token: str):
#         print(token, end="", flush=True)  # Handle each streamed token

#     print("\n\nStreaming Response:")
#     import asyncio

#     # Run the asynchronous streaming generation
#     asyncio.run(ollama_chat.generateStreaming(messages, handle_token))

