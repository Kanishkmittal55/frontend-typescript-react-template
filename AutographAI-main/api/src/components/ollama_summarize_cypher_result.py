import httpx
from typing import Any, Awaitable, Callable, Dict, List

system = f"""
You are an assistant that helps to generate text to form nice and human understandable answers based on some contextual information.
The latest prompt contains the information, and you need to generate a human readable response based on the given information.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
Do not add any additional information that is not explicitly provided in the latest prompt.
I repeat, do not add any information that is not explicitly given.
Make the answer as concise as possible and do not use more than 500 words.
"""

OLLAMA_URI = "http://host.docker.internal:11434/api/generate"  # Replace with your Docker URI for Ollama

def remove_large_lists(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    The idea is to remove all properties that have large lists (embeddings) or text as values
    """
    LIST_CUTOFF = 56
    CHARACTER_CUTOFF = 5000
    for key, value in d.items():
        if isinstance(value, list) and len(value) > LIST_CUTOFF:
            d[key] = None
        if isinstance(value, str) and len(value) > CHARACTER_CUTOFF:
            d[key] = d[key][:CHARACTER_CUTOFF]
        elif isinstance(value, dict):
            remove_large_lists(d[key])
    return d


class OllamaSummarizeCypherResult:
    def __init__(self, exclude_embeddings: bool = True) -> None:
        self.exclude_embeddings = exclude_embeddings

    def generate_user_prompt(self, question: str, results: List[Dict[str, str]]) -> str:
        """
        Generate a prompt for Ollama using the question and results.
        """
        cleaned_results = (
            [remove_large_lists(el) for el in results] if self.exclude_embeddings else results
        )
        return f"""
        The question was: {question}
        Answer the question by using the following results:
        {cleaned_results}
        """

    async def run_async(
        self,
        question: str,
        results: List[Dict[str, Any]],
        callback: Callable[[str], Awaitable[Any]] = None,
    ) -> str:
        """
        Sends the prompt to Ollama's endpoint and returns the streaming response.
        """
        print("The db context and question asked again")
        async with httpx.AsyncClient() as client:
            payload = {
                "model": "auto",  # Specify the model
                "system_message": system,
                "prompt":self.generate_user_prompt(question, results),
            }

            async with client.stream("POST", "http://localhost:7860/ollama/chat", json=payload) as response:
                if response.status_code != 200:
                    raise ValueError(f"Error from Ollama: {await response.aread()}")

                output = []
                async for line in response.aiter_lines():
                    if callback:
                        await callback(line)
                    output.append(line)

                return "".join(output)

    def run(
        self,
        question: str,
        results: List[Dict[str, Any]],
    ) -> str:
        """
        Sends the prompt to Ollama's endpoint and returns the complete response.
        """
        with httpx.Client() as client:
            payload = {
                "model": "llama3.2",  # Specify the model
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": self.generate_user_prompt(question, results)},
                ],
                "stream": False,
            }

            response = client.post(OLLAMA_URI, json=payload)
            if response.status_code != 200:
                raise ValueError(f"Error from Ollama: {response.text}")

            return response.json().get("generated_text", "")

