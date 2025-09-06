import re
import logging
import tiktoken
from typing import Callable, List, Dict, Any
from utils.unstructured_data_utils import (
    nodesTextToListOfDict,
    relationshipTextToListOfDict,
)
import httpx
import psutil
from typing import Optional
import json
import csv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def generate_system_message() -> str:
    return """
You are a data scientist working for a company that is building a graph database. Your task is to extract information from data and convert it into a graph database.
Provide a set of Nodes in the form [ENTITY_ID, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES].
It is important that the ENTITY_ID_1 and ENTITY_ID_2 exists as nodes with a matching ENTITY_ID. If you can't pair a relationship with a pair of nodes don't add it.
When you find a node or relationship you want to add try to create a generic TYPE for it that  describes the entity you can also think of it as a label.

Example:
Data: Alice lawyer and is 25 years old and Bob is her roommate since 2001. Bob works as a journalist. Alice owns a the webpage www.alice.com and Bob owns the webpage www.bob.com.
Nodes: ["alice", "Person", {"age": 25, "occupation": "lawyer", "name":"Alice"}], ["bob", "Person", {"occupation": "journalist", "name": "Bob"}], ["alice.com", "Webpage", {"url": "www.alice.com"}], ["bob.com", "Webpage", {"url": "www.bob.com"}]
Relationships: ["alice", "roommate", "bob", {"start": 2021}], ["alice", "owns", "alice.com", {}], ["bob", "owns", "bob.com", {}]
"""


def num_tokens_from_string(string: str) -> int: 
    """
    Estimate the number of tokens in a string using the LLaMA tokenizer.
    :param string: The input string.
    :return: The number of tokens in the string.
    """

    # Use tiktoken's cl100k_base tokenizer to encode the input string
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(string)
    return len(tokens)

def max_allowed_token_length() -> int:
    """
    Return the maximum number of tokens the model can handle.
    :return: The maximum allowed token length.
    """
    return 1024


def splitString(string, max_length) -> List[str]:
    return [string[i : i + max_length] for i in range(0, len(string), max_length)]


def splitStringToFitTokenSpace(string: str, token_use_per_string: int) -> List[str]:
    allowed_tokens = max_allowed_token_length() - token_use_per_string
    chunked_data = splitString(string, 500)  # Split based on approximate length
    # print("The length of the splitted text array is: ", len(chunked_data))

    # print(chunked_data[0])

    combined_chunks = []
    current_chunk = ""
    
    for chunk in chunked_data:
        # Calculate token count for the combined chunk
        current_chunk_tokens = num_tokens_from_string(current_chunk)
        # print("Length of the current chunk of 500 words based on regex is in tokens is :", current_chunk_tokens)
        chunk_tokens = num_tokens_from_string(chunk)
        # print("Length of the chunk token of 500 words based on regex is in tokens is :", current_chunk_tokens)

        if current_chunk_tokens + chunk_tokens <= allowed_tokens:
            current_chunk += chunk
        else:
            # Append the non-empty current chunk and start a new one
            if current_chunk.strip():
                combined_chunks.append(current_chunk.strip())
            current_chunk = chunk
    
    # Append any remaining chunk after the loop
    if current_chunk.strip():
        combined_chunks.append(current_chunk.strip())

    return combined_chunks



async def old_process(chunk: str) -> str:
    """
    Process a single chunk by making an asynchronous request to the Ollama endpoint.
    :param chunk: The chunk to process.
    :return: The processed response as a string.
    """
    try:
        # Prepare the request payload with only the chunk (prompt)
        payload = {
            "prompt": chunk  # Assuming the API expects a field named "chunk" for the prompt
        }

        # Log the payload
        logging.debug(f"Sending request to http://localhost:7860/ollama/chat with payload: {payload}")

        # Use httpx for asynchronous requests
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:7860/ollama/chat", json=payload)

        # Raise an exception for HTTP errors
        response.raise_for_status()

        # Parse and return the generated response
        # Access the 'message' field and then 'content
        result = response.json()

        print("The response is :",result)        
        # From generated text
        message = result.get('generated_text', {}).get('message', {})

        # Extract the content
        content = message.get('content', None)

        # print(content)
        
        return content
    except httpx.RequestError as e:
        logging.error(f"Error communicating with Ollama: {e}")
        return f"Error: {e}"
    except ValueError as e:  # Handle JSON decoding errors
        logging.error(f"Invalid JSON received from Ollama: {response.text}")
        return f"Invalid JSON received: {response.text}"
        

def generate_prompt(data) -> str:
    return f"""
Data: {data}"""


def clean_llm_response(raw_response: str) -> str:
    """
    Cleans the LLM response to remove any notes or extra comments,
    and extracts only the Nodes and Relationships section.
    Handles bullet points and additional formatting issues.
    
    :param raw_response: The raw response from the LLM.
    :return: Cleaned string containing only nodes and relationships.
    """
    # Use regex to extract the **Nodes** and **Relationships** sections
    match = re.search(r"Nodes:\s*(.*?)Relationships:\s*(.*)", raw_response, re.S)
    if match:
        nodes = re.sub(r"^\s*-\s*", "", match.group(1).strip(), flags=re.M)  # Remove bullet points from Nodes
        relationships = re.sub(r"^\s*-\s*", "", match.group(2).strip(), flags=re.M)  # Remove bullet points from Relationships

        # Remove any trailing note or comments
        relationships = re.split(r"(?i)\n(?:Note|Let me know)", relationships, maxsplit=1)[0].strip()

        # Reconstruct cleaned response in the expected format
        cleaned_response = f"Nodes: {nodes}\nRelationships: {relationships}"
        return cleaned_response
    else:
        # Return empty structure if no match is found
        return "Nodes: []\nRelationships: []"




def getNodesAndRelationshipsFromResult(result):
    regex = "Nodes:\s+(.*?)\s?\s?Relationships:\s?\s?(.*)"
    internalRegex = "\[(.*?)\]"
    nodes = []
    relationships = []

    print("Starting to parse :" , result, len(result))
    
    for row in result:
        print(f"Processing row: {row}")
        
        # Match using regex
        parsing = re.match(regex, row, flags=re.S)
        if parsing is None:
            print(f"No match found for row: {row}")
            continue
        
        print("Regex match found. Extracting nodes and relationships.")
        
        # Extract raw nodes and relationships
        rawNodes = str(parsing.group(1))
        rawRelationships = parsing.group(2)
        
        print(f"Extracted rawNodes: {rawNodes}")
        print(f"Extracted rawRelationships: {rawRelationships}")
        
        # Find individual elements using internal regex
        nodes.extend(re.findall(internalRegex, rawNodes))
        relationships.extend(re.findall(internalRegex, rawRelationships))
        
        print(f"Current list of nodes: {nodes}")
        print(f"Current list of relationships: {relationships}")

    # Validate extracted nodes and relationships before inserting them
    nodes = [node for node in nodes if len(node.split(",")) >= 2]  # Ensure each node has at least name and label
    relationships = [rel for rel in relationships if len(rel.split(",")) >= 3]  # Ensure valid relationship structure
    
    if not nodes:
        print("Warning: No valid nodes found.")
    if not relationships:
        print("Warning: No valid relationships found.")

    # Convert raw text data to structured dictionaries
    print("Converting extracted nodes and relationships to structured dictionaries.")
    try:
        result = dict()
        result["nodes"] = nodesTextToListOfDict(nodes)
        result["relationships"] = relationshipTextToListOfDict(relationships)
    except Exception as e:
        logging.error(f"Error during conversion of nodes or relationships: {e}")
        raise
    
    print("Inside the function nodes are : ", result["nodes"])

    logging.info("Successfully parsed nodes and relationships.")
    return result


async def run_with_chunk_logging(data: str) -> List[str]:
        print("Process Started with the patent text")

        system_message = generate_system_message() # Currently 324 tokens are only used for the system message or the intruction of the task.
        prompt_string = generate_prompt(data)
        print("No. of tokens in the prompt input string :", num_tokens_from_string(prompt_string))
        print("No. of tokens in the system message string :", num_tokens_from_string(system_message) )
        
        token_usage_per_prompt = num_tokens_from_string(
            system_message + prompt_string
        )

        print("token usage per prompt :",  token_usage_per_prompt)

        # Once we get the total token to be used including the system prompt we split it to fit the token space.
        chunked_data = splitStringToFitTokenSpace(
            string=data, token_use_per_string=num_tokens_from_string(system_message)
        )

        print("So number of chunks created from the text are : ",len(chunked_data))

        results = []
        labels = set()
        chunks = []  # To store chunk metadata
        # print ( "Chunks have been created")

        print("Starting chunkwise processing")
        
        chunkResult = {} # getNodesAndRelationshipsFromResult returns this -> dict[str, list[dict[str, Any]]]

        for i,chunk in enumerate(chunked_data, start=1):
            print(f"Chunk number {i} chunk sent: {chunk}")
            
            # Log memory usage before processing the chunk
            memory_info = psutil.virtual_memory()
            print(f"Memory before processing chunk {i}: {memory_info.used / (1024**2):.2f} MB")

            processedChunk = await old_process(chunk)
            print(f"Chunk number {i} processedChunk : ", processedChunk)
            cleaned_response = clean_llm_response(processedChunk)
            print("The cleaned response is : ", cleaned_response)
            chunkResult = getNodesAndRelationshipsFromResult([cleaned_response])
            print("chunkResult- nodes and relationships : ", chunkResult.get("nodes", []))
            newLabels = [node["label"] for node in chunkResult["nodes"]]
            print("newLabels", newLabels)
            results.append(cleaned_response) # Why cleaned reponse ? Speculate
            labels.update(newLabels)
            # Append chunk metadata and result to chunks array
            chunks.append({
                "chunk_number": i,
                "system_prompt": system_message,
                "input_chunk_text": chunk,
                "chunk_result_nodes": chunkResult.get("nodes", []),  
                # Ensure no KeyError ,by providing a default empty key [] 
                # The empty brakcets serve as a default value in case the key "nodes" is missing from the dictionary.
                "chunk_result_relationships": chunkResult.get("relationships", [])
            })

            # Log memory usage after processing the chunk
            memory_info = psutil.virtual_memory()
            print(f"Memory after processing chunk {i}: {memory_info.used / (1024**2):.2f} MB")

        ## So No matter how many time one runs this "getNodesAndRelationshipsFromResult" function
        ## it will lead to further refinement of results.
        final_result = getNodesAndRelationshipsFromResult(results)
        return final_result, chunks

#####################################################################################
# The product report and discovery workflow
#####################################################################################

from llm.openai import OpenAIChat
import os
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")



# Now retrieve the API key as a string
api_key = openai_key
# Initialize LLM 
llm = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", max_tokens=4096
)

# Helper Functions - 
async def openai_generate(prompt: str) -> str:
    """
    Sends a prompt to the OpenAI endpoint and returns the raw output.

    :param prompt: The input prompt to send to the OpenAI API.
    :return: The raw output generated by the OpenAI API.
    """
    messages = [
        {"role": "user", "content": prompt}  # Construct the message payload
    ]
    print(f"Sending request to OpenAI endpoint with messages: {messages}")
    
    # Assuming `llm.generate()` is correctly configured to accept the messages
    output = llm.generate(messages)
    print("The output is:", output)
    return output

async def process(chunk: str, provider: str) -> str:
    """
    Process a single chunk by making an asynchronous request to the specified provider's endpoint.
    :param chunk: The chunk to process.
    :param provider: The provider name (e.g., "ollama", "openai", "groq").
    :return: The processed response as a string.
    """
    try:
        # Prepare the request payload with only the chunk (prompt)
        payload = {
            "prompt": chunk  # Assuming the API expects a field named "chunk" for the prompt
        }

        # Route to appropriate provider using a match-case structure
        if provider == "ollama":
            logging.debug(f"Sending request to http://localhost:7860/ollama/chat with payload: {payload}")
            async with httpx.AsyncClient() as client:
                response = await client.post("http://localhost:7860/ollama/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            message = result.get('generated_text', {}).get('message', {})
            content = message.get('content', None)

        elif provider == "openai":
            output = await openai_generate(chunk)
            return output

        elif provider == "groq":
            logging.debug(f"Sending request to Groq endpoint with payload: {payload}")
            async with httpx.AsyncClient() as client:
                response = await client.post("http://localhost:7860/groq/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            content = result.get('response', None)  # Adjust based on Groq's API response structure

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        return content

    except httpx.RequestError as e:
        logging.error(f"Error communicating with {provider}: {e}")
        return f"Error: {e}"
    except ValueError as e:
        logging.error(f"Invalid JSON received from {provider}: {e}")
        return f"Invalid JSON received: {e}"


# # Define the reprocess() function
# async def reprocess(text: str, provider: str) -> str:
#     """
#     Reprocess the extracted text to refine and condense the summary based on the provider.
#     Input: string of text, provider name
#     Output: a refined and shorter version of the input string
#     """
#     try:
#         # Format prompt to include previous context and the current chunk
#         prompt = f"""
#         ### Overlimit Information:
#         {text}

#         ### Instruction:
#         You are a data scientist working for a company that is building a report for a cosmetic patent document. Summarize the overlimited information in not more than 30 words, focusing only on product type, description. 
#         If no relevant information is found, respond: "No new information found." Only add the words that would be used at
# the last chunk to create one holistic product name and description from the complete information.
#         """

#         payload = {
#             "prompt": prompt
#         }

#         content = None

#         # Route to appropriate provider using a match-case structure
#         if provider == "ollama":
#             logging.debug(f"Sending request to http://localhost:7860/ollama/chat with payload: {payload}")
#             async with httpx.AsyncClient() as client:
#                 response = await client.post("http://localhost:7860/ollama/chat", json=payload)
#             response.raise_for_status()
#             result = response.json()
#             message = result.get('generated_text', {}).get('message', {})
#             content = message.get('content', None)

#         elif provider == "openai":
#             output = await openai_generate(prompt)
#             return output

#         elif provider == "groq":
#             logging.debug(f"Sending request to Groq endpoint with payload: {payload}")
#             async with httpx.AsyncClient() as client:
#                 response = await client.post("http://localhost:7860/groq/chat", json=payload)
#             response.raise_for_status()
#             result = response.json()
#             content = result.get('response', None)  # Adjust based on Groq's API response structure

#         else:
#             raise ValueError(f"Unsupported provider: {provider}")

#         return content

#     except httpx.RequestError as e:
#         logging.error(f"Error communicating with {provider}: {e}")
#         return f"Error: {e}"
#     except ValueError as e:
#         logging.error(f"Invalid JSON received from {provider}: {e}")
#         return f"Invalid JSON received: {e}"


def cleaned_name_and_description(response):
    """
    Extracts the product name and description from the response.
    
    :param response: The raw response string from the OpenAI API.
    :return: A dictionary containing the product name and description.
    """
    import re

    # Regular expressions to capture the name and description
    name_pattern = r"\*\*Product Type Name:\*\* (.+)"
    description_pattern = r"\*\*Description:\*\* (.+)"

    # Extract name and description
    name_match = re.search(name_pattern, response)
    description_match = re.search(description_pattern, response)

    # Get the matched content or default to None
    product_name = name_match.group(1).strip() if name_match else None
    product_description = description_match.group(1).strip() if description_match else None

    # Return as a dictionary
    return {
        "name": product_name,
        "description": product_description
    }


async def extract_name_description(extracted_info: str, provider: str) -> Optional[Dict[str, str]]:
    """
    Process the complete extracted information to finalize product name and description based on the provider.
    Input:
        extracted_info: string of extracted information
        provider: provider name (e.g., 'openai', 'groq', 'ollama')
    Output:
        A final refined string with product name and description
    """
    try:
        # Format the prompt
        prompt = f"""
        ### Extracted Information:
        {extracted_info}

        ### Instruction:
        You are a data scientist working for a company that is building a report for a cosmetic patent document. You are an author and have the capability to provide a very appropriate type of patent products based on gathered description. Provide an appropriate product type name describing very aptly what the product is about in a few words and a 2-sentence description of the product based on the extracted information, which represents a summary of the patent document.
        """

        # Prepare the request payload
        payload = {
            "prompt": prompt
        }

        logging.debug(f"Prepared payload: {payload}")

        # Route to appropriate provider using a match-case structure
        content = None

        if provider == "ollama":
            logging.debug(f"Sending request to http://localhost:7860/ollama/chat with payload: {payload}")
            async with httpx.AsyncClient() as client:
                response = await client.post("http://localhost:7860/ollama/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            message = result.get('generated_text', {}).get('message', {})
            content = message.get('content', None)

        elif provider == "openai":
            output = await openai_generate(prompt)
            cleaned_output = cleaned_name_and_description(output)
            return cleaned_output

        elif provider == "groq":
            logging.debug(f"Sending request to Groq endpoint with payload: {payload}")
            async with httpx.AsyncClient() as client:
                response = await client.post("http://localhost:7860/groq/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            content = result.get('response', None)  # Adjust based on Groq's API response structure

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if content:
            return content
        else:
            logging.error("No content received in the response.")
            return None

    except httpx.RequestError as e:
        logging.error(f"Error communicating with {provider}: {e}")
        return f"Error: {e}"

    except ValueError as e:  # Handle JSON decoding errors and invalid provider
        logging.error(f"ValueError: {e}")
        return f"ValueError: {e}"

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return f"Unexpected error: {e}"


def clean_functional_role_info(received_text: str) -> Optional[Dict]:
    """
    Transforms the received functional role information text into a JSON object.
    
    :param received_text: The raw string containing functional role information.
    :return: A JSON object with functional roles, or None if parsing fails.
    """
    try:
        # Attempt to parse the received text as JSON
        functional_roles_json = json.loads(received_text)

        # Validate that the required "functional_roles" key exists
        if "functional_roles" in functional_roles_json:
            return functional_roles_json  # Return the JSON object if valid

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")

    # Return None if parsing fails or required key is missing
    return None

async def final_composition_information(extracted_info: str, provider: str) -> Optional[Dict[str, str]]:
    """
    Process the complete extracted information to finalize product composition report based on the provider.
    Input:
        extracted_info: string of extracted information
        provider: provider name (e.g., 'openai', 'groq', 'ollama')
    Output:
        A JSON object with the product's functional roles, chemicals, and weights
    """
    output_format = """
     {
        "functional_roles": {
            "Emollient": [
            {"chemical": "Dimethicone", "weight": "15%"},
            {"chemical": "Caprylic Triglyceride", "weight": "10%"}
            ],
            "Humectant": [
            {"chemical": "Glycerin", "weight": "20%"}
            ],
            "Preservative": [
            {"chemical": "Phenoxyethanol", "weight": "1%"}
            ]
        }
        }
    """
    try:
        # Format the prompt
        prompt = f"""
        ### Extracted Information:
        {extracted_info}

        ### Instruction:
        You are a data scientist working for a company that is building a report for a cosmetic patent document. Your job is to provide the final composition report of the product mentioned in the patent. Using the provided extracted functional role information, compile a comprehensive JSON object detailing the product's functional roles, the chemicals belonging to those roles, and their weights in percentages or ranges.Include all the unique extracted functional roles, their chemicals, and weights (or weight ranges). Ensure the JSON is clean, well-structured, and free of duplicates.If no functional roles are found in the complete data, respond with an empty functional_roles JSON object. Dont add comments, notes, or suggestions. Only provide the JSON object in the specified format.

        ### Output Format:
        Provide the response strictly in the following JSON format:
        {output_format}
        """

        # Prepare the request payload
        payload = {
            "prompt": prompt
        }

        logging.debug(f"Prepared payload: {payload}")

        # Route to appropriate provider using a match-case structure
        content = None

        if provider == "ollama":
            logging.debug(f"Sending request to http://localhost:7860/ollama/chat with payload: {payload}")
            async with httpx.AsyncClient() as client:
                response = await client.post("http://localhost:7860/ollama/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            message = result.get('generated_text', {}).get('message', {})
            content = message.get('content', None)

        elif provider == "openai":
            output = await openai_generate(prompt)
            cleaned_output = clean_functional_role_info(output)
            return cleaned_output

        elif provider == "groq":
            logging.debug(f"Sending request to Groq endpoint with payload: {payload}")
            async with httpx.AsyncClient() as client:
                response = await client.post("http://localhost:7860/groq/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            content = result.get('response', None)  # Adjust based on Groq's API response structure

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if content:
            return content
        else:
            logging.error("No content received in the response.")
            return None

    except httpx.RequestError as e:
        logging.error(f"Error communicating with {provider}: {e}")
        return f"Error: {e}"

    except ValueError as e:  # Handle JSON decoding errors and invalid provider
        logging.error(f"ValueError: {e}")
        return f"ValueError: {e}"

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return f"Unexpected error: {e}"

# Function to clean and tokenize text into words
def clean_and_tokenize(text: str) -> set:
    """
    Cleans and tokenizes input text into a set of words.
    Removes punctuation and converts to lowercase.
    """
    return set(re.findall(r'\b\w+\b', text.lower()))

# Function to check and append only unique words
def information_checker(information_extracted: str, new_response: str) -> str:
    """
    Compares existing extracted information with new response, appends only unique words.
    """
    # Tokenize and clean existing and new responses
    existing_tokens = clean_and_tokenize(information_extracted)
    new_tokens = clean_and_tokenize(new_response)
    
    # Find unique words from new response
    unique_tokens = new_tokens - existing_tokens

    if unique_tokens:
        # Append unique words to the extracted information
        updated_information = f"{information_extracted.strip()} {' '.join(unique_tokens)}"
        print("New unique words added:", unique_tokens)
        return updated_information.strip()
    else:
        print("No new unique words to append.")
        return information_extracted.strip()

functional_roles = "Antioxidant, Humectant, Emollient, Surfactant, Emulsifier, Preservative, Fragrance, Colorant, UV Filter (Sunscreen Agent), Thickener/Viscosity Modifier, Conditioning Agent, Astringent, Film-Former, Opacifier, Solvent, Exfoliant, Antimicrobial Agent, Chelating Agent, Antifoaming Agent, Moisturizer, Absorbent, Mattifier, Skin Protectant, Soothing Agent, Exfoliating Enzyme, Wetting Agent, Texturizer, Anti-inflammatory, Desensitizer, Penetration Enhancer, Hair Fixative, Antidandruff Agent, Anti-aging Agent, Brightening Agent, Anti-acne Agent, Lubricant, Deodorant Agent, Toning Agent, Antiperspirant, Styling Agent, Hair Growth Stimulator, Anti-hair Loss Agent, Nail Hardener, Plasticizer, Peptide/Protein Agent, Anti-pollution Agent, Anti-oxidative Stress Agent"

def extract_document_details(text_input):
    """
    Helper function to extract patent document details from the input text
    and return the extracted JSON object.

    Args:
        text_input (str): Text containing the patent details in JSON-like format.

    Returns:
        dict: Extracted patent document details as a JSON object.
    """
    try:
        # Extract JSON object from the text input
        start_index = text_input.find("{")
        end_index = text_input.rfind("}") + 1
        json_text = text_input[start_index:end_index]
        document_details = json.loads(json_text)
        print("The final document details are :", document_details)
        return document_details.get("patent", {})

    except (ValueError, KeyError, AttributeError) as e:
        print(f"Error processing document details: {e}")
        return {}

# Initialize the embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_relevance_from_query_response(query_response_data: list) -> str:
    """
    Calculates relevance scores between consecutive responses in the query-response data.

    Parameters:
    - query_response_data: list of dictionaries containing "query" and "response" keys.

    Returns:
    - A string summarizing the chunk pairs and their relevance scores.
    """
    # Extract responses from the query-response data
    responses = [entry['response'] for entry in query_response_data]
    chunks = [entry['query'] for entry in query_response_data]

    # Encode responses into embeddings
    embeddings = model.encode(responses)

    # Calculate relevance scores
    relevance_scores = []
    for i in range(len(embeddings) - 1):
        query_embedding = embeddings[i]
        key_embedding = embeddings[i + 1]
        score = cosine_similarity([query_embedding], [key_embedding])[0][0]
        relevance_scores.append(f"Chunk Pair: [{chunks[i]}] -> [{chunks[i + 1]}], Relevance Score: {score:.4f}")

    # Combine all scores into a single string
    result_string = "\n".join(relevance_scores)
    return result_string

# Example usage in the self_attention_chunking workflow
async def self_attention_chunking(data: str, provider: str) -> dict:
    print("Process Started with the patent text")

    # Split data into chunks to fit token space
    max_tokens_per_chunk = 2048  # Total token budget
    chunked_data = splitStringToFitTokenSpace(string=data, token_use_per_string=0)  # Adjust chunk size
    print("Number of chunks created from the text:", len(chunked_data))

    query_response_data = []

    print("Starting chunkwise processing")

    for i, chunk in enumerate(chunked_data, start=1):
        print(f"\nProcessing Chunk {i}:")

        composition_prompt = f"""
        ### New Chunk:
        {chunk}

        ### Instruction:
        You are a data scientist working for a company that is building a report for a cosmetic patent document. Your job is to find the chemicals used to make the product and in what percentage weights or weight ranges they have been used in the product formulation. Your task is to extract chemicals and their respective functional roles (e.g., Emollient, Humectant, Preservative) along with their weights in percentages or ranges.

        ### Output Format:
        Provide the response in a simple text format with headings and bullet points, as follows:

        Functional Role: <Role Name>
        - Chemical: <Chemical Name>, Weight: <Percentage or Range>
        - Chemical: <Chemical Name>, Weight: <Percentage or Range>

        Functional Role: <Role Name>
        - Chemical: <Chemical Name>, Weight: <Percentage or Range>

        If no relevant information is found, respond with a message saying **"No New Functional roles found"** exactly and nothing else. Do not provide any comments, notes, or suggestions. Only provide the extracted data in the specified format.

        """

        tokens_in_prompt_2 = num_tokens_from_string(composition_prompt)
     
        print(f"Tokens sent for Chunk {i}: {tokens_in_prompt_2}")
       
        processedChunk_2 = await process(composition_prompt, provider)
        print(f"Chunk {i} processed response: {processedChunk_2}")

        # Append the chunk and the processed response to the query-response list
        query_response_data.append({"query": chunk, "response": processedChunk_2})

    # Calculate relevance scores based on query-response data
    relevance_score = calculate_relevance_from_query_response(query_response_data)
    print("Relevance score is : ", relevance_score)

    # Finalize extracted information
    finalized_information = {
        "functional_roles": ""
    }

    print(finalized_information)

    return finalized_information
