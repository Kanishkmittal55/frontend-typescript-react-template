import ollama 

response = ollama.list()
print(response)


modelfile = """
FROM llama3.1
SYSTEM You are very smart assitant who knows everything about oceans
PARAMETER temperature 0.1
"""

ollama.create(model="knowitall", modelfile=modelfile)

