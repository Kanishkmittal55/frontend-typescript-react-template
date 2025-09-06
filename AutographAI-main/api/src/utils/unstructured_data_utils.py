import json
import re
import csv
import tiktoken

regex = "Nodes:\s+(.*?)\s?\s?Relationships:\s+(.*)"
internalRegex = "\[(.*?)\]"
jsonRegex = "\{.*\}"


def nodesTextToListOfDict(nodes):
    result = []
    for node in nodes:
        nodeList = node.split(",")
        if len(nodeList) < 2:
            continue

        name = nodeList[0].strip().replace('"', "")
        label = nodeList[1].strip().replace('"', "")
        properties = re.search(jsonRegex, node)
        if properties == None:
            properties = "{}"
        else:
            properties = properties.group(0)
        properties = properties.replace("True", "true")
        try:
            properties = json.loads(properties)
        except:
            properties = {}
        result.append({"name": name, "label": label, "properties": properties})
    return result


def relationshipTextToListOfDict(relationships):
    result = []
    for relation in relationships:
        relationList = relation.split(",")
        if len(relation) < 3:
            continue
        start = relationList[0].strip().replace('"', "")
        end = relationList[2].strip().replace('"', "")
        type = relationList[1].strip().replace('"', "")

        properties = re.search(jsonRegex, relation)
        if properties == None:
            properties = "{}"
        else:
            properties = properties.group(0)
        properties = properties.replace("True", "true")
        try:
            properties = json.loads(properties)
        except:
            properties = {}
        result.append(
            {"start": start, "end": end, "type": type, "properties": properties}
        )
    return result

def save_intermediate_results_to_csv(intermediate_results):
    """
    Save intermediate processing stages to a CSV file.
    """
    csv_file = "intermediate_results.csv"
    fieldnames = ["stage", "chunk_number", "system_prompt", "input_chunk_text", "nodes", "relationships"]

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in intermediate_results:
            if result.get("chunks"):
                for chunk in result["chunks"]:  # Iterate chunks if available
                    if result["stage"] == "Extraction":
                        # Write row for Extraction stage
                        writer.writerow({
                            "stage": result["stage"],
                            "chunk_number": chunk.get("chunk_number", "N/A"),
                            "system_prompt": chunk.get("system_prompt", "N/A"),
                            "input_chunk_text": chunk.get("input_chunk_text", "N/A"),
                            "nodes": chunk.get("chunk_result_nodes", "N/A"),
                            "relationships": chunk.get("chunk_result_relationships", "N/A")
                        })
                    elif result["stage"] == "Disambiguation":
                        # Write row for Disambiguation stage
                        writer.writerow({
                        "stage": result["stage"],
                        "chunk_number": chunk.get("chunk_number", "N/A"),
                        "system_prompt": chunk.get("system_prompt", "N/A"),
                        "input_chunk_text": chunk.get("input_chunk_text", "N/A"),
                        "nodes": f"Raw: {chunk['nodes']['raw']}, Transformed: {chunk['nodes']['transformed']}",
                        "relationships": f"Raw: {chunk['relationships']['raw']}, Transformed: {chunk['relationships']['transformed']}"
                    })
            else:  # Log stages without chunks
                writer.writerow({
                    "stage": result["stage"],
                    "chunk_number": "N/A",
                    "system_prompt": "N/A",
                    "input_chunk_text": "N/A",
                    "nodes": result.get("data", "N/A"),
                    "relationships": "N/A"
                })
    print(f"Intermediate results saved to {csv_file}")



def data_to_cypher(data):
    """
    Converts nodes and relationships into Cypher statements.
    """
    cypher_statements = []

    # Create node statements
    for node in data["nodes"]:
        properties = ", ".join([f'{key}: "{value}"' for key, value in node["properties"].items()])
        node_statement = f'CREATE ({node["name"]}:{node["label"]} {{ {properties} }});'
        cypher_statements.append(node_statement)

    # Create relationship statements
    for relationship in data["relationships"]:
        properties = ", ".join([f'{key}: "{value}"' for key, value in relationship["properties"].items()])
        relationship_statement = (
            f'MATCH (a:{relationship["start"]}), (b:{relationship["end"]}) '
            f'CREATE (a)-[:{relationship["type"]} {{ {properties} }}]->(b);'
        )
        cypher_statements.append(relationship_statement)

    return "\n".join(cypher_statements)