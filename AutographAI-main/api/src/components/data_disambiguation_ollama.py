import json
import re
from itertools import groupby
from utils.unstructured_data_utils import (
    nodesTextToListOfDict,
    relationshipTextToListOfDict,
)

internalRegex = r"\[(.*?)\]"

# Global registries for unique nodes and relationships
global_nodes_registry = {} # Dictionaries in python are unique by default w.r.t. their keys.
global_relationships_registry = set() # So is a set.


def generate_system_message_for_nodes() -> str:
    return """Your task is to identify if there are duplicated nodes and if so merge them into one node. Only merge the nodes that refer to the same entity.
            You will be given different datasets of nodes and some of these nodes may be duplicated or refer to the same entity. 
            The datasets contains nodes in the form [ENTITY_ID, TYPE, PROPERTIES]. When you have completed your task please give me the 
            resulting nodes in the same format. Only return the nodes and relationships, no other text. If there are no duplicated nodes return the original nodes.

            Here is an example of the input you will be given:
            ["alice", "Person", {"age": 25, "occupation": "lawyer", "name":"Alice"}], 
            ["bob", "Person", {"occupation": "journalist", "name": "Bob"}], 
            ["alice.com", "Webpage", {"url": "www.alice.com"}], ["bob.com", "Webpage", {"url": "www.bob.com"}]
"""


def generate_system_message_for_relationships() -> str:
    return """
Your task is to identify if a set of relationships make sense.
If they do not make sense please remove them from the dataset.
Some relationships may be duplicated or refer to the same entity. 
Please merge relationships that refer to the same entity.
The datasets contains relationships in the form [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES].
You will also be given a set of ENTITY_IDs that are valid.
Some relationships may use ENTITY_IDs that are not in the valid set but refer to an entity in the valid set.
If a relationship refers to an ENTITY_ID in the valid set, please change the ID so it matches the valid ID.
When you have completed your task, please give me the valid relationships in the same format. Only return the relationships, no other text.

Here is an example of the input you will be given:
["alice", "roommate", "bob", {"start": 2021}], ["alice", "owns", "alice.com", {}], ["bob", "owns", "bob.com", {}]
"""


def generate_prompt(data) -> str:
    return f"""Here is the data:
{data}
"""


def process(chunk_data):
    """
    Placeholder for processing a chunk of nodes or relationships.
    This function will handle API calls or other mechanisms to disambiguate the given data.
    """
    # Perform API call or disambiguation logic
    # Return processed nodes or relationships
    pass


def add_to_global_registry(nodes, relationships):
    """
    Add new nodes and relationships to the global registry if they are not already present.
    """
    # print("The recieved nodes are : ", nodes)
    for node in nodes:
        if node["name"] not in global_nodes_registry:
            global_nodes_registry[node["name"]] = node

    for relationship in relationships:
        # Use a tuple to ensure relationships are stored uniquely
        rel_tuple = (
            relationship["start"],
            relationship["type"],
            relationship["end"],
            json.dumps(relationship["properties"]),
        )
        global_relationships_registry.add(rel_tuple)


# This function removes nodes and relationships that are already present in the global registry , retuning only unique items for further processing.N
def filter_existing_entries(nodes, relationships):
    """
    Filter out nodes and relationships that are already in the global registry.
    """
    unique_nodes = [
        node for node in nodes if node["name"] not in global_nodes_registry
    ]

    unique_relationships = []
    for relationship in relationships:
        rel_tuple = (
            relationship["start"],
            relationship["type"],
            relationship["end"],
            json.dumps(relationship["properties"]),
        )
        if rel_tuple not in global_relationships_registry:
            unique_relationships.append(relationship)

    return unique_nodes, unique_relationships


def run_disambiguation(data):
    """
    Main function to perform data disambiguation on nodes and relationships.
    """
    print("I have come inside the function :",data)

    # data["nodes"] extracts a list of nodes from the input data , and sorted() - 
    # sorts the nodes alphabetically by their label, and key=lamda x : x["label"] specified the sorting criterion ( the label field of each node ).
    # Just this is not working right now for some reason.
    nodes = sorted(data["nodes"], key=lambda x: x["label"])

    print("\033[32mThe sorted nodes are :\033[0m", nodes)  # Green text

    relationships = data["relationships"]
    new_nodes = []
    new_relationships = []

    # Process nodes group by label
    node_groups = groupby(nodes, lambda x: x["label"])

    
    for group in node_groups:
        print("The node groups are : ", group)
        disString = ""
        nodes_in_group = list(group[1])
        print("nodes_in_group :" , nodes_in_group)
        if len(nodes_in_group) == 1:
            new_nodes.extend(nodes_in_group)
            continue

        for node in nodes_in_group:
            disString += (
                '["'
                + node["name"]
                + '", "'
                + node["label"]
                + '", '
                + json.dumps(node["properties"])
                + "]\n"
            )

            print("Disstring : ", disString)

        # Call the process function (API call or other disambiguation logic)
        processed_nodes = process(disString)
        processed_nodes = nodesTextToListOfDict(
            re.findall(internalRegex, processed_nodes)
        )

        # Filter and add unique nodes to the list
        unique_nodes, _ = filter_existing_entries(processed_nodes, [])
        new_nodes.extend(unique_nodes)

    # Update global nodes registry
    add_to_global_registry(new_nodes, [])

    # Process relationships
    relationship_data = "Relationships:\n"
    for relation in relationships:
        relationship_data += (
            '["'
            + relation["start"]
            + '", "'
            + relation["type"]
            + '", "'
            + relation["end"]
            + '", '
            + json.dumps(relation["properties"])
            + "]\n"
        )

    # Add valid node labels to relationship data
    node_labels = [node["name"] for node in global_nodes_registry.values()]
    relationship_data += "Valid Nodes:\n" + "\n".join(node_labels)

    # Call the process function (API call or other disambiguation logic)
    processed_relationships = process(relationship_data)
    processed_relationships = relationshipTextToListOfDict(
        re.findall(internalRegex, processed_relationships)
    )

    # Filter and add unique relationships
    _, unique_relationships = filter_existing_entries([], processed_relationships)
    new_relationships.extend(unique_relationships)

    # Update global relationships registry
    add_to_global_registry([], new_relationships)

    # Return global results
    return {
        "nodes": list(global_nodes_registry.values()),
        "relationships": [
            {
                "start": rel[0],
                "type": rel[1],
                "end": rel[2],
                "properties": json.loads(rel[3]),
            }
            for rel in global_relationships_registry
        ],
    }
