import json
import re
import os
from itertools import groupby
from llm.openai import OpenAIChat
import logging
from components.base_component import BaseComponent
from utils.unstructured_data_utils import (
    nodesTextToListOfDict,
    relationshipTextToListOfDict,
)
from dotenv import load_dotenv
from pathlib import Path
import os

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is required")

api_key = openai_api_key 


def generate_system_message_for_nodes() -> str:
    return """Your task is to identify if there are duplicated nodes and if so merge them into one nod. Only merge the nodes that refer to the same entity.
You will be given different datasets of nodes and some of these nodes may be duplicated or refer to the same entity. 
The datasets contains nodes in the form [ENTITY_ID, TYPE, PROPERTIES]. When you have completed your task please give me the 
resulting nodes in the same format. Only return the nodes and relationships no other text. If there is no duplicated nodes return the original nodes.

Here is an example of the input you will be given:
["alice", "Person", {"age": 25, "occupation": "lawyer", "name":"Alice"}], ["bob", "Person", {"occupation": "journalist", "name": "Bob"}], ["alice.com", "Webpage", {"url": "www.alice.com"}], ["bob.com", "Webpage", {"url": "www.bob.com"}]
"""


def generate_system_message_for_relationships() -> str:
    return """
Your task is to identify if a set of relationships make sense.
If they do not make sense please remove them from the dataset.
Some relationships may be duplicated or refer to the same entity. 
Please merge relationships that refer to the same entity.
The datasets contains relationships in the form [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES].
You will also be given a set of ENTITY_IDs that are valid.
Some relationships may use ENTITY_IDs that are not in the valid set but refer to a entity in the valid set.
If a relationships refer to a ENTITY_ID in the valid set please change the ID so it matches the valid ID.
When you have completed your task please give me the valid relationships in the same format. Only return the relationships no other text.

Here is an example of the input you will be given:
["alice", "roommate", "bob", {"start": 2021}], ["alice", "owns", "alice.com", {}], ["bob", "owns", "bob.com", {}]
"""


def generate_prompt(data) -> str:
    return f""" Here is the data:
{data}
"""


internalRegex = "\[(.*?)\]"


class DataDisambiguation(BaseComponent):
    def __init__(self, llm) -> None:
        self.llm = llm

    def run(self, data: dict) -> dict:
        print("=== Process Started: Node and Relationship Processing ===")

        # Initial data log
        print("Initial Data Received:")
        print(f"Nodes: {len(data['nodes'])} nodes")
        print(f"Relationships: {len(data['relationships'])} relationships")

        nodes = sorted(data["nodes"], key=lambda x: x["label"])
        print(f"Nodes sorted by label: {len(nodes)} nodes sorted.")

        relationships = data["relationships"]
        print(f"Relationships retained as-is: {len(relationships)} relationships.")

        new_nodes = []
        new_relationships = []
        chunks = []  # To store chunk metadata

        print("\n--- Starting Node Processing ---")
        node_groups = groupby(nodes, lambda x: x["label"])
        for i, group in enumerate(node_groups, start=1):
            disString = ""
            nodes_in_group = list(group[1])
            print(f"\nGroup {i}: Label = {group[0]}, Nodes in group = {len(nodes_in_group)}")

            if len(nodes_in_group) == 1:
                print(f"Single node group. Adding node: {nodes_in_group[0]}")
                new_nodes.extend(nodes_in_group)
                continue

            # Prepare disString for group
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
            print(f"Constructed disString for group {i}:\n{disString}")

            system_message = generate_system_message_for_nodes()
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": generate_prompt(disString)},
            ]
            print(f"System message for group {i}:\n{system_message}")

            rawNodes = self.llm.generate(messages)
            print(f"Raw response for group {i}:\n{rawNodes}")

            n = re.findall(internalRegex, rawNodes)
            print(f"Extracted node information for group {i}:\n{n}")

            # Transform raw response into node dicts
            transformed_nodes = nodesTextToListOfDict(n)
            print(f"Transformed nodes for group {i}:\n{transformed_nodes}")
            new_nodes.extend(transformed_nodes)

            # Append chunk metadata with both raw and transformed responses
            # Append chunk metadata with both raw and transformed nodes
            chunks.append({
                "chunk_number": len(chunks) + 1,
                "system_prompt": system_message,
                "input_chunk_text": disString,
                "nodes": {
                    "raw": json.dumps(rawNodes),
                    "transformed": json.dumps(transformed_nodes)
                },
                "relationships": {
                    "raw": None,  # No relationships for node processing
                    "transformed": None
                }
            })

        print("\n--- Starting Relationship Processing ---")
        relationship_data = "Relationships:\n"
        for relation in relationships:
            relation_string = (
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
            relationship_data += relation_string
        print(f"Constructed relationship data:\n{relationship_data}")

        system_message = generate_system_message_for_relationships()
        node_labels = [node["name"] for node in new_nodes]
        relationship_data += "Valid Nodes:\n" + "\n".join(node_labels)
        print(f"Final relationship data with valid nodes:\n{relationship_data}")

        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {"role": "user", "content": generate_prompt(relationship_data)},
        ]
        print(f"System message for relationships:\n{system_message}")

        rawRelationships = self.llm.generate(messages)
        print(f"Raw response for relationships:\n{rawRelationships}")

        rels = re.findall(internalRegex, rawRelationships)
        print(f"Extracted relationship information:\n{rels}")

        # Transform raw response into relationship dicts
        transformed_relationships = relationshipTextToListOfDict(rels)
        print(f"Transformed relationships:\n{transformed_relationships}")
        new_relationships.extend(transformed_relationships)

        # Append chunk metadata for relationships with both raw and transformed responses
        chunks.append({
            "chunk_number": len(chunks) + 1,
            "system_prompt": system_message,
            "input_chunk_text": relationship_data,
            "nodes": {
                "raw": None,  # No nodes for relationship processing
                "transformed": None
            },
            "relationships": {
                "raw": json.dumps(rawRelationships),
                "transformed": json.dumps(transformed_relationships)
            }
        })

        print("\n=== Process Completed ===")
        print(f"Total nodes processed: {len(new_nodes)}")
        print(f"Total relationships processed: {len(new_relationships)}")
        print(f"Total chunks logged: {len(chunks)}")

        return {"nodes": new_nodes, "relationships": new_relationships, "chunks": chunks}

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

if __name__ == "__main__":
    # Initialize LLM 

        llm = OpenAIChat(
            openai_api_key=api_key, model_name="gpt-4o", max_tokens=4000
        )

        client = OpenAI(
            api_key=api_key,  # This is the default and can be omitted
        )   

        disambiguation = DataDisambiguation(llm)

        # Provide custom input data for testing
        test_data = [
            """Nodes: [
                ["US12161213B2", "Patent", {"documentId": "US 12161213 B2", "publishedDate": "2024-12-10"}],
                ["Varale; Aditya", "Inventor", {"name": "Aditya Varale", "city": "Mumbai", "country": "IN"}], 
                ["Sheregar; Venkatesh", "Inventor", {"name": "Venkatesh Sheregar", "city": "Shanghai", "country": "CN"}], 
                ["L'Oreal Paris", "Company", {"name": "L'Oreal Paris", "city": "Paris", "country": "FR"}],
                ["Application_17/906804", "Application", {"applicationNumber": "17/906804", "dateFiled": "2021-03-15"}],
                ["ForeignApplication_IN", "ForeignApplication", {"country": "IN", "applicationNumber": "202021012636", "applicationDate": "2020-03-23"}],
                ["ForeignApplication_FR", "ForeignApplication", {"country": "FR", "applicationNumber": "2008614", "applicationDate": "2020-08-21"}],
                ["InternationalApplication", "PatentApplication", {"application_number": "PCT/EP2021/056493", "jurisdiction": "European Patent Office", "filing_date": "2021-03-15"}],
                ["IndianApplication", "PatentApplication", {"application_number": "202021012636", "jurisdiction": "India", "filing_date": "2020-03-23"}],
                ["FrenchApplication", "PatentApplication", {"application_number": "2008614", "jurisdiction": "France", "filing_date": "2020-08-21"}],
                ["PriorityRight", "LegalConcept", {"definition": "The right to claim the earliest filing date for the invention."}],
                ["ParisConvention", "InternationalAgreement", {"purpose": "Establishes rules for claiming priority among participating countries."}],
                ["NationalPhase", "PatentProcessPhase", {"definition": "Stage where an international application is transitioned into national jurisdictions for patent granting."}],
                ["IncorporationByReference", "LegalConcept", {"definition": "Treating earlier applications as if fully included in the current application."}],
                ["invention1", "Invention", {"field": "Cosmetic Product Packaging", "description": "Container for packaging and dispensing a cosmetic product."}],
                ["cosmetic_product", "ProductType", {"definition": "As defined in Article 2 of Regulation No. 1223/2009 of the European Parliament and of the Council."}],
                ["container1", "Container", {"purpose": "Packaging a cosmetic product such as make-up, compact powder, or pasty products.", "examples": ["Eye shadow", "Blusher", "Foundation"]}],
                ["packaging_device", "Device", {"features": "Upper cap pivotably mounted on a base."}],
                ["us_patent_6901937", "PriorArt", {"title": "Make-up case with pivotable lid and bottom", "document_number": "US6901937B2"}],
                ["pivotable_cap", "Component", {"function": "Allows movement of the lid and cap in different positions."}],
                ["horizontal_pins", "Component", {"function": "Reversible fastening elements that cooperate with horizontal recesses."}],
                ["complex_mould", "ManufacturingProcess", {"description": "Mould needed to create horizontal recesses with sliders apart from the core."}],
                ["regulation1223_2009", "Regulation", {"name": "Regulation No. 1223/2009", "source": "European Parliament and Council", "date": "2009-11-30"}],
                ["article2", "LegalClause", {"description": "Defines cosmetic products under Regulation 1223/2009."}],
                ["european_parliament", "Entity", {"type": "Legislative Body", "region": "European Union"}],
                ["council_of_eu", "Entity", {"type": "Legislative Body", "region": "European Union"}],
                ["invention2", "Invention", {"field": "Cosmetic Product Packaging", "description": "Single-piece moulded container with hinge platform and snapping mechanism for packaging and dispensing products."}],
                ["us_patent_2002162565", "PriorArt", {"title": "Case for packaging a product with 360° pivot lid", "document_number": "US2002/0162565-A1"}],
                ["container2", "Container", {"type": "Single-piece moulded container", "features": ["Cover", "Base", "Hinge Platform"]}],
                ["hinge_platform", "Component", {"features": ["First hinge", "Second hinge"], "purpose": "Connects cover and base, enabling pivoting between positions."}],
                ["snapping_mechanism", "Component", {"features": ["Snap beads", "Snapping member"], "purpose": "Locks base and hinge platform, preventing easy reversal."}],
                ["snap_beads", "Component", {"type": "Vertical Snap Beads", "function": "Provides significant snapping force."}],
                ["moulding_core", "ManufacturingProcess", {"description": "Simplified mould core for single-piece container, no additional sliders needed."}],
                ["usage_policy1", "Policy", {"description": "Reduce manufacturing cost and improve ergonomics of product packaging."}],
                ["snapping_member", "Component", {"shape": "Elongated bar or rib", "features": ["Fastening members", "Groove"], "position": "Opened"}],
                ["fastening_members", "Component", {"features": ["Groove depth: 0.55-0.65 mm", "Groove length: 0.2-0.23 mm"], "function": "Cooperate with vertical snap beads"}],
                ["snap_beads", "Component", {"type": "Vertical beads", "position": "Base", "features": ["Rounded edge", "Protruding towards each other"], "function": "Significant snapping force"}],
                ["vertical_groove", "Component", {"shape": "Parallelepiped", "position": "Rear wall cut-out"}],
                ["ribs", "Component", {"position": "Snapping member edges", "features": ["Flat surface", "Protruding"]}],
                ["hinge_platform", "Component", {"features": ["First hinge", "Second hinge"], "position": "Base to cover", "function": "Pivoting"}],
                ["hinge_types", "Component", {"types": ["Dead hinge"], "features": ["U-groove upper", "V-notch lower"]}],
                ["v_notch", "Component", {"angle": "100-140°", "depth": "0.1 mm", "position": "Lower hinge surface"}],
                ["u_groove", "Component", {"position": "Upper hinge surface"}],
                ["cover", "Component", {"features": ["Front wall", "Rear wall", "Inner cavity"], "function": "Encloses product or mirror"}],
                ["base", "Component", {"features": ["Front wall", "Rear wall", "Inner cavity"], "function": "Holds product"}],
                ["alignment", "Feature", {"function": "Align cover and base", "position": "Front and rear walls"}],
                ["cut_out", "Feature", {"position": "Rear wall edge", "function": "Prevents scraping during rotation"}],
                ["angles", "Feature", {"first_angle": "180°", "second_angle": "90-150°"}],
                ["ergonomic_design", "Policy", {"objective": "Ease of manufacturing"}],
                ["manufacturing_cost", "Policy", {"objective": "Cost reduction"}]
            ]

            Relationships: [
                ["US12161213B2", "inventedBy", "Varale; Aditya", {}], 
                {"US12161213B2", "inventedBy", "Sheregar; Venkatesh", {}}, 
                ["L'Oreal Paris", "appliedFor", "US12161213B2", {"applicationDate": "2021-03-15"}], 
                ["L'Oreal Paris", "assignedTo", "US12161213B2", {}], ["Application_17/906804", "hasPriorityClaim", "ForeignApplication_IN", {}],
                ["Application_17/906804", "hasPriorityClaim", "ForeignApplication_FR", {}],
                ["InternationalApplication", "claims_benefit_of", "IndianApplication", {"priority_date": "2020-03-23"}],
                ["InternationalApplication", "claims_benefit_of", "FrenchApplication", {"priority_date": "2020-08-21"}],
                ["IndianApplication", "establishes", "PriorityRight", {"date": "2020-03-23"}],
                ["PriorityRight", "governed_by", "ParisConvention", {}],
                ["InternationalApplication", "proceeds_to", "NationalPhase", {"jurisdictions": ["Europe", "USA", "others"]}],
                ["InternationalApplication", "uses", "IncorporationByReference", {"source_applications": ["IndianApplication", "FrenchApplication"]}],
                ["invention1", "targets", "cosmetic_product", {}],
                ["invention1", "utilizes", "container1", {}],
                ["container1", "includes", "packaging_device", {}],
                ["packaging_device", "contains", "pivotable_cap", {}],
                ["pivotable_cap", "features", "horizontal_pins", {}],
                ["horizontal_pins", "formed_by", "complex_mould", {}],
                ["invention1", "references", "us_patent_6901937", {"relationship_type": "Prior Art"}],
                ["cosmetic_product", "defined_by", "article2", {}],
                ["article2", "part_of", "regulation1223_2009", {}],
                ["regulation1223_2009", "issued_by", "european_parliament", {}],
                ["regulation1223_2009", "issued_by", "council_of_eu", {}],
                ["invention2", "references", "us_patent_2002162565", {"relationship_type": "Prior Art"}],
                ["invention2", "utilizes", "container2", {}],
                ["container2", "includes", "hinge_platform", {}],
                ["container2", "includes", "snapping_mechanism", {}],
                ["hinge_platform", "contains", "snap_beads", {}],
                ["snapping_mechanism", "formed_by", "moulding_core", {}],
                ["hinge_platform", "enables", "pivoting", {"positions": ["Moulding Flat", "Opened", "Closed"]}],
                ["snap_beads", "provides", "snapping_force", {}],
                ["invention2", "complies_with", "usage_policy1", {}],
                ["usage_policy1", "targets", "container2", {}],
                ["snapping_member", "includes", "fastening_members", {}],
                ["fastening_members", "cooperate_with", "snap_beads", {"position": "Opened"}],
                ["snap_beads", "extends_along", "vertical_groove", {"height": "Entire base height"}],
                ["snapping_member", "covers", "hinge_platform", {"position": "Opened"}],
                ["snapping_member", "protrudes", "ribs", {"axis": "Transversal"}],
                ["ribs", "resist", "de-snapping", {}],
                ["hinge_platform", "contains", "hinge_types", {}],
                ["hinge_types", "includes", "v_notch", {}],
                ["hinge_types", "includes", "u_groove", {}],
                ["cover", "delimited_by", "base", {"alignment": "Front and rear walls"}],
                ["cut_out", "prevents", "scraping", {"movement": "Rotation"}],
                ["angles", "defines", "hinge_positions", {}],
                ["snapping_member", "aligns_with", "ergonomic_design", {}],
                ["snapping_member", "reduces", "manufacturing_cost", {}]
            ]
            """
          
        ]

#         processed_Nodes = [       
#     {
#         "name": "Application_17/906804",
#         "label": "Application",
#         "properties": {
#         "applicationNumber": "17/906804",
#         "dateFiled": "2021-03-15"
#         }
#     },
#     {
#         "name": "L'Oreal Paris",
#         "label": "Company",
#         "properties": {
#         "name": "L'Oreal Paris",
#         "city": "Paris",
#         "country": "FR"
#         }
#     },
#     {
#         "name": "pivotable_cap",
#         "label": "Component",
#         "properties": {
#         "function": "Allows movement of the lid and cap in different positions."
#         }
#     },
#     {
#         "name": "horizontal_pins",
#         "label": "Component",
#         "properties": {
#         "function": "Reversible fastening elements that cooperate with horizontal recesses."
#         }
#     },
#     {
#         "name": "hinge_platform",
#         "label": "Component",
#         "properties": {}
#     },
#     {
#         "name": "snapping_mechanism",
#         "label": "Component",
#         "properties": {}
#     },
#     {
#         "name": "snap_beads",
#         "label": "Component",
#         "properties": {
#         "type": "Vertical Snap Beads",
#         "function": "Provides significant snapping force."
#         }
#     },
#     {
#         "name": "snapping_member",
#         "label": "Component",
#         "properties": {}
#     },
#     {
#         "name": "fastening_members",
#         "label": "Component",
#         "properties": {}
#     },
#     {
#         "name": "vertical_groove",
#         "label": "Component",
#         "properties": {
#         "shape": "Parallelepiped",
#         "position": "Rear wall cut-out"
#         }
#     },
#     {
#         "name": "ribs",
#         "label": "Component",
#         "properties": {}
#     },
#     {
#         "name": "hinge_types",
#         "label": "Component",
#         "properties": {}
#     },
#     {
#         "name": "v_notch",
#         "label": "Component",
#         "properties": {
#         "angle": "100-140\u00b0",
#         "depth": "0.1 mm",
#         "position": "Lower hinge surface"
#         }
#     },
#     {
#         "name": "u_groove",
#         "label": "Component",
#         "properties": {
#         "position": "Upper hinge surface"
#         }
#     },
#     {
#         "name": "cover",
#         "label": "Component",
#         "properties": {}
#     },
#     {
#         "name": "base",
#         "label": "Component",
#         "properties": {}
#     },
#     {
#         "name": "container1",
#         "label": "Container",
#         "properties": {}
#     },
#     {
#         "name": "container2",
#         "label": "Container",
#         "properties": {}
#     },
#     {
#         "name": "packaging_device",
#         "label": "Device",
#         "properties": {
#         "features": "Upper cap pivotably mounted on a base."
#         }
#     },
#     {
#         "name": "european_parliament",
#         "label": "Entity",
#         "properties": {
#         "type": "Legislative Body",
#         "region": "European Union"
#         }
#     },
#     {
#         "name": "council_of_eu",
#         "label": "Entity",
#         "properties": {
#         "type": "Legislative Body",
#         "region": "European Union"
#         }
#     },
#     {
#         "name": "alignment",
#         "label": "Feature",
#         "properties": {
#         "function": "Align cover and base",
#         "position": "Front and rear walls"
#         }
#     },
#     {
#         "name": "cut_out",
#         "label": "Feature",
#         "properties": {
#         "position": "Rear wall edge",
#         "function": "Prevents scraping during rotation"
#         }
#     },
#     {
#         "name": "angles",
#         "label": "Feature",
#         "properties": {
#         "first_angle": "180\u00b0",
#         "second_angle": "90-150\u00b0"
#         }
#     },
#     {
#         "name": "ForeignApplication_IN",
#         "label": "ForeignApplication",
#         "properties": {
#         "country": "IN",
#         "applicationNumber": "202021012636",
#         "applicationDate": "2020-03-23"
#         }
#     },
#     {
#         "name": "ForeignApplication_FR",
#         "label": "ForeignApplication",
#         "properties": {
#         "country": "FR",
#         "applicationNumber": "2008614",
#         "applicationDate": "2020-08-21"
#         }
#     },
#     {
#         "name": "ParisConvention",
#         "label": "InternationalAgreement",
#         "properties": {
#         "purpose": "Establishes rules for claiming priority among participating countries."
#         }
#     },
#     {
#         "name": "invention1",
#         "label": "Invention",
#         "properties": {
#         "field": "Cosmetic Product Packaging",
#         "description": "Container for packaging and dispensing a cosmetic product."
#         }
#     },
#     {
#         "name": "invention2",
#         "label": "Invention",
#         "properties": {
#         "field": "Cosmetic Product Packaging",
#         "description": "Single-piece moulded container with hinge platform and snapping mechanism for packaging and dispensing products."
#         }
#     },
#     {
#         "name": "Varale; Aditya",
#         "label": "Inventor",
#         "properties": {
#         "name": "Aditya Varale",
#         "city": "Mumbai",
#         "country": "IN"
#         }
#     },
#     {
#         "name": "Sheregar; Venkatesh",
#         "label": "Inventor",
#         "properties": {
#         "name": "Venkatesh Sheregar",
#         "city": "Shanghai",
#         "country": "CN"
#         }
#     },
#     {
#         "name": "article2",
#         "label": "LegalClause",
#         "properties": {
#         "description": "Defines cosmetic products under Regulation 1223/2009."
#         }
#     },
#     {
#         "name": "PriorityRight",
#         "label": "LegalConcept",
#         "properties": {
#         "definition": "The right to claim the earliest filing date for the invention."
#         }
#     },
#     {
#         "name": "IncorporationByReference",
#         "label": "LegalConcept",
#         "properties": {
#         "definition": "Treating earlier applications as if fully included in the current application."
#         }
#     },
#     {
#         "name": "complex_mould",
#         "label": "ManufacturingProcess",
#         "properties": {
#         "description": "Mould needed to create horizontal recesses with sliders apart from the core."
#         }
#     },
#     {
#         "name": "moulding_core",
#         "label": "ManufacturingProcess",
#         "properties": {
#         "description": "Simplified mould core for single-piece container, no additional sliders needed."
#         }
#     },
#     {
#         "name": "US12161213B2",
#         "label": "Patent",
#         "properties": {
#         "documentId": "US 12161213 B2",
#         "publishedDate": "2024-12-10"
#         }
#     },
#     {
#         "name": "InternationalApplication",
#         "label": "PatentApplication",
#         "properties": {
#         "application_number": "PCT/EP2021/056493",
#         "jurisdiction": "European Patent Office",
#         "filing_date": "2021-03-15"
#         }
#     },
#     {
#         "name": "IndianApplication",
#         "label": "PatentApplication",
#         "properties": {
#         "application_number": "202021012636",
#         "jurisdiction": "India",
#         "filing_date": "2020-03-23"
#         }
#     },
#     {
#         "name": "FrenchApplication",
#         "label": "PatentApplication",
#         "properties": {
#         "application_number": "2008614",
#         "jurisdiction": "France",
#         "filing_date": "2020-08-21"
#         }
#     },
#     {
#         "name": "NationalPhase",
#         "label": "PatentProcessPhase",
#         "properties": {
#         "definition": "Stage where an international application is transitioned into national jurisdictions for patent granting."
#         }
#     },
#     {
#         "name": "usage_policy1",
#         "label": "Policy",
#         "properties": {
#         "description": "Reduce manufacturing cost and improve ergonomics of product packaging."
#         }
#     },
#     {
#         "name": "ergonomic_design",
#         "label": "Policy",
#         "properties": {
#         "objective": "Ease of manufacturing"
#         }
#     },
#     {
#         "name": "manufacturing_cost",
#         "label": "Policy",
#         "properties": {
#         "objective": "Cost reduction"
#         }
#     },
#     {
#         "name": "us_patent_6901937",
#         "label": "PriorArt",
#         "properties": {
#         "title": "Make-up case with pivotable lid and bottom",
#         "document_number": "US6901937B2"
#         }
#     },
#     {
#         "name": "us_patent_2002162565",
#         "label": "PriorArt",
#         "properties": {
#         "title": "Case for packaging a product with 360\u00b0 pivot lid",
#         "document_number": "US2002/0162565-A1"
#         }
#     },
#     {
#         "name": "cosmetic_product",
#         "label": "ProductType",
#         "properties": {
#         "definition": "As defined in Article 2 of Regulation No. 1223/2009 of the European Parliament and of the Council."
#         }
#     },
#     {
#         "name": "regulation1223_2009",
#         "label": "Regulation",
#         "properties": {
#         "name": "Regulation No. 1223/2009",
#         "source": "European Parliament and Council",
#         "date": "2009-11-30"
#         }
#     },
#     {
#         "name": "U-groove upper",
#         "label": "V-notch lower",
#         "properties": {}
#     }
#     ]

#         processed_Relationships = [
#             {
#                 "start": "US12161213B2",
#                 "end": "Varale; Aditya",
#                 "type": "inventedBy",
#                 "properties": {}
#             },
#             {
#                 "start": "L'Oreal Paris",
#                 "end": "US12161213B2",
#                 "type": "appliedFor",
#                 "properties": {
#                 "applicationDate": "2021-03-15"
#                 }
#             },
#             {
#                 "start": "L'Oreal Paris",
#                 "end": "US12161213B2",
#                 "type": "assignedTo",
#                 "properties": {}
#             },
#             {
#                 "start": "Application_17/906804",
#                 "end": "ForeignApplication_IN",
#                 "type": "hasPriorityClaim",
#                 "properties": {}
#             },
#             {
#                 "start": "Application_17/906804",
#                 "end": "ForeignApplication_FR",
#                 "type": "hasPriorityClaim",
#                 "properties": {}
#             },
#             {
#                 "start": "InternationalApplication",
#                 "end": "IndianApplication",
#                 "type": "claims_benefit_of",
#                 "properties": {
#                 "priority_date": "2020-03-23"
#                 }
#             },
#             {
#                 "start": "InternationalApplication",
#                 "end": "FrenchApplication",
#                 "type": "claims_benefit_of",
#                 "properties": {
#                 "priority_date": "2020-08-21"
#                 }
#             },
#             {
#                 "start": "IndianApplication",
#                 "end": "PriorityRight",
#                 "type": "establishes",
#                 "properties": {
#                 "date": "2020-03-23"
#                 }
#             },
#             {
#                 "start": "PriorityRight",
#                 "end": "ParisConvention",
#                 "type": "governed_by",
#                 "properties": {}
#             },
#             {
#                 "start": "InternationalApplication",
#                 "end": "NationalPhase",
#                 "type": "proceeds_to",
#                 "properties": {}
#             },
#             {
#                 "start": "InternationalApplication",
#                 "end": "IncorporationByReference",
#                 "type": "uses",
#                 "properties": {}
#             },
#             {
#                 "start": "invention1",
#                 "end": "cosmetic_product",
#                 "type": "targets",
#                 "properties": {}
#             },
#             {
#                 "start": "invention1",
#                 "end": "container1",
#                 "type": "utilizes",
#                 "properties": {}
#             },
#             {
#                 "start": "container1",
#                 "end": "packaging_device",
#                 "type": "includes",
#                 "properties": {}
#             },
#             {
#                 "start": "packaging_device",
#                 "end": "pivotable_cap",
#                 "type": "contains",
#                 "properties": {}
#             },
#             {
#                 "start": "pivotable_cap",
#                 "end": "horizontal_pins",
#                 "type": "features",
#                 "properties": {}
#             },
#             {
#                 "start": "horizontal_pins",
#                 "end": "complex_mould",
#                 "type": "formed_by",
#                 "properties": {}
#             },
#             {
#                 "start": "invention1",
#                 "end": "us_patent_6901937",
#                 "type": "references",
#                 "properties": {
#                 "relationship_type": "Prior Art"
#                 }
#             },
#             {
#                 "start": "cosmetic_product",
#                 "end": "article2",
#                 "type": "defined_by",
#                 "properties": {}
#             },
#             {
#                 "start": "article2",
#                 "end": "regulation1223_2009",
#                 "type": "part_of",
#                 "properties": {}
#             },
#             {
#                 "start": "regulation1223_2009",
#                 "end": "european_parliament",
#                 "type": "issued_by",
#                 "properties": {}
#             },
#             {
#                 "start": "regulation1223_2009",
#                 "end": "council_of_eu",
#                 "type": "issued_by",
#                 "properties": {}
#             },
#             {
#                 "start": "invention2",
#                 "end": "us_patent_2002162565",
#                 "type": "references",
#                 "properties": {
#                 "relationship_type": "Prior Art"
#                 }
#             },
#             {
#                 "start": "invention2",
#                 "end": "container2",
#                 "type": "utilizes",
#                 "properties": {}
#             },
#             {
#                 "start": "container2",
#                 "end": "hinge_platform",
#                 "type": "includes",
#                 "properties": {}
#             },
#             {
#                 "start": "container2",
#                 "end": "snapping_mechanism",
#                 "type": "includes",
#                 "properties": {}
#             },
#             {
#                 "start": "hinge_platform",
#                 "end": "snap_beads",
#                 "type": "contains",
#                 "properties": {}
#             },
#             {
#                 "start": "snapping_mechanism",
#                 "end": "moulding_core",
#                 "type": "formed_by",
#                 "properties": {}
#             },
#             {
#                 "start": "hinge_platform",
#                 "end": "pivoting",
#                 "type": "enables",
#                 "properties": {}
#             },
#             {
#                 "start": "snap_beads",
#                 "end": "snapping_force",
#                 "type": "provides",
#                 "properties": {}
#             },
#             {
#                 "start": "invention2",
#                 "end": "usage_policy1",
#                 "type": "complies_with",
#                 "properties": {}
#             },
#             {
#                 "start": "usage_policy1",
#                 "end": "container2",
#                 "type": "targets",
#                 "properties": {}
#             },
#             {
#                 "start": "snapping_member",
#                 "end": "fastening_members",
#                 "type": "includes",
#                 "properties": {}
#             },
#             {
#                 "start": "fastening_members",
#                 "end": "snap_beads",
#                 "type": "cooperate_with",
#                 "properties": {
#                 "position": "Opened"
#                 }
#             },
#             {
#                 "start": "snap_beads",
#                 "end": "vertical_groove",
#                 "type": "extends_along",
#                 "properties": {
#                 "height": "Entire base height"
#                 }
#             },
#             {
#                 "start": "snapping_member",
#                 "end": "hinge_platform",
#                 "type": "covers",
#                 "properties": {
#                 "position": "Opened"
#                 }
#             },
#             {
#                 "start": "snapping_member",
#                 "end": "ribs",
#                 "type": "protrudes",
#                 "properties": {
#                 "axis": "Transversal"
#                 }
#             },
#             {
#                 "start": "ribs",
#                 "end": "de-snapping",
#                 "type": "resist",
#                 "properties": {}
#             },
#             {
#                 "start": "hinge_platform",
#                 "end": "hinge_types",
#                 "type": "contains",
#                 "properties": {}
#             },
#             {
#                 "start": "hinge_types",
#                 "end": "v_notch",
#                 "type": "includes",
#                 "properties": {}
#             },
#             {
#                 "start": "hinge_types",
#                 "end": "u_groove",
#                 "type": "includes",
#                 "properties": {}
#             },
#             {
#                 "start": "cover",
#                 "end": "base",
#                 "type": "delimited_by",
#                 "properties": {
#                 "alignment": "Front and rear walls"
#                 }
#             },
#             {
#                 "start": "cut_out",
#                 "end": "scraping",
#                 "type": "prevents",
#                 "properties": {
#                 "movement": "Rotation"
#                 }
#             },
#             {
#                 "start": "angles",
#                 "end": "hinge_positions",
#                 "type": "defines",
#                 "properties": {}
#             },
#             {
#                 "start": "snapping_member",
#                 "end": "ergonomic_design",
#                 "type": "aligns_with",
#                 "properties": {}
#             },
#             {
#                 "start": "snapping_member",
#                 "end": "manufacturing_cost",
#                 "type": "reduces",
#                 "properties": {}
#             }
# ]

        

        # We need to convert it into the right format.
        formatted_results = getNodesAndRelationshipsFromResult(test_data)

        # Run the processing
        result = disambiguation.run(formatted_results)

        cypher_script = data_to_cypher(result)

        print("The Cypher Script : ", cypher_script)

        # Print the results
        print("\nProcessed Nodes:")
        print(json.dumps(result["nodes"], indent=2))
        print("\nProcessed Relationships:")
        print(json.dumps(result["relationships"], indent=2))
        print("\nProcessing Chunks:")
        print(json.dumps(result["chunks"], indent=2))








        