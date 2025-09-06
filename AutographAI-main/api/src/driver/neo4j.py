from typing import Any, Dict, List, Optional
import json
import traceback

from neo4j import GraphDatabase, exceptions

node_properties_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "node"
WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
RETURN {labels: nodeLabels, properties: properties} AS output

"""

rel_properties_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
RETURN {type: nodeLabels, properties: properties} AS output
"""

rel_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE type = "RELATIONSHIP" AND elementType = "node"
RETURN "(:" + label + ")-[:" + property + "]->(:" + toString(other[0]) + ")" AS output
"""


def schema_text(node_props, rel_props, rels) -> str:
    return f"""
  This is the schema representation of the Neo4j database.
  Node properties are the following:
  {node_props}
  Relationship properties are the following:
  {rel_props}
  The relationships are the following
  {rels}
  """


class Neo4jDatabase:
    def __init__(
        self,
        host: str = "bolt://127.0.0.1:7687", # For the driver setup we need to use the localhost , only in main.py where we call the docker container running the neo4j we need to use the container name.
        user: str = "neo4j",
        password: str = "your12345",
        database: str = "neo4j",
        read_only: bool = True,
    ) -> None:
        """Initialize a neo4j database"""
        self._driver = GraphDatabase.driver(host, auth=(user, password))
        self._database = database
        self._read_only = read_only
        self.schema = ""
        # Verify connection
        try:
            self._driver.verify_connectivity()
        except exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the url is correct"
            )
        except exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the username and password are correct"
            )
        try:
            self.refresh_schema()
        except Exception as e:
             print("Error encountered while refreshing the schema. Below is the stack trace:")
             traceback.print_exc()  # prints a detailed stack trace
             raise ValueError("Missing APOC Core plugin") from e

    @staticmethod
    def _execute_read_only_query(tx, cypher_query: str, params: Optional[Dict] = {}):
        result = tx.run(cypher_query, params)
        return [r.data() for r in result]

    def query(
        self, cypher_query: str, params: Optional[Dict] = {}
    ) -> List[Dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            try:
                if self._read_only:
                    result = session.read_transaction(
                        self._execute_read_only_query, cypher_query, params
                    )
                    return result
                else:
                    result = session.run(cypher_query, params)
                    # Limit to at most 10 results
                    return [r.data() for r in result]

            # Catch Cypher syntax errors
            except exceptions.CypherSyntaxError as e:
                return [
                    {
                        "code": "invalid_cypher",
                        "message": f"Invalid Cypher statement due to an error: {e}",
                    }
                ]

            except exceptions.ClientError as e:
                # Catch access mode errors
                if e.code == "Neo.ClientError.Statement.AccessMode":
                    return [
                        {
                            "code": "error",
                            "message": "Couldn't execute the query due to the read only access to Neo4j",
                        }
                    ]
                else:
                    return [{"code": "error", "message": e}]

    def refresh_schema(self) -> None:
        node_props = [el["output"] for el in self.query(node_properties_query)]
        rel_props = [el["output"] for el in self.query(rel_properties_query)]
        rels = [el["output"] for el in self.query(rel_query)]
        schema = schema_text(node_props, rel_props, rels)
        self.schema = schema
        print(schema)

    def check_if_empty(self) -> bool:
        data = self.query(
            """
        MATCH (n)
        WITH count(n) as c
        RETURN CASE WHEN c > 0 THEN true ELSE false END AS output
        """
        )
        return data[0]["output"]
    
    def insert_patent_data(self, json_object: Dict[str, Any]) -> None:
        queries = []

        print("Received JSON object:", json_object)

        # Validate all required keys
        for key in ["patent_no", "cpcc_codes", "inventor_names", "assignee", "properties"]:
            if key not in json_object:
                print(f"Missing key in JSON object: {key}")
                raise ValueError(f"Key '{key}' is missing in the JSON object")

        # Extract properties from JSON
        properties = json_object.get("properties", {})
        functional_roles = properties.get("functional_roles", {})
        if not functional_roles:
            print("Error: 'functional_roles' key is missing or empty in 'properties'")
            raise ValueError("Key 'functional_roles' is missing or empty in 'properties'")

        patent_no = json_object["patent_no"]
        product_name = json_object["properties"]["product_name"]
        description = json_object["properties"]["description"]
        cpcc_codes = json_object["cpcc_codes"]
        inventors = json_object["inventor_names"]
        assignee = json_object["assignee"]
        product_type = json_object["type"]

        print(f"Patent Number: {patent_no}")
        print(f"Product Name: {product_name}")
        print(f"Description: {description}")
        print(f"Product Type: {product_type}")

        # Queries
        queries = []

        # Create or update head_patents node
        queries.append(
            """
            MERGE (head:patents {d_type: "patents"})
            ON CREATE SET head.length = 0
            SET head.length = head.length + 1
            """
        )

        assignee = assignee.replace("'", "")  # Escaping single quotes for Cypher compatibility

        # Create individual patent node and link to head_patents
        queries.append(
            f"""
            MERGE (patent:patent {{
                aa_patent_no: '{patent_no}',
                inventor_names: {json.dumps(inventors)},
                cpcc_codes: {json.dumps(cpcc_codes)},
                the_assignee: '{assignee}'
            }})
            ON CREATE SET patent.created_at = timestamp()
            WITH patent
            MATCH (head:patents {{d_type: "patents"}})
            MERGE (head)-[:HAS]->(patent)
            """
        )

        # Explicit check query to ensure patent exists
        check_query = f"""
        MATCH (patent:patent {{aa_patent_no: '{patent_no}'}})
        RETURN COUNT(patent) AS patent_count
        """

        # Execute queries sequentially
        for query in queries:
            try:
                print("Executing query:", query)
                result = self.query(query)
                print(f"Query executed successfully. Result: {result}")
            except Exception as e:
                print(f"Error executing query: {query}\nError: {e}")
                return  # Stop execution if a query fails

        # Explicitly check if the patent node was created
        try:
            print("Executing check query to ensure patent exists:", check_query)
            check_result = self.query(check_query)
            if check_result[0]["patent_count"] > 0:
                print(f"Patent node creation confirmed. Patent Number: {patent_no}")
            else:
                print(f"Patent node creation failed. Patent Number: {patent_no}")
                return  # Stop execution if patent node is not found
        except Exception as e:
            print(f"Error executing check query: {check_query}\nError: {e}")
            return

        # Create product node and link to patent
        try:
            product_query = f"""
            MERGE (product:product {{
                aa_product_name: '{product_name}', 
                description: '{description}',
                product_type: '{product_type}'
            }})
            ON CREATE
                SET product.created_at = timestamp()
            WITH product
            MATCH (patent:patent {{aa_patent_no: '{patent_no}'}})
            MERGE (patent)-[:PROTECTS]->(product)
            """
            print("Executing product query:", product_query)
            product_result = self.query(product_query)
            print(f"Product query executed successfully. Result: {product_result}")
        except Exception as e:
            print(f"Error executing product query: {e}")
            return


        # Create functional_role nodes and link to product
        for role, chemicals_list in functional_roles.items():
            role_name = role.lower()  # Normalize role name

            try:
                role_query = f"""
                MERGE (role:functional_role {{name: '{role_name}'}})
                MERGE (product:product {{aa_product_name: '{product_name}'}})
                MERGE (product)-[of:OF]->(role)
                """
                print("Executing role query:", role_query)
                role_result = self.query(role_query)
                print(f"Role query executed successfully. Result: {role_result}")
            except Exception as e:
                print(f"Error executing role query for role '{role_name}': {e}")
                return

            # Create chemical nodes and link to functional_role
            for chem in chemicals_list:
                try:
                    chemical_name = chem["chemical"].lower()  # Normalize chemical name
                    weight = chem.get("weight", "null")  # Extract weight (if available)

                    chemical_query = f"""
                    MERGE (chemical:chemical {{name: '{chemical_name}'}})
                    MERGE (product:product {{aa_product_name: '{product_name}'}})
                    MERGE (product)-[:CONTAINS {{functional_role: '{role_name}', weight: '{weight}'}}]->(chemical)
                    """
                    print("Executing chemical query:", chemical_query)
                    chemical_result = self.query(chemical_query)
                    print(f"Chemical query executed successfully. Result: {chemical_result}")
                except Exception as e:
                    print(f"Error executing chemical query for chemical '{chemical_name}': {e}")
                    return

    def insert_real_world_product(self, json_object: Dict[str, Any]) -> None:
            queries = []

            print("Received JSON object:", json_object)

            # Validate all required keys
            for key in ["website_name", "properties"]:
                if key not in json_object:
                    print(f"Missing key in JSON object: {key}")
                    raise ValueError(f"Key '{key}' is missing in the JSON object")

            # Extract properties from JSON
            properties = json_object.get("properties", {})
            functional_roles = properties.get("functional_roles", {})
            if not functional_roles:
                print("Error: 'functional_roles' key is missing or empty in 'properties'")
                raise ValueError("Key 'functional_roles' is missing or empty in 'properties'")

            website_name = json_object["website_name"]
            product_name = json_object["properties"]["product_name"]
            description = json_object["properties"]["description"]
            product_type = json_object["type"]

            print(f"Website Name: {website_name}")
            print(f"Product Name: {product_name}")
            print(f"Description: {description}")
            print(f"Product Type: {product_type}")

            # Queries
            queries = []

            # Create or update head_websites node
            queries.append(
                """
                MERGE (head:websites {d_type: "websites"})
                ON CREATE SET head.length = 0
                SET head.length = head.length + 1
                """
            )

            website_name = website_name.replace("'", "")  # Escaping single quotes for Cypher compatibility

            # Create individual website node and link to head_websites
            queries.append(
                f"""
                MERGE (website:website {{
                    name: '{website_name}'
                }})
                ON CREATE SET website.created_at = timestamp()
                WITH website
                MATCH (head:websites {{d_type: "websites"}})
                MERGE (head)-[:HAS]->(website)
                """
            )

            # Execute queries sequentially
            for query in queries:
                try:
                    print("Executing query:", query)
                    result = self.query(query)
                    print(f"Query executed successfully. Result: {result}")
                except Exception as e:
                    print(f"Error executing query: {query}\nError: {e}")
                    return  # Stop execution if a query fails

            # Create product node and link to website
            try:
                product_query = f"""
                MERGE (product:product {{
                    aa_product_name: '{product_name}', 
                    description: '{description}',
                    product_type: '{product_type}'
                }})
                ON CREATE
                    SET product.created_at = timestamp()
                WITH product
                MATCH (website:website {{name: '{website_name}'}})
                MERGE (website)-[:OFFERS]->(product)
                """
                print("Executing product query:", product_query)
                product_result = self.query(product_query)
                print(f"Product query executed successfully. Result: {product_result}")
            except Exception as e:
                print(f"Error executing product query: {e}")
                return

            # Create functional_role nodes and link to product
            for role, chemicals_list in functional_roles.items():
                role_name = role.lower()  # Normalize role name

                try:
                    role_query = f"""
                    MERGE (role:functional_role {{name: '{role_name}'}})
                    MERGE (product:product {{aa_product_name: '{product_name}'}})
                    MERGE (product)-[of:OF]->(role)
                    """
                    print("Executing role query:", role_query)
                    role_result = self.query(role_query)
                    print(f"Role query executed successfully. Result: {role_result}")
                except Exception as e:
                    print(f"Error executing role query for role '{role_name}': {e}")
                    return

                # Create chemical nodes and link to functional_role
                for chem in chemicals_list:
                    try:
                        chemical_name = chem["chemical"].lower()  # Normalize chemical name
                        weight = chem.get("weight", "null")  # Extract weight (if available)

                        chemical_query = f"""
                        MERGE (chemical:chemical {{name: '{chemical_name}'}})
                        MERGE (product:product {{aa_product_name: '{product_name}'}})
                        MERGE (product)-[:CONTAINS {{functional_role: '{role_name}', weight: '{weight}'}}]->(chemical)
                        """
                        print("Executing chemical query:", chemical_query)
                        chemical_result = self.query(chemical_query)
                        print(f"Chemical query executed successfully. Result: {chemical_result}")
                    except Exception as e:
                        print(f"Error executing chemical query for chemical '{chemical_name}': {e}")
                        return

