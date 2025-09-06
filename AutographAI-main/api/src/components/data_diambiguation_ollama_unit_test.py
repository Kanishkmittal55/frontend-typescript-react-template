import unittest
import json

# Assuming the function and global variables are imported from your code file
from components.data_disambiguation_ollama import (
    run_disambiguation,
    add_to_global_registry,
    global_nodes_registry,  # Import the global registry variables
    global_relationships_registry,
)

# The patch function in unittest.moc is used to temporarily replace an object or a function with a mock during the execution of a test.
from unittest.mock import patch

# # Mock global registries
# global_nodes_registry = {}
# global_relationships_registry = set()

class TestAddToGlobalRegistry(unittest.TestCase):


    def setUp(self):
        """
        Reset global registries before each test.
        """
        global global_nodes_registry, global_relationships_registry
        global_nodes_registry.clear()  # Clears the dictionary, retaining the reference
        global_relationships_registry.clear()  # Clears the set, retaining the reference


    def test_add_specific_nodes(self):
        """
        Test adding specific nodes to the global registry.
        """
        nodes = [
            {"name": "alice", "label": "Person", "properties": {"age": 30, "occupation": "lawyer"}},
            {"name": "bob", "label": "Person", "properties": {"age": 25, "occupation": "journalist"}},
        ]
        
        add_to_global_registry(nodes, [])

         # Log the global registry
        # print("Global Nodes Registry is :", global_nodes_registry)

        # Check the global registry size and specific entries
        self.assertEqual(len(global_nodes_registry), 2)
        self.assertIn("alice", global_nodes_registry)
        self.assertIn("bob", global_nodes_registry)
        self.assertEqual(global_nodes_registry["alice"]["properties"]["occupation"], "lawyer")
        self.assertEqual(global_nodes_registry["bob"]["properties"]["age"], 25)

    def test_add_duplicate_nodes(self):
        """
        Test that adding duplicate nodes does not overwrite existing ones.
        """
        global_nodes_registry["alice"] = {"name": "alice", "label": "Person", "properties": {"age": 30, "occupation": "lawyer"}}
        
        nodes = [
            {"name": "alice", "label": "Person", "properties": {"age": 35, "occupation": "lawyer"}},  # Duplicate entry
            {"name": "charlie", "label": "Person", "properties": {"age": 40, "occupation": "engineer"}},
        ]
        add_to_global_registry(nodes, [])

        # Check the global registry size and ensure no overwrites
        self.assertEqual(len(global_nodes_registry), 2)
        self.assertEqual(global_nodes_registry["alice"]["properties"]["age"], 30)  # Original value retained
        self.assertIn("charlie", global_nodes_registry)
        self.assertEqual(global_nodes_registry["charlie"]["label"], "Person")

    def test_add_specific_relationships(self):
        """
        Test adding specific relationships to the global registry.
        """
        relationships = [
            {"start": "alice", "type": "friend", "end": "bob", "properties": {"since": "2020"}},
            {"start": "bob", "type": "works_with", "end": "charlie", "properties": {"project": "Apollo"}},
        ]
        add_to_global_registry([], relationships)

        # Check the global registry size and specific entries
        self.assertEqual(len(global_relationships_registry), 2)
        self.assertIn(
            ("alice", "friend", "bob", json.dumps({"since": "2020"})),
            global_relationships_registry,
        )
        self.assertIn(
            ("bob", "works_with", "charlie", json.dumps({"project": "Apollo"})),
            global_relationships_registry,
        )

    def test_add_duplicate_relationships(self):
        """
        Test that adding duplicate relationships does not create duplicates.
        """
        global_relationships_registry.add(
            ("alice", "friend", "bob", json.dumps({"since": "2020"}))
        )

        relationships = [
            {"start": "alice", "type": "friend", "end": "bob", "properties": {"since": "2020"}},  # Duplicate entry
            {"start": "bob", "type": "mentor", "end": "charlie", "properties": {"duration": "2 years"}},
        ]
        add_to_global_registry([], relationships)

        # Check the global registry size and ensure no duplicates
        self.assertEqual(len(global_relationships_registry), 2)
        self.assertIn(
            ("alice", "friend", "bob", json.dumps({"since": "2020"})),
            global_relationships_registry,
        )
        self.assertIn(
            ("bob", "mentor", "charlie", json.dumps({"duration": "2 years"})),
            global_relationships_registry,
        )

class TestRunDisambiguation(unittest.TestCase):
    def setup(self):
        """
        Reset the global registries before each test.
        """
        global_nodes_registry.clear()
        global_relationships_registry.clear()

    @patch("components.data_disambiguation_ollama.process")
    def test_unique_nodes_and_relationships(self, mock_process):
        """
        Test with unique nodes and relationships.
        """
        mock_process.side_effect = lambda x: x  # Mock the process function to return input data directly.

        data = {
            "nodes": [
                {"name": "alice", "label": "Person", "properties": {"age": 30}},
                {"name": "bob", "label": "Person", "properties": {"age": 25}},
                {"name": "kanishk", "label": "Person", "properties": {"age": 30}},
                {"name": "ferrari", "label": "car", "properties": {"color": "red"}},
                {"name": "lamborgini", "label": "car", "properties": {"color": "grey"}},
                {"name": "macbook", "label": "laptop", "properties": {"ram": "25gb"}},
                {"name": "dell xps", "label": "laptop", "properties": {"ram": "30gb"}},
                {"name": "lenovo yogabook", "label": "laptop", "properties": {"ram": "40gb"}},
            ],
            "relationships": [
                {"start": "alice", "type": "friend", "end": "bob", "properties": {"since": "2020"}}
            ],
        }

        result = run_disambiguation(data)

        print("The result is : ", result)

        # Verify nodes and relationships in the result
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(len(result["relationships"]), 1)
        self.assertIn("alice", global_nodes_registry)
        self.assertIn("bob", global_nodes_registry)
        self.assertIn(
            ("alice", "friend", "bob", json.dumps({"since": "2020"})),
            global_relationships_registry,
        )

    # @patch("components.data_disambiguation_ollama.process")
    # def test_duplicate_nodes_and_relationships(self, mock_process):
    #     """
    #     Test with duplicate nodes and relationships.
    #     """
    #     mock_process.side_effect = lambda x: x  # Mock the process function to return input data directly.

    #     data = {
    #         "nodes": [
    #             {"name": "alice", "label": "Person", "properties": {"age": 30}},
    #             {"name": "alice", "label": "Person", "properties": {"age": 35}},
    #             {"name": "bob", "label": "Person", "properties": {"age": 25}},
    #         ],
    #         "relationships": [
    #             {"start": "alice", "type": "friend", "end": "bob", "properties": {"since": "2020"}},
    #             {"start": "alice", "type": "friend", "end": "bob", "properties": {"since": "2021"}},
    #         ],
    #     }

    #     result = run_disambiguation(data)

    #     # Verify that duplicates are handled correctly
    #     self.assertEqual(len(result["nodes"]), 2)  # Only unique nodes
    #     self.assertEqual(len(result["relationships"]), 1)  # Only unique relationships
    #     self.assertIn("alice", global_nodes_registry)
    #     self.assertIn("bob", global_nodes_registry)

    # @patch("components.data_disambiguation_ollama.process")
    # def test_disambiguation_logic(self, mock_process):
    #     """
    #     Test with disambiguation logic to merge nodes.
    #     """
    #     mock_process.side_effect = lambda x: (
    #         '[["alice", "Person", {"age": 30}], ["bob", "Person", {"age": 25}]]'
    #     ) if "alice" in x else x

    #     data = {
    #         "nodes": [
    #             {"name": "alice", "label": "Person", "properties": {"age": 35}},
    #             {"name": "bob", "label": "Person", "properties": {"age": 25}},
    #         ],
    #         "relationships": [
    #             {"start": "alice", "type": "friend", "end": "bob", "properties": {"since": "2020"}}
    #         ],
    #     }

    #     result = run_disambiguation(data)

    #     # Verify that disambiguation was applied
    #     self.assertEqual(len(result["nodes"]), 2)
    #     self.assertEqual(global_nodes_registry["alice"]["properties"]["age"], 30)

    # @patch("components.data_disambiguation_ollama.process")
    # def test_empty_input(self, mock_process):
    #     """
    #     Test with empty input data.
    #     """
    #     mock_process.side_effect = lambda x: x  # Mock the process function to return input data directly.

    #     data = {"nodes": [], "relationships": []}
    #     result = run_disambiguation(data)

    #     # Verify that the result is empty
    #     self.assertEqual(len(result["nodes"]), 0)
    #     self.assertEqual(len(result["relationships"]), 0)
    #     self.assertEqual(len(global_nodes_registry), 0)
    #     self.assertEqual(len(global_relationships_registry), 0)

    # @patch("components.data_disambiguation_ollama.process")
    # def test_mixed_valid_and_invalid_relationships(self, mock_process):
    #     """
    #     Test with mixed valid and invalid relationships.
    #     """
    #     mock_process.side_effect = lambda x: x  # Mock the process function to return input data directly.

    #     data = {
    #         "nodes": [
    #             {"name": "alice", "label": "Person", "properties": {"age": 30}},
    #             {"name": "bob", "label": "Person", "properties": {"age": 25}},
    #         ],
    #         "relationships": [
    #             {"start": "alice", "type": "friend", "end": "bob", "properties": {"since": "2020"}},
    #             {"start": "alice", "type": "friend", "end": "unknown", "properties": {"since": "2021"}},
    #         ],
    #     }

    #     result = run_disambiguation(data)

    #     # Verify that only valid relationships are added
    #     self.assertEqual(len(result["relationships"]), 1)
    #     self.assertIn(
    #         ("alice", "friend", "bob", json.dumps({"since": "2020"})),
    #         global_relationships_registry,
    #     )
    #     self.assertNotIn(
    #         ("alice", "friend", "unknown", json.dumps({"since": "2021"})),
    #         global_relationships_registry,
    #     )


if __name__ == "__main__":
    unittest.main()
