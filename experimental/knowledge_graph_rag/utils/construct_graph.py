import os
import concurrent.futures
import json
import time
from typing import List, Dict, Any, Tuple
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import networkx as nx
import pandas as pd
from pyvis.network import Network


def get_repo_structure(path: str) -> Dict[str, Any]:
    """
    Recursively get the structure of the repository.
    """
    if not os.path.isdir(path):
        return None
    
    result = {'name': os.path.basename(path), 'path': path}
    if os.path.isdir(path):
        result['type'] = "directory"
        result['children'] = [get_repo_structure(os.path.join(path, x)) for x in os.listdir(path)]
        result['children'] = [x for x in result['children'] if x is not None]
    else:
        result['type'] = "file"
    return result


def print_repo_structure(structure: Dict[str, Any], indent: str = "") -> None:
    """
    Print the repository structure in a readable format.
    """
    print(f"{indent}{structure['name']}/")
    if structure['type'] == "directory":
        for child in structure['children']:
            print_repo_structure(child, indent + "  ")
            
            
def save_checkpoint(triples: List[Dict[str, str]], processed_files: List[str], checkpoint_file: str):
    """Save the current state of extracted triples and processed files to a single JSON file."""
    with open(checkpoint_file, 'w') as f:
        json.dump({"triples": triples, "processed_files": processed_files}, f)
    print(f"Checkpoint saved: {checkpoint_file}")


def load_checkpoint(checkpoint_file: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Load the checkpoint from a single JSON file."""
    if not os.path.exists(checkpoint_file):
        print("No existing checkpoint found. Starting from scratch.")
        return [], []
    print(f"Loading checkpoint: {checkpoint_file}")
    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
    return data.get("triples", []), data.get("processed_files", [])


def process_document(doc: str, file_path: str, repo_structure: Dict[str, Any], llm: Any) -> List[Dict[str, str]]:
    """
    Process a single document and extract triples, incorporating file path information.
    """
    repo_path = repo_structure['path']
    relative_path = os.path.relpath(file_path, repo_path)

    prompt = f"""
    Analyze the following AV testing code from the file {relative_path}:

    {doc.page_content}

    Extract key information as triples, focusing primarily on EVALUATORS and their relationships to features, test cases, and AV components. 
    Pay special attention to the folder structure as it may indicate features.

    Entities should be classified into categories such as:
    EVALUATOR, TEST_CASE, FEATURE, AV_COMPONENT, FUNCTION, CLASS, PARAMETER, DYNAMIC_CHANGE

    Relationships should be represented by verbs such as:
    Evaluates, Tests, Implements, DependsOn, Uses, Configures, Simulates, Validates, Changes

    For each EVALUATOR found, try to identify:
    1. Which feature or AV component it evaluates
    2. Which test cases use this evaluator
    3. Any specific parameters or dynamic changes it handles

    Return only a Python list of tuples, where each tuple is structured as:
    ('subject', 'subject_type', 'relation', 'object', 'object_type')

    Always include at least one triple that relates the file to its location in the repository, and try to infer the feature from the folder structure.
    """


    try:
        response = llm.invoke(prompt)
        print(f"Raw response type for {relative_path}: {type(response)}")  # Debug print
        
        # Extract content from ChatMessage object
        if hasattr(response, 'content'):
            content = response.content
        elif isinstance(response, dict) and 'content' in response:
            content = response['content']
        else:
            print(f"Unexpected response format for {relative_path}: {response}")
            return []

        print(f"Content for {relative_path}: {content}")  # Debug print
        
        # Extract the list from the content
        triples_str = content.strip()
        if triples_str.startswith("```python"):
            triples_str = triples_str.split("```python")[1]
        if triples_str.endswith("```"):
            triples_str = triples_str[:-3]
        
        triples = eval(triples_str)
        if not isinstance(triples, list):
            raise ValueError("Response is not a list")
        return triples
    except Exception as e:
        print(f"Error processing file {relative_path}: {e}")
        return []


def process_documents(directory: str, repo_structure: Dict[str, Any], llm: Any, checkpoint_file: str) -> List[Dict[str, str]]:
    """Process all documents in the given directory and extract triples, with checkpointing."""

    loader = DirectoryLoader(directory, glob="**/*.py", loader_cls=TextLoader)
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_docs)
    
    # Load the latest checkpoint
    all_triples, processed_files = load_checkpoint(checkpoint_file)
    
    for doc in documents:
        try:
            if doc.metadata['source'] in processed_files:
                # print(f"Skipping already processed file: {doc.metadata['source']}")
                continue  # Skip already processed files
            
            triples = process_document(doc, doc.metadata['source'], repo_structure, llm)
            all_triples.extend(triples)
            processed_files.append(doc.metadata['source'])
            print(f"Processed {doc.metadata['source']}, extracted {len(triples)} triples")
            
            # Save checkpoint after each document
            save_checkpoint(all_triples, processed_files, checkpoint_file)
        except Exception as e:
            print(f"Error processing document {doc.metadata.get('source', 'Unknown')}: {str(e)}")
            # Optionally, you can save a checkpoint here to preserve progress up to the error
    
    # Save final checkpoint
    save_checkpoint(all_triples, processed_files, checkpoint_file)
    
    return all_triples


def create_knowledge_graph(triples: List[Dict[str, str]], repo_structure: Dict[str, Any]) -> nx.DiGraph:
    """
    Create a knowledge graph from the extracted triples and repository structure.
    """
    G = nx.DiGraph()
    skipped_triples = 0
    
    for i, triple in enumerate(triples):
        try:
            if isinstance(triple, (list, tuple)):
                if len(triple) >= 5:
                    subject, subject_type, relation, obj, object_type, *extra = triple
                    # Handle class imports
                    if len(extra) == 1 and extra[0] in ['pacsim']:
                        obj = f"{extra[0]}.{obj}"
                    # Handle file locations
                    elif len(extra) > 1 and 'DIRECTORY' in extra:
                        obj = '/'.join([obj] + extra[::2])  # Combine path elements
                        object_type = 'FILE'
                    # Handle the special 'Transforms' case
                    elif relation == 'Transforms' and len(extra) == 3 and extra[1] == 'To':
                        obj = f"{obj} to {extra[2]}"
                        object_type = 'TRANSFORMATION'
                elif len(triple) < 5:
                    print(f"Skipping triple with insufficient elements: {triple}")
                    skipped_triples += 1
                    continue
            else:
                print(f"Skipping triple with unexpected type: {type(triple)}")
                skipped_triples += 1
                continue

            # Replace None with a placeholder string
            subject = 'Unknown' if subject is None else str(subject)
            obj = 'Unknown' if obj is None else str(obj)
            subject_type = 'Unknown' if subject_type is None else str(subject_type)
            object_type = 'Unknown' if object_type is None else str(object_type)

            # Add nodes and edge
            G.add_node(subject, type=subject_type)
            G.add_node(obj, type=object_type)
            G.add_edge(subject, obj, relation=relation)
        except Exception as e:
            print(f"Error processing triple {i}: {triple}")
            print(f"Error message: {str(e)}")
            skipped_triples += 1

    print(f"Processed {len(triples)} triples. Skipped {skipped_triples} problematic triples.")
    
    # Add repository structure to the graph
    def add_structure_to_graph(structure, parent=None):
        G.add_node(structure['name'], type='DIRECTORY' if structure['type'] == 'directory' else 'FILE')
        if parent:
            G.add_edge(parent, structure['name'], relation='Contains')
        if structure['type'] == 'directory':
            for child in structure['children']:
                add_structure_to_graph(child, structure['name'])
    
    add_structure_to_graph(repo_structure)
    
    return G



def visualize_graph(G: nx.DiGraph, output_file: str):
    """Visualize the graph using pyvis and save as an HTML file."""
    net = Network(notebook=True, width="100%", height="800px", directed=True)
    
    # Add nodes
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'Unknown')
        color = "#97c2fc" if node_type in ['DIRECTORY', 'FILE'] else "#ffaa00"
        net.add_node(node, label=f"{node}\n({node_type})", color=color)
    
    # Add edges
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, title=data.get('relation', ''))
    
    # Set physics layout
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      }
    }
    """)
    
    net.save_graph(output_file)
    print(f"Graph visualization saved to {output_file}")


def generate_graphml_from_checkpoint(checkpoint_file: str, repo_path: str, output_dir: str):
    """Generate GraphML and visualization from a checkpoint file."""
    triples, processed_files = load_checkpoint(checkpoint_file)
    if not triples:
        print("No triples found in the checkpoint.")
        return

    print(f"Loaded {len(triples)} triples and {len(processed_files)} processed files from checkpoint.")
    
    # Print detailed information about triples
    print("Detailed triple information:")
    triple_types = {}
    for i, triple in enumerate(triples):
        triple_type = type(triple).__name__
        triple_types[triple_type] = triple_types.get(triple_type, 0) + 1
        if i < 10:  # Print details for first 10 triples
            print(f"Triple {i}:")
            print(f"  Type: {triple_type}")
            print(f"  Length: {len(triple)}")
            print(f"  Content: {triple}")
    print("\nTriple type summary:")
    for t_type, count in triple_types.items():
        print(f"  {t_type}: {count}")
    print()

    # Get repo structure
    repo_structure = get_repo_structure(repo_path)

    # Create knowledge graph
    G = create_knowledge_graph(triples, repo_structure)

    # Print graph statistics
    print("\nGraph statistics:")
    print(f"  Number of nodes: {G.number_of_nodes()}")
    print(f"  Number of edges: {G.number_of_edges()}")
    print("  Node types:")
    node_types = {}
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'Unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    for n_type, count in node_types.items():
        print(f"    {n_type}: {count}")
    print("  Edge types:")
    edge_types = {}
    for _, _, data in G.edges(data=True):
        edge_type = data.get('relation', 'Unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    for e_type, count in edge_types.items():
        print(f"    {e_type}: {count}")

    # Save as GraphML
    graphml_file = os.path.join(output_dir, "pacsim_knowledge_graph_from_checkpoint.graphml")
    nx.write_graphml(G, graphml_file)
    print(f"\nGraphML saved to {graphml_file}")

    # Visualize
    html_file = os.path.join(output_dir, "pacsim_knowledge_graph_from_checkpoint.html")
    visualize_graph(G, html_file)
    print(f"Graph visualization saved to {html_file}")


def save_graph_data(G: nx.DiGraph, output_dir: str) -> None:
    """
    Save the graph data as CSV files and GraphML.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save nodes
    nodes_data = [(node, attr['type']) for node, attr in G.nodes(data=True)]
    nodes_df = pd.DataFrame(nodes_data, columns=['entity', 'type'])
    nodes_df.to_csv(os.path.join(output_dir, 'entities.csv'), index=False)

    # Save edges (relations)
    edges_data = [(u, v, attr['relation']) for u, v, attr in G.edges(data=True)]
    edges_df = pd.DataFrame(edges_data, columns=['source', 'target', 'relation'])
    edges_df.to_csv(os.path.join(output_dir, 'relations.csv'), index=False)

    # Save as GraphML
    nx.write_graphml(G, os.path.join(output_dir, 'av_test_knowledge_graph.graphml'))


def main():
    # Set up the NVIDIA AI Endpoints LLM
    llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")  
    
    repo_path = "/home/april/av-llm/GenerativeAIExamples/experimental/knowledge_graph_rag/ndas-refs_heads_main-tests/behaviorplanner"
    output_dir = "/home/april/av-llm/GenerativeAIExamples/experimental/knowledge_graph_rag" 
    checkpoint_file = os.path.join(output_dir, "pacsim_checkpoint.json")
    
    # Generate GraphML from checkpoint
    # generate_graphml_from_checkpoint(checkpoint_file, repo_path, output_dir)

    # If you want to process documents and create a new checkpoint, uncomment the following lines:
    repo_structure = get_repo_structure(repo_path)
    print("Repository Structure:")
    print_repo_structure(repo_structure)
    triples = process_documents(repo_path, repo_structure, llm, checkpoint_file)
    print(f"Total triples extracted: {len(triples)}")
    
    if triples:
        G = create_knowledge_graph(triples, repo_structure)
        graphml_file = os.path.join(output_dir, "knowledge_graph.graphml")
        nx.write_graphml(G, graphml_file)
        print(f"GraphML saved to {graphml_file}")
        
        html_file = os.path.join(output_dir, "knowledge_graph.html")
        visualize_graph(G, html_file)
    else:
        print("No triples extracted. Cannot create knowledge graph.")

    
if __name__ == "__main__":
    main()