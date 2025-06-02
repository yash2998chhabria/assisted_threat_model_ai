import os
import ast
import json
import networkx as nx

def find_python_files(repo_root):
    python_files = []
    for root, _, files in os.walk(repo_root):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def parse_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        return ast.parse(source), source
    except SyntaxError as e:
        print(f"Skipping {file_path}: {e}")
        return None, None

def process_ast(tree, source_code, file_path, root_dir, graph):
    chunks = []
    relative_path = os.path.relpath(file_path, root_dir)

    # Map for function/class names in this file
    local_defs = set()

    for node in ast.walk(tree):
        # Extract chunks (functions and classes)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
            local_defs.add(name)
            chunk = {
                'name': name,
                'type': type(node).__name__,
                'lineno': node.lineno,
                'file': relative_path,
                'docstring': ast.get_docstring(node),
                'source': ast.get_source_segment(source_code, node)
            }
            chunks.append(chunk)
            graph.add_node(name, type=type(node).__name__, file=relative_path)

        # Capture calls: FunctionDef or ClassDef as parent
        if isinstance(node, ast.FunctionDef):
            caller = node.name
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func_name = None
                    if isinstance(child.func, ast.Name):
                        func_name = child.func.id
                    elif isinstance(child.func, ast.Attribute):
                        func_name = child.func.attr
                    if func_name:
                        graph.add_edge(caller, func_name, type='calls')

        # Capture inheritance: ClassDef.bases
        if isinstance(node, ast.ClassDef):
            subclass = node.name
            for base in node.bases:
                if isinstance(base, ast.Name):
                    graph.add_edge(subclass, base.id, type='inherits')

            # Capture methods within classes
            for body_node in node.body:
                if isinstance(body_node, ast.FunctionDef):
                    graph.add_edge(subclass, body_node.name, type='defines_method')

    return chunks

# -------- Main Pipeline --------

repo_root = '/Users/yashchhabria/Projects/assisted_threat_model_ai/qa_code_context/transformers'
files = find_python_files(repo_root)

all_chunks = []
code_graph = nx.DiGraph()

for file_path in files:
    tree, source_code = parse_file(file_path)
    if tree:
        chunks = process_ast(tree, source_code, file_path, repo_root, code_graph)
        all_chunks.extend(chunks)

# Write JSONL of code chunks
with open('transformers_chunks.jsonl', 'w', encoding='utf-8') as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk) + '\n')

# Save GraphML for graph relationships
nx.write_graphml(code_graph, 'transformers_code_graph.graphml')
