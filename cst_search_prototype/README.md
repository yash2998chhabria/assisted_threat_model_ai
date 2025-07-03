# 🔍 CST Search Prototype

A powerful semantic code search system that combines **Concrete Syntax Tree (CST) parsing** with **AI embeddings** to enable intelligent code discovery and analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Workflows](#core-workflows)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Development Status](#development-status)

## 🎯 Overview

This system allows you to:
- **Parse codebases** into structured chunks using CST analysis
- **Generate semantic embeddings** for intelligent code search
- **Search code by meaning** rather than just keywords
- **Analyze code patterns** and complexity across large codebases
- **Find similar code segments** and architectural patterns

## 📁 Project Structure

```
cst_search_prototype/
├── 📊 Data Processing
│   ├── codebase_cst_parser.py      # Main CST parser for codebases
│   └── cst_to_vector_chunks_async.py  # Convert CST to vector chunks
│
├── 🗄️ Vector Database
│   ├── vector_load.py              # Load chunks into ChromaDB
│   ├── vector_search.py            # Advanced semantic search (🚧 WIP)
│   └── vector_search_basic.py      # Basic search functionality
│
├── 🔧 Utilities
│   ├── legacy_chunk_converter.py  # Legacy chunk converter
│   ├── debug_loading.ipynb         # Debug notebook
│   └── requirements.txt            # Python dependencies
│
├── 📂 Output Directories
│   ├── cst_outputs/               # Parsed CST results
│   ├── vector_chunks/             # Vector chunk outputs
│   └── chroma_db/                 # Vector database storage
│
└── 📝 Documentation
    └── README.md                  # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Git

### Setup

1. **Clone and navigate:**
   ```bash
   cd cst_search_prototype
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import chromadb, sentence_transformers; print('✅ Setup complete!')"
   ```

## ⚡ Quick Start

### 1. Parse Your Codebase
```bash
# Parse a codebase into CST chunks
python codebase_cst_parser.py /path/to/your/codebase

# Example: Parse a TypeScript project
python codebase_cst_parser.py ../my_project
```

**Output:** `cst_outputs/my_project_cst_parsed.json`

### 2. Convert to Vector Chunks
```bash
# Convert CST output to vector-ready chunks
python cst_to_vector_chunks_async.py cst_outputs/my_project_cst_parsed.json

# This creates optimized chunks with metadata
```

**Output:** `vector_chunks/my_project_chunks.json`

### 3. Load into Vector Database
```bash
# Load chunks into ChromaDB for semantic search
python vector_load.py vector_chunks/my_project_chunks.json

# Optional: Use different embedding model
python vector_load.py my_chunks.json --model=all-MiniLM-L6-v2
```

**Output:** Vector database in `chroma_db/`

### 4. Search Your Code
```bash
# Basic semantic search
python vector_search_basic.py "websocket authentication" 5

# Test multiple queries
python vector_search_basic.py test
```

## 🔄 Core Workflows

### Workflow 1: New Codebase Analysis
```bash
# 1. Parse codebase
python codebase_cst_parser.py /path/to/codebase

# 2. Convert to chunks  
python cst_to_vector_chunks_async.py cst_outputs/codebase_cst_parsed.json

# 3. Load into database
python vector_load.py vector_chunks/codebase_chunks.json

# 4. Search and analyze
python vector_search_basic.py "authentication logic" 10
```

### Workflow 2: Update Existing Database
```bash
# Parse new/updated code
python codebase_cst_parser.py /path/to/updated/codebase

# Reload database (clears existing)
python vector_load.py new_chunks.json --model=all-mpnet-base-v2
```

### Workflow 3: Multiple Codebases
```bash
# Parse multiple codebases
python codebase_cst_parser.py /path/to/frontend
python codebase_cst_parser.py /path/to/backend  
python codebase_cst_parser.py /path/to/mobile

# Combine and load (manual JSON merging required)
python vector_load.py combined_chunks.json
```

## 🎛️ Advanced Usage

### CST Parser Options
```bash
# Parse with custom file extensions
python codebase_cst_parser.py /path/to/code --extensions .ts,.js,.py

# Include/exclude patterns
python codebase_cst_parser.py /path/to/code --include "src/**" --exclude "node_modules/**"

# Adjust chunk size and complexity analysis
python codebase_cst_parser.py /path/to/code --max-chunk-size 2000 --complexity-analysis
```

### Vector Loading Options
```bash
# Different embedding models
python vector_load.py chunks.json --model=all-MiniLM-L6-v2      # Faster, smaller
python vector_load.py chunks.json --model=all-mpnet-base-v2     # Better quality

# Custom database location
python vector_load.py chunks.json --db-path=./my_custom_db

# Batch size optimization
python vector_load.py chunks.json --batch-size=50  # For memory-constrained systems
```

### Search Options
```bash
# Basic searches
python vector_search_basic.py "error handling" 10
python vector_search_basic.py "async function" 5
python vector_search_basic.py "database query" 15

# Test comprehensive search patterns
python vector_search_basic.py test
```

### Database Management
```bash
# Check database status
python vector_search.py --stats

# Validate database health  
python vector_search.py --validate

# Run diagnostics
python vector_search.py --diagnose
```

## 🔍 Search Examples

### Finding Specific Patterns
```bash
# Authentication code
python vector_search_basic.py "authentication login jwt token" 10

# Error handling patterns  
python vector_search_basic.py "try catch error handling" 8

# Database operations
python vector_search_basic.py "database query sql insert update" 12

# WebSocket implementations
python vector_search_basic.py "websocket realtime connection" 6

# Test code
python vector_search_basic.py "unit test describe it expect" 15
```

### Architectural Analysis
```bash
# Find design patterns
python vector_search_basic.py "factory pattern" 5
python vector_search_basic.py "observer pattern subscribe" 5  
python vector_search_basic.py "singleton instance" 5

# Performance-critical code
python vector_search_basic.py "optimization performance cache" 10

# Security-related code
python vector_search_basic.py "security validation sanitization" 8
```

## 🐛 Troubleshooting

### Common Issues

#### "Collection not found"
```bash
# Verify database exists
ls chroma_db/

# Recreate database
python vector_load.py your_chunks.json
```

#### "Dimension mismatch"
```bash
# Use same model for search as loading
python vector_search_basic.py "query" --model=all-mpnet-base-v2
```

#### "No results found"
```bash
# Check database has data
python vector_search.py --stats

# Try broader search with lower threshold
python vector_search_basic.py "function" 20
```

#### "Memory errors during loading"
```bash
# Reduce batch size
python vector_load.py chunks.json --batch-size=25

# Use smaller embedding model  
python vector_load.py chunks.json --model=all-MiniLM-L6-v2
```

### Debug Mode

Use the Jupyter notebook for interactive debugging:
```bash
jupyter notebook debug_loading.ipynb
```

This allows step-by-step diagnosis of:
- Data loading issues
- Embedding generation problems  
- Search query troubleshooting
- Database connectivity issues

## 🚧 Development Status

### ✅ Completed Features
- **CST Parsing:** Full codebase analysis with tree-sitter
- **Chunk Generation:** Optimized code chunks with metadata
- **Vector Loading:** ChromaDB integration with batching
- **Basic Search:** Semantic search functionality
- **Database Management:** Stats, validation, diagnostics

### 🔄 In Progress  
- **Advanced Search:** Enhanced filtering and ranking
- **Web Interface:** Browser-based search interface
- **Performance Optimization:** Faster indexing and queries
- **Multi-language Support:** Extended language coverage

### 📋 Planned Features
- **Code Similarity Analysis:** Find duplicate/similar code blocks
- **Architectural Insights:** Pattern detection and analysis
- **Integration APIs:** REST API for external tools
- **Export/Import:** Database backup and migration tools
- **Collaborative Features:** Team-based code exploration

## 💡 Tips & Best Practices

### Performance Optimization
- Use `all-MiniLM-L6-v2` model for faster processing with reasonable quality
- Use `all-mpnet-base-v2` model for highest search quality
- Process large codebases in smaller batches
- Exclude unnecessary directories (node_modules, .git, etc.)

### Search Effectiveness
- Use natural language descriptions of what you're looking for
- Combine multiple related keywords for better results
- Start with broad searches, then narrow down
- Try different phrasings if initial searches don't work

### Database Management
- Regularly validate your database with `--validate`
- Monitor database size and performance with `--stats`
- Backup your `chroma_db` directory for important projects
- Consider separate databases for different projects/teams

## 📚 Additional Resources

- **ChromaDB Documentation:** [docs.trychroma.com](https://docs.trychroma.com)
- **Sentence Transformers:** [sbert.net](https://sbert.net)
- **Tree-sitter Parsers:** [tree-sitter.github.io](https://tree-sitter.github.io)

## 🤝 Contributing

This is a prototype system under active development. Contributions, bug reports, and feature requests are welcome!

---

**Happy Code Searching!** 🚀✨