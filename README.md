# 🔍 Goal: Language-Agnostic Vulnerability Analysis Using CSTs, Chunks, and RAG

This project aims to build a static vulnerability analysis system that can parse any codebase into a Concrete Syntax Tree (CST), chunk the CST into semantically meaningful sections, embed them using a sentence transformer model, and store them in a vector database. The final goal is to enable retrieval-augmented analysis (RAG) that can identify security vulnerabilities — including taint flows from user-controlled sources to dangerous sinks — using both deterministic rules and LLM-based validation.

The CST parser has reached a working prototype stage, and search over vector chunks is live. Upcoming phases include taint propagation tracking, call graph + dataflow graph construction, sanitizer identification, and automated threat modeling.

---

## ✅ What’s Working

### `codebase_cst_parser.py`
- Converts a codebase into per-file CSTs using Tree-sitter.
- Outputs structured CSTs in JSON format.

### `cst_to_vector_chunks_async.py`
- Processes CSTs into chunks representing functions, classes, or logical blocks.
- Uses `nltk` for dynamic chunk summarization and complexity tagging.

### `vector_load.py`
- Embeds chunks using a SentenceTransformer model (e.g., `all-MiniLM-L6-v2`).
- Stores them in a Chroma vector store for semantic search.

### `vector_search_basic.py`
- Runs top-K nearest neighbor search against the vector store.
- Returns semantically relevant code blocks for natural language or structured queries.

---

## 🔧 In Progress

### `vector_search_advanced_to_fix.py`
- Under development to support contextual filters (e.g., by taint, decorator, or complexity).

### Next Steps
- Taint analysis engine for propagating user-controlled input through variables and function calls.
- Call graph + dataflow graph construction to support interprocedural tracing.
- Sanitizer detection using both static heuristics and optional LLM assistance.
- Threat model graph generation per finding.

---

> ⚠️ This is an active work in progress focused on making static vulnerability detection explainable, language-agnostic, and RAG-friendly. The foundation is complete and ready to expand toward flow tracking and threat modeling.
