#!/usr/bin/env python3
"""
Convert CST output to vector store chunks for RAG
Extracts semantic chunks with rich metadata and context preservation
"""

import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import re
from collections import Counter, defaultdict
import string

# Optional dependencies for better keyword extraction
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    HAS_NLTK = True
    # Download required data if not present
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
except ImportError:
    HAS_NLTK = False

# For better keyword extraction (optional)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

@dataclass
class CodeChunk:
    """A semantic code chunk ready for vector embedding"""
    id: str  # Unique identifier
    text: str  # The actual code content
    chunk_type: str  # function, class, module, file, etc.
    language: str
    filepath: str
    
    # Structural metadata
    name: str  # Function/class/module name
    parent_name: Optional[str]  # Parent class/module
    line_start: int
    line_end: int
    
    # Context for better retrieval
    summary: str  # Human-readable description
    keywords: List[str]  # Extracted keywords for search
    dependencies: List[str]  # Imports, function calls, etc.
    
    # Additional metadata
    complexity_score: float  # Simple complexity metric
    docstring: Optional[str]  # Extracted documentation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for vector store"""
        return {
            'id': self.id,
            'text': self.text,
            'metadata': {
                'chunk_type': self.chunk_type,
                'language': self.language,
                'filepath': self.filepath,
                'name': self.name,
                'parent_name': self.parent_name,
                'line_start': self.line_start,
                'line_end': self.line_end,
                'summary': self.summary,
                'keywords': self.keywords,
                'dependencies': self.dependencies,
                'complexity_score': self.complexity_score,
                'docstring': self.docstring
            }
        }


class DynamicKeywordExtractor:
    """Extract domain-specific keywords from code and text"""
    
    def __init__(self):
        # Common programming keywords to filter out
        self.programming_stopwords = {
            'function', 'class', 'def', 'return', 'if', 'else', 'for', 'while',
            'try', 'catch', 'finally', 'import', 'from', 'as', 'with', 'pass',
            'break', 'continue', 'yield', 'async', 'await', 'lambda', 'global',
            'nonlocal', 'assert', 'del', 'exec', 'print', 'len', 'str', 'int',
            'float', 'bool', 'list', 'dict', 'set', 'tuple', 'range', 'enumerate',
            'zip', 'map', 'filter', 'reduce', 'any', 'all', 'sum', 'max', 'min',
            'const', 'let', 'var', 'true', 'false', 'null', 'undefined', 'this',
            'new', 'typeof', 'instanceof', 'constructor', 'prototype', 'extends',
            'super', 'static', 'public', 'private', 'protected', 'abstract',
            'interface', 'enum', 'namespace', 'module', 'export', 'default'
        }
        
        # Get English stopwords if NLTK is available
        if HAS_NLTK:
            try:
                self.english_stopwords = set(stopwords.words('english'))
            except:
                self.english_stopwords = set()
        else:
            # Basic English stopwords
            self.english_stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
                'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'can', 'must',
                'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
            }
        
        # Combined stopwords
        self.all_stopwords = self.programming_stopwords | self.english_stopwords
        
        # Domain vocabulary (built dynamically)
        self.domain_vocab = set()
        self.tfidf_vectorizer = None
    
    def build_domain_vocabulary(self, texts: List[str], max_features: int = 1000) -> Set[str]:
        """Build domain-specific vocabulary from a collection of texts"""
        if not texts:
            return set()
        
        # Clean and prepare texts
        cleaned_texts = []
        for text in texts:
            cleaned = self._clean_text(text)
            if cleaned.strip():
                cleaned_texts.append(cleaned)
        
        if not cleaned_texts:
            return set()
        
        # Use TF-IDF if sklearn is available
        if HAS_SKLEARN:
            return self._build_tfidf_vocabulary(cleaned_texts, max_features)
        else:
            return self._build_frequency_vocabulary(cleaned_texts, max_features)
    
    def _build_tfidf_vocabulary(self, texts: List[str], max_features: int) -> Set[str]:
        """Build vocabulary using TF-IDF"""
        try:
            # Create TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words=None,  # We'll handle stopwords ourselves
                ngram_range=(1, 2),  # Include bigrams
                min_df=2,  # Must appear in at least 2 documents
                max_df=0.8,  # Don't include terms in >80% of docs
                token_pattern=r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'  # Code-friendly tokens
            )
            
            # Fit and get feature names
            self.tfidf_vectorizer.fit(texts)
            features = self.tfidf_vectorizer.get_feature_names_out()
            
            # Filter out stopwords and single characters
            domain_terms = set()
            for term in features:
                # Split bigrams and check each part
                words = term.split()
                valid_words = []
                for word in words:
                    if (len(word) > 1 and 
                        word.lower() not in self.all_stopwords and
                        not word.isdigit() and
                        not all(c in string.punctuation for c in word)):
                        valid_words.append(word)
                
                # Add back the term if all words are valid
                if len(valid_words) == len(words):
                    domain_terms.add(term)
            
            self.domain_vocab = domain_terms
            return domain_terms
            
        except Exception as e:
            print(f"TF-IDF failed, falling back to frequency: {e}")
            return self._build_frequency_vocabulary(texts, max_features)
    
    def _build_frequency_vocabulary(self, texts: List[str], max_features: int) -> Set[str]:
        """Build vocabulary using simple frequency analysis"""
        word_freq = Counter()
        
        for text in texts:
            words = self._tokenize_text(text)
            for word in words:
                if (len(word) > 1 and 
                    word.lower() not in self.all_stopwords and
                    not word.isdigit() and
                    not all(c in string.punctuation for c in word)):
                    word_freq[word.lower()] += 1
        
        # Get most common words, but exclude too common ones
        total_docs = len(texts)
        domain_terms = set()
        
        for word, freq in word_freq.most_common(max_features * 2):
            # Skip words that appear in >80% of documents (too common)
            # and words that appear only once (too rare)
            if 1 < freq < total_docs * 0.8:
                domain_terms.add(word)
                if len(domain_terms) >= max_features:
                    break
        
        self.domain_vocab = domain_terms
        return domain_terms
    
    def extract_keywords_from_text(self, text: str, context: str = "", max_keywords: int = 10) -> List[str]:
        """Extract keywords from a single text"""
        if not text.strip():
            return []
        
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        # Tokenize
        words = self._tokenize_text(cleaned_text)
        
        # Extract candidate keywords
        candidates = []
        
        # 1. Add words that are in domain vocabulary
        for word in words:
            if word.lower() in self.domain_vocab:
                candidates.append(word.lower())
        
        # 2. Add camelCase/snake_case identifiers
        identifiers = self._extract_identifiers(text)
        candidates.extend(identifiers)
        
        # 3. Add context-specific terms
        if context:
            context_terms = self._get_context_terms(context)
            candidates.extend(context_terms)
        
        # 4. Use POS tagging if available
        if HAS_NLTK:
            pos_keywords = self._extract_pos_keywords(cleaned_text)
            candidates.extend(pos_keywords)
        
        # 5. Extract meaningful n-grams
        ngrams = self._extract_ngrams(words)
        candidates.extend(ngrams)
        
        # Remove duplicates and filter
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            candidate_lower = candidate.lower()
            if (candidate_lower not in seen and 
                candidate_lower not in self.all_stopwords and
                len(candidate) > 1 and
                not candidate.isdigit()):
                unique_candidates.append(candidate)
                seen.add(candidate_lower)
        
        # Score and rank candidates
        scored_candidates = self._score_keywords(unique_candidates, text, context)
        
        # Return top keywords
        return [kw for kw, score in scored_candidates[:max_keywords]]
    
    def _clean_text(self, text: str) -> str:
        """Clean text for processing"""
        # Remove comments (basic patterns)
        text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
        
        # Remove string literals to avoid extracting content
        text = re.sub(r'"[^"]*"', '', text)
        text = re.sub(r"'[^']*'", '', text)
        text = re.sub(r'`[^`]*`', '', text)
        
        # Remove special characters but keep underscores and dots
        text = re.sub(r'[^\w\s\.]', ' ', text)
        
        return text
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if HAS_NLTK:
            try:
                return word_tokenize(text)
            except:
                pass
        
        # Basic tokenization
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
        return words
    
    def _extract_identifiers(self, text: str) -> List[str]:
        """Extract camelCase and snake_case identifiers"""
        identifiers = []
        
        # camelCase
        camel_pattern = r'\b[a-z]+(?:[A-Z][a-z]*)+\b'
        camel_matches = re.findall(camel_pattern, text)
        identifiers.extend(camel_matches)
        
        # snake_case
        snake_pattern = r'\b[a-z]+(?:_[a-z]+)+\b'
        snake_matches = re.findall(snake_pattern, text)
        identifiers.extend(snake_matches)
        
        # CONSTANTS
        const_pattern = r'\b[A-Z]+(?:_[A-Z]+)*\b'
        const_matches = re.findall(const_pattern, text)
        identifiers.extend([c.lower() for c in const_matches if len(c) > 2])
        
        return identifiers
    
    def _get_context_terms(self, context: str) -> List[str]:
        """Get relevant terms based on context"""
        context_map = {
            'function': ['function', 'method', 'procedure', 'routine'],
            'class': ['class', 'object', 'type', 'struct'],
            'file': ['module', 'package', 'library'],
            'variable': ['variable', 'field', 'property', 'attribute']
        }
        return context_map.get(context, [])
    
    def _extract_pos_keywords(self, text: str) -> List[str]:
        """Extract keywords using POS tagging"""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Extract nouns, verbs, and adjectives
            keywords = []
            for word, pos in pos_tags:
                if pos.startswith(('NN', 'VB', 'JJ')) and len(word) > 2:
                    keywords.append(word.lower())
            
            return keywords
        except:
            return []
    
    def _extract_ngrams(self, words: List[str], n: int = 2) -> List[str]:
        """Extract meaningful n-grams"""
        if len(words) < n:
            return []
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = '_'.join(words[i:i+n])
            # Only include if all words are meaningful
            if all(len(w) > 1 and w.lower() not in self.all_stopwords for w in words[i:i+n]):
                ngrams.append(ngram.lower())
        
        return ngrams
    
    def _score_keywords(self, candidates: List[str], text: str, context: str) -> List[tuple]:
        """Score and rank keyword candidates"""
        scored = []
        text_lower = text.lower()
        
        for candidate in candidates:
            score = 0.0
            candidate_lower = candidate.lower()
            
            # Frequency in text
            freq = text_lower.count(candidate_lower)
            score += min(freq * 0.1, 1.0)  # Cap frequency bonus
            
            # Length bonus (prefer longer, more specific terms)
            score += min(len(candidate) * 0.05, 0.5)
            
            # Domain vocabulary bonus
            if candidate_lower in self.domain_vocab:
                score += 0.5
            
            # Context bonus
            if context and context in candidate_lower:
                score += 0.3
            
            # Identifier pattern bonus
            if ('_' in candidate or 
                any(c.isupper() for c in candidate[1:])):  # camelCase
                score += 0.3
            
            # Position bonus (earlier = more important)
            first_pos = text_lower.find(candidate_lower)
            if first_pos != -1:
                position_bonus = max(0, 0.2 - (first_pos / len(text_lower)) * 0.2)
                score += position_bonus
            
            scored.append((candidate, score))
        
        # Sort by score descending
        return sorted(scored, key=lambda x: x[1], reverse=True)


class CSTToVectorChunks:
    """Convert CST trees to semantic chunks for vector embedding"""
    
    def __init__(self):
        # Node types that represent semantic units we want to extract
        self.semantic_node_types = {
            # Python
            'function_definition', 'async_function_definition', 'class_definition',
            'module',
            # JavaScript/TypeScript  
            'function_declaration', 'method_definition', 'class_declaration',
            'arrow_function', 'function_expression', 'program',
            # Java
            'method_declaration', 'class_declaration', 'interface_declaration',
            # C/C++
            'function_definition', 'struct_specifier', 'class_specifier',
            # Rust
            'function_item', 'struct_item', 'impl_item', 'trait_item',
            # Go
            'function_declaration', 'method_declaration', 'type_declaration',
            # C#
            'method_declaration', 'class_declaration', 'interface_declaration'
        }
        
        # Initialize dynamic keyword extraction system
        self.domain_vocabulary = set()  # Will be built from the codebase
        self.keyword_extractor = DynamicKeywordExtractor()
    
    def convert_cst_to_chunks(self, cst_data: Dict[str, Any]) -> List[CodeChunk]:
        """Convert full CST output to semantic chunks"""
        all_chunks = []
        
        # First pass: collect all text for domain vocabulary building
        print("🔍 Building domain vocabulary from codebase...")
        all_texts = []
        for file_data in cst_data.get('files', []):
            file_text = file_data['cst'].get('text', '')
            if file_text.strip():
                all_texts.append(file_text)
        
        # Build domain-specific vocabulary
        if all_texts:
            self.domain_vocabulary = self.keyword_extractor.build_domain_vocabulary(all_texts)
            print(f"📚 Built domain vocabulary with {len(self.domain_vocabulary)} terms")
        
        # Second pass: extract chunks with enhanced keyword extraction
        print("⚡ Extracting semantic chunks...")
        for file_data in cst_data.get('files', []):
            filepath = file_data['filepath']
            language = file_data['language']
            cst_tree = file_data['cst']
            
            # Extract chunks from this file
            file_chunks = self._extract_file_chunks(filepath, language, cst_tree)
            all_chunks.extend(file_chunks)
        
        return all_chunks
    
    def _extract_file_chunks(self, filepath: str, language: str, cst_tree: Dict[str, Any]) -> List[CodeChunk]:
        """Extract semantic chunks from a single file's CST"""
        chunks = []
        
        # Get file content and metadata
        file_text = cst_tree.get('text', '')
        file_lines = file_text.split('\n')
        
        # Extract file-level dependencies
        file_dependencies = self._extract_dependencies(cst_tree, language)
        
        # 1. File-level chunk (for broad context)
        file_chunk = self._create_file_chunk(filepath, language, cst_tree, file_dependencies)
        chunks.append(file_chunk)
        
        # 2. Extract semantic units (functions, classes, etc.)
        semantic_chunks = self._extract_semantic_units(
            filepath, language, cst_tree, file_lines, file_dependencies
        )
        chunks.extend(semantic_chunks)
        
        return chunks
    
    def _create_file_chunk(self, filepath: str, language: str, cst_tree: Dict[str, Any], 
                          dependencies: List[str]) -> CodeChunk:
        """Create a file-level chunk for broad context"""
        file_text = cst_tree.get('text', '')
        filename = Path(filepath).name
        
        # Create summary from filename and dependencies
        summary = f"File {filename} ({language})"
        if dependencies:
            summary += f" - imports: {', '.join(dependencies[:5])}"
        
        # Extract keywords from filename and content
        keywords = self._extract_keywords(file_text, filepath, context="file")
        
        return CodeChunk(
            id=f"file:{filepath}",
            text=file_text,
            chunk_type="file",
            language=language,
            filepath=filepath,
            name=filename,
            parent_name=None,
            line_start=1,
            line_end=len(file_text.split('\n')),
            summary=summary,
            keywords=keywords,
            dependencies=dependencies,
            complexity_score=self._calculate_complexity(cst_tree),
            docstring=self._extract_file_docstring(cst_tree, language)
        )
    
    def _extract_semantic_units(self, filepath: str, language: str, node: Dict[str, Any], 
                               file_lines: List[str], file_dependencies: List[str],
                               parent_name: Optional[str] = None) -> List[CodeChunk]:
        """Recursively extract semantic units from CST nodes"""
        chunks = []
        node_type = node.get('type', '')
        
        # Check if this node is a semantic unit we want to extract
        if node_type in self.semantic_node_types and node_type != 'module' and node_type != 'program':
            chunk = self._create_semantic_chunk(
                filepath, language, node, file_lines, file_dependencies, parent_name
            )
            if chunk:
                chunks.append(chunk)
                # Use this chunk's name as parent for children
                parent_name = chunk.name
        
        # Recursively process children
        for child in node.get('children', []):
            child_chunks = self._extract_semantic_units(
                filepath, language, child, file_lines, file_dependencies, parent_name
            )
            chunks.extend(child_chunks)
        
        return chunks
    
    def _create_semantic_chunk(self, filepath: str, language: str, node: Dict[str, Any],
                              file_lines: List[str], file_dependencies: List[str],
                              parent_name: Optional[str]) -> Optional[CodeChunk]:
        """Create a chunk for a semantic unit (function, class, etc.)"""
        node_type = node.get('type', '')
        node_text = node.get('text', '')
        
        if not node_text.strip():
            return None
        
        # Extract name
        name = self._extract_name(node, node_type, language)
        if not name:
            name = f"anonymous_{node_type}"
        
        # Calculate line numbers
        start_line = node.get('start_point', [0, 0])[0] + 1
        end_line = node.get('end_point', [0, 0])[0] + 1
        
        # Extract docstring
        docstring = self._extract_docstring(node, language)
        
        # Create summary
        summary = self._create_summary(name, node_type, parent_name, docstring)
        
        # Extract keywords
        keywords = self._extract_keywords(node_text, name, context=node_type)
        
        # Extract local dependencies (function calls, etc.)
        local_deps = self._extract_local_dependencies(node, language)
        all_deps = list(set(file_dependencies + local_deps))
        
        return CodeChunk(
            id=f"{filepath}:{node_type}:{name}:{start_line}",
            text=node_text,
            chunk_type=node_type,
            language=language,
            filepath=filepath,
            name=name,
            parent_name=parent_name,
            line_start=start_line,
            line_end=end_line,
            summary=summary,
            keywords=keywords,
            dependencies=all_deps,
            complexity_score=self._calculate_complexity(node),
            docstring=docstring
        )
    
    def _extract_name(self, node: Dict[str, Any], node_type: str, language: str) -> Optional[str]:
        """Extract the name of a semantic unit"""
        # Look for identifier nodes in children
        def find_identifier(node):
            if node.get('type') == 'identifier':
                return node.get('text', '').strip()
            for child in node.get('children', []):
                result = find_identifier(child)
                if result:
                    return result
            return None
        
        return find_identifier(node)
    
    def _extract_dependencies(self, node: Dict[str, Any], language: str) -> List[str]:
        """Extract import statements and dependencies"""
        dependencies = []
        
        def extract_imports(node):
            node_type = node.get('type', '')
            
            # Python imports
            if node_type in ['import_statement', 'import_from_statement']:
                text = node.get('text', '').strip()
                # Extract module names
                if 'import ' in text:
                    parts = text.replace('from ', '').replace('import ', '').split()
                    if parts:
                        dependencies.append(parts[0].split('.')[0])
            
            # JavaScript imports
            elif node_type == 'import_statement':
                text = node.get('text', '').strip()
                # Extract from "from 'module'" or "require('module')"
                match = re.search(r"['\"]([^'\"]+)['\"]", text)
                if match:
                    dependencies.append(match.group(1))
            
            # Recursively check children
            for child in node.get('children', []):
                extract_imports(child)
        
        extract_imports(node)
        return list(set(dependencies))  # Remove duplicates
    
    def _extract_local_dependencies(self, node: Dict[str, Any], language: str) -> List[str]:
        """Extract function calls and local dependencies within a semantic unit"""
        dependencies = []
        
        def extract_calls(node):
            node_type = node.get('type', '')
            
            # Function calls
            if node_type in ['call_expression', 'function_call']:
                # Try to get the function name
                for child in node.get('children', []):
                    if child.get('type') in ['identifier', 'member_expression']:
                        call_text = child.get('text', '').strip()
                        if call_text and not call_text.startswith('('):
                            dependencies.append(call_text.split('(')[0])
                        break
            
            # Recursively check children
            for child in node.get('children', []):
                extract_calls(child)
        
        extract_calls(node)
        return list(set(dependencies))
    
    def _extract_keywords(self, text: str, name: str = "", context: str = "") -> List[str]:
        """Extract relevant keywords using dynamic extraction"""
        # Combine text and name for context
        full_context = f"{name} {text}" if name else text
        
        # Use dynamic keyword extractor
        keywords = self.keyword_extractor.extract_keywords_from_text(
            full_context, 
            context=context
        )
        
        # Add domain vocabulary terms found in text
        text_lower = text.lower()
        for domain_term in self.domain_vocabulary:
            if domain_term.lower() in text_lower:
                keywords.append(domain_term)
        
        # Remove duplicates and return
        return list(set(keywords))
    
    def _extract_docstring(self, node: Dict[str, Any], language: str) -> Optional[str]:
        """Extract docstring/comments from a semantic unit"""
        # Look for string literals or comments at the beginning
        for child in node.get('children', [])[:3]:  # Check first few children
            child_type = child.get('type', '')
            if child_type in ['string_literal', 'comment', 'expression_statement']:
                text = child.get('text', '').strip()
                if language == 'python' and (text.startswith('"""') or text.startswith("'''")):
                    return text.strip('"""\'')
                elif text.startswith('//') or text.startswith('/*'):
                    return text
        return None
    
    def _extract_file_docstring(self, node: Dict[str, Any], language: str) -> Optional[str]:
        """Extract file-level docstring"""
        # Similar to function docstring but look at top of file
        return self._extract_docstring(node, language)
    
    def _create_summary(self, name: str, node_type: str, parent_name: Optional[str], 
                       docstring: Optional[str]) -> str:
        """Create a human-readable summary"""
        summary = f"{node_type.replace('_', ' ').title()}: {name}"
        
        if parent_name:
            summary += f" (in {parent_name})"
        
        if docstring:
            # Take first line of docstring
            first_line = docstring.split('\n')[0].strip().strip('"""\'/*')
            if first_line and len(first_line) < 100:
                summary += f" - {first_line}"
        
        return summary
    
    def _calculate_complexity(self, node: Dict[str, Any]) -> float:
        """Simple complexity score based on node count and nesting"""
        def count_nodes(node, depth=0):
            count = 1
            max_depth = depth
            for child in node.get('children', []):
                child_count, child_depth = count_nodes(child, depth + 1)
                count += child_count
                max_depth = max(max_depth, child_depth)
            return count, max_depth
        
        node_count, max_depth = count_nodes(node)
        return min(10.0, (node_count / 10) + (max_depth / 5))  # Scale to 0-10


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cst_to_vector.py <cst_output.json> [chunks_output.json]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "vector_chunks.json"
    
    print(f"🔄 Converting CST output to vector chunks...")
    print(f"📁 Input: {input_file}")
    print(f"💾 Output: {output_file}")
    
    # Check for optional dependencies
    if not HAS_NLTK:
        print("💡 Install NLTK for better keyword extraction: pip install nltk")
    if not HAS_SKLEARN:
        print("💡 Install scikit-learn for domain vocabulary: pip install scikit-learn")
    
    # Load CST data
    with open(input_file, 'r') as f:
        cst_data = json.load(f)
    
    # Convert to chunks
    converter = CSTToVectorChunks()
    chunks = converter.convert_cst_to_chunks(cst_data)
    
    print(f"✅ Generated {len(chunks)} chunks")
    
    if hasattr(converter, 'domain_vocabulary') and converter.domain_vocabulary:
        print(f"📚 Domain vocabulary: {len(converter.domain_vocabulary)} terms")
        # Show sample domain terms
        sample_terms = list(converter.domain_vocabulary)[:10]
        print(f"   Sample terms: {', '.join(sample_terms)}")
    
    # Show most common keywords across all chunks
    all_keywords = []
    for chunk in chunks:
        all_keywords.extend(chunk.keywords)
    
    if all_keywords:
        from collections import Counter
        common_keywords = Counter(all_keywords).most_common(10)
        print(f"🔍 Most common keywords: {', '.join([k for k, _ in common_keywords])}")
    
    # Show chunk type breakdown
    chunk_types = {}
    for chunk in chunks:
        chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
    
    print("📊 Chunk breakdown:")
    for chunk_type, count in sorted(chunk_types.items()):
        print(f"   {chunk_type}: {count}")
    
    # Save chunks
    chunks_data = {
        'chunks': [chunk.to_dict() for chunk in chunks],
        'metadata': {
            'total_chunks': len(chunks),
            'chunk_types': chunk_types
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(chunks_data, f, indent=2)
    
    print(f"💾 Chunks saved to {output_file}")
    
    # Show example chunk
    if chunks:
        print("\n📝 Example chunk:")
        example = chunks[0]
        print(f"   ID: {example.id}")
        print(f"   Type: {example.chunk_type}")
        print(f"   Name: {example.name}")
        print(f"   Summary: {example.summary}")
        print(f"   Keywords: {example.keywords}")


if __name__ == "__main__":
    main()