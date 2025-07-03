#!/usr/bin/env python3
"""
Simplified Codebase CST Parser - Parse and output CST as JSON
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import concurrent.futures
from multiprocessing import cpu_count

# Core language imports
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_rust as tsrust
import tree_sitter_go as tsgo
import tree_sitter_cpp as tscpp
import tree_sitter_java as tsjava
import tree_sitter_c_sharp as tscsharp

# TypeScript - simplified approach
try:
    import tree_sitter_typescript as tstypescript
    HAS_TYPESCRIPT = True
except ImportError:
    HAS_TYPESCRIPT = False

from tree_sitter import Language, Parser, Node


@dataclass
class CSTNode:
    """CST node representation"""
    type: str
    text: str
    start_byte: int
    end_byte: int
    start_point: tuple[int, int]
    end_point: tuple[int, int]
    is_named: bool
    children: List['CSTNode']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'type': self.type,
            'text': self.text if len(self.text) <= 1000 else self.text[:997] + '...',
            'start_byte': self.start_byte,
            'end_byte': self.end_byte,
            'start_point': self.start_point,
            'end_point': self.end_point,
            'is_named': self.is_named,
            'children': [child.to_dict() for child in self.children]
        }


@dataclass
class ParseResult:
    """Result of parsing a file"""
    filepath: str
    language: str
    success: bool
    cst: Optional[CSTNode] = None
    error: Optional[str] = None
    parse_time: float = 0.0
    file_size: int = 0


class CodebaseCST:
    """Simple CST parser for codebases"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(cpu_count(), 8)
        self.parsers: Dict[str, Parser] = {}
        self.extensions = self._get_extensions()
        self._init_parsers()
    
    def _get_extensions(self) -> Dict[str, str]:
        """Get file extension to language mapping"""
        extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.mjs': 'javascript',
            '.rs': 'rust',
            '.go': 'go',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c++': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.hxx': 'cpp',
            '.java': 'java',
            '.cs': 'csharp',
        }
        
        # Add TypeScript if available
        if HAS_TYPESCRIPT:
            extensions['.ts'] = 'typescript'
            extensions['.tsx'] = 'tsx'
        
        return extensions
    
    def _init_parsers(self):
        """Initialize parsers for supported languages"""
        languages = {
            'python': tspython.language(),
            'javascript': tsjavascript.language(),
            'rust': tsrust.language(),
            'go': tsgo.language(),
            'cpp': tscpp.language(),
            'c': tscpp.language(),
            'java': tsjava.language(),
            'csharp': tscsharp.language(),
        }
        
        # Add TypeScript parsers
        if HAS_TYPESCRIPT:
            try:
                # Try different TypeScript loading methods
                if hasattr(tstypescript, 'language_typescript'):
                    languages['typescript'] = tstypescript.language_typescript()
                elif hasattr(tstypescript, 'typescript'):
                    languages['typescript'] = tstypescript.typescript()
                else:
                    languages['typescript'] = tstypescript.language()
                
                if hasattr(tstypescript, 'language_tsx'):
                    languages['tsx'] = tstypescript.language_tsx()
                elif hasattr(tstypescript, 'tsx'):
                    languages['tsx'] = tstypescript.tsx()
                else:
                    languages['tsx'] = languages['typescript']  # Fallback
                    
            except Exception as e:
                print(f"TypeScript parser init failed: {e}")
        
        # Initialize parsers with proper Language wrapping
        successful_parsers = 0
        for lang_name, language_obj in languages.items():
            try:
                parser = Parser()
                
                # Handle different tree-sitter package versions
                if isinstance(language_obj, Language):
                    # Already a Language object
                    parser.language = language_obj
                    print(f"✅ {lang_name} parser initialized (Language object)")
                else:
                    # PyCapsule or other - wrap with Language
                    try:
                        language = Language(language_obj)
                        parser.language = language
                        print(f"✅ {lang_name} parser initialized (wrapped PyCapsule)")
                    except Exception as wrap_error:
                        # Skip this language if we can't initialize it
                        print(f"❌ {lang_name} parser failed: {wrap_error}")
                        continue
                
                self.parsers[lang_name] = parser
                successful_parsers += 1
                
            except Exception as e:
                print(f"❌ {lang_name} parser failed: {e}")
        
        print(f"🎯 Successfully initialized {successful_parsers}/{len(languages)} parsers")
    
    def detect_language(self, filepath: Path) -> Optional[str]:
        """Detect language from file extension"""
        return self.extensions.get(filepath.suffix.lower())
    
    def find_source_files(self, 
                         directory: Union[str, Path],
                         recursive: bool = True,
                         max_file_size: int = 10 * 1024 * 1024) -> List[Path]:
        """Find all source files in directory"""
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        excludes = {
            '.git', '.svn', '__pycache__', 'node_modules', 
            'target', 'build', 'dist', '.next', '.terraform'
        }
        
        source_files = []
        pattern = "**/*" if recursive else "*"
        
        for filepath in directory.glob(pattern):
            if not filepath.is_file():
                continue
            
            if filepath.suffix.lower() not in self.extensions:
                continue
            
            if any(exclude in str(filepath) for exclude in excludes):
                continue
            
            try:
                if filepath.stat().st_size > max_file_size:
                    continue
            except OSError:
                continue
            
            source_files.append(filepath)
        
        return sorted(source_files)
    
    def parse_file(self, filepath: Union[str, Path]) -> ParseResult:
        """Parse a single file and return CST"""
        filepath = Path(filepath)
        start_time = time.time()
        
        try:
            language = self.detect_language(filepath)
            if language is None:
                return ParseResult(
                    filepath=str(filepath),
                    language='unknown',
                    success=False,
                    error=f"Unsupported extension: {filepath.suffix}",
                    parse_time=time.time() - start_time
                )
            
            if language not in self.parsers:
                return ParseResult(
                    filepath=str(filepath),
                    language=language,
                    success=False,
                    error=f"No parser for language: {language}",
                    parse_time=time.time() - start_time
                )
            
            # Read file
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            file_size = len(content.encode('utf-8'))
            
            # Parse
            parser = self.parsers[language]
            tree = parser.parse(bytes(content, 'utf-8'))
            
            if tree.root_node is None:
                return ParseResult(
                    filepath=str(filepath),
                    language=language,
                    success=False,
                    error="No root node",
                    file_size=file_size,
                    parse_time=time.time() - start_time
                )
            
            # Convert to CST
            cst = self._convert_node(tree.root_node, content.encode('utf-8'))
            
            return ParseResult(
                filepath=str(filepath),
                language=language,
                success=True,
                cst=cst,
                file_size=file_size,
                parse_time=time.time() - start_time
            )
            
        except Exception as e:
            return ParseResult(
                filepath=str(filepath),
                language='unknown',
                success=False,
                error=str(e),
                parse_time=time.time() - start_time
            )
    
    def _convert_node(self, node: Node, source: bytes) -> CSTNode:
        """Convert Tree-sitter node to CST format"""
        node_text = source[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
        
        children = []
        for child in node.children:
            children.append(self._convert_node(child, source))
        
        return CSTNode(
            type=node.type,
            text=node_text,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_point=(node.start_point[0], node.start_point[1]),
            end_point=(node.end_point[0], node.end_point[1]),
            is_named=node.is_named,
            children=children
        )
    
    def parse_codebase(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """Parse entire codebase and return CST trees as JSON"""
        directory = Path(directory)
        print(f"🔍 Scanning directory: {directory}")
        
        # Find source files
        source_files = self.find_source_files(directory)
        print(f"📁 Found {len(source_files)} source files")
        
        if not source_files:
            print("❌ No source files found")
            return {'files': []}
        
        # Show language breakdown
        lang_counts = {}
        for filepath in source_files:
            lang = self.detect_language(filepath)
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        print("📊 Language breakdown:")
        for lang, count in sorted(lang_counts.items()):
            available = "✅" if lang in self.parsers else "❌"
            print(f"   {available} {lang}: {count} files")
        
        # Parse files in parallel
        print(f"⚡ Starting parallel parsing with {self.max_workers} workers...")
        results = []
        successful = 0
        failed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.parse_file, filepath): filepath 
                for filepath in source_files
            }
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                result = future.result()
                results.append(result)
                
                if result.success:
                    successful += 1
                else:
                    failed += 1
                
                # Progress updates
                processed = i + 1
                if processed % 10 == 0 or processed == len(source_files):
                    print(f"   📈 Progress: {processed}/{len(source_files)} ({processed/len(source_files)*100:.1f}%) - ✅{successful} ❌{failed}")
        
        # Format output
        files = []
        for result in results:
            if result.success and result.cst:
                files.append({
                    'filepath': result.filepath,
                    'language': result.language,
                    'cst': result.cst.to_dict()
                })
        
        print(f"🎉 Completed! Successfully parsed {len(files)} files")
        return {'files': files}


def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cst_parser.py <directory> [output_file]")
        sys.exit(1)
    
    directory = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("🚀 Starting CST Parser")
    print(f"📂 Target directory: {directory}")
    if output_file:
        print(f"💾 Output file: {output_file}")
    
    parser = CodebaseCST()
    result = parser.parse_codebase(directory)
    
    if output_file:
        print(f"💾 Writing results to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✅ Results saved to {output_file}")
    else:
        print("📤 Outputting to console...")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()