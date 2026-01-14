"""
Enhanced LLM-Based Dependency Graph Builder
Generates comprehensive dependency analysis with all requested metrics
"""
import os
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FileAnalysis:
    """Complete analysis of a single file"""
    file: str  # Filename only
    full_path: str  # Relative path from repo root
    language: str
    num_functions: int
    functions_defined: List[str]  # Function names
    dependencies: List[str]  # Import statements
    dependency_files: List[str]  # Files this depends on
    dependency_with_functions: List[str]  # Dependencies with function details
    dependency_full_paths: List[str]  # Full paths of dependencies
    dependents: List[str] = field(default_factory=list)  # Import statements of dependents
    dependent_files: List[str] = field(default_factory=list)  # Files that depend on this
    dependent_with_functions: List[str] = field(default_factory=list)  # Dependents with function details
    dependent_full_paths: List[str] = field(default_factory=list)  # Full paths of dependents
    total_connections: int = 0  # Total dependencies + dependents
    
    # Additional details
    num_classes: int = 0
    classes_defined: List[str] = field(default_factory=list)
    global_variables: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    
    # Detailed structure
    functions_with_details: List[Dict] = field(default_factory=list)
    classes_with_details: List[Dict] = field(default_factory=list)


class EnhancedLLMDependencyBuilder:
    """Build comprehensive dependency graphs using LLM analysis"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the builder
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.file_analyses: Dict[str, FileAnalysis] = {}
        self.repo_path: Optional[Path] = None
        
        # Language detection mapping
        self.language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'JavaScript (React)',
            '.tsx': 'TypeScript (React)',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++ Header',
            '.go': 'Go',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.cs': 'C#',
            '.rs': 'Rust',
            '.kt': 'Kotlin',
            '.swift': 'Swift',
        }
    
    def detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        return self.language_map.get(file_path.suffix, 'Unknown')
    
    def analyze_file_with_llm(self, file_path: Path, code: str) -> Optional[FileAnalysis]:
        """
        Analyze a single file using LLM
        
        Args:
            file_path: Path to the file (relative to repo root)
            code: File content
            
        Returns:
            FileAnalysis object or None if analysis fails
        """
        language = self.detect_language(file_path)
        
        if language == 'Unknown':
            logger.warning(f"Skipping {file_path} - unknown language")
            return None
        
        prompt = f"""Analyze this {language} code file and extract COMPLETE dependency and structure information.

File: {file_path}

Code:
```{language.lower()}
{code}
```

Extract and return as JSON with these EXACT fields:

1. **imports**: List of ALL import/include/require statements (complete statements as strings)
2. **dependency_file_paths**: List of FILE PATHS this imports from (resolve relative paths like './utils', '../config')
3. **functions**: Array of objects with:
   - name: function name
   - start_line: starting line number
   - end_line: ending line number
   - parameters: list of parameter names
   - calls: list of function names this function calls
4. **classes**: Array of objects with:
   - name: class name
   - start_line: starting line number
   - end_line: ending line number
   - methods: list of method names
   - inherits_from: list of parent class names
5. **global_variables**: List of global/module-level variable names
6. **exports**: List of what this file exports (functions, classes, variables)

Return ONLY this JSON structure (no markdown, no explanations):
{{
    "imports": ["import os", "from utils import helper", "const config = require('./config')"],
    "dependency_file_paths": ["./utils.py", "../config.js", "./helpers/formatter.py"],
    "functions": [
        {{
            "name": "process_data",
            "start_line": 10,
            "end_line": 25,
            "parameters": ["data", "options"],
            "calls": ["validate", "transform", "save"]
        }}
    ],
    "classes": [
        {{
            "name": "DataProcessor",
            "start_line": 30,
            "end_line": 50,
            "methods": ["__init__", "process", "validate"],
            "inherits_from": ["BaseProcessor"]
        }}
    ],
    "global_variables": ["DEFAULT_CONFIG", "CACHE_SIZE"],
    "exports": ["process_data", "DataProcessor"]
}}

IMPORTANT:
- Include ALL imports, even standard library
- Resolve relative imports to file paths (e.g., './utils' -> './utils.py' or './utils.js')
- List ALL functions and methods with accurate line numbers
- Include ALL function calls within each function
- Be comprehensive - don't skip anything
"""
        
        try:
            logger.info(f"Analyzing {file_path} with LLM...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert code analyzer. Extract COMPLETE and ACCURATE dependency information. Return only valid JSON with no markdown formatting."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=4096
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
                if content.startswith('json'):
                    content = content[4:].strip()
            
            result = json.loads(content)
            
            # Extract data
            functions = result.get('functions', [])
            classes = result.get('classes', [])
            function_names = [f['name'] for f in functions]
            class_names = [c['name'] for c in classes]
            
            # Get dependency file paths
            dependency_files = result.get('dependency_file_paths', [])
            
            # Create dependency_with_functions format: "file.py::function1,function2"
            dependency_with_funcs = []
            for dep_file in dependency_files:
                # We'll update this later when we know what functions are in each file
                dependency_with_funcs.append(f"{dep_file}::[analyzing]")
            
            # Get full paths
            dependency_full_paths = []
            for dep_file in dependency_files:
                if dep_file.startswith('./') or dep_file.startswith('../'):
                    # Resolve relative to current file's directory
                    current_dir = file_path.parent if file_path.parent != Path('.') else Path('.')
                    resolved = (current_dir / dep_file).resolve()
                    try:
                        rel_path = resolved.relative_to(self.repo_path.resolve())
                        dependency_full_paths.append(str(rel_path))
                    except ValueError:
                        dependency_full_paths.append(dep_file)
                else:
                    dependency_full_paths.append(dep_file)
            
            # Create FileAnalysis object
            analysis = FileAnalysis(
                file=file_path.name,
                full_path=str(file_path),
                language=language,
                num_functions=len(functions),
                functions_defined=function_names,
                dependencies=result.get('imports', []),
                dependency_files=dependency_files,
                dependency_with_functions=dependency_with_funcs,
                dependency_full_paths=dependency_full_paths,
                num_classes=len(classes),
                classes_defined=class_names,
                global_variables=result.get('global_variables', []),
                exports=result.get('exports', []),
                functions_with_details=functions,
                classes_with_details=classes
            )
            
            logger.info(f"✓ {file_path}: {len(functions)} functions, {len(classes)} classes, {len(dependency_files)} dependencies")
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response for {file_path}: {e}")
            logger.error(f"Response content: {content[:500]}")
            return None
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return None
    
    def scan_repository(self, repo_path: Path, extensions: Optional[List[str]] = None) -> List[Path]:
        """
        Scan repository for code files
        
        Args:
            repo_path: Path to repository
            extensions: List of extensions to include (None = all supported)
            
        Returns:
            List of file paths
        """
        if extensions is None:
            extensions = list(self.language_map.keys())
        
        files = []
        skip_dirs = {'.git', 'node_modules', '__pycache__', 'venv', '.venv', 
                     'build', 'dist', '.next', 'target', 'bin', 'obj', 'vendor'}
        
        for file_path in repo_path.rglob("*"):
            # Skip hidden files and directories (except allow some hidden files)
            if any(part.startswith('.') and part not in ['.github'] for part in file_path.parts):
                continue
            
            # Skip common build/dependency directories
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            
            # Check if it's a file with matching extension
            if file_path.is_file() and file_path.suffix in extensions:
                files.append(file_path)
        
        logger.info(f"Found {len(files)} files to analyze")
        return files
    
    def build_dependency_graph(self, repo_path: Path, max_workers: int = 3) -> Dict[str, FileAnalysis]:
        """
        Build complete dependency graph for repository
        
        Args:
            repo_path: Path to repository
            max_workers: Number of parallel workers
            
        Returns:
            Dictionary of file_path -> FileAnalysis
        """
        self.repo_path = Path(repo_path).resolve()
        files = self.scan_repository(self.repo_path)
        
        logger.info(f"Building dependency graph for {len(files)} files...")
        
        # Analyze files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    # Get relative path from repo root
                    relative_path = file_path.relative_to(self.repo_path)
                    
                    future = executor.submit(
                        self.analyze_file_with_llm,
                        relative_path,
                        code
                    )
                    futures[future] = relative_path
                    
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
            
            # Collect results
            for future in as_completed(futures):
                relative_path = futures[future]
                try:
                    analysis = future.result()
                    if analysis:
                        self.file_analyses[str(relative_path)] = analysis
                except Exception as e:
                    logger.error(f"Error processing {relative_path}: {e}")
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
        
        logger.info(f"✓ Completed analysis of {len(self.file_analyses)} files")
        
        # Build reverse dependencies (dependents)
        self._build_reverse_dependencies()
        
        # Update dependency_with_functions with actual function names
        self._update_dependency_functions()
        
        # Calculate total connections
        self._calculate_connections()
        
        return self.file_analyses
    
    def _build_reverse_dependencies(self):
        """Build reverse dependency relationships"""
        logger.info("Building reverse dependencies (dependents)...")
        
        # Create mapping of normalized paths to analyses
        path_to_analysis = {}
        for path, analysis in self.file_analyses.items():
            normalized = str(Path(path).as_posix())
            path_to_analysis[normalized] = analysis
        
        # For each file, find who depends on it
        for file_path, analysis in self.file_analyses.items():
            normalized_path = str(Path(file_path).as_posix())
            
            # Check each file's dependencies
            for other_path, other_analysis in self.file_analyses.items():
                if file_path == other_path:
                    continue
                
                # Check if other_file depends on this file
                for dep_path in other_analysis.dependency_full_paths:
                    normalized_dep = str(Path(dep_path).as_posix())
                    
                    # Normalize and compare paths
                    if normalized_dep == normalized_path or dep_path == file_path:
                        # other_file depends on this file
                        # So this file is a dependent of other_file
                        if other_path not in analysis.dependent_files:
                            analysis.dependent_files.append(other_path)
                            analysis.dependent_full_paths.append(other_path)
                            
                            # Add imports from dependent file
                            for imp in other_analysis.dependencies:
                                if file_path in imp or analysis.file in imp:
                                    if imp not in analysis.dependents:
                                        analysis.dependents.append(imp)
    
    def _update_dependency_functions(self):
        """Update dependency_with_functions with actual function names from dependent files"""
        logger.info("Updating dependency function details...")
        
        for file_path, analysis in self.file_analyses.items():
            updated_deps = []
            
            for i, dep_file in enumerate(analysis.dependency_full_paths):
                # Find the analysis for this dependency
                dep_analysis = self.file_analyses.get(dep_file)
                
                if dep_analysis and dep_analysis.functions_defined:
                    # Format: "file.py::func1,func2,func3"
                    func_list = ','.join(dep_analysis.functions_defined[:5])  # Limit to first 5
                    if len(dep_analysis.functions_defined) > 5:
                        func_list += "..."
                    updated_deps.append(f"{dep_file}::{func_list}")
                else:
                    updated_deps.append(f"{dep_file}::[no functions]")
            
            analysis.dependency_with_functions = updated_deps
            
            # Same for dependents
            updated_dependents = []
            for dep_file in analysis.dependent_full_paths:
                dep_analysis = self.file_analyses.get(dep_file)
                
                if dep_analysis and dep_analysis.functions_defined:
                    func_list = ','.join(dep_analysis.functions_defined[:5])
                    if len(dep_analysis.functions_defined) > 5:
                        func_list += "..."
                    updated_dependents.append(f"{dep_file}::{func_list}")
                else:
                    updated_dependents.append(f"{dep_file}::[no functions]")
            
            analysis.dependent_with_functions = updated_dependents
    
    def _calculate_connections(self):
        """Calculate total connections for each file"""
        for analysis in self.file_analyses.values():
            analysis.total_connections = len(analysis.dependency_files) + len(analysis.dependent_files)
    
    def save_comprehensive_csv(self, output_path: str):
        """
        Save comprehensive dependency graph to CSV with all requested fields
        
        Args:
            output_path: Path to output CSV file
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header - exact fields requested
            writer.writerow([
                'file',
                'full_path',
                'language',
                'num_functions',
                'functions_defined',
                'dependencies',
                'dependency_files',
                'dependency_with_functions',
                'dependency_full_paths',
                'dependents',
                'dependent_files',
                'dependent_with_functions',
                'dependent_full_paths',
                'total_connections',
                # Additional useful fields
                'num_classes',
                'classes_defined',
                'global_variables',
                'exports'
            ])
            
            # Data rows
            for file_path, analysis in sorted(self.file_analyses.items()):
                writer.writerow([
                    analysis.file,
                    analysis.full_path,
                    analysis.language,
                    analysis.num_functions,
                    '; '.join(analysis.functions_defined),
                    '; '.join(analysis.dependencies),
                    '; '.join(analysis.dependency_files),
                    '; '.join(analysis.dependency_with_functions),
                    '; '.join(analysis.dependency_full_paths),
                    '; '.join(analysis.dependents),
                    '; '.join(analysis.dependent_files),
                    '; '.join(analysis.dependent_with_functions),
                    '; '.join(analysis.dependent_full_paths),
                    analysis.total_connections,
                    analysis.num_classes,
                    '; '.join(analysis.classes_defined),
                    '; '.join(analysis.global_variables),
                    '; '.join(analysis.exports)
                ])
        
        logger.info(f"✓ Saved comprehensive dependency graph to {output_path}")
    
    def save_detailed_functions_csv(self, output_path: str):
        """
        Save detailed function-level analysis
        
        Args:
            output_path: Path to output CSV file
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'file',
                'full_path',
                'type',
                'name',
                'start_line',
                'end_line',
                'parameters',
                'calls',
                'details'
            ])
            
            # Write function and class details
            for file_path, analysis in sorted(self.file_analyses.items()):
                # Functions
                for func in analysis.functions_with_details:
                    writer.writerow([
                        analysis.file,
                        analysis.full_path,
                        'function',
                        func['name'],
                        func.get('start_line', ''),
                        func.get('end_line', ''),
                        ', '.join(func.get('parameters', [])),
                        ', '.join(func.get('calls', [])),
                        f"Calls {len(func.get('calls', []))} functions"
                    ])
                
                # Classes
                for cls in analysis.classes_with_details:
                    methods = ', '.join(cls.get('methods', []))
                    inherits = ', '.join(cls.get('inherits_from', []))
                    
                    writer.writerow([
                        analysis.file,
                        analysis.full_path,
                        'class',
                        cls['name'],
                        cls.get('start_line', ''),
                        cls.get('end_line', ''),
                        '',
                        methods,
                        f"Methods: [{methods}]; Inherits: [{inherits}]"
                    ])
        
        logger.info(f"✓ Saved detailed function analysis to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        total_functions = sum(a.num_functions for a in self.file_analyses.values())
        total_classes = sum(a.num_classes for a in self.file_analyses.values())
        total_dependencies = sum(len(a.dependency_files) for a in self.file_analyses.values())
        total_dependents = sum(len(a.dependent_files) for a in self.file_analyses.values())
        
        languages = {}
        for analysis in self.file_analyses.values():
            languages[analysis.language] = languages.get(analysis.language, 0) + 1
        
        # Find most connected files
        most_connected = sorted(
            self.file_analyses.values(),
            key=lambda a: a.total_connections,
            reverse=True
        )[:10]
        
        # Find files with most dependents (most depended upon)
        most_depended = sorted(
            self.file_analyses.values(),
            key=lambda a: len(a.dependent_files),
            reverse=True
        )[:10]
        
        return {
            'total_files': len(self.file_analyses),
            'total_functions': total_functions,
            'total_classes': total_classes,
            'total_dependencies': total_dependencies,
            'total_dependents': total_dependents,
            'total_connections': total_dependencies + total_dependents,
            'languages': languages,
            'avg_functions_per_file': total_functions / max(len(self.file_analyses), 1),
            'avg_classes_per_file': total_classes / max(len(self.file_analyses), 1),
            'avg_connections_per_file': (total_dependencies + total_dependents) / max(len(self.file_analyses), 1),
            'most_connected_files': [(a.full_path, a.total_connections) for a in most_connected],
            'most_depended_upon': [(a.full_path, len(a.dependent_files)) for a in most_depended]
        }
    
    def print_statistics(self):
        """Print comprehensive statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE DEPENDENCY GRAPH STATISTICS")
        print("="*80)
        
        print(f"\n📊 Overview:")
        print(f"  Total Files: {stats['total_files']}")
        print(f"  Total Functions: {stats['total_functions']}")
        print(f"  Total Classes: {stats['total_classes']}")
        print(f"  Total Dependencies: {stats['total_dependencies']}")
        print(f"  Total Dependents: {stats['total_dependents']}")
        print(f"  Total Connections: {stats['total_connections']}")
        
        print(f"\n📈 Averages:")
        print(f"  Functions per File: {stats['avg_functions_per_file']:.2f}")
        print(f"  Classes per File: {stats['avg_classes_per_file']:.2f}")
        print(f"  Connections per File: {stats['avg_connections_per_file']:.2f}")
        
        print(f"\n🔤 Language Distribution:")
        for lang, count in sorted(stats['languages'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_files']) * 100
            print(f"  {lang}: {count} files ({percentage:.1f}%)")
        
        print(f"\n🔗 Most Connected Files (Top 10):")
        for i, (file_path, connections) in enumerate(stats['most_connected_files'], 1):
            print(f"  {i}. {file_path}: {connections} connections")
        
        print(f"\n⭐ Most Depended Upon Files (Top 10):")
        for i, (file_path, dependents) in enumerate(stats['most_depended_upon'], 1):
            print(f"  {i}. {file_path}: {dependents} dependents")
        
        print("="*80 + "\n")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Build comprehensive dependency graph using LLM analysis'
    )
    parser.add_argument(
        '--repo-path',
        required=True,
        help='Path to repository to analyze'
    )
    parser.add_argument(
        '--output',
        default='dependency_graph_comprehensive.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--functions-output',
        default='functions_detailed.csv',
        help='Detailed functions output CSV'
    )
    parser.add_argument(
        '--api-key',
        help='OpenAI API key (or set OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o',
        help='OpenAI model to use'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=3,
        help='Number of parallel workers'
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = ""
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY or use --api-key")
        return 1
    
    # Build dependency graph
    builder = EnhancedLLMDependencyBuilder(api_key=api_key, model=args.model)
    
    try:
        builder.build_dependency_graph(
            repo_path=Path(args.repo_path),
            max_workers=args.max_workers
        )
        
        # Save outputs
        builder.save_comprehensive_csv(args.output)
        builder.save_detailed_functions_csv(args.functions_output)
        
        # Print statistics
        builder.print_statistics()
        
        print(f"\n✓ Success! Outputs saved:")
        print(f"  - {args.output}")
        print(f"  - {args.functions_output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to build dependency graph: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())