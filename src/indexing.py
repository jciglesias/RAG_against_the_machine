from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from src.chunking import get_chunker, Chunk
from src.retrieval import BM25Retriever, TFIDFRetriever
from src.config import VLLM_REPO_PATH


class Indexer:

    def __init__(self, retrieval_method: str = "bm25"):
        self.retrieval_method = retrieval_method
        if retrieval_method == "bm25":
            self.retriever = BM25Retriever()
        elif retrieval_method == "tfidf":
            self.retriever = TFIDFRetriever()
        else:
            raise ValueError(f"Unknown retrieval method: {retrieval_method}")

    def index_repository(self, repo_path: Path = VLLM_REPO_PATH, selective_files: Optional[List[str]] = None):
        print(f"Indexing repository: {repo_path}")

        if selective_files:
            files_to_index = [repo_path / f for f in selective_files if (repo_path / f).exists()]
        else:
            files_to_index = self._collect_files(repo_path)

        print(f"Found {len(files_to_index)} files to index")

        all_chunks: List[Chunk] = []

        for file_path in tqdm(files_to_index, desc="Chunking files"):
            try:
                chunks = self._chunk_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error chunking {file_path}: {e}")
                continue

        print(f"Created {len(all_chunks)} chunks")

        print("Building retrieval index...")
        self.retriever.build_index(all_chunks)

        print("Indexing complete!")

    def _collect_files(self, repo_path: Path) -> List[Path]:
        files = []

        include_patterns = ['**/*.py', '**/*.md', '**/*.rst', '**/*.txt']

        exclude_patterns = [
            '**/test_*.py',
            '**/*_test.py',
            '**/tests/**',
            '**/__pycache__/**',
            '**/.git/**',
            '**/node_modules/**',
            '**/.venv/**',
            '**/venv/**',
            '**/*.pyc',
            '**/.DS_Store'
        ]

        for pattern in include_patterns:
            for file_path in repo_path.glob(pattern):
                if file_path.is_file():
                    should_exclude = False
                    for exclude_pattern in exclude_patterns:
                        if file_path.match(exclude_pattern):
                            should_exclude = True
                            break

                    if not should_exclude:
                        files.append(file_path)

        return files

    def _chunk_file(self, file_path: Path) -> List[Chunk]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            chunker = get_chunker(str(file_path))
            chunks = chunker.chunk(content, str(file_path))

            return chunks
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

    def save_index(self, output_path: Path):
        self.retriever.save(output_path)
        print(f"Index saved to {output_path}")

    def load_index(self, index_path: Path):
        self.retriever.load(index_path)
        print(f"Index loaded from {index_path}")
