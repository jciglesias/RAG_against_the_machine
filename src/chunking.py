import ast
import re
from typing import List, Tuple
from pathlib import Path
from src.config import MAX_CHUNK_SIZE


class Chunk:

    def __init__(self, content: str, file_path: str, start_idx: int, end_idx: int):
        self.content = content
        self.file_path = file_path
        self.start_idx = start_idx
        self.end_idx = end_idx


class PythonCodeChunker:

    def __init__(self, max_chunk_size: int = MAX_CHUNK_SIZE):
        self.max_chunk_size = max_chunk_size

    def chunk(self, code: str, file_path: str) -> List[Chunk]:
        chunks = []

        try:
            tree = ast.parse(code)

            for node in ast.iter_child_nodes(tree):
                if isinstance(
                    node,
                    (ast.FunctionDef,
                     ast.ClassDef,
                     ast.AsyncFunctionDef)):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno

                    lines = code.split('\n')
                    chunk_content = '\n'.join(lines[start_line:end_line])

                    start_idx = len(
                        '\n'.join(lines[:start_line])) + (1 if start_line > 0 else 0)
                    end_idx = start_idx + len(chunk_content)

                    if len(chunk_content) <= self.max_chunk_size:
                        chunks.append(
                            Chunk(
                                chunk_content,
                                file_path,
                                start_idx,
                                end_idx))
                    else:
                        chunks.extend(
                            self._split_large_chunk(
                                chunk_content, file_path, start_idx))

        except SyntaxError:
            chunks = self._simple_chunk(code, file_path)

        return chunks

    def _split_large_chunk(
            self,
            content: str,
            file_path: str,
            start_idx: int) -> List[Chunk]:
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        chunk_start_idx = start_idx

        for line in lines:
            line_size = len(line) + 1

            if current_size + line_size > self.max_chunk_size and current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunk_end_idx = chunk_start_idx + len(chunk_content)
                chunks.append(
                    Chunk(
                        chunk_content,
                        file_path,
                        chunk_start_idx,
                        chunk_end_idx))

                chunk_start_idx = chunk_end_idx + 1
                current_chunk = []
                current_size = 0

            current_chunk.append(line)
            current_size += line_size

        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunk_end_idx = chunk_start_idx + len(chunk_content)
            chunks.append(
                Chunk(
                    chunk_content,
                    file_path,
                    chunk_start_idx,
                    chunk_end_idx))

        return chunks

    def _simple_chunk(self, content: str, file_path: str) -> List[Chunk]:
        chunks = []
        start_idx = 0

        while start_idx < len(content):
            end_idx = min(start_idx + self.max_chunk_size, len(content))
            chunk_content = content[start_idx:end_idx]
            chunks.append(Chunk(chunk_content, file_path, start_idx, end_idx))
            start_idx = end_idx

        return chunks


class DocumentationChunker:

    def __init__(self, max_chunk_size: int = MAX_CHUNK_SIZE, overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk(self, content: str, file_path: str) -> List[Chunk]:
        chunks = []

        sections = self._split_by_headers(content)

        for section_content, section_start in sections:
            if len(section_content) <= self.max_chunk_size:
                chunks.append(Chunk(
                    section_content,
                    file_path,
                    section_start,
                    section_start + len(section_content)
                ))
            else:
                chunks.extend(self._split_with_overlap(
                    section_content,
                    file_path,
                    section_start
                ))

        return chunks

    def _split_by_headers(self, content: str) -> List[Tuple[str, int]]:
        header_pattern = re.compile(r'^#{1,6}\s+.+$', re.MULTILINE)

        sections = []
        matches = list(header_pattern.finditer(content))

        if not matches:
            return [(content, 0)]

        if matches[0].start() > 0:
            sections.append((content[:matches[0].start()], 0))

        for i, match in enumerate(matches):
            start_idx = match.start()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start_idx:end_idx]
            sections.append((section_content, start_idx))

        return sections

    def _split_with_overlap(self, content: str, file_path: str,
                            base_start_idx: int) -> List[Chunk]:
        chunks = []
        sentences = self._split_sentences(content)

        current_chunk = []
        current_size = 0
        chunk_start_idx = base_start_idx

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunk_content = ''.join(current_chunk)
                chunk_end_idx = chunk_start_idx + len(chunk_content)
                chunks.append(
                    Chunk(
                        chunk_content,
                        file_path,
                        chunk_start_idx,
                        chunk_end_idx))

                overlap_size = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break

                chunk_start_idx = chunk_end_idx - overlap_size
                current_chunk = overlap_sentences
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunk_content = ''.join(current_chunk)
            chunk_end_idx = chunk_start_idx + len(chunk_content)
            chunks.append(
                Chunk(
                    chunk_content,
                    file_path,
                    chunk_start_idx,
                    chunk_end_idx))

        return chunks

    def _split_sentences(self, content: str) -> List[str]:
        sentence_pattern = re.compile(r'[.!?]\s+|\n\n')
        sentences = sentence_pattern.split(content)

        result = []
        for i, sentence in enumerate(sentences[:-1]):
            result.append(sentence + '. ')
        if sentences:
            result.append(sentences[-1])

        return result


def get_chunker(file_path: str):
    path = Path(file_path)

    if path.suffix == '.py':
        return PythonCodeChunker()
    elif path.suffix in ['.md', '.rst', '.txt']:
        return DocumentationChunker()
    else:
        return DocumentationChunker()
