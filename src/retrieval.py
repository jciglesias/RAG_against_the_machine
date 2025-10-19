import pickle
from pathlib import Path
from typing import List, Dict
from collections import Counter
import math
import re
from tqdm import tqdm

from src.models import MinimalSource
from src.chunking import Chunk


class BM25Retriever:

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_lens: List[int] = []
        self.avgdl: float = 0
        self.tokenized_docs: List[List[str]] = []

    def tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'\b\w+\b', text.lower())

        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'should', 'could', 'may', 'might', 'must', 'can'}

        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]

        return tokens

    def build_index(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.tokenized_docs = []
        self.doc_lens = []

        print("Tokenizing documents...")
        for chunk in tqdm(chunks):
            tokens = self.tokenize(chunk.content)
            self.tokenized_docs.append(tokens)
            self.doc_lens.append(len(tokens))

        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0

        print("Calculating document frequencies...")
        self.doc_freqs = {}
        for tokens in tqdm(self.tokenized_docs):
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        print("Calculating IDF scores...")
        num_docs = len(self.chunks)
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)

    def search(self, query: str, k: int = 10) -> List[MinimalSource]:
        query_tokens = self.tokenize(query)

        scores = []
        for idx, (tokens, doc_len) in enumerate(zip(self.tokenized_docs, self.doc_lens)):
            score = self._calculate_bm25_score(query_tokens, tokens, doc_len)
            scores.append((score, idx))

        scores.sort(reverse=True, key=lambda x: x[0])

        results = []
        for score, idx in scores[:k]:
            if score > 0:
                chunk = self.chunks[idx]
                results.append(MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.start_idx,
                    last_character_index=chunk.end_idx
                ))

        return results

    def _calculate_bm25_score(self, query_tokens: List[str], doc_tokens: List[str], doc_len: int) -> float:
        score = 0.0
        doc_token_counts = Counter(doc_tokens)

        for token in query_tokens:
            if token not in self.idf:
                continue

            tf = doc_token_counts.get(token, 0)
            idf = self.idf[token]

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))

            score += idf * (numerator / denominator)

        return score

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'doc_freqs': self.doc_freqs,
                'idf': self.idf,
                'doc_lens': self.doc_lens,
                'avgdl': self.avgdl,
                'tokenized_docs': self.tokenized_docs,
                'k1': self.k1,
                'b': self.b
            }, f)

    def load(self, path: Path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.doc_freqs = data['doc_freqs']
            self.idf = data['idf']
            self.doc_lens = data['doc_lens']
            self.avgdl = data['avgdl']
            self.tokenized_docs = data['tokenized_docs']
            self.k1 = data['k1']
            self.b = data['b']


class TFIDFRetriever:

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.tokenized_docs: List[List[str]] = []
        self.tf_idf_vectors: List[Dict[str, float]] = []

    def tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'\b\w+\b', text.lower())

        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'should', 'could', 'may', 'might', 'must', 'can'}

        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]

        return tokens

    def build_index(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.tokenized_docs = []

        print("Tokenizing documents...")
        for chunk in tqdm(chunks):
            tokens = self.tokenize(chunk.content)
            self.tokenized_docs.append(tokens)

        print("Calculating document frequencies...")
        self.doc_freqs = {}
        for tokens in tqdm(self.tokenized_docs):
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        print("Calculating IDF scores...")
        num_docs = len(self.chunks)
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log(num_docs / freq)

        print("Calculating TF-IDF vectors...")
        self.tf_idf_vectors = []
        for tokens in tqdm(self.tokenized_docs):
            token_counts = Counter(tokens)
            tf_idf_vector = {}

            for token, count in token_counts.items():
                tf = count / len(tokens) if tokens else 0
                idf = self.idf.get(token, 0)
                tf_idf_vector[token] = tf * idf

            self.tf_idf_vectors.append(tf_idf_vector)

    def search(self, query: str, k: int = 10) -> List[MinimalSource]:
        query_tokens = self.tokenize(query)
        query_token_counts = Counter(query_tokens)

        query_vector = {}
        for token, count in query_token_counts.items():
            tf = count / len(query_tokens) if query_tokens else 0
            idf = self.idf.get(token, 0)
            query_vector[token] = tf * idf

        scores = []
        for idx, doc_vector in enumerate(self.tf_idf_vectors):
            score = self._cosine_similarity(query_vector, doc_vector)
            scores.append((score, idx))

        scores.sort(reverse=True, key=lambda x: x[0])

        results = []
        for score, idx in scores[:k]:
            if score > 0:
                chunk = self.chunks[idx]
                results.append(MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.start_idx,
                    last_character_index=chunk.end_idx
                ))

        return results

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        common_terms = set(vec1.keys()) & set(vec2.keys())

        if not common_terms:
            return 0.0

        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)

        mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'doc_freqs': self.doc_freqs,
                'idf': self.idf,
                'tokenized_docs': self.tokenized_docs,
                'tf_idf_vectors': self.tf_idf_vectors
            }, f)

    def load(self, path: Path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.doc_freqs = data['doc_freqs']
            self.idf = data['idf']
            self.tokenized_docs = data['tokenized_docs']
            self.tf_idf_vectors = data['tf_idf_vectors']
