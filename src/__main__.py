import json
import fire
from pathlib import Path
from typing import Optional
import time

from src.indexing import Indexer
from src.models import (
    AnsweredQuestion,
    RagDataset,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)
from src.evaluation import evaluate_dataset
from src.config import (
    DEFAULT_K,
    OUTPUT_DIR,
    OVERLAP_THRESHOLD,
    VLLM_REPO_PATH,
    DEFAULT_MODEL
)
from src.llm_generator import LLMGenerator, ContextManager


class RAGSystem:

    def __init__(self, retrieval_method: str = "bm25", model: str = DEFAULT_MODEL):
        self.retrieval_method = retrieval_method
        self.indexer = Indexer(retrieval_method=retrieval_method)
        self.index_path = OUTPUT_DIR / f"index_{retrieval_method}.pkl"
        self.llm_generator = LLMGenerator(model=model)
        self.context_manager = ContextManager()

    def index(self, repo_path: Optional[str] = None):
        from src.config import VLLM_REPO_PATH

        start_time = time.time()

        if repo_path:
            repo_path = Path(repo_path)
        else:
            repo_path = VLLM_REPO_PATH

        self.indexer.index_repository(repo_path)
        self.indexer.save_index(self.index_path)

        elapsed_time = time.time() - start_time
        print(f"Indexing completed in {elapsed_time:.2f} seconds")

    def search(self, query: str, k: int = DEFAULT_K):
        if not hasattr(self.indexer.retriever, 'chunks') or not self.indexer.retriever.chunks:
            print(f"Loading index from {self.index_path}")
            self.indexer.load_index(self.index_path)

        start_time = time.time()

        results = self.indexer.retriever.search(query, k=k)

        elapsed_time = time.time() - start_time

        print(f"\nFound {len(results)} results in {elapsed_time:.2f} seconds:\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. {result.file_path}")
            print(f"   Characters {result.first_character_index}-{result.last_character_index}\n")

    def search_dataset(self, dataset_path: str, k: int = DEFAULT_K, output_path: Optional[str] = None):
        if not hasattr(self.indexer.retriever, 'chunks') or not self.indexer.retriever.chunks:
            print(f"Loading index from {self.index_path}")
            self.indexer.load_index(self.index_path)

        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        dataset = RagDataset(**data)

        search_results = []

        print(f"Searching {len(dataset.rag_questions)} questions...")

        for question in dataset.rag_questions:
            start_time = time.time()

            results = self.indexer.retriever.search(question.question, k=k)

            elapsed_time = time.time() - start_time

            search_results.append(MinimalSearchResults(
                question_id=question.question_id,
                retrieved_sources=results
            ))

            print(f"Question {question.question_id}: {len(results)} results in {elapsed_time:.2f}s")

        output = StudentSearchResults(
            search_results=search_results,
            k=k
        )

        if output_path is None:
            dataset_name = Path(dataset_path).stem
            output_path = OUTPUT_DIR / "search_results" / f"{dataset_name}.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output.model_dump(), f, indent=2)

        print(f"\nResults saved to {output_path}")

    def measure_recall_at_k_on_dataset(self, search_results_path: str, answered_questions_path: str):
        print(f"Loading search results from {search_results_path}")
        with open(search_results_path, 'r') as f:
            search_data = json.load(f)

        search_results = StudentSearchResults(**search_data)

        print(f"Loading answered questions from {answered_questions_path}")
        with open(answered_questions_path, 'r') as f:
            answer_data = json.load(f)

        dataset = RagDataset(**answer_data)
        answered_questions = [q for q in dataset.rag_questions if isinstance(q, AnsweredQuestion)]

        print("\nEvaluating...")
        metrics = evaluate_dataset(
            search_results.search_results,
            answered_questions,
            overlap_threshold=OVERLAP_THRESHOLD
        )

        print("\n" + "="*50)
        print("Evaluation Results")
        print("="*50)
        print(f"Number of questions: {metrics['num_questions']}")
        print(f"Average Recall@{search_results.k}: {metrics['average_recall_at_k']:.2%}")
        if 'min_recall' in metrics:
            print(f"Min Recall: {metrics['min_recall']:.2%}")
            print(f"Max Recall: {metrics['max_recall']:.2%}")
        print("="*50)

    def answer_dataset(self, search_results_path: str, dataset_path: Optional[str] = None, output_path: Optional[str] = None, repo_path: Optional[str] = None):
        print(f"Loading search results from {search_results_path}")
        with open(search_results_path, 'r') as f:
            search_data = json.load(f)

        search_results = StudentSearchResults(**search_data)

        question_map = {}
        if dataset_path:
            print(f"Loading questions from {dataset_path}")
            with open(dataset_path, 'r') as f:
                dataset_data = json.load(f)
            dataset = RagDataset(**dataset_data)
            question_map = {q.question_id: q.question for q in dataset.rag_questions}

        if repo_path:
            repo_path = Path(repo_path)
        else:
            repo_path = VLLM_REPO_PATH

        answers = []

        print(f"Generating answers for {len(search_results.search_results)} questions...")

        for i, result in enumerate(search_results.search_results, 1):
            start_time = time.time()

            question_text = question_map.get(result.question_id, f"Question {result.question_id}")

            prioritized_sources = self.context_manager.prioritize_sources(
                result.retrieved_sources,
                repo_path,
                question_text
            )

            answer_text = self.llm_generator.generate_answer(
                question_text,
                prioritized_sources,
                repo_path
            )

            elapsed_time = time.time() - start_time

            answer = MinimalAnswer(
                question_id=result.question_id,
                retrieved_sources=result.retrieved_sources,
                answer=answer_text
            )
            answers.append(answer)

            print(f"  [{i}/{len(search_results.search_results)}] Generated answer in {elapsed_time:.2f}s")

        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_results.k
        )

        if output_path is None:
            dataset_name = Path(search_results_path).stem
            output_path = OUTPUT_DIR / "answers" / f"{dataset_name}_answers.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output.model_dump(), f, indent=2)

        print(f"\nAnswers saved to {output_path}")

    def answer(self, query: str, k: int = DEFAULT_K, repo_path: Optional[str] = None):
        if not hasattr(self.indexer.retriever, 'chunks') or not self.indexer.retriever.chunks:
            print(f"Loading index from {self.index_path}")
            self.indexer.load_index(self.index_path)

        if repo_path:
            repo_path = Path(repo_path)
        else:
            repo_path = VLLM_REPO_PATH

        print(f"Searching for relevant sources...")
        start_search = time.time()
        results = self.indexer.retriever.search(query, k=k)
        search_time = time.time() - start_search

        print(f"\nFound {len(results)} relevant sources in {search_time:.2f}s:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.file_path}")
            print(f"     Characters {result.first_character_index}-{result.last_character_index}")

        prioritized_sources = self.context_manager.prioritize_sources(
            results,
            repo_path,
            query
        )

        print(f"\nGenerating answer...")
        start_gen = time.time()
        answer_text = self.llm_generator.generate_answer(
            query,
            prioritized_sources,
            repo_path
        )
        gen_time = time.time() - start_gen

        print(f"\n{'='*60}")
        print(f"Answer (generated in {gen_time:.2f}s):")
        print("=" * 60)
        print(answer_text)
        print("=" * 60)


def main():
    fire.Fire(RAGSystem)


if __name__ == "__main__":
    main()
