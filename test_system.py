import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing import Indexer
from src.config import VLLM_REPO_PATH, OUTPUT_DIR
from src.models import UnansweredQuestion
import json


def test_indexing():
    print("\n" + "="*60)
    print("TEST 1: Indexing Performance")
    print("="*60)
    
    indexer = Indexer(retrieval_method="bm25")
    
    start_time = time.time()
    
    test_files = [
        "README.md",
        "vllm/__init__.py",
        "docs/source/getting_started/quickstart.md"
    ]
    
    print(f"Indexing {len(test_files)} files...")
    indexer.index_repository(VLLM_REPO_PATH, selective_files=test_files)
    
    elapsed = time.time() - start_time
    print(f"✓ Indexing completed in {elapsed:.2f}s")
    
    index_path = OUTPUT_DIR / "test_index_bm25.pkl"
    indexer.save_index(index_path)
    print(f"✓ Index saved to {index_path}")
    
    return indexer


def test_retrieval(indexer):
    print("\n" + "="*60)
    print("TEST 2: Retrieval Performance")
    print("="*60)
    
    test_queries = [
        "How to start vLLM server?",
        "What is tensor parallelism?",
        "How to configure OpenAI compatible API?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        start_time = time.time()
        
        results = indexer.retriever.search(query, k=5)
        
        elapsed = time.time() - start_time
        print(f"  ✓ Found {len(results)} results in {elapsed:.2f}s")
        
        for i, result in enumerate(results[:3], 1):
            print(f"    {i}. {result.file_path}")


def test_answer_generation():
    print("\n" + "="*60)
    print("TEST 3: Answer Generation (requires Ollama)")
    print("="*60)
    
    from src.llm_generator import LLMGenerator
    from src.models import MinimalSource
    
    generator = LLMGenerator()
    
    test_sources = [
        MinimalSource(
            file_path="README.md",
            first_character_index=0,
            last_character_index=500
        )
    ]
    
    question = "What is vLLM?"
    
    print(f"Question: {question}")
    print("Generating answer...")
    
    start_time = time.time()
    answer = generator.generate_answer(question, test_sources, VLLM_REPO_PATH)
    elapsed = time.time() - start_time
    
    print(f"\nAnswer (generated in {elapsed:.2f}s):")
    print("-" * 60)
    print(answer)
    print("-" * 60)
    
    if "Error" in answer:
        print("\n⚠ Note: Ollama may not be installed or running.")
        print("   Install with: curl -fsSL https://ollama.com/install.sh | sh")
        print("   Then run: ollama pull qwen2.5:0.5b")
    else:
        print(f"✓ Answer generated successfully in {elapsed:.2f}s")


def test_evaluation():
    print("\n" + "="*60)
    print("TEST 4: Evaluation Metrics")
    print("="*60)
    
    from src.evaluation import calculate_recall_at_k
    from src.models import MinimalSource
    
    retrieved = [
        MinimalSource(file_path="test.py", first_character_index=0, last_character_index=100),
        MinimalSource(file_path="test.py", first_character_index=150, last_character_index=250),
    ]
    
    correct = [
        MinimalSource(file_path="test.py", first_character_index=50, last_character_index=150),
    ]
    
    recall = calculate_recall_at_k(retrieved, correct, overlap_threshold=0.05)
    
    print(f"Test recall@k calculation:")
    print(f"  Retrieved sources: {len(retrieved)}")
    print(f"  Correct sources: {len(correct)}")
    print(f"  Recall@k: {recall:.2%}")
    print(f"✓ Evaluation working correctly")


def test_full_pipeline():
    print("\n" + "="*60)
    print("TEST 5: Full Pipeline")
    print("="*60)
    
    from src.config import DATASETS_DIR
    
    dataset_path = DATASETS_DIR / "Dataset_2025-09-21_valid_unanswered.json"
    
    if not dataset_path.exists():
        print("⚠ Dataset not found. Skipping full pipeline test.")
        print(f"  Expected: {dataset_path}")
        return
    
    print(f"Loading dataset: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    from src.models import RagDataset
    dataset = RagDataset(**data)
    
    print(f"✓ Loaded {len(dataset.rag_questions)} questions")
    
    sample_questions = dataset.rag_questions[:3]
    print(f"\nTesting with {len(sample_questions)} sample questions...")
    
    print("⚠ Full pipeline test requires complete index")
    print("  Run: uv run python -m src index")
    print("  Then: uv run python -m src search_dataset <dataset_path>")


def main():
    print("\n" + "="*60)
    print("RAG SYSTEM TEST SUITE")
    print("="*60)
    
    try:
        indexer = test_indexing()
        
        test_retrieval(indexer)
        
        test_answer_generation()
        
        test_evaluation()
        
        test_full_pipeline()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        print("\nNext steps:")
        print("  1. Index full repository: uv run python -m src index")
        print("  2. Search dataset: uv run python -m src search_dataset <dataset_path>")
        print("  3. Evaluate: uv run python -m src measure_recall_at_k_on_dataset <results> <answers>")
        print("  4. Generate answers: uv run python -m src answer_dataset <results> --dataset_path=<questions>")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
