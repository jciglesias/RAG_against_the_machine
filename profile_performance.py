import time
import sys
from pathlib import Path
from typing import List
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing import Indexer
from src.config import VLLM_REPO_PATH, OUTPUT_DIR, DATASETS_DIR, MAX_INDEXING_TIME
from src.models import RagDataset


def profile_indexing(method: str = "bm25") -> dict:
    print(f"\n{'='*60}")
    print(f"Profiling Indexing ({method.upper()})")
    print(f"{'='*60}")
    
    indexer = Indexer(retrieval_method=method)
    
    start_time = time.time()
    indexer.index_repository(VLLM_REPO_PATH)
    elapsed = time.time() - start_time
    
    num_chunks = len(indexer.retriever.chunks)
    
    index_path = OUTPUT_DIR / f"index_{method}.pkl"
    indexer.save_index(index_path)
    
    results = {
        "method": method,
        "elapsed_time": elapsed,
        "num_chunks": num_chunks,
        "chunks_per_second": num_chunks / elapsed if elapsed > 0 else 0,
        "within_target": elapsed <= MAX_INDEXING_TIME,
        "target_time": MAX_INDEXING_TIME
    }
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s (target: {MAX_INDEXING_TIME}s)")
    print(f"  Chunks: {num_chunks}")
    print(f"  Speed: {results['chunks_per_second']:.2f} chunks/s")
    print(f"  Status: {'✓ PASS' if results['within_target'] else '✗ FAIL'}")
    
    return results


def profile_retrieval(method: str = "bm25", num_queries: int = 100) -> dict:
    print(f"\n{'='*60}")
    print(f"Profiling Retrieval ({method.upper()})")
    print(f"{'='*60}")
    
    indexer = Indexer(retrieval_method=method)
    index_path = OUTPUT_DIR / f"index_{method}.pkl"
    
    if not index_path.exists():
        print(f"Index not found at {index_path}")
        print("Run indexing first!")
        return {}
    
    indexer.load_index(index_path)
    
    dataset_path = DATASETS_DIR / "Dataset_2025-09-21_valid_unanswered.json"
    
    if not dataset_path.exists():
        queries = [
            "How to start vLLM server?",
            "What is tensor parallelism?",
            "How to configure OpenAI API?",
            "What models are supported?",
            "How to enable GPU?"
        ] * 20
        queries = queries[:num_queries]
    else:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        dataset = RagDataset(**data)
        queries = [q.question for q in dataset.rag_questions[:num_queries]]
    
    print(f"Testing with {len(queries)} queries...")
    
    times = []
    results_counts = []
    
    for query in queries:
        start = time.time()
        results = indexer.retriever.search(query, k=10)
        elapsed = time.time() - start
        
        times.append(elapsed)
        results_counts.append(len(results))
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    avg_results = sum(results_counts) / len(results_counts)
    
    results = {
        "method": method,
        "num_queries": len(queries),
        "avg_time": avg_time,
        "max_time": max_time,
        "min_time": min_time,
        "avg_results": avg_results,
        "queries_per_second": 1 / avg_time if avg_time > 0 else 0
    }
    
    print(f"\nResults:")
    print(f"  Queries: {len(queries)}")
    print(f"  Avg time: {avg_time:.4f}s")
    print(f"  Max time: {max_time:.4f}s")
    print(f"  Min time: {min_time:.4f}s")
    print(f"  Throughput: {results['queries_per_second']:.2f} queries/s")
    print(f"  Avg results: {avg_results:.1f}")
    
    return results


def compare_methods():
    print(f"\n{'='*60}")
    print("Comparing Retrieval Methods")
    print(f"{'='*60}")
    
    methods = ["bm25", "tfidf"]
    comparison = []
    
    for method in methods:
        print(f"\nTesting {method.upper()}...")
        
        indexer = Indexer(retrieval_method=method)
        
        test_files = [
            "README.md",
            "vllm/__init__.py",
            "docs/source/getting_started/quickstart.md"
        ]
        
        start = time.time()
        indexer.index_repository(VLLM_REPO_PATH, selective_files=test_files)
        index_time = time.time() - start
        
        query = "How to start vLLM server?"
        start = time.time()
        results = indexer.retriever.search(query, k=10)
        retrieval_time = time.time() - start
        
        comparison.append({
            "method": method,
            "index_time": index_time,
            "retrieval_time": retrieval_time,
            "num_results": len(results)
        })
    
    print(f"\n{'='*60}")
    print("Comparison Results")
    print(f"{'='*60}")
    print(f"\n{'Method':<10} {'Index Time':<15} {'Retrieval Time':<15}")
    print("-" * 40)
    
    for c in comparison:
        print(f"{c['method'].upper():<10} {c['index_time']:.4f}s{'':<8} {c['retrieval_time']:.4f}s")
    
    fastest_retrieval = min(comparison, key=lambda x: x['retrieval_time'])
    print(f"\nRecommendation: {fastest_retrieval['method'].upper()} for fastest retrieval")


def optimization_tips():
    print(f"\n{'='*60}")
    print("Optimization Tips")
    print(f"{'='*60}")
    
    tips = [
        ("Indexing Speed", [
            "- Use selective file indexing for testing",
            "- Reduce MAX_CHUNK_SIZE for fewer chunks",
            "- Consider parallel processing for large repos",
            "- Cache tokenization results"
        ]),
        ("Retrieval Speed", [
            "- Use smaller k values when possible",
            "- Pre-compute and cache common queries",
            "- Consider TF-IDF for speed vs BM25 for accuracy",
            "- Optimize tokenization (fewer stopwords)"
        ]),
        ("Memory Usage", [
            "- Serialize and load indexes instead of rebuilding",
            "- Use sparse representations for TF-IDF vectors",
            "- Limit max context window size",
            "- Stream large result sets"
        ]),
        ("Answer Quality", [
            "- Tune BM25 parameters (k1, b)",
            "- Increase chunk overlap for better context",
            "- Use hybrid retrieval (lexical + semantic)",
            "- Fine-tune LLM prompts"
        ])
    ]
    
    for category, tip_list in tips:
        print(f"\n{category}:")
        for tip in tip_list:
            print(f"  {tip}")


def main():
    print("RAG System Performance Profiling")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["index", "retrieval", "compare", "all"], 
                       default="all", help="What to profile")
    parser.add_argument("--method", choices=["bm25", "tfidf"], default="bm25",
                       help="Retrieval method to use")
    parser.add_argument("--queries", type=int, default=100,
                       help="Number of queries for retrieval profiling")
    
    args = parser.parse_args()
    
    try:
        if args.profile in ["index", "all"]:
            profile_indexing(args.method)
        
        if args.profile in ["retrieval", "all"]:
            profile_retrieval(args.method, args.queries)
        
        if args.profile in ["compare", "all"]:
            compare_methods()
        
        optimization_tips()
        
        print(f"\n{'='*60}")
        print("Profiling Complete!")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\nProfiling interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
