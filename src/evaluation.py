from typing import List, Dict
from src.models import MinimalSource, AnsweredQuestion, MinimalSearchResults


def calculate_overlap(retrieved: MinimalSource, correct: MinimalSource) -> float:
    if retrieved.file_path != correct.file_path:
        return 0.0

    start = max(retrieved.first_character_index, correct.first_character_index)
    end = min(retrieved.last_character_index, correct.last_character_index)

    if start >= end:
        return 0.0

    overlap_length = end - start
    correct_length = correct.last_character_index - correct.first_character_index

    if correct_length == 0:
        return 0.0

    return overlap_length / correct_length


def calculate_recall_at_k(
    retrieved_sources: List[MinimalSource],
    correct_sources: List[MinimalSource],
    overlap_threshold: float = 0.05
) -> float:
    if not correct_sources:
        return 1.0

    found_count = 0

    for correct_source in correct_sources:
        for retrieved_source in retrieved_sources:
            overlap = calculate_overlap(retrieved_source, correct_source)
            if overlap >= overlap_threshold:
                found_count += 1
                break

    return found_count / len(correct_sources)


def evaluate_dataset(
    search_results: List[MinimalSearchResults],
    answered_questions: List[AnsweredQuestion],
    overlap_threshold: float = 0.05
) -> Dict[str, float]:
    question_map = {q.question_id: q for q in answered_questions}

    recalls = []

    for result in search_results:
        if result.question_id not in question_map:
            continue

        correct_question = question_map[result.question_id]
        recall = calculate_recall_at_k(
            result.retrieved_sources,
            correct_question.sources,
            overlap_threshold
        )
        recalls.append(recall)

    if not recalls:
        return {
            'average_recall_at_k': 0.0,
            'num_questions': 0
        }

    return {
        'average_recall_at_k': sum(recalls) / len(recalls),
        'num_questions': len(recalls),
        'min_recall': min(recalls),
        'max_recall': max(recalls)
    }
