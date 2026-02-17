#!/usr/bin/env python3
"""
Test script to verify RAG retrieval logic for the test question.
"""

import json
import difflib

def _norm(text: str) -> str:
    """Normalize text by lowercasing and collapsing whitespace."""
    return " ".join((text or "").lower().split())

def retrieve_context(query: str, dataset: list[dict], k: int = 3) -> tuple[str, str]:
    """Retrieve relevant context from dataset using similarity matching.
    Returns: (context_string, best_match_answer)"""
    q = _norm(query)
    if not q or not dataset:
        return "", ""
    
    scored = []
    for entry in dataset:
        question = _norm(entry.get("question", ""))
        answer = _norm(entry.get("answer", ""))
        
        # Prioritize question matching over answer matching
        question_score = difflib.SequenceMatcher(None, q, question).ratio()
        # Also check if query words appear in question
        q_words = set(q.split())
        question_words = set(question.split())
        word_overlap = len(q_words & question_words) / max(len(q_words), 1)
        
        # Combined score: 70% question match, 20% word overlap, 10% answer match
        combined_score = (question_score * 0.7) + (word_overlap * 0.2) + (difflib.SequenceMatcher(None, q, answer).ratio() * 0.1)
        
        scored.append((combined_score, question_score, entry))

    # Sort by combined score, then by question score
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = [e for s, qs, e in scored[:k] if s > 0]
    if not top:
        return "", ""
    
    # Get the best match answer (highest combined score)
    best_match_answer = top[0].get('answer', '') if top else ''
    best_score = scored[0][0] if scored else 0.0
    best_question_score = scored[0][1] if scored else 0.0
    
    print(f"🔵 [DEBUG] Best match - Combined score: {best_score:.3f}, Question score: {best_question_score:.3f}")
    if top:
        print(f"🔵 [DEBUG] Best match question: '{top[0].get('question', '')[:80]}...'")
    
    # Use dataset answer if question match is good (>= 0.5) or combined score is high (>= 0.4)
    # Lowered threshold to catch more matches
    use_dataset_answer = best_question_score >= 0.4 or best_score >= 0.3
    
    context = "\n\n".join([f"[{i+1}] Question: {e.get('question','')}\nAnswer: {e.get('answer','')}" for i, e in enumerate(top)])
    return context, best_match_answer if use_dataset_answer else ""

# Load dataset
with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Test question
test_question = "Can you give a concrete example of a balanced schedule for a student?"

print("=" * 80)
print("RAG RETRIEVAL TEST")
print("=" * 80)
print(f"\nTest Question: {test_question}\n")

context, dataset_answer = retrieve_context(test_question, dataset, k=3)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

if dataset_answer:
    print(f"\n✅ Dataset Answer Found (Length: {len(dataset_answer)} chars)")
    print(f"\nAnswer Preview (first 200 chars):")
    print(dataset_answer[:200] + "...")
    
    # Check if it matches expected answer
    expected_entry = next((e for e in dataset if _norm(e.get("question", "")) == _norm(test_question)), None)
    if expected_entry:
        expected_answer = expected_entry.get("answer", "")
        if dataset_answer.strip() == expected_answer.strip():
            print("\n✅ PERFECT MATCH: Retrieved answer matches expected answer exactly!")
        else:
            print("\n⚠️ MISMATCH: Retrieved answer differs from expected answer")
            print(f"Expected length: {len(expected_answer)}, Retrieved length: {len(dataset_answer)}")
else:
    print("\n❌ No dataset answer found - would fall back to LLM")

print(f"\n\nContext Retrieved (Top 3 matches):")
print("-" * 80)
print(context[:500] + "..." if len(context) > 500 else context)
