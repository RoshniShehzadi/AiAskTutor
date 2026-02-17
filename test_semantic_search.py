#!/usr/bin/env python3
"""
Test script to verify semantic search with paraphrased questions.
"""

import json
import sys
import os

# Add the current directory to path to import app functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from app.py
from app import retrieve_context, _norm, get_embedding_model, generate_dataset_embeddings, cosine_similarity
import numpy as np

def test_paraphrased_questions():
    """Test retrieval with paraphrased questions."""
    
    # Load dataset
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print("=" * 80)
    print("SEMANTIC SEARCH TEST - Paraphrased Questions")
    print("=" * 80)
    
    # Test cases: (paraphrased_question, expected_question_keywords)
    test_cases = [
        {
            "query": "Can you give a concrete example of a balanced schedule for a student?",
            "description": "Original question (exact match)",
            "expected_keywords": ["balanced", "schedule", "student", "example"]
        },
        {
            "query": "Show me a sample daily schedule for students",
            "description": "Paraphrase 1: Different wording",
            "expected_keywords": ["schedule", "student"]
        },
        {
            "query": "What does a well-balanced student timetable look like?",
            "description": "Paraphrase 2: Using 'timetable' instead of 'schedule'",
            "expected_keywords": ["balanced", "student", "timetable"]
        },
        {
            "query": "Give me an example of how students can balance their day",
            "description": "Paraphrase 3: Different structure",
            "expected_keywords": ["student", "balance", "day", "example"]
        },
        {
            "query": "I need a sample routine for a student's balanced lifestyle",
            "description": "Paraphrase 4: Using 'routine' and 'lifestyle'",
            "expected_keywords": ["student", "balanced", "routine", "lifestyle"]
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {test_case['description']}")
        print(f"{'='*80}")
        print(f"Query: {test_case['query']}\n")
        
        # Test with semantic search enabled
        context, dataset_answer = retrieve_context(
            test_case['query'], 
            dataset, 
            k=3, 
            use_semantic=True
        )
        
        # Check if answer was found
        if dataset_answer:
            print(f"✅ Answer Found (Length: {len(dataset_answer)} chars)")
            print(f"\nAnswer Preview:")
            print(dataset_answer[:300] + "..." if len(dataset_answer) > 300 else dataset_answer)
            
            # Check if it contains expected keywords
            answer_lower = dataset_answer.lower()
            found_keywords = [kw for kw in test_case['expected_keywords'] if kw.lower() in answer_lower]
            print(f"\nExpected keywords found: {found_keywords}/{test_case['expected_keywords']}")
            
            results.append({
                "test": i,
                "query": test_case['query'],
                "found": True,
                "answer_length": len(dataset_answer),
                "keywords_found": len(found_keywords),
                "total_keywords": len(test_case['expected_keywords'])
            })
        else:
            print("❌ No answer found - would fall back to LLM")
            results.append({
                "test": i,
                "query": test_case['query'],
                "found": False,
                "answer_length": 0,
                "keywords_found": 0,
                "total_keywords": len(test_case['expected_keywords'])
            })
        
        print(f"\nContext Retrieved (Top 3):")
        print("-" * 80)
        if context:
            lines = context.split('\n')
            for line in lines[:10]:  # Show first 10 lines
                print(line)
            if len(lines) > 10:
                print("...")
        else:
            print("No context retrieved")
    
    # Summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['found'])
    print(f"Total Tests: {total_tests}")
    print(f"Passed (Answer Found): {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"Failed (No Answer): {total_tests - passed_tests}/{total_tests}")
    
    avg_keywords = sum(r['keywords_found']/r['total_keywords'] if r['total_keywords'] > 0 else 0 for r in results) / total_tests
    print(f"Average Keyword Match: {avg_keywords*100:.1f}%")
    
    print("\nDetailed Results:")
    for r in results:
        status = "✅" if r['found'] else "❌"
        print(f"{status} Test {r['test']}: {r['query'][:60]}...")
        if r['found']:
            print(f"   Answer length: {r['answer_length']} chars, Keywords: {r['keywords_found']}/{r['total_keywords']}")

if __name__ == "__main__":
    try:
        test_paraphrased_questions()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
