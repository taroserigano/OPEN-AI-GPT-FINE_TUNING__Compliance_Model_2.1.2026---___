#!/usr/bin/env python3
"""
Round 2: Additional 25 Questions for Fine-Tuned Model Evaluation
Different scenarios and edge cases from training data
"""

import os
import sys
import time
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_header(text):
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(70)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_info(text):
    print(f"{CYAN}{text}{RESET}")

# Round 2: 25 NEW questions focusing on edge cases and combinations
TEST_QUESTIONS_ROUND2 = [
    # Pricing edge cases and combinations
    {
        "category": "Pricing Authority",
        "question": "What happens if I need to change price by exactly 15%?",
        "expected": "15% is within autonomous authority, no approval needed",
        "keywords": ["15", "autonomous", "within"],
        "weight": 1.0
    },
    {
        "category": "Pricing Authority",
        "question": "Can I reduce a product by 15.5%?",
        "expected": "No, exceeds 15% threshold, requires senior director approval",
        "keywords": ["15", "exceeds", "senior", "approval"],
        "weight": 1.0
    },
    
    # Margin combinations
    {
        "category": "Margin Requirements",
        "question": "What's the margin requirement difference between retail and marketplace products?",
        "expected": "Retail 25%, marketplace 15% - 10% difference",
        "keywords": ["25", "15", "retail", "marketplace"],
        "weight": 1.0
    },
    {
        "category": "Margin Requirements",
        "question": "If a marketplace product has 22% margin, can I list it?",
        "expected": "Yes, marketplace minimum is 15%",
        "keywords": ["15", "marketplace", "yes"],
        "weight": 1.0
    },
    {
        "category": "Margin Requirements",
        "question": "What margin is required for a product that's both retail and marketplace?",
        "expected": "25% retail margin must be met",
        "keywords": ["25", "retail", "higher"],
        "weight": 0.9
    },
    
    # In-stock timing scenarios
    {
        "category": "In-Stock Management",
        "question": "If a product goes out of stock Monday morning, when's the latest it can be restocked?",
        "expected": "Thursday morning - 72 hours maximum",
        "keywords": ["72", "three", "day"],
        "weight": 1.0
    },
    {
        "category": "In-Stock Management",
        "question": "We're at 94% in-stock rate. Is this acceptable?",
        "expected": "No, target is 95%",
        "keywords": ["95", "target", "below"],
        "weight": 1.0
    },
    {
        "category": "In-Stock Management",
        "question": "Can I have 96% in-stock rate?",
        "expected": "Yes, exceeds 95% target",
        "keywords": ["95", "yes", "exceeds"],
        "weight": 0.8
    },
    
    # Vendor gift combinations
    {
        "category": "Vendor Management",
        "question": "A vendor gave me $20 yesterday and $15 today. Can I keep both?",
        "expected": "No, total exceeds $25 limit",
        "keywords": ["25", "total", "no", "exceeds"],
        "weight": 1.0
    },
    {
        "category": "Vendor Management",
        "question": "What if a vendor offers a $25 gift card?",
        "expected": "Acceptable, exactly at $25 limit",
        "keywords": ["25", "acceptable", "limit"],
        "weight": 0.9
    },
    {
        "category": "Vendor Management",
        "question": "Can I sign a $250,000 contract without VP approval?",
        "expected": "Yes, up to $250K is within authority",
        "keywords": ["250", "yes", "within"],
        "weight": 1.0
    },
    {
        "category": "Vendor Management",
        "question": "What about a $250,001 contract?",
        "expected": "No, exceeds $250K threshold, requires VP approval",
        "keywords": ["250", "exceeds", "VP", "approval"],
        "weight": 1.0
    },
    
    # Product quality thresholds
    {
        "category": "Product Selection",
        "question": "A product has exactly 3.0 stars. Can I add it?",
        "expected": "Yes, meets 3.0 minimum",
        "keywords": ["3", "yes", "meets", "minimum"],
        "weight": 1.0
    },
    {
        "category": "Product Selection",
        "question": "Product has 2.95 stars. Is this acceptable?",
        "expected": "No, minimum 3.0 required",
        "keywords": ["3", "no", "minimum"],
        "weight": 1.0
    },
    {
        "category": "Product Selection",
        "question": "Return rate is exactly 15%. Can we continue selling?",
        "expected": "Yes, at maximum threshold but acceptable",
        "keywords": ["15", "yes", "maximum", "acceptable"],
        "weight": 0.9
    },
    
    # KPI combinations
    {
        "category": "KPI Targets",
        "question": "We hit 14.5% revenue growth. Did we meet target?",
        "expected": "No, target is 15%",
        "keywords": ["15", "no", "target"],
        "weight": 1.0
    },
    {
        "category": "KPI Targets",
        "question": "Perfect order rate is 94.5%. Is this good enough?",
        "expected": "No, target is 95%",
        "keywords": ["95", "no", "target"],
        "weight": 1.0
    },
    {
        "category": "KPI Targets",
        "question": "Return rate is 8.5%. Is this within target?",
        "expected": "No, exceeds 8% maximum",
        "keywords": ["8", "no", "exceeds"],
        "weight": 1.0
    },
    
    # Compliance timing edge cases
    {
        "category": "Compliance",
        "question": "Safety issue reported at 9am Monday. Latest removal time?",
        "expected": "1pm Monday - 4 hours",
        "keywords": ["4", "hour"],
        "weight": 1.0
    },
    {
        "category": "Compliance",
        "question": "IP violation found Friday 5pm. When must we respond?",
        "expected": "Saturday 5pm - 24 hours",
        "keywords": ["24", "hour"],
        "weight": 1.0
    },
    
    # Reporting deadlines
    {
        "category": "Reporting",
        "question": "Can I submit weekly report Monday 11am?",
        "expected": "No, due Monday 10am PST",
        "keywords": ["10", "monday", "no"],
        "weight": 1.0
    },
    {
        "category": "Reporting",
        "question": "Today is February 14th. When's the forecast due?",
        "expected": "Tomorrow, February 15th",
        "keywords": ["15", "tomorrow"],
        "weight": 0.8
    },
    
    # Promotion planning
    {
        "category": "Promotions",
        "question": "Can I plan a promotion 5 weeks in advance?",
        "expected": "No, minimum 6 weeks required",
        "keywords": ["6", "week", "no", "minimum"],
        "weight": 1.0
    },
    {
        "category": "Promotions",
        "question": "Lightning Deal with exactly 20% discount - is this allowed?",
        "expected": "Yes, meets 20% minimum",
        "keywords": ["20", "yes", "minimum"],
        "weight": 1.0
    },
    
    # Training edge cases
    {
        "category": "Training",
        "question": "New hire completes training in 35 days. Is this acceptable?",
        "expected": "No, must complete within 30 days",
        "keywords": ["30", "no", "within"],
        "weight": 1.0
    },
]


def test_model(client, model_name, question):
    """Test a specific model with a question"""
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about company policies and compliance requirements. Be concise and specific."},
                {"role": "user", "content": question}
            ],
            temperature=0,
            max_tokens=300
        )
        
        duration = time.time() - start_time
        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        return {
            "answer": answer,
            "duration": duration,
            "tokens": tokens_used,
            "success": True
        }
    except Exception as e:
        return {
            "answer": f"ERROR: {str(e)}",
            "duration": time.time() - start_time,
            "tokens": 0,
            "success": False
        }


def evaluate_answer(answer, expected, keywords, weight=1.0):
    """Evaluate answer quality using keyword matching"""
    answer_lower = answer.lower()
    
    if answer.startswith("ERROR"):
        return 0.0
        
    keyword_matches = sum(1 for kw in keywords if kw.lower() in answer_lower)
    keyword_score = keyword_matches / len(keywords) if keywords else 0
    
    if keyword_score >= 0.5:
        return weight
    elif keyword_score >= 0.3:
        return weight * 0.5
    else:
        return 0.0


def run_round2_test():
    """Run Round 2 evaluation with 25 new questions"""
    print_header("ROUND 2: 25 ADDITIONAL QUESTIONS")
    print(f"{CYAN}Testing edge cases and combinations{RESET}")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY not found!")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    base_model = "gpt-4o-mini"
    finetuned_model = os.getenv("FINETUNED_MODEL_ID")
    
    if not finetuned_model:
        print_error("FINETUNED_MODEL_ID not found!")
        sys.exit(1)
    
    print(f"\n{BOLD}Models:{RESET}")
    print(f"  Base: {CYAN}{base_model}{RESET}")
    print(f"  Fine-tuned: {CYAN}{finetuned_model}{RESET}\n")
    
    base_results = []
    finetuned_results = []
    base_scores = []
    finetuned_scores = []
    categories = {}
    
    total = len(TEST_QUESTIONS_ROUND2)
    print(f"{BOLD}Running {total} tests...{RESET}")
    
    for idx, test_case in enumerate(TEST_QUESTIONS_ROUND2, 1):
        question = test_case["question"]
        category = test_case["category"]
        expected = test_case["expected"]
        keywords = test_case["keywords"]
        weight = test_case["weight"]
        
        if category not in categories:
            categories[category] = {"base": [], "finetuned": []}
        
        progress = f"[{idx}/{total}]"
        print(f"\r{CYAN}{progress}{RESET} Testing: {question[:60]}...", end="", flush=True)
        
        base_result = test_model(client, base_model, question)
        finetuned_result = test_model(client, finetuned_model, question)
        
        base_score = evaluate_answer(base_result["answer"], expected, keywords, weight)
        finetuned_score = evaluate_answer(finetuned_result["answer"], expected, keywords, weight)
        
        base_results.append(base_result)
        finetuned_results.append(finetuned_result)
        base_scores.append(base_score)
        finetuned_scores.append(finetuned_score)
        
        categories[category]["base"].append(base_score)
        categories[category]["finetuned"].append(finetuned_score)
    
    print(f"\r{' ' * 100}\r")
    
    # Calculate metrics
    total_possible = sum(t["weight"] for t in TEST_QUESTIONS_ROUND2)
    base_total_score = sum(base_scores)
    finetuned_total_score = sum(finetuned_scores)
    
    base_accuracy = (base_total_score / total_possible) * 100
    finetuned_accuracy = (finetuned_total_score / total_possible) * 100
    
    base_avg_tokens = sum(r["tokens"] for r in base_results) / len(base_results)
    finetuned_avg_tokens = sum(r["tokens"] for r in finetuned_results) / len(finetuned_results)
    
    base_avg_time = sum(r["duration"] for r in base_results) / len(base_results)
    finetuned_avg_time = sum(r["duration"] for r in finetuned_results) / len(finetuned_results)
    
    # Print results
    print_header("ROUND 2 RESULTS")
    
    print(f"{BOLD}Overall Performance:{RESET}")
    print(f"  Base Model:       {base_accuracy:.1f}% accuracy ({base_total_score:.1f}/{total_possible:.1f} points)")
    print(f"  Fine-tuned Model: {finetuned_accuracy:.1f}% accuracy ({finetuned_total_score:.1f}/{total_possible:.1f} points)")
    
    accuracy_diff = finetuned_accuracy - base_accuracy
    if accuracy_diff > 0:
        print_success(f"  Improvement: +{accuracy_diff:.1f}%")
    elif accuracy_diff < 0:
        print_warning(f"  Difference: {accuracy_diff:.1f}%")
    else:
        print_info("  Performance: Equal")
    
    print(f"\n{BOLD}Token Usage:{RESET}")
    print(f"  Base Model:       {base_avg_tokens:.1f} avg tokens")
    print(f"  Fine-tuned Model: {finetuned_avg_tokens:.1f} avg tokens")
    token_diff = ((finetuned_avg_tokens - base_avg_tokens) / base_avg_tokens) * 100
    if token_diff < 0:
        print_success(f"  Efficiency: {token_diff:.1f}% fewer tokens")
    else:
        print_warning(f"  Usage: +{token_diff:.1f}% more tokens")
    
    print(f"\n{BOLD}Response Time:{RESET}")
    print(f"  Base Model:       {base_avg_time:.2f}s avg")
    print(f"  Fine-tuned Model: {finetuned_avg_time:.2f}s avg")
    
    # Category breakdown
    print(f"\n{BOLD}Category Breakdown:{RESET}")
    for category, scores in sorted(categories.items()):
        base_cat_score = (sum(scores["base"]) / len(scores["base"])) * 100
        finetuned_cat_score = (sum(scores["finetuned"]) / len(scores["finetuned"])) * 100
        diff = finetuned_cat_score - base_cat_score
        
        print(f"\n  {BOLD}{category}:{RESET}")
        print(f"    Base: {base_cat_score:.1f}%  |  Fine-tuned: {finetuned_cat_score:.1f}%  |  Diff: {diff:+.1f}%")
    
    print(f"\n{BOLD}{'='*70}{RESET}")
    if accuracy_diff > 5:
        print_success(f"🏆 Significant improvement in Round 2!")
    elif accuracy_diff > 0:
        print_success(f"✓ Modest improvement")
    else:
        print_warning(f"⚠ Mixed results on edge cases")
    
    # Save results
    results_data = {
        "round": 2,
        "timestamp": datetime.now().isoformat(),
        "models": {
            "base": base_model,
            "finetuned": finetuned_model
        },
        "overall": {
            "base_accuracy": base_accuracy,
            "finetuned_accuracy": finetuned_accuracy,
            "accuracy_difference": accuracy_diff,
            "base_avg_tokens": base_avg_tokens,
            "finetuned_avg_tokens": finetuned_avg_tokens,
            "base_avg_time": base_avg_time,
            "finetuned_avg_time": finetuned_avg_time
        },
        "categories": {
            cat: {
                "base_accuracy": (sum(scores["base"]) / len(scores["base"])) * 100,
                "finetuned_accuracy": (sum(scores["finetuned"]) / len(scores["finetuned"])) * 100
            }
            for cat, scores in categories.items()
        },
        "questions": [
            {
                "question": test_case["question"],
                "category": test_case["category"],
                "expected": test_case["expected"],
                "base_answer": base_results[i]["answer"],
                "finetuned_answer": finetuned_results[i]["answer"],
                "base_score": base_scores[i],
                "finetuned_score": finetuned_scores[i]
            }
            for i, test_case in enumerate(TEST_QUESTIONS_ROUND2)
        ]
    }
    
    output_file = "evaluation_results_round2.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n{CYAN}Round 2 results saved to: {output_file}{RESET}\n")


if __name__ == "__main__":
    run_round2_test()
