#!/usr/bin/env python3
"""
Comprehensive GPT-4o-mini Fine-Tuned Model Evaluation
Tests base model vs fine-tuned model on compliance policy questions
"""

import os
import sys
import time
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_header(text):
    """Print colored header"""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(70)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

def print_success(text):
    """Print success message"""
    print(f"{GREEN}✓ {text}{RESET}")

def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_error(text):
    """Print error message"""
    print(f"{RED}✗ {text}{RESET}")

def print_info(text):
    """Print info message"""
    print(f"{CYAN}{text}{RESET}")

# Test questions aligned with actual training data
# Based on compliance_finetuning_gpt4o_mini.jsonl (164 examples)
TEST_QUESTIONS = [
    # PRICING AUTHORITY (5 questions - 15% threshold from training)
    {
        "category": "Pricing Authority",
        "question": "If I want to reduce a product price by 12%, do I need senior approval?",
        "expected": "No, price changes up to 15% can be done autonomously",
        "keywords": ["15", "autonomous", "no", "approval"],
        "weight": 1.0
    },
    {
        "category": "Pricing Authority",
        "question": "Can I independently approve an 18% price increase?",
        "expected": "No, price changes above 15% require senior director approval",
        "keywords": ["15", "senior", "director", "approval"],
        "weight": 1.0
    },
    {
        "category": "Pricing Authority",
        "question": "What's the maximum percentage I can change pricing without escalation?",
        "expected": "15%",
        "keywords": ["15", "percent", "%"],
        "weight": 1.0
    },
    {
        "category": "Pricing Authority",
        "question": "Who approves price changes beyond 15%?",
        "expected": "Senior Director of Merchandise",
        "keywords": ["senior", "director", "merchandise"],
        "weight": 0.9
    },
    {
        "category": "Pricing Authority",
        "question": "I need to drop a price by 20% for a clearance sale. What do I do?",
        "expected": "Get approval from Senior Director - changes over 15% require escalation",
        "keywords": ["senior", "director", "approval", "15"],
        "weight": 0.9
    },
    
    # MARGIN REQUIREMENTS (5 questions - 25% minimum from training)
    {
        "category": "Margin Requirements",
        "question": "What is the minimum margin percentage for retail products?",
        "expected": "25%",
        "keywords": ["25", "percent", "%", "margin"],
        "weight": 1.0
    },
    {
        "category": "Margin Requirements",
        "question": "Can I accept a product with 23% margin?",
        "expected": "No, minimum 25% required for retail",
        "keywords": ["25", "minimum", "no"],
        "weight": 1.0
    },
    {
        "category": "Margin Requirements",
        "question": "What's the margin requirement for marketplace sellers?",
        "expected": "15% minimum",
        "keywords": ["15", "marketplace", "percent"],
        "weight": 1.0
    },
    {
        "category": "Margin Requirements",
        "question": "I found a great deal but margin is only 20%. Can I proceed?",
        "expected": "No, retail requires 25% minimum margin",
        "keywords": ["25", "minimum", "retail", "no"],
        "weight": 0.9
    },
    {
        "category": "Margin Requirements",
        "question": "What's the difference in margin requirements between retail and marketplace?",
        "expected": "Retail requires 25%, marketplace requires 15%",
        "keywords": ["25", "15", "retail", "marketplace"],
        "weight": 0.8
    },
    
    # IN-STOCK MANAGEMENT (5 questions - 95% target, 72hr limit from training)
    {
        "category": "In-Stock Management",
        "question": "What's our in-stock rate target?",
        "expected": "95%",
        "keywords": ["95", "percent", "%"],
        "weight": 1.0
    },
    {
        "category": "In-Stock Management",
        "question": "How long can a product be out of stock before escalation?",
        "expected": "72 hours maximum",
        "keywords": ["72", "hours", "three days"],
        "weight": 1.0
    },
    {
        "category": "In-Stock Management",
        "question": "A product has been out of stock for 5 days. Is this acceptable?",
        "expected": "No, maximum allowed is 72 hours",
        "keywords": ["72", "hours", "no", "maximum"],
        "weight": 1.0
    },
    {
        "category": "In-Stock Management",
        "question": "What happens if we miss the 95% in-stock target?",
        "expected": "Requires review and corrective action plan",
        "keywords": ["review", "action", "corrective"],
        "weight": 0.7
    },
    {
        "category": "In-Stock Management",
        "question": "Can I let a slow-moving item stay out of stock for a week?",
        "expected": "No, 72-hour maximum applies to all products",
        "keywords": ["72", "maximum", "no", "all"],
        "weight": 0.9
    },
    
    # VENDOR MANAGEMENT (5 questions - $25 gift limit, $250K authority from training)
    {
        "category": "Vendor Management",
        "question": "A vendor offered me a $50 gift. Can I accept it?",
        "expected": "No, maximum allowed is $25",
        "keywords": ["25", "maximum", "no"],
        "weight": 1.0
    },
    {
        "category": "Vendor Management",
        "question": "What's the maximum gift value I can accept from vendors?",
        "expected": "$25",
        "keywords": ["25", "dollar"],
        "weight": 1.0
    },
    {
        "category": "Vendor Management",
        "question": "Up to what dollar amount can I negotiate contracts independently?",
        "expected": "$250,000",
        "keywords": ["250", "thousand"],
        "weight": 1.0
    },
    {
        "category": "Vendor Management",
        "question": "I need to sign a $300K vendor contract. What's the process?",
        "expected": "Requires VP approval - over $250K threshold",
        "keywords": ["VP", "approval", "250"],
        "weight": 0.9
    },
    {
        "category": "Vendor Management",
        "question": "Can I accept a $20 coffee mug from a supplier?",
        "expected": "Yes, under $25 limit",
        "keywords": ["25", "yes", "under"],
        "weight": 0.8
    },
    
    # PRODUCT SELECTION (5 questions - 3.0 stars, 15% return rate from training)
    {
        "category": "Product Selection",
        "question": "What's the minimum acceptable product rating?",
        "expected": "3.0 stars",
        "keywords": ["3", "star", "rating"],
        "weight": 1.0
    },
    {
        "category": "Product Selection",
        "question": "Can I add a product with 2.8 star rating to our catalog?",
        "expected": "No, minimum 3.0 stars required",
        "keywords": ["3", "no", "minimum"],
        "weight": 1.0
    },
    {
        "category": "Product Selection",
        "question": "What's the maximum acceptable return rate for new products?",
        "expected": "15%",
        "keywords": ["15", "percent", "return"],
        "weight": 1.0
    },
    {
        "category": "Product Selection",
        "question": "A product has 18% return rate. Should we continue carrying it?",
        "expected": "No, exceeds 15% maximum threshold",
        "keywords": ["15", "no", "exceeds", "maximum"],
        "weight": 0.9
    },
    {
        "category": "Product Selection",
        "question": "What criteria determine if we can add a new product?",
        "expected": "Minimum 3.0 stars, maximum 15% return rate, 25% margin",
        "keywords": ["3", "15", "25", "star", "return", "margin"],
        "weight": 0.8
    },
    
    # KPI TARGETS (5 questions - specific targets from training)
    {
        "category": "KPI Targets",
        "question": "What's our annual revenue growth target?",
        "expected": "15% year-over-year",
        "keywords": ["15", "percent", "yoy", "year"],
        "weight": 1.0
    },
    {
        "category": "KPI Targets",
        "question": "What perfect order rate should we achieve?",
        "expected": "95%",
        "keywords": ["95", "percent", "%"],
        "weight": 1.0
    },
    {
        "category": "KPI Targets",
        "question": "What's the target return rate we should not exceed?",
        "expected": "8%",
        "keywords": ["8", "percent", "%", "return"],
        "weight": 1.0
    },
    {
        "category": "KPI Targets",
        "question": "What customer satisfaction rating do we aim for?",
        "expected": "4.0 or higher",
        "keywords": ["4", "rating", "satisfaction"],
        "weight": 0.9
    },
    {
        "category": "KPI Targets",
        "question": "Our return rate hit 10% last quarter. Is this acceptable?",
        "expected": "No, target is maximum 8%",
        "keywords": ["8", "no", "maximum", "target"],
        "weight": 0.9
    },
    
    # COMPLIANCE (5 questions - specific timelines from training)
    {
        "category": "Compliance",
        "question": "How quickly must we remove products with safety concerns?",
        "expected": "4 hours",
        "keywords": ["4", "hour", "immediate"],
        "weight": 1.0
    },
    {
        "category": "Compliance",
        "question": "What's the response time for intellectual property violations?",
        "expected": "24 hours",
        "keywords": ["24", "hour"],
        "weight": 1.0
    },
    {
        "category": "Compliance",
        "question": "We found a counterfeit product listing. How long do we have to act?",
        "expected": "24 hours for IP violations",
        "keywords": ["24", "hour", "IP"],
        "weight": 0.9
    },
    {
        "category": "Compliance",
        "question": "How long to respond to inaccurate product content reports?",
        "expected": "72 hours",
        "keywords": ["72", "hour", "three days"],
        "weight": 1.0
    },
    {
        "category": "Compliance",
        "question": "A customer reported incorrect product specs. What's my deadline?",
        "expected": "72 hours to investigate and correct",
        "keywords": ["72", "hour", "correct"],
        "weight": 0.8
    },
    
    # REPORTING (5 questions - specific deadlines from training)
    {
        "category": "Reporting",
        "question": "When are weekly sales reports due?",
        "expected": "Monday 10 AM PST",
        "keywords": ["monday", "10", "am", "pst"],
        "weight": 1.0
    },
    {
        "category": "Reporting",
        "question": "What day of the month is the revenue forecast due?",
        "expected": "15th of each month",
        "keywords": ["15", "month"],
        "weight": 1.0
    },
    {
        "category": "Reporting",
        "question": "I'm late submitting my weekly report on Tuesday. Is this on time?",
        "expected": "No, due Monday 10 AM PST",
        "keywords": ["monday", "10", "no"],
        "weight": 0.9
    },
    {
        "category": "Reporting",
        "question": "When should I submit next month's forecast?",
        "expected": "By the 15th of current month",
        "keywords": ["15", "month"],
        "weight": 0.8
    },
    {
        "category": "Reporting",
        "question": "What's the deadline for submitting quarterly business reviews?",
        "expected": "Within 5 business days of quarter end",
        "keywords": ["5", "business", "day", "quarter"],
        "weight": 0.7
    },
    
    # PROMOTIONS (5 questions - 6 weeks advance, 20% discount from training)
    {
        "category": "Promotions",
        "question": "How far in advance must promotional campaigns be planned?",
        "expected": "6 weeks minimum",
        "keywords": ["6", "week", "advance"],
        "weight": 1.0
    },
    {
        "category": "Promotions",
        "question": "What's the minimum discount for Lightning Deals?",
        "expected": "20%",
        "keywords": ["20", "percent", "%"],
        "weight": 1.0
    },
    {
        "category": "Promotions",
        "question": "Can I launch a promotion with 3 weeks notice?",
        "expected": "No, requires 6 weeks advance planning",
        "keywords": ["6", "week", "no"],
        "weight": 1.0
    },
    {
        "category": "Promotions",
        "question": "I want to run a Lightning Deal with 15% off. Is this allowed?",
        "expected": "No, Lightning Deals require minimum 20% discount",
        "keywords": ["20", "minimum", "no"],
        "weight": 0.9
    },
    {
        "category": "Promotions",
        "question": "What's required to launch a promotional campaign?",
        "expected": "6 weeks advance notice, approved discount structure",
        "keywords": ["6", "week", "advance", "approval"],
        "weight": 0.8
    },
    
    # TRAINING (3 questions - 85% pass rate, 30 days from training)
    {
        "category": "Training",
        "question": "What's the required pass rate for compliance training?",
        "expected": "85%",
        "keywords": ["85", "percent", "%", "pass"],
        "weight": 1.0
    },
    {
        "category": "Training",
        "question": "How long do new employees have to complete initial training?",
        "expected": "30 days from start date",
        "keywords": ["30", "day"],
        "weight": 1.0
    },
    {
        "category": "Training",
        "question": "How often is refresher training required?",
        "expected": "Annually",
        "keywords": ["annual", "yearly", "refresher", "training"],
        "weight": 0.7
    },
]


def test_model(client, model_name, question, is_finetuned=False):
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
    
    # Check for errors
    if answer.startswith("ERROR"):
        return 0.0
        
    # Count keyword matches
    keyword_matches = sum(1 for kw in keywords if kw.lower() in answer_lower)
    keyword_score = keyword_matches / len(keywords) if keywords else 0
    
    # If most keywords present, give high score
    if keyword_score >= 0.5:
        return weight
    elif keyword_score >= 0.3:
        return weight * 0.5
    else:
        return 0.0


def run_comparison_test(verbose=False, quick_mode=False):
    """Run comprehensive comparison test"""
    print_header("COMPREHENSIVE GPT-4o-mini MODEL EVALUATION")
    print(f"{CYAN}Testing {len(TEST_QUESTIONS)} policy compliance questions{RESET}")
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY not found in environment!")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Model IDs
    base_model = "gpt-4o-mini"
    finetuned_model = os.getenv("FINETUNED_MODEL_ID")
    
    if not finetuned_model:
        print_error("FINETUNED_MODEL_ID not found in environment!")
        sys.exit(1)
    
    print(f"\n{BOLD}Models:{RESET}")
    print(f"  Base: {CYAN}{base_model}{RESET}")
    print(f"  Fine-tuned: {CYAN}{finetuned_model}{RESET}\n")
    
    # Results storage
    base_results = []
    finetuned_results = []
    base_scores = []
    finetuned_scores = []
    
    # Category tracking
    categories = {}
    
    # Test each question
    total = len(TEST_QUESTIONS)
    print(f"{BOLD}Running tests...{RESET}")
    
    for idx, test_case in enumerate(TEST_QUESTIONS, 1):
        question = test_case["question"]
        category = test_case["category"]
        expected = test_case["expected"]
        keywords = test_case["keywords"]
        weight = test_case["weight"]
        
        # Initialize category tracking
        if category not in categories:
            categories[category] = {"base": [], "finetuned": []}
        
        # Progress indicator
        progress = f"[{idx}/{total}]"
        print(f"\r{CYAN}{progress}{RESET} Testing: {question[:60]}...", end="", flush=True)
        
        # Test both models
        base_result = test_model(client, base_model, question, False)
        finetuned_result = test_model(client, finetuned_model, question, True)
        
        # Evaluate answers
        base_score = evaluate_answer(base_result["answer"], expected, keywords, weight)
        finetuned_score = evaluate_answer(finetuned_result["answer"], expected, keywords, weight)
        
        # Store results
        base_results.append(base_result)
        finetuned_results.append(finetuned_result)
        base_scores.append(base_score)
        finetuned_scores.append(finetuned_score)
        
        # Category tracking
        categories[category]["base"].append(base_score)
        categories[category]["finetuned"].append(finetuned_score)
        
        # Verbose output
        if verbose:
            print(f"\n\n{BOLD}Question {idx}:{RESET} {question}")
            print(f"{BOLD}Expected:{RESET} {expected}")
            print(f"\n{BOLD}Base Model:{RESET}\n{base_result['answer']}")
            print(f"{BOLD}Score:{RESET} {base_score}/{weight}")
            print(f"\n{BOLD}Fine-tuned Model:{RESET}\n{finetuned_result['answer']}")
            print(f"{BOLD}Score:{RESET} {finetuned_score}/{weight}\n")
            print("-" * 70)
    
    print(f"\r{' ' * 100}\r")  # Clear progress line
    
    # Calculate metrics
    total_possible = sum(t["weight"] for t in TEST_QUESTIONS)
    base_total_score = sum(base_scores)
    finetuned_total_score = sum(finetuned_scores)
    
    base_accuracy = (base_total_score / total_possible) * 100
    finetuned_accuracy = (finetuned_total_score / total_possible) * 100
    
    base_avg_tokens = sum(r["tokens"] for r in base_results) / len(base_results)
    finetuned_avg_tokens = sum(r["tokens"] for r in finetuned_results) / len(finetuned_results)
    
    base_avg_time = sum(r["duration"] for r in base_results) / len(base_results)
    finetuned_avg_time = sum(r["duration"] for r in finetuned_results) / len(finetuned_results)
    
    # Print results
    print_header("EVALUATION RESULTS")
    
    print(f"{BOLD}Overall Performance:{RESET}")
    print(f"  Base Model:      {base_accuracy:.1f}% accuracy ({base_total_score:.1f}/{total_possible:.1f} points)")
    print(f"  Fine-tuned Model: {finetuned_accuracy:.1f}% accuracy ({finetuned_total_score:.1f}/{total_possible:.1f} points)")
    
    accuracy_diff = finetuned_accuracy - base_accuracy
    if accuracy_diff > 0:
        print_success(f"  Improvement: +{accuracy_diff:.1f}%")
    elif accuracy_diff < 0:
        print_warning(f"  Difference: {accuracy_diff:.1f}%")
    else:
        print_info("  Performance: Equal")
    
    print(f"\n{BOLD}Token Usage:{RESET}")
    print(f"  Base Model:      {base_avg_tokens:.1f} avg tokens/question")
    print(f"  Fine-tuned Model: {finetuned_avg_tokens:.1f} avg tokens/question")
    token_diff = ((finetuned_avg_tokens - base_avg_tokens) / base_avg_tokens) * 100
    if token_diff < 0:
        print_success(f"  Efficiency: {token_diff:.1f}% fewer tokens")
    else:
        print_warning(f"  Usage: +{token_diff:.1f}% more tokens")
    
    print(f"\n{BOLD}Response Time:{RESET}")
    print(f"  Base Model:      {base_avg_time:.2f}s avg")
    print(f"  Fine-tuned Model: {finetuned_avg_time:.2f}s avg")
    time_diff = ((finetuned_avg_time - base_avg_time) / base_avg_time) * 100
    if time_diff < 0:
        print_success(f"  Speed: {abs(time_diff):.1f}% faster")
    else:
        print_warning(f"  Speed: +{time_diff:.1f}% slower")
    
    # Category breakdown
    print(f"\n{BOLD}Category Breakdown:{RESET}")
    for category, scores in sorted(categories.items()):
        base_cat_score = (sum(scores["base"]) / len(scores["base"])) * 100
        finetuned_cat_score = (sum(scores["finetuned"]) / len(scores["finetuned"])) * 100
        diff = finetuned_cat_score - base_cat_score
        
        print(f"\n  {BOLD}{category}:{RESET}")
        print(f"    Base: {base_cat_score:.1f}%  |  Fine-tuned: {finetuned_cat_score:.1f}%  |  Diff: {diff:+.1f}%")
    
    # Final verdict
    print(f"\n{BOLD}{'='*70}{RESET}")
    if accuracy_diff > 5:
        print_success(f"🏆 Significant accuracy improvement with fine-tuning!")
    elif accuracy_diff > 0:
        print_success(f"✓ Modest improvement with fine-tuning")
    elif accuracy_diff < -5:
        print_error(f"⚠ Base model outperformed fine-tuned model")
    else:
        print_info(f"≈ Similar performance between models")
    
    # Save detailed results
    results_data = {
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
            for i, test_case in enumerate(TEST_QUESTIONS)
        ]
    }
    
    output_file = "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n{CYAN}Detailed results saved to: {output_file}{RESET}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GPT-4o-mini fine-tuned model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quick", action="store_true", help="Quick mode (fewer questions)")
    
    args = parser.parse_args()
    
    run_comparison_test(verbose=args.verbose, quick_mode=args.quick)
