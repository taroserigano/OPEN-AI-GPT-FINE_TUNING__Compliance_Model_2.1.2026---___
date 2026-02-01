#!/usr/bin/env python3
"""
Round 3: 50 Questions - Comprehensive Fine-Tuned Model Evaluation
Mix of scenarios, combinations, and realistic use cases
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

# Round 3: 50 comprehensive questions
TEST_QUESTIONS_ROUND3 = [
    # Pricing scenarios (10 questions)
    {
        "category": "Pricing Authority",
        "question": "I need to increase price by 8%. Do I need approval?",
        "expected": "No, within 15% autonomous authority",
        "keywords": ["15", "no", "autonomous"],
        "weight": 1.0
    },
    {
        "category": "Pricing Authority",
        "question": "What's the process for a 22% price reduction?",
        "expected": "Requires Senior Director approval - exceeds 15% limit",
        "keywords": ["senior", "director", "15", "approval"],
        "weight": 1.0
    },
    {
        "category": "Pricing Authority",
        "question": "Can I approve a 14.9% price change?",
        "expected": "Yes, under 15% threshold",
        "keywords": ["15", "yes", "under"],
        "weight": 0.9
    },
    {
        "category": "Pricing Authority",
        "question": "Who has authority for price changes between 10-15%?",
        "expected": "You do - autonomous up to 15%",
        "keywords": ["15", "autonomous", "you"],
        "weight": 0.8
    },
    {
        "category": "Pricing Authority",
        "question": "I want to drop price from $100 to $84. Need approval?",
        "expected": "Yes, 16% reduction requires senior director approval",
        "keywords": ["15", "16", "approval", "yes"],
        "weight": 1.0
    },
    {
        "category": "Pricing Authority",
        "question": "Can I raise price by 15.1%?",
        "expected": "No, exceeds 15% threshold",
        "keywords": ["15", "no", "exceeds"],
        "weight": 1.0
    },
    {
        "category": "Pricing Authority",
        "question": "What if I need to change price by 10% twice?",
        "expected": "Each change evaluated separately if cumulative exceeds 15%",
        "keywords": ["15", "separate", "cumulative"],
        "weight": 0.7
    },
    {
        "category": "Pricing Authority",
        "question": "Emergency price drop of 25% - what's required?",
        "expected": "Senior Director approval - far exceeds 15%",
        "keywords": ["senior", "director", "15", "approval"],
        "weight": 0.9
    },
    {
        "category": "Pricing Authority",
        "question": "Can I independently adjust prices during sales events?",
        "expected": "Only within 15% limit",
        "keywords": ["15", "within", "limit"],
        "weight": 0.8
    },
    {
        "category": "Pricing Authority",
        "question": "What documentation is needed for 20% price changes?",
        "expected": "Senior Director approval and business justification",
        "keywords": ["senior", "director", "approval", "justification"],
        "weight": 0.7
    },
    
    # Margin requirements (10 questions)
    {
        "category": "Margin Requirements",
        "question": "Is 24.5% margin acceptable for retail?",
        "expected": "No, minimum 25% required",
        "keywords": ["25", "no", "minimum"],
        "weight": 1.0
    },
    {
        "category": "Margin Requirements",
        "question": "Can marketplace seller have 14% margin?",
        "expected": "No, marketplace minimum is 15%",
        "keywords": ["15", "no", "minimum"],
        "weight": 1.0
    },
    {
        "category": "Margin Requirements",
        "question": "What's acceptable margin for retail: 26% or 23%?",
        "expected": "26% - meets 25% minimum requirement",
        "keywords": ["25", "26", "meets"],
        "weight": 1.0
    },
    {
        "category": "Margin Requirements",
        "question": "New product has 30% margin. Can I add it?",
        "expected": "Yes, exceeds 25% minimum for retail",
        "keywords": ["25", "yes", "exceeds"],
        "weight": 0.8
    },
    {
        "category": "Margin Requirements",
        "question": "Marketplace product with 18% margin - acceptable?",
        "expected": "Yes, exceeds 15% marketplace minimum",
        "keywords": ["15", "yes", "marketplace"],
        "weight": 1.0
    },
    {
        "category": "Margin Requirements",
        "question": "What if retail margin drops to 24% after discount?",
        "expected": "Below 25% minimum - not acceptable",
        "keywords": ["25", "below", "not"],
        "weight": 0.9
    },
    {
        "category": "Margin Requirements",
        "question": "Can I launch with 25% margin exactly?",
        "expected": "Yes, meets minimum requirement",
        "keywords": ["25", "yes", "meets"],
        "weight": 1.0
    },
    {
        "category": "Margin Requirements",
        "question": "Wholesale product needs what margin?",
        "expected": "25% retail standard applies",
        "keywords": ["25", "retail"],
        "weight": 0.7
    },
    {
        "category": "Margin Requirements",
        "question": "Is 15.5% enough for marketplace items?",
        "expected": "Yes, exceeds 15% minimum",
        "keywords": ["15", "yes", "exceeds"],
        "weight": 0.9
    },
    {
        "category": "Margin Requirements",
        "question": "Product has 22% margin but high volume. Acceptable?",
        "expected": "No, must meet 25% minimum regardless of volume",
        "keywords": ["25", "no", "minimum"],
        "weight": 1.0
    },
    
    # In-stock management (5 questions)
    {
        "category": "In-Stock Management",
        "question": "Product out of stock for 60 hours - issue?",
        "expected": "No issue yet, under 72-hour limit",
        "keywords": ["72", "no", "under"],
        "weight": 1.0
    },
    {
        "category": "In-Stock Management",
        "question": "Can product be unavailable for 80 hours?",
        "expected": "No, exceeds 72-hour maximum",
        "keywords": ["72", "no", "exceeds"],
        "weight": 1.0
    },
    {
        "category": "In-Stock Management",
        "question": "Our in-stock rate is 96%. Is this good?",
        "expected": "Yes, exceeds 95% target",
        "keywords": ["95", "yes", "exceeds"],
        "weight": 1.0
    },
    {
        "category": "In-Stock Management",
        "question": "What happens if we're at 93% in-stock?",
        "expected": "Below 95% target - requires corrective action",
        "keywords": ["95", "below", "target"],
        "weight": 0.9
    },
    {
        "category": "In-Stock Management",
        "question": "Maximum time for stockout before escalation?",
        "expected": "72 hours",
        "keywords": ["72", "hours", "maximum"],
        "weight": 1.0
    },
    
    # Vendor management (10 questions)
    {
        "category": "Vendor Management",
        "question": "Vendor offers $30 holiday gift. Can I accept?",
        "expected": "No, exceeds $25 maximum",
        "keywords": ["25", "no", "exceeds"],
        "weight": 1.0
    },
    {
        "category": "Vendor Management",
        "question": "Is a $15 vendor lunch acceptable?",
        "expected": "Yes, under $25 limit",
        "keywords": ["25", "yes", "under"],
        "weight": 0.9
    },
    {
        "category": "Vendor Management",
        "question": "Can I negotiate a $200K contract alone?",
        "expected": "Yes, within $250K authority",
        "keywords": ["250", "yes", "within"],
        "weight": 1.0
    },
    {
        "category": "Vendor Management",
        "question": "Contract value is $300K. What do I need?",
        "expected": "VP approval - exceeds $250K threshold",
        "keywords": ["250", "VP", "approval", "exceeds"],
        "weight": 1.0
    },
    {
        "category": "Vendor Management",
        "question": "Two vendors each give $15 gifts. Acceptable?",
        "expected": "Yes, each under $25 individual limit",
        "keywords": ["25", "yes", "individual"],
        "weight": 0.8
    },
    {
        "category": "Vendor Management",
        "question": "Can I accept $25.50 gift?",
        "expected": "No, exceeds $25 maximum",
        "keywords": ["25", "no", "exceeds"],
        "weight": 1.0
    },
    {
        "category": "Vendor Management",
        "question": "What's my contract signing authority limit?",
        "expected": "$250,000",
        "keywords": ["250", "thousand"],
        "weight": 1.0
    },
    {
        "category": "Vendor Management",
        "question": "Can I approve a $250,000 deal?",
        "expected": "Yes, exactly at authority limit",
        "keywords": ["250", "yes", "limit"],
        "weight": 1.0
    },
    {
        "category": "Vendor Management",
        "question": "Vendor gave me $20 last year. Can I accept $20 now?",
        "expected": "Yes, each under annual $25 limit",
        "keywords": ["25", "yes"],
        "weight": 0.7
    },
    {
        "category": "Vendor Management",
        "question": "Multi-year contract totaling $500K. Need approval?",
        "expected": "Depends on annual value vs total - if annual under $250K, yes",
        "keywords": ["250", "annual", "value"],
        "weight": 0.6
    },
    
    # Product selection (5 questions)
    {
        "category": "Product Selection",
        "question": "Product has 3.2 stars and 12% returns. Can I add it?",
        "expected": "Yes, meets 3.0 stars and under 15% return rate",
        "keywords": ["3", "15", "yes", "meets"],
        "weight": 1.0
    },
    {
        "category": "Product Selection",
        "question": "Item has 4.5 stars but 18% return rate. Acceptable?",
        "expected": "No, exceeds 15% return rate maximum",
        "keywords": ["15", "no", "exceeds"],
        "weight": 1.0
    },
    {
        "category": "Product Selection",
        "question": "Can I list product with 2.9 stars and 5% returns?",
        "expected": "No, below 3.0 star minimum",
        "keywords": ["3", "no", "below"],
        "weight": 1.0
    },
    {
        "category": "Product Selection",
        "question": "Product at exactly 15% return rate - OK to keep?",
        "expected": "Yes, at maximum threshold but acceptable",
        "keywords": ["15", "yes", "maximum"],
        "weight": 0.9
    },
    {
        "category": "Product Selection",
        "question": "New item: 3.1 stars, 14% returns, 26% margin. Add it?",
        "expected": "Yes, meets all criteria (3.0 stars, 15% returns, 25% margin)",
        "keywords": ["yes", "meets", "criteria"],
        "weight": 0.8
    },
    
    # KPI targets (5 questions)
    {
        "category": "KPI Targets",
        "question": "We achieved 16% revenue growth. Did we hit target?",
        "expected": "Yes, exceeds 15% target",
        "keywords": ["15", "yes", "exceeds"],
        "weight": 1.0
    },
    {
        "category": "KPI Targets",
        "question": "Perfect order rate is 95.5%. Is this acceptable?",
        "expected": "Yes, exceeds 95% target",
        "keywords": ["95", "yes", "exceeds"],
        "weight": 1.0
    },
    {
        "category": "KPI Targets",
        "question": "Return rate hit 7.5%. Within target?",
        "expected": "Yes, under 8% maximum",
        "keywords": ["8", "yes", "under"],
        "weight": 1.0
    },
    {
        "category": "KPI Targets",
        "question": "Customer rating is 3.9. Do we meet the goal?",
        "expected": "No, target is 4.0 or higher",
        "keywords": ["4", "no", "target"],
        "weight": 1.0
    },
    {
        "category": "KPI Targets",
        "question": "What revenue growth rate should we target annually?",
        "expected": "15% year-over-year",
        "keywords": ["15", "percent", "year"],
        "weight": 1.0
    },
    
    # Compliance timelines (5 questions)
    {
        "category": "Compliance",
        "question": "Safety recall announced. How fast must we act?",
        "expected": "4 hours to remove products",
        "keywords": ["4", "hour", "remove"],
        "weight": 1.0
    },
    {
        "category": "Compliance",
        "question": "Copyright complaint received. Response deadline?",
        "expected": "24 hours for IP violations",
        "keywords": ["24", "hour", "IP"],
        "weight": 1.0
    },
    {
        "category": "Compliance",
        "question": "Can we take 48 hours to fix product description error?",
        "expected": "Yes, 72-hour limit for content corrections",
        "keywords": ["72", "yes"],
        "weight": 0.9
    },
    {
        "category": "Compliance",
        "question": "Safety issue found at noon. Deadline to remove?",
        "expected": "4pm same day - 4 hours",
        "keywords": ["4", "hour"],
        "weight": 1.0
    },
    {
        "category": "Compliance",
        "question": "How long to respond to trademark infringement?",
        "expected": "24 hours",
        "keywords": ["24", "hour"],
        "weight": 1.0
    },
    
    # Reporting deadlines (3 questions)
    {
        "category": "Reporting",
        "question": "Can I submit weekly report Tuesday morning?",
        "expected": "No, due Monday 10 AM PST",
        "keywords": ["monday", "10", "no"],
        "weight": 1.0
    },
    {
        "category": "Reporting",
        "question": "Monthly forecast due date?",
        "expected": "15th of each month",
        "keywords": ["15", "month"],
        "weight": 1.0
    },
    {
        "category": "Reporting",
        "question": "What time on Monday is the weekly report due?",
        "expected": "10 AM PST",
        "keywords": ["10", "am", "pst"],
        "weight": 1.0
    },
    
    # Promotions (2 questions)
    {
        "category": "Promotions",
        "question": "Can I schedule promotion 7 weeks out?",
        "expected": "Yes, exceeds 6-week minimum",
        "keywords": ["6", "yes", "exceeds"],
        "weight": 1.0
    },
    {
        "category": "Promotions",
        "question": "Lightning Deal needs minimum what discount?",
        "expected": "20%",
        "keywords": ["20", "percent"],
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


def run_round3_test():
    """Run Round 3 evaluation with 50 questions"""
    print_header("ROUND 3: 50 COMPREHENSIVE QUESTIONS")
    print(f"{CYAN}Diverse scenarios and realistic use cases{RESET}")
    
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
    
    total = len(TEST_QUESTIONS_ROUND3)
    print(f"{BOLD}Running {total} tests...{RESET}")
    
    for idx, test_case in enumerate(TEST_QUESTIONS_ROUND3, 1):
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
    total_possible = sum(t["weight"] for t in TEST_QUESTIONS_ROUND3)
    base_total_score = sum(base_scores)
    finetuned_total_score = sum(finetuned_scores)
    
    base_accuracy = (base_total_score / total_possible) * 100
    finetuned_accuracy = (finetuned_total_score / total_possible) * 100
    
    base_avg_tokens = sum(r["tokens"] for r in base_results) / len(base_results)
    finetuned_avg_tokens = sum(r["tokens"] for r in finetuned_results) / len(finetuned_results)
    
    base_avg_time = sum(r["duration"] for r in base_results) / len(base_results)
    finetuned_avg_time = sum(r["duration"] for r in finetuned_results) / len(finetuned_results)
    
    # Print results
    print_header("ROUND 3 RESULTS")
    
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
    
    # Summary statistics
    print(f"\n{BOLD}Test Summary:{RESET}")
    print(f"  Total Questions: {total}")
    print(f"  Base Correct: {sum(1 for s in base_scores if s > 0.5)} questions")
    print(f"  Fine-tuned Correct: {sum(1 for s in finetuned_scores if s > 0.5)} questions")
    
    print(f"\n{BOLD}{'='*70}{RESET}")
    if accuracy_diff > 5:
        print_success(f"🏆 Significant improvement in Round 3!")
    elif accuracy_diff > 0:
        print_success(f"✓ Consistent positive results")
    else:
        print_warning(f"⚠ Variable performance")
    
    # Save results
    results_data = {
        "round": 3,
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
            for i, test_case in enumerate(TEST_QUESTIONS_ROUND3)
        ]
    }
    
    output_file = "evaluation_results_round3.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n{CYAN}Round 3 results saved to: {output_file}{RESET}\n")


if __name__ == "__main__":
    run_round3_test()
