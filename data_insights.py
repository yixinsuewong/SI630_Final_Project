#!/usr/bin/env python3
"""
Data Insights for RAG Project
------------------------------
This script analyzes the data collected and annotated for the RAG project,
providing an overview of documents, questions, answers, and other relevant metrics.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import re

# Define project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
DEV_DIR = os.path.join(BASE_DIR, "dev")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Create output directory for figures
FIGURES_DIR = os.path.join(BASE_DIR, "data_insights_figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_metadata() -> List[Dict]:
    """Load and return the metadata file."""
    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Metadata file not found at {metadata_path}")
        return []
    except json.JSONDecodeError:
        print(f"Warning: Metadata file at {metadata_path} is not valid JSON")
        return []


def count_documents() -> Tuple[int, int, int]:
    """Count the number of text and PDF documents in the data directory."""
    txt_count = len(list(Path(DATA_DIR).glob("**/*.txt")))
    pdf_count = len(list(Path(DATA_DIR).glob("**/*.pdf")))
    total_count = txt_count + pdf_count
    return txt_count, pdf_count, total_count


def analyze_qa_files(dir_path: str) -> Dict[str, Any]:
    """Analyze question and answer files in the specified directory."""
    questions_path = os.path.join(dir_path, "questions.txt")
    answers_path = os.path.join(dir_path, "reference_answers.txt")
    
    results = {
        "question_count": 0,
        "answer_count": 0,
        "avg_question_length": 0,
        "avg_answer_length": 0,
        "question_length_dist": [],
        "answer_length_dist": [],
        "questions": [],
        "answers": []
    }
    
    # Read questions
    if os.path.exists(questions_path):
        with open(questions_path, "r", encoding="utf-8") as f:
            questions = [q.strip() for q in f if q.strip()]
            results["questions"] = questions
            results["question_count"] = len(questions)
            
            if questions:
                q_lengths = [len(q.split()) for q in questions]
                results["avg_question_length"] = sum(q_lengths) / len(q_lengths)
                results["question_length_dist"] = q_lengths
    
    # Read answers
    if os.path.exists(answers_path):
        with open(answers_path, "r", encoding="utf-8") as f:
            answers = [a.strip() for a in f if a.strip()]
            results["answers"] = answers
            results["answer_count"] = len(answers)
            
            if answers:
                a_lengths = [len(a.split()) for a in answers]
                results["avg_answer_length"] = sum(a_lengths) / len(a_lengths)
                results["answer_length_dist"] = a_lengths
    
    return results


def analyze_metadata(metadata: List[Dict]) -> Dict[str, Any]:
    """Analyze the metadata to extract categories, sources, etc."""
    if not metadata:
        return {
            "categories": Counter(),
            "sources": Counter(),
            "document_lengths": [],
            "has_title": 0,
            "has_category": 0,
            "has_source": 0
        }
    
    results = {
        "categories": Counter(),
        "sources": Counter(),
        "document_lengths": [],
        "has_title": 0,
        "has_category": 0,
        "has_source": 0
    }
    
    for item in metadata:
        # Check for key fields
        if "title" in item and item["title"]:
            results["has_title"] += 1
        
        if "category" in item and item["category"]:
            results["has_category"] += 1
            results["categories"][item["category"]] += 1
        
        if "source" in item and item["source"]:
            results["has_source"] += 1
            results["sources"][item["source"]] += 1
        
        # Estimate document length from filename
        if "filename" in item:
            file_path = os.path.join(DATA_DIR, item["filename"])
            if os.path.exists(file_path):
                if file_path.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        results["document_lengths"].append(len(content.split()))
    
    return results


def analyze_question_types(questions: List[str]) -> Dict[str, int]:
    """Analyze the types of questions being asked."""
    question_types = Counter()
    
    # Define patterns for different question types
    patterns = {
        "what": r"^what\b|^what's\b",
        "how": r"^how\b",
        "who": r"^who\b|^who's\b",
        "where": r"^where\b",
        "when": r"^when\b",
        "why": r"^why\b",
        "which": r"^which\b",
        "can/could": r"^can\b|^could\b",
        "is/are": r"^is\b|^are\b|^isn't\b|^aren't\b",
        "do/does": r"^do\b|^does\b|^don't\b|^doesn't\b",
        "other": r".*"
    }
    
    for question in questions:
        q_lower = question.lower().strip()
        
        # Check for question mark
        has_question_mark = "?" in q_lower
        
        # Match question type
        matched = False
        for q_type, pattern in patterns.items():
            if re.match(pattern, q_lower):
                if has_question_mark:
                    question_types[f"{q_type} (?)"] += 1
                else:
                    question_types[f"{q_type} (no ?)"] += 1
                matched = True
                break
        
        if not matched:
            question_types["other"] += 1
    
    return question_types


def plot_document_distribution(metadata_analysis: Dict) -> None:
    """Plot the distribution of document categories and sources."""
    if not metadata_analysis["categories"] and not metadata_analysis["sources"]:
        print("No category or source data available for plotting")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Plot categories
    if metadata_analysis["categories"]:
        plt.subplot(2, 1, 1)
        categories = metadata_analysis["categories"].most_common(10)
        names = [c[0] for c in categories]
        values = [c[1] for c in categories]
        
        plt.bar(names, values)
        plt.title("Top 10 Document Categories")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
    
    # Plot sources
    if metadata_analysis["sources"]:
        plt.subplot(2, 1, 2)
        sources = metadata_analysis["sources"].most_common(10)
        names = [s[0] for s in sources]
        values = [s[1] for s in sources]
        
        plt.bar(names, values)
        plt.title("Top 10 Document Sources")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
    
    plt.savefig(os.path.join(FIGURES_DIR, "document_distribution.png"))
    plt.close()


def plot_qa_distribution(train_qa: Dict, dev_qa: Dict, test_qa: Dict) -> None:
    """Plot the distribution of question and answer lengths across datasets."""
    plt.figure(figsize=(15, 10))
    
    # Plot question length distributions
    plt.subplot(2, 1, 1)
    if train_qa["question_length_dist"]:
        plt.hist(train_qa["question_length_dist"], alpha=0.5, label="Train", bins=20)
    if dev_qa["question_length_dist"]:
        plt.hist(dev_qa["question_length_dist"], alpha=0.5, label="Dev", bins=20)
    if test_qa["question_length_dist"]:
        plt.hist(test_qa["question_length_dist"], alpha=0.5, label="Test", bins=20)
    
    plt.title("Question Length Distribution (words)")
    plt.xlabel("Length (words)")
    plt.ylabel("Count")
    plt.legend()
    
    # Plot answer length distributions
    plt.subplot(2, 1, 2)
    if train_qa["answer_length_dist"]:
        plt.hist(train_qa["answer_length_dist"], alpha=0.5, label="Train", bins=20)
    if dev_qa["answer_length_dist"]:
        plt.hist(dev_qa["answer_length_dist"], alpha=0.5, label="Dev", bins=20)
    if test_qa["answer_length_dist"]:
        plt.hist(test_qa["answer_length_dist"], alpha=0.5, label="Test", bins=20)
    
    plt.title("Answer Length Distribution (words)")
    plt.xlabel("Length (words)")
    plt.ylabel("Count")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "qa_distribution.png"))
    plt.close()


def plot_question_types(train_qa: Dict, dev_qa: Dict, test_qa: Dict) -> None:
    """Plot the distribution of question types across datasets."""
    # Analyze question types
    train_types = analyze_question_types(train_qa["questions"])
    dev_types = analyze_question_types(dev_qa["questions"])
    test_types = analyze_question_types(test_qa["questions"])
    
    # Combine all question types
    all_types = set()
    for types_dict in [train_types, dev_types, test_types]:
        all_types.update(types_dict.keys())
    
    all_types = sorted(all_types)
    
    # Create dataframe for plotting
    data = {
        "Question Type": all_types,
        "Train": [train_types.get(t, 0) for t in all_types],
        "Dev": [dev_types.get(t, 0) for t in all_types],
        "Test": [test_types.get(t, 0) for t in all_types]
    }
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(12, 8))
    df.plot(x="Question Type", kind="bar", stacked=False)
    plt.title("Question Types Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "question_types.png"))
    plt.close()


def generate_insights_report(
    txt_count: int, 
    pdf_count: int, 
    total_count: int,
    metadata_analysis: Dict,
    train_qa: Dict,
    dev_qa: Dict,
    test_qa: Dict
) -> str:
    """Generate a textual report of the data insights."""
    report = []
    
    # Document statistics
    report.append("=" * 50)
    report.append("DATA INSIGHTS REPORT")
    report.append("=" * 50)
    report.append("\n1. DOCUMENT STATISTICS")
    report.append("-" * 50)
    report.append(f"Total Documents: {total_count}")
    report.append(f"Text Files: {txt_count}")
    report.append(f"PDF Files: {pdf_count}")
    
    # Metadata statistics
    report.append("\n2. METADATA STATISTICS")
    report.append("-" * 50)
    report.append(f"Documents with Title: {metadata_analysis['has_title']}")
    report.append(f"Documents with Category: {metadata_analysis['has_category']}")
    report.append(f"Documents with Source: {metadata_analysis['has_source']}")
    
    # Category distribution
    if metadata_analysis['categories']:
        report.append("\nTop 5 Categories:")
        for category, count in metadata_analysis['categories'].most_common(5):
            report.append(f"  - {category}: {count} documents")
    
    # Document length statistics
    if metadata_analysis['document_lengths']:
        doc_lengths = metadata_analysis['document_lengths']
        report.append("\nDocument Length Statistics (words):")
        report.append(f"  - Average: {sum(doc_lengths) / len(doc_lengths):.2f}")
        report.append(f"  - Minimum: {min(doc_lengths)}")
        report.append(f"  - Maximum: {max(doc_lengths)}")
        report.append(f"  - Median: {sorted(doc_lengths)[len(doc_lengths)//2]}")
    
    # QA statistics
    report.append("\n3. QUESTION-ANSWER STATISTICS")
    report.append("-" * 50)
    report.append(f"Training Set: {train_qa['question_count']} QA pairs")
    report.append(f"Development Set: {dev_qa['question_count']} QA pairs")
    report.append(f"Test Set: {test_qa['question_count']} QA pairs")
    report.append(f"Total QA Pairs: {train_qa['question_count'] + dev_qa['question_count'] + test_qa['question_count']}")
    
    # Question length statistics
    report.append("\nAverage Question Length (words):")
    if train_qa['question_count']:
        report.append(f"  - Training: {train_qa['avg_question_length']:.2f}")
    if dev_qa['question_count']:
        report.append(f"  - Development: {dev_qa['avg_question_length']:.2f}")
    if test_qa['question_count']:
        report.append(f"  - Test: {test_qa['avg_question_length']:.2f}")
    
    # Answer length statistics
    report.append("\nAverage Answer Length (words):")
    if train_qa['answer_count']:
        report.append(f"  - Training: {train_qa['avg_answer_length']:.2f}")
    if dev_qa['answer_count']:
        report.append(f"  - Development: {dev_qa['avg_answer_length']:.2f}")
    if test_qa['answer_count']:
        report.append(f"  - Test: {test_qa['avg_answer_length']:.2f}")
    
    # Question type analysis
    train_types = analyze_question_types(train_qa["questions"])
    dev_types = analyze_question_types(dev_qa["questions"])
    test_types = analyze_question_types(test_qa["questions"])
    
    # Combine all sets
    all_questions = train_qa["questions"] + dev_qa["questions"] + test_qa["questions"]
    all_types = analyze_question_types(all_questions)
    
    report.append("\nQuestion Type Distribution (All Sets):")
    for q_type, count in all_types.most_common():
        report.append(f"  - {q_type}: {count} questions ({count/len(all_questions)*100:.1f}%)")
    
    # Summary
    report.append("\n4. SUMMARY")
    report.append("-" * 50)
    report.append(f"The dataset consists of {total_count} documents ({txt_count} text files, {pdf_count} PDFs)")
    report.append(f"with a total of {train_qa['question_count'] + dev_qa['question_count'] + test_qa['question_count']} question-answer pairs.")
    
    if metadata_analysis['categories']:
        main_category = metadata_analysis['categories'].most_common(1)[0][0]
        report.append(f"The main document category is '{main_category}'.")
    
    qa_splits = [
        f"{train_qa['question_count']} in training",
        f"{dev_qa['question_count']} in development",
        f"{test_qa['question_count']} in test"
    ]
    report.append(f"QA pairs are split as: {', '.join(qa_splits)}.")
    
    # Most common question type
    if all_types:
        most_common_type = all_types.most_common(1)[0][0]
        report.append(f"The most common question type is '{most_common_type}'.")
    
    # Figures info
    report.append("\nVisualizations have been saved to the 'data_insights_figures' directory.")
    
    return "\n".join(report)


def main():
    """Main function to run the data analysis."""
    print("Analyzing RAG project data...")
    
    # Load metadata
    metadata = load_metadata()
    print(f"Loaded metadata for {len(metadata)} documents")
    
    # Count documents
    txt_count, pdf_count, total_count = count_documents()
    print(f"Found {total_count} documents: {txt_count} text files, {pdf_count} PDFs")
    
    # Analyze metadata
    metadata_analysis = analyze_metadata(metadata)
    
    # Analyze QA pairs
    train_qa = analyze_qa_files(TRAIN_DIR)
    dev_qa = analyze_qa_files(DEV_DIR)
    test_qa = analyze_qa_files(TEST_DIR)
    
    print(f"Analyzed QA pairs: {train_qa['question_count']} training, {dev_qa['question_count']} development, {test_qa['question_count']} test")
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_document_distribution(metadata_analysis)
    plot_qa_distribution(train_qa, dev_qa, test_qa)
    plot_question_types(train_qa, dev_qa, test_qa)
    
    # Generate report
    report = generate_insights_report(
        txt_count, pdf_count, total_count,
        metadata_analysis, train_qa, dev_qa, test_qa
    )
    
    # Save report to file
    report_path = os.path.join(BASE_DIR, "data_insights_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Data insights report saved to {report_path}")
    print(f"Visualizations saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main() 