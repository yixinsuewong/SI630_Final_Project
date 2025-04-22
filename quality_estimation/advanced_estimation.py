import re
import numpy as np
import pandas as pd
from collections import Counter
import json
import os
import matplotlib.pyplot as plt
from scipy import stats
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def normalize_answer(text):
    """Normalize answer text for comparison"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_tokens(text):
    """Split text into tokens"""
    text = normalize_answer(text)
    return text.split()

def calculate_f1(answer1, answer2):
    """Calculate token-level F1 score between two answers"""
    tokens1 = get_tokens(answer1)
    tokens2 = get_tokens(answer2)
    
    if len(tokens1) == 0 or len(tokens2) == 0:
        return int(tokens1 == tokens2)  # Return 1 if both empty, 0 otherwise
    
    # Count token occurrences
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)
    
    # Find tokens that appear in both answers
    common_tokens = counter1 & counter2
    num_same = sum(common_tokens.values())
    
    # If no overlap, F1 is 0
    if num_same == 0:
        return 0
    
    # Calculate precision and recall
    precision = num_same / len(tokens2)
    recall = num_same / len(tokens1)
    
    # Calculate F1
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def calculate_jaccard(answer1, answer2):
    """Calculate Jaccard similarity between two answers (token-level)"""
    tokens1 = set(get_tokens(answer1))
    tokens2 = set(get_tokens(answer2))
    
    if len(tokens1) == 0 or len(tokens2) == 0:
        return int(tokens1 == tokens2)  # Return 1 if both empty, 0 otherwise
    
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    return intersection / union

def calculate_exact_match(answer1, answer2):
    """Calculate exact match after normalization"""
    norm1 = normalize_answer(answer1)
    norm2 = normalize_answer(answer2)
    return int(norm1 == norm2)

def calculate_token_overlap(answer1, answer2):
    """Calculate percentage of tokens in answer1 that appear in answer2"""
    tokens1 = get_tokens(answer1)
    tokens2 = get_tokens(answer2)
    
    if len(tokens1) == 0:
        return int(len(tokens2) == 0)  # Return 1 if both empty, 0 otherwise
    
    # Count token occurrences
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)
    
    # Find tokens that appear in both answers
    common_tokens = counter1 & counter2
    num_same = sum(common_tokens.values())
    
    # Calculate overlap as percentage of tokens in answer1 that appear in answer2
    return num_same / len(tokens1)

def calculate_bleu(answer1, answer2):
    """Calculate BLEU score between two answers"""
    # Normalize and tokenize
    reference = [get_tokens(answer1)]  # BLEU expects a list of references
    hypothesis = get_tokens(answer2)
    
    # Check for empty sequences
    if len(hypothesis) == 0:
        return int(len(reference[0]) == 0)  # Return 1 if both empty, 0 otherwise
    
    if len(reference[0]) == 0:
        return 0  # If reference is empty but hypothesis isn't, return 0
    
    # Use smoothing to handle cases where there are no n-gram overlaps
    smoothing = SmoothingFunction().method1
    
    # Calculate BLEU with different n-gram weights
    # This uses equal weights for 1-gram, 2-gram, 3-gram, and 4-gram matches
    try:
        # BLEU-1 (only unigrams)
        bleu1 = sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        
        # BLEU-2 (unigrams and bigrams)
        if len(hypothesis) > 1 and len(reference[0]) > 1:
            bleu2 = sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        else:
            bleu2 = 0
            
        # BLEU-4 (standard with equal weights for 1,2,3,4-grams)
        bleu4 = sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        return {
            'bleu1': bleu1,
            'bleu2': bleu2,
            'bleu4': bleu4
        }
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        return {
            'bleu1': 0,
            'bleu2': 0,
            'bleu4': 0
        }

def calculate_rouge(answer1, answer2):
    """Calculate ROUGE scores between two answers"""
    if not answer1.strip() or not answer2.strip():
        # Handle empty answers
        is_both_empty = not answer1.strip() and not answer2.strip()
        score_value = 1.0 if is_both_empty else 0.0
        return {
            'rouge1_precision': score_value,
            'rouge1_recall': score_value,
            'rouge1_fmeasure': score_value,
            'rouge2_fmeasure': score_value,
            'rougeL_fmeasure': score_value
        }
    
    try:
        # Initialize the ROUGE scorer with specific metrics
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Calculate scores
        scores = scorer.score(answer1, answer2)
        
        # Extract the desired metrics
        result = {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_fmeasure': scores['rouge1'].fmeasure,
            'rouge2_fmeasure': scores['rouge2'].fmeasure,
            'rougeL_fmeasure': scores['rougeL'].fmeasure  # ROUGE-L measures longest common subsequence
        }
        
        return result
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        return {
            'rouge1_precision': 0,
            'rouge1_recall': 0,
            'rouge1_fmeasure': 0,
            'rouge2_fmeasure': 0,
            'rougeL_fmeasure': 0
        }

def process_dataset(dataset_name, questions_file, annotator1_file, annotator2_file, output_dir):
    """Process a single dataset (train or dev) and return results"""
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Load data with error handling
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        with open(annotator1_file, 'r', encoding='utf-8') as f:
            annotator1_answers = [line.strip() for line in f if line.strip()]
        
        with open(annotator2_file, 'r', encoding='utf-8') as f:
            annotator2_answers = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading files for {dataset_name}: {e}")
        return None
    
    # Check if files have content
    if not questions:
        print(f"Error: Questions file is empty or contains no valid lines: {questions_file}")
        return None
    
    if not annotator1_answers:
        print(f"Error: Annotator 1 file is empty or contains no valid lines: {annotator1_file}")
        return None
    
    if not annotator2_answers:
        print(f"Error: Annotator 2 file is empty or contains no valid lines: {annotator2_file}")
        return None
    
    # Ensure all files have the same number of non-empty lines
    min_length = min(len(questions), len(annotator1_answers), len(annotator2_answers))
    
    if min_length == 0:
        print(f"Error: No valid question-answer pairs found in the {dataset_name} files.")
        return None
    
    if len(questions) != min_length or len(annotator1_answers) != min_length or len(annotator2_answers) != min_length:
        print(f"Warning: Files for {dataset_name} have different numbers of lines. Using only the first {min_length} lines from each.")
    
    questions = questions[:min_length]
    annotator1_answers = annotator1_answers[:min_length]
    annotator2_answers = annotator2_answers[:min_length]
    
    print(f"Loaded {len(questions)} question-answer pairs for {dataset_name}")
    
    # Calculate scores
    f1_scores = []
    jaccard_scores = []
    exact_match_scores = []
    token_overlap_scores = []
    bleu1_scores = []
    bleu2_scores = []
    bleu4_scores = []
    rouge1_fmeasure_scores = []
    rouge2_fmeasure_scores = []
    rougeL_fmeasure_scores = []
    results = []
    
    for i, (q, a1, a2) in enumerate(zip(questions, annotator1_answers, annotator2_answers)):
        try:
            # Basic metrics
            f1 = calculate_f1(a1, a2)
            jaccard = calculate_jaccard(a1, a2)
            exact_match = calculate_exact_match(a1, a2)
            token_overlap = calculate_token_overlap(a1, a2)
            
            # BLEU scores
            bleu_scores = calculate_bleu(a1, a2)
            bleu1 = bleu_scores['bleu1']
            bleu2 = bleu_scores['bleu2']
            bleu4 = bleu_scores['bleu4']
            
            # ROUGE scores
            rouge_scores = calculate_rouge(a1, a2)
            rouge1_fmeasure = rouge_scores['rouge1_fmeasure']
            rouge2_fmeasure = rouge_scores['rouge2_fmeasure']
            rougeL_fmeasure = rouge_scores['rougeL_fmeasure']
            
            # Append to lists
            f1_scores.append(f1)
            jaccard_scores.append(jaccard)
            exact_match_scores.append(exact_match)
            token_overlap_scores.append(token_overlap)
            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu4_scores.append(bleu4)
            rouge1_fmeasure_scores.append(rouge1_fmeasure)
            rouge2_fmeasure_scores.append(rouge2_fmeasure)
            rougeL_fmeasure_scores.append(rougeL_fmeasure)
            
            # Store detailed results
            results.append({
                'question_id': i,
                'question': q,
                'annotator1_answer': a1,
                'annotator2_answer': a2,
                'f1_score': f1,
                'jaccard_score': jaccard,
                'exact_match': exact_match,
                'token_overlap': token_overlap,
                'bleu1': bleu1,
                'bleu2': bleu2,
                'bleu4': bleu4,
                'rouge1_fmeasure': rouge1_fmeasure,
                'rouge2_fmeasure': rouge2_fmeasure,
                'rougeL_fmeasure': rougeL_fmeasure
            })
        except Exception as e:
            print(f"Error calculating scores for question {i} in {dataset_name}: {e}")
    
    # Check if any scores were calculated
    if not f1_scores:
        print(f"Error: No scores could be calculated for {dataset_name}. Check that files contain valid data.")
        return None
    
    # Calculate aggregate statistics
    stats = {
        'dataset': dataset_name,
        'total_questions': len(questions),
        'total_scores_calculated': len(f1_scores),
        
        # F1 statistics
        'average_f1': float(np.mean(f1_scores)),
        'median_f1': float(np.median(f1_scores)),
        'min_f1': float(np.min(f1_scores)),
        'max_f1': float(np.max(f1_scores)),
        'std_dev_f1': float(np.std(f1_scores)),
        
        # Jaccard statistics
        'average_jaccard': float(np.mean(jaccard_scores)),
        'median_jaccard': float(np.median(jaccard_scores)),
        
        # Exact match statistics
        'exact_match_rate': float(np.mean(exact_match_scores)),
        
        # Token overlap statistics
        'average_token_overlap': float(np.mean(token_overlap_scores)),
        
        # BLEU statistics
        'average_bleu1': float(np.mean(bleu1_scores)),
        'average_bleu2': float(np.mean(bleu2_scores)),
        'average_bleu4': float(np.mean(bleu4_scores)),
        
        # ROUGE statistics
        'average_rouge1_fmeasure': float(np.mean(rouge1_fmeasure_scores)),
        'average_rouge2_fmeasure': float(np.mean(rouge2_fmeasure_scores)),
        'average_rougeL_fmeasure': float(np.mean(rougeL_fmeasure_scores)),
        
        # Detailed results
        'detailed_results': results
    }
    
    # Save results to files
    output_json = os.path.join(output_dir, f"iaa_results_{dataset_name}.json")
    output_csv = os.path.join(output_dir, f"iaa_results_{dataset_name}.csv")
    
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            # Save a version without the detailed results (might be too large)
            summary_stats = {k: v for k, v in stats.items() if k != 'detailed_results'}
            json.dump(summary_stats, f, indent=2)
        print(f"Summary results saved to {output_json}")
        
        # Save detailed results separately
        detailed_json = os.path.join(output_dir, f"iaa_detailed_{dataset_name}.json")
        with open(detailed_json, 'w', encoding='utf-8') as f:
            json.dump({'detailed_results': results}, f, indent=2)
        print(f"Detailed results saved to {detailed_json}")
    except Exception as e:
        print(f"Error saving JSON results for {dataset_name}: {e}")
    
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Detailed results saved to {output_csv}")
    except Exception as e:
        print(f"Error saving CSV results for {dataset_name}: {e}")
    
    # Generate histogram of F1 scores
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(f1_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of F1 Scores - {dataset_name} Dataset')
        plt.xlabel('F1 Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"f1_histogram_{dataset_name}.png"))
        print(f"F1 histogram saved to {os.path.join(output_dir, f'f1_histogram_{dataset_name}.png')}")
    except Exception as e:
        print(f"Error generating F1 histogram for {dataset_name}: {e}")
    
    # Generate multi-metric visualization
    try:
        plt.figure(figsize=(12, 8))
        
        # Create two subplots side by side
        plt.subplot(1, 2, 1)
        metrics1 = ['F1', 'BLEU-1', 'BLEU-2', 'BLEU-4']
        values1 = [
            stats['average_f1'], 
            stats['average_bleu1'], 
            stats['average_bleu2'], 
            stats['average_bleu4']
        ]
        plt.bar(metrics1, values1, color=['skyblue', 'lightgreen', 'salmon', 'lightpurple'])
        plt.title(f'BLEU Metrics - {dataset_name}')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        metrics2 = ['F1', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        values2 = [
            stats['average_f1'],
            stats['average_rouge1_fmeasure'],
            stats['average_rouge2_fmeasure'],
            stats['average_rougeL_fmeasure']
        ]
        plt.bar(metrics2, values2, color=['skyblue', 'orange', 'green', 'red'])
        plt.title(f'ROUGE Metrics - {dataset_name}')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"metrics_comparison_{dataset_name}.png"))
        print(f"Metrics comparison saved to {os.path.join(output_dir, f'metrics_comparison_{dataset_name}.png')}")
    except Exception as e:
        print(f"Error generating metrics comparison for {dataset_name}: {e}")
    
    return stats

def main():
    # Base directory
    base_dir = "/Users/yixin/Desktop/SI 630/Project_RAG/quality_estimation"

    # File paths
    train_questions_file = os.path.join(base_dir, "/Users/yixin/Desktop/SI 630/Project_RAG/quality_estimation/quality_question_train.txt")
    train_annotator1_file = os.path.join(base_dir, "/Users/yixin/Desktop/SI 630/Project_RAG/quality_estimation/quality_answer_train_1.txt")
    train_annotator2_file = os.path.join(base_dir, "/Users/yixin/Desktop/SI 630/Project_RAG/quality_estimation/quality_answer_train_2.txt")
    
    dev_questions_file = os.path.join(base_dir, "/Users/yixin/Desktop/SI 630/Project_RAG/quality_estimation/quality_question_dev.txt")
    dev_annotator1_file = os.path.join(base_dir, "/Users/yixin/Desktop/SI 630/Project_RAG/quality_estimation/quality_answer_dev_1.txt")
    dev_annotator2_file = os.path.join(base_dir, "/Users/yixin/Desktop/SI 630/Project_RAG/quality_estimation/quality_answer_dev_2.txt")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(base_dir, "iaa_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Install required packages if not already installed
    try:
        import nltk
        nltk.download('punkt', quiet=True)
    except ImportError:
        print("NLTK not installed. Please install required packages:")
        print("pip install nltk rouge-score matplotlib scipy pandas numpy")
        print("Then run 'import nltk; nltk.download(\"punkt\")'")
        return
    
    try:
        import rouge_score
    except ImportError:
        print("rouge-score not installed. Please install required package:")
        print("pip install rouge-score")
        return
    
    # Process train dataset
    train_stats = process_dataset("train", train_questions_file, train_annotator1_file, train_annotator2_file, output_dir)
    
    # Process dev dataset
    dev_stats = process_dataset("dev", dev_questions_file, dev_annotator1_file, dev_annotator2_file, output_dir)
    
    # Combine results and compare datasets
    if train_stats and dev_stats:
        print("\n=== COMPARING TRAIN AND DEV DATASETS ===")
        
        comparison = {
            'train_size': train_stats['total_questions'],
            'dev_size': dev_stats['total_questions'],
            
            # F1 comparisons
            'train_avg_f1': train_stats['average_f1'],
            'dev_avg_f1': dev_stats['average_f1'],
            'f1_difference': abs(train_stats['average_f1'] - dev_stats['average_f1']),
            
            # BLEU comparisons
            'train_avg_bleu1': train_stats['average_bleu1'],
            'dev_avg_bleu1': dev_stats['average_bleu1'],
            'bleu1_difference': abs(train_stats['average_bleu1'] - dev_stats['average_bleu1']),
            
            'train_avg_bleu4': train_stats['average_bleu4'],
            'dev_avg_bleu4': dev_stats['average_bleu4'],
            'bleu4_difference': abs(train_stats['average_bleu4'] - dev_stats['average_bleu4']),
            
            # ROUGE comparisons
            'train_avg_rouge1': train_stats['average_rouge1_fmeasure'],
            'dev_avg_rouge1': dev_stats['average_rouge1_fmeasure'],
            'rouge1_difference': abs(train_stats['average_rouge1_fmeasure'] - dev_stats['average_rouge1_fmeasure']),
            
            'train_avg_rougeL': train_stats['average_rougeL_fmeasure'],
            'dev_avg_rougeL': dev_stats['average_rougeL_fmeasure'],
            'rougeL_difference': abs(train_stats['average_rougeL_fmeasure'] - dev_stats['average_rougeL_fmeasure']),
            
            # Exact match comparison
            'train_exact_match_rate': train_stats['exact_match_rate'],
            'dev_exact_match_rate': dev_stats['exact_match_rate'],
            'exact_match_difference': abs(train_stats['exact_match_rate'] - dev_stats['exact_match_rate']),
        }
        
        # Perform statistical test to compare F1 distributions
        try:
            # Extract F1 scores from detailed results
            train_f1 = [item['f1_score'] for item in train_stats['detailed_results']]
            dev_f1 = [item['f1_score'] for item in dev_stats['detailed_results']]
            
            # Mann-Whitney U test (non-parametric test for comparing distributions)
            u_stat, p_value = stats.mannwhitneyu(train_f1, dev_f1, alternative='two-sided')
            comparison['mann_whitney_u_stat'] = float(u_stat)
            comparison['mann_whitney_p_value'] = float(p_value)
            comparison['distributions_significantly_different'] = p_value < 0.05
        except Exception as e:
            print(f"Error performing statistical test: {e}")
        
        # Save comparison to file
        comparison_json = os.path.join(output_dir, "train_vs_dev_comparison.json")
        try:
            with open(comparison_json, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2)
            print(f"Train vs. Dev comparison saved to {comparison_json}")
        except Exception as e:
            print(f"Error saving comparison results: {e}")
        
        # Print key findings
        print(f"\nTrain dataset average F1: {train_stats['average_f1']:.4f}")
        print(f"Dev dataset average F1: {dev_stats['average_f1']:.4f}")
        print(f"Absolute difference in average F1: {comparison['f1_difference']:.4f}")
        
        print(f"\nTrain dataset average BLEU-1: {train_stats['average_bleu1']:.4f}")
        print(f"Dev dataset average BLEU-1: {dev_stats['average_bleu1']:.4f}")
        
        print(f"\nTrain dataset average ROUGE-L: {train_stats['average_rougeL_fmeasure']:.4f}")
        print(f"Dev dataset average ROUGE-L: {dev_stats['average_rougeL_fmeasure']:.4f}")
        
        if 'distributions_significantly_different' in comparison:
            if comparison['distributions_significantly_different']:
                print(f"Statistical test shows the F1 score distributions are significantly different (p={comparison['mann_whitney_p_value']:.4f})")
            else:
                print(f"Statistical test shows no significant difference in F1 score distributions (p={comparison['mann_whitney_p_value']:.4f})")
        
        # Create comparison visualization
        try:
            plt.figure(figsize=(14, 10))
            
            # Set up metrics to compare
            metrics = [
                'F1', 'BLEU-1', 'BLEU-4', 
                'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 
                'Exact Match'
            ]
            
            train_values = [
                train_stats['average_f1'],
                train_stats['average_bleu1'],
                train_stats['average_bleu4'],
                train_stats['average_rouge1_fmeasure'],
                train_stats['average_rouge2_fmeasure'],
                train_stats['average_rougeL_fmeasure'],
                train_stats['exact_match_rate']
            ]
            
            dev_values = [
                dev_stats['average_f1'],
                dev_stats['average_bleu1'],
                dev_stats['average_bleu4'],
                dev_stats['average_rouge1_fmeasure'],
                dev_stats['average_rouge2_fmeasure'],
                dev_stats['average_rougeL_fmeasure'],
                dev_stats['exact_match_rate']
            ]
            
            # Set up bar positions
            x = np.arange(len(metrics))
            width = 0.35
            
            # Create bars
            plt.bar(x - width/2, train_values, width, label='Train', color='skyblue')
            plt.bar(x + width/2, dev_values, width, label='Dev', color='orange')
            
            # Add labels and title
            plt.xlabel('Evaluation Metric')
            plt.ylabel('Score')
            plt.title('Inter-Annotator Agreement: Train vs Dev Comparison')
            plt.xticks(x, metrics)
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add value labels on top of bars
            for i, v in enumerate(train_values):
                plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
            
            for i, v in enumerate(dev_values):
                plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
            
            # Save the chart
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "train_vs_dev_all_metrics.png"))
            print(f"Comprehensive metrics comparison saved to {os.path.join(output_dir, 'train_vs_dev_all_metrics.png')}")
        except Exception as e:
            print(f"Error generating comprehensive comparison visualization: {e}")

if __name__ == "__main__":
    main()