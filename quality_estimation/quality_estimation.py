import re
import numpy as np
import pandas as pd
from collections import Counter
import json
import os
import matplotlib.pyplot as plt
from scipy import stats

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
    results = []
    
    for i, (q, a1, a2) in enumerate(zip(questions, annotator1_answers, annotator2_answers)):
        try:
            f1 = calculate_f1(a1, a2)
            jaccard = calculate_jaccard(a1, a2)
            exact_match = calculate_exact_match(a1, a2)
            token_overlap = calculate_token_overlap(a1, a2)
            
            f1_scores.append(f1)
            jaccard_scores.append(jaccard)
            exact_match_scores.append(exact_match)
            token_overlap_scores.append(token_overlap)
            
            results.append({
                'question_id': i,
                'question': q,
                'annotator1_answer': a1,
                'annotator2_answer': a2,
                'f1_score': f1,
                'jaccard_score': jaccard,
                'exact_match': exact_match,
                'token_overlap': token_overlap
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
        'min_jaccard': float(np.min(jaccard_scores)),
        'max_jaccard': float(np.max(jaccard_scores)),
        'std_dev_jaccard': float(np.std(jaccard_scores)),
        
        # Exact match statistics
        'exact_match_rate': float(np.mean(exact_match_scores)),
        
        # Token overlap statistics
        'average_token_overlap': float(np.mean(token_overlap_scores)),
        'median_token_overlap': float(np.median(token_overlap_scores)),
        
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
            'train_avg_f1': train_stats['average_f1'],
            'dev_avg_f1': dev_stats['average_f1'],
            'train_median_f1': train_stats['median_f1'],
            'dev_median_f1': dev_stats['median_f1'],
            'train_exact_match_rate': train_stats['exact_match_rate'],
            'dev_exact_match_rate': dev_stats['exact_match_rate'],
            'f1_difference': abs(train_stats['average_f1'] - dev_stats['average_f1']),
            'jaccard_difference': abs(train_stats['average_jaccard'] - dev_stats['average_jaccard']),
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
        
        if 'distributions_significantly_different' in comparison:
            if comparison['distributions_significantly_different']:
                print(f"Statistical test shows the F1 score distributions are significantly different (p={comparison['mann_whitney_p_value']:.4f})")
            else:
                print(f"Statistical test shows no significant difference in F1 score distributions (p={comparison['mann_whitney_p_value']:.4f})")
        
        # Create comparison visualization
        try:
            plt.figure(figsize=(12, 6))
            
            # Data for the chart
            datasets = ['Train', 'Dev']
            avg_f1 = [train_stats['average_f1'], dev_stats['average_f1']]
            med_f1 = [train_stats['median_f1'], dev_stats['median_f1']]
            avg_jaccard = [train_stats['average_jaccard'], dev_stats['average_jaccard']]
            exact_match = [train_stats['exact_match_rate'], dev_stats['exact_match_rate']]
            
            # Set up bar positions
            bar_width = 0.2
            r1 = np.arange(len(datasets))
            r2 = [x + bar_width for x in r1]
            r3 = [x + bar_width for x in r2]
            r4 = [x + bar_width for x in r3]
            
            # Create bars
            plt.bar(r1, avg_f1, width=bar_width, label='Average F1', color='skyblue')
            plt.bar(r2, med_f1, width=bar_width, label='Median F1', color='lightgreen')
            plt.bar(r3, avg_jaccard, width=bar_width, label='Average Jaccard', color='salmon')
            plt.bar(r4, exact_match, width=bar_width, label='Exact Match Rate', color='purple')
            
            # Add labels and title
            plt.xlabel('Dataset')
            plt.ylabel('Score')
            plt.title('Inter-Annotator Agreement Comparison: Train vs Dev')
            plt.xticks([r + bar_width*1.5 for r in range(len(datasets))], datasets)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the chart
            plt.savefig(os.path.join(output_dir, "train_vs_dev_comparison.png"))
            print(f"Comparison visualization saved to {os.path.join(output_dir, 'train_vs_dev_comparison.png')}")
        except Exception as e:
            print(f"Error generating comparison visualization: {e}")

if __name__ == "__main__":
    main()