import re
import string
from typing import List, Callable, Tuple


def normalize_answer(s: str) -> str:
    """
    Normalize answer string for fair comparison
    1. Convert to lowercase
    2. Remove punctuation, articles and extra whitespace
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truths: List[str], normalize_fn: Callable = None) -> Tuple[float, float]:
    """
    Calculate F1 and recall scores comparing prediction to ground truths
    """
    if normalize_fn is None:
        normalize_fn = lambda x: x
    
    prediction = normalize_fn(prediction)
    f1_scores = []
    recall_scores = []
    
    for ground_truth in ground_truths:
        ground_truth = normalize_fn(ground_truth)
        
        prediction_tokens = prediction.split()
        ground_truth_tokens = ground_truth.split()
        
        # Calculate precision and recall
        common = set(prediction_tokens) & set(ground_truth_tokens)
        num_same = len(common)
        
        if num_same == 0:
            f1_scores.append(0)
            recall_scores.append(0)
            continue
            
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        
        # Calculate F1
        f1 = 2 * precision * recall / (precision + recall)
        
        f1_scores.append(f1)
        recall_scores.append(recall)
    
    # Return the maximum F1 and recall across all ground truths
    return max(f1_scores), max(recall_scores)


def exact_match_score(prediction: str, ground_truths: List[str], normalize_fn: Callable = None) -> float:
    """
    Calculate exact match score comparing prediction to ground truths
    """
    if normalize_fn is None:
        normalize_fn = lambda x: x
    
    prediction = normalize_fn(prediction)
    
    for ground_truth in ground_truths:
        if prediction == normalize_fn(ground_truth):
            return 1
    
    return 0


def evaluate_predictions(predictions: List[str], references: List[str]) -> Tuple[float, float, float]:
    """
    Evaluate a list of predictions against reference answers
    Returns tuple of (F1, Recall, Exact Match) scores
    """
    assert len(predictions) == len(references), "Number of predictions and references must match"
    
    f1_total = 0
    recall_total = 0
    em_total = 0
    
    for pred, ref in zip(predictions, references):
        # Handle multiple reference answers separated by '|'
        if '|' in ref:
            ref_list = ref.split('|')
        else:
            ref_list = [ref]
        
        f1, recall = f1_score(pred, ref_list, normalize_fn=normalize_answer)
        em = exact_match_score(pred, ref_list, normalize_fn=normalize_answer)
        
        f1_total += f1
        recall_total += recall
        em_total += em
    
    # Calculate averages
    num_samples = len(predictions)
    return f1_total / num_samples, recall_total / num_samples, em_total / num_samples