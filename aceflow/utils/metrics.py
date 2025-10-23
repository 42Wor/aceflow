import numpy as np
from collections import Counter

def calculate_bleu(reference, candidate, n=4):
    """
    Calculate BLEU score between reference and candidate sequences
    """
    reference_tokens = reference
    candidate_tokens = candidate
    
    # Brevity penalty
    if len(candidate_tokens) < len(reference_tokens):
        bp = np.exp(1 - len(reference_tokens) / len(candidate_tokens))
    else:
        bp = 1.0
    
    # Calculate modified n-gram precision
    precisions = []
    for i in range(1, n + 1):
        ref_ngrams = _get_ngrams(reference_tokens, i)
        cand_ngrams = _get_ngrams(candidate_tokens, i)
        
        if not cand_ngrams:
            precisions.append(0)
            continue
            
        # Count matching n-grams
        match_count = 0
        for ngram in cand_ngrams:
            if ngram in ref_ngrams:
                match_count += 1
                # Remove the matched n-gram to avoid double counting
                ref_ngrams.remove(ngram)
        
        precisions.append(match_count / len(cand_ngrams))
    
    # Geometric mean of precisions
    if min(precisions) > 0:
        geo_mean = np.exp(np.sum(np.log(precisions)) / n)
    else:
        geo_mean = 0
    
    bleu = bp * geo_mean
    return bleu

def _get_ngrams(tokens, n):
    """Extract n-grams from token sequence"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def calculate_accuracy(reference, candidate):
    """Calculate token-level accuracy"""
    if len(reference) != len(candidate):
        return 0.0
    
    correct = sum(1 for ref, cand in zip(reference, candidate) if ref == cand)
    return correct / len(reference)