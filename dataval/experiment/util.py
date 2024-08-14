import inspect
from itertools import accumulate
from typing import Any, Callable, Sequence

import numpy as np


def filter_kwargs(func: Callable, **kwargs) -> dict[str, Any]:
    """Filters out non-arguments of a specific function out of kwargs.

    Parameters
    ----------
    func : Callable
        Function with a specified signature, whose kwargs can be extracted from kwargs
    kwargs : dict[str, Any]
        Key word arguments passed to the function

    Returns
    -------
    dict[str, Any]
        Key word arguments of func that are passed in as kwargs
    """
    params = inspect.signature(func).parameters.values()
    filter_keys = [p.name for p in params if p.kind == p.POSITIONAL_OR_KEYWORD]
    return {key: kwargs[key] for key in filter_keys if key in kwargs}


def oned_twonn_clustering(vals: Sequence[float], outlier_fraction: float = 0.1) -> tuple[Sequence[int], Sequence[int]]:
    """O(nlog(n)) sort, O(n) pass exact 2-NN clustering of 1 dimensional input data.

    References
    ----------
    .. [1] A. Grønlund, K. G. Larsen, A. Mathiasen, J. S. Nielsen, S. Schneider,
        and M. Song,
        Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D,
        arXiv.org, 2017. https://arxiv.org/abs/1701.07204.

    Parameters
    ----------
    vals : Sequence[float]
        Input floats which to cluster

    Returns
    -------
    tuple[Sequence[int], Sequence[int]]
        Indices of the data points in each cluster, because of the convexity of KMeans,
        the first sequence represents the lower value group and the second the higher
    """
    sid = np.argsort(vals, kind="stable")
    n = len(vals)
    

    psums = list(accumulate((vals[sid[i]] for i in range(n)), initial=0.0))
    psqsums = list(accumulate((vals[sid[i]] ** 2 for i in range(n)), initial=0.0))

    def cost(i: int, j: int):
        sij = psums[j + 1] - psums[i]
        uij = sij / (j - i + 1)
        return (uij**2) * (j - i + 1) + (psqsums[j + 1] - psqsums[i]) - 2 * uij * sij
    
    # split = min((i for i in range(1, n)), key=lambda i: cost(0, i - 1) + cost(i, n - 1))
    
    # Keep splitting until the lower value group is at least as large as the outlier fraction
    start = 0
    split = 0
    while split < outlier_fraction * n:
        # Calculate the new split point within the remaining data
        new_split = min((i for i in range(start + 1, n)), key=lambda i: cost(start, i - 1) + cost(i, n - 1))
        
        # If no more splits can be made, break the loop to avoid infinite loop
        if new_split == split:
            break
        
        split = new_split
        start = split
    
    return sid[range(0, split)], sid[range(split, n)]


def oned_twonn_clustering_filter(vals: Sequence[float], outlier_fraction: float = 0.5) -> tuple[Sequence[int], Sequence[int]]:
    """O(nlog(n)) sort, O(n) pass exact 2-NN clustering of 1 dimensional input data.

    References
    ----------
    .. [1] A. Grønlund, K. G. Larsen, A. Mathiasen, J. S. Nielsen, S. Schneider,
        and M. Song,
        Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D,
        arXiv.org, 2017. https://arxiv.org/abs/1701.07204.

    Parameters
    ----------
    vals : Sequence[float]
        Input floats which to cluster

    Returns
    -------
    tuple[Sequence[int], Sequence[int]]
        Indices of the data points in each cluster, because of the convexity of KMeans,
        the first sequence represents the lower value group and the second the higher
    """
    sid = np.argsort(vals, kind="stable")
    n = len(vals)
    

    psums = list(accumulate((vals[sid[i]] for i in range(n)), initial=0.0))
    psqsums = list(accumulate((vals[sid[i]] ** 2 for i in range(n)), initial=0.0))

    def cost(i: int, j: int):
        sij = psums[j + 1] - psums[i]
        uij = sij / (j - i + 1)
        return (uij**2) * (j - i + 1) + (psqsums[j + 1] - psqsums[i]) - 2 * uij * sij
    
    # Keep splitting until the lower value group is at least as large as the outlier fraction
    start = 0
    split = 0
    while split < outlier_fraction * n:
        # Calculate the new split point within the remaining data
        new_split = min((i for i in range(start + 1, n)), key=lambda i: cost(start, i - 1) + cost(i, n - 1))
        
        # If no more splits can be made, break the loop to avoid infinite loop
        if new_split == split:
            break
        
        split = new_split
        start = split
    
    return sid[range(0, split)], sid[range(split, n)]
    

def f1_score(predicted: Sequence[float], actual: Sequence[float], total: int) -> float:
    """Computes the F1 score based on the indices of values found."""
    predicted_set, actual_set = set(predicted), set(actual)

    tp, fp, fn = 0, 0, 0
    for i in range(total):
        if i in predicted_set and i in actual_set:
            tp += 1
        elif i in predicted_set:
            fp += 1
        elif i in actual_set:
            fn += 1
    
    return 2 * tp / (2 * tp + fp + fn)

def EOp_score(predicted: Sequence[float], actual: Sequence[float], label: Sequence[float], total: int) -> float:
    """Computes the equal opportunity score based on the indices of values found."""
    predicted_set, actual_set = set(predicted), set(actual)

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(total):
        if i in predicted_set and i in actual_set:
            tp += 1
        elif i in predicted_set:
            fp += 1
        elif i in actual_set:
            fn += 1
        else:
            tn += 1
            
    tpr = tp / (tp + fn)
    
    EOp_list = []
    for label_indices in label:
        label_set = set(label_indices) 
        tpa, fpa, tna, fna = 0, 0, 0, 0
        for i in range(total):
            if i in predicted_set and i in label_set:
                tpa += 1
            elif i in predicted_set:
                fpa += 1
            elif i in label_set:
                fna += 1
            else:
                tna += 1
    
        tpr_a = tpa / (tpa + fna)
        EOp_list.append(abs(tpr - tpr_a))
    
    return max(EOp_list)

def EOdds_score(predicted: Sequence[float], actual: Sequence[float], label: Sequence[float], total: int) -> float:
    """Computes the equalized odds score based on the indices of values found."""
    predicted_set, actual_set = set(predicted), set(actual)

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(total):
        if i in predicted_set and i in actual_set:
            tp += 1
        elif i in predicted_set:
            fp += 1
        elif i in actual_set:
            fn += 1
        else:
            tn += 1
            
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    EOdds_list = []
    for label_indices in label:
        label_set = set(label_indices) 
        tpa, fpa, tna, fna = 0, 0, 0, 0
        for i in range(total):
            if i in predicted_set and i in label_set:
                tpa += 1
            elif i in predicted_set:
                fpa += 1
            elif i in label_set:
                fna += 1
            else:
                tna += 1
    
        tpr_a = tpa / (tpa + fna)
        fpr_a = fpa / (fpa + tna)
        EOdds_list.append(1/2 * (abs(tpr - tpr_a) + abs(fpr - fpr_a)))
    
    return max(EOdds_list)