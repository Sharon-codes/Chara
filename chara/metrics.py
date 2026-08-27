"""Survival evaluation metrics and Concordance Index calculation."""
import numpy as np

def concordance_index(risk_scores, time, event=None) -> float:
    """
    Computes Harrell's Concordance Index (C-Index) natively in pure NumPy.
    
    Parameters:
    -----------
    risk_scores : 1D array-like
        Predicted patient risk scores (higher means worse prognosis/earlier event).
    time : 1D array-like
        Observed survival times or follow-up duration.
    event : 1D array-like, optional
        Censoring indicator (1 = death/event observed, 0 = censored). 
        If None, assumes all events are observed (uncensored).
        
    Returns:
    --------
    c_index : float
        Concordance index in [0, 1] (0.50 = random guessing, 1.0 = perfect prediction).
    """
    risks = np.asarray(risk_scores, dtype=float)
    times = np.asarray(time, dtype=float)
    
    if event is None:
        events = np.ones_like(times, dtype=bool)
    else:
        events = np.asarray(event, dtype=bool)
        
    n = len(times)
    if n < 2:
        return 0.5

    concordant = 0.0
    tied_risk = 0.0
    valid_pairs = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            # Pair is evaluable if one died before the other was censored or both had events at different times
            if times[i] < times[j] and events[i]:
                valid_pairs += 1.0
                if risks[i] > risks[j]:
                    concordant += 1.0
                elif risks[i] == risks[j]:
                    tied_risk += 0.5
            elif times[j] < times[i] and events[j]:
                valid_pairs += 1.0
                if risks[j] > risks[i]:
                    concordant += 1.0
                elif risks[j] == risks[i]:
                    tied_risk += 0.5
            elif times[i] == times[j] and (events[i] or events[j]):
                # Tied times with event
                if events[i] and events[j]:
                    if risks[i] == risks[j]:
                        valid_pairs += 1.0
                        concordant += 1.0

    if valid_pairs == 0:
        return 0.5

    return float((concordant + tied_risk) / valid_pairs)
