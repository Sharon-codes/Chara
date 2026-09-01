"""Survival evaluation metrics including Harrell's Concordance Index and Time-Dependent Brier Scores."""
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
                if events[i] and events[j]:
                    if risks[i] == risks[j]:
                        valid_pairs += 1.0
                        concordant += 1.0

    if valid_pairs == 0:
        return 0.5

    return float((concordant + tied_risk) / valid_pairs)


def brier_score_at_time(survival_probs_at_t, time, event, eval_time=60.0) -> float:
    """
    Computes Time-Dependent Brier Score calibration metric at a specified time horizon t.
    
    Parameters:
    -----------
    survival_probs_at_t : 1D array-like
        Predicted survival probability S_i(t) for each patient at time horizon eval_time.
    time : 1D array-like
        Observed survival times.
    event : 1D array-like
        Censoring indicator (1 = death/event observed, 0 = censored).
    eval_time : float
        Evaluation milestone in months (default 60.0 for 5-year benchmark).
        
    Returns:
    --------
    brier_score : float
        Calibration Brier score (lower is better, 0.0 = perfect probabilistic calibration).
    """
    probs = np.asarray(survival_probs_at_t, dtype=float)
    times = np.asarray(time, dtype=float)
    events = np.asarray(event, dtype=bool) if event is not None else np.ones_like(times, dtype=bool)
    
    n = len(times)
    if n == 0:
        return 0.0

    # True binary status at eval_time
    # Patient alive at eval_time: time > eval_time -> Y = 1
    # Patient died before eval_time: time <= eval_time & event == 1 -> Y = 0
    scores = []
    for i in range(n):
        if times[i] > eval_time:
            # Alive at eval_time
            scores.append((1.0 - probs[i]) ** 2)
        elif events[i] and times[i] <= eval_time:
            # Event occurred prior to eval_time
            scores.append((0.0 - probs[i]) ** 2)
        # Censored prior to eval_time are omitted from unweighted estimate
        
    if len(scores) == 0:
        return float(np.mean((probs - 0.5) ** 2))
        
    return float(np.mean(scores))
