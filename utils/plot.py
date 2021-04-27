import numpy as np
import plotly.graph_objects as go
from utils.metrics import precision, recall

def precision_recall_curve(x, y, start, end):
    """A function draws precision_recall_curve.

    Args:
        x (list or numpy array): 2D inputs, personalized recommendation list, sorted by the probabilities
        y (list or numpy array): 2D inputs, actual selection list, sorted by the probabilities
        start (int): start length of the recommendation list
        end (int): end length of the recommendation list
    """
    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError("start and end must be an positive integer")
    
    if start > end:
        raise ValueError("start must be less than or equal to end")
    
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    if x.shape[0] != y.shape[0]:
        raise ValueError("inputs must be same length")
    
    if (x.shape[0] == 0) or (y.shape[0] == 0):
        raise ValueError("inputs must not be empty")

    precisions, recalls = [], []
    for i in range(start, end + 1):
        precisions.append(precision(x, y, i))
        recalls.append(recall(x, y, i))
    
    fig = go.Figure(go.Scatter(x = recalls, y = precisions, mode = "lines+markers"))
    fig.update_layout(
        title = "precision-recall curve",
        xaxis_title = "recall",
        yaxis_title = "precision",
    )
    fig.update_xaxes(range=(0.0, 1.1))
    fig.update_yaxes(range=(0.0, 1.1))
    fig.show()