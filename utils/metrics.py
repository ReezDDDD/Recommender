import numpy as np

def mse(x, y):
    """A function calculates mean squared errors.

    Args:
        x (list or numpy array): 2D inputs
        y (list or numpy array): 2D inputs

    Returns:
        np.float32 : MSE
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    if not isinstance(y, np.ndarray):
        y = np.array(y)
        
    if x.shape[0] != y.shape[0]:
        raise ValueError("inputs must be same length")
    
    if x.shape[0] == 0:
        raise ValueError("inputs must not be empty")
    
    return np.float32(np.mean(np.power((x - y), 2)))

def rmse(x, y):
    """A function calculates root mean squared errors.

    Args:
        x (list or numpy array): 2D inputs
        y (list or numpy array): 2D inputs

    Returns:
        np.float32 : RMSE
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    if not isinstance(y, np.ndarray):
        y = np.array(y)
        
    if x.shape[0] != y.shape[0]:
        raise ValueError("inputs must be same length")
    
    if x.shape[0] == 0:
        raise ValueError("inputs must not be empty")
    
    return np.sqrt(mse(x, y))

def mae(x, y):
    """A function calculates mean absolute errors.

    Args:
        x (list or numpy array): 2D inputs
        y (list or numpy array): 2D inputs

    Returns:
        np.float32 : MAE
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    if not isinstance(y, np.ndarray):
        y = np.array(y)
        
    if x.shape[0] != y.shape[0]:
        raise ValueError("inputs must be same length")
    
    if x.shape[0] == 0:
        raise ValueError("inputs must not be empty")
    
    return np.float32(np.mean(np.absolute(x - y)))

def precision(x, y, N = None):
    """A function calculates precision score.

    Args:
        x (list or numpy array): 2D inputs, personalized recommendation list, sorted by the probabilities
        y (list or numpy array): 2D inputs, actual selection list, sorted by the probabilities
        N (int, optional): precision at N. Defaults to None.

    Returns:
        np.float32: Precision
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    if x.shape[0] != y.shape[0]:
        raise ValueError("inputs must be same length")
    
    if (x.shape[0] == 0) or (y.shape[0] == 0):
        raise ValueError("inputs must not be empty")

    if N:
        if not isinstance(N, int):
            raise TypeError("N must be an positive integer")
        x = x[:, :N]
    
    hit, n = 0.0, 0.0
    for a, b in zip(x, y):
        hit += len(np.intersect1d(a, b))
        n += len(a)
     
    # * for debugging
    # print(hit, n)   
    return np.float32(hit / n)

def recall(x, y, N = None):
    """A function calculates recall score.

    Args:
        x (list or numpy array): 2D inputs, personalized recommendation list, sorted by the probabilities
        y (list or numpy array): 2D inputs, actual selection list, sorted by the probabilities
        N (int, optional): recall at N. Defaults to None.

    Returns:
        np.float32: Precision
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    if x.shape[0] != y.shape[0]:
        raise ValueError("inputs must be same length")
    
    if (x.shape[0] == 0) or (y.shape[0] == 0):
        raise ValueError("inputs must not be empty")

    if N:
        if not isinstance(N, int):
            raise TypeError("N must be an positive integer")
        x = x[:, :N]
    
    hit, n = 0.0, 0.0
    for a, b in zip(x, y):
        hit += len(np.intersect1d(a, b))
        n += len(b)
     
    # * for debugging
    # print(hit, n)
    return np.float32(hit / n)

# todo
def ap(x, y, N):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    if x.shape[0] != y.shape[0]:
        raise ValueError("inputs must be same length")
    
    if (x.shape[0] == 0) or (y.shape[0] == 0):
        raise ValueError("inputs must not be empty")

    if N:
        if not isinstance(N, int):
            raise TypeError("N must be an positive integer")
        x = x[:, :N]
