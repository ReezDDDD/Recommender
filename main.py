import numpy as np
from utils.metrics import mse, rmse, mae, precision, recall
from utils.plot import precision_recall_curve

# x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
# y = np.array([[4, 3, 2, 1], [8, 7, 6, 5]])   
# print(mse(x, y))
# print(rmse(x, y))
# print(mae(x, y))

x = np.array([[1, 2, 6, 5, 8, 3, 9, 10, 4], [5, 6, 3, 2, 1, 7, 9, 4, 8]])
y = np.array([[4, 3, 2, 1], [8, 7, 6, 5]])  
# N = 3
# print(precision(x, y, N))
# print(recall(x, y, N))

# N = None
# print(precision(x, y, N))
# print(recall(x, y, N))

# precision_recall_curve(x, y, 1, 9)