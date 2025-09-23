import numpy as np
import gurobipy as gp
from gurobipy import GRB

def select_sum_min_diversity(k: int, complete: np.ndarray) -> tuple:
    """
    Select k points from n points to maximize the Sum-Min Diversification objective.
    
    Parameters:
        k: Number of points to select
        complete: n*n adjacency matrix containing pairwise Euclidean distances
        
    Returns:
        selected_points: Indices of the selected k points
        sum_min_dist: Maximum diversity value
    """
    n = complete.shape[0]
    if k < 2 or k > n:
        return np.array([]), 0.0
    
    model = gp.Model("SumMinDiversity")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 1800)
    
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    d = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="d")
    
    model.setObjective(gp.quicksum(d[i] for i in range(n)), GRB.MAXIMIZE)
    
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == k, "select_k")
    
    M = complete.max()
    for i in range(n):
        for j in range(n):
            if i != j:
                model.addConstr(d[i] <= complete[i,j] + M * (2 - x[i] - x[j]), f"min_dist_{i}_{j}")
        model.addConstr(d[i] <= M * x[i], f"active_{i}")
    
    model.optimize()
    
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        if model.SolCount > 0:
            selected_points = np.array([i for i in range(n) if x[i].x > 0.5])
            sum_min_dist = model.ObjVal
            return selected_points, sum_min_dist
    return np.array([]), 0.0

CelebA = np.load("dataset/CelebA_complete.npy")
glove = np.load("dataset/glove_complete.npy")
movielens = np.load("dataset/movielens_complete.npy")
Gaussian_blob = np.load("dataset/Gaussian_blob_complete.npy")

complete_col = [CelebA, glove, movielens, Gaussian_blob]
k_col = [5, 10, 20]
diversity = []
for complete in complete_col:
    for k in k_col:
        selected, diversity_value = select_sum_min_diversity(k, complete)
        diversity.append(diversity_value)

np.save("results/sum_min_opt_div.npy", np.array(diversity))