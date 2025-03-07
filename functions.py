import numpy as np
import random
import gurobipy as gp
from gurobipy import GRB

def MaxMinDiversity(subset: np.ndarray, complete: np.ndarray) -> float:
    """
    Calculate Max-Min Diversity, the minimum distance between all pairs of points in the given subset.
    
    Parameters:
        subset: Subset of points
        complete: Complete graph adjacency matrix containing distances between all pairs of points
    
    Returns:
        min_dist: Max-Min Diversity, the minimum distance between all pairs of points in the subset
    """
    if len(subset) < 2:
        return 0
    
    n = len(subset)
    min_dist = float('inf')
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = complete[subset[i]][subset[j]]
            min_dist = min(min_dist, dist)
    
    return min_dist
def FairRadius(k: int, complete: np.ndarray) -> np.ndarray:
    """
    Calculate Fair Radius for all points in the dataset.

    Parameters:
        k: Integer parameter k, used to calculate minimum number of points each point needs to cover ⌈n/k⌉
        complete: Complete graph adjacency matrix representing distances between all pairs of points
        
    Returns:
        fair_radius: Fair Radius for all points in the dataset
    """
    n = len(complete)
    min_points = int(np.ceil(n / k))
    fair_radius = []

    for i in range(n):
        distances = complete[i]
        sorted_distances = np.sort(distances)
        point_radius = sorted_distances[min_points-1]
        fair_radius.append(point_radius)
        
    return np.array(fair_radius)

def CriticalRegion(alpha: float, fair_radius: np.ndarray, complete: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Generate the Critical Regions by finding a set of centers that cover all points.
    
    Parameters:
        alpha: Fairness parameter
        fair_radius: Array containing fair radius values for all points
        complete: Complete graph adjacency matrix representing distances between all pairs of points
        
    Returns:
        selected_centers: Centers of critical regions
        critical_regions: Dictionary mapping each center to its covered points
    """
    covered_points = set()
    n = len(fair_radius)
    points = set(range(n))
    selected_centers = []
    critical_regions = {}

    while covered_points != points:
        minus = points - covered_points
        c = min(minus, key=lambda x: fair_radius[x])
        selected_centers.append(c)
        for point in minus:
            if complete[point][c] <= 2 * alpha * fair_radius[point]:
                covered_points.add(point)
        print(len(covered_points))
    selected_centers = np.array(selected_centers)

    for center in selected_centers:
        r_c = fair_radius[center]
        distances = complete[center]
        points_in_circle = np.array([i for i, dist in enumerate(distances) if dist <= alpha * r_c])
        critical_regions[center] = points_in_circle

    return selected_centers, critical_regions

def IFDM(k: int, complete: np.ndarray, epsilon: float, beta: float, selected_centers: np.ndarray, critical_regions: dict) -> tuple[np.ndarray, tuple[tuple[np.ndarray, int], ...], np.ndarray, dict]:
    """
    Construct an instance of k-clustering under partition matroid constraint corresponding to the given instance of alpha-fair k-clustering.
    
    Parameters:
        k: Target number of clusters
        alpha: Fairness parameter
        complete: Complete graph adjacency matrix with distances
        epsilon: Accuracy parameter (< 1/2)
        beta: Approximation guarantee parameter (≤ 1)
        selected_centers: Centers of critical regions
        critical_regions: Dictionary mapping centers to covered points
        
    Returns:
        P_prime: Augmented point set with duplicated points
        centers_info: Tuple containing center selection info
        d_prime: Modified graph adjacency matrix with distances
        corresponding: The dictionary for getting the corresponding point in originial data
    """
    n = len(complete)
    P_0 = list(range(n))
    P_1 = list(range(n))
    current_max_id = n
    
    B_copies = {}
    corresponding = {}
    for center in selected_centers:
        copies = []
        for point in critical_regions[center]:
            corresponding[current_max_id] = point
            copies.append(current_max_id)
            P_1.append(point)
            current_max_id += 1
        B_copies[center] = np.array(copies)

    P_prime = np.arange(current_max_id)
    
    k_i = {center: 1 for center in selected_centers}
    
    k_0 = k - len(selected_centers)

    delta = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            if complete[i][j] > 0:
                delta = min(delta, complete[i][j])

    n_prime = len(P_prime)
    d_prime = np.zeros((n_prime, n_prime))

    for i in range(n_prime):
        for j in range(n_prime):
            if i == j:
                d_prime[i][j] = 0
            elif P_1[i] != P_1[j]:
                d_prime[i][j] = complete[P_1[i]][P_1[j]]
            else:
                d_prime[i][j] = epsilon * beta * delta
                
    return P_prime, ((P_0, k_0), *((B_copies[c], k_i[c]) for c in selected_centers)), d_prime, corresponding

def distancePS(centerSet: np.ndarray, i: int, complete: np.ndarray) -> float:
    """
    Returns the distance between a certain point and a certain set.
    
    Parameters:
        centerSet: A numpy array containing confirmed center indexes
        i: The index of any point
        complete : Complete graph adjacency matrix containing distances between all pairs of points
    
    Returns:
        min_distance: The distance between point and center set
    """
    min_distance = float("inf")
    for center in centerSet:
        distance = complete[center][i]
        if (distance < min_distance):
            min_distance = distance
    
    return min_distance
def GMM(points_index: np.ndarray, k: int, complete: np.ndarray) -> np.ndarray:
    """
    Returns indexes of k centers after running GMM Algorithm.
    
    Parameters: 
        points_index: The indexes of data
        k: A decimal integer, the number of centers
        complete: Complete graph adjacency matrix containing distances between all pairs of points
        initial: An initial set of elements
    
    Returns:
        centers: A numpy array with k indexes as center point indexes
    """
    centers = []
    initial_point_index = random.choice(points_index)
    centers.append(initial_point_index)
    while (len(centers) < k):
        max_distance = 0
        max_distance_vector_index = None
        for i in points_index:
            distance = distancePS(centers, i, complete)
            if distance > max_distance:
                max_distance = distance
                max_distance_vector_index = i
        centers.append(max_distance_vector_index)
    centers = np.array(centers)

    return centers
def ILP(n, E, k, color_constraints, color_sets):
    """
    Returns an ILP solution for FMMD-S.
    
    Parameters:
        n: The number of points
        E: The undirected graph generated by FMMD-S
        k: A decimal integer, the number of centers 
        color_constraints: The groups that the data are divided into
        color_sets: The points in different groups
    
    Returns:
        selected_nodes: A numpy array containing selected elements that maximize the minimum pairwise distance    
    """
    model = gp.Model("ILP")

    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    model.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)

    for (i, j) in E:
        model.addConstr(x[i] + x[j] <= 1, f"edge_{i}_{j}")

    model.addConstr(gp.quicksum(x[i] for i in range(n)) <= k, "max_selection")

    for c, number in color_constraints.items():
        nodes_in_color = color_sets[c]
        model.addConstr(gp.quicksum(x[i] for i in nodes_in_color) == number, f"number_color_{c}")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        selected_nodes = np.array([i for i in range(n) if x[i].x > 0.5])
        return selected_nodes
    else:
        raise ValueError("No optimal solution found.")
        
def FMMDS(sets: tuple, k: int, error: float, complete: np.ndarray) -> np.ndarray:
    """
    Returns a subset with high Max-min diversity under partition matroid constraint.
    
    Parameters:
        sets: A tuple containing partitioned sets returned by IFDM function
        k: A decimal integer, the number of elements to be selected
        error: A float number indicating the error parameter
        complete: Complete graph adjacency matrix containing distances between all pairs of points
    
    Returns:
        solution: A numpy array containing selected elements that maximize the minimum pairwise distance
    """
    amount = complete.shape[0]
    complete_array = np.arange(amount)
    U_gmm = GMM(complete_array, k, complete)
    critical_number = len(sets)
    U_c = []
    for i in range(critical_number):
        U_c.append(np.intersect1d(sets[i][0], U_gmm))
    distance_p = 2 * MaxMinDiversity(U_gmm, complete)
    print(U_c)

    S = []
    while len(S) == 0:
        for c in range(critical_number):
            while len(U_c[c]) < k and any(v for v in sets[c][0] if MaxMinDiversity(np.union1d(U_c[c], [v]), complete) >= distance_p):
                max_distance = 0
                max_distance_vector_index = None
                for i in sets[c][0]:
                    distance = distancePS(U_c[c], i, complete)
                    if distance > max_distance:
                        max_distance = distance
                        max_distance_vector_index = i
                U_c[c] = np.append(U_c[c], max_distance_vector_index)

        V_p = np.concatenate(U_c)
        E = []
        for i in range(len(V_p)):
             for j in range(i + 1, len(V_p)):
                  if complete[V_p[i]][V_p[j]] < distance_p / 2:
                       E.append((V_p[i], V_p[j]))

        color_constraints = dict()
        color_sets = dict()
        for group in range(critical_number):
            color_constraints[group] = sets[group][1]
            color_sets[group] = sets[group][0]
        
        solution = ILP(amount, E, k, color_constraints, color_sets)
        if len(solution) < k:
             distance_p = (1 - error) * distance_p
        else:
             S = solution
    return S

def SOL(S: np.ndarray, corresponding: dict, complete: np.ndarray) -> np.ndarray:
    """
    Returns a np.ndarray with high Max-min diversity in the original data.
    
    Parameters:
        S: The solution from FMMDS
        corresponding: The dictionary for getting the corresponding point in originial data
        complete: Complete graph adjacency matrix containing distances between all pairs of points
    
    Returns:
        solution: A numpy array containing selected elements that maximize the minimum pairwise distance
    """
    solution = []
    for point in S:
        if point < complete.shape[0]:
            solution.append(point)
        elif corresponding[point] not in solution:
            solution.append(corresponding[point])
        else:
            continue
    solution = np.array(solution)

    return solution