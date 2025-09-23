import numpy as np
import random
import gurobipy as gp
from gurobipy import GRB
import time

def MaxMinDiversity(subset: np.ndarray, complete: np.ndarray) -> float:
    """
    Calculate diversity for Max-Min Diversification objective, the minimum distance between all pairs of points in the given subset.
    
    Parameters:
        subset: Subset of points (array of indices)
        complete: Complete graph adjacency matrix containing distances between all pairs of points
    
    Returns:
        min_dist: Diversity for Max-Min Diversification objective, the minimum distance between all pairs of points in the subset
    """
    if len(subset) < 2:
        return 0.0
    
    submatrix = complete[subset][:, subset]
    np.fill_diagonal(submatrix, np.inf)
    min_dist = np.min(submatrix)
    
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
    n = complete.shape[0]
    min_points = int(np.ceil(n / k))
    fair_radius = []

    for i in range(n):
        distances = complete[i]
        sorted_distances = np.sort(distances)
        point_radius = sorted_distances[min_points-1]
        fair_radius.append(point_radius)
        
    return np.array(fair_radius)

def IFRegion(alpha: float, fair_radius: np.ndarray, complete: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Generate the Individual Fairness Regions by finding a set of centers that cover all points.
    
    Parameters:
        alpha: Fairness parameter
        fair_radius: Array containing fair radius values for all points
        complete: Complete graph adjacency matrix representing distances between all pairs of points
        
    Returns:
        selected_centers: Centers of individual fairness regions
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
    selected_centers = np.array(selected_centers)

    for center in selected_centers:
        r_c = fair_radius[center]
        distances = complete[center]
        points_in_circle = np.array([i for i, dist in enumerate(distances) if dist <= alpha * r_c])
        critical_regions[center] = points_in_circle

    return selected_centers, critical_regions

def OriginalMetric(k: int, complete: np.ndarray, selected_centers: np.ndarray, critical_regions: dict) -> tuple[tuple[np.ndarray, int], ...]:
    """
    Generate annotations for Individual Fairness Regions in the original metric space, ensuring total selected points equal k.
    
    Parameters:
        k: Target number of clusters
        complete: Complete graph adjacency matrix representing distances between all pairs of points
        selected_centers: Centers of individual fairness regions
        critical_regions: Dictionary mapping each center to its covered points
        
    Returns:
        A tuple containing:
        - (P_0, points_to_take): Points not in any individual fairness region and the maximal number of points to take
        - Multiple tuples of (critical_region_points, points_to_take) for each center
    """
    n = complete.shape[0]
    m = len(selected_centers)
    
    P_0 = np.arange(n)
    for center in selected_centers:
        P_0 = np.setdiff1d(P_0, critical_regions[center])
    
    base = 1
    while True:
        p0_points = k - base * m
        if p0_points <= 0:
            points_per_region = k // m
            extra_points = k % m
            result = []
            for i, center in enumerate(selected_centers):
                region_points = len(critical_regions[center])
                target = points_per_region + (1 if i < extra_points else 0)
                if region_points < target:
                    raise ValueError(f"Critical region {center} has {region_points} points, but needs {target}")
                result.append((critical_regions[center], target))
            return tuple(result)
        
        if len(P_0) >= p0_points:
            valid = True
            for center in selected_centers:
                if len(critical_regions[center]) < base:
                    valid = False
                    break
            if valid:
                return ((P_0, p0_points), *((critical_regions[c], base) for c in selected_centers))
        
        base += 1

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
    try:
        model = gp.Model("ILP")
        model.setParam('OutputFlag', 0)
        x = model.addVars(n, vtype=GRB.BINARY, name="x")

        model.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)

        eid = 0
        for e in E:
            model.addConstr(x[e[0]] + x[e[1]] <= 1, f"edge_{str(eid)}")
            eid += 1

        model.addConstr(gp.quicksum(x[i] for i in range(n)) <= k, "max_selection")

        for c, number in color_constraints.items():
            nodes_in_color = color_sets[c]
            if c != 0:
                model.addConstr(gp.quicksum(x[i] for i in nodes_in_color) <= number, f"number_color_{c}")
                model.addConstr(gp.quicksum(x[i] for i in nodes_in_color) >= 1, f"number_color_{c}")

        model.optimize()

        selected_nodes = np.array([i for i in range(n) if x[i].x > 0.5])

        if len(selected_nodes) < k:
            return 0, 0
        else:
            return 1, selected_nodes

    except gp.GurobiError as error:
        print("Error code " + str(error))
    
    except AttributeError:
        return 0, 0
        
def FMMDS(sets: tuple, k: int, error: float, complete: np.ndarray) -> np.ndarray:
    """
    Returns a subset with high diversity under partition matroid constraints.
    
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
    category_number = len(sets)
    U_c = []
    for i in range(category_number):
        intersection = np.intersect1d(sets[i][0], U_gmm)
        U_c.append(intersection)
            
    distance_p = 2 * MaxMinDiversity(U_gmm, complete)

    S = []
    while len(S) == 0:
        for c in range(category_number):  
            if len(U_c[c]) == 0:
                random_element = np.random.choice(sets[c][0])
                U_c[c] = np.append(U_c[c], random_element)
            max_div_uc_v = 0
            for v in sets[c][0]:
                new_div_uc_v = MaxMinDiversity(np.union1d(U_c[c], [v]), complete)
                if new_div_uc_v > max_div_uc_v:
                    max_div_uc_v = new_div_uc_v
            while len(U_c[c]) < k and max_div_uc_v >= distance_p:
                max_distance = 0
                max_distance_vector_index = 0
                for i in sets[c][0]:
                    distance = distancePS(U_c[c], i, complete)
                    if distance > max_distance:
                        max_distance = distance
                        max_distance_vector_index = i
                U_c[c] = np.append(U_c[c], max_distance_vector_index)

        V_p = np.concatenate(U_c)
        index2V_p = dict()
        V_p2index = dict()
        for index, point in enumerate(V_p):
            index2V_p[index] = point
            V_p2index[point] = index

        E = []
        for i in range(len(V_p)):
             for j in range(i + 1, len(V_p)):
                  if complete[V_p[i]][V_p[j]] < distance_p / 2:
                       E.append((i, j))

        color_constraints = dict()
        color_sets = dict()
        for group in range(category_number):
            color_constraints[group] = sets[group][1]
            color_sets_original = np.intersect1d(sets[group][0], V_p)
            color_sets_ilp = []
            for point in color_sets_original:
                color_sets_ilp.append(V_p2index[point])
            color_sets[group] = np.array(color_sets_ilp)

        flag, solution = ILP(len(V_p), E, k, color_constraints, color_sets)
        if flag == 0:
            distance_p = (1 - error) * distance_p
        else:
            S = solution

    for i in range(len(S)):
        S[i] = index2V_p[S[i]]
    
    return S

def Examine(complete: np.ndarray, solution: np.ndarray, fair_radius: np.ndarray, alpha: float, gamma: float) -> bool:
    """
    Examines whether a solution satisfies individual fairness constraints on the dataset.
    
    Parameters:
        complete: Complete graph adjacency matrix containing distances between all pairs of points
        solution: The solution to be examined
        fair_radius: Fair radius for each point
        alpha: Fairness parameter
        gamma: Amplification coefficient for fairness constraints
        
    Returns:
        bool: True if the solution satisfies individual fairness constraints, False otherwise
    """
    points = np.arange(complete.shape[0])
    for point in points:
        if distancePS(solution, point, complete) > alpha * gamma * fair_radius[point]:
            print(distancePS(solution, point, complete))
            print(alpha * gamma * fair_radius[point])
            return False
    
    return True

def ExactAlpha(complete: np.ndarray, solution: np.ndarray, fair_radius: np.ndarray) -> float:
    """
    Examines whether a solution satisfies individual fairness constraints on the dataset.
    
    Parameters:
        complete: Complete graph adjacency matrix containing distances between all pairs of points
        solution: The solution to be examined
        fair_radius: Fair radius for each point
        
    Returns:
        bool: True if the solution satisfies individual fairness constraints, False otherwise
    """
    points = np.arange(complete.shape[0])
    alpha_list = []
    for point in points:
        alpha_list.append(distancePS(solution, point, complete) / fair_radius[point]) 
    
    return max(alpha_list)

def AverageDistance(complete: np.ndarray, solution: np.ndarray) -> float:
    """
    Calculate the average distance from points not in the solution to the solution set.
    
    Parameters:
        complete: Complete graph adjacency matrix containing distances between all pairs of points
        solution: A numpy array containing the indices of selected points (solution set)
    
    Returns:
        avg_distance: The average distance from all points not in the solution to the closest point in the solution
    """
    points = np.arange(complete.shape[0])
    non_solution_points = np.setdiff1d(points, solution)
    if len(non_solution_points) == 0:
        return 0
    
    distances = []
    for point in non_solution_points:
        min_distance = distancePS(solution, point, complete)
        distances.append(min_distance)
    
    avg_distance = np.mean(distances)
    return avg_distance

def MaxDistance(complete: np.ndarray, solution: np.ndarray) -> float:
    """
    Returns the k-center loss value with centers and data points.

    Parameters:
        complete: Complete graph adjacency matrix containing distances between all pairs of points
        solution: A numpy array containing the indices of selected points (solution set)

    Returns:
        max_distance (float): The k-center loss value of certain centers and data points
    """
    max_distance = float("-inf")
    points = np.arange(complete.shape[0])
    non_solution_points = np.setdiff1d(points, solution)
    for i in non_solution_points:
        distance = distancePS(solution, i, complete)
        if (distance > max_distance):
            max_distance = distance
    
    return max_distance

CelebA = np.load("dataset/CelebA_complete.npy")
glove = np.load("dataset/glove_complete.npy")
movielens = np.load("dataset/movielens_complete.npy")
Gaussian_blob = np.load("dataset/Gaussian_blob_complete.npy")

alpha_CelebA = []
average_CelebA = []
averageMax_CelebA = []
alpha_constrain_CelebA = []
average_constrain_CelebA = []
averageMax_constrain_CelebA = []

alpha_glove = []
average_glove = []
averageMax_glove = []
alpha_constrain_glove = []
average_constrain_glove = []
averageMax_constrain_glove = []

alpha_movielens = []
average_movielens = []
averageMax_movielens = []
alpha_constrain_movielens = []
average_constrain_movielens = []
averageMax_constrain_movielens = []

alpha_gaussian = []
average_gaussian = []
averageMax_gaussian = []
alpha_constrain_gaussian = []
average_constrain_gaussian = []
averageMax_constrain_gaussian = []

alpha = 1
complete = CelebA
for K in range(2, 51):
    start_time = time.time()
    alpha_real = 0
    average_real = 0
    averageMax_real = 0
    alpha_constrain = 0
    average_constrain = 0
    averageMax_constrain = 0
    for i in range(10):
        random.seed(42 + i)
        np.random.seed(42 + i)
        b = GMM(np.arange(1000), K, complete)
        alpha_constrain += ExactAlpha(complete, b, FairRadius(K, complete)) / 10
        average_constrain += AverageDistance(complete, b) / 10
        averageMax_constrain += MaxDistance(complete, b) / 10
    fairradius = FairRadius(K, complete)
    critical = IFRegion(alpha, fairradius, complete)
    space = OriginalMetric(K, complete, critical[0], critical[1])
    for random_number in range(10):
        random.seed(142 + random_number*9)
        np.random.seed(142 + random_number*9)
        solution = FMMDS(space, K, 0.05, complete)
        alpha_real += ExactAlpha(complete, solution, fairradius) / 10
        average_real += AverageDistance(complete, solution) / 10
        averageMax_real += MaxDistance(complete, solution) / 10
    alpha_constrain_CelebA.append(alpha_constrain)
    average_constrain_CelebA.append(average_constrain)
    averageMax_constrain_CelebA.append(averageMax_constrain)
    alpha_CelebA.append(alpha_real)
    average_CelebA.append(average_real)
    averageMax_CelebA.append(averageMax_real)
    print(f"Ratio of alpha: {alpha_real / alpha_constrain}, Ratio of Average: {average_real / average_constrain}, Ratio of AverageMax: {averageMax_real / averageMax_constrain}")

print("----------------------------------------------")

complete = glove
for K in range(2, 51):
    start_time = time.time()
    alpha_real = 0
    average_real = 0
    averageMax_real = 0
    alpha_constrain = 0
    average_constrain = 0
    averageMax_constrain = 0
    for i in range(10):
        random.seed(42 + i)
        np.random.seed(42 + i)
        b = GMM(np.arange(1000), K, complete)
        alpha_constrain += ExactAlpha(complete, b, FairRadius(K, complete)) / 10
        average_constrain += AverageDistance(complete, b) / 10
        averageMax_constrain += MaxDistance(complete, b) / 10
    fairradius = FairRadius(K, complete)
    critical = IFRegion(alpha, fairradius, complete)
    space = OriginalMetric(K, complete, critical[0], critical[1])
    for random_number in range(10):
        random.seed(142 + random_number*9)
        np.random.seed(142 + random_number*9)
        solution = FMMDS(space, K, 0.05, complete)
        alpha_real += ExactAlpha(complete, solution, fairradius) / 10
        average_real += AverageDistance(complete, solution) / 10
        averageMax_real += MaxDistance(complete, solution) / 10
    alpha_constrain_glove.append(alpha_constrain)
    average_constrain_glove.append(average_constrain)
    averageMax_constrain_glove.append(averageMax_constrain)
    alpha_glove.append(alpha_real)
    average_glove.append(average_real)
    averageMax_glove.append(averageMax_real)
    print(f"Ratio of alpha: {alpha_real / alpha_constrain}, Ratio of Average: {average_real / average_constrain}, Ratio of AverageMax: {averageMax_real / averageMax_constrain}")

print("----------------------------------------------")

complete = movielens
for K in range(2, 51):
    start_time = time.time()
    alpha_real = 0
    average_real = 0
    averageMax_real = 0
    alpha_constrain = 0
    average_constrain = 0
    averageMax_constrain = 0
    for i in range(10):
        random.seed(42 + i)
        np.random.seed(42 + i)
        b = GMM(np.arange(1000), K, complete)
        alpha_constrain += ExactAlpha(complete, b, FairRadius(K, complete)) / 10
        average_constrain += AverageDistance(complete, b) / 10
        averageMax_constrain += MaxDistance(complete, b) / 10
    fairradius = FairRadius(K, complete)
    critical = IFRegion(alpha, fairradius, complete)
    space = OriginalMetric(K, complete, critical[0], critical[1])
    for random_number in range(10):
        random.seed(142 + random_number*9)
        np.random.seed(142 + random_number*9)
        solution = FMMDS(space, K, 0.05, complete)
        alpha_real += ExactAlpha(complete, solution, fairradius) / 10
        average_real += AverageDistance(complete, solution) / 10
        averageMax_real += MaxDistance(complete, solution) / 10
    alpha_constrain_movielens.append(alpha_constrain)
    average_constrain_movielens.append(average_constrain)
    averageMax_constrain_movielens.append(averageMax_constrain)
    alpha_movielens.append(alpha_real)
    average_movielens.append(average_real)
    averageMax_movielens.append(averageMax_real)
    print(f"Ratio of alpha: {alpha_real / alpha_constrain}, Ratio of Average: {average_real / average_constrain}, Ratio of AverageMax: {averageMax_real / averageMax_constrain}")

print("----------------------------------------------")

complete = Gaussian_blob
for K in range(2, 51):
    start_time = time.time()
    alpha_real = 0
    average_real = 0
    averageMax_real = 0
    alpha_constrain = 0
    average_constrain = 0
    averageMax_constrain = 0
    for i in range(10):
        random.seed(42 + i)
        np.random.seed(42 + i)
        b = GMM(np.arange(1000), K, complete)
        alpha_constrain += ExactAlpha(complete, b, FairRadius(K, complete)) / 10
        average_constrain += AverageDistance(complete, b) / 10
        averageMax_constrain += MaxDistance(complete, b) / 10
    fairradius = FairRadius(K, complete)
    critical = IFRegion(alpha, fairradius, complete)
    space = OriginalMetric(K, complete, critical[0], critical[1])
    for random_number in range(10):
        random.seed(142 + random_number*9)
        np.random.seed(142 + random_number*9)
        solution = FMMDS(space, K, 0.05, complete)
        alpha_real += ExactAlpha(complete, solution, fairradius) / 10
        average_real += AverageDistance(complete, solution) / 10
        averageMax_real += MaxDistance(complete, solution) / 10
    alpha_constrain_gaussian.append(alpha_constrain)
    average_constrain_gaussian.append(average_constrain)
    averageMax_constrain_gaussian.append(averageMax_constrain)
    alpha_gaussian.append(alpha_real)
    average_gaussian.append(average_real)
    averageMax_gaussian.append(averageMax_real)
    print(f"Ratio of alpha: {alpha_real / alpha_constrain}, Ratio of Average: {average_real / average_constrain}, Ratio of AverageMax: {averageMax_real / averageMax_constrain}")

np.save("results_extended/max_min_alpha_unconstrained_CelebA.npy", np.array(alpha_constrain_CelebA))
np.save("results_extended/max_min_average_unconstrained_CelebA.npy", np.array(average_constrain_CelebA))
np.save("results_extended/max_min_averageMax_unconstrained_CelebA.npy", np.array(averageMax_constrain_CelebA))
np.save("results_extended/max_min_alpha_real_CelebA.npy", np.array(alpha_CelebA))
np.save("results_extended/max_min_average_real_CelebA.npy", np.array(average_CelebA))
np.save("results_extended/max_min_averageMax_real_CelebA.npy", np.array(averageMax_CelebA))

np.save("results_extended/max_min_alpha_unconstrained_glove.npy", np.array(alpha_constrain_glove))
np.save("results_extended/max_min_average_unconstrained_glove.npy", np.array(average_constrain_glove))
np.save("results_extended/max_min_averageMax_unconstrained_glove.npy", np.array(averageMax_constrain_glove))
np.save("results_extended/max_min_alpha_real_glove.npy", np.array(alpha_glove))
np.save("results_extended/max_min_average_real_glove.npy", np.array(average_glove))
np.save("results_extended/max_min_averageMax_real_glove.npy", np.array(averageMax_glove))

np.save("results_extended/max_min_alpha_unconstrained_movielens.npy", np.array(alpha_constrain_movielens))
np.save("results_extended/max_min_average_unconstrained_movielens.npy", np.array(average_constrain_movielens))
np.save("results_extended/max_min_averageMax_unconstrained_movielens.npy", np.array(averageMax_constrain_movielens))
np.save("results_extended/max_min_alpha_real_movielens.npy", np.array(alpha_movielens))
np.save("results_extended/max_min_average_real_movielens.npy", np.array(average_movielens))
np.save("results_extended/max_min_averageMax_real_movielens.npy", np.array(averageMax_movielens))

np.save("results_extended/max_min_alpha_unconstrained_gaussian.npy", np.array(alpha_constrain_gaussian))
np.save("results_extended/max_min_average_unconstrained_gaussian.npy", np.array(average_constrain_gaussian))
np.save("results_extended/max_min_averageMax_unconstrained_gaussian.npy", np.array(averageMax_constrain_gaussian))
np.save("results_extended/max_min_alpha_real_gaussian.npy", np.array(alpha_gaussian))
np.save("results_extended/max_min_average_real_gaussian.npy", np.array(average_gaussian))
np.save("results_extended/max_min_averageMax_real_gaussian.npy", np.array(averageMax_gaussian))
