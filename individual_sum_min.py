import numpy as np
import random
import math
import itertools
import copy
import time

def div_score_sum_nn(subset, complete) -> float:
    """
    Calculate the diversity score for a subset in the notion of sum-min distances.

    Args:
        complete: n*n numpy array, symmetric distance matrix satisfying triangle inequality
        subset: list or set of indices representing the selected subset
    Returns:
        div_score: the diversity score expressed in the notion of sum-min diversification
    """
    subset = list(subset) if isinstance(subset, set) else subset
    dist_mod = complete[np.ix_(subset, subset)]
    dist_mod[dist_mod == 0] = np.inf
    div_score = np.sum(np.min(dist_mod, axis=0))

    return div_score

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

def GMM_radii(points_index: np.ndarray, k: int, complete: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    R = []
    initial_point_index = random.choice(points_index)
    centers.append(initial_point_index)
    R.append(float('inf'))
    while (len(centers) < k):
        max_distance = 0
        max_distance_vector_index = None
        for i in points_index:
            distance = distancePS(centers, i, complete)
            if distance > max_distance:
                max_distance = distance
                max_distance_vector_index = i
        if max_distance_vector_index != None:
            centers.append(max_distance_vector_index)
        else:
            break

        min_distance = float('inf')
        length_centers = len(centers)
        for i in range(length_centers):
            for j in range(i + 1, length_centers):
                if complete[centers[i]][centers[j]] < min_distance:
                    min_distance = complete[centers[i]][centers[j]]
        R.append(min_distance)
    centers = np.array(centers)
    R = np.array(R)

    return centers, R

def is_fair_solution(S_union: set, original: tuple) -> bool:
    """
    Check if a solution satisfies the fairness constraints specified by the original tuple.
    
    Parameters:
        S_union: Set of selected points
        original: Tuple of tuples [(P_0, k_0), ..., (P_m, k_m)], where P_i is a numpy array of point indices
                  and k_i is the desired maximal number of points for group i
        
    Returns:
        bool: True if the solution satisfies all fairness constraints, False otherwise
    """
    for P_i, k_i in original:
        count = len(S_union.intersection(P_i))
        if count != k_i:
            return False
    return True

def ApproximateLargestSubset(B, complete, original, i) -> np.ndarray:
    """
    Given a set of balls B, a distance matrix complete, group information original, and a target group index i,
    return an approximately largest subset of balls B' such that the remaining points (P \ B') satisfy:
    - At least k_i - |B'| points from group i.
    - At least k_l points from group l ≠ i.

    Args:
        B: List of tuples [(center, radius)], where center is an int (point index) and radius is a float.
        complete: n*n numpy array, symmetric distance matrix.
        original: Tuple of tuples [(P_0, k_0), ..., (P_m, k_m)], where P_i is a numpy array of point indices
                  and k_i is the desired maximal number of points for group i.
        i: Integer, index of the target group.

    Returns:
        np.ndarray: Array of selected balls [(center, radius)].
    """
    B_points = []
    for center, radius in B:
        covered = [idx for idx in range(complete.shape[0]) if complete[idx, center] <= radius]
        B_points.append(set(covered))
    B_flat = set(itertools.chain.from_iterable(B_points))

    n_groups = len(original)
    demands = [original[l][1] for l in range(n_groups)]
    P_prime = [list(original[l][0]) for l in range(n_groups)]

    for l in range(n_groups):
        uncovered = set(P_prime[l]).difference(B_flat)
        a = min(demands[l], len(uncovered))
        demands[l] -= a
        P_prime[l] = list(set(P_prime[l]).difference(uncovered))[:len(P_prime[l]) - a]

    SOL1 = []
    fin_ind_b = -1
    for ind_b, (center, radius) in enumerate(B):
        B_diff_b = set(B_flat).difference(B_points[ind_b])
        if len(set(original[i][0]).difference(B_diff_b)) < demands[i] - 1:
            continue
        valid = True
        for l in range(n_groups):
            if l == i:
                continue
            if len(set(original[l][0]).difference(B_diff_b)) < demands[l]:
                valid = False
                break
        if valid:
            SOL1 = [(center, radius)]
            fin_ind_b = ind_b
            break

    SOL2 = []
    fin_ind_b1, fin_ind_b2 = -1, -1
    for ind_b1, (center1, radius1) in enumerate(B):
        for ind_b2, (center2, radius2) in enumerate(B):
            if ind_b1 == ind_b2:
                continue
            B_diff_b = set(B_flat).difference(B_points[ind_b1]).difference(B_points[ind_b2])
            if len(set(original[i][0]).difference(B_diff_b)) < demands[i] - 2:
                continue
            valid = True
            for l in range(n_groups):
                if l == i:
                    continue
                if len(set(original[l][0]).difference(B_diff_b)) < demands[l]:
                    valid = False
                    break
            if valid:
                SOL2 = [(center1, radius1), (center2, radius2)]
                fin_ind_b1, fin_ind_b2 = ind_b1, ind_b2
                break
        if SOL2:
            break

    SOLM = copy.deepcopy(B)
    demands_m = demands.copy()
    selected_indices = []

    for l in range(n_groups):
        if l == i:
            continue
        if demands_m[l] > 0:
            max_contrib, max_ind = -1, -1
            for ind_b, (center, radius) in enumerate(SOLM):
                contrib = len(set(original[l][0]).intersection(B_points[ind_b]))
                if contrib > max_contrib:
                    max_contrib = contrib
                    max_ind = ind_b
            if max_ind != -1:
                selected_indices.append(max_ind)
                covered_points = B_points[max_ind]
                for p in covered_points:
                    group_id = next(idx for idx, g in enumerate(original) if p in g[0])
                    if demands_m[group_id] > 0:
                        demands_m[group_id] -= 1
                SOLM.pop(max_ind)
                B_points.pop(max_ind)

    SOLM_flat = set(itertools.chain.from_iterable(
        [set(idx for idx in range(complete.shape[0]) if complete[idx, c] <= r) for c, r in SOLM]
    ))
    all_contributions_zero = all(demands_m[l] == 0 for l in range(n_groups) if l != i)
    len_SOLM = len(SOLM)

    if len_SOLM > 2 and demands[i] < len_SOLM and all_contributions_zero:
        return np.array(SOLM, dtype=object)
    elif SOL2:
        return np.array(SOL2, dtype=object)
    elif SOL1:
        return np.array(SOL1, dtype=object)
    else:
        return np.array([], dtype=object)
    
def SumNN(original: tuple, complete: np.ndarray, k: int) -> np.ndarray:
    """
    Implements the algorithm for sum-min diversification under partition matroid constraints for fair subset selection.
    
    Parameters:
        original: Tuple of tuples [(P_0, k_0), ..., (P_m, k_m)], where P_i is a numpy array of point indices
                  and k_i is the desired maximal number of points for group i.
        complete: Complete graph adjacency matrix containing distances between all pairs of points
        k: Total number of points to select across all groups
        
    Returns:
        solution: Array of selected point indices that maximizes sum-min diversification objective while satisfying fairness
    """
    SOL = []
    number_groups = len(original)
    for group in original:
        SOL.extend(np.random.choice(group[0], size = group[1], replace = False))
    
    for i in range(number_groups):
        G_points, G_radii = GMM_radii(original[i][0], k, complete)
        for j in range(1, len(G_points)):
            subsets = [set() for _ in range(number_groups)]
            t_j = len(G_points)            
            for t in range(len(G_points) - 1, j - 1, -1):
                if G_radii[t] >= G_radii[j] / 2:
                    t_j = t
                    break
            B = [(G_points[l], G_radii[t_j] / 2) for l in range(t_j)]
            B_pr = ApproximateLargestSubset(B, complete, original, i)
            if len(B_pr) > original[i][1]:
                C = [int(b[0]) for b in B_pr]
                subsets[i] = set(np.random.choice(C, size=original[i][1], replace=False))
            else:
                subsets[i] = set([int(b[0]) for b in B_pr])
                remaining = set(p for p in original[i][0] if all(complete[p][int(b[0])] > b[1] for b in B_pr))
                if len(remaining) >= original[i][1] - len(B_pr):
                    subsets[i].update(np.random.choice(list(remaining), size=original[i][1] - len(B_pr), replace=False))
            
            for l in range(number_groups):
                if l != i:
                    remaining = set(p for p in original[l][0] if all(complete[p][int(b[0])] > b[1] for b in B_pr))
                    if len(remaining) >= original[l][1]:
                        subsets[l] = set(np.random.choice(list(remaining), size=original[l][1], replace=False))
            
            S_union = set()
            for l in range(number_groups):
                S_union.update(subsets[l])
            if len(S_union) == k and is_fair_solution(S_union, original):
                current_div = div_score_sum_nn(np.array(list(S_union)), complete)
                sol_div = div_score_sum_nn(np.array(SOL), complete) if SOL else -1
                if current_div > sol_div:
                    SOL = list(S_union)
    
    return np.array(SOL)

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
    alpha_real = 0
    average_real = 0
    averageMax_real = 0
    alpha_constrain = 0
    average_constrain = 0
    averageMax_constrain = 0
    fairradius = FairRadius(K, complete)
    critical = IFRegion(alpha, fairradius, complete)
    space = OriginalMetric(K, complete, critical[0], critical[1])
    Best_max = 0
    array = np.arange(complete.shape[0])
    for i in range(10):
        random.seed(42 + i)
        np.random.seed(42 + i)
        new_tuple = (array, K)
        big_tuple = (new_tuple, )
        Best_solution = SumNN(big_tuple, complete, K)
        alpha_constrain = alpha_constrain + ExactAlpha(complete, Best_solution, fairradius) / 10
        average_constrain = average_constrain + AverageDistance(complete, Best_solution) / 10
        averageMax_constrain = averageMax_constrain + MaxDistance(complete, Best_solution) / 10
    start_time = time.time()
    for random_number in range(10):
        random.seed(1434 + random_number*K)
        np.random.seed(1434 + random_number*K)
        solution = SumNN(space, complete, K)
        alpha_real = alpha_real + ExactAlpha(complete, solution, fairradius) / 10
        average_real = average_real + AverageDistance(complete, solution) / 10
        averageMax_real = averageMax_real + MaxDistance(complete, solution) / 10
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
    alpha_real = 0
    average_real = 0
    averageMax_real = 0
    alpha_constrain = 0
    average_constrain = 0
    averageMax_constrain = 0
    fairradius = FairRadius(K, complete)
    critical = IFRegion(alpha, fairradius, complete)
    space = OriginalMetric(K, complete, critical[0], critical[1])
    Best_max = 0
    array = np.arange(complete.shape[0])
    for i in range(10):
        random.seed(42 + i)
        np.random.seed(42 + i)
        new_tuple = (array, K)
        big_tuple = (new_tuple, )
        Best_solution = SumNN(big_tuple, complete, K)
        alpha_constrain = alpha_constrain + ExactAlpha(complete, Best_solution, fairradius) / 10
        average_constrain = average_constrain + AverageDistance(complete, Best_solution) / 10
        averageMax_constrain = averageMax_constrain + MaxDistance(complete, Best_solution) / 10
    start_time = time.time()
    for random_number in range(10):
        random.seed(1434 + random_number*K)
        np.random.seed(1434 + random_number*K)
        solution = SumNN(space, complete, K)
        alpha_real = alpha_real + ExactAlpha(complete, solution, fairradius) / 10
        average_real = average_real + AverageDistance(complete, solution) / 10
        averageMax_real = averageMax_real + MaxDistance(complete, solution) / 10
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
    alpha_real = 0
    average_real = 0
    averageMax_real = 0
    alpha_constrain = 0
    average_constrain = 0
    averageMax_constrain = 0
    fairradius = FairRadius(K, complete)
    critical = IFRegion(alpha, fairradius, complete)
    space = OriginalMetric(K, complete, critical[0], critical[1])
    Best_max = 0
    array = np.arange(complete.shape[0])
    for i in range(10):
        random.seed(42 + i)
        np.random.seed(42 + i)
        new_tuple = (array, K)
        big_tuple = (new_tuple, )
        Best_solution = SumNN(big_tuple, complete, K)
        alpha_constrain = alpha_constrain + ExactAlpha(complete, Best_solution, fairradius) / 10
        average_constrain = average_constrain + AverageDistance(complete, Best_solution) / 10
        averageMax_constrain = averageMax_constrain + MaxDistance(complete, Best_solution) / 10
    start_time = time.time()
    for random_number in range(10):
        random.seed(1434 + random_number*K)
        np.random.seed(1434 + random_number*K)
        solution = SumNN(space, complete, K)
        alpha_real = alpha_real + ExactAlpha(complete, solution, fairradius) / 10
        average_real = average_real + AverageDistance(complete, solution) / 10
        averageMax_real = averageMax_real + MaxDistance(complete, solution) / 10
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
    alpha_real = 0
    average_real = 0
    averageMax_real = 0
    alpha_constrain = 0
    average_constrain = 0
    averageMax_constrain = 0
    fairradius = FairRadius(K, complete)
    critical = IFRegion(alpha, fairradius, complete)
    space = OriginalMetric(K, complete, critical[0], critical[1])
    Best_max = 0
    array = np.arange(complete.shape[0])
    for i in range(10):
        random.seed(42 + i)
        np.random.seed(42 + i)
        new_tuple = (array, K)
        big_tuple = (new_tuple, )
        Best_solution = SumNN(big_tuple, complete, K)
        alpha_constrain = alpha_constrain + ExactAlpha(complete, Best_solution, fairradius) / 10
        average_constrain = average_constrain + AverageDistance(complete, Best_solution) / 10
        averageMax_constrain = averageMax_constrain + MaxDistance(complete, Best_solution) / 10
    start_time = time.time()
    for random_number in range(10):
        random.seed(1434 + random_number*K)
        np.random.seed(1434 + random_number*K)
        solution = SumNN(space, complete, K)
        alpha_real = alpha_real + ExactAlpha(complete, solution, fairradius) / 10
        average_real = average_real + AverageDistance(complete, solution) / 10
        averageMax_real = averageMax_real + MaxDistance(complete, solution) / 10
    alpha_constrain_gaussian.append(alpha_constrain)
    average_constrain_gaussian.append(average_constrain)
    averageMax_constrain_gaussian.append(averageMax_constrain)
    alpha_gaussian.append(alpha_real)
    average_gaussian.append(average_real)
    averageMax_gaussian.append(averageMax_real)
    print(f"Ratio of alpha: {alpha_real / alpha_constrain}, Ratio of Average: {average_real / average_constrain}, Ratio of AverageMax: {averageMax_real / averageMax_constrain}")

np.save("results_extended/sum_min_alpha_unconstrained_CelebA.npy", np.array(alpha_constrain_CelebA))
np.save("results_extended/sum_min_average_unconstrained_CelebA.npy", np.array(average_constrain_CelebA))
np.save("results_extended/sum_min_averageMax_unconstrained_CelebA.npy", np.array(averageMax_constrain_CelebA))
np.save("results_extended/sum_min_alpha_real_CelebA.npy", np.array(alpha_CelebA))
np.save("results_extended/sum_min_average_real_CelebA.npy", np.array(average_CelebA))
np.save("results_extended/sum_min_averageMax_real_CelebA.npy", np.array(averageMax_CelebA))

np.save("results_extended/sum_min_alpha_unconstrained_glove.npy", np.array(alpha_constrain_glove))
np.save("results_extended/sum_min_average_unconstrained_glove.npy", np.array(average_constrain_glove))
np.save("results_extended/sum_min_averageMax_unconstrained_glove.npy", np.array(averageMax_constrain_glove))
np.save("results_extended/sum_min_alpha_real_glove.npy", np.array(alpha_glove))
np.save("results_extended/sum_min_average_real_glove.npy", np.array(average_glove))
np.save("results_extended/sum_min_averageMax_real_glove.npy", np.array(averageMax_glove))

np.save("results_extended/sum_min_alpha_unconstrained_movielens.npy", np.array(alpha_constrain_movielens))
np.save("results_extended/sum_min_average_unconstrained_movielens.npy", np.array(average_constrain_movielens))
np.save("results_extended/sum_min_averageMax_unconstrained_movielens.npy", np.array(averageMax_constrain_movielens))
np.save("results_extended/sum_min_alpha_real_movielens.npy", np.array(alpha_movielens))
np.save("results_extended/sum_min_average_real_movielens.npy", np.array(average_movielens))
np.save("results_extended/sum_min_averageMax_real_movielens.npy", np.array(averageMax_movielens))

np.save("results_extended/sum_min_alpha_unconstrained_gaussian.npy", np.array(alpha_constrain_gaussian))
np.save("results_extended/sum_min_average_unconstrained_gaussian.npy", np.array(average_constrain_gaussian))
np.save("results_extended/sum_min_averageMax_unconstrained_gaussian.npy", np.array(averageMax_constrain_gaussian))
np.save("results_extended/sum_min_alpha_real_gaussian.npy", np.array(alpha_gaussian))
np.save("results_extended/sum_min_average_real_gaussian.npy", np.array(average_gaussian))
np.save("results_extended/sum_min_averageMax_real_gaussian.npy", np.array(averageMax_gaussian))
