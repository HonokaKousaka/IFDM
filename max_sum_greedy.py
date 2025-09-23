import numpy as np
import random
import time

def MaxSumDiversity(subset: np.ndarray, complete: np.ndarray) -> float:
    """
    Calculate diversity for Max-Sum Diversification objective, the sum of distance between all pairs of points in the given subset.
    
    Parameters:
        subset: Subset of points
        complete: Complete graph adjacency matrix containing distances between all pairs of points
    
    Returns:
        sum_dist: Diversity for Max-Sum Diversification objective, the sum of distance between all pairs of points in the subset
    """
    if len(subset) < 2:
        return 0.0
    sub_matrix = complete[np.ix_(subset, subset)]
    sum_dist = np.triu(sub_matrix, k=1).sum()
    
    return float(sum_dist)

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

def LocalSearch(original: tuple, complete: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Performs local search to get a Max-sum solution.
    
    Parameters:
        original: A tuple containing critical regions and their target sizes
        complete: Complete graph adjacency matrix containing distances between all pairs of points
        epsilon: Small constant controlling improvement threshold
        
    Returns:
        subsets: A numpy array containing the solution points
    """
    n = complete.shape[0]
    subsets = np.array([], dtype=int)
    R = []
    for critical_region in original:
        data_points = critical_region[0]
        number = critical_region[1]
        R_i = np.random.choice(data_points, size = number, replace = False)
        R.append(R_i)
        subsets = np.union1d(subsets, R_i)
    for index, critical_region in enumerate(original):
        data_points = critical_region[0]
        number = critical_region[1]
        improved = True
        while improved:
            improved = False
            for d in R[index]:
                for d_prime in np.setdiff1d(data_points, R[index]):
                    R_i_new = R[index].copy()
                    R_i_new = np.setdiff1d(R_i_new, [d])
                    R_i_new = np.union1d(R_i_new, [d_prime])
                    
                    old_diversity = MaxSumDiversity(subsets, complete)
                    new_subsets = np.union1d(np.setdiff1d(subsets, [d]), [d_prime])
                    new_diversity = MaxSumDiversity(new_subsets, complete)
                    
                    if new_diversity > (1 + epsilon / n) * old_diversity:
                        R[index] = R_i_new
                        subsets = new_subsets
                        improved = True
                        break
                if improved:
                    break
    
    return subsets

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

def MaxSumNoConstraint(complete_ori: np.ndarray, k: int) -> np.ndarray:
    """
    Finds a solution for Max-Sum Diversification without fairness constraints by greedily selecting pairs of points with maximum distances.
    
    Parameters:
        complete_ori: Complete graph adjacency matrix containing distances between all pairs of points
        k: Number of points to select
        
    Returns:
        np.ndarray: Array of selected point indices that maximize sum of pairwise distances
    """
    S = set()
    complete = complete_ori.copy()
    n = complete.shape[0]
    times = k // 2
    for i in range(times):
        max_index = np.unravel_index(np.argmax(complete), complete.shape)
        S.add(max_index[0])
        S.add(max_index[1])
        complete[max_index[0], :] = 0
        complete[max_index[1], :] = 0
        complete[:, max_index[0]] = 0
        complete[:, max_index[1]] = 0
    if k % 2 == 1:
        remaining = list(set(range(n)) - S)
        if remaining:
            rand_point = np.random.choice(remaining)
            S.add(rand_point)
    
    return np.array(list(S))

CelebA = np.load("dataset/CelebA_complete.npy")
glove = np.load("dataset/glove_complete.npy")
movielens = np.load("dataset/movielens_complete.npy")
Gaussian_blob = np.load("dataset/Gaussian_blob_complete.npy")
div_CelebA = []
alpha_CelebA = []
ratio_CelebA = []
div_glove = []
alpha_glove = []
ratio_glove = []
div_movielens = []
alpha_movielens = []
ratio_movielens = []
div_gaussian = []
alpha_gaussian = []
ratio_gaussian = []
time_CelebA = []
time_glove = []
time_movielens = []
time_gaussian = []

alpha = 1
complete = CelebA
for K in range(2, 51):
    final_diversity = 0
    final_ratio = 0
    final_alpha = 0
    fairradius = FairRadius(K, complete)
    critical = IFRegion(alpha, fairradius, complete)
    space = OriginalMetric(K, complete, critical[0], critical[1])
    Best_max = 0
    array = np.arange(complete.shape[0])
    for i in range(10):
        random.seed(42 + i)
        np.random.seed(42 + i)
        Best_solution = MaxSumNoConstraint(complete, K)
        if MaxSumDiversity(Best_solution, complete) > Best_max:
            Best_max = MaxSumDiversity(Best_solution, complete)
    start_time = time.time()
    for random_number in range(10):
        random.seed(142 + random_number * K)
        np.random.seed(142 + random_number * K)
        real_sol = LocalSearch(space, complete, 0.05)
        final_diversity = final_diversity + MaxSumDiversity(real_sol, complete) / 10
        final_alpha = final_alpha + ExactAlpha(complete, real_sol, fairradius) / 10
        final_ratio = final_ratio + MaxSumDiversity(real_sol, complete) / Best_max / 10
    end_time = time.time()
    print("The ratio: ", final_ratio, "alpha = ", final_alpha)
    div_CelebA.append(final_diversity)
    alpha_CelebA.append(final_alpha)
    ratio_CelebA.append(final_ratio)
    time_CelebA.append(end_time - start_time)

alpha = 1
complete = glove
for K in range(2, 51):
    final_diversity = 0
    final_ratio = 0
    final_alpha = 0
    fairradius = FairRadius(K, complete)
    critical = IFRegion(alpha, fairradius, complete)
    space = OriginalMetric(K, complete, critical[0], critical[1])
    Best_max = 0
    array = np.arange(complete.shape[0])
    for i in range(10):
        random.seed(42 + i)
        np.random.seed(42 + i)
        Best_solution = MaxSumNoConstraint(complete, K)
        if MaxSumDiversity(Best_solution, complete) > Best_max:
            Best_max = MaxSumDiversity(Best_solution, complete)
    start_time = time.time()
    for random_number in range(10):
        random.seed(142 + random_number * K)
        np.random.seed(142 + random_number * K)
        real_sol = LocalSearch(space, complete, 0.05)
        final_diversity = final_diversity + MaxSumDiversity(real_sol, complete) / 10
        final_alpha = final_alpha + ExactAlpha(complete, real_sol, fairradius) / 10
        final_ratio = final_ratio + MaxSumDiversity(real_sol, complete) / Best_max / 10
    end_time = time.time()
    print("The ratio: ", final_ratio, "alpha = ", final_alpha)
    div_glove.append(final_diversity)
    alpha_glove.append(final_alpha)
    ratio_glove.append(final_ratio)
    time_glove.append(end_time - start_time)

alpha = 1
complete = movielens
for K in range(2, 51):
    final_diversity = 0
    final_ratio = 0
    final_alpha = 0
    fairradius = FairRadius(K, complete)
    critical = IFRegion(alpha, fairradius, complete)
    space = OriginalMetric(K, complete, critical[0], critical[1])
    Best_max = 0
    array = np.arange(complete.shape[0])
    for i in range(10):
        random.seed(42 + i)
        np.random.seed(42 + i)
        Best_solution = MaxSumNoConstraint(complete, K)
        if MaxSumDiversity(Best_solution, complete) > Best_max:
            Best_max = MaxSumDiversity(Best_solution, complete)
    start_time = time.time()
    for random_number in range(10):
        random.seed(142 + random_number * K)
        np.random.seed(142 + random_number * K)
        real_sol = LocalSearch(space, complete, 0.05)
        final_diversity = final_diversity + MaxSumDiversity(real_sol, complete) / 10
        final_alpha = final_alpha + ExactAlpha(complete, real_sol, fairradius) / 10
        final_ratio = final_ratio + MaxSumDiversity(real_sol, complete) / Best_max / 10
    end_time = time.time()
    print("The ratio: ", final_ratio, "alpha = ", final_alpha)
    div_movielens.append(final_diversity)
    alpha_movielens.append(final_alpha)
    ratio_movielens.append(final_ratio)
    time_movielens.append(end_time - start_time)

alpha = 1
complete = Gaussian_blob
for K in range(2, 51):
    final_diversity = 0
    final_ratio = 0
    final_alpha = 0
    fairradius = FairRadius(K, complete)
    critical = IFRegion(alpha, fairradius, complete)
    space = OriginalMetric(K, complete, critical[0], critical[1])
    Best_max = 0
    array = np.arange(complete.shape[0])
    for i in range(10):
        random.seed(42 + i)
        np.random.seed(42 + i)
        Best_solution = MaxSumNoConstraint(complete, K)
        if MaxSumDiversity(Best_solution, complete) > Best_max:
            Best_max = MaxSumDiversity(Best_solution, complete)
    start_time = time.time()
    for random_number in range(10):
        random.seed(142 + random_number * K)
        np.random.seed(142 + random_number * K)
        real_sol = LocalSearch(space, complete, 0.05)
        final_diversity = final_diversity + MaxSumDiversity(real_sol, complete) / 10
        final_alpha = final_alpha + ExactAlpha(complete, real_sol, fairradius) / 10
        final_ratio = final_ratio + MaxSumDiversity(real_sol, complete) / Best_max / 10
    end_time = time.time()
    print("The ratio: ", final_ratio, "alpha = ", final_alpha)
    div_gaussian.append(final_diversity)
    alpha_gaussian.append(final_alpha)
    ratio_gaussian.append(final_ratio)
    time_gaussian.append(end_time - start_time)

np.save("results/max_sum_greedy_div_CelebA.npy", np.array(div_CelebA))
np.save("results/max_sum_greedy_alpha_CelebA.npy", np.array(alpha_CelebA))
np.save("results/max_sum_greedy_ratio_CelebA.npy", np.array(ratio_CelebA))
np.save("results/max_sum_greedy_time_CelebA.npy", np.array(time_CelebA))
np.save("results/max_sum_greedy_div_glove.npy", np.array(div_glove))
np.save("results/max_sum_greedy_alpha_glove.npy", np.array(alpha_glove))
np.save("results/max_sum_greedy_ratio_glove.npy", np.array(ratio_glove))
np.save("results/max_sum_greedy_time_glove.npy", np.array(time_glove))
np.save("results/max_sum_greedy_div_movielens.npy", np.array(div_movielens))
np.save("results/max_sum_greedy_alpha_movielens.npy", np.array(alpha_movielens))
np.save("results/max_sum_greedy_ratio_movielens.npy", np.array(ratio_movielens))
np.save("results/max_sum_greedy_time_movielens.npy", np.array(time_movielens))
np.save("results/max_sum_greedy_div_gaussian.npy", np.array(div_gaussian))
np.save("results/max_sum_greedy_alpha_gaussian.npy", np.array(alpha_gaussian))
np.save("results/max_sum_greedy_ratio_gaussian.npy", np.array(ratio_gaussian))
np.save("results/max_sum_greedy_time_gaussian.npy", np.array(time_gaussian))